//! Codelor Color 1.2 — segmentation d'une couleur cible (chroma key), réimplémenté avec Candle.

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use std::collections::HashMap;
use std::path::Path;

/// BatchNorm en mode évaluation, replié en (échelle, décalage) par canal.
struct Bn {
    scale: Tensor,
    shift: Tensor,
}

impl Bn {
    fn load(w: &mut Weights, p: &str) -> Result<Self> {
        let g = w.get(&format!("{p}.weight"))?;
        let b = w.get(&format!("{p}.bias"))?;
        let m = w.get(&format!("{p}.running_mean"))?;
        let v = w.get(&format!("{p}.running_var"))?;
        let scale = g.div(&(v + 1e-5)?.sqrt()?)?;
        let shift = (b - m.mul(&scale)?)?;
        let n = scale.dim(0)?;
        Ok(Self { scale: scale.reshape((1, n, 1, 1))?, shift: shift.reshape((1, n, 1, 1))? })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        x.broadcast_mul(&self.scale)?.broadcast_add(&self.shift)
    }
}

struct Weights(HashMap<String, Tensor>);

impl Weights {
    fn get(&mut self, name: &str) -> Result<Tensor> {
        self.0
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("tenseur manquant : {name}"))?
            .to_dtype(DType::F32)
            .map_err(Into::into)
    }
}

struct Conv {
    w: Tensor,
    b: Option<Tensor>,
    pad: usize,
    stride: usize,
    groups: usize,
}

impl Conv {
    fn load(w: &mut Weights, p: &str, pad: usize, stride: usize, groups: usize, bias: bool) -> Result<Self> {
        let weight = w.get(&format!("{p}.weight"))?;
        let b = if bias {
            let b = w.get(&format!("{p}.bias"))?;
            let n = b.dim(0)?;
            Some(b.reshape((1, n, 1, 1))?)
        } else {
            None
        };
        Ok(Self { w: weight, b, pad, stride, groups })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = x.conv2d(&self.w, self.pad, self.stride, 1, self.groups)?;
        match &self.b {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

fn rms_norm(x: &Tensor, w: &Tensor) -> candle_core::Result<Tensor> {
    let r = (x.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
    x.broadcast_div(&r)?.broadcast_mul(w)
}

struct Lin {
    wt: Tensor,
    b: Option<Tensor>,
}

impl Lin {
    fn load(w: &mut Weights, p: &str, bias: bool) -> Result<Self> {
        let wt = w.get(&format!("{p}.weight"))?.t()?.contiguous()?;
        let b = if bias { Some(w.get(&format!("{p}.bias"))?) } else { None };
        Ok(Self { wt, b })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = x.broadcast_matmul(&self.wt)?;
        match &self.b {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

struct Block {
    norm1: Tensor,
    qkv: Lin,
    q_norm: Tensor,
    k_norm: Tensor,
    proj: Lin,
    ls1: Tensor,
    norm2: Tensor,
    w12: Lin,
    w3: Lin,
    ls2: Tensor,
    // Attention croisée sur la couleur (un seul token de condition)
    xa_v: Lin,
    xa_proj: Lin,
    dim: usize,
}

pub struct Codelor {
    patch: Conv,
    patch_bn: Bn,
    b1_dw: Conv,
    b1_bn1: Bn,
    b1_pw: Conv,
    b1_bn2: Bn,
    d8_blur: Conv,
    d8_conv: Conv,
    d8_bn: Bn,
    d16_blur: Conv,
    d16_conv: Conv,
    d16_bn: Bn,
    color1: Lin,
    color2: Lin,
    pos: Tensor,
    blocks: Vec<Block>,
    depth: usize,
    heads: usize,
    norm_out: Tensor,
    proj16: Conv,
    up8: Conv,
    up8_bn: Bn,
    up4: Conv,
    up4_bn: Bn,
    up1a: Conv,
    up1a_bn: Bn,
    up1b: Conv,
    up1b_bn: Bn,
    head: Conv,
    gate8: f32,
    gate4: f32,
    sobel_x: Tensor,
    sobel_y: Tensor,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Matrice d'interpolation bilinéaire 1D (align_corners = false), forme [out, in].
fn interp_matrix(out: usize, inp: usize) -> Vec<f32> {
    let mut m = vec![0f32; out * inp];
    let scale = inp as f32 / out as f32;
    for i in 0..out {
        let src = ((i as f32 + 0.5) * scale - 0.5).max(0.0);
        let i0 = (src.floor() as usize).min(inp - 1);
        let i1 = (i0 + 1).min(inp - 1);
        let l = src - i0 as f32;
        m[i * inp + i0] += 1.0 - l;
        m[i * inp + i1] += l;
    }
    m
}

/// Redimensionne [1, C, H, W] vers [1, C, oh, ow] (bilinéaire).
fn resize(x: &Tensor, oh: usize, ow: usize) -> candle_core::Result<Tensor> {
    let (_, c, h, w) = x.dims4()?;
    if h == oh && w == ow {
        return Ok(x.clone());
    }
    let dev = x.device();
    let ry = Tensor::from_vec(interp_matrix(oh, h), (oh, h), dev)?;
    let rxt = Tensor::from_vec(interp_matrix(ow, w), (ow, w), dev)?.t()?.contiguous()?;
    let x = x.reshape((c, h, w))?;
    let x = x.broadcast_matmul(&rxt)?; // [C, H, ow]
    let x = ry.broadcast_matmul(&x)?; // [C, oh, ow]
    x.reshape((1, c, oh, ow))
}

/// Bilinéaire sur un simple buffer f32 (pour remettre le masque à la taille d'origine).
pub fn resize_plane(src: &[f32], w: usize, h: usize, ow: usize, oh: usize) -> Vec<f32> {
    let mut out = vec![0f32; ow * oh];
    let sx = w as f32 / ow as f32;
    let sy = h as f32 / oh as f32;
    for y in 0..oh {
        let fy = ((y as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (fy as usize).min(h - 1);
        let y1 = (y0 + 1).min(h - 1);
        let ly = fy - y0 as f32;
        for x in 0..ow {
            let fx = ((x as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = (fx as usize).min(w - 1);
            let x1 = (x0 + 1).min(w - 1);
            let lx = fx - x0 as f32;
            let a = src[y0 * w + x0] * (1.0 - lx) + src[y0 * w + x1] * lx;
            let b = src[y1 * w + x0] * (1.0 - lx) + src[y1 * w + x1] * lx;
            out[y * ow + x] = a * (1.0 - ly) + b * ly;
        }
    }
    out
}

impl Codelor {
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).context("config.json")?)?;
        let hp = &cfg["hyperparameters"];
        let dim = hp["dim"].as_u64().unwrap_or(128) as usize;
        let depth = hp["depth"].as_u64().unwrap_or(8) as usize;
        let heads = hp["heads"].as_u64().unwrap_or(8) as usize;
        let tied_after = hp["tied_after"].as_u64().unwrap_or(4) as usize;
        let n_unique = if tied_after == 0 { depth } else { depth.min(tied_after) };
        let map = candle_core::safetensors::load(dir.join("model.safetensors"), &Device::Cpu)?;
        let mut w = Weights(map);

        let c32 = 32;
        let c64 = 64;
        let mut blocks = Vec::new();
        for i in 0..n_unique {
            let p = format!("blocks.{i}");
            let x = format!("color_xattn.{i}");
            // kv : [2*dim, dim] -> on ne garde que la partie V (le softmax sur un seul token vaut 1).
            let kv = w.get(&format!("{x}.kv.weight"))?;
            let v_w = kv.narrow(0, dim, dim)?.t()?.contiguous()?;
            blocks.push(Block {
                norm1: w.get(&format!("{p}.norm1.w"))?,
                qkv: Lin::load(&mut w, &format!("{p}.attn.qkv"), false)?,
                q_norm: w.get(&format!("{p}.attn.q_norm.w"))?,
                k_norm: w.get(&format!("{p}.attn.k_norm.w"))?,
                proj: Lin::load(&mut w, &format!("{p}.attn.proj"), false)?,
                ls1: w.get(&format!("{p}.ls1.gamma"))?,
                norm2: w.get(&format!("{p}.norm2.w"))?,
                w12: Lin::load(&mut w, &format!("{p}.ffn.w12"), false)?,
                w3: Lin::load(&mut w, &format!("{p}.ffn.w3"), false)?,
                ls2: w.get(&format!("{p}.ls2.gamma"))?,
                xa_v: Lin { wt: v_w, b: None },
                xa_proj: Lin::load(&mut w, &format!("{x}.proj"), false)?,
                dim,
            });
        }
        let sobel_x = Tensor::from_vec(vec![-1f32, 0., 1., -2., 0., 2., -1., 0., 1.], (1, 1, 3, 3), &Device::Cpu)?;
        let sobel_y = Tensor::from_vec(vec![-1f32, -2., -1., 0., 0., 0., 1., 2., 1.], (1, 1, 3, 3), &Device::Cpu)?;
        let gate8 = w.get("decoder.gate8")?.to_vec1::<f32>()?[0];
        let gate4 = w.get("decoder.gate4")?.to_vec1::<f32>()?[0];
        Ok(Self {
            patch: Conv::load(&mut w, "stem.patch.0", 0, 4, 1, false)?,
            patch_bn: Bn::load(&mut w, "stem.patch.1")?,
            b1_dw: Conv::load(&mut w, "stem.block1.0", 3, 1, c32, false)?,
            b1_bn1: Bn::load(&mut w, "stem.block1.1")?,
            b1_pw: Conv::load(&mut w, "stem.block1.3", 0, 1, 1, false)?,
            b1_bn2: Bn::load(&mut w, "stem.block1.4")?,
            d8_blur: Conv::load(&mut w, "stem.down8.blur", 1, 1, c32, false)?,
            d8_conv: Conv::load(&mut w, "stem.down8.conv", 0, 2, 1, false)?,
            d8_bn: Bn::load(&mut w, "stem.down8.bn")?,
            d16_blur: Conv::load(&mut w, "stem.down16.blur", 1, 1, c64, false)?,
            d16_conv: Conv::load(&mut w, "stem.down16.conv", 0, 2, 1, false)?,
            d16_bn: Bn::load(&mut w, "stem.down16.bn")?,
            color1: Lin::load(&mut w, "color_mlp.0", true)?,
            color2: Lin::load(&mut w, "color_mlp.2", true)?,
            pos: w.get("pos")?,
            blocks,
            depth,
            heads,
            norm_out: w.get("norm_out.w")?,
            proj16: Conv::load(&mut w, "decoder.proj16", 0, 1, 1, false)?,
            up8: Conv::load(&mut w, "decoder.up8.0", 1, 1, 1, false)?,
            up8_bn: Bn::load(&mut w, "decoder.up8.1")?,
            up4: Conv::load(&mut w, "decoder.up4.0", 1, 1, 1, false)?,
            up4_bn: Bn::load(&mut w, "decoder.up4.1")?,
            up1a: Conv::load(&mut w, "decoder.up1.0", 1, 1, 1, false)?,
            up1a_bn: Bn::load(&mut w, "decoder.up1.1")?,
            up1b: Conv::load(&mut w, "decoder.up1.3", 1, 1, 1, false)?,
            up1b_bn: Bn::load(&mut w, "decoder.up1.4")?,
            head: Conv::load(&mut w, "decoder.head", 0, 1, 1, true)?,
            gate8,
            gate4,
            sobel_x,
            sobel_y,
        })
    }

    fn rope(&self, x: &Tensor, h: usize, w: usize) -> candle_core::Result<Tensor> {
        // x : [1, heads, N, D]
        let d = x.dim(3)?;
        let d4 = d / 4;
        let n = h * w;
        let mut cos = Vec::with_capacity(n * d / 2);
        let mut sin = Vec::with_capacity(n * d / 2);
        let freqs: Vec<f32> = (0..d4).map(|i| 1.0 / 100f32.powf(i as f32 / d4 as f32)).collect();
        for yy in 0..h {
            for xx in 0..w {
                for f in &freqs {
                    let a = xx as f32 * f;
                    cos.push(a.cos());
                    sin.push(a.sin());
                }
                for f in &freqs {
                    let a = yy as f32 * f;
                    cos.push(a.cos());
                    sin.push(a.sin());
                }
            }
        }
        let cos = Tensor::from_vec(cos, (1, 1, n, d / 2), x.device())?;
        let sin = Tensor::from_vec(sin, (1, 1, n, d / 2), x.device())?;
        let x1 = x.narrow(3, 0, d / 2)?;
        let x2 = x.narrow(3, d / 2, d / 2)?;
        let a = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let b = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        Tensor::cat(&[a, b], 3)
    }

    fn block(&self, blk: &Block, x: &Tensor, c_tok: &Tensor, h: usize, w: usize) -> candle_core::Result<Tensor> {
        let (b, n, c) = x.dims3()?;
        let dh = c / self.heads;
        // Self-attention
        let hn = rms_norm(x, &blk.norm1)?;
        let qkv = blk.qkv.forward(&hn)?.reshape((b, n, 3, self.heads, dh))?.permute((2, 0, 3, 1, 4))?;
        let q = rms_norm(&qkv.get(0)?.contiguous()?, &blk.q_norm)?;
        let k = rms_norm(&qkv.get(1)?.contiguous()?, &blk.k_norm)?;
        let v = qkv.get(2)?.contiguous()?;
        let q = self.rope(&q, h, w)?;
        let k = self.rope(&k, h, w)?;
        let att = (q.matmul(&k.t()?)? * (1.0 / (dh as f64).sqrt()))?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v)?.transpose(1, 2)?.reshape((b, n, c))?;
        let y = blk.proj.forward(&y)?;
        let x = (x + y.broadcast_mul(&blk.ls1)?)?;
        // SwiGLU
        let hn = rms_norm(&x, &blk.norm2)?;
        let ab = blk.w12.forward(&hn)?;
        let hidden = ab.dim(2)? / 2;
        let a = ab.narrow(2, 0, hidden)?;
        let bb = ab.narrow(2, hidden, hidden)?;
        let f = blk.w3.forward(&(candle_nn::ops::silu(&a)? * bb)?)?;
        let x = (x + f.broadcast_mul(&blk.ls2)?)?;
        // Attention croisée couleur : avec un seul token de condition, la sortie vaut proj(V).
        let v = blk.xa_v.forward(c_tok)?; // [1,1,C]
        let _ = blk.dim;
        let xa = blk.xa_proj.forward(&v)?;
        x.broadcast_add(&xa)
    }

    /// `img` : RGB [0,1] au format [1,3,H,W] avec H, W multiples de 16.
    /// Renvoie la probabilité [H*W] que chaque pixel soit de la couleur cible.
    pub fn predict(&self, img: &Tensor, color: [f32; 3]) -> Result<Vec<f32>> {
        let (_, _, hh, ww) = img.dims4()?;
        let silu = |t: Tensor| candle_nn::ops::silu(&t);
        // Stem
        let x = silu(self.patch_bn.forward(&self.patch.forward(img)?)?)?;
        let y = silu(self.b1_bn1.forward(&self.b1_dw.forward(&x)?)?)?;
        let y = self.b1_bn2.forward(&self.b1_pw.forward(&y)?)?;
        let f4 = (x + y)?;
        let f8 = silu(self.d8_bn.forward(&self.d8_conv.forward(&self.d8_blur.forward(&f4)?)?)?)?;
        let f16 = silu(self.d16_bn.forward(&self.d16_conv.forward(&self.d16_blur.forward(&f8)?)?)?)?;
        let (_, c, h, w) = f16.dims4()?;
        let n = h * w;
        let mut x = f16.flatten_from(2)?.transpose(1, 2)?.contiguous()?; // [1,N,C]
        if n <= self.pos.dim(1)? {
            x = x.broadcast_add(&self.pos.narrow(1, 0, n)?)?;
        }
        let ref_t = Tensor::new(&[color], &Device::Cpu)?; // [1,3]
        let c_tok = self.color2.forward(&silu(self.color1.forward(&ref_t)?)?)?.unsqueeze(1)?; // [1,1,C]
        for i in 0..self.depth {
            let blk = &self.blocks[i % self.blocks.len()];
            x = self.block(blk, &x, &c_tok, h, w)?;
        }
        let x = rms_norm(&x, &self.norm_out)?;
        let latent = x.transpose(1, 2)?.reshape((1, c, h, w))?;

        // Décodeur
        let x = self.proj16.forward(&latent)?;
        let (_, _, h8, w8) = f8.dims4()?;
        let (_, _, h4, w4) = f4.dims4()?;
        let x = resize(&x, h8, w8)?;
        let x = Tensor::cat(&[x, (f8 * sigmoid(self.gate8) as f64)?], 1)?;
        let x = silu(self.up8_bn.forward(&self.up8.forward(&x)?)?)?;
        let x = resize(&x, h4, w4)?;
        let x = Tensor::cat(&[x, (f4 * sigmoid(self.gate4) as f64)?], 1)?;
        let x = silu(self.up4_bn.forward(&self.up4.forward(&x)?)?)?;
        let x = resize(&x, hh, ww)?;
        // Similarité cosinus avec la couleur cible
        let r = Tensor::new(&color, &Device::Cpu)?.reshape((1, 3, 1, 1))?;
        let rn = (color.iter().map(|v| v * v).sum::<f32>().sqrt()).max(1e-6);
        let img_n = img.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(1e-6f32, f32::MAX)?;
        let cos = (img.broadcast_mul(&r)?.sum_keepdim(1)?.broadcast_div(&img_n)? / rn as f64)?;
        // Contours (Sobel)
        let gray = img.mean_keepdim(1)?;
        let ex = gray.conv2d(&self.sobel_x, 1, 1, 1, 1)?;
        let ey = gray.conv2d(&self.sobel_y, 1, 1, 1, 1)?;
        let edge = ((ex.sqr()? + ey.sqr()?)? + 1e-6)?.sqrt()?;
        let x = Tensor::cat(&[x, img.clone(), cos, edge], 1)?;
        let x = silu(self.up1a_bn.forward(&self.up1a.forward(&x)?)?)?;
        let x = silu(self.up1b_bn.forward(&self.up1b.forward(&x)?)?)?;
        let logits = self.head.forward(&x)?;
        let prob = candle_nn::ops::sigmoid(&logits)?.flatten_all()?.to_vec1::<f32>()?;
        Ok(prob)
    }

    /// Traite une image RGB8 de taille quelconque.
    /// Renvoie le masque (probabilités) à la résolution de traitement : `(masque, largeur, hauteur)`.
    pub fn segment(&self, rgb: &image::RgbImage, color: [f32; 3], max_side: u32) -> Result<(Vec<f32>, usize, usize)> {
        let (w0, h0) = rgb.dimensions();
        let scale = (max_side as f32 / w0.max(h0) as f32).min(1.0);
        let round16 = |v: f32| (((v / 16.0).round() as u32).max(1) * 16) as usize;
        let (pw, ph) = (round16(w0 as f32 * scale), round16(h0 as f32 * scale));
        let small = image::imageops::resize(rgb, pw as u32, ph as u32, image::imageops::FilterType::Triangle);
        let mut data = vec![0f32; 3 * pw * ph];
        for (i, p) in small.pixels().enumerate() {
            for ch in 0..3 {
                data[ch * pw * ph + i] = p[ch] as f32 / 255.0;
            }
        }
        let t = Tensor::from_vec(data, (1, 3, ph, pw), &Device::Cpu)?;
        let prob = self.predict(&t, color)?;
        Ok((prob, pw, ph))
    }
}
