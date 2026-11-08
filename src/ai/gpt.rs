//! Transformer décodeur style GPT, commun aux deux modèles Gladios.
//!
//! - `GptFlavor::Hf` : `GPT2LMHeadModel` Hugging Face (safetensors, Conv1D, biais, gelu tanh)
//! - `GptFlavor::Gladios` : GPT "from scratch" (checkpoint PyTorch `.pt`, Linear sans biais, gelu erf)

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GptFlavor {
    Hf,
    Gladios,
}

struct Linear {
    /// Poids sous la forme [in, out]
    w: Tensor,
    b: Option<Tensor>,
}

impl Linear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = x.broadcast_matmul(&self.w)?;
        match &self.b {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

struct LayerNorm {
    w: Tensor,
    b: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let xc = x.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
        let xn = xc.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        xn.broadcast_mul(&self.w)?.broadcast_add(&self.b)
    }
}

struct Block {
    ln1: LayerNorm,
    qkv: Linear,
    proj: Linear,
    ln2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

pub struct Gpt {
    flavor: GptFlavor,
    wte: Tensor,
    wpe: Tensor,
    head_t: Tensor,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    n_head: usize,
    pub n_ctx: usize,
    pub vocab: usize,
    cache: Vec<Option<(Tensor, Tensor)>>,
    cache_len: usize,
}

fn take(map: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    map.remove(name)
        .ok_or_else(|| anyhow!("tenseur manquant : {name}"))?
        .to_dtype(DType::F32)
        .map_err(Into::into)
}

impl Gpt {
    pub fn load_hf(dir: &Path) -> Result<Self> {
        let cfg: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).context("config.json")?)?;
        let n_layer = cfg["n_layer"].as_u64().unwrap_or(12) as usize;
        let n_head = cfg["n_head"].as_u64().unwrap_or(12) as usize;
        let eps = cfg["layer_norm_epsilon"].as_f64().unwrap_or(1e-5);
        let mut map: HashMap<String, Tensor> =
            candle_core::safetensors::load(dir.join("model.safetensors"), &Device::Cpu)?;
        // Certains exports n'ont pas le préfixe "transformer."
        let keys: Vec<String> = map.keys().cloned().collect();
        for k in keys {
            if !k.starts_with("transformer.") && !k.starts_with("lm_head") {
                let t = map.remove(&k).unwrap();
                map.insert(format!("transformer.{k}"), t);
            }
        }
        let ln = |map: &mut HashMap<String, Tensor>, p: &str| -> Result<LayerNorm> {
            Ok(LayerNorm { w: take(map, &format!("{p}.weight"))?, b: take(map, &format!("{p}.bias"))?, eps })
        };
        let conv1d = |map: &mut HashMap<String, Tensor>, p: &str| -> Result<Linear> {
            Ok(Linear { w: take(map, &format!("{p}.weight"))?, b: Some(take(map, &format!("{p}.bias"))?) })
        };
        let mut blocks = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = format!("transformer.h.{i}");
            blocks.push(Block {
                ln1: ln(&mut map, &format!("{p}.ln_1"))?,
                qkv: conv1d(&mut map, &format!("{p}.attn.c_attn"))?,
                proj: conv1d(&mut map, &format!("{p}.attn.c_proj"))?,
                ln2: ln(&mut map, &format!("{p}.ln_2"))?,
                fc1: conv1d(&mut map, &format!("{p}.mlp.c_fc"))?,
                fc2: conv1d(&mut map, &format!("{p}.mlp.c_proj"))?,
            });
        }
        let wte = take(&mut map, "transformer.wte.weight")?;
        let wpe = take(&mut map, "transformer.wpe.weight")?;
        let head_t = match map.remove("lm_head.weight") {
            Some(h) => h.to_dtype(DType::F32)?.t()?.contiguous()?,
            None => wte.t()?.contiguous()?,
        };
        let ln_f = ln(&mut map, "transformer.ln_f")?;
        Ok(Self::assemble(GptFlavor::Hf, wte, wpe, head_t, blocks, ln_f, n_head))
    }

    pub fn load_gladios(weights: &Path, n_head: usize) -> Result<Self> {
        let tensors = candle_core::pickle::read_all_with_key(weights, Some("model"))
            .or_else(|_| candle_core::pickle::read_all_with_key(weights, None))
            .context("lecture du checkpoint PyTorch")?;
        let mut map: HashMap<String, Tensor> = tensors
            .into_iter()
            .map(|(k, v)| (k.trim_start_matches("_orig_mod.").trim_start_matches("module.").to_string(), v))
            .collect();
        let eps = 1e-5;
        let n_layer = (0..)
            .take_while(|i| map.contains_key(&format!("blocks.{i}.ln1.weight")))
            .count();
        if n_layer == 0 {
            return Err(anyhow!("aucun bloc trouvé dans le checkpoint"));
        }
        let ln = |map: &mut HashMap<String, Tensor>, p: &str| -> Result<LayerNorm> {
            Ok(LayerNorm { w: take(map, &format!("{p}.weight"))?, b: take(map, &format!("{p}.bias"))?, eps })
        };
        let linear = |map: &mut HashMap<String, Tensor>, p: &str| -> Result<Linear> {
            let w = take(map, &format!("{p}.weight"))?.t()?.contiguous()?;
            let b = match map.remove(&format!("{p}.bias")) {
                Some(b) => Some(b.to_dtype(DType::F32)?),
                None => None,
            };
            Ok(Linear { w, b })
        };
        let mut blocks = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = format!("blocks.{i}");
            blocks.push(Block {
                ln1: ln(&mut map, &format!("{p}.ln1"))?,
                qkv: linear(&mut map, &format!("{p}.attn.qkv"))?,
                proj: linear(&mut map, &format!("{p}.attn.proj"))?,
                ln2: ln(&mut map, &format!("{p}.ln2"))?,
                fc1: linear(&mut map, &format!("{p}.mlp.fc1"))?,
                fc2: linear(&mut map, &format!("{p}.mlp.fc2"))?,
            });
            map.remove(&format!("{p}.attn.mask"));
        }
        let wte = take(&mut map, "tok_emb.weight")?;
        let wpe = take(&mut map, "pos_emb.weight")?;
        let head_t = match map.remove("head.weight") {
            Some(h) => h.to_dtype(DType::F32)?.t()?.contiguous()?,
            None => wte.t()?.contiguous()?,
        };
        let ln_f = ln(&mut map, "ln_f")?;
        Ok(Self::assemble(GptFlavor::Gladios, wte, wpe, head_t, blocks, ln_f, n_head))
    }

    fn assemble(
        flavor: GptFlavor,
        wte: Tensor,
        wpe: Tensor,
        head_t: Tensor,
        blocks: Vec<Block>,
        ln_f: LayerNorm,
        n_head: usize,
    ) -> Self {
        let n_ctx = wpe.dim(0).unwrap_or(256);
        let vocab = wte.dim(0).unwrap_or(0);
        let n = blocks.len();
        Self { flavor, wte, wpe, head_t, blocks, ln_f, n_head, n_ctx, vocab, cache: vec![None; n], cache_len: 0 }
    }

    pub fn clear_cache(&mut self) {
        self.cache = vec![None; self.blocks.len()];
        self.cache_len = 0;
    }

    pub fn cached_len(&self) -> usize {
        self.cache_len
    }

    /// Traite `ids` (à la suite du cache) et renvoie les logits du dernier token.
    pub fn forward(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        let t = ids.len();
        let start = self.cache_len;
        if start + t > self.n_ctx {
            return Err(anyhow!("contexte dépassé"));
        }
        let idx = Tensor::new(ids, &Device::Cpu)?;
        let tok = self.wte.index_select(&idx, 0)?;
        let pos = self.wpe.narrow(0, start, t)?;
        let mut x = (tok + pos)?.unsqueeze(0)?; // [1, T, C]
        let c = x.dim(2)?;
        let hd = c / self.n_head;
        let scale = 1.0 / (hd as f64).sqrt();
        let total = start + t;
        // Masque causal [T, total]
        let mask: Vec<f32> = (0..t)
            .flat_map(|i| (0..total).map(move |j| if j > start + i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from_vec(mask, (t, total), &Device::Cpu)?;
        let exact_gelu = self.flavor == GptFlavor::Gladios;

        for (li, blk) in self.blocks.iter().enumerate() {
            let h = blk.ln1.forward(&x)?;
            let qkv = blk.qkv.forward(&h)?; // [1,T,3C]
            let q = qkv.narrow(2, 0, c)?.reshape((1, t, self.n_head, hd))?.transpose(1, 2)?.contiguous()?;
            let k = qkv.narrow(2, c, c)?.reshape((1, t, self.n_head, hd))?.transpose(1, 2)?.contiguous()?;
            let v = qkv.narrow(2, 2 * c, c)?.reshape((1, t, self.n_head, hd))?.transpose(1, 2)?.contiguous()?;
            let (k, v) = match &self.cache[li] {
                Some((pk, pv)) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?),
                None => (k, v),
            };
            self.cache[li] = Some((k.clone(), v.clone()));
            let att = (q.matmul(&k.t()?)? * scale)?.broadcast_add(&mask)?;
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            let y = att.matmul(&v)?.transpose(1, 2)?.reshape((1, t, c))?;
            x = (x + blk.proj.forward(&y)?)?;
            let h = blk.ln2.forward(&x)?;
            let h = blk.fc1.forward(&h)?;
            let h = if exact_gelu { h.gelu_erf()? } else { h.gelu()? };
            x = (x + blk.fc2.forward(&h)?)?;
        }
        self.cache_len = total;
        let last = x.i((0, t - 1))?.unsqueeze(0)?;
        let last = self.ln_f.forward(&last)?;
        let logits = last.matmul(&self.head_t)?.squeeze(0)?;
        Ok(logits.to_vec1::<f32>()?)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

pub fn sample(logits: &mut [f32], history: &[u32], p: &SamplingParams, rng: &mut impl rand::Rng) -> u32 {
    if p.repetition_penalty > 1.0 {
        for &id in history.iter().rev().take(64) {
            if let Some(l) = logits.get_mut(id as usize) {
                *l = if *l > 0.0 { *l / p.repetition_penalty } else { *l * p.repetition_penalty };
            }
        }
    }
    if p.temperature <= 0.01 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    let mut idx: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, l)| (i as u32, l / p.temperature)).collect();
    idx.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    let k = if p.top_k == 0 { idx.len() } else { p.top_k.min(idx.len()) };
    idx.truncate(k);
    let max = idx[0].1;
    let mut probs: Vec<f32> = idx.iter().map(|(_, l)| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|x| *x /= sum);
    if p.top_p < 1.0 {
        let mut acc = 0.0;
        let mut cut = probs.len();
        for (i, pr) in probs.iter().enumerate() {
            acc += pr;
            if acc >= p.top_p {
                cut = i + 1;
                break;
            }
        }
        probs.truncate(cut);
        let s: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|x| *x /= s);
    }
    let r: f32 = rng.random();
    let mut acc = 0.0;
    for (i, pr) in probs.iter().enumerate() {
        acc += pr;
        if r <= acc {
            return idx[i].0;
        }
    }
    idx[probs.len() - 1].0
}
