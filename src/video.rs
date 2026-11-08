//! Lecture et écriture de vidéos MP4 / H.264 en Rust (démultiplexage `mp4`, codec OpenH264).
//!
//! Limites : seul le codec vidéo H.264 est pris en charge en entrée (pas le HEVC/H.265),
//! et la piste audio n'est pas conservée.

use anyhow::{anyhow, bail, Context, Result};
use image::RgbImage;
use mp4::{AvcConfig, MediaConfig, Mp4Config, Mp4Reader, Mp4Sample, Mp4Writer, TrackConfig, TrackType};
use openh264::decoder::{Decoder, DecoderConfig, Flush};
use openh264::encoder::{BitRate, Encoder, EncoderConfig, FrameRate};
use openh264::formats::{RgbSliceU8, YUVBuffer, YUVSource};
use openh264::OpenH264API;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub frames: u32,
    pub seconds: f64,
}

pub struct VideoReader {
    mp4: Mp4Reader<BufReader<File>>,
    track: u32,
    next_sample: u32,
    count: u32,
    decoder: Decoder,
    nal_len: usize,
    header: Option<Vec<u8>>,
    /// Rotation à appliquer (0, 90, 180, 270), lue dans la matrice de la piste.
    rotation: u32,
    queue: VecDeque<RgbImage>,
    flushed: bool,
    pub info: VideoInfo,
}

fn annexb_push(out: &mut Vec<u8>, nal: &[u8]) {
    out.extend_from_slice(&[0, 0, 0, 1]);
    out.extend_from_slice(nal);
}

impl VideoReader {
    pub fn open(path: &Path) -> Result<Self> {
        let f = File::open(path).context("ouverture de la vidéo")?;
        let size = f.metadata()?.len();
        let mp4 = Mp4Reader::read_header(BufReader::new(f), size).map_err(|e| anyhow!("fichier MP4 illisible : {e}"))?;
        let (track_id, t) = mp4
            .tracks()
            .iter()
            .find(|(_, t)| matches!(t.track_type(), Ok(TrackType::Video)))
            .ok_or_else(|| anyhow!("aucune piste vidéo"))?;
        let track = *track_id;
        let fourcc = t.box_type().map(|b| b.to_string()).unwrap_or_default();
        if t.trak.mdia.minf.stbl.stsd.avc1.is_none() {
            bail!(crate::tf!(
                "codec \"{fourcc}\" is not supported: only H.264 videos are accepted (turn off HEVC/H.265 in your camera settings)",
                "codec « {fourcc} » non pris en charge : seules les vidéos H.264 sont acceptées (désactivez le HEVC/H.265 dans l'appareil photo)"
            ));
        }
        let avc1 = t.trak.mdia.minf.stbl.stsd.avc1.as_ref().unwrap();
        let nal_len = (avc1.avcc.length_size_minus_one & 3) as usize + 1;
        let mut header = Vec::new();
        for sps in &avc1.avcc.sequence_parameter_sets {
            annexb_push(&mut header, &sps.bytes);
        }
        for pps in &avc1.avcc.picture_parameter_sets {
            annexb_push(&mut header, &pps.bytes);
        }
        let m = &t.trak.tkhd.matrix;
        let angle = (m.b as f64).atan2(m.a as f64).to_degrees().round() as i32;
        let rotation = (((angle % 360) + 360) % 360) as u32;
        let rotation = [0, 90, 180, 270].into_iter().min_by_key(|r| (*r as i32 - rotation as i32).abs()).unwrap_or(0);
        let (mut w, mut h) = (t.width() as u32, t.height() as u32);
        if rotation == 90 || rotation == 270 {
            std::mem::swap(&mut w, &mut h);
        }
        let count = t.sample_count();
        let seconds = t.duration().as_secs_f64();
        let fps = if seconds > 0.0 { count as f64 / seconds } else { t.frame_rate().max(1.0) };
        Ok(Self {
            mp4,
            track,
            next_sample: 1,
            count,
            decoder: Decoder::with_api_config(OpenH264API::from_source(), DecoderConfig::new().flush_after_decode(Flush::NoFlush))
                .map_err(|e| anyhow!("décodeur H.264 : {e}"))?,
            nal_len,
            header: Some(header),
            rotation,
            queue: VecDeque::new(),
            flushed: false,
            info: VideoInfo { width: w, height: h, fps, frames: count, seconds },
        })
    }

}

fn yuv_to_rgb(rotation: u32, yuv: &openh264::decoder::DecodedYUV<'_>) -> RgbImage {
    let (w, h) = yuv.dimensions();
    let mut buf = vec![0u8; w * h * 3];
    yuv.write_rgb8(&mut buf);
    let img = RgbImage::from_raw(w as u32, h as u32, buf).unwrap_or_default();
    match rotation {
        90 => image::imageops::rotate90(&img),
        180 => image::imageops::rotate180(&img),
        270 => image::imageops::rotate270(&img),
        _ => img,
    }
}

impl VideoReader {
    /// Renvoie l'image suivante, ou `None` à la fin de la vidéo.
    pub fn next_frame(&mut self) -> Result<Option<RgbImage>> {
        loop {
            if let Some(f) = self.queue.pop_front() {
                return Ok(Some(f));
            }
            if self.next_sample > self.count {
                if !self.flushed {
                    self.flushed = true;
                    let frames: Vec<RgbImage> = match self.decoder.flush_remaining() {
                        Ok(v) => v.iter().map(|y| yuv_to_rgb(self.rotation, y)).collect(),
                        Err(_) => Vec::new(),
                    };
                    self.queue.extend(frames);
                    continue;
                }
                return Ok(None);
            }
            let id = self.next_sample;
            self.next_sample += 1;
            let Some(sample) = self.mp4.read_sample(self.track, id).map_err(|e| anyhow!("lecture : {e}"))? else {
                continue;
            };
            let mut packet = self.header.take().unwrap_or_default();
            let data = &sample.bytes[..];
            let mut i = 0;
            while i + self.nal_len <= data.len() {
                let mut n = 0usize;
                for b in &data[i..i + self.nal_len] {
                    n = (n << 8) | *b as usize;
                }
                i += self.nal_len;
                if n == 0 || i + n > data.len() {
                    break;
                }
                annexb_push(&mut packet, &data[i..i + n]);
                i += n;
            }
            let frame = match self.decoder.decode(&packet) {
                Ok(Some(yuv)) => Some(yuv_to_rgb(self.rotation, &yuv)),
                Ok(None) => None,
                Err(e) => {
                    log::warn!("image {id} non décodée : {e}");
                    if std::env::var("PLIMB_VDEBUG").is_ok() { eprintln!("image {id} : {e}"); }
                    None
                }
            };
            if let Some(f) = frame {
                return Ok(Some(f));
            }
        }
    }
}

/// Lit uniquement la première image (aperçu).
pub fn first_frame(path: &Path) -> Result<(RgbImage, VideoInfo)> {
    let mut r = VideoReader::open(path)?;
    let f = r.next_frame()?.ok_or_else(|| anyhow!("aucune image décodable"))?;
    Ok((f, r.info))
}

pub struct VideoWriter {
    mp4: Option<Mp4Writer<BufWriter<File>>>,
    file: Option<File>,
    encoder: Encoder,
    width: usize,
    height: usize,
    frame: u64,
    frame_dur: u32,
}

const TIMESCALE: u32 = 90_000;

fn split_nals(bitstream: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= bitstream.len() {
        if bitstream[i] == 0 && bitstream[i + 1] == 0 && bitstream[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::new();
    for (k, s) in starts.iter().enumerate() {
        let mut end = starts.get(k + 1).map(|n| n - 3).unwrap_or(bitstream.len());
        while end > *s && bitstream[end - 1] == 0 {
            end -= 1;
        }
        out.push(&bitstream[*s..end]);
    }
    out
}

impl VideoWriter {
    /// Crée un MP4 H.264. Les dimensions sont arrondies au pair inférieur.
    pub fn create(path: &Path, width: u32, height: u32, fps: f64) -> Result<Self> {
        let (w, h) = ((width & !1).max(2) as usize, (height & !1).max(2) as usize);
        let fps = fps.clamp(1.0, 120.0);
        let bitrate = ((w * h) as f64 * fps * 0.12).clamp(800_000.0, 16_000_000.0) as u32;
        let cfg = EncoderConfig::new().skip_frames(false).bitrate(BitRate::from_bps(bitrate)).max_frame_rate(FrameRate::from_hz(fps as f32));
        let encoder = Encoder::with_api_config(OpenH264API::from_source(), cfg).map_err(|e| anyhow!("encodeur H.264 : {e}"))?;
        let file = File::create(path).context("création de la vidéo")?;
        Ok(Self {
            mp4: None,
            file: Some(file),
            encoder,
            width: w,
            height: h,
            frame: 0,
            frame_dur: (TIMESCALE as f64 / fps).round() as u32,
        })
    }

    pub fn write(&mut self, img: &RgbImage) -> Result<()> {
        let img = if img.width() as usize != self.width || img.height() as usize != self.height {
            image::imageops::crop_imm(img, 0, 0, self.width as u32, self.height as u32).to_image()
        } else {
            img.clone()
        };
        let yuv = YUVBuffer::from_rgb8_source(RgbSliceU8::new(img.as_raw(), (self.width, self.height)));
        let bs = self.encoder.encode(&yuv).map_err(|e| anyhow!("encodage : {e}"))?.to_vec();
        let nals = split_nals(&bs);
        let (mut sps, mut pps, mut sample, mut sync) = (None, None, Vec::new(), false);
        for n in nals {
            if n.is_empty() {
                continue;
            }
            match n[0] & 0x1f {
                7 => sps = Some(n.to_vec()),
                8 => pps = Some(n.to_vec()),
                t => {
                    if t == 5 {
                        sync = true;
                    }
                    sample.extend_from_slice(&(n.len() as u32).to_be_bytes());
                    sample.extend_from_slice(n);
                }
            }
        }
        if self.mp4.is_none() {
            let (Some(sps), Some(pps)) = (sps, pps) else {
                bail!("l'encodeur n'a pas produit de SPS/PPS");
            };
            let config = Mp4Config {
                major_brand: "isom".parse().unwrap(),
                minor_version: 512,
                compatible_brands: vec!["isom".parse().unwrap(), "iso2".parse().unwrap(), "avc1".parse().unwrap(), "mp41".parse().unwrap()],
                timescale: 1000,
            };
            let file = self.file.take().ok_or_else(|| anyhow!("fichier déjà fermé"))?;
            let mut w = Mp4Writer::write_start(BufWriter::new(file), &config).map_err(|e| anyhow!("{e}"))?;
            w.add_track(&TrackConfig {
                track_type: TrackType::Video,
                timescale: TIMESCALE,
                language: "und".into(),
                media_conf: MediaConfig::AvcConfig(AvcConfig {
                    width: self.width as u16,
                    height: self.height as u16,
                    seq_param_set: sps,
                    pic_param_set: pps,
                }),
            })
            .map_err(|e| anyhow!("{e}"))?;
            self.mp4 = Some(w);
        }
        if sample.is_empty() {
            return Ok(()); // image sautée par le contrôle de débit
        }
        let start = self.frame * self.frame_dur as u64;
        self.frame += 1;
        self.mp4
            .as_mut()
            .unwrap()
            .write_sample(
                1,
                &Mp4Sample { start_time: start, duration: self.frame_dur, rendering_offset: 0, is_sync: sync, bytes: bytes::Bytes::from(sample) },
            )
            .map_err(|e| anyhow!("{e}"))?;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        let mut w = self.mp4.take().ok_or_else(|| anyhow!("aucune image écrite"))?;
        w.write_end().map_err(|e| anyhow!("{e}"))?;
        Ok(())
    }
}
