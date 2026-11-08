//! Test du pipeline vidéo : `cargo run --release --example video_smoke -- <codelor_dir> <in.mp4> <out.mp4>`
use plimb_studio::ai::codelor::{resize_plane, Codelor};
use plimb_studio::video::{VideoReader, VideoWriter};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let model = Codelor::load(&PathBuf::from(&a[1]))?;
    let mut r = VideoReader::open(&PathBuf::from(&a[2]))?;
    println!("{}x{} {:.2} i/s {} images {:.2}s", r.info.width, r.info.height, r.info.fps, r.info.frames, r.info.seconds);
    let mut w = VideoWriter::create(&PathBuf::from(&a[3]), r.info.width, r.info.height, r.info.fps)?;
    let t = std::time::Instant::now();
    let mut n = 0;
    while let Some(f) = r.next_frame()? {
        let (p, pw, ph) = model.segment(&f, [0.2, 0.85, 0.3], 384)?;
        let prob = resize_plane(&p, pw, ph, f.width() as usize, f.height() as usize);
        let mut out = f.clone();
        for (i, px) in out.pixels_mut().enumerate() {
            let a = (0.5 + (0.5 - prob[i]) / 0.3).clamp(0.0, 1.0);
            for c in 0..3 { px[c] = (px[c] as f32 * a + 255.0 * (1.0 - a)) as u8; }
        }
        if n == 10 { out.save(PathBuf::from(&a[3]).with_extension("png"))?; }
        w.write(&out)?;
        n += 1;
    }
    w.finish()?;
    println!("{n} images en {:?}", t.elapsed());
    Ok(())
}
