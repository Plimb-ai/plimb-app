//! Test rapide des modèles : `cargo run --release --example smoke -- <dossier>`
use plimb_studio::ai::{codelor::Codelor, gpt::SamplingParams, text::TextModel};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let root = PathBuf::from(std::env::args().nth(1).expect("dossier"));
    let which = std::env::args().nth(2).unwrap_or_else(|| "all".into());
    let params = SamplingParams { temperature: 0.0, top_k: 50, top_p: 1.0, repetition_penalty: 1.0 };
    if which == "all" || which == "story" {
        let t = Instant::now();
        let mut m = TextModel::load_story(&root.join("story"))?;
        println!("story chargé en {:?}", t.elapsed());
        let t = Instant::now();
        let out = m.generate(&[], "Once upon a time", 60, params, |s| { print!("{s}"); true })?;
        println!("\n[{} chars en {:?}]", out.len(), t.elapsed());
    }
    if which == "all" || which == "chat" {
        let t = Instant::now();
        let mut m = TextModel::load_gladios_chat(&root.join("chat"), "gladios-0.3b-eng-prototype.pt", 16)?;
        println!("chat chargé en {:?}", t.elapsed());
        let t = Instant::now();
        let out = m.generate(&[], "What is the capital of France?", 60, params, |s| { print!("{s}"); true })?;
        println!("\n[{} chars en {:?}]", out.len(), t.elapsed());
    }
    if which == "all" || which == "codelor" {
        let m = Codelor::load(&root.join("codelor"))?;
        let img = image::ImageReader::open(root.join("codelor/example.png"))?.with_guessed_format()?.decode()?.to_rgb8();
        let t = Instant::now();
        let (p, pw, ph) = m.segment(&img, [0.2, 0.85, 0.3], 512)?;
        let prob = plimb_studio::ai::codelor::resize_plane(&p, pw, ph, img.width() as usize, img.height() as usize);
        println!("codelor {:?} en {:?}; moyenne masque {:.4}", img.dimensions(), t.elapsed(), prob.iter().sum::<f32>() / prob.len() as f32);
        let mask = image::GrayImage::from_fn(img.width(), img.height(), |x, y| image::Luma([(prob[(y * img.width() + x) as usize] * 255.0) as u8]));
        mask.save(root.join("codelor_mask.png"))?;
    }
    Ok(())
}
