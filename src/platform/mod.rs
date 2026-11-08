//! Fonctions dépendantes de la plateforme (bureau / Android).

use std::path::PathBuf;
use std::sync::OnceLock;

#[cfg(target_os = "android")]
pub mod android;
#[cfg(not(target_os = "android"))]
mod desktop;

#[cfg(target_os = "android")]
pub use android::*;
#[cfg(not(target_os = "android"))]
pub use desktop::*;

static DATA_DIR: OnceLock<PathBuf> = OnceLock::new();

pub fn set_data_dir(p: PathBuf) {
    let _ = std::fs::create_dir_all(&p);
    let _ = DATA_DIR.set(p);
}

/// Dossier racine des données de l'application (modèles, réglages…).
pub fn data_dir() -> PathBuf {
    DATA_DIR.get().cloned().unwrap_or_else(|| PathBuf::from("."))
}

pub fn models_dir() -> PathBuf {
    data_dir().join("models")
}

pub fn is_mobile() -> bool {
    cfg!(target_os = "android")
}

pub const IMAGE_EXT: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp"];
pub const VIDEO_EXT: &[&str] = &["mp4", "m4v", "mov"];

pub fn is_video(p: &std::path::Path) -> bool {
    p.extension().map(|e| VIDEO_EXT.contains(&e.to_string_lossy().to_lowercase().as_str())).unwrap_or(false)
}

/// Liste les images et vidéos présentes dans les dossiers courants (galerie intégrée).
pub fn list_images(roots: &[PathBuf]) -> Vec<PathBuf> {
    fn walk(dir: &std::path::Path, depth: usize, out: &mut Vec<(std::time::SystemTime, PathBuf)>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().to_lowercase();
            if name.starts_with('.') {
                continue;
            }
            if p.is_dir() {
                if depth > 0 {
                    walk(&p, depth - 1, out);
                }
            } else if IMAGE_EXT.iter().chain(VIDEO_EXT).any(|x| name.ends_with(&format!(".{x}"))) {
                let t = e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
                out.push((t, p));
            }
        }
    }
    let mut out = Vec::new();
    for r in roots {
        walk(r, 2, &mut out);
    }
    out.sort_by_key(|e| std::cmp::Reverse(e.0));
    out.dedup_by(|a, b| a.1 == b.1);
    out.into_iter().take(120).map(|(_, p)| p).collect()
}
