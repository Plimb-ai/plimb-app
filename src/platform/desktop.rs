use std::path::PathBuf;

pub fn default_data_dir() -> PathBuf {
    dirs::data_dir().unwrap_or_else(|| PathBuf::from(".")).join("PlimbStudio")
}

pub fn open_url(url: &str) {
    if let Err(e) = open::that_detached(url) {
        log::warn!("impossible d'ouvrir {url} : {e}");
    }
}

pub fn open_folder(path: &std::path::Path) {
    let _ = std::fs::create_dir_all(path);
    let _ = open::that_detached(path);
}

/// Sélecteur de fichier natif. `None` si annulé.
pub fn pick_media() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .set_title("Choisir une image ou une vidéo")
        .add_filter("Images et vidéos", &["png", "jpg", "jpeg", "webp", "bmp", "mp4", "m4v", "mov"])
        .pick_file()
}

/// Le bureau utilise le sélecteur natif : pas de galerie intégrée.
pub fn has_native_picker() -> bool {
    true
}

pub fn gallery_roots() -> Vec<PathBuf> {
    [dirs::picture_dir(), dirs::download_dir(), dirs::desktop_dir()].into_iter().flatten().collect()
}

pub fn request_media_permission() -> bool {
    true
}

pub fn has_media_permission() -> bool {
    true
}

/// Enregistre une vidéo produite ; renvoie le chemin choisi.
pub fn save_video(src: &std::path::Path, suggested: &str) -> anyhow::Result<Option<PathBuf>> {
    let Some(path) = rfd::FileDialog::new()
        .set_title("Enregistrer la vidéo")
        .set_file_name(suggested)
        .add_filter("Vidéo MP4", &["mp4"])
        .save_file()
    else {
        return Ok(None);
    };
    let path = if path.extension().is_none() { path.with_extension("mp4") } else { path };
    std::fs::copy(src, &path)?;
    Ok(Some(path))
}

/// Enregistre une image PNG ; renvoie le chemin choisi.
pub fn save_png(img: &image::RgbaImage, suggested: &str) -> anyhow::Result<Option<PathBuf>> {
    let Some(path) = rfd::FileDialog::new()
        .set_title("Enregistrer l'image")
        .set_file_name(suggested)
        .add_filter("PNG", &["png"])
        .save_file()
    else {
        return Ok(None);
    };
    let path = if path.extension().is_none() { path.with_extension("png") } else { path };
    img.save_with_format(&path, image::ImageFormat::Png)?;
    Ok(Some(path))
}

/// Remplace l'exécutable courant par `new_exe` puis relance l'application.
pub fn install_update(new_exe: &std::path::Path) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(new_exe, std::fs::Permissions::from_mode(0o755))?;
    }
    self_replace::self_replace(new_exe)?;
    let _ = std::fs::remove_file(new_exe);
    let exe = std::env::current_exe()?;
    std::process::Command::new(exe).spawn()?;
    std::process::exit(0);
}

/// Nom de l'asset de release à télécharger pour cette plateforme.
pub fn release_asset_name() -> Option<&'static str> {
    if cfg!(all(windows, target_arch = "x86_64")) {
        Some("plimb-studio-windows-x86_64.exe")
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Some("plimb-studio-linux-x86_64")
    } else {
        None
    }
}
