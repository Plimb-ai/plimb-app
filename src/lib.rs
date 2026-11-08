//! Plimb Studio : application d'IA locale pour les modèles Plimb.

pub mod ai;
mod app;
pub mod catalog;
mod net;
pub mod platform;
mod settings;
mod update;

/// Point d'entrée bureau (Windows / Linux / macOS).
#[cfg(not(target_os = "android"))]
pub fn run_desktop() -> Result<(), slint::PlatformError> {
    platform::set_data_dir(platform::default_data_dir());
    app::run()
}

/// Point d'entrée Android (NativeActivity).
#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(android_app: slint::android::AndroidApp) {
    android_logger::init_once(android_logger::Config::default().with_max_level(log::LevelFilter::Info).with_tag("PlimbStudio"));
    let dir = android_app
        .internal_data_path()
        .unwrap_or_else(|| std::path::PathBuf::from("/data/data/ai.plimb.studio/files"));
    platform::set_data_dir(dir);
    if let Err(e) = slint::android::init(android_app) {
        log::error!("initialisation Slint : {e}");
        return;
    }
    if let Err(e) = app::run() {
        log::error!("erreur : {e}");
    }
}
