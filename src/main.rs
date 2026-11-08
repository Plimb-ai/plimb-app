// Pas de console sous Windows en mode release.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

#[cfg(not(target_os = "android"))]
fn main() -> Result<(), slint::PlatformError> {
    plimb_studio::run_desktop()
}

#[cfg(target_os = "android")]
fn main() {}
