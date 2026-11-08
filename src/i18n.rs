//! Langue de l'application (français / anglais).
//!
//! L'interface Slint utilise `@tr()` (source anglaise, traduction dans `lang/fr`).
//! Les textes produits côté Rust passent par [`t`] et la macro [`tf!`].

use std::sync::atomic::{AtomicBool, Ordering};

static FRENCH: AtomicBool = AtomicBool::new(true);

pub fn fr() -> bool {
    FRENCH.load(Ordering::Relaxed)
}

/// Langue du système : vrai si elle commence par « fr ».
pub fn system_is_french() -> bool {
    sys_locale::get_locale().map(|l| l.to_lowercase().starts_with("fr")).unwrap_or(false)
}

/// Applique le réglage (0 = langue du système, 1 = français, 2 = anglais).
pub fn apply(setting: i32) {
    let french = match setting {
        1 => true,
        2 => false,
        _ => system_is_french(),
    };
    FRENCH.store(french, Ordering::Relaxed);
    if let Err(e) = slint::select_bundled_translation(if french { "fr" } else { "" }) {
        log::warn!("traduction : {e}");
    }
}

/// Choisit le texte selon la langue active.
pub fn t(en: &'static str, fr_text: &'static str) -> &'static str {
    if fr() {
        fr_text
    } else {
        en
    }
}

/// `tf!("English {x}", "Français {x}", …)` : `format!` dans la langue active.
#[macro_export]
macro_rules! tf {
    ($en:literal, $fr:literal $(, $arg:expr)* $(,)?) => {
        if $crate::i18n::fr() { format!($fr $(, $arg)*) } else { format!($en $(, $arg)*) }
    };
}
