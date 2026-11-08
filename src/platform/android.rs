use jni::objects::{JObject, JValue};
use jni::JavaVM;
use std::path::PathBuf;
use std::sync::OnceLock;

static APP: OnceLock<slint::android::AndroidApp> = OnceLock::new();

/// À appeler depuis `android_main` : donne accès à l'Activity pour les appels Java.
/// (Le contexte global de `ndk_context` est l'Application, qui ne peut ni demander
/// de permissions ni lancer d'activité sans FLAG_ACTIVITY_NEW_TASK.)
pub fn set_android_app(app: slint::android::AndroidApp) {
    let _ = APP.set(app);
}

fn with_env<R>(f: impl FnOnce(&mut jni::JNIEnv, &JObject) -> jni::errors::Result<R>) -> Option<R> {
    let app = APP.get()?;
    let vm = unsafe { JavaVM::from_raw(app.vm_as_ptr().cast()) }.ok()?;
    let mut env = vm.attach_current_thread().ok()?;
    let activity = unsafe { JObject::from_raw(app.activity_as_ptr().cast()) };
    let r = f(&mut env, &activity);
    if env.exception_check().unwrap_or(false) {
        let _ = env.exception_describe();
        let _ = env.exception_clear();
    }
    match r {
        Ok(v) => Some(v),
        Err(e) => {
            log::warn!("JNI : {e}");
            None
        }
    }
}

fn sdk_int() -> i32 {
    with_env(|env, _| env.get_static_field("android/os/Build$VERSION", "SDK_INT", "I")?.i()).unwrap_or(30)
}

/// Ouvre un lien dans le navigateur. Renvoie `false` si Android a refusé.
pub fn open_url(url: &str) -> bool {
    let url = url.to_string();
    with_env(|env, activity| {
        let jurl = env.new_string(&url)?;
        let uri = env
            .call_static_method("android/net/Uri", "parse", "(Ljava/lang/String;)Landroid/net/Uri;", &[JValue::Object(&jurl)])?
            .l()?;
        let action = env.new_string("android.intent.action.VIEW")?;
        let intent = env.new_object(
            "android/content/Intent",
            "(Ljava/lang/String;Landroid/net/Uri;)V",
            &[JValue::Object(&action), JValue::Object(&uri)],
        )?;
        // FLAG_ACTIVITY_NEW_TASK
        env.call_method(&intent, "addFlags", "(I)Landroid/content/Intent;", &[JValue::Int(0x1000_0000)])?;
        env.call_method(activity, "startActivity", "(Landroid/content/Intent;)V", &[JValue::Object(&intent)])?;
        Ok(())
    })
    .is_some()
}

pub fn open_folder(_path: &std::path::Path) {}

pub fn pick_media() -> Option<PathBuf> {
    None
}

/// Android : pas de sélecteur natif accessible depuis une NativeActivity, on utilise la galerie intégrée.
pub fn has_native_picker() -> bool {
    false
}

fn external_root() -> PathBuf {
    PathBuf::from("/storage/emulated/0")
}

pub fn gallery_roots() -> Vec<PathBuf> {
    let r = external_root();
    ["DCIM", "Pictures", "Movies", "Download", "Screenshots"].iter().map(|d| r.join(d)).collect()
}

fn media_permissions() -> &'static [&'static str] {
    if sdk_int() >= 33 {
        &["android.permission.READ_MEDIA_IMAGES", "android.permission.READ_MEDIA_VIDEO"]
    } else {
        &["android.permission.READ_EXTERNAL_STORAGE"]
    }
}

/// Vrai si l'application peut lire les photos et vidéos.
pub fn has_media_permission() -> bool {
    media_permissions().iter().all(|p| has_permission(p))
}

fn has_permission(p: &str) -> bool {
    with_env(|env, activity| {
        let jp = env.new_string(p)?;
        env.call_method(activity, "checkSelfPermission", "(Ljava/lang/String;)I", &[JValue::Object(&jp)])?.i()
    })
    .map(|r| r == 0)
    .unwrap_or(false)
}

/// Demande l'accès aux photos et vidéos. Renvoie `true` si l'accès est déjà accordé.
pub fn request_media_permission() -> bool {
    if has_media_permission() {
        return true;
    }
    let perms = media_permissions();
    let r = with_env(|env, activity| {
        let arr = env.new_object_array(perms.len() as i32, "java/lang/String", JObject::null())?;
        for (i, p) in perms.iter().enumerate() {
            let jp = env.new_string(p)?;
            env.set_object_array_element(&arr, i as i32, jp)?;
        }
        env.call_method(activity, "requestPermissions", "([Ljava/lang/String;I)V", &[JValue::Object(&arr), JValue::Int(42)])?;
        Ok(())
    });
    if r.is_none() {
        log::warn!("requestPermissions a échoué");
    }
    false
}

fn export_dir(sub: &str) -> anyhow::Result<PathBuf> {
    let dir = external_root().join(sub).join("Plimb");
    if std::fs::create_dir_all(&dir).is_ok() {
        return Ok(dir);
    }
    let dir = super::data_dir().join("exports");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Copie une vidéo produite dans Movies/Plimb.
pub fn save_video(src: &std::path::Path, suggested: &str) -> anyhow::Result<Option<PathBuf>> {
    let mut dest = export_dir("Movies")?.join(suggested);
    if std::fs::copy(src, &dest).is_err() {
        let dir = super::data_dir().join("exports");
        std::fs::create_dir_all(&dir)?;
        dest = dir.join(suggested);
        std::fs::copy(src, &dest)?;
    }
    Ok(Some(dest))
}

pub fn save_png(img: &image::RgbaImage, suggested: &str) -> anyhow::Result<Option<PathBuf>> {
    let mut dir = external_root().join("Pictures").join("Plimb");
    if std::fs::create_dir_all(&dir).is_err() {
        dir = super::data_dir().join("exports");
        std::fs::create_dir_all(&dir)?;
    }
    let mut path = dir.join(suggested);
    if let Err(e) = img.save_with_format(&path, image::ImageFormat::Png) {
        log::warn!("écriture dans Pictures impossible ({e}), repli dans le stockage de l'app");
        dir = super::data_dir().join("exports");
        std::fs::create_dir_all(&dir)?;
        path = dir.join(suggested);
        img.save_with_format(&path, image::ImageFormat::Png)?;
    }
    Ok(Some(path))
}

pub fn install_update(_file: &std::path::Path) -> anyhow::Result<()> {
    anyhow::bail!("non supporté sur Android")
}

pub fn release_asset_name() -> Option<&'static str> {
    Some("plimb-studio-android-arm64.apk")
}
