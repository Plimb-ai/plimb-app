use jni::objects::{JObject, JValue};
use jni::JavaVM;
use std::path::PathBuf;

fn with_env<R>(f: impl FnOnce(&mut jni::JNIEnv, &JObject) -> jni::errors::Result<R>) -> Option<R> {
    let ctx = ndk_context::android_context();
    let vm = unsafe { JavaVM::from_raw(ctx.vm().cast()) }.ok()?;
    let mut env = vm.attach_current_thread().ok()?;
    let activity = unsafe { JObject::from_raw(ctx.context().cast()) };
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

pub fn open_url(url: &str) {
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
        env.call_method(activity, "startActivity", "(Landroid/content/Intent;)V", &[JValue::Object(&intent)])?;
        Ok(())
    });
}

pub fn open_folder(_path: &std::path::Path) {}

pub fn pick_image() -> Option<PathBuf> {
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
    ["DCIM", "Pictures", "Download", "Screenshots"].iter().map(|d| r.join(d)).collect()
}

fn media_permission() -> &'static str {
    if sdk_int() >= 33 {
        "android.permission.READ_MEDIA_IMAGES"
    } else {
        "android.permission.READ_EXTERNAL_STORAGE"
    }
}

fn has_permission(p: &str) -> bool {
    with_env(|env, activity| {
        let jp = env.new_string(p)?;
        env.call_method(activity, "checkSelfPermission", "(Ljava/lang/String;)I", &[JValue::Object(&jp)])?.i()
    })
    .map(|r| r == 0)
    .unwrap_or(false)
}

/// Demande l'accès aux photos. Renvoie `true` si l'accès est déjà accordé.
pub fn request_media_permission() -> bool {
    let p = media_permission();
    if has_permission(p) {
        return true;
    }
    with_env(|env, activity| {
        let arr = env.new_object_array(1, "java/lang/String", JObject::null())?;
        let jp = env.new_string(p)?;
        env.set_object_array_element(&arr, 0, jp)?;
        env.call_method(activity, "requestPermissions", "([Ljava/lang/String;I)V", &[JValue::Object(&arr), JValue::Int(42)])?;
        Ok(())
    });
    false
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
