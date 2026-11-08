fn main() {
    let config = slint_build::CompilerConfiguration::new()
        .with_style("material".into())
        .with_bundled_translations("lang");
    slint_build::compile_with_config("ui/app.slint", config).expect("compilation de l'interface Slint");

    // Icône de l'exécutable Windows
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        #[cfg(windows)]
        {
            let mut res = winresource::WindowsResource::new();
            res.set_icon("assets/icon.ico");
            res.set("ProductName", "Plimb Studio");
            res.set("FileDescription", "Plimb Studio");
            if let Err(e) = res.compile() {
                println!("cargo:warning=icône Windows non intégrée : {e}");
            }
        }
    }
}
