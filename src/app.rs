//! Logique de l'application : relie l'interface Slint au catalogue, aux téléchargements,
//! au moteur d'inférence (thread dédié) et au système de mise à jour.

use crate::ai::codelor::{resize_plane, Codelor};
use crate::ai::gpt::SamplingParams;
use crate::ai::text::{TextModel, Turn};
use crate::catalog::{self, human_size, Engine, ModelInfo, Task};
use crate::settings::{GenSettings, Settings};
use crate::{net, platform, update};
use slint::{ComponentHandle, Model, ModelRc, SharedPixelBuffer, SharedString, VecModel, Weak};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::Arc;
use std::time::Instant;

slint::include_modules!();

// ------------------------------------------------------------------ état (thread UI)

#[derive(Clone)]
struct DlState {
    progress: f32,
    text: String,
    error: Option<String>,
    cancel: Arc<AtomicBool>,
}

struct ChromaState {
    source: Option<image::RgbImage>,
    preview: Option<image::RgbImage>,
    name: String,
    prob: Option<(Vec<f32>, usize, usize)>,
    preview_prob: Option<Vec<f32>>,
}

struct State {
    settings: Settings,
    downloads: HashMap<&'static str, DlState>,
    task: Task,
    active: Option<&'static ModelInfo>,
    loaded: Option<&'static str>,
    loading: Option<&'static str>,
    load_error: Option<String>,
    messages: Rc<VecModel<ChatMsg>>,
    turns: Vec<Turn>,
    stop: Arc<AtomicBool>,
    generating: bool,
    chroma: ChromaState,
    release: Option<update::Release>,
    worker: Sender<Job>,
    toast_id: u64,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> R {
    STATE.with(|s| f(s.borrow_mut().as_mut().expect("état initialisé")))
}

// ------------------------------------------------------------------ thread d'inférence

enum Job {
    Load(&'static ModelInfo),
    Unload(&'static str),
    Chat { history: Vec<Turn>, message: String, max: usize, params: SamplingParams, stop: Arc<AtomicBool> },
    Story { prompt: String, max: usize, params: SamplingParams, stop: Arc<AtomicBool> },
    Segment { image: Arc<image::RgbImage>, color: [f32; 3], max_side: u32 },
}

enum Loaded {
    Text(TextModel),
    Codelor(Codelor),
}

fn model_dir(m: &ModelInfo) -> PathBuf {
    platform::models_dir().join(m.id)
}

fn is_installed(m: &ModelInfo) -> bool {
    let d = model_dir(m);
    m.files.iter().all(|f| d.join(f.name).is_file())
}

fn load_model(m: &ModelInfo) -> anyhow::Result<Loaded> {
    let dir = model_dir(m);
    Ok(match m.engine {
        Engine::HfGpt2 => Loaded::Text(TextModel::load_story(&dir)?),
        Engine::GladiosPt { weights, n_head } => Loaded::Text(TextModel::load_gladios_chat(&dir, weights, n_head)?),
        Engine::Codelor => Loaded::Codelor(Codelor::load(&dir)?),
    })
}

fn spawn_worker(ui: Weak<AppWindow>) -> Sender<Job> {
    let (tx, rx) = channel::<Job>();
    std::thread::Builder::new()
        .name("inference".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            let mut current: Option<(&'static str, Loaded)> = None;
            for job in rx {
                match job {
                    Job::Unload(id) => {
                        if current.as_ref().is_some_and(|(cid, _)| *cid == id) {
                            current = None;
                        }
                    }
                    Job::Load(m) => {
                        if current.as_ref().is_some_and(|(cid, _)| *cid == m.id) {
                            let _ = ui.upgrade_in_event_loop(move |ui| on_loaded(&ui, m.id, Ok(0.0)));
                            continue;
                        }
                        current = None; // libère la mémoire avant de charger le suivant
                        let t = Instant::now();
                        let r = load_model(m);
                        let secs = t.elapsed().as_secs_f32();
                        let res = match r {
                            Ok(l) => {
                                current = Some((m.id, l));
                                Ok(secs)
                            }
                            Err(e) => Err(format!("{e:#}")),
                        };
                        let _ = ui.upgrade_in_event_loop(move |ui| on_loaded(&ui, m.id, res));
                    }
                    Job::Chat { history, message, max, params, stop } => {
                        let Some((_, Loaded::Text(model))) = current.as_mut() else {
                            let _ = ui.upgrade_in_event_loop(|ui| on_text_done(&ui, Err("modèle non chargé".into()), 0, 0.0));
                            continue;
                        };
                        let t = Instant::now();
                        let mut n = 0usize;
                        let weak = ui.clone();
                        let r = model.generate(&history, &message, max, params, |piece| {
                            n += 1;
                            let p = piece.to_string();
                            let _ = weak.upgrade_in_event_loop(move |ui| on_piece(&ui, &p));
                            !stop.load(Ordering::Relaxed)
                        });
                        let secs = t.elapsed().as_secs_f32();
                        let _ = ui.upgrade_in_event_loop(move |ui| on_text_done(&ui, r.map_err(|e| format!("{e:#}")), n, secs));
                    }
                    Job::Story { prompt, max, params, stop } => {
                        let Some((_, Loaded::Text(model))) = current.as_mut() else {
                            let _ = ui.upgrade_in_event_loop(|ui| on_text_done(&ui, Err("modèle non chargé".into()), 0, 0.0));
                            continue;
                        };
                        let t = Instant::now();
                        let mut n = 0usize;
                        let weak = ui.clone();
                        let r = model.generate(&[], &prompt, max, params, |piece| {
                            n += 1;
                            let p = piece.to_string();
                            let _ = weak.upgrade_in_event_loop(move |ui| on_piece(&ui, &p));
                            !stop.load(Ordering::Relaxed)
                        });
                        let secs = t.elapsed().as_secs_f32();
                        let _ = ui.upgrade_in_event_loop(move |ui| on_text_done(&ui, r.map_err(|e| format!("{e:#}")), n, secs));
                    }
                    Job::Segment { image, color, max_side } => {
                        let r = match current.as_ref() {
                            Some((_, Loaded::Codelor(m))) => {
                                let t = Instant::now();
                                m.segment(&image, color, max_side).map(|p| (p, t.elapsed().as_secs_f32())).map_err(|e| format!("{e:#}"))
                            }
                            _ => Err("modèle non chargé".into()),
                        };
                        let _ = ui.upgrade_in_event_loop(move |ui| on_segmented(&ui, r));
                    }
                }
            }
        })
        .expect("thread d'inférence");
    tx
}

// ------------------------------------------------------------------ conversions vers l'UI

fn model_data(m: &'static ModelInfo, st: &State) -> ModelData {
    let installed = is_installed(m);
    let (state, progress, text) = match st.downloads.get(m.id) {
        Some(d) if d.error.is_some() => (3, 0.0, format!("Échec du téléchargement : {}", d.error.clone().unwrap_or_default())),
        Some(d) => (1, d.progress, d.text.clone()),
        None if installed => (2, 1.0, String::new()),
        None => (0, 0.0, String::new()),
    };
    ModelData {
        id: m.id.into(),
        name: m.name.into(),
        repo_url: m.url().into(),
        task_id: m.task.id().into(),
        task_title: m.task.title().into(),
        description: m.description.into(),
        size: human_size(m.total_size()).into(),
        details: m.details.join(" · ").into(),
        state,
        progress,
        progress_text: text.into(),
        recommended: m.recommended,
        license: m.license.into(),
        color_index: m.task.icon(),
    }
}

fn category_data(t: Task) -> CategoryData {
    CategoryData {
        id: t.id().into(),
        title: t.title().into(),
        subtitle: t.subtitle().into(),
        count: catalog::for_task(t).count() as i32,
        icon: match t {
            Task::Chat => "chat",
            Task::Story => "book",
            Task::ChromaKey => "drop",
        }
        .into(),
        color_index: t.icon(),
    }
}

fn dir_size(p: &std::path::Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(p) else { return 0 };
    rd.flatten()
        .map(|e| {
            let p = e.path();
            if p.is_dir() {
                dir_size(&p)
            } else {
                e.metadata().map(|m| m.len()).unwrap_or(0)
            }
        })
        .sum()
}

fn refresh_models(ui: &AppWindow) {
    with_state(|st| {
        let all: Vec<ModelData> = catalog::MODELS.iter().map(|m| model_data(m, st)).collect();
        ui.set_models(ModelRc::new(VecModel::from(all)));
        let cat: Vec<ModelData> = catalog::for_task(st.task).map(|m| model_data(m, st)).collect();
        ui.set_category_models(ModelRc::new(VecModel::from(cat)));
        if let Some(m) = st.active {
            ui.set_active_model(model_data(m, st));
        }
        let installed = catalog::MODELS.iter().filter(|m| is_installed(m)).count();
        let used = dir_size(&platform::models_dir());
        let mut text = format!(
            "{installed} modèle{} installé{} sur {} · {} utilisés",
            if installed > 1 { "s" } else { "" },
            if installed > 1 { "s" } else { "" },
            catalog::MODELS.len(),
            human_size(used)
        );
        if !platform::is_mobile() {
            text.push_str(&format!(" · {}", platform::models_dir().display()));
        }
        ui.set_storage_text(text.into());
    });
}

fn show_toast(ui: &AppWindow, msg: impl Into<String>) {
    let id = with_state(|st| {
        st.toast_id += 1;
        st.toast_id
    });
    ui.set_toast(msg.into().into());
    let weak = ui.as_weak();
    slint::Timer::single_shot(std::time::Duration::from_millis(3200), move || {
        if with_state(|st| st.toast_id) == id {
            if let Some(ui) = weak.upgrade() {
                ui.set_toast(SharedString::new());
            }
        }
    });
}

fn gen_settings(st: &State) -> GenSettings {
    match st.task {
        Task::Story => st.settings.story.clone(),
        _ => st.settings.chat.clone(),
    }
}

fn params_to_ui(g: &GenSettings) -> GenParams {
    GenParams {
        max_tokens: g.max_tokens as f32,
        top_k: g.top_k as f32,
        top_p: g.top_p,
        temperature: g.temperature,
        repetition_penalty: g.repetition_penalty,
    }
}

fn sampling(g: &GenSettings) -> SamplingParams {
    SamplingParams {
        temperature: g.temperature,
        top_k: g.top_k as usize,
        top_p: g.top_p,
        repetition_penalty: g.repetition_penalty,
    }
}

// ------------------------------------------------------------------ modèles : chargement

fn sync_model_status(ui: &AppWindow) {
    with_state(|st| {
        let Some(m) = st.active else { return };
        let (status, text) = if !is_installed(m) {
            (0, String::new())
        } else if st.loaded == Some(m.id) {
            (2, String::new())
        } else if st.loading == Some(m.id) {
            (1, "Chargement du modèle en mémoire…".to_string())
        } else if let Some(e) = &st.load_error {
            (3, format!("Impossible de charger le modèle : {e}"))
        } else {
            (1, "Préparation…".to_string())
        };
        ui.set_model_status(status);
        ui.set_model_status_text(text.into());
    });
}

fn ensure_loaded(ui: &AppWindow) {
    with_state(|st| {
        let Some(m) = st.active else { return };
        if is_installed(m) && st.loaded != Some(m.id) && st.loading != Some(m.id) {
            st.loading = Some(m.id);
            st.load_error = None;
            let _ = st.worker.send(Job::Load(m));
        }
    });
    sync_model_status(ui);
}

fn on_loaded(ui: &AppWindow, id: &'static str, res: Result<f32, String>) {
    with_state(|st| {
        if st.loading == Some(id) {
            st.loading = None;
        }
        match res {
            Ok(_) => {
                st.loaded = Some(id);
                st.load_error = None;
            }
            Err(e) => {
                st.loaded = None;
                st.load_error = Some(e);
            }
        }
    });
    sync_model_status(ui);
}

// ------------------------------------------------------------------ génération de texte

fn on_piece(ui: &AppWindow, piece: &str) {
    with_state(|st| match st.task {
        Task::Chat => {
            let n = st.messages.row_count();
            if n > 0 {
                let mut m = st.messages.row_data(n - 1).unwrap();
                m.text = format!("{}{}", m.text, piece).into();
                st.messages.set_row_data(n - 1, m);
            }
        }
        Task::Story => {
            ui.set_story_output(format!("{}{}", ui.get_story_output(), piece).into());
        }
        _ => {}
    });
}

fn on_text_done(ui: &AppWindow, res: Result<String, String>, n: usize, secs: f32) {
    let speed = if secs > 0.0 { n as f32 / secs } else { 0.0 };
    let info = format!("{n} tokens · {:.1} tokens/s", speed).replace('.', ",");
    with_state(|st| {
        st.generating = false;
        match st.task {
            Task::Chat => {
                let n = st.messages.row_count();
                if n > 0 {
                    let mut m = st.messages.row_data(n - 1).unwrap();
                    match &res {
                        Ok(text) => {
                            let user = if n >= 2 { st.messages.row_data(n - 2).unwrap().text.to_string() } else { String::new() };
                            st.turns.push(Turn { user, bot: text.clone() });
                            if text.trim().is_empty() {
                                m.text = "(pas de réponse)".into();
                            } else {
                                m.text = text.trim().into();
                            }
                            m.info = info.clone().into();
                        }
                        Err(e) => m.text = format!("Erreur : {e}").into(),
                    }
                    st.messages.set_row_data(n - 1, m);
                }
            }
            Task::Story => match &res {
                Ok(_) => ui.set_story_info(info.clone().into()),
                Err(e) => ui.set_story_info(format!("Erreur : {e}").into()),
            },
            _ => {}
        }
    });
    ui.set_generating(false);
}

fn start_chat(ui: &AppWindow, text: String) {
    let text = text.trim().to_string();
    if text.is_empty() {
        return;
    }
    let started = with_state(|st| {
        if st.generating || st.active.map(|m| st.loaded != Some(m.id)).unwrap_or(true) {
            return false;
        }
        st.generating = true;
        st.stop = Arc::new(AtomicBool::new(false));
        st.messages.push(ChatMsg { user: true, text: text.clone().into(), info: SharedString::new() });
        st.messages.push(ChatMsg { user: false, text: SharedString::new(), info: SharedString::new() });
        let g = st.settings.chat.clone();
        let history: Vec<Turn> = st.turns.iter().map(|t| Turn { user: t.user.clone(), bot: t.bot.clone() }).collect();
        let _ = st.worker.send(Job::Chat {
            history,
            message: text,
            max: g.max_tokens as usize,
            params: sampling(&g),
            stop: st.stop.clone(),
        });
        true
    });
    if started {
        ui.set_chat_draft(SharedString::new());
        ui.set_generating(true);
    }
}

fn start_story(ui: &AppWindow, prompt: String) {
    let prompt = prompt.trim_end().to_string();
    if prompt.trim().is_empty() {
        return;
    }
    let started = with_state(|st| {
        if st.generating || st.active.map(|m| st.loaded != Some(m.id)).unwrap_or(true) {
            return false;
        }
        st.generating = true;
        st.stop = Arc::new(AtomicBool::new(false));
        let g = st.settings.story.clone();
        let _ = st.worker.send(Job::Story { prompt: prompt.clone(), max: g.max_tokens as usize, params: sampling(&g), stop: st.stop.clone() });
        true
    });
    if started {
        ui.set_story_output(prompt.into());
        ui.set_story_info("Écriture en cours…".into());
        ui.set_generating(true);
    }
}

// ------------------------------------------------------------------ détourage

fn to_slint_rgba(img: &image::RgbaImage) -> slint::Image {
    let buf = SharedPixelBuffer::<slint::Rgba8Pixel>::clone_from_slice(img.as_raw(), img.width(), img.height());
    slint::Image::from_rgba8(buf)
}

fn to_slint_rgb(img: &image::RgbImage) -> slint::Image {
    let buf = SharedPixelBuffer::<slint::Rgb8Pixel>::clone_from_slice(img.as_raw(), img.width(), img.height());
    slint::Image::from_rgb8(buf)
}

/// Compose le résultat : `view` 1 = image détourée, 2 = masque. `checker` dessine un damier sous la transparence.
fn compose(src: &image::RgbImage, prob: &[f32], threshold: f32, soft: bool, bg: i32, view: i32, despill: bool, checker: bool) -> image::RgbaImage {
    let (w, h) = src.dimensions();
    let mut out = image::RgbaImage::new(w, h);
    for (i, (x, y, p)) in src.enumerate_pixels().enumerate() {
        let pr = prob[i];
        let a = if soft { (0.5 + (threshold - pr) / 0.3).clamp(0.0, 1.0) } else if pr < threshold { 1.0 } else { 0.0 };
        if view == 2 {
            let v = (a * 255.0) as u8;
            out.put_pixel(x, y, image::Rgba([v, v, v, 255]));
            continue;
        }
        let [mut r, mut g, b] = p.0;
        if despill && a < 1.0 {
            g = g.min(r.max(b));
        }
        let (br, bgc, bb, ba) = match bg {
            1 => (255.0, 255.0, 255.0, 1.0),
            2 => (0.0, 0.0, 0.0, 1.0),
            _ if checker => {
                let c = if ((x / 12) + (y / 12)) % 2 == 0 { 200.0 } else { 150.0 };
                (c, c, c, 1.0)
            }
            _ => (0.0, 0.0, 0.0, 0.0),
        };
        if ba == 0.0 {
            out.put_pixel(x, y, image::Rgba([r, g, b, (a * 255.0) as u8]));
        } else {
            let mix = |f: u8, bgv: f32| (f as f32 * a + bgv * (1.0 - a)) as u8;
            r = mix(r, br);
            let g2 = mix(g, bgc);
            out.put_pixel(x, y, image::Rgba([r, g2, mix(b, bb), 255]));
        }
    }
    out
}

fn chroma_refresh(ui: &AppWindow) {
    with_state(|st| {
        let c = &st.chroma;
        let Some(preview) = &c.preview else {
            ui.set_chroma_has_image(false);
            return;
        };
        ui.set_chroma_has_image(true);
        let view = ui.get_chroma_view();
        match (&c.preview_prob, view) {
            (Some(prob), 1 | 2) => {
                let img = compose(
                    preview,
                    prob,
                    ui.get_chroma_threshold(),
                    ui.get_chroma_soft(),
                    ui.get_chroma_bg(),
                    view,
                    ui.get_chroma_preset() == 0,
                    true,
                );
                ui.set_chroma_preview(to_slint_rgba(&img));
            }
            _ => ui.set_chroma_preview(to_slint_rgb(preview)),
        }
        if let Some(prob) = &c.preview_prob {
            let t = ui.get_chroma_threshold();
            let pct = prob.iter().filter(|p| **p >= t).count() as f32 / prob.len().max(1) as f32 * 100.0;
            let base = ui.get_chroma_info().to_string();
            let base = base.split(" · Couleur détectée").next().unwrap_or("").to_string();
            ui.set_chroma_info(format!("{base} · Couleur détectée sur {:.1} % de l'image", pct).replace('.', ",").into());
        }
    });
}

fn chroma_set_image(ui: &AppWindow, path: PathBuf) {
    let img = image::ImageReader::open(&path).and_then(|r| r.with_guessed_format()).map_err(|e| e.to_string()).and_then(|r| r.decode().map_err(|e| e.to_string()));
    match img {
        Ok(img) => {
            let rgb = img.to_rgb8();
            let preview = if rgb.width().max(rgb.height()) > 1280 {
                image::DynamicImage::ImageRgb8(rgb.clone()).resize(1280, 1280, image::imageops::FilterType::Triangle).to_rgb8()
            } else {
                rgb.clone()
            };
            let name = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "image".into());
            let (w, h) = rgb.dimensions();
            with_state(|st| {
                st.chroma = ChromaState { source: Some(rgb), preview: Some(preview), name, prob: None, preview_prob: None };
            });
            ui.set_chroma_view(0);
            ui.set_chroma_has_result(false);
            ui.set_chroma_info(format!("{w} × {h} px").into());
            chroma_refresh(ui);
        }
        Err(e) => show_toast(ui, format!("Image illisible : {e}")),
    }
}

fn chroma_run(ui: &AppWindow) {
    let color = [ui.get_chroma_r(), ui.get_chroma_g(), ui.get_chroma_b()];
    let ok = with_state(|st| {
        let Some(src) = &st.chroma.source else { return false };
        if st.active.map(|m| st.loaded != Some(m.id)).unwrap_or(true) {
            return false;
        }
        let _ = st.worker.send(Job::Segment { image: Arc::new(src.clone()), color, max_side: st.settings.chroma_max_side });
        true
    });
    if ok {
        ui.set_chroma_busy(true);
    }
}

fn on_segmented(ui: &AppWindow, res: Result<((Vec<f32>, usize, usize), f32), String>) {
    ui.set_chroma_busy(false);
    match res {
        Ok(((prob, pw, ph), secs)) => {
            with_state(|st| {
                if let Some(pv) = &st.chroma.preview {
                    st.chroma.preview_prob = Some(resize_plane(&prob, pw, ph, pv.width() as usize, pv.height() as usize));
                }
                st.chroma.prob = Some((prob, pw, ph));
                if let Some(src) = &st.chroma.source {
                    ui.set_chroma_info(format!("{} × {} px · traité en {:.2} s", src.width(), src.height(), secs).replace('.', ",").into());
                }
            });
            ui.set_chroma_has_result(true);
            ui.set_chroma_view(1);
            chroma_refresh(ui);
        }
        Err(e) => show_toast(ui, format!("Erreur : {e}")),
    }
}

fn chroma_save(ui: &AppWindow) {
    let (img, name) = match with_state(|st| {
        let c = &st.chroma;
        let (src, (prob, pw, ph)) = (c.source.as_ref()?, c.prob.as_ref()?);
        let full = resize_plane(prob, *pw, *ph, src.width() as usize, src.height() as usize);
        let img = compose(src, &full, ui.get_chroma_threshold(), ui.get_chroma_soft(), ui.get_chroma_bg(), 1, ui.get_chroma_preset() == 0, false);
        Some((img, format!("{}-detoure.png", c.name)))
    }) {
        Some(v) => v,
        None => return,
    };
    match platform::save_png(&img, &name) {
        Ok(Some(p)) => show_toast(ui, format!("Image enregistrée : {}", p.display())),
        Ok(None) => {}
        Err(e) => show_toast(ui, format!("Échec de l'enregistrement : {e}")),
    }
}

fn open_gallery(ui: &AppWindow) {
    ui.set_gallery(ModelRc::new(VecModel::<GalleryItem>::default()));
    ui.set_gallery_open(true);
    if !platform::request_media_permission() {
        ui.set_gallery_message("Autorisez l'accès à vos photos dans la fenêtre qui s'affiche, puis appuyez sur « Réessayer ».".into());
        return;
    }
    ui.set_gallery_message("Recherche des images…".into());
    let weak = ui.as_weak();
    std::thread::spawn(move || {
        let files = platform::list_images(&platform::gallery_roots());
        if files.is_empty() {
            let _ = weak.upgrade_in_event_loop(|ui| ui.set_gallery_message("Aucune image trouvée dans DCIM, Pictures ou Download.".into()));
            return;
        }
        let mut batch = Vec::new();
        let total = files.len();
        for (i, f) in files.into_iter().enumerate() {
            let thumb = image::ImageReader::open(&f)
                .ok()
                .and_then(|r| r.with_guessed_format().ok())
                .and_then(|r| r.decode().ok())
                .map(|im| im.thumbnail(220, 220).to_rgba8());
            if let Some(t) = thumb {
                let buf = SharedPixelBuffer::<slint::Rgba8Pixel>::clone_from_slice(t.as_raw(), t.width(), t.height());
                batch.push((f.to_string_lossy().to_string(), buf));
            }
            if batch.len() >= 6 || i + 1 == total {
                let items = std::mem::take(&mut batch);
                let ok = weak
                    .upgrade_in_event_loop(move |ui| {
                        if !ui.get_gallery_open() {
                            return;
                        }
                        let model = ui.get_gallery();
                        let vm = model.as_any().downcast_ref::<VecModel<GalleryItem>>();
                        if let Some(vm) = vm {
                            for (p, b) in items {
                                vm.push(GalleryItem { path: p.into(), thumb: slint::Image::from_rgba8(b) });
                            }
                        }
                    })
                    .is_ok();
                if !ok {
                    break;
                }
            }
        }
    });
}

// ------------------------------------------------------------------ téléchargements

fn start_download(ui: &AppWindow, id: &str) {
    let Some(m) = catalog::find(id) else { return };
    let cancel = Arc::new(AtomicBool::new(false));
    let already = with_state(|st| {
        if st.downloads.get(m.id).is_some_and(|d| d.error.is_none()) {
            return true;
        }
        st.downloads.insert(m.id, DlState { progress: 0.0, text: "Connexion…".into(), error: None, cancel: cancel.clone() });
        false
    });
    if already {
        return;
    }
    refresh_models(ui);
    let weak = ui.as_weak();
    std::thread::spawn(move || {
        let dir = model_dir(m);
        let total = m.total_size().max(1);
        let mut done = 0u64;
        let mut result = Ok(());
        let t0 = Instant::now();
        for f in m.files {
            let dest = dir.join(f.name);
            if dest.is_file() {
                done += f.size;
                continue;
            }
            let base = done;
            let w = weak.clone();
            let r = net::download(&m.file_url(f.name), &dest, None, &cancel, |got, _| {
                let cur = base + got;
                let p = cur as f32 / total as f32;
                let rate = cur.saturating_sub(base) as f32 / t0.elapsed().as_secs_f32().max(0.1);
                let text = format!("{} / {} · {}/s", human_size(cur), human_size(total), human_size(rate as u64));
                let _ = w.upgrade_in_event_loop(move |ui| {
                    with_state(|st| {
                        if let Some(d) = st.downloads.get_mut(m.id) {
                            d.progress = p.min(1.0);
                            d.text = text;
                        }
                    });
                    refresh_models(&ui);
                });
            });
            if let Err(e) = r {
                result = Err(e);
                break;
            }
            done += std::fs::metadata(&dest).map(|md| md.len()).unwrap_or(f.size);
        }
        let cancelled = cancel.load(Ordering::Relaxed);
        let _ = weak.upgrade_in_event_loop(move |ui| {
            with_state(|st| match &result {
                Err(e) if !cancelled => {
                    if let Some(d) = st.downloads.get_mut(m.id) {
                        d.error = Some(e.to_string());
                    }
                }
                _ => {
                    st.downloads.remove(m.id);
                }
            });
            refresh_models(&ui);
            if result.is_ok() {
                show_toast(&ui, format!("{} est prêt", m.name));
                let is_active = with_state(|st| st.active.map(|a| a.id) == Some(m.id));
                if is_active && ui.get_page() >= 3 {
                    ensure_loaded(&ui);
                }
            }
            sync_model_status(&ui);
        });
    });
}

fn cancel_download(ui: &AppWindow, id: &str) {
    with_state(|st| {
        if let Some(d) = st.downloads.get(id) {
            d.cancel.store(true, Ordering::Relaxed);
        }
        if st.downloads.get(id).is_some_and(|d| d.error.is_some()) {
            st.downloads.remove(id);
        }
    });
    refresh_models(ui);
}

fn delete_model(ui: &AppWindow, id: &str) {
    let Some(m) = catalog::find(id) else { return };
    with_state(|st| {
        let _ = st.worker.send(Job::Unload(m.id));
        if st.loaded == Some(m.id) {
            st.loaded = None;
        }
        st.downloads.remove(m.id);
    });
    let _ = std::fs::remove_dir_all(model_dir(m));
    refresh_models(ui);
    sync_model_status(ui);
    show_toast(ui, format!("{} a été supprimé", m.name));
}

// ------------------------------------------------------------------ mises à jour

fn check_update(ui: &AppWindow, silent: bool) {
    if ui.get_update_state() == 1 || ui.get_update_state() == 2 {
        return;
    }
    ui.set_update_state(1);
    let weak = ui.as_weak();
    std::thread::spawn(move || {
        let r = update::check();
        let _ = weak.upgrade_in_event_loop(move |ui| match r {
            Ok(Some(rel)) => {
                ui.set_update_state(0);
                ui.set_update_available(true);
                ui.set_update_version(rel.version.clone().into());
                ui.set_update_notes(rel.notes.clone().into());
                ui.set_update_message(SharedString::new());
                with_state(|st| st.release = Some(rel));
                if !silent {
                    ui.set_update_open(true);
                }
            }
            Ok(None) => {
                ui.set_update_state(4);
                ui.set_update_message(format!("Vous utilisez la dernière version ({}).", update::CURRENT).into());
            }
            Err(e) => {
                ui.set_update_state(if silent { 0 } else { 3 });
                ui.set_update_message(format!("Impossible de vérifier les mises à jour : {e}").into());
            }
        });
    });
}

fn install_update(ui: &AppWindow) {
    let Some(rel) = with_state(|st| st.release.clone()) else { return };
    let Some(url) = rel.asset_url.clone() else {
        platform::open_url(&rel.page_url);
        return;
    };
    if platform::is_mobile() {
        // Le navigateur télécharge l'APK ; Android propose ensuite l'installation.
        platform::open_url(&url);
        show_toast(ui, "Ouvrez le fichier téléchargé pour installer la mise à jour.");
        return;
    }
    ui.set_update_state(2);
    ui.set_update_progress(0.0);
    ui.set_update_message("Téléchargement…".into());
    let weak = ui.as_weak();
    std::thread::spawn(move || {
        let dest = platform::data_dir().join("update").join(platform::release_asset_name().unwrap_or("update.bin"));
        let _ = std::fs::remove_file(&dest);
        let cancel = AtomicBool::new(false);
        let w = weak.clone();
        let size = (rel.asset_size > 0).then_some(rel.asset_size);
        let r = net::download(&url, &dest, size, &cancel, |got, total| {
            let p = if total > 0 { got as f32 / total as f32 } else { 0.0 };
            let msg = format!("Téléchargement… {} / {}", human_size(got), human_size(total));
            let _ = w.upgrade_in_event_loop(move |ui| {
                ui.set_update_progress(p);
                ui.set_update_message(msg.into());
            });
        });
        let r = r.and_then(|_| {
            let _ = weak.upgrade_in_event_loop(|ui| ui.set_update_message("Installation et redémarrage…".into()));
            platform::install_update(&dest)
        });
        if let Err(e) = r {
            let _ = weak.upgrade_in_event_loop(move |ui| {
                ui.set_update_state(3);
                ui.set_update_message(format!("La mise à jour a échoué : {e}").into());
            });
        }
    });
}

// ------------------------------------------------------------------ point d'entrée

pub fn run() -> Result<(), slint::PlatformError> {
    let ui = AppWindow::new()?;
    let settings = Settings::load();
    let worker = spawn_worker(ui.as_weak());
    let messages = Rc::new(VecModel::<ChatMsg>::default());
    STATE.with(|s| {
        *s.borrow_mut() = Some(State {
            settings: settings.clone(),
            downloads: HashMap::new(),
            task: Task::Chat,
            active: None,
            loaded: None,
            loading: None,
            load_error: None,
            messages: messages.clone(),
            turns: Vec::new(),
            stop: Arc::new(AtomicBool::new(false)),
            generating: false,
            chroma: ChromaState { source: None, preview: None, name: String::new(), prob: None, preview_prob: None },
            release: None,
            worker,
            toast_id: 0,
        })
    });

    ui.set_app_version(update::CURRENT.into());
    ui.set_mobile(platform::is_mobile());
    ui.set_native_picker(platform::has_native_picker());
    ui.set_messages(ModelRc::from(messages.clone()));
    ui.set_auto_update(settings.auto_update_check);
    ui.global::<Theme>().set_mode(settings.theme);
    ui.invoke_apply_theme();
    let cats: Vec<CategoryData> = Task::ALL.iter().map(|t| category_data(*t)).collect();
    ui.set_categories(ModelRc::new(VecModel::from(cats)));
    refresh_models(&ui);

    // --- navigation
    let w = ui.as_weak();
    ui.on_open_category(move |id| {
        let ui = w.unwrap();
        if let Some(t) = Task::from_id(&id) {
            with_state(|st| st.task = t);
            ui.set_current_category(category_data(t));
            refresh_models(&ui);
        }
    });
    let w = ui.as_weak();
    ui.on_try_model(move |id| {
        let ui = w.unwrap();
        let Some(m) = catalog::find(&id) else { return };
        let switched = with_state(|st| {
            let switched = st.active.map(|a| a.id) != Some(m.id);
            st.task = m.task;
            st.active = Some(m);
            if switched {
                st.messages.set_vec(Vec::new());
                st.turns.clear();
            }
            switched
        });
        ui.set_current_category(category_data(m.task));
        ui.set_gen_params(with_state(|st| params_to_ui(&gen_settings(st))));
        if switched {
            ui.set_story_output(SharedString::new());
            ui.set_story_info(SharedString::new());
        }
        refresh_models(&ui);
        ui.invoke_go(match m.task {
            Task::Chat => 3,
            Task::Story => 4,
            Task::ChromaKey => 5,
        });
        ensure_loaded(&ui);
    });
    let w = ui.as_weak();
    ui.on_navigated(move |p| {
        let ui = w.unwrap();
        if p < 3 {
            with_state(|st| st.stop.store(true, Ordering::Relaxed));
        }
        if p == 1 || p == 2 {
            refresh_models(&ui);
        }
    });
    ui.on_open_url(|u| platform::open_url(&u));
    ui.on_open_data_folder(|| platform::open_folder(&platform::models_dir()));

    // --- modèles
    let w = ui.as_weak();
    ui.on_download_model(move |id| start_download(&w.unwrap(), &id));
    let w = ui.as_weak();
    ui.on_cancel_download(move |id| cancel_download(&w.unwrap(), &id));
    let w = ui.as_weak();
    ui.on_delete_model(move |id| delete_model(&w.unwrap(), &id));

    // --- discussion & histoires
    let w = ui.as_weak();
    ui.on_send_message(move |t| start_chat(&w.unwrap(), t.to_string()));
    ui.on_stop_generation(|| with_state(|st| st.stop.store(true, Ordering::Relaxed)));
    ui.on_new_chat(|| {
        with_state(|st| {
            st.messages.set_vec(Vec::new());
            st.turns.clear();
        })
    });
    let w = ui.as_weak();
    ui.on_story_generate(move |p| start_story(&w.unwrap(), p.to_string()));
    let w = ui.as_weak();
    ui.on_story_clear(move || {
        let ui = w.unwrap();
        ui.set_story_prompt(SharedString::new());
        ui.set_story_output(SharedString::new());
        ui.set_story_info(SharedString::new());
    });
    let w = ui.as_weak();
    ui.on_apply_params(move |p| {
        let ui = w.unwrap();
        let g = GenSettings {
            max_tokens: p.max_tokens.round() as u32,
            top_k: p.top_k.round() as u32,
            top_p: p.top_p,
            temperature: p.temperature,
            repetition_penalty: p.repetition_penalty,
        };
        with_state(|st| {
            match st.task {
                Task::Story => st.settings.story = g.clone(),
                _ => st.settings.chat = g.clone(),
            }
            st.settings.save();
        });
        ui.set_gen_params(params_to_ui(&g));
    });
    let w = ui.as_weak();
    ui.on_reset_params(move || {
        let ui = w.unwrap();
        let g = with_state(|st| match st.task {
            Task::Story => GenSettings::story_default(),
            _ => GenSettings::chat_default(),
        });
        ui.set_gen_params(params_to_ui(&g));
    });

    // --- détourage
    let w = ui.as_weak();
    ui.on_chroma_pick(move || {
        let ui = w.unwrap();
        if platform::has_native_picker() {
            if let Some(p) = platform::pick_image() {
                chroma_set_image(&ui, p);
            }
        } else {
            open_gallery(&ui);
        }
    });
    let w = ui.as_weak();
    ui.on_gallery_retry(move || open_gallery(&w.unwrap()));
    let w = ui.as_weak();
    ui.on_gallery_select(move |p| {
        let ui = w.unwrap();
        ui.set_gallery_open(false);
        ui.set_gallery(ModelRc::new(VecModel::<GalleryItem>::default()));
        chroma_set_image(&ui, PathBuf::from(p.as_str()));
    });
    let w = ui.as_weak();
    ui.on_chroma_run(move || chroma_run(&w.unwrap()));
    let w = ui.as_weak();
    ui.on_chroma_refresh(move || chroma_refresh(&w.unwrap()));
    let w = ui.as_weak();
    ui.on_chroma_save(move || chroma_save(&w.unwrap()));

    // --- réglages & mises à jour
    let w = ui.as_weak();
    ui.on_set_theme(move |m| {
        let ui = w.unwrap();
        ui.global::<Theme>().set_mode(m);
        ui.invoke_apply_theme();
        with_state(|st| {
            st.settings.theme = m;
            st.settings.save();
        });
    });
    ui.on_set_auto_update(|v| {
        with_state(|st| {
            st.settings.auto_update_check = v;
            st.settings.save();
        })
    });
    let w = ui.as_weak();
    ui.on_check_update(move || check_update(&w.unwrap(), false));
    let w = ui.as_weak();
    ui.on_install_update(move || install_update(&w.unwrap()));

    if settings.auto_update_check {
        check_update(&ui, true);
    }
    // Nettoie un éventuel binaire de mise à jour resté sur le disque.
    let _ = std::fs::remove_dir_all(platform::data_dir().join("update"));

    // Scénario de démonstration (développement / captures d'écran) : PLIMB_DEMO="action;action…"
    if let Ok(script) = std::env::var("PLIMB_DEMO") {
        run_demo(&ui, script);
    }

    ui.run()
}

fn run_demo(ui: &AppWindow, script: String) {
    let steps: Vec<String> = script.split(';').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    let weak = ui.as_weak();
    let timer = Box::leak(Box::new(slint::Timer::default()));
    let idx = std::cell::Cell::new(0usize);
    timer.start(slint::TimerMode::Repeated, std::time::Duration::from_millis(400), move || {
        let Some(ui) = weak.upgrade() else { return };
        let i = idx.get();
        let Some(step) = steps.get(i) else { return };
        // Attend que le modèle soit prêt / la génération finie avant l'étape suivante.
        if ui.get_generating() || ui.get_chroma_busy() || ui.get_model_status() == 1 {
            return;
        }
        idx.set(i + 1);
        let (cmd, arg) = step.split_once(':').unwrap_or((step.as_str(), ""));
        match cmd {
            "category" => {
                ui.invoke_open_category(arg.into());
                ui.invoke_go(1);
            }
            "page" => ui.invoke_go(arg.parse().unwrap_or(0)),
            "try" => ui.invoke_try_model(arg.into()),
            "send" => ui.invoke_send_message(arg.into()),
            "story" => {
                ui.set_story_prompt(arg.into());
                ui.invoke_story_generate(arg.into());
            }
            "image" => chroma_set_image(&ui, PathBuf::from(arg)),
            "chroma" => ui.invoke_chroma_run(),
            "settings" => ui.set_settings_open(true),
            "drawer" => ui.set_drawer_open(true),
            "params" => {
                ui.set_params_open(true);
            }
            "theme" => ui.invoke_set_theme(arg.parse().unwrap_or(0)),
            "close" => {
                ui.invoke_go_back();
            }
            "size" => {
                let (w, h) = arg.split_once('x').unwrap_or(("400", "800"));
                ui.window().set_size(slint::LogicalSize::new(w.parse().unwrap_or(400.0), h.parse().unwrap_or(800.0)));
            }
            "wait" => {}
            "quit" => {
                let _ = slint::quit_event_loop();
            }
            _ => log::warn!("étape inconnue : {step}"),
        }
    });
}
