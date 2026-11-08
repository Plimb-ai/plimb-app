use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct GenSettings {
    pub max_tokens: u32,
    pub top_k: u32,
    pub top_p: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
}

impl GenSettings {
    pub fn chat_default() -> Self {
        Self { max_tokens: 128, top_k: 40, top_p: 0.95, temperature: 0.8, repetition_penalty: 1.1 }
    }
    pub fn story_default() -> Self {
        Self { max_tokens: 300, top_k: 50, top_p: 1.0, temperature: 0.8, repetition_penalty: 1.0 }
    }
}

impl Default for GenSettings {
    fn default() -> Self {
        Self::chat_default()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Settings {
    /// 0 = auto, 1 = clair, 2 = sombre
    pub theme: i32,
    pub auto_update_check: bool,
    pub chat: GenSettings,
    pub story: GenSettings,
    pub chroma_max_side: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            theme: 0,
            auto_update_check: true,
            chat: GenSettings::chat_default(),
            story: GenSettings::story_default(),
            chroma_max_side: 512,
        }
    }
}

impl Settings {
    fn path() -> std::path::PathBuf {
        crate::platform::data_dir().join("settings.json")
    }

    pub fn load() -> Self {
        std::fs::read(Self::path()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
    }

    pub fn save(&self) {
        if let Ok(s) = serde_json::to_vec_pretty(self) {
            let _ = std::fs::write(Self::path(), s);
        }
    }
}
