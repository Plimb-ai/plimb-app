//! Catalogue des modèles publiés par Plimb sur Hugging Face (https://huggingface.co/plimb).

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Task {
    Chat,
    Story,
    ChromaKey,
}

impl Task {
    pub const ALL: [Task; 3] = [Task::Chat, Task::Story, Task::ChromaKey];

    pub fn id(self) -> &'static str {
        match self {
            Task::Chat => "chat",
            Task::Story => "story",
            Task::ChromaKey => "chroma",
        }
    }

    pub fn from_id(id: &str) -> Option<Task> {
        Task::ALL.into_iter().find(|t| t.id() == id)
    }

    pub fn title(self) -> &'static str {
        match self {
            Task::Chat => "Discussion",
            Task::Story => "Atelier d'histoires",
            Task::ChromaKey => "Détourage couleur",
        }
    }

    pub fn subtitle(self) -> &'static str {
        match self {
            Task::Chat => "Converser avec un modèle de langage sur l'appareil",
            Task::Story => "Écrire de courtes histoires à partir d'un début de phrase",
            Task::ChromaKey => "Retirer un fond vert (ou d'une autre couleur) d'une image",
        }
    }

    /// Index de l'icône dans l'interface (voir `ui/icons.slint`).
    pub fn icon(self) -> i32 {
        match self {
            Task::Chat => 0,
            Task::Story => 1,
            Task::ChromaKey => 2,
        }
    }
}

pub struct ModelFile {
    pub name: &'static str,
    pub size: u64,
}

pub enum Engine {
    /// GPT-2 Hugging Face (safetensors + tokenizer.json)
    HfGpt2,
    /// GPT Gladios (checkpoint PyTorch + vocab.json)
    GladiosPt { weights: &'static str, n_head: usize },
    /// Codelor Color (safetensors)
    Codelor,
}

pub struct ModelInfo {
    pub id: &'static str,
    pub repo: &'static str,
    pub name: &'static str,
    pub task: Task,
    pub description: &'static str,
    pub details: &'static [&'static str],
    pub files: &'static [ModelFile],
    pub engine: Engine,
    pub recommended: bool,
    pub license: &'static str,
}

impl ModelInfo {
    pub fn total_size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    pub fn url(&self) -> String {
        format!("https://huggingface.co/{}", self.repo)
    }

    pub fn file_url(&self, file: &str) -> String {
        format!("https://huggingface.co/{}/resolve/main/{}", self.repo, file)
    }
}

pub static MODELS: &[ModelInfo] = &[
    ModelInfo {
        id: "gladios-0.3b-eng-prototype",
        repo: "plimb/gladios-0.3b-eng-prototype",
        name: "Gladios 0.3B ENG",
        task: Task::Chat,
        description: "Modèle conversationnel d'environ 300 M de paramètres, entraîné de zéro sur des \
                      données de dialogue en anglais. Prototype : réponses simples, contexte de 256 tokens.",
        details: &["~300 M paramètres", "Anglais", "Contexte 256", "Prototype"],
        files: &[
            ModelFile { name: "vocab.json", size: 113_900 },
            ModelFile { name: "gladios-0.3b-eng-prototype.pt", size: 1_248_553_879 },
        ],
        engine: Engine::GladiosPt { weights: "gladios-0.3b-eng-prototype.pt", n_head: 16 },
        recommended: true,
        license: "MIT",
    },
    ModelInfo {
        id: "gladios-tiny.story-0.1B",
        repo: "plimb/gladios-tiny.story-0.1B",
        name: "Gladios Tiny Story 0.1B",
        task: Task::Story,
        description: "Transformer de type GPT-2 (~124 M de paramètres) entraîné de zéro sur TinyStories. \
                      Écrit de courtes histoires pour enfants, en anglais.",
        details: &["~124 M paramètres", "Anglais", "Contexte 512", "TinyStories"],
        files: &[
            ModelFile { name: "config.json", size: 809 },
            ModelFile { name: "tokenizer.json", size: 3_557_680 },
            ModelFile { name: "model.safetensors", size: 496_201_344 },
        ],
        engine: Engine::HfGpt2,
        recommended: true,
        license: "MIT",
    },
    ModelInfo {
        id: "codelor-color-1.2",
        repo: "plimb/codelor-color-1.2",
        name: "Codelor Color 1.2",
        task: Task::ChromaKey,
        description: "Modèle de segmentation ultra-léger (1,6 M de paramètres) spécialisé dans le chroma key : \
                      il détecte une couleur cible (vert par défaut) et produit un masque alpha propre.",
        details: &["1,6 M paramètres", "6,3 Mo", "Optimisé CPU", "Image → masque"],
        files: &[
            ModelFile { name: "config.json", size: 519 },
            ModelFile { name: "model.safetensors", size: 6_575_768 },
        ],
        engine: Engine::Codelor,
        recommended: true,
        license: "MIT",
    },
];

pub fn find(id: &str) -> Option<&'static ModelInfo> {
    MODELS.iter().find(|m| m.id == id)
}

pub fn for_task(task: Task) -> impl Iterator<Item = &'static ModelInfo> {
    MODELS.iter().filter(move |m| m.task == task)
}

pub fn human_size(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.2} Go", b / 1e9).replace('.', ",")
    } else if b >= 1e6 {
        format!("{:.1} Mo", b / 1e6).replace('.', ",")
    } else {
        format!("{:.0} Ko", b / 1e3)
    }
}
