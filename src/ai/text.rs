//! Génération de texte (chat et histoires) au-dessus de [`Gpt`].

use super::bpe::{ByteBpe, StreamDecoder};
use super::gpt::{sample, Gpt, SamplingParams};
use anyhow::{anyhow, Result};
use std::path::Path;

pub enum PromptStyle {
    /// `<|user|> question <|bot|> réponse <|endoftext|>`
    Chat { user: u32, bot: u32 },
    /// Simple complétion de texte
    Completion,
}

pub struct TextModel {
    gpt: Gpt,
    tok: ByteBpe,
    style: PromptStyle,
    eos: u32,
}

pub struct Turn {
    pub user: String,
    pub bot: String,
}

impl TextModel {
    pub fn load_story(dir: &Path) -> Result<Self> {
        let tok = ByteBpe::from_tokenizer_json(&dir.join("tokenizer.json"))?;
        let gpt = Gpt::load_hf(dir)?;
        let eos = tok.token_id("<|endoftext|>").unwrap_or(50256);
        Ok(Self { gpt, tok, style: PromptStyle::Completion, eos })
    }

    pub fn load_gladios_chat(dir: &Path, weights_file: &str, n_head: usize) -> Result<Self> {
        let tok = ByteBpe::from_vocab_json(&dir.join("vocab.json"))?;
        let gpt = Gpt::load_gladios(&dir.join(weights_file), n_head)?;
        let user = tok.token_id("<|user|>").ok_or_else(|| anyhow!("token <|user|> absent"))?;
        let bot = tok.token_id("<|bot|>").ok_or_else(|| anyhow!("token <|bot|> absent"))?;
        let eos = tok.token_id("<|endoftext|>").unwrap_or(0);
        Ok(Self { gpt, tok, style: PromptStyle::Chat { user, bot }, eos })
    }

    pub fn context_len(&self) -> usize {
        self.gpt.n_ctx
    }

    fn build_prompt(&self, history: &[Turn], message: &str, budget: usize) -> Vec<u32> {
        match self.style {
            PromptStyle::Completion => {
                let mut ids = vec![self.eos];
                ids.extend(self.tok.encode(message));
                if ids.len() > budget {
                    ids.drain(..ids.len() - budget);
                }
                ids
            }
            PromptStyle::Chat { user, bot } => {
                let mut current = vec![user];
                current.extend(self.tok.encode(message.trim()));
                current.push(bot);
                if current.len() > budget {
                    // Trop long : on garde la fin du message.
                    let mut ids = vec![user];
                    let keep = budget.saturating_sub(2);
                    let enc = self.tok.encode(message.trim());
                    ids.extend_from_slice(&enc[enc.len().saturating_sub(keep)..]);
                    ids.push(bot);
                    return ids;
                }
                // Ajoute l'historique récent tant qu'il tient dans le budget.
                let mut prefix: Vec<u32> = Vec::new();
                for turn in history.iter().rev() {
                    let mut t = vec![user];
                    t.extend(self.tok.encode(turn.user.trim()));
                    t.push(bot);
                    t.extend(self.tok.encode(turn.bot.trim()));
                    t.push(self.eos);
                    if t.len() + prefix.len() + current.len() > budget {
                        break;
                    }
                    t.extend(prefix);
                    prefix = t;
                }
                prefix.extend(current);
                prefix
            }
        }
    }

    /// Génère une réponse. `on_text` reçoit chaque fragment et renvoie `false` pour arrêter.
    pub fn generate(
        &mut self,
        history: &[Turn],
        message: &str,
        max_new: usize,
        params: SamplingParams,
        mut on_text: impl FnMut(&str) -> bool,
    ) -> Result<String> {
        let n_ctx = self.gpt.n_ctx;
        let max_new = max_new.clamp(1, n_ctx.saturating_sub(16));
        let budget = n_ctx - max_new.min(n_ctx / 2);
        let mut ids = self.build_prompt(history, message, budget);
        let mut rng = rand::rng();
        let mut dec = StreamDecoder::default();
        let mut out = String::new();
        let mut generated: Vec<u32> = Vec::new();

        self.gpt.clear_cache();
        let mut logits = self.gpt.forward(&ids)?;
        for _ in 0..max_new {
            let mut hist = ids.clone();
            hist.extend(&generated);
            let next = sample(&mut logits, &hist, &params, &mut rng);
            if next == self.eos {
                break;
            }
            if let PromptStyle::Chat { user, bot } = self.style {
                if next == user || next == bot {
                    break;
                }
            }
            generated.push(next);
            let piece = dec.push(self.tok.token_bytes(next));
            if !piece.is_empty() {
                out.push_str(&piece);
                if !on_text(&piece) {
                    break;
                }
            }
            if self.gpt.cached_len() + 1 > n_ctx {
                // Fenêtre glissante : on recalcule avec la fin du contexte.
                ids.extend(&generated);
                generated.clear();
                let keep = n_ctx * 3 / 4;
                ids.drain(..ids.len().saturating_sub(keep));
                self.gpt.clear_cache();
                logits = self.gpt.forward(&ids)?;
            } else {
                logits = self.gpt.forward(&[next])?;
            }
        }
        Ok(out)
    }
}
