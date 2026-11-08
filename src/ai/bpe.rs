//! Tokenizer BPE "byte-level" (style GPT-2), écrit en Rust pur.
//!
//! Utilisé par les deux modèles Gladios :
//! - `gladios-tiny.story-0.1B` : `tokenizer.json` HF (vocab + merges)
//! - `gladios-0.3b-eng-prototype` : `vocab.json` seul. Le tokenizer a été
//!   entraîné avec `ByteLevelBPETokenizer`, qui attribue les identifiants dans
//!   l'ordre des fusions : le rang d'une fusion est donc l'id du token produit.

use anyhow::{anyhow, Context, Result};
use fancy_regex::Regex;
use std::collections::HashMap;
use std::path::Path;

const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

pub struct ByteBpe {
    encoder: HashMap<String, u32>,
    decoder: Vec<Vec<u8>>,
    /// Rang des fusions : (gauche, droite) -> rang (plus petit = prioritaire)
    ranks: HashMap<(String, String), u32>,
    byte_to_char: [char; 256],
    pattern: Regex,
}

fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        out[*b as usize] = char::from_u32(*c).unwrap();
    }
    out
}

impl ByteBpe {
    fn build(encoder: HashMap<String, u32>, ranks: HashMap<(String, String), u32>) -> Result<Self> {
        let byte_to_char = bytes_to_unicode();
        let char_to_byte: HashMap<char, u8> =
            byte_to_char.iter().enumerate().map(|(b, c)| (*c, b as u8)).collect();
        let max_id = encoder.values().copied().max().unwrap_or(0) as usize;
        let mut decoder = vec![Vec::new(); max_id + 1];
        for (tok, id) in &encoder {
            // Les tokens spéciaux (<|user|>…) ne sont pas encodés octet par octet.
            let bytes: Option<Vec<u8>> = tok.chars().map(|c| char_to_byte.get(&c).copied()).collect();
            decoder[*id as usize] = bytes.unwrap_or_else(|| tok.as_bytes().to_vec());
        }
        Ok(Self {
            encoder,
            decoder,
            ranks,
            byte_to_char,
            pattern: Regex::new(GPT2_PATTERN)?,
        })
    }

    /// Charge un `tokenizer.json` Hugging Face (modèle BPE).
    pub fn from_tokenizer_json(path: &Path) -> Result<Self> {
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).context("lecture tokenizer.json")?)?;
        let model = &json["model"];
        let vocab = model["vocab"].as_object().ok_or_else(|| anyhow!("vocab manquant"))?;
        let encoder: HashMap<String, u32> =
            vocab.iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0) as u32)).collect();
        let mut ranks = HashMap::new();
        for (i, m) in model["merges"].as_array().ok_or_else(|| anyhow!("merges manquants"))?.iter().enumerate() {
            let pair = match m {
                serde_json::Value::String(s) => {
                    let mut it = s.splitn(2, ' ');
                    (it.next().unwrap_or("").to_string(), it.next().unwrap_or("").to_string())
                }
                serde_json::Value::Array(a) => (
                    a.first().and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    a.get(1).and_then(|v| v.as_str()).unwrap_or("").to_string(),
                ),
                _ => continue,
            };
            ranks.insert(pair, i as u32);
        }
        Self::build(encoder, ranks)
    }

    /// Charge un `vocab.json` seul : les fusions sont reconstruites à partir
    /// de l'ordre des identifiants.
    pub fn from_vocab_json(path: &Path) -> Result<Self> {
        let encoder: HashMap<String, u32> =
            serde_json::from_slice(&std::fs::read(path).context("lecture vocab.json")?)?;
        let mut by_id: Vec<(&String, u32)> = encoder.iter().map(|(k, v)| (k, *v)).collect();
        by_id.sort_by_key(|(_, id)| *id);
        let mut ranks = HashMap::new();
        for (tok, id) in by_id {
            let chars: Vec<char> = tok.chars().collect();
            if chars.len() < 2 || tok.starts_with("<|") {
                continue;
            }
            // Choisit la découpe dont les deux moitiés existent et sont les plus "anciennes".
            let mut best: Option<(u32, String, String)> = None;
            for i in 1..chars.len() {
                let l: String = chars[..i].iter().collect();
                let r: String = chars[i..].iter().collect();
                if let (Some(&a), Some(&b)) = (encoder.get(&l), encoder.get(&r)) {
                    if a < id && b < id {
                        let score = a.max(b);
                        if best.as_ref().map_or(true, |(s, _, _)| score > *s) {
                            best = Some((score, l, r));
                        }
                    }
                }
            }
            if let Some((_, l, r)) = best {
                ranks.entry((l, r)).or_insert(id);
            }
        }
        Self::build(encoder, ranks)
    }

    pub fn token_id(&self, tok: &str) -> Option<u32> {
        self.encoder.get(tok).copied()
    }

    fn bpe(&self, word: &str, out: &mut Vec<u32>) {
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        loop {
            if parts.len() < 2 {
                break;
            }
            let mut best: Option<(u32, usize)> = None;
            for i in 0..parts.len() - 1 {
                if let Some(&r) = self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) {
                    if best.map_or(true, |(br, _)| r < br) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", parts[i], parts[i + 1]);
            parts[i] = merged;
            parts.remove(i + 1);
        }
        for p in parts {
            if let Some(&id) = self.encoder.get(&p) {
                out.push(id);
            } else {
                // Repli : caractère par caractère
                for c in p.chars() {
                    if let Some(&id) = self.encoder.get(&c.to_string()) {
                        out.push(id);
                    }
                }
            }
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for m in self.pattern.find_iter(text).flatten() {
            let mapped: String = m.as_str().bytes().map(|b| self.byte_to_char[b as usize]).collect();
            self.bpe(&mapped, &mut out);
        }
        out
    }

    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.decoder.get(id as usize).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|i| self.token_bytes(*i).to_vec()).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Décodeur incrémental : renvoie uniquement le texte UTF-8 complet.
#[derive(Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let s = s.to_string();
                self.pending.clear();
                s
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if e.error_len().is_some() {
                    // Séquence invalide : on la remplace.
                    let s = String::from_utf8_lossy(&self.pending).into_owned();
                    self.pending.clear();
                    return s;
                }
                let s = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                self.pending.drain(..valid);
                s
            }
        }
    }
}
