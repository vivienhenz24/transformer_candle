use std::collections::{HashMap, HashSet};
use std::io::{self, Write};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::TokenizerConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveBpeConfig {
    pub vocab_size: usize,
    pub max_merges: usize,
}

impl From<TokenizerConfig> for AdaptiveBpeConfig {
    fn from(value: TokenizerConfig) -> Self {
        Self {
            vocab_size: value.vocab_size,
            max_merges: value.vocab_size.saturating_sub(256).max(64),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveBpeTokenizer {
    pub token_to_id: HashMap<String, u32>,
    pub id_to_token: Vec<String>,
    merges: Vec<(String, String, String)>,
    unk_token: String,
    pad_token: Option<String>,
}

impl AdaptiveBpeTokenizer {
    pub fn train_from_text(text: &str, tokenizer_config: TokenizerConfig) -> Result<Self> {
        let mut config: AdaptiveBpeConfig = tokenizer_config.clone().into();
        println!(
            "Tokenizer training: preparing character vocabulary from {} chars",
            text.len()
        );
        let _ = io::stdout().flush();
        let tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        let mut working = tokens.clone();

        if working.is_empty() {
            working.push(tokenizer_config.unk_token.clone());
        }

        // Heuristic: limit merge count for very large corpora to keep training tractable.
        if working.len() > 1_000_000 {
            config.max_merges = config.max_merges.min(128);
        } else if working.len() > 200_000 {
            config.max_merges = config.max_merges.min(256);
        } else if working.len() > 100_000 {
            config.max_merges = config.max_merges.min(512);
        }
        println!(
            "Tokenizer training: initial symbols {} | max merges {} | target vocab {}",
            working.len(),
            config.max_merges,
            config.vocab_size
        );
        let _ = io::stdout().flush();

        let mut vocab: HashSet<String> = working.iter().cloned().collect();
        let mut merges = Vec::new();
        let mut merges_added = 0usize;

        while vocab.len() < config.vocab_size && merges_added < config.max_merges {
            let counts = bigram_counts(&working);
            if counts.is_empty() {
                break;
            }
            let (best_pair, _) = counts.into_iter().max_by(|a, b| a.1.cmp(&b.1)).unwrap();
            let new_token = format!("{}{}", best_pair.0, best_pair.1);
            if vocab.contains(&new_token) {
                break;
            }
            working = apply_merge(&working, &best_pair);
            vocab.insert(new_token.clone());
            merges.push((best_pair.0, best_pair.1, new_token));
            merges_added += 1;

            if merges_added % 128 == 0 {
                println!(
                    "Tokenizer training: performed {} merges (target vocab: {})",
                    merges_added, config.vocab_size
                );
                let _ = io::stdout().flush();
            }
        }

        let mut id_to_token = Vec::new();
        let mut token_to_id = HashMap::new();

        if let Some(pad) = &tokenizer_config.pad_token {
            id_to_token.push(pad.clone());
            token_to_id.insert(pad.clone(), 0);
        }

        let unk_id = id_to_token.len() as u32;
        id_to_token.push(tokenizer_config.unk_token.clone());
        token_to_id.insert(tokenizer_config.unk_token.clone(), unk_id);

        for token in vocab.iter() {
            if token_to_id.contains_key(token) {
                continue;
            }
            let id = id_to_token.len() as u32;
            id_to_token.push(token.clone());
            token_to_id.insert(token.clone(), id);
        }

        Ok(Self {
            token_to_id,
            id_to_token,
            merges,
            unk_token: tokenizer_config.unk_token,
            pad_token: tokenizer_config.pad_token,
        })
    }

    pub fn encode_tokens(&self, text: &str) -> Vec<String> {
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        for (left, right, _) in &self.merges {
            tokens = apply_merge(&tokens, &(left.clone(), right.clone()));
        }
        tokens
    }

    pub fn encode_to_ids(&self, text: &str) -> Vec<u32> {
        self.encode_tokens(text)
            .into_iter()
            .map(|token| {
                self.token_to_id
                    .get(&token)
                    .cloned()
                    .unwrap_or_else(|| self.token_to_id[&self.unk_token])
            })
            .collect()
    }

    pub fn encode_to_ids_with_progress(&self, text: &str, chunk: usize) -> Vec<u32> {
        if chunk == 0 {
            return self.encode_to_ids(text);
        }
        let total = text.len();
        let mut tokens = Vec::with_capacity(total);
        let mut start = 0;
        while start < total {
            let end = (start + chunk).min(total);
            let slice = &text[start..end];
            let ids = self.encode_to_ids(slice);
            tokens.extend_from_slice(&ids);
            println!(
                "Tokenizer: encoded {}/{} chars into {} tokens",
                end,
                total,
                tokens.len()
            );
            let _ = io::stdout().flush();
            start = end;
        }
        tokens
    }

    pub fn decode_ids(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            let token = self
                .id_to_token
                .get(id as usize)
                .cloned()
                .unwrap_or_else(|| self.unk_token.clone());
            if Some(token.clone()) == self.pad_token {
                continue;
            }
            out.push_str(&token);
        }
        out
    }
}

fn bigram_counts(tokens: &[String]) -> HashMap<(String, String), usize> {
    let mut counts = HashMap::new();
    for pair in tokens.windows(2) {
        let key = (pair[0].clone(), pair[1].clone());
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

fn apply_merge(tokens: &[String], pair: &(String, String)) -> Vec<String> {
    let mut merged = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if i + 1 < tokens.len() && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
            merged.push(format!("{}{}", pair.0, pair.1));
            i += 2;
        } else {
            merged.push(tokens[i].clone());
            i += 1;
        }
    }
    merged
}
