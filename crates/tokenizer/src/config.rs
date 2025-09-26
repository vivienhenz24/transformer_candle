use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub model: ModelCfg,
    pub pretokenizer: ByteLevelCfg,
    pub postprocessor: Option<PostCfg>,
    #[cfg(feature = "train")]
    pub training: Option<TrainingCfg>,
    pub artifacts: ArtifactsCfg,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCfg {
    pub vocab_size: usize,
    pub min_frequency: u32,
    pub dropout: Option<f32>,
    pub special_tokens: Vec<String>,
    pub byte_fallback_on_decode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByteLevelCfg {
    pub add_prefix_space: bool,
    pub trim_offsets: bool,
    pub use_regex: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostCfg {
    pub add_bos: bool,
    pub add_eos: bool,
    pub pair_template: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactsCfg {
    pub dir: PathBuf,
    pub tokenizer_json: Option<PathBuf>,
    pub vocab_json: Option<PathBuf>,
    pub merges_txt: Option<PathBuf>,
    pub manifest: Option<PathBuf>,
}

#[cfg(feature = "train")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCfg {
    pub inputs: Vec<PathBuf>,
    pub seed: u64,
    pub shuffle: bool,
    pub max_lines: Option<usize>,
    pub num_threads: Option<usize>,
}

impl Config {
    pub fn expects_single_file(&self) -> bool {
        self.artifacts.tokenizer_json.is_some()
    }

    pub fn expects_split_files(&self) -> bool {
        self.artifacts.vocab_json.is_some() && self.artifacts.merges_txt.is_some()
    }
}
