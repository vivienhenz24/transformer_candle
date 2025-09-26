use crate::config::ByteLevelCfg;
use crate::errors::Result;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

pub fn build_byte_level(cfg: &ByteLevelCfg) -> Result<ByteLevel> {
    let byte_level = ByteLevel::new(cfg.add_prefix_space, cfg.trim_offsets, cfg.use_regex);
    Ok(byte_level)
}

pub fn name() -> &'static str {
    "byte-level"
}
