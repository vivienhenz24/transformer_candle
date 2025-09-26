use crate::config::ByteLevelCfg;
use crate::errors::Result;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

pub fn build_byte_level(cfg: &ByteLevelCfg) -> Result<ByteLevel> {
    let byte_level = ByteLevel::new(cfg.add_prefix_space, cfg.trim_offsets, cfg.use_regex);
    Ok(byte_level)
}

pub fn build_byte_level_decoder() -> ByteLevelDecoder {
    ByteLevelDecoder::default()
}
