use anyhow::Result;

use crate::AdvancedTokenizer;

#[derive(Debug, Default)]
pub struct StreamingTokenizer;

impl StreamingTokenizer {
    pub fn encode_stream<'a>(
        &self,
        tokenizer: &'a AdvancedTokenizer,
        chunks: impl Iterator<Item = &'a str>,
    ) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        for chunk in chunks {
            tokens.extend(tokenizer.encode(chunk)?);
        }
        Ok(tokens)
    }
}
