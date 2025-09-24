#[derive(Debug, Clone)]
pub struct MicroBatchConfig {
    pub micro_batches: usize,
}

impl Default for MicroBatchConfig {
    fn default() -> Self {
        Self { micro_batches: 1 }
    }
}
