#[derive(Debug, Clone)]
pub struct CurriculumStage {
    pub block_size: usize,
    pub batch_size: usize,
}

#[derive(Debug, Clone)]
pub struct CurriculumPlan {
    pub stages: Vec<CurriculumStage>,
}

impl CurriculumPlan {
    pub fn default(block_size: usize, batch_size: usize) -> Self {
        Self {
            stages: vec![
                CurriculumStage {
                    block_size: block_size / 2,
                    batch_size,
                },
                CurriculumStage {
                    block_size,
                    batch_size,
                },
            ],
        }
    }
}
