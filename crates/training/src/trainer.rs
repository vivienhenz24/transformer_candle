use crate::{TrainingConfig, TrainingError};

pub struct Trainer {
    config: TrainingConfig,
}

impl Trainer {
    pub fn new(config: TrainingConfig) -> Result<Self, TrainingError> {
        // Placeholder for wiring up model, optimizer, and data sources.
        config.validate()?;
        Ok(Self { config })
    }

    pub fn train(&self) -> Result<(), TrainingError> {
        // Placeholder for the main training loop.
        println!(
            "Starting training with batch size {} and learning rate {}",
            self.config.data.batch_size, self.config.optimizer.learning_rate
        );
        Ok(())
    }

    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }
}
