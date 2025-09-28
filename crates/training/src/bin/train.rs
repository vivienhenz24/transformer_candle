use std::env;
use std::process;

use training::{TrainingConfig, TrainingError, Trainer};

fn main() {
    if let Err(err) = run() {
        eprintln!("training failed: {}", err);
        process::exit(1);
    }
}

fn run() -> Result<(), TrainingError> {
    let config_path = env::args().nth(1).ok_or_else(|| {
        TrainingError::initialization("expected path to training config as the first argument")
    })?;

    let config = TrainingConfig::load(&config_path)?;
    let trainer = Trainer::new(config)?;
    trainer.train()?;

    Ok(())
}
