use std::f64::consts::PI;

use crate::{config, TrainingError};

pub trait LRScheduler: Send {
    fn step(&mut self) -> f64;
    fn learning_rate(&self) -> f64;
}

#[derive(Debug, Clone)]
pub enum SchedulerConfig {
    LinearWarmupCosine {
        base_lr: f64,
        total_steps: usize,
        warmup_proportion: f64,
        min_lr: f64,
    },
    ConstantWithWarmup {
        base_lr: f64,
        total_steps: usize,
        warmup_proportion: f64,
    },
    PolynomialDecay {
        base_lr: f64,
        total_steps: usize,
        warmup_proportion: f64,
        min_lr: f64,
        power: f64,
    },
}

impl SchedulerConfig {
    pub fn from_training_config(
        cfg: &config::SchedulerConfig,
        base_lr: f64,
        total_steps: usize,
    ) -> Result<Self, TrainingError> {
        if total_steps == 0 {
            return Err(TrainingError::initialization(
                "scheduler requires total_steps greater than zero",
            ));
        }

        if let Some(cfg_total) = cfg.total_steps {
            if cfg_total as usize != total_steps {
                return Err(TrainingError::initialization(
                    "scheduler total_steps mismatch between runtime and config",
                ));
            }
        }

        let warmup_steps = cfg.warmup_steps.unwrap_or(0) as usize;
        let warmup_proportion = (warmup_steps as f64 / total_steps as f64).clamp(0.0, 1.0);
        let min_lr = cfg.min_lr.unwrap_or(0.0) as f64;

        match cfg.strategy {
            config::LearningRateSchedule::Constant | config::LearningRateSchedule::LinearWarmup => {
                Ok(SchedulerConfig::ConstantWithWarmup {
                    base_lr,
                    total_steps,
                    warmup_proportion,
                })
            }
            config::LearningRateSchedule::Cosine
            | config::LearningRateSchedule::CosineWithWarmup => {
                Ok(SchedulerConfig::LinearWarmupCosine {
                    base_lr,
                    total_steps,
                    warmup_proportion,
                    min_lr,
                })
            }
            config::LearningRateSchedule::Polynomial => {
                let power = cfg.max_lr.unwrap_or(1.0) as f64;
                Ok(SchedulerConfig::PolynomialDecay {
                    base_lr,
                    total_steps,
                    warmup_proportion,
                    min_lr,
                    power: if power <= 0.0 { 1.0 } else { power },
                })
            }
        }
    }

    pub fn build(self) -> Result<Box<dyn LRScheduler>, TrainingError> {
        match self {
            SchedulerConfig::LinearWarmupCosine {
                base_lr,
                total_steps,
                warmup_proportion,
                min_lr,
            } => Ok(Box::new(LinearWarmupCosine::new(
                base_lr,
                min_lr,
                total_steps,
                warmup_proportion,
            )?)),
            SchedulerConfig::ConstantWithWarmup {
                base_lr,
                total_steps,
                warmup_proportion,
            } => Ok(Box::new(ConstantWithWarmup::new(
                base_lr,
                total_steps,
                warmup_proportion,
            )?)),
            SchedulerConfig::PolynomialDecay {
                base_lr,
                total_steps,
                warmup_proportion,
                min_lr,
                power,
            } => Ok(Box::new(PolynomialDecay::new(
                base_lr,
                min_lr,
                total_steps,
                warmup_proportion,
                power,
            )?)),
        }
    }
}

struct LinearWarmupCosine {
    base_lr: f64,
    min_lr: f64,
    total_steps: usize,
    warmup_steps: usize,
    step: usize,
    current_lr: f64,
}

impl LinearWarmupCosine {
    fn new(
        base_lr: f64,
        min_lr: f64,
        total_steps: usize,
        warmup_proportion: f64,
    ) -> Result<Self, TrainingError> {
        if base_lr <= 0.0 {
            return Err(TrainingError::initialization(
                "scheduler requires base learning rate > 0",
            ));
        }
        if min_lr < 0.0 || min_lr > base_lr {
            return Err(TrainingError::initialization(
                "scheduler min_lr must be in [0, base_lr]",
            ));
        }
        let warmup_steps = compute_warmup_steps(total_steps, warmup_proportion);
        Ok(Self {
            base_lr,
            min_lr,
            total_steps: total_steps.max(1),
            warmup_steps,
            step: 0,
            current_lr: 0.0,
        })
    }

    fn compute_lr(&self, step: usize) -> f64 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            let progress = (step + 1) as f64 / self.warmup_steps as f64;
            return self.base_lr * progress;
        }

        if self.total_steps <= self.warmup_steps {
            return self.base_lr.max(self.min_lr);
        }

        let effective = step.saturating_sub(self.warmup_steps) as f64;
        let denom = (self.total_steps - self.warmup_steps).max(1) as f64;
        let progress = (effective / denom).clamp(0.0, 1.0);
        let cosine = 0.5 * (1.0 + f64::cos(PI * progress));
        self.min_lr + (self.base_lr - self.min_lr) * cosine
    }
}

impl LRScheduler for LinearWarmupCosine {
    fn step(&mut self) -> f64 {
        let lr = self.compute_lr(self.step);
        self.current_lr = lr;
        self.step = self.step.saturating_add(1);
        lr
    }

    fn learning_rate(&self) -> f64 {
        self.current_lr
    }
}

struct ConstantWithWarmup {
    base_lr: f64,
    warmup_steps: usize,
    step: usize,
    current_lr: f64,
}

impl ConstantWithWarmup {
    fn new(
        base_lr: f64,
        total_steps: usize,
        warmup_proportion: f64,
    ) -> Result<Self, TrainingError> {
        if base_lr <= 0.0 {
            return Err(TrainingError::initialization(
                "scheduler requires base learning rate > 0",
            ));
        }
        let warmup_steps = compute_warmup_steps(total_steps, warmup_proportion);
        Ok(Self {
            base_lr,
            warmup_steps,
            step: 0,
            current_lr: 0.0,
        })
    }

    fn compute_lr(&self, step: usize) -> f64 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            let progress = (step + 1) as f64 / self.warmup_steps as f64;
            self.base_lr * progress
        } else {
            self.base_lr
        }
    }
}

impl LRScheduler for ConstantWithWarmup {
    fn step(&mut self) -> f64 {
        let lr = self.compute_lr(self.step);
        self.current_lr = lr;
        self.step = self.step.saturating_add(1);
        lr
    }

    fn learning_rate(&self) -> f64 {
        self.current_lr
    }
}

struct PolynomialDecay {
    base_lr: f64,
    min_lr: f64,
    power: f64,
    total_steps: usize,
    warmup_steps: usize,
    step: usize,
    current_lr: f64,
}

impl PolynomialDecay {
    fn new(
        base_lr: f64,
        min_lr: f64,
        total_steps: usize,
        warmup_proportion: f64,
        power: f64,
    ) -> Result<Self, TrainingError> {
        if base_lr <= 0.0 {
            return Err(TrainingError::initialization(
                "scheduler requires base learning rate > 0",
            ));
        }
        if min_lr < 0.0 || min_lr > base_lr {
            return Err(TrainingError::initialization(
                "scheduler min_lr must be in [0, base_lr]",
            ));
        }
        let warmup_steps = compute_warmup_steps(total_steps, warmup_proportion);
        Ok(Self {
            base_lr,
            min_lr,
            power: if power <= 0.0 { 1.0 } else { power },
            total_steps: total_steps.max(1),
            warmup_steps,
            step: 0,
            current_lr: 0.0,
        })
    }

    fn compute_lr(&self, step: usize) -> f64 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            let progress = (step + 1) as f64 / self.warmup_steps as f64;
            return self.base_lr * progress;
        }

        if self.total_steps <= self.warmup_steps {
            return self.min_lr;
        }

        let denom = (self.total_steps - self.warmup_steps).max(1) as f64;
        let effective = step.saturating_sub(self.warmup_steps) as f64;
        let progress = (effective / denom).clamp(0.0, 1.0);
        let decay = (1.0 - progress).max(0.0).powf(self.power);
        self.min_lr + (self.base_lr - self.min_lr) * decay
    }
}

impl LRScheduler for PolynomialDecay {
    fn step(&mut self) -> f64 {
        let lr = self.compute_lr(self.step);
        self.current_lr = lr;
        self.step = self.step.saturating_add(1);
        lr
    }

    fn learning_rate(&self) -> f64 {
        self.current_lr
    }
}

fn compute_warmup_steps(total_steps: usize, proportion: f64) -> usize {
    if total_steps == 0 {
        return 0;
    }
    let clamped = proportion.clamp(0.0, 1.0);
    let steps = (clamped * total_steps as f64).round() as usize;
    steps.min(total_steps)
}
