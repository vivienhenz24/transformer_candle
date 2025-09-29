use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct ExponentialMovingAverage {
    alpha: f64,
    value: Option<f64>,
}

impl ExponentialMovingAverage {
    pub fn new(alpha: f64) -> Self {
        Self { alpha, value: None }
    }

    pub fn update(&mut self, sample: f64) -> f64 {
        let v = match self.value {
            Some(prev) => self.alpha * sample + (1.0 - self.alpha) * prev,
            None => sample,
        };
        self.value = Some(v);
        v
    }

    pub fn value(&self) -> Option<f64> {
        self.value
    }
}

#[derive(Debug)]
pub struct TrainingMetrics {
    step_timer: Instant,
    start_time: Instant,
    tokens_processed: u64,
    loss_ema: ExponentialMovingAverage,
    throughput_ema: ExponentialMovingAverage,
    grad_norm_ema: ExponentialMovingAverage,
}

impl TrainingMetrics {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            step_timer: now,
            start_time: now,
            tokens_processed: 0,
            loss_ema: ExponentialMovingAverage::new(0.1),
            throughput_ema: ExponentialMovingAverage::new(0.1),
            grad_norm_ema: ExponentialMovingAverage::new(0.1),
        }
    }

    pub fn record_step(&mut self, tokens: u64, loss: f64, grad_norm: f64) -> StepSnapshot {
        let now = Instant::now();
        let step_duration = now.duration_since(self.step_timer);
        self.step_timer = now;

        self.tokens_processed = self.tokens_processed.saturating_add(tokens);
        let step_tokens_per_sec = if step_duration > Duration::ZERO {
            tokens as f64 / step_duration.as_secs_f64()
        } else {
            0.0
        };
        let loss_avg = self.loss_ema.update(loss);
        let throughput_avg = self.throughput_ema.update(step_tokens_per_sec);
        let grad_norm_avg = self.grad_norm_ema.update(grad_norm);

        StepSnapshot {
            loss: loss_avg,
            step_loss: loss,
            tokens,
            step_tokens_per_sec,
            tokens_per_sec: throughput_avg,
            grad_norm: grad_norm_avg,
            raw_grad_norm: grad_norm,
            total_tokens: self.tokens_processed,
            wall_time: now.duration_since(self.start_time),
            step_duration,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StepSnapshot {
    pub loss: f64,
    pub step_loss: f64,
    pub tokens: u64,
    pub step_tokens_per_sec: f64,
    pub tokens_per_sec: f64,
    pub grad_norm: f64,
    pub raw_grad_norm: f64,
    pub total_tokens: u64,
    pub wall_time: Duration,
    pub step_duration: Duration,
}

#[derive(Debug, Default)]
pub struct EvaluationMetrics {
    loss_sum: f64,
    token_count: u64,
    correct_tokens: u64,
}

impl EvaluationMetrics {
    pub fn update(&mut self, loss: f64, tokens: u64, correct: u64) {
        self.loss_sum += loss * tokens as f64;
        self.token_count += tokens;
        self.correct_tokens += correct;
    }

    pub fn finalize(self) -> Option<EvaluationSummary> {
        if self.token_count == 0 {
            None
        } else {
            let avg_loss = self.loss_sum / self.token_count as f64;
            Some(EvaluationSummary {
                average_loss: avg_loss,
                perplexity: avg_loss.exp(),
                accuracy: (self.correct_tokens as f64) / (self.token_count as f64),
                tokens: self.token_count,
            })
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvaluationSummary {
    pub average_loss: f64,
    pub perplexity: f64,
    pub accuracy: f64,
    pub tokens: u64,
}
