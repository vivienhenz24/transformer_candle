# Training Crate

This crate orchestrates end-to-end transformer pre-training using the shared
components from the workspace. It wires together data streaming, tokenizer
loading, model instantiation, optimization, scheduling, metrics, logging, and
checkpoint management.

## Example Configuration

```toml
[tokenizer]
tokenizer_json = "./artifacts/tokenizer.json"
special_tokens = "./artifacts/special_tokens.txt"

[data]
train_shards = ["./data/train.txt"]
validation_shards = ["./data/valid.txt"]
batch_size = 8
gradient_accumulation_steps = 1
shuffle_buffer_size = 128

[optimizer]
learning_rate = 0.0003
beta1 = 0.9
beta2 = 0.95
epsilon = 1e-8

[scheduler]
strategy = "constant"
total_steps = 1000

[runtime]
seed = 42
log_every_n_steps = 10

[runtime.checkpoint]
directory = "./checkpoints"
every_n_steps = 100
max_keep = 3

[runtime.evaluation]
every_n_steps = 200
max_batches = 2

[runtime.evaluation.best]
directory = "./checkpoints/best"
max_keep = 1

[runtime.logging]
enable_stdout = true
tensorboard = "./runs/train"
tensorboard_flush_every_n = 20

[model]
hidden_size = 512
intermediate_size = 2048
num_layers = 6
num_attention_heads = 8
max_position_embeddings = 512
```

## CLI Usage

The `training` crate ships a `train` binary that consumes configuration files
and runs the orchestrated trainer:

```bash
cargo run -p training --bin train -- --config config.toml
```

You can override individual configuration keys via `--override` using
`dot.separated.paths` (array indices are supported):

```bash
cargo run -p training --bin train -- \
  --config config.toml \
  --override runtime.log_every_n_steps=5 \
  --override optimizer.learning_rate=0.0001 \
  --override scheduler.total_steps=500 \
  --override data.train_shards[0]=./alt/train.txt
```

Passing `--resume` will restart training from the latest checkpoint (when
available). Checkpoints contain model weights, optimizer state, scheduler
progress, gradient-scaler state, progress metadata, and integrity hashes. Best
checkpoints can be maintained independently using the evaluation configuration.

## Expected Artifacts

* `checkpoints/step_*`: versioned checkpoints including manifests, optimizer
  state, scaler state, and `safetensors` weights.
* `checkpoints/best/step_*`: optional best checkpoints tracked by perplexity.
* `runs/train/events.out.tfevents.*`: TensorBoard event files containing
  smoothed training metrics and evaluation summaries.

## Metrics & Evaluation

Training logs include running loss, throughput, gradient norms, and learning
rates. Evaluations are performed on held-out shards without gradient updates,
reporting average loss, perplexity, and accuracy. Evaluation cadence is
controlled via `runtime.evaluation.every_n_steps` or
`runtime.evaluation.every_n_epochs` and may be limited via `max_batches`.

## Smoke Test

A synthetic integration test (`tests/smoke.rs`) exercises tokenizer training,
model setup, short training runs, checkpoint creation, resume flow, and basic
loss improvement. Use it as inspiration for constructing additional end-to-end
scenarios.
