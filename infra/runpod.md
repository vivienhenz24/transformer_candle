# RunPod Setup

This repository now ships a RunPod oriented configuration that targets the ~2B parameter decoder-only model. Use the steps below to stand up a cloud pod and launch training with the `configs/runpod_2b.yaml` profile.

## Prerequisites
- RunPod account with access to A100 80 GB class GPUs (the configuration assumes at least one 80 GB card).
- Ensure the pod type you launch exposes an A100 80 GB; the `runpod_2b` profile will not fit comfortably on smaller GPUs.
- `runpodctl` installed and authenticated (`pip install runpodctl` or download the binary, then `runpodctl login`).
- A RunPod volume provisioned (≥1 TB recommended) to hold the repository, tokenizer artifacts, datasets, and checkpoints. The sample manifests use the name `transformer-shared`.

## Prepare the Workspace
1. Clone this repository into the mounted volume so that it lives at `/workspace/transformer` inside the pod.
2. Place tokenizer assets under `/workspace/artifacts/tokenizer/` to match the paths referenced by `configs/runpod_2b.yaml`.
3. Stage your training shards under `/workspace/datasets/train/` and validation shards under `/workspace/datasets/val/`. Update the YAML if your filenames differ.
4. (Optional) Pre-create `/workspace/cache`, `/workspace/hf_cache`, and `/workspace/runs` if you want deterministic mount permissions.

## Launching a Pod
1. Review `infra/runpod-pod.yaml` and update:
   - `volumeMounts[0].volumeName` to your actual RunPod volume ID.
   - `gpuType`, CPU, and memory requests to match the instance type you intend to rent.
   - The shell command block if you plan to invoke a different binary or add environment setup.
2. Create the pod: `runpodctl create pod -f infra/runpod-pod.yaml`.
3. Inspect status: `runpodctl get pods` and wait for the pod to enter the `RUNNING` state.
4. (Optional) Port-forward TensorBoard once training starts: `runpodctl port-forward <pod-id> 6006:6006`.

## Pod Lifecycle Overview
`infra/runpod-pod.yaml` orchestrates a four-stage boot sequence:
1. **Directory prep** – creates `/workspace/{datasets,artifacts,cache,runs,hf_cache}` and experiment-specific subdirectories if missing.
2. **Optional shard streaming** – when `RUN_SHARD=1`, the pod invokes `fineweb_sharder.py` to materialise new FineWeb shards using `SHARD_LINES` (default 250k lines) into `/workspace/datasets/{train,val}/$RUN_EXPERIMENT`.
3. **Tokenizer orchestration** – runs the Rust orchestrator in cloud mode to reuse or train the tokenizer and emit `training.toml`/`training.yaml` for the selected experiment. Existing tokenizer bundles under `/workspace/artifacts/tokenizer/$RUN_EXPERIMENT` are re-used automatically.
4. **Training launch** – starts `cargo run -p training --bin train` with `RUN_CONFIG` (defaults to `configs/runpod_2b.yaml`).

Environment knobs:
- `RUN_SHARD=1` to re-stream FineWeb before training; leave at `0` to reuse existing shards.
- `RUN_EXPERIMENT` controls experiment naming for datasets/artifacts/runs (default `runpod-2b`).
- `RUN_CONFIG` selects the training profile (`configs/runpod_2b.yaml` by default; swap for `configs/runpod_4b.yaml` as needed).
- `SHARD_LINES` tunes the shard size passed to the sharder and orchestrator (default `250000`).

## Training Commands Inside the Pod
The pod spec invokes the trainer automatically. If you prefer to attach manually:
1. `runpodctl exec -it <pod-id> -- bash`.
2. `cd /workspace/transformer`.
3. `cargo run -p training --bin train -- --config configs/runpod_2b.yaml`.
## Cloud Pipeline Entrypoint
You can hand control of dataset sharding, tokenizer preparation, and training launch to the orchestrator's new cloud mode. It standardizes the RunPod volume layout:

```
/workspace/datasets/train/<experiment>
/workspace/datasets/val/<experiment>
/workspace/artifacts/tokenizer/<experiment>
/workspace/runs/<experiment>
/workspace/cache/<experiment>
/workspace/hf_cache
```

Invoke it with:

```
cargo run -p training --bin orchestrate -- \
  --mode cloud \
  --input /workspace/raw_corpus.txt \
  --experiment runpod-2b \
  --train-shard-lines 250000
```

Pass `--validation` if you have a separate evaluation corpus, `--reuse-shards` to keep existing shard folders, or provide `--tokenizer-json` to reuse a published artifact. The orchestrator writes both `training.toml` and `training.yaml` under `/workspace/runs/<experiment>` before starting the trainer.

## FineWeb Ingestion
Use the FineWeb sharder to hydrate `/workspace/datasets/{train,val}` directly on the pod. The utility streams the dataset and rolls newline-delimited shards on disk while keeping Hugging Face caches isolated under `/workspace/hf_cache`.

```
python crates/pretraining-data/scripts/fineweb_sharder.py \
  --mode smoke \
  --split train \
  --lines-per-shard 50000 \
  --take 100000
```

Modes:
- `smoke` limits ingestion to a small sample (override with `--take`).
- `prod` streams the full split until interrupted.
- `split` routes each line to validation with probability `--val-ratio` (e.g. `0.01`).

Add `--gzip` to emit `.txt.gz` shards. Each output directory receives a `manifest.json` summarizing shard counts, line totals, and byte sizes for quick bookkeeping.

### Hugging Face Credentials & Cache
- Set `HF_HOME=/workspace/hf_cache` and `HF_DATASETS_CACHE=/workspace/hf_cache/datasets` to keep metadata on the shared volume. The orchestrator now enforces these defaults when running in cloud mode.
- Provide an access token via `HUGGINGFACEHUB_API_TOKEN` (preferred) or `HF_TOKEN`. A `.env` sourced before launching the pod works well: `export HF_TOKEN=...`.

### Config Selection Cheat Sheet
- `configs/m3_medium.yaml`: 22M parameter reference for local smoke tests or CPU/GPU dev boxes.
- `configs/runpod_1b.yaml`: default single-GPU cloud target (~1B params on A100 80 GB) with faster turnaround.
- `configs/runpod_2b.yaml`: larger single-GPU cloud training target (~2B params) when you need more capacity.
- `configs/runpod_4b.yaml`: experimental large-model configuration for future multi-GPU scaling; expect slow iterations on a single card.

### Quickstart
1. **Clone & configure** the repo on your RunPod volume (`/workspace/transformer`).
2. **Edit `infra/runpod-pod.yaml`**:
   - Set the correct `volumeName`.
   - (Optional) Toggle `RUN_SHARD=1` for initial dataset ingestion.
   - Adjust `RUN_EXPERIMENT`, `RUN_CONFIG`, or `SHARD_LINES` to suit your run.
3. **Launch the pod**: `runpodctl create pod -f infra/runpod-pod.yaml`.
4. **Monitor**:
   - `runpodctl get pods` for status.
   - `runpodctl logs <pod-id>` for orchestrator/train output.
   - `runpodctl port-forward <pod-id> 6006:6006` for TensorBoard at `/workspace/runs/<experiment>/tensorboard`.
5. **Pause or delete** idle pods (`runpodctl delete pod <pod-id>`) to avoid unnecessary spend. Confirm shards exist (`ls /workspace/datasets/train/<experiment>`) before resuming long runs.

The manifest defaults to the `runpod_1b` configuration; update `RUN_CONFIG`/`RUN_EXPERIMENT` if you want to step up to the 2B or 4B profiles.

## Configuration Notes
- `configs/runpod_2b.yaml` builds a 1.97B parameter model (hidden size 2560, 24 layers, 40 heads) which fits on a single A100 80 GB when training in `bf16` with gradient accumulation (`batch_size=2`, `gradient_accumulation_steps=64`). Adjust these numbers if you scale pods up or down.
- `configs/runpod_4b.yaml` targets a larger ~4B parameter stack (hidden size 3072, 32 layers, 48 heads). Single A100 runs require aggressive accumulation (`batch_size=1`, `gradient_accumulation_steps=256`) and deliver low throughput; the profile is provided for future multi-GPU scale outs.
- Checkpointing and TensorBoard artifacts land under `/workspace/runs/runpod-2b/`; they remain on the mounted volume after the pod stops.
- The scheduler is configured for a long cosine schedule (`total_steps=400000`). Shorten this for smoke tests to avoid unnecessary GPU spend.
- Always confirm shard paths resolve before launching long runs (`ls /workspace/datasets/train`). Missing shards cause the configuration loader to abort.

## Cost and Monitoring Tips
- Pause or delete the pod when idle; RunPod bills while the GPU is allocated.
- Tail logs with `runpodctl logs <pod-id>` to watch training progress, and port-forward TensorBoard (`runpodctl port-forward <pod-id> 6006:6006`) for metrics.
- Enable RunPod metrics streaming or attach `nvidia-smi --loop=5` in another shell to watch memory usage, especially during the first optimizer step.
- Before launching long runs, sanity check shard contents (`ls /workspace/datasets/train/${RUN_EXPERIMENT}`) so the loader does not abort on missing or empty files.

Start with the 1B profile to validate throughput and loss behaviour, then scale to the 2B/4B configs once you're confident in data quality and spend.
