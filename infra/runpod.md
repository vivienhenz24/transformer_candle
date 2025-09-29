# RunPod Setup

This repository now ships a RunPod oriented configuration that targets the ~2B parameter decoder-only model. Use the steps below to stand up a cloud pod and launch training with the `configs/runpod_2b.yaml` profile.

## Prerequisites
- RunPod account with access to A100 80 GB class GPUs (the configuration assumes at least one 80 GB card).
- `runpodctl` installed and authenticated (`pip install runpodctl` or download the binary, then `runpodctl login`).
- A RunPod volume provisioned (≥1 TB recommended) to hold the repository, tokenizer artifacts, datasets, and checkpoints. The sample manifests use the name `transformer-shared`.

## Prepare the Workspace
1. Clone this repository into the mounted volume so that it lives at `/workspace/transformer` inside the pod.
2. Place tokenizer assets under `/workspace/artifacts/tokenizer/` to match the paths referenced by `configs/runpod_2b.yaml`.
3. Stage your training shards under `/workspace/datasets/train/` and validation shards under `/workspace/datasets/val/`. Update the YAML if your filenames differ.
4. (Optional) Pre-create `/workspace/cache` and `/workspace/runs` if you want deterministic mount permissions.

## Launching a Pod
1. Review `infra/runpod-pod.yaml` and update:
   - `volumeMounts[0].volumeName` to your actual RunPod volume ID.
   - `gpuType`, CPU, and memory requests to match the instance type you intend to rent.
   - The shell command block if you plan to invoke a different binary or add environment setup.
2. Create the pod: `runpodctl create pod -f infra/runpod-pod.yaml`.
3. Inspect status: `runpodctl get pods` and wait for the pod to enter the `RUNNING` state.
4. (Optional) Port-forward TensorBoard once training starts: `runpodctl port-forward <pod-id> 6006:6006`.

## Training Commands Inside the Pod
The pod spec invokes the trainer automatically. If you prefer to attach manually:
1. `runpodctl exec -it <pod-id> -- bash`.
2. `cd /workspace/transformer`.
3. `cargo run -p training --bin train -- --config configs/runpod_2b.yaml`.

## Configuration Notes
- `configs/runpod_2b.yaml` builds a 1.97B parameter model (hidden size 2560, 24 layers, 40 heads) which fits on a single A100 80 GB when training in `bf16` with gradient accumulation (`batch_size=2`, `gradient_accumulation_steps=64`). Adjust these numbers if you scale pods up or down.
- Checkpointing and TensorBoard artifacts land under `/workspace/runs/runpod-2b/`; they remain on the mounted volume after the pod stops.
- The scheduler is configured for a long cosine schedule (`total_steps=400000`). Shorten this for smoke tests to avoid unnecessary GPU spend.
- Always confirm shard paths resolve before launching long runs (`ls /workspace/datasets/train`). Missing shards cause the configuration loader to abort.

## Cost and Monitoring Tips
- Pause or delete the pod when idle; RunPod bills while the GPU is allocated.
- Enable RunPod metrics streaming or attach `nvidia-smi --loop=5` in another shell to watch memory usage, especially during the first optimizer step.
- TensorBoard (`runpodctl port-forward`) and the trainer stdout logs (`runpodctl logs <pod-id>`) remain the quickest way to monitor loss curves remotely.
