#!/usr/bin/env bash
# End-to-end RunPod bootstrap + streaming training for the 550M profile

set -euo pipefail
IFS=$'\n\t'

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

require_root() {
    if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
        echo "This script must run as root inside the RunPod container." >&2
        exit 1
    fi
}

script_dir() {
    local src="${BASH_SOURCE[0]}"
    while [[ -h "$src" ]]; do
        local dir
        dir=$(cd -P "$(dirname "$src")" && pwd)
        src=$(readlink "$src")
        [[ $src != /* ]] && src="$dir/$src"
    done
    cd -P "$(dirname "$src")" && pwd
}

main() {
    require_root

    local repo_dir
    repo_dir=$(script_dir)
    cd "$repo_dir"

    # ---- Tunables ----
    local run_name="${RUN_NAME:-rtx5090-550m}"
    local experiment="${EXPERIMENT:-$run_name}"
    local run_root="${RUN_DIR:-/workspace/runs/$run_name}"
    local dataset="${DATASET:-HuggingFaceFW/fineweb}"
    local split="${DATASET_SPLIT:-train}"
    local vocab_size="${VOCAB_SIZE:-50000}"
    local tokenizer_max_lines="${TOKENIZER_MAX_LINES:-1000000}"
    local stream_batch_size="${STREAM_BATCH_SIZE:-8000}"
    local stream_max_samples="${STREAM_MAX_SAMPLES:-8500000}"
    local skip_tokenizer="${SKIP_TOKENIZER:-0}"
    local cache_root="/workspace/cache/$run_name"
    local hf_cache_root="/workspace/hf_cache"
    local placeholder_shard="/workspace/tmp_streaming_shards/.placeholder"
    local training_config="$run_root/training.yaml"
    local streaming_config="$run_root/streaming_config.json"

    log "Using run directory: $run_root"

    install_system_packages
    install_rust_toolchain
    install_python_packages
    configure_cuda_env

    mkdir -p "$run_root" "$cache_root" "${hf_cache_root}/datasets" "${hf_cache_root}/hub" \
        "$(dirname "$placeholder_shard")"
    touch "$placeholder_shard"

    export HF_HOME="$hf_cache_root"
    export HF_DATASETS_CACHE="${hf_cache_root}/datasets"
    export HF_HUB_CACHE="${hf_cache_root}/hub"
    export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

    if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACEHUB_API_TOKEN:-}" ]]; then
        log "HF_TOKEN is not set. Public datasets only; set HF_TOKEN for gated corpora."
    fi

    build_training_binary

    train_or_reuse_tokenizer \
        "$dataset" "$split" "$vocab_size" "$tokenizer_max_lines" \
        "$stream_batch_size" "$stream_max_samples" \
        "$experiment" "$run_root"

    emit_special_tokens "$run_root/special_tokens.txt"
    assemble_training_config "$training_config" "$run_root" "$cache_root" "$placeholder_shard"

    if [[ ! -f "$streaming_config" ]]; then
        log "Streaming config missing; generating manually."
        write_streaming_config "$streaming_config" "$dataset" "$split" \
            "$stream_max_samples" "$stream_batch_size" "$experiment"
    fi

    log "Launching training run"
    cargo run --release -p training --bin train -- \
        --config "$training_config" --resume
}

install_system_packages() {
    log "Installing system dependencies"
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential pkg-config libssl-dev cmake curl unzip git \
        python3 python3-pip tmux
}

install_rust_toolchain() {
    if ! command -v cargo >/dev/null 2>&1; then
        log "Installing Rust toolchain via rustup"
        curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain nightly
        source "$HOME/.cargo/env"
    else
        source "$HOME/.cargo/env"
    fi
    rustup default nightly
    rustup component add rustfmt clippy >/dev/null 2>&1 || true
}

install_python_packages() {
    log "Installing Python dependencies"
    python3 -m pip install --upgrade pip
    python3 -m pip install --upgrade \
        datasets==3.0.0 huggingface-hub tokenizers tqdm hf_transfer
}

configure_cuda_env() {
    if [[ -d /usr/local/cuda ]]; then
        export CUDA_HOME=/usr/local/cuda
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
        export PATH="/usr/local/cuda/bin:${PATH}"
    fi
}

build_training_binary() {
    log "Building training binary (release)"
    cargo build --release -p training
}

train_or_reuse_tokenizer() {
    local dataset="$1"
    local split="$2"
    local vocab_size="$3"
    local max_lines="$4"
    local stream_batch="$5"
    local stream_max_samples="$6"
    local experiment="$7"
    local run_root="$8"

    log "Preparing tokenizer + artifact layout"
    local cmd=(cargo run --release -p training --bin orchestrate --
        --stream
        --dataset "$dataset"
        --split "$split"
        --vocab-size "$vocab_size"
        --tokenizer-max-lines "$max_lines"
        --stream-batch-size "$stream_batch"
        --batch-size 64
        --grad-accum 16
        --hidden-size 896
        --intermediate-size 3584
        --layers 12
        --heads 14
        --seq-len 1024
        --steps 500000
        --learning-rate 6e-4
        --weight-decay 0.1
        --warmup-steps 2000
        --schedule cosine-with-warmup
        --checkpoint-every 5000
        --max-checkpoints 3
        --evaluate-every 5000
        --eval-batches 20
        --log-every 50
        --shuffle-buffer 32768
        --loader-workers 6
        --precision bf16
        --mode cloud
        --experiment "$experiment"
        --run-dir "$run_root"
    )

    if [[ -n "$stream_max_samples" && "$stream_max_samples" != "0" ]]; then
        cmd+=(--max-samples "$stream_max_samples")
    fi

    if [[ "$skip_tokenizer" == "1" ]]; then
        cmd+=(--skip-tokenizer)
    fi

    "${cmd[@]}"
}

emit_special_tokens() {
    local path="$1"
    log "Writing special tokens artifact: $path"
    cat >"$path" <<'JSON'
[
  "<pad>",
  "<bos>",
  "<eos>"
]
JSON
}

assemble_training_config() {
    local config_path="$1"
    local run_root="$2"
    local cache_root="$3"
    local placeholder="$4"

    log "Generating training configuration: $config_path"
    cat >"$config_path" <<EOF
model:
  hidden_size: 896
  intermediate_size: 3584
  num_layers: 12
  num_attention_heads: 14
  num_key_value_heads: 14
  max_position_embeddings: 1024
  attn_dropout: null
  residual_dropout: 0.0
  rope_mode: null
  rope_theta: null

tokenizer:
  tokenizer_json: "$run_root/tokenizer/tokenizer.json"
  vocab: "$run_root/tokenizer/vocab.json"
  merges: "$run_root/tokenizer/merges.txt"
  special_tokens: "$run_root/special_tokens.txt"

data:
  train_shards:
    - "$placeholder"
  validation_shards:
    - "$placeholder"
  sequence_length: 1024
  batch_size: 64
  gradient_accumulation_steps: 16
  shuffle_buffer_size: 32768
  num_workers: 6
  cache_dir: "$cache_root"

optimizer:
  algorithm: adam_w
  learning_rate: 0.0006
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  epsilon: 1.0e-8
  max_grad_norm: 1.0

scheduler:
  strategy: cosine-with-warmup
  warmup_steps: 2000
  total_steps: 500000
  total_epochs: null
  min_lr: 6.0e-5
  max_lr: null

runtime:
  seed: 42
  precision: bf16
  log_every_n_steps: 50
  checkpoint:
    directory: "$run_root/checkpoints"
    every_n_steps: 5000
    every_n_epochs: null
    max_keep: 3
  evaluation:
    every_n_steps: 5000
    every_n_epochs: null
    max_batches: 20
    best:
      directory: "$run_root/best"
      max_keep: 2
  logging:
    enable_stdout: true
    tensorboard: "$run_root/tensorboard"
    tensorboard_flush_every_n: 100
EOF
}

write_streaming_config() {
    local path="$1"
    local dataset="$2"
    local split="$3"
    local max_samples="$4"
    local batch="$5"
    local experiment="$6"

    log "Creating streaming config: $path"
    local max_value
    if [[ -z "$max_samples" || "$max_samples" == "0" ]]; then
        max_value="null"
    else
        max_value="$max_samples"
    fi

    cat >"$path" <<JSON
{
  "dataset": "$dataset",
  "split": "$split",
  "max_samples": $max_value,
  "stream_batch_size": $batch,
  "experiment": "$experiment"
}
JSON
}

main "$@"
