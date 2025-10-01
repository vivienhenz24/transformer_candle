#!/usr/bin/env bash
# End-to-end pipeline: download FineWeb shards, convert to text, train 550M model on RunPod

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

    # ---- Tunables (override via environment) ----
    local run_name="${RUN_NAME:-rtx5090-550m}"
    local run_root="${RUN_DIR:-/workspace/runs/$run_name}"
    local dataset_id="${HF_DATASET_ID:-HuggingFaceFW/fineweb}"
    local dataset_dir="${DATASET_DIR:-/workspace/datasets/fineweb}"
    local shards_dir="${SHARDS_DIR:-/workspace/tmp_streaming_shards}"
    local lines_per_shard="${LINES_PER_SHARD:-600000}"
    local hf_workers="${HF_MAX_WORKERS:-32}"
    local parquet_limit="${HF_PARQUET_LIMIT:-20}"
    local vocab_size="${VOCAB_SIZE:-50000}"
    local tokenizer_max_lines="${TOKENIZER_MAX_LINES:-1000000}"
    local skip_download="${SKIP_DOWNLOAD:-0}"
    local skip_shard_convert="${SKIP_SHARD_CONVERT:-0}"
    local target_shards="${TARGET_SHARDS:-85}"

    log "Run directory: $run_root"
    log "Dataset cache: $dataset_dir"
    log "Shards directory: $shards_dir"

    install_system_packages
    install_rust_toolchain
    install_python_packages
    configure_env_caches

    mkdir -p "$run_root" "$dataset_dir" "$shards_dir" "$run_root/checkpoints" \
        "$run_root/best" "$run_root/tensorboard"

    if [[ "$skip_download" != "1" ]]; then
        download_dataset "$dataset_id" "$dataset_dir" "$hf_workers" "$parquet_limit"
    else
        log "Skipping dataset download (SKIP_DOWNLOAD=1)"
    fi

    if [[ "$skip_shard_convert" != "1" ]]; then
        convert_parquet_to_shards "$dataset_dir" "$shards_dir" "$lines_per_shard" "$target_shards"
    else
        log "Skipping shard conversion (SKIP_SHARD_CONVERT=1)"
    fi

    generate_training_config \
        "$run_root" "$shards_dir" "$vocab_size" "$tokenizer_max_lines"

    build_training_binary

    log "Launching training run"
    cargo run --release -p training --bin train -- \
        --config "$run_root/training.yaml" --resume
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
        datasets==3.0.0 huggingface-hub tokenizers tqdm pyarrow pandas hf_transfer pyyaml
}

configure_env_caches() {
    export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
    export HF_DATASETS_CACHE="$HF_HOME/datasets"
    export HF_HUB_CACHE="$HF_HOME/hub"
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
    mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"
}

download_dataset() {
    local dataset_id="$1"
    local dataset_dir="$2"
    local max_workers="$3"
    local parquet_limit="$4"

    log "Downloading dataset $dataset_id -> $dataset_dir"

    DATASET_ID="$dataset_id" DATASET_DIR="$dataset_dir" HF_WORKERS="$max_workers" PARQUET_LIMIT="$parquet_limit" \
    python3 - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_url, snapshot_download

try:
    import hf_transfer
except ImportError:
    hf_transfer = None

dataset_id = os.environ['DATASET_ID']
local_dir = Path(os.environ['DATASET_DIR'])
max_workers = int(os.environ['HF_WORKERS'])
parquet_limit = int(os.environ.get('PARQUET_LIMIT', '0'))

api = HfApi()
files = [
    file
    for file in api.list_repo_files(dataset_id, repo_type='dataset')
    if file.startswith('data/') and file.endswith('.parquet')
]
files.sort()
if parquet_limit > 0:
    files = files[:parquet_limit]
if not files:
    raise SystemExit('No parquet files found in dataset')

local_dir.mkdir(parents=True, exist_ok=True)

if hf_transfer is not None:
    chunk_size = 32 * 1024 * 1024
    for idx, filename in enumerate(files, 1):
        url = hf_hub_url(dataset_id, filename, repo_type='dataset')
        dest_path = local_dir / filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(dest_path.suffix + '.tmp')
        if dest_path.exists():
            print(f"[{idx}/{len(files)}] Skipping existing {dest_path}")
            continue
        print(f"[{idx}/{len(files)}] hf_transfer downloading {filename}")
        hf_transfer.download(
            url,
            str(tmp_path),
            max_files=max(4, min(max_workers, 64)),
            chunk_size=chunk_size,
        )
        tmp_path.replace(dest_path)
else:
    allow_patterns = files
    snapshot_download(
        repo_id=dataset_id,
        repo_type='dataset',
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        max_workers=max_workers,
        resume_download=True,
    )
PY
}

convert_parquet_to_shards() {
    local dataset_dir="$1"
    local shards_dir="$2"
    local lines_per_shard="$3"
    local target_shards="$4"

    log "Converting parquet to text shards"
    PARQUET_DIR="$dataset_dir" SHARDS_DIR="$shards_dir" LINES_PER_SHARD="$lines_per_shard" TARGET_SHARDS="$target_shards" \
    python3 - <<'PY'
import os
from pathlib import Path
import pyarrow.dataset as ds

DATASET_DIR = Path(os.environ['PARQUET_DIR'])
SHARDS_DIR = Path(os.environ['SHARDS_DIR'])
LINES_PER_SHARD = int(os.environ['LINES_PER_SHARD'])
TARGET_SHARDS = int(os.environ.get('TARGET_SHARDS', '0'))

SHARDS_DIR.mkdir(parents=True, exist_ok=True)
paths = sorted(DATASET_DIR.glob('data/**/*.parquet'))
if not paths:
    raise SystemExit(f"no parquet files found under {DATASET_DIR}")

buffer = []
shard_idx = 0
stop_processing = False

def flush():
    global buffer, shard_idx, stop_processing
    if not buffer:
        return
    shard_path = SHARDS_DIR / f"shard_{shard_idx:04}.txt"
    with shard_path.open('w', encoding='utf-8') as f:
        for line in buffer:
            f.write(line.replace('\n', ' ') + '\n')
    print(f"wrote {shard_path} with {len(buffer)} lines")
    buffer.clear()
    shard_idx += 1
    if TARGET_SHARDS and shard_idx >= TARGET_SHARDS:
        stop_processing = True

def process_file(path: Path):
    dataset = ds.dataset(path, format='parquet')
    for batch in dataset.to_batches(columns=['text'], batch_size=2048):
        for value in batch.column(0):
            text = value.as_py()
            if not text:
                continue
            buffer.append(text)
            if len(buffer) >= LINES_PER_SHARD:
                flush()
                if stop_processing:
                    return
        if stop_processing:
            return

for parquet_path in paths:
    if stop_processing:
        break
    print(f"processing {parquet_path}")
    process_file(parquet_path)

if not stop_processing and buffer:
    flush()

print(f"conversion complete -> total shards: {shard_idx}")
PY
}

generate_training_config() {
    local run_root="$1"
    local shards_dir="$2"
    local vocab_size="$3"
    local tokenizer_max_lines="$4"

    log "Generating training config"
    RUN_ROOT="$run_root" \
    SHARDS_DIR="$shards_dir" \
    python3 <<'PY'
import os
import sys
from pathlib import Path
import yaml

run_root = Path(os.environ['RUN_ROOT'])
shards_dir = Path(os.environ['SHARDS_DIR'])
shards = sorted(str(p) for p in shards_dir.glob('shard_*.txt'))
if not shards:
    sys.stderr.write(f"No shards found in {shards_dir}\n")
    raise SystemExit(1)

config = {
    'model': {
        'hidden_size': 896,
        'intermediate_size': 3584,
        'num_layers': 12,
        'num_attention_heads': 14,
        'num_key_value_heads': 14,
        'max_position_embeddings': 1024,
        'attn_dropout': None,
        'residual_dropout': 0.0,
        'rope_mode': None,
        'rope_theta': None,
    },
    'tokenizer': {
        'tokenizer_json': str(run_root / 'tokenizer' / 'tokenizer.json'),
        'vocab': str(run_root / 'tokenizer' / 'vocab.json'),
        'merges': str(run_root / 'tokenizer' / 'merges.txt'),
        'special_tokens': str(run_root / 'special_tokens.txt'),
    },
    'data': {
        'train_shards': shards,
        'validation_shards': shards,
        'sequence_length': 1024,
        'batch_size': 64,
        'gradient_accumulation_steps': 16,
        'shuffle_buffer_size': 32768,
        'num_workers': 6,
        'cache_dir': str(run_root / 'cache'),
    },
    'optimizer': {
        'algorithm': 'adam_w',
        'learning_rate': 6e-4,
        'weight_decay': 0.1,
        'beta1': 0.9,
        'beta2': 0.95,
        'epsilon': 1e-8,
        'max_grad_norm': 1.0,
    },
    'scheduler': {
        'strategy': 'cosine_with_warmup',
        'warmup_steps': 2000,
        'total_steps': 500000,
        'total_epochs': None,
        'min_lr': 6e-5,
        'max_lr': None,
    },
    'runtime': {
        'seed': 42,
        'precision': 'fp32',
        'log_every_n_steps': 50,
        'checkpoint': {
            'directory': str(run_root / 'checkpoints'),
            'every_n_steps': 5000,
            'every_n_epochs': None,
            'max_keep': 2,
        },
        'evaluation': {
            'every_n_steps': 5000,
            'every_n_epochs': None,
            'max_batches': 20,
            'best': {
                'directory': str(run_root / 'best'),
                'max_keep': 2,
            },
        },
        'logging': {
            'enable_stdout': True,
            'tensorboard': str(run_root / 'tensorboard'),
            'tensorboard_flush_every_n': 100,
        },
    },
}

training_yaml = run_root / 'training.yaml'
run_root.mkdir(parents=True, exist_ok=True)
training_yaml.write_text(yaml.dump(config, sort_keys=False))
print(f"wrote {training_yaml}")
PY

    cat >"$run_root/special_tokens.txt" <<'JSON'
[
  "<|endoftext|>",
  "<|pad|>"
]
JSON

    if [[ -f "$run_root/streaming_config.json" ]]; then
        rm -f "$run_root/streaming_config.json"
        log "Removed legacy streaming_config.json from $run_root"
    fi
}

build_training_binary() {
    log "Building training binary (release)"
    cargo build --release -p training
}

main "$@"
