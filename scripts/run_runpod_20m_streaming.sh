#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_FILE="$REPO_ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

PROFILE="${MODEL_PROFILE:-20m}"

case "$PROFILE" in
  20m)
    DEFAULT_RUN_NAME="runpod-20m-streaming"
    CONFIG_TEMPLATE="$REPO_ROOT/configs/runpod_20m_streaming.yaml"
    STREAM_TEMPLATE="$REPO_ROOT/configs/runpod_20m_streaming_config.json"
    STREAM_BATCH_SIZE="${STREAM_BATCH_SIZE:-512}"
    MAX_STREAM_SAMPLES="${MAX_STREAM_SAMPLES:-600000}"
    TOKENIZER_MAX_LINES="${TOKENIZER_MAX_LINES:-600000}"
    HIDDEN_SIZE=256
    INTERMEDIATE_SIZE=1024
    NUM_LAYERS=16
    NUM_HEADS=8
    BASE_GLOBAL_BATCH=64
    BASE_GRAD_ACCUM=16
    BASE_TOTAL_STEPS=6400
    BASE_WARMUP_STEPS=800
    BASE_CHECKPOINT_EVERY=400
    BASE_EVALUATE_EVERY=800
    LOG_EVERY=10
    SHUFFLE_BUFFER=16384
    LOADER_WORKERS=8
    DATASET="${DATASET:-HuggingFaceFW/fineweb}"
    SPLIT="${SPLIT:-train}"
    VOCAB_SIZE="${VOCAB_SIZE:-32000}"
    GLOBAL_BATCH=$BASE_GLOBAL_BATCH
    GRAD_ACCUM=$BASE_GRAD_ACCUM
    TOTAL_STEPS=$BASE_TOTAL_STEPS
    WARMUP_STEPS=$BASE_WARMUP_STEPS
    CHECKPOINT_EVERY=$BASE_CHECKPOINT_EVERY
    EVALUATE_EVERY=$BASE_EVALUATE_EVERY
    ;;
  400m)
    DEFAULT_RUN_NAME="runpod-400m-streaming"
    CONFIG_TEMPLATE="$REPO_ROOT/configs/runpod_400m_streaming.yaml"
    STREAM_TEMPLATE="$REPO_ROOT/configs/runpod_400m_streaming_config.json"
    STREAM_BATCH_SIZE="${STREAM_BATCH_SIZE:-512}"
    MAX_STREAM_SAMPLES="${MAX_STREAM_SAMPLES:-2500000}"
    TOKENIZER_MAX_LINES="${TOKENIZER_MAX_LINES:-1200000}"
    HIDDEN_SIZE=1024
    INTERMEDIATE_SIZE=4096
    NUM_LAYERS=30
    NUM_HEADS=16
    BASE_GLOBAL_BATCH=48
    BASE_GRAD_ACCUM=48
    BASE_TOTAL_STEPS=20000
    BASE_WARMUP_STEPS=2000
    BASE_CHECKPOINT_EVERY=1000
    BASE_EVALUATE_EVERY=2000
    LOG_EVERY=20
    SHUFFLE_BUFFER=65536
    LOADER_WORKERS=8
    DATASET="${DATASET:-HuggingFaceFW/fineweb}"
    SPLIT="${SPLIT:-train}"
    VOCAB_SIZE="${VOCAB_SIZE:-32000}"
    GLOBAL_BATCH=$BASE_GLOBAL_BATCH
    GRAD_ACCUM=$BASE_GRAD_ACCUM
    TOTAL_STEPS=$BASE_TOTAL_STEPS
    WARMUP_STEPS=$BASE_WARMUP_STEPS
    CHECKPOINT_EVERY=$BASE_CHECKPOINT_EVERY
    EVALUATE_EVERY=$BASE_EVALUATE_EVERY
    ;;
  *)
    echo "Unsupported MODEL_PROFILE '$PROFILE'. Supported profiles: 20m, 400m." >&2
    exit 1
    ;;
esac

if [[ "$PROFILE" == "400m" ]]; then
  GPU_MEM_TOTAL=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 || true)
  fi

  if [[ -n "$GPU_MEM_TOTAL" ]]; then
    if [[ "$GPU_MEM_TOTAL" =~ ^[0-9]+$ ]]; then
      if (( GPU_MEM_TOTAL < 24000 )); then
        echo "Detected GPU memory ${GPU_MEM_TOTAL} MiB < 24 GiB; the 400M profile requires at least 24 GiB." >&2
        echo "Please switch to the 20M profile or attach a larger GPU." >&2
        exit 1
      elif (( GPU_MEM_TOTAL < 36000 )); then
        GLOBAL_BATCH=16
        GRAD_ACCUM=16
      elif (( GPU_MEM_TOTAL < 64000 )); then
        GLOBAL_BATCH=24
        GRAD_ACCUM=24
      else
        GLOBAL_BATCH=$BASE_GLOBAL_BATCH
        GRAD_ACCUM=$BASE_GRAD_ACCUM
      fi
    fi
  fi

  if (( GLOBAL_BATCH != BASE_GLOBAL_BATCH )); then
    TOTAL_STEPS=$(( BASE_TOTAL_STEPS * BASE_GLOBAL_BATCH / GLOBAL_BATCH ))
    WARMUP_STEPS=$(( BASE_WARMUP_STEPS * BASE_GLOBAL_BATCH / GLOBAL_BATCH ))
    CHECKPOINT_EVERY=$(( BASE_CHECKPOINT_EVERY * BASE_GLOBAL_BATCH / GLOBAL_BATCH ))
    EVALUATE_EVERY=$(( BASE_EVALUATE_EVERY * BASE_GLOBAL_BATCH / GLOBAL_BATCH ))
    echo "Adjusted 400M profile for ${GPU_MEM_TOTAL:-unknown} MiB GPU: global_batch=${GLOBAL_BATCH}, grad_accum=${GRAD_ACCUM}, total_steps=${TOTAL_STEPS}" >&2
  fi
fi

RUN_NAME="${DEFAULT_RUN_NAME}"
declare -a TRAIN_ARGS=()
if [[ $# -gt 0 && "$1" != "--" && "$1" != -* ]]; then
  RUN_NAME="$1"
  shift
fi
if [[ $# -gt 0 && "$1" == "--" ]]; then
  shift
fi
TRAIN_ARGS=("$@")

RUNS_BASE="${RUNS_BASE:-/workspace/runs}"
RUN_ROOT="$RUNS_BASE/$RUN_NAME"
TOKENIZER_DIR="$RUN_ROOT/tokenizer"
DATA_DIR="$RUN_ROOT/data"
CACHE_DIR="$RUN_ROOT/cache"
CHECKPOINT_DIR="$RUN_ROOT/checkpoints"
BEST_DIR="$RUN_ROOT/best"
TB_DIR="$RUN_ROOT/tensorboard"

VOCAB_SIZE="${VOCAB_SIZE:-32000}"

BASE_PYTHON="${BASE_PYTHON:-}"
if [[ -z "$BASE_PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    BASE_PYTHON="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    BASE_PYTHON="$(command -v python)"
  else
    echo "Python 3 is required but was not found on PATH." >&2
    exit 1
  fi
fi

VENV_DIR="${PYTHON_VENV_DIR:-$REPO_ROOT/.venv}"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating Python virtual environment at $VENV_DIR"
  "$BASE_PYTHON" -m venv "$VENV_DIR"
fi

export PATH="$VENV_DIR/bin:$PATH"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python3}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$VENV_DIR/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$BASE_PYTHON"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Failed to locate usable python interpreter" >&2
  exit 1
fi

echo "Using python interpreter: $PYTHON_BIN"

DEPS_SENTINEL="$VENV_DIR/.deps-installed"
if [[ ! -f "$DEPS_SENTINEL" ]]; then
  echo "Installing Python dependencies (datasets, huggingface_hub)"
  "$PYTHON_BIN" -m pip install --upgrade pip >/dev/null
  "$PYTHON_BIN" -m pip install --upgrade datasets huggingface_hub pyyaml >/dev/null
  touch "$DEPS_SENTINEL"
else
  if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util
missing = [m for m in ("datasets", "huggingface_hub", "yaml") if importlib.util.find_spec(m) is None]
raise SystemExit(0 if not missing else 1)
PY
  then
    echo "Updating Python dependencies (datasets, huggingface_hub)"
    "$PYTHON_BIN" -m pip install --upgrade datasets huggingface_hub pyyaml >/dev/null
  fi
fi

if [[ "${HF_HUB_ENABLE_HF_TRANSFER:-}" == "1" ]]; then
  if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("hf_transfer") else 1)
PY
  then
    echo "Installing Python dependency: hf_transfer (required for HF_HUB_ENABLE_HF_TRANSFER=1)"
    "$PYTHON_BIN" -m pip install --upgrade hf_transfer >/dev/null
  fi
fi

export PYTHON_BIN

if [[ ! -f "$CONFIG_TEMPLATE" ]]; then
  echo "Expected template config at $CONFIG_TEMPLATE" >&2
  exit 1
fi
if [[ ! -f "$STREAM_TEMPLATE" ]]; then
  echo "Expected streaming template at $STREAM_TEMPLATE" >&2
  exit 1
fi

if command -v huggingface-cli >/dev/null 2>&1; then
  if ! huggingface-cli whoami >/dev/null 2>&1; then
    echo "Hugging Face CLI is not logged in. Run 'huggingface-cli login' if the dataset requires auth." >&2
  fi
else
  if [[ -z "${HUGGINGFACEHUB_API_TOKEN:-}" ]]; then
    echo "huggingface-cli not found on PATH. Install it or export HUGGINGFACEHUB_API_TOKEN if FineWeb access requires auth." >&2
  fi
fi

echo "Preparing run directory at $RUN_ROOT"
mkdir -p "$TOKENIZER_DIR" "$DATA_DIR" "$CACHE_DIR" "$CHECKPOINT_DIR" "$BEST_DIR" "$TB_DIR"

touch "$DATA_DIR/train_placeholder.txt" "$DATA_DIR/val_placeholder.txt"

TOKENIZER_JSON="$TOKENIZER_DIR/tokenizer.json"

if [[ ! -f "$TOKENIZER_JSON" ]]; then
  echo "Training tokenizer and preparing streaming shards"
  ORCH_ROOT="$RUN_ROOT/.orchestrate"
  mkdir -p "$ORCH_ROOT"
  export ORCHESTRATE_RUNS_ROOT="$RUNS_BASE"
  export ORCHESTRATE_CACHE_ROOT="$ORCH_ROOT/cache"
  export ORCHESTRATE_TOKENIZER_ROOT="$ORCH_ROOT/tokenizers"
  export ORCHESTRATE_TRAIN_ROOT="$ORCH_ROOT/train"
  export ORCHESTRATE_VAL_ROOT="$ORCH_ROOT/val"
  export ORCHESTRATE_HF_CACHE_ROOT="$ORCH_ROOT/hf-cache"

  mkdir -p "$ORCHESTRATE_CACHE_ROOT" "$ORCHESTRATE_TOKENIZER_ROOT" "$ORCHESTRATE_HF_CACHE_ROOT"

  declare -a ORCH_ARGS=(
    --stream
    --dataset "$DATASET"
    --split "$SPLIT"
    --stream-batch-size "$STREAM_BATCH_SIZE"
    --run-dir "$RUN_ROOT"
    --experiment "$RUN_NAME"
    --vocab-size "$VOCAB_SIZE"
    --tokenizer-max-lines "$TOKENIZER_MAX_LINES"
    --hidden-size "$HIDDEN_SIZE"
    --intermediate-size "$INTERMEDIATE_SIZE"
    --layers "$NUM_LAYERS"
    --heads "$NUM_HEADS"
    --seq-len 1024
    --batch-size "$GLOBAL_BATCH"
    --grad-accum "$GRAD_ACCUM"
    --steps "$TOTAL_STEPS"
    --learning-rate 3e-4
    --weight-decay 0.05
    --warmup-steps "$WARMUP_STEPS"
    --schedule cosine-with-warmup
    --checkpoint-every "$CHECKPOINT_EVERY"
    --max-checkpoints 5
    --evaluate-every "$EVALUATE_EVERY"
    --eval-batches 4
    --log-every "$LOG_EVERY"
    --shuffle-buffer "$SHUFFLE_BUFFER"
    --loader-workers "$LOADER_WORKERS"
    --precision fp32
    --seed 42
    --tokenizer-seed 42
  )

  if [[ -n "$MAX_STREAM_SAMPLES" ]]; then
    ORCH_ARGS+=(--max-samples "$MAX_STREAM_SAMPLES")
  fi

  cargo run --release -p training --no-default-features --features cuda --bin orchestrate -- "${ORCH_ARGS[@]}"

  unset ORCHESTRATE_RUNS_ROOT ORCHESTRATE_CACHE_ROOT ORCHESTRATE_TOKENIZER_ROOT ORCHESTRATE_TRAIN_ROOT ORCHESTRATE_VAL_ROOT ORCHESTRATE_HF_CACHE_ROOT
else
  echo "Found existing tokenizer at $TOKENIZER_JSON; skipping tokenizer training"
fi

if [[ ! -f "$TOKENIZER_JSON" ]]; then
  echo "Tokenizer artifact missing at $TOKENIZER_JSON after orchestration" >&2
  exit 1
fi

cat >"$TOKENIZER_DIR/special_tokens.txt" <<'JSON'
[
  "<|endoftext|>",
  "<|pad|>"
]
JSON

BASE_PREFIX="/workspace/runs/${DEFAULT_RUN_NAME}"
RUN_ROOT_ABS="$(cd "$RUN_ROOT" && pwd)"

export CONFIG_ENV_TEMPLATE="$CONFIG_TEMPLATE"
export CONFIG_ENV_OUTPUT="$RUN_ROOT/training.yaml"
export CONFIG_ENV_BASE="$BASE_PREFIX"
export CONFIG_ENV_TARGET="$RUN_ROOT_ABS"
export CONFIG_ENV_BATCH="$GLOBAL_BATCH"
export CONFIG_ENV_GRAD_ACCUM="$GRAD_ACCUM"
export CONFIG_ENV_TOTAL_STEPS="$TOTAL_STEPS"
export CONFIG_ENV_WARMUP_STEPS="$WARMUP_STEPS"
export CONFIG_ENV_CHECKPOINT_EVERY="$CHECKPOINT_EVERY"
export CONFIG_ENV_EVALUATE_EVERY="$EVALUATE_EVERY"
export CONFIG_ENV_LOG_EVERY="$LOG_EVERY"
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
import yaml

template_path = Path(os.environ["CONFIG_ENV_TEMPLATE"])
output_path = Path(os.environ["CONFIG_ENV_OUTPUT"])
base = os.environ["CONFIG_ENV_BASE"]
target = os.environ["CONFIG_ENV_TARGET"]
cfg = yaml.safe_load(template_path.read_text())

def replace_path(value):
    if isinstance(value, str):
        return value.replace(base, target)
    return value

cfg["tokenizer"]["tokenizer_json"] = replace_path(cfg["tokenizer"]["tokenizer_json"])
cfg["tokenizer"]["special_tokens"] = replace_path(cfg["tokenizer"]["special_tokens"])

cfg["data"]["train_shards"] = [replace_path(path) for path in cfg["data"]["train_shards"]]
cfg["data"]["validation_shards"] = [replace_path(path) for path in cfg["data"]["validation_shards"]]
cfg["data"]["cache_dir"] = replace_path(cfg["data"].get("cache_dir"))

cfg["runtime"]["checkpoint"]["directory"] = replace_path(cfg["runtime"]["checkpoint"]["directory"])
cfg["runtime"]["evaluation"]["best"]["directory"] = replace_path(cfg["runtime"]["evaluation"]["best"]["directory"])
cfg["runtime"]["logging"]["tensorboard"] = replace_path(cfg["runtime"]["logging"]["tensorboard"])

cfg["data"]["batch_size"] = int(os.environ["CONFIG_ENV_BATCH"])
cfg["data"]["gradient_accumulation_steps"] = int(os.environ["CONFIG_ENV_GRAD_ACCUM"])
cfg["scheduler"]["total_steps"] = int(os.environ["CONFIG_ENV_TOTAL_STEPS"])
cfg["scheduler"]["warmup_steps"] = int(os.environ["CONFIG_ENV_WARMUP_STEPS"])
cfg["runtime"]["checkpoint"]["every_n_steps"] = int(os.environ["CONFIG_ENV_CHECKPOINT_EVERY"])
cfg["runtime"]["evaluation"]["every_n_steps"] = int(os.environ["CONFIG_ENV_EVALUATE_EVERY"])
cfg["runtime"]["log_every_n_steps"] = int(os.environ["CONFIG_ENV_LOG_EVERY"])

output_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

if [[ ! -f "$RUN_ROOT/streaming_config.json" ]]; then
  cp "$STREAM_TEMPLATE" "$RUN_ROOT/streaming_config.json"
fi

echo "Config ready at $RUN_ROOT/training.yaml"

echo "Building training binary (release)"
cargo build --release -p training --no-default-features --features cuda

echo "Starting training"
if ((${#TRAIN_ARGS[@]} > 0)); then
  cargo run --release -p training --no-default-features --features cuda --bin train -- --config "$RUN_ROOT/training.yaml" "${TRAIN_ARGS[@]}"
else
  cargo run --release -p training --no-default-features --features cuda --bin train -- --config "$RUN_ROOT/training.yaml"
fi
