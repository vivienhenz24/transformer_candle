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

RUN_NAME="m3-20m-streaming"
declare -a TRAIN_ARGS=()
if [[ $# -gt 0 && "$1" != "--" && "$1" != -* ]]; then
  RUN_NAME="$1"
  shift
fi
if [[ $# -gt 0 && "$1" == "--" ]]; then
  shift
fi
TRAIN_ARGS=("$@")

RUN_ROOT="$REPO_ROOT/runs/$RUN_NAME"
TOKENIZER_DIR="$RUN_ROOT/tokenizer"
DATA_DIR="$RUN_ROOT/data"
CACHE_DIR="$RUN_ROOT/cache"
CHECKPOINT_DIR="$RUN_ROOT/checkpoints"
BEST_DIR="$RUN_ROOT/best"
TB_DIR="$RUN_ROOT/tensorboard"

DATASET="${DATASET:-HuggingFaceFW/fineweb}"
SPLIT="${SPLIT:-train}"
STREAM_BATCH_SIZE="${STREAM_BATCH_SIZE:-512}"
MAX_STREAM_SAMPLES="${MAX_STREAM_SAMPLES:-600000}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"
TOKENIZER_MAX_LINES="${TOKENIZER_MAX_LINES:-600000}"

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
  if ! "$PYTHON_BIN" -m pip install --upgrade pip >/dev/null; then
    echo "Failed to upgrade pip inside virtualenv" >&2
    exit 1
  fi
  if ! "$PYTHON_BIN" -m pip install --upgrade datasets huggingface_hub >/dev/null; then
    echo "Failed to install required Python packages (datasets, huggingface_hub)." >&2
    echo "Install them manually and re-run." >&2
    exit 1
  fi
  touch "$DEPS_SENTINEL"
else
  if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
missing = [m for m in ("datasets", "huggingface_hub") if importlib.util.find_spec(m) is None]
sys.exit(0 if not missing else 1)
PY
  then
    echo "Updating Python dependencies (datasets, huggingface_hub)"
    if ! "$PYTHON_BIN" -m pip install --upgrade datasets huggingface_hub >/dev/null; then
      echo "Failed to update Python packages (datasets, huggingface_hub)." >&2
      echo "Install them manually and re-run." >&2
      exit 1
    fi
  fi
fi

export PYTHON_BIN

CONFIG_TEMPLATE="$REPO_ROOT/configs/m3_20m_streaming.yaml"
STREAM_TEMPLATE="$REPO_ROOT/configs/m3_20m_streaming_config.json"

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
  if [[ -n "${HUGGINGFACEHUB_API_TOKEN:-}" ]]; then
    echo "huggingface-cli not found; relying on HUGGINGFACEHUB_API_TOKEN from environment/.env." >&2
  else
    echo "huggingface-cli not found on PATH. Install it or set HUGGINGFACEHUB_API_TOKEN if FineWeb access requires auth." >&2
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-}"
echo "Preparing run directory at $RUN_ROOT"
mkdir -p "$TOKENIZER_DIR" "$DATA_DIR" "$CACHE_DIR" "$CHECKPOINT_DIR" "$BEST_DIR" "$TB_DIR"

touch "$DATA_DIR/train_placeholder.txt" "$DATA_DIR/val_placeholder.txt"

TOKENIZER_JSON="$TOKENIZER_DIR/tokenizer.json"

if [[ ! -f "$TOKENIZER_JSON" ]]; then
  echo "Training tokenizer from $DATASET ($SPLIT split)"
  ORCH_ROOT="$RUN_ROOT/.orchestrate"
  mkdir -p "$ORCH_ROOT"
  export ORCHESTRATE_RUNS_ROOT="$(dirname "$RUN_ROOT")"
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
    --hidden-size 256
    --intermediate-size 1024
    --layers 16
    --heads 8
    --seq-len 1024
    --batch-size 8
    --grad-accum 4
    --steps 12500
    --learning-rate 3e-4
    --weight-decay 0.05
    --warmup-steps 1000
    --schedule cosine-with-warmup
    --checkpoint-every 500
    --max-checkpoints 5
    --evaluate-every 1000
    --eval-batches 4
    --log-every 10
    --shuffle-buffer 8192
    --loader-workers 2
    --precision fp32
    --seed 42
    --tokenizer-seed 42
  )

  if [[ -n "$MAX_STREAM_SAMPLES" ]]; then
    ORCH_ARGS+=(--max-samples "$MAX_STREAM_SAMPLES")
  fi

  if ! cargo run --release -p training --bin orchestrate -- "${ORCH_ARGS[@]}"; then
    echo "Tokenizer training via orchestrate failed" >&2
    exit 1
  fi

  unset ORCHESTRATE_RUNS_ROOT ORCHESTRATE_CACHE_ROOT ORCHESTRATE_TOKENIZER_ROOT \
    ORCHESTRATE_TRAIN_ROOT ORCHESTRATE_VAL_ROOT ORCHESTRATE_HF_CACHE_ROOT
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

RUN_ROOT_ABS="$RUN_ROOT"
TMP_RUN_ROOT="$(cd "$RUN_ROOT" 2>/dev/null && pwd)" || TMP_RUN_ROOT=""
if [[ -n "$TMP_RUN_ROOT" ]]; then
  RUN_ROOT_ABS="$TMP_RUN_ROOT"
fi
export CONFIG_ENV_RUN_ROOT="$RUN_ROOT_ABS"
export CONFIG_ENV_TEMPLATE="$CONFIG_TEMPLATE"
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

template_path = Path(os.environ["CONFIG_ENV_TEMPLATE"])
run_root = Path(os.environ["CONFIG_ENV_RUN_ROOT"])
text = template_path.read_text()
text = text.replace("../runs/m3-20m-streaming", str(run_root))
(run_root / "training.yaml").write_text(text)
PY

if [[ ! -f "$RUN_ROOT/streaming_config.json" ]]; then
  cp "$STREAM_TEMPLATE" "$RUN_ROOT/streaming_config.json"
fi

echo "Config ready at $RUN_ROOT/training.yaml"

echo "Building training binary (release)"
if ! cargo build --release -p training; then
  echo "cargo build failed. On macOS you may need to disable CUDA features or enable Metal in crates/training/Cargo.toml." >&2
  exit 1
fi

echo "Starting training"
if ((${#TRAIN_ARGS[@]} > 0)); then
  cargo run --release -p training --bin train -- --config "$RUN_ROOT/training.yaml" "${TRAIN_ARGS[@]}"
else
  cargo run --release -p training --bin train -- --config "$RUN_ROOT/training.yaml"
fi
