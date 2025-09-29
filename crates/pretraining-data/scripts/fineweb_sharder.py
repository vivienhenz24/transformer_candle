#!/usr/bin/env python3
"""Stream HuggingFace's FineWeb dataset into newline-delimited shards.

The script targets pod environments where `/workspace` is the mounted volume.
Shards are written to `/workspace/datasets/train` and `/workspace/datasets/val`
by default, matching the training loader expectations in this repository.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterable, List, Optional

from datasets import load_dataset  # type: ignore


DEFAULT_DATASET = "HuggingFaceFW/fineweb"
DEFAULT_CACHE_ROOT = Path("/workspace/hf_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", help="FineWeb split to stream")
    parser.add_argument(
        "--mode",
        choices=["smoke", "prod", "split"],
        default="smoke",
        help="smoke: tiny sample, prod: full stream, split: probabilistic val sampling",
    )
    parser.add_argument(
        "--lines-per-shard",
        type=int,
        default=200_000,
        help="Maximum lines per shard before rolling to a new file",
    )
    parser.add_argument(
        "--take",
        type=int,
        default=None,
        help="Optional cap on total lines (train + val). Overrides mode defaults",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Probability of routing a line to validation when in split/prod mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed used for validation sampling",
    )
    parser.add_argument(
        "--gzip",
        action="store_true",
        help="Compress shards with gzip (writes *.txt.gz)",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Dataset repository to stream (default: HuggingFaceFW/fineweb)",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("/workspace/datasets/train"),
        help="Directory for training shards",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=Path("/workspace/datasets/val"),
        help="Directory for validation shards",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token (falls back to env vars)",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="Base directory for HF_HOME/HF_DATASETS_CACHE",
    )

    return parser.parse_args()


def configure_hf_env(cache_root: Path, token: Optional[str]) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    datasets_cache = cache_root / "datasets"
    hub_cache = cache_root / "hub"
    datasets_cache.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))

    if token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    elif "HF_TOKEN" in os.environ and "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["HF_TOKEN"]


@dataclass
class ShardRecord:
    path: str
    lines: int
    bytes: int
    sha256: Optional[str]


@dataclass
class ShardWriter:
    output_dir: Path
    prefix: str
    lines_per_shard: int
    gzip_enabled: bool
    index: int = 0
    lines_in_current: int = 0
    current_file: Optional[IO[str]] = None
    current_path: Optional[Path] = None
    shards: List[ShardRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, text: str) -> None:
        if self.current_file is None or self.lines_in_current >= self.lines_per_shard:
            self._open_next()

        assert self.current_file is not None  # for type checkers
        self.current_file.write(text.rstrip("\n") + "\n")
        self.lines_in_current += 1

    def finish(self) -> List[ShardRecord]:
        if self.current_file is not None:
            self._close_current()
        return self.shards

    # Internal helpers -------------------------------------------------
    def _open_next(self) -> None:
        if self.current_file is not None:
            self._close_current()

        suffix = ".txt.gz" if self.gzip_enabled else ".txt"
        filename = f"{self.prefix}_{self.index:04d}{suffix}"
        self.current_path = self.output_dir / filename
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.gzip_enabled:
            self.current_file = gzip.open(self.current_path, "wt", encoding="utf-8")
        else:
            self.current_file = open(self.current_path, "w", encoding="utf-8")
        self.lines_in_current = 0
        self.index += 1

    def _close_current(self) -> None:
        assert self.current_file is not None
        assert self.current_path is not None

        self.current_file.close()
        size_bytes = self.current_path.stat().st_size
        sha256 = sha256_file(self.current_path)
        self.shards.append(
            ShardRecord(
                path=str(self.current_path),
                lines=self.lines_in_current,
                bytes=size_bytes,
                sha256=sha256,
            )
        )
        self.current_file = None
        self.current_path = None
        self.lines_in_current = 0


def sha256_file(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def compute_mode_defaults(args: argparse.Namespace) -> tuple[Optional[int], float]:
    cap = args.take
    val_ratio = args.val_ratio if args.val_ratio is not None else 0.0

    if args.mode == "smoke":
        cap = cap or 20_000
        val_ratio = args.val_ratio if args.val_ratio is not None else 0.1
    elif args.mode == "prod":
        # user-provided settings stand; default to no validation sampling
        val_ratio = args.val_ratio if args.val_ratio is not None else 0.0
    elif args.mode == "split":
        if args.val_ratio is None:
            raise SystemExit("--mode split requires --val-ratio to be provided")
        val_ratio = args.val_ratio

    if val_ratio < 0.0 or val_ratio >= 1.0:
        raise SystemExit("--val-ratio must be in [0, 1) when specified")

    return cap, val_ratio


def stream_fineweb(args: argparse.Namespace) -> None:
    cap, val_ratio = compute_mode_defaults(args)
    rng = random.Random(args.seed)

    configure_hf_env(args.cache_root, args.hf_token)

    dataset = load_dataset(args.dataset, split=args.split, streaming=True)

    train_writer: Optional[ShardWriter] = None
    val_writer: Optional[ShardWriter] = None

    if args.train_output:
        train_writer = ShardWriter(
            output_dir=args.train_output,
            prefix="train",
            lines_per_shard=args.lines_per_shard,
            gzip_enabled=args.gzip,
        )

    write_val = val_ratio > 0.0
    if write_val:
        val_writer = ShardWriter(
            output_dir=args.val_output,
            prefix="val",
            lines_per_shard=args.lines_per_shard,
            gzip_enabled=args.gzip,
        )

    processed = 0
    for sample in dataset:
        text = sample.get("text") if isinstance(sample, dict) else None
        if not text:
            continue

        processed += 1

        should_take_val = write_val and rng.random() < val_ratio
        if should_take_val and val_writer is not None:
            val_writer.write(text)
        elif train_writer is not None:
            train_writer.write(text)

        if cap is not None and processed >= cap:
            break

    manifests = {}
    if train_writer is not None:
        train_shards = train_writer.finish()
        manifests["train"] = summarise_shards(train_shards, args)
    if val_writer is not None:
        val_shards = val_writer.finish()
        manifests["val"] = summarise_shards(val_shards, args)

    for split, manifest in manifests.items():
        target = (args.train_output if split == "train" else args.val_output) / "manifest.json"
        target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    total_lines = sum(m["total_lines"] for m in manifests.values())
    print(f"Completed streaming {processed} records -> {total_lines} retained lines", file=sys.stderr)


def summarise_shards(shards: List[ShardRecord], args: argparse.Namespace) -> dict:
    return {
        "dataset": args.dataset,
        "split": args.split,
        "mode": args.mode,
        "seed": args.seed,
        "lines_per_shard": args.lines_per_shard,
        "gzip": args.gzip,
        "total_shards": len(shards),
        "total_lines": sum(s.lines for s in shards),
        "total_bytes": sum(s.bytes for s in shards),
        "shards": [s.__dict__ for s in shards],
}


def main() -> None:
    args = parse_args()
    try:
        stream_fineweb(args)
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)


if __name__ == "__main__":
    main()


def sha256_file(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None
