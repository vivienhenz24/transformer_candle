#!/usr/bin/env python3
"""Plot visualizations for attention benchmark results."""

from __future__ import annotations

import argparse
import ast
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - fail fast when matplotlib missing
    raise SystemExit(
        "matplotlib is required to run this script. Install it with `pip install matplotlib`."
    ) from exc


SECTION_MARKERS = {
    "correctness": ("<!-- CORRECTNESS_START -->", "<!-- CORRECTNESS_END -->"),
    "memory": ("<!-- MEMORY_START -->", "<!-- MEMORY_END -->"),
    "throughput": ("<!-- THROUGHPUT_START -->", "<!-- THROUGHPUT_END -->"),
}


def sanitize_header(header: str) -> str:
    """Convert a markdown table header into a pythonic key."""
    cleaned = header.strip().lower()
    cleaned = cleaned.replace("Δ", "delta")
    cleaned = cleaned.replace("/sec", "_per_sec")
    cleaned = cleaned.replace("/", "_")
    cleaned = cleaned.replace(" ", "_")
    cleaned = cleaned.replace("|", "")
    cleaned = re.sub(r"[^0-9a-z_()]", "", cleaned)
    cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def parse_markdown_table(raw: str) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    """Parse a GitHub-flavored markdown table into rows and header metadata."""
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if len(lines) < 3:
        return [], {}

    header_cells = [cell.strip() for cell in lines[0].strip("|").split("|")]
    header_keys = [sanitize_header(cell) for cell in header_cells]
    header_map = dict(zip(header_keys, header_cells))

    rows: List[Dict[str, str]] = []
    for line in lines[2:]:  # skip header + alignment line
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(header_cells):  # skip malformed rows
            continue
        row = {header_keys[idx]: cells[idx] for idx in range(len(header_keys))}
        rows.append(row)
    return rows, header_map


def extract_section(markdown: str, start: str, end: str) -> str:
    try:
        start_idx = markdown.index(start) + len(start)
        end_idx = markdown.index(end, start_idx)
    except ValueError as exc:
        raise ValueError(f"Could not locate section between {start} and {end}.") from exc
    return markdown[start_idx:end_idx]


def parse_number(raw: str) -> float:
    """Convert tokens with SI-style suffixes (K/M) or scientific notation to float."""
    token = raw.replace(",", "").strip()
    match = re.match(r"^([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*([KMG]?)$", token)
    if not match:
        raise ValueError(f"Cannot parse numeric value from '{raw}'.")
    value = float(match.group(1))
    suffix = match.group(2)
    scale = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9}
    return value * scale[suffix]


def parse_shape(raw: str) -> Tuple[int, ...]:
    try:
        return tuple(int(num) for num in ast.literal_eval(raw))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse shape tuple from '{raw}'.") from exc


def ensure_sequence(key: str, row: Dict[str, str]) -> str:
    if key in row:
        return row[key]
    raise KeyError(f"Expected column '{key}' in table row: {row}")


def save_figure(fig: "plt.Figure", output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".png", {"dpi": 200}), (".pdf", {})):
        target = output_dir / f"{name}{suffix}"
        fig.savefig(target, bbox_inches="tight", **kwargs)
        print(f"Saved {target}")
    plt.close(fig)


def plot_correctness(rows: List[Dict[str, str]], headers: Dict[str, str], output_dir: Path) -> None:
    if not rows:
        return
    shape_key = "shape"
    dtype_key = "dtype"
    metrics = [col for col in ("max_abs", "max_rel", "max_deltalog") if col in rows[0]]

    numeric_rows = []
    for row in rows:
        converted = dict(row)
        for metric in metrics:
            converted[metric] = parse_number(row[metric])
        converted[shape_key] = ensure_sequence(shape_key, row)
        numeric_rows.append(converted)

    shapes = sorted({row[shape_key] for row in numeric_rows}, key=lambda s: parse_shape(s))
    dtypes = sorted({row[dtype_key] for row in numeric_rows})
    x_positions = range(len(shapes))

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for axis, metric in zip(axes, metrics):
        for dtype in dtypes:
            y_values = []
            for shape in shapes:
                match_row = next(
                    (row for row in numeric_rows if row[shape_key] == shape and row[dtype_key] == dtype),
                    None,
                )
                y_values.append(match_row[metric] if match_row else math.nan)
            axis.plot(list(x_positions), y_values, marker="o", label=dtype)
        axis.set_ylabel(headers.get(metric, metric))
        axis.set_yscale("log")
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
        axis.legend(title=headers.get(dtype_key, "dtype"))
    axes[-1].set_xticks(list(x_positions))
    axes[-1].set_xticklabels(shapes, rotation=45, ha="right")
    axes[-1].set_xlabel(headers.get(shape_key, "shape"))
    fig.suptitle("Attention Correctness Metrics")
    fig.tight_layout()
    save_figure(fig, output_dir, "attention_correctness")


def plot_memory(rows: List[Dict[str, str]], headers: Dict[str, str], output_dir: Path) -> None:
    if not rows:
        return
    backend_key = "backend"
    shape_key = "shape"
    dtype_key = "dtype"

    metrics = [col for col in ("tokens_per_sec", "peak_mb", "steady_mb") if col in rows[0]]

    numeric_rows = []
    for row in rows:
        converted = dict(row)
        for metric in metrics:
            converted[metric] = parse_number(row[metric])
        numeric_rows.append(converted)

    pairs = sorted(
        { (row[shape_key], row[dtype_key]) for row in numeric_rows },
        key=lambda item: (parse_shape(item[0]), item[1]),
    )
    backends = sorted({row[backend_key] for row in numeric_rows})
    x_positions = range(len(pairs))
    width = 0.8 / max(len(backends), 1)

    for metric in metrics:
        fig, axis = plt.subplots(figsize=(12, 4))
        for idx, backend in enumerate(backends):
            offsets = [pos + (idx - (len(backends) - 1) / 2) * width for pos in x_positions]
            values = []
            for shape, dtype in pairs:
                match_row = next(
                    (
                        row
                        for row in numeric_rows
                        if row[shape_key] == shape and row[dtype_key] == dtype and row[backend_key] == backend
                    ),
                    None,
                )
                values.append(match_row[metric] if match_row else 0.0)
            axis.bar(offsets, values, width=width, label=backend)
        axis.set_xticks(list(x_positions))
        axis.set_xticklabels([f"{shape}\n{dtype}" for shape, dtype in pairs], rotation=45, ha="right")
        axis.set_ylabel(headers.get(metric, metric))
        axis.set_xlabel(f"{headers.get(shape_key, 'shape')} + {headers.get(dtype_key, 'dtype')}")
        axis.set_title(f"Attention Memory Benchmark — {headers.get(metric, metric)}")
        axis.grid(True, axis="y", linestyle="--", alpha=0.3)
        axis.legend(title=headers.get(backend_key, "backend"))
        fig.tight_layout()
        save_figure(fig, output_dir, f"attention_memory_{metric}")


def plot_throughput(rows: List[Dict[str, str]], headers: Dict[str, str], output_dir: Path) -> None:
    if not rows:
        return
    backend_key = "backend"
    shape_key_candidates = [key for key in rows[0] if key.startswith("shape")]
    if not shape_key_candidates:
        raise KeyError("Failed to locate shape column in throughput table.")
    shape_key = shape_key_candidates[0]
    dtype_key = "dtype"
    metric_key = "tokens_per_sec"

    numeric_rows = []
    for row in rows:
        converted = dict(row)
        converted[metric_key] = parse_number(row[metric_key])
        numeric_rows.append(converted)

    labels = sorted(
        { (row[shape_key], row[dtype_key]) for row in numeric_rows },
        key=lambda item: (parse_shape(item[0]), item[1]),
    )
    backends = sorted({row[backend_key] for row in numeric_rows})
    x_positions = range(len(labels))
    width = 0.8 / max(len(backends), 1)

    fig, axis = plt.subplots(figsize=(14, 5))
    for idx, backend in enumerate(backends):
        offsets = [pos + (idx - (len(backends) - 1) / 2) * width for pos in x_positions]
        values = []
        for shape, dtype in labels:
            match_row = next(
                (
                    row
                    for row in numeric_rows
                    if row[shape_key] == shape and row[dtype_key] == dtype and row[backend_key] == backend
                ),
                None,
            )
            values.append(match_row[metric_key] if match_row else 0.0)
        axis.bar(offsets, values, width=width, label=backend)
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels([f"{shape}\n{dtype}" for shape, dtype in labels], rotation=45, ha="right")
    axis.set_ylabel(headers.get(metric_key, metric_key))
    axis.set_xlabel(f"{headers.get(shape_key, 'shape')} + {headers.get(dtype_key, 'dtype')}")
    axis.set_title("Attention Throughput (tokens/sec)")
    axis.grid(True, axis="y", linestyle="--", alpha=0.3)
    axis.legend(title=headers.get(backend_key, "backend"))
    fig.tight_layout()
    save_figure(fig, output_dir, "attention_throughput")


def load_results(path: Path) -> Dict[str, Tuple[List[Dict[str, str]], Dict[str, str]]]:
    markdown = path.read_text(encoding="utf-8")
    tables: Dict[str, Tuple[List[Dict[str, str]], Dict[str, str]]] = {}
    for key, (start, end) in SECTION_MARKERS.items():
        section = extract_section(markdown, start, end)
        rows, headers = parse_markdown_table(section)
        tables[key] = (rows, headers)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_results = Path(__file__).resolve().parent.parent / "crates" / "attention" / "benchmarks" / "RESULTS.md"
    parser.add_argument(
        "--results",
        type=Path,
        default=default_results,
        help=f"Path to RESULTS.md (default: {default_results})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots",
        help="Directory where plots will be written (default: benches/plots)",
    )
    args = parser.parse_args()

    tables = load_results(args.results)
    plot_correctness(*tables["correctness"], output_dir=args.output_dir)
    plot_memory(*tables["memory"], output_dir=args.output_dir)
    plot_throughput(*tables["throughput"], output_dir=args.output_dir)


if __name__ == "__main__":
    main()
