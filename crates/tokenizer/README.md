tokenizer


We're going to use a byte-level BPE (Byte Pair Encoding) to tokenize the text corpus.
More info at https://arxiv.org/pdf/1909.03341
We're going to use the tokenizer lib from huggingface.

-----

## Overview
This crate wraps Hugging Face's `tokenizers` library to provide a byte-level BPE
pipeline with a strongly typed configuration. Consumers can either load existing
artifacts (`tokenizer.json` or `vocab.json`/`merges.txt`) or, with the optional
`train` feature enabled, train a new model directly from text corpora.

## Configuration
All behaviour is driven by `Config`, which bundles:
- `model`: vocab size, minimum merge frequency, optional dropout, declared special
tokens, and whether byte fallback should be enforced on decode.
- `pretokenizer`: byte-level options (prefix space, offset trimming, regex mode)
mirroring Hugging Face defaults.
- `postprocessor`: template toggles for injecting `<bos>`/`<eos>` and pairing
logic.
- `artifacts`: base directory plus relative or absolute paths to tokenizer
artifacts and optional manifests.
- `training` (feature `train` only): corpus paths, shuffling seed, concurrency,
and corpus size limits.

## Loading Artifacts
`build_from_artifacts` first validates the configuration, resolves artifact paths
relative to `artifacts.dir`, and checks that every required file exists. If a
`tokenizer.json` is provided it is loaded directly; otherwise the crate reads the
vocabulary/merge pair, instantiates a Hugging Face `BPE` model, and applies the
configured dropout and byte-fallback flags. A byte-level pretokenizer and
matching decoder are always attached so encode/decode follow Hugging Face byte
normalization rules. If template post-processing is requested, the helper builds
it using the resolved special-token IDs.

## Training Mode
When compiled with `--features train`, `train_bbpe` streams one or more text
files through a bounded-memory iterator, optionally shuffling in reproducible chunks.
It trains a `BpeTrainer`, reattaches dropout/byte-fallback settings, then writes
the resulting artifacts (JSON or split files) alongside an optional manifest that
records a configuration hash, creation timestamp, and token count.

## Validation & Testing
Configuration and produced tokenizers are validated to catch missing special
tokens, inconsistent byte-fallback settings, oversized vocabularies, and
uncreatable artifact directories. The test suite (`cargo test -p tokenizer --features train`)
includes round-trip, Unicode, pair-template, and whitespace stability checks,
ensuring the end-to-end pipeline behaves as expected.
