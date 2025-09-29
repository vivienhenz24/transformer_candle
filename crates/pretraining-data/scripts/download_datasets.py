#!/usr/bin/env python3
"""
Download and preprocess datasets from Hugging Face for tokenizer training.
This script downloads datasets in chunks to work with the existing Rust preprocessing pipeline.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datasets import load_dataset, Dataset
from tqdm import tqdm
import time
from huggingface_hub import login

class DatasetDownloader:
    def __init__(self, output_dir: str = "data/raw_hf", hf_token: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Handle authentication for private datasets
        if hf_token:
            print("🔐 Authenticating with Hugging Face...")
            login(token=hf_token)
        elif os.getenv("HUGGINGFACE_HUB_TOKEN"):
            print("🔐 Using HF token from environment...")
            login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
        else:
            print("ℹ️  No authentication - using public datasets only")

    def download_openwebtext(self, max_examples: Optional[int] = None, chunk_size: int = 10000):
        """Download OpenWebText dataset in chunks."""
        print("🔄 Downloading OpenWebText dataset...")

        dataset = load_dataset("openwebtext", streaming=True, split="train")
        chunk_num = 0
        total_examples = 0

        current_chunk = []

        for example in tqdm(dataset, desc="Processing OpenWebText"):
            if max_examples and total_examples >= max_examples:
                break

            text = example["text"].strip()
            if len(text) > 50:  # Filter very short texts
                current_chunk.append(text)

            if len(current_chunk) >= chunk_size:
                self._save_chunk("openwebtext", chunk_num, current_chunk)
                current_chunk = []
                chunk_num += 1

            total_examples += 1

        # Save remaining texts
        if current_chunk:
            self._save_chunk("openwebtext", chunk_num, current_chunk)

        print(f"✅ Downloaded {total_examples} examples in {chunk_num + 1} chunks")
        return chunk_num + 1

    def download_paul_graham_essays(self, max_examples: Optional[int] = None, chunk_size: int = 10000):
        """
        Download Paul Graham Essays dataset in chunks.
        Requires authentication (see Hugging Face instructions).
        """
        print("🔄 Downloading Paul Graham Essays dataset...")

        try:
            dataset = load_dataset("baber/paul_graham_essays", split="train", streaming=True)
        except Exception as e:
            print(f"❌ Could not load Paul Graham Essays: {e}")
            return 0

        chunk_num = 0
        total_examples = 0
        current_chunk = []

        for example in tqdm(dataset, desc="Processing Paul Graham Essays"):
            if max_examples and total_examples >= max_examples:
                break

            # The dataset has a "text" field
            text = example.get("text", "").strip()
            if len(text) > 50:  # Filter very short essays
                current_chunk.append(text)

            if len(current_chunk) >= chunk_size:
                self._save_chunk("paul_graham_essays", chunk_num, current_chunk)
                current_chunk = []
                chunk_num += 1

            total_examples += 1

        # Save any remaining essays
        if current_chunk:
            self._save_chunk("paul_graham_essays", chunk_num, current_chunk)

        print(f"✅ Downloaded {total_examples} examples in {chunk_num + 1} chunks")
        return chunk_num + 1

    def download_wikipedia(self, language: str = "en", max_examples: Optional[int] = None, chunk_size: int = 5000):
        """Download Wikipedia dataset in chunks."""
        print(f"🔄 Downloading Wikipedia ({language}) dataset...")

        dataset = load_dataset("wikipedia", f"20220301.{language}", streaming=True, split="train")
        chunk_num = 0
        total_examples = 0

        current_chunk = []

        for example in tqdm(dataset, desc=f"Processing Wikipedia ({language})"):
            if max_examples and total_examples >= max_examples:
                break

            text = example["text"].strip()
            title = example.get("title", "").strip()

            # Combine title and text
            if title and text:
                combined_text = f"{title}\n\n{text}"
                if len(combined_text) > 100:  # Filter very short articles
                    current_chunk.append(combined_text)

            if len(current_chunk) >= chunk_size:
                self._save_chunk("wikipedia", chunk_num, current_chunk)
                current_chunk = []
                chunk_num += 1

            total_examples += 1

        # Save remaining texts
        if current_chunk:
            self._save_chunk("wikipedia", chunk_num, current_chunk)

        print(f"✅ Downloaded {total_examples} examples in {chunk_num + 1} chunks")
        return chunk_num + 1

    def download_bookcorpus(self, max_examples: Optional[int] = None, chunk_size: int = 1000):
        """Download BookCorpus dataset in chunks."""
        print("🔄 Downloading BookCorpus dataset...")

        try:
            dataset = load_dataset("bookcorpus", streaming=True, split="train")
        except Exception as e:
            print(f"❌ Could not load BookCorpus: {e}")
            print("💡 Trying alternative: using 'pg19' dataset instead...")
            dataset = load_dataset("pg19", streaming=True, split="train")

        chunk_num = 0
        total_examples = 0
        current_chunk = []

        for example in tqdm(dataset, desc="Processing BookCorpus/PG19"):
            if max_examples and total_examples >= max_examples:
                break

            text = example["text"].strip()
            if len(text) > 500:  # Books should have substantial content
                current_chunk.append(text)

            if len(current_chunk) >= chunk_size:
                self._save_chunk("books", chunk_num, current_chunk)
                current_chunk = []
                chunk_num += 1

            total_examples += 1

        # Save remaining texts
        if current_chunk:
            self._save_chunk("books", chunk_num, current_chunk)

        print(f"✅ Downloaded {total_examples} examples in {chunk_num + 1} chunks")
        return chunk_num + 1

    def download_code_dataset(self, max_examples: Optional[int] = None, chunk_size: int = 5000):
        """Download code dataset (subset of The Stack)."""
        print("🔄 Downloading code dataset...")

        # Use a smaller subset for now
        languages = ["Python", "JavaScript", "Rust", "Go", "Java"]
        total_chunks = 0

        for lang in languages:
            print(f"  Processing {lang} code...")
            try:
                dataset = load_dataset(
                    "bigcode/the-stack-dedup",
                    data_dir=f"data/{lang.lower()}",
                    streaming=True,
                    split="train"
                )

                chunk_num = 0
                examples_count = 0
                current_chunk = []

                for example in dataset:
                    if max_examples and examples_count >= max_examples // len(languages):
                        break

                    content = example.get("content", "").strip()
                    if len(content) > 100:  # Filter very short code files
                        # Add file context
                        file_path = example.get("path", "unknown")
                        formatted_content = f"// File: {file_path}\n{content}"
                        current_chunk.append(formatted_content)

                    if len(current_chunk) >= chunk_size:
                        self._save_chunk(f"code_{lang.lower()}", chunk_num, current_chunk)
                        current_chunk = []
                        chunk_num += 1
                        total_chunks += 1

                    examples_count += 1

                # Save remaining
                if current_chunk:
                    self._save_chunk(f"code_{lang.lower()}", chunk_num, current_chunk)
                    total_chunks += 1

            except Exception as e:
                print(f"  ⚠️  Could not download {lang}: {e}")
                continue

        print(f"✅ Downloaded code in {total_chunks} total chunks")
        return total_chunks

    def download_private_dataset(self, dataset_name: str, max_examples: Optional[int] = None, chunk_size: int = 5000):
        """Download a private dataset by name."""
        print(f"🔄 Downloading private dataset: {dataset_name}...")

        try:
            dataset = load_dataset(dataset_name, streaming=True, split="train")
            chunk_num = 0
            total_examples = 0
            current_chunk = []

            for example in tqdm(dataset, desc=f"Processing {dataset_name}"):
                if max_examples and total_examples >= max_examples:
                    break

                # Try to get text content from common field names
                text = None
                for field in ["text", "content", "passage", "document", "article"]:
                    if field in example:
                        text = example[field].strip()
                        break

                if text and len(text) > 50:
                    current_chunk.append(text)

                if len(current_chunk) >= chunk_size:
                    safe_name = dataset_name.replace("/", "_").replace("-", "_")
                    self._save_chunk(f"private_{safe_name}", chunk_num, current_chunk)
                    current_chunk = []
                    chunk_num += 1

                total_examples += 1

            # Save remaining texts
            if current_chunk:
                safe_name = dataset_name.replace("/", "_").replace("-", "_")
                self._save_chunk(f"private_{safe_name}", chunk_num, current_chunk)

            print(f"✅ Downloaded {total_examples} examples in {chunk_num + 1} chunks")
            return chunk_num + 1

        except Exception as e:
            print(f"❌ Failed to download {dataset_name}: {e}")
            print("💡 Make sure you have access to this dataset and are authenticated")
            return 0

    def _save_chunk(self, dataset_name: str, chunk_num: int, texts: list):
        """Save a chunk of texts to a file."""
        filename = f"{dataset_name}_chunk_{chunk_num:04d}.txt"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            for text in texts:
                # Ensure each text is on its own line and clean
                clean_text = text.replace('\n\n\n', '\n\n').strip()
                f.write(clean_text + '\n')

        print(f"  💾 Saved {len(texts)} texts to {filename}")

    def create_manifest(self, dataset_info: Dict[str, Any]):
        """Create a manifest file with download information."""
        manifest_path = self.output_dir / "download_manifest.json"

        manifest = {
            "download_timestamp": time.time(),
            "datasets": dataset_info,
            "total_files": len(list(self.output_dir.glob("*.txt"))),
            "output_directory": str(self.output_dir)
        }

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"📄 Created manifest at {manifest_path}")

def main():
    parser = argparse.ArgumentParser(description="Download datasets from Hugging Face")
    parser.add_argument("--datasets", nargs="+",
                       choices=["openwebtext", "wikipedia", "books", "code", "all"],
                       default=["openwebtext"],
                       help="Datasets to download")
    parser.add_argument("--output-dir", default="data/raw_hf",
                       help="Output directory for downloaded data")
    parser.add_argument("--max-examples", type=int,
                       help="Maximum examples per dataset (for testing)")
    parser.add_argument("--chunk-size", type=int, default=10000,
                       help="Number of examples per chunk file")
    parser.add_argument("--private-dataset", type=str,
                       help="Download a specific private dataset by name (e.g., 'username/dataset-name')")
    parser.add_argument("--hf-token", type=str,
                       help="Hugging Face token for private datasets")

    args = parser.parse_args()

    if "all" in args.datasets:
        args.datasets = ["openwebtext", "wikipedia", "books", "code"]

    downloader = DatasetDownloader(args.output_dir, args.hf_token)
    dataset_info = {}

    print(f"📥 Starting dataset download to: {args.output_dir}")
    print(f"📊 Datasets to download: {', '.join(args.datasets)}")

    if args.max_examples:
        print(f"🎯 Max examples per dataset: {args.max_examples:,}")

    start_time = time.time()

    # Handle private dataset download
    if args.private_dataset:
        print(f"🔒 Downloading private dataset: {args.private_dataset}")
        chunks = downloader.download_private_dataset(args.private_dataset, args.max_examples, args.chunk_size)
        dataset_info[args.private_dataset] = {"chunks": chunks, "chunk_size": args.chunk_size}

    if "openwebtext" in args.datasets:
        chunks = downloader.download_openwebtext(args.max_examples, args.chunk_size)
        dataset_info["openwebtext"] = {"chunks": chunks, "chunk_size": args.chunk_size}

    if "wikipedia" in args.datasets:
        chunks = downloader.download_wikipedia("en", args.max_examples, args.chunk_size // 2)
        dataset_info["wikipedia"] = {"chunks": chunks, "chunk_size": args.chunk_size // 2}

    if "books" in args.datasets:
        chunks = downloader.download_bookcorpus(args.max_examples, args.chunk_size // 10)
        dataset_info["books"] = {"chunks": chunks, "chunk_size": args.chunk_size // 10}

    if "code" in args.datasets:
        chunks = downloader.download_code_dataset(args.max_examples, args.chunk_size)
        dataset_info["code"] = {"chunks": chunks, "chunk_size": args.chunk_size}

    # Create manifest
    downloader.create_manifest(dataset_info)

    elapsed = time.time() - start_time
    print(f"\n🎉 Download complete! Total time: {elapsed/60:.1f} minutes")
    print(f"📁 Files saved to: {args.output_dir}")
    print("\n🚀 Next steps:")
    print("1. Run the Rust preprocessor to combine all datasets")
    print("2. Train your tokenizer on the combined corpus")

if __name__ == "__main__":
    main()