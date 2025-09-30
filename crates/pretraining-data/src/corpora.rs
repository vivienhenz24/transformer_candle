use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::PathBuf;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

/// Trait for corpus types that can stream text data
pub trait TextCorpus {
    type Stream: Iterator<Item = io::Result<String>>;
    fn stream(&self) -> io::Result<Self::Stream>;
}

/// Streaming helper for sharded text corpora.
///
/// The iterator yields trimmed UTF-8 lines from each shard in order.
#[derive(Clone, Debug)]
pub struct StreamingCorpus {
    shards: Vec<PathBuf>,
}

impl StreamingCorpus {
    pub fn new(shards: Vec<PathBuf>) -> io::Result<Self> {
        if shards.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "expected at least one shard for corpus",
            ));
        }

        println!(
            "[pretraining-data crate] StreamingCorpus::new with {} shard(s)",
            shards.len()
        );

        Ok(Self { shards })
    }

    pub fn shard_paths(&self) -> &[PathBuf] {
        &self.shards
    }
}

impl TextCorpus for StreamingCorpus {
    type Stream = CorpusStream;

    fn stream(&self) -> io::Result<Self::Stream> {
        println!("[pretraining-data crate] StreamingCorpus::stream reading {} shards in parallel", self.shards.len());
        
        // Read all shards in parallel using rayon
        use std::sync::Mutex;
        let lines_mutex = Arc::new(Mutex::new(Vec::new()));
        
        self.shards
            .par_iter()
            .enumerate()
            .for_each(|(idx, path)| {
                if idx % 10 == 0 {
                    println!("[pretraining-data crate] Reading shard batch starting at {}", idx);
                }
                
                match File::open(path) {
                    Ok(file) => {
                        let reader = BufReader::new(file);
                        let shard_lines: Vec<String> = reader
                            .lines()
                            .filter_map(|line| line.ok())
                            .filter(|line| !line.trim().is_empty())
                            .collect();
                        
                        if let Ok(mut lines) = lines_mutex.lock() {
                            lines.extend(shard_lines);
                        }
                    }
                    Err(e) => {
                        eprintln!("[pretraining-data crate] Failed to open shard {}: {}", path.display(), e);
                    }
                }
            });
        
        let all_lines = Arc::try_unwrap(lines_mutex)
            .unwrap_or_else(|arc| arc.lock().unwrap().clone())
            .into_inner()
            .unwrap();
        
        println!("[pretraining-data crate] Loaded {} total lines from {} shards", all_lines.len(), self.shards.len());
        
        CorpusStream::from_lines(all_lines)
    }
}

pub struct CorpusStream {
    lines: VecDeque<String>,
    shards: Option<Vec<PathBuf>>,
    current_reader: Option<BufReader<File>>,
    next_shard: usize,
}

impl CorpusStream {
    fn from_lines(lines: Vec<String>) -> io::Result<Self> {
        Ok(Self {
            lines: VecDeque::from(lines),
            shards: None,
            current_reader: None,
            next_shard: 0,
        })
    }
    
    fn new_parallel(shards: Vec<PathBuf>) -> io::Result<Self> {
        println!("[pretraining-data crate] Using parallel streaming approach");
        let mut stream = Self {
            lines: VecDeque::new(),
            shards: Some(shards),
            current_reader: None,
            next_shard: 0,
        };
        stream.open_next_shard()?;
        Ok(stream)
    }
    
    fn open_next_shard(&mut self) -> io::Result<bool> {
        let shards = self.shards.as_ref().ok_or_else(|| {
            io::Error::new(io::ErrorKind::Other, "No shards available")
        })?;
        
        while self.next_shard < shards.len() {
            let path = &shards[self.next_shard];
            self.next_shard += 1;
            match File::open(path) {
                Ok(file) => {
                    self.current_reader = Some(BufReader::new(file));
                    println!(
                        "[pretraining-data crate] StreamingCorpus opening shard {}: {}",
                        self.next_shard - 1,
                        path.display()
                    );
                    return Ok(true);
                }
                Err(err) if err.kind() == io::ErrorKind::NotFound => continue,
                Err(err) => return Err(err),
            }
        }

        self.current_reader = None;
        Ok(false)
    }
}

impl Iterator for CorpusStream {
    type Item = io::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        // If we have pre-loaded lines, use those first
        if let Some(line) = self.lines.pop_front() {
            return Some(Ok(line));
        }
        
        // Otherwise, stream from files
        if self.shards.is_some() {
            loop {
                let reader = match self.current_reader.as_mut() {
                    Some(reader) => reader,
                    None => return None,
                };

                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => match self.open_next_shard() {
                        Ok(true) => continue,
                        Ok(false) => return None,
                        Err(err) => return Some(Err(err)),
                    },
                    Ok(_) => {
                        while line.ends_with(['\n', '\r']) {
                            line.pop();
                        }
                        if !line.trim().is_empty() {
                            return Some(Ok(line));
                        }
                        // Continue to next line if empty
                    }
                    Err(err) => return Some(Err(err)),
                }
            }
        }
        
        None
    }
}

/// Streaming corpus that fetches data directly from HuggingFace without downloading files.
///
/// This uses Python's datasets library via a long-lived subprocess to stream data efficiently.
/// The data is fetched in batches and buffered in memory.
pub struct HuggingFaceStreamingCorpus {
    dataset_name: String,
    split: String,
    batch_size: usize,
    max_samples: Option<usize>,
}

impl HuggingFaceStreamingCorpus {
    pub fn new(
        dataset_name: String,
        split: String,
        batch_size: usize,
        max_samples: Option<usize>,
    ) -> io::Result<Self> {
        println!(
            "[pretraining-data crate] HuggingFaceStreamingCorpus::new dataset={} split={} batch_size={} max_samples={:?}",
            dataset_name, split, batch_size, max_samples
        );

        Ok(Self {
            dataset_name,
            split,
            batch_size,
            max_samples,
        })
    }
}

impl TextCorpus for HuggingFaceStreamingCorpus {
    type Stream = HFCorpusStream;

    fn stream(&self) -> io::Result<Self::Stream> {
        println!("[pretraining-data crate] HuggingFaceStreamingCorpus::stream creating iterator");
        HFCorpusStream::new(
            self.dataset_name.clone(),
            self.split.clone(),
            self.batch_size,
            self.max_samples,
        )
    }
}

pub struct HFCorpusStream {
    current_index: usize,
    max_samples: Option<usize>,
    buffer: VecDeque<String>,
    python_process: Option<std::process::Child>,
    stdout_reader: Option<std::io::BufReader<std::process::ChildStdout>>,
    finished: bool,
}

impl HFCorpusStream {
    fn new(
        dataset_name: String,
        split: String,
        batch_size: usize,
        max_samples: Option<usize>,
    ) -> io::Result<Self> {
        use std::process::{Command, Stdio};
        
        // Python script that runs as a long-lived process
        let python_script = format!(
            r#"
import sys
import json
from datasets import load_dataset

def main():
    dataset_name = "{}"
    split = "{}"
    batch_size = {}
    max_samples = {}
    
    try:
        # Load dataset in streaming mode  
        print(json.dumps({{"status": "loading"}}), flush=True)
        ds = load_dataset(dataset_name, split=split, streaming=True)
        print(json.dumps({{"status": "ready"}}), flush=True)
        
        batch = []
        count = 0
        
        for sample in ds:
            text = sample.get("text", "")
            if text:
                batch.append(text)
                count += 1
                
                # Send batch when full
                if len(batch) >= batch_size:
                    print(json.dumps({{"batch": batch}}), flush=True)
                    batch = []
                    
                    # Progress update every 10k samples
                    if count % 10000 == 0:
                        print(json.dumps({{"progress": count}}), file=sys.stderr, flush=True)
                
                # Check max samples limit
                if max_samples is not None and count >= max_samples:
                    break
        
        # Send remaining batch
        if batch:
            print(json.dumps({{"batch": batch}}), flush=True)
        
        # Send completion signal
        print(json.dumps({{"done": True, "total": count}}), flush=True)
        
    except Exception as e:
        print(json.dumps({{"error": str(e)}}), file=sys.stderr, flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
"#,
            dataset_name,
            split,
            batch_size,
            max_samples
                .map(|n| format!("{}", n))
                .unwrap_or_else(|| "None".to_string())
        );

        // Start Python process
        let mut child = Command::new("python3")
            .arg("-c")
            .arg(&python_script)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("Failed to spawn Python streaming process: {}", e),
                )
            })?;

        let stdout = child.stdout.take().ok_or_else(|| {
            io::Error::new(io::ErrorKind::Other, "Failed to capture stdout")
        })?;

        let mut stream = Self {
            current_index: 0,
            max_samples,
            buffer: VecDeque::new(),
            python_process: Some(child),
            stdout_reader: Some(std::io::BufReader::new(stdout)),
            finished: false,
        };

        // Wait for "ready" status
        if let Err(e) = stream.wait_for_ready() {
            stream.cleanup();
            return Err(e);
        }

        Ok(stream)
    }

    fn wait_for_ready(&mut self) -> io::Result<()> {
        use std::io::BufRead;
        
        let reader = self.stdout_reader.as_mut().ok_or_else(|| {
            io::Error::new(io::ErrorKind::Other, "No stdout reader available")
        })?;

        let mut line = String::new();
        while reader.read_line(&mut line)? > 0 {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&line) {
                if let Some(status) = parsed.get("status").and_then(|s| s.as_str()) {
                    if status == "ready" {
                        println!("[pretraining-data crate] HuggingFace streaming ready");
                        return Ok(());
                    } else if status == "loading" {
                        println!("[pretraining-data crate] Loading HuggingFace dataset...");
                    }
                }
            }
            line.clear();
        }

        Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "Python process ended before becoming ready",
        ))
    }

    fn fetch_next_batch(&mut self) -> io::Result<bool> {
        use std::io::BufRead;
        
        if self.finished {
            return Ok(false);
        }

        let reader = self.stdout_reader.as_mut().ok_or_else(|| {
            io::Error::new(io::ErrorKind::Other, "No stdout reader available")
        })?;

        let mut line = String::new();
        while reader.read_line(&mut line)? > 0 {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&line) {
                // Check for batch data
                if let Some(batch) = parsed.get("batch") {
                    if let Some(texts) = batch.as_array() {
                        for text in texts {
                            if let Some(s) = text.as_str() {
                                self.buffer.push_back(s.to_string());
                            }
                        }
                        return Ok(true);
                    }
                }
                
                // Check for completion
                if parsed.get("done").and_then(|v| v.as_bool()) == Some(true) {
                    if let Some(total) = parsed.get("total").and_then(|v| v.as_u64()) {
                        println!(
                            "[pretraining-data crate] HuggingFace streaming completed: {} samples",
                            total
                        );
                    }
                    self.finished = true;
                    return Ok(false);
                }
            }
            line.clear();
        }

        // EOF reached
        self.finished = true;
        Ok(false)
    }

    fn cleanup(&mut self) {
        if let Some(mut child) = self.python_process.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for HFCorpusStream {
    fn drop(&mut self) {
        self.cleanup();
    }
}

impl Iterator for HFCorpusStream {
    type Item = io::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if we've reached max samples
        if let Some(max) = self.max_samples {
            if self.current_index >= max {
                return None;
            }
        }

        // Try to get from buffer first
        if let Some(text) = self.buffer.pop_front() {
            self.current_index += 1;
            return Some(Ok(text));
        }

        // Buffer is empty, fetch next batch
        if self.finished {
            return None;
        }

        match self.fetch_next_batch() {
            Ok(true) => {
                // Successfully fetched a batch
                if let Some(text) = self.buffer.pop_front() {
                    self.current_index += 1;
                    Some(Ok(text))
                } else {
                    None
                }
            }
            Ok(false) => {
                // No more data
                None
            }
            Err(e) => Some(Err(e)),
        }
    }
}
