use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::PathBuf;

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

        Ok(Self { shards })
    }

    pub fn stream(&self) -> io::Result<CorpusStream> {
        CorpusStream::new(self.shards.clone())
    }

    pub fn shard_paths(&self) -> &[PathBuf] {
        &self.shards
    }
}

pub struct CorpusStream {
    shards: Vec<PathBuf>,
    current_reader: Option<BufReader<File>>,
    next_shard: usize,
}

impl CorpusStream {
    fn new(shards: Vec<PathBuf>) -> io::Result<Self> {
        let mut stream = Self {
            shards,
            current_reader: None,
            next_shard: 0,
        };
        stream.open_next_shard()?;
        Ok(stream)
    }

    fn open_next_shard(&mut self) -> io::Result<bool> {
        while self.next_shard < self.shards.len() {
            let path = &self.shards[self.next_shard];
            self.next_shard += 1;
            match File::open(path) {
                Ok(file) => {
                    self.current_reader = Some(BufReader::new(file));
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
                    return Some(Ok(line));
                }
                Err(err) => return Some(Err(err)),
            }
        }
    }
}
