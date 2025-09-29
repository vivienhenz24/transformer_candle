use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Split a text corpus into numbered shards, each capped at `max_lines` lines.
/// Returns the ordered list of shard paths.
pub fn shard_text_by_lines(
    source: &Path,
    destination: &Path,
    prefix: &str,
    max_lines: usize,
) -> io::Result<Vec<PathBuf>> {
    if max_lines == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "max_lines must be greater than zero",
        ));
    }

    fs::create_dir_all(destination)?;

    let file = File::open(source)?;
    let reader = BufReader::new(file);

    let mut shards = Vec::new();
    let mut shard_index = 0usize;
    let mut lines_in_shard = 0usize;
    let mut writer: Option<BufWriter<File>> = None;

    for line in reader.lines() {
        let line = line?;

        if writer.is_none() || lines_in_shard >= max_lines {
            if let Some(mut existing) = writer.take() {
                existing.flush()?;
            }
            let (new_writer, path) = open_shard(destination, prefix, shard_index)?;
            shards.push(path);
            writer = Some(new_writer);
            shard_index += 1;
            lines_in_shard = 0;
        }

        if let Some(current) = writer.as_mut() {
            writeln!(current, "{}", line)?;
        }
        lines_in_shard += 1;
    }

    match writer {
        Some(mut active) => {
            active.flush()?;
        }
        None => {
            let (mut writer, path) = open_shard(destination, prefix, shard_index)?;
            writer.flush()?;
            shards.push(path);
        }
    }

    Ok(shards)
}

fn open_shard(
    destination: &Path,
    prefix: &str,
    index: usize,
) -> io::Result<(BufWriter<File>, PathBuf)> {
    let filename = format!("{prefix}-{index:05}.txt");
    let path = destination.join(filename);
    let file = File::create(&path)?;
    Ok((BufWriter::new(file), path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn shards_respect_line_cap() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("corpus.txt");
        let mut file = File::create(&source).unwrap();
        for i in 0..5 {
            writeln!(file, "line-{i}").unwrap();
        }

        let dest = dir.path().join("output");
        let shards = shard_text_by_lines(&source, &dest, "train", 2).unwrap();
        assert_eq!(shards.len(), 3);

        let contents: Vec<_> = shards
            .iter()
            .map(|path| {
                let data = std::fs::read_to_string(path).unwrap();
                data.lines()
                    .map(|line| line.to_string())
                    .collect::<Vec<_>>()
            })
            .collect();

        assert_eq!(contents[0], vec!["line-0", "line-1"]);
        assert_eq!(contents[1], vec!["line-2", "line-3"]);
        assert_eq!(contents[2], vec!["line-4"]);
    }

    #[test]
    fn creates_single_empty_shard_for_empty_corpus() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("empty.txt");
        File::create(&source).unwrap();

        let dest = dir.path().join("output");
        let shards = shard_text_by_lines(&source, &dest, "val", 3).unwrap();

        assert_eq!(shards.len(), 1);
        let data = std::fs::read_to_string(&shards[0]).unwrap();
        assert!(data.is_empty());
    }

    #[test]
    fn reject_zero_line_cap() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("file.txt");
        File::create(&source).unwrap();

        let dest = dir.path().join("output");
        let err = shard_text_by_lines(&source, &dest, "train", 0).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }
}
