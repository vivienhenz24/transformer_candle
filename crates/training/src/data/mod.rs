use std::{collections::VecDeque, sync::Arc};

use candle_core::{DType, Device, Tensor};
use futures::future::BoxFuture;
use pretraining_data::corpora::StreamingCorpus;
use pretraining_data::TextCorpus;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use rayon::prelude::*;
use tokenizers::Tokenizer;

use crate::TrainingError;

/// Result alias for data pipeline fallible operations.
pub type Result<T> = std::result::Result<T, TrainingError>;

/// Batch returned by dataset loaders.
#[derive(Debug)]
pub struct DataBatch {
    pub input_ids: Tensor,
    pub attention_mask: Tensor,
    pub lengths: Vec<usize>,
    pub micro_batch_size: usize,
    pub global_batch_size: usize,
    pub micro_batch_index: usize,
    pub micro_batches_per_step: usize,
    pub global_step: usize,
    pub epoch: usize,
}

/// Asynchronous-compatible loader abstraction.
pub trait DataLoader: Send {
    fn next_batch(&mut self) -> BoxFuture<'_, Result<Option<DataBatch>>>;
}

/// Blocking adapter around an async-friendly loader.
pub struct BlockingDataLoader<L>
where
    L: DataLoader,
{
    inner: L,
}

impl<L> BlockingDataLoader<L>
where
    L: DataLoader,
{
    pub fn new(inner: L) -> Self {
        Self { inner }
    }

    pub fn next_batch(&mut self) -> Result<Option<DataBatch>> {
        futures::executor::block_on(self.inner.next_batch())
    }

    pub fn into_inner(self) -> L {
        self.inner
    }
}

/// Streaming text loader that tokenizes shards on the fly and surfaces
/// micro-batches aligned with gradient accumulation.
pub struct StreamingTextDataLoader {
    corpus: StreamingCorpus,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    sequence_length: usize,
    micro_batch_size: usize,
    micro_batches_per_step: usize,
    global_batch_size: usize,
    epoch_seed: u64,
    current_epoch: usize,
    prepared_epoch: usize,
    global_step: usize,
    micro_batch_index: usize,
    trim_padding: bool,
    shuffle_buffer_size: usize,
    pad_token_id: u32,
    separator_token_id: Option<u32>,
    token_buffer: VecDeque<u32>,
    document_queue: VecDeque<Vec<u32>>,
    active_stream: Option<Box<dyn Iterator<Item = std::io::Result<String>> + Send>>,
    stream_rng: StdRng,
}

impl StreamingTextDataLoader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        corpus: StreamingCorpus,
        tokenizer: Arc<Tokenizer>,
        device: Device,
        sequence_length: usize,
        global_batch_size: usize,
        gradient_accumulation_steps: usize,
        seed: u64,
        shuffle_buffer_size: Option<usize>,
    ) -> Result<Self> {
        if sequence_length == 0 {
            return Err(TrainingError::initialization(
                "sequence_length must be greater than zero",
            ));
        }

        if global_batch_size == 0 {
            return Err(TrainingError::initialization(
                "global batch size must be greater than zero",
            ));
        }

        // Data loader initialized

        let micro_batches_per_step = gradient_accumulation_steps.max(1);
        let micro_batch_size = if global_batch_size % micro_batches_per_step == 0 {
            global_batch_size / micro_batches_per_step
        } else {
            return Err(TrainingError::initialization(
                "global batch size must be divisible by gradient accumulation steps",
            ));
        };

        let pad_token_id = tokenizer
            .get_padding()
            .map(|params| params.pad_id as u32)
            .unwrap_or(0);

        let mut loader = Self {
            corpus,
            tokenizer,
            device,
            sequence_length,
            micro_batch_size,
            micro_batches_per_step,
            global_batch_size,
            epoch_seed: seed,
            current_epoch: 0,
            prepared_epoch: 0,
            global_step: 0,
            micro_batch_index: 0,
            trim_padding: true,
            shuffle_buffer_size: shuffle_buffer_size.unwrap_or(4096),
            pad_token_id,
            separator_token_id: None,
            token_buffer: VecDeque::with_capacity(sequence_length * micro_batch_size),
            document_queue: VecDeque::new(),
            active_stream: None,
            stream_rng: StdRng::seed_from_u64(seed),
        };

        // Initialize streaming - load enough for initial shuffle buffer
        if !loader.load_more_documents()? {
            return Err(TrainingError::initialization(
                "training corpus is empty; no documents available",
            ));
        }

        Ok(loader)
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    pub fn set_trim_padding(&mut self, enabled: bool) {
        self.trim_padding = enabled;
    }

    pub fn set_separator_token(&mut self, token_id: Option<u32>) {
        self.separator_token_id = token_id;
    }

    fn load_more_documents(&mut self) -> Result<bool> {
        // Initialize stream if needed
        if self.active_stream.is_none() {
            println!("📚 Initializing data stream...");
            self.active_stream = Some(Box::new(self.corpus.stream()?));
        }

        let buffer_target = self.shuffle_buffer_size.max(self.micro_batch_size).max(1);

        // Collect lines from stream
        let mut temp_buffer = Vec::new();
        let mut stream_exhausted = false;
        let mut lines_loaded = 0;

        while lines_loaded < buffer_target && self.document_queue.len() < buffer_target {
            // Take stream temporarily to avoid borrow conflicts
            let mut stream = self.active_stream.take().unwrap();

            match stream.next() {
                Some(Ok(line)) => {
                    self.active_stream = Some(stream);
                    if !line.is_empty() {
                        temp_buffer.push(line);
                        lines_loaded += 1;

                        // Flush when buffer is full
                        if temp_buffer.len() >= buffer_target {
                            temp_buffer.shuffle(&mut self.stream_rng);
                            self.tokenize_and_queue(temp_buffer)?;
                            temp_buffer = Vec::new();
                        }
                    }
                }
                Some(Err(e)) => {
                    return Err(TrainingError::runtime(format!("stream error: {}", e)));
                }
                None => {
                    stream_exhausted = true;
                    break;
                }
            }
        }

        // Flush any remaining lines
        if !temp_buffer.is_empty() {
            temp_buffer.shuffle(&mut self.stream_rng);
            self.tokenize_and_queue(temp_buffer)?;
        }

        // Handle stream exhaustion
        if stream_exhausted {
            if self.document_queue.is_empty() {
                return Ok(false);
            }
            // Reset stream for next epoch
            self.active_stream = None;
            self.current_epoch += 1;
            self.prepared_epoch = self.current_epoch + 1;
            println!(
                "✅ Epoch {} complete, starting epoch {}",
                self.current_epoch - 1,
                self.current_epoch
            );
        }

        Ok(!self.document_queue.is_empty())
    }

    fn tokenize_and_queue(&mut self, buffer: Vec<String>) -> Result<()> {
        // Parallel tokenization for speed
        let tokenizer = self.tokenizer.clone();
        let separator_token = self.separator_token_id;

        let tokenized: Vec<Vec<u32>> = buffer
            .par_iter()
            .filter_map(|text| match tokenizer.encode(text.as_str(), true) {
                Ok(encoding) => {
                    let mut ids = encoding.get_ids().to_vec();
                    if let Some(token) = separator_token {
                        ids.push(token);
                    }
                    if !ids.is_empty() {
                        Some(ids)
                    } else {
                        None
                    }
                }
                Err(_) => None,
            })
            .collect();

        // Add to queue
        for ids in tokenized {
            self.document_queue.push_back(ids);
        }

        Ok(())
    }

    fn next_sequence(&mut self) -> Result<Option<(Vec<u32>, usize)>> {
        loop {
            if self.token_buffer.len() >= self.sequence_length {
                let mut seq = Vec::with_capacity(self.sequence_length);
                for _ in 0..self.sequence_length {
                    if let Some(token) = self.token_buffer.pop_front() {
                        seq.push(token);
                    }
                }
                return Ok(Some((seq, self.sequence_length)));
            }

            if let Some(mut doc) = self.document_queue.pop_front() {
                if !doc.is_empty() {
                    self.token_buffer.extend(doc.drain(..));
                    continue;
                }
            } else {
                if !self.token_buffer.is_empty() {
                    let mut seq = Vec::with_capacity(self.sequence_length);
                    let mut valid = 0;
                    while let Some(token) = self.token_buffer.pop_front() {
                        seq.push(token);
                        valid += 1;
                    }
                    if seq.is_empty() {
                        return Ok(None);
                    }
                    seq.resize(self.sequence_length, self.pad_token_id);
                    return Ok(Some((seq, valid)));
                }

                if !self.load_more_documents()? {
                    return Ok(None);
                }
            }
        }
    }

    fn build_batch(&mut self) -> Result<Option<DataBatch>> {
        let mut sequences = Vec::with_capacity(self.micro_batch_size);
        while sequences.len() < self.micro_batch_size {
            match self.next_sequence()? {
                Some(seq) => sequences.push(seq),
                None => break,
            }
        }

        if sequences.len() < self.micro_batch_size {
            return Ok(None);
        }

        let effective_lengths: Vec<usize> = sequences.iter().map(|(_, len)| *len).collect();
        let target_len = if self.trim_padding {
            effective_lengths.iter().copied().max().unwrap_or(1).max(1)
        } else {
            self.sequence_length
        };

        let mut tokens = Vec::with_capacity(sequences.len() * target_len);
        let mut mask: Vec<u32> = Vec::with_capacity(sequences.len() * target_len);

        for (mut seq, valid) in sequences {
            if seq.len() < target_len {
                seq.resize(target_len, self.pad_token_id);
            }
            for idx in 0..target_len {
                let token = seq.get(idx).copied().unwrap_or(self.pad_token_id);
                tokens.push(token);
                mask.push(if idx < valid { 1 } else { 0 });
            }
        }

        let tokens_i64: Vec<i64> = tokens.into_iter().map(|t| t as i64).collect();
        let mask_u32: Vec<u32> = mask;

        let cpu_device = Device::Cpu;

        let batch_tokens =
            Tensor::from_vec(tokens_i64, (self.micro_batch_size, target_len), &cpu_device)
                .and_then(|t| t.to_device(&self.device))
                .map_err(|err| {
                    TrainingError::runtime(format!("failed to materialize token tensor: {}", err))
                })?;

        let batch_mask =
            Tensor::from_vec(mask_u32, (self.micro_batch_size, target_len), &cpu_device)
                .and_then(|t| t.to_device(&self.device))
                .map_err(|err| {
                    TrainingError::runtime(format!(
                        "failed to materialize attention mask tensor: {}",
                        err
                    ))
                })?;

        let micro_batch_index = self.micro_batch_index;
        let micro_batches_per_step = self.micro_batches_per_step;
        let epoch = self.current_epoch;
        let global_step = self.global_step;

        if micro_batches_per_step == 0 || micro_batch_index + 1 == micro_batches_per_step {
            self.micro_batch_index = 0;
            self.global_step = self.global_step.wrapping_add(1);
        } else {
            self.micro_batch_index += 1;
        }

        Ok(Some(DataBatch {
            input_ids: batch_tokens,
            attention_mask: batch_mask,
            lengths: effective_lengths,
            micro_batch_size: self.micro_batch_size,
            global_batch_size: self.global_batch_size,
            micro_batch_index,
            micro_batches_per_step,
            global_step,
            epoch,
        }))
    }
}

impl DataLoader for StreamingTextDataLoader {
    fn next_batch(&mut self) -> BoxFuture<'_, Result<Option<DataBatch>>> {
        Box::pin(async move { self.build_batch() })
    }
}
