use std::{collections::VecDeque, sync::Arc};

use candle_core::{Device, Tensor};
use futures::future::BoxFuture;
use pretraining_data::corpora::StreamingCorpus;
use pretraining_data::TextCorpus;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
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
        };

        if !loader.prepare_next_epoch()? {
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

    fn prepare_next_epoch(&mut self) -> Result<bool> {
        let epoch = self.prepared_epoch;
        let mut stream = self.corpus.stream()?;

        let mut rng = StdRng::seed_from_u64(self.epoch_seed.wrapping_add(epoch as u64));
        let buffer_target = self.shuffle_buffer_size.max(self.micro_batch_size).max(1);
        let mut buffer = Vec::with_capacity(buffer_target);
        self.document_queue.clear();

        // Preparing epoch

        while let Some(line) = stream.next() {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            buffer.push(line);
            if buffer.len() >= buffer_target {
                self.flush_buffer(&mut buffer, &mut rng)?;
            }
        }

        if !buffer.is_empty() {
            self.flush_buffer(&mut buffer, &mut rng)?;
        }

        if self.document_queue.is_empty() {
            return Ok(false);
        }

        println!(
            "📚 Epoch {} ready: {} documents loaded",
            epoch,
            self.document_queue.len()
        );

        self.current_epoch = epoch;
        self.prepared_epoch = epoch + 1;
        Ok(true)
    }

    fn flush_buffer(&mut self, buffer: &mut Vec<String>, rng: &mut StdRng) -> Result<()> {
        buffer.shuffle(rng);
        for text in buffer.drain(..) {
            let encoding = self
                .tokenizer
                .encode(text, true)
                .map_err(|err| TrainingError::runtime(format!("tokenization failed: {}", err)))?;
            let mut ids = encoding.get_ids().to_vec();
            if let Some(token) = self.separator_token_id {
                ids.push(token);
            }
            if !ids.is_empty() {
                self.document_queue.push_back(ids);
                let _total = self.document_queue.len();
                // Removed verbose buffering logs
            }
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

                if !self.prepare_next_epoch()? {
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

        // Convert u32 tokens to i64 to avoid negative token ID issues
        let tokens_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        
        let batch_tokens =
            Tensor::from_slice(&tokens_i64, (self.micro_batch_size, target_len), &self.device)
                .map_err(|err| {
                    TrainingError::runtime(format!("failed to materialize token tensor: {}", err))
                })?;

        let batch_mask =
            Tensor::from_slice(&mask, (self.micro_batch_size, target_len), &self.device).map_err(
                |err| {
                    TrainingError::runtime(format!(
                        "failed to materialize attention mask tensor: {}",
                        err
                    ))
                },
            )?;

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
