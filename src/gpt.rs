use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{loss, Dropout, Embedding, LayerNorm, Linear, Module, VarBuilder};

/// Single head of self-attention
#[derive(Debug)]
pub struct Head {
    /// Linear layer for keys
    key: Linear,
    /// Linear layer for queries  
    query: Linear,
    /// Linear layer for values
    value: Linear,
    /// Dropout layer
    dropout: Dropout,
    /// Head size (dimension of each attention head)
    head_size: usize,
    /// Block size (sequence length)
    block_size: usize,
    /// Scale factor for attention scores (1/sqrt(head_size))
    scale: f64,
}

impl Head {
    /// Create a new attention head
    pub fn new(
        n_embd: usize,     // Embedding dimension
        head_size: usize,  // Size of this attention head
        block_size: usize, // Maximum sequence length
        dropout_rate: f32, // Dropout probability
        vb: VarBuilder,    // Variable builder for parameters
    ) -> CandleResult<Self> {
        // Create linear layers without bias
        let key = candle_nn::linear_no_bias(n_embd, head_size, vb.pp("key"))?;
        let query = candle_nn::linear_no_bias(n_embd, head_size, vb.pp("query"))?;
        let value = candle_nn::linear_no_bias(n_embd, head_size, vb.pp("value"))?;

        // Create dropout layer
        let dropout = Dropout::new(dropout_rate);

        // Scale factor for attention scores
        let scale = 1.0 / (head_size as f64).sqrt();

        Ok(Head {
            key,
            query,
            value,
            dropout,
            head_size,
            block_size,
            scale,
        })
    }

    /// Create causal mask (lower triangular matrix)
    /// This prevents attention to future tokens
    pub fn create_causal_mask(&self, seq_len: usize, device: &Device) -> CandleResult<Tensor> {
        // Create a lower triangular matrix filled with 1s
        let mut mask_data = vec![0.0f32; seq_len * seq_len];

        for i in 0..seq_len {
            for j in 0..=i {
                mask_data[i * seq_len + j] = 1.0;
            }
        }

        let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), device)?;

        // Convert to boolean-like mask where 0 means mask out (set to -inf)
        // We'll use this by setting masked positions to a very negative number
        let ones = Tensor::ones((seq_len, seq_len), DType::F32, device)?;
        let mask = ones.sub(&mask)?; // Flip: 1 where we want to mask, 0 where we don't
        let neg_inf = mask.affine(-1e9, 0.0)?; // Very negative number for masked positions

        Ok(neg_inf)
    }

    /// Forward pass through the attention head
    pub fn forward(&self, x: &Tensor, train: bool) -> CandleResult<Tensor> {
        // x shape: (batch_size, seq_len, n_embd)
        let (_batch_size, seq_len, _n_embd) = x.dims3()?;

        // Compute queries, keys, and values
        // Each has shape: (batch_size, seq_len, head_size)
        let queries = self.query.forward(x)?; // Q
        let keys = self.key.forward(x)?; // K
        let values = self.value.forward(x)?; // V

        // Compute attention scores: Q @ K^T
        // queries: (B, T, head_size), keys: (B, T, head_size)
        // We want: (B, T, T) where T = seq_len
        let keys_transposed = keys.transpose(1, 2)?; // (B, head_size, T)
        let attention_scores = queries.matmul(&keys_transposed)?; // (B, T, T)

        // Scale the attention scores
        let scaled_scores = attention_scores.affine(self.scale as f64, 0.0)?;

        // Apply causal mask to prevent looking at future tokens
        let mask = self.create_causal_mask(seq_len, x.device())?;
        // Broadcast mask to match the batch dimension
        let mask = mask.unsqueeze(0)?.broadcast_as(scaled_scores.shape())?;
        let masked_scores = scaled_scores.add(&mask)?;

        // Apply softmax to get attention weights
        let attention_weights = candle_nn::ops::softmax_last_dim(&masked_scores)?;

        // Apply dropout during training
        let attention_weights = if train {
            self.dropout.forward(&attention_weights, train)?
        } else {
            attention_weights
        };

        // Apply attention weights to values: attention_weights @ V
        // attention_weights: (B, T, T), values: (B, T, head_size)
        // Result: (B, T, head_size)
        let output = attention_weights.matmul(&values)?;

        Ok(output)
    }

    /// Size of this attention head (primarily for inspection and testing)
    pub fn head_size(&self) -> usize {
        self.head_size
    }

    /// Maximum sequence length this head was configured for
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

/// Multi-head attention layer
#[derive(Debug)]
pub struct MultiHeadAttention {
    heads: Vec<Head>,
    proj: Linear,
    dropout: Dropout,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer
    pub fn new(
        n_embd: usize,     // Embedding dimension
        n_head: usize,     // Number of attention heads
        block_size: usize, // Maximum sequence length
        dropout_rate: f32, // Dropout probability
        vb: VarBuilder,    // Variable builder for parameters
    ) -> CandleResult<Self> {
        assert_eq!(n_embd % n_head, 0, "n_embd must be divisible by n_head");

        let head_size = n_embd / n_head;
        let mut heads = Vec::new();

        // Create individual attention heads
        for i in 0..n_head {
            let head = Head::new(
                n_embd,
                head_size,
                block_size,
                dropout_rate,
                vb.pp(&format!("head_{}", i)),
            )?;
            heads.push(head);
        }

        // Output projection layer
        let proj = candle_nn::linear(n_embd, n_embd, vb.pp("proj"))?;
        let dropout = Dropout::new(dropout_rate);

        Ok(MultiHeadAttention {
            heads,
            proj,
            dropout,
        })
    }

    /// Forward pass through multi-head attention
    pub fn forward(&self, x: &Tensor, train: bool) -> CandleResult<Tensor> {
        // Compute attention for each head
        let mut head_outputs = Vec::new();
        for head in &self.heads {
            let head_output = head.forward(x, train)?;
            head_outputs.push(head_output);
        }

        // Concatenate all head outputs along the last dimension
        let concatenated = Tensor::cat(&head_outputs, 2)?;

        // Apply output projection and dropout
        let projected = self.proj.forward(&concatenated)?;
        let output = if train {
            self.dropout.forward(&projected, train)?
        } else {
            projected
        };

        Ok(output)
    }

    /// Number of attention heads in this layer
    pub fn num_heads(&self) -> usize {
        self.heads.len()
    }
}

/// Feed-forward neural network (MLP) component of transformer block
#[derive(Debug)]
pub struct FeedForward {
    /// First linear layer: n_embd -> 4 * n_embd
    c_fc: Linear,
    /// Second linear layer: 4 * n_embd -> n_embd  
    c_proj: Linear,
    /// Dropout layer for regularization
    dropout: Dropout,
}

impl FeedForward {
    /// Create a new feed-forward network
    pub fn new(
        n_embd: usize,     // Embedding dimension
        dropout_rate: f32, // Dropout probability
        vb: VarBuilder,    // Variable builder for parameters
    ) -> CandleResult<Self> {
        let inner_dim = 4 * n_embd; // Expand to 4x the embedding dimension

        // Create the two linear layers
        let c_fc = candle_nn::linear(n_embd, inner_dim, vb.pp("c_fc"))?; // Expansion layer
        let c_proj = candle_nn::linear(inner_dim, n_embd, vb.pp("c_proj"))?; // Contraction layer
        let dropout = Dropout::new(dropout_rate);

        Ok(FeedForward {
            c_fc,
            c_proj,
            dropout,
        })
    }

    /// Forward pass through the feed-forward network
    /// Architecture: Input -> Linear -> ReLU -> Dropout -> Linear -> Output
    pub fn forward(&self, x: &Tensor, train: bool) -> CandleResult<Tensor> {
        // First linear layer: n_embd -> 4 * n_embd
        let x = self.c_fc.forward(x)?;

        // ReLU activation
        let x = x.relu()?;

        // Dropout (applied in training mode)
        let x = if train {
            self.dropout.forward(&x, train)?
        } else {
            x
        };

        // Second linear layer: 4 * n_embd -> n_embd
        let x = self.c_proj.forward(&x)?;

        Ok(x)
    }
}

/// Transformer block combining self-attention and feed-forward with residual connections
#[derive(Debug)]
pub struct Block {
    /// Multi-head self-attention layer
    sa: MultiHeadAttention,
    /// Feed-forward network
    ffwd: FeedForward,
    /// Layer normalization before self-attention
    ln1: LayerNorm,
    /// Layer normalization before feed-forward
    ln2: LayerNorm,
}

impl Block {
    /// Create a new transformer block
    pub fn new(
        n_embd: usize,     // Embedding dimension
        n_head: usize,     // Number of attention heads
        block_size: usize, // Maximum sequence length
        dropout_rate: f32, // Dropout probability
        vb: VarBuilder,    // Variable builder for parameters
    ) -> CandleResult<Self> {
        // Create multi-head attention
        let sa = MultiHeadAttention::new(n_embd, n_head, block_size, dropout_rate, vb.pp("attn"))?;

        // Create feed-forward network
        let ffwd = FeedForward::new(n_embd, dropout_rate, vb.pp("mlp"))?;

        // Create layer normalization layers
        let ln1 = candle_nn::layer_norm(n_embd, 1e-5, vb.pp("ln1"))?;
        let ln2 = candle_nn::layer_norm(n_embd, 1e-5, vb.pp("ln2"))?;

        Ok(Block { sa, ffwd, ln1, ln2 })
    }

    /// Forward pass through the transformer block
    /// Uses pre-layer normalization with residual connections:
    /// x = x + attention(layer_norm(x))
    /// x = x + ffwd(layer_norm(x))
    pub fn forward(&self, x: &Tensor, train: bool) -> CandleResult<Tensor> {
        // First residual connection: x + self_attention(layer_norm(x))
        let normed_x1 = self.ln1.forward(x)?;
        let attn_out = self.sa.forward(&normed_x1, train)?;
        let x = x.add(&attn_out)?; // Residual connection

        // Second residual connection: x + feed_forward(layer_norm(x))
        let normed_x2 = self.ln2.forward(&x)?;
        let ffwd_out = self.ffwd.forward(&normed_x2, train)?;
        let x = x.add(&ffwd_out)?; // Residual connection

        Ok(x)
    }
}

/// Configuration for GPT language model
#[derive(Debug, Clone)]
pub struct GPTConfig {
    pub vocab_size: usize, // Size of vocabulary
    pub block_size: usize, // Maximum sequence length
    pub n_embd: usize,     // Embedding dimension
    pub n_head: usize,     // Number of attention heads
    pub n_layer: usize,    // Number of transformer blocks
    pub dropout_rate: f32, // Dropout probability
}

impl Default for GPTConfig {
    fn default() -> Self {
        GPTConfig {
            vocab_size: 65,    // Character-level vocabulary
            block_size: 256,   // Sequence length
            n_embd: 384,       // Embedding dimension
            n_head: 6,         // Number of heads
            n_layer: 6,        // Number of layers
            dropout_rate: 0.1, // Dropout rate
        }
    }
}

/// GPT Language Model combining all transformer components
#[derive(Debug)]
pub struct GPTLanguageModel {
    /// Configuration
    config: GPTConfig,
    /// Token embedding table: vocab_size -> n_embd
    token_embedding_table: Embedding,
    /// Position embedding table: block_size -> n_embd
    position_embedding_table: Embedding,
    /// Stack of transformer blocks
    blocks: Vec<Block>,
    /// Final layer normalization
    ln_f: LayerNorm,
    /// Language modeling head: n_embd -> vocab_size
    lm_head: Linear,
}

impl GPTLanguageModel {
    /// Create a new GPT language model
    pub fn new(config: GPTConfig, vb: VarBuilder) -> CandleResult<Self> {
        // Create token embedding table
        let token_embedding_table =
            candle_nn::embedding(config.vocab_size, config.n_embd, vb.pp("wte"))?;

        // Create position embedding table
        let position_embedding_table =
            candle_nn::embedding(config.block_size, config.n_embd, vb.pp("wpe"))?;

        // Create transformer blocks
        let mut blocks = Vec::new();
        for i in 0..config.n_layer {
            let block = Block::new(
                config.n_embd,
                config.n_head,
                config.block_size,
                config.dropout_rate,
                vb.pp(&format!("h.{}", i)),
            )?;
            blocks.push(block);
        }

        // Create final layer normalization
        let ln_f = candle_nn::layer_norm(config.n_embd, 1e-5, vb.pp("ln_f"))?;

        // Create language modeling head
        let lm_head = candle_nn::linear(config.n_embd, config.vocab_size, vb.pp("lm_head"))?;

        Ok(GPTLanguageModel {
            config,
            token_embedding_table,
            position_embedding_table,
            blocks,
            ln_f,
            lm_head,
        })
    }

    /// Forward pass through the GPT model
    /// Returns logits and optionally loss if targets are provided
    pub fn forward(
        &self,
        idx: &Tensor,
        targets: Option<&Tensor>,
        train: bool,
    ) -> CandleResult<(Tensor, Option<Tensor>)> {
        let (batch_size, seq_len) = idx.dims2()?;

        // Check sequence length doesn't exceed block size
        if seq_len > self.config.block_size {
            return Err(candle_core::Error::Msg(format!(
                "Sequence length {} exceeds block size {}",
                seq_len, self.config.block_size
            )));
        }

        // Token embeddings: (batch_size, seq_len) -> (batch_size, seq_len, n_embd)
        let tok_emb = self.token_embedding_table.forward(idx)?;

        // Position embeddings: (seq_len,) -> (seq_len, n_embd) -> broadcast to (batch_size, seq_len, n_embd)
        let pos_indices = Tensor::arange(0u32, seq_len as u32, idx.device())?;
        let pos_emb = self.position_embedding_table.forward(&pos_indices)?; // (seq_len, n_embd)
        let pos_emb = pos_emb.unsqueeze(0)?; // (1, seq_len, n_embd)
        let pos_emb = pos_emb.broadcast_as(tok_emb.shape())?; // (batch_size, seq_len, n_embd)

        // Combine token and position embeddings
        let mut x = tok_emb.add(&pos_emb)?;

        // Pass through transformer blocks
        for block in &self.blocks {
            x = block.forward(&x, train)?;
        }

        // Final layer normalization
        x = self.ln_f.forward(&x)?;

        // Language modeling head: (batch_size, seq_len, n_embd) -> (batch_size, seq_len, vocab_size)
        let logits = self.lm_head.forward(&x)?;

        // Compute loss if targets are provided
        let loss = if let Some(targets) = targets {
            // Reshape for cross-entropy loss computation
            // logits: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
            // targets: (batch_size, seq_len) -> (batch_size * seq_len,)
            let logits_flat = logits.reshape(&[batch_size * seq_len, self.config.vocab_size])?;
            let targets_flat = targets.reshape(&[batch_size * seq_len])?;

            // Compute cross-entropy loss
            let loss = loss::cross_entropy(&logits_flat, &targets_flat)?;
            Some(loss)
        } else {
            None
        };

        Ok((logits, loss))
    }

    /// Sample from a probability distribution using multinomial sampling
    /// Returns indices sampled from the distribution
    fn multinomial_sample(&self, probs: &Tensor) -> CandleResult<Tensor> {
        let (batch_size, vocab_size) = probs.dims2()?;
        let device = probs.device();

        // Convert probabilities to Vec for sampling
        let mut sampled_tokens = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            // Get probabilities for this batch element
            let batch_probs = probs.get(b)?;
            let prob_vec = batch_probs.to_vec1::<f32>()?;

            // Create cumulative distribution
            let mut cum_prob = 0.0f32;
            let mut cumulative = Vec::with_capacity(vocab_size);
            for &p in &prob_vec {
                cum_prob += p;
                cumulative.push(cum_prob);
            }

            // Sample using uniform random number
            let random_val = fastrand::f32();

            // Find the token corresponding to this random value
            let mut sampled_token = vocab_size - 1; // Default to last token
            for (i, &cum_p) in cumulative.iter().enumerate() {
                if random_val <= cum_p {
                    sampled_token = i;
                    break;
                }
            }

            sampled_tokens.push(sampled_token as u32);
        }

        // Convert back to tensor
        let result = Tensor::from_vec(sampled_tokens, (batch_size, 1), device)?;
        Ok(result)
    }

    /// Generate new tokens (inference mode)
    /// Takes a sequence and generates the next token using multinomial sampling
    pub fn generate_next(&self, idx: &Tensor) -> CandleResult<Tensor> {
        let (_batch_size, seq_len) = idx.dims2()?;

        // If sequence is longer than block_size, take the last block_size tokens
        let idx = if seq_len > self.config.block_size {
            let start = seq_len - self.config.block_size;
            idx.narrow(1, start, self.config.block_size)?
        } else {
            idx.clone()
        };

        // Forward pass without targets (inference mode)
        let (logits, _) = self.forward(&idx, None, false)?;

        // Take logits from the last position
        let last_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?; // (batch_size, 1, vocab_size)
        let last_logits = last_logits.squeeze(1)?; // (batch_size, vocab_size)

        // Apply softmax to get probabilities
        let probs = candle_nn::ops::softmax_last_dim(&last_logits)?;

        // Sample from the distribution using multinomial sampling
        let next_token = self.multinomial_sample(&probs)?; // (batch_size, 1)

        Ok(next_token)
    }

    /// Generate a sequence of tokens using specified sampling parameters.
    pub fn generate_with_sampling(
        &self,
        context: &Tensor,
        max_new_tokens: usize,
        temperature: f64,
        top_k: Option<usize>,
        top_p: Option<f64>,
    ) -> CandleResult<Tensor> {
        let (_batch_size, _initial_seq_len) = context.dims2()?;

        // Start with the input context
        let mut current_sequence = context.clone();

        // Generate tokens one by one
        for _token_idx in 0..max_new_tokens {
            let (_, current_seq_len) = current_sequence.dims2()?;

            // Crop context to last block_size tokens if it's too long
            let model_input = if current_seq_len > self.config.block_size {
                let start_idx = current_seq_len - self.config.block_size;
                current_sequence.narrow(1, start_idx, self.config.block_size)?
            } else {
                current_sequence.clone()
            };

            // Forward pass to get logits for next token
            let (logits, _) = self.forward(&model_input, None, false)?;

            // Extract logits for the last position (next token prediction)
            let last_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?; // (batch_size, 1, vocab_size)
            let last_logits = last_logits.squeeze(1)?; // (batch_size, vocab_size)

            let next_token = sample_next_token(&last_logits, temperature, top_k, top_p)?; // (batch_size, 1)

            // Append the sampled token to the current sequence
            current_sequence = Tensor::cat(&[current_sequence, next_token], 1)?;
        }

        Ok(current_sequence)
    }

    /// Generate using default sampling parameters (temperature=1.0, no top-k).
    pub fn generate(&self, context: &Tensor, max_new_tokens: usize) -> CandleResult<Tensor> {
        self.generate_with_sampling(context, max_new_tokens, 1.0, None, None)
    }

    /// Get the model configuration
    pub fn config(&self) -> &GPTConfig {
        &self.config
    }

    /// Get the number of parameters in the model
    pub fn count_parameters(&self) -> usize {
        // This is a rough estimate - in practice you'd iterate through all parameters
        let token_emb_params = self.config.vocab_size * self.config.n_embd;
        let pos_emb_params = self.config.block_size * self.config.n_embd;
        let block_params = self.config.n_layer
            * (
                // Attention: 3 * (n_embd * head_size) * n_head + n_embd * n_embd (projection)
                4 * self.config.n_embd * self.config.n_embd +
            // Feed-forward: n_embd * 4*n_embd + 4*n_embd * n_embd
            8 * self.config.n_embd * self.config.n_embd +
            // Layer norms: 2 * n_embd
            2 * self.config.n_embd
            );
        let lm_head_params = self.config.n_embd * self.config.vocab_size;
        let final_ln_params = self.config.n_embd;

        token_emb_params + pos_emb_params + block_params + lm_head_params + final_ln_params
    }

    /// Sample the next token from logits using the provided sampling parameters.
    pub fn sample_next_token(
        &self,
        logits: &Tensor,
        temperature: f64,
        top_k: Option<usize>,
        top_p: Option<f64>,
    ) -> CandleResult<Tensor> {
        sample_next_token(logits, temperature, top_k, top_p)
    }
}

fn sample_next_token(
    logits: &Tensor,
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
) -> CandleResult<Tensor> {
    let (batch_size, _vocab_size) = logits.dims2()?;
    let device = logits.device();
    let logits_vec = logits.to_vec2::<f32>()?;
    let mut sampled_tokens = Vec::with_capacity(batch_size);

    for row in logits_vec.iter() {
        let idx = sample_from_logits_row(row, temperature, top_k, top_p);
        sampled_tokens.push(idx as u32);
    }

    Tensor::from_vec(sampled_tokens, (batch_size, 1), device)
}

fn sample_from_logits_row(
    logits: &[f32],
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
) -> usize {
    if logits.is_empty() {
        return 0;
    }

    if temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
    }

    let inv_temp = (1.0 / temperature.max(1e-4)) as f32;
    let mut adjusted: Vec<f32> = logits.iter().map(|&logit| logit * inv_temp).collect();

    if let Some(mut k) = top_k {
        if k == 0 {
            k = 1;
        }
        if k < adjusted.len() {
            let mut indices: Vec<usize> = (0..adjusted.len()).collect();
            indices.sort_unstable_by(|a, b| {
                adjusted[*b]
                    .partial_cmp(&adjusted[*a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for &idx in indices.iter().skip(k) {
                adjusted[idx] = f32::NEG_INFINITY;
            }
        }
    }

    let max_logit = adjusted.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_values = Vec::with_capacity(adjusted.len());
    let mut sum = 0.0;
    for &logit in &adjusted {
        let value = if logit.is_finite() {
            (logit - max_logit).exp()
        } else {
            0.0
        };
        exp_values.push(value);
        sum += value;
    }

    if sum <= f32::EPSILON {
        return fastrand::usize(0..adjusted.len());
    }

    let mut probabilities: Vec<f32> = exp_values.iter().map(|value| value / sum).collect();

    if let Some(p_threshold) = top_p {
        let mut pairs: Vec<(usize, f32)> = probabilities.iter().cloned().enumerate().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cumulative = 0.0f32;
        let mut allowed = vec![false; probabilities.len()];
        for (idx, prob) in pairs {
            cumulative += prob;
            allowed[idx] = true;
            if cumulative >= p_threshold as f32 {
                break;
            }
        }

        for (idx, prob) in probabilities.iter_mut().enumerate() {
            if !allowed[idx] {
                *prob = 0.0;
            }
        }

        let renorm: f32 = probabilities.iter().sum();
        if renorm > f32::EPSILON {
            for prob in probabilities.iter_mut() {
                *prob /= renorm;
            }
        }
    }

    let mut cumulative = 0.0;
    let sample = fastrand::f32();
    for (idx, prob) in probabilities.iter().enumerate() {
        cumulative += *prob;
        if sample <= cumulative {
            return idx;
        }
    }

    probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}
