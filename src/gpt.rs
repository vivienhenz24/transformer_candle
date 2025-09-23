use candle_core::{Device, Tensor, Result as CandleResult, DType};
use candle_nn::{Linear, VarBuilder, Module, Dropout};

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
        n_embd: usize,      // Embedding dimension
        head_size: usize,   // Size of this attention head
        block_size: usize,  // Maximum sequence length
        dropout_rate: f32,  // Dropout probability
        vb: VarBuilder,     // Variable builder for parameters
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
    fn create_causal_mask(&self, seq_len: usize, device: &Device) -> CandleResult<Tensor> {
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
        let queries = self.query.forward(x)?;  // Q
        let keys = self.key.forward(x)?;       // K  
        let values = self.value.forward(x)?;   // V
        
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
}

/// Multi-head attention layer
#[derive(Debug)]
pub struct MultiHeadAttention {
    /// Vector of attention heads
    heads: Vec<Head>,
    /// Output projection layer
    proj: Linear,
    /// Dropout for output
    dropout: Dropout,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer
    pub fn new(
        n_embd: usize,      // Embedding dimension
        n_head: usize,      // Number of attention heads
        block_size: usize,  // Maximum sequence length
        dropout_rate: f32,  // Dropout probability
        vb: VarBuilder,     // Variable builder for parameters
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;
    
    #[test]
    fn test_head_creation() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let head = Head::new(512, 64, 128, 0.1, vb).unwrap();
        assert_eq!(head.head_size, 64);
        assert_eq!(head.block_size, 128);
    }
    
    #[test]
    fn test_causal_mask() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let head = Head::new(512, 64, 128, 0.1, vb).unwrap();
        let mask = head.create_causal_mask(4, &device).unwrap();
        
        // Check that mask has correct shape
        assert_eq!(mask.dims(), &[4, 4]);
    }
    
    #[test]
    fn test_multi_head_attention() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let mha = MultiHeadAttention::new(512, 8, 128, 0.1, vb).unwrap();
        assert_eq!(mha.heads.len(), 8);
    }
    
    #[test]
    fn test_six_head_attention() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        // Test with 6 heads as specifically requested
        let n_embd = 72;  // Divisible by 6
        let n_head = 6;
        let mha = MultiHeadAttention::new(n_embd, n_head, 128, 0.1, vb).unwrap();
        
        assert_eq!(mha.heads.len(), 6);
        
        // Test forward pass with shape preservation
        let batch_size = 2;
        let seq_len = 10;
        let input = Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len, n_embd), &device)
            .unwrap().to_dtype(DType::F32).unwrap();
        
        let output = mha.forward(&input, false).unwrap();
        
        // Verify input and output shapes match
        assert_eq!(input.shape(), output.shape());
        assert_eq!(output.dims3().unwrap(), (batch_size, seq_len, n_embd));
    }
}