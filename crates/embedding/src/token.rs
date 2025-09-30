//! Token embedding layer and optional tied readout head.

use candle_core::{bail, DType, Device, Error, Result, Tensor, Var};
use layers::PrecisionPolicy;

/// Configuration for building a token embedding table.
#[derive(Debug, Clone)]
pub struct TokenEmbeddingConfig {
    /// Size of the vocabulary (number of distinct tokens).
    pub vocab_size: usize,
    /// Dimensionality of each embedding vector.
    pub hidden_dim: usize,
    /// Storage dtype used for the underlying parameters and outputs.
    pub dtype: DType,
    /// Device hosting the parameters.
    pub device: Device,
}

impl TokenEmbeddingConfig {
    /// Precision policy mirroring the layers crate behaviour for mixed precision.
    fn policy(&self) -> PrecisionPolicy {
        PrecisionPolicy::from_parameter_dtype(self.dtype)
    }
}

/// Learnable token embedding table with optional tied projection head.
#[derive(Debug, Clone)]
pub struct TokenEmbedding {
    config: TokenEmbeddingConfig,
    weight: Var,
    policy: PrecisionPolicy,
}

impl TokenEmbedding {
    /// Builds a new token embedding table and samples the parameters from `N(0, 1)`.
    pub fn new(config: TokenEmbeddingConfig) -> Result<Self> {
        if config.vocab_size == 0 {
            bail!("token embedding requires vocab_size > 0");
        }
        if config.hidden_dim == 0 {
            bail!("token embedding requires hidden_dim > 0");
        }

        let policy = config.policy();
        let shape = (config.vocab_size, config.hidden_dim);
        let initial = Var::randn(0f32, 1f32, shape, &config.device)?;
        let weight = if initial.dtype() == config.dtype {
            initial
        } else {
            let cast = initial.to_dtype(config.dtype)?;
            Var::from_tensor(&cast)?
        };

        Ok(Self {
            config,
            weight,
            policy,
        })
    }

    /// Returns the embedding configuration.
    pub fn config(&self) -> &TokenEmbeddingConfig {
        &self.config
    }

    /// Returns a clone of the underlying weight tensor.
    pub fn weight(&self) -> Tensor {
        self.weight.as_tensor().clone()
    }

    /// Looks up embeddings for the provided token ids.
    ///
    /// Inputs must be shaped `(batch, seq)` with an integer dtype. Outputs follow the
    /// `(batch, seq, hidden)` layout using the configured storage dtype.
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.validate_token_ids(token_ids)?;
        let dims = token_ids.dims();

        let ids = token_ids.to_dtype(DType::I64)?;
        let flat = ids.flatten_all()?;
        self.ensure_id_range(&flat)?;

        let weight = self.weight.as_tensor();
        let gathered = weight.index_select(&flat, 0)?;
        let mut output_dims = dims.to_vec();
        output_dims.push(self.config.hidden_dim);
        gathered.reshape(output_dims)
    }

    /// Returns the trainable parameters for this embedding with an optional scope prefix.
    pub fn named_parameters(&self, scope: &str) -> Vec<(String, Var)> {
        let prefix = if scope.is_empty() {
            "embedding".to_string()
        } else {
            scope.to_string()
        };
        vec![(format!("{}.weight", prefix), self.weight.clone())]
    }

    /// Applies a tied linear projection using the transpose of the embedding weight.
    pub fn linear_out(&self, hidden: &Tensor) -> Result<Tensor> {
        let dims = hidden.dims();
        let (batch, seq, hidden_dim) = match dims {
            [batch, seq, hidden_dim] => (*batch, *seq, *hidden_dim),
            _ => {
                return Err(Error::Msg(
                    "linear_out expects input shaped [batch, seq, hidden]".into(),
                ))
            }
        };

        if hidden_dim != self.config.hidden_dim {
            return Err(Error::Msg(format!(
                "linear_out expected hidden dim {} but received {}",
                self.config.hidden_dim, hidden_dim
            )));
        }

        let policy = &self.policy;
        let input = policy.cast_for_matmul(hidden)?;
        let weight = policy.cast_for_matmul(self.weight.as_tensor())?;
        let weight_t = weight.t()?;

        let flat = input.reshape((batch * seq, hidden_dim))?;
        let logits = flat.matmul(&weight_t)?;
        let logits = logits.reshape((batch, seq, self.config.vocab_size))?;
        policy.cast_to_storage(&logits)
    }

    fn validate_token_ids(&self, token_ids: &Tensor) -> Result<()> {
        let dims = token_ids.dims();
        match dims {
            [batch, seq] => {
                if *batch == 0 || *seq == 0 {
                    return Err(Error::Msg(
                        "token_ids must have non-zero batch and seq dimensions".into(),
                    ));
                }
            }
            _ => return Err(Error::Msg("token_ids must be shaped [batch, seq]".into())),
        }

        if !is_integer_dtype(token_ids.dtype()) {
            Err(Error::Msg(format!(
                "token_ids expected integer dtype but received {:?}",
                token_ids.dtype()
            )))
        } else {
            Ok(())
        }
    }

    fn ensure_id_range(&self, flat_ids: &Tensor) -> Result<()> {
        if flat_ids.elem_count() == 0 {
            return Ok(());
        }

        // Debug: Print tensor info
        println!("DEBUG: Token tensor shape: {:?}, dtype: {:?}", flat_ids.shape(), flat_ids.dtype());
        
        let min_id = flat_ids.min_all()?.to_scalar::<i64>()?;
        if min_id < 0 {
            // Debug: Print first few values and find negative ones
            if flat_ids.elem_count() > 0 {
                let first_10 = flat_ids.flatten_all()?.narrow(0, 0, (flat_ids.elem_count().min(10) as usize))?;
                println!("DEBUG: First 10 token values: {:?}", first_10.to_vec1::<i64>());
                
                // Find and print negative values
                let all_values = flat_ids.flatten_all()?.to_vec1::<i64>()?;
                println!("DEBUG: Total elements: {}", all_values.len());
                let mut negative_count = 0;
                for (i, &val) in all_values.iter().enumerate() {
                    if val < 0 {
                        println!("DEBUG: Negative token at index {}: {}", i, val);
                        negative_count += 1;
                        if negative_count >= 5 { // Limit output
                            break;
                        }
                    }
                }
            }
            return Err(Error::Msg(format!(
                "encountered negative token id {} (minimum)",
                min_id
            )));
        }

        let max_id = flat_ids.max_all()?.to_scalar::<i64>()?;
        let vocab = self.config.vocab_size as i64;
        if max_id >= vocab {
            return Err(Error::Msg(format!(
                "token id {} exceeds vocab size {}",
                max_id, vocab
            )));
        }
        Ok(())
    }
}

fn is_integer_dtype(dtype: DType) -> bool {
    dtype.is_int()
}
