use candle_core::{DType, Device, Result, Tensor};
use embedding::token::{TokenEmbedding, TokenEmbeddingConfig};

fn make_ids(data: &[i64], shape: (usize, usize)) -> Result<Tensor> {
    Tensor::from_slice(data, shape, &Device::Cpu)
}

#[test]
fn forward_shape_and_dtype_match_config() -> Result<()> {
    let config = TokenEmbeddingConfig {
        vocab_size: 8,
        hidden_dim: 4,
        dtype: DType::F16,
        device: Device::Cpu,
    };
    let embedding = TokenEmbedding::new(config.clone())?;
    let token_ids = make_ids(&[0, 1, 2, 3], (2, 2))?;

    let output = embedding.forward(&token_ids)?;

    assert_eq!(output.dims(), &[2, 2, config.hidden_dim]);
    assert_eq!(output.dtype(), config.dtype);
    Ok(())
}

#[test]
fn forward_rejects_out_of_range_ids() -> Result<()> {
    let config = TokenEmbeddingConfig {
        vocab_size: 4,
        hidden_dim: 3,
        dtype: DType::F32,
        device: Device::Cpu,
    };
    let embedding = TokenEmbedding::new(config)?;
    let token_ids = make_ids(&[0, 4], (1, 2))?;

    let err = embedding.forward(&token_ids).unwrap_err();
    assert!(err.to_string().contains("token id 4 exceeds vocab size"));
    Ok(())
}

#[test]
fn tied_linear_head_produces_finite_logits() -> Result<()> {
    let dtypes = [DType::F32, DType::F16, DType::BF16];
    for &dtype in &dtypes {
        let config = TokenEmbeddingConfig {
            vocab_size: 6,
            hidden_dim: 5,
            dtype,
            device: Device::Cpu,
        };
        let embedding = TokenEmbedding::new(config.clone())?;
        let token_ids = make_ids(&[0, 1, 2, 3], (2, 2))?;

        let hidden = embedding.forward(&token_ids)?;
        let logits = embedding.linear_out(&hidden)?;

        assert_eq!(logits.dims(), &[2, 2, config.vocab_size]);
        assert_eq!(logits.dtype(), config.dtype);

        let logits_f32 = logits.to_dtype(DType::F32)?;
        let values = logits_f32.flatten_all()?.to_vec1::<f32>()?;
        assert!(values.iter().all(|v| v.is_finite()));
    }
    Ok(())
}
