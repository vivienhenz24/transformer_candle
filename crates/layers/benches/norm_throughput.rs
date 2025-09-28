use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use layers::{
    dtypes::PrecisionPolicy,
    norm::{LayerNorm, NormConfig, NormKind, NormalizationLayer, RmsNorm},
};

fn bench_norms(c: &mut Criterion) {
    let device = Device::Cpu;
    let batch = 8usize;
    let seq = 32usize;
    let hidden_sizes = &[1024usize, 2048, 4096];
    let dtypes = &[DType::F16, DType::BF16, DType::F32];

    for &dtype in dtypes {
        let mut group = c.benchmark_group(format!("norm/{dtype:?}"));
        for &hidden in hidden_sizes {
            let input = Tensor::randn(0f32, 1.0, (batch, seq, hidden), &device)
                .expect("input")
                .to_dtype(dtype)
                .expect("cast input");
            let policy = PrecisionPolicy::from_parameter_dtype(dtype);

            let mut layer_cfg = NormConfig::new(hidden, NormKind::LayerNorm);
            layer_cfg.elementwise_affine = true;
            let weight = Tensor::ones((hidden,), dtype, &device).expect("weight");
            let bias = Tensor::zeros((hidden,), dtype, &device).expect("bias");
            let layer_norm = LayerNorm::new(weight.clone(), bias, layer_cfg).expect("layer norm");

            let mut rms_cfg = NormConfig::new(hidden, NormKind::RmsNorm);
            rms_cfg.elementwise_affine = true;
            let rms_norm = RmsNorm::new(weight, rms_cfg).expect("rms norm");

            let elements = (batch * seq * hidden) as u64;
            group.throughput(Throughput::Elements(elements));

            group.bench_with_input(
                BenchmarkId::new("layer", hidden),
                &(layer_norm.clone(), input.clone(), policy.clone()),
                |b, (norm, input, policy)| {
                    b.iter(|| {
                        let out = norm.forward(black_box(input), policy).expect("forward");
                        black_box(out);
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("rms", hidden),
                &(rms_norm.clone(), input.clone(), policy.clone()),
                |b, (norm, input, policy)| {
                    b.iter(|| {
                        let out = norm.forward(black_box(input), policy).expect("forward");
                        black_box(out);
                    });
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches, bench_norms);
criterion_main!(benches);
