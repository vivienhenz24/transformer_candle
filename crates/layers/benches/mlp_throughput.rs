use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use layers::{
    activations::ActivationKind,
    dtypes::PrecisionPolicy,
    linear::LinearInit,
    mlp::{FeedForward, FeedForwardConfig, FeedForwardLayer},
};

fn bench_mlp(c: &mut Criterion) {
    let device = Device::Cpu;
    let batch = 4usize;
    let seq = 16usize;
    let hidden = 4096usize;
    let dtypes = &[DType::F16, DType::BF16, DType::F32];
    let variants = [
        ("gelu", 4.0f32, ActivationKind::Gelu, false),
        ("swiglu", 4.0f32, ActivationKind::SwiGlu, true),
        ("gelu", 8.0f32, ActivationKind::Gelu, false),
    ];

    for &dtype in dtypes {
        let mut group = c.benchmark_group(format!("mlp/{dtype:?}"));
        for &(label, ratio, activation, gated) in &variants {
            let mut config = FeedForwardConfig::with_expansion_ratio(hidden, ratio, activation);
            config.gated = gated;
            let mlp = FeedForward::with_init(
                config,
                activation,
                &LinearInit::XavierNormal,
                &LinearInit::scaled(
                    LinearInit::KaimingUniform {
                        negative_slope: 0.0,
                    },
                    0.5,
                ),
                &device,
                dtype,
            )
            .expect("mlp init");
            let input = Tensor::randn(0f32, 1.0, (batch, seq, hidden), &device)
                .expect("input")
                .to_dtype(dtype)
                .expect("cast input");
            let policy = PrecisionPolicy::from_parameter_dtype(dtype);
            let elements = (batch * seq * hidden) as u64;
            group.throughput(Throughput::Elements(elements));
            group.bench_with_input(
                BenchmarkId::new(label, ratio),
                &(mlp.clone(), input.clone(), policy.clone()),
                |b, (mlp, input, policy)| {
                    b.iter(|| {
                        let out = mlp.forward(black_box(input), policy).expect("forward");
                        black_box(out);
                    });
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches, bench_mlp);
criterion_main!(benches);
