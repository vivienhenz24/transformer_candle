use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use layers::{
    dtypes::PrecisionPolicy,
    linear::{Linear, LinearConfig, LinearInit, LinearLayer},
};

fn bench_linear(c: &mut Criterion) {
    let device = Device::Cpu;
    let batch = 4usize;
    let seq = 16usize;
    let shapes = &[(1024usize, 1024usize), (2048, 4096), (4096, 4096)];
    let dtypes = &[DType::F16, DType::BF16, DType::F32];

    for &dtype in dtypes {
        let mut group = c.benchmark_group(format!("linear/{dtype:?}"));
        for &(input_dim, output_dim) in shapes {
            let mut cfg = LinearConfig::new(input_dim, output_dim);
            cfg.bias = false;
            let linear = Linear::with_init(cfg, &LinearInit::XavierNormal, &device, dtype)
                .expect("linear init");
            let input = Tensor::randn(0f32, 1.0, (batch, seq, input_dim), &device)
                .expect("input")
                .to_dtype(dtype)
                .expect("cast input");
            let policy = PrecisionPolicy::from_parameter_dtype(dtype);
            let elements = (batch * seq * input_dim * output_dim) as u64;
            group.throughput(Throughput::Elements(elements));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}x{}", input_dim, output_dim)),
                &(linear.clone(), input.clone(), policy.clone()),
                |b, (linear, input, policy)| {
                    b.iter(|| {
                        let out = linear.forward(black_box(input), policy).expect("forward");
                        black_box(out);
                    });
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches, bench_linear);
criterion_main!(benches);
