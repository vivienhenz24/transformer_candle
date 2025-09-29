use super::{CacheConfig, KeyValueCache, PagedKeyValueCache};
use candle_core::{DType, Device, Tensor};
use static_assertions::{assert_impl_all, assert_not_impl_any};

fn allclose(a: &Tensor, b: &Tensor, tol: f32) {
    let diff = a
        .to_dtype(DType::F32)
        .unwrap()
        .sub(&b.to_dtype(DType::F32).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let max = diff.into_iter().fold(0.0_f32, |acc, v| acc.max(v));
    assert!(max <= tol, "max diff {max} exceeded tolerance {tol}");
}

fn build_config(device: &Device, page_size: usize) -> CacheConfig {
    CacheConfig::new(2, 2, 3, 4, page_size, DType::F32, device.clone()).unwrap()
}

fn random_prefill(device: &Device, seq_len: usize) -> (Tensor, Tensor) {
    let shape = (2, 2, 3, seq_len, 4);
    let keys = Tensor::rand(0.0f32, 1.0, shape, device).unwrap();
    let values = Tensor::rand(0.0f32, 1.0, shape, device).unwrap();
    (keys, values)
}

#[test]
fn prefill_then_decode_transition() {
    let device = Device::Cpu;
    let mut cache = PagedKeyValueCache::new(build_config(&device, 3));

    let (mut expected_keys, mut expected_values) = random_prefill(&device, 5);
    cache
        .prefill(&expected_keys, &expected_values)
        .expect("prefill succeeds");

    for layer in 0..2 {
        let positions: Vec<usize> = (0..5).collect();
        let (keys, values) = cache.gather(layer, &positions).unwrap();
        let layer_keys = expected_keys
            .narrow(1, layer, 1)
            .unwrap()
            .squeeze(1)
            .unwrap();
        let layer_values = expected_values
            .narrow(1, layer, 1)
            .unwrap()
            .squeeze(1)
            .unwrap();
        allclose(&keys, &layer_keys, 1e-5);
        allclose(&values, &layer_values, 1e-5);
    }

    let mut stats = cache.stats();
    assert_eq!(stats.tokens, 5 * 2);
    assert_eq!(stats.hits, 10);
    assert_eq!(stats.misses, 0);

    for _ in 0..3 {
        let step_keys = Tensor::rand(0.0f32, 1.0, (2, 2, 3, 4), &device).unwrap();
        let step_values = Tensor::rand(0.0f32, 1.0, (2, 2, 3, 4), &device).unwrap();
        cache
            .append_decode_step(&step_keys, &step_values)
            .expect("append succeeds");

        let step_keys_expanded = step_keys.unsqueeze(3).unwrap();
        expected_keys = Tensor::cat(&[&expected_keys, &step_keys_expanded], 3).unwrap();

        let step_values_expanded = step_values.unsqueeze(3).unwrap();
        expected_values = Tensor::cat(&[&expected_values, &step_values_expanded], 3).unwrap();
    }

    for layer in 0..2 {
        let positions: Vec<usize> = (0..8).collect();
        let (keys, values) = cache.gather(layer, &positions).unwrap();
        let layer_keys = expected_keys
            .narrow(1, layer, 1)
            .unwrap()
            .squeeze(1)
            .unwrap();
        let layer_values = expected_values
            .narrow(1, layer, 1)
            .unwrap()
            .squeeze(1)
            .unwrap();
        allclose(&keys, &layer_keys, 1e-5);
        allclose(&values, &layer_values, 1e-5);
    }

    stats = cache.stats();
    assert_eq!(stats.tokens, 8 * 2);
    assert_eq!(stats.hits, 26);
    assert_eq!(stats.misses, 0);
}

#[test]
fn paged_matches_contiguous() {
    let device = Device::Cpu;
    let page_sizes = [1, 2, 4, 8];
    let (keys, values) = random_prefill(&device, 6);

    for &page in &page_sizes {
        let mut cache = PagedKeyValueCache::new(build_config(&device, page));
        cache.prefill(&keys, &values).unwrap();
        let mut contiguous_cache = PagedKeyValueCache::new(build_config(&device, 64));
        contiguous_cache.prefill(&keys, &values).unwrap();

        for layer in 0..2 {
            let positions: Vec<usize> = (0..6).collect();
            let (paged_k, paged_v) = cache.gather(layer, &positions).unwrap();
            let (contig_k, contig_v) = contiguous_cache.gather(layer, &positions).unwrap();
            allclose(&paged_k, &contig_k, 1e-5);
            allclose(&paged_v, &contig_v, 1e-5);
        }

        let step_keys = Tensor::rand(0.0f32, 1.0, (2, 2, 3, 4), &device).unwrap();
        let step_values = Tensor::rand(0.0f32, 1.0, (2, 2, 3, 4), &device).unwrap();
        cache.append_decode_step(&step_keys, &step_values).unwrap();
        contiguous_cache
            .append_decode_step(&step_keys, &step_values)
            .unwrap();

        for layer in 0..2 {
            let positions: Vec<usize> = (0..7).collect();
            let (paged_k, paged_v) = cache.gather(layer, &positions).unwrap();
            let (contig_k, contig_v) = contiguous_cache.gather(layer, &positions).unwrap();
            allclose(&paged_k, &contig_k, 1e-5);
            allclose(&paged_v, &contig_v, 1e-5);
        }
    }
}

#[test]
fn stats_capture_misses_and_segment_churn() {
    let device = Device::Cpu;
    let mut cache = PagedKeyValueCache::new(build_config(&device, 2));
    let (keys, values) = random_prefill(&device, 3);
    cache.prefill(&keys, &values).unwrap();

    let stats = cache.stats();
    assert_eq!(stats.segment_churn, 2 * 2); // two layers, ceil(3/2)=2 pages each

    assert!(cache.gather(0, &[0, 1]).is_ok());
    assert!(cache.gather(1, &[0, 2]).is_ok());
    assert!(cache.gather(0, &[5]).is_err());

    let stats = cache.stats();
    assert_eq!(stats.hits, 4);
    assert_eq!(stats.misses, 1);
}

#[test]
fn concurrency_traits_documented() {
    assert_impl_all!(PagedKeyValueCache: Send);
    assert_not_impl_any!(PagedKeyValueCache: Sync);
}
