
use embedding::positional::rope::RopeConfig;

fn main() {
    println!("Transformer started");
    pretraining_data::init();

    let _rope_config = RopeConfig::default();
    println!(
        "RoPE module placeholder ready (head_dim = {}, theta = {})",
        _rope_config.head_dim, _rope_config.rope_theta
    );

    run_rope_demo();
}

/// Minimal RoPE demo executed from the binary.
fn run_rope_demo() {
    println!("--- RoPE demo (stub) ---");

    // Plan:
    // 1. Allocate synthetic query/key tensors shaped [batch=1, heads=2, seq_len=8, head_dim=??]
    //    filled with deterministic data to make norm checks straightforward.
    // 2. Build a `RopeConfig` specifying the target `head_dim` and optional `rotate_dim`.
    // 3. Invoke `get_sin_cos` with `max_seq_len >= 8` on CPU to warm the cache and capture
    //    whether the allocation path was a miss or hit (expose a counter/log for cache diagnostics).
    // 4. Apply `apply_rope_to_qk` with `pos_start = 0` to obtain rotated Q/K tensors.
    // 5. Compute and print the L2 norm delta for the first `rotate_dim` features and verify
    //    the trailing `(head_dim - rotate_dim)` slice remains unchanged (exact comparison).
    // 6. Re-run steps 3-4 to confirm sin/cos tensors are reused (log "cache hit" metric).

    println!("RoPE demo is not implemented yet; see comments for the execution plan.");
}
