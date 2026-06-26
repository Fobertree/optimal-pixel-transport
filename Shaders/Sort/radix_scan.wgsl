// Exclusive prefix scan over per-digit block sums produced by radix_sort.
// Converts counts in block_sums to exclusive offsets in-place.

override WORKGROUP_COUNT: u32;

@group(1) @binding(1) var<storage, read_write> block_sums: array<u32>;

@compute @workgroup_size(4, 1, 1)
fn radixScanBlockSums(@builtin(global_invocation_id) gid: vec3<u32>) {
    let digit = gid.x;
    if (digit >= 4u) {
        return;
    }

    let base = digit * WORKGROUP_COUNT;
    var running: u32 = 0u;
    for (var w: u32 = 0u; w < WORKGROUP_COUNT; w++) {
        let count = block_sums[base + w];
        block_sums[base + w] = running;
        running += count;
    }
}