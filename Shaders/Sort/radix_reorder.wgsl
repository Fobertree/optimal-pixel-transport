// Impl taken from: https://github.com/kishimisu/WebGPU-Radix-Sort/blob/main/src/shaders/radix_sort_reorder.js
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

@group(0) @binding(0) var<storage, read> inputKeys: array<u32>;        // bin hashes
@group(0) @binding(2) var<storage, read> local_prefix_sum: array<u32>;
@group(0) @binding(3) var<storage, read> prefix_block_sum: array<u32>;
// input/output values
@group(0) @binding(4) var<storage, read> inputParticles: array<Particle>;
@group(0) @binding(5) var<storage, read_write> outputParticles: array<Particle>;

override WORKGROUP_COUNT: u32;
override THREADS_PER_WORKGROUP: u32;
override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override CURRENT_BIT: u32;
override ELEMENT_COUNT: u32;

// 3: REORDER
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn radix_sort_reorder(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID

    if (GID >= ELEMENT_COUNT) {
        return;
    }

    let k = inputKeys[GID];
    let v = inputParticles[GID];

    let local_prefix = local_prefix_sum[GID];

    // Calculate new position
    let extract_bits = (k >> CURRENT_BIT) & 0x3;
    let pid = extract_bits * WORKGROUP_COUNT + WORKGROUP_ID;
    // true prefix sum = local_prefix + prefix block sum
    let sorted_position = prefix_block_sum[pid] + local_prefix;

    outputParticles[sorted_position] = v;
}