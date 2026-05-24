// Impl taken from: https://github.com/kishimisu/WebGPU-Radix-Sort/blob/main/src/shaders/radix_sort_reorder.js
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

@group(0) @binding(0) var<storage, read_write> inputParticles: array<Particle>;
@group(0) @binding(1) var<storage, read> params : Params;

@group(0) @binding(0) var<storage, read> local_prefix_sum: array<u32>;
@group(0) @binding(1) var<storage, read> prefix_block_sum: array<u32>;
// this is a tmp buffer in my case. I do an inefficient copy to the input buffer for ease of binding groups
// will optimize this out later
// TODO: rm outputParticles buffer from radix BG, then swap particle buffer on global param bg every iteration
@group(0) @binding(2) var<storage, read_write> outputParticles: array<Particle>;
@group(0) @binding(3) var<storage, read_write> binStart: array<u32>;
@group(0) @binding(4) var<storage, read_write> binEnd: array<u32>;
@group(0) @binding(5) var<storage, read_write> outputHashes: array<u32>;

struct Params {
    // Below: unused (just for BG consistency)
    size : u32,
    rho0 : f32,
    H: f32,         // kernel smoothing radius
    dt: f32,        // TODO: instead of hardcoding this, maybe expose this to CFL conditions
    solverIterations: u32,
    cellSize: f32,
    numBins: u32,
}

override WORKGROUP_COUNT: u32;
override THREADS_PER_WORKGROUP: u32;
override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override CURRENT_BIT: u32;
override ELEMENT_COUNT: u32;

/* utils */
fn hashCoords(pos: vec2f) -> u32 {
    // 10 minute physics hash, can replace with Z-order
    let xi = i32(floor(pos.x / params.cellSize));
    let yi = i32(floor(pos.y / params.cellSize));
    // Believe you can arbitrarily xor these numbers * any set of arbitrarily large prime (or coprime) numbers
    // Wonder if there's any analytical/non-empirical way to validate effectiveness against hash-collisions
    let h = (xi * 92837111) ^ (yi * 689287499);
    return u32(abs(h)) % params.numBins;
}
/* end utils */

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

    if (GID < ELEMENT_COUNT) {
        let k = hashCoords(inputParticles[GID]);
        let v = inputParticles[GID];

        let local_prefix = local_prefix_sum[GID];

        // Calculate new position
        let extract_bits = (k >> CURRENT_BIT) & 0x3;
        let pid = extract_bits * WORKGROUP_COUNT + WORKGROUP_ID;
        // true prefix sum = local_prefix + prefix block sum
        let sorted_position = prefix_block_sum[pid] + local_prefix;

        // TODO: modify outputParticles to workgroup tile cache if possible
        // TODO: see if webgpu can support buffer swap logic like opengl
        outputParticles[sorted_position] = v;
    }

    if (GID < ELEMENT_COUNT) {
        // NAIVE SLOPPY COPY CODE - will optimize later
        inputParticles[GID] = outputParticles[GID];
        outputHashes[GID] = k;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn radix_sort_reorder(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID

    if (GID < ELEMENT_COUNT) {
        // TODO: see if can swap buffers
        // I think it's diff from opengl where best approach is to rebind between buffers A and B every iteration
        inputParticles[GID] = outputParticles[GID];

        let cur = hashes[GID];
        let prev = hashes[GID-1];
        let next = hashes[GID+1];

        // thread-safe despite duplicate hashes - only one index where writes can occur in both cases
        if (GID == 0 || cur != prev) {
            binStart[cur] = GID;
        }

        if (GID == ELEMENT_COUNT-1 || cur != next) {
            binEnd[cur] = GID+1;
        }
    }
}