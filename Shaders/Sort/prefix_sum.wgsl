// Impl taken from: https://github.com/kishimisu/WebGPU-Radix-Sort/blob/main/src/shaders/prefix_sum.js
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// Almost counting sort subroutine - prefix sum on bin counts for starting position for each bin
// Get starting position of each digit in the array

struct Particle {
    position: vec2f,
    velocity: vec2f,
    color: vec4f
};

// output should be prefix sum/accumulation of counts of each bin (based on hash % num bins)
@group(0) @binding(0) var<storage, read_write> output: array<u32>; // output - should be prefix by starting pos
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;
@group(0) @binding(2) var<storage, read_write> particles: Particle;

override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override THREADS_PER_WORKGROUP: u32;
override ITEMS_PER_WORKGROUP: u32;
override ELEMENT_COUNT: u32;

var<workgroup> temp: array<u32, ITEMS_PER_WORKGROUP*2>;

struct Particle {
    position: vec2f,
    velocity: vec2f,
    color: vec4f,
};

/* utils */
fn hashCoords(particle: Particle) -> u32 {
    // 10 minute physics hash, can replace with Z-order
    let xi = i32(floor(pos.x / params.cellSize));
    let yi = i32(floor(pos.y / params.cellSize));
    let h = (xi * 92837111) ^ (yi * 689287499);
    return u32(abs(h)) % params.numBins;
}
/* end utils */

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn reduce_downsweep(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID

    let ELM_TID = TID * 2; // Element pair local ID
    let ELM_GID = GID * 2; // Element pair global ID

    // Load input to shared memory
    temp[ELM_TID]     = select(hashCoords(particles[ELM_GID]), 0, ELM_GID >= ELEMENT_COUNT);
    temp[ELM_TID + 1] = select(hashCoords(particles[ELM_GID + 1]), 0, ELM_GID + 1 >= ELEMENT_COUNT);

    var offset: u32 = 1;

    // Up-sweep (reduce) phase
    // Propagate sums from leaves to root of balanced binary tree (similar to segment tree)
    for (var d: u32 = ITEMS_PER_WORKGROUP >> 1; d > 0; d >>= 1) {
        workgroupBarrier();

        if (TID < d) {
            var ai: u32 = offset * (ELM_TID + 1) - 1;
            var bi: u32 = offset * (ELM_TID + 2) - 1;
            temp[bi] += temp[ai];
        }

        offset *= 2;
    }

    // Save workgroup sum and clear last element
    if (TID == 0) {
        let last_offset = ITEMS_PER_WORKGROUP - 1;

        blockSums[WORKGROUP_ID] = temp[last_offset];
        temp[last_offset] = 0;
    }

    // Down-sweep phase
    // Propogate values downwards from root to leaves as prefix sum
    // Each level multiplies read-access by 2, offset logic mitigates bank conflicts
    for (var d: u32 = 1; d < ITEMS_PER_WORKGROUP; d *= 2) {
        offset >>= 1;
        workgroupBarrier();

        if (TID < d) {
            var ai: u32 = offset * (ELM_TID + 1) - 1;
            var bi: u32 = offset * (ELM_TID + 2) - 1;

            let t: u32 = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    workgroupBarrier();

    // Copy result from shared memory to global memory
    if (ELM_GID >= ELEMENT_COUNT) {
        return;
    }
    output[ELM_GID] = temp[ELM_TID];

    if (ELM_GID + 1 >= ELEMENT_COUNT) {
        return;
    }
    output[ELM_GID + 1] = temp[ELM_TID + 1];
}

/*
 * For arbitrarily large buffers, divide into several blocks
 * Then add block sum to blocks on RHS
*/
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn add_block_sums(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID

    let ELM_ID = GID * 2;

    if (ELM_ID >= ELEMENT_COUNT) {
        return;
    }

    let blockSum = blockSums[WORKGROUP_ID];

    output[ELM_ID] += blockSum;

    if (ELM_ID + 1 >= ELEMENT_COUNT) {
        return;
    }

    output[ELM_ID + 1] += blockSum;
}