// Impl taken from: https://github.com/kishimisu/WebGPU-Radix-Sort/blob/main/src/shaders/prefix_sum.js
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// https://www.sci.utah.edu/~csilva/papers/cgf.pdf
// Base-4 radix instead of base-2 for efficiency, meaning 0x3 mask

struct Particle {
    position: vec2f,
    velocity: vec2f,
    color: vec4f
};

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

// Global BG
@group(0) @binding(0) var<storage, read_write> inputParticles: array<Particle>;
@group(0) @binding(1) var<uniform> params : Params;

// Radix BG
@group(1) @binding(0) var<storage, read_write> local_prefix_sums: array<u32>; // sort hashes
@group(1) @binding(1) var<storage, read_write> block_sums: array<u32>;
@group(1) @binding(2) var<storage, read_write> binStart: array<u32>;
@group(1) @binding(3) var<storage, read_write> binEnd: array<u32>;
// outputs
@group(1) @binding(4) var<storage, read_write> outputHashes: array<u32>;
// argsort because it's a PITA to manage a ton of swap buffers and all the assignments might lead to even worse performance than lost cache locality
@group(1) @binding(5) var<storage, read_write> sortIndices: array<u32>;

override WORKGROUP_COUNT: u32;
override THREADS_PER_WORKGROUP: u32;
override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override CURRENT_BIT: u32;

var<workgroup> s_prefix_sum: array<u32, 2 * (THREADS_PER_WORKGROUP + 1)>;

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

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn radix_sort(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID
    let size = params.size;

    // Extract 2 bits from the input
    let elm = select(hashCoords(inputParticles[GID].position), 0, GID >= size);
    let extract_bits: u32 = (elm >> CURRENT_BIT) & 0x3;

    var bit_prefix_sums = array<u32, 4>(0, 0, 0, 0);

    // If the workgroup is inactive, prevent block_sums buffer update
    var LAST_THREAD: u32 = 0xffffffff;

    if (WORKGROUP_ID < WORKGROUP_COUNT) {
        // Otherwise store the index of the last active thread in the workgroup
        LAST_THREAD = min(THREADS_PER_WORKGROUP, size - WID) - 1;
    }

    // Initialize parameters for double-buffering
    let TPW = THREADS_PER_WORKGROUP + 1;
    var swapOffset: u32 = 0;
    var inOffset:  u32 = TID;
    var outOffset: u32 = TID + TPW;

    // 4-way prefix sum
    for (var b: u32 = 0; b < 4; b++) {
        // Initialize local prefix with bitmask
        let bitmask = select(0u, 1u, extract_bits == b);
        s_prefix_sum[inOffset + 1] = bitmask;
        workgroupBarrier();

        var prefix_sum: u32 = 0;

        // Prefix sum
        for (var offset: u32 = 1; offset < THREADS_PER_WORKGROUP; offset *= 2) {
            if (TID >= offset) {
                prefix_sum = s_prefix_sum[inOffset] + s_prefix_sum[inOffset - offset];
            } else {
                prefix_sum = s_prefix_sum[inOffset];
            }

            s_prefix_sum[outOffset] = prefix_sum;

            // Swap buffers
            outOffset = inOffset;
            swapOffset = TPW - swapOffset;
            inOffset = TID + swapOffset;

            workgroupBarrier();
        }

        // Store prefix sum for current bit
        bit_prefix_sums[b] = prefix_sum;

        if (TID == LAST_THREAD) {
            // Store block sum to global memory
            let total_sum: u32 = prefix_sum + bitmask;
            block_sums[b * WORKGROUP_COUNT + WORKGROUP_ID] = total_sum;
        }

        // Swap buffers
        outOffset = inOffset;
        swapOffset = TPW - swapOffset;
        inOffset = TID + swapOffset;
    }

    if (GID < size) {
        // Store local prefix sum to global memory
        local_prefix_sums[GID] = bit_prefix_sums[extract_bits];
    }
}