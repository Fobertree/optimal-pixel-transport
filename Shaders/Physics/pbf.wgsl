/*
Resources:
- https://mmacklin.com/pbf_sig_preprint.pdf (too lazy to find final print)
- https://people.engr.tamu.edu/sueda/courses/CSCE450/2022F/projects/Brandon_Nguyen/index.html#:~:text=Description%C2%A7,one%20physics%20step%20per%20frame.
*/

const TILE_SIZE : u32 = 256u; // workgroup length
const NEIGHBORHOOD_SIZE : f32 = 0.05; // hyperparam for calc particle neighborhood via counting sort

struct Particle {
    position: vec2f,
    color: vec4f
};

struct Params {
    size : u32
}

// group 0 - params
@group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(1) var<storage, read> params : Params;

// group 1 - solver (assignments, costs)
@group(1) @binding(0) var<storage, read> assignments : array<i32>;

// Pre-computed on CPU, since very in-expensive and I like the templating
@group(1) @binding(1) var<storage, read> cost_matrix : array<i32>; // TODO: drop cost_matrix from here - redundant

fn cost(i : u32, j: u32) -> i32 {
    return cost_matrix[i * size + j];
}

// group 2 - target particle
@group(2) @binding(0) var<storage, read> target_particles : array<Particle>;

fn get_force_from_assignment() {
    // only call on neighborhood to follow PBF's O(N)
}

fn get_neighborhood() {
    // O(N) counting sort w/ discretized bins + prefix sum
    // Won't parallelize for now - just do naive

}

@compute @workgroup_size(TILE_SIZE)
fn computeMain(
    @builtin(global_invocation_id) global_invocation_id : vec3<u32>
    ) {
    let size = params.size;
    let particle_idx = global_id.x;

    if (particle_idx >= size) {
        return;
    }
    // TODO: impl

    // Skip step 1 - apply external forces

    // Find neighboring particles

    // Calculate lambda_i

    // calc \Inc P_i + collision

    // Position update

    // Euler integration velocity update
}