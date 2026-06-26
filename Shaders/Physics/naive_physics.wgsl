
// Directly assign particles based on assignment

const TILE_SIZE : u32 = 256u; // workgroup length

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

@compute @workgroup_size(TILE_SIZE)
fn computeMain(
    @builtin(global_invocation_id) global_invocation_id : vec3<u32>
    ) {
    let size = params.size;
    let particle_idx = global_id.x;

    if (particle_idx >= size) {
        return;
    }

    // cast to u32 from i32 (shouldn't matter but better practice here)
    let target_idx = u32(assignments[particle_idx]);
    let assigned_particle = target_particles[target_idx];

    particles[particle_idx].position = assigned_particle.position;
}