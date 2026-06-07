// naive particle shader (adapted to BG)
// abstract-float
const particle_size = 0.088;

struct Particle {
    position: vec2f,
    velocity: vec2f,
    targetPos: vec2f,
    color: vec4f
}

struct Params {
    size : u32,
    // Below: unused (just for BG consistency)
    rho0 : f32,
    H: f32,         // kernel smoothing radius
    dt: f32,        // TODO: instead of hardcoding this, maybe expose this to CFL conditions
    solverIterations: u32,
    cellSize: f32,
    numBins: u32,
}

struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
    @builtin(instance_index) instance_index : u32
};

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) color: vec4f
    // other inter-stage variables alongside @location
};

@group(0) @binding(0) var<storage, read> particles : array<Particle>;
@group(0) @binding(1) var<uniform> params : Params;

@vertex
fn vertexMain(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // look up tesselation for circle in future
    let quad = array(
        vec2f(-particle_size,  particle_size),
        vec2f( particle_size,  particle_size),
        vec2f( particle_size, -particle_size),
        vec2f(-particle_size, -particle_size)
    );

    let particle = particles[in.instance_index];

    let offset = quad[in.vertex_index] * particle_size;

    let world_pos = particle.position + offset;

    out.clip_position = vec4f(world_pos, 0.0, 1.0);
    out.color = particle.color;

    return out;
}

@fragment
fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
    return in.color;
}