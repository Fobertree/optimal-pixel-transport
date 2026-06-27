/*
Resources:
- https://mmacklin.com/pbf_sig_preprint.pdf (too lazy to find final print)
- https://people.engr.tamu.edu/sueda/courses/CSCE450/2022F/projects/Brandon_Nguyen/index.html#:~:text=Description%C2%A7,one%20physics%20step%20per%20frame.
- https://gpuopen.com/download/Introduction_to_GPU_Radix_Sort.pdf
*/

const TILE_SIZE : u32 = 256u; // workgroup length
const NEIGHBORHOOD_SIZE : f32 = 0.05; // hyperparam for calc particle neighborhood via counting sort
const NUM_BINS : u32 = 50000u;
const TRANSPORT_RATE : f32 = 10.0;        // 1/s — Hungarian morph rate (always applied)
const TRANSPORT_MAX_SPEED : f32 = 2.0;    // sim units/s — transport speed cap
const MAX_PARTICLE_SPEED : f32 = 0.75;    // sim units/s — PBF velocity cap
const SIM_BOUNDS : f32 = 1.05;            // keep particles in clip-space neighborhood
const MIN_EFFECTIVE_DT : f32 = 1.0 / 120.0; // floor for disp→vel (prevents blow-up when dt is tiny)
const MAX_LAMBDA : f32 = 25.0;            // soft cap on Lagrange multipliers
const MAX_PBF_CORRECTION : f32 = 0.006;   // per-iter position correction cap (sim units)
const MAX_PBF_DISP_FRAME : f32 = 0.012;   // per-frame PBF displacement cap
const MAX_PBF_VEL : f32 = 0.35;           // max PBF velocity contribution (sim units/s)
const PBF_CONSTRAINT_GAIN : f32 = 6.0;    // disp→vel gain (1/s), dt-independent
const PBF_VELOCITY_BLEND : f32 = 0.025;   // blend PBF vel into assignment vel
const MAX_VISCOSITY_IMPULSE : f32 = 0.08; // per-frame viscosity velocity cap
const MAX_VORTICITY_IMPULSE : f32 = 0.04; // per-frame vorticity velocity cap
const MORPH_RAMP_FRAMES : f32 = 120.0;    // ~2s at 60fps: PBF influence ramps in (assignment pull is always on)
const STENCIL_RADIUS : i32 = 2;           // 5x5 bins — cubic spline support is 2H, cellSize = H
const VORTICITY_EPS : f32 = 0.004;        // vorticity confinement strength
const VISCOSITY_SOFTMAX_TEMP : f32 = 0.12; // neighbor logit temperature for XSPH softmax
const VISCOSITY_GAIN : f32 = 0.08;        // max viscosity impulse after softmax activation
const KERNEL_W0 : f32 = 2.0 / 3.0;        // cubic spline W(0) self-density contribution

struct Particle {
    position: vec2f,
    velocity: vec2f,
    color: vec4f,
};

struct TargetPos {
    position: vec2f,
    color: vec4f,
}

// minimal redundancy overhead
struct Params {
    size : u32,
    rho0 : f32,
    H: f32,         // kernel smoothing radius
    dt: f32,        // TODO: instead of hardcoding this, maybe expose this to CFL conditions
    solverIterations: u32,
    cellSize: f32,
    numBins: u32,
    frameCount: u32,
}

// 0 → 1 over MORPH_RAMP_FRAMES (quadratic ease-in)
fn morphRamp() -> f32 {
    let t = f32(params.frameCount) / MORPH_RAMP_FRAMES;
    return min(t * t, 1.0);
}

// PBF off during morph; ramps in over the last 25% of MORPH_RAMP_FRAMES
fn pbfInfluence() -> f32 {
    let t = morphRamp();
    return clamp((t - 0.75) / 0.25, 0.0, 1.0);
}

// group 0 - params
@group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(1) var<uniform> params : Params;

// group 1 - simulation state (hot loop)
// main pbf simulation stuff
@group(1) @binding(0) var<storage, read_write> lambdas: array<f32>;
@group(1) @binding(1) var<storage, read_write> deltaPos: array<vec2f>;
@group(1) @binding(2) var<storage, read_write> posStar: array<vec2f>;           // predicted positions
@group(1) @binding(3) var<storage, read_write> binStart: array<u32>;
@group(1) @binding(4) var<storage, read_write> binEnd: array<u32>;
@group(1) @binding(5) var<storage, read_write> omega: array<f32>;          // memoize for vorticity confinement
// solver
@group(1) @binding(6) var<storage, read_write> assignments: array<atomic<i32>>;
@group(1) @binding(7) var<storage, read> targetParticles : array<TargetPos>;    // constant, don't need to swap
// single layer of indirection
@group(1) @binding(8) var<storage, read> sortIndices: array<u32>;

/* utils */
// TODO: migrate to Morton code/Z-order
fn hashCoords(pos: vec2f) -> u32 {
    // 10 minute physics hash, can replace with Z-order
    let xi = i32(floor(pos.x / params.cellSize));
    let yi = i32(floor(pos.y / params.cellSize));
    // Believe you can arbitrarily xor these numbers * any set of arbitrarily large prime (or coprime) numbers
    // Wonder if there's any analytical/non-empirical way to validate effectiveness against hash-collisions
    let h = (xi * 92837111) ^ (yi * 689287499);
    return u32(abs(h)) % params.numBins;
}

fn clampSimPos(p: vec2f) -> vec2f {
    return clamp(p, vec2f(-SIM_BOUNDS), vec2f(SIM_BOUNDS));
}

// Linear near zero (preserves local gradient), soft-saturates at large magnitude
fn softCompressScalar(x: f32, maxAbs: f32) -> f32 {
    let a = abs(x);
    if (a < 1e-8) {
        return x;
    }
    return sign(x) * maxAbs * tanh(a / maxAbs);
}

fn softCompressVec(v: vec2f, maxLen: f32) -> vec2f {
    let len = length(v);
    if (len < 1e-8) {
        return v;
    }
    let newLen = maxLen * tanh(len / maxLen);
    return v * (newLen / len);
}

fn effectiveDt(dt: f32) -> f32 {
    return max(dt, MIN_EFFECTIVE_DT);
}

fn transportVelFromDisp(disp: vec2f, dt: f32) -> vec2f {
    return softCompressVec(disp / effectiveDt(dt), TRANSPORT_MAX_SPEED);
}

fn velFromDisp(disp: vec2f, dt: f32) -> vec2f {
    return softCompressVec(disp / effectiveDt(dt), MAX_PARTICLE_SPEED);
}

fn clampSpeed(v: vec2f) -> vec2f {
    return softCompressVec(v, MAX_PARTICLE_SPEED);
}

// Symmetric softmax saturation: gain * tanh(||v||/temp) * direction
fn softmaxClampVec(v: vec2f, temperature: f32, gain: f32) -> vec2f {
    let len = length(v);
    if (len < 1e-8) {
        return vec2f(0.0);
    }
    let activated = tanh(len / max(temperature, 1e-6));
    return (v / len) * gain * activated;
}

fn assignmentSlot(idx: u32, n: u32) -> u32 {
    let targetIdx = atomicLoad(&assignments[idx]);
    if (targetIdx >= 0 && targetIdx < i32(n)) {
        return u32(targetIdx);
    }
    return idx;
}

/* end utils */
@compute @workgroup_size(TILE_SIZE)
fn pbfExternalForces(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Seed PBF neighborhood search from committed positions — transport happens in assignmentTransport
    let idx = gid.x;
    if (idx >= params.size) {
        return;
    }
    posStar[idx] = particles[idx].position;
}

@compute @workgroup_size(TILE_SIZE)
fn clearSpatialBins(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.numBins) {
        return;
    }
    binStart[idx] = params.size;
    binEnd[idx] = 0u;
}

@compute @workgroup_size(TILE_SIZE)
fn buildSpatialBins(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = params.size;
    if (idx >= n) {
        return;
    }

    let pIdx = sortIndices[idx];
    let bin = hashCoords(posStar[pIdx]);

    if (idx == 0u) {
        binStart[bin] = 0u;
    } else {
        let prev_p = sortIndices[idx - 1u];
        let prev_bin = hashCoords(posStar[prev_p]);
        if (bin != prev_bin) {
            binStart[bin] = idx;
        }
    }

    if (idx == n - 1u) {
        binEnd[bin] = n;
    } else {
        let next_p = sortIndices[idx + 1u];
        let next_bin = hashCoords(posStar[next_p]);
        if (bin != next_bin) {
            binEnd[bin] = idx + 1u;
        }
    }
}

@compute @workgroup_size(TILE_SIZE)
fn pbfSolverPass(@builtin(global_invocation_id) gid: vec3<u32>) {
    // run AFTER sort
    let idx = gid.x;
    let n = params.size;

    if (idx >= n) {return;}

    let pIdx = sortIndices[idx];

    let dt = params.dt;
    let H = params.H;
    let rho0 = params.rho0;
    let invRho0 = 1.0 / rho0;
    let eps = 1e-8;
    let cellSize = params.cellSize;

    // calc density + lambda
    var density: f32 = KERNEL_W0;
    var gradSum: f32 = 0.0;

    let pos = posStar[pIdx];
    for (var dx = -STENCIL_RADIUS; dx <= STENCIL_RADIUS; dx++) {
        for (var dy = -STENCIL_RADIUS; dy <= STENCIL_RADIUS; dy++) {
            let posPrime = vec2(pos.x+f32(dx)*cellSize, pos.y+f32(dy)*cellSize);
            if (min(posPrime.x, posPrime.y) < -1 || max(posPrime.x, posPrime.y) > 1) {
                // OOB, no clamp
                continue;
            }
            let nb = hashCoords(posPrime);
            let start = binStart[nb];
            let end = binEnd[nb];

            // iterate over neighbors
            for (var jIdx: u32 = start; jIdx < end; jIdx++) {
                if (jIdx == idx) {continue;}
                let pjIdx = sortIndices[jIdx];

                let neiPos = posStar[pjIdx];
                let r = pos - neiPos;
                let dist = length(r);

                if (dist > H || dist < 0.0001) { continue; }
                let q = dist / H;
                var w: f32 = 0.0;
                var dw: f32 = 0.0;

                // cubic spline kernel
                if (q < 1.0) {
                    w = 2.0/3.0 - q*q + 0.5 * q*q*q;
                    dw = -3.0 * q + 2.25 * q*q;
                } else if (q < 2.0) {
                    w = 1.0/6.0 * pow(2.0-q, 3.0);
                    dw = -0.75*pow(2.0-q,2.0);
                }

                density += w;

                // gradient contribution for lambda
                let grad = (dw / (H*dist)) * r;
                gradSum += dot(grad, grad);
            }
        }
    } // end neighbor bin accumulation
    // this C is different from the c (in the later XSPH viscosity stage)
    // this C is for density constraints C_i (p_1, ..., p_n)
    let C = density * invRho0 - 1.0;
    lambdas[pIdx] = softCompressScalar(-C / (gradSum + eps), MAX_LAMBDA);
}

@compute @workgroup_size(TILE_SIZE)
fn pbfSolverPassTwo(@builtin(global_invocation_id) gid: vec3<u32>) {
    // run AFTER sort
    let idx = gid.x;
    let n = params.size;

    if (idx >= n) {return;}
    let pIdx = sortIndices[idx];
    let pos = posStar[pIdx];

    let dt = params.dt;
    let H = params.H;
    let rho0 = params.rho0;
    let invRho0 = 1.0 / rho0;
    let eps = 1e-8;
    let cellSize = params.cellSize;

    // deltaPos (with tensile instability)
    var dPos = vec2f(0.0);
    let lambda_i = softCompressScalar(lambdas[pIdx], MAX_LAMBDA);

    for (var dx: i32 = -STENCIL_RADIUS; dx <= STENCIL_RADIUS; dx++) {
        for (var dy: i32 = -STENCIL_RADIUS; dy <= STENCIL_RADIUS; dy++) {
            let posPrime = vec2(pos.x+f32(dx)*cellSize, pos.y+f32(dy)*cellSize);
            if (min(posPrime.x, posPrime.y) < -1 || max(posPrime.x, posPrime.y) > 1) {
                continue;
            }
            let nb = hashCoords(posPrime);
            let start = binStart[nb];
            let end = binEnd[nb];

            for (var jIdx: u32 = start; jIdx < end; jIdx++) {
                if (jIdx == idx) {continue;}
                let pjIdx = sortIndices[jIdx];

                let neiPos = posStar[pjIdx];
                let r = pos - neiPos;
                let dist = length(r);
                if (dist > H || dist < 0.0001) {continue;}

                let q = dist/H;
                var w: f32 = 0.0;
                var dw: f32 = 0.0;
                if (q < 1.0) {
                    w = 2.0/3.0 - q*q + 0.5 * q*q*q;
                    dw = -3.0 * q + 2.25 * q*q;
                } else if (q < 2.0) {
                    w = 1.0/6.0 * pow(2.0 - q, 3.0);
                    dw = -0.75 * pow(2.0 - q, 2.0);
                }

                let grad = (dw / (H * dist)) * r;
                let lambda_j = softCompressScalar(lambdas[pjIdx], MAX_LAMBDA);

                let k = 0.1;
                let delta_q = 0.2 * H;
                let n = 4.0;
                let W_delta = 1.0/6.0 * pow(2.0 - delta_q/H, 3.0);
                let s_corr = -k * pow(w / W_delta, n);

                dPos += (lambda_i + lambda_j + s_corr) * grad;
            }
        }
    }

    var correction = softCompressVec(dPos * invRho0, MAX_PBF_CORRECTION);
    deltaPos[pIdx] = correction;

    let influence = pbfInfluence();
    correction = softCompressVec(correction * influence, MAX_PBF_CORRECTION * influence);

    posStar[pIdx] = clampSimPos(pos + correction);
}

@compute @workgroup_size(TILE_SIZE)
fn assignmentTransport(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = params.size;
    if (idx >= n) {
        return;
    }

    let dt = params.dt;
    let slot = assignmentSlot(idx, n);
    let assignedPos = targetParticles[slot].position;
    let assignedColor = targetParticles[slot].color;

    let pos = particles[idx].position;
    let toTarget = assignedPos - pos;
    let dist = length(toTarget);

    var newPos = pos;
    let alpha = 1.0 - exp(-TRANSPORT_RATE * dt);
    if (dist > 1e-6) {
        newPos = pos + toTarget * alpha;
    } else {
        newPos = assignedPos;
    }
    newPos = clampSimPos(newPos);

    let disp = newPos - pos;
    particles[idx].position = newPos;
    posStar[idx] = newPos;
    particles[idx].velocity = transportVelFromDisp(disp, dt);
}

@compute @workgroup_size(TILE_SIZE)
fn pbfSolverPassThree(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Runs AFTER assignmentTransport — XSPH viscosity + vorticity on velocity only
    let idx = gid.x;
    let n = params.size;

    if (idx >= n) {return;}
    let pIdx = sortIndices[idx];
    let pos = particles[pIdx].position;

    let H = params.H;
    let cellSize = params.cellSize;
    let influence = pbfInfluence();
    var omega_i: f32 = 0.0;
    var viscLogitMax = -1e9;

    for (var dx: i32 = -STENCIL_RADIUS; dx <= STENCIL_RADIUS; dx++) {
        for (var dy: i32 = -STENCIL_RADIUS; dy <= STENCIL_RADIUS; dy++) {
            let posPrime = vec2(pos.x+f32(dx)*cellSize, pos.y+f32(dy)*cellSize);
            if (min(posPrime.x, posPrime.y) < -1 || max(posPrime.x, posPrime.y) > 1) {
                continue;
            }
            let nb = hashCoords(posPrime);
            let start = binStart[nb];
            let end = binEnd[nb];

            for (var jIdx: u32 = start; jIdx < end; jIdx++) {
                if (jIdx == idx) {continue;}
                let pjIdx = sortIndices[jIdx];

                let neiPos = particles[pjIdx].position;
                let r = pos - neiPos;
                let dist = length(r);
                if (dist > H || dist < 0.0001) {continue;}

                let v_ij = particles[pjIdx].velocity - particles[pIdx].velocity;
                let q = dist / H;
                var dw: f32 = 0.0;
                if (q < 1.0) {
                    dw = -3.0 * q + 2.25 * q * q;
                } else if (q < 2.0) {
                    dw = -0.75 * pow(2.0 - q, 2.0);
                }

                let scale = dw / (H * q);
                let grad = scale * r;
                omega_i += v_ij.x * grad.y - v_ij.y * grad.x;
                viscLogitMax = max(viscLogitMax, dot(v_ij, grad));
            }
        }
    }

    var viscWeighted = vec2f(0.0);
    var viscWeightSum = 0.0;

    for (var dx2: i32 = -STENCIL_RADIUS; dx2 <= STENCIL_RADIUS; dx2++) {
        for (var dy2: i32 = -STENCIL_RADIUS; dy2 <= STENCIL_RADIUS; dy2++) {
            let posPrime = vec2(pos.x+f32(dx2)*cellSize, pos.y+f32(dy2)*cellSize);
            if (min(posPrime.x, posPrime.y) < -1 || max(posPrime.x, posPrime.y) > 1) {
                continue;
            }
            let nb = hashCoords(posPrime);
            let start = binStart[nb];
            let end = binEnd[nb];

            for (var jIdx2: u32 = start; jIdx2 < end; jIdx2++) {
                if (jIdx2 == idx) {continue;}
                let pjIdx = sortIndices[jIdx2];

                let neiPos = particles[pjIdx].position;
                let r = pos - neiPos;
                let dist = length(r);
                if (dist > H || dist < 0.0001) {continue;}

                let v_ij = particles[pjIdx].velocity - particles[pIdx].velocity;
                let q = dist / H;
                var dw: f32 = 0.0;
                if (q < 1.0) {
                    dw = -3.0 * q + 2.25 * q * q;
                } else if (q < 2.0) {
                    dw = -0.75 * pow(2.0 - q, 2.0);
                }

                let scale = dw / (H * q);
                let grad = scale * r;
                let logit = dot(v_ij, grad);
                let w = exp((logit - viscLogitMax) / VISCOSITY_SOFTMAX_TEMP);
                viscWeightSum += w;
                viscWeighted += w * v_ij;
            }
        }
    }

    omega[pIdx] = omega_i;

    if (influence > 1e-6 && viscWeightSum > 1e-8) {
        let viscosity_sum = viscWeighted / viscWeightSum;
        let viscImpulse = softmaxClampVec(viscosity_sum, VISCOSITY_SOFTMAX_TEMP, VISCOSITY_GAIN * influence);
        particles[pIdx].velocity = clampSpeed(particles[pIdx].velocity + viscImpulse);
    }
}