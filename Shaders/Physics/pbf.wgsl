/*
Resources:
- https://mmacklin.com/pbf_sig_preprint.pdf (too lazy to find final print)
- https://people.engr.tamu.edu/sueda/courses/CSCE450/2022F/projects/Brandon_Nguyen/index.html#:~:text=Description%C2%A7,one%20physics%20step%20per%20frame.
- https://gpuopen.com/download/Introduction_to_GPU_Radix_Sort.pdf
*/

const TILE_SIZE : u32 = 256u; // workgroup length
const NEIGHBORHOOD_SIZE : f32 = 0.05; // hyperparam for calc particle neighborhood via counting sort
const NUM_BINS : u32 = 50000u;
const ASSIGNMENT_PULL : f32 = 42.0;       // spring constant toward assigned target
const ASSIGNMENT_DAMPING : f32 = 10.0;    // velocity damping — prevents oscillation / teleport feel
const MAX_PBF_CORRECTION : f32 = 0.06;    // cap density correction per iter to avoid position spikes
const PBF_VELOCITY_BLEND : f32 = 0.15;    // blend constraint vel into assignment-driven vel (low = smoother)

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

/* end utils */
@compute @workgroup_size(TILE_SIZE)
fn pbfExternalForces(@builtin(global_invocation_id) gid: vec3<u32>) {
    // run BEFORE sort
    // this doesn't utilize sorted indices bc no neighborhood search
    let idx = gid.x;
    let n = params.size;

    if (idx >= n) {return;}

    let dt = params.dt;
    let H = params.H;
    let rho0 = params.rho0;
    let invRho0 = 1.0 / rho0;
    let eps = 1e-8;
    let cellSize = params.cellSize;

    var pos = particles[idx].position;
    var vel = particles[idx].velocity;

    // Apply assignment pull via spring-damper on velocity (smooth even when auction reassigns)
    let targetIdx = atomicLoad(&assignments[idx]);
    var targetPos = targetParticles[idx].position; // identity slot while unassigned
    if (targetIdx >= 0 && targetIdx < i32(n)) {
        targetPos = targetParticles[u32(targetIdx)].position;
    }
    let dir = targetPos - pos;
    vel += (dir * ASSIGNMENT_PULL - vel * ASSIGNMENT_DAMPING) * dt;

    pos += vel * dt;
    posStar[idx] = pos;
    particles[idx].position = pos;
    particles[idx].velocity = vel;
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
    var density: f32 = 0.0;
    var gradSum: f32 = 0.0;             // lambda denominator

    // 3x3 bin neighborhood search
    // TODO: check if i should set pos to pos or posStar
    let pos = posStar[pIdx];
    for (var dx = -1; dx <= i32(1); dx++) {
        for (var dy = -1; dy <= i32(1); dy++) {
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
    lambdas[pIdx] = -C / (gradSum + eps);
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
    let lambda_i = lambdas[pIdx];

    for (var dx: i32 = -1; dx <= 1; dx++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            let posPrime = vec2(pos.x+f32(dx)*cellSize, pos.y+f32(dy)*cellSize);
            if (min(posPrime.x, posPrime.y) < -1 || max(posPrime.x, posPrime.y) > 1) {
                // OOB, no clamp
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
                let lambda_j = lambdas[pjIdx];

                // Tensile instability correction
                let k = 0.1;
                let delta_q = 0.2 * H;
                let n = 4.0;
                let W_delta = 1.0/6.0 * pow(2.0 - delta_q/H, 3.0);
                let s_corr = -k * pow(w / W_delta, n);

                dPos += (lambda_i + lambda_j + s_corr) * grad;
            }
        }
    } // end neighbor bin accumulation

    // calculate delta pos
    var correction = dPos * invRho0;
    deltaPos[pIdx] = correction;

    // skip collision detection + response - no solids

    // Cap density correction — uncapped lambda steps cause occasional teleports
    let corrLen = length(correction);
    if (corrLen > MAX_PBF_CORRECTION) {
        correction = correction * (MAX_PBF_CORRECTION / corrLen);
    }

    let prevVel = particles[pIdx].velocity;
    posStar[pIdx] = pos + correction;
    let constraintVel = correction / dt;
    particles[pIdx].velocity = mix(prevVel, constraintVel, PBF_VELOCITY_BLEND);

    var omega_i: f32 = 0;

    for (var dx: i32 = -1; dx <= 1; dx++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
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

                let v_ij = particles[pjIdx].velocity - particles[pIdx].velocity;

                let q = dist/H;
                var dw: f32 = 0;

                if (q < 1) {
                    dw = -3 * q + 2.25 * q*q;
                } else if (q < 2) {
                    dw = -0.75 * pow(2.0 - q, 2);
                } else {
                    dw = 0;
                }

                let scale = dw / (H * q);
                let grad = scale * r;

                omega_i += v_ij.x * grad.y - v_ij.y * grad.x;
            }
        }
    }
    omega[pIdx] = omega_i;
}
@compute @workgroup_size(TILE_SIZE)
fn pbfSolverPassThree(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    var eta = vec2f(0,0);
    var viscosity_sum = vec2f(0,0);

    // TODO: migrate this to params
    let C: f32 = 0.01;

    for (var dx: i32 = -1; dx <= 1; dx++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            let posPrime = vec2(pos.x+f32(dx)*cellSize, pos.y+f32(dy)*cellSize);
            if (min(posPrime.x, posPrime.y) < -1 || max(posPrime.x, posPrime.y) > 1) {
                // OOB, no clamp
                continue;
            }
            let nb = hashCoords(posPrime);
            let start = binStart[nb];
            let end = binEnd[nb];

            // another loop for vorticity
            for (var jIdx: u32 = start; jIdx < end; jIdx++) {
                if (jIdx == idx) {continue;}
                let pjIdx = sortIndices[jIdx];

                let neiPos = posStar[pjIdx];
                let r = pos - neiPos;
                let dist = length(r);
                if (dist > H || dist < 0.0001) {continue;}

                let v_ij = particles[pjIdx].velocity - particles[pIdx].velocity;

                var q = dist/H;
                var dw: f32 = 0;

                if (q < 1.0) {
                    dw = -3 * q + 2.25 * q*q;
                } else if (q < 2.0) {
                    dw = -0.75 * pow(2.0 - q, 2);
                } else {
                    dw = 0;
                }

                let scale = dw / (H * q);
                let grad = scale * r;

                viscosity_sum += v_ij * grad;
                eta += ((abs(omega[pIdx]) - abs(omega[pjIdx])) / (rho0 + eps)) * grad;
            }
        }
    }

    particles[pIdx].velocity += C * viscosity_sum;

    let eta_len = length(eta);
    if (eta_len > eps) {
        eta = eta / eta_len;
        let f_vorticity = eps * omega[pIdx] * vec2(-eta.y, eta.x);
        particles[pIdx].velocity += dt * f_vorticity;
    }

    particles[pIdx].position = pos + particles[pIdx].velocity * dt;
    posStar[pIdx] = particles[pIdx].position;
}