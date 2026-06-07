/*
Resources:
- https://mmacklin.com/pbf_sig_preprint.pdf (too lazy to find final print)
- https://people.engr.tamu.edu/sueda/courses/CSCE450/2022F/projects/Brandon_Nguyen/index.html#:~:text=Description%C2%A7,one%20physics%20step%20per%20frame.
- https://gpuopen.com/download/Introduction_to_GPU_Radix_Sort.pdf
*/

const TILE_SIZE : u32 = 256u; // workgroup length
const NEIGHBORHOOD_SIZE : f32 = 0.05; // hyperparam for calc particle neighborhood via counting sort
const NUM_BINS : u32 = 50000u;

struct Particle {
    position: vec2f,
    velocity: vec2f,
    color: vec4f,
};

struct TargetPos {
    position: vec2f
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
@group(1) @binding(3) var<storage, read> binStart: array<i32>;
@group(1) @binding(4) var<storage, read> binEnd: array<i32>;
@group(1) @binding(5) var<storage, read_write> omega: array<f32>;          // memoize for vorticity confinement
// solver
@group(1) @binding(6) var<storage, read> assignments: array<i32>;
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

    let dt = params.dt;
    let H = params.H;
    let rho0 = params.rho0;
    let invRho0 = 1.0 / rho0;
    let eps = 1e-8;
    let cellSize = params.cellSize;

    var pos = posStar[idx];
    let bin = hashCoords(pos);

    // Apply external forces
    // don't need sorted particles here
    var vel = particles[idx].velocity;
    let targetIdx = assignments[idx];
    if (targetIdx >= 0 && targetIdx < i32(n)) {
        let targetPos = targetParticles[u32(targetIdx)].position;
        let dir = targetPos - pos;
        let dist = length(dir);
        if (dist > 0.0001) {
            // 0.5 "pull strength" hyperparameter
            pos += normalize(dir) * min(dist, 0.02) * dt * 0.5;
        }
    }

    posStar[idx] = pos + vel * dt;
}

@compute @workgroup_size(TILE_SIZE)
fn pbfSolverPass(@builtin(global_invocation_id) gid: vec3<u32>) {
    // run AFTER sort
    let idx = gid.x;
    let n = params.size;

    if (idx >= n) {return;}

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
    let pIdx = sortIndices[idx];
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
            for (var jIdx: u32 = u32(start); jIdx < u32(end); jIdx++) {
                if (jIdx == idx) {continue;}
                let pjIdx = sortIndices[jIdx];

                let neiPos = posStar[pjIdx];
                let r = pos - neiPos;
                let dist = length(r);

                // avoid obvious 0
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
    // TODO: check if i should set pos to pos or posStar
    let pos = particles[pIdx].position;

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

            for (var jIdx: u32 = u32(start); jIdx < u32(end); jIdx++) {
                if (jIdx == idx) {continue;}
                let pjIdx = sortIndices[jIdx];

                let neiPos = posStar[pjIdx];
                let r = pos - neiPos;
                let dist = length(r);
                if (dist > H || dist < 0.0001) {continue;}

                let q = dist/H;
                var w: f32 = 0.0;
                var dw: f32 = 0.0;

                // kernel
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
    deltaPos[pIdx] = dPos * invRho0;

    // skip collision detection + response - no solids

    // Position correction + velocity update
    posStar[pIdx] = posStar[pIdx] + deltaPos[pIdx];

    let newVel = (posStar[pIdx] - pos) / dt;
    particles[pIdx].velocity = newVel;

    // XSPH viscosity
    var omega_i: f32 = 0;
    // for vorticity
    var f_vorticity = vec2(0,0);

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

            // iterate over neighbors
            for (var jIdx: u32 = u32(start); jIdx < u32(end); jIdx++) {
                if (jIdx == idx) {continue;}
                if (jIdx == idx) {continue;}
                let pjIdx = sortIndices[jIdx];

                let neiPos = posStar[pjIdx];
                let r = pos - neiPos;
                let dist = length(r);
                if (dist > H || dist < 0.0001) {continue;}

                // diff in velocity between particle and neighbor
                let v_ij = particles[pjIdx].velocity - particles[pIdx].velocity;

                // gradient cubic spline kernel

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
            } // end neighbor search
            omega[pIdx] = omega_i;
        }
    }
}
@compute @workgroup_size(TILE_SIZE)
fn pbfSolverPassThree(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = params.size;

    if (idx >= n) {return;}
    let pIdx = sortIndices[idx];
    // TODO: check if i should set pos to pos or posStar
    let pos = particles[pIdx].position;

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
            for (var jIdx: u32 = u32(start); jIdx < u32(end); jIdx++) {
                if (jIdx == idx) {continue;}
                if (jIdx == idx) {continue;}
                let pjIdx = sortIndices[jIdx];

                let neiPos = posStar[pjIdx];
                let r = pos - neiPos;
                let dist = length(r);
                if (dist > H || dist < 0.0001) {continue;}

                let v_ij = particles[pjIdx].velocity - particles[pIdx].velocity;

                // gradient cubic spline kernel
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

                // XSPH viscosity in same loop
                viscosity_sum += v_ij * grad;

                // since in 2D, we modify the math a little
                // eta for location vector
                // TODO: check for correctness
                eta += ((abs(omega[pIdx]) - abs(omega[pjIdx])) / (rho0 + eps)) * grad;
            } // end neighbor search for bin
        }
    } // end neighbor bin accumulation
    // viscosity velocity update
    particles[pIdx].velocity += C * viscosity_sum;

    // cross product with scalar on RHS here intuitively becomes 90 degrees CCW by right-hand-rule
    // perp vector: (-Ny, Nx)
    eta = eta / length(eta);
    let f_vorticity = eps * omega[pIdx] * vec2(-eta.y,eta.x);

    particles[pIdx].velocity += dt * f_vorticity;

    // euler step position
    particles[pIdx].position += particles[pIdx].velocity * dt;
    posStar[pIdx] = particles[pIdx].position;
}