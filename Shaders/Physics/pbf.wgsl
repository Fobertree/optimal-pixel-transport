/*
Resources:
- https://mmacklin.com/pbf_sig_preprint.pdf (too lazy to find final print)
- https://people.engr.tamu.edu/sueda/courses/CSCE450/2022F/projects/Brandon_Nguyen/index.html#:~:text=Description%C2%A7,one%20physics%20step%20per%20frame.
*/

const TILE_SIZE : u32 = 256u; // workgroup length
const NEIGHBORHOOD_SIZE : f32 = 0.05; // hyperparam for calc particle neighborhood via counting sort
const NUM_BINS : u32 = 50000u;

struct Particle {
    position: vec2f,
    velocity: vec2f, // TODO: consolidate velocity (rm binding)
    color: vec4f
};

// minimal redundancy overhead
struct Params {
    size : u32,
    rho0 : f32,
    H: f32,         // kernel smoothing radius
    dt: f32,
    solverIterations: u32,
    cellSize: f32,
    numBins: u32,
}

// TODO: update BG grouping

// group 0 - params
@group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(1) var<storage, read> params : Params;

// group 1 - solver (assignments, costs)
@group(1) @binding(0) var<storage, read> assignments : array<i32>;

// TODO: cleanup cost_matrix from BG

// group 2 - simulation state (hot loop) + target particle
// TODO: migrate to target positions + figure out which BG this belongs in
@group(2) @binding(0) var<storage, read> target_particles : array<Particle>; // not sure if should move to solver BG
// main pbf simulation stuff
// TODO: move positions + velocities to particles struct
@group(2) @binding(0) var<storage, read_write> positions: array<vec2f>;
@group(2) @binding(1) var<storage, read_write> velocities: array<vec2f>;

// Not sure if these need to be binded
@group(2) @binding(2) var<storage, read_write> lambdas: array<f32>;
@group(2) @binding(3) var<storage, read_write> deltaPos: array<vec2f>;
@group(2) @binding(4) var<storage, read_write> posStar: array<vec2f>;       // predicted positions
@group(2) @binding(5) var<storage, read_write> binStart: array<i32>;
@group(2) @binding(6) var<storage, read_write> binCount: array<atomic<u32>>;
@group(2) @binding(7) var<storage, read_write> omega: array<f32>;          // memoize for vorticity confinement

/* utils */
fn hashCoords(pos: vec2f) -> u32 {
    // 10 minute physics hash, can replace with Z-order
    let xi = i32(floor(pos.x / params.cellSize));
    let yi = i32(floor(pos.y / params.cellSize));
    let h = (xi * 92837111) ^ (yi * 689287499);
    return u32(abs(h)) % params.numBins;
}
/* end utils */

@compute @workgroup_size(TILE_SIZE)
fn computeMain(
    @builtin(global_invocation_id) gid : vec3<u32>
    ) {
    let idx = gid.x;
    let n = params.size;

    if (idx >= n) {
        return;
    }

    let dt = params.dt;
    let H = params.H;
    let rho0 = params.rho0;
    let invRho0 = 1.0 / rho0;
    let eps = 1e-8;

    let pos  = posStar[idx];

    // Apply external forces
    var vel = velocities[idx];
    let targetIdx = assignments[idx];
    if (targetIdx >= 0 && targetIdx < i32(n)) {
        let targetPos = posStar[u32(targetIdx)];
        let dir = targetPos - pos;
        let dist = length(dir);
        if (dist > 0.0001) {
            // 0.5 "pull strength" hyperparameter
            pos += normalize(dir) * min(dist, 0.02) * dt * 0.5;
        }
    }

    posStar[idx] = pos + vel * dt;

    // Counting pass
    // particles must already be sorted by hash coords
    // TODO: radix sort wgsl impl in another shader
    let bin = hashCoords(posStar[idx]);
    atomicAdd(&binCount[bin], 1u);
}

@compute @workgroup_size(TILE_SIZE)
fn pbfSolverPass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = params.size;
    if (idx >= n) {return;}

    let H = params.H;
    let rho0 = params.rho0;
    let invRho0 = 1.0 / rho0;
    let eps = 1e-8;

    let pos = posStar[idx];
    let bin = hashCoords(pos);

    // calc density + lambda
    var density: f32 = 0.0;
    var gradSum: f32 = 0.0;             // lambda denominator

    // 3x3 bin neighborhood search
    for (var dx = -1; dx <= i32(1); dx++) {
        for (var dy = -1; dy <= i32(1); dy++) {
            // TODO: add support for local_invocation_id and modify control flow to not screw over workgroup barrier
            let posPrime = vec2(pos.x+dx, pos.y+dy);
            if (min(posPrime.x, posPrime.y) < 0 || max(posPrime.x, posPrime.y) >= params.numBins) {
                // OOB, no clamp
                continue;
            }
            let nb = hashCoords(posPrime);
            let start = binStart[nb];
            let cnt = atomicLoad(&binCount[nb]);

            // iterate over neighbors
            for (var j: u32 = 0u; j < cnt; j++) {
                let jIdx = u32(start) + j;
                if (jIdx == idx) {continue;}

                let neiPos = posStar[jIdx];
                let r = pos - neiPos;
                let dist = length(r);

                // avoid obvious 0
                if (dist > H || dist < 0.0001) { continue; }
                let q = dist / H;

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
    let C = density * invRho0 - 1.0;
    lambdas[idx] = -C / (gradSum + eps); // TODO: Need to compute all lambdas

    // Sync to ensure all lambdas are calculated
    workgroupBarrier();

    // deltaPos (with tensile instability)
    var dPos = vec2f(0.0);
    let lambda_i = lambdas[idx];

    for (var dx: i32 = -1; dx <= 1; dx++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            let posPrime = vec2(pos.x+dx, pos.y+dy);
            if (min(posPrime.x, posPrime.y) < 0 || max(posPrime.x, posPrime.y) >= params.numBins) {
                // OOB, no clamp
                continue;
            }
            let nb = hashCoords(posPrime);
            let start = binStart[nb];
            let cnt = atomicLoad(&binCount[nb]);

            for (var j: u32 = 0u; j < cnt; j++) {
                let jIdx = u32(start) + j;
                if (jIdx == idx) {continue;}

                let neiPos = posStar[jIdx];
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
                let lambda_j = lambdas[jIdx];

                // Tensile insability correction
                let k = 0.1;
                let delta_q = 0.2 * H;
                let n = 4.0;
                let W_delta = 1.0/6.0 * pow(2.0 - delta_q/H, 3.0);
                let s_corr = -k * pow(w / W_delta, n);

                dPos += (lambda_i + lambda_j + s_corr) * grad;
            }
        }
    } // end neighbor bin accumulation

    deltaPos[idx] = dPos * invRho0;

    // Position correction + velocity update
    let newPos = posStar[idx] + deltaPos[idx];
    positions[idx] = newPos;

    let newVel = (newPos - pos) / dt;
    velocities[idx] = newVel;

    // XSPH viscosity
    let C: f32 = 0.01;
    var omega_i: f32 = 0;
    var viscosity_sum: f32 = 0;
    // for vorticity
    var eta = vec2(0,0);
    var f_vorticity = vec2(0,0);

    for (var dx: i32 = -1; dx <= 1; dx++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            let posPrime = vec2(pos.x+dx, pos.y+dy);
            if (min(posPrime.x, posPrime.y) < 0 || max(posPrime.x, posPrime.y) >= params.numBins) {
                // OOB, no clamp
                continue;
            }
            let nb = hashCoords(posPrime);
            let start = binStart[nb];
            let cnt = atomicLoad(&binCout[nb]);

            for (var j: u32 = 0u; j < cnt; j++) {
                let jIdx = u32(start) + j;
                if (jIdx == idx) {continue;}

                let neiPos = posStar[jIdx];
                let r = pos - neiPos;
                let dist = length(r);
                if (dist > H || dist < 0.0001) {continue;}

                let v_ij = velocities[jIdx] - velocities[idx];

                // gradient cubic spline kernel
                let r = pos - neiPos;
                let dist = length(r);
                if (dist > H || dist < 0.0001) {continue;}

                let q = dist/H;
                let dW: f32 = 0;

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

                // XSPH viscosity in same loop
                viscosity_sum += v_ij * grad;
            } // end neighbor search
            omega[jIdx] = omega_i;

            // another loop for vorticity

            for (var j: u32 = 0u; j < cnt; j++) {
                let jIdx = u32(start) + j;
                if (jIdx == idx) {continue;}

                let neiPos = posStar[jIdx];
                let r = pos - neiPos;
                let dist = length(r);
                if (dist > H || dist < 0.0001) {continue;}

                let v_ij = velocities[jIdx] - velocities[idx];

                // gradient cubic spline kernel
                let r = pos - neiPos;
                let dist = length(r);
                if (dist > H || dist < 0.0001) {continue;}

                let q = dist/H;
                let dw: f32 = 0;

                if (q < 1.0) {
                    dw = -3 * q + 2.25 * q*q;
                } else if (q < 2.0) {
                    dw = -0.75 * pow(2.0 - q, 2);
                } else {
                    dw = 0;
                }

                let scale = dw / (H * q);
                let grad = scale * r;

                // since in 2D, we modify the math a little
                eta += ((abs(omega[idx]) - abs(omega[jIdx])) / (rho[jIdx] + eps)) * grad;
            } // end neighbor search for bin
        }
    } // end neighbor bin accumulation
    // viscosity velocity update
    velocities[idx] += C * viscosity_sum;

    // cross product with scalar on RHS here intuitively becomes 90 degrees CCW by right-hand-rule
    // perp vector: (-Ny, Nx)
    eta = eta / length(eta);
    let f_vorticity = eps * omega[idx] * vec2(-eta.y,eta.x);

    velocity[idx] += dt * f_vorticity;

    // euler step position
    positions[idx] += velocity[idx] * dt;
}