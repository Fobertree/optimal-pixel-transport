// Bertsekas - Parallel Auction ALgorithm (Jacobi-style)
// Bertsekas auction algorithm only parallelizes bidding stage, update is synchronous
// To simulate synchronous nature, we use a CAS weak

// epsilon-complementary slackness: treat LAP as dual problem by adding epsilon to every bid price
// This prevents infinite loops and improves convergence
// Retain optimality by noting that this dual problem has greater cost by n * epsilon -> n * epsilon < 1 leads to optimality for integer costs
// i.e., floor(dual = total cost + n * epsilon) = total_cost -> optimality

// OG papel: https://www.columbia.edu/~cs2035/courses/ieor8100.F12/auction-alg.pdf
// https://stanford.edu/~rezab/classes/cme323/S16/projects_reports/jin.pdf: only parallel Jacobi improves perf
// https://www.cs.columbia.edu/~sedwards/classes/2024/4995-fall/reports/assignment-report.pdf
// https://web.eecs.umich.edu/~pettie/matching/Bertsekas-auction-algorithms-for-network-flow.pdf

// Reverse auction to modify to min LP - traditional auction is max LP by maximizing bids

const INT_MAX : i32 = 2147483647;
const INT_MIN : i32 = -100000;
const TILE_SIZE : u32 = 256;

// Matches RGB_DIST_INT_HYBRID on CPU (cost_function.h)
const RGB_WEIGHT : f32 = 0.9;
const INT_FACTOR : f32 = 100000.0;
const RGB_SCALE : f32 = 500.0;
const DIST_SCALE : f32 = 50.0;

override EPSILON : i32;
// CPU previously multiplied stored costs by 2 * NUM_PARTICLES for complementary slackness
override COST_MULTIPLIER : i32;

struct Particle {
    position: vec2f,
    velocity: vec2f,
    color: vec4f
};

struct TargetParticle {
    position: vec2f,
    color: vec4f,
};

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

// group 0 - params (unchanged)
@group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(1) var<uniform> params : Params;

// group 1 - solver
@group(1) @binding(0) var<storage, read_write> assignments : array<atomic<i32>>; // row → col (-1 = unassigned)
@group(1) @binding(1) var<storage, read> target_particles : array<TargetParticle>;

// WGSL does not support atomic<f32>.
// Auction-specific buffers (all size MAX_SIZE)
@group(1) @binding(2) var<storage, read_write> prices : array<i32>;           // column prices, init to 0 on CPU
@group(1) @binding(3) var<storage, read_write> bid_value : array<i32>; // highest bid per column this round
@group(1) @binding(4) var<storage, read_write> bid_from_row : array<i32>; // who placed the highest bid
@group(1) @binding(5) var<storage, read_write> owner : array<i32>;  // col idx -> row idx - basically prev for assignments

fn cost(i : u32, j : u32) -> i32 {
    let src = particles[i];
    let tgt = target_particles[j];

    let dr = src.color.r - tgt.color.r;
    let dg = src.color.g - tgt.color.g;
    let db = src.color.b - tgt.color.b;
    let rgb_cost = i32((dr * dr + dg * dg + db * db) * RGB_SCALE);

    let dx = src.position.x - tgt.position.x;
    let dy = src.position.y - tgt.position.y;
    let dist_cost = i32((dx * dx + dy * dy) * DIST_SCALE);

    let blended = RGB_WEIGHT * f32(rgb_cost) + (1.0 - RGB_WEIGHT) * f32(dist_cost);
    let scaled = i32(blended * INT_FACTOR);
    return scaled * COST_MULTIPLIER;
}

@compute @workgroup_size(TILE_SIZE)
fn auctionBiddingPhase(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let j = local_id.x + workgroup_id.x;
    let sz = params.size;
    if (j >= sz) {return;}

    var best_value :i32 = INT_MIN;
    var best_i: i32 = -1; // row assignment
    // Referred to by Bertsekas as either w or omega
    var second_best: i32 = INT_MIN;

    // a_ij is cost
    for (var i = 0; i < i32(sz); i++) {
        // skip already assigned rows
        let assignment = atomicLoad(&assignments[i]);
        if (assignment != -1) {
            continue;
        }

        // v_ij = a_ij - p_j
        var value: i32 = cost(u32(i),j) - prices[j];
        if (value > best_value) {
            // omega_j = max_{i in B(j), i != i_j} (a_ij - pi_i)
            second_best = best_value;
            // v_ij* = max_{i in A(i)} v_ij
            best_i = i;
            // beta_j = max_{i in B(j)} (a_ij - pi_i})
            best_value = value;
        } else if (value > second_best) {
            second_best = value;
        }
    }

    if (second_best == INT_MIN) {
        // prevent explosion if only one candidate
        second_best = best_value;
    }

    if (best_i != -1) {
        // beta_j - omega_j + epsilon
        // Preparation for assignment phase: j_i = argmax_{j in P(i)} (beta_j - omega_j + epsilon)
        bid_value[j] = best_value - second_best + EPSILON;
        bid_from_row[j] = best_i;
    } else {
        bid_value[j] = INT_MIN;
    }
}

@compute @workgroup_size(TILE_SIZE)
fn auctionUpdatePhase(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let j = local_id.x + workgroup_id.x;
    let sz = params.size;
    if (j >= sz) {return;}

    if (bid_value[j] == INT_MIN) {
        // no bid
        return;
    }

    let bidder = bid_from_row[j];

    prices[j] += bid_value[j];

    // CAS for race condition with multiple bidders matching
    let prev = atomicCompareExchangeWeak(
        &assignments[bidder],
        -1,
        i32(j)
    );

    // exchanged - assignments[bidder] unassigned
    if (prev.exchanged) {
        let old_owner = owner[j];

        owner[j] = bidder;

        if (old_owner != -1 && old_owner != bidder) {
            // unassign old owner
            atomicStore(&assignments[old_owner], -1);
        }
    }
}

@compute @workgroup_size(TILE_SIZE)
fn auctionInit(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    // j indexes target particle
    let j = local_id.x + workgroup_id.x;
    let sz = params.size;
    if (j >= sz) {return;}
    // p_j = min_i a_ij for all j
    for (var i = 0; i < i32(sz); i++) {
        prices[j] = min(prices[j], cost(u32(i), j));
    }
}