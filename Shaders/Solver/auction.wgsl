// Bertsekas - Parallel Auction ALgorithm (Jacobi-style)
// Bertsekas auction algorithm only parallelizes bidding stage, update is synchronous
// To simulate synchronous nature, we use a CAS weak
//
// epsilon-complementary slackness: treat LAP as dual problem by adding epsilon to every bid price
// This prevents infinite loops and improves convergence
// Retain optimality by noting that this dual problem has greater cost by n * epsilon -> n * epsilon < 1 leads to optimality for integer costs
// i.e., floor(dual = total cost + n * epsilon) = total_cost -> optimality
//
// OG papel: https://www.columbia.edu/~cs2035/courses/ieor8100.F12/auction-alg.pdf
// https://stanford.edu/~rezab/classes/cme323/S16/projects_reports/jin.pdf: only parallel Jacobi improves perf
// https://www.cs.columbia.edu/~sedwards/classes/2024/4995-fall/reports/assignment-report.pdf
// https://web.eecs.umich.edu/~pettie/matching/Bertsekas-auction-algorithms-for-network-flow.pdf
//
// Reverse auction to modify to min LP - traditional auction is max LP by maximizing bids
//
// Min-cost LAP via forward ε-auction on benefits a_ij = -cost_ij:
//   row i maximizes  a_ij - π_j  =  -cost_ij - π_j
//   bid increment    (v* - v_second) + ε
// Costs are integer with min gap 1 (tiebreak); ε is passed from CPU (typically 1).
//
// When CANDIDATE_RADIUS < GRID_DIM, bidding scans only a local 2D grid window (O(N) per iter).

const INT_MAX : i32 = 2147483647;
const INT_MIN : i32 = -100000;
const TILE_SIZE : u32 = 256;

const INT_FACTOR : f32 = 1000.0;  // keep i32 costs well below INT_MAX
const RGB_SCALE : f32 = 500.0;
const RGB_WEIGHT : f32 = 0.9;
const DIST_SCALE : f32 = 50.0;

override EPSILON : i32;
override GRID_DIM : u32;
override CANDIDATE_RADIUS : u32;
override USE_DISTANCE : i32; // 1 = RGB+distance (convergence), 0 = RGB only (live)

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
    rho0 : f32,
    H: f32,
    dt: f32,
    solverIterations: u32,
    cellSize: f32,
    numBins: u32,
}

@group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(1) var<uniform> params : Params;

@group(1) @binding(0) var<storage, read_write> assignments : array<atomic<i32>>;
@group(1) @binding(1) var<storage, read> target_particles : array<TargetParticle>;
@group(1) @binding(2) var<storage, read_write> prices : array<i32>;
@group(1) @binding(3) var<storage, read_write> bid_value : array<i32>;
@group(1) @binding(4) var<storage, read_write> bid_from_row : array<i32>;
@group(1) @binding(5) var<storage, read_write> owner : array<atomic<i32>>;

fn cost(i : u32, j : u32) -> i32 {
    let src_color = particles[i].color;
    let tgt = target_particles[j];

    let dr = src_color.r - tgt.color.r;
    let dg = src_color.g - tgt.color.g;
    let db = src_color.b - tgt.color.b;
    let rgb_cost = i32((dr * dr + dg * dg + db * db) * RGB_SCALE);

    var blended = f32(rgb_cost);
    if (USE_DISTANCE != 0) {
        // Fixed grid geometry during convergence — not live particle positions.
        let src_pos = target_particles[i].position;
        let dx = src_pos.x - tgt.position.x;
        let dy = src_pos.y - tgt.position.y;
        let dist_cost = i32((dx * dx + dy * dy) * DIST_SCALE);
        blended = RGB_WEIGHT * f32(rgb_cost) + (1.0 - RGB_WEIGHT) * f32(dist_cost);
    }

    let tiebreak = abs(i32(i) - i32(j));
    return i32(blended * INT_FACTOR) + tiebreak;
}

// Benefit for min-cost forward auction: a_ij - π_j = -cost_ij - π_j
fn net_value(i : u32, j : u32) -> i32 {
    return -cost(i, j) - prices[j];
}

fn index_to_rc(idx: u32) -> vec2<i32> {
    return vec2<i32>(i32(idx / GRID_DIM), i32(idx % GRID_DIM));
}

fn rc_to_index(row: i32, col: i32) -> u32 {
    return u32(row) * GRID_DIM + u32(col);
}

fn in_bounds(row: i32, col: i32, grid_dim: i32) -> bool {
    return row >= 0 && col >= 0 && row < grid_dim && col < grid_dim;
}

@compute @workgroup_size(TILE_SIZE)
fn auctionBiddingPhase(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let i = local_id.x + workgroup_id.x * TILE_SIZE;
    let sz = params.size;
    if (i >= sz) {
        return;
    }

    bid_value[i] = INT_MIN;
    bid_from_row[i] = -1;

    if (atomicLoad(&assignments[i]) != -1) {
        return;
    }

    let grid_dim = i32(GRID_DIM);
    let radius = i32(CANDIDATE_RADIUS);
    let full_scan = CANDIDATE_RADIUS >= GRID_DIM;
    let pi = index_to_rc(i);

    var best_value: i32 = INT_MIN;
    var second_best: i32 = INT_MIN;
    var best_j: i32 = -1;

    if (full_scan) {
        for (var j = 0; j < i32(sz); j++) {
            if (atomicLoad(&owner[u32(j)]) >= 0) {
                continue;
            }
            let value = net_value(i, u32(j));
            if (value > best_value) {
                second_best = best_value;
                best_j = j;
                best_value = value;
            } else if (value > second_best) {
                second_best = value;
            }
        }
    } else {
        for (var dr = -radius; dr <= radius; dr++) {
            for (var dc = -radius; dc <= radius; dc++) {
                let rj = pi.x + dr;
                let cj = pi.y + dc;
                if (!in_bounds(rj, cj, grid_dim)) {
                    continue;
                }
                let j = i32(rc_to_index(rj, cj));
                if (atomicLoad(&owner[u32(j)]) >= 0) {
                    continue;
                }
                let value = net_value(i, u32(j));
                if (value > best_value) {
                    second_best = best_value;
                    best_j = j;
                    best_value = value;
                } else if (value > second_best) {
                    second_best = value;
                }
            }
        }
    }

    if (best_j < 0) {
        return;
    }

    if (second_best == INT_MIN) {
        second_best = best_value;
    }

    bid_value[i] = best_value - second_best + EPSILON;
    bid_from_row[i] = best_j;
}

@compute @workgroup_size(TILE_SIZE)
fn auctionUpdatePhase(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let j = local_id.x + workgroup_id.x * TILE_SIZE;
    let sz = params.size;
    if (j >= sz) {
        return;
    }

    var best_bid: i32 = INT_MIN;
    var best_row: i32 = -1;

    for (var i = 0; i < i32(sz); i++) {
        if (bid_from_row[u32(i)] != i32(j)) {
            continue;
        }
        let bid = bid_value[u32(i)];
        if (bid > best_bid) {
            best_bid = bid;
            best_row = i;
        }
    }

    if (best_row < 0 || best_bid == INT_MIN) {
        return;
    }

    // Fill-only: never evict an existing owner (eviction caused per-frame churn).
    if (atomicLoad(&owner[j]) >= 0) {
        return;
    }

    let prev = atomicCompareExchangeWeak(
        &assignments[u32(best_row)],
        -1,
        i32(j)
    );

    if (prev.exchanged) {
        prices[j] += best_bid;
        atomicStore(&owner[j], best_row);
    }
}

// Reconcile owner[] with assignments[] — ghost owners block repair from finding free columns.
@compute @workgroup_size(TILE_SIZE)
fn auctionSyncPhase(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let k = local_id.x + workgroup_id.x * TILE_SIZE;
    let sz = params.size;
    if (k >= sz) {
        return;
    }

    let col_owner = atomicLoad(&owner[k]);
    if (col_owner >= 0 && atomicLoad(&assignments[u32(col_owner)]) != i32(k)) {
        atomicStore(&owner[k], -1);
    }

    let assigned_col = atomicLoad(&assignments[k]);
    if (assigned_col < 0) {
        return;
    }

    let ucol = u32(assigned_col);
    let owner_of_col = atomicLoad(&owner[ucol]);
    if (owner_of_col == i32(k)) {
        return;
    }
    if (owner_of_col < 0) {
        let claimed = atomicCompareExchangeWeak(&owner[ucol], -1, i32(k));
        if (!claimed.exchanged) {
            atomicStore(&assignments[k], -1);
        }
        return;
    }
    atomicStore(&assignments[k], -1);
}

// Greedy bijection repair: assign any still-unassigned rows to free columns (min cost).
@compute @workgroup_size(TILE_SIZE)
fn auctionRepairPhase(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let i = local_id.x + workgroup_id.x * TILE_SIZE;
    let sz = params.size;
    if (i >= sz) {
        return;
    }

    if (atomicLoad(&assignments[i]) != -1) {
        return;
    }

    var best_j: i32 = -1;
    var best_cost: i32 = INT_MAX;

    for (var j = 0; j < i32(sz); j++) {
        if (atomicLoad(&owner[u32(j)]) >= 0) {
            continue;
        }
        let c = cost(i, u32(j));
        if (c < best_cost) {
            best_cost = c;
            best_j = j;
        }
    }

    if (best_j < 0) {
        return;
    }

    let claimed = atomicCompareExchangeWeak(&owner[u32(best_j)], -1, i32(i));
    if (!claimed.exchanged) {
        return;
    }

    let assigned = atomicCompareExchangeWeak(&assignments[i], -1, best_j);
    if (!assigned.exchanged) {
        atomicStore(&owner[u32(best_j)], -1);
    }
}

@compute @workgroup_size(TILE_SIZE)
fn auctionInit(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let j = local_id.x + workgroup_id.x * TILE_SIZE;
    let sz = params.size;
    if (j >= sz) {
        return;
    }
    // Forward ε-auction on a_ij = -cost_ij starts with object prices at 0
    prices[j] = 0;
}