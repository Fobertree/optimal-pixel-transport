// Parallel Auction ALgorithm (Jacobi-style)
// epsilon-complementary slackness: treat LAP as dual problem by adding epsilon to every bid price
// This prevents infinite loops and improves convergence
// Retain optimality by noting that this dual problem has greater cost by n * epsilon -> n * epsilon < 1 leads to optimality for integer costs
// i.e., floor(dual = total cost + n * epsilon) = total_cost -> optimality

// https://stanford.edu/~rezab/classes/cme323/S16/projects_reports/jin.pdf: only parallel Jacobi improves perf

const INT_MAX : i32 = 2147483647;
const TILE_SIZE : u32 = 256u;
const MAX_SIZE : u32 = 25000; // 500^2
// no guaranteed optimality but in practice this should be good (don't want to convert everything to float for slower calc)
const EPSILON : i32 = 1;

struct Particle {
    position: vec2f,
    velocity: vec2f,
    color: vec4f
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

// tile caches
var<workgroup> tileA: array<i32, TILE_SIZE>;     // costs - prices slice
var<workgroup> tileB: array<i32, TILE_SIZE>;     // column indices for argmax

// group 0 - params (unchanged)
@group(0) @binding(0) var<storage, read> particles : array<Particle>;
@group(0) @binding(1) var<storage, read> params : Params;

// group 1 - solver (assignments + cost_matrix)
@group(1) @binding(0) var<storage, read_write> assignments : array<i32>; // row → col (-1 = unassigned)

// Pre-computed on CPU
@group(1) @binding(1) var<storage, read> cost_matrix : array<i32>;

// WGSL does not support atomic<f32>.
// TODO: Roundabout way to ensure completmentary slackness - multiply all costs on CPU by value > N
// Auction-specific buffers (all size MAX_SIZE)
@group(1) @binding(2) var<storage, read_write> prices : array<i32, MAX_SIZE>;           // column prices, init to 0 on CPU
@group(1) @binding(3) var<storage, read_write> bid_value : array<atomic<i32>, MAX_SIZE>; // highest bid per column this round
@group(1) @binding(4) var<storage, read_write> bid_from_row : array<atomic<i32>, MAX_SIZE>; // who placed the highest bid
// shouldn't need the indices here

var<workgroup> match_count : atomic<i32>;

fn cost(i : u32, j : u32) -> i32 {
    return cost_matrix[i * params.size + j];
}

// bidding phase - each unassigned row scans all columns and submits a bid
@compute @workgroup_size(TILE_SIZE)
fn auctionBiddingPhase(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tid = local_id.x;
    let row = workgroup_id.x; // one WG per row
    let size = params.size;

    if (assignments[row] != -1) {
        // already assigned. Skip
        return;
    }
    var best_profit : i32 = -INT_MAX;
    var best_col : i32 = -1;
    var second_best : i32 = -INT_MAX;

    // if assignments[row] == -1, then it is unassigned
    for (var tileStart = 0u; tileStart < size; tileStart += TILE_SIZE) {
        // load tile of (cost - price) and column index
        if (assignments[row] == -1) {
            let col = tileStart + tid;
            if (col < size) {
                let profit = cost(row, col) - i32(prices[col]);
                tileA[tid] = profit;
                tileB[tid] = i32(col);
            } else {
                tileA[tid] = -INT_MAX;
                tileB[tid] = -1;
            }
        }

//        workgroupBarrier();

        if (assignments[row] == -1) {
            // local parallel reduction for max + argmax + second max
            for (var k = 0u; k < TILE_SIZE; k++) {
                let p = tileA[k];
                if (p > best_profit) {
                    second_best = best_profit;
                    best_profit = p;
                    best_col = tileB[k];
                } else if (p > second_best) {
                    second_best = p;
                }
            }
        }

//        workgroupBarrier();
    }

    let bid_amount: i32 = best_profit - second_best + EPSILON;

    // submit bid to best column w/ atomics
    if (best_col != -1) {
        let j = u32(best_col);
        // atomic max on bid value
        let old = atomicMax(&bid_value[j], bid_amount);

        // PROBLEM: atomic store not immune to TOCTOU - race condition: B > A, both > old, B writes first, A overwrites B
        // TOCTOU window between conditional and atomic store
        // don't see an immediate way for CAS to fix issue - argmax
        // don't need to address this actually - as long as we're improving bid every iteration
        if (bid_amount > old) {
            // new highest bid: bid_amount
            // store argmax_{row} bid
            atomicStore(&bid_from_row[j], i32(row));
        }
    }
}

// Update/Assignment Phase (one thread per column)
// Each column accepts highest bidder, updates price + assignment
@compute @workgroup_size(TILE_SIZE)
fn auctionUpdatePhase(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tid = local_id.x;
    let size = params.size;

    if (tid == 0u && workgroup_id.x == 0u) {
        // exec only once
        atomicStore(&match_count, 0);
    }
//    workgroupBarrier();

    for (var tileStart = 0u; tileStart < size; tileStart += TILE_SIZE) {
        let col = tileStart + tid;
        if (col >= size) {continue;}

        var highest_bid : i32 = -INT_MAX;
        var bidder_row : i32 = -1;

        if (col < size) {
            highest_bid = i32(atomicLoad(&bid_value[col]));
            bidder_row = i32(atomicLoad(&bid_from_row[col]));
        }

//        workgroupBarrier();

        if (col < size && highest_bid > i32(prices[col])) {
            // not critical section - no writes to reads
            // accept the bid
            let old_owner = assignments[col];
            assignments[col] = bidder_row;

            // update price
            prices[col] = highest_bid;

            if (bidder_row != -1) {
                assignments[bidder_row] = i32(col);
            }

            // un-assign old owner (if any)
            if (old_owner != -1 && old_owner != bidder_row) {
                assignments[old_owner] = -1;
            }
        }

        // not sure if possible/efficient to workgroupBarrier then have only first wg + tid atomic store
        // reset bids for next round
        if (col < size) {
            atomicStore(&bid_value[col], -INT_MAX);
            atomicStore(&bid_from_row[col], -1);
        }

        if (col < size && assignments[col] != -1) {
            atomicAdd(&match_count, 1);
        }

//        workgroupBarrier();
    }
}

@compute @workgroup_size(TILE_SIZE)
fn auctionInit(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tid = local_id.x;
    let size = params.size;

    for (var tileStart = 0u; tileStart < size; tileStart += TILE_SIZE) {
        let idx = tileStart + tid;
        if (idx < size) {
            assignments[idx] = -1;
            prices[idx] = 0;
            atomicStore(&bid_value[idx], -INT_MAX);
            atomicStore(&bid_from_row[idx], -1);
        }
        workgroupBarrier();
    }
}