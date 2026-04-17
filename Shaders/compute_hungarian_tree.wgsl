// Parallelized Hungarian - Alternating Tree
// Augmenting Tree should be faster than classical for problems as large as ours
// fluid sim shader will handle physics + fragment

const INT_MAX : i32 = 2147483647;
const TILE_SIZE : u32 = 256u; // workgroup length
const MAX_SIZE : u32 = 25000;

struct Particle {
    position: vec2f,
    color: vec4f
};

struct Input {

};

struct Output {
    // to pipe to next shader in pipeline
};

struct Params {
// TODO replace direct size pass, set params as @binding(2)
    size : u32
}

// tile caches
var<workgroup> tileA: array<u32, TILE_SIZE>;
var<workgroup> tileB: array<u32, TILE_SIZE>;

@group(0) @binding(0) var<storage, read> particles : array<Particle>;

// Pre-computed on CPU, since very in-expensive and I like the templating
@group(0) @binding(1) var<storage, read> cost_matrix : array<i32>;
@group(0) @binding(2) var<storage, read> params : Params;

//var<storage, read_write> tile : array<i32, TILE_SIZE>;

// auxilliary
var<storage, read_write> Dr : array<i32, MAX_SIZE>;
var<storage, read_write> Dc : array<i32, MAX_SIZE>;

var<storage, read_write> At : array<i32, MAX_SIZE>;

// forward pass (alg 4)
var<storage, read_write> Ac : array<i32, MAX_SIZE>;
var<storage, read_write> Vr : array<i32, MAX_SIZE>;
var<storage, read_write> Vc : array<i32, MAX_SIZE>;
var<storage, read_write> slack : array<i32, MAX_SIZE>;
var<storage, read_write> Pr : array<i32, MAX_SIZE>;
var<storage, read_write> Pc : array<i32, MAX_SIZE>;

// assignment vectors for augmentation pass
var<storage, read_write> Ar : array<i32, MAX_SIZE>;
var<storage, read_write> Sr : array<i32, MAX_SIZE>;
var<storage, read_write> Sc : array<i32, MAX_SIZE>;

// Minimum uncovered slack
var<workgroup> theta : i32;

var<workgroup> match_count : atomic<i32>;

// Frontier arrays
var<storage, read_write> Fin : array<bool, MAX_SIZE>;
var<storage, read_write> Fout : array<bool, MAX_SIZE>;
var<storage, read_write> Faug : array<bool, MAX_SIZE>;

fn cost(i : u32, j: u32) -> i32 {
    return cost_matrix[i * size + j];
}

fn getMinUncoveredSlack() {
    theta = INT_MAX;
    let size = params.size;

    for (var j = 0; j < size; j++) {
        if (slack[j] > 0 && slack[j] < theta) {
            theta = slack[j];
        }
    }
}

fn reduction(
        local_id : vec3<u32>,
        workgroup_id : vec3<u32>
    ) {
    // thread id
    let tid = local_id.x;
    let size = params.size;

    // one workgroup per row
    let row = workgroup_id.x;
    var row_min = INT_MAX;

    // TODO: parallel min-reduction
    // tileA - Dr - write to tile first to rm one storage buffer read

    /* ROW REDUCTION */
    // parallelized column iteration
    for (var tileStart = 0u; tileStart < size; tileStart += TILE_SIZE) {
        let col = tileStart + tid;

        if (col < size) {
            tileA[col] = cost(row, col);
        } else {
            tileA[col] = INT_MAX;
        }

        Dr[col] = tileA[col];

        workgroupBarrier();
    }

    /* COLUMN ITERATION */
    // assume row = col
    let col = workgroup_id.x;
    for (var tileStart = 0u; tileStart < size; tileStart += TILE_SIZE) {
        let rrow = tileStart + tid;

        if (rrow < size) {
            Dc[rrow] = cost(rrow, col) - tileA[rrow];
        } else {
            Dc[rrow] = INT_MAX;
        }

        workgroupBarrier();
    }
}

fn checkOptimality(local_id : vec3<u32>) {
    let tid = local_id.x;
    let size = params.size;

    for (var tileStart = 0u; tileStart < size; tileStart += TILE_SIZE) {
        let i = tileStart + tid;

        // cannot short-circuit due to joining
        if (i < size && Ar[i] != -1) { // row is assigned
            Vr[i] = 1; // update row cover
            // TODO: probably better to just have single-threaded check instead of atomic
            // try both, but this probably sucks and might as well be an escape hatch
            atomicAdd(&match_count, 1);
        }

        workgroupBarrier();
    }
}

fn forwardPass(local_id : vec3<u32>) {
    let tid = local_id.x;
    let size = params.size;

    // assign all unassigned row indices to Fin
    for (var tileStart = 0; tileStart < size; tileStart += size) {
        let i = tid + tileStart;
        if (i < size) {
            if (Ar[i] == -1) {
                Fin[i] = true;
            } else {
                Fin[i] = false;
            }
        }
    }

    workgroupBarrier();

    // Algorithm 4 - parallel bfs
    for (var tileStart = 0; tileStart < size; tileStart += size) {
        let j = tid + tileStart;
        if (j < size) {
            if (Vc[j] == 0) {
                // TODO: do we need more tiling for nested for loop?
                for (var i = 0; i < size; i++) {
                // TODO Fin[i] in tileA
                // TODO Dr tileB, Dc in tileC
                // loading Ac into tile might be overkill
                    if (Fin[i]) {
                        // 'active', i.e., neither visited NOR matched
                        if (slack[j] > cost(i,j) - Dr[i] - Dc[j]) {
                            slack[j] = cost(i,j) - Dr[i] - Dc[j];
                            Pc[j] = i;
                        }
                        let inew = Ac[j];
                        if (slack[j] == 0) {
                            if (inew != -1) {
                                Pr[inew] = j;
                                // Not critical section: Ac[i] should be unique for assigned columns
                                // Shouldn't be critical section based on workgroupBarrier but good to check
                                Vr[inew] = 0;
                                Vc[j] = 1;
                                Fout[inew] = true;
                            }
                            else {
                                Frev[j] = true;
                            }
                        }
                    }
                    workgroupBarrier();
                }
            } // End Vc[j] == 0
            Fout[i] = false;
        }

        workgroupBarrier();
    }

    for (var tileStart = 0; tileStart < size; tileStart += size) {
        let i = tid + tileStart;
        if (i < size) {
            // TODO: this is kind of ugly - Consider copyBufferToBuffer outside the shader (main.cpp)
            // move Fout to Fin
            Fin[i] = Fout[i];

            // reset Fout for next iteration
            Fout[i] = false;
        }
        workgroupBarrier();
    }
}

// Handling F_{whatever} by boolean arr, since dynamic data structures unsupported

fn reversePass(local_id : vec3<u32>) {
    let tid = local_id.x;
    let size = params.size;

    // TODO: push active row indices to frontier array F_in
    for (var tileStart = 0; tileStart < size; tileStart += TILE_SIZE) {
        let j = tid + tileStart;
        if (j < start) {
            rcur = -1;
            ccur = j;
            while (ccur != -1) {
                Sc[ccur] = rcur;
                Sr[rcur] = ccur;
                ccur = Pr[rcur];
            }
            // mark rcur as "augment"
            Faug[rcur] = true;
        }
        workgroupBarrier();
    }
}

fn augmentationPass(local_id : vec3<u32>) {
    // Alternating Tree Variant, not classical variant
    let tid = local_id.x;
    let size = params.size;
    for (var tileStart = 0u; tileStart < size; tileStart += TILE_SIZE) {
        let i = tid + tileStart;
        if (i < size) {
            var rcur = i;
            var ccur = -1;
            while (rcur != -1) {
                ccur = Sr[rcur];
                Ar[rcur] = ccur;
                Ac[ccur] = rcur;
                rcur = Sc[ccur];
            }
        }
        workgroupBarrier();
    }
}

fn dualUpdate(
            local_id : vec3<u32>,
            global_invocation_id : vec3<u32>
            ) {
    if (global_invocation_id.x == 0u) {
        getMinUncoveredSlack(); // get theta
    }
    let halfTheta : f32 = theta * .5;
    let tid = local_id.x;
    let size = params.size;
    for (let tileStart = 0u; tileStart < size; tileStart += TILE_SIZE) {
        let k = tid + tileStart;
        if (k < size) {
            if (Vr[k] == 0) {
                Dr[k] += halfTheta;
            } else {
                Dr[k] -= halfTheta;
            }

            if (Vc[k] == 0) {
                Dc[k] += halfTheta;
            } else {
                Dc[k] -= halfTheta;
            }

            if (slack[j] > 0) {
                slack[j] -= theta;
                if (slack[j] == 0) {
                    // TODO: Mark Pc[j] as "active"
                    Fin[Pc[j]] = 1;
                }
            }
        }
        workgroupBarrier();
    }

    // after this, return to augmenting path search step
    // we want to pipe real-time results here to physics
    // TODO: add a buffer to write tmp matching to, bindable to next compute shader in pipeline
}

// main
@compute @workgroup_size(TILE_SIZE)
fn computeMain(
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id : vec3<u32>,
        @builtin(global_invocation_id) global_invocation_id : vec3<u32>
    ) {
    let size = params.size;
    reduction(local_id, workgroup_id);

    while (true) {
        checkOptimality(local_id);

        if (match_count == size) {
            // solved
            break;
        }
        // forward pass is main bottleneck in alternating tree
        forwardPass(local_id);
        reversePass(local_id);
        augmentationPass(local_id);
        dualUpdate(local_id, global_invocation_id);
    }
}

