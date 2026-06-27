// Bertsekas-Castanon Parallel Hungarian (classical primal-dual, NOT LAPJV)
// https://web.mit.edu/dimitrib/www/Bertsekas_Castanon_Parallel_Hungarian_1993.pdf
//
// Duals: row_u[i] + col_v[j] <= cost(i,j). Slack = cost - row_u - col_v.
// Each augmenting-path step is a parallel column sweep (multi-pass).

const INT_MAX : i32 = 2147483647;
const TILE_SIZE : u32 = 256u;
const IN_Z_FLAG : i32 = 0x40000000;
const META_COUNT : u32 = 6u;

const INT_FACTOR : f32 = 1000.0;
const RGB_SCALE : f32 = 500.0;
const RGB_WEIGHT : f32 = 0.9;
const DIST_SCALE : f32 = 50.0;

override GRID_DIM : u32;
override USE_DISTANCE : i32;

struct Particle {
    position: vec2f,
    velocity: vec2f,
    color: vec4f,
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
    frameCount: u32,
}

@group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(1) var<uniform> params : Params;

@group(1) @binding(0) var<storage, read_write> assignments : array<atomic<i32>>;
@group(1) @binding(1) var<storage, read> target_particles : array<TargetParticle>;
@group(1) @binding(2) var<storage, read_write> col_v : array<i32>;
@group(1) @binding(3) var<storage, read_write> min_to : array<atomic<i32>>;
@group(1) @binding(4) var<storage, read_write> row_u : array<i32>;
@group(1) @binding(5) var<storage, read_write> owner : array<atomic<i32>>;
@group(1) @binding(6) var<storage, read_write> prev_col : array<i32>;

fn cost(i : u32, j : u32) -> i32 {
    let src_color = particles[i].color;
    let tgt = target_particles[j];

    let dr = src_color.r - tgt.color.r;
    let dg = src_color.g - tgt.color.g;
    let db = src_color.b - tgt.color.b;
    let rgb_cost = i32((dr * dr + dg * dg + db * db) * RGB_SCALE);

    var blended = f32(rgb_cost);
    if (USE_DISTANCE != 0) {
        let src_pos = target_particles[i].position;
        let dx = src_pos.x - tgt.position.x;
        let dy = src_pos.y - tgt.position.y;
        let dist_cost = i32((dx * dx + dy * dy) * DIST_SCALE);
        blended = RGB_WEIGHT * f32(rgb_cost) + (1.0 - RGB_WEIGHT) * f32(dist_cost);
    }

    let tiebreak = abs(i32(i) - i32(j));
    return i32(blended * INT_FACTOR) + tiebreak;
}

fn slack(i : u32, j : u32) -> i32 {
    return cost(i, j) - row_u[i] - col_v[j];
}

fn min_val(raw : i32) -> i32 {
    return raw & 0x3FFFFFFF;
}

fn in_z(raw : i32) -> bool {
    return (raw & IN_Z_FLAG) != 0;
}

fn set_val(raw : i32, v : i32) -> i32 {
    return (raw & IN_Z_FLAG) | (v & 0x3FFFFFFF);
}

fn mark_z(raw : i32) -> i32 {
    return raw | IN_Z_FLAG;
}

fn meta_base(sz : u32) -> u32 {
    return sz - META_COUNT;
}

fn meta_get(sz : u32, slot : u32) -> i32 {
    return row_u[meta_base(sz) + slot];
}

fn meta_set(sz : u32, slot : u32, v : i32) {
    row_u[meta_base(sz) + slot] = v;
}

fn col_active(sz : u32, j : u32) -> bool {
    return j < meta_base(sz);
}

@compute @workgroup_size(TILE_SIZE)
fn hungarianInit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let sz = params.size;
    if (i >= sz) {
        return;
    }

    atomicStore(&assignments[i], -1);
    atomicStore(&owner[i], -1);
    row_u[i] = 0;
    col_v[i] = 0;
    atomicStore(&min_to[i], 0);
}

@compute @workgroup_size(TILE_SIZE)
fn hungarianRowReduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let sz = params.size;
    if (i >= sz) {
        return;
    }

    var min_c = INT_MAX;
    for (var j = 0; j < i32(sz); j++) {
        min_c = min(min_c, cost(i, u32(j)));
    }
    row_u[i] = min_c;
}

@compute @workgroup_size(TILE_SIZE)
fn hungarianColReduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let sz = params.size;
    if (j >= sz) {
        return;
    }

    var min_s = INT_MAX;
    for (var i = 0; i < i32(sz); i++) {
        min_s = min(min_s, cost(u32(i), j) - row_u[u32(i)]);
    }
    col_v[j] = min_s;
}

@compute @workgroup_size(TILE_SIZE)
fn hungarianGreedyMatch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let sz = params.size;
    if (i >= sz) {
        return;
    }

    var best_j: i32 = -1;
    var best_s = INT_MAX;
    for (var j = 0; j < i32(meta_base(sz)); j++) {
        let s = slack(i, u32(j));
        if (s < best_s) {
            best_s = s;
            best_j = j;
        }
    }

    if (best_j < 0) {
        return;
    }

    let claimed_col = atomicCompareExchangeWeak(&owner[u32(best_j)], -1, i32(i));
    if (claimed_col.exchanged) {
        let claimed_row = atomicCompareExchangeWeak(&assignments[i], -1, best_j);
        if (!claimed_row.exchanged) {
            atomicStore(&owner[u32(best_j)], -1);
        }
    }
}

@compute @workgroup_size(TILE_SIZE)
fn hungarianClearOwners(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= params.size) {
        return;
    }
    atomicStore(&owner[k], -1);
}

@compute @workgroup_size(TILE_SIZE)
fn hungarianApplyOwners(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let sz = params.size;
    if (i >= sz) {
        return;
    }
    let j = atomicLoad(&assignments[i]);
    if (j >= 0) {
        atomicStore(&owner[u32(j)], i32(i));
    }
}

// meta slots: 0=src row, 1=found, 2=end_col, 3=delta, 4=pivot, 5=saved row_u[meta_base]
@compute @workgroup_size(TILE_SIZE)
fn hungarianAugmentSetup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let sz = params.size;
    let mb = meta_base(sz);
    if (idx >= sz) {
        return;
    }

    if (idx == 0u) {
        meta_set(sz, 0u, -1);
        meta_set(sz, 1u, 0);
        meta_set(sz, 2u, -1);
        meta_set(sz, 3u, INT_MAX);
        meta_set(sz, 4u, -1);
        meta_set(sz, 5u, row_u[mb]);
        for (var r = 0; r < i32(mb); r++) {
            if (atomicLoad(&assignments[u32(r)]) < 0) {
                meta_set(sz, 0u, r);
                break;
            }
        }
    }

    let src = meta_get(sz, 0u);
    if (src < 0) {
        return;
    }

    let usrc = u32(src);
    if (col_active(sz, idx)) {
        atomicStore(&min_to[idx], slack(usrc, idx));
        prev_col[idx] = -1;
    }
}

@compute @workgroup_size(TILE_SIZE)
fn hungarianAugmentFindDelta(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let sz = params.size;
    let mb = meta_base(sz);

    if (j == 0u) {
        atomicStore(&min_to[mb], INT_MAX);
    }

    if (!col_active(sz, j)) {
        return;
    }

    let raw = atomicLoad(&min_to[j]);
    if (!in_z(raw)) {
        atomicMin(&min_to[mb], min_val(raw));
    }
}

@compute @workgroup_size(TILE_SIZE)
fn hungarianAugmentCheckFree(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let sz = params.size;
    let mb = meta_base(sz);

    if (j == 0u) {
        meta_set(sz, 3u, atomicLoad(&min_to[mb]));
    }

    if (!col_active(sz, j)) {
        return;
    }

    if (meta_get(sz, 1u) != 0) {
        return;
    }

    let delta = meta_get(sz, 3u);
    if (delta == INT_MAX) {
        return;
    }

    let raw = atomicLoad(&min_to[j]);
    if (in_z(raw) || min_val(raw) != delta) {
        return;
    }

    if (atomicLoad(&owner[j]) < 0) {
        meta_set(sz, 1u, 1);
        meta_set(sz, 2u, i32(j));
    }
}

@compute @workgroup_size(TILE_SIZE)
fn hungarianAugmentRelax(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let sz = params.size;
    if (!col_active(sz, j)) {
        return;
    }

    if (meta_get(sz, 1u) != 0) {
        return;
    }

    let delta = meta_get(sz, 3u);
    if (delta == INT_MAX) {
        return;
    }

    if (j == 0u) {
        meta_set(sz, 4u, -1);
        for (var c = 0; c < i32(meta_base(sz)); c++) {
            let raw = atomicLoad(&min_to[u32(c)]);
            if (in_z(raw) || min_val(raw) != delta) {
                continue;
            }
            if (atomicLoad(&owner[u32(c)]) >= 0) {
                meta_set(sz, 4u, c);
                break;
            }
        }
    }

    let pivot = meta_get(sz, 4u);
    if (pivot < 0) {
        return;
    }

    let pu = u32(pivot);
    let raw_p = atomicLoad(&min_to[pu]);
    if (!in_z(raw_p)) {
        atomicStore(&min_to[pu], mark_z(raw_p));
    }

    let i1 = u32(atomicLoad(&owner[pu]));
    let h = cost(i1, pu) - col_v[pu] - row_u[i1] - delta;

    let raw_j = atomicLoad(&min_to[j]);
    if (in_z(raw_j)) {
        return;
    }

    let v2 = cost(i1, j) - col_v[j] - row_u[i1] - h;
    if (v2 < min_val(raw_j)) {
        atomicStore(&min_to[j], set_val(raw_j, v2));
        prev_col[j] = pivot;
        if (v2 == delta && atomicLoad(&owner[j]) < 0) {
            meta_set(sz, 1u, 1);
            meta_set(sz, 2u, i32(j));
        }
    }
}

@compute @workgroup_size(TILE_SIZE)
fn hungarianAugmentDualUpdate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let sz = params.size;
    if (!col_active(sz, j)) {
        return;
    }

    if (meta_get(sz, 1u) != 0) {
        return;
    }

    let delta = meta_get(sz, 3u);
    if (delta == INT_MAX) {
        return;
    }

    let raw = atomicLoad(&min_to[j]);
    if (in_z(raw)) {
        col_v[j] += min_val(raw) - delta;
        let matched_row = atomicLoad(&owner[j]);
        if (matched_row >= 0) {
            row_u[u32(matched_row)] += delta;
        }
    }
}

@compute @workgroup_size(TILE_SIZE)
fn hungarianAugmentCommit(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }

    let sz = params.size;
    let mb = meta_base(sz);

    if (meta_get(sz, 1u) == 0) {
        row_u[mb] = meta_get(sz, 5u);
        return;
    }

    let src = meta_get(sz, 0u);
    var tar_cur = meta_get(sz, 2u);

    loop {
        let pcol = prev_col[u32(tar_cur)];
        var new_row: i32;
        if (pcol < 0) {
            new_row = src;
        } else {
            new_row = atomicLoad(&owner[u32(pcol)]);
        }

        atomicStore(&owner[u32(tar_cur)], new_row);
        if (new_row >= 0) {
            atomicStore(&assignments[u32(new_row)], tar_cur);
        }

        if (pcol < 0) {
            break;
        }
        tar_cur = pcol;
    }

    row_u[mb] = meta_get(sz, 5u);
}

@compute @workgroup_size(TILE_SIZE)
fn hungarianRepair(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let sz = params.size;
    let mb = meta_base(sz);
    if (i >= mb) {
        return;
    }

    if (atomicLoad(&assignments[i]) >= 0) {
        return;
    }

    var best_j: i32 = -1;
    var best_cost = INT_MAX;
    for (var j = 0; j < i32(mb); j++) {
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

    let claimed_col = atomicCompareExchangeWeak(&owner[u32(best_j)], -1, i32(i));
    if (!claimed_col.exchanged) {
        return;
    }

    let claimed_row = atomicCompareExchangeWeak(&assignments[i], -1, best_j);
    if (!claimed_row.exchanged) {
        atomicStore(&owner[u32(best_j)], -1);
    }
}