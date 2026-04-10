# Optimal Pixel Transport

WIP

Inspired by Obamify

## Build from Source

Emcc toolchain

# TODO

- Explore parallelizing Hungarian algorithm
    - CPU-bound: thread-pool + std::thread
    - GPU-bound: no native emcc support for OpenMP, must build compute shaders (seems a bit too much for now)
- Run `exert_impulse()` on every iteration to give illusion of real-time. Add flags for both
- Code refactors (general TODOs)
- LAPJV (Jonker-Volgenant) algorithm
- Gale Shapley would require N^2logN preference precompute - very easily parallelizable so maybe applicable still?
    - Problem: not LAP algorithm - only proposer optimality
- Explore Genetic Algorithm
- Voronoi tesselation shader instead of particle
- Explore migration to particle-based fluid sim
- Absurdly dense cost matrix. Wonder if there's some way/transformation to induce sparsity without ruining results
- PBF and physics as Compute Shader