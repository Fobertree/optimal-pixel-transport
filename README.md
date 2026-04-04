# Optimal Pixel Transport

WIP

Inspired by Obamify

## Build from Source

Emcc toolchain

# TODO

- Explore parallelizing Hungarian algorithm
- Run `exert_impulse()` on every iteration to give illusion of real-time. Add flags for both
- Code refactors (general TODOs)
- LAPJV (Jonker-Volgenant) algorithm
- Find whether Sinkhorn can be used at all here
- Gale Shapley would require N^2logN preference precompute - very easily parallelizable so maybe applicable still?
- Explore Genetic Algorithm
- Voronoi tesselation shader instead of particle
- Explore migration to particle-based fluid sim
- Absurdly dense cost matrix. Wonder if there's some way/transformation to induce sparsity without ruining results