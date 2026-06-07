# Optimal Pixel Transport

WIP

Inspired by Obamify

## Build from Source

Emcc toolchain

# TODO

- Code refactors (general TODOs)
- Voronoi tesselation shader instead of particle

# Notes

WebGPU spec officially defaults to 8 storage buffers for compute to guarantee portability to older hardware

- Refactor is compressing relevant buffers into structs (PITA)
- Defaulting to 10

AoS and SoA