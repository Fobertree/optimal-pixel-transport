# Shaders

Inputs

Solver (BG0, BG1)

- params
- particles
- assignments
- cost_matrix

Physics

- params
- particles
- assignments
- PBF/simulation specific

Want to define BGs in terms of update frequency (whole world of optimization)

Universal (BG0) (Render - fragment/vertex only needs this)

- params
- particles

BG1 (solver only)

- cost_matrix
- auction specific

BG2 (assignments only)

- assignments

BG3 (physics only - could try divide into two based on how "tightly" they are coupled with rendering loop)

- Simulation-specific

BG4 (remainder of radix sort)

- Bin hashes
- Output particles (since not in-place)
    - Arguably better to have a buffer of indices and supply it to render
- Local prefix sum & block sum

// global
wgpu::Buffer particleBuffer;
wgpu::Buffer paramsBuffer;
// solver
wgpu::Buffer assignmentsBuffer;
wgpu::Buffer costBuffer;
// pbf
wgpu::Buffer lambdasBuffer;
wgpu::Buffer deltaPosBuffer;
wgpu::Buffer posStarBuffer;
wgpu::Buffer binStartBuffer;
wgpu::Buffer binCountBuffer;
wgpu::Buffer omegaBuffer;
// radix
wgpu::Buffer localPrefixSumBuffer;
wgpu::Buffer prefixBlockSumBuffer;
wgpu::Buffer auxParticleSortBuffer; // naive auxiliary buffer since radix sort is not in-place, so copy back

Next steps:

- Particle swap buffer
- fragment shader? (current one should work fine, not sure if should try point splatting/voronoi splatting)