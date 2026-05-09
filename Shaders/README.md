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

Render (BG0)

- params
- particles

BG0

- params
- particles

BG1 (solver only)

- cost_matrix
- auction specific

BG2 (assignments only)

- assignments

BG3 (physics only - could try divide into two based on how "tightly" they are coupled with rendering loop)

- Simulation-specific