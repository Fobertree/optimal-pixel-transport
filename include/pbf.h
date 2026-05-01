//
// Created by Alexander Liu on 4/3/26.
//

#ifndef OPTIMALPIXELTRANSPORT_PBF_H
#define OPTIMALPIXELTRANSPORT_PBF_H

// prob redundancy with include guard, used to gate main.cpp pbf
#define CPP_PBF

#include <span>
#include <numeric>
#include <cmath>

#include "particle_buffer.h"

// TODO: refactor to factory if want to test other kernels
// https://interactivecomputergraphics.github.io/physics-simulation/examples/sph_kernel.html
namespace SmoothingKernel {
    // (x,y) pos vec
    float cubicSplineKernel(float dx, float dy, float h) noexcept {
        float q = std::sqrt(dx * dx + dy * dy) / h;

        // std::pow doesn't throw (returns nan)
        // check via std::isnan not necessary since will never hit numbers that overflow
        if (q < 1) {
            return 2 / 3. - q * q + 0.5 * std::pow(q, 3);
        } else if (q < 2) {
            return 1 / 6. * std::pow(2 - q, 3);
        }
        return 0;
    }

    float gradientCubicSplineKernel(float dx, float dy, float h) noexcept {
        // derived from analytic solution, unlike gradient of field quantities
        float q = std::sqrt(dx * dx + dy * dy) / h;

        if (q < 1) {
            return -3 * q + 2.25 * q * q;
        } else if (q < 2) {
            return -0.75 * std::pow(2 - q, 2);
        }
        return 0;
    }
}

// store hash table as 1d array

// contains logic to populate buffers
// will gradually migrate logic to wgsl shaders as needed on profiling

class PBF {
public:
    explicit PBF() = default;

    explicit PBF(const std::vector<ParticleCPU> &particles, int solverIterations = 10) : particles_(particles),
                                                                                         solverIterations_(
                                                                                                 solverIterations) {
        n_ = particles.size();
        constraintGradients_ = std::vector<float>(n_);
        lambdas_ = std::vector<float>(n_);
        firstInBin_ = std::vector<int>(NUM_BINS_, -1);
    }

    void iterate();

private:
    [[nodiscard]] std::vector<int> getSortedParticleIndices(std::span<ParticleCPU> particles);

    [[nodiscard]] float getConstraints() noexcept;

    [[nodiscard]] float getDensities() noexcept;

    int hashCoords(float xi, float yi, int table_sz) noexcept;

    constexpr static int NUM_BINS_ = 10.;
    constexpr static float CELL_SIZE_ = 1 / NUM_BINS_;

    std::vector<ParticleCPU> particles_;
    std::vector<float> constraintGradients_;
    std::vector<float> lambdas_;
    std::vector<float> rho_;
    // TODO: maybe an auxilliary vector that stores first particle in vector within sorted bin
    std::vector<int> firstInBin_; // aux vector storing first particle idx in vector within sorted bin
    int solverIterations_;
    size_t n_;
    float rho_0_; // resting density
};

#endif //OPTIMALPIXELTRANSPORT_PBF_H
