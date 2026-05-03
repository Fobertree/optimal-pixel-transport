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
// https://pysph.readthedocs.io/en/1.0a1/reference/kernels.html
// TODO: check all the kernel impls - probably wrong
namespace SmoothingKernel {
    // (x,y) pos vec
    float cubicSplineKernel(const ParticleCPU &p1, const ParticleCPU &p2, float h) noexcept {
        // h is smoothing parameter/radius
        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;
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

    float cubicSplineKernel(float delta_q, float h) noexcept {
        // h is smoothing parameter/radius
        float q = delta_q / h;

        // std::pow doesn't throw (returns nan)
        // check via std::isnan not necessary since will never hit numbers that overflow
        if (q < 1) {
            return 2 / 3. - q * q + 0.5 * std::pow(q, 3);
        } else if (q < 2) {
            return 1 / 6. * std::pow(2 - q, 3);
        }
        return 0;
    }

    std::pair<float, float> gradientCubicSplineKernel(const ParticleCPU &p1, const ParticleCPU &p2, float h) noexcept {
        // analytic gradient
        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;
        float q = std::sqrt(dx * dx + dy * dy) / h; // r/h
        float dW_dq;

        if (q < 1) {
            dW_dq = -3 * q + 2.25 * q * q;
        } else if (q < 2) {
            dW_dq = -0.75f * std::pow(2.0f - q, 2.0f);
        } else {
            return {0.0f, 0.0f};
        }

        float scale = dW_dq / (h * q);
        return {scale * dx, scale * dy};
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
        constraints_ = std::vector<float>(n_);
        lambdas_ = std::vector<float>(n_);
        binStart_ = std::vector<int>(NUM_BINS_, -1);
        binEnd_ = std::vector<int>(NUM_BINS_, -1); // not necessary, just for readability
        binIDs_ = std::vector<int>(n_, -1);
        deltaPosX_ = std::vector<float>(n_, -1);
        deltaPosY_ = std::vector<float>(n_, -1);
        veloX_ = std::vector<float>(n_);
        veloY_ = std::vector<float>(n_);
        vorticityForceX_ = std::vector<float>(n_);
        vorticityForceY_ = std::vector<float>(n_);
        veloNewX_ = std::vector<float>(n_);
        veloNewY_ = std::vector<float>(n_);
        omega_ = std::vector<float>(n_);
    }

    void iterate();

private:
    [[nodiscard]] std::vector<int> getSortedParticleIndices(std::span<ParticleCPU> particles);

    void getLambdas(const std::vector<int> &sortedParticleIndices);

    void getDeltaPos(const std::vector<int> &sortedParticleIndices);

    void updatePositionStar();

    void updateVelocities();

    void vorticityConfinementAndViscosity(const std::vector<int> &sortedParticleIndices);

    void updatePosition();


    int hashCoords(float xi, float yi, int table_sz) noexcept;

    // TODO: figure out what this should be set to. Too low - unintended collisions + slow
    // Also looking into a way to identify or pass assertions that detect collision vulnerabilities
    constexpr static int NUM_BINS_ = 100000;
    constexpr static float CELL_SIZE_ = 1 / NUM_BINS_;
    constexpr static float EPS_ = 1e-8;
    constexpr static float DT_ = 1;

    std::vector<ParticleCPU> particles_;
    std::vector<float> constraints_;
    std::vector<float> lambdas_;
    std::vector<float> rho_;
    std::vector<float> deltaPosX_;
    std::vector<float> deltaPosY_;
    // x^*
    std::vector<float> posStarX_;
    std::vector<float> posStarY_;
    // v
    std::vector<float> veloX_;
    std::vector<float> veloY_;

    std::vector<float> vorticityForceX_;
    std::vector<float> vorticityForceY_;

    std::vector<float> veloNewX_;
    std::vector<float> veloNewY_;

    std::vector<float> omega_;
    // TODO: maybe an auxilliary vector that stores first particle in vector within sorted bin
    std::vector<int> binIDs_;
    std::vector<int> binStart_; // aux vector storing first particle idx in vector within sorted bin
    int solverIterations_;
    size_t n_;
    float rho_0_; // resting density

    std::vector<int> binEnd_;    // index inclusive
};

#endif //OPTIMALPIXELTRANSPORT_PBF_H
