//
// Created by Alexander Liu on 4/3/26.
//

#ifndef OPTIMALPIXELTRANSPORT_PBF_H
#define OPTIMALPIXELTRANSPORT_PBF_H

// prob redundancy with include guard, used to gate main.cpp pbf
#define CPP_PBF

#include <span>
#include <numeric>

#include "particle_buffer.h"
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
    }

    void iterate();

private:
    [[nodiscard]] std::vector<int> getSortedParticleIndices(std::span<ParticleCPU> particles);

    [[nodiscard]] float getDensity() noexcept;

    int hashCoords(float xi, float yi, int table_sz) noexcept;

    std::vector<ParticleCPU> particles_;
    std::vector<float> constraintGradients_;
    std::vector<float> lambdas_;
    std::vector<float> rho_;
    int solverIterations_;
    size_t n_;
    float rho_0_; // resting density
};

#endif //OPTIMALPIXELTRANSPORT_PBF_H
