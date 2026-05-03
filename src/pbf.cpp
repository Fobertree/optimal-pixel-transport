//
// Created by Alexander Liu on 4/29/26.
//

#include "pbf.h"

// this file is basically garbage temp code that I will refactor later
void PBF::iterate() {
    // skip apply ext forces

    // find neighboring particles
    // Argsort to match future compute shader logic - argsort buffer of structs, dk tradeoff in spatial locality
    auto particleIndices = getSortedParticleIndices(particles_);

    // core
    int iters = 0;
    while (iters++ < solverIterations_) {
        // TODO: calculate lambda_i
        getLambdas(particleIndices);
        getDeltaPos(particleIndices);
        // since we don't have solids, can skip collision detection
        // SPH already pushes particles away via density constraints, so no extra logic needed
        updatePositions();
        // TODO: vorticity confinement
    }
}

void PBF::getLambdas(const std::vector<int> &sortedParticleIndices) { // TODO: impl
    for (int pid = 0; pid < sortedParticleIndices.size(); pid++) { // i
        int binID = binIDs_[pid];
        // lambda

        // Density
        // given neighbors p_1 ... p_n, rho_i
        // \rho_i = \sum_j m_j W(p_i - p_j, h)
        // summation over neighbors, with W as smoothing kernel
        float total_density = 0;
        int start = binStart_[binID];
        int end = binEnd_[binID];
        auto particle = particles_[sortedParticleIndices[pid]];

        for (int npid = start; npid <= end; npid++) {
            auto nei_particle = particles_[sortedParticleIndices[npid]];
            auto kernel_output = SmoothingKernel::cubicSplineKernel(particle, nei_particle, 0.5); // TODO: adjust h

            // density
            // assume equal unit mass
            total_density += kernel_output;
        }
        rho_[pid] = total_density; // rho vec might be unnecessary

        // Constraint + Gradient Constraint
        float lambda_denom = 0;
        float recip_rho_0 = 1 / rho_0_; // C_i
        constraints_[pid] = (total_density * recip_rho_0) - 1;
        for (int k_npid = start; k_npid <= end; k_npid++) { // k
            if (k_npid == pid) { // k == i
                for (int j_npid = start; j_npid <= end; j_npid++) { // j
                    auto nei_particle = particles_[sortedParticleIndices[j_npid]];
                    auto [gradient_kernel_x, gradient_kernel_y] = SmoothingKernel::gradientCubicSplineKernel(particle,
                                                                                                             nei_particle,
                                                                                                             0.5); // TODO: adjust h
                    float sq_norm = gradient_kernel_x * gradient_kernel_x + gradient_kernel_y * gradient_kernel_y;
                    lambda_denom += recip_rho_0 * recip_rho_0 * sq_norm;
                }
            }
                // this can be else if because i = j yields 0
            else if (binIDs_[k_npid] == binIDs_[pid]) { // k = j, i.e. k is a neighbor of i
                auto nei_particle = particles_[sortedParticleIndices[k_npid]];
                auto [gradient_kernel_x, gradient_kernel_y] = SmoothingKernel::gradientCubicSplineKernel(particle,
                                                                                                         nei_particle,
                                                                                                         0.5); // TODO: adjust h
                float sq_norm = gradient_kernel_x * gradient_kernel_x + gradient_kernel_y * gradient_kernel_y;
                lambda_denom += recip_rho_0 * recip_rho_0 * sq_norm;
            }
        }

        lambdas_[pid] = constraints_[pid] / (lambda_denom + EPS_);
    }
}

void PBF::getDeltaPos(const std::vector<int> &sortedParticleIndices) {
    float rho_0_recip = 1 / rho_0_;

    for (int i = 0; i < n_; i++) {
        float delta_p = 0;
        int i_pid = sortedParticleIndices[i];
        auto particle = particles_[i_pid];

        int start = binStart_[i];
        int end = binEnd_[i];
        float totalX = 0;
        float totalY = 0;

        for (int j = start; j <= end; j++) {
            if (i == j) continue; // 0 kernel output
            int j_pid = sortedParticleIndices[j];
            auto nei_particle = particles_[j_pid];
            auto [gradient_kernel_x, gradient_kernel_y] = SmoothingKernel::gradientCubicSplineKernel(particle,
                                                                                                     nei_particle,
                                                                                                     0.5); // TODO: change h

            float lambda_i = lambdas_[i_pid];
            float lambda_j = lambdas_[j_pid];

            totalX += (lambda_i + lambda_j) * gradient_kernel_x;
            totalY += (lambda_i + lambda_j) * gradient_kernel_y;
        }
        deltaPosX_[i_pid] = rho_0_recip * totalX;
        deltaPosY_[i_pid] = rho_0_recip * totalY;
    }
}

void PBF::updatePositions() {
    // do not need argsort indices. Being direct for efficiency
    for (int i = 0; i < n_; i++) {
        particles_[i].x += deltaPosX_[i];
        particles_[i].y += deltaPosY_[i];
    }
}

int PBF::hashCoords(float x, float y, int table_sz) noexcept {
    // https://matthias-research.github.io/pages/tenMinutePhysics/11-hashing.pdf
    // Not sure if Z-order/Morton or Hilbert curves are better here
    // To handle float w/ Z-order multiply with large integer then floor/round to int
    // Using this 10-minute physics hash function for simplicity
    int xi = std::floor(x / CELL_SIZE_);
    int yi = std::floor(y / CELL_SIZE_);
    int h = (xi * 92837111) ^ (yi * 689287499);
    return std::abs(h) % table_sz;
}

std::vector<int> PBF::getSortedParticleIndices(std::span<ParticleCPU> particles) {

    // get the indices, then in the actual logic we check direct neighbors with index equality on 27 point lattice
    // bin into neighborhood via sort
    // counting sort particleIDs
    // should benchmark vs radix sort later
    size_t sz = particles.size();
    std::vector<int> hashes(sz);
    std::vector<int> count(NUM_BINS_ + 1);
    std::vector<int> sortedParticleIDs(NUM_BINS_ + 1); // output

    // populate hashes
    std::transform(particles.begin(), particles.end(), hashes.begin(), [&](const ParticleCPU &particle) {
        return hashCoords(particle.x, particle.y, NUM_BINS_);
    });

    for (int i = 0; i < sz; i++) {
        int key = hashes[i];
        count[key]++;
    }

    for (int i = 1; i <= NUM_BINS_; i++) {
        // prefix
        count[i] = count[i] + count[i - 1];
    }

    for (int i = sz - 1; sz >= 0; sz--) {
        int key = hashes[i];
        count[key]--;
        sortedParticleIDs[count[key]] = i;
        binIDs_[count[key]] = key;

        // store bin starts + ends in indices for easy neighborhood access
        if (binStart_[key] == -1 || count[key] < binStart_[key]) {
            binStart_[key] = count[key];
        }

        if (binEnd_[key] == -1 || count[key] > binEnd_[key]) {
            binEnd_[key] = i;
        }
    }

    return sortedParticleIDs;
}