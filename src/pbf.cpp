//
// Created by Alexander Liu on 4/29/26.
//

#include "pbf.h"

// this file is basically garbage temp code that I will refactor later
void PBF::iterate(std::span<int> assignments) {
    applyAssignmentForces(assignments);

    // find neighboring particles
    // Argsort to match future compute shader logic - argsort buffer of structs, dk tradeoff in spatial locality
    // TODO: clean up indexing everywhere or drop argsort for direct sort - argsort + direct indexing is bug-prone
    auto particleIndices = getSortedParticleIndices(particles_);

    // core
    int iters = 0;
    while (iters++ < solverIterations_) {
        getLambdas(particleIndices);
        getDeltaPos(particleIndices);
        // since we don't have solids, can skip collision detection
        // SPH already pushes particles away via density constraints, so no extra logic needed for collision
        updatePositionStar();
        updateVelocities();
        vorticityConfinementAndViscosity(particleIndices);
        updatePosition();
    }
}

void PBF::applyAssignmentForces(std::span<int> assignments) {
    for (int i = 0; i < n_; i++) {
        auto particle = particles_[i];
        auto assigned_particle = particles_[assignments[i]];

        float dx = assigned_particle.x - particle.x;
        float dy = assigned_particle.y - particle.y;

        posStarX_[i] += dx * DT_;
        posStarY_[i] += dy * DT_;
    }
}

void PBF::getLambdas(const std::vector<int> &sortedParticleIndices) {
    for (int i = 0; i < sortedParticleIndices.size(); i++) { // i
        int pid = sortedParticleIndices[i];
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
            auto kernel_output = SmoothingKernel::cubicSplineKernel(particle, nei_particle, H_);

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
                                                                                                             H_);
                    float sq_norm = gradient_kernel_x * gradient_kernel_x + gradient_kernel_y * gradient_kernel_y;
                    lambda_denom += recip_rho_0 * recip_rho_0 * sq_norm;
                }
            }
                // this can be else if because i = j yields 0
            else if (binIDs_[k_npid] == binIDs_[pid]) { // k = j, i.e. k is a neighbor of i
                auto nei_particle = particles_[sortedParticleIndices[k_npid]];
                auto [gradient_kernel_x, gradient_kernel_y] = SmoothingKernel::gradientCubicSplineKernel(particle,
                                                                                                         nei_particle,
                                                                                                         H_);
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
        int i_pid = sortedParticleIndices[i];
        auto particle = particles_[i_pid];

        int binID = binIDs_[i_pid];
        int start = binStart_[binID];
        int end = binEnd_[binID];

        float totalX = 0;
        float totalY = 0;

        for (int j = start; j <= end; j++) {
            if (i == j) continue; // 0 kernel output
            int j_pid = sortedParticleIndices[j];
            auto nei_particle = particles_[j_pid];
            auto [gradient_kernel_x, gradient_kernel_y] = SmoothingKernel::gradientCubicSplineKernel(particle,
                                                                                                     nei_particle,
                                                                                                     H_);

            float lambda_i = lambdas_[i_pid];
            float lambda_j = lambdas_[j_pid];

            // tensile instability
            float k = 0.1;
            float delta_q = 0.2 * H_;
            int n = 4;
            float kernel_output = SmoothingKernel::cubicSplineKernel(particle, nei_particle, H_);
            float W_delta = SmoothingKernel::cubicSplineKernel(delta_q, H_);
            float s_corr = -k * std::pow(kernel_output / W_delta, n);

            totalX += (lambda_i + lambda_j + s_corr) * gradient_kernel_x;
            totalY += (lambda_i + lambda_j + s_corr) * gradient_kernel_y;
        }
        deltaPosX_[i_pid] = rho_0_recip * totalX;
        deltaPosY_[i_pid] = rho_0_recip * totalY;
    }
}

void PBF::updatePositionStar() {
    // do not need argsort indices. Being direct for efficiency
    for (int i = 0; i < n_; i++) {
        posStarX_[i] = particles_[i].x + deltaPosX_[i];
        posStarY_[i] = particles_[i].y + deltaPosY_[i];
    }
}

void PBF::updateVelocities() {
    for (int i = 0; i < n_; i++) {
        float x_ix = particles_[i].x;
        float x_iy = particles_[i].y;
        veloX_[i] = 1 / DT_ * (posStarX_[i] - x_ix);
        veloY_[i] = 1 / DT_ * (posStarY_[i] - x_iy);
    }
}

void PBF::vorticityConfinementAndViscosity(const std::vector<int> &sortedParticleIndices) {
    for (int i = 0; i < n_; i++) {
        int pid = sortedParticleIndices[i];
        int binID = binIDs_[pid];
        int start = binStart_[binID];
        int end = binEnd_[binID];
        auto particle = particles_[pid];
        float omega_i = 0;
        const float C = 0.01;
        float viscosity_sum = 0;
        for (int pid_j = start; pid_j <= end; pid_j++) {
            auto nei_particle = particles_[pid_j];

            float v_ijx = veloX_[pid_j] - veloX_[pid];
            float v_ijy = veloY_[pid_j] - veloY_[pid];
            auto [grad_kernel_x, grad_kernel_y] = SmoothingKernel::gradientCubicSplineKernel(particle, nei_particle,
                                                                                             H_);

            // a x b = x1y2 - y1x2
            omega_i += v_ijx * grad_kernel_y - v_ijy * grad_kernel_x;

            // can also do XSPH viscosity in same loop
            viscosity_sum += v_ijx * grad_kernel_x + v_ijy * grad_kernel_y;
        }
        veloNewX_[pid] = veloX_[pid] + C * viscosity_sum;
        veloNewY_[pid] = veloY_[pid] + C * viscosity_sum;
        omega_[pid] = omega_i;

        float eta_x = 0;
        float eta_y = 0;
        // Need another loop since we needed omega to be precomputed for eta gradient
        // \eta_i = \nabla_i |\omega|_i = \sum_j m_j \frac{|\omega|_j - |\omega|_i}{\rho_j} \nabla_i W_{ij}
        for (int pid_j = start; pid_j <= end; pid_j++) {
            auto nei_particle = particles_[pid_j];
            auto [grad_kernel_x, grad_kernel_y] = SmoothingKernel::gradientCubicSplineKernel(particle, nei_particle,
                                                                                             H_);
            eta_x += ((std::abs(omega_[pid]) - std::abs(omega_[pid_j])) / rho_[pid_j]) * grad_kernel_x;
            eta_y += ((std::abs(omega_[pid]) - std::abs(omega_[pid_j])) / rho_[pid_j]) * grad_kernel_y;
        }
        // norm to unit
        float norm_divisor = std::sqrt(eta_x * eta_x + eta_y * eta_y);
        eta_x /= norm_divisor;
        eta_y /= norm_divisor;

        // cross product with scalar on RHS here intuitively becomes 90 degrees CCW by right-hand-rule
        // perp vector: (-Ny, Nx)
        float f_vorticity_x = EPS_ * omega_[pid] * (-eta_y);
        float f_vorticity_y = EPS_ * omega_[pid] * eta_x;

        // direct euler step force into velocity
        // TODO: veloNewX may be redundant with posStar. Should be able to /= DT then add vorticity + viscosity contributions
        veloNewX_[pid] += DT_ * f_vorticity_x;
        veloNewY_[pid] += DT_ * f_vorticity_y;
    }
}

void PBF::updatePosition() {
    for (int i = 0; i < n_; i++) {
        particles_[i].x += veloNewX_[i] * DT_;
        particles_[i].y += veloNewY_[i] * DT_;
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