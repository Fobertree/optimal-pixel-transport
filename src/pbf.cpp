//
// Created by Alexander Liu on 4/29/26.
//

#include "pbf.h"

void PBF::iterate() {
    // skip apply ext forces

    // find neighboring particles
    auto particleIndices = getSortedParticleIndices(particles_);

    // core
    int iters = 0;
    while (iters++ < solverIterations_) {
        for (int i = 0; i < n_; i++) {
            auto particle = particles_[i];
            // get constrain
            lambdas_[i] = -1;
        }
    }
}

float PBF::getDensity() noexcept {
    // given neighbors p_1 ... p_n, rho_i
    // \rho_i = \sum_j m_j W(p_i - p_j, h)
    // summation over neighbors, with W as smoothing kernel
}

int PBF::hashCoords(float xi, float yi, int table_sz) noexcept {
    // https://matthias-research.github.io/pages/tenMinutePhysics/11-hashing.pdf
    // Not sure if Z-order/Morton or Hilbert curves are better here
    // To handle float w/ Z-order multiply with large integer then floor/round to int
    // Using this 10-minute physics hash function for simplicity
    int h = static_cast<int>(xi * 92837111.) ^ static_cast<int>(yi * 689287499.);
    return std::abs(h) % table_sz;
}

std::vector<int> PBF::getSortedParticleIndices(std::span<ParticleCPU> particles) {

    // get the indices, then in the actual logic we check direct neighbors with index equality on 27 point lattice
    // bin into neighborhood via sort
    // counting sort particleIDs
    // should benchmark vs radix sort later
    size_t sz = particles.size();
    const static int NUM_BINS = 10; // k
    std::vector<int> hashes(sz);
    std::vector<int> count(NUM_BINS + 1);
    std::vector<int> sortedParticleIDs(NUM_BINS + 1); // output

    // populate hashes
    std::transform(particles.begin(), particles.end(), hashes.begin(), [&](const ParticleCPU &particle) {
        return hashCoords(particle.x, particle.y, sz);
    });

    for (int i = 0; i < sz; i++) {
        int key = hashes[i];
        count[key]++;
    }

    for (int i = 1; i <= NUM_BINS; i++) {
        // prefix
        count[i] = count[i] + count[i - 1];
    }

    for (int i = sz - 1; sz >= 0; sz--) {
        int key = hashes[i];
        count[key]--;
        sortedParticleIDs[count[key]] = i;
    }

    return sortedParticleIDs;
}