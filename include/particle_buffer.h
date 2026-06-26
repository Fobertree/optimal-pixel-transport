//
// Created by Alexander Liu on 2/19/26.
//

#ifndef OPTIMALPIXELTRANSPORT_PARTICLE_BUFFER_H
#define OPTIMALPIXELTRANSPORT_PARTICLE_BUFFER_H

#include <particle.h>
#include <vector>
#include <iterator>
#include <array>
#include <memory>
#include <ranges>

#include "particle.h"
#include "image.h"

namespace {

[[nodiscard]] std::array<float, 2> gridToSimPos(float pi, float pj) {
    return {(pj - .5f) * 1.5f, (pi - .5f) * 1.5f};
}

} // namespace

// Lightweight loaders for GPU upload — decode image once, sample grid, free pixels.
[[nodiscard]] inline std::vector<ParticleCPU> loadSourceParticlesFromImage(
        const std::string &path, int gridW, int gridH) {
    Image image(path);
    std::vector<ParticleCPU> out;
    out.reserve(static_cast<size_t>(gridW) * gridH);
    for (int i = 0; i < gridH; i++) {
        for (int j = 0; j < gridW; j++) {
            const float pi = static_cast<float>(i) / static_cast<float>(gridH);
            const float pj = static_cast<float>(j) / static_cast<float>(gridW);
            const auto rgb = image.rgb_interpolate(pi, pj);
            const auto pos = gridToSimPos(pi, pj);
            out.emplace_back(pos[0], pos[1], std::array<float, 4>{rgb[0], rgb[1], rgb[2], 1});
        }
    }
    return out;
}

[[nodiscard]] inline std::vector<TargetParticleCPU> loadTargetParticlesFromImage(
        const std::string &path, int gridW, int gridH) {
    Image image(path);
    std::vector<TargetParticleCPU> out;
    out.reserve(static_cast<size_t>(gridW) * gridH);
    for (int i = 0; i < gridH; i++) {
        for (int j = 0; j < gridW; j++) {
            const float pi = static_cast<float>(i) / static_cast<float>(gridH);
            const float pj = static_cast<float>(j) / static_cast<float>(gridW);
            const auto rgb = image.rgb_interpolate(pi, pj);
            const auto pos = gridToSimPos(pi, pj);
            out.emplace_back(pos[0], pos[1], std::array<float, 4>{rgb[0], rgb[1], rgb[2], 1});
        }
    }
    return out;
}

/*
 * Wrapper around VBO
 * Also contains physics - collision, Particle attr update
 */

// TODO: read image into particle buffer
class ParticleBuffer {
public:
    ParticleBuffer() = default;

    ParticleBuffer(const std::string &path, int pWidth, int pHeight) {
        Image image = Image(path);

        for (int i = 0; i < pHeight; i++) {
            for (int j = 0; j < pWidth; j++) {
                // can be easily optimized
                float pi = static_cast<float>(i) / static_cast<float>(pHeight);
                float pj = static_cast<float>(j) / static_cast<float>(pWidth);

                auto rgb = image.rgb_interpolate(pi, pj);

                pi = (pi - .5f) * 1.5f;
                pj = (pj - .5f) * 1.5f;
                // TODO: fix spaghetti
                this->pushParticle(std::array<double, 2>{pj, pi}, std::array<float, 4>{rgb[0], rgb[1], rgb[2], 1});
            }
        }
    }

    void reserve(int n) {
        buf_.reserve(n);
        particleCPUBuffer_.reserve(n);
    }

    [[nodiscard]] size_t length() const { return buf_.size(); }

    // TODO: replace this with [] operator overload
    Particle *getParticle(int i) {
        if (i >= length())
            throw std::out_of_range("Particle::get - Out of range");
        return buf_.at(i).get();
    }

    template<typename... Args>
    void pushParticle(Args &&... args) {
        auto p = Particle(std::forward<Args>(args)...);
        auto particle = std::make_unique<Particle>(std::forward<Args>(args)...);
        particleCPUBuffer_.push_back(particle->getParticleCPU());
        buf_.push_back(std::move(particle));
    }

//    void iterate();

    void refreshParticleCPUBuffer() {
        // this function is stupid, but idk how to pass subparts of Particle by reference yet
        // maybe needs a redesign for such a refactor
        // PROPOSED CHANGE - absorb everything into just having the ParticleCPU directly as a member struct
        // delete pos_
//        std::cout << length() << std::endl;
        for (int i = 0; i < length(); i++) {
            ParticleCPU updatedParticleCPU = buf_[i]->getParticleCPU();
            particleCPUBuffer_[i] = updatedParticleCPU;
        }
    }

    void printDebug() {
        std::cout << "PRINTING PARTICLE BUFFER\n";
        for (auto &particle: buf_) {
            std::cout << particle->getInfo() << " - ";
        }
        std::cout << "\n";
    }

    [[nodiscard]] const std::vector<ParticleCPU> &getParticleCPUBuffer() const {
        return particleCPUBuffer_;
    }

    [[nodiscard]] const std::vector<TargetParticleCPU> getTargetParticleCPUBuffer() const {
        size_t N = buf_.size();
        std::vector<TargetParticleCPU> res;
        res.reserve(N);
        for (auto &particle: buf_) {
            auto pos = particle->getPos();
            auto color = particle->getColor();
            res.emplace_back(pos[0], pos[1], color);
        }
        return res;
    }

    // don't need to pop particles for out use-case
    std::vector<std::unique_ptr<Particle>>::iterator begin() {
        return buf_.begin();
    }

    std::vector<std::unique_ptr<Particle>>::iterator end() {
        return buf_.end();
    }

private:
    std::vector<std::unique_ptr<Particle>> buf_;
    std::vector<ParticleCPU> particleCPUBuffer_;
};

// TODO
//std::ostream &operator<<(std::ostream &os, const ParticleBuffer &particleBuffer) {
//    os << "OK";
//    return os;
//}

#endif //OPTIMALPIXELTRANSPORT_PARTICLE_BUFFER_H
