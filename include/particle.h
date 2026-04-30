//
// Created by Alexander Liu on 2/19/26.
//

#ifndef OPTIMALPIXELTRANSPORT_PARTICLE_H
#define OPTIMALPIXELTRANSPORT_PARTICLE_H

// https://www.cs.cmu.edu/~baraff/pbm/particles.pdf
// wgsl, ubo

#include <array>
#include <iostream>

template<typename T>
struct Vec2 {
    Vec2() = default;

    Vec2(T x, T y) : x_(x), y_(y) {};

    Vec2(const Vec2 &other) : x_(other.x_), y_(other.y_) {};

    // TODO: pass by ref if possible. I think we would have to convert internal to std array
    [[nodiscard]] std::array<T, 2> get() const {
        return {x_, y_};
    }

    T dot(const Vec2 &other) {
        return x_ * other.x_ + y_ * other.y_;
    }

    void operator+=(const Vec2 &other) {
        x_ += other.x_;
        y_ += other.y_;
    }

    void operator-=(const Vec2 &other) {
        x_ -= other.x_;
        y_ -= other.y_;
    }

    Vec2<T> operator+(const Vec2 &other) {
        return {x_ + other.x_, y_ + other.y_};
    }

    Vec2<T> operator*(T scalar) {
        return {x_ * scalar, y_ * scalar};
    }

    Vec2<T> operator*(Vec2<T> other) {
        return {x_ * other.x_, y_ * other.y_};
    }

    T x_, y_;
};

struct ParticleCPU {
    // have to use float since double only works for const
    ParticleCPU(float x, float y) : x(x), y(y) {};

    ParticleCPU(float x, float y, std::array<float, 4> color) : x(x), y(y), r(color[0]), g(color[1]), b(color[2]),
                                                                a(color[3]) {};
    float x, y;
    float pad[2]{}; // need padding for vec4f
    // vec4f must start at a 16-byte aligned address
    float r{}, g{}, b{}, a{};
//    float color[4];
};


// https://www.cg.tuwien.ac.at/research/publications/2023/PETER-2023-PSW/PETER-2023-PSW-.pdf
// https://www.youtube.com/watch?v=wy57zjpaCYc
class Particle {
public:
    Particle() = default;

    // TODO: clean up messy constructors
    explicit Particle(std::array<double, 2> pos) : pos_(pos[0], pos[1]), id_(NEXT_ID_++) {};

    explicit Particle(std::array<double, 2> pos, std::array<float, 4> color) : pos_(pos[0], pos[1]), id_(NEXT_ID_++),
                                                                               color_(color) {};

    // TODO: add color+position constructor

    void setPos(float x, float y);

    // TODO: replace below with verlet, or change RK4 to include acceleration (otherwise equiv to Euler)
    void stepPos(float dt); // based on velocity

    void setVelo(const std::array<float, 2> &velo);

    void addVelo(const std::array<float, 2> &velo, double dt);

    [[nodiscard]] double dist(const Particle &other) const; // make private?

    void collisionImpulse(Particle *other);

    void applyBoundary();

    // TODO: FIX
    [[nodiscard]] std::array<float, 2> getPos() const;

    [[nodiscard]] int getID() const { return id_; }

    [[nodiscard]] inline ParticleCPU getParticleCPU() const {
        // return lvalue for RVO
        // TODO: refactor to float for Vec2
        return {pos_.x_, pos_.y_, color_};
    };

    std::string getInfo() {
        return std::format("ID: {}, Pos: ({:.2f}, {:.2f}), Velo: ({:.2f}, {:.2f})", id_, pos_.x_, pos_.y_, velo_.x_,
                           velo_.y_);
    }

    [[nodiscard]] const std::array<float, 4> &getColor() const { return color_; }

private:
    static int NEXT_ID_;
    int id_{-1};                    // for debugging
    std::array<float, 4> color_{1, 0, 0, 0};  // rgba color (to be interpolated by shader)
    // physics
    float mass_{100};
    float radius_{0.005};
    Vec2<float> pos_{};            // position
    Vec2<float> velo_{};           // velocity
    Vec2<float> accel_{};          // force accumulator
};

#endif //OPTIMALPIXELTRANSPORT_PARTICLE_H
