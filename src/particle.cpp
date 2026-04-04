//
// Created by Alexander Liu on 2/26/26.
//

#include "particle.h"
#include "consts.h"
#include <iostream>

int Particle::NEXT_ID_ = 0;

constexpr double COR = 0.5; // coefficient of restitution

void Particle::setPos(float x, float y) {
    this->pos_ = {x, y};
}

void Particle::stepPos(float dt) {
    // numerical integrator
    // Euler
    // this->pos_.x_ += this->velo_.x_ * dt;
    // this->pos_.y_ += this->velo_.y_ * dt;
//    printf("%d pos: (%f, %f), velo: (%f, %f)\n", this->id_, this->pos_.x_, this->pos_.y_, this->velo_.x_,
//           this->velo_.y_);

    // RK4
    const static float coeff = dt / 6.0;

    // velocity is initial k1
    // TODO: generalize this to include acceleration, this is only for case for x' is v,
    // Without acceleration, RK4 is the same as Euler
    Vec2<float> k1 = velo_;
    Vec2<float> k2 = velo_;
    Vec2<float> k3 = velo_;
    Vec2<float> k4 = velo_;

    pos_ += (k1 + k2 * 2 + k3 * 2 + k4) * coeff;
//    pos_.y_ += 1e-4 * dt;
//    this->velo_ += Vec2(0.f, -10.f) * dt;

    // TODO: Verlet - O(n^4) precision, n = dt
    // Want to replace RK4 with Verlet because Verlet conserves energy
}

void Particle::setVelo(const std::array<float, 2> &velo) {
    velo_.x_ = velo[0];
    velo_.y_ = velo[1];
}

void Particle::addVelo(const std::array<float, 2> &velo, double dt) {
    velo_.x_ += velo[0];
    velo_.y_ += velo[1];
}

double Particle::dist(const Particle &other) const {
    double dx = this->pos_.x_ - other.pos_.x_;
    double dy = this->pos_.y_ - other.pos_.y_;
    return dx * dx + dy * dy; // ret w/o sqrt as optimization
}

void Particle::collisionImpulse(Particle *other) {
    float dx = this->pos_.x_ - other->pos_.x_;
    float dy = this->pos_.y_ - other->pos_.y_;
    float distSq = dx * dx + dy * dy;

    float combinedRad = this->radius_ + other->radius_;

    if (distSq >= combinedRad * combinedRad) return; // no collision

    float dist = std::sqrt(distSq);
    if (dist < EPSILON) return; // avoid division by zero

    float nx = dx / dist;
    float ny = dy / dist;

    // relative velocity
    float rx = velo_.x_ - other->velo_.x_;
    float ry = velo_.y_ - other->velo_.y_;

    float veloAlongNormal = nx * rx + ny * ry; // dot normal vector
    if (veloAlongNormal > 0) return; // moving away, return

    // compute impulse J
    float j = -(1.0f + COR) * veloAlongNormal;
    j /= (1.0f / mass_) + (1.0 / other->mass_);

    float impulseX = j * nx;
    float impulseY = j * ny;

    // these 2 LOC: Euler - replace if using verlet
    velo_.x_ += impulseX / mass_;
    velo_.y_ += impulseY / mass_;

    other->velo_.x_ -= impulseX / other->mass_;
    other->velo_.y_ -= impulseY / other->mass_;

    // position correction
    float penetration = combinedRad - dist;
    float correction = penetration / (mass_ + other->mass_);

    pos_ += {nx * correction * other->mass_,
             ny * correction * other->mass_};

    other->pos_ -= {nx * correction * mass_,
                    ny * correction * mass_};

    // TODO: verlet integration - drop the velocity, or use RK4
}

std::array<float, 2> Particle::getPos() const {
    return pos_.get();
}

void Particle::applyBoundary() {
    if ((pos_.x_ < -1 && velo_.x_ < 0) || (pos_.x_ > 1 && velo_.x_ > 0)) velo_.x_ *= -1;
    if ((pos_.y_ < -1 && velo_.y_ < 0) || (pos_.y_ > 1 && velo_.y_ > 0)) velo_.y_ *= -1;
}

//inline const ParticleCPU &&Particle::getParticleCPU() const {
//    return std::move(ParticleCPU(pos_.x_, pos_.y_));
//}
