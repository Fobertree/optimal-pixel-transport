//
// Created by Alexander Liu on 3/11/26.
//

#include "particle_buffer.h"
#include "consts.h"

// main loop
// TODO: move main loop to SolverBase class
//void ParticleBuffer::iterate() {
//    // accumulate collisions
//    ParticlePhysics::computeCollisions(this);
//
//    // apply boundaries
//    ParticlePhysics::applyBoundaries(this);
//
//    // stepPos on each particle
//    for (auto &particle: buf_) {
//        particle->stepPos(DT);
//    }
//
//    // debug print
////    printDebug();
//
//    // temp: refresh particle
//    // TODO: maybe this can be done via some sort of view?
//    refreshParticleCPUBuffer();
//}