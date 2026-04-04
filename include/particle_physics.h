//
// Created by Alexander Liu on 2/26/26.
//

#ifndef OPTIMALPIXELTRANSPORT_PARTICLE_PHYSICS_H
#define OPTIMALPIXELTRANSPORT_PARTICLE_PHYSICS_H

#include <vector>

#include "particle_buffer.h"
#include "particle.h"


namespace ParticlePhysics {
    void computeCollisions(ParticleBuffer *buffer) {
        // TODO: optimize this. This is very naive just to get smth running
        for (int i = 0; i < buffer->length(); i++) {
            for (int j = i + 1; j < buffer->length(); j++) {
                Particle *p1 = buffer->getParticle(i);
                Particle *p2 = buffer->getParticle(j);

                p1->collisionImpulse(p2);
            }
        }
    }

    void applyBoundaries(ParticleBuffer *buffer) {
        for (auto &particle: *buffer) {
            particle->applyBoundary();
        }
    }
    // TODO: migrate derivative update routines here (ex. RK4)
}

#endif //OPTIMALPIXELTRANSPORT_PARTICLE_PHYSICS_H
