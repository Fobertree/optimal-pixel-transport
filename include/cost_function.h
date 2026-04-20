//
// Created by Alexander Liu on 3/20/26.
//

#ifndef OPTIMALPIXELTRANSPORT_COST_FUNCTION_H
#define OPTIMALPIXELTRANSPORT_COST_FUNCTION_H

#include <Eigen/Dense>

#include "particle.h"

// templated compile-time functor factory design built on traits

static int DEBUG_ID = 0;

enum class COST_TYPE {
    ZERO,
    RGB,
    RGB_DIST_HYBRID,
    RGB_DIST_INT_HYBRID
};
// sinkhorn -> have struct save ineternal data - or convert to class

// functors
struct Zero_Cost {
    // control
    float operator()(const Particle &p1, const Particle &p2) {
        return 0;
    };
};

struct RGB_Cost {
    float operator()(const Particle &p1, const Particle &p2) {
        auto c1 = p1.getColor();
        auto c2 = p2.getColor();

        float dr = c1[0] - c2[0];
        float dg = c1[1] - c2[1];
        float db = c1[2] - c2[2];

        return dr * dr + dg * dg + db * db;
    }
};

struct Dist_Cost {
    float operator()(const Particle &p1, const Particle &p2) {
        auto p1_pos = p1.getPos();
        auto p2_pos = p2.getPos();

        float dx = p1_pos[0] - p2_pos[0];
        float dy = p1_pos[1] - p2_pos[1];

        return dx * dx + dy * dy;
    }
};

struct RGB_Dist_Hybrid_Cost {
    float operator()(const Particle &p1, const Particle &p2) const {
        float rgb_cost = RGB_Cost()(p1, p2) * 255;
        float dist_cost = Dist_Cost()(p1, p2); // tiebreaker

        if (DEBUG_ID < 10) {
            std::cout << std::format("RGB: {}, DIST: {}\n", rgb_cost, dist_cost);
            DEBUG_ID++;
        }

        return (rgb_weight * rgb_cost + (1 - rgb_weight) * dist_cost);
    }

    float rgb_weight = 0.99;
};

struct RGB_Dist_Hybrid_Dist_Cost {
    using T = int64_t;

    T operator()(const Particle &p1, const Particle &p2) const {
        T rgb_cost = static_cast<T>(RGB_Cost()(p1, p2) * 500);
        T dist_cost = static_cast<T>(Dist_Cost()(p1, p2) * 50); // tiebreaker

        double cost = rgb_weight * (double) rgb_cost + (1. - rgb_weight) * (double) dist_cost;
        T scaled_cost = static_cast<T>(cost * int_factor);

        if (DEBUG_ID < 0) {
            std::cout << std::format("Orig: {}, Cost: {}\n", cost, scaled_cost);
            DEBUG_ID++;
        }

        if (scaled_cost < 0) {
            std::cout << "ERROR::" << scaled_cost << std::endl;
            throw std::runtime_error("Bad cost, negative");
        }

        return scaled_cost;
    }

    int int_factor = 1e5;
    float rgb_weight = 0.9;
};

// traits
template<COST_TYPE T>
struct CostFunctionTraits;

template<>
struct CostFunctionTraits<COST_TYPE::ZERO> {
    using type = Zero_Cost;
};

template<>
struct CostFunctionTraits<COST_TYPE::RGB> {
    using type = RGB_Cost;
};

template<>
struct CostFunctionTraits<COST_TYPE::RGB_DIST_HYBRID> {
    using type = RGB_Dist_Hybrid_Cost;
};

template<>
struct CostFunctionTraits<COST_TYPE::RGB_DIST_INT_HYBRID> {
    using type = RGB_Dist_Hybrid_Dist_Cost;
};

// factory function
template<COST_TYPE T>
auto get_cost_function() {
    return typename CostFunctionTraits<T>::type{};
}

template<COST_TYPE COST_T, typename T>
std::vector<T> get_cost_buffer(ParticleBuffer &src_buf, ParticleBuffer &target_buf) {
    assert(src_buf.length() == target_buf.length());
    size_t n = src_buf.length();
    auto cost = get_cost_function<COST_T>();
    std::vector<T> out;
    out.reserve(n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            auto p1 = src_buf.getParticle(i);
            auto p2 = target_buf.getParticle(j);
            out.push_back(cost(*p1, *p2));
        }
    }

    return out;
}

#endif //OPTIMALPIXELTRANSPORT_COST_FUNCTION_H
