//
// Created by Alexander Liu on 2/20/26.
//

#ifndef OPTIMALPIXELTRANSPORT_CONSTS_H
#define OPTIMALPIXELTRANSPORT_CONSTS_H

#include <cstdint>

#define ERROR_SCOPE (true)

constexpr uint32_t kWidth = 512;
constexpr uint32_t kHeight = 512;

constexpr int MAX_CPU_PARTICLES = 10000; // for bind group

constexpr double DT = 5e-1;
constexpr float EPSILON = 1e-9; // prevent collision blowup

#endif //OPTIMALPIXELTRANSPORT_CONSTS_H
