/*
 * Copyright 2021 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file math_helper.hpp
 * @brief enable sw-emu on embedded platforms.
 */

#ifndef XF_SOLVER_MATH_HELPER_HPP
#define XF_SOLVER_MATH_HELPER_HPP

#include "hls_math.h"

namespace xf {
namespace solver {
namespace internal {
namespace m {

// abs
template <typename T>
T abs(T a) {
    return hls::abs(a);
}

double abs(double a) {
#if defined(__aarch64__) || defined(__arm__)
    return std::abs(a);
#else
    return hls::abs(a);
#endif
}

float abs(float a) {
#if defined(__aarch64__) || defined(__arm__)
    return std::abs(a);
#else
    return hls::abs(a);
#endif
}

// fabs
template <typename T>
T fabs(T a) {
    return hls::fabs(a);
}

double fabs(double a) {
#if defined(__aarch64__) || defined(__arm__)
    return std::fabs(a);
#else
    return hls::fabs(a);
#endif
}

float fabs(float a) {
#if defined(__aarch64__) || defined(__arm__)
    return std::fabs(a);
#else
    return hls::fabs(a);
#endif
}

// sqrt
template <typename T>
T sqrt(T a) {
    return hls::sqrt(a);
}

double sqrt(double a) {
#if defined(__aarch64__) || defined(__arm__)
    return std::sqrt(a);
#else
    return hls::sqrt(a);
#endif
}

float sqrt(float a) {
#if defined(__aarch64__) || defined(__arm__)
    return std::sqrt(a);
#else
    return hls::sqrt(a);
#endif
}

} // m
} // internal
} // solver
} // xf

#endif
