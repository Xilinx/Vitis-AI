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

#ifndef XF_DATAANALYTICS_MATH_HELPER_HPP
#define XF_DATAANALYTICS_MATH_HELPER_HPP

#include "hls_math.h"

namespace xf {
namespace data_analytics {
namespace internal {
namespace m {

// abs
template <typename T>
inline T abs(T a) {
    return hls::abs(a);
}

inline double abs(double a) {
#if defined(__aarch64__) || defined(__arm__)
    return std::abs(a);
#else
    return hls::abs(a);
#endif
}

inline float abs(float a) {
#if defined(__aarch64__) || defined(__arm__)
    return std::abs(a);
#else
    return hls::abs(a);
#endif
}

// fabs
template <typename T>
inline T fabs(T a) {
    return hls::fabs(a);
}

inline double fabs(double a) {
#if defined(__aarch64__) || defined(__arm__)
    return std::fabs(a);
#else
    return hls::fabs(a);
#endif
}

inline float fabs(float a) {
#if defined(__aarch64__) || defined(__arm__)
    return std::fabs(a);
#else
    return hls::fabs(a);
#endif
}

// sqrt
template <typename T>
inline T sqrt(T a) {
    return hls::sqrt(a);
}

inline double sqrt(double a) {
#if defined(__aarch64__) || defined(__arm__)
    return std::sqrt(a);
#else
    return hls::sqrt(a);
#endif
}

inline float sqrt(float a) {
#if defined(__aarch64__) || defined(__arm__)
    return std::sqrt(a);
#else
    return hls::sqrt(a);
#endif
}

// exp
template <typename T>
inline T exp(T a) {
    return hls::exp(a);
}

inline double exp(double a) {
#if defined(__aarch64__) || defined(__arm__)
    return std::exp(a);
#else
    return hls::exp(a);
#endif
}

inline float exp(float a) {
#if defined(__aarch64__) || defined(__arm__)
    return std::exp(a);
#else
    return hls::exp(a);
#endif
}

// log
template <typename T>
inline T log(T a) {
    return hls::log(a);
}

inline double log(double a) {
#if defined(__aarch64__) || defined(__arm__)
    return std::log(a);
#else
    return hls::log(a);
#endif
}

inline float log(float a) {
#if defined(__aarch64__) || defined(__arm__)
    return std::log(a);
#else
    return hls::log(a);
#endif
}

} // m
} // internal
} // data_analytics
} // xf

#endif
