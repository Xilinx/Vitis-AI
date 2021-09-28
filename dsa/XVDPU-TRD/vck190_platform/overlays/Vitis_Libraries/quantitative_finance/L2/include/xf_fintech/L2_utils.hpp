/*
 * Copyright 2019 Xilinx, Inc.
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
 * @file L2_utils.hpp
 * @brief This file contains the utilities for L2.
 */
#ifndef HLS_FINTECH_L2_UTILS_HPP
#define HLS_FINTECH_L2_UTILS_HPP

// #include "hls_math.h"

// #ifndef __SYNTHESIS__
#include "math.h"
// #endif

namespace xf {
namespace fintech {
namespace internal {

// min & max templates
template <class T>
const T& MAX(const T& a, const T& b) {
    return (a < b) ? b : a;
}

template <class T>
const T& MIN(const T& a, const T& b) {
    return !(b < a) ? a : b;
}

// template specialization for float math functions
template <typename DT>
DT EXP(DT x) {
    return exp(x);
}

template <>
inline float EXP(float x) {
    return expf(x);
}

template <typename DT>
DT SQRT(DT x) {
    return sqrt(x);
}

template <>
inline float SQRT(float x) {
    return sqrtf(x);
}

template <typename DT>
DT LOG(DT x) {
    return log(x);
}

template <>
inline float LOG(float x) {
    return logf(x);
}

template <typename DT>
DT POW(DT x, DT y) {
    return pow(x, y);
}

template <>
inline float POW(float x, float y) {
    return powf(x, y);
}

template <typename DT>
DT ATAN2(DT x, DT y) {
    return atan2(x, y);
}

template <>
inline float ATAN2(float x, float y) {
    return atan2f(x, y);
}

template <typename DT>
DT SIN(DT x) {
    return sin(x);
}

template <>
inline float SIN(float x) {
    return sinf(x);
}

template <typename DT>
DT COS(DT x) {
    return cos(x);
}

template <>
inline float COS(float x) {
    return cosf(x);
}

// complex number arithmetic
template <typename DT>
struct complex_num {
    DT real;
    DT imag;
};

template <typename DT>
struct complex_num<DT> cn_init(DT real, DT imag) {
    struct complex_num<DT> Z;
    Z.real = real;
    Z.imag = imag;
    return Z;
}

template <typename DT>
struct complex_num<DT> cn_add(struct complex_num<DT> Z, struct complex_num<DT> Y) {
    struct complex_num<DT> res;
    res.real = Z.real + Y.real;
    res.imag = Z.imag + Y.imag;
    return res;
}

template <typename DT>
struct complex_num<DT> cn_sub(struct complex_num<DT> Z, struct complex_num<DT> Y) {
    struct complex_num<DT> res;
    res.real = Z.real - Y.real;
    res.imag = Z.imag - Y.imag;
    return res;
}

template <typename DT>
struct complex_num<DT> cn_mul(struct complex_num<DT> Z, struct complex_num<DT> Y) {
    struct complex_num<DT> res;
    res.real = (Z.real * Y.real) - (Z.imag * Y.imag);
    res.imag = (Z.imag * Y.real) + (Z.real * Y.imag);
    return res;
}

template <typename DT>
struct complex_num<DT> cn_scalar_mul(struct complex_num<DT> Z, DT val) {
    struct complex_num<DT> res;
    res.real = Z.real * val;
    res.imag = Z.imag * val;
    return res;
}

template <typename DT>
DT cn_mod(struct complex_num<DT> Z) {
    DT tmp = (Z.real * Z.real) + (Z.imag * Z.imag);
    return SQRT(tmp);
}

template <typename DT>
DT cn_arg(struct complex_num<DT> Z) {
    return ATAN2(Z.imag, Z.real);
}

template <typename DT>
struct complex_num<DT> cn_div(struct complex_num<DT> Z, struct complex_num<DT> Y) {
    struct complex_num<DT> res;
    DT tmp = (Y.real * Y.real) + (Y.imag * Y.imag);
    res.real = (Z.real * Y.real) + (Z.imag * Y.imag);
    res.real /= tmp;
    res.imag = (-Z.real * Y.imag) + (Z.imag * Y.real);
    res.imag /= tmp;
    return res;
}

template <typename DT>
struct complex_num<DT> cn_exp(struct complex_num<DT> Z) {
    struct complex_num<DT> res;
    DT e = EXP(Z.real);
    res.real = e * COS(Z.imag);
    res.imag = e * SIN(Z.imag);
    return res;
}

template <typename DT>
struct complex_num<DT> cn_ln(struct complex_num<DT> Z) {
    struct complex_num<DT> res;
    res.real = 0.5 * LOG((Z.real * Z.real) + (Z.imag * Z.imag));
    res.imag = ATAN2(Z.imag, Z.real);
    return res;
}

template <typename DT>
struct complex_num<DT> cn_sqrt(struct complex_num<DT> Z) {
    struct complex_num<DT> res;
    DT mod = cn_mod(Z);
    DT arg = cn_arg(Z);

    DT sqrt_mod = SQRT(mod);
    res.real = sqrt_mod * COS(arg / 2);
    res.imag = sqrt_mod * SIN(arg / 2);

    /* ensure the "positive" answer is supplied */
    if (res.real < 0.0) {
        res.real = -res.real;
        res.imag = -res.imag;
    }
    return res;
}

template <typename DT>
DT cn_real(struct complex_num<DT> Z) {
    return Z.real;
}

} // namespace internal
} // namespace fintech
} // namespace xf
#endif // ifndef HLS_FINTECH_L2_UTILS_H
