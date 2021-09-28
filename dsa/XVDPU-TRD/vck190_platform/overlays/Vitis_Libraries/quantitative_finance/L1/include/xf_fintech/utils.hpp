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
 * @file utils.hpp
 * @brief This file contains the utilities.
 */
#ifndef HLS_FINTECH_UNTILS_H
#define HLS_FINTECH_UNTILS_H
#include <stdint.h>
#include "hls_math.h"

namespace xf {
namespace fintech {
namespace internal {

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
inline float bitsToFloat(uint32_t in) {
    union {
        uint32_t __I;
        float __F;
    } __T;
    __T.__I = in;
    return __T.__F;
}

inline uint32_t floatToBits(float __V) {
    union {
        uint32_t __I;
        float __F;
    } __T;
    __T.__F = __V;
    return __T.__I;
}

inline double bitsToDouble(uint64_t in) {
    union {
        uint64_t __I;
        double __D;
    } __T;
    __T.__I = in;
    return __T.__D;
}

inline uint64_t doubleToBits(double __V) {
    union {
        uint64_t __I;
        double __D;
    } __T;
    __T.__D = __V;
    return __T.__I;
}
template <typename DT>
inline DT FPTwoAdd(DT in1, DT in2) {
    DT r = 0;
#ifdef DPRAGMA
#pragma HLS resource variable = r core = DAddSub_nodsp
#else
#pragma HLS resource variable = r core = FAddSub_nodsp
#endif
    r = in1 + in2;
    return r;
}
template <typename DT>
inline DT FPTwoSub(DT in1, DT in2) {
    DT r = 0;
#ifdef DPRAGMA
#pragma HLS resource variable = r core = DAddSub_nodsp
#else
#pragma HLS resource variable = r core = FAddSub_nodsp
#endif
    r = in1 - in2;
    return r;
}
template <typename DT>
inline DT FPTwoMul(DT in1, DT in2) {
    DT r = 0;
#ifdef DPRAGMA
#pragma HLS resource variable = r core = DMul_meddsp
#else
#pragma HLS resource variable = r core = FMul_fulldsp
#endif
    r = in1 * in2;
    return r;
}
template <typename DT>
inline DT FPExp(DT in) {
    DT r = 0;
#ifdef DPRAGMA
#pragma HLS resource variable = r core = DExp_meddsp
#else
#pragma HLS resource variable = r core = FExp_meddsp
#endif
#ifndef __SYNTHESIS__
    r = std::exp(in);
#else
    r = hls::exp(in);
#endif

    return r;
}
template <typename DT>
inline DT divide_by_2(DT input) {
    DT result = FPTwoMul(input, (DT)0.5);
    return result;
}

template <typename DT>
inline DT mul_by_2(DT input) {
    DT result = FPTwoMul(input, (DT)2);
    return result;
}

template <typename DT, int N, int M>
struct xf_2D_array {
#ifndef __SYNTHESIS__
    DT* _2d_array;
#else
    DT _2d_array[N][M];
#endif

    xf_2D_array() {
#ifndef __SYNTHESIS__
        _2d_array = new DT[N * M];
#endif
    }
#ifndef __SYNTHESIS__
    ~xf_2D_array() { delete[] _2d_array; }
#endif

    DT read(int n, int m) {
#pragma HLS inline
#ifndef __SYNTHESIS__
        return _2d_array[n * M + m];
#else
        return _2d_array[n][m];
#endif
    }

    void write(DT in, int n, int m) {
#pragma HLS inline
#ifndef __SYNTHESIS__
        _2d_array[n * M + m] = in;
#else
        _2d_array[n][m] = in;
#endif
    }
};

} // internal
} // fintech
} // xf
#endif // ifndef HLS_FINTECH_UNTILS_H
