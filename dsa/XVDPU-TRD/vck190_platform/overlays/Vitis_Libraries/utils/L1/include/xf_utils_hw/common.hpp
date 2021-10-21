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
#ifndef XF_UTILS_HW_COMMON_H
#define XF_UTILS_HW_COMMON_H

/**
 * @file common.hpp
 * @brief Shared logic in utilities library.
 *
 * This file is part of Vitis Utility Library.
 */

#include "xf_utils_hw/types.hpp"

namespace xf {
namespace common {
namespace utils_hw {

/**
 * @struct PowerOf2 common.hpp "xf_utils_hw/common.hpp"
 * @brief Template to calculate power of 2.
 * @tparam _N the power to be calculated.
 */
template <int _N>
struct PowerOf2 {
    /// value of 2^(_N)
    static const unsigned value = (1u << _N);
};

/**
 * @struct GCD common.hpp "xf_utils_hw/common.hpp"
 * @brief Template to calculate Greatest Common Divisor (GCD) of two integers.
 * @tparam _A an integer
 * @tparam _B the other integer
 */
template <int _A, int _B>
struct GCD {
    /// value of GCD of _A and _B
    static const int value = GCD<_B, _A % _B>::value;
};
template <int _A>
struct GCD<_A, 0> {
    static const int value = _A;
};

/**
 * @struct LCM common.hpp "xf_utils_hw/common.hpp"
 * @brief Template to calculate Least Common Multiple(LCM) of two integers.
 * @tparam _A an integer
 * @tparam _B the other integer
 */
template <int _A, int _B>
struct LCM {
    /// value of LCM of _A and _B
    static const int value = _A * _B / GCD<_B, _A % _B>::value;
};

namespace details {

/**
 * @struct UpBound common.hpp "xf_utils_hw/common.hpp"
 * @brief Template to calculate next power of 2, up to 128.
 * @tparam _N the value to be calculated.
 */
template <int _N>
struct UpBound {
    // clang-format off
    static const int value = (_N >=128) ? 128 :
                              _N > 64   ? 128 :
                              _N > 32   ? 64  :
                              _N > 16   ? 32  :
                              _N > 8    ? 16  :
                              _N > 4    ? 8   :
                              _N > 2    ? 4   :
                              _N > 1    ? 2   : 1 ;
    // clang-format on
};

/**
 * @brief Count ones in an integer.
 * @tparam _N width of the integer in number of bits.
 * @param y input
 * @return the number of asserted bits.
 */
inline ap_uint<1> countOnes(ap_uint<1> y) {
#pragma HLS inline

    return y & 0x1;
}

/**
 * @brief Count ones in an integer.
 * @tparam _N width of the integer in number of bits.
 * @param y input
 * @return the number of asserted bits.
 */
inline ap_uint<2> countOnes(ap_uint<2> y) {
#pragma HLS inline

    ap_uint<2> x0 = y;
    ap_uint<2> x1 = (x0 & 0x1) + ((x0 & 0x2) >> 1);
    return x1;
}

/**
 * @brief Count ones in an integer.
 * @tparam _N width of the integer in number of bits.
 * @param y input
 * @return the number of asserted bits.
 */
inline ap_uint<4> countOnes(ap_uint<4> y) {
#pragma HLS inline

    ap_uint<4> x0 = y;
    ap_uint<4> x1 = (x0 & 0x5) + ((x0 & 0xa) >> 1);
    ap_uint<4> x2 = (x1 & 0x3) + ((x1 & 0xc) >> 2);
    return x2;
}

/**
 * @brief Count ones in an integer.
 * @tparam _N width of the integer in number of bits.
 * @param y input
 * @return the number of asserted bits.
 */
inline ap_uint<8> countOnes(ap_uint<8> y) {
#pragma HLS inline

    ap_uint<8> x0 = y;
    ap_uint<8> x1 = (x0 & 0x55) + ((x0 & 0xaa) >> 1);
    ap_uint<8> x2 = (x1 & 0x33) + ((x1 & 0xcc) >> 2);
    ap_uint<8> x3 = (x2 & 0x0f) + ((x2 & 0xf0) >> 4);
    return x3;
}

/**
 * @brief Count ones in an integer.
 * @tparam _N width of the integer in number of bits.
 * @param y input
 * @return the number of asserted bits.
 */
inline ap_uint<16> countOnes(ap_uint<16> y) {
#pragma HLS inline

    ap_uint<16> x0 = y;
    ap_uint<16> x1 = (x0 & 0x5555) + ((x0 & 0xaaaa) >> 1);
    ap_uint<16> x2 = (x1 & 0x3333) + ((x1 & 0xcccc) >> 2);
    ap_uint<16> x3 = (x2 & 0x0f0f) + ((x2 & 0xf0f0) >> 4);
    ap_uint<16> x4 = (x3 & 0x00ff) + ((x3 & 0xff00) >> 8);
    return x4;
}

/**
 * @brief Count ones in an integer.
 * @tparam _N width of the integer in number of bits.
 * @param y input
 * @return the number of asserted bits.
 */
inline ap_uint<32> countOnes(ap_uint<32> y) {
#pragma HLS inline

    ap_uint<32> x0 = y;
    ap_uint<32> x1 = (x0 & 0x55555555UL) + ((x0 & 0xaaaaaaaaUL) >> 1);
    ap_uint<32> x2 = (x1 & 0x33333333UL) + ((x1 & 0xccccccccUL) >> 2);
    ap_uint<32> x3 = (x2 & 0x0f0f0f0fUL) + ((x2 & 0xf0f0f0f0UL) >> 4);
    ap_uint<32> x4 = (x3 & 0x00ff00ffUL) + ((x3 & 0xff00ff00UL) >> 8);
    ap_uint<32> x5 = (x4 & 0x0000ffffUL) + ((x4 & 0xffff0000UL) >> 16);
    return x5;
}

/**
 * @brief Count ones in an integer.
 * @tparam _N width of the integer in number of bits.
 * @param y input
 * @return the number of asserted bits.
 */
inline ap_uint<64> countOnes(ap_uint<64> y) {
#pragma HLS inline
    ap_uint<128> x0 = y;
    ap_uint<128> x1 = (x0 & 0x5555555555555555ULL) + ((x0 & 0xaaaaaaaaaaaaaaaaULL) >> 1);
    ap_uint<128> x2 = (x1 & 0x3333333333333333ULL) + ((x1 & 0xccccccccccccccccULL) >> 2);
    ap_uint<128> x3 = (x2 & 0x0f0f0f0f0f0f0f0fULL) + ((x2 & 0xf0f0f0f0f0f0f0f0ULL) >> 4);
    ap_uint<128> x4 = (x3 & 0x00ff00ff00ff00ffULL) + ((x3 & 0xff00ff00ff00ff00ULL) >> 8);
    ap_uint<128> x5 = (x4 & 0x0000ffff0000ffffULL) + ((x4 & 0xffff0000ffff0000ULL) >> 16);
    ap_uint<128> x6 = (x5 & 0x00000000ffffffffULL) + ((x5 & 0xffffffff00000000ULL) >> 32);
    return x6;
}

} // details

} // utils_hw
} // common
} // xf

#ifndef __SYNTHESIS__
// for assert function.
#include <cassert>
#define XF_UTILS_HW_ASSERT(b) assert((b))
#else
#define XF_UTILS_HW_ASSERT(b) ((void)0)
#endif

#if __cplusplus >= 201103L
#define XF_UTILS_HW_STATIC_ASSERT(b, m) static_assert((b), m)
#else
#define XF_UTILS_HW_STATIC_ASSERT(b, m) XF_UTILS_HW_ASSERT((b) && (m))
#endif

#define XF_UTILS_HW_MACRO_QUOTE(s) #s
#define XF_UTILS_HW_MACRO_STR(s) XF_UTILS_HW_MACRO_QUOTE(s)

#if !defined(__SYNTHESIS__) && XF_UTILS_HW_DEBUG == 1
#define XF_UTILS_HW_PRINT(...)               \
    do {                                     \
        fprintf(stderr, "TX: " __VA_ARGS__); \
        fprintf(stderr, "\n");               \
    } while (0)

#else
#define XF_UTILS_HW_PRINT(...) ((void)(0))
#endif

#endif // XF_UTILS_HW_COMMON_H
