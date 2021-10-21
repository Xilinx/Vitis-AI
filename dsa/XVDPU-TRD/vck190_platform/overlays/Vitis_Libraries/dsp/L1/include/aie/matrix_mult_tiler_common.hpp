/*
 * Copyright 2021 Xilinx, Inc.
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
#ifndef _DSPLIB_MATRIX_MULT_TILER_COMMON_HPP_
#define _DSPLIB_MATRIX_MULT_TILER_COMMON_HPP_

#include <adf.h>

#ifndef __NEW_WINDOW_H__
#define __NEW_WINDOW_H__ 1
#endif
// if we use 1kb registers -> aie api uses 2x512b registers for 1024b so we need this for QoR
#ifndef __AIE_API_USE_NATIVE_1024B_VECTOR__
#define __AIE_API_USE_NATIVE_1024B_VECTOR__
#endif
#include "aie_api/aie_adf.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace blas {
namespace matrix_mult {

namespace aie = ::aie;

// struct to hold offsets for shuffle intrinsic.
struct loHi {
    unsigned int lo = 0;
    unsigned int hi = 0;
    unsigned int square = 0x3210; // default to no permute.
};

// We might need to take a const reference to offsets to avoid implicit copy.
template <typename T_D, unsigned Elems>
inline aie::vector<T_D, Elems> doShuffle(aie::vector<T_D, Elems> data, const unsigned start, const loHi offsets) {
    // static_assert(false, "Unsupported vectorSize / type combo");#
    printf("\nERROR: shouldn't be here\n");
    return data;
}
template <>
inline aie::vector<cint32, 8> doShuffle(aie::vector<cint32, 8> data, const unsigned start, const loHi offsets) {
    // don't need offsets hi
    return shuffle8(data, start, offsets.lo);
}
template <>
inline aie::vector<int32, 16> doShuffle(aie::vector<int32, 16> data, const unsigned start, const loHi offsets) {
    return shuffle16(data, start, offsets.lo, offsets.hi);
}
template <>
inline aie::vector<cint16, 16> doShuffle(aie::vector<cint16, 16> data, const unsigned start, const loHi offsets) {
    return shuffle16(data, start, offsets.lo, offsets.hi);
}
template <>
inline aie::vector<int16, 32> doShuffle(aie::vector<int16, 32> data, const unsigned start, const loHi offsets) {
    // we don't use square permutes (no usecase for it right now - thankfully)
    // This only works for N=4
    return shuffle32(data, start, offsets.lo, offsets.hi, offsets.square);
    // For N=2, we should rework makeShuffleOffsets
    //  If M=4 and N=2:
    //    lo=0x39313830 and high=0x3B333A32
    // if M=2 and N=2:
    //    lo=0x33323130 and high=0x3B3A3938
    // Basically If N=2; result = ((offsets.lo/hi & 0x0F0F0F0F) | 0x30303030)
}
template <>
inline aie::vector<float, 16> doShuffle(aie::vector<float, 16> data, const unsigned start, const loHi offsets) {
    // we don't use square permutes (no usecase for it right now - thankfully)
    // This only works for N=4
    return fpshuffle16(data, start, offsets.lo, offsets.hi);
}
template <>
inline aie::vector<cfloat, 8> doShuffle(aie::vector<cfloat, 8> data, const unsigned start, const loHi offsets) {
    // we don't use square permutes (no usecase for it right now - thankfully)
    // This only works for N=4
    return fpshuffle8(data, start, offsets.lo);
}

// shuffle8()  //cint32
// shuffle16() //int32 cint16
// shuffle32() //int16
/* -------int16 example ----------
  first full buffer
  idx
  0:   0  1  // 32b chunks
  1:   2  3
  2:   4  5
  3:   6  7
  4:  16 17
  5:  18 19
  6:  20 21
  7:  22 23
  8:  32 33
  9:  34 35
  A:  36 37
  B:  38 39
  C:  48 49
  D:  50 51
  E:  52 53
  F:  54 55


  lo= 0xD951C840
  hi= 0xFB73EA62
  square= 0x3210 (default - no permute)
  produces

  0:   0  1  // 32b chunks
  4:  16 17
  8:  32 33
  C:  48 49
  1:   2  3
  5:  18 19
  9:  34 35
  D:  50 51
  2:   4  5
  6:  20 21
  A:  36 37
  E:  52 53
  3:   6  7
  7:  22 23
  B:  38 39
  F:  54 55

   ----------------------------*/
/*
Coloumn Mjaor storage
  0 16 32 48 64 80 96 112 128 144 160 176 192 208 224 240
  1 17 33 ..                                          241

// Will be loaded like this
  0 16 32 48 1 17 33 49 2 18 34 50 3 19 35 51
  4 20 36 52 5 21 37 53 6 22 38 54 7 23 39 55
  ..
                                  15 31 47 63
  64
// Then shuffled to this.
  0 1 2 3 16 17 18 19 32 .. 50 51
  lo 0xD951C840  hi 0xFB73EA62
*/

// copied from AIE_API, had issues with namespaces / scope.
template <typename T, unsigned Elems>
void myprint(const aie::vector<T, Elems>& v, bool nl = false, const char* prefix = nullptr) {
    if (prefix) printf("%s", prefix);

    using vector_type = aie::vector<T, Elems>;

    for (unsigned i = 0; i < Elems; ++i) {
        T e = v[i];

        if
            constexpr(vector_type::is_complex()) {
                if
                    constexpr(vector_type::is_floating_point()) printf("%f %f, ", (float)e.real, (float)e.imag);
                else
                    printf("%d %d, ", (int)e.real, (int)e.imag);
            }
        else {
            if
                constexpr(vector_type::is_floating_point()) printf("%f ", (float)e);
            else if
                constexpr(!vector_type::is_signed()) printf("%u ", (unsigned)e);
            else
                printf("%d ", (int)e);
        }
    }

    if (nl) printf("\n");
}
}
}
}
}
}

#endif