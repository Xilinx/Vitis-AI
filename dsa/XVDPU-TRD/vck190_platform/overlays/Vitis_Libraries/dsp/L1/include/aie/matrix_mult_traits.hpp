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
#ifndef _DSPLIB_MATRIX_MULT_TRAITS_HPP_
#define _DSPLIB_MATRIX_MULT_TRAITS_HPP_

namespace xf {
namespace dsp {
namespace aie {
namespace blas {
namespace matrix_mult {
/*
Asymmetrical Interpolation FIR traits.
This file contains sets of overloaded, templatized and specialized templatized functions which
encapsulate properties of the intrinsics used by the main kernal class. Specifically,
this file does not contain any vector types or intrinsics since it is required for construction
and therefore must be suitable for the aie compiler graph-level compilation.
*/

// The following is a set of type-specialized functions which return the number of accumulator registers
// available in the processor. Since these may be 384 or 768 bit registers the number could vary by type.
template <typename TT_DATA_A, typename TT_DATA_B>
unsigned int fnAccRegsMatMult() {
    return 0;
}; // default error trap
template <>
inline constexpr unsigned int fnAccRegsMatMult<int16, int16>() {
    return 4;
};
template <>
inline constexpr unsigned int fnAccRegsMatMult<cint16, int16>() {
    return 4;
};
template <>
inline constexpr unsigned int fnAccRegsMatMult<cint16, cint16>() {
    return 4;
};
template <>
inline constexpr unsigned int fnAccRegsMatMult<int32, int16>() {
    return 4;
};
template <>
inline constexpr unsigned int fnAccRegsMatMult<int32, int32>() {
    return 4;
};
template <>
inline constexpr unsigned int fnAccRegsMatMult<cint32, int16>() {
    return 4;
};
template <>
inline constexpr unsigned int fnAccRegsMatMult<cint32, cint16>() {
    return 4;
};
template <>
inline constexpr unsigned int fnAccRegsMatMult<cint32, int32>() {
    return 4;
};
template <>
inline constexpr unsigned int fnAccRegsMatMult<cint32, cint32>() {
    return 4;
};
template <>
inline constexpr unsigned int fnAccRegsMatMult<float, float>() {
    return 4;
};
template <>
inline constexpr unsigned int fnAccRegsMatMult<cfloat, float>() {
    return 4;
};
template <>
inline constexpr unsigned int fnAccRegsMatMult<cfloat, cfloat>() {
    return 4;
};

// function to return the number of lanes for a type combo
// The default is effectively an error trap, but adding an error message to a constexpr return results in a warning.
template <typename TT_DATA_A, typename TT_DATA_B>
inline constexpr unsigned int fnNumLanesMatMult() {
    return 0;
};
template <>
inline constexpr unsigned int fnNumLanesMatMult<int16, int16>() {
    return 16;
};
template <>
inline constexpr unsigned int fnNumLanesMatMult<cint16, int16>() {
    return 8;
};
template <>
inline constexpr unsigned int fnNumLanesMatMult<cint16, cint16>() {
    return 8;
};
template <>
inline constexpr unsigned int fnNumLanesMatMult<int32, int16>() {
    return 8;
};
template <>
inline constexpr unsigned int fnNumLanesMatMult<int32, int32>() {
    return 8;
};
template <>
inline constexpr unsigned int fnNumLanesMatMult<cint32, int16>() {
    return 4;
};
template <>
inline constexpr unsigned int fnNumLanesMatMult<cint32, cint16>() {
    return 4;
};
template <>
inline constexpr unsigned int fnNumLanesMatMult<cint32, int32>() {
    return 4;
};
template <>
inline constexpr unsigned int fnNumLanesMatMult<cint32, cint32>() {
    return 4;
};
template <>
inline constexpr unsigned int fnNumLanesMatMult<float, float>() {
    return 8;
};
template <>
inline constexpr unsigned int fnNumLanesMatMult<cfloat, float>() {
    return 4;
};
template <>
inline constexpr unsigned int fnNumLanesMatMult<cfloat, cfloat>() {
    return 4;
};

// Function to return the lowest common multiple of two numbers
// A full implementation of this would entail prime factor decomposition, but here
// The maximum integer size is 16, so a simpler brute force method will do.
template <typename TT_DATA_A, typename TT_DATA_B, unsigned int TP_FACTOR>
inline constexpr unsigned int fnLCMMatMult() {
    return ((fnNumLanesMatMult<TT_DATA_A, TT_DATA_B>() == 2)
                ? ((TP_FACTOR % 2 == 0) ? TP_FACTOR : (TP_FACTOR * 2))
                : (fnNumLanesMatMult<TT_DATA_A, TT_DATA_B>() == 4)
                      ? ((TP_FACTOR % 4 == 0) ? TP_FACTOR : ((TP_FACTOR % 2 == 0) ? (TP_FACTOR * 2) : (TP_FACTOR * 4)))
                      : (fnNumLanesMatMult<TT_DATA_A, TT_DATA_B>() == 8)
                            ? ((TP_FACTOR % 8 == 0)
                                   ? TP_FACTOR
                                   : ((TP_FACTOR % 4 == 0) ? (TP_FACTOR * 2)
                                                           : ((TP_FACTOR % 2 == 0) ? (TP_FACTOR * 4) : TP_FACTOR * 8)))
                            : 0);
};

// function to return the number of samples in an output vector for a type combo
template <typename TT_DATA_A, typename TT_DATA_B>
inline constexpr unsigned int fnVOutSizeMatMult() {
    return fnNumLanesMatMult<TT_DATA_A, TT_DATA_B>();
};
}
}
}
}
}

#endif // _DSPLIB_MATRIX_MULT_TRAITS_HPP_
