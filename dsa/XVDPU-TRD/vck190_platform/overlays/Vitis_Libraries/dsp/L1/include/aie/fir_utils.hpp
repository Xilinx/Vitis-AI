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
#pragma once

#ifndef _DSPLIB_FIR_UTILS_HPP_
#define _DSPLIB_FIR_UTILS_HPP_

/* This file exists to hold utility functions for kernels in general, not specific
   to one library element. Also, the functions in this file do not use vector types
   or intrinsics so are fit for use by aiecompiler and kernel constructors.
*/

#include <stdio.h>
#include <adf.h>

// The following maximums are the maximums tested. The function may work for larger values.
#ifndef FIR_LEN_MAX
#define FIR_LEN_MAX 240
#endif
#define FIR_LEN_MIN 4
#define SHIFT_MAX 62
#define SHIFT_MIN 0
#define ROUND_MAX 7
#define ROUND_MIN 0
#define INTERPOLATE_FACTOR_MAX 16
#define INTERPOLATE_FACTOR_MIN 1

#define FRACT_INTERPOLATE_FACTOR_MAX 15
#define FRACT_INTERPOLATE_FACTOR_MIN 3
#define DECIMATE_FACTOR_MAX 7
#define DECIMATE_FACTOR_MIN 2

// CEIL and TRUNC are common utilities where x is rounded up (CEIL) or down (TRUNC)
// until a value which is multiple of y is found. This may be x.
// e.g. CEIL(10,8) = 16, TRUNC(10, 8) = 8
#define CEIL(x, y) (((x + y - 1) / y) * y)
#define TRUNC(x, y) (((x) / y) * y)
// Pragma unroll complains if you try to unroll(0);
// It's safe to just unroll(1) in this circumstance.
#define GUARD_ZERO(x) ((x) > 0 ? (x) : 1)

#define CASC_IN_TRUE true
#define CASC_IN_FALSE false
#define CASC_OUT_TRUE true
#define CASC_OUT_FALSE false

#define DUAL_IP_SINGLE 0
#define DUAL_IP_DUAL 1

#define USE_COEFF_RELOAD_FALSE 0
#define USE_COEFF_RELOAD_TRUE 1

namespace xf {
namespace dsp {
namespace aie {

// function to return the size of the acc,
template <typename TT_DATA, typename TT_COEFF>
inline constexpr unsigned int fnAccSize() {
    return 0;
};
template <>
inline constexpr unsigned int fnAccSize<int16, int16>() {
    return 48;
};
template <>
inline constexpr unsigned int fnAccSize<cint16, int16>() {
    return 48;
};
template <>
inline constexpr unsigned int fnAccSize<cint16, cint16>() {
    return 48;
};
template <>
inline constexpr unsigned int fnAccSize<int32, int16>() {
    return 80;
};
template <>
inline constexpr unsigned int fnAccSize<int32, int32>() {
    return 80;
};
template <>
inline constexpr unsigned int fnAccSize<cint32, int16>() {
    return 80;
};
template <>
inline constexpr unsigned int fnAccSize<cint32, cint16>() {
    return 80;
};
template <>
inline constexpr unsigned int fnAccSize<cint32, int32>() {
    return 80;
};
template <>
inline constexpr unsigned int fnAccSize<cint32, cint32>() {
    return 80;
};
template <>
inline constexpr unsigned int fnAccSize<float, float>() {
    return 32;
};
template <>
inline constexpr unsigned int fnAccSize<cfloat, float>() {
    return 32;
};
template <>
inline constexpr unsigned int fnAccSize<cfloat, cfloat>() {
    return 32;
};

// function to return the number of 768-bit acc's lanes for a type combo
template <typename TT_DATA, typename TT_COEFF>
inline constexpr unsigned int fnNumLanes() {
    return 0;
};
template <>
inline constexpr unsigned int fnNumLanes<int16, int16>() {
    return 16;
}; // 16x2
template <>
inline constexpr unsigned int fnNumLanes<cint16, int16>() {
    return 8;
}; // 8x2
template <>
inline constexpr unsigned int fnNumLanes<cint16, cint16>() {
    return 8;
}; // 8x1
template <>
inline constexpr unsigned int fnNumLanes<int32, int16>() {
    return 8;
}; // 8x2
template <>
inline constexpr unsigned int fnNumLanes<int32, int32>() {
    return 8;
}; // 8x1
template <>
inline constexpr unsigned int fnNumLanes<cint32, int16>() {
    return 4;
}; // 4x2
template <>
inline constexpr unsigned int fnNumLanes<cint32, cint16>() {
    return 4;
}; // 4x1
template <>
inline constexpr unsigned int fnNumLanes<cint32, int32>() {
    return 4;
}; // 4x1
template <>
inline constexpr unsigned int fnNumLanes<cint32, cint32>() {
    return 2;
}; // 2x1
template <>
inline constexpr unsigned int fnNumLanes<float, float>() {
    return 8;
};
template <>
inline constexpr unsigned int fnNumLanes<cfloat, float>() {
    return 4;
};
template <>
inline constexpr unsigned int fnNumLanes<cfloat, cfloat>() {
    return 4;
};

// function to return the number of 384-bit acc's lanes for a type combo
// 80-bit accs (for 32-bit input types) are always 768-bit
template <typename TT_DATA, typename TT_COEFF>
inline constexpr unsigned int fnNumLanes384() {
    return 0;
};
template <>
inline constexpr unsigned int fnNumLanes384<int16, int16>() {
    return 8;
}; // 8x4
template <>
inline constexpr unsigned int fnNumLanes384<cint16, int16>() {
    return 4;
}; // 4x4
template <>
inline constexpr unsigned int fnNumLanes384<cint16, cint16>() {
    return 4;
}; // 4x2
template <>
inline constexpr unsigned int fnNumLanes384<int32, int16>() {
    return 8;
}; // 8x2 - 80-bit
template <>
inline constexpr unsigned int fnNumLanes384<int32, int32>() {
    return 4;
}; // 4x2
template <>
inline constexpr unsigned int fnNumLanes384<cint32, int16>() {
    return 4;
}; // 4x2
template <>
inline constexpr unsigned int fnNumLanes384<cint32, cint16>() {
    return 4;
}; // 4x1
template <>
inline constexpr unsigned int fnNumLanes384<cint32, int32>() {
    return 4;
}; // 4x1 or 2x2
template <>
inline constexpr unsigned int fnNumLanes384<cint32, cint32>() {
    return 2;
}; // 2x1
template <>
inline constexpr unsigned int fnNumLanes384<float, float>() {
    return 8;
};
template <>
inline constexpr unsigned int fnNumLanes384<cfloat, float>() {
    return 4;
};
template <>
inline constexpr unsigned int fnNumLanes384<cfloat, cfloat>() {
    return 4;
};

// function to return the number of columns for a tall-narrow atomic intrinsic for a type combo
template <typename TT_DATA, typename TT_COEFF>
inline constexpr unsigned int fnNumCols() {
    return sizeof(TT_COEFF) == 2 ? 2 : 1;
};

// function to return the number of columns for a short-wide atomic intrinsic for a type combo
template <typename TT_DATA, typename TT_COEFF>
inline constexpr unsigned int fnNumCols384() {
    return 2 * (sizeof(TT_COEFF) == 2 ? 2 : 1);
};
template <>
inline constexpr unsigned int fnNumCols384<int16, int16>() {
    return 4;
}; // 8x4
template <>
inline constexpr unsigned int fnNumCols384<cint16, int16>() {
    return 4;
}; // 4x4
template <>
inline constexpr unsigned int fnNumCols384<cint16, cint16>() {
    return 2;
}; // 4x2
template <>
inline constexpr unsigned int fnNumCols384<int32, int16>() {
    return 2;
}; // 8x2 - 80-bit
template <>
inline constexpr unsigned int fnNumCols384<int32, int32>() {
    return 2;
}; // 4x2 or 8x1
template <>
inline constexpr unsigned int fnNumCols384<cint32, int16>() {
    return 2;
}; // 4x2
template <>
inline constexpr unsigned int fnNumCols384<cint32, cint16>() {
    return 1;
}; // 4x1
template <>
inline constexpr unsigned int fnNumCols384<cint32, int32>() {
    return 1;
}; // 4x1 or 2x2
template <>
inline constexpr unsigned int fnNumCols384<cint32, cint32>() {
    return 1;
}; // 2x1
template <>
inline constexpr unsigned int fnNumCols384<float, float>() {
    return 1;
};
template <>
inline constexpr unsigned int fnNumCols384<cfloat, float>() {
    return 1;
};
template <>
inline constexpr unsigned int fnNumCols384<cfloat, cfloat>() {
    return 1;
};

namespace fir {
// Hardware constants and functions.
static constexpr unsigned int kAIEYRegSizeInBytes = 128; // The size of a Y register in bytes

// Functions to support defensive checks
enum { enumUnknownType = 0, enumInt16, enumCint16, enumInt32, enumCint32, enumFloat, enumCfloat };
// function to return an enumeration of the data or coefficient type
template <typename TT_INPUT>
inline constexpr unsigned int fnEnumType() {
    return enumUnknownType;
}; // returns 0 as default. This can be trapped as an error;
template <>
inline constexpr unsigned int fnEnumType<int16>() {
    return enumInt16;
};
template <>
inline constexpr unsigned int fnEnumType<cint16>() {
    return enumCint16;
};
template <>
inline constexpr unsigned int fnEnumType<int32>() {
    return enumInt32;
};
template <>
inline constexpr unsigned int fnEnumType<cint32>() {
    return enumCint32;
};
template <>
inline constexpr unsigned int fnEnumType<float>() {
    return enumFloat;
};
template <>
inline constexpr unsigned int fnEnumType<cfloat>() {
    return enumCfloat;
};

// Function to trap illegal precision of DATA vs COEFF
// This defaults to legal which would be fail-unsafe if not used in conjunction with fnEnumType trap
template <typename TT_DATA, typename TT_COEFF>
inline constexpr unsigned int fnTypeCheckDataCoeffSize() {
    return 1;
}; // default here is a legal combo
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffSize<int16, int32>() {
    return 0;
};
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffSize<int16, cint32>() {
    return 0;
};
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffSize<cint16, int32>() {
    return 0;
};
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffSize<cint16, cint32>() {
    return 0;
};

// Function to trap illegal real DATA vs complex COEFF
// This defaults to legal which would be fail-unsage is not used in conjunction with functions above.
template <typename TT_DATA, typename TT_COEFF>
inline constexpr unsigned int fnTypeCheckDataCoeffCmplx() {
    return 1;
}; // default here is a legal combo
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffCmplx<int16, cint16>() {
    return 0;
};
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffCmplx<int32, cint32>() {
    return 0;
};
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffCmplx<int32, cint16>() {
    return 0;
};
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffCmplx<float, cfloat>() {
    return 0;
};

// Function to trap illegal combo of real and float types
template <typename TT_DATA, typename TT_COEFF>
inline constexpr unsigned int fnTypeCheckDataCoeffFltInt() {
    return 1;
}; // default here is a legal combo
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffFltInt<int16, float>() {
    return 0;
};
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffFltInt<int16, cfloat>() {
    return 0;
};
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffFltInt<cint16, float>() {
    return 0;
};
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffFltInt<cint16, cfloat>() {
    return 0;
};
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffFltInt<int32, float>() {
    return 0;
};
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffFltInt<int32, cfloat>() {
    return 0;
};
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffFltInt<cint32, float>() {
    return 0;
};
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffFltInt<cint32, cfloat>() {
    return 0;
};

template <>
inline constexpr unsigned int fnTypeCheckDataCoeffFltInt<float, int16>() {
    return 0;
};
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffFltInt<cfloat, int16>() {
    return 0;
};
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffFltInt<float, cint16>() {
    return 0;
};
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffFltInt<cfloat, cint16>() {
    return 0;
};
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffFltInt<float, int32>() {
    return 0;
};
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffFltInt<cfloat, int32>() {
    return 0;
};
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffFltInt<float, cint32>() {
    return 0;
};
template <>
inline constexpr unsigned int fnTypeCheckDataCoeffFltInt<cfloat, cint32>() {
    return 0;
};

// Function to trap illegal  types
template <typename TT_DATA, typename TT_COEFF>
inline constexpr unsigned int fnUnsupportedTypeCombo() {
    return 1;
}; // default here is a legal combo
template <>
inline constexpr unsigned int fnUnsupportedTypeCombo<int16, int16>() {
    return 0;
};

// IF input type
template <bool T_CASC_IN, typename T_D, unsigned int T_DUAL_IP = 0>
struct T_inputIF {};
template <>
struct T_inputIF<CASC_IN_FALSE, int16, 0> {
    input_window<int16>* inWindow;
};
template <>
struct T_inputIF<CASC_IN_FALSE, int32, 0> {
    input_window<int32>* inWindow;
};
template <>
struct T_inputIF<CASC_IN_FALSE, cint16, 0> {
    input_window<cint16>* inWindow;
};
template <>
struct T_inputIF<CASC_IN_FALSE, cint32, 0> {
    input_window<cint32>* inWindow;
};
template <>
struct T_inputIF<CASC_IN_FALSE, float, 0> {
    input_window<float>* inWindow;
};
template <>
struct T_inputIF<CASC_IN_FALSE, cfloat, 0> {
    input_window<cfloat>* inWindow;
};
template <>
struct T_inputIF<CASC_IN_TRUE, int16, 0> {
    input_window<int16>* inWindow;
    input_stream_cacc48* inCascade;
};
template <>
struct T_inputIF<CASC_IN_TRUE, int32, 0> {
    input_window<int32>* inWindow;
    input_stream_cacc48* inCascade;
};
template <>
struct T_inputIF<CASC_IN_TRUE, cint16, 0> {
    input_window<cint16>* inWindow;
    input_stream_cacc48* inCascade;
};
template <>
struct T_inputIF<CASC_IN_TRUE, cint32, 0> {
    input_window<cint32>* inWindow;
    input_stream_cacc48* inCascade;
};
template <>
struct T_inputIF<CASC_IN_TRUE, float, 0> {
    input_window<float>* inWindow;
    input_stream_cacc48* inCascade;
};
template <>
struct T_inputIF<CASC_IN_TRUE, cfloat, 0> {
    input_window<cfloat>* inWindow;
    input_stream_cacc48* inCascade;
};
template <>
struct T_inputIF<CASC_IN_FALSE, int16, 1> {
    input_window<int16>* inWindow;
    input_window<int16>* inWindowReverse;
};
template <>
struct T_inputIF<CASC_IN_FALSE, int32, 1> {
    input_window<int32>* inWindow;
    input_window<int32>* inWindowReverse;
};
template <>
struct T_inputIF<CASC_IN_FALSE, cint16, 1> {
    input_window<cint16>* inWindow;
    input_window<cint16>* inWindowReverse;
};
template <>
struct T_inputIF<CASC_IN_FALSE, cint32, 1> {
    input_window<cint32>* inWindow;
    input_window<cint32>* inWindowReverse;
};
template <>
struct T_inputIF<CASC_IN_FALSE, float, 1> {
    input_window<float>* inWindow;
    input_window<float>* inWindowReverse;
};
template <>
struct T_inputIF<CASC_IN_FALSE, cfloat, 1> {
    input_window<cfloat>* inWindow;
    input_window<cfloat>* inWindowReverse;
};
template <>
struct T_inputIF<CASC_IN_TRUE, int16, 1> {
    input_window<int16>* inWindow;
    input_window<int16>* inWindowReverse;
    input_stream_cacc48* inCascade;
};
template <>
struct T_inputIF<CASC_IN_TRUE, int32, 1> {
    input_window<int32>* inWindow;
    input_window<int32>* inWindowReverse;
    input_stream_cacc48* inCascade;
};
template <>
struct T_inputIF<CASC_IN_TRUE, cint16, 1> {
    input_window<cint16>* inWindow;
    input_window<cint16>* inWindowReverse;
    input_stream_cacc48* inCascade;
};
template <>
struct T_inputIF<CASC_IN_TRUE, cint32, 1> {
    input_window<cint32>* inWindow;
    input_window<cint32>* inWindowReverse;
    input_stream_cacc48* inCascade;
};
template <>
struct T_inputIF<CASC_IN_TRUE, float, 1> {
    input_window<float>* inWindow;
    input_window<float>* inWindowReverse;
    input_stream_cacc48* inCascade;
};
template <>
struct T_inputIF<CASC_IN_TRUE, cfloat, 1> {
    input_window<cfloat>* inWindow;
    input_window<cfloat>* inWindowReverse;
    input_stream_cacc48* inCascade;
};

// IF output type
template <bool T_CASC_IN, typename T_D>
struct T_outputIF {};
template <>
struct T_outputIF<CASC_OUT_FALSE, int16> {
    output_window<int16>* restrict outWindow;
    output_window<int16>* restrict outWindow2;
};
template <>
struct T_outputIF<CASC_OUT_FALSE, int32> {
    output_window<int32>* restrict outWindow;
    output_window<int32>* restrict outWindow2;
};
template <>
struct T_outputIF<CASC_OUT_FALSE, cint16> {
    output_window<cint16>* restrict outWindow;
    output_window<cint16>* restrict outWindow2;
};
template <>
struct T_outputIF<CASC_OUT_FALSE, cint32> {
    output_window<cint32>* restrict outWindow;
    output_window<cint32>* restrict outWindow2;
};
template <>
struct T_outputIF<CASC_OUT_FALSE, float> {
    output_window<float>* restrict outWindow;
    output_window<float>* restrict outWindow2;
};
template <>
struct T_outputIF<CASC_OUT_FALSE, cfloat> {
    output_window<cfloat>* restrict outWindow;
    output_window<cfloat>* restrict outWindow2;
};
template <>
struct T_outputIF<CASC_OUT_TRUE, int16> {
    output_stream_cacc48* outCascade;
    output_window<int16>* broadcastWindow;
};
template <>
struct T_outputIF<CASC_OUT_TRUE, int32> {
    output_stream_cacc48* outCascade;
    output_window<int32>* broadcastWindow;
};
template <>
struct T_outputIF<CASC_OUT_TRUE, cint16> {
    output_stream_cacc48* outCascade;
    output_window<cint16>* broadcastWindow;
};
template <>
struct T_outputIF<CASC_OUT_TRUE, cint32> {
    output_stream_cacc48* outCascade;
    output_window<cint32>* broadcastWindow;
};
template <>
struct T_outputIF<CASC_OUT_TRUE, float> {
    output_stream_cacc48* outCascade;
    output_window<float>* broadcastWindow;
};
template <>
struct T_outputIF<CASC_OUT_TRUE, cfloat> {
    output_stream_cacc48* outCascade;
    output_window<cfloat>* broadcastWindow;
};

//----------------------------------------------------------------------
// nullElem
template <typename T_RDATA>
inline T_RDATA nullElem() {
    return 0;
};

// Null cint16_t element
template <>
inline cint16_t nullElem() {
    cint16_t d;
    d.real = 0;
    d.imag = 0;
    return d;
};

// Null cint32 element
template <>
inline cint32 nullElem() {
    cint32 d;
    d.real = 0;
    d.imag = 0;
    return d;
};

// Null float element
template <>
inline float nullElem() {
    return 0.0;
};

// Null cint32 element
template <>
inline cfloat nullElem() {
    cfloat retVal;

    retVal.real = 0.0;
    retVal.imag = 0.0;
    return retVal;
};

// function to return Margin length.
template <size_t TP_FIR_LEN, typename TT_DATA>
inline constexpr unsigned int fnFirMargin() {
    return CEIL(TP_FIR_LEN, (32 / sizeof(TT_DATA)));
};

// Truncation. This function rounds x down to the next multiple of y (which may be x)
inline constexpr int fnTrunc(unsigned int x, unsigned int y) {
    return TRUNC(x, y);
};

// Calculate FIR range for cascaded kernel
template <unsigned int TP_FL, unsigned int TP_CL, int TP_KP, int TP_Rnd = 1>
inline constexpr int fnFirRange() {
    // TP_FL - FIR Length, TP_CL - Cascade Length, TP_KP - Kernel Position
    return ((fnTrunc(TP_FL, TP_Rnd * TP_CL) / TP_CL) +
            ((TP_FL - fnTrunc(TP_FL, TP_Rnd * TP_CL)) >= TP_Rnd * (TP_KP + 1) ? TP_Rnd : 0));
}
template <unsigned int TP_FL, unsigned int TP_CL, int TP_KP, int TP_Rnd = 1>
inline constexpr int fnFirRangeRem() {
    // TP_FL - FIR Length, TP_CL - Cascade Length, TP_KP - Kernel Position
    // this is for last in the cascade
    return ((fnTrunc(TP_FL, TP_Rnd * TP_CL) / TP_CL) + ((TP_FL - fnTrunc(TP_FL, TP_Rnd * TP_CL)) % TP_Rnd));
}

// Calculate FIR range offset for cascaded kernel
template <unsigned int TP_FL, unsigned int TP_CL, int TP_KP, int TP_Rnd = 1, int TP_Sym = 1>
inline constexpr int fnFirRangeOffset() {
    // TP_FL - FIR Length, TP_CL - Cascade Length, TP_KP - Kernel Position
    return (TP_KP * (fnTrunc(TP_FL, TP_Rnd * TP_CL) / TP_CL) +
            ((TP_FL - fnTrunc(TP_FL, TP_Rnd * TP_CL)) >= TP_Rnd * TP_KP
                 ? TP_Rnd * TP_KP
                 : (fnTrunc(TP_FL, TP_Rnd) - fnTrunc(TP_FL, TP_Rnd * TP_CL)))) /
           TP_Sym;
}

//
template <typename T_D>
inline T_D formatUpshiftCt(T_D inVal) {
    // Do nothing for types other than 16-bit integers
    return inVal;
}

template <>
inline int16 formatUpshiftCt(int16 inVal) {
    const unsigned int kMaxUpshiftVal = 16;
    int16 retVal;
    // Make sure value is within UCT supported range (0 - 16).
    retVal = inVal % kMaxUpshiftVal;
    return retVal;
}
template <>
inline cint16 formatUpshiftCt(cint16 inVal) {
    const unsigned int kMaxUpshiftVal = 16;
    cint16 retVal;
    // Make sure value is within UCT supported range (0 - 16).
    retVal.real = inVal.real % kMaxUpshiftVal;
    retVal.imag = 0;
    return retVal;
}

template <typename T_D>
inline int16 getUpshiftCt(T_D inVal) {
    // Do nothing for types other than 16-bit integers
    return 0;
}

template <>
inline int16 getUpshiftCt(int16 inVal) {
    const unsigned int kMaxUpshiftVal = 16;
    int16 retVal;
    // Make sure value is within UCT supported range (0 - 16).
    retVal = inVal % kMaxUpshiftVal;
    return retVal;
}
template <>
inline int16 getUpshiftCt(cint16 inVal) {
    const unsigned int kMaxUpshiftVal = 16;
    int16 retVal;
    // Make sure value is within UCT supported range (0 - 16).
    retVal = inVal.real % kMaxUpshiftVal;
    return retVal;
}
}
}
}
}
#endif // _DSPLIB_FIR_UTILS_HPP_
