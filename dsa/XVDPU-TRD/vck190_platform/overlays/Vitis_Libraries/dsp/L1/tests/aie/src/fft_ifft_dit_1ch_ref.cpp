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
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include "fft_ifft_dit_1ch_ref.hpp"
#include "fft_ifft_dit_twiddle_lut.h"
#include "fir_ref_utils.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace fft {
namespace dit_1ch {
/*
  FFT/iFFT DIT single channel reference model
*/
static constexpr unsigned int kMaxPointSize = 4096;

//-----------------------------------------------------------------------------------------------------
// Utility functions for FFT/iFFT single channel reference model

template <typename TT_TWIDDLE>
TT_TWIDDLE* fnGetTwiddleMasterBase(){};
template <>
cint16* fnGetTwiddleMasterBase<cint16>() {
    return (cint16*)twiddle_master_cint16;
};
template <>
cfloat* fnGetTwiddleMasterBase<cfloat>() {
    return (cfloat*)twiddle_master_cfloat;
};

// function to query type -used to avoid template specializing the whole class.
template <typename T_D>
inline constexpr bool is_cfloat() {
    return false;
};
template <>
inline constexpr bool is_cfloat<cfloat>() {
    return true;
};

// Function has to be fully specialized, which is excessive, so integer template parameters are used simply as
// parameters (function arguments) to reduce duplication
template <typename TT_TWIDDLE>
TT_TWIDDLE get_twiddle(int i, unsigned int TP_POINT_SIZE, unsigned int TP_FFT_NIFFT) {
    return 0; // never used
}
template <>
cint16 get_twiddle<cint16>(int i, unsigned int TP_POINT_SIZE, unsigned int TP_FFT_NIFFT) {
    int step = kMaxPointSize / TP_POINT_SIZE;
    cint16 raw_twiddle = twiddle_master_cint16[i * step];
    if (TP_FFT_NIFFT == 0) {
        raw_twiddle.imag = -raw_twiddle.imag;
    }
    return raw_twiddle;
}
template <>
cint32 get_twiddle<cint32>(int i, unsigned int TP_POINT_SIZE, unsigned int TP_FFT_NIFFT) {
    int step = kMaxPointSize / TP_POINT_SIZE;
    cint32 raw_twiddle = twiddle_master_cint32[i * step];
    if (TP_FFT_NIFFT == 0) {
        raw_twiddle.imag = -raw_twiddle.imag;
    }
    return raw_twiddle;
}
template <>
cfloat get_twiddle<cfloat>(int i, unsigned int TP_POINT_SIZE, unsigned int TP_FFT_NIFFT) {
    int step = kMaxPointSize / TP_POINT_SIZE;
    cfloat raw_twiddle = twiddle_master_cfloat[i * step];
    if (TP_FFT_NIFFT == 0) {
        raw_twiddle.imag = -raw_twiddle.imag;
    }
    return raw_twiddle;
}

//--------castInput
// converts the input type to the type used for internal data processing
template <typename TT_DATA>
inline T_int_data<TT_DATA> castInput(TT_DATA sampleIn) {
    T_int_data<TT_DATA> retVal; // default to mute warnings
    return retVal;
}
template <>
inline T_int_data<int16> castInput<int16>(int16 sampleIn) {
    T_int_data<int16> retVal;
    retVal.real = sampleIn;
    retVal.imag = 0;
    return retVal;
}
template <>
inline T_int_data<cint16> castInput<cint16>(cint16 sampleIn) {
    T_int_data<cint16> retVal;
    retVal.real = sampleIn.real;
    retVal.imag = sampleIn.imag;
    return retVal;
}
template <>
inline T_int_data<int32> castInput<int32>(int32 sampleIn) {
    T_int_data<int32> retVal;
    retVal.real = sampleIn;
    retVal.imag = 0;
    return retVal;
}
template <>
inline T_int_data<cint32> castInput<cint32>(cint32 sampleIn) {
    T_int_data<cint32> retVal;
    retVal.real = sampleIn.real;
    retVal.imag = sampleIn.imag;
    return retVal;
}
template <>
inline T_int_data<float> castInput<float>(float sampleIn) {
    T_int_data<float> retVal;
    retVal.real = sampleIn;
    retVal.imag = 0;
    return retVal;
}
template <>
inline T_int_data<cfloat> castInput<cfloat>(cfloat sampleIn) {
    T_int_data<cfloat> retVal;
    retVal.real = sampleIn.real;
    retVal.imag = sampleIn.imag;
    return retVal;
}

// Derivation of base type of a complex type.
template <typename TT_TWIDDLE = cint16>
struct T_base_type_struct {
    using T_base_type = int16;
};
template <>
struct T_base_type_struct<cint32> {
    using T_base_type = int32;
};
template <>
struct T_base_type_struct<cfloat> {
    using T_base_type = float;
};

// fnMaxPos Maximum positive value of a twiddle component. This is arcane because twiddle is a complex type, but the
// return is real.
template <typename TT_TWIDDLE = cint16>
typename T_base_type_struct<TT_TWIDDLE>::T_base_type fnMaxPos() {
    return C_PMAX16;
}
template <>
T_base_type_struct<cint32>::T_base_type fnMaxPos<cint32>() {
    return C_PMAX32;
}
template <>
T_base_type_struct<cfloat>::T_base_type fnMaxPos<cfloat>() {
    return 3.4028234664e+32;
}

// fnMaxNeg Maximum negative value of a twiddle component. This is arcane because twiddle is a complex type, but the
// return is real.
template <typename TT_TWIDDLE = cint16>
typename T_base_type_struct<TT_TWIDDLE>::T_base_type fnMaxNeg() {
    return C_NMAX16;
}
template <>
T_base_type_struct<cint32>::T_base_type fnMaxNeg<cint32>() {
    return C_NMAX32;
}
template <>
T_base_type_struct<cfloat>::T_base_type fnMaxNeg<cfloat>() {
    return -3.4028234664e+32;
}

//--------castOutput
// converts the input type to the type used for internal data processing
template <typename TT_DATA>
inline TT_DATA castOutput(T_int_data<TT_DATA> sampleIn, const unsigned shift) {
    TT_DATA retVal; // default to mute warnings
    return retVal;
}
template <>
inline int16 castOutput<int16>(T_int_data<int16> sampleIn, const unsigned shift) {
    int16 retVal;
    // rounding is performed in the radix stages
    // retVal = (sampleIn.real + (1 << (shift-1))) >> shift;
    retVal = sampleIn.real;
    if (retVal >= C_PMAX16) {
        retVal = C_PMAX16;
    } else if (retVal < C_NMAX16) {
        retVal = C_NMAX16;
    }
    return retVal;
}
template <>
inline cint16 castOutput<cint16>(T_int_data<cint16> sampleIn, const unsigned shift) {
    cint16 retVal;
    // rounding is performed in the radix stages
    retVal.real = sampleIn.real;
    retVal.imag = sampleIn.imag;
    if (sampleIn.real >= C_PMAX16) {
        retVal.real = C_PMAX16;
    } else if (sampleIn.real < C_NMAX16) {
        retVal.real = C_NMAX16;
    }
    if (sampleIn.imag >= C_PMAX16) {
        retVal.imag = C_PMAX16;
    } else if (sampleIn.imag < C_NMAX16) {
        retVal.imag = C_NMAX16;
    }
    return retVal;
}
template <>
inline int32 castOutput<int32>(T_int_data<int32> sampleIn, const unsigned shift) {
    int32 retVal;
    // rounding is performed in the radix stages
    retVal = sampleIn.real;
    if (retVal >= C_PMAX32) {
        retVal = C_PMAX32;
    } else if (retVal < C_NMAX32) {
        retVal = C_NMAX32;
    }
    return retVal;
}
template <>
inline cint32 castOutput<cint32>(T_int_data<cint32> sampleIn, const unsigned shift) {
    cint32 retVal;
    // rounding is performed in the radix stages
    retVal.real = sampleIn.real;
    retVal.imag = sampleIn.imag;
    if (retVal.real >= C_PMAX32) {
        retVal.real = C_PMAX32;
    } else if (retVal.real < C_NMAX32) {
        retVal.real = C_NMAX32;
    }
    if (retVal.imag >= C_PMAX32) {
        retVal.imag = C_PMAX32;
    } else if (retVal.imag < C_NMAX32) {
        retVal.imag = C_NMAX32;
    }
    return retVal;
}
template <>
inline float castOutput<float>(T_int_data<float> sampleIn, const unsigned shift) {
    float retVal;
    // rounding is performed in the radix stages
    // retVal = (sampleIn.real + (float)(1<<(shift-1)))/(float)(1 << shift);
    retVal = sampleIn.real;
    return retVal;
}
template <>
inline cfloat castOutput<cfloat>(T_int_data<cfloat> sampleIn, const unsigned shift) {
    cfloat retVal;
    // rounding is performed in the radix stages
    retVal.real = sampleIn.real;
    retVal.imag = sampleIn.imag;
    // retVal.real = (sampleIn.real + (float)(1<<(shift-1)))/(float)(1 << shift);
    // retVal.imag = (sampleIn.imag + (float)(1<<(shift-1)))/(float)(1 << shift);
    return retVal;
}

//--------butterfly
template <typename TT_DATA, typename TT_TWIDDLE>
inline void btflynonbitacc(TT_TWIDDLE twiddle,
                           T_int_data<TT_DATA> A,
                           T_int_data<TT_DATA> B,
                           T_int_data<TT_DATA>& outA,
                           T_int_data<TT_DATA>& outB) {
    return;
}
template <>
inline void btflynonbitacc<cint16, cint16>(
    cint16 twiddle, T_int_data<cint16> A, T_int_data<cint16> B, T_int_data<cint16>& outA, T_int_data<cint16>& outB) {
    T_int_data<cint16> rotB, Ao, Bo;
    rotB.real = (int32)(((int64)twiddle.real * B.real - (int64)twiddle.imag * B.imag + 16384) >> 15);
    rotB.imag = (int32)(((int64)twiddle.real * B.imag + (int64)twiddle.imag * B.real + 16384) >> 15);
#ifdef _DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
    printf("rotB = (%6d, %6d), ", rotB.real, rotB.imag);
#endif //_DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_

    Ao.real = A.real + rotB.real;
    Ao.imag = A.imag + rotB.imag;
    Bo.real = A.real - rotB.real;
    Bo.imag = A.imag - rotB.imag;
    outA = Ao;
    outB = Bo;
}

//----------------log2 point size
template <unsigned int TP_POINT_SIZE>
inline constexpr unsigned int getPointSizePower() {
    switch (TP_POINT_SIZE) {
        case 16:
            return 4;
            break;
        case 32:
            return 5;
            break;
        case 64:
            return 6;
            break;
        case 128:
            return 7;
            break;
        case 256:
            return 8;
            break;
        case 512:
            return 9;
            break;
        case 1024:
            return 10;
            break;
        case 2048:
            return 11;
            break;
        case 4096:
            return 12;
            break;
        case 8192:
            return 13;
            break;
        case 16384:
            return 14;
            break;
        case 32768:
            return 15;
            break;
        case 65536:
            return 16;
            break;
        default:
            printf("Error in pointSizePower\n");
            return 0;
            break;
    }
}

unsigned int bitRev(unsigned int len, unsigned int val) {
    unsigned int retVal = 0;
    unsigned int ip = val;
    for (int i = 0; i < len; i++) {
        retVal = retVal << 1;
        if (ip % 2 == 1) {
            retVal++;
        }
        ip >>= 1;
    }
    return retVal;
}
inline void fftScale(int rndMode, int shift, T_accRef<int16>& accum) {
    // cint64_t ret;
    accum.real = rounding(rndMode, shift, accum.real);
    // return ret;
};

inline void fftScale(int rndMode, int shift, T_accRef<int32>& accum) {
    // cint64_t ret;
    accum.real = rounding(rndMode, shift, accum.real);
    // return ret;
};

inline void fftScale(int rndMode, int shift, T_accRef<cint32>& accum) {
    // cint64_t ret;
    accum.real = rounding(rndMode, shift, accum.real);
    accum.imag = rounding(rndMode, shift, accum.imag);
    // return ret;
};
inline void fftScale(int rndMode, int shift, T_accRef<cint16>& accum) {
    // cint64_t ret;
    accum.real = rounding(rndMode, shift, accum.real);
    accum.imag = rounding(rndMode, shift, accum.imag);
    // return ret;
};

// Rounding and shift - do apply to float in the FFT
inline void fftScale(int rndMode, int shift, T_accRef<float>& accum) {
    accum.real = (accum.real + (float)(1 << (shift - 1))) / (float)(1 << (shift));
};
inline void fftScale(int rndMode, int shift, T_accRef<cfloat>& accum) {
    accum.real = (accum.real + (float)(1 << (shift - 1))) / (float)(1 << (shift));
    accum.imag = (accum.imag + (float)(1 << (shift - 1))) / (float)(1 << (shift));
};

//------------------------------------------------------
// Templatized complex multiply, returning uniform cint64.
// inv is actually compile-time constant, but passed here as a run-time because it reduces code verbosity.
template <typename T_D, typename T_TW>
cint64 cmpy(T_D d, T_TW tw, bool inv) {
    return {0, 0};
};
template <>
cint64 cmpy<cint16, cint16>(cint16 d, cint16 tw, bool inv) {
    cint64 retVal;
    if (inv == true) {
        retVal.real = (int64)d.real * (int64)tw.real + (int64)d.imag * (int64)tw.imag;
        retVal.imag = (int64)d.imag * (int64)tw.real - (int64)d.real * (int64)tw.imag;
    } else {
        retVal.real = (int64)d.real * (int64)tw.real - (int64)d.imag * (int64)tw.imag;
        retVal.imag = (int64)d.imag * (int64)tw.real + (int64)d.real * (int64)tw.imag;
    }
    return retVal;
}
template <>
cint64 cmpy<cint32, cint16>(cint32 d, cint16 tw, bool inv) {
    cint64 retVal;
    if (inv == true) {
        retVal.real = (int64)d.real * (int64)tw.real + (int64)d.imag * (int64)tw.imag;
        retVal.imag = (int64)d.imag * (int64)tw.real - (int64)d.real * (int64)tw.imag;
    } else {
        retVal.real = (int64)d.real * (int64)tw.real - (int64)d.imag * (int64)tw.imag;
        retVal.imag = (int64)d.imag * (int64)tw.real + (int64)d.real * (int64)tw.imag;
    }
    return retVal;
}

//-----------------------------------------------------------------
// templatized butterfly function.
template <typename TI_D, typename T_TW>
void btfly(T_int_data<TI_D>& q0,
           T_int_data<TI_D>& q1,
           T_int_data<TI_D> d0,
           T_int_data<TI_D> d1,
           T_TW tw,
           bool inv,
           unsigned int shift) {
    printf("wrong\n");
}; // assumes downshift by weight of twiddle type, e.g. 15 for cint16

template <>
void btfly<cfloat, cfloat>(T_int_data<cfloat>& q0,
                           T_int_data<cfloat>& q1,
                           T_int_data<cfloat> d0,
                           T_int_data<cfloat> d1,
                           cfloat tw,
                           bool inv,
                           unsigned int shift) {
    cfloat d1up;
    cfloat d1rot;
    d1rot.real = d0.real * tw.real - d0.imag * tw.imag;
    d1rot.imag = d0.imag * tw.real + d0.real * tw.imag;
#ifdef _DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
    printf("32 d1rot = (%f, %f)\n", d1rot.real, d1rot.imag);
#endif // _DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
    q0.real = d0.real + d1rot.real;
    q0.imag = d0.imag + d1rot.imag;
    q1.real = d0.real - d1rot.real;
    q1.imag = d0.imag - d1rot.imag;
};

template <>
void btfly<cint32, cint16>(T_int_data<cint32>& q0,
                           T_int_data<cint32>& q1,
                           T_int_data<cint32> d0,
                           T_int_data<cint32> d1,
                           cint16 tw,
                           bool inv,
                           unsigned int shift) {
    cint32 d1up;
    cint32 d1rot;
    cint64 d1rot64;
    const int64 kRoundConst = ((int64)1 << (shift - 1));
    d1up.real = (int32)d1.real;
    d1up.imag = (int32)d1.imag;
    const unsigned int shft = 15; // from cint16 for twiddle
    d1rot64 = cmpy<cint32, cint16>(d1up, tw, inv);
#ifdef _DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
    d1rot.real = (int32)(d1rot64.real);
    d1rot.imag = (int32)(d1rot64.imag);
// printf("32 d1rot = (%d, %d)\n", d1rot.real, d1rot.imag);
#endif // _DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
    q0.real = (int32)((((int64)d0.real << shft) + d1rot64.real + kRoundConst) >> shift);
    q0.imag = (int32)((((int64)d0.imag << shft) + d1rot64.imag + kRoundConst) >> shift);
    q1.real = (int32)((((int64)d0.real << shft) - d1rot64.real + kRoundConst) >> shift);
    q1.imag = (int32)((((int64)d0.imag << shft) - d1rot64.imag + kRoundConst) >> shift);
};

template <>
void btfly<cint16, cint16>(T_int_data<cint16>& q0,
                           T_int_data<cint16>& q1,
                           T_int_data<cint16> d0,
                           T_int_data<cint16> d1,
                           cint16 tw,
                           bool inv,
                           unsigned int shift) // the upshift variant
{
    cint32 d0up;
    cint32 d1up;
    cint64 d1rot64;
    cint32 d1rot;
    const int64 kRoundConst = ((int64)1 << (shift - 1));
    const unsigned int shft =
        15; // from cint16 for twiddle.Not to be confused with shift which can include scaling factor
    d0up.real = (int32)d0.real;
    d0up.imag = (int32)d0.imag;
    d1up.real = (int32)d1.real;
    d1up.imag = (int32)d1.imag;
    d1rot64 = cmpy<cint32, cint16>(d1up, tw, inv);
#ifdef _DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
    d1rot.real = (int32)(d1rot64.real);
    d1rot.imag = (int32)(d1rot64.imag);
// printf("16 round const = %d, shift = %d, d0up = (%d, %d), d1rot = (%d, %d)\n", (int32)kRoundConst, shift, d0up.real,
// d0up.imag, d1rot.real, d1rot.imag);
#endif // _DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
    q0.real = (int32)((((int64)d0up.real << shft) + d1rot64.real + kRoundConst) >> shift);
    q0.imag = (int32)((((int64)d0up.imag << shft) + d1rot64.imag + kRoundConst) >> shift);
    q1.real = (int32)((((int64)d0up.real << shft) - d1rot64.real + kRoundConst) >> shift);
    q1.imag = (int32)((((int64)d0up.imag << shft) - d1rot64.imag + kRoundConst) >> shift);
};

//---------------------------------------------------------
// templatized Radix2 stage
// First stage in DIT has trivial twiddles (1,0), but this is the one twiddle which isn't exact, so cannot be treated as
// trivial.
template <typename TT_DATA,
          typename TT_TWIDDLE, // type of coefficients           (e.g. int16, cint32)
          unsigned int TP_POINT_SIZE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
void fft_ifft_dit_1ch_ref<TT_DATA, TT_TWIDDLE, TP_POINT_SIZE, TP_FFT_NIFFT, TP_SHIFT, TP_DYN_PT_SIZE, TP_WINDOW_VSIZE>::
    r2StageInt(
        T_int_data<TT_DATA>* samplesA, T_int_data<TT_DATA>* samplesB, TT_TWIDDLE* twiddles, int pptSize, bool inv) {
#ifdef _DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
    printf("Entering r2StageInt\n");
#endif //_DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
    int ptSize = TP_DYN_PT_SIZE == 0 ? TP_POINT_SIZE : pptSize;
    T_int_data<TT_DATA> sam1, sam2, sam2rot;
    int64 sum;
    TT_TWIDDLE tw;
    const unsigned int shift = 15;
    const unsigned int round_const = (1 << (shift - 1));
    for (int op = 0; op < (ptSize >> 1); op++) {
        tw.real = twiddles[0].real;
        tw.imag = inv ? -twiddles[0].imag : twiddles[0].imag;
        sam1.real = (int64)samplesA[2 * op].real << shift;
        sam1.imag = (int64)samplesA[2 * op].imag << shift;
        sam2 = samplesA[2 * op + 1];
        if (inv) {
            sam2rot.real = (int64)sam2.real * tw.real + (int64)sam2.imag * tw.imag;
            sam2rot.imag = (int64)sam2.imag * tw.real - (int64)sam2.real * tw.imag;
        } else {
            sam2rot.real = (int64)sam2.real * tw.real - (int64)sam2.imag * tw.imag;
            sam2rot.imag = (int64)sam2.real * tw.imag + (int64)sam2.imag * tw.real;
        }
        sum = (int64)sam1.real + (int64)sam2rot.real + (int64)round_const;
        samplesB[2 * op].real = (int32)(sum >> shift);
        samplesB[2 * op].imag = (int32)(((int64)sam1.imag + (int64)sam2rot.imag + (int64)round_const) >> shift);
        samplesB[2 * op + 1].real = (int32)(((int64)sam1.real - (int64)sam2rot.real + (int64)round_const) >> shift);
        samplesB[2 * op + 1].imag = (int32)(((int64)sam1.imag - (int64)sam2rot.imag + (int64)round_const) >> shift);
        // printf("in[%d] = (%d, %d) and (%d,%d), out = [%d, %d],[%d, %d]\n",op, sam1.real, sam1.imag, sam2.real,
        // sam2.imag, samplesB[2%op].real, samplesB[2%op].imag,  samplesB[2%op+1].real, samplesB[2%op+1].imag );
    }
}

template <typename TT_DATA,
          typename TT_TWIDDLE, // type of coefficients           (e.g. int16, cint32)
          unsigned int TP_POINT_SIZE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
void fft_ifft_dit_1ch_ref<TT_DATA, TT_TWIDDLE, TP_POINT_SIZE, TP_FFT_NIFFT, TP_SHIFT, TP_DYN_PT_SIZE, TP_WINDOW_VSIZE>::
    r2StageFloat(T_int_data<TT_DATA>* samplesA,
                 T_int_data<TT_DATA>* samplesB,
                 TT_TWIDDLE* twiddles,
                 unsigned int rank,
                 int pptSize,
                 bool inv) {
    int ptSize = TP_DYN_PT_SIZE == 0 ? TP_POINT_SIZE : pptSize;
    constexpr unsigned int kRadix = 2;
    unsigned int opLowMask = (1 << rank) - 1;
    unsigned int opHiMask = ptSize - 1 - opLowMask;
    T_int_data<cfloat> sam1, sam2, sam2rot;
    unsigned int inIndex[kRadix];
    cfloat tw;
    unsigned int temp1, temp2, twIndex;
    for (int op = 0; op < (ptSize >> 1); op++) {
        for (int i = 0; i < 2; i++) {
            inIndex[i] = ((op & opHiMask) << 1) + (i << rank) + (op & opLowMask);
        }
        temp1 = inIndex[0] << (kMaxLogPtSize - 1 - rank);
        temp2 = temp1 & ((1 << (kMaxLogPtSize - 1)) - 1);
        twIndex = temp2;
        tw.real = twiddles[twIndex].real;
        tw.imag = inv ? -twiddles[twIndex].imag : twiddles[twIndex].imag;
        sam1.real = samplesA[inIndex[0]].real;
        sam1.imag = samplesA[inIndex[0]].imag;
        sam2.real = samplesA[inIndex[1]].real;
        sam2.imag = samplesA[inIndex[1]].imag;
        // sam2rot.real = sam2.real * tw.real - sam2.imag * tw.imag;
        // sam2rot.imag = sam2.real * tw.imag + sam2.imag * tw.real;
        samplesB[inIndex[0]].real = +sam1.real + (-sam2.imag * tw.imag + sam2.real * tw.real); // sam2rot.real;
        samplesB[inIndex[0]].imag = +sam1.imag + (+sam2.imag * tw.real + sam2.real * tw.imag); // sam2rot.imag;
        samplesB[inIndex[1]].real = +sam1.real + (+sam2.imag * tw.imag - sam2.real * tw.real); // sam2rot.real;
        samplesB[inIndex[1]].imag = +sam1.imag + (-sam2.imag * tw.real - sam2.real * tw.imag); // sam2rot.imag;
#ifdef _DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
        printf("index 0 = %d, index 1 = %d, twIndex = %d, tw = (%f, %f)", inIndex[0], inIndex[1], twIndex, tw.real,
               tw.imag);
//    printf("A = (%f, %f), B = (%f, %f), tw = (%f, %f), Brot = (%f, %f), resA = (%f, %f), resB = (%f, %f)\n",
//    sam1.real, sam1.imag, sam2.real, sam2.imag, tw.real, tw.imag, sam2rot.real, sam2rot.imag,
//    samplesB[inIndex[0]].real, samplesB[inIndex[0]].imag, samplesB[inIndex[1]].real, samplesB[inIndex[1]].imag );
#endif //_DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
    }
}

template <typename TT_DATA,
          typename TT_TWIDDLE, // type of coefficients           (e.g. int16, cint32)
          unsigned int TP_POINT_SIZE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
void fft_ifft_dit_1ch_ref<TT_DATA, TT_TWIDDLE, TP_POINT_SIZE, TP_FFT_NIFFT, TP_SHIFT, TP_DYN_PT_SIZE, TP_WINDOW_VSIZE>::
    r4StageInt(T_int_data<TT_DATA>* samplesIn,
               TT_TWIDDLE* twiddles1,
               TT_TWIDDLE* twiddles2,
               unsigned int n,
               unsigned int r,
               unsigned int shift,
               unsigned int rank,
               T_int_data<TT_DATA>* samplesOut,
               int pptSize,
               bool inv) {
    int ptSize = TP_DYN_PT_SIZE == 0 ? TP_POINT_SIZE : pptSize;
    constexpr unsigned int kMaxPointSize = 4096;
    constexpr unsigned int kRadix = 4;    // actually spoofed by 4 radix2 operations.
    constexpr unsigned int stdShift = 15; // derived from cint16's binary point position.
    T_int_data<TT_DATA> sam0, sam1, sam2, sam3;
    TT_TWIDDLE tw[kRadix];
    unsigned int inIndex[kRadix];
    unsigned int twIndex[kRadix];
    unsigned int temp1, temp2;
    cint32 sam2raw, sam3raw;
    cint64 sam2rot, sam3rot;
    cint64 a0, a1, a2, a3, o0, o1, o2, o3;
    T_int_data<TT_DATA> y0, y1, y2, y3;
    T_int_data<TT_DATA> yd0, yd1, yd2, yd3;
    T_int_data<TT_DATA> z0, z1, z2, z3;
    cint64 y1rot, y3rot;
    unsigned int opLowMask = (1 << rank) - 1;
    unsigned int opHiMask = ptSize - 1 - opLowMask;
    const unsigned int round_const = (1 << (shift - 1));

#ifdef _DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
// printf("opHiMask = %d, opLowMask%d\n",opHiMask, opLowMask);
// printf("calcs: (1 << (kMaxLogPtSize-rank-2)) = %d, & ((1<<(kMaxLogPtSize-1))-1) = %d = %d\n", (1 <<
// (kMaxLogPtSize-rank-2)), ((1<<(kMaxLogPtSize-1))-1), (1 << (kMaxLogPtSize-rank-2)) & ((1<<(kMaxLogPtSize-1))-1));
#endif //_DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_

    for (int op = 0; op < (ptSize >> 2); op++) {
        for (int i = 0; i < 4; i++) {
            inIndex[i] = ((op & opHiMask) << 2) + (i << rank) + (op & opLowMask);
        }
        temp1 = inIndex[0] << (kMaxLogPtSize - 1 - rank);
        temp2 = temp1 & ((1 << (kMaxLogPtSize - 1)) - 1);
        twIndex[0] = temp2;

        temp1 = inIndex[1] << (kMaxLogPtSize - 1 - rank);
        temp2 = temp1 & ((1 << (kMaxLogPtSize - 1)) - 1);
        twIndex[1] = temp2;

        temp1 = inIndex[2] << (kMaxLogPtSize - 1 - (rank + 1));
        temp2 = temp1 & ((1 << (kMaxLogPtSize - 1)) - 1);
        twIndex[2] = temp2;

        temp1 = inIndex[3] << (kMaxLogPtSize - 1 - (rank + 1));
        temp2 = temp1 & ((1 << (kMaxLogPtSize - 1)) - 1);
        twIndex[3] = temp2;
        for (int i = 0; i < 4; i++) {
            tw[i].real = twiddles1[twIndex[i]].real;
            tw[i].imag = twiddles1[twIndex[i]].imag;
        }
// for second rank butterflies, minus j intrinsic is used, but sometimes due to saturation, table entries in different
// quadrants are not exactly
// the same as each other rotated by j, so it is necessary to mimic the UUT behaviour here
#ifdef _DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
// printf("original twIndexs = %d, %d, %d, %d\n",twIndex[0], twIndex[1], twIndex[2], twIndex[3]);
#endif //_DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
        if (twIndex[2] >= kMaxPointSize >> 2) {
            twIndex[2] -= (kMaxPointSize >> 2);
            tw[2].real = twiddles2[twIndex[2]].imag;
            tw[2].imag = -twiddles2[twIndex[2]].real;
        }
        if (twIndex[3] >= kMaxPointSize >> 2) {
            twIndex[3] -= (kMaxPointSize >> 2);
            tw[3].real = twiddles2[twIndex[3]].imag;
            tw[3].imag = -twiddles2[twIndex[3]].real;
        }

#ifdef _DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
// printf("in indexes = %d, %d, %d, %d\n",inIndex[0],inIndex[1],inIndex[2],inIndex[3]);
// printf("twIndexs = %d, %d, %d, %d\n",twIndex[0], twIndex[1], twIndex[2], twIndex[3]);
// printf("tws = (%d,%d), (%d,%d), (%d,%d), (%d,%d)\n",tw[0].real, tw[0].imag, tw[1].real, tw[1].imag, tw[2].real,
// tw[2].imag, tw[3].real, tw[3].imag);
#endif //_DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_

        sam0 = samplesIn[inIndex[0]];
        sam1 = samplesIn[inIndex[1]];
        sam2 = samplesIn[inIndex[2]];
        sam3 = samplesIn[inIndex[3]];
#ifdef _DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
// printf("sam0 = (%d,%d), sam1 = (%d,%d), sam2 = (%d,%d), sam3 = (%d,%d)\n",
//       (int32)sam0.real, (int32)sam0.imag, (int32)sam1.real, (int32)sam1.imag, (int32)sam2.real, (int32)sam2.imag,
//       (int32)sam3.real, (int32)sam3.imag);
#endif //_DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
#ifdef _DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
// sam2raw.real = sam2.real;
// sam2raw.imag = sam2.imag;
// sam3raw.real = sam3.real;
// sam3raw.imag = sam3.imag;
// sam2rot = cmpy<cint32,TT_TWIDDLE>(sam2raw, tw[0]);
// sam3rot = cmpy<cint32,TT_TWIDDLE>(sam3raw, tw[1]);
// printf("sam2rot = (%d,%d), sam3rot =
// (%d,%d)\n",(int32)(sam2rot.real>>15),(int32)(sam2rot.imag>>15),(int32)(sam3rot.real>>15),(int32)(sam3rot.imag>>15));
#endif //_DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
        btfly<TT_DATA, TT_TWIDDLE>(y0, y1, sam0, sam1, tw[0], inv, stdShift);
        btfly<TT_DATA, TT_TWIDDLE>(y2, y3, sam2, sam3, tw[1], inv, stdShift);
#ifdef _DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
//    printf("End of first R2 y0  = (%d, %d), y1  = (%d, %d), y2  = (%d, %d), y3  =
//    (%d,%d)\n",y0.real,y0.imag,y1.real,y1.imag,y2.real,y2.imag,y3.real,y3.imag);
#endif //_DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
        btfly<TT_DATA, TT_TWIDDLE>(z0, z2, y0, y2, tw[2], inv, shift);
        btfly<TT_DATA, TT_TWIDDLE>(z1, z3, y1, y3, tw[3], inv, shift);
#ifdef _DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
// printf("End of second R2 z0  = (%d, %d), z1  = (%d, %d), z2  = (%d, %d), z3  =
// (%d,%d)\n",z0.real,z0.imag,z1.real,z1.imag,z2.real,z2.imag,z3.real,z3.imag);
#endif //_DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
        samplesOut[inIndex[0]] = z0;
        samplesOut[inIndex[1]] = z1;
        samplesOut[inIndex[2]] = z2;
        samplesOut[inIndex[3]] = z3;
    }
}

// Bit-accurate REF FFT DIT function
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_TWIDDLE, // type of coefficients           (e.g. int16, cint32)
          unsigned int TP_POINT_SIZE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
void fft_ifft_dit_1ch_ref<TT_DATA, TT_TWIDDLE, TP_POINT_SIZE, TP_FFT_NIFFT, TP_SHIFT, TP_DYN_PT_SIZE, TP_WINDOW_VSIZE>::
    fft(input_window<TT_DATA>* inWindow, output_window<TT_DATA>* outWindow) {
    constexpr unsigned int kMaxPtSizePwr = 12;                   // largest is 4096 = 1<<12;
    constexpr unsigned int kMinPtSizePwr = 4;                    // largest is 16 = 1<<4;
    constexpr unsigned int kHeaderSize = 32 / (sizeof(TT_DATA)); // dynamic point size header size in samples
    constexpr unsigned int kPtSizePwr = fnGetPointSizePower<TP_POINT_SIZE>();
    constexpr unsigned int kScaleFactor = kPtSizePwr - 1; // 1 is for initial rotation factor of 1/sqrt(2).
    constexpr unsigned int kSampleRanks = kRanks + 1;

    constexpr unsigned int kR2Stages =
        std::is_same<TT_DATA, cfloat>::value ? kPtSizePwr : (kPtSizePwr % 2 == 1 ? 1 : 0); // There is one radix 2 stage
                                                                                           // if we have an odd power of
                                                                                           // 2 point size, but for
                                                                                           // cfloat all stages are R2.
    constexpr unsigned int kR4Stages = std::is_same<TT_DATA, cfloat>::value ? 0 : kPtSizePwr / 2;
    constexpr unsigned int shift = 15; // unsigned weight (binary point position) of TT_TWIDDLE
    unsigned int stageShift = 0;

    TT_DATA sampleIn;
    T_int_data<TT_DATA> rotB; // Sample B after rotation.
    TT_TWIDDLE twiddle;
    TT_TWIDDLE twiddles[1 << (kMaxPtSizePwr - 1)];
    T_int_data<TT_DATA> chess_storage(% chess_alignof(cint32)) samplesStoreA[TP_POINT_SIZE];
    T_int_data<TT_DATA> chess_storage(% chess_alignof(cint32)) samplesStoreB[TP_POINT_SIZE];
    T_int_data<TT_DATA> inSampleA, inSampleB, outSampleA, outSampleB;
    unsigned int posLoMask, posHiMask;
    unsigned int twiddleMask, twiddleIndex, twiddlePos;
    unsigned int posLo, posHi;
    unsigned int posA, posB;
    unsigned int rank = 0;
    TT_DATA* headerPtr;
    TT_DATA header;
    int16 ptSizePwr =
        kPtSizePwr;      // default to static point size value. May be overwritten if dynamic point size selected.
    TT_DATA dummyttdata; // used to consume blank data in header.
    unsigned int r2Stages =
        kR2Stages; // default to static point size value. May be overwritten if dynamic point size selected.
    unsigned int r4Stages =
        kR4Stages; // default to static point size value. May be overwritten if dynamic point size selected.
    unsigned int ptSize =
        TP_POINT_SIZE; // default to static point size value. May be overwritten if dynamic point size selected.
    bool inv = TP_FFT_NIFFT == 1 ? false : true; // may be overwritten if dyn_pt_size is set
    TT_DATA headerOut;

    T_accRef<T_int_data<TT_DATA> > accum;
    T_int_data<TT_DATA> *samplesA, *samplesB, *tempPtr;
    const TT_TWIDDLE* twiddle_master = fnGetTwiddleMasterBase<TT_TWIDDLE>();

    if
        constexpr(TP_DYN_PT_SIZE == 1) {
            headerPtr = (TT_DATA*)inWindow->ptr;
            header = *headerPtr++; // saved for later when output to outWindow
            window_writeincr(outWindow, header);
            inv = header.real == 0 ? true : false;
            header = *headerPtr++; // saved for later when output to outWindow
            ptSizePwr = (int32)header.real;
            window_writeincr(outWindow, header);
            dummyttdata = nullElem<TT_DATA>();
            for (int i = 2; i < kHeaderSize - 1; i++) {
                window_writeincr(outWindow, dummyttdata);
            }
            window_writeincr(outWindow, dummyttdata); // Status word. 0 indicated all ok.
            window_incr(inWindow, kHeaderSize);
            // override values set for constant point size with values derived from the header in a dynamic point size
            // frame
            r2Stages = std::is_same<TT_DATA, cfloat>::value
                           ? ptSizePwr
                           : (ptSizePwr % 2 == 1 ? 1 : 0); // There is one radix 2 stage if we have an odd power of 2
                                                           // point size, but for cfloat all stages are R2.
            r4Stages = std::is_same<TT_DATA, cfloat>::value ? 0 : ptSizePwr / 2;
            ptSize = ((unsigned int)1) << ptSizePwr;

            // write out header
        }
    for (int opIndex = 0; opIndex < TP_WINDOW_VSIZE / TP_POINT_SIZE; opIndex++) {
        rank = 0;
        if ((ptSizePwr >= kMinPtSizePwr) && (ptSizePwr <= kMaxPtSizePwr)) {
            // read samples in
            for (unsigned int i = 0; i < ptSize; i++) {
                window_readincr(inWindow, sampleIn);
                samplesStoreA[bitRev(ptSizePwr, i)] = castInput<TT_DATA>(sampleIn);
            }
            window_incr(inWindow, TP_POINT_SIZE - ptSize);
            for (unsigned int i = 0; i < (1 << (kMaxPtSizePwr - 1)); i++) {
                twiddles[i] = twiddle_master[i];
            }

            samplesA = samplesStoreA;
            samplesB = samplesStoreB;

            for (unsigned int r2StageCnt = 0; r2StageCnt < r2Stages; r2StageCnt++) {
                if
                    constexpr(is_cfloat<TT_DATA>()) {
                        r2StageFloat(samplesA, samplesB, twiddles, r2StageCnt, ptSize, inv);
                    }
                else {
                    r2StageInt(samplesA, samplesB, twiddles, ptSize,
                               inv); // only called for the first stage so stage is implied
                }
                // Now watch carefully. The pea is under the cup labelled samplesB (the output).
                tempPtr = samplesA;
                samplesA = samplesB;
                samplesB = tempPtr;
                rank++;
                // The pea is now under the cup labelled samplesA. The next rank's input or the actual output
            }

            for (int r4StageCnt = 0; r4StageCnt < r4Stages; r4StageCnt++) {
                if (r4StageCnt == r4Stages - 1) {
                    stageShift = shift + TP_SHIFT;
                } else {
                    stageShift = shift;
                }
                r4StageInt(samplesA, twiddles, twiddles, ptSize, ptSize >> 2, stageShift, rank, samplesB, ptSize,
                           inv); //<TT_DATA,TT_TWIDDLE,TP_POINT_SIZE,TP_FFT_NIFFT>, but not required because this is a
                                 //member function.

                // Now watch carefully. The pea is under the cup labelled samplesB (the output).
                tempPtr = samplesA;
                samplesA = samplesB;
                samplesB = tempPtr;
                rank += 2;
                // The pea is now under the cup labelled samplesA. The next rank's input or the actual output
            }

            // Write samples out (natural order)
            TT_DATA outSample;
            for (unsigned int i = 0; i < ptSize; i++) {
                outSample = castOutput<TT_DATA>(samplesA[i], 0);
                window_writeincr(outWindow, outSample);
            }
            for (int i = ptSize; i < TP_POINT_SIZE; i++) {
                window_writeincr(outWindow, nullElem<TT_DATA>());
            }

        } else {                                 // ptSizePwr is out of range
            window_writeincr(outWindow, header); // pass input definition header to output
            dummyttdata = nullElem<TT_DATA>();
            for (int i = sizeof(TT_DATA) / 2; i < kHeaderSize - 1; i++) {
                window_writeincr(outWindow, dummyttdata);
            }
            dummyttdata.real = std::is_same<TT_DATA, cfloat>::value, 1.0, 1;
            window_writeincr(outWindow, dummyttdata); // error flag out
            dummyttdata = nullElem<TT_DATA>();
            // write out blank frame
            for (int i = 0; i < ptSize * sizeof(TT_DATA) / sizeof(int32); i++) {
                window_writeincr(outWindow, dummyttdata);
            }

            rank += 2;
            // The pea is now under the cup labelled samplesA. The next rank's input or the actual output
        }
    }
};

// Non Bit-accurate REF FFT DIT function
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_TWIDDLE, // type of coefficients           (e.g. int16, cint32)
          unsigned int TP_POINT_SIZE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
void fft_ifft_dit_1ch_ref<TT_DATA, TT_TWIDDLE, TP_POINT_SIZE, TP_FFT_NIFFT, TP_SHIFT, TP_DYN_PT_SIZE, TP_WINDOW_VSIZE>::
    nonBitAccfft(input_window<TT_DATA>* inWindow, output_window<TT_DATA>* outWindow) {
#ifdef _DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
    printf("Ref model params:\n");
    printf("TP_POINT_SIZE = %d\n", TP_POINT_SIZE);
    printf("TP_FFT_NIFFT = %d\n", TP_FFT_NIFFT);
#endif //_DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
    constexpr unsigned int kPtSizePwr = fnGetPointSizePower<TP_POINT_SIZE>();
    constexpr unsigned int kScaleFactor = TP_SHIFT; // was kPtSizePwr -1; //1 is for initial rotation factor of
                                                    // 1/sqrt(2), but with TP_SHIFT this is user-config
    constexpr unsigned int kSampleRanks = kRanks + 1;

#ifdef _DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_
    printf("kranks = %d, kPtSizePwr = %d, kScaleFactor = %d\n", kRanks, kPtSizePwr, kScaleFactor);
#endif

    TT_DATA sampleIn;
    T_int_data<TT_DATA> rotB; // Sample B after rotation.
    TT_TWIDDLE twiddle;
    TT_TWIDDLE twiddles[TP_POINT_SIZE / 2];
    T_int_data<TT_DATA> chess_storage(% chess_alignof(cint16)) samplesStoreA[TP_POINT_SIZE];
    T_int_data<TT_DATA> chess_storage(% chess_alignof(cint16)) samplesStoreB[TP_POINT_SIZE];
    T_int_data<TT_DATA> inSampleA, inSampleB, outSampleA, outSampleB;
    unsigned int posLoMask, posHiMask;
    unsigned int twiddleMask, twiddleIndex, twiddlePos;
    unsigned int posLo, posHi;
    unsigned int posA, posB;
    T_accRef<T_int_data<TT_DATA> > accum;
    T_int_data<TT_DATA> *samplesA, *samplesB, *tempPtr;

    // Form twiddle table for this point size;
    for (int i = 0; i < TP_POINT_SIZE / 2; i++) {
        twiddles[i] = get_twiddle<TT_TWIDDLE>(i, TP_POINT_SIZE, TP_FFT_NIFFT);
    }

    // A window may contain multiple FFT data sets. This dilutes the overheads
    for (int iter = 0; iter < TP_WINDOW_VSIZE / TP_POINT_SIZE; iter++) {
        // read samples in
        for (unsigned int i = 0; i < TP_POINT_SIZE; i++) {
            window_readincr(inWindow, sampleIn);
            samplesStoreA[i] = castInput<TT_DATA>(sampleIn);
        }

        samplesA = samplesStoreA;
        samplesB = samplesStoreB;

        // Perform FFT
        for (unsigned int rank = 0; rank < kRanks; rank++) {
            posLoMask = (1 << (kRanks - 1 - rank)) - 1;      // e.g. 000111
            posHiMask = (1 << (kRanks - 1)) - 1 - posLoMask; // e.g. 111000
            for (unsigned int op = 0; op < TP_POINT_SIZE / 2; op++) {
                posLo = op & posLoMask;
                posHi = op & posHiMask;
                posA = (posHi << 1) + 0 + posLo;
                posB = (posHi << 1) + (1 << (kRanks - 1 - rank)) + posLo;
                if (posA < TP_POINT_SIZE && posB < TP_POINT_SIZE) {
                    inSampleA = samplesA[posA];
                    inSampleB = samplesA[posB];
                }
                twiddleMask = TP_POINT_SIZE / 2 - 1;
                twiddleIndex = bitRev(kPtSizePwr - 1, op) << (kRanks - rank - 1);
                twiddlePos = twiddleIndex & twiddleMask; // << (kRanks-rank-1);
                twiddle = twiddles[twiddlePos];
                // printf("twiddlePos = %2d, twiddle = (%6d, %6d) ",
                //       twiddlePos,       twiddle.real, twiddle.imag);
                btflynonbitacc<TT_DATA, TT_TWIDDLE>(twiddle, inSampleA, inSampleB, outSampleA, outSampleB);
                if (posA < TP_POINT_SIZE && posB < TP_POINT_SIZE) {
                    samplesB[posA] = outSampleA;
                    samplesB[posB] = outSampleB;
                }
            }
            // Now watch carefully. The pea is under the cup labelled samplesB (the output).
            tempPtr = samplesA;
            samplesA = samplesB;
            samplesB = tempPtr;
            // The pea is now under the cup labelled samplesA. The next rank's input or the actual output
        }
        // Write samples out (natural order)
        for (unsigned int i = 0; i < TP_POINT_SIZE; i++) {
            window_writeincr((output_window<TT_DATA>*)outWindow,
                             castOutput<TT_DATA>(samplesA[bitRev(kPtSizePwr, i)], kScaleFactor));
        }
    }
};
}
}
}
}
}
