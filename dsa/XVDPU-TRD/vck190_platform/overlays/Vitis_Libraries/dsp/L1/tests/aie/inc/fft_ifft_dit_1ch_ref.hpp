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
#ifndef _DSPLIB_FFT_IFFT_DIT_1CH_REF_HPP_
#define _DSPLIB_FFT_IFFT_DIT_1CH_REF_HPP_

/*
FFT/iFFT DIT single channel reference model
*/

//#define _DSPLIB_FFT_IFFT_DIT_1CH_REF_DEBUG_

#include <adf.h>
#include <limits>
namespace xf {
namespace dsp {
namespace aie {
namespace fft {
namespace dit_1ch {

constexpr int kFftDynHeadBytes = 32;

//---------------------------------
// Templatized types
template <typename T_D>
struct T_int_data {};
template <>
struct T_int_data<int16> {
    int32 real;
    int32 imag;
};
template <>
struct T_int_data<cint16> {
    int32 real;
    int32 imag;
};
template <>
struct T_int_data<int32> {
    int32 real;
    int32 imag;
};
template <>
struct T_int_data<cint32> {
    int32 real;
    int32 imag;
};
template <>
struct T_int_data<float> {
    float real;
    float imag;
};
template <>
struct T_int_data<cfloat> {
    float real;
    float imag;
};

template <typename T_D>
struct T_accfftRef {};
template <>
struct T_accfftRef<int16> {
    int32 real;
    int32 imag;
};
template <>
struct T_accfftRef<cint16> {
    int32 real;
    int32 imag;
};
template <>
struct T_accfftRef<int32> {
    int32 real;
    int32 imag;
};
template <>
struct T_accfftRef<cint32> {
    int32 real;
    int32 imag;
};
template <>
struct T_accfftRef<float> {
    float real;
    float imag;
};
template <>
struct T_accfftRef<cfloat> {
    float real;
    float imag;
};

// Fn to perform log2 on TP_POINT_SIZE to get #ranks
template <unsigned int TP_POINT_SIZE>
inline constexpr unsigned int fnGetPointSizePower() {
    return TP_POINT_SIZE == 16
               ? 4
               : TP_POINT_SIZE == 32
                     ? 5
                     : TP_POINT_SIZE == 64
                           ? 6
                           : TP_POINT_SIZE == 128
                                 ? 7
                                 : TP_POINT_SIZE == 256
                                       ? 8
                                       : TP_POINT_SIZE == 512
                                             ? 9
                                             : TP_POINT_SIZE == 1024
                                                   ? 10
                                                   : TP_POINT_SIZE == 2048 ? 11 : TP_POINT_SIZE == 4096 ? 12 : 0;
}

//-----------------------------------------------------------------------------------------------------
// FFT/iFFT DIT single channel reference model class
template <typename TT_DATA,    // type of data input and output
          typename TT_TWIDDLE, // type of twiddle factor
          unsigned int TP_POINT_SIZE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE = TP_POINT_SIZE>
class fft_ifft_dit_1ch_ref {
   private:
    TT_TWIDDLE twiddles[TP_POINT_SIZE];
    static constexpr unsigned int kMaxLogPtSize = 12;
    static constexpr unsigned int kRanks =
        fnGetPointSizePower<TP_POINT_SIZE>(); // while each rank is radix2 this is true
    void r2StageInt(
        T_int_data<TT_DATA>* samplesA, T_int_data<TT_DATA>* samplesB, TT_TWIDDLE* twiddles, int pptSize, bool inv);
    void r2StageFloat(T_int_data<TT_DATA>* samplesA,
                      T_int_data<TT_DATA>* samplesB,
                      TT_TWIDDLE* twiddles,
                      unsigned int rank,
                      int pptSize,
                      bool inv);
    void r4StageInt(T_int_data<TT_DATA>* samplesIn,
                    TT_TWIDDLE* twiddles1,
                    TT_TWIDDLE* twiddles2,
                    unsigned int n,
                    unsigned int r,
                    unsigned int shift,
                    unsigned int rank,
                    T_int_data<TT_DATA>* samplesOut,
                    int pptSize,
                    bool inv);

   public:
    // Constructor
    fft_ifft_dit_1ch_ref() {}
    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fft_ifft_dit_1ch_ref::fft); }
    // FFT
    void fft(input_window<TT_DATA>* inWindow, output_window<TT_DATA>* outWindow);
    void nonBitAccfft(input_window<TT_DATA>* inWindow, output_window<TT_DATA>* outWindow);
};
}
}
}
}
} // namespace closing braces

#endif // _DSPLIB_FFT_IFFT_DIT_1CH_REF_HPP_
