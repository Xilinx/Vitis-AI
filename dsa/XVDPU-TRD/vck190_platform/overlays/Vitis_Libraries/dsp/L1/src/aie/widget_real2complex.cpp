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
/*
Widget real2complex kernal code.
This file captures the body of run-time code for the kernel class.

Coding conventions
  TT_      template type suffix
  TP_      template parameter suffix
*/

#pragma once
#include <adf.h>

#define __AIEARCH__ 1
#define __AIENGINE__ 1
#define __AIE_API_USE_NATIVE_1024B_VECTOR__
#include "aie_api/aie_adf.hpp"
//#include "widget_real2complex_traits.hpp"
#include "widget_real2complex.hpp"

#include "widget_real2complex_utils.hpp"

#include "kernel_api_utils.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace widget {
namespace real2complex {

// Base specialization, used for
// real to complex (all 3 variants)
template <typename TT_DATA, typename TT_OUT_DATA, unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelClass<TT_DATA, TT_OUT_DATA, TP_WINDOW_VSIZE>::kernelClassMain(const TT_DATA* __restrict inBuff,
                                                                                     TT_OUT_DATA* __restrict outBuff) {
    using inReal128VectorType = ::aie::vector<TT_DATA, 128 / 8 / sizeof(TT_DATA)>;
    using inReal256VectorType = ::aie::vector<TT_DATA, 256 / 8 / sizeof(TT_DATA)>;
    using outCplx256VectorType = ::aie::vector<TT_OUT_DATA, 256 / 8 / sizeof(TT_OUT_DATA)>;

    constexpr unsigned int inStep = 16 / sizeof(TT_DATA);
    constexpr unsigned int outStep = 32 / sizeof(TT_OUT_DATA);
    constexpr unsigned int kLsize = TP_WINDOW_VSIZE / inStep;

    const inReal128VectorType inZeroes = ::aie::zeros<TT_DATA, 128 / 8 / sizeof(TT_DATA)>();
    inReal128VectorType inReal;
    ::std::pair<inReal128VectorType, inReal128VectorType> inRealIntlv;
    outCplx256VectorType outCplx;
    inReal256VectorType realLarge;

    inReal128VectorType* __restrict inPtr = (inReal128VectorType*)inBuff;
    outCplx256VectorType* __restrict outPtr = (outCplx256VectorType*)outBuff;

    for (int i = 0; i < kLsize; i++) chess_prepare_for_pipelining chess_loop_range(kLsize, ) {
            inReal = *inPtr++; // load
            inRealIntlv =
                ::aie::interleave_zip(inReal, inZeroes, 1); // convert to complex by interleaving zeros for imag parts
            realLarge = ::aie::concat<inReal128VectorType, inReal128VectorType>(inRealIntlv.first, inRealIntlv.second);
            outCplx = ::aie::vector_cast<TT_OUT_DATA, inReal256VectorType>(realLarge); // cast
            *outPtr++ = outCplx;
        }
};

template <unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelClass<cint16, int16, TP_WINDOW_VSIZE>::kernelClassMain(const cint16* __restrict inBuff,
                                                                              int16* __restrict outBuff) {
    typedef cint16 TT_DATA;
    typedef int16 TT_OUT_DATA;
    using inCplx256VectorType = ::aie::vector<TT_DATA, 256 / 8 / sizeof(TT_DATA)>;
    using outReal256VectorType = ::aie::vector<TT_OUT_DATA, 256 / 8 / sizeof(TT_OUT_DATA)>;
    using outReal128VectorType = ::aie::vector<TT_OUT_DATA, 128 / 8 / sizeof(TT_OUT_DATA)>;

    constexpr unsigned int inStep = 32 / sizeof(TT_DATA);      // numsamples in 256b read
    constexpr unsigned int outStep = 16 / sizeof(TT_OUT_DATA); // numsamples in 128b write
    constexpr unsigned int kLsize = TP_WINDOW_VSIZE / inStep;
    inCplx256VectorType inCplx;
    outReal256VectorType realLarge;
    outReal128VectorType outReal;
    inCplx256VectorType* __restrict inPtr = (inCplx256VectorType*)inBuff;
    outReal128VectorType* __restrict outPtr = (outReal128VectorType*)outBuff;

    for (int i = 0; i < kLsize; i++) chess_prepare_for_pipelining chess_loop_range(kLsize, ) {
            inCplx = *inPtr++;                                                        // load
            realLarge = ::aie::vector_cast<TT_OUT_DATA, inCplx256VectorType>(inCplx); // convert to real
            outReal = ::aie::filter_even<outReal256VectorType>(realLarge);            // cast
            *outPtr++ = outReal;
        }
};

template <unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelClass<cint32, int32, TP_WINDOW_VSIZE>::kernelClassMain(const cint32* __restrict inBuff,
                                                                              int32* __restrict outBuff) {
    typedef cint32 TT_DATA;
    typedef int32 TT_OUT_DATA;
    using inCplx256VectorType = ::aie::vector<TT_DATA, 256 / 8 / sizeof(TT_DATA)>;
    using outReal256VectorType = ::aie::vector<TT_OUT_DATA, 256 / 8 / sizeof(TT_OUT_DATA)>;
    using outReal128VectorType = ::aie::vector<TT_OUT_DATA, 128 / 8 / sizeof(TT_OUT_DATA)>;

    constexpr unsigned int inStep = 32 / sizeof(TT_DATA);      // numsamples in 256b read
    constexpr unsigned int outStep = 16 / sizeof(TT_OUT_DATA); // numsamples in 128b write
    constexpr unsigned int kLsize = TP_WINDOW_VSIZE / inStep;
    inCplx256VectorType inCplx;
    outReal256VectorType realLarge;
    outReal128VectorType outReal;
    inCplx256VectorType* __restrict inPtr = (inCplx256VectorType*)inBuff;
    outReal128VectorType* __restrict outPtr = (outReal128VectorType*)outBuff;

    for (int i = 0; i < kLsize; i++) chess_prepare_for_pipelining chess_loop_range(kLsize, ) {
            inCplx = *inPtr++;                                                        // load
            realLarge = ::aie::vector_cast<TT_OUT_DATA, inCplx256VectorType>(inCplx); // convert to real
            outReal = ::aie::filter_even<outReal256VectorType>(realLarge);            // cast
            // outReal = realLarge.template extract<outStep>(0); //dummy code for sake of CRVO
            *outPtr++ = outReal;
        }
};

template <unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelClass<cfloat, float, TP_WINDOW_VSIZE>::kernelClassMain(const cfloat* __restrict inBuff,
                                                                              float* __restrict outBuff) {
    typedef cfloat TT_DATA;
    typedef float TT_OUT_DATA;
    using inCplx256VectorType = ::aie::vector<TT_DATA, 256 / 8 / sizeof(TT_DATA)>;
    using outReal256VectorType = ::aie::vector<TT_OUT_DATA, 256 / 8 / sizeof(TT_OUT_DATA)>;
    using outReal128VectorType = ::aie::vector<TT_OUT_DATA, 128 / 8 / sizeof(TT_OUT_DATA)>;

    constexpr unsigned int inStep = 32 / sizeof(TT_DATA);      // numsamples in 256b read
    constexpr unsigned int outStep = 16 / sizeof(TT_OUT_DATA); // numsamples in 128b write
    constexpr unsigned int kLsize = TP_WINDOW_VSIZE / inStep;
    inCplx256VectorType inCplx;
    outReal256VectorType realLarge;
    outReal128VectorType outReal;
    inCplx256VectorType* __restrict inPtr = (inCplx256VectorType*)inBuff;
    outReal128VectorType* __restrict outPtr = (outReal128VectorType*)outBuff;

    for (int i = 0; i < kLsize; i++) chess_prepare_for_pipelining chess_loop_range(kLsize, ) {
            inCplx = *inPtr++;                                                        // load
            realLarge = ::aie::vector_cast<TT_OUT_DATA, inCplx256VectorType>(inCplx); // convert to real
            outReal = ::aie::filter_even<outReal256VectorType>(realLarge);            // cast
            //    outReal = realLarge.template extract<outStep>(0); //dummy code for sake of CRVO
            *outPtr++ = outReal;
        }
};

//-------------------------------------------------------------------------------------------------------
// This is the base specialization of the main class for when there is
template <typename TT_DATA,
          typename TT_OUT_DATA,
          unsigned int TP_WINDOW_VSIZE>
__attribute__((noinline))   //This function is the hook for QoR profiling, so must be identifiable after compilation.
void widget_real2complex<TT_DATA, TT_OUT_DATA, TP_WINDOW_VSIZE>::convertData
                (input_window<TT_DATA>* __restrict inWindow0,
                 output_window<TT_OUT_DATA>* __restrict outWindow0
                )
    {
    TT_DATA* inPtr = (TT_DATA*)inWindow0->ptr;
    TT_OUT_DATA* outPtr = (TT_OUT_DATA*)outWindow0->ptr;
    this->kernelClassMain(inPtr, outPtr);
};
}
}
}
}
}
