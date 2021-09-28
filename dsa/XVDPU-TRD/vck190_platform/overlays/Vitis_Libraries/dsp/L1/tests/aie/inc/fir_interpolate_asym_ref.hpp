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
#ifndef _DSPLIB_FIR_INTERPOLATE_ASYM_REF_HPP_
#define _DSPLIB_FIR_INTERPOLATE_ASYM_REF_HPP_

// This file holds the definition of the Asymmetric Interpolation FIR reference model kernel class.
// Design Notes
// Note that the AIE intrinsics operate on increasing indices, but in a conventional FIR there is a convolution of data
// and coefficients.
// So as to achieve the impulse response from the filter which matches the coefficeint set, the coefficient array has to
// be reversed
// to compensate for the action of the intrinsics. This reversal is performed in the constructor where possible, or in
// the reference
// model's run-time filter function.

#include <adf.h>
#include <limits>
#include "fir_ref_utils.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace interpolate_asym {

//-----------------------------------------------------------------------------------------------------
// Interpolate Asym Reference Model Class - static coefficients, single output
template <typename TT_DATA,  // type of data input and output
          typename TT_COEFF, // type of coefficients           (e.g. int16, cint32)
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_USE_COEFF_RELOAD = 0, // 1 = use coeff reload, 0 = don't use coeff reload
          unsigned int TP_NUM_OUTPUTS = 1>
class fir_interpolate_asym_ref {
   public:
    // Constructor
    fir_interpolate_asym_ref(const TT_COEFF (&taps)[TP_FIR_LEN]) {
        // This reference model uses taps directly. It does not need to pad the taps array
        // to the column width because the concept of columns does not apply to the ref model.
        for (int i = 0; i < FIR_LEN; ++i) {
            m_internalTapsRef[i] = taps[FIR_LEN - 1 - i];
        }
    }
    // Register Kernel Class
    static void registerKernelClass() {
        REGISTER_FUNCTION(fir_interpolate_asym_ref::filter);
        // REGISTER_PARAMETER(m_internalTapsRef);
    }
    // FIR
    void filter(input_window<TT_DATA>* inWindow, output_window<TT_DATA>* outWindow);

   private:
    TT_COEFF chess_storage(% chess_alignof(v8cint16)) m_internalTapsRef[TP_FIR_LEN];
};

// Interpolate Asym Reference Model Class - static coefficients, dual output
template <typename TT_DATA,  // type of data input and output
          typename TT_COEFF, // type of coefficients           (e.g. int16, cint32)
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
class fir_interpolate_asym_ref<TT_DATA,
                               TT_COEFF,
                               TP_FIR_LEN,
                               TP_INTERPOLATE_FACTOR,
                               TP_SHIFT,
                               TP_RND,
                               TP_INPUT_WINDOW_VSIZE,
                               USE_COEFF_RELOAD_FALSE,
                               2> {
   public:
    // Constructor
    fir_interpolate_asym_ref(const TT_COEFF (&taps)[TP_FIR_LEN]) {
        // This reference model uses taps directly. It does not need to pad the taps array
        // to the column width because the concept of columns does not apply to the ref model.
        for (int i = 0; i < FIR_LEN; ++i) {
            m_internalTapsRef[i] = taps[FIR_LEN - 1 - i];
        }
    }
    // Register Kernel Class
    static void registerKernelClass() {
        REGISTER_FUNCTION(fir_interpolate_asym_ref::filter);
        // REGISTER_PARAMETER(m_internalTapsRef);
    }
    // FIR
    void filter(input_window<TT_DATA>* inWindow, output_window<TT_DATA>* outWindow, output_window<TT_DATA>* outWindow2);

   private:
    TT_COEFF chess_storage(% chess_alignof(v8cint16)) m_internalTapsRef[TP_FIR_LEN];
};

//-----------------------------------------------------------------------------------------------------
// Interpolate Asym Reference Model Class - reloadable coefficients, single output
template <typename TT_DATA,  // type of data input and output
          typename TT_COEFF, // type of coefficients           (e.g. int16, cint32)
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
class fir_interpolate_asym_ref<TT_DATA,
                               TT_COEFF,
                               TP_FIR_LEN,
                               TP_INTERPOLATE_FACTOR,
                               TP_SHIFT,
                               TP_RND,
                               TP_INPUT_WINDOW_VSIZE,
                               USE_COEFF_RELOAD_TRUE,
                               1> {
   public:
    // Constructor
    fir_interpolate_asym_ref() {}
    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_asym_ref::filter); }
    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_window<TT_DATA>* outWindow,
                const TT_COEFF (&inTaps)[TP_FIR_LEN]);

   private:
    TT_COEFF chess_storage(% chess_alignof(v8cint16)) m_internalTapsRef[TP_FIR_LEN];
};

// Interpolate Asym Reference Model Class - reloadable coefficients, dual output
template <typename TT_DATA,  // type of data input and output
          typename TT_COEFF, // type of coefficients           (e.g. int16, cint32)
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
class fir_interpolate_asym_ref<TT_DATA,
                               TT_COEFF,
                               TP_FIR_LEN,
                               TP_INTERPOLATE_FACTOR,
                               TP_SHIFT,
                               TP_RND,
                               TP_INPUT_WINDOW_VSIZE,
                               USE_COEFF_RELOAD_TRUE,
                               2> {
   public:
    // Constructor
    fir_interpolate_asym_ref() {}
    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_asym_ref::filter); }
    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_window<TT_DATA>* outWindow,
                output_window<TT_DATA>* outWindow2,
                const TT_COEFF (&inTaps)[TP_FIR_LEN]);

   private:
    TT_COEFF chess_storage(% chess_alignof(v8cint16)) m_internalTapsRef[TP_FIR_LEN];
};
}
}
}
}
}

#endif // fir_interpolate_asym_ref
