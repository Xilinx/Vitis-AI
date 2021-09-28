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
#ifndef _DSPLIB_fir_sr_asym_REF_HPP_
#define _DSPLIB_fir_sr_asym_REF_HPP_

/*
Single rate asymetric FIR filter reference model
*/

#include <adf.h>
#include <limits>
#include "fir_ref_utils.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace sr_asym {

//-----------------------------------------------------------------------------------------------------
// Single Rate class
// Static coefficients
template <typename TT_DATA,  // type of data input and output
          typename TT_COEFF, // type of coefficients           (e.g. int16, cint32)
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_USE_COEFF_RELOAD = 0, // 1 = use coeff reload, 0 = don't use coeff reload
          unsigned int TP_NUM_OUTPUTS = 1>
class fir_sr_asym_ref {
   private:
    TT_COEFF internalTaps[TP_FIR_LEN] = {};

   public:
    // Constructor
    fir_sr_asym_ref(const TT_COEFF (&taps)[TP_FIR_LEN]) {
        for (int i = 0; i < TP_FIR_LEN; i++) {
            internalTaps[i] = taps[i];
        }
    }

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_asym_ref::filter); }
    // FIR
    void filter(input_window<TT_DATA>* inWindow, output_window<TT_DATA>* outWindow);
};

// static coefficients, dual output
template <typename TT_DATA,  // type of data input and output
          typename TT_COEFF, // type of coefficients           (e.g. int16, cint32)
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
class fir_sr_asym_ref<TT_DATA,
                      TT_COEFF,
                      TP_FIR_LEN,
                      TP_SHIFT,
                      TP_RND,
                      TP_INPUT_WINDOW_VSIZE,
                      0 /* USE_COEFF_RELOAD_FALSE*/,
                      2> {
   private:
    TT_COEFF internalTaps[TP_FIR_LEN] = {};

   public:
    // Constructor
    fir_sr_asym_ref(const TT_COEFF (&taps)[TP_FIR_LEN]) {
        for (int i = 0; i < TP_FIR_LEN; i++) {
            internalTaps[i] = taps[i];
        }
    }

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_asym_ref::filter); }
    // FIR
    void filter(input_window<TT_DATA>* inWindow, output_window<TT_DATA>* outWindow, output_window<TT_DATA>* outWindow2);
};
//-----------------------------------------------------------------------------------------------------
// Single Rate class
// Reloadable coefficients, single output
template <typename TT_DATA,  // type of data input and output
          typename TT_COEFF, // type of coefficients           (e.g. int16, cint32)
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
class fir_sr_asym_ref<TT_DATA,
                      TT_COEFF,
                      TP_FIR_LEN,
                      TP_SHIFT,
                      TP_RND,
                      TP_INPUT_WINDOW_VSIZE,
                      1 /*USE_COEFF_RELOAD_TRUE*/,
                      1> {
   private:
    TT_COEFF internalTaps[TP_FIR_LEN] = {};

   public:
    // Constructor
    fir_sr_asym_ref() {}

    void firReload(const TT_COEFF (&taps)[TP_FIR_LEN]) {
        for (int i = 0; i < TP_FIR_LEN; i++) {
            internalTaps[i] = taps[i];
        }
    }

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_asym_ref::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_window<TT_DATA>* outWindow,
                const TT_COEFF (&inTaps)[TP_FIR_LEN]);
};

// Specialization for Reloadable coefficients, dual  output
template <typename TT_DATA,  // type of data input and output
          typename TT_COEFF, // type of coefficients           (e.g. int16, cint32)
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
class fir_sr_asym_ref<TT_DATA,
                      TT_COEFF,
                      TP_FIR_LEN,
                      TP_SHIFT,
                      TP_RND,
                      TP_INPUT_WINDOW_VSIZE,
                      1 /*USE_COEFF_RELOAD_TRUE*/,
                      2> {
   private:
    TT_COEFF internalTaps[TP_FIR_LEN] = {};

   public:
    // Constructor
    fir_sr_asym_ref() {}

    void firReload(const TT_COEFF (&taps)[TP_FIR_LEN]) {
        for (int i = 0; i < TP_FIR_LEN; i++) {
            internalTaps[i] = taps[i];
        }
    }

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_asym_ref::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_window<TT_DATA>* outWindow,
                output_window<TT_DATA>* outWindow2,
                const TT_COEFF (&inTaps)[TP_FIR_LEN]);
};
}
}
}
}
}

#endif // _DSPLIB_fir_sr_asym_REF_HPP_
