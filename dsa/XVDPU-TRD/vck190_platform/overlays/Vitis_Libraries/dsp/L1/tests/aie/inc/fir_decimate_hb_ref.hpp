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
#ifndef _DSPLIB_fir_decimate_hb_REF_HPP_
#define _DSPLIB_fir_decimate_hb_REF_HPP_

/*
Halfband decimation FIR Kernel Reference model.
This file holds the declaration of the reference model class. The reference model
is functionally equivalent to the kernel class with intrinics. The reference model
does not use intrinsics or vector operations. The reference mode, once validated
acts as the golden reference to verify the AIE-targetting kernel class.
*/

#include <adf.h>
#include <limits>
#include "fir_ref_utils.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace decimate_hb {

//-----------------------------------------------------------------------------------------------------
// Behavioural model class for Halfband decimation FIR, static coeffs, single output
template <typename TT_DATA,  // type of data input and output
          typename TT_COEFF, // type of coefficients           (e.g. int16, cint32)
          size_t TP_FIR_LEN,
          size_t TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_USE_COEFF_RELOAD = 0, // 1 = use coeff reload, 0 = don't use coeff reload
          unsigned int TP_NUM_OUTPUTS = 1>
class fir_decimate_hb_ref {
   private:
    TT_COEFF m_internalTaps[TP_FIR_LEN];
    static constexpr unsigned int m_kCentreTapInputPos =
        (TP_FIR_LEN + 1) / 4; // e.g.for 11 taps, 3 taps then ct are given. 11+1/4 gives index 3.
    static constexpr unsigned int m_kCentreTapInternalPos =
        TP_FIR_LEN / 2; // e.g.for 11 taps (with zeros), centre tap is index 5.
    static constexpr unsigned int m_kDataSampleCentre = TP_FIR_LEN / 4; // Index of data sample for centre tap
    bool m_useCentreTapShift = false;
    unsigned int m_ctShift = 0;

   public:
    // Constructor
    // The constructor here reads only as far as the centre tap. Given that this is a symmetrical FIR
    // the constructor constructs a full array of coefficients from a sparse array. The sparse array is
    // only the first half of the taps array, since it is symmetrical, and only the non-zero values since
    // this is a half band.
    // e.g. for input of (1, 2, 3, 64) the constructor will have an 11 tap array of (1, 0, 2, 0, 3, 64, 4, 0, 2, 0, 1)
    // In this variant of the constructor the centre tap is expected and may be denormalized.
    fir_decimate_hb_ref(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / 4 + 1]);

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb_ref::filter); }
    // FIR
    void filter(input_window<TT_DATA>* inWindow, output_window<TT_DATA>* outWindow);
};

// Specialized for static coefficients, dual output
template <typename TT_DATA,  // type of data input and output
          typename TT_COEFF, // type of coefficients           (e.g. int16, cint32)
          size_t TP_FIR_LEN,
          size_t TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
class fir_decimate_hb_ref<TT_DATA,
                          TT_COEFF,
                          TP_FIR_LEN,
                          TP_SHIFT,
                          TP_RND,
                          TP_INPUT_WINDOW_VSIZE,
                          USE_COEFF_RELOAD_FALSE,
                          2> {
   private:
    TT_COEFF m_internalTaps[TP_FIR_LEN];
    static constexpr unsigned int m_kCentreTapInputPos =
        (TP_FIR_LEN + 1) / 4; // e.g.for 11 taps, 3 taps then ct are given. 11+1/4 gives index 3.
    static constexpr unsigned int m_kCentreTapInternalPos =
        TP_FIR_LEN / 2; // e.g.for 11 taps (with zeros), centre tap is index 5.
    static constexpr unsigned int m_kDataSampleCentre = TP_FIR_LEN / 4; // Index of data sample for centre tap
    bool m_useCentreTapShift = false;
    unsigned int m_ctShift = 0;

   public:
    // Constructor
    // The constructor here reads only as far as the centre tap. Given that this is a symmetrical FIR
    // the constructor constructs a full array of coefficients from a sparse array. The sparse array is
    // only the first half of the taps array, since it is symmetrical, and only the non-zero values since
    // this is a half band.
    // e.g. for input of (1, 2, 3, 64) the constructor will have an 11 tap array of (1, 0, 2, 0, 3, 64, 4, 0, 2, 0, 1)
    // In this variant of the constructor the centre tap is expected and may be denormalized.
    fir_decimate_hb_ref(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / 4 + 1]);

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb_ref::filter); }
    // FIR
    void filter(input_window<TT_DATA>* inWindow, output_window<TT_DATA>* outWindow, output_window<TT_DATA>* outWindow2);
};

// Specialized for reloadable coefficients, single output
template <typename TT_DATA,  // type of data input and output
          typename TT_COEFF, // type of coefficients           (e.g. int16, cint32)
          size_t TP_FIR_LEN,
          size_t TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
class fir_decimate_hb_ref<TT_DATA,
                          TT_COEFF,
                          TP_FIR_LEN,
                          TP_SHIFT,
                          TP_RND,
                          TP_INPUT_WINDOW_VSIZE,
                          USE_COEFF_RELOAD_TRUE,
                          1> {
   private:
    TT_COEFF m_internalTaps[TP_FIR_LEN];
    static constexpr unsigned int m_kCentreTapInputPos =
        (TP_FIR_LEN + 1) / 4; // e.g.for 11 taps, 3 taps then ct are given. 11+1/4 gives index 3.
    static constexpr unsigned int m_kCentreTapInternalPos =
        TP_FIR_LEN / 2; // e.g.for 11 taps (with zeros), centre tap is index 5.
    static constexpr unsigned int m_kDataSampleCentre = TP_FIR_LEN / 4; // Index of data sample for centre tap
    bool m_useCentreTapShift = false;
    unsigned int m_ctShift = 0;

   public:
    // Constructor
    // The constructor here reads only as far as the centre tap. Given that this is a symmetrical FIR
    // the constructor constructs a full array of coefficients from a sparse array. The sparse array is
    // only the first half of the taps array, since it is symmetrical, and only the non-zero values since
    // this is a half band.
    // e.g. for input of (1, 2, 3, 64) the constructor will have an 11 tap array of (1, 0, 2, 0, 3, 64, 4, 0, 2, 0, 1)
    // In this variant of the constructor the centre tap is expected and may be denormalized.
    fir_decimate_hb_ref();

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb_ref::filter); }
    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_window<TT_DATA>* outWindow,
                const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / 4 + 1]);
};

// Specialized for reloadable coefficients, dual output
template <typename TT_DATA,  // type of data input and output
          typename TT_COEFF, // type of coefficients           (e.g. int16, cint32)
          size_t TP_FIR_LEN,
          size_t TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
class fir_decimate_hb_ref<TT_DATA,
                          TT_COEFF,
                          TP_FIR_LEN,
                          TP_SHIFT,
                          TP_RND,
                          TP_INPUT_WINDOW_VSIZE,
                          USE_COEFF_RELOAD_TRUE,
                          2> {
   private:
    TT_COEFF m_internalTaps[TP_FIR_LEN];
    static constexpr unsigned int m_kCentreTapInputPos =
        (TP_FIR_LEN + 1) / 4; // e.g.for 11 taps, 3 taps then ct are given. 11+1/4 gives index 3.
    static constexpr unsigned int m_kCentreTapInternalPos =
        TP_FIR_LEN / 2; // e.g.for 11 taps (with zeros), centre tap is index 5.
    static constexpr unsigned int m_kDataSampleCentre = TP_FIR_LEN / 4; // Index of data sample for centre tap
    bool m_useCentreTapShift = false;
    unsigned int m_ctShift = 0;

   public:
    // Constructor
    // The constructor here reads only as far as the centre tap. Given that this is a symmetrical FIR
    // the constructor constructs a full array of coefficients from a sparse array. The sparse array is
    // only the first half of the taps array, since it is symmetrical, and only the non-zero values since
    // this is a half band.
    // e.g. for input of (1, 2, 3, 64) the constructor will have an 11 tap array of (1, 0, 2, 0, 3, 64, 4, 0, 2, 0, 1)
    // In this variant of the constructor the centre tap is expected and may be denormalized.
    fir_decimate_hb_ref();

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb_ref::filter); }
    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_window<TT_DATA>* outWindow,
                output_window<TT_DATA>* outWindow2,
                const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / 4 + 1]);
};
}
}
}
}
}

#endif // _DSPLIB_fir_decimate_hb_REF_HPP_
