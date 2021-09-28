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
#include "fir_interpolate_asym_ref.hpp"
#include "fir_ref_utils.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace interpolate_asym {
// body of the Asymmetric Interpolation FIR reference model kernel class, static coeffs, single output
template <typename TT_DATA,  // type of data input and output
          typename TT_COEFF, // type of coefficients           (e.g. int16, cint32)
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_USE_COEFF_RELOAD,
          unsigned int TP_NUM_OUTPUTS>
void fir_interpolate_asym_ref<TT_DATA,
                              TT_COEFF,
                              TP_FIR_LEN,
                              TP_INTERPOLATE_FACTOR,
                              TP_SHIFT,
                              TP_RND,
                              TP_INPUT_WINDOW_VSIZE,
                              TP_USE_COEFF_RELOAD,
                              TP_NUM_OUTPUTS>::filter(input_window<TT_DATA>* inWindow,
                                                      output_window<TT_DATA>* outWindow) {
    const unsigned int shift = TP_SHIFT;
    T_accRef<TT_DATA> accum;
    TT_DATA d_in[TP_FIR_LEN / TP_INTERPOLATE_FACTOR];
    TT_DATA accum_srs;

    printf("Ref model params:\n");
    printf("TP_INTERPOLATE_FACTOR = %d\n", TP_INTERPOLATE_FACTOR);
    printf("TP_SHIFT = %lu\n", TP_SHIFT);
    printf("TP_RND = %d\n", TP_RND);
    printf("TP_INPUT_WINDOW_SIZE = %d\n", TP_INPUT_WINDOW_VSIZE);
    const unsigned int kFirLen = TP_FIR_LEN / TP_INTERPOLATE_FACTOR;
    const unsigned int kFirMarginOffset = fnFirMargin<kFirLen, TT_DATA>() - kFirLen + 1; // FIR Margin Offset.
    window_incr(inWindow, kFirMarginOffset); // move input data pointer past the margin padding

    for (unsigned int i = 0; i < TP_INPUT_WINDOW_VSIZE; i++) {
        for (unsigned int j = 0; j < TP_FIR_LEN / TP_INTERPOLATE_FACTOR; ++j) {
            d_in[j] = window_readincr(inWindow); // read input data
        }
        for (int k = TP_INTERPOLATE_FACTOR - 1; k >= 0; --k) {
            accum = null_accRef<TT_DATA>(); // reset accumulator at the start of the mult-add for each output sample
            for (unsigned int j = 0; j < TP_FIR_LEN / TP_INTERPOLATE_FACTOR; ++j) {
                multiplyAcc<TT_DATA, TT_COEFF>(accum, d_in[j], m_internalTapsRef[j * TP_INTERPOLATE_FACTOR + k]);
            }
            // prior to output, the final accumulated value must be downsized to the same type
            // as was input. To do this, the final result is rounded, saturated and shifted down
            roundAcc(TP_RND, shift, accum);
            saturateAcc(accum);
            accum_srs = castAcc(accum);
            window_writeincr((output_window<TT_DATA>*)outWindow, accum_srs);
        }
        // Revert data pointer for next sample
        window_decr(inWindow, TP_FIR_LEN / TP_INTERPOLATE_FACTOR - 1);
    }
};

// specialization, static coeffs, dual output
template <typename TT_DATA,  // type of data input and output
          typename TT_COEFF, // type of coefficients           (e.g. int16, cint32)
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
void fir_interpolate_asym_ref<TT_DATA,
                              TT_COEFF,
                              TP_FIR_LEN,
                              TP_INTERPOLATE_FACTOR,
                              TP_SHIFT,
                              TP_RND,
                              TP_INPUT_WINDOW_VSIZE,
                              USE_COEFF_RELOAD_FALSE,
                              2>::filter(input_window<TT_DATA>* inWindow,
                                         output_window<TT_DATA>* outWindow,
                                         output_window<TT_DATA>* outWindow2) {
    const unsigned int shift = TP_SHIFT;
    T_accRef<TT_DATA> accum;
    TT_DATA d_in[TP_FIR_LEN / TP_INTERPOLATE_FACTOR];
    TT_DATA accum_srs;

    printf("Ref model params:\n");
    printf("TP_INTERPOLATE_FACTOR = %d\n", TP_INTERPOLATE_FACTOR);
    printf("TP_SHIFT = %lu\n", TP_SHIFT);
    printf("TP_RND = %d\n", TP_RND);
    printf("TP_INPUT_WINDOW_SIZE = %d\n", TP_INPUT_WINDOW_VSIZE);
    const unsigned int kFirLen = TP_FIR_LEN / TP_INTERPOLATE_FACTOR;
    const unsigned int kFirMarginOffset = fnFirMargin<kFirLen, TT_DATA>() - kFirLen + 1; // FIR Margin Offset.
    window_incr(inWindow, kFirMarginOffset); // move input data pointer past the margin padding

    for (unsigned int i = 0; i < TP_INPUT_WINDOW_VSIZE; i++) {
        for (unsigned int j = 0; j < TP_FIR_LEN / TP_INTERPOLATE_FACTOR; ++j) {
            d_in[j] = window_readincr(inWindow); // read input data
        }
        for (int k = TP_INTERPOLATE_FACTOR - 1; k >= 0; --k) {
            accum = null_accRef<TT_DATA>(); // reset accumulator at the start of the mult-add for each output sample
            for (unsigned int j = 0; j < TP_FIR_LEN / TP_INTERPOLATE_FACTOR; ++j) {
                multiplyAcc<TT_DATA, TT_COEFF>(accum, d_in[j], m_internalTapsRef[j * TP_INTERPOLATE_FACTOR + k]);
            }
            // prior to output, the final accumulated value must be downsized to the same type
            // as was input. To do this, the final result is rounded, saturated and shifted down
            roundAcc(TP_RND, shift, accum);
            saturateAcc(accum);
            accum_srs = castAcc(accum);
            window_writeincr((output_window<TT_DATA>*)outWindow, accum_srs);
            window_writeincr((output_window<TT_DATA>*)outWindow2, accum_srs);
        }
        // Revert data pointer for next sample
        window_decr(inWindow, TP_FIR_LEN / TP_INTERPOLATE_FACTOR - 1);
    }
};

// specialization, reload coeffs, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
void fir_interpolate_asym_ref<TT_DATA,
                              TT_COEFF,
                              TP_FIR_LEN,
                              TP_INTERPOLATE_FACTOR,
                              TP_SHIFT,
                              TP_RND,
                              TP_INPUT_WINDOW_VSIZE,
                              USE_COEFF_RELOAD_TRUE,
                              1>::filter(input_window<TT_DATA>* inWindow,
                                         output_window<TT_DATA>* outWindow,
                                         const TT_COEFF (&inTaps)[TP_FIR_LEN]) {
    // Coefficient reload
    for (int i = 0; i < TP_FIR_LEN; i++) {
        m_internalTapsRef[i] = inTaps[FIR_LEN - 1 - i];
        printf("inTaps[%d] = %d\n", i, inTaps[i]);
    }
    const unsigned int shift = TP_SHIFT;
    T_accRef<TT_DATA> accum;
    TT_DATA d_in[TP_FIR_LEN / TP_INTERPOLATE_FACTOR];
    TT_DATA accum_srs;

    printf("Ref model params:\n");
    printf("TP_INTERPOLATE_FACTOR = %d\n", TP_INTERPOLATE_FACTOR);
    printf("TP_SHIFT = %lu\n", TP_SHIFT);
    printf("TP_RND = %d\n", TP_RND);
    printf("TP_INPUT_WINDOW_SIZE = %d\n", TP_INPUT_WINDOW_VSIZE);
    const unsigned int kFirLen = TP_FIR_LEN / TP_INTERPOLATE_FACTOR;
    const unsigned int kFirMarginOffset = fnFirMargin<kFirLen, TT_DATA>() - kFirLen + 1; // FIR Margin Offset.
    window_incr(inWindow, kFirMarginOffset); // move input data pointer past the margin padding

    for (unsigned int i = 0; i < TP_INPUT_WINDOW_VSIZE; i++) {
        for (unsigned int j = 0; j < TP_FIR_LEN / TP_INTERPOLATE_FACTOR; ++j) {
            d_in[j] = window_readincr(inWindow); // read input data
        }
        for (int k = TP_INTERPOLATE_FACTOR - 1; k >= 0; --k) {
            accum = null_accRef<TT_DATA>(); // reset accumulator at the start of the mult-add for each output sample
            for (unsigned int j = 0; j < TP_FIR_LEN / TP_INTERPOLATE_FACTOR; ++j) {
                multiplyAcc<TT_DATA, TT_COEFF>(accum, d_in[j], m_internalTapsRef[j * TP_INTERPOLATE_FACTOR + k]);
            }
            // prior to output, the final accumulated value must be downsized to the same type
            // as was input. To do this, the final result is rounded, saturated and shifted down
            roundAcc(TP_RND, shift, accum);
            saturateAcc(accum);
            accum_srs = castAcc(accum);
            window_writeincr((output_window<TT_DATA>*)outWindow, accum_srs);
        }
        // Revert data pointer for next sample
        window_decr(inWindow, TP_FIR_LEN / TP_INTERPOLATE_FACTOR - 1);
    }
};

// specialization, reload coeffs, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
void fir_interpolate_asym_ref<TT_DATA,
                              TT_COEFF,
                              TP_FIR_LEN,
                              TP_INTERPOLATE_FACTOR,
                              TP_SHIFT,
                              TP_RND,
                              TP_INPUT_WINDOW_VSIZE,
                              USE_COEFF_RELOAD_TRUE,
                              2>::filter(input_window<TT_DATA>* inWindow,
                                         output_window<TT_DATA>* outWindow,
                                         output_window<TT_DATA>* outWindow2,
                                         const TT_COEFF (&inTaps)[TP_FIR_LEN]) {
    // Coefficient reload
    for (int i = 0; i < TP_FIR_LEN; i++) {
        m_internalTapsRef[i] = inTaps[FIR_LEN - 1 - i];
        printf("inTaps[%d] = %d\n", i, inTaps[i]);
    }
    const unsigned int shift = TP_SHIFT;
    T_accRef<TT_DATA> accum;
    TT_DATA d_in[TP_FIR_LEN / TP_INTERPOLATE_FACTOR];
    TT_DATA accum_srs;

    printf("Ref model params:\n");
    printf("TP_INTERPOLATE_FACTOR = %d\n", TP_INTERPOLATE_FACTOR);
    printf("TP_SHIFT = %lu\n", TP_SHIFT);
    printf("TP_RND = %d\n", TP_RND);
    printf("TP_INPUT_WINDOW_SIZE = %d\n", TP_INPUT_WINDOW_VSIZE);
    const unsigned int kFirLen = TP_FIR_LEN / TP_INTERPOLATE_FACTOR;
    const unsigned int kFirMarginOffset = fnFirMargin<kFirLen, TT_DATA>() - kFirLen + 1; // FIR Margin Offset.
    window_incr(inWindow, kFirMarginOffset); // move input data pointer past the margin padding

    for (unsigned int i = 0; i < TP_INPUT_WINDOW_VSIZE; i++) {
        for (unsigned int j = 0; j < TP_FIR_LEN / TP_INTERPOLATE_FACTOR; ++j) {
            d_in[j] = window_readincr(inWindow); // read input data
        }
        for (int k = TP_INTERPOLATE_FACTOR - 1; k >= 0; --k) {
            accum = null_accRef<TT_DATA>(); // reset accumulator at the start of the mult-add for each output sample
            for (unsigned int j = 0; j < TP_FIR_LEN / TP_INTERPOLATE_FACTOR; ++j) {
                multiplyAcc<TT_DATA, TT_COEFF>(accum, d_in[j], m_internalTapsRef[j * TP_INTERPOLATE_FACTOR + k]);
            }
            // prior to output, the final accumulated value must be downsized to the same type
            // as was input. To do this, the final result is rounded, saturated and shifted down
            roundAcc(TP_RND, shift, accum);
            saturateAcc(accum);
            accum_srs = castAcc(accum);
            window_writeincr((output_window<TT_DATA>*)outWindow, accum_srs);
            window_writeincr((output_window<TT_DATA>*)outWindow2, accum_srs);
        }
        // Revert data pointer for next sample
        window_decr(inWindow, TP_FIR_LEN / TP_INTERPOLATE_FACTOR - 1);
    }
};
}
}
}
}
}
