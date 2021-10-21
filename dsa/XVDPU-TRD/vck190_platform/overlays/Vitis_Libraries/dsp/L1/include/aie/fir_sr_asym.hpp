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
#ifndef _DSPLIB_FIR_SR_ASYM_HPP_
#define _DSPLIB_FIR_SR_ASYM_HPP_

/*
Single Rate Asymmetric FIR.
This file exists to capture the definition of the single rate asymmetric FIR
filter kernel class.
The class definition holds defensive checks on parameter range and other
legality.
The constructor definition is held in this class because this class must be
accessible to graph level aie compilation.
The main runtime filter function is captured elsewhere as it contains aie
intrinsics which are not included in aie graph level
compilation.
*/

/* Coding conventions
   TT_      template type suffix
   TP_      template parameter suffix
*/

/* Design Notes
   Note that the AIE intrinsics operate on increasing indices, but in a conventional FIR there is a convolution of data
   and coefficients.
   So as to achieve the impulse response from the filter which matches the coefficient set, the coefficient array has to
   be reversed
   to compensate for the action of the intrinsics. This reversal is performed in the constructor. To avoid common-mode
   errors
   the reference model performs this reversal at run-time.
*/

#include <adf.h>
#include "fir_utils.hpp"
#include "fir_sr_asym_traits.hpp"
#include <vector>

//#define _DSPLIB_FIR_SR_ASYM_HPP_DEBUG_

namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace sr_asym {

//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          bool TP_CASC_IN = CASC_IN_FALSE,
          bool TP_CASC_OUT = CASC_OUT_FALSE,
          unsigned int TP_FIR_RANGE_LEN = TP_FIR_LEN,
          unsigned int TP_KERNEL_POSITION = 0,
          unsigned int TP_CASC_LEN = 1,
          unsigned int TP_USE_COEFF_RELOAD = 0, // 1 = use coeff reload, 0 = don't use coeff reload
          unsigned int TP_NUM_OUTPUTS = 1>
class kernelFilterClass {
   private:
    // Parameter value defensive and legality checks
    static_assert(TP_FIR_LEN <= FIR_LEN_MAX, "ERROR: Max supported FIR length exceeded. ");
    static_assert(TP_FIR_RANGE_LEN >= FIR_LEN_MIN,
                  "ERROR: Illegal combination of design FIR length and cascade length, resulting in kernel FIR length "
                  "below minimum required value. ");
    static_assert(TP_SHIFT >= SHIFT_MIN && TP_SHIFT <= SHIFT_MAX, "ERROR: TP_SHIFT is out of the supported range.");
    static_assert(TP_RND >= ROUND_MIN && TP_RND <= ROUND_MAX, "ERROR: TP_RND is out of the supported range.");
    static_assert(fnEnumType<TT_DATA>() != enumUnknownType, "ERROR: TT_DATA is not a supported type.");
    static_assert(fnEnumType<TT_COEFF>() != enumUnknownType, "ERROR: TT_COEFF is not a supported type.");
    static_assert(fnTypeCheckDataCoeffSize<TT_DATA, TT_COEFF>() != 0,
                  "ERROR: TT_DATA type less precise than TT_COEFF is not supported.");
    static_assert(fnTypeCheckDataCoeffCmplx<TT_DATA, TT_COEFF>() != 0,
                  "ERROR: real TT_DATA with complex TT_COEFF is not supported.");
    static_assert(fnTypeCheckDataCoeffFltInt<TT_DATA, TT_COEFF>() != 0,
                  "ERROR: a mix of float and integer types of TT_DATA and TT_COEFF is not supported.");
    static_assert(TP_NUM_OUTPUTS > 0 && TP_NUM_OUTPUTS <= 2, "ERROR: only single or dual outputs are supported.");
    // There are additional defensive checks after architectural constants have been calculated.

    static constexpr unsigned int m_kColumns =
        fnNumColumnsSrAsym<TT_DATA, TT_COEFF>(); // number of mult-adds per lane for main intrinsic
    static constexpr unsigned int m_kLanes = fnNumLanesSrAsym<TT_DATA, TT_COEFF>(); // number of operations in parallel
                                                                                    // of this type combinations that
                                                                                    // the vector processor can do.
    static constexpr unsigned int m_kVOutSize =
        fnVOutSizeSrAsym<TT_DATA, TT_COEFF>(); // This differs from m_kLanes for cint32/cint32
    static constexpr unsigned int m_kFirLenCeilCols = CEIL(TP_FIR_RANGE_LEN, m_kColumns);
    static constexpr unsigned int m_kDataLoadsInReg =
        fnDataLoadsInRegSrAsym<TT_DATA>(); // 4;  //ratio of sbuff to load size.
    static constexpr unsigned int m_kDataLoadVsize =
        fnDataLoadVsizeSrAsym<TT_DATA>(); // 16;  //ie. upd_w loads a v4 of cint16
    static constexpr unsigned int m_kSamplesInBuff = m_kDataLoadsInReg * m_kDataLoadVsize;
    static constexpr unsigned int m_kFirRangeOffset =
        fnFirRangeOffset<TP_FIR_LEN, TP_CASC_LEN, TP_KERNEL_POSITION>(); // FIR Cascade Offset for this kernel position
    static constexpr unsigned int m_kFirMarginOffset =
        fnFirMargin<TP_FIR_LEN, TT_DATA>() - TP_FIR_LEN + 1; // FIR Margin Offset.
    static constexpr unsigned int m_kFirInitOffset = m_kFirRangeOffset + m_kFirMarginOffset;
    static constexpr unsigned int m_kDataBuffXOffset =
        m_kFirInitOffset % (16 / sizeof(TT_DATA)); // Remainder of m_kFirInitOffset divided by 128bit
    static constexpr unsigned int m_kArchFirLen =
        TP_FIR_RANGE_LEN + m_kDataBuffXOffset; // This will check if only the portion of the FIR (TP_FIR_RANGE_LEN +
                                               // m_kDataBuffXOffset - due to xoffset alignment) fits.
    static constexpr unsigned int m_kFirCoeffByteSize = TP_FIR_RANGE_LEN * sizeof(TT_COEFF);
    static constexpr unsigned int m_kInitDataNeeded = ((m_kArchFirLen + m_kLanes * m_kDataLoadVsize / m_kVOutSize) - 1);
    static constexpr eArchType m_kArchBufSize = m_kInitDataNeeded > m_kSamplesInBuff ? kArchZigZag : kArchIncLoads;
    static constexpr eArchType m_kArch = (m_kFirCoeffByteSize < 512 && // Revert to Basic for "big" FIRs.
                                          TP_INPUT_WINDOW_VSIZE % (m_kLanes * m_kDataLoadsInReg) == 0)
                                             ? m_kArchBufSize
                                             : kArchBasic;
    static constexpr unsigned int m_kZigZagFactor = m_kArch == kArchZigZag ? 2 : 1;
    static constexpr unsigned int m_kRepeatFactor =
        m_kArch == kArchZigZag ? (m_kSamplesInBuff / m_kVOutSize) / m_kZigZagFactor : 1;
    static constexpr unsigned int m_kZbuffSize =
        kBuffSize32Byte; // kZbuffSize (256bit) - const for all data/coeff types
    static constexpr unsigned int m_kCoeffRegVsize = m_kZbuffSize / sizeof(TT_COEFF);
    static constexpr unsigned int m_kLsize = ((TP_INPUT_WINDOW_VSIZE) / m_kVOutSize) /
                                             (m_kZigZagFactor * m_kRepeatFactor); // m_kRepeatFactor;  // loop length,
                                                                                  // given that <m_kVOutSize> samples
                                                                                  // are output per iteration of loop
    static constexpr unsigned int m_kInitialLoads =
        m_kArch == kArchZigZag
            ? m_kDataLoadsInReg - 1
            : m_kArch == kArchIncLoads ? CEIL(m_kInitDataNeeded, m_kDataLoadVsize) / m_kDataLoadVsize
                                       : CEIL(m_kDataBuffXOffset + m_kLanes + m_kColumns - 1, m_kDataLoadVsize) /
                                             m_kDataLoadVsize; // Mimic end of zag (m_kDataBuffXOffset +
                                                               // m_kVOutSize+m_kColumns-1+(m_kLanes-1))/m_kLanes;
                                                               // //effectively ceil[(m_kDataBuffXOffset
                                                               // m_+kVOutsize+m_kColumns-1)/m_kLanes]
    static constexpr unsigned int m_kIncLoadsRptFactor =
        m_kDataLoadsInReg * m_kDataLoadVsize /
        m_kVOutSize; // 8 times for cint32/cint32; 4 times for other data/coeff types

    // Additional defensive checks
    static_assert(TP_INPUT_WINDOW_VSIZE % m_kLanes == 0,
                  "ERROR: TP_INPUT_WINDOW_VSIZE must be an integer multiple of the number of lanes for this data type");

    // Coefficient Load Size - number of samples in 256-bits
    static constexpr unsigned int m_kCoeffLoadSize = 256 / 8 / sizeof(TT_COEFF);

    // The coefficients array must include zero padding up to a multiple of the number of columns
    // the MAC intrinsic used to eliminate the accidental inclusion of terms beyond the FIR length.
    // Since this zero padding cannot be applied to the class-external coefficient array
    // the supplied taps are copied to an internal array, m_internalTaps, which can be padded.
    TT_COEFF chess_storage(% chess_alignof(v16int16)) m_internalTaps[CEIL(TP_FIR_RANGE_LEN, m_kCoeffLoadSize)];

    void filterSelectArch(T_inputIF<TP_CASC_IN, TT_DATA> inInterface, T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);
    // Implementations
    void filterBasic(T_inputIF<TP_CASC_IN, TT_DATA> inInterface, T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);
    void filterIncLoads(T_inputIF<TP_CASC_IN, TT_DATA> inInterface, T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);
    void filterZigZag(T_inputIF<TP_CASC_IN, TT_DATA> inInterface, T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);

    TT_COEFF chess_storage(% chess_alignof(v8cint16))
        m_oldInTaps[CEIL(TP_FIR_LEN, m_kCoeffLoadSize)]; // Previous user input coefficients with zero padding
    bool m_coeffnEq;                                     // Are coefficients sets equal?

   public:
    // Access function for AIE Synthesizer
    unsigned int get_m_kArch() { return m_kArch; };

    // Constructor
    kernelFilterClass(const TT_COEFF (&taps)[TP_FIR_LEN]) : m_internalTaps{} {
        // Loads taps/coefficients
        firReload(taps);
    }

    // Constructors
    kernelFilterClass() : m_oldInTaps{}, m_internalTaps{} {}

    void firReload(const TT_COEFF* taps) {
        TT_COEFF* tapsPtr = (TT_COEFF*)taps;
        firReload(tapsPtr);
    }

    void firReload(TT_COEFF* taps) {
        for (int i = 0; i < TP_FIR_RANGE_LEN; i++) {
            m_internalTaps[i] =
                taps[TP_FIR_LEN - 1 - fnFirRangeOffset<TP_FIR_LEN, TP_CASC_LEN, TP_KERNEL_POSITION>() - i];
        }
    };

    // FIR
    void filterKernel(T_inputIF<TP_CASC_IN, TT_DATA> inInterface, T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);
    // Filter kernel for reloadable coefficient designs
    void filterKernel(T_inputIF<TP_CASC_IN, TT_DATA> inInterface,
                      T_outputIF<TP_CASC_OUT, TT_DATA> outInterface,
                      const TT_COEFF (&inTaps)[TP_FIR_LEN]);
    void filterKernelRtp(T_inputIF<TP_CASC_IN, TT_DATA> inInterface, T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);
};

//-----------------------------------------------------------------------------------------------------
// Single kernel base specialization. No cascade ports. Static coefficients
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          bool TP_CASC_IN = CASC_IN_FALSE,
          bool TP_CASC_OUT = CASC_OUT_FALSE,
          unsigned int TP_FIR_RANGE_LEN = TP_FIR_LEN,
          unsigned int TP_KERNEL_POSITION = 0,
          unsigned int TP_CASC_LEN = 1,
          unsigned int TP_USE_COEFF_RELOAD = 0, // 1 = use coeff reload, 0 = don't use coeff reload
          unsigned int TP_NUM_OUTPUTS = 1>
class fir_sr_asym : public kernelFilterClass<TT_DATA,
                                             TT_COEFF,
                                             TP_FIR_LEN,
                                             TP_SHIFT,
                                             TP_RND,
                                             TP_INPUT_WINDOW_VSIZE,
                                             TP_CASC_IN,
                                             TP_CASC_OUT,
                                             TP_FIR_RANGE_LEN,
                                             TP_KERNEL_POSITION,
                                             TP_CASC_LEN,
                                             TP_USE_COEFF_RELOAD,
                                             TP_NUM_OUTPUTS> {
   public:
    // Constructor
    fir_sr_asym(const TT_COEFF (&taps)[TP_FIR_LEN])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            TP_CASC_IN,
                            TP_CASC_OUT,
                            TP_FIR_RANGE_LEN,
                            TP_KERNEL_POSITION,
                            TP_CASC_LEN,
                            TP_USE_COEFF_RELOAD,
                            TP_NUM_OUTPUTS>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow, output_window<TT_DATA>* outWindow);
};

// Single kernel specialization. No cascade ports. Static coefficients, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN>
class fir_sr_asym<TT_DATA,
                  TT_COEFF,
                  TP_FIR_LEN,
                  TP_SHIFT,
                  TP_RND,
                  TP_INPUT_WINDOW_VSIZE,
                  false,
                  false,
                  TP_FIR_RANGE_LEN,
                  0,
                  1,
                  0,
                  2> : public kernelFilterClass<TT_DATA,
                                                TT_COEFF,
                                                TP_FIR_LEN,
                                                TP_SHIFT,
                                                TP_RND,
                                                TP_INPUT_WINDOW_VSIZE,
                                                false,
                                                false,
                                                TP_FIR_RANGE_LEN,
                                                0,
                                                1,
                                                0,
                                                2> {
   public:
    // Constructor
    fir_sr_asym(const TT_COEFF (&taps)[TP_FIR_LEN])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            false,
                            false,
                            TP_FIR_RANGE_LEN,
                            0,
                            1,
                            0,
                            2>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow, output_window<TT_DATA>* outWindow, output_window<TT_DATA>* outWindow2);
};

//-----------------------------------------------------------------------------------------------------
// Single kernel specialization. No cascade ports. Using coefficient reload, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_sr_asym<TT_DATA,
                  TT_COEFF,
                  TP_FIR_LEN,
                  TP_SHIFT,
                  TP_RND,
                  TP_INPUT_WINDOW_VSIZE,
                  CASC_IN_FALSE,
                  CASC_OUT_FALSE,
                  TP_FIR_RANGE_LEN,
                  TP_KERNEL_POSITION,
                  TP_CASC_LEN,
                  USE_COEFF_RELOAD_TRUE,
                  1> : public kernelFilterClass<TT_DATA,
                                                TT_COEFF,
                                                TP_FIR_LEN,
                                                TP_SHIFT,
                                                TP_RND,
                                                TP_INPUT_WINDOW_VSIZE,
                                                CASC_IN_FALSE,
                                                CASC_OUT_FALSE,
                                                TP_FIR_RANGE_LEN,
                                                TP_KERNEL_POSITION,
                                                TP_CASC_LEN,
                                                USE_COEFF_RELOAD_TRUE,
                                                1> {
   public:
    // Constructor
    fir_sr_asym()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_FALSE,
                            CASC_OUT_FALSE,
                            TP_FIR_RANGE_LEN,
                            TP_KERNEL_POSITION,
                            TP_CASC_LEN,
                            USE_COEFF_RELOAD_TRUE,
                            1>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* restrict inWindow,
                output_window<TT_DATA>* restrict outWindow,
                const TT_COEFF (&inTaps)[TP_FIR_LEN]);
};

// Single kernel specialization. No cascade ports. Using coefficient reload, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_sr_asym<TT_DATA,
                  TT_COEFF,
                  TP_FIR_LEN,
                  TP_SHIFT,
                  TP_RND,
                  TP_INPUT_WINDOW_VSIZE,
                  CASC_IN_FALSE,
                  CASC_OUT_FALSE,
                  TP_FIR_RANGE_LEN,
                  TP_KERNEL_POSITION,
                  TP_CASC_LEN,
                  USE_COEFF_RELOAD_TRUE,
                  2> : public kernelFilterClass<TT_DATA,
                                                TT_COEFF,
                                                TP_FIR_LEN,
                                                TP_SHIFT,
                                                TP_RND,
                                                TP_INPUT_WINDOW_VSIZE,
                                                CASC_IN_FALSE,
                                                CASC_OUT_FALSE,
                                                TP_FIR_RANGE_LEN,
                                                TP_KERNEL_POSITION,
                                                TP_CASC_LEN,
                                                USE_COEFF_RELOAD_TRUE,
                                                2> {
   public:
    // Constructor
    fir_sr_asym()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_FALSE,
                            CASC_OUT_FALSE,
                            TP_FIR_RANGE_LEN,
                            TP_KERNEL_POSITION,
                            TP_CASC_LEN,
                            USE_COEFF_RELOAD_TRUE,
                            2>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* restrict inWindow,
                output_window<TT_DATA>* restrict outWindow,
                output_window<TT_DATA>* restrict outWindow2,
                const TT_COEFF (&inTaps)[TP_FIR_LEN]);
};

//-----------------------------------------------------------------------------------------------------
// Partially specialized classes for cascaded interface - final kernel. Static coefficients single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_sr_asym<TT_DATA,
                  TT_COEFF,
                  TP_FIR_LEN,
                  TP_SHIFT,
                  TP_RND,
                  TP_INPUT_WINDOW_VSIZE,
                  CASC_IN_TRUE,
                  CASC_OUT_FALSE,
                  TP_FIR_RANGE_LEN,
                  TP_KERNEL_POSITION,
                  TP_CASC_LEN,
                  USE_COEFF_RELOAD_FALSE,
                  1> : public kernelFilterClass<TT_DATA,
                                                TT_COEFF,
                                                TP_FIR_LEN,
                                                TP_SHIFT,
                                                TP_RND,
                                                TP_INPUT_WINDOW_VSIZE,
                                                CASC_IN_TRUE,
                                                CASC_OUT_FALSE,
                                                TP_FIR_RANGE_LEN,
                                                TP_KERNEL_POSITION,
                                                TP_CASC_LEN,
                                                USE_COEFF_RELOAD_FALSE,
                                                1> {
   public:
    // Constructor
    fir_sr_asym(const TT_COEFF (&taps)[TP_FIR_LEN])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_TRUE,
                            CASC_OUT_FALSE,
                            TP_FIR_RANGE_LEN,
                            TP_KERNEL_POSITION,
                            TP_CASC_LEN,
                            USE_COEFF_RELOAD_FALSE,
                            1>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow, input_stream_cacc48* inCascade, output_window<TT_DATA>* outWindow);
};

// Partially specialized classes for cascaded interface - final kernel. Static coefficients, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_sr_asym<TT_DATA,
                  TT_COEFF,
                  TP_FIR_LEN,
                  TP_SHIFT,
                  TP_RND,
                  TP_INPUT_WINDOW_VSIZE,
                  CASC_IN_TRUE,
                  CASC_OUT_FALSE,
                  TP_FIR_RANGE_LEN,
                  TP_KERNEL_POSITION,
                  TP_CASC_LEN,
                  USE_COEFF_RELOAD_FALSE,
                  2> : public kernelFilterClass<TT_DATA,
                                                TT_COEFF,
                                                TP_FIR_LEN,
                                                TP_SHIFT,
                                                TP_RND,
                                                TP_INPUT_WINDOW_VSIZE,
                                                CASC_IN_TRUE,
                                                CASC_OUT_FALSE,
                                                TP_FIR_RANGE_LEN,
                                                TP_KERNEL_POSITION,
                                                TP_CASC_LEN,
                                                USE_COEFF_RELOAD_FALSE,
                                                2> {
   public:
    // Constructor
    fir_sr_asym(const TT_COEFF (&taps)[TP_FIR_LEN])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_TRUE,
                            CASC_OUT_FALSE,
                            TP_FIR_RANGE_LEN,
                            TP_KERNEL_POSITION,
                            TP_CASC_LEN,
                            USE_COEFF_RELOAD_FALSE,
                            2>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_stream_cacc48* inCascade,
                output_window<TT_DATA>* outWindow,
                output_window<TT_DATA>* outWindow2);
};

//-----------------------------------------------------------------------------------------------------
// Partially specialized classes for cascaded interface - first kernel. Static coefficients
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_sr_asym<TT_DATA,
                  TT_COEFF,
                  TP_FIR_LEN,
                  TP_SHIFT,
                  TP_RND,
                  TP_INPUT_WINDOW_VSIZE,
                  CASC_IN_FALSE,
                  CASC_OUT_TRUE,
                  TP_FIR_RANGE_LEN,
                  TP_KERNEL_POSITION,
                  TP_CASC_LEN,
                  USE_COEFF_RELOAD_FALSE,
                  1> : public kernelFilterClass<TT_DATA,
                                                TT_COEFF,
                                                TP_FIR_LEN,
                                                TP_SHIFT,
                                                TP_RND,
                                                TP_INPUT_WINDOW_VSIZE,
                                                CASC_IN_FALSE,
                                                CASC_OUT_TRUE,
                                                TP_FIR_RANGE_LEN,
                                                TP_KERNEL_POSITION,
                                                TP_CASC_LEN,
                                                USE_COEFF_RELOAD_FALSE,
                                                1> {
   public:
    // Constructor
    fir_sr_asym(const TT_COEFF (&taps)[TP_FIR_LEN])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_FALSE,
                            CASC_OUT_TRUE,
                            TP_FIR_RANGE_LEN,
                            TP_KERNEL_POSITION,
                            TP_CASC_LEN,
                            USE_COEFF_RELOAD_FALSE,
                            1>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_stream_cacc48* outCascade,
                output_window<TT_DATA>* broadcastWindow);
};

//-----------------------------------------------------------------------------------------------------
// Partially specialized classes for cascaded interface - middle kernel. Static coefficients
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_sr_asym<TT_DATA,
                  TT_COEFF,
                  TP_FIR_LEN,
                  TP_SHIFT,
                  TP_RND,
                  TP_INPUT_WINDOW_VSIZE,
                  CASC_IN_TRUE,
                  CASC_OUT_TRUE,
                  TP_FIR_RANGE_LEN,
                  TP_KERNEL_POSITION,
                  TP_CASC_LEN,
                  USE_COEFF_RELOAD_FALSE,
                  1> : public kernelFilterClass<TT_DATA,
                                                TT_COEFF,
                                                TP_FIR_LEN,
                                                TP_SHIFT,
                                                TP_RND,
                                                TP_INPUT_WINDOW_VSIZE,
                                                CASC_IN_TRUE,
                                                CASC_OUT_TRUE,
                                                TP_FIR_RANGE_LEN,
                                                TP_KERNEL_POSITION,
                                                TP_CASC_LEN,
                                                USE_COEFF_RELOAD_FALSE,
                                                1> {
   public:
    // Constructor
    fir_sr_asym(const TT_COEFF (&taps)[TP_FIR_LEN])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_TRUE,
                            CASC_OUT_TRUE,
                            TP_FIR_RANGE_LEN,
                            TP_KERNEL_POSITION,
                            TP_CASC_LEN,
                            USE_COEFF_RELOAD_FALSE,
                            1>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_stream_cacc48* inCascade,
                output_stream_cacc48* outCascade,
                output_window<TT_DATA>* broadcastWindow);
};

//-----------------------------------------------------------------------------------------------------
// Partially specialized classes for cascaded interface - final kernel. Reloadable coefficients
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_sr_asym<TT_DATA,
                  TT_COEFF,
                  TP_FIR_LEN,
                  TP_SHIFT,
                  TP_RND,
                  TP_INPUT_WINDOW_VSIZE,
                  CASC_IN_TRUE,
                  CASC_OUT_FALSE,
                  TP_FIR_RANGE_LEN,
                  TP_KERNEL_POSITION,
                  TP_CASC_LEN,
                  USE_COEFF_RELOAD_TRUE,
                  1> : public kernelFilterClass<TT_DATA,
                                                TT_COEFF,
                                                TP_FIR_LEN,
                                                TP_SHIFT,
                                                TP_RND,
                                                TP_INPUT_WINDOW_VSIZE,
                                                CASC_IN_TRUE,
                                                CASC_OUT_FALSE,
                                                TP_FIR_RANGE_LEN,
                                                TP_KERNEL_POSITION,
                                                TP_CASC_LEN,
                                                USE_COEFF_RELOAD_TRUE,
                                                1> {
   public:
    // Constructor
    fir_sr_asym()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_TRUE,
                            CASC_OUT_FALSE,
                            TP_FIR_RANGE_LEN,
                            TP_KERNEL_POSITION,
                            TP_CASC_LEN,
                            USE_COEFF_RELOAD_TRUE,
                            1>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow, input_stream_cacc48* inCascade, output_window<TT_DATA>* outWindow);
};

// Partially specialized classes for cascaded interface - final kernel. Reloadable coefficients, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_sr_asym<TT_DATA,
                  TT_COEFF,
                  TP_FIR_LEN,
                  TP_SHIFT,
                  TP_RND,
                  TP_INPUT_WINDOW_VSIZE,
                  CASC_IN_TRUE,
                  CASC_OUT_FALSE,
                  TP_FIR_RANGE_LEN,
                  TP_KERNEL_POSITION,
                  TP_CASC_LEN,
                  USE_COEFF_RELOAD_TRUE,
                  2> : public kernelFilterClass<TT_DATA,
                                                TT_COEFF,
                                                TP_FIR_LEN,
                                                TP_SHIFT,
                                                TP_RND,
                                                TP_INPUT_WINDOW_VSIZE,
                                                CASC_IN_TRUE,
                                                CASC_OUT_FALSE,
                                                TP_FIR_RANGE_LEN,
                                                TP_KERNEL_POSITION,
                                                TP_CASC_LEN,
                                                USE_COEFF_RELOAD_TRUE,
                                                2> {
   public:
    // Constructor
    fir_sr_asym()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_TRUE,
                            CASC_OUT_FALSE,
                            TP_FIR_RANGE_LEN,
                            TP_KERNEL_POSITION,
                            TP_CASC_LEN,
                            USE_COEFF_RELOAD_TRUE,
                            2>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_stream_cacc48* inCascade,
                output_window<TT_DATA>* outWindow,
                output_window<TT_DATA>* outWindow2);
};

//-----------------------------------------------------------------------------------------------------
// Partially specialized classes for cascaded interface - first kernel. Reloadable coefficients
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_sr_asym<TT_DATA,
                  TT_COEFF,
                  TP_FIR_LEN,
                  TP_SHIFT,
                  TP_RND,
                  TP_INPUT_WINDOW_VSIZE,
                  CASC_IN_FALSE,
                  CASC_OUT_TRUE,
                  TP_FIR_RANGE_LEN,
                  TP_KERNEL_POSITION,
                  TP_CASC_LEN,
                  USE_COEFF_RELOAD_TRUE,
                  1> : public kernelFilterClass<TT_DATA,
                                                TT_COEFF,
                                                TP_FIR_LEN,
                                                TP_SHIFT,
                                                TP_RND,
                                                TP_INPUT_WINDOW_VSIZE,
                                                CASC_IN_FALSE,
                                                CASC_OUT_TRUE,
                                                TP_FIR_RANGE_LEN,
                                                TP_KERNEL_POSITION,
                                                TP_CASC_LEN,
                                                USE_COEFF_RELOAD_TRUE,
                                                1> {
   public:
    // Constructor
    fir_sr_asym()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_FALSE,
                            CASC_OUT_TRUE,
                            TP_FIR_RANGE_LEN,
                            TP_KERNEL_POSITION,
                            TP_CASC_LEN,
                            USE_COEFF_RELOAD_TRUE,
                            1>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_stream_cacc48* outCascade,
                output_window<TT_DATA>* broadcastWindow,
                const TT_COEFF (&inTaps)[TP_FIR_LEN]);
};

//-----------------------------------------------------------------------------------------------------
// Partially specialized classes for cascaded interface - middle kernel. Reeloadable coefficients
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_sr_asym<TT_DATA,
                  TT_COEFF,
                  TP_FIR_LEN,
                  TP_SHIFT,
                  TP_RND,
                  TP_INPUT_WINDOW_VSIZE,
                  CASC_IN_TRUE,
                  CASC_OUT_TRUE,
                  TP_FIR_RANGE_LEN,
                  TP_KERNEL_POSITION,
                  TP_CASC_LEN,
                  USE_COEFF_RELOAD_TRUE,
                  1> : public kernelFilterClass<TT_DATA,
                                                TT_COEFF,
                                                TP_FIR_LEN,
                                                TP_SHIFT,
                                                TP_RND,
                                                TP_INPUT_WINDOW_VSIZE,
                                                CASC_IN_TRUE,
                                                CASC_OUT_TRUE,
                                                TP_FIR_RANGE_LEN,
                                                TP_KERNEL_POSITION,
                                                TP_CASC_LEN,
                                                USE_COEFF_RELOAD_TRUE,
                                                1> {
   public:
    // Constructor
    fir_sr_asym()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_TRUE,
                            CASC_OUT_TRUE,
                            TP_FIR_RANGE_LEN,
                            TP_KERNEL_POSITION,
                            TP_CASC_LEN,
                            USE_COEFF_RELOAD_TRUE,
                            1>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_stream_cacc48* inCascade,
                output_stream_cacc48* outCascade,
                output_window<TT_DATA>* broadcastWindow);
};
}
}
}
}
}

#endif // _DSPLIB_FIR_SR_ASYM_HPP_
