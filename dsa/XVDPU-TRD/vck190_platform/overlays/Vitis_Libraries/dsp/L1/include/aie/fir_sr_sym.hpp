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
#ifndef _DSPLIB_FIR_SR_SYM_HPP_
#define _DSPLIB_FIR_SR_SYM_HPP_

/*
Single Rate Symmetric FIR.
This file exists to capture the definition of the single rate symmetric FIR
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

#include <adf.h>
#include "fir_utils.hpp"
#include "fir_sr_sym_traits.hpp"
#include <vector>

namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace sr_sym {
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
          unsigned int TP_USE_COEFF_RELOAD = 0,
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

    static constexpr unsigned int m_kDataLoadVsize = fnDataLoadVsizeSrSym<TT_DATA>(); // ie. upd_w loads a v4 of cint16
    static constexpr unsigned int m_kDataLoadsInReg1Buff =
        fnDataLoadsInRegSrSym<TT_DATA>(); // commands needed to fill 1024-bit input vector register using a 256-bit
                                          // upd_w
    static constexpr unsigned int m_kWinAccessByteSize =
        fnWinAccessByteSize<TT_DATA, TT_COEFF>(); // The memory data path is min 128-bits wide for vector operations
    static constexpr unsigned int m_kFirRangeOffset =
        fnFirRangeOffsetSym<TP_FIR_LEN, TP_CASC_LEN, TP_KERNEL_POSITION>(); // FIR Cascade Offset for this kernel
                                                                            // position
    static constexpr unsigned int m_kFirMarginOffset =
        fnFirMargin<TP_FIR_LEN, TT_DATA>() - TP_FIR_LEN + 1; // FIR Margin Offset.
    static constexpr unsigned int m_kFirInitOffset = m_kFirRangeOffset + m_kFirMarginOffset;
    static constexpr unsigned int m_kSBuffXOffset =
        m_kFirInitOffset % (m_kWinAccessByteSize /
                            sizeof(TT_DATA)); // Remainder of m_kFirInitOffset divided by m_kWinAccessByteSize (128bit )
    static constexpr unsigned int m_kTBuffXOffset =
        m_kFirRangeOffset % (m_kWinAccessByteSize / sizeof(TT_DATA)); // don't include margin offset
    static constexpr int m_kColumns =
        fnNumColumnsSym<TT_DATA, TT_COEFF>(); // number of mult-adds per lane for main intrinsic
    static constexpr unsigned int m_kLanes = fnNumLanesSym<TT_DATA, TT_COEFF>(); // number of operations in parallel of
                                                                                 // this type combinations that the
                                                                                 // vector processor performs.
    static constexpr unsigned int m_kVOutSize =
        fnNumLanesSym<TT_DATA, TT_COEFF>(); // Output vector size, equal to number of operations in parallel of this
                                            // type combinations that the vector processor performs.
    // static constexpr unsigned int m_kArchFirLen             =
    // TP_KERNEL_POSITION==TP_CASC_LEN-1?TP_FIR_RANGE_LEN:TP_FIR_LEN;  //Check if only the remainder of the FIR
    // (TP_FIR_RANGE_LEN + m_Lanes - due to xoffset alignment) for last kernel in chain fits, otherwise check if full
    // FIR (TP_FIR_LEN) fits.
    static constexpr unsigned int m_kArchFirLen = TP_FIR_LEN; // Debug.
    static constexpr eArchType m_kSmallFirArch =
        (TP_INPUT_WINDOW_VSIZE % (m_kLanes * m_kDataLoadsInReg1Buff) == 0)
            ? kArch1Buff
            : kArch2Buff; // is 1buff architecture supported for this data/coeff type combination?
    static constexpr eArchType m_kArch =
        ((m_kSBuffXOffset + m_kArchFirLen - 1 + m_kDataLoadVsize) <= kBuffSize128Byte / sizeof(TT_DATA))
            ? m_kSmallFirArch
            : kArch2Buff; // will all data fit in a 1024b reg
    static constexpr int m_kFirLenCeilCols = TRUNC((TP_FIR_RANGE_LEN) / kSymmetryFactor, m_kColumns);
    static constexpr unsigned int m_kXbuffByteSize =
        m_kArch == kArch1Buff ? kBuffSize128Byte
                              : kBuffSize64Byte; // 128B buffer for 1buff arch, 2 small 64B buffers for 2buff arch.
    static constexpr unsigned int m_kDataRegVsize = m_kXbuffByteSize / (sizeof(TT_DATA)); // buff size in Bytes
    static constexpr unsigned int m_kDataLoadsInReg =
        m_kDataRegVsize / m_kDataLoadVsize; // 2 loads for 64Byte size, 4 loads for 128B size
    static constexpr unsigned int m_kZbuffByteSize =
        kBuffSize32Byte; // m_kZbuffByteSize (256bit) - const for all data/coeff types
    static constexpr unsigned int m_kCoeffRegVsize = m_kZbuffByteSize / sizeof(TT_COEFF);
    static constexpr unsigned int m_kLsize =
        (TP_INPUT_WINDOW_VSIZE /
         m_kVOutSize); // loop length, given that <m_kVOutSize> samples are output per iteration of loop
    static constexpr unsigned int m_kSDataLoadInitOffset =
        TRUNC((m_kFirInitOffset),
              (m_kWinAccessByteSize / sizeof(TT_DATA))); // Ybuff window offset, aligned to load vector size.
    static constexpr unsigned int m_kTDataLoadInitOffset =
        CEIL((TP_FIR_LEN - 1 + m_kFirMarginOffset - m_kFirRangeOffset + m_kLanes - 1), m_kLanes) -
        m_kDataLoadVsize; // tbuff initial offset, aligned to m_kLanes (not necessarily m_kDataLoadVsize), so that last
                          // m_kLanes tbuff samples are used during first MAC operation
    static constexpr unsigned int m_kTStartOffset =
        m_kDataRegVsize - m_kLanes - m_kTBuffXOffset; // Ystart offset. Ybuff is always aligned to m_kDataLoadVsize.
    static constexpr unsigned int m_kSInitialLoads = m_kDataLoadsInReg; // Number of initial data loads to sbuff. Always
                                                                        // extend to 2, to avoid case where both buffers
                                                                        // schedule update after init MAC
    static constexpr unsigned int m_kTInitialLoads =
        (m_kColumns + m_kLanes + m_kTBuffXOffset) > m_kDataLoadVsize ? 2 : 1; //  Number of initial data loads to tbuff.
    static constexpr unsigned int m_kIncrRepeatFactor =
        m_kDataLoadsInReg * m_kDataLoadVsize /
        m_kVOutSize; // 8 phases for cint16/int16 data/coeff type combo, 4 phases for all others
    static constexpr unsigned int m_kInitialLoads =
        (CEIL((fnFirMargin<TP_FIR_LEN, TT_DATA>() - m_kFirRangeOffset - m_kSDataLoadInitOffset) + m_kDataLoadVsize,
              m_kDataLoadVsize)) /
        m_kDataLoadVsize; // Number of initial data loads - enough data gets loaded to perform all MACs, until next data
                          // buffer phase iteration (second iteration for data/coeff type combos where
                          // m_kDataLoadVsize/=m_kLanes)
    static constexpr unsigned int m_kYstartInitOffset =
        (fnFirMargin<TP_FIR_LEN, TT_DATA>() - m_kFirRangeOffset -
         m_kSDataLoadInitOffset); // Ystart offset relative to first sample loaded in the data buffer (1buff arch)

    // Additional defensive checks
    static_assert(TP_INPUT_WINDOW_VSIZE % m_kLanes == 0,
                  "ERROR: TP_INPUT_WINDOW_VSIZE must be an integer multiple of the number of lanes for this data type");

    // Coefficient Load Size - number of samples in 256-bits
    static constexpr unsigned int m_kCoeffLoadSize = 256 / 8 / sizeof(TT_COEFF);

    // The coefficients array must include zero padding up to a multiple of the number of columns
    // the MAC intrinsic used to eliminate the accidental inclusion of terms beyond the FIR length.
    // Since this zero padding cannot be applied to the class-external coefficient array
    // the supplied taps are copied to an internal array, m_internalTaps, which can be padded.
    TT_COEFF chess_storage(% chess_alignof(v16int16))
        m_internalTaps[CEIL((TP_FIR_LEN + 1) / kSymmetryFactor, m_kCoeffLoadSize)];

    void filterSelectArch(T_inputIF<TP_CASC_IN, TT_DATA> inInterface, T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);
    void filterKernel1buff(T_inputIF<TP_CASC_IN, TT_DATA> inInterface, T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);
    void filterKernel2buff(T_inputIF<TP_CASC_IN, TT_DATA> inInterface, T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);

    TT_COEFF chess_storage(% chess_alignof(v8cint16)) m_oldInTaps[CEIL(
        (TP_FIR_LEN + 1) / kSymmetryFactor, m_kCoeffLoadSize)]; // Previous user input coefficients with zero padding
    bool m_coeffnEq;                                            // Are coefficients sets equal?

   public:
    // Access function for AIE Synthesizer
    unsigned int get_m_kArch() { return m_kArch; };

    // Constructor
    kernelFilterClass(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / kSymmetryFactor]) : m_internalTaps{} {
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
        for (int i = 0; i < CEIL((TP_FIR_RANGE_LEN + 1) / kSymmetryFactor, m_kColumns); i++) {
            if (i < (TP_FIR_RANGE_LEN + 1) / kSymmetryFactor) {
                m_internalTaps[i] = taps[i + fnFirRangeOffsetSym<TP_FIR_LEN, TP_CASC_LEN, TP_KERNEL_POSITION>()];
            } else {
                m_internalTaps[i] = nullElem<TT_COEFF>(); // 0 for the type.
            }
        }
    };

    // Filter kernel for static coefficient designs
    void filterKernel(T_inputIF<TP_CASC_IN, TT_DATA> inInterface, T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);
    // Filter kernel for reloadable coefficient designs
    void filterKernel(T_inputIF<TP_CASC_IN, TT_DATA> inInterface,
                      T_outputIF<TP_CASC_OUT, TT_DATA> outInterface,
                      const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / kSymmetryFactor]);
    void filterKernelRtp(T_inputIF<TP_CASC_IN, TT_DATA> inInterface, T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);
};

//-----------------------------------------------------------------------------------------------------
// Single kernel specialization. No cascade ports. Static coefficients
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
class fir_sr_sym : public kernelFilterClass<TT_DATA,
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
    fir_sr_sym(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / kSymmetryFactor])
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_sym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow, output_window<TT_DATA>* outWindow);
};

// Single kernel specialization. No cascade ports. Static coefficients, dual outputs
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN>
class fir_sr_sym<TT_DATA,
                 TT_COEFF,
                 TP_FIR_LEN,
                 TP_SHIFT,
                 TP_RND,
                 TP_INPUT_WINDOW_VSIZE,
                 CASC_IN_FALSE,
                 CASC_OUT_FALSE,
                 TP_FIR_RANGE_LEN,
                 0,
                 1,
                 USE_COEFF_RELOAD_FALSE,
                 2> : public kernelFilterClass<TT_DATA,
                                               TT_COEFF,
                                               TP_FIR_LEN,
                                               TP_SHIFT,
                                               TP_RND,
                                               TP_INPUT_WINDOW_VSIZE,
                                               CASC_IN_FALSE,
                                               CASC_OUT_FALSE,
                                               TP_FIR_RANGE_LEN,
                                               0,
                                               1,
                                               USE_COEFF_RELOAD_FALSE,
                                               2> {
   public:
    // Constructor
    fir_sr_sym(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / kSymmetryFactor])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_FALSE,
                            CASC_OUT_FALSE,
                            TP_FIR_RANGE_LEN,
                            0,
                            1,
                            USE_COEFF_RELOAD_FALSE,
                            2>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_sym::filter); }

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
class fir_sr_sym<TT_DATA,
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
    fir_sr_sym()
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_sym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_window<TT_DATA>* outWindow,
                const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / kSymmetryFactor]);
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
class fir_sr_sym<TT_DATA,
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
    fir_sr_sym()
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_sym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_window<TT_DATA>* outWindow,
                output_window<TT_DATA>* outWindow2,
                const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / kSymmetryFactor]);
};

//-----------------------------------------------------------------------------------------------------
// Partially specialized classes for cascaded interface - final kernel. Static coefficients, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_sr_sym<TT_DATA,
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
    fir_sr_sym(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / kSymmetryFactor])
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_sym::filter); }

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
class fir_sr_sym<TT_DATA,
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
    fir_sr_sym(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / kSymmetryFactor])
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_sym::filter); }

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
class fir_sr_sym<TT_DATA,
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
    fir_sr_sym(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / kSymmetryFactor])
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_sym::filter); }

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
class fir_sr_sym<TT_DATA,
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
    fir_sr_sym(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / kSymmetryFactor])
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_sym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_stream_cacc48* inCascade,
                output_stream_cacc48* outCascade,
                output_window<TT_DATA>* broadcastWindow);
};

//-----------------------------------------------------------------------------------------------------
// Partially specialized classes for cascaded interface - final kernel. Reloadable coefficients, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_sr_sym<TT_DATA,
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
    fir_sr_sym()
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_sym::filter); }

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
class fir_sr_sym<TT_DATA,
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
    fir_sr_sym()
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_sym::filter); }

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
class fir_sr_sym<TT_DATA,
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
    fir_sr_sym()
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_sym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_stream_cacc48* outCascade,
                const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / kSymmetryFactor],
                output_window<TT_DATA>* broadcastWindow);
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
class fir_sr_sym<TT_DATA,
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
    fir_sr_sym()
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_sr_sym::filter); }

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

#endif // _DSPLIB_FIR_SR_SYM_HPP_
