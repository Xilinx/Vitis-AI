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
#ifndef FIR_INTERPOLATE_ASYM_HPP
#define FIR_INTERPOLATE_ASYM_HPP
/*
Interpolating FIR class definition

The file holds the definition of the Asymmetric Interpolation FIR kernel class.

Note on Coefficient reversal. The AIE processor intrinsics naturally sum data and coefficients in the same order,
but the conventional definition of a FIR has data and coefficient indices in opposite order. The order of
coefficients is therefore reversed during construction to yield conventional FIR behaviour.
*/

/* Coding conventions
  TT_      template type suffix
  TP_      template parameter suffix
*/

#include <adf.h>
#include <assert.h>
#include <array>
#include <cstdint>
#include "fir_utils.hpp"
#include "fir_interpolate_asym_traits.hpp"

// CEIL rounds x up to the next multiple of y, which may be x itself.
#define CEIL(x, y) (((x + y - 1) / y) * y)

namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace interpolate_asym {
//#define _DSPLIB_FIR_INTERPOLATE_ASYM_HPP_DEBUG_

//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
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
    // Two implementations have been written for this filter. They have identical behaviour, but one is optimised for an
    // Interpolation factor
    // greater than the number of accumulator registers available.
    static constexpr unsigned int kArchIncr = 1;
    static constexpr unsigned int kArchPhaseSeries = 2;
    static constexpr unsigned int kArchPhaseParallel = 3;

    // Parameter value defensive and legality checks
    static_assert(TP_FIR_LEN <= FIR_LEN_MAX, "ERROR: Max supported FIR length exceeded. ");
    static_assert(TP_FIR_RANGE_LEN >= FIR_LEN_MIN,
                  "ERROR: Illegal combination of design FIR length and cascade length, resulting in kernel FIR length "
                  "below minimum required value. ");
    static_assert(TP_SHIFT >= SHIFT_MIN && TP_SHIFT <= SHIFT_MAX, "ERROR: SHIFT is out of the supported range.");
    static_assert(TP_RND >= ROUND_MIN && TP_RND <= ROUND_MAX, "ERROR: RND is out of the supported range.");
    static_assert((TP_FIR_LEN % TP_INTERPOLATE_FACTOR) == 0,
                  "ERROR: TP_FIR_LEN must be an integer multiple of INTERPOLATE_FACTOR.");
    static_assert(fnEnumType<TT_DATA>() != enumUnknownType, "ERROR: TT_DATA is not a supported type.");
    static_assert(fnEnumType<TT_COEFF>() != enumUnknownType, "ERROR: TT_COEFF is not a supported type.");
    static_assert(fnTypeCheckDataCoeffSize<TT_DATA, TT_COEFF>() != 0,
                  "ERROR: TT_DATA type less precise than TT_COEFF is not supported.");
    static_assert(fnTypeCheckDataCoeffCmplx<TT_DATA, TT_COEFF>() != 0,
                  "ERROR: real TT_DATA with complex TT_COEFF is not supported.");
    static_assert(fnTypeCheckDataCoeffFltInt<TT_DATA, TT_COEFF>() != 0,
                  "ERROR: a mix of float and integer types of TT_DATA and TT_COEFF is not supported.");
    //
    static_assert(TP_INTERPOLATE_FACTOR >= INTERPOLATE_FACTOR_MIN && TP_INTERPOLATE_FACTOR <= INTERPOLATE_FACTOR_MAX,
                  "ERROR: TP_INTERPOLATE_FACTOR is out of the supported range");
    static_assert(fnUnsupportedTypeCombo<TT_DATA, TT_COEFF>() != 0,
                  "ERROR: The combination of TT_DATA and TT_COEFF is not supported for this class.");
    static_assert(TP_NUM_OUTPUTS > 0 && TP_NUM_OUTPUTS <= 2, "ERROR: only single or dual outputs are supported.");
    // There are additional defensive checks after architectural constants have been calculated.

    // The interpolation FIR calculates over multiple phases where such that the total number of lanes is an integer
    // multiple of the
    // interpolation factor. Hence an array of accumulators is needed for this set of lanes.
    static constexpr unsigned int m_kNumAccRegs = fnAccRegsIntAsym<TT_DATA, TT_COEFF>();
    static constexpr unsigned int m_kWinAccessByteSize =
        16; // 16 Bytes. The memory data path is min 128-bits wide for vector operations
    static constexpr unsigned int m_kColumns =
        sizeof(TT_COEFF) == 2 ? 2 : 1; // number of mult-adds per lane for main intrinsic
    static constexpr unsigned int m_kLanes = fnNumLanesIntAsym<TT_DATA, TT_COEFF>(); // number of operations in parallel
                                                                                     // of this type combinations that
                                                                                     // the vector processor can do.
    static constexpr unsigned int m_kDataLoadsInReg = 4; // ratio of 1024-bit data buffer to 256-bit load size.
    static constexpr unsigned int m_kDataLoadVsize =
        (32 / sizeof(TT_DATA)); // number of samples in a single 256-bit load
    static constexpr unsigned int m_kSamplesInBuff = m_kDataLoadsInReg * m_kDataLoadVsize;

    static constexpr unsigned int m_kFirRangeOffset =
        fnFirRangeOffset<TP_FIR_LEN, TP_CASC_LEN, TP_KERNEL_POSITION, TP_INTERPOLATE_FACTOR>() /
        TP_INTERPOLATE_FACTOR; // FIR Cascade Offset for this kernel position
    static constexpr unsigned int m_kFirMarginLen = TP_FIR_LEN / TP_INTERPOLATE_FACTOR;
    static constexpr unsigned int m_kFirMarginOffset =
        fnFirMargin<m_kFirMarginLen, TT_DATA>() - m_kFirMarginLen + 1; // FIR Margin Offset.
    static constexpr unsigned int m_kFirInitOffset = m_kFirRangeOffset + m_kFirMarginOffset;
    static constexpr unsigned int m_kDataWindowOffset =
        TRUNC((m_kFirInitOffset), (m_kWinAccessByteSize / sizeof(TT_DATA))); // Window offset - increments by 128bit
    static constexpr unsigned int m_kDataBuffXOffset =
        m_kFirInitOffset % (m_kWinAccessByteSize / sizeof(TT_DATA)); // Remainder of m_kFirInitOffset divided by 128bit
    // In some cases, the number of accumulators needed exceeds the number available in the processor leading to
    // inefficency as the
    // accumulators are loaded and stored on the stack. An alternative implementation is used to avoid this.
    static constexpr unsigned int m_kArch = (((m_kDataBuffXOffset + TP_FIR_RANGE_LEN + m_kLanes) < m_kSamplesInBuff) &&
                                             (TP_INPUT_WINDOW_VSIZE % (m_kLanes * m_kDataLoadsInReg) == 0))
                                                ? kArchIncr
                                                :                 // execute incremental load architecture
                                                kArchPhaseSeries; // execute each phase in series (reloads data)
    static constexpr unsigned int m_kZbuffSize = 32;
    static constexpr unsigned int m_kCoeffRegVsize = m_kZbuffSize / sizeof(TT_COEFF);
    static constexpr unsigned int m_kTotalLanes =
        fnLCMIntAsym<TT_DATA, TT_COEFF, TP_INTERPOLATE_FACTOR>(); // Lowest common multiple of Lanes and
                                                                  // Interpolatefactor
    static constexpr unsigned int m_kLCMPhases = m_kTotalLanes / m_kLanes;
    static constexpr unsigned int m_kPhases = TP_INTERPOLATE_FACTOR;
    static constexpr unsigned int m_kNumOps = CEIL(TP_FIR_RANGE_LEN / TP_INTERPOLATE_FACTOR, m_kColumns) / m_kColumns;
    static constexpr unsigned int m_kVOutSize =
        fnVOutSizeIntAsym<TT_DATA, TT_COEFF>();       // This differs from kLanes for cint32/cint32
    static constexpr unsigned int m_kXbuffSize = 128; // data buffer size in Bytes
    static constexpr unsigned int m_kDataRegVsize = m_kXbuffSize / sizeof(TT_DATA); // samples in data buffer
    static constexpr unsigned int m_kLsize = TP_INPUT_WINDOW_VSIZE / m_kLanes;      // loops required to consume input
    static constexpr unsigned int m_kInitialLoads =
        (m_kDataBuffXOffset + (m_kPhases * m_kLanes + m_kVOutSize) / TP_INTERPOLATE_FACTOR + m_kColumns - 1 +
         (m_kLanes - 1)) /
        m_kLanes; // effectively ceil[(kVOutsize+m_kColumns-1)/kLanes]
    static constexpr unsigned int m_kInitialLoadsIncr =
        CEIL(m_kDataBuffXOffset + TP_FIR_RANGE_LEN + m_kLanes * m_kDataLoadVsize / m_kVOutSize - 1, m_kDataLoadVsize) /
        m_kDataLoadVsize;
    static constexpr unsigned int m_kRepeatFactor = m_kDataLoadsInReg * m_kDataLoadVsize / m_kVOutSize;

    // Additional defensive checks
    static_assert(TP_INPUT_WINDOW_VSIZE % m_kLanes == 0,
                  "ERROR: TP_INPUT_WINDOW_VSIZE must be an integer multiple of the number of lanes for this data type");

    // The m_internalTaps is defined in terms of samples, but loaded into a vector, so has to be memory-aligned to the
    // vector size.
    TT_COEFF chess_storage(% chess_alignof(v16int16)) m_internalTaps[m_kLCMPhases][m_kNumOps][m_kColumns][m_kLanes];

    // Two implementations have been written for this filter. They have identical behaviour, but one is optimised for an
    // Interpolation factor
    // greater than the number of accumulator registers available.
    void filter_impl1(T_inputIF<TP_CASC_IN, TT_DATA> inInterface,
                      T_outputIF<TP_CASC_OUT, TT_DATA> outInterface); // Each phase is calculated in turn which avoids
                                                                      // need for multiple accumulators, but requires
                                                                      // data reloading.
    void filter_impl2(
        T_inputIF<TP_CASC_IN, TT_DATA> inInterface,
        T_outputIF<TP_CASC_OUT, TT_DATA> outInterface); // Parallel phase execution, requires multiple accumulators
    void filterIncr(
        T_inputIF<TP_CASC_IN, TT_DATA> inInterface,
        T_outputIF<TP_CASC_OUT, TT_DATA> outInterface); // Incremental load architecture which applies for short FIR_LEN
    void filterSelectArch(T_inputIF<TP_CASC_IN, TT_DATA> inInterface, T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);

    // Constants for coeff reload
    static constexpr unsigned int m_kCoeffLoadSize = 256 / 8 / sizeof(TT_COEFF);
    TT_COEFF chess_storage(% chess_alignof(v8cint16))
        m_oldInTaps[CEIL(TP_FIR_LEN, m_kCoeffLoadSize)]; // Previous user input coefficients with zero padding
    bool m_coeffnEq;                                     // Are coefficients sets equal?

   public:
    // Access function for AIE Synthesizer
    unsigned int get_m_kArch() { return m_kArch; };

    // Constructors
    kernelFilterClass() : m_oldInTaps{} {}

    kernelFilterClass(const TT_COEFF (&taps)[TP_FIR_LEN]) {
        // Loads taps/coefficients
        firReload(taps);
    };

    void firReload(const TT_COEFF* taps) {
        TT_COEFF* tapsPtr = (TT_COEFF*)taps;
        firReload(tapsPtr);
    }

    void firReload(TT_COEFF* taps) {
        const unsigned int bitsInNibble = 4;

        // Since the intrinsics can have columns, any values in memory beyond the end of the taps array could
        // contaminate the calculation.
        // To avoid this hazard, the class has its own taps array which is zero-padded to the column width for the type
        // of coefficient.
        int tapIndex;

        // Coefficients are pre-arranged such that during filter execution they may simply be read from a lookup table.
        for (int phase = 0; phase < m_kLCMPhases; ++phase) {
            for (int op = 0; op < m_kNumOps; ++op) {
                for (int column = 0; column < m_kColumns; ++column) {
                    for (int lane = 0; lane < m_kLanes; ++lane) {
                        tapIndex = TP_INTERPOLATE_FACTOR - 1 -
                                   ((lane + phase * m_kLanes) % TP_INTERPOLATE_FACTOR) + // datum index of lane
                                   (column * TP_INTERPOLATE_FACTOR) +                    // column offset is additive
                                   ((op * m_kColumns * TP_INTERPOLATE_FACTOR));
                        if (tapIndex < TP_FIR_RANGE_LEN && tapIndex >= 0) {
                            tapIndex = TP_FIR_LEN - 1 - tapIndex -
                                       fnFirRangeOffset<TP_FIR_LEN, TP_CASC_LEN, TP_KERNEL_POSITION,
                                                        TP_INTERPOLATE_FACTOR>(); // Reverse coefficients and apply
                                                                                  // cascade range offset. See note at
                                                                                  // head of file.
                            m_internalTaps[phase][op][column][lane] = taps[tapIndex];
                        } else {
                            m_internalTaps[phase][op][column][lane] = nullElem<TT_COEFF>(); // 0 for the type.
                        }
                    }
                }
            }
        }
    };

    // Filter kernel for static coefficient designs
    void filterKernel(T_inputIF<TP_CASC_IN, TT_DATA> inInterface, T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);

    // Filter kernel for reloadable coefficient designs
    void filterKernel(T_inputIF<TP_CASC_IN, TT_DATA> inInterface,
                      T_outputIF<TP_CASC_OUT, TT_DATA> outInterface,
                      const TT_COEFF (&inTaps)[TP_FIR_LEN]);
    void filterKernelRtp(T_inputIF<TP_CASC_IN, TT_DATA> inInterface, T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);
};

//-----------------------------------------------------------------------------------------------------
// Cascade layer class and specializations

//-----------------------------------------------------------------------------------------------------
// This is the main declaration of the fir_interpolate_asym class, and is also used for the Standalone kernel
// specialization with no cascade ports, no reload, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
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
class fir_interpolate_asym : public kernelFilterClass<TT_DATA,
                                                      TT_COEFF,
                                                      TP_FIR_LEN,
                                                      TP_INTERPOLATE_FACTOR,
                                                      TP_SHIFT,
                                                      TP_RND,
                                                      TP_INPUT_WINDOW_VSIZE,
                                                      CASC_IN_FALSE,
                                                      CASC_OUT_FALSE,
                                                      TP_FIR_LEN,
                                                      0,
                                                      1,
                                                      USE_COEFF_RELOAD_FALSE,
                                                      TP_NUM_OUTPUTS> {
   private:
   public:
    // Constructor
    fir_interpolate_asym(const TT_COEFF (&taps)[TP_FIR_LEN])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_FALSE,
                            CASC_OUT_FALSE,
                            TP_FIR_LEN,
                            0,
                            1,
                            USE_COEFF_RELOAD_FALSE>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow, output_window<TT_DATA>* restrict outWindow);
};

// Single kernel specialization. No cascade ports, with reload coefficients.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_interpolate_asym<TT_DATA,
                           TT_COEFF,
                           TP_FIR_LEN,
                           TP_INTERPOLATE_FACTOR,
                           TP_SHIFT,
                           TP_RND,
                           TP_INPUT_WINDOW_VSIZE,
                           CASC_IN_FALSE,
                           CASC_OUT_FALSE,
                           TP_FIR_RANGE_LEN,
                           TP_KERNEL_POSITION,
                           TP_CASC_LEN,
                           USE_COEFF_RELOAD_FALSE,
                           2> : public kernelFilterClass<TT_DATA,
                                                         TT_COEFF,
                                                         TP_FIR_LEN,
                                                         TP_INTERPOLATE_FACTOR,
                                                         TP_SHIFT,
                                                         TP_RND,
                                                         TP_INPUT_WINDOW_VSIZE,
                                                         CASC_IN_FALSE,
                                                         CASC_OUT_FALSE,
                                                         TP_FIR_RANGE_LEN,
                                                         TP_KERNEL_POSITION,
                                                         TP_CASC_LEN,
                                                         USE_COEFF_RELOAD_FALSE,
                                                         2> {
   private:
   public:
    // Constructor
    fir_interpolate_asym(const TT_COEFF (&taps)[TP_FIR_LEN])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_FALSE,
                            CASC_OUT_FALSE,
                            TP_FIR_LEN,
                            0,
                            1,
                            USE_COEFF_RELOAD_FALSE,
                            2>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_window<TT_DATA>* restrict outWindow,
                output_window<TT_DATA>* restrict outWindow2);
};

//-----------------------------------------------------------------------------------------------------
// Single kernel specialization. No cascade ports, with reload coefficients, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_interpolate_asym<TT_DATA,
                           TT_COEFF,
                           TP_FIR_LEN,
                           TP_INTERPOLATE_FACTOR,
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
                                                         TP_INTERPOLATE_FACTOR,
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
   private:
   public:
    // Constructor
    fir_interpolate_asym()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_FALSE,
                            CASC_OUT_FALSE,
                            TP_FIR_LEN,
                            0,
                            1,
                            USE_COEFF_RELOAD_TRUE,
                            1>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_window<TT_DATA>* outWindow,
                const TT_COEFF (&inTaps)[TP_FIR_LEN]);
};

// Single kernel specialization. No cascade ports, with reload coefficients, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_interpolate_asym<TT_DATA,
                           TT_COEFF,
                           TP_FIR_LEN,
                           TP_INTERPOLATE_FACTOR,
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
                                                         TP_INTERPOLATE_FACTOR,
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
   private:
   public:
    // Constructor
    fir_interpolate_asym()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_FALSE,
                            CASC_OUT_FALSE,
                            TP_FIR_LEN,
                            0,
                            1,
                            USE_COEFF_RELOAD_TRUE,
                            2>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_window<TT_DATA>* outWindow,
                output_window<TT_DATA>* outWindow2,
                const TT_COEFF (&inTaps)[TP_FIR_LEN]);
};

//-----------------------------------------------------------------------------------------------------
// Partially specialized classes for cascaded interface (final kernel in cascade), no reload, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_interpolate_asym<TT_DATA,
                           TT_COEFF,
                           TP_FIR_LEN,
                           TP_INTERPOLATE_FACTOR,
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
                                                         TP_INTERPOLATE_FACTOR,
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
   private:
   public:
    // Constructor
    fir_interpolate_asym(const TT_COEFF (&taps)[TP_FIR_LEN])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_stream_cacc48* inCascade,
                output_window<TT_DATA>* restrict outWindow);
};

// Partially specialized classes for cascaded interface (final kernel in cascade), no reload, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_interpolate_asym<TT_DATA,
                           TT_COEFF,
                           TP_FIR_LEN,
                           TP_INTERPOLATE_FACTOR,
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
                                                         TP_INTERPOLATE_FACTOR,
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
   private:
   public:
    // Constructor
    fir_interpolate_asym(const TT_COEFF (&taps)[TP_FIR_LEN])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_stream_cacc48* inCascade,
                output_window<TT_DATA>* restrict outWindow,
                output_window<TT_DATA>* restrict outWindow2);
};

// Partially specialized classes for cascaded interface (final kernel in cascade), with reload, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_interpolate_asym<TT_DATA,
                           TT_COEFF,
                           TP_FIR_LEN,
                           TP_INTERPOLATE_FACTOR,
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
                                                         TP_INTERPOLATE_FACTOR,
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
   private:
   public:
    // Constructor
    fir_interpolate_asym()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow, input_stream_cacc48* inCascade, output_window<TT_DATA>* outWindow);
    // output_window<TT_DATA>* restrict outWindow,
    // const TT_COEFF (&inTaps)[TP_FIR_LEN]);
};

// Partially specialized classes for cascaded interface (final kernel in cascade), with reload, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_interpolate_asym<TT_DATA,
                           TT_COEFF,
                           TP_FIR_LEN,
                           TP_INTERPOLATE_FACTOR,
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
                                                         TP_INTERPOLATE_FACTOR,
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
   private:
   public:
    // Constructor
    fir_interpolate_asym()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_stream_cacc48* inCascade,
                output_window<TT_DATA>* outWindow,
                output_window<TT_DATA>* outWindow2);
    // output_window<TT_DATA>* restrict outWindow,
    // const TT_COEFF (&inTaps)[TP_FIR_LEN]);
};

//-----------------------------------------------------------------------------------------------------
// Partially specialized classes for cascaded interface (First kernel in cascade), no reload
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_interpolate_asym<TT_DATA,
                           TT_COEFF,
                           TP_FIR_LEN,
                           TP_INTERPOLATE_FACTOR,
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
                                                         TP_INTERPOLATE_FACTOR,
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
   private:
   public:
    // Constructor
    fir_interpolate_asym(const TT_COEFF (&taps)[TP_FIR_LEN])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_stream_cacc48* outCascade,
                output_window<TT_DATA>* broadcastWindow);
};

// Partially specialized classes for cascaded interface (First kernel in cascade), with reload
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_interpolate_asym<TT_DATA,
                           TT_COEFF,
                           TP_FIR_LEN,
                           TP_INTERPOLATE_FACTOR,
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
                                                         TP_INTERPOLATE_FACTOR,
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
   private:
   public:
    // Constructor
    fir_interpolate_asym()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_stream_cacc48* outCascade,
                output_window<TT_DATA>* broadcastWindow,
                const TT_COEFF (&inTaps)[TP_FIR_LEN]);
};

//-----------------------------------------------------------------------------------------------------
// Partially specialized classes for cascaded interface (middle kernels in cascade), no reload
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_interpolate_asym<TT_DATA,
                           TT_COEFF,
                           TP_FIR_LEN,
                           TP_INTERPOLATE_FACTOR,
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
                                                         TP_INTERPOLATE_FACTOR,
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
   private:
   public:
    // Constructor
    fir_interpolate_asym(const TT_COEFF (&taps)[TP_FIR_LEN])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_stream_cacc48* inCascade,
                output_stream_cacc48* outCascade,
                output_window<TT_DATA>* broadcastWindow);
};

// Partially specialized classes for cascaded interface (middle kernels in cascade), with reload
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_interpolate_asym<TT_DATA,
                           TT_COEFF,
                           TP_FIR_LEN,
                           TP_INTERPOLATE_FACTOR,
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
                                                         TP_INTERPOLATE_FACTOR,
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
   private:
   public:
    // Constructor
    fir_interpolate_asym()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_asym::filter); }

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
} // namespaces
#endif // fir_interpolate_asym_HPP
