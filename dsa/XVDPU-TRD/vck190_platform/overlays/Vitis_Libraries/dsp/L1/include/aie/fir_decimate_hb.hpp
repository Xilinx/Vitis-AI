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
#ifndef _DSPLIB_fir_decimate_hb_HPP_
#define _DSPLIB_fir_decimate_hb_HPP_

/*
    Half band decimatiion FIR filter.
    This file exists to capture the definition of the FIR filter kernel class.
    The class definition holds defensive checks on parameter range and other
    legality.
    The constructor definition is held in this class because this class must be
    accessible to graph level aie compilation.
    The main runtime filter function is captured elsewhere as it contains aie
    intrinsics which are not included in aie graph level compilation.

    TT_      template type suffix
    TP_      template parameter suffix
*/

#include <adf.h>
#include "fir_utils.hpp"
#include "fir_decimate_hb_traits.hpp"

#define K_BITS_IN_BYTE 8
#define K_MAX_PASSES 2

//#define _DSPLIB_FIR_DECIMATE_HB_HPP_DEBUG_

namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace decimate_hb {
//-----------------------------------------------------------------------------------------------------
// Half band decimation FIR class definition.
template <typename TT_DATA,
          typename TT_COEFF,
          size_t TP_FIR_LEN,
          size_t TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          bool TP_CASC_IN = CASC_IN_FALSE,
          bool TP_CASC_OUT = CASC_OUT_FALSE,
          unsigned int TP_FIR_RANGE_LEN = TP_FIR_LEN,
          unsigned int TP_KERNEL_POSITION = 0,
          unsigned int TP_CASC_LEN = 1,
          unsigned int TP_DUAL_IP = 0,
          unsigned int TP_USE_COEFF_RELOAD = 0, // 1 = use coeff reload, 0 = don't use coeff reload
          unsigned int TP_NUM_OUTPUTS = 1>
class kernelFilterClass {
   private:
    // Parameter value defensive and legality checks
    static_assert(TP_FIR_LEN <= FIR_LEN_MAX, "ERROR: Max supported FIR length exceeded. ");
    static_assert(TP_FIR_RANGE_LEN >= FIR_LEN_MIN,
                  "ERROR: Illegal combination of design FIR length and cascade length, resulting in kernel FIR length "
                  "below minimum required value. ");
    static_assert(((TP_FIR_LEN + 1) % 4) == 0, "ERROR: TP_FIR_LEN must be 4N-1 where N is a positive integer.");
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
    static_assert(fnUnsupportedTypeComboFirDecHb<TT_DATA, TT_COEFF>() != 0,
                  "ERROR: The combination of TT_DATA and TT_COEFF is not supported for this class.");
    static_assert(TP_NUM_OUTPUTS > 0 && TP_NUM_OUTPUTS <= 2, "ERROR: only single or dual outputs are supported.");

    static constexpr unsigned int m_kDataReadGranByte = 16; // Data reads occur on 16 byte (128 bit boundary)
    static constexpr unsigned int m_kFirRangeOffset =
        fnFirRangeOffsetSym<TP_FIR_LEN, TP_CASC_LEN, TP_KERNEL_POSITION>(); //
    static constexpr unsigned int m_kFirMarginLen = TP_FIR_LEN;
    static constexpr unsigned int m_kFirMarginOffset =
        fnFirMargin<m_kFirMarginLen, TT_DATA>() - m_kFirMarginLen + 1; // FIR Margin Offset.
    static constexpr unsigned int m_kFirInitOffset = m_kFirRangeOffset + m_kFirMarginOffset;
    static constexpr unsigned int m_kDataWindowOffset =
        TRUNC((m_kFirInitOffset), (m_kDataReadGranByte / sizeof(TT_DATA))); // Window offset - increments by 128bit
    static constexpr unsigned int m_kDataBuffXOffset =
        m_kFirInitOffset % (m_kDataReadGranByte / sizeof(TT_DATA)); // Remainder of m_kFirInitOffset divided by 128bit

    static constexpr unsigned int m_kLanes1buff =
        fnNumOpLanesDecHb<TT_DATA, TT_COEFF, kArch1Buff>(); // kMaxMacs/(sizeof(TT_DATA)*sizeof(TT_COEFF)*m_kColumns);
                                                            // //number of operations in parallel of this type
                                                            // combinations that the vector processor can do.
    static constexpr unsigned int m_kLanes2buff =
        fnNumOpLanesDecHb<TT_DATA, TT_COEFF, kArch2Buff>(); // kMaxMacs/(sizeof(TT_DATA)*sizeof(TT_COEFF)*m_kColumns);
                                                            // //number of operations in parallel of this type
                                                            // combinations that the vector processor can do.
    // Use 1 buffer architecture if all required samples fit into a single 1024b buffer
    static constexpr unsigned int m_kArchFirLen =
        TP_KERNEL_POSITION == TP_CASC_LEN - 1 ? TP_FIR_RANGE_LEN : TP_FIR_LEN; // This will check if only the remainder
                                                                               // of the FIR (TP_FIR_RANGE_LEN + m_Lanes
                                                                               // - due to xoffset alignment) for last
                                                                               // kernel in chain fits, otherwise check
                                                                               // if full FIR (TP_FIR_LEN) fits.
    static constexpr eArchType m_kArch =
        (m_kDataBuffXOffset + TP_FIR_LEN + (m_kLanes1buff - 1) * kDecimateFactor) <= 128 / sizeof(TT_DATA)
            ? kArch1Buff
            : kArch2Buff; // will all data fit in a 1024b reg
    static constexpr eArchType m_kArchZigZag =
        fnSupportZigZag<TT_DATA, TT_COEFF>() == 1
            ? kArchZigZag
            : kArch2Buff; // ZigZag architecture supports cint16/int16 data/coeff type combination
    static constexpr unsigned int m_kLoadSize = fnLoadSize<TT_DATA, TT_COEFF, m_kArch>();
    static constexpr unsigned int m_kXbuffByteSize =
        fnXBuffBSize<TT_DATA, TT_COEFF, m_kArch>(); // kXbuffSize in Bytes (1024bit/512bit)
    static constexpr unsigned int m_kPasses = fnNumPassesDecHb<TT_DATA, TT_COEFF, m_kArch>(); //
    static constexpr unsigned int m_kLanes = m_kArch == kArch1Buff ? m_kLanes1buff : m_kLanes2buff;
    static constexpr unsigned int m_kColumns = fnNumColumnsDecHb<TT_DATA, TT_COEFF, m_kArch>();
    static constexpr unsigned int m_kZbuffSize = 32; // bytes
    static constexpr unsigned int m_kCoeffRegVsize = m_kZbuffSize / sizeof(TT_COEFF);
    static constexpr unsigned int m_kNumOps =
        CEIL((TP_FIR_RANGE_LEN + 1) / kDecimateFactor / kSymmetryFactor + 1, m_kColumns) /
        m_kColumns; //+1 for centre tap
    static constexpr unsigned int m_kDataLoadsInReg = fnDataLoadsInReg<TT_DATA, TT_COEFF, kArch2Buff>();
    static constexpr unsigned int m_kDataLoadVsize =
        m_kLoadSize / (sizeof(TT_DATA) * K_BITS_IN_BYTE); // m_kArch == kArch1Buff ? (32 / sizeof(TT_DATA)):(16 /
                                                          // sizeof(TT_DATA));//kdataLoadVsize<TT_DATA>();// 16; //ie.
                                                          // upd_w loads a v4 of cint16
    static constexpr unsigned int m_kSamplesInDataBuff = m_kDataLoadVsize * m_kDataLoadsInReg;
    static constexpr unsigned int m_kDataNeededEachOp =
        m_kColumns * kDecimateFactor; // Each operation needs this much MORE data, except the last
    static constexpr unsigned int m_kDataNeededLastOp =
        (m_kColumns - 1) * kDecimateFactor + 1; // This isn't strictly true (X/Y need differing amount, but doesn't
                                                // affect output). We need a reduced amount of data because of centre
                                                // tap, but also potentially less due to reduced number of columns used.
                                                // This isn't taken into account because we use those columns for 0
                                                // coefs.
    static constexpr unsigned int m_kOpsEachLoad =
        m_kDataLoadVsize / m_kDataNeededEachOp; // If this is 0, then invalid config. implied flooring
    static constexpr unsigned int m_kRepeatFactor = (m_kArchZigZag == kArchZigZag && m_kArch == kArch2Buff)
                                                        ? m_kDataLoadsInReg / 2
                                                        : 1; // enough to repeat m_kDataLoadsInReg/2 - as the buff gets
                                                             // one load on zig and one on zag, i.e. =2 for 1024bit
                                                             // buffers, =1 for 512 bit buffers.
    static constexpr unsigned int m_kVOutSize = fnVOutSizeDecHb<TT_DATA, TT_COEFF, m_kArch>();
    static constexpr unsigned int m_kLsize =
        (TP_INPUT_WINDOW_VSIZE / (m_kVOutSize * kDecimateFactor)) / m_kRepeatFactor; // loops required to consume input
    static constexpr unsigned int m_kDataRegVsize =
        m_kXbuffByteSize / sizeof(TT_DATA); // sbuff samples, for small architecture

    // The plus 1 in the equation below is the fencepost idea - the equation calculated the index of the top data value
    // to load
    // but index 8, say, requires 3 loads of 4, not 2.
    static constexpr unsigned int m_kInitialLoads =
        CEIL(m_kDataBuffXOffset + 1 + TP_FIR_LEN - 1 + (m_kLanes - 1) * kDecimateFactor, m_kDataLoadVsize) /
        m_kDataLoadVsize; // filter1buff specific.
    static constexpr unsigned int m_kFirstYdata =
        (m_kFirMarginOffset + (TP_FIR_LEN - 1) - (m_kColumns - 1) * kDecimateFactor) - m_kFirRangeOffset;
    static constexpr unsigned int m_kySpliceStart =
        ((m_kFirstYdata / (m_kDataReadGranByte / sizeof(TT_DATA)))) * (m_kDataReadGranByte / sizeof(TT_DATA));
    static constexpr unsigned int m_kFirstYdataOffset = m_kFirstYdata - m_kySpliceStart;
    static constexpr unsigned int m_kyStart = m_kFirstYdata + (m_kColumns - 1) * kDecimateFactor - m_kySpliceStart;
    static constexpr unsigned int m_kInitialLoadsX =
        CEIL(m_kDataBuffXOffset + (m_kLanes + m_kColumns - 1) * kDecimateFactor, m_kDataLoadVsize) / m_kDataLoadVsize;
    static constexpr unsigned int m_kInitLoadsXneeded =
        (m_kInitialLoadsX == 3 && m_kLoadSize == 128)
            ? 4
            : m_kInitialLoadsX; // force to use upd_w and load more than we need at init stage
    static constexpr unsigned int m_kInitialLoadsY =
        CEIL(m_kFirstYdataOffset + (m_kLanes + m_kColumns - 1) * kDecimateFactor, m_kDataLoadVsize) / m_kDataLoadVsize;
    static constexpr unsigned int m_kInitLoadsYneeded =
        (m_kInitialLoadsY == 3 && m_kLoadSize == 128)
            ? 4
            : m_kInitialLoadsY; // force to use upd_w and load more than we need at init stage
    // TODO: Figure this out for m_kOpsEachLoad > 1
    static constexpr signed int m_kWindowModX = (m_kInitialLoadsX + 1) - CEIL(m_kNumOps, m_kOpsEachLoad);
    // The final operation is either a single non-symmetric mac for single column macs, or a partial pre-add for
    // multi-column macs.
    // For the former, xstart has to point to the ct. For the latter, xstart points to sym data and ct is a separate
    // argument
    static constexpr unsigned int m_kFinalOpSkew =
        m_kColumns == 1
            ? 1
            : ((m_kColumns - 1) - (((TP_FIR_RANGE_LEN + 1) / (kDecimateFactor * kSymmetryFactor))) % m_kColumns) *
                  kDecimateFactor;
    static constexpr unsigned int m_kCtOffset =
        (m_kColumns - 2) * kDecimateFactor +
        1; // mtap - center tap data offset relative to xstart. 5 for 4-column, 1 for 2-column intrinsics.
    static constexpr unsigned int m_kCtPresent =
        (TP_KERNEL_POSITION + 1 == TP_CASC_LEN)
            ? 1
            : 0; // Flag presence of coeff center tap operation - only last kernel in cascade cain
    static constexpr unsigned int m_kIncrRepeatFactor = 4;
    static constexpr unsigned int m_kIncrLoadsTopUp = (m_kVOutSize * kDecimateFactor) / m_kDataLoadVsize;

    TT_COEFF chess_storage(% chess_alignof(v16int16)) m_phaseOneTaps[m_kNumOps][m_kColumns];
    TT_COEFF chess_storage(% chess_alignof(v16int16)) m_phaseOneTapsCt[m_kColumns];

    // Constants for coefficient reload
    static constexpr unsigned int m_kCoeffLoadSize = 256 / 8 / sizeof(TT_COEFF);
    TT_COEFF chess_storage(% chess_alignof(v8cint16)) m_oldInTaps[CEIL(
        (TP_FIR_LEN + 1) / 4 + 1, m_kCoeffLoadSize)]; // Previous user input coefficients with zero padding
    bool m_coeffnEq;                                  // Are coefficients sets equal?

    void filterSelectArch(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                          T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);
    // FIR filter function variant utilizing single 1024-bit buffer.
    void filter1buff(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                     T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);
    // FIR filter function variant utilizing two 512-bit buffers.
    void filter2buff(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                     T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);
    // FIR filter function variant utilizing two 512-bit buffers, with a forward and reverse direction FIR and
    // incremental loads.
    void filter2buffzigzag(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                           T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);

    // Additional defensive checks
    static_assert(TP_INPUT_WINDOW_VSIZE % m_kLanes == 0,
                  "ERROR: TP_INPUT_WINDOW_VSIZE must be an integer multiple of the number of lanes for this data type");

   public:
    // Access function for AIE Synthesizer
    unsigned int get_m_kArch() { return m_kArch; };
    unsigned int get_m_kArchZigZag() { return m_kArchZigZag; };

    // Constructor used for reloadable coefficients
    kernelFilterClass() : m_oldInTaps{}, m_phaseOneTaps{} {}

    // Constractor used for static coefficients
    kernelFilterClass(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / 4 + 1]) : m_phaseOneTaps{} { firReload(taps); }

    void firReload(const TT_COEFF* taps) {
        TT_COEFF* tapsPtr = (TT_COEFF*)taps;
        firReload(tapsPtr);
    }

    void firReload(TT_COEFF* taps) {
        // Since the intrinsics can have columns, any values in memory beyond the end of the taps array could
        // contaminate the calculation.
        // To avoid this hazard, the class has its own taps array which is zero-padded to the column width for the type
        // of coefficient.
        int tapIndex;
        int op;
        int column;

        // Coefficients are pre-arranged such that during filter execution they may simply be read from a lookup table.
        for (op = 0; op < m_kNumOps; ++op) {
            for (column = 0; column < m_kColumns; ++column) {
                tapIndex = (op * m_kColumns + column);
                if (tapIndex < (TP_FIR_RANGE_LEN + 1) / 4) { // symmetry and halfband mean only 1/4 coeffs matter. CT
                                                             // coeff ignored here, as it gets special handling below.
                    m_phaseOneTaps[op][column] =
                        taps[tapIndex + fnFirRangeOffsetSym<TP_FIR_LEN, TP_CASC_LEN, TP_KERNEL_POSITION>() / 2];
                } else {
                    m_phaseOneTaps[op][column] = nullElem<TT_COEFF>(); // 0 for the type.
                }
            }
        }
        // The final operation, including the centre tap, requires special handling.
        // since the centre tap is in the final column, not all columns may be in use
        // so must be zeroed.
        // tapIndexMin = ((m_kNumOps-2)*m_kColumns + m_kColumns-1);
        int tapIndexMin = TRUNC((TP_FIR_RANGE_LEN + 1) / 4, m_kColumns);
        int tapIndexMax = (TP_KERNEL_POSITION + 1 == TP_CASC_LEN)
                              ? (TP_FIR_RANGE_LEN + 1) / 4
                              : (TP_FIR_RANGE_LEN - 1) / 4; // Max tap index, only last kernel in cascade gets CT.
        tapIndex = (TP_FIR_RANGE_LEN + 1) / 4;
        // m_phaseOneTapsCt[m_kColumns-1] = taps[tapIndex +
        // fnFirRangeOffsetSym<TP_FIR_LEN,TP_CASC_LEN,TP_KERNEL_POSITION>()/2];
        // tapIndex--;
        for (column = m_kColumns - 1; column >= 0; column--) {
            if (tapIndex >= tapIndexMin && tapIndex <= tapIndexMax) {
                m_phaseOneTapsCt[column] =
                    taps[tapIndex + fnFirRangeOffsetSym<TP_FIR_LEN, TP_CASC_LEN, TP_KERNEL_POSITION>() / 2];
            } else {
                m_phaseOneTapsCt[column] = nullElem<TT_COEFF>();
            }
            tapIndex--;
        }
    }

    // FIR
    void filterKernel(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                      T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);

    // with taps for reload
    void filterKernel(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                      T_outputIF<TP_CASC_OUT, TT_DATA> outInterface,
                      const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / 4 + 1]);

    void filterKernelRtp(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                         T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);
};

//-----------------------------------------------------------------------------------------------------
// Cascade layer class and specializations

//-----------------------------------------------------------------------------------------------------
// This is the main declaration of the fir_decimate_hb class, and is also used for the Standalone kernel specialization
// with no cascade ports, a single input, no reload, single output
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
          unsigned int TP_DUAL_IP = 0,
          unsigned int TP_USE_COEFF_RELOAD = 0, // 1 = use coeff reload, 0 = don't use coeff reload
          unsigned int TP_NUM_OUTPUTS = 1>
class fir_decimate_hb : public kernelFilterClass<TT_DATA,
                                                 TT_COEFF,
                                                 TP_FIR_LEN,
                                                 TP_SHIFT,
                                                 TP_RND,
                                                 TP_INPUT_WINDOW_VSIZE,
                                                 CASC_IN_FALSE,
                                                 CASC_OUT_FALSE,
                                                 TP_FIR_LEN,
                                                 0,
                                                 1,
                                                 DUAL_IP_SINGLE,
                                                 USE_COEFF_RELOAD_FALSE,
                                                 TP_NUM_OUTPUTS> {
   private:
   public:
    // Constructor
    fir_decimate_hb(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / 4 + 1])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_FALSE,
                            CASC_OUT_FALSE,
                            TP_FIR_LEN,
                            0,
                            1,
                            DUAL_IP_SINGLE,
                            USE_COEFF_RELOAD_FALSE,
                            TP_NUM_OUTPUTS>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* restrict inWindow, output_window<TT_DATA>* restrict outWindow);
};

//-----------------------------------------------------------------------------------------------------
// Single kernel specialization. No cascade ports, single input, with static coefficients, dual outputs
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
class fir_decimate_hb<TT_DATA,
                      TT_COEFF,
                      TP_FIR_LEN,
                      TP_SHIFT,
                      TP_RND,
                      TP_INPUT_WINDOW_VSIZE,
                      CASC_IN_FALSE,
                      CASC_OUT_FALSE,
                      TP_FIR_LEN,
                      0,
                      1,
                      DUAL_IP_SINGLE,
                      USE_COEFF_RELOAD_FALSE,
                      2> : public kernelFilterClass<TT_DATA,
                                                    TT_COEFF,
                                                    TP_FIR_LEN,
                                                    TP_SHIFT,
                                                    TP_RND,
                                                    TP_INPUT_WINDOW_VSIZE,
                                                    CASC_IN_FALSE,
                                                    CASC_OUT_FALSE,
                                                    TP_FIR_LEN,
                                                    0,
                                                    1,
                                                    DUAL_IP_SINGLE,
                                                    USE_COEFF_RELOAD_FALSE,
                                                    2> {
   private:
   public:
    // Constructor
    fir_decimate_hb(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / 4 + 1])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_FALSE,
                            CASC_OUT_FALSE,
                            TP_FIR_LEN,
                            0,
                            1,
                            DUAL_IP_SINGLE,
                            USE_COEFF_RELOAD_FALSE,
                            2>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* restrict inWindow,
                output_window<TT_DATA>* restrict outWindow,
                output_window<TT_DATA>* restrict outWindow2);
};

//-----------------------------------------------------------------------------------------------------
// Single kernel specialization. No cascade ports, single input, with reload coefficients, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
class fir_decimate_hb<TT_DATA,
                      TT_COEFF,
                      TP_FIR_LEN,
                      TP_SHIFT,
                      TP_RND,
                      TP_INPUT_WINDOW_VSIZE,
                      CASC_IN_FALSE,
                      CASC_OUT_FALSE,
                      TP_FIR_LEN,
                      0,
                      1,
                      DUAL_IP_SINGLE,
                      USE_COEFF_RELOAD_TRUE,
                      1> : public kernelFilterClass<TT_DATA,
                                                    TT_COEFF,
                                                    TP_FIR_LEN,
                                                    TP_SHIFT,
                                                    TP_RND,
                                                    TP_INPUT_WINDOW_VSIZE,
                                                    CASC_IN_FALSE,
                                                    CASC_OUT_FALSE,
                                                    TP_FIR_LEN,
                                                    0,
                                                    1,
                                                    DUAL_IP_SINGLE,
                                                    USE_COEFF_RELOAD_TRUE,
                                                    1> {
   private:
   public:
    // Constructor
    fir_decimate_hb()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_FALSE,
                            CASC_OUT_FALSE,
                            TP_FIR_LEN,
                            0,
                            1,
                            DUAL_IP_SINGLE,
                            USE_COEFF_RELOAD_TRUE,
                            1>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_window<TT_DATA>* outWindow,
                const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / 4 + 1]);
};

// Single kernel specialization. No cascade ports, single input, with reload coefficients, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
class fir_decimate_hb<TT_DATA,
                      TT_COEFF,
                      TP_FIR_LEN,
                      TP_SHIFT,
                      TP_RND,
                      TP_INPUT_WINDOW_VSIZE,
                      CASC_IN_FALSE,
                      CASC_OUT_FALSE,
                      TP_FIR_LEN,
                      0,
                      1,
                      DUAL_IP_SINGLE,
                      USE_COEFF_RELOAD_TRUE,
                      2> : public kernelFilterClass<TT_DATA,
                                                    TT_COEFF,
                                                    TP_FIR_LEN,
                                                    TP_SHIFT,
                                                    TP_RND,
                                                    TP_INPUT_WINDOW_VSIZE,
                                                    CASC_IN_FALSE,
                                                    CASC_OUT_FALSE,
                                                    TP_FIR_LEN,
                                                    0,
                                                    1,
                                                    DUAL_IP_SINGLE,
                                                    USE_COEFF_RELOAD_TRUE,
                                                    2> {
   private:
   public:
    // Constructor
    fir_decimate_hb()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_FALSE,
                            CASC_OUT_FALSE,
                            TP_FIR_LEN,
                            0,
                            1,
                            DUAL_IP_SINGLE,
                            USE_COEFF_RELOAD_TRUE,
                            2>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_window<TT_DATA>* outWindow,
                output_window<TT_DATA>* outWindow2,
                const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / 4 + 1]);
};

// Single kernel specialization. No cascade ports, dual input, no reload, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
class fir_decimate_hb<TT_DATA,
                      TT_COEFF,
                      TP_FIR_LEN,
                      TP_SHIFT,
                      TP_RND,
                      TP_INPUT_WINDOW_VSIZE,
                      CASC_IN_FALSE,
                      CASC_OUT_FALSE,
                      TP_FIR_LEN,
                      0,
                      1,
                      DUAL_IP_DUAL,
                      USE_COEFF_RELOAD_FALSE,
                      1> : public kernelFilterClass<TT_DATA,
                                                    TT_COEFF,
                                                    TP_FIR_LEN,
                                                    TP_SHIFT,
                                                    TP_RND,
                                                    TP_INPUT_WINDOW_VSIZE,
                                                    CASC_IN_FALSE,
                                                    CASC_OUT_FALSE,
                                                    TP_FIR_LEN,
                                                    0,
                                                    1,
                                                    DUAL_IP_DUAL,
                                                    USE_COEFF_RELOAD_FALSE,
                                                    1> {
   private:
   public:
    // Constructor
    fir_decimate_hb(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / 4 + 1])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_FALSE,
                            CASC_OUT_FALSE,
                            TP_FIR_LEN,
                            0,
                            1,
                            DUAL_IP_DUAL,
                            USE_COEFF_RELOAD_FALSE,
                            1>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow, input_window<TT_DATA>* inWindowRev, output_window<TT_DATA>* outWindow);
};

// Single kernel specialization. No cascade ports, dual input, no reload, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
class fir_decimate_hb<TT_DATA,
                      TT_COEFF,
                      TP_FIR_LEN,
                      TP_SHIFT,
                      TP_RND,
                      TP_INPUT_WINDOW_VSIZE,
                      CASC_IN_FALSE,
                      CASC_OUT_FALSE,
                      TP_FIR_LEN,
                      0,
                      1,
                      DUAL_IP_DUAL,
                      USE_COEFF_RELOAD_FALSE,
                      2> : public kernelFilterClass<TT_DATA,
                                                    TT_COEFF,
                                                    TP_FIR_LEN,
                                                    TP_SHIFT,
                                                    TP_RND,
                                                    TP_INPUT_WINDOW_VSIZE,
                                                    CASC_IN_FALSE,
                                                    CASC_OUT_FALSE,
                                                    TP_FIR_LEN,
                                                    0,
                                                    1,
                                                    DUAL_IP_DUAL,
                                                    USE_COEFF_RELOAD_FALSE,
                                                    2> {
   private:
   public:
    // Constructor
    fir_decimate_hb(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / 4 + 1])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_FALSE,
                            CASC_OUT_FALSE,
                            TP_FIR_LEN,
                            0,
                            1,
                            DUAL_IP_DUAL,
                            USE_COEFF_RELOAD_FALSE,
                            2>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_window<TT_DATA>* inWindowRev,
                output_window<TT_DATA>* outWindow,
                output_window<TT_DATA>* outWindow2);
};

//-----------------------------------------------------------------------------------------------------
// Single kernel specialization. No cascade ports, dual input, with reload, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
class fir_decimate_hb<TT_DATA,
                      TT_COEFF,
                      TP_FIR_LEN,
                      TP_SHIFT,
                      TP_RND,
                      TP_INPUT_WINDOW_VSIZE,
                      CASC_IN_FALSE,
                      CASC_OUT_FALSE,
                      TP_FIR_LEN,
                      0,
                      1,
                      DUAL_IP_DUAL,
                      USE_COEFF_RELOAD_TRUE,
                      1> : public kernelFilterClass<TT_DATA,
                                                    TT_COEFF,
                                                    TP_FIR_LEN,
                                                    TP_SHIFT,
                                                    TP_RND,
                                                    TP_INPUT_WINDOW_VSIZE,
                                                    CASC_IN_FALSE,
                                                    CASC_OUT_FALSE,
                                                    TP_FIR_LEN,
                                                    0,
                                                    1,
                                                    DUAL_IP_DUAL,
                                                    USE_COEFF_RELOAD_TRUE,
                                                    1> {
   private:
   public:
    // Constructor
    fir_decimate_hb()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_FALSE,
                            CASC_OUT_FALSE,
                            TP_FIR_LEN,
                            0,
                            1,
                            DUAL_IP_DUAL,
                            USE_COEFF_RELOAD_TRUE,
                            1>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_window<TT_DATA>* inWindowRev,
                output_window<TT_DATA>* outWindow,
                const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / 4 + 1]);
};

// Single kernel specialization. No cascade ports, dual input, with reload, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
class fir_decimate_hb<TT_DATA,
                      TT_COEFF,
                      TP_FIR_LEN,
                      TP_SHIFT,
                      TP_RND,
                      TP_INPUT_WINDOW_VSIZE,
                      CASC_IN_FALSE,
                      CASC_OUT_FALSE,
                      TP_FIR_LEN,
                      0,
                      1,
                      DUAL_IP_DUAL,
                      USE_COEFF_RELOAD_TRUE,
                      2> : public kernelFilterClass<TT_DATA,
                                                    TT_COEFF,
                                                    TP_FIR_LEN,
                                                    TP_SHIFT,
                                                    TP_RND,
                                                    TP_INPUT_WINDOW_VSIZE,
                                                    CASC_IN_FALSE,
                                                    CASC_OUT_FALSE,
                                                    TP_FIR_LEN,
                                                    0,
                                                    1,
                                                    DUAL_IP_DUAL,
                                                    USE_COEFF_RELOAD_TRUE,
                                                    2> {
   private:
   public:
    // Constructor
    fir_decimate_hb()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_FALSE,
                            CASC_OUT_FALSE,
                            TP_FIR_LEN,
                            0,
                            1,
                            DUAL_IP_DUAL,
                            USE_COEFF_RELOAD_TRUE,
                            2>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_window<TT_DATA>* inWindowRev,
                output_window<TT_DATA>* outWindow,
                output_window<TT_DATA>* outWindow2,
                const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / 4 + 1]);
};

//-----------------------------------------------------------------------------------------------------
// Partially specialized classes for cascaded interface (final kernel in cascade), single input, no reload, single
// output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP>
class fir_decimate_hb<TT_DATA,
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
                      TP_DUAL_IP,
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
                                                    TP_DUAL_IP,
                                                    USE_COEFF_RELOAD_FALSE,
                                                    1> {
   private:
   public:
    // Constructor
    fir_decimate_hb(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / 4 + 1])
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
                            TP_DUAL_IP,
                            USE_COEFF_RELOAD_FALSE,
                            1>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_stream_cacc48* inCascade,
                output_window<TT_DATA>* restrict outWindow);
};

// Partially specialized classes for cascaded interface (final kernel in cascade), single input, no reload, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP>
class fir_decimate_hb<TT_DATA,
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
                      TP_DUAL_IP,
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
                                                    TP_DUAL_IP,
                                                    USE_COEFF_RELOAD_FALSE,
                                                    2> {
   private:
   public:
    // Constructor
    fir_decimate_hb(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / 4 + 1])
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
                            TP_DUAL_IP,
                            USE_COEFF_RELOAD_FALSE,
                            2>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_stream_cacc48* inCascade,
                output_window<TT_DATA>* restrict outWindow,
                output_window<TT_DATA>* restrict outWindow2);
};

// Partially specialized classes for cascaded interface (final kernel in cascade), dual input, no reload, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_decimate_hb<TT_DATA,
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
                      DUAL_IP_DUAL,
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
                                                    DUAL_IP_DUAL,
                                                    USE_COEFF_RELOAD_FALSE,
                                                    1> {
   private:
   public:
    // Constructor
    fir_decimate_hb(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / 4 + 1])
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
                            DUAL_IP_DUAL,
                            USE_COEFF_RELOAD_FALSE,
                            1>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_window<TT_DATA>* inWindowReverse,
                input_stream_cacc48* inCascade,
                output_window<TT_DATA>* restrict outWindow);
};

// Partially specialized classes for cascaded interface (final kernel in cascade), dual input, no reload, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_decimate_hb<TT_DATA,
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
                      DUAL_IP_DUAL,
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
                                                    DUAL_IP_DUAL,
                                                    USE_COEFF_RELOAD_FALSE,
                                                    2> {
   private:
   public:
    // Constructor
    fir_decimate_hb(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / 4 + 1])
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
                            DUAL_IP_DUAL,
                            USE_COEFF_RELOAD_FALSE,
                            2>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_window<TT_DATA>* inWindowReverse,
                input_stream_cacc48* inCascade,
                output_window<TT_DATA>* restrict outWindow,
                output_window<TT_DATA>* restrict outWindow2);
};

// Partially specialized classes for cascaded interface (final kernel in cascade), single input, with reload, single
// output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP>
class fir_decimate_hb<TT_DATA,
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
                      TP_DUAL_IP,
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
                                                    TP_DUAL_IP,
                                                    USE_COEFF_RELOAD_TRUE,
                                                    1> {
   private:
   public:
    // Constructor
    fir_decimate_hb()
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
                            TP_DUAL_IP,
                            USE_COEFF_RELOAD_TRUE,
                            1>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_stream_cacc48* inCascade,
                output_window<TT_DATA>* restrict outWindow);
};

// Partially specialized classes for cascaded interface (final kernel in cascade), single input, with reload, dual
// output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP>
class fir_decimate_hb<TT_DATA,
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
                      TP_DUAL_IP,
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
                                                    TP_DUAL_IP,
                                                    USE_COEFF_RELOAD_TRUE,
                                                    2> {
   private:
   public:
    // Constructor
    fir_decimate_hb()
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
                            TP_DUAL_IP,
                            USE_COEFF_RELOAD_TRUE,
                            2>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_stream_cacc48* inCascade,
                output_window<TT_DATA>* restrict outWindow,
                output_window<TT_DATA>* restrict outWindow2);
};

// Partially specialized classes for cascaded interface (final kernel in cascade), dual input, with reload, single
// output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_decimate_hb<TT_DATA,
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
                      DUAL_IP_DUAL,
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
                                                    DUAL_IP_DUAL,
                                                    USE_COEFF_RELOAD_TRUE,
                                                    1> {
   private:
   public:
    // Constructor
    fir_decimate_hb()
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
                            DUAL_IP_DUAL,
                            USE_COEFF_RELOAD_TRUE,
                            1>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_window<TT_DATA>* inWindowReverse,
                input_stream_cacc48* inCascade,
                output_window<TT_DATA>* restrict outWindow);
};

// Partially specialized classes for cascaded interface (final kernel in cascade), dual input, with reload, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_decimate_hb<TT_DATA,
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
                      DUAL_IP_DUAL,
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
                                                    DUAL_IP_DUAL,
                                                    USE_COEFF_RELOAD_TRUE,
                                                    2> {
   private:
   public:
    // Constructor
    fir_decimate_hb()
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
                            DUAL_IP_DUAL,
                            USE_COEFF_RELOAD_TRUE,
                            2>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_window<TT_DATA>* inWindowReverse,
                input_stream_cacc48* inCascade,
                output_window<TT_DATA>* restrict outWindow,
                output_window<TT_DATA>* restrict outWindow2);
};

//-----------------------------------------------------------------------------------------------------
// Partially specialized classes for cascaded interface (First kernel in cascade), single input, no reload
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP>
class fir_decimate_hb<TT_DATA,
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
                      TP_DUAL_IP,
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
                                                    TP_DUAL_IP,
                                                    USE_COEFF_RELOAD_FALSE,
                                                    1> {
   private:
   public:
    // Constructor
    fir_decimate_hb(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / 4 + 1])
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
                            TP_DUAL_IP,
                            USE_COEFF_RELOAD_FALSE,
                            1>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_stream_cacc48* outCascade,
                output_window<TT_DATA>* broadcastWindow);
};

// Partially specialized classes for cascaded interface (First kernel in cascade), dual input, no reload
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_decimate_hb<TT_DATA,
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
                      DUAL_IP_DUAL,
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
                                                    DUAL_IP_DUAL,
                                                    USE_COEFF_RELOAD_FALSE,
                                                    1> {
   private:
   public:
    // Constructor
    fir_decimate_hb(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / 4 + 1])
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
                            DUAL_IP_DUAL,
                            USE_COEFF_RELOAD_FALSE,
                            1>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_window<TT_DATA>* inWindowReverse,
                output_stream_cacc48* outCascade,
                output_window<TT_DATA>* broadcastWindow);
};

// Partially specialized classes for cascaded interface (First kernel in cascade), single input, with reload
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP>
class fir_decimate_hb<TT_DATA,
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
                      TP_DUAL_IP,
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
                                                    TP_DUAL_IP,
                                                    USE_COEFF_RELOAD_TRUE,
                                                    1> {
   private:
   public:
    // Constructor
    fir_decimate_hb()
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
                            TP_DUAL_IP,
                            USE_COEFF_RELOAD_TRUE,
                            1>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_stream_cacc48* outCascade,
                output_window<TT_DATA>* broadcastWindow,
                const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / 4 + 1]);
};

// Partially specialized classes for cascaded interface (First kernel in cascade), dual input, with reload
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_decimate_hb<TT_DATA,
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
                      DUAL_IP_DUAL,
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
                                                    DUAL_IP_DUAL,
                                                    USE_COEFF_RELOAD_TRUE,
                                                    1> {
   private:
   public:
    // Constructor
    fir_decimate_hb()
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
                            DUAL_IP_DUAL,
                            USE_COEFF_RELOAD_TRUE,
                            1>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_window<TT_DATA>* inWindowReverse,
                output_stream_cacc48* outCascade,
                output_window<TT_DATA>* broadcastWindow,
                const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / 4 + 1]);
};

//-----------------------------------------------------------------------------------------------------
// Partially specialized classes for cascaded interface (middle kernels in cascade), single input, no reload
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP>
class fir_decimate_hb<TT_DATA,
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
                      TP_DUAL_IP,
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
                                                    TP_DUAL_IP,
                                                    USE_COEFF_RELOAD_FALSE,
                                                    1> {
   private:
   public:
    // Constructor
    fir_decimate_hb(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / 4 + 1])
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
                            TP_DUAL_IP,
                            USE_COEFF_RELOAD_FALSE,
                            1>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_stream_cacc48* inCascade,
                output_stream_cacc48* outCascade,
                output_window<TT_DATA>* broadcastWindow);
};

// Partially specialized classes for cascaded interface (middle kernels in cascade), dual input, no reload
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_decimate_hb<TT_DATA,
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
                      DUAL_IP_DUAL,
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
                                                    DUAL_IP_DUAL,
                                                    USE_COEFF_RELOAD_FALSE,
                                                    1> {
   private:
   public:
    // Constructor
    fir_decimate_hb(const TT_COEFF (&taps)[(TP_FIR_LEN + 1) / 4 + 1])
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
                            DUAL_IP_DUAL,
                            USE_COEFF_RELOAD_FALSE,
                            1>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_window<TT_DATA>* inWindowReverse,
                input_stream_cacc48* inCascade,
                output_stream_cacc48* outCascade,
                output_window<TT_DATA>* broadcastWindow);
};

// Partially specialized classes for cascaded interface (middle kernels in cascade), single input, with reload
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP>
class fir_decimate_hb<TT_DATA,
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
                      TP_DUAL_IP,
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
                                                    TP_DUAL_IP,
                                                    USE_COEFF_RELOAD_TRUE,
                                                    1> {
   private:
   public:
    // Constructor
    fir_decimate_hb()
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
                            TP_DUAL_IP,
                            USE_COEFF_RELOAD_TRUE,
                            1>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_stream_cacc48* inCascade,
                output_stream_cacc48* outCascade,
                output_window<TT_DATA>* broadcastWindow);
};

// Partially specialized classes for cascaded interface (middle kernels in cascade), dual input, with reload
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_decimate_hb<TT_DATA,
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
                      DUAL_IP_DUAL,
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
                                                    DUAL_IP_DUAL,
                                                    USE_COEFF_RELOAD_TRUE,
                                                    1> {
   private:
   public:
    // Constructor
    fir_decimate_hb()
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
                            DUAL_IP_DUAL,
                            USE_COEFF_RELOAD_TRUE,
                            1>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_decimate_hb::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_window<TT_DATA>* inWindowReverse,
                input_stream_cacc48* inCascade,
                output_stream_cacc48* outCascade,
                output_window<TT_DATA>* broadcastWindow);
};
}
}
}
}
}
#endif // _DSPLIB_fir_decimate_hb_HPP_
