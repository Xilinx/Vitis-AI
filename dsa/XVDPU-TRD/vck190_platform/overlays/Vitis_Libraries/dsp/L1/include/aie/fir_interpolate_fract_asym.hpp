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
#ifndef _DSPLIB_fir_interpolate_fract_asym_HPP_
#define _DSPLIB_fir_interpolate_fract_asym_HPP_

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
#include "fir_interpolate_fract_asym_traits.hpp"

#include <vector>
#include <numeric> //for lcm calc
#include <array>   //for phase arrays

namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace interpolate_fract_asym {

//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_DECIMATE_FACTOR,
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
    static_assert(TP_INTERPOLATE_FACTOR >= FRACT_INTERPOLATE_FACTOR_MIN &&
                      TP_INTERPOLATE_FACTOR <= FRACT_INTERPOLATE_FACTOR_MAX,
                  "ERROR: TP_INTERPOLATE_FACTOR is out of the supported range.");
    static_assert(TP_DECIMATE_FACTOR < TP_INTERPOLATE_FACTOR,
                  "ERROR: TP_DECIMATE_FACTOR must be less than TP_INTERPOLATE_FACTOR for fractional interpolator.");
    static_assert(TP_FIR_LEN <= FIR_LEN_MAX, "ERROR: Max supported FIR length exceeded. ");
    static_assert(TP_FIR_RANGE_LEN >= FIR_LEN_MIN,
                  "ERROR: Illegal combination of design FIR length and cascade length, resulting in kernel FIR length "
                  "below minimum required value. ");
    static_assert(TP_SHIFT >= SHIFT_MIN && TP_SHIFT <= SHIFT_MAX, "ERROR: TP_SHIFT is out of the supported range.");
    static_assert(TP_RND >= ROUND_MIN && TP_RND <= ROUND_MAX, "ERROR: TP_RND is out of the supported range.");
    static_assert(fnEnumType<TT_DATA>() != enumUnknownType, "ERROR: TT_DATA is not a supported type.");
    static_assert(fnEnumType<TT_COEFF>() != enumUnknownType, "ERROR: TT_COEFF is not a supported type.");
    static_assert(fnFirInterpFractTypeSupport<TT_DATA, TT_COEFF>() != 0,
                  "ERROR: This library element currently supports TT_DATA of cint16 and TT_COEFF of int16.");
    static_assert(fnTypeCheckDataCoeffSize<TT_DATA, TT_COEFF>() != 0,
                  "ERROR: TT_DATA type less precise than TT_COEFF is not supported.");
    static_assert(fnTypeCheckDataCoeffCmplx<TT_DATA, TT_COEFF>() != 0,
                  "ERROR: real TT_DATA with complex TT_COEFF is not supported.");
    static_assert(fnTypeCheckDataCoeffFltInt<TT_DATA, TT_COEFF>() != 0,
                  "ERROR: a mix of float and integer types of TT_DATA and TT_COEFF is not supported.");
    static_assert((((TP_INPUT_WINDOW_VSIZE * TP_INTERPOLATE_FACTOR) % TP_DECIMATE_FACTOR) == 0),
                  "Number of input samples must give an integer number of output samples based on Interpolate Factor "
                  "and Decimate Factor");
    static_assert((((TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA)) % (128 / 8)) == 0),
                  "Number of input samples must align to 128 bits.");
    static_assert(TP_NUM_OUTPUTS > 0 && TP_NUM_OUTPUTS <= 2, "ERROR: only single or dual outputs are supported.");

    static constexpr unsigned int m_kColumns =
        fnNumColumnsIntFract<TT_DATA, TT_COEFF>(); // number of mult-adds per lane for main intrinsic
    static constexpr unsigned int m_kLanes =
        fnNumLanesIntFract<TT_DATA, TT_COEFF>(); // number of operations in parallel of this type combinations that the
                                                 // vector processor can do.
    static constexpr unsigned int m_kFirMarginLen = (TP_FIR_LEN + TP_INTERPOLATE_FACTOR - 1) / TP_INTERPOLATE_FACTOR;
    static constexpr unsigned int m_kPolyLen = (TP_FIR_RANGE_LEN + TP_INTERPOLATE_FACTOR - 1) / TP_INTERPOLATE_FACTOR;
    static constexpr unsigned int m_kFirLenCeilCols = CEIL(m_kPolyLen, m_kColumns);
    static constexpr unsigned int m_kNumOps = CEIL(m_kPolyLen, m_kColumns) / m_kColumns;
    static constexpr unsigned int m_kDataLoadsInReg =
        fnDataLoadsInRegIntFract<TT_DATA>(); // 4;  //ratio of sbuff to load size.
    static constexpr unsigned int m_kDataLoadVsize =
        fnDataLoadVsizeIntFract<TT_DATA>();               // 16;  //ie. upd_w loads a v4 of cint16
    static constexpr unsigned int m_kZbuffSize = 256 / 8; // kZbuffSize (256bit) - const for all data/coeff types
    static constexpr unsigned int m_kCoeffRegVsize = m_kZbuffSize / sizeof(TT_COEFF);
    static constexpr unsigned int m_kVOutSize =
        fnVOutSizeIntFract<TT_DATA, TT_COEFF>(); // This differs from m_kLanes for cint32/cint32
    static constexpr unsigned int m_kFirCoeffOffset =
        fnFirRangeOffset<CEIL(TP_FIR_LEN, TP_INTERPOLATE_FACTOR),
                         TP_CASC_LEN,
                         TP_KERNEL_POSITION,
                         TP_INTERPOLATE_FACTOR>(); // FIR Cascade Offset for this kernel position
    // static constexpr unsigned int  m_kFirCoeffOffset    =
    // fnFirRangeOffset<TP_FIR_LEN,TP_CASC_LEN,TP_KERNEL_POSITION,TP_INTERPOLATE_FACTOR>();  // FIR Cascade Offset for
    // this kernel position
    // static constexpr unsigned int  m_kFirRangeOffset    =
    // fnFirRangeOffset<TP_FIR_LEN,TP_CASC_LEN,TP_KERNEL_POSITION,TP_INTERPOLATE_FACTOR>()/TP_INTERPOLATE_FACTOR;  //
    // FIR Cascade Offset for this kernel position
    static constexpr unsigned int m_kFirRangeOffset =
        m_kFirCoeffOffset / TP_INTERPOLATE_FACTOR; // FIR Cascade Offset for this kernel position
    static constexpr unsigned int m_kFirMarginOffset =
        fnFirMargin<m_kFirMarginLen, TT_DATA>() - m_kFirMarginLen + 1; // FIR Margin Offset.
    static constexpr unsigned int m_kWinAccessByteSize =
        fnWinAccessByteSize<TT_DATA, TT_COEFF>(); // The memory data path is min 128-bits wide for vector operations
    static constexpr unsigned int m_kFirInitOffset = m_kFirRangeOffset + m_kFirMarginOffset;
    static constexpr unsigned int m_kDataBuffXOffset =
        m_kFirInitOffset % (m_kWinAccessByteSize / sizeof(TT_DATA)); // Remainder of m_kFirInitOffset divided by 128bit
    //   int marginOffset = ((int)((firLen+(I-1))/I)-1);

    // the lowest common multiple defines how many output samples it would take
    // to have lane 0 with polyphase 0 again. Divide by lanes to get number of
    // vector outputs.
    // Ensure that we have at least 2 polyphases. Saw an issue with windowDecs and xStart where data is awkward with
    // alignment of 128bits. Fractional rate of 4/3.
    // Data is still awkward for interp = 8..

    // The hardcoded 4 and 2 are probably to do with number of loads in databuffer.
    static constexpr unsigned int m_kPolyphaseLaneAlias =
        (my_lcm(TP_INTERPOLATE_FACTOR, m_kLanes) / m_kLanes > 1)
            ? my_lcm(TP_INTERPOLATE_FACTOR, m_kLanes) / m_kLanes
            : (TP_INTERPOLATE_FACTOR == 4)
                  ? 2
                  : 4; // 4 works for interp 8, and probably any other interp that has this behaviour.
    // If interp fits within one set of lanes, then we don't need to duplicate
    // coeff storage into phases, as this can be acheived through zoffsets only.
    static constexpr unsigned int m_kPolyphaseCoeffAlias =
        TP_INTERPOLATE_FACTOR <= m_kLanes ? 1 : m_kPolyphaseLaneAlias;

    static constexpr unsigned int m_kNumOutputs =
        ((unsigned int)((TP_INPUT_WINDOW_VSIZE * TP_INTERPOLATE_FACTOR) / TP_DECIMATE_FACTOR));
    static constexpr unsigned int m_kLsize =
        m_kNumOutputs /
        (m_kPolyphaseLaneAlias *
         m_kVOutSize); // loop length, given that <m_kVOutSize> samples are output per m_kPolyphaseLaneAlias loop

    static_assert(m_kNumOutputs % (m_kVOutSize) == 0,
                  "ERROR: output window size must be a multiple of number of lanes. ");
    static_assert(m_kNumOutputs % (m_kPolyphaseLaneAlias * m_kVOutSize) == 0,
                  "ERROR: due to architectural optimisation, this window size is not currently supported. Please use a "
                  "INPUT_WINDOW_SIZE that will give a number of output samples which is a multiple of lowest common "
                  "multiple of Lanes and Interpolate Factor.");

    // Typically, we need less data than coefficients for each operation in this fractional interpolator.
    // we're going to need MIN(interp,lanes)*num_cols coefficients and MIN(lanes,rndUp(((LCM*interp)/(Deci*lanes)))
    static_assert(sizeof(TT_DATA) * m_kLanes <= m_kZbuffSize,
                  "ERROR: Invalid assumption in archtecture. Can't fit enough data into selected (Z) buffer.");

    // Coefficient Load Size - number of samples in 256-bits
    static constexpr unsigned int m_kCoeffLoadSize = 256 / 8 / sizeof(TT_COEFF);
    TT_COEFF chess_storage(% chess_alignof(v8cint16))
        m_oldInTaps[CEIL(TP_FIR_LEN, m_kCoeffLoadSize)]; // Previous user input coefficients with zero padding
    bool m_coeffnEq;                                     // Are coefficients sets equal?

    // inline definition of the struct which is declared in traits.
    static constexpr firParamsTrait params{
        sizeof(TT_DATA),                         // dataSizeBytes;
        TP_FIR_LEN,                              // firLen ;
        m_kDataLoadVsize,                        // loadSize ;
        m_kDataLoadsInReg* m_kDataLoadVsize,     // dataBuffSamples ;
        m_kWinAccessByteSize,                    // alignWindowReadBytes ;
        m_kFirMarginLen - 1 - m_kFirRangeOffset, //  marginOffsetIndex ;
        0,                                       //  rangeOffsetIndex ;
    };

    template <typename T>
    using polyphaseArray = std::array<T, m_kPolyphaseLaneAlias>;

    // The coefficients array must include zero padding up to a multiple of the number of columns
    // the MAC intrinsic used to eliminate the accidental inclusion of terms beyond the FIR length.
    // Since this zero padding cannot be applied to the class-external coefficient array
    // the supplied taps are copied to an internal array, m_internalTaps, which can be padded.
    TT_COEFF chess_storage(% chess_alignof(v8cint16))
        m_internalTaps[m_kPolyphaseCoeffAlias][m_kNumOps][m_kColumns][m_kLanes] = {
            nullElem<TT_COEFF>()}; // Filter taps/coefficients

    // Filter kernel architecture
    void filterIntFractAsym(T_inputIF<TP_CASC_IN, TT_DATA> inInterface, T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);
    void filterSelectArch(T_inputIF<TP_CASC_IN, TT_DATA> inInterface, T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);

   public:
    // Access function for AIE Synthesizer
    unsigned int get_m_kArch() { return 0; }; // no distinct architectures yet

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
        // Loads taps/coefficients
        // Need nColumns more coefficients each op (including padding)
        // there are FirLenCeilCols/InterpRate columns split over numOps at nColumns per op
        for (int coeffPhase = 0; coeffPhase < m_kPolyphaseCoeffAlias; coeffPhase++) {
            for (int op = 0; op < m_kNumOps; op++) {
                for (int col = 0; col < m_kColumns; col++) {
                    // polyphaseI
                    // We know that the interpolation rate can not be > number of lanes
                    for (int poly = 0; poly < m_kLanes; poly++) {
                        unsigned I = TP_INTERPOLATE_FACTOR;
                        unsigned D = TP_DECIMATE_FACTOR;
                        // non-reversed indexes
                        // Reorder coefficients so that they are in the order they will be
                        // used in due to decimate factor. This means that zoffsets will
                        // always be 01234012,34012340 regardless of decimation factor.
                        // Instead of 04321043,21..
                        // If you draw the polyphase diagram, this is the cascade column index for each polyphase.
                        int polyPhaseCol = op * m_kColumns + col;

                        int polyphaseIndex = (((coeffPhase * m_kLanes + poly) * D) % I);
                        // We could modulus poly by interpRate, but we're
                        // already going to pad taps array for values over interpRate.
                        int tapIndexFwd = polyphaseIndex + polyPhaseCol * I;
                        // Coefficient reversal, retaining order of polyphases
                        // int tapIndexRev = TP_FIR_LEN +
                        int tapIndexRev = CEIL(TP_FIR_LEN, TP_INTERPOLATE_FACTOR) + polyphaseIndex -
                                          (polyPhaseCol + 1) * I - m_kFirCoeffOffset;

                        if (poly < TP_INTERPOLATE_FACTOR &&
                            // tapIndexRev >= TP_FIR_LEN - m_kFirCoeffOffset - TP_FIR_RANGE_LEN &&
                            tapIndexRev >= CEIL(TP_FIR_LEN, TP_INTERPOLATE_FACTOR) - m_kFirCoeffOffset -
                                               CEIL(TP_FIR_RANGE_LEN, TP_INTERPOLATE_FACTOR) &&
                            tapIndexRev < TP_FIR_LEN) {
                            m_internalTaps[coeffPhase][op][col][poly] = taps[tapIndexRev];
                        } else {
                            // padding, interpRate doesn't fit into m_kLanes
                            // or fir_len/interp doesn't fit into m_kColummns
                            // or fir_len doesn't fit into interpolate factor
                            // This padding is nessecary in order to have coef reads at a
                            // 256b boundary.
                            m_internalTaps[coeffPhase][op][col][poly] = nullElem<TT_COEFF>(); // 0 for the type.
                        }
                    }
                }
            }
        }
    }

    // Filter kernel for static coefficient designs
    void filterKernel(T_inputIF<TP_CASC_IN, TT_DATA> inInterface, T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);

    // Filter kernel for reloadable coefficient designs
    void filterKernel(T_inputIF<TP_CASC_IN, TT_DATA> inInterface,
                      T_outputIF<TP_CASC_OUT, TT_DATA> outInterface,
                      const TT_COEFF (&inTaps)[TP_FIR_LEN]);
    void filterKernelRtp(T_inputIF<TP_CASC_IN, TT_DATA> inInterface, T_outputIF<TP_CASC_OUT, TT_DATA> outInterface);
};

//-----------------------------------------------------------------------------------------------------
// Single kernel specialization. No cascade ports, static coefficients
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          bool TP_CASC_IN = CASC_IN_FALSE,
          bool TP_CASC_OUT = CASC_OUT_FALSE,
          unsigned int TP_FIR_RANGE_LEN = TP_FIR_LEN,
          unsigned int TP_KERNEL_POSITION = 0,
          unsigned int TP_CASC_LEN = 1,
          unsigned int TP_USE_COEFF_RELOAD = USE_COEFF_RELOAD_FALSE,
          unsigned int TP_NUM_OUTPUTS = 1>
class fir_interpolate_fract_asym : public kernelFilterClass<TT_DATA,
                                                            TT_COEFF,
                                                            TP_FIR_LEN,
                                                            TP_INTERPOLATE_FACTOR,
                                                            TP_DECIMATE_FACTOR,
                                                            TP_SHIFT,
                                                            TP_RND,
                                                            TP_INPUT_WINDOW_VSIZE,
                                                            CASC_IN_FALSE,
                                                            CASC_OUT_FALSE,
                                                            TP_FIR_LEN,
                                                            0,
                                                            1,
                                                            USE_COEFF_RELOAD_FALSE,
                                                            1> {
   private:
   public:
    // Constructor
    fir_interpolate_fract_asym(const TT_COEFF (&taps)[TP_FIR_LEN])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
                            TP_DECIMATE_FACTOR,
                            TP_SHIFT,
                            TP_RND,
                            TP_INPUT_WINDOW_VSIZE,
                            CASC_IN_FALSE,
                            CASC_OUT_FALSE,
                            TP_FIR_LEN,
                            0,
                            1,
                            USE_COEFF_RELOAD_FALSE,
                            1>(taps) {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_fract_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow, output_window<TT_DATA>* outWindow);
};

// specialization, single kernel, static coeff, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
class fir_interpolate_fract_asym<TT_DATA,
                                 TT_COEFF,
                                 TP_FIR_LEN,
                                 TP_INTERPOLATE_FACTOR,
                                 TP_DECIMATE_FACTOR,
                                 TP_SHIFT,
                                 TP_RND,
                                 TP_INPUT_WINDOW_VSIZE,
                                 CASC_IN_FALSE,
                                 CASC_OUT_FALSE,
                                 TP_FIR_LEN,
                                 0,
                                 1,
                                 USE_COEFF_RELOAD_FALSE,
                                 2> : public kernelFilterClass<TT_DATA,
                                                               TT_COEFF,
                                                               TP_FIR_LEN,
                                                               TP_INTERPOLATE_FACTOR,
                                                               TP_DECIMATE_FACTOR,
                                                               TP_SHIFT,
                                                               TP_RND,
                                                               TP_INPUT_WINDOW_VSIZE,
                                                               CASC_IN_FALSE,
                                                               CASC_OUT_FALSE,
                                                               TP_FIR_LEN,
                                                               0,
                                                               1,
                                                               USE_COEFF_RELOAD_FALSE,
                                                               2> {
   private:
   public:
    // Constructor
    fir_interpolate_fract_asym(const TT_COEFF (&taps)[TP_FIR_LEN])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
                            TP_DECIMATE_FACTOR,
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_fract_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow, output_window<TT_DATA>* outWindow, output_window<TT_DATA>* outWindow2);
};

// Partially specialized classes for cascaded interface. Single kernel, reloadable coefficients, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
class fir_interpolate_fract_asym<TT_DATA,
                                 TT_COEFF,
                                 TP_FIR_LEN,
                                 TP_INTERPOLATE_FACTOR,
                                 TP_DECIMATE_FACTOR,
                                 TP_SHIFT,
                                 TP_RND,
                                 TP_INPUT_WINDOW_VSIZE,
                                 CASC_IN_FALSE,
                                 CASC_OUT_FALSE,
                                 TP_FIR_LEN,
                                 0,
                                 1,
                                 USE_COEFF_RELOAD_TRUE,
                                 1> : public kernelFilterClass<TT_DATA,
                                                               TT_COEFF,
                                                               TP_FIR_LEN,
                                                               TP_INTERPOLATE_FACTOR,
                                                               TP_DECIMATE_FACTOR,
                                                               TP_SHIFT,
                                                               TP_RND,
                                                               TP_INPUT_WINDOW_VSIZE,
                                                               CASC_IN_FALSE,
                                                               CASC_OUT_FALSE,
                                                               TP_FIR_LEN,
                                                               0,
                                                               1,
                                                               USE_COEFF_RELOAD_TRUE,
                                                               1> {
   private:
   public:
    // Constructor
    fir_interpolate_fract_asym()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
                            TP_DECIMATE_FACTOR,
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_fract_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_window<TT_DATA>* outWindow,
                const TT_COEFF (&inTaps)[TP_FIR_LEN]);
};

// Partially specialized classes for cascaded interface. Single kernel, reloadable coefficients, dual  output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
class fir_interpolate_fract_asym<TT_DATA,
                                 TT_COEFF,
                                 TP_FIR_LEN,
                                 TP_INTERPOLATE_FACTOR,
                                 TP_DECIMATE_FACTOR,
                                 TP_SHIFT,
                                 TP_RND,
                                 TP_INPUT_WINDOW_VSIZE,
                                 CASC_IN_FALSE,
                                 CASC_OUT_FALSE,
                                 TP_FIR_LEN,
                                 0,
                                 1,
                                 USE_COEFF_RELOAD_TRUE,
                                 2> : public kernelFilterClass<TT_DATA,
                                                               TT_COEFF,
                                                               TP_FIR_LEN,
                                                               TP_INTERPOLATE_FACTOR,
                                                               TP_DECIMATE_FACTOR,
                                                               TP_SHIFT,
                                                               TP_RND,
                                                               TP_INPUT_WINDOW_VSIZE,
                                                               CASC_IN_FALSE,
                                                               CASC_OUT_FALSE,
                                                               TP_FIR_LEN,
                                                               0,
                                                               1,
                                                               USE_COEFF_RELOAD_TRUE,
                                                               2> {
   private:
   public:
    // Constructor
    fir_interpolate_fract_asym()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
                            TP_DECIMATE_FACTOR,
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_fract_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_window<TT_DATA>* outWindow,
                output_window<TT_DATA>* outWindow2,
                const TT_COEFF (&inTaps)[TP_FIR_LEN]);
};

// Partially specialized classes for cascaded interface (final kernel in cascade) with static coefficients, single
// output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_interpolate_fract_asym<TT_DATA,
                                 TT_COEFF,
                                 TP_FIR_LEN,
                                 TP_INTERPOLATE_FACTOR,
                                 TP_DECIMATE_FACTOR,
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
                                                               TP_DECIMATE_FACTOR,
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
    fir_interpolate_fract_asym(const TT_COEFF (&taps)[TP_FIR_LEN])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
                            TP_DECIMATE_FACTOR,
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_fract_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow, input_stream_cacc48* inCascade, output_window<TT_DATA>* outWindow);
};

// Partially specialized classes for cascaded interface (final kernel in cascade) with static coefficients, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_interpolate_fract_asym<TT_DATA,
                                 TT_COEFF,
                                 TP_FIR_LEN,
                                 TP_INTERPOLATE_FACTOR,
                                 TP_DECIMATE_FACTOR,
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
                                                               TP_DECIMATE_FACTOR,
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
    fir_interpolate_fract_asym(const TT_COEFF (&taps)[TP_FIR_LEN])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
                            TP_DECIMATE_FACTOR,
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_fract_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_stream_cacc48* inCascade,
                output_window<TT_DATA>* outWindow,
                output_window<TT_DATA>* outWindow2);
};

// Partially specialized classes for cascaded interface (final kernel in cascade) with reloadable coefficients, single
// output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_interpolate_fract_asym<TT_DATA,
                                 TT_COEFF,
                                 TP_FIR_LEN,
                                 TP_INTERPOLATE_FACTOR,
                                 TP_DECIMATE_FACTOR,
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
                                                               TP_DECIMATE_FACTOR,
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
    fir_interpolate_fract_asym()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
                            TP_DECIMATE_FACTOR,
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_fract_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow, input_stream_cacc48* inCascade, output_window<TT_DATA>* outWindow);
};

// Partially specialized classes for cascaded interface (final kernel in cascade) with reloadable coefficients, dual
// output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_interpolate_fract_asym<TT_DATA,
                                 TT_COEFF,
                                 TP_FIR_LEN,
                                 TP_INTERPOLATE_FACTOR,
                                 TP_DECIMATE_FACTOR,
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
                                                               TP_DECIMATE_FACTOR,
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
    fir_interpolate_fract_asym()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
                            TP_DECIMATE_FACTOR,
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_fract_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_stream_cacc48* inCascade,
                output_window<TT_DATA>* outWindow,
                output_window<TT_DATA>* outWindow2);
};

// Partially specialized classes for cascaded interface (First kernel in cascade) with static coefficients
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_interpolate_fract_asym<TT_DATA,
                                 TT_COEFF,
                                 TP_FIR_LEN,
                                 TP_INTERPOLATE_FACTOR,
                                 TP_DECIMATE_FACTOR,
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
                                                               TP_DECIMATE_FACTOR,
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
    fir_interpolate_fract_asym(const TT_COEFF (&taps)[TP_FIR_LEN])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
                            TP_DECIMATE_FACTOR,
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_fract_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_stream_cacc48* outCascade,
                output_window<TT_DATA>* broadcastWindow);
};

// Partially specialized classes for cascaded interface (First kernel in cascade) with reloadable coefficients
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_interpolate_fract_asym<TT_DATA,
                                 TT_COEFF,
                                 TP_FIR_LEN,
                                 TP_INTERPOLATE_FACTOR,
                                 TP_DECIMATE_FACTOR,
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
                                                               TP_DECIMATE_FACTOR,
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
    fir_interpolate_fract_asym()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
                            TP_DECIMATE_FACTOR,
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_fract_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                output_stream_cacc48* outCascade,
                output_window<TT_DATA>* broadcastWindow,
                const TT_COEFF (&inTaps)[TP_FIR_LEN]);
};

//-----------------------------------------------------------------------------------------------------
// Partially specialized classes for cascaded interface (middle kernels in cascade) with static coefficients
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_interpolate_fract_asym<TT_DATA,
                                 TT_COEFF,
                                 TP_FIR_LEN,
                                 TP_INTERPOLATE_FACTOR,
                                 TP_DECIMATE_FACTOR,
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
                                                               TP_DECIMATE_FACTOR,
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
    fir_interpolate_fract_asym(const TT_COEFF (&taps)[TP_FIR_LEN])
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
                            TP_DECIMATE_FACTOR,
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_fract_asym::filter); }

    // FIR
    void filter(input_window<TT_DATA>* inWindow,
                input_stream_cacc48* inCascade,
                output_stream_cacc48* outCascade,
                output_window<TT_DATA>* broadcastWindow);
};

// Partially specialized classes for cascaded interface (middle kernels in cascade) with reloadable coefficients
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
class fir_interpolate_fract_asym<TT_DATA,
                                 TT_COEFF,
                                 TP_FIR_LEN,
                                 TP_INTERPOLATE_FACTOR,
                                 TP_DECIMATE_FACTOR,
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
                                                               TP_DECIMATE_FACTOR,
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
    fir_interpolate_fract_asym()
        : kernelFilterClass<TT_DATA,
                            TT_COEFF,
                            TP_FIR_LEN,
                            TP_INTERPOLATE_FACTOR,
                            TP_DECIMATE_FACTOR,
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
    static void registerKernelClass() { REGISTER_FUNCTION(fir_interpolate_fract_asym::filter); }

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
#endif // _DSPLIB_fir_interpolate_fract_asym_HPP_
