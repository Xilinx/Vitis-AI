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
Asymmetric Decimator FIR kernal code.
This file captures the body of run-time code for the kernal class.

Coding conventions
  TT_      template type suffix
  TP_      template parameter suffix
*/

#pragma once
#include <adf.h>

#define __NEW_WINDOW_H__ 1
// #define __AIEARCH__ 1
// #define __AIENGINE__ 1
#define __AIE_API_USE_NATIVE_1024B_VECTOR__
#include "aie_api/aie_adf.hpp"
#include "kernel_api_utils.hpp"
#include "fir_decimate_asym.hpp"
#include "fir_decimate_asym_utils.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace decimate_asym {

#define Y_BUFFER ya
#define X_BUFFER xd
#define Z_BUFFER wc0

// Asymmetrical Decimation FIR Kernel Function
// According to template parameter the input may be a window, or window and cascade input
// Similarly the output interface may be a window or a cascade output
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          bool TP_CASC_IN,
          bool TP_CASC_OUT,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_USE_COEFF_RELOAD,
          unsigned int TP_NUM_OUTPUTS>
inline void kernelFilterClass<TT_DATA,
                              TT_COEFF,
                              TP_FIR_LEN,
                              TP_DECIMATE_FACTOR,
                              TP_SHIFT,
                              TP_RND,
                              TP_INPUT_WINDOW_VSIZE,
                              TP_CASC_IN,
                              TP_CASC_OUT,
                              TP_FIR_RANGE_LEN,
                              TP_KERNEL_POSITION,
                              TP_CASC_LEN,
                              TP_USE_COEFF_RELOAD,
                              TP_NUM_OUTPUTS>::filterKernel(T_inputIF<TP_CASC_IN, TT_DATA> inInterface,
                                                            T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    windowBroadcast<TT_DATA, TP_INPUT_WINDOW_VSIZE + fnFirMargin<TP_FIR_LEN, TT_DATA>()>(inInterface, outInterface);
    filterSelectArch(inInterface, outInterface);
}

// Asymmetrical Decimation FIR Kernel Function - overloaded (not specialised)
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          bool TP_CASC_IN,
          bool TP_CASC_OUT,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_USE_COEFF_RELOAD,
          unsigned int TP_NUM_OUTPUTS>
inline void kernelFilterClass<TT_DATA,
                              TT_COEFF,
                              TP_FIR_LEN,
                              TP_DECIMATE_FACTOR,
                              TP_SHIFT,
                              TP_RND,
                              TP_INPUT_WINDOW_VSIZE,
                              TP_CASC_IN,
                              TP_CASC_OUT,
                              TP_FIR_RANGE_LEN,
                              TP_KERNEL_POSITION,
                              TP_CASC_LEN,
                              TP_USE_COEFF_RELOAD,
                              TP_NUM_OUTPUTS>::filterKernel(T_inputIF<TP_CASC_IN, TT_DATA> inInterface,
                                                            T_outputIF<TP_CASC_OUT, TT_DATA> outInterface,
                                                            const TT_COEFF (&inTaps)[TP_FIR_LEN]) {
    windowBroadcast<TT_DATA, TP_INPUT_WINDOW_VSIZE + fnFirMargin<TP_FIR_LEN, TT_DATA>()>(inInterface, outInterface);
    m_coeffnEq = rtpCompare(inTaps, m_oldInTaps);

    sendRtpTrigger(m_coeffnEq, outInterface);
    if (m_coeffnEq) { // Coefficients have changed
        bufferReload(inTaps, m_oldInTaps, outInterface);
        firReload(inTaps);
        chess_memory_fence();
    }
    filterSelectArch(inInterface, outInterface);
}

// Asymmetrical Decimation FIR Kernel Function
// According to template parameter the input may be a window, or window and cascade input
// Similarly the output interface may be a window or a cascade output
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          bool TP_CASC_IN,
          bool TP_CASC_OUT,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_USE_COEFF_RELOAD,
          unsigned int TP_NUM_OUTPUTS>
inline void kernelFilterClass<TT_DATA,
                              TT_COEFF,
                              TP_FIR_LEN,
                              TP_DECIMATE_FACTOR,
                              TP_SHIFT,
                              TP_RND,
                              TP_INPUT_WINDOW_VSIZE,
                              TP_CASC_IN,
                              TP_CASC_OUT,
                              TP_FIR_RANGE_LEN,
                              TP_KERNEL_POSITION,
                              TP_CASC_LEN,
                              TP_USE_COEFF_RELOAD,
                              TP_NUM_OUTPUTS>::filterKernelRtp(T_inputIF<TP_CASC_IN, TT_DATA> inInterface,
                                                               T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    windowBroadcast<TT_DATA, TP_INPUT_WINDOW_VSIZE + fnFirMargin<TP_FIR_LEN, TT_DATA>()>(inInterface, outInterface);
    m_coeffnEq = getRtpTrigger(); // 0 - equal, 1 - not equal

    sendRtpTrigger(m_coeffnEq, outInterface);
    if (m_coeffnEq) { // Coefficients have changed
        bufferReload<TT_DATA, TT_COEFF, TP_FIR_LEN>(inInterface, m_oldInTaps, outInterface);
        firReload(m_oldInTaps);
        chess_memory_fence();
    }
    filterSelectArch(inInterface, outInterface);
};
// Asymmetrical Decimation FIR Kernel Function - overloaded (not specialised)
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          bool TP_CASC_IN,
          bool TP_CASC_OUT,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_USE_COEFF_RELOAD,
          unsigned int TP_NUM_OUTPUTS>
inline void kernelFilterClass<TT_DATA,
                              TT_COEFF,
                              TP_FIR_LEN,
                              TP_DECIMATE_FACTOR,
                              TP_SHIFT,
                              TP_RND,
                              TP_INPUT_WINDOW_VSIZE,
                              TP_CASC_IN,
                              TP_CASC_OUT,
                              TP_FIR_RANGE_LEN,
                              TP_KERNEL_POSITION,
                              TP_CASC_LEN,
                              TP_USE_COEFF_RELOAD,
                              TP_NUM_OUTPUTS>::filterSelectArch(T_inputIF<TP_CASC_IN, TT_DATA> inInterface,
                                                                T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    set_rnd(TP_RND);
    set_sat();

    windowAcquire(inInterface);
    if
        constexpr(m_kArch == kArch1BuffBasic) { filter1BuffBasic(inInterface, outInterface); }
    else if
        constexpr(m_kArch == kArch1BuffIncrStrobe) { filter1BuffIncrStrobe(inInterface, outInterface); }
    windowRelease(inInterface);
}

// -------------------------------------------------------------- Basic
// -------------------------------------------------------------- //
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          bool TP_CASC_IN,
          bool TP_CASC_OUT,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_USE_COEFF_RELOAD,
          unsigned int TP_NUM_OUTPUTS>
void kernelFilterClass<TT_DATA,
                       TT_COEFF,
                       TP_FIR_LEN,
                       TP_DECIMATE_FACTOR,
                       TP_SHIFT,
                       TP_RND,
                       TP_INPUT_WINDOW_VSIZE,
                       TP_CASC_IN,
                       TP_CASC_OUT,
                       TP_FIR_RANGE_LEN,
                       TP_KERNEL_POSITION,
                       TP_CASC_LEN,
                       TP_USE_COEFF_RELOAD,
                       TP_NUM_OUTPUTS>::filter1BuffBasic(T_inputIF<TP_CASC_IN, TT_DATA> inInterface,
                                                         T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    // The plus one in this calculation converts index to width
    static constexpr unsigned int m_kInitialLoads1Buff =
        CEIL(m_kDataBuffXOffset + (TP_DECIMATE_FACTOR * (m_kLanes - 1)) + (m_kColumns - 1) + 1, m_kInitLoadVsize) /
        m_kInitLoadVsize;
    // static constexpr unsigned int  m_kInitialLoads1Buff= m_kInitLoads1BBasic;

    using coe_type = typename T_buff_256b<TT_COEFF>::v_type;
    using buf_type = typename T_buff_1024b<TT_DATA>::v_type;

    T_buff_256b<TT_COEFF>* restrict coeff = (T_buff_256b<TT_COEFF>*)m_internalTaps;
    T_buff_256b<TT_COEFF> coe0; // register for coeff values.
    T_buff_1024b<TT_DATA> sbuff;
    T_accDecAsym<TT_DATA, TT_COEFF> acc;
    T_outValDecAsym<TT_DATA, TT_COEFF> outVal;
    T_buff_256b<TT_DATA> readData; // input data read from window, bound for sbuff
    unsigned int dataLoaded, dataNeeded, numDataLoads, initNumDataLoads;
    unsigned int xstart, xstartUpper, splice; // upper is used for upper lanes in high decimation factor handling

    // Move data pointer away from data consumed by previous cascades
    // Move only by  multiples of 128bit. Cascade phase remainder goes to m_kDataBuffXOffset
    window_incr(inInterface.inWindow, (TRUNC((m_kFirInitOffset), (m_kWinAccessByteSize / sizeof(TT_DATA)))));

    // This loop creates the output window data. In each iteration a vector of samples is output
    for (unsigned i = 0; i < m_kLsize; i++) chess_prepare_for_pipelining chess_loop_range(m_kLsize, ) {
            coeff = ((T_buff_256b<TT_COEFF>*)m_internalTaps);
            coe_type chess_storage(Z_BUFFER) coe_h = coeff->val;
            coeff++;
            coe0.val = coe_h;

            // Preamble, calc number of samples for first mul.
            splice = 0;
            numDataLoads = 0;
            initNumDataLoads = 0;
            dataLoaded = 0;
            dataNeeded = m_kDataBuffXOffset + (m_kLanes - 1) * TP_DECIMATE_FACTOR + m_kColumns - 1 +
                         1; // final plus one turns index into width

#pragma unroll(m_kInitialLoads1Buff)
            for (int initLoads = 0; initLoads < m_kInitialLoads1Buff; ++initLoads) {
                readData = window_readincr_256b<TT_DATA>(inInterface.inWindow); // Read 256b from input window
                buf_type chess_storage(Y_BUFFER) sb =
                    upd_w(sbuff.val, initNumDataLoads % m_kInitLoadsInReg, readData.val);
                sbuff.val = sb;
                dataLoaded += m_kInitLoadVsize;
                initNumDataLoads++;
            }

            // Read cascade input. Do nothing if cascade input not present.
            acc = readCascade<TT_DATA, TT_COEFF>(inInterface, acc);

            xstart = m_kDataBuffXOffset;
            xstartUpper = xstart + m_kLanes / 2 * TP_DECIMATE_FACTOR; // upper half of lanes

            // Init Vector operation. VMUL if cascade not present, otherwise VMAC
            acc = initMacDecAsym1Buff<TT_DATA, TT_COEFF, m_kDFX, TP_DECIMATE_FACTOR>(
                inInterface, acc, sbuff, xstart, coe0, 0, m_kDecimateOffsets, xstartUpper);

#pragma unroll(GUARD_ZERO((m_kFirLenCeilCols / (m_kColumns) - 1)))
            for (int op = m_kColumns; op < m_kFirLenCeilCols; op += m_kColumns) {
                dataNeeded += m_kColumns;
                if (dataNeeded > dataLoaded) {
                    splice =
                        (numDataLoads + initNumDataLoads * m_kInitLoadVsize / m_kDataLoadVsize) % m_kDataLoadsInReg;
                    fnLoadXIpData<TT_DATA, m_kDataLoadSize>(sbuff, splice, inInterface.inWindow);
                    dataLoaded += m_kDataLoadVsize;
                    numDataLoads++;
                }
                if (op % m_kCoeffRegVsize == 0) {
                    coe_type chess_storage(Z_BUFFER) coe_h = coeff->val;
                    coeff++;
                    coe0.val = coe_h;
                }
                xstart += m_kColumns;
                xstartUpper += m_kColumns;

                acc = macDecAsym1Buff<TT_DATA, TT_COEFF, m_kDFX, TP_DECIMATE_FACTOR>(
                    acc, sbuff, xstart, coe0, (op % m_kCoeffRegVsize), m_kDecimateOffsets, xstartUpper);
            }

            // Go back by the number of input samples loaded minus  (i.e forward) by the number of samples consumed
            window_decr(inInterface.inWindow,
                        (initNumDataLoads * m_kInitLoadVsize + m_kDataLoadVsize * numDataLoads -
                         m_kVOutSize * TP_DECIMATE_FACTOR)); // return read pointer to start of next chunk of window.
            // Write cascade. Do nothing if cascade not present.
            writeCascade<TT_DATA, TT_COEFF>(outInterface, acc);

            outVal = shiftAndSaturateDecAsym(acc, TP_SHIFT);
            // Write to output window
            writeWindow<TT_DATA, TT_COEFF, TP_NUM_OUTPUTS>(outInterface, outVal);
        } // for i
};

// -------------------------------------------------------------- IncrLoads
// -------------------------------------------------------------- //
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          bool TP_CASC_IN,
          bool TP_CASC_OUT,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_USE_COEFF_RELOAD,
          unsigned int TP_NUM_OUTPUTS>
void kernelFilterClass<TT_DATA,
                       TT_COEFF,
                       TP_FIR_LEN,
                       TP_DECIMATE_FACTOR,
                       TP_SHIFT,
                       TP_RND,
                       TP_INPUT_WINDOW_VSIZE,
                       TP_CASC_IN,
                       TP_CASC_OUT,
                       TP_FIR_RANGE_LEN,
                       TP_KERNEL_POSITION,
                       TP_CASC_LEN,
                       TP_USE_COEFF_RELOAD,
                       TP_NUM_OUTPUTS>::filter1BuffIncrStrobe(T_inputIF<TP_CASC_IN, TT_DATA> inInterface,
                                                              T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    static constexpr unsigned int m_kRepeatFactor =
        TP_DECIMATE_FACTOR % 2 == 0 ? m_kInitLoadsInReg
                                    : m_kSamplesInDataBuff / m_kVOutSize; // only FACTORS of 2 or 3 supported
    static constexpr unsigned int m_kDataUsedRptp1 =
        m_kDataBuffXOffset + TP_FIR_RANGE_LEN - 1 + TP_DECIMATE_FACTOR * m_kVOutSize * (m_kRepeatFactor + 1);
    static constexpr unsigned int m_kDataUsedRpt =
        m_kDataBuffXOffset + TP_FIR_RANGE_LEN - 1 + TP_DECIMATE_FACTOR * m_kVOutSize * (m_kRepeatFactor);
    static constexpr unsigned int m_kLoadsAfterRepeatP1 = CEIL(m_kDataUsedRptp1, m_kInitLoadVsize) / m_kInitLoadVsize;
    static constexpr unsigned int m_kLoadsAfterRepeat = CEIL(m_kDataUsedRpt, m_kInitLoadVsize) / m_kInitLoadVsize;
    static constexpr unsigned int m_kInitDataNeeded =
        m_kDataBuffXOffset + TP_FIR_RANGE_LEN + TP_DECIMATE_FACTOR * (m_kLanes - 1);
    static constexpr unsigned int m_kInitLoads1BIncLd =
        CEIL(m_kInitDataNeeded, m_kInitLoadVsize) / m_kInitLoadVsize + m_kLoadsAfterRepeat - m_kLoadsAfterRepeatP1;
    static constexpr unsigned int m_kInitialLoads1Buff = m_kInitLoads1BIncLd;

    T_buff_256b<TT_COEFF>* restrict coeff = (T_buff_256b<TT_COEFF>*)m_internalTaps;
    T_buff_256b<TT_COEFF> coe0;  // register for coeff values.
    T_buff_1024b<TT_DATA> sbuff; // input data value cache.
    T_accDecAsym<TT_DATA, TT_COEFF> acc;
    T_outValDecAsym<TT_DATA, TT_COEFF> outVal;
    T_buff_256b<TT_DATA> readData; // input data read from window, bound for sbuff
    unsigned int dataLoaded, dataNeeded, numDataLoads;
    unsigned int initDataLoaded, initDataNeeded, initNumDataLoads;
    unsigned int xstart, xstartUpper; // upper is used for upper lanes in high decimation factor handling

    // Move data pointer away from data consumed by previous cascades
    // Move only by  multiples of 128bit. Cascade phase remainder goes to m_kDataBuffXOffset
    window_incr(inInterface.inWindow, (TRUNC((m_kFirInitOffset), (m_kWinAccessByteSize / sizeof(TT_DATA)))));

    initNumDataLoads = 0;
    // Preamble, calc number of samples for first mul.
    initDataLoaded = 0;
    initDataNeeded = m_kInitDataNeeded - TP_DECIMATE_FACTOR * m_kVOutSize;
// numDataLoads = m_kInitialLoads1Buff % m_kInitLoadsInReg;

#pragma unroll(m_kInitialLoads1Buff)
    for (int initLoads = 0; initLoads < m_kInitialLoads1Buff; ++initLoads) {
        readData = window_readincr_256b<TT_DATA>(inInterface.inWindow); // Read 256b from input window
        sbuff.val = upd_w(sbuff.val, (initNumDataLoads % m_kInitLoadsInReg),
                          readData.val); // Update sbuff with data from input window. 00++|____|____|____
        initDataLoaded += m_kInitLoadVsize;
        initNumDataLoads++;
    }
    numDataLoads = initNumDataLoads;

    // This loop creates the output window data. In each iteration a vector of samples is output
    for (unsigned i = 0; i < m_kLsize / m_kRepeatFactor; i++)
        chess_prepare_for_pipelining chess_loop_range(m_kLsize / m_kRepeatFactor, ) {
            dataNeeded = initDataNeeded;
            dataLoaded = initDataLoaded;
            numDataLoads = initNumDataLoads;

// The strobe loop is the number of iterations of Lsize required for conditions to return to the same state (typically 4
// due to the ratio of
// load size(256b) to xbuff size(1024b))
#pragma unroll(m_kRepeatFactor)
            for (int strobe = 0; strobe < m_kRepeatFactor; strobe++) {
                dataNeeded += TP_DECIMATE_FACTOR * m_kVOutSize;

                // it might take more than one load to top up the buffer of input data to satisfy the need for the next
                // vector of outputs
                if (dataNeeded > dataLoaded) {
                    readData = window_readincr_256b<TT_DATA>(inInterface.inWindow); // Read 256b from input window
                    sbuff.val = upd_w(sbuff.val, numDataLoads % m_kInitLoadsInReg,
                                      readData.val); // Update sbuff with data from input window
                    dataLoaded += m_kInitLoadVsize;
                    numDataLoads++;

                    if (dataNeeded > dataLoaded) {
                        readData = window_readincr_256b<TT_DATA>(inInterface.inWindow); // Read 256b from input window
                        sbuff.val = upd_w(sbuff.val, numDataLoads % m_kInitLoadsInReg,
                                          readData.val); // Update sbuff with data from input window
                        dataLoaded += m_kInitLoadVsize;
                        numDataLoads++;
                    }
                }

                coeff = ((T_buff_256b<TT_COEFF>*)m_internalTaps);
                coe0 = *coeff;
                coeff++;

                // Read cascade input. Do nothing if cascade input not present.
                acc = readCascade<TT_DATA, TT_COEFF>(inInterface, acc);
                xstart = m_kDataBuffXOffset + strobe * TP_DECIMATE_FACTOR * m_kLanes;
                xstartUpper = xstart + m_kLanes / 2 * TP_DECIMATE_FACTOR; // upper half of lanes

                // Init Vector operation. VMUL if cascade not present, otherwise VMAC
                acc = initMacDecAsym1Buff<TT_DATA, TT_COEFF, m_kDFX, TP_DECIMATE_FACTOR>(
                    inInterface, acc, sbuff, xstart, coe0, 0, m_kDecimateOffsets, xstartUpper);

#pragma unroll(GUARD_ZERO((m_kFirLenCeilCols / (m_kColumns) - 1)))
                for (int op = m_kColumns; op < m_kFirLenCeilCols; op += m_kColumns) {
                    if (op % m_kCoeffRegVsize == 0) {
                        coe0 = *coeff++;
                    }
                    xstart += m_kColumns;
                    xstartUpper += m_kColumns;

                    acc = macDecAsym1Buff<TT_DATA, TT_COEFF, m_kDFX, TP_DECIMATE_FACTOR>(
                        acc, sbuff, xstart, coe0, (op % m_kCoeffRegVsize), m_kDecimateOffsets, xstartUpper);
                }
                // Write cascade. Do nothing if cascade not present.
                writeCascade<TT_DATA, TT_COEFF>(outInterface, acc);

                outVal = shiftAndSaturateDecAsym(acc, TP_SHIFT);
                // Write to output window
                writeWindow<TT_DATA, TT_COEFF, TP_NUM_OUTPUTS>(outInterface, outVal);
            } // for strobe
        }     // for i
};

// This is a specialization of the main class for when there is only one kernel for the whole filter.
// STATIC COEFFICIENTS, single_output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          bool TP_CASC_IN,
          bool TP_CASC_OUT,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_USE_COEFF_RELOAD,
          unsigned int TP_NUM_OUTPUTS>
void fir_decimate_asym<TT_DATA,
                       TT_COEFF,
                       TP_FIR_LEN,
                       TP_DECIMATE_FACTOR,
                       TP_SHIFT,
                       TP_RND,
                       TP_INPUT_WINDOW_VSIZE,
                       TP_CASC_IN,
                       TP_CASC_OUT,
                       TP_FIR_RANGE_LEN,
                       TP_KERNEL_POSITION,
                       TP_CASC_LEN,
                       TP_USE_COEFF_RELOAD,
                       TP_NUM_OUTPUTS>::filter(input_window<TT_DATA>* inWindow, output_window<TT_DATA>* outWindow) {
    T_inputIF<CASC_IN_FALSE, TT_DATA> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outWindow = outWindow;
    this->filterKernel(inInterface, outInterface);
};

// This is a specialization of the main class for when there is only one kernel for the whole filter.
// STATIC COEFFICIENTS, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
void fir_decimate_asym<TT_DATA,
                       TT_COEFF,
                       TP_FIR_LEN,
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
                       2>::filter(input_window<TT_DATA>* inWindow,
                                  output_window<TT_DATA>* outWindow,
                                  output_window<TT_DATA>* outWindow2) {
    T_inputIF<CASC_IN_FALSE, TT_DATA> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outWindow = outWindow;
    outInterface.outWindow2 = outWindow2;
    this->filterKernel(inInterface, outInterface);
};

// This is a specialization of the main class for when there is only one kernel for the whole filter.
// RELOADABLE COEFFICIENTS, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
void fir_decimate_asym<TT_DATA,
                       TT_COEFF,
                       TP_FIR_LEN,
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
                       1>::filter(input_window<TT_DATA>* inWindow,
                                  output_window<TT_DATA>* outWindow,
                                  const TT_COEFF (&inTaps)[TP_FIR_LEN]) {
    T_inputIF<CASC_IN_FALSE, TT_DATA> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outWindow = outWindow;
    this->filterKernel(inInterface, outInterface, inTaps);
};

// This is a specialization of the main class for when there is only one kernel for the whole filter.
// RELOADABLE COEFFICIENTS, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
void fir_decimate_asym<TT_DATA,
                       TT_COEFF,
                       TP_FIR_LEN,
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
                       2>::filter(input_window<TT_DATA>* inWindow,
                                  output_window<TT_DATA>* outWindow,
                                  output_window<TT_DATA>* outWindow2,
                                  const TT_COEFF (&inTaps)[TP_FIR_LEN]) {
    T_inputIF<CASC_IN_FALSE, TT_DATA> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outWindow = outWindow;
    outInterface.outWindow2 = outWindow2;
    this->filterKernel(inInterface, outInterface, inTaps);
};

// This is a specialization of the main class for the final kernel in a cascade chain.
// STATIC COEFFICIENTS, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_decimate_asym<TT_DATA,
                       TT_COEFF,
                       TP_FIR_LEN,
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
                       1>::filter(input_window<TT_DATA>* inWindow,
                                  input_stream_cacc48* inCascade,
                                  output_window<TT_DATA>* outWindow) {
    T_inputIF<CASC_IN_TRUE, TT_DATA> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inCascade = inCascade;
    outInterface.outWindow = outWindow;
    this->filterKernel(inInterface, outInterface);
};

// This is a specialization of the main class for the final kernel in a cascade chain.
// STATIC COEFFICIENTS, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_decimate_asym<TT_DATA,
                       TT_COEFF,
                       TP_FIR_LEN,
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
                       2>::filter(input_window<TT_DATA>* inWindow,
                                  input_stream_cacc48* inCascade,
                                  output_window<TT_DATA>* outWindow,
                                  output_window<TT_DATA>* outWindow2) {
    T_inputIF<CASC_IN_TRUE, TT_DATA> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inCascade = inCascade;
    outInterface.outWindow = outWindow;
    outInterface.outWindow2 = outWindow2;
    this->filterKernel(inInterface, outInterface);
};

// This is a specialization of the main class for the final kernel in a cascade chain.
// RELOADABLE COEFFICIENTS, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_decimate_asym<TT_DATA,
                       TT_COEFF,
                       TP_FIR_LEN,
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
                       1>::filter(input_window<TT_DATA>* inWindow,
                                  input_stream_cacc48* inCascade,
                                  output_window<TT_DATA>* outWindow) {
    T_inputIF<CASC_IN_TRUE, TT_DATA> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inCascade = inCascade;
    outInterface.outWindow = outWindow;
    this->filterKernelRtp(inInterface, outInterface);
};

// This is a specialization of the main class for the final kernel in a cascade chain.
// RELOADABLE COEFFICIENTS, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_decimate_asym<TT_DATA,
                       TT_COEFF,
                       TP_FIR_LEN,
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
                       2>::filter(input_window<TT_DATA>* inWindow,
                                  input_stream_cacc48* inCascade,
                                  output_window<TT_DATA>* outWindow,
                                  output_window<TT_DATA>* outWindow2) {
    T_inputIF<CASC_IN_TRUE, TT_DATA> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inCascade = inCascade;
    outInterface.outWindow = outWindow;
    outInterface.outWindow2 = outWindow2;
    this->filterKernelRtp(inInterface, outInterface);
};

// This is a specialization of the main class for the first kernel in a cascade chain.
// STATIC COEFFICIENTS
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_decimate_asym<TT_DATA,
                       TT_COEFF,
                       TP_FIR_LEN,
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
                       1>::filter(input_window<TT_DATA>* inWindow,
                                  output_stream_cacc48* outCascade,
                                  output_window<TT_DATA>* broadcastWindow) {
    T_inputIF<CASC_IN_FALSE, TT_DATA> inInterface;
    T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outCascade = outCascade;
    outInterface.broadcastWindow = broadcastWindow;
    this->filterKernel(inInterface, outInterface);
};

// This is a specialization of the main class for the first kernel in a cascade chain.
// RELOADABLE COEFFICIENTS
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_decimate_asym<TT_DATA,
                       TT_COEFF,
                       TP_FIR_LEN,
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
                       1>::filter(input_window<TT_DATA>* inWindow,
                                  output_stream_cacc48* outCascade,
                                  output_window<TT_DATA>* broadcastWindow,
                                  const TT_COEFF (&inTaps)[TP_FIR_LEN]) {
    T_inputIF<CASC_IN_FALSE, TT_DATA> inInterface;
    T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outCascade = outCascade;
    outInterface.broadcastWindow = broadcastWindow;
    this->filterKernel(inInterface, outInterface, inTaps);
};

// This is a specialization of the main class for any kernel within a cascade chain, but neither first nor last.
// STATIC COEFFICIENTS
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_decimate_asym<TT_DATA,
                       TT_COEFF,
                       TP_FIR_LEN,
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
                       1>::filter(input_window<TT_DATA>* inWindow,
                                  input_stream_cacc48* inCascade,
                                  output_stream_cacc48* outCascade,
                                  output_window<TT_DATA>* broadcastWindow) {
    T_inputIF<CASC_IN_TRUE, TT_DATA> inInterface;
    T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inCascade = inCascade;
    outInterface.outCascade = outCascade;
    outInterface.broadcastWindow = broadcastWindow;
    this->filterKernel(inInterface, outInterface);
};

// This is a specialization of the main class for any kernel within a cascade chain, but neither first nor last.
// RELOADABLE COEFFICIENTS
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_decimate_asym<TT_DATA,
                       TT_COEFF,
                       TP_FIR_LEN,
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
                       1>::filter(input_window<TT_DATA>* inWindow,
                                  input_stream_cacc48* inCascade,
                                  output_stream_cacc48* outCascade,
                                  output_window<TT_DATA>* broadcastWindow) {
    T_inputIF<CASC_IN_TRUE, TT_DATA> inInterface;
    T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inCascade = inCascade;
    outInterface.outCascade = outCascade;
    outInterface.broadcastWindow = broadcastWindow;
    this->filterKernelRtp(inInterface, outInterface);
};
}
}
}
}
} // namespaces
