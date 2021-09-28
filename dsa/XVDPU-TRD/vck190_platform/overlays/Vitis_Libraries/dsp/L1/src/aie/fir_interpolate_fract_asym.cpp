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
FIR kernel code.
This file captures the body of run-time code for the kernal class.

Coding conventions
  TT_      template type suffix
  TP_      template parameter suffix
*/

#pragma once
#include <adf.h>

#define __NEW_WINDOW_H__ 1

#define __AIEARCH__ 1
#define __AIENGINE__ 1
#define __AIE_API_USE_NATIVE_1024B_VECTOR__
#include "aie_api/aie_adf.hpp"

#include "kernel_api_utils.hpp"

#include "fir_interpolate_fract_asym.hpp"
#include "fir_interpolate_fract_asym_utils.hpp"

#include <cmath> // For power function

// According to template parameter the input may be a window, or window and cascade input
// Similarly the output interface may be a window or a cascade output
//-----------------------------------------------------------------------------------------------------

namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace interpolate_fract_asym {

template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
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
                       TP_INTERPOLATE_FACTOR,
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
    windowBroadcast<TT_DATA,
                    TP_INPUT_WINDOW_VSIZE +
                        fnFirMargin<((TP_FIR_LEN + TP_INTERPOLATE_FACTOR - 1) / TP_INTERPOLATE_FACTOR), TT_DATA>()>(
        inInterface, outInterface);
    windowAcquire(inInterface);
    filterIntFractAsym(inInterface, outInterface);
    windowRelease(inInterface);
}

// Asymmetric Fractional Interpolation FIR Kernel Function - overloaded (not specialised)
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
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
                              TP_INTERPOLATE_FACTOR,
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
    windowBroadcast<TT_DATA,
                    TP_INPUT_WINDOW_VSIZE +
                        fnFirMargin<((TP_FIR_LEN + TP_INTERPOLATE_FACTOR - 1) / TP_INTERPOLATE_FACTOR), TT_DATA>()>(
        inInterface, outInterface);
    m_coeffnEq = rtpCompare(inTaps, m_oldInTaps);

    sendRtpTrigger(m_coeffnEq, outInterface);
    if (m_coeffnEq) { // Coefficients have changed
        bufferReload(inTaps, m_oldInTaps, outInterface);
        firReload(inTaps);
        chess_memory_fence();
    }

    windowAcquire(inInterface);
    filterIntFractAsym(inInterface, outInterface);
    windowRelease(inInterface);
}

// Asymmetric Fractional Interpolation FIR Kernel Function - overloaded (not specialised)
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
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
                              TP_INTERPOLATE_FACTOR,
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
    windowBroadcast<TT_DATA,
                    TP_INPUT_WINDOW_VSIZE +
                        fnFirMargin<((TP_FIR_LEN + TP_INTERPOLATE_FACTOR - 1) / TP_INTERPOLATE_FACTOR), TT_DATA>()>(
        inInterface, outInterface);
    m_coeffnEq = getRtpTrigger(); // 0 - equal, 1 - not equal

    sendRtpTrigger(m_coeffnEq, outInterface);
    if (m_coeffnEq) { // Coefficients have changed
        bufferReload<TT_DATA, TT_COEFF, TP_FIR_LEN>(inInterface, m_oldInTaps, outInterface);
        firReload(m_oldInTaps);
        chess_memory_fence();
    }

    windowAcquire(inInterface);
    filterIntFractAsym(inInterface, outInterface);
    windowRelease(inInterface);
}

template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
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
                       TP_INTERPOLATE_FACTOR,
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
                       TP_NUM_OUTPUTS>::filterIntFractAsym(T_inputIF<TP_CASC_IN, TT_DATA> inInterface,
                                                           T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    set_rnd(TP_RND);
    set_sat();

    static constexpr polyphaseArray<unsigned int> m_kDataNeededPhase =
        getDataNeeded<m_kPolyphaseLaneAlias>(params, TP_DECIMATE_FACTOR, TP_INTERPOLATE_FACTOR, m_kLanes, m_kColumns);
    static constexpr polyphaseArray<unsigned int> m_kInitialLoads =
        getInitialLoads<m_kPolyphaseLaneAlias>(params, TP_DECIMATE_FACTOR, TP_INTERPOLATE_FACTOR, m_kLanes, m_kColumns);
    static constexpr polyphaseArray<int> xstartPhase =
        getXStarts<m_kPolyphaseLaneAlias>(params, TP_DECIMATE_FACTOR, TP_INTERPOLATE_FACTOR, m_kLanes, m_kColumns);
    static constexpr polyphaseArray<int> windowDecPhase = getWindowDecrements<m_kPolyphaseLaneAlias>(
        params, TP_DECIMATE_FACTOR, TP_INTERPOLATE_FACTOR, m_kLanes, m_kColumns, m_kNumOps);
    // static constexpr polyphaseArray<int> windowDecPhase                 = {12};

    // static constexpr std::array<int,m_kPolyphaseLaneAlias>  windowDecPhase ={12,8,8,12,8};
    static constexpr polyphaseArray<unsigned int> xoffsets =
        getXOffsets<m_kPolyphaseLaneAlias>(TP_DECIMATE_FACTOR, TP_INTERPOLATE_FACTOR, m_kLanes);
    static constexpr polyphaseArray<unsigned int> zoffsets =
        getZOffsets<m_kPolyphaseLaneAlias>(TP_INTERPOLATE_FACTOR, m_kLanes);

#ifndef _DSPLIB_FIR_INTERPOLATE_FRACT_ASYM_HPP_DEBUG_
    static_assert(windowDecPhase[0] % (params.alignWindowReadBytes / params.dataSizeBytes) == 0,
                  "ERROR: Solution doesn't meet alignment requirements. Window decrements must be aligned to 128b "
                  "boundary. Increase m_kPolyphaseLaneAlias usually solves this. ");
    static_assert(windowDecPhase[m_kPolyphaseLaneAlias - 1] % (params.alignWindowReadBytes / params.dataSizeBytes) == 0,
                  "ERROR: Solution doesn't meet alignment requirements. Window decrements must be aligned to 128b "
                  "boundary. Increase m_kPolyphaseLaneAlias usually solves this. ");
#endif

    T_buff_256b<TT_COEFF>* restrict coeff = (T_buff_256b<TT_COEFF>*)m_internalTaps;
    T_buff_256b<TT_COEFF> coe0;                               // register for coeff values.
    T_buff_1024b<TT_DATA> sbuff = null_buff_1024b<TT_DATA>(); // input data value cache.
    T_acc<TT_DATA, TT_COEFF> acc;
    T_outVal<TT_DATA, TT_COEFF> outVal;
    T_buff_256b<TT_DATA> readData; // input data read from window, bound for sbuff
    unsigned int dataNeeded;
    // unsigned int                dataLoaded,numDataLoads;
    unsigned int xstart = 0;
    // Display constants for debug

    // Move data pointer away from data consumed by previous cascades
    // Move only by  multiples of 128bit. Cascade phase remainder goes to m_kDataBuffXOffset
    window_incr(inInterface.inWindow, (TRUNC((m_kFirInitOffset), (m_kWinAccessByteSize / sizeof(TT_DATA)))));
    // Incremental loads cause a very un-wanted new loop (strobeFactor) because upd_w idx has to be compile time
    // constant
    // this essentially puts a requirement of having at least strobeFactor*PhaseLaneAlias*FirLen window length i think
    // numDataLoads =0;
    // dataLoaded = 0;
    // ideally only add this once
    // dataNeeded = m_kDataBuffXOffset;
    // This loop creates the output window data. In each iteration a vector of samples is output
    for (unsigned i = 0; i < m_kLsize; i++) chess_prepare_for_pipelining chess_loop_range(m_kLsize, ) {
// Allows us to keep upd_w as compile tile constant
// HAZARD : The splice in buffer is unlikely to line up very well,
// but right now, we don't use more than dataLoadSize for a given op
// dataLoaded = 0;
// numDataLoads =0;
// How many operations until the 0th polyphase is the first lane again.
#pragma unroll(m_kPolyphaseLaneAlias)
            for (unsigned offsetPhase = 0; offsetPhase < m_kPolyphaseLaneAlias; ++offsetPhase) {
                coeff = ((T_buff_256b<TT_COEFF>*)&m_internalTaps[offsetPhase % m_kPolyphaseCoeffAlias][0][0]);

                coe0 = *coeff;
                // This doesn't give enough granularity
                // coeff++;

                // Preamble, calculate and load data from window into register
                // A given offsetPhase may require more samples than another.
                // It shouldn't make a difference for ops/load, but accountancy might skew if we take the larger
                xstart = xstartPhase[offsetPhase]; //+m_kDataBuffXOffset (i've inadvertnly already added this)
                dataNeeded = m_kDataNeededPhase[offsetPhase] + xstart; // Adding the xstart to show that some samples
                                                                       // that are loaded are not used at all, but we
                                                                       // need them anyway.
                // dataNeeded = m_kDataBuffXOffset + m_kVOutSize + m_kColumns-1;

                // pragma unroll needs completely constant, but offsetPhase doesn't look constant (even though it will
                // be due to pahse unroll..)
                //#pragma unroll (m_kInitialLoads[offsetPhase])
                for (int initLoads = 0; initLoads < m_kInitialLoads[offsetPhase]; ++initLoads) {
                    readData = window_readincr_256b<TT_DATA>(inInterface.inWindow); // Read 256b from input window
                    sbuff.val = upd_w(sbuff.val, initLoads % m_kDataLoadsInReg,
                                      readData.val); // Update sbuff with data from input window. 00++|____|____|____
                }
                // Ensures that these can be treated as compile time constant in the next unrolled loop.
                unsigned int dataLoaded = m_kDataLoadVsize * m_kInitialLoads[offsetPhase];
                unsigned int numDataLoads = m_kInitialLoads[offsetPhase];
                // Read cascade input. Do nothing if cascade input not present.
                acc = readCascade<TT_DATA, TT_COEFF>(inInterface, acc);
                // Init Vector operation. VMUL if cascade not present, otherwise VMAC
                acc = initMacIntFract<TT_DATA, TT_COEFF>(inInterface, acc, sbuff, xstart, xoffsets[offsetPhase], coe0,
                                                         0, zoffsets[offsetPhase]);

#pragma unroll(GUARD_ZERO((m_kNumOps - 1)))
                for (int op = m_kColumns; op < m_kFirLenCeilCols; op += m_kColumns) {
                    dataNeeded += m_kColumns;

                    if (dataNeeded > dataLoaded) {
                        readData = window_readincr_256b<TT_DATA>(inInterface.inWindow); // Read 256b from input window
                        sbuff.val = upd_w(sbuff.val, (numDataLoads % m_kDataLoadsInReg),
                                          readData.val); // Update sbuff with data from input window
                        dataLoaded += m_kDataLoadVsize;
                        numDataLoads++;
                    }
                    if (1) {
                        coeff = ((T_buff_256b<TT_COEFF>*)&m_internalTaps[offsetPhase % m_kPolyphaseCoeffAlias]
                                                                        [op / m_kColumns][0][0]);
                        coe0 = *coeff;
                        // coe0 = *coeff++;
                    }
                    acc = macIntFract(acc, sbuff, (op + xstart), xoffsets[offsetPhase], coe0,
                                      ((op * m_kLanes % m_kLanes) % m_kCoeffRegVsize), zoffsets[offsetPhase]);
                }
                // Use dataNeededPhase to show how many data samples were consumed
                window_decr(inInterface.inWindow,
                            windowDecPhase[offsetPhase]); // return read pointer to start of next chunk of window.

                // Write cascade. Do nothing if cascade not present.
                writeCascade<TT_DATA, TT_COEFF>(outInterface, acc);

                outVal = shiftAndSaturate(acc, TP_SHIFT);

                // Write to output window
                writeWindow<TT_DATA, TT_COEFF, TP_NUM_OUTPUTS>(outInterface, outVal);
            } // m_kPolyphaseLaneAlias

        } // LSize
};

// This is a specialization of the main class for when there is only one kernel for the whole filter, static
// coefficients, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
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
void fir_interpolate_fract_asym<TT_DATA,
                                TT_COEFF,
                                TP_FIR_LEN,
                                TP_INTERPOLATE_FACTOR,
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
                                TP_NUM_OUTPUTS>::filter(input_window<TT_DATA>* inWindow,
                                                        output_window<TT_DATA>* outWindow) {
    T_inputIF<CASC_IN_FALSE, TT_DATA> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outWindow = outWindow;
    this->filterKernel(inInterface, outInterface);
};

// This is a specialization of the main class for when there is only one kernel for the whole filter, static
// coefficients, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
void fir_interpolate_fract_asym<TT_DATA,
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

// This is a specialization of the main class for when there is only one kernel for the whole filter, reloadable
// coefficients, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
void fir_interpolate_fract_asym<TT_DATA,
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
                                1>::filter(input_window<TT_DATA>* inWindow,
                                           output_window<TT_DATA>* outWindow,
                                           const TT_COEFF (&inTaps)[TP_FIR_LEN]) {
    T_inputIF<CASC_IN_FALSE, TT_DATA> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outWindow = outWindow;
    this->filterKernel(inInterface, outInterface, inTaps);
};

// This is a specialization of the main class for when there is only one kernel for the whole filter, reloadable
// coefficients, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_INTERPOLATE_FACTOR,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
void fir_interpolate_fract_asym<TT_DATA,
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

// This is a specialization of the main class for the final kernel in a cascade chain. static coefficients, single
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
void fir_interpolate_fract_asym<TT_DATA,
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

// This is a specialization of the main class for the final kernel in a cascade chain. static coefficients, dual output
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
void fir_interpolate_fract_asym<TT_DATA,
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

// This is a specialization of the main class for the final kernel in a cascade chain, reloadable coefficients, single
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
void fir_interpolate_fract_asym<TT_DATA,
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

// This is a specialization of the main class for the final kernel in a cascade chain, reloadable coefficients, dual
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
void fir_interpolate_fract_asym<TT_DATA,
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

// This is a specialization of the main class for the first kernel in a cascade chain, static coefficients
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
void fir_interpolate_fract_asym<TT_DATA,
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

// This is a specialization of the main class for the first kernel in a cascade chain, reloadable coefficients
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
void fir_interpolate_fract_asym<TT_DATA,
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

// This is a specialization of the main class for any kernel within a cascade chain, but neither first nor last, static
// coefficients
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
void fir_interpolate_fract_asym<TT_DATA,
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

// This is a specialization of the main class for any kernel within a cascade chain, but neither first nor last,
// reloadable coefficients
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
void fir_interpolate_fract_asym<TT_DATA,
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
}
