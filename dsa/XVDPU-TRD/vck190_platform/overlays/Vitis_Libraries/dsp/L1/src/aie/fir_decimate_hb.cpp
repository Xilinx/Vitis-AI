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
Halfband decimation FIR kernel code.
This file captures the body of run-time code for the kernel class and a higher wrapping 'cascade' layer which has
specializations for
 combinations of inputs and outputs. That is, in a chain of kernels, the first will have an input window, and a cascade
out stream.
 The next, potentially multiple, kernel(s) will each have an input window and cascade stream and will output a cascade
steam. The final kernel
 will have an input window and cascade stream and an output window only.
 The cascade layer class is called fir_interpolate_hb with the kernel-layer (operational) class called
kernelFilterClass.
 The fir_interpolate_hb class has a member of the kernelFilterClass..

Coding conventions
  TT_      template type suffix
  TP_      template parameter suffix
*/

#pragma once
#include <adf.h>

#define __NEW_WINDOW_H__ 1
#include "aie_api/aie_adf.hpp"

#include "kernel_api_utils.hpp"
#include "fir_decimate_hb.hpp"
#include "fir_decimate_hb_utils.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace decimate_hb {

// FIR function
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          size_t TP_FIR_LEN,
          size_t TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          bool TP_CASC_IN,
          bool TP_CASC_OUT,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP,
          unsigned int TP_USE_COEFF_RELOAD,
          unsigned int TP_NUM_OUTPUTS>
inline void kernelFilterClass<TT_DATA,
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
                              TP_DUAL_IP,
                              TP_USE_COEFF_RELOAD,
                              TP_NUM_OUTPUTS>::filterKernel(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                                                            T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    windowBroadcast<TT_DATA, TP_INPUT_WINDOW_VSIZE + fnFirMargin<TP_FIR_LEN, TT_DATA>()>(inInterface, outInterface);
    filterSelectArch(inInterface, outInterface);
};

// FIR function - overloaded (not specialised) with taps for reload
template <typename TT_DATA,
          typename TT_COEFF,
          size_t TP_FIR_LEN,
          size_t TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          bool TP_CASC_IN,
          bool TP_CASC_OUT,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP,
          unsigned int TP_USE_COEFF_RELOAD,
          unsigned int TP_NUM_OUTPUTS>
inline void kernelFilterClass<TT_DATA,
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
                              TP_DUAL_IP,
                              TP_USE_COEFF_RELOAD,
                              TP_NUM_OUTPUTS>::filterKernel(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                                                            T_outputIF<TP_CASC_OUT, TT_DATA> outInterface,
                                                            const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / 4 + 1]) {
    windowBroadcast<TT_DATA, TP_INPUT_WINDOW_VSIZE + fnFirMargin<TP_FIR_LEN, TT_DATA>()>(inInterface, outInterface);
    m_coeffnEq = rtpCompare(inTaps, m_oldInTaps);

    sendRtpTrigger(m_coeffnEq, outInterface);
    if (m_coeffnEq) { // Coefficients have changed
        bufferReload(inTaps, m_oldInTaps, outInterface);
        firReload(inTaps);
    }
    filterSelectArch(inInterface, outInterface);
};

// FIR function
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          size_t TP_FIR_LEN,
          size_t TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          bool TP_CASC_IN,
          bool TP_CASC_OUT,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP,
          unsigned int TP_USE_COEFF_RELOAD,
          unsigned int TP_NUM_OUTPUTS>
inline void kernelFilterClass<TT_DATA,
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
                              TP_DUAL_IP,
                              TP_USE_COEFF_RELOAD,
                              TP_NUM_OUTPUTS>::filterKernelRtp(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                                                               T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    windowBroadcast<TT_DATA, TP_INPUT_WINDOW_VSIZE + fnFirMargin<TP_FIR_LEN, TT_DATA>()>(inInterface, outInterface);
    m_coeffnEq = getRtpTrigger(); // 0 - equal, 1 - not equal

    sendRtpTrigger(m_coeffnEq, outInterface);
    if (m_coeffnEq) { // Coefficients have changed
        bufferReload<TT_DATA, TT_COEFF, (TP_FIR_LEN + 1) / 4 + 1, TP_DUAL_IP>(inInterface, m_oldInTaps, outInterface);
        firReload(m_oldInTaps);
    }
    filterSelectArch(inInterface, outInterface);
};

// FIR function - overloaded (not specialised) with taps for reload
template <typename TT_DATA,
          typename TT_COEFF,
          size_t TP_FIR_LEN,
          size_t TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          bool TP_CASC_IN,
          bool TP_CASC_OUT,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP,
          unsigned int TP_USE_COEFF_RELOAD,
          unsigned int TP_NUM_OUTPUTS>
inline void kernelFilterClass<TT_DATA,
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
                              TP_DUAL_IP,
                              TP_USE_COEFF_RELOAD,
                              TP_NUM_OUTPUTS>::filterSelectArch(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                                                                T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    set_rnd(TP_RND);
    set_sat();

    windowAcquire(inInterface);
    // 3 possible architectures depending on size of data/coef types, fir_len and input window size
    // Using a single data buffer for x and y (forward & reverse) or seperate
    if
        constexpr(m_kArch == kArch1Buff) { filter1buff(inInterface, outInterface); }
    else if
        constexpr((m_kArch == kArch2Buff) && (m_kArchZigZag == kArchZigZag)) {
            filter2buffzigzag(inInterface, outInterface);
        }
    else {
        filter2buff(inInterface, outInterface);
    }
    windowRelease(inInterface);
};

//-----------------------------------------------------------------------------------------------------
// The filterBig1 variant of this function is for cases where 2 separate buffers must be used, one for forward data
// and the other for reverse data. This is for int32 and smaller
template <typename TT_DATA,
          typename TT_COEFF,
          size_t TP_FIR_LEN,
          size_t TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          bool TP_CASC_IN,
          bool TP_CASC_OUT,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP,
          unsigned int TP_USE_COEFF_RELOAD,
          unsigned int TP_NUM_OUTPUTS>
inline void kernelFilterClass<TT_DATA,
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
                              TP_DUAL_IP,
                              TP_USE_COEFF_RELOAD,
                              TP_NUM_OUTPUTS>::filter2buff(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                                                           T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    // Pointers to coefficient storage and explicit registers to hold values
    T_buff_256b<TT_COEFF>* restrict coeff = (T_buff_256b<TT_COEFF>*)m_phaseOneTaps;
    T_buff_256b<TT_COEFF> coe0; // register for coeff values.
    T_buff_256b<TT_COEFF>* restrict coeffCt = (T_buff_256b<TT_COEFF>*)m_phaseOneTapsCt;
    T_buff_256b<TT_COEFF> coeCt = *coeffCt; // register for coeff values.
    T_buff_128b<TT_DATA> xReadData;
    T_buff_128b<TT_DATA> yReadData;
    T_buff_FirDecHb<TT_DATA, TT_COEFF, m_kArch> xbuff;
    T_buff_FirDecHb<TT_DATA, TT_COEFF, m_kArch> ybuff;
    T_accFirDecHb<TT_DATA, TT_COEFF, m_kArch> acc[K_MAX_PASSES];
    T_acc768<TT_DATA, TT_COEFF> accConcat;
    T_outValFiRDecHb<TT_DATA, TT_COEFF, m_kArch> outVal;
    unsigned int xDataNeeded, xDataLoaded, xNumDataLoads;
    unsigned int yDataNeeded, yDataLoaded, yNumDataLoads, ySplice;
    unsigned int xstart, ystart, coeffstart;
    unsigned int lastXstart, lastYstart;

    input_window<TT_DATA>* restrict inWindow = inInterface.inWindow;
    input_window<TT_DATA> yinWindowActual;
    input_window<TT_DATA>* restrict yinWindow;
    yinWindow = &yinWindowActual;
    if
        constexpr(TP_DUAL_IP == 0) { window_copy(yinWindow, inWindow); }
    else {
        window_copy(yinWindow, inInterface.inWindowReverse);
    }

    window_incr(inWindow, m_kDataWindowOffset); // move input data pointer past the margin padding
    window_incr(yinWindow, m_kySpliceStart);    // Fast forward to 128b boundary containing first Y data

    for (int i = 0; i < m_kLsize; i++) chess_prepare_for_pipelining chess_loop_range(m_kLsize, ) {
            for (int pass = 0; pass < m_kPasses; ++pass) {
                coeff = ((T_buff_256b<TT_COEFF>*)&m_phaseOneTaps[0][0]);
                coe0 = *coeff;

                xNumDataLoads = 0;
                xDataLoaded = 0;
                xstart = m_kDataBuffXOffset;
                ystart = m_kyStart; // first column of first load
                coeffstart = 0;

                // Pre-loading the ybuff differs from the xbuff load because ystart is not in general aligned to loads.
                yNumDataLoads = 0;
                yDataLoaded = 0;
                ySplice = 0;

// preload xdata from window into xbuff register
#pragma unroll(m_kInitLoadsXneeded)
                for (unsigned int initLoads = 0; initLoads < m_kInitLoadsXneeded; ++initLoads) {
                    if ((m_kInitLoadsXneeded - initLoads) >= 2 && m_kLoadSize == 128) {
                        fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, (m_kLoadSize * 2)>(
                            xbuff, (xNumDataLoads / 2) % m_kDataLoadsInReg, inWindow);
                        xNumDataLoads++;
                        initLoads++;
                        xDataLoaded += m_kDataLoadVsize;
                    } else {
                        fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, m_kLoadSize>(xbuff, xNumDataLoads % m_kDataLoadsInReg,
                                                                               inWindow);
                    }
                    // fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, m_kLoadSize>(xbuff, xNumDataLoads % m_kDataLoadsInReg,
                    // inWindow);
                    xNumDataLoads++;
                    xDataLoaded += m_kDataLoadVsize;
                }
// preload ydata from window into ybuff register
#pragma unroll(m_kInitLoadsYneeded)
                for (unsigned int initLoads = 0; initLoads < m_kInitLoadsYneeded; ++initLoads) {
                    if ((m_kInitLoadsYneeded - initLoads) >= 2 && m_kLoadSize == 128) {
                        fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, (m_kLoadSize * 2)>(
                            ybuff, (ySplice / 2) % m_kDataLoadsInReg, yinWindow);
                        initLoads++;
                        ySplice++;

                    } else {
                        // Note that initial Y loads are forwards, hence use the forward direction load function.
                        fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, m_kLoadSize>(ybuff, ySplice % m_kDataLoadsInReg,
                                                                               yinWindow);
                    }
                    ySplice++;
                }
                xDataNeeded = m_kDataBuffXOffset + (m_kLanes + m_kColumns - 2) * kDecimateFactor +
                              1;                                      // e.g. D0 to D8 is 9, not 10.
                yDataNeeded = (m_kColumns - 1) * kDecimateFactor - 1; // datum is lane 0, but y needs go backwards.
                yDataLoaded += ystart % m_kDataLoadVsize;             // i.e. how many do we have in hand
                window_decr(yinWindow, (m_kInitLoadsYneeded + 1) * m_kDataLoadVsize);
                ySplice = m_kNumOps * m_kDataLoadsInReg - 1; // allow ysplice to count down to 0

                // Read cascade input. Do nothing if cascade input not present.
                acc[pass] = readCascade<TT_DATA, TT_COEFF>(inInterface, acc[pass]);

                // Initial multiply
                if (m_kNumOps == 1 && m_kCtPresent == 1) {
                    lastXstart = (xstart - m_kFinalOpSkew + m_kSamplesInDataBuff) % m_kSamplesInDataBuff;
                    lastYstart = (ystart + m_kFinalOpSkew + m_kSamplesInDataBuff) % m_kSamplesInDataBuff;
                    acc[pass] = initMacDecHbCt(inInterface, acc[pass], xbuff, lastXstart, ybuff, lastYstart,
                                               (m_kColumns - 1) * kDecimateFactor - 1, // m_kCtOffset
                                               coeCt, 0);
                } else {
                    acc[pass] = initMacDecHb(inInterface, acc[pass], xbuff, xstart, ybuff, ystart, coe0, coeffstart);
                }

// In the operations loop, x and y buffs load at different times because y can start mid-splice.
#pragma unroll(GUARD_ZERO((m_kNumOps - 1)))
                for (int op = 1; op < m_kNumOps; ++op) {
                    if (op < m_kNumOps - 1) {
                        xDataNeeded += m_kColumns * kDecimateFactor;
                        yDataNeeded += m_kColumns * kDecimateFactor;
                    } else {
                        xDataNeeded += (m_kColumns - 1) * kDecimateFactor + 1;
                        yDataNeeded += (m_kColumns - 1) * kDecimateFactor + 1;
                    }
                    if (xDataNeeded > xDataLoaded) {
                        // Load xdata from window into xbuff register
                        fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, m_kLoadSize>(xbuff, xNumDataLoads % m_kDataLoadsInReg,
                                                                               inWindow);
                        xNumDataLoads++;
                        xDataLoaded += m_kDataLoadVsize;
                    }
                    if (yDataNeeded > yDataLoaded) {
                        // Load ydata from window into ybuff register
                        fnLoadYIpData<TT_DATA, TT_COEFF, m_kArch, m_kLoadSize>(ybuff, ySplice % m_kDataLoadsInReg,
                                                                               yinWindow);
                        ySplice--;
                        yNumDataLoads++;
                        yDataLoaded += m_kDataLoadVsize; // 0 maps to mkDataLoadsVsize
                    }
                    xstart += m_kColumns * kDecimateFactor;
                    ystart -= m_kColumns * kDecimateFactor;
                    coeffstart += m_kColumns;
                    if (op % (m_kCoeffRegVsize / m_kColumns) == 0) {
                        // Load coefficients coe0 register
                        coeff = ((T_buff_256b<TT_COEFF>*)&m_phaseOneTaps[op][0]);
                        coe0 = *coeff;
                    }
                    if ((op >= m_kNumOps - 1) && (m_kCtPresent == 1)) {
                        // Last operation includes the centre tap
                        lastXstart = (xstart - m_kFinalOpSkew + m_kSamplesInDataBuff) % m_kSamplesInDataBuff;
                        lastYstart = (ystart + m_kFinalOpSkew + m_kSamplesInDataBuff) % m_kSamplesInDataBuff;
                        // Final multiply operation
                        acc[pass] = firDecHbMacSymCt(acc[pass], xbuff, lastXstart, ybuff, lastYstart,
                                                     (m_kColumns - 1) * kDecimateFactor - 1, // m_kCtOffset
                                                     coeCt, 0);
                    } else {
                        // Multiply operation
                        acc[pass] = firDecHbMacSym(acc[pass], xbuff, xstart, ybuff, ystart, coe0, coeffstart);
                    }
                } // operations loop
                window_decr(inWindow,
                            xNumDataLoads * m_kDataLoadVsize -
                                kDecimateFactor * m_kLanes); // return read pointer to start of next chunk of window.
                window_incr(yinWindow,
                            (yNumDataLoads + 1) * m_kDataLoadVsize +
                                kDecimateFactor * m_kLanes); // return read pointer to start of next chunk of window.
            }                                                // passes loop
            // Write cascade. Do nothing if cascade not present.
            writeCascade<TT_DATA, TT_COEFF, m_kArch>(outInterface, acc[0]);

            // The data for the centre tap is the same data as required for the last op of the top phase, so is already
            // loaded
            outVal = firDecHbWriteOut<TT_DATA, TT_COEFF, m_kArch>(acc, TP_SHIFT);
            // writeWindow<TT_DATA,TT_COEFF>(outInterface, outVal);
            writeWindow<TT_DATA, TT_COEFF, m_kArch, TP_NUM_OUTPUTS>(
                outInterface, outVal); // NOT from kernel_utils, but specific to dec

        } // i loop (splice of window)
};

//-----------------------------------------------------------------------------------------------------
// This function processes data in the forward direction as normal, but then
// recognizes the fact that only one further incremental load is needed to start
// from the centre tap of the next set of outputs.
template <typename TT_DATA,
          typename TT_COEFF,
          size_t TP_FIR_LEN,
          size_t TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          bool TP_CASC_IN,
          bool TP_CASC_OUT,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP,
          unsigned int TP_USE_COEFF_RELOAD,
          unsigned int TP_NUM_OUTPUTS>
inline void kernelFilterClass<TT_DATA,
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
                              TP_DUAL_IP,
                              TP_USE_COEFF_RELOAD,
                              TP_NUM_OUTPUTS>::filter2buffzigzag(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                                                                 T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    // Pointers to coefficient storage and explicit registers to hold values
    T_buff_256b<TT_COEFF>* restrict coeff = (T_buff_256b<TT_COEFF>*)m_phaseOneTaps;
    T_buff_256b<TT_COEFF> coe0; // register for coeff values.
    T_buff_256b<TT_COEFF>* restrict coeffCt = (T_buff_256b<TT_COEFF>*)m_phaseOneTapsCt;
    T_buff_256b<TT_COEFF> coeCt = *coeffCt; // register for coeff values.
    T_buff_128b<TT_DATA> yReadData;
    T_buff_FirDecHb<TT_DATA, TT_COEFF, m_kArch> xbuff;
    T_buff_FirDecHb<TT_DATA, TT_COEFF, m_kArch> ybuff;
    T_accFirDecHb<TT_DATA, TT_COEFF, m_kArch> acc[K_MAX_PASSES];
    T_acc768<TT_DATA, TT_COEFF> accConcat;
    T_outValFiRDecHb<TT_DATA, TT_COEFF, m_kArch> outVal;
    unsigned int xDataNeeded, xDataLoaded, xNumDataLoads;
    unsigned int yDataNeeded, yDataLoaded, yNumDataLoads, ySplice;
    unsigned int xstart, ystart, coeffstart;
    unsigned int lastXstart, lastYstart;

    input_window<TT_DATA>* restrict inWindow = inInterface.inWindow;
    input_window<TT_DATA> yinWindowActual;
    input_window<TT_DATA>* restrict yinWindow;
    yinWindow = &yinWindowActual;
    if
        constexpr(TP_DUAL_IP == 0) { window_copy(yinWindow, inWindow); }
    else {
        window_copy(yinWindow, inInterface.inWindowReverse);
    }

    // use separate window pointers for zag (reverse computation iterations)
    input_window<TT_DATA> inRevWindowActual;
    input_window<TT_DATA>* restrict inRevWindow;
    inRevWindow = &inRevWindowActual;
    window_copy(inRevWindow, inWindow);
    // Starting chunk + forward loads.
    window_incr(inRevWindow, m_kDataWindowOffset + (m_kNumOps - 1) * m_kDataLoadVsize - m_kDataLoadVsize * 0);

    input_window<TT_DATA> yRevinWindowActual;
    input_window<TT_DATA>* restrict yRevinWindow;
    yRevinWindow = &yRevinWindowActual;
    if
        constexpr(TP_DUAL_IP == 0) { window_copy(yRevinWindow, inInterface.inWindow); }
    else {
        window_copy(yRevinWindow, inInterface.inWindowReverse);
    }
    // Starting chunk - reverse loads + realignment to xwindowPointer
    window_incr(yRevinWindow, m_kySpliceStart + (m_kInitLoadsYneeded - (m_kNumOps - 1)) * m_kDataLoadVsize);

    window_incr(inWindow, m_kDataWindowOffset); // move input data pointer past the margin padding
    window_incr(yinWindow, m_kySpliceStart);    // Fast forward to 128b boundary containing first Y data

    xNumDataLoads = 0;
    ySplice = 0;
// preload ydata from window into ybuff register
#pragma unroll(m_kInitLoadsYneeded)
    for (unsigned int initLoads = 0; initLoads < m_kInitLoadsYneeded; ++initLoads) {
        if ((m_kInitLoadsYneeded - initLoads) >= 2 && m_kLoadSize == 128) {
            fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, (m_kLoadSize * 2)>(ybuff, (ySplice / 2) % m_kDataLoadsInReg,
                                                                         yinWindow);
            initLoads++;
            ySplice++;

        } else {
            // Note that initial Y loads are forwards, hence use the forward direction load function.
            fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, m_kLoadSize>(ybuff, ySplice % m_kDataLoadsInReg, yinWindow);
        }
        ySplice++;
    }
    // Reset y window pointer
    window_decr(yinWindow, (m_kInitLoadsYneeded + 1) * m_kDataLoadVsize);

// preload xdata from window into xbuff register
#pragma unroll(m_kInitLoadsXneeded)
    for (unsigned int initLoads = 0; initLoads < m_kInitLoadsXneeded; ++initLoads) {
        if ((m_kInitLoadsXneeded - initLoads) >= 2 && m_kLoadSize == 128) {
            fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, (m_kLoadSize * 2)>(xbuff, (xNumDataLoads / 2) % m_kDataLoadsInReg,
                                                                         inWindow);
            xNumDataLoads++;
            initLoads++;
            // xDataLoaded += m_kDataLoadVsize;
        } else {
            fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, m_kLoadSize>(xbuff, xNumDataLoads % m_kDataLoadsInReg, inWindow);
        }
        // fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, m_kLoadSize>(xbuff, xNumDataLoads % m_kDataLoadsInReg, inWindow);
        xNumDataLoads++;
        // xDataLoaded += m_kDataLoadVsize;
    }

    for (int i = 0; i < m_kLsize / 2; i++) chess_prepare_for_pipelining chess_loop_range(m_kLsize / 2, ) {
            coeff = ((T_buff_256b<TT_COEFF>*)&m_phaseOneTaps[0][0]);
            coe0 = *coeff;
            xNumDataLoads = m_kInitLoadsXneeded;
            xDataLoaded = m_kInitLoadsXneeded * m_kDataLoadVsize;
            xstart = m_kDataBuffXOffset;
            ystart = m_kyStart; // first column of first load
            coeffstart = 0;
            yNumDataLoads = 0;
            // Pre-loading the ybuff differs from the xbuff load because ystart is not in general aligned to loads.
            xDataNeeded =
                m_kDataBuffXOffset + (m_kLanes + m_kColumns - 2) * kDecimateFactor + 1; // e.g. D0 to D8 is 9, not 10.
            yDataNeeded = (m_kColumns - 1) * kDecimateFactor - 1; // datum is lane 0, but y needs go backwards.
            yDataLoaded = ystart % m_kDataLoadVsize;              // i.e. how many do we have in hand
            ySplice = m_kNumOps * m_kDataLoadsInReg - 1;          // allow ysplice to count down to 0

// Unroll this so that we see constant splices.
#pragma unroll(m_kRepeatFactor)
            for (int dataLoadPhase = 0; dataLoadPhase < m_kRepeatFactor; dataLoadPhase++) {
                ///////////////////////// Forward /////////////////////////////////////////////////
                // Mostly deprecated. passes loop was for multiple accumulators when we wanted to use double accumulator
                // for (int pass = 0; pass < m_kPasses; ++pass) {

                // Read cascade input. Do nothing if cascade input not present.
                acc[0] = readCascade<TT_DATA, TT_COEFF>(inInterface, acc[0]);

                // Initial multiply
                if (m_kNumOps == 1 && m_kCtPresent == 1) {
                    lastXstart = (xstart - m_kFinalOpSkew + m_kSamplesInDataBuff) % m_kSamplesInDataBuff;
                    lastYstart = (ystart + m_kFinalOpSkew + m_kSamplesInDataBuff) % m_kSamplesInDataBuff;
                    acc[0] = initMacDecHbCt(inInterface, acc[0], xbuff, lastXstart, ybuff, lastYstart,
                                            (m_kColumns - 1) * kDecimateFactor - 1, // m_kCtOffset
                                            coeCt, 0);
                } else {
                    acc[0] = initMacDecHb(inInterface, acc[0], xbuff, xstart, ybuff, ystart, coe0, coeffstart);
                }

// In the operations loop, x and y buffs load at different times because y can start mid-splice.
#pragma unroll(GUARD_ZERO((m_kNumOps - 1)))
                for (int op = 1; op < m_kNumOps; ++op) {
                    if (op < m_kNumOps - 1) {
                        xDataNeeded += m_kColumns * kDecimateFactor;
                        yDataNeeded += m_kColumns * kDecimateFactor;
                    } else {
                        xDataNeeded += (m_kColumns - 1) * kDecimateFactor + 1;
                        yDataNeeded += (m_kColumns - 1) * kDecimateFactor + 1;
                    }
                    if (xDataNeeded > xDataLoaded) {
                        // Load xdata from window into xbuff register
                        fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, m_kLoadSize>(xbuff, xNumDataLoads % m_kDataLoadsInReg,
                                                                               inWindow);
                        xNumDataLoads++;
                        xDataLoaded += m_kDataLoadVsize;
                    }
                    if (yDataNeeded > yDataLoaded) {
                        // Load ydata from window into ybuff register
                        fnLoadYIpData<TT_DATA, TT_COEFF, m_kArch, m_kLoadSize>(ybuff, ySplice % m_kDataLoadsInReg,
                                                                               yinWindow);
                        ySplice--;
                        yNumDataLoads++;
                        yDataLoaded += m_kDataLoadVsize; // 0 maps to mkDataLoadsVsize
                    }

                    xstart += m_kColumns * kDecimateFactor;
                    ystart -= m_kColumns * kDecimateFactor;
                    coeffstart += m_kColumns;
                    if (op % (m_kCoeffRegVsize / m_kColumns) == 0) {
                        // Load coefficients coe0 register
                        coeff = ((T_buff_256b<TT_COEFF>*)&m_phaseOneTaps[op][0]);
                        coe0 = *coeff;
                    }
                    if ((op >= m_kNumOps - 1) && (m_kCtPresent == 1)) {
                        // Last operation includes the centre tap
                        lastXstart = (xstart - m_kFinalOpSkew + m_kSamplesInDataBuff) % m_kSamplesInDataBuff;
                        lastYstart = (ystart + m_kFinalOpSkew + m_kSamplesInDataBuff) % m_kSamplesInDataBuff;
                        // Final multiply operation
                        acc[0] = firDecHbMacSymCt(acc[0], xbuff, lastXstart, ybuff, lastYstart,
                                                  (m_kColumns - 1) * kDecimateFactor - 1, // m_kCtOffset
                                                  coeCt, 0);
                    } else {
                        // Multiply operation
                        acc[0] = firDecHbMacSym(acc[0], xbuff, xstart, ybuff, ystart, coe0, coeffstart);
                    }
                } // operations loop

                window_incr(yinWindow,
                            ((m_kNumOps + 1) / m_kOpsEachLoad) * m_kDataLoadVsize); // moved from prep for forward stage

                // Write cascade. Do nothing if cascade not present.
                writeCascade<TT_DATA, TT_COEFF, m_kArch>(outInterface, acc[0]);

                // The data for the centre tap is the same data as required for the last op of the top phase, so is
                // already loaded
                outVal = firDecHbWriteOut<TT_DATA, TT_COEFF, m_kArch>(acc, TP_SHIFT);
                writeWindow<TT_DATA, TT_COEFF, m_kArch, TP_NUM_OUTPUTS>(outInterface, outVal); // NOT from kernel_utils

                ////////////////// Prepare for Reverse /////////////////////
                // Need a chunk more data for reverse midpoint calc
                xDataNeeded += m_kDataNeededLastOp;
                yDataNeeded += m_kDataNeededLastOp;

                if (xDataNeeded > xDataLoaded) {
                    // Load xdata from window into xbuff register - read dec
                    fnLoadYIpData<TT_DATA, TT_COEFF, m_kArch, m_kLoadSize>(xbuff, xNumDataLoads % m_kDataLoadsInReg,
                                                                           inWindow);
                    // we don't ajust splice for this load.
                    // xNumDataLoads++;
                    xDataLoaded += m_kDataLoadVsize;
                }
                // This could be a decrement or increment depending on num_ops (fir_len)
                // 23 tap cint16 int16 would be +1, 55 tap cint16 int16 would be -1
                window_incr(inWindow, m_kDataLoadVsize * m_kWindowModX);
                if (yDataNeeded > yDataLoaded) {
                    // Load ydata from window into ybuff register, in forward direction for next chunk of outputs
                    // need to adjust ySplice
                    fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, m_kLoadSize>(ybuff, (ySplice - 1) % m_kDataLoadsInReg,
                                                                           yRevinWindow);
                    // we don't ajust splice for this load.
                    yDataLoaded += m_kDataLoadVsize;
                }
                ////////////////// Reverse ////////////////////////////////

                // Final xystart values
                xstart = m_kDataBuffXOffset + m_kDataNeededEachOp * (m_kNumOps);
                ystart = m_kyStart - m_kDataNeededEachOp * (m_kNumOps);

                // coefstart already starts at last mid point.

                lastXstart = (xstart - m_kFinalOpSkew + m_kSamplesInDataBuff) % m_kSamplesInDataBuff;
                lastYstart = (ystart + m_kFinalOpSkew + m_kSamplesInDataBuff) % m_kSamplesInDataBuff;

                // Read cascade input. Do nothing if cascade input not present.
                acc[0] = readCascade<TT_DATA, TT_COEFF>(inInterface, acc[0]);
                // midpoint multiply operation
                if (m_kCtPresent == 1) {
                    acc[0] = initMacDecHbCt(inInterface, acc[0], xbuff, lastXstart, ybuff, lastYstart,
                                            (m_kColumns - 1) * kDecimateFactor - 1, // m_kCtOffset
                                            coeCt, 0);
                } else {
                    acc[0] = initMacDecHb(inInterface, acc[0], xbuff, xstart, ybuff, ystart, coe0, coeffstart);
                }

#pragma unroll(GUARD_ZERO((m_kNumOps - 1)))
                for (int op = 1; op < m_kNumOps; ++op) {
                    xDataNeeded += m_kDataNeededEachOp;
                    yDataNeeded += m_kDataNeededEachOp;

                    if (xDataNeeded > xDataLoaded) {
                        // Load xdata from window into xbuff register
                        fnLoadYIpData<TT_DATA, TT_COEFF, m_kArch, m_kLoadSize>(
                            xbuff, (m_kDataLoadsInReg - xNumDataLoads) % m_kDataLoadsInReg, inRevWindow);
                        xNumDataLoads++;
                        xDataLoaded += m_kDataLoadVsize;
                    }
                    if (yDataNeeded > yDataLoaded) {
                        // Load ydata from window into ybuff register
                        fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, m_kLoadSize>(
                            ybuff, (m_kDataLoadsInReg - ySplice) % m_kDataLoadsInReg, yRevinWindow);
                        ySplice--;
                        yNumDataLoads++;
                        yDataLoaded += m_kDataLoadVsize; // 0 maps to mkDataLoadsVsize
                    }

                    // reverse direction
                    xstart -= m_kDataNeededEachOp;
                    ystart += m_kDataNeededEachOp;
                    coeffstart -= m_kColumns;
                    // m_kNumOps-op gives the forward-equivalent op.
                    if ((m_kNumOps - op) % (m_kCoeffRegVsize / m_kColumns) == 0) {
                        // Load a CoeffRegSize worth of coefficients up to (m_kNumOps - op)
                        coeff = ((T_buff_256b<TT_COEFF>*)&m_phaseOneTaps[(m_kNumOps - op) -
                                                                         (m_kCoeffRegVsize / m_kColumns)][0]);
                        coe0 = *coeff;
                    }
                    // Reverse loop always has standard mac, as centre tap calc is initial calc
                    acc[0] = firDecHbMacSym(acc[0], xbuff, xstart, ybuff, ystart, coe0, coeffstart);
                } // operations loop

                window_incr(inRevWindow,
                            ((m_kNumOps + 1) / m_kOpsEachLoad) * m_kDataLoadVsize); // moved from prep for forward stage

                // Write cascade. Do nothing if cascade not present.
                writeCascade<TT_DATA, TT_COEFF, m_kArch>(outInterface, acc[0]);

                outVal = firDecHbWriteOut<TT_DATA, TT_COEFF, m_kArch>(acc, TP_SHIFT);
                writeWindow<TT_DATA, TT_COEFF, m_kArch, TP_NUM_OUTPUTS>(outInterface, outVal); // NOT from kernel_utils

                ///////////////// Prepare for Forward //////////////////

                xNumDataLoads = m_kInitLoadsXneeded - 1; // we only need one load
                xDataLoaded = (m_kInitLoadsXneeded - 1) * m_kDataLoadVsize;
                yNumDataLoads = m_kInitLoadsYneeded - 1;                    // we only need one load;
                yDataLoaded = (m_kInitLoadsYneeded - 1) * m_kDataLoadVsize; // i.e. how many do we have in hand
                ySplice = m_kInitLoadsYneeded - 1;                          // we only need one load

// preload xdata from window into xbuff register
#pragma unroll(1)
                for (unsigned int initLoads = 0; initLoads < 1; ++initLoads) {
                    if ((1 - initLoads) >= 2 && m_kLoadSize == 128) {
                        fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, (m_kLoadSize * 2)>(
                            xbuff, (xNumDataLoads / 2) % m_kDataLoadsInReg, inWindow);
                        xNumDataLoads++;
                        initLoads++;
                        xDataLoaded += m_kDataLoadVsize;
                    } else {
                        fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, m_kLoadSize>(xbuff, xNumDataLoads % m_kDataLoadsInReg,
                                                                               inWindow);
                    }
                    // fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, m_kLoadSize>(xbuff, xNumDataLoads % m_kDataLoadsInReg,
                    // inWindow);
                    xNumDataLoads++;
                    xDataLoaded += m_kDataLoadVsize;
                }
// preload ydata from window into ybuff register
#pragma unroll(1)
                for (unsigned int initLoads = 0; initLoads < 1; ++initLoads) {
                    if ((1 - initLoads) >= 2 && m_kLoadSize == 128) {
                        fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, (m_kLoadSize * 2)>(
                            ybuff, (ySplice / 2) % m_kDataLoadsInReg, yRevinWindow);
                        initLoads++;
                        ySplice++;

                    } else {
                        // Note that initial Y loads are forwards, hence use the forward direction load function.
                        fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, m_kLoadSize>(ybuff, ySplice % m_kDataLoadsInReg,
                                                                               yRevinWindow);
                    }
                    ySplice++;
                }

                xNumDataLoads = m_kInitLoadsXneeded;
                xDataLoaded = m_kInitLoadsXneeded * m_kDataLoadVsize;
                xstart = m_kDataBuffXOffset;
                ystart = m_kyStart; // first column of first load
                coeffstart = 0;
                // Pre-loading the ybuff differs from the xbuff load because ystart is not in general aligned to loads.
                xDataNeeded = m_kDataBuffXOffset + (m_kLanes + m_kColumns - 2) * kDecimateFactor +
                              1;                                      // e.g. D0 to D8 is 9, not 10.
                yDataNeeded = (m_kColumns - 1) * kDecimateFactor - 1; // datum is lane 0, but y needs go backwards.
                yDataLoaded = ystart % m_kDataLoadVsize;              // i.e. how many do we have in hand
                ySplice = m_kNumOps * m_kDataLoadsInReg - 1;          // allow ysplice to count down to 0

                // reset window pointers to be reading from correct slices
                // TODO: check if we need floor or ceil num loads calc
                window_decr(yRevinWindow, ((m_kNumOps - 1) / m_kOpsEachLoad) * m_kDataLoadVsize);

            } // strobe loop
        }     // i loop (splice of window)
};

//-----------------------------------------------------------------------------------------------------
// The filterSmall variant of this function is for cases where all the data samples required may be loaded into
// the sbuff such that the single buffer may be used for both xbuff and ybuff.
template <typename TT_DATA,
          typename TT_COEFF,
          size_t TP_FIR_LEN,
          size_t TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          bool TP_CASC_IN,
          bool TP_CASC_OUT,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP,
          unsigned int TP_USE_COEFF_RELOAD,
          unsigned int TP_NUM_OUTPUTS>
inline void kernelFilterClass<TT_DATA,
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
                              TP_DUAL_IP,
                              TP_USE_COEFF_RELOAD,
                              TP_NUM_OUTPUTS>::filter1buff(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                                                           T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    // Pointers to coefficient storage and explicit registers to hold values
    T_buff_256b<TT_COEFF>* restrict coeff = (T_buff_256b<TT_COEFF>*)m_phaseOneTaps;
    T_buff_256b<TT_COEFF> coe0; // register for coeff values.
    T_buff_256b<TT_COEFF>* restrict coeffCt = (T_buff_256b<TT_COEFF>*)m_phaseOneTapsCt;
    T_buff_256b<TT_COEFF> coeCt = *coeffCt; // register for coeff values.
    T_buff_FirDecHb<TT_DATA, TT_COEFF, m_kArch> sbuff;
    T_accFirDecHb<TT_DATA, TT_COEFF, m_kArch> acc[m_kPasses]; // 1 for small 1buff algo.
    T_acc768<TT_DATA, TT_COEFF> accConcat;
    T_outValFiRDecHb<TT_DATA, TT_COEFF, m_kArch> outVal;

    unsigned int xDataLoaded, xDataNeeded, numDataLoads;
    unsigned int xstart, ystart, coeffstart;

    input_window<TT_DATA>* restrict inWindow = inInterface.inWindow;

    window_incr(inWindow, m_kDataWindowOffset); // move input data pointer past the margin padding

    // Architecture never requires more than 1 register set of coeffs.
    coeff = ((T_buff_256b<TT_COEFF>*)&m_phaseOneTaps[0][0]);
    coe0 = *coeff;

// preamble, load data from window into register
#pragma unroll(m_kInitialLoads - m_kIncrLoadsTopUp)
    for (unsigned int initLoads = 0; initLoads < m_kInitialLoads - m_kIncrLoadsTopUp; ++initLoads) {
        fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, m_kLoadSize>(sbuff, initLoads, inWindow);
    }

    for (unsigned i = 0; i < m_kLsize / m_kIncrRepeatFactor; i++)
        chess_prepare_for_pipelining chess_loop_range(m_kLsize / m_kIncrRepeatFactor, ) {
            numDataLoads = 0;
            xDataLoaded = (m_kInitialLoads - m_kIncrLoadsTopUp) * m_kDataLoadVsize;
            xDataNeeded = m_kInitialLoads * m_kDataLoadVsize;
#pragma unroll(m_kIncrRepeatFactor)
            for (unsigned dataLoadPhase = 0; dataLoadPhase < m_kIncrRepeatFactor; dataLoadPhase++) {
                if (xDataNeeded > xDataLoaded) {
                    // Load xdata from window into xbuff register
                    fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, m_kLoadSize>(
                        sbuff, (m_kInitialLoads - m_kIncrLoadsTopUp) + numDataLoads, inWindow);
                    numDataLoads++;
                    xDataLoaded += m_kDataLoadVsize;
                }
                if (xDataNeeded > xDataLoaded) {
                    // Load xdata from window into xbuff register
                    fnLoadXIpData<TT_DATA, TT_COEFF, m_kArch, m_kLoadSize>(
                        sbuff, (m_kInitialLoads - m_kIncrLoadsTopUp) + numDataLoads, inWindow);
                    numDataLoads++;
                    xDataLoaded += m_kDataLoadVsize;
                }
                xDataNeeded += m_kVOutSize * kDecimateFactor;

                xstart = dataLoadPhase * m_kVOutSize * kDecimateFactor + m_kDataBuffXOffset;
                ystart = dataLoadPhase * m_kVOutSize * kDecimateFactor + m_kDataBuffXOffset + (TP_FIR_LEN - 1) -
                         m_kFirRangeOffset * kDecimateFactor;
                coeffstart = 0;

                // Read cascade input. Do nothing if cascade input not present.
                acc[0] = readCascade(inInterface, acc[0]);
                // Initial multiply
                if (m_kNumOps == 1 && m_kCtPresent == 1) {
                    acc[0] =
                        initMacDecHbCt(inInterface, acc[0], sbuff, xstart - m_kFinalOpSkew, ystart + m_kFinalOpSkew,
                                       m_kCtOffset, coeCt, 0); // Final multiply, if there's only one
                } else {
                    acc[0] = initMacDecHb(inInterface, acc[0], sbuff, xstart, ystart, coe0, coeffstart);
                }

#pragma unroll(GUARD_ZERO((m_kNumOps - 1)))
                for (int op = 1; op < m_kNumOps; ++op) {
                    xstart += m_kColumns * kDecimateFactor;
                    ystart -= m_kColumns * kDecimateFactor;
                    coeffstart += m_kColumns;
                    if (op == m_kNumOps - 1 && m_kCtPresent == 1) {
                        // Final multiply
                        acc[0] = firDecHbMacSymCt1buff(acc[0], sbuff, xstart - m_kFinalOpSkew, ystart + m_kFinalOpSkew,
                                                       m_kCtOffset, coeCt, 0);
                    } else {
                        // Multiply operation
                        acc[0] = firDecHbMacSym1buff(acc[0], sbuff, xstart, ystart, coe0, coeffstart);
                    }
                }
                // Write cascade. Do nothing if cascade not present.
                writeCascade<TT_DATA, TT_COEFF, m_kArch>(outInterface, acc[0]);

                outVal = firDecHbWriteOut<TT_DATA, TT_COEFF, m_kArch>(acc, TP_SHIFT);
                writeWindow<TT_DATA, TT_COEFF, m_kArch, TP_NUM_OUTPUTS>(outInterface, outVal); // NOT from kernel_utils
            }
        }
};

//----------------------------------------------------------------------------
// Start of Cascade-layer class body and specializations

// This is a (default) specialization of the main class for when there is only one kernel for the whole filter, single
// input and static coeffs, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          bool TP_CASC_IN,
          bool TP_CASC_OUT,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP,
          unsigned int TP_USE_COEFF_RELOAD,
          unsigned int TP_NUM_OUTPUTS>
void fir_decimate_hb<TT_DATA,
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
                     TP_DUAL_IP,
                     TP_USE_COEFF_RELOAD,
                     TP_NUM_OUTPUTS>::filter(input_window<TT_DATA>* restrict inWindow,
                                             output_window<TT_DATA>* restrict outWindow) {
    T_inputIF<CASC_IN_FALSE, TT_DATA, TP_DUAL_IP> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outWindow = outWindow;
    this->filterKernel(inInterface, outInterface);
};

// specialization for one kernel for the whole filter, single input and static coeffs, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
void fir_decimate_hb<TT_DATA,
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
                     2>::filter(input_window<TT_DATA>* restrict inWindow,
                                output_window<TT_DATA>* restrict outWindow,
                                output_window<TT_DATA>* restrict outWindow2) {
    T_inputIF<CASC_IN_FALSE, TT_DATA, DUAL_IP_SINGLE> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outWindow = outWindow;
    outInterface.outWindow2 = outWindow2;
    this->filterKernel(inInterface, outInterface);
};

// specialization for one kernel for the whole filter, but with dual inputs, static coeffs and single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
void fir_decimate_hb<TT_DATA,
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
                     1>::filter(input_window<TT_DATA>* restrict inWindow,
                                input_window<TT_DATA>* restrict inWindowReverse,
                                output_window<TT_DATA>* restrict outWindow) {
    T_inputIF<CASC_IN_FALSE, TT_DATA, DUAL_IP_DUAL> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inWindowReverse = inWindowReverse;
    outInterface.outWindow = outWindow;
    this->filterKernel(inInterface, outInterface);
};

// specialization for one kernel for the whole filter, but with dual inputs, static coeffs and dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
void fir_decimate_hb<TT_DATA,
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
                     2>::filter(input_window<TT_DATA>* restrict inWindow,
                                input_window<TT_DATA>* restrict inWindowReverse,
                                output_window<TT_DATA>* restrict outWindow,
                                output_window<TT_DATA>* restrict outWindow2) {
    T_inputIF<CASC_IN_FALSE, TT_DATA, DUAL_IP_DUAL> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inWindowReverse = inWindowReverse;
    outInterface.outWindow = outWindow;
    outInterface.outWindow2 = outWindow2;
    this->filterKernel(inInterface, outInterface);
};

// specialization for one kernel for the whole filter, but with single input, reload, single output.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
void fir_decimate_hb<TT_DATA,
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
                     1>::filter(input_window<TT_DATA>* restrict inWindow,
                                output_window<TT_DATA>* restrict outWindow,
                                const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / 4 + 1]) {
    T_inputIF<CASC_IN_FALSE, TT_DATA, DUAL_IP_SINGLE> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outWindow = outWindow;
    this->filterKernel(inInterface, outInterface, inTaps);
};

// specialization for one kernel for the whole filter, but with single input, reload, dual output.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
void fir_decimate_hb<TT_DATA,
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
                     2>::filter(input_window<TT_DATA>* restrict inWindow,
                                output_window<TT_DATA>* restrict outWindow,
                                output_window<TT_DATA>* restrict outWindow2,
                                const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / 4 + 1]) {
    T_inputIF<CASC_IN_FALSE, TT_DATA, DUAL_IP_SINGLE> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outWindow = outWindow;
    outInterface.outWindow2 = outWindow2;
    this->filterKernel(inInterface, outInterface, inTaps);
};

// specialization for one kernel for the whole filter, but with dual inputs, reload, single output.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
void fir_decimate_hb<TT_DATA,
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
                     1>::filter(input_window<TT_DATA>* restrict inWindow,
                                input_window<TT_DATA>* restrict inWindowReverse,
                                output_window<TT_DATA>* restrict outWindow,
                                const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / 4 + 1]) {
    T_inputIF<CASC_IN_FALSE, TT_DATA, DUAL_IP_DUAL> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inWindowReverse = inWindowReverse;
    outInterface.outWindow = outWindow;
    this->filterKernel(inInterface, outInterface, inTaps);
};

// specialization for one kernel for the whole filter, but with dual inputs, reload, dual output.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE>
void fir_decimate_hb<TT_DATA,
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
                     2>::filter(input_window<TT_DATA>* restrict inWindow,
                                input_window<TT_DATA>* restrict inWindowReverse,
                                output_window<TT_DATA>* restrict outWindow,
                                output_window<TT_DATA>* restrict outWindow2,
                                const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / 4 + 1]) {
    T_inputIF<CASC_IN_FALSE, TT_DATA, DUAL_IP_DUAL> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inWindowReverse = inWindowReverse;
    outInterface.outWindow = outWindow;
    outInterface.outWindow2 = outWindow2;
    this->filterKernel(inInterface, outInterface, inTaps);
};

// specialization for the final kernel in a cascade chain with single input and no coefficient reloads, single output
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
void fir_decimate_hb<TT_DATA,
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
                     1>::filter(input_window<TT_DATA>* restrict inWindow,
                                input_stream_cacc48* inCascade,
                                output_window<TT_DATA>* restrict outWindow) {
    T_inputIF<CASC_IN_TRUE, TT_DATA, TP_DUAL_IP> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inCascade = inCascade;
    outInterface.outWindow = outWindow;
    this->filterKernel(inInterface, outInterface);
};

// specialization for the final kernel in a cascade chain with single input and no coefficient reloads, dual output
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
void fir_decimate_hb<TT_DATA,
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
                     2>::filter(input_window<TT_DATA>* restrict inWindow,
                                input_stream_cacc48* inCascade,
                                output_window<TT_DATA>* restrict outWindow,
                                output_window<TT_DATA>* restrict outWindow2) {
    T_inputIF<CASC_IN_TRUE, TT_DATA, TP_DUAL_IP> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inCascade = inCascade;
    outInterface.outWindow = outWindow;
    outInterface.outWindow2 = outWindow2;
    this->filterKernel(inInterface, outInterface);
};

// specialization for the final kernel in a cascade chain with dual inputs and no coefficient reloads, single output.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_decimate_hb<TT_DATA,
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
                     1>::filter(input_window<TT_DATA>* restrict inWindow,
                                input_window<TT_DATA>* restrict inWindowReverse,
                                input_stream_cacc48* inCascade,
                                output_window<TT_DATA>* restrict outWindow) {
    // The cascade layer inputs must be in atomic types supported by adf. The kernel class inputs can be templatized.
    T_inputIF<CASC_IN_TRUE, TT_DATA, DUAL_IP_DUAL> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inWindowReverse = inWindowReverse;
    inInterface.inCascade = inCascade;
    outInterface.outWindow = outWindow;
    this->filterKernel(inInterface, outInterface);
};

// specialization for the final kernel in a cascade chain with dual inputs and no coefficient reloads, dual output.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_decimate_hb<TT_DATA,
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
                     2>::filter(input_window<TT_DATA>* restrict inWindow,
                                input_window<TT_DATA>* restrict inWindowReverse,
                                input_stream_cacc48* inCascade,
                                output_window<TT_DATA>* restrict outWindow,
                                output_window<TT_DATA>* restrict outWindow2) {
    // The cascade layer inputs must be in atomic types supported by adf. The kernel class inputs can be templatized.
    T_inputIF<CASC_IN_TRUE, TT_DATA, DUAL_IP_DUAL> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inWindowReverse = inWindowReverse;
    inInterface.inCascade = inCascade;
    outInterface.outWindow = outWindow;
    outInterface.outWindow2 = outWindow2;
    this->filterKernel(inInterface, outInterface);
};

// This is a specialization of the main class for the final kernel in a cascade chain with single input and coefficient
// reloads, single output
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
void fir_decimate_hb<TT_DATA,
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
                     1>::filter(input_window<TT_DATA>* restrict inWindow,
                                input_stream_cacc48* inCascade,
                                output_window<TT_DATA>* restrict outWindow) {
    T_inputIF<CASC_IN_TRUE, TT_DATA, TP_DUAL_IP> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inCascade = inCascade;
    outInterface.outWindow = outWindow;
    this->filterKernelRtp(inInterface, outInterface);
};

// This is a specialization of the main class for the final kernel in a cascade chain with single input and coefficient
// reloads, dual output
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
void fir_decimate_hb<TT_DATA,
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
                     2>::filter(input_window<TT_DATA>* restrict inWindow,
                                input_stream_cacc48* inCascade,
                                output_window<TT_DATA>* restrict outWindow,
                                output_window<TT_DATA>* restrict outWindow2) {
    T_inputIF<CASC_IN_TRUE, TT_DATA, TP_DUAL_IP> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inCascade = inCascade;
    outInterface.outWindow = outWindow;
    outInterface.outWindow2 = outWindow2;
    this->filterKernelRtp(inInterface, outInterface);
};

// This is a specialization of the main class for the final kernel in a cascade chain with dual inputs and coefficient
// reloads, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_decimate_hb<TT_DATA,
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
                     1>::filter(input_window<TT_DATA>* restrict inWindow,
                                input_window<TT_DATA>* restrict inWindowReverse,
                                input_stream_cacc48* inCascade,
                                output_window<TT_DATA>* restrict outWindow) {
    // The cascade layer inputs must be in atomic types supported by adf. The kernel class inputs can be templatized.
    T_inputIF<CASC_IN_TRUE, TT_DATA, DUAL_IP_DUAL> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inWindowReverse = inWindowReverse;
    inInterface.inCascade = inCascade;
    outInterface.outWindow = outWindow;
    this->filterKernelRtp(inInterface, outInterface);
};

// This is a specialization of the main class for the final kernel in a cascade chain with dual inputs and coefficient
// reloads, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_decimate_hb<TT_DATA,
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
                     2>::filter(input_window<TT_DATA>* restrict inWindow,
                                input_window<TT_DATA>* restrict inWindowReverse,
                                input_stream_cacc48* inCascade,
                                output_window<TT_DATA>* restrict outWindow,
                                output_window<TT_DATA>* restrict outWindow2) {
    // The cascade layer inputs must be in atomic types supported by adf. The kernel class inputs can be templatized.
    T_inputIF<CASC_IN_TRUE, TT_DATA, DUAL_IP_DUAL> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inWindowReverse = inWindowReverse;
    inInterface.inCascade = inCascade;
    outInterface.outWindow = outWindow;
    outInterface.outWindow2 = outWindow2;
    this->filterKernelRtp(inInterface, outInterface);
};

// This is a specialization of the main class for the first kernel in a cascade chain and no coefficient reloads.
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
void fir_decimate_hb<TT_DATA,
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
                     1>::filter(input_window<TT_DATA>* inWindow,
                                output_stream_cacc48* outCascade,
                                output_window<TT_DATA>* broadcastWindow) {
    T_inputIF<CASC_IN_FALSE, TT_DATA, TP_DUAL_IP> inInterface;
    T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outCascade = outCascade;
    outInterface.broadcastWindow = broadcastWindow;
    this->filterKernel(inInterface, outInterface);
};

// This is a specialization of the main class for the first kernel in a cascade chain with dual inputs and no
// coefficient reloads.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_decimate_hb<TT_DATA,
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
                     1>::filter(input_window<TT_DATA>* inWindow,
                                input_window<TT_DATA>* inWindowReverse,
                                output_stream_cacc48* outCascade,
                                output_window<TT_DATA>* broadcastWindow) {
    T_inputIF<CASC_IN_FALSE, TT_DATA, DUAL_IP_DUAL> inInterface;
    T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inWindowReverse = inWindowReverse;
    outInterface.outCascade = outCascade;
    outInterface.broadcastWindow = broadcastWindow;
    this->filterKernel(inInterface, outInterface);
};

// This is a specialization of the main class for the first kernel in a cascade chain and coefficient reloads.
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
void fir_decimate_hb<TT_DATA,
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
                     1>::filter(input_window<TT_DATA>* inWindow,
                                output_stream_cacc48* outCascade,
                                output_window<TT_DATA>* broadcastWindow,
                                const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / 4 + 1]) {
    T_inputIF<CASC_IN_FALSE, TT_DATA, TP_DUAL_IP> inInterface;
    T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outCascade = outCascade;
    outInterface.broadcastWindow = broadcastWindow;
    this->filterKernel(inInterface, outInterface, inTaps);
};

// This is a specialization of the main class for the first kernel in a cascade chain with dual inputs and coefficient
// reloads.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_decimate_hb<TT_DATA,
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
                     1>::filter(input_window<TT_DATA>* inWindow,
                                input_window<TT_DATA>* inWindowReverse,
                                output_stream_cacc48* outCascade,
                                output_window<TT_DATA>* broadcastWindow,
                                const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / 4 + 1]) {
    T_inputIF<CASC_IN_FALSE, TT_DATA, DUAL_IP_DUAL> inInterface;
    T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inWindowReverse = inWindowReverse;
    outInterface.outCascade = outCascade;
    outInterface.broadcastWindow = broadcastWindow;
    this->filterKernel(inInterface, outInterface, inTaps);
};

// This is a specialization of the main class for any kernel within a cascade chain, but neither first nor last and no
// coefficient reloads.
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
void fir_decimate_hb<TT_DATA,
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
                     1>::filter(input_window<TT_DATA>* inWindow,
                                input_stream_cacc48* inCascade,
                                output_stream_cacc48* outCascade,
                                output_window<TT_DATA>* broadcastWindow) {
    T_inputIF<CASC_IN_TRUE, TT_DATA, TP_DUAL_IP> inInterface;
    T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inCascade = inCascade;
    outInterface.outCascade = outCascade;
    outInterface.broadcastWindow = broadcastWindow;
    this->filterKernel(inInterface, outInterface);
};

// This is a specialization of the main class for any kernel within a cascade chain, but neither first nor last, with
// dual inputs and no coefficient reloads.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_decimate_hb<TT_DATA,
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
                     1>::filter(input_window<TT_DATA>* inWindow,
                                input_window<TT_DATA>* inWindowReverse,
                                input_stream_cacc48* inCascade,
                                output_stream_cacc48* outCascade,
                                output_window<TT_DATA>* broadcastWindow) {
    T_inputIF<CASC_IN_TRUE, TT_DATA, DUAL_IP_DUAL> inInterface;
    T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inWindowReverse = inWindowReverse;
    inInterface.inCascade = inCascade;
    outInterface.outCascade = outCascade;
    outInterface.broadcastWindow = broadcastWindow;
    this->filterKernel(inInterface, outInterface);
};

// This is a specialization of the main class for any kernel within a cascade chain, but neither first nor last, with
// single input and coefficient reloads.
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
void fir_decimate_hb<TT_DATA,
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
                     1>::filter(input_window<TT_DATA>* inWindow,
                                input_stream_cacc48* inCascade,
                                output_stream_cacc48* outCascade,
                                output_window<TT_DATA>* broadcastWindow) {
    T_inputIF<CASC_IN_TRUE, TT_DATA, TP_DUAL_IP> inInterface;
    T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inCascade = inCascade;
    outInterface.outCascade = outCascade;
    outInterface.broadcastWindow = broadcastWindow;
    this->filterKernelRtp(inInterface, outInterface);
};

// This is a specialization of the main class for any kernel within a cascade chain, but neither first nor last, with
// dual inputs and coefficient reloads.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_decimate_hb<TT_DATA,
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
                     1>::filter(input_window<TT_DATA>* inWindow,
                                input_window<TT_DATA>* inWindowReverse,
                                input_stream_cacc48* inCascade,
                                output_stream_cacc48* outCascade,
                                output_window<TT_DATA>* broadcastWindow) {
    T_inputIF<CASC_IN_TRUE, TT_DATA, DUAL_IP_DUAL> inInterface;
    T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inWindowReverse = inWindowReverse;
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
