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
Halfband interpolating FIR kernel code.
 This file captures the body of run-time code for the kernel class and a higher wrapping 'cascade' layer which has
specializations for
 combinations of inputs and outputs. That is, in a chain of kernels, the first will have an input window, and a cascade
out stream.
 The next, potentially multiple, kernel(s) will each have an input window and cascade stream and will output a cascade
steam. The final kernel
 will have an input window and cascade stream and an output window only.
 The cascade layer class is called fir_interpolate_hb with the kernel-layer (operational) class called
kernelFilterClass.
 The fir_interpolate_hb class has a member of the kernelFilterClass.

 Coding conventions
  TT_      template type suffix
  TP_      template parameter suffix
*/

#pragma once
#include <adf.h>

#define __NEW_WINDOW_H__ 1
#define __AIE_API_USE_NATIVE_1024B_VECTOR__
#include "aie_api/aie_adf.hpp"

#include "kernel_api_utils.hpp"
#include "fir_interpolate_hb.hpp"
#include "fir_interpolate_hb_utils.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace interpolate_hb {

// FIR function
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
          unsigned int TP_NUM_OUTPUTS,
          unsigned int TP_UPSHIFT_CT>
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
                              TP_NUM_OUTPUTS,
                              TP_UPSHIFT_CT>::filterKernel(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                                                           T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    windowBroadcast<TT_DATA, TP_INPUT_WINDOW_VSIZE + fnFirMargin<TP_FIR_LEN / kInterpolateFactor, TT_DATA>()>(
        inInterface, outInterface);
    filterSelectArch(inInterface, outInterface);
};

// FIR function - overloaded (not specialized) with taps for reload
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
          unsigned int TP_NUM_OUTPUTS,
          unsigned int TP_UPSHIFT_CT>
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
                              TP_NUM_OUTPUTS,
                              TP_UPSHIFT_CT>::filterKernel(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                                                           T_outputIF<TP_CASC_OUT, TT_DATA> outInterface,
                                                           const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / 4 + 1]) {
    windowBroadcast<TT_DATA, TP_INPUT_WINDOW_VSIZE + fnFirMargin<TP_FIR_LEN / kInterpolateFactor, TT_DATA>()>(
        inInterface, outInterface);
    m_coeffnEq = rtpCompare(inTaps, m_oldInTaps);

    sendRtpTrigger(m_coeffnEq, outInterface);
    if (m_coeffnEq) { // Coefficients have changed
        bufferReload(inTaps, m_oldInTaps, outInterface);
        firReload(inTaps);
    }
    filterSelectArch(inInterface, outInterface);
};

// FIR function
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
          unsigned int TP_NUM_OUTPUTS,
          unsigned int TP_UPSHIFT_CT>
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
                              TP_NUM_OUTPUTS,
                              TP_UPSHIFT_CT>::filterKernelRtp(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                                                              T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    windowBroadcast<TT_DATA, TP_INPUT_WINDOW_VSIZE + fnFirMargin<TP_FIR_LEN / kInterpolateFactor, TT_DATA>()>(
        inInterface, outInterface);
    m_coeffnEq = getRtpTrigger(); // 0 - equal, 1 - not equal

    sendRtpTrigger(m_coeffnEq, outInterface);
    if (m_coeffnEq) { // Coefficients have changed
        bufferReload<TT_DATA, TT_COEFF, (TP_FIR_LEN + 1) / 4 + 1, TP_DUAL_IP>(inInterface, m_oldInTaps, outInterface);
        firReload(m_oldInTaps);
    }
    filterSelectArch(inInterface, outInterface);
};

// FIR function - overloaded (not specialized) with taps for reload
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
          unsigned int TP_NUM_OUTPUTS,
          unsigned int TP_UPSHIFT_CT>
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
                              TP_NUM_OUTPUTS,
                              TP_UPSHIFT_CT>::filterSelectArch(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                                                               T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    windowAcquire(inInterface);
    // Two possible architectures depending on size of data/coef types & fir_len
    // Using a single data buffer for x and y (forward & reverse) or seperate
    if
        constexpr(m_kArch == kArch1Buff) { filter1buff(inInterface, outInterface); }
    else if
        constexpr(m_kArchZigZag == kArch2BuffZigZag) { filter2buffZigZag(inInterface, outInterface); }
    else {
        filter2buff(inInterface, outInterface);
    }
    windowRelease(inInterface);
};

// --------------------------------------------------- filter2buff -------------------------------------------------- //
// The filter2buff variant of this function is for cases where 2 separate buffers must be used, one for forward data
// and the other for reverse data.
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
          unsigned int TP_NUM_OUTPUTS,
          unsigned int TP_UPSHIFT_CT>
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
                              TP_NUM_OUTPUTS,
                              TP_UPSHIFT_CT>::filter2buff(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                                                          T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    set_rnd(TP_RND);
    set_sat();
    // const unsigned int     kSamplesInWindow = window_size(inWindow);  //number of samples in window

    // Pointers to coefficient storage and explicit registers to hold values
    T_buff_256b<TT_COEFF>* restrict coeff = (T_buff_256b<TT_COEFF>*)m_internalTapsFSA;
    T_buff_256b<TT_COEFF> coe0; // register for coeff values.
    T_buff_256b<TT_COEFF>* restrict ctCoeffptr =
        (T_buff_256b<TT_COEFF>*)m_phaseTwoTap; // register for centre tap coeff value.
    T_buff_256b<TT_COEFF> ctCoeff = *ctCoeffptr;
    T_buff_128b<TT_DATA> xReadData;
    T_buff_128b<TT_DATA> yReadData;
    T_buff_512b<TT_DATA> xbuff;
    T_buff_512b<TT_DATA> ybuff;
    T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> accHighPolyphase, accLowPolyphase;
    unsigned int xDataNeeded, xDataLoaded, xNumDataLoads;
    unsigned int yDataNeeded, yDataLoaded, yNumDataLoads, ySplice;
    unsigned int xstart, ystart, coeffstart;

    input_window<TT_DATA>* restrict inWindow = inInterface.inWindow;
    input_window<TT_DATA> yinWindowActual;
    input_window<TT_DATA>* restrict yinWindow;
    yinWindow = &yinWindowActual;
    if
        constexpr(TP_DUAL_IP == 0) { window_copy(yinWindow, inWindow); }
    else {
        window_copy(yinWindow, inInterface.inWindowReverse);
    }

    window_incr(inWindow, m_kXDataLoadInitOffset); // move input data pointer past the margin padding
    window_incr(yinWindow, m_kYDataLoadInitOffset);

    for (unsigned i = 0; i < m_kLsize; i++) chess_prepare_for_pipelining chess_loop_range(m_kLsize, ) {
            // Coeff is the pointer to the m_internalTapsFSA array, whereas coe0 is the register which holds a cache of
            // that.
            coeff = ((T_buff_256b<TT_COEFF>*)&m_internalTapsFSA);
            coe0 = *coeff++;

            xNumDataLoads = 0;
            xDataLoaded = 0;
            xstart = m_kDataBuffXOffset;
            ystart = m_kyStart;
            coeffstart = 0;

            // Pre-loading the ybuff differs from the xbuff load because ystart is not in general aligned to loads.
            yNumDataLoads = 0;
            yDataLoaded = 0;

// preload xdata from window into xbuff register
#pragma unroll(m_kInitialLoadsX)
            for (unsigned int initLoads = 0; initLoads < m_kInitialLoadsX; ++initLoads) {
                xReadData = window_readincr_128b<TT_DATA>(inWindow);
                xbuff.val = upd_v(xbuff.val, xNumDataLoads % m_kDataLoadsInReg, xReadData.val);
                xNumDataLoads++;
                xDataLoaded += m_kDataLoadVsize;
            }

            // preload ydata from window into ybuff register
            ySplice = m_kySpliceStart; //  m_kInitialLoadsY -1 already included
#pragma unroll(m_kInitialLoadsY)
            for (unsigned int initLoads = 0; initLoads < m_kInitialLoadsY; ++initLoads) {
                yReadData = window_readdecr_128b<TT_DATA>(yinWindow);
                ybuff.val = upd_v(ybuff.val, ySplice % m_kDataLoadsInReg, yReadData.val);
                ySplice--;
            }

            yNumDataLoads = m_kInitialLoadsY;

            // xNumDataLoads = m_kInitialLoadsX;
            xDataNeeded = m_kDataBuffXOffset + m_kLanes + m_kColumns - 1 + 1;
            yDataLoaded = m_kInitialLoadsY * m_kDataLoadVsize;
            yDataNeeded = yDataLoaded - ((ystart - (m_kColumns - 1)) % m_kDataLoadVsize);

            // Initial multiply
            accHighPolyphase = readCascade<TT_DATA, TT_COEFF, TP_DUAL_IP>(inInterface, accHighPolyphase);

            accHighPolyphase = initMacIntHb<TT_DATA, TT_COEFF, TP_DUAL_IP, TP_FIR_RANGE_LEN>(
                inInterface, accHighPolyphase, xbuff, xstart, ybuff, ystart, coe0, coeffstart, m_ctShift);

// In the operations loop, x and y buffs load at different times because y can start mid-splice.
#pragma unroll(GUARD_ZERO((m_kNumOps - 1)))
            for (int op = 1; op < m_kNumOps; ++op) {
                xDataNeeded += m_kColumns;
                yDataNeeded += m_kColumns;
                if (xDataNeeded > xDataLoaded) {
                    xReadData = window_readincr_128b<TT_DATA>(inWindow);
                    xbuff.val = upd_v(xbuff.val, xNumDataLoads % m_kDataLoadsInReg, xReadData.val);
                    xNumDataLoads++;
                    xDataLoaded += m_kDataLoadVsize;
                }
                if (yDataNeeded > yDataLoaded) {
                    yReadData = window_readdecr_128b<TT_DATA>(yinWindow);
                    ybuff.val = upd_v(ybuff.val, ySplice % m_kDataLoadsInReg, yReadData.val);
                    ySplice--;
                    yNumDataLoads++;
                    yDataLoaded += m_kDataLoadVsize;
                }
                xstart += m_kColumns;
                ystart -= m_kColumns;
                coeffstart += m_kColumns;
                // m_kCoeffRegVsize indicates number of coeffs that can fit in zbuff
                // for a given operation, we may need to load a new 256b chunk of coeffs
                if (op % (m_kCoeffRegVsize / m_kColumns) == 0) {
                    coe0 = *coeff++;
                }
                accHighPolyphase = macSym2buffIntHb<TT_DATA, TT_COEFF, TP_FIR_RANGE_LEN, TP_UPSHIFT_CT>(
                    accHighPolyphase, xbuff, xstart, ybuff, ystart, coe0, coeffstart, m_ctShift);
            }
            writeCascade<TT_DATA, TT_COEFF>(outInterface, accHighPolyphase);
            // The data for the centre tap is the same data as required for the last op of the top phase, so is already
            // loaded
            if (TP_KERNEL_POSITION + 1 == TP_CASC_LEN) {
                // check if enough data for center tap operation
                xDataNeeded += 1;
                if (xDataNeeded > xDataLoaded) {
                    xReadData = window_readincr_128b<TT_DATA>(inWindow);
                    xbuff.val = upd_v(xbuff.val, xNumDataLoads % m_kDataLoadsInReg, xReadData.val);
                    xNumDataLoads++;
                    xDataLoaded += m_kDataLoadVsize;
                }
                // Only last kernel in cascade chain has a non-zero ct coeff.
                accLowPolyphase =
                    mulCentreTap2buffIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT>(xbuff, m_kCentreTapDataCol, ctCoeff);
            }
            writeWindow<TT_DATA, TT_COEFF, TP_NUM_OUTPUTS>(outInterface, accHighPolyphase, accLowPolyphase, TP_SHIFT);
            // writeOutputIntHb(outWindow, accLowPolyphase, accHighPolyphase, TP_SHIFT);
            window_decr(inWindow, xNumDataLoads * m_kDataLoadVsize -
                                      m_kLanes); // return read pointer to start of next chunk of window.
            window_incr(yinWindow, (yNumDataLoads)*m_kDataLoadVsize +
                                       m_kLanes); // return read pointer to start of next chunk of window.
        }
};

// -------------------------------------------------- filter1buff -------------------------------------------------- //
// The filter1buff variant of this function is for cases where all the data samples required may be loaded into
// the sbuff such that the single buffer may be used for both xbuff and ybuff.
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
          unsigned int TP_NUM_OUTPUTS,
          unsigned int TP_UPSHIFT_CT>
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
                              TP_NUM_OUTPUTS,
                              TP_UPSHIFT_CT>::filter1buff(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                                                          T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    set_rnd(TP_RND);
    set_sat();
    // const unsigned int     kSamplesInWindow = window_size(inWindow);  //number of samples in window
    T_buff_256b<TT_COEFF>* restrict coeff = (T_buff_256b<TT_COEFF>*)m_internalTapsFSA;
    T_buff_256b<TT_COEFF> coe0, coe1; // register for coeff values.
    T_buff_256b<TT_COEFF>* restrict ctCoeffptr =
        (T_buff_256b<TT_COEFF>*)m_phaseTwoTap; // register for centre tap coeff value.
    T_buff_256b<TT_COEFF> ctCoeff = *ctCoeffptr;
    T_buff_256b<TT_DATA> readData;
    T_buff_128b<TT_DATA> readData_128;
    T_buff_1024b<TT_DATA> sbuff;
    T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> accHighPolyphase, accLowPolyphase;
    unsigned int xDataLoaded, xDataNeeded, numDataLoads;
    unsigned int xstart, ystart, coeffstart;
    unsigned int xdatum, ydatum; // point within buffer for each output data point's first input
    unsigned int loadSplice;
    unsigned int dataLoaded, dataNeeded;
    unsigned int ctPos;

    input_window<TT_DATA>* restrict inWindow = inInterface.inWindow;

    window_incr(inWindow, m_kXDataLoadInitOffset); // move input data pointer past the margin padding
// preamble, load data from window into register m_kInitialLoads1buff-1 times
#pragma unroll(m_kInitialLoads1buff - 1)
    for (unsigned int initLoads = 0; initLoads < m_kInitialLoads1buff - 1; ++initLoads) {
        readData = window_readincr_256b<TT_DATA>(inWindow);
        sbuff.val = upd_w(sbuff.val, initLoads % m_kDataLoadsInReg, readData.val);
    }
    xdatum = m_kDataBuffXOffset;
    ydatum = m_kDataBuffXOffset + (TP_FIR_LEN) / 2 - kInterpolateFactor * m_kFirRangeOffset; //

    coeff = ((T_buff_256b<TT_COEFF>*)&m_internalTapsFSA);
    coe0 = *coeff++;
    // Load more coeffs. Is this needed at all?
    if (m_kNumOps >= m_kCoeffRegVsize / m_kColumns) {
        coe1 = *coeff;
    }
    // load half of the data final m_kInitialLoads1buff load,
    // while the other half will be done at the start of strobe loop.
    if (m_kDataLoadsInReg == kUse128bitLoads) {
        readData_128 = window_readincr_128b<TT_DATA>(inWindow);
        sbuff.val = upd_v(sbuff.val, ((2 * (m_kInitialLoads1buff - 1)) % (m_kDataLoadsInReg)), readData_128.val);
    }

    // Loop through window outputting a vector of values each time.
    // Lsize is therefore a ratio of window size to output vector size
    for (unsigned i = 0; i < m_kLsize / (m_kDataLoadsInReg); i++)
        // Allow optimizations in the kernel compilation for this loop
        chess_prepare_for_pipelining chess_loop_range(m_kLsize / (m_kDataLoadsInReg), ) {
#pragma unroll(m_kDataLoadsInReg)
            for (unsigned strobe = 0; strobe < (m_kDataLoadsInReg); strobe++) {
                if (m_kDataLoadsInReg == kUse128bitLoads) {
                    readData_128 = window_readincr_128b<TT_DATA>(inWindow);
                    sbuff.val = upd_v(sbuff.val, ((2 * (m_kInitialLoads1buff - 1) + 1 + strobe) % (m_kDataLoadsInReg)),
                                      readData_128.val);
                } else {
                    readData = window_readincr_256b<TT_DATA>(inWindow);
                    sbuff.val =
                        upd_w(sbuff.val, (((m_kInitialLoads1buff - 1) + strobe) % (m_kDataLoadsInReg)), readData.val);
                }

                // incremental load data from window into register
                // The full FIR length may be more than one load, but most of the data
                // required overlaps with the data used in the last iteration of i
                xstart = xdatum + strobe * m_kLanes;
                ystart = ydatum + strobe * m_kLanes;
                ctPos = xdatum + strobe * m_kLanes + ((TP_FIR_RANGE_LEN) / 2 - (TP_FIR_RANGE_LEN) / 4);
                coeffstart = 0;
                accHighPolyphase = readCascade<TT_DATA, TT_COEFF, TP_DUAL_IP>(inInterface, accHighPolyphase);

                accHighPolyphase = initMacIntHb<TT_DATA, TT_COEFF, TP_DUAL_IP, TP_FIR_RANGE_LEN>(
                    inInterface, accHighPolyphase, sbuff, xstart, ystart, coe0, coeffstart, m_ctShift);

#pragma unroll(GUARD_ZERO((m_kNumOps - 1)))
                for (int op = 1; op < m_kNumOps; ++op) {
                    xstart += m_kColumns;
                    ystart -= m_kColumns;
                    coeffstart += m_kColumns;
                    accHighPolyphase = macSym1buffIntHb<TT_DATA, TT_COEFF>(
                        accHighPolyphase, sbuff, xstart, ystart, op >= m_kCoeffRegVsize / m_kColumns ? coe1 : coe0,
                        coeffstart); // common with single rate
                }

                // Write cascade. Do nothing if cascade not present.
                writeCascade<TT_DATA, TT_COEFF>(outInterface, accHighPolyphase);
                if (TP_KERNEL_POSITION + 1 == TP_CASC_LEN) {
                    // Only last kernel in cascade chain has a non-zero ct coeff.
                    // TODO: hide the condition in template/argument
                    accLowPolyphase = mulCentreTap1buffIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT>(sbuff, ctPos, ctCoeff);
                }
                writeWindow<TT_DATA, TT_COEFF, TP_NUM_OUTPUTS>(outInterface, accHighPolyphase, accLowPolyphase,
                                                               TP_SHIFT);
            }
        }
// Deal with remaining operations. Probably a cleaner way to do this.
// Just a copy-paste.
// Guard against unroll 0.
#pragma unroll(GUARD_ZERO((m_kLsize % (m_kDataLoadsInReg))))
    for (unsigned strobe = 0; strobe < (m_kLsize % (m_kDataLoadsInReg)); strobe++) {
        if (m_kDataLoadsInReg == kUse128bitLoads) {
            readData_128 = window_readincr_128b<TT_DATA>(inWindow);
            sbuff.val = upd_v(sbuff.val, ((2 * (m_kInitialLoads1buff - 1) + 1 + strobe) % (m_kDataLoadsInReg)),
                              readData_128.val);
        } else {
            readData = window_readincr_256b<TT_DATA>(inWindow);
            sbuff.val = upd_w(sbuff.val, (((m_kInitialLoads1buff - 1) + strobe) % (m_kDataLoadsInReg)), readData.val);
        }

        // incremental load data from window into register
        // The full FIR length may be more than one load, but most of the data
        // required overlaps with the data used in the last iteration of i
        xstart = xdatum + strobe * m_kLanes;
        ystart = ydatum + strobe * m_kLanes;
        ctPos = xdatum + strobe * m_kLanes +
                ((TP_FIR_RANGE_LEN) / kInterpolateFactor - ((TP_FIR_RANGE_LEN) / kInterpolateFactor) / kSymmetryFactor);
        coeffstart = 0;
        accHighPolyphase = readCascade<TT_DATA, TT_COEFF, TP_DUAL_IP>(inInterface, accHighPolyphase);

        accHighPolyphase = initMacIntHb<TT_DATA, TT_COEFF, TP_DUAL_IP, TP_FIR_RANGE_LEN>(
            inInterface, accHighPolyphase, sbuff, xstart, ystart, coe0, coeffstart, m_ctShift);

#pragma unroll(GUARD_ZERO((m_kNumOps - 1)))
        for (int op = 1; op < m_kNumOps; ++op) {
            xstart += m_kColumns;
            ystart -= m_kColumns;
            coeffstart += m_kColumns;
            accHighPolyphase = macSym1buffIntHb<TT_DATA, TT_COEFF>(accHighPolyphase, sbuff, xstart, ystart,
                                                                   op >= m_kCoeffRegVsize / m_kColumns ? coe1 : coe0,
                                                                   coeffstart); // common with single rate
        }

        // Write cascade. Do nothing if cascade not present.
        writeCascade<TT_DATA, TT_COEFF>(outInterface, accHighPolyphase);
        if (TP_KERNEL_POSITION + 1 == TP_CASC_LEN) {
            // Only last kernel in cascade chain has a non-zero ct coeff.
            // TODO: hide the condition in template/argument
            accLowPolyphase = mulCentreTap1buffIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT>(sbuff, ctPos, ctCoeff);
        }
        writeWindow<TT_DATA, TT_COEFF, TP_NUM_OUTPUTS, TP_UPSHIFT_CT>(outInterface, accHighPolyphase, accLowPolyphase,
                                                                      TP_SHIFT);
    }
};

// --------------------------------------------------- filter2buffZigZag
// -------------------------------------------------- //
// The filter2buffZigZag variant of this function is for cases where 2 separate buffers must be used, one for forward
// data
// and the other for reverse data.
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
          unsigned int TP_NUM_OUTPUTS,
          unsigned int TP_UPSHIFT_CT>
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
                              TP_NUM_OUTPUTS,
                              TP_UPSHIFT_CT>::filter2buffZigZag(T_inputIF<TP_CASC_IN, TT_DATA, TP_DUAL_IP> inInterface,
                                                                T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    set_rnd(TP_RND);
    set_sat();

    // Pointers to coefficient storage and explicit registers to hold values
    T_buff_256b<TT_COEFF>* restrict coeff = (T_buff_256b<TT_COEFF>*)m_internalTapsFSA;
    T_buff_256b<TT_COEFF> coe0; // register for coeff values.
    T_buff_256b<TT_COEFF>* restrict ctCoeffptr =
        (T_buff_256b<TT_COEFF>*)m_phaseTwoTap; // register for centre tap coeff value.
    T_buff_256b<TT_COEFF> ctCoeff = *ctCoeffptr;
    T_buff_128b<TT_DATA> xReadData;
    T_buff_128b<TT_DATA> yReadData;
    T_buff_512b<TT_DATA> xbuff;
    T_buff_512b<TT_DATA> ybuff;
    T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> accHighPolyphase, accLowPolyphase;
    unsigned int xDataNeeded, xDataLoaded, xNumDataLoads;
    unsigned int yDataNeeded, yDataLoaded, yNumDataLoads;
    unsigned int coeffstart;
    int xstart, xRevinWinOffset, xDataLoads, xDataLoadCap, xSplice;
    int ystart, yRevinWinOffset, yDataLoads, yDataLoadCap, ySplice;
    int xinWinRet, xRevinWinRet;
    int yinWinRet, yRevinWinRet;

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
    input_window<TT_DATA> yRevinWindowActual;
    input_window<TT_DATA>* restrict yRevinWindow;
    yRevinWindow = &yRevinWindowActual;
    window_copy(yRevinWindow, yinWindow);

    xDataLoads = m_kNumOps * m_kColumns - 3 * m_kDataLoadVsize; // needed data - valid xbuff parts from prev stage

    xRevinWinOffset = m_kXDataLoadInitOffset + xDataLoads + (m_kInitialLoadsX - 2) * m_kDataLoadVsize;
    window_incr(inRevWindow, xRevinWinOffset);

    yDataLoads = (((m_kNumOps)*m_kColumns) / m_kDataLoadVsize); // Calculate Data loads
    yDataLoadCap = (yDataLoads < m_kDataLoadsInReg
                        ? yDataLoads
                        : m_kDataLoadsInReg -
                              1); // valid databuffer parts - reused at the Zag stage, capped at m_kDataLoadsInReg-1
    yRevinWinOffset = m_kYDataLoadInitOffset - CEIL((m_kNumOps)*m_kColumns, m_kDataLoadVsize) +
                      yDataLoadCap * m_kDataLoadVsize + m_kDataLoadVsize;

    // Starting chunk - reverse loads + realignment to xwindowPointer
    window_incr(yRevinWindow, yRevinWinOffset);

    window_incr(inWindow, m_kXDataLoadInitOffset); // move input data pointer past the margin padding
    window_incr(yinWindow, m_kYDataLoadInitOffset);

    xNumDataLoads = 0;
    xDataLoaded = 0;
    yNumDataLoads = 0;
    yDataLoaded = 0;
// preload xdata from window into xbuff register
#pragma unroll(m_kInitialLoadsX)
    for (unsigned int initLoads = 0; initLoads < m_kInitialLoadsX; ++initLoads) {
        xReadData = window_readincr_128b<TT_DATA>(inWindow);
        xbuff.val = upd_v(xbuff.val, xNumDataLoads % m_kDataLoadsInReg, xReadData.val);
        xNumDataLoads++;
        xDataLoaded += m_kDataLoadVsize;
    }

    // preload ydata from window into ybuff register
    ySplice = m_kySpliceStart;
#pragma unroll(m_kInitialLoadsY)
    for (unsigned int initLoads = 0; initLoads < m_kInitialLoadsY; ++initLoads) {
        yReadData = window_readdecr_128b<TT_DATA>(yinWindow);
        ybuff.val = upd_v(ybuff.val, ySplice % m_kDataLoadsInReg, yReadData.val);
        ySplice--;
    }

    for (unsigned i = 0; i < m_kLsize / (2 * m_kRepeatFactor); i++)
        chess_prepare_for_pipelining chess_loop_range(m_kLsize / (2 * m_kRepeatFactor), ) {
            // Coeff is the pointer to the m_internalTapsFSA array, whereas coe0 is the register which holds a cache of
            // that.
            coeff = ((T_buff_256b<TT_COEFF>*)&m_internalTapsFSA);
            coe0 = *coeff++;

            xstart = m_kDataBuffXOffset;
            ystart = m_kyStart;

            coeffstart = 0;
            xNumDataLoads = m_kInitialLoadsX;
            xDataLoaded = m_kInitialLoadsX * m_kDataLoadVsize;
            xDataNeeded = m_kDataBuffXOffset + m_kLanes + m_kColumns - 1;

            yNumDataLoads = m_kInitialLoadsY;
            ySplice = (m_kDataLoadsInReg + m_kySpliceStart - m_kInitialLoadsY) % m_kDataLoadsInReg;
            yDataLoaded = m_kInitialLoadsY * m_kDataLoadVsize;
            yDataNeeded = yDataLoaded - ((ystart - (m_kColumns - 1)) % m_kDataLoadVsize);

// Unroll this so that we see constant splices.
#pragma unroll(m_kRepeatFactor)
            for (int dataLoadPhase = 0; dataLoadPhase < m_kRepeatFactor; dataLoadPhase++) {
                ////////////////// Forward /////////////////////

                // Read cascade
                accHighPolyphase = readCascade<TT_DATA, TT_COEFF, TP_DUAL_IP>(inInterface, accHighPolyphase);

                // Initial multiply
                accHighPolyphase = initMacIntHb<TT_DATA, TT_COEFF, TP_DUAL_IP, TP_FIR_RANGE_LEN>(
                    inInterface, accHighPolyphase, xbuff, xstart, ybuff, ystart, coe0, coeffstart, m_ctShift);

// In the operations loop, x and y buffs load at different times because y can start mid-splice.
#pragma unroll(GUARD_ZERO((m_kNumOps - 1)))
                for (int op = 1; op < m_kNumOps; ++op) {
                    xDataNeeded += m_kColumns;
                    yDataNeeded += m_kColumns;
                    if (xDataNeeded > xDataLoaded) {
                        xReadData = window_readincr_128b<TT_DATA>(inWindow);
                        xbuff.val = upd_v(xbuff.val, xNumDataLoads % m_kDataLoadsInReg, xReadData.val);
                        xNumDataLoads++;
                        xDataLoaded += m_kDataLoadVsize;
                    }
                    if (yDataNeeded > yDataLoaded) {
                        yReadData = window_readdecr_128b<TT_DATA>(yinWindow);
                        ybuff.val = upd_v(ybuff.val, ySplice % m_kDataLoadsInReg, yReadData.val);
                        ySplice = (ySplice - 1 + m_kDataLoadsInReg) % m_kDataLoadsInReg;
                        yNumDataLoads++;
                        yDataLoaded += m_kDataLoadVsize;
                    }
                    xstart += m_kColumns;
                    ystart -= m_kColumns;
                    coeffstart += m_kColumns;
                    // m_kCoeffRegVsize indicates number of coeffs that can fit in zbuff
                    // for a given operation, we may need to load a new 256b chunk of coeffs
                    if (op % (m_kCoeffRegVsize / m_kColumns) == 0) {
                        if (op / (m_kCoeffRegVsize / m_kColumns) == m_kNumOps / (m_kCoeffRegVsize / m_kColumns)) {
                            // last update, load and prepare for next stage
                            coe0 = *coeff--;
                        } else {
                            coe0 = *coeff++;
                        }
                    }
                    accHighPolyphase = macSym2buffIntHb<TT_DATA, TT_COEFF, TP_FIR_RANGE_LEN, TP_UPSHIFT_CT>(
                        accHighPolyphase, xbuff, xstart, ybuff, ystart, coe0, coeffstart, m_ctShift);
                }
                writeCascade<TT_DATA, TT_COEFF>(outInterface, accHighPolyphase); // TODO: extract only the low half. UCT
                                                                                 // contents are only calculated on last
                                                                                 // kernel.

                writeWindow<TT_DATA, TT_COEFF, TP_NUM_OUTPUTS>(outInterface, accHighPolyphase, accLowPolyphase,
                                                               TP_SHIFT);

                ////////////////// Prepare for Reverse /////////////////////

                xDataNeeded += m_kDataLoadVsize;
                if (xDataNeeded > xDataLoaded) {
                    xReadData = window_readincr_128b<TT_DATA>(inWindow);
                    xbuff.val = upd_v(xbuff.val, xNumDataLoads % m_kDataLoadsInReg, xReadData.val);
                    xNumDataLoads++;
                    xDataLoaded += m_kDataLoadVsize;
                }

                yDataNeeded = m_kLanes + m_kColumns - 1;
                xDataLoadCap = xNumDataLoads < m_kDataLoadsInReg
                                   ? xNumDataLoads
                                   : m_kDataLoadsInReg; // xDataLoaded = xNumDataLoads*m_kDataLoadVsize, unless all
                                                        // register segments have been written, then cap at max
                yDataLoadCap = yNumDataLoads + 2 * dataLoadPhase * m_kDataLoadVsize < m_kDataLoadsInReg
                                   ? yNumDataLoads
                                   : m_kDataLoadsInReg; // yDataLoaded = yNumDataLoads*m_kDataLoadVsize, unless all
                                                        // register segments have been written, then cap at max
                yDataLoaded = yDataLoadCap * m_kDataLoadVsize - m_kDataLoadVsize;

                xDataNeeded = m_kDataBuffXOffset + m_kLanes + m_kColumns - 1;
                xDataLoaded = xDataLoadCap * m_kDataLoadVsize; //

                // Final xystart values
                xstart += m_kColumns;
                ystart += m_kColumns;

                xSplice = (xNumDataLoads - 1) % m_kDataLoadsInReg; // First loaded chunk not required for X's Zag phase

                ySplice++; // last loaded chunk not required for Y's Zag phase.

                int xZagDataNeeded =
                    (m_kNumOps - 1) * m_kColumns - (xDataLoadCap - m_kInitialLoadsX) * m_kDataLoadVsize; //
                int xZagDataLoads = xZagDataNeeded / m_kDataLoadVsize;                                   //
                xinWinRet =
                    xZagDataNeeded +
                    (3 - m_kInitialLoadsX) * dataLoadPhase *
                        m_kDataLoadVsize; // Revert by how much Zag will consume + the diff between Zig2 and init stage.
                xNumDataLoads = xNumDataLoads - xZagDataLoads;
                yinWinRet = (m_kNumOps - 1) * m_kColumns + 2 * dataLoadPhase * m_kDataLoadVsize;

                // Decrement window by how much has been loaded in Zig stage minus what will be reused after the Zag
                // stage
                window_decr(inWindow, xinWinRet); //
                // move pointer back to the start + prep for next iteration
                window_incr(yinWindow, yinWinRet);
                ////////////////// Reverse /////////////////////

                accHighPolyphase = readCascade<TT_DATA, TT_COEFF, TP_DUAL_IP>(inInterface, accHighPolyphase);

                // Initial multiply
                accHighPolyphase = initMacIntHb<TT_DATA, TT_COEFF, TP_DUAL_IP, TP_FIR_RANGE_LEN>(
                    inInterface, accHighPolyphase, xbuff, xstart, ybuff, ystart, coe0, coeffstart, m_ctShift);

// In the operations loop, x and y buffs load at different times because y can start mid-splice.
#pragma unroll(GUARD_ZERO((m_kNumOps - 1)))
                for (int op = 1; op < m_kNumOps; ++op) {
                    xDataNeeded += m_kColumns;
                    yDataNeeded += m_kColumns;
                    if (xDataNeeded > xDataLoaded) {
                        xReadData = window_readdecr_128b<TT_DATA>(inRevWindow);
                        xbuff.val = upd_v(xbuff.val, xSplice, xReadData.val);
                        xSplice =
                            (xSplice - 1 + m_kDataLoadsInReg) % m_kDataLoadsInReg; // wrapped around m_kDataLoadsInReg
                        xDataLoaded += m_kDataLoadVsize;
                    }
                    if (yDataNeeded > yDataLoaded) {
                        yReadData = window_readincr_128b<TT_DATA>(yRevinWindow);
                        ybuff.val = upd_v(ybuff.val, (ySplice) % m_kDataLoadsInReg, yReadData.val);
                        ySplice++;
                        yDataLoaded += m_kDataLoadVsize;
                    }
                    xstart -= m_kColumns;
                    ystart += m_kColumns;
                    coeffstart -= m_kColumns;
                    // m_kCoeffRegVsize indicates number of coeffs that can fit in zbuff
                    // for a given operation, we may need to load a new 256b chunk of coeffs
                    if ((coeffstart) % (m_kCoeffRegVsize) == (m_kCoeffRegVsize - m_kColumns)) {
                        if (coeffstart < m_kCoeffRegVsize) {
                            // last update, load and prepare for next stage
                            coe0 = *coeff++;
                        } else {
                            coe0 = *coeff--;
                        }
                    }
                    accHighPolyphase = macSym2buffIntHb<TT_DATA, TT_COEFF, TP_FIR_LEN, TP_UPSHIFT_CT>(
                        accHighPolyphase, xbuff, xstart, ybuff, ystart, coe0, coeffstart);
                }
                writeCascade<TT_DATA, TT_COEFF>(outInterface, accHighPolyphase);
                writeWindow<TT_DATA, TT_COEFF, TP_NUM_OUTPUTS>(outInterface, accHighPolyphase, accLowPolyphase,
                                                               TP_SHIFT);
                ////////////////// Prepare for Forward /////////////////////
                yDataNeeded += m_kDataLoadVsize;

                if (yDataNeeded > yDataLoaded) {
                    yReadData = window_readincr_128b<TT_DATA>(yRevinWindow);
                    ybuff.val = upd_v(ybuff.val, (ySplice) % m_kDataLoadsInReg, yReadData.val);
                    ySplice++;
                    yDataLoaded += m_kDataLoadVsize;
                }

                yNumDataLoads = m_kDataLoadsInReg; // All parts of buffer are filled

                xDataLoadCap = xNumDataLoads < m_kDataLoadsInReg ? xNumDataLoads : m_kDataLoadsInReg;
                xDataLoaded =
                    xDataLoadCap * m_kDataLoadVsize - m_kDataLoadVsize; // All parts of buffer are filled, apart from 1

                xstart = m_kDataBuffXOffset + 2 * m_kDataLoadVsize; // Prep for Zig stage
                ystart = m_kyStart + 2 * m_kDataLoadVsize;          // Prep for Zig stage
                coeffstart = 0;
                xDataNeeded = m_kDataBuffXOffset + m_kLanes + m_kColumns - 1;
                yDataNeeded = m_kLanes + m_kColumns - 1;
                yDataLoaded = m_kDataLoadsInReg * m_kDataLoadVsize; // All parts of buffer are filled
                ySplice--;                                          // last loaded chunk not required for Y's Zag phase.

                xRevinWinRet =
                    xZagDataNeeded + 2 * m_kDataLoadVsize; // Revert Zag loads + 2 more to prepare for next Zag start
                yRevinWinRet = (m_kNumOps)*m_kColumns - 3 * m_kDataLoadVsize;

                window_incr(inRevWindow, xRevinWinRet);  //
                window_decr(yRevinWindow, yRevinWinRet); //
            }
        }
};

//----------------------------------------------------------------------------
// Start of Cascade-layer class body and specializations

// FIR filter function overloaded with cascade interface variations
// This is a (default) specialization of the main class for when there is only one kernel for the whole filter.
// single output, static coeffs, single output
//-----------------------------------------------------------------------------------------------------
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
          unsigned int TP_NUM_OUTPUTS,
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        TP_NUM_OUTPUTS,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* restrict inWindow, // T_inputIF<CASC_IN_FALSE,
                                                                                         // TT_DATA, TP_DUAL_IP>
                                                                                         // inInterface,
                                               output_window<TT_DATA>* restrict outWindow) {
    T_inputIF<CASC_IN_FALSE, TT_DATA, TP_DUAL_IP> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outWindow = outWindow;
    this->filterKernel(inInterface, outInterface);
};

// specialization for one kernel for the whole filter, but with single input static coeffs, dual output.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        2,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* restrict inWindow,
                                               output_window<TT_DATA>* restrict outWindow,
                                               output_window<TT_DATA>* restrict outWindow2) {
    T_inputIF<CASC_IN_FALSE, TT_DATA, DUAL_IP_SINGLE> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outWindow = outWindow;
    outInterface.outWindow2 = outWindow2;
    this->filterKernel(inInterface, outInterface);
};

// specialization for one kernel for the whole filter, but with dual inputs static coeffs, single output.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        1,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* restrict inWindow,
                                               input_window<TT_DATA>* restrict inWindowReverse,
                                               output_window<TT_DATA>* restrict outWindow) {
    T_inputIF<CASC_IN_FALSE, TT_DATA, DUAL_IP_DUAL> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inWindowReverse = inWindowReverse;
    outInterface.outWindow = outWindow;
    this->filterKernel(inInterface, outInterface);
};

// specialization for one kernel for the whole filter, but with dual inputs static coeffs, dual output.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        2,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* restrict inWindow,
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

// specialization for one kernel for the whole filter, but with single input and reload, single output.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        1,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* restrict inWindow,
                                               output_window<TT_DATA>* restrict outWindow,
                                               const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / 4 + 1]) {
    T_inputIF<CASC_IN_FALSE, TT_DATA, DUAL_IP_SINGLE> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outWindow = outWindow;
    this->filterKernel(inInterface, outInterface, inTaps);
};

// specialization for one kernel for the whole filter, but with single input and reload, dual output.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        2,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* restrict inWindow,
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

// specialization for one kernel for the whole filter, but with dual inputs and reload, single output.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        1,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* restrict inWindow,
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

// specialization for one kernel for the whole filter, but with dual inputs and reload, dual output.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        2,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* restrict inWindow,
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

// FIR filter function overloaded with cascade interface variations
// This is a specialization of the main class for the final kernel in a cascade chain, with single input and no
// coefficient reloads, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        DUAL_IP_SINGLE,
                        USE_COEFF_RELOAD_FALSE,
                        1,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* restrict inWindow,
                                               input_stream_cacc48* inCascade,
                                               output_window<TT_DATA>* restrict outWindow) {
    T_inputIF<CASC_IN_TRUE, TT_DATA, DUAL_IP_SINGLE> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inCascade = inCascade;
    outInterface.outWindow = outWindow;
    this->filterKernel(inInterface, outInterface);
};

// This is a specialization of the main class for the final kernel in a cascade chain, with single input and no
// coefficient reloads, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        DUAL_IP_SINGLE,
                        USE_COEFF_RELOAD_FALSE,
                        2,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* restrict inWindow,
                                               input_stream_cacc48* inCascade,
                                               output_window<TT_DATA>* restrict outWindow,
                                               output_window<TT_DATA>* restrict outWindow2) {
    T_inputIF<CASC_IN_TRUE, TT_DATA, DUAL_IP_SINGLE> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inCascade = inCascade;
    outInterface.outWindow = outWindow;
    outInterface.outWindow2 = outWindow2;
    this->filterKernel(inInterface, outInterface);
};

// This is a specialization of the main class for the final kernel in a cascade chain, with dual inputs and static
// coefficient, single output.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        1,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* restrict inWindow,
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

// This is a specialization of the main class for the final kernel in a cascade chain, with dual inputs and static
// coefficient, dual output.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        2,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* restrict inWindow,
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

// This is a specialization of the main class for the final kernel in a cascade chain, with single input and coefficient
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
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        DUAL_IP_SINGLE,
                        USE_COEFF_RELOAD_TRUE,
                        1,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* restrict inWindow,
                                               input_stream_cacc48* inCascade,
                                               output_window<TT_DATA>* restrict outWindow) {
    T_inputIF<CASC_IN_TRUE, TT_DATA, DUAL_IP_SINGLE> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inCascade = inCascade;
    outInterface.outWindow = outWindow;
    this->filterKernelRtp(inInterface, outInterface);
};

// This is a specialization of the main class for the final kernel in a cascade chain, with single input and coefficient
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
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        DUAL_IP_SINGLE,
                        USE_COEFF_RELOAD_TRUE,
                        2,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* restrict inWindow,
                                               input_stream_cacc48* inCascade,
                                               output_window<TT_DATA>* restrict outWindow,
                                               output_window<TT_DATA>* restrict outWindow2) {
    T_inputIF<CASC_IN_TRUE, TT_DATA, DUAL_IP_SINGLE> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inCascade = inCascade;
    outInterface.outWindow = outWindow;
    outInterface.outWindow2 = outWindow2;
    this->filterKernelRtp(inInterface, outInterface);
};

// This is a specialization of the main class for the final kernel in a cascade chain, with dual inputs and coefficient
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
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        1,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* restrict inWindow,
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

// This is a specialization of the main class for the final kernel in a cascade chain, with dual inputs and coefficient
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
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        2,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* restrict inWindow,
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

// FIR filter function overloaded with cascade interface variations
// This is a specialization of the main class for the first kernel in a cascade chain, with single input and no
// coefficient reloads.
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        DUAL_IP_SINGLE,
                        USE_COEFF_RELOAD_FALSE,
                        1,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* inWindow,
                                               output_stream_cacc48* outCascade,
                                               output_window<TT_DATA>* broadcastWindow) {
    T_inputIF<CASC_IN_FALSE, TT_DATA, DUAL_IP_SINGLE> inInterface;
    T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outCascade = outCascade;
    outInterface.broadcastWindow = broadcastWindow;
    this->filterKernel(inInterface, outInterface);
};

// This is a specialization of the main class for the first kernel in a cascade chain, with dual inputs and no
// coefficient reloads.
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        1,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* inWindow,
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

// This is a specialization of the main class for the first kernel in a cascade chain, with single input and coefficient
// reloads.
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        DUAL_IP_SINGLE,
                        USE_COEFF_RELOAD_TRUE,
                        1,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* inWindow,
                                               output_stream_cacc48* outCascade,
                                               output_window<TT_DATA>* broadcastWindow,
                                               const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / 4 + 1]) {
    T_inputIF<CASC_IN_FALSE, TT_DATA, DUAL_IP_SINGLE> inInterface;
    T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outCascade = outCascade;
    outInterface.broadcastWindow = broadcastWindow;
    this->filterKernel(inInterface, outInterface, inTaps);
};

// This is a specialization of the main class for the first kernel in a cascade chain, with dual inputs and coefficient
// reloads.
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        1,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* inWindow,
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

// FIR filter function overloaded with cascade interface variations
// This is a specialization of the main class for any kernel within a cascade chain, but neither first nor last, with
// single input and no coefficient reloads.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        DUAL_IP_SINGLE,
                        USE_COEFF_RELOAD_FALSE,
                        1,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* inWindow,
                                               input_stream_cacc48* inCascade,
                                               output_stream_cacc48* outCascade,
                                               output_window<TT_DATA>* broadcastWindow) {
    T_inputIF<CASC_IN_TRUE, TT_DATA, DUAL_IP_SINGLE> inInterface;
    T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inCascade = inCascade;
    outInterface.outCascade = outCascade;
    outInterface.broadcastWindow = broadcastWindow;
    this->filterKernel(inInterface, outInterface);
};

// This is a specialization of the main class for any kernel within a cascade chain, but neither first nor last, with
// dual inputs and no coefficient reload.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        1,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* inWindow,
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
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        DUAL_IP_SINGLE,
                        USE_COEFF_RELOAD_TRUE,
                        1,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* inWindow,
                                               input_stream_cacc48* inCascade,
                                               output_stream_cacc48* outCascade,
                                               output_window<TT_DATA>* broadcastWindow) {
    T_inputIF<CASC_IN_TRUE, TT_DATA, DUAL_IP_SINGLE> inInterface;
    T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    inInterface.inCascade = inCascade;
    outInterface.outCascade = outCascade;
    outInterface.broadcastWindow = broadcastWindow;
    this->filterKernelRtp(inInterface, outInterface);
};

// This is a specialization of the main class for any kernel within a cascade chain, but neither first nor last, with
// dual inputs and coefficient reload.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN,
          unsigned int TP_UPSHIFT_CT>
void fir_interpolate_hb<TT_DATA,
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
                        1,
                        TP_UPSHIFT_CT>::filter(input_window<TT_DATA>* inWindow,
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
