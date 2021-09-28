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
Single Rate Symmetrical FIR kernal code.
This file captures the body of run-time code for the kernal class.

Coding conventions
  TT_      template type suffix
  TP_      template parameter suffix
*/
#pragma once

#define __NEW_WINDOW_H__ 1

#define __AIEARCH__ 1
#define __AIENGINE__ 1
#define __AIE_API_USE_NATIVE_1024B_VECTOR__
#include "aie_api/aie_adf.hpp"

#include <adf.h>
#include "fir_sr_sym.hpp"
#include "kernel_api_utils.hpp"
#include "fir_sr_sym_utils.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace sr_sym {
// FIR function
//----------------------------------------------------------------------------------------------------------------------
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
          unsigned int TP_USE_COEFF_RELOAD,
          unsigned int TP_NUM_OUTPUTS

          >
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
                              TP_USE_COEFF_RELOAD,
                              TP_NUM_OUTPUTS>::filterKernel(T_inputIF<TP_CASC_IN, TT_DATA> inInterface,
                                                            T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    windowBroadcast<TT_DATA, TP_INPUT_WINDOW_VSIZE + fnFirMargin<TP_FIR_LEN, TT_DATA>()>(inInterface, outInterface);

    filterSelectArch(inInterface, outInterface);
};

// FIR function
//----------------------------------------------------------------------------------------------------------------------
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
                              TP_USE_COEFF_RELOAD,
                              TP_NUM_OUTPUTS>::filterKernel(T_inputIF<TP_CASC_IN, TT_DATA> inInterface,
                                                            T_outputIF<TP_CASC_OUT, TT_DATA> outInterface,
                                                            const TT_COEFF (
                                                                &inTaps)[(TP_FIR_LEN + 1) / kSymmetryFactor]) {
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
//----------------------------------------------------------------------------------------------------------------------
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
                              TP_USE_COEFF_RELOAD,
                              TP_NUM_OUTPUTS>::filterKernelRtp(T_inputIF<TP_CASC_IN, TT_DATA> inInterface,
                                                               T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    windowBroadcast<TT_DATA, TP_INPUT_WINDOW_VSIZE + fnFirMargin<TP_FIR_LEN, TT_DATA>()>(inInterface, outInterface);

    m_coeffnEq = getRtpTrigger(); // 0 - equal, 1 - not equal

    sendRtpTrigger(m_coeffnEq, outInterface);
    if (m_coeffnEq) { // Coefficients have changed
        bufferReload<TT_DATA, TT_COEFF, (TP_FIR_LEN + 1) / kSymmetryFactor>(inInterface, m_oldInTaps, outInterface);
        firReload(m_oldInTaps);
    }
    filterSelectArch(inInterface, outInterface);
};

// FIR function
//----------------------------------------------------------------------------------------------------------------------
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
                              TP_USE_COEFF_RELOAD,
                              TP_NUM_OUTPUTS>::filterSelectArch(T_inputIF<TP_CASC_IN, TT_DATA> inInterface,
                                                                T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    windowAcquire(inInterface);
    // chess_memory_fence();

    // Two possible architectures depending on size of data/coef types & fir_len
    // Using a single data buffer for x and y (forward & reverse) or seperate
    if
        constexpr(m_kArch == kArch1Buff) { filterKernel1buff(inInterface, outInterface); }
    else {
        filterKernel2buff(inInterface, outInterface);
    }
    windowRelease(inInterface);
};

// 1buff architecture.
// Used when data samples required for FIR calculation fit fully into 1024-bit input vector register.
// Architecture is characterized by the usage of single 1024-bit buffer
//----------------------------------------------------------------------------------------------------------------------
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
                              TP_USE_COEFF_RELOAD,
                              TP_NUM_OUTPUTS>::filterKernel1buff(T_inputIF<TP_CASC_IN, TT_DATA> inInterface,
                                                                 T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    set_rnd(TP_RND);
    set_sat();

    T_buff_256b<TT_COEFF>* restrict coeff = (T_buff_256b<TT_COEFF>*)m_internalTaps;
    T_buff_256b<TT_COEFF> coe0, coe1; // register for coeff values.
    // T_buff_1024b<TT_DATA>           sbuff = null_buff_1024b<TT_DATA>();
    T_buff_1024b<TT_DATA> sbuff;
    T_accSym<TT_DATA, TT_COEFF> acc;
    T_outValSym<TT_DATA, TT_COEFF> outVal;
    T_buff_256b<TT_DATA> readDataS;
    T_buff_256b<TT_DATA> readDataT;
    T_buff_128b<TT_DATA> readDataV;
    unsigned int sNumDataLoads = 0;
    unsigned int sDataNeeded = 0;
    unsigned int sDataLoaded = 0;
    unsigned int xstart = 0;
    unsigned int ystart = 0;

    input_window<TT_DATA>* restrict inWindow = inInterface.inWindow;

    // Move data pointer away from data consumed by previous cascades
    // Move only by  multiples of 128bit.
    window_incr(inWindow, m_kSDataLoadInitOffset); // Cascade phase remainder goes to m_kSBuffXOffset

#pragma unroll(m_kInitialLoads - 1)
    for (int initLoads = 0; initLoads < m_kInitialLoads - 1; ++initLoads) {
        readDataS = window_readincr_256b<TT_DATA>(inWindow);
        sbuff.val = upd_w(sbuff.val, initLoads % m_kDataLoadsInReg, readDataS.val);
    }
    coeff = ((T_buff_256b<TT_COEFF>*)m_internalTaps);
    coe0 = *coeff++;
    // Only load more for int16/int16
    if (m_kFirLenCeilCols >= m_kCoeffRegVsize) {
        coe1 = *coeff;
    }
    for (unsigned i = 0; i < m_kLsize / m_kIncrRepeatFactor; i++)
        // Allow optimizations in the kernel compilation for this loop
        chess_prepare_for_pipelining chess_loop_range(m_kLsize / m_kIncrRepeatFactor, ) {
            sNumDataLoads = 0;
            sDataNeeded = 0; // m_kDataLoadVsize still needed due to only performing m_kInitialLoads-1
            sDataLoaded = 0;
// unroll m_kDataLoadsInReg times
#pragma unroll(m_kIncrRepeatFactor)
            for (unsigned dataLoadPhase = 0; dataLoadPhase < m_kIncrRepeatFactor; dataLoadPhase++) {
                if (sDataNeeded >= sDataLoaded) {
                    readDataS = window_readincr_256b<TT_DATA>(inWindow);
                    sbuff.val =
                        upd_w(sbuff.val, ((m_kInitialLoads - 1) + sNumDataLoads) % m_kDataLoadsInReg, readDataS.val);
                    sNumDataLoads++;
                    sDataLoaded += m_kDataLoadVsize;
                }
                sDataNeeded += m_kVOutSize;
                xstart = dataLoadPhase * m_kVOutSize + m_kSBuffXOffset;
                ystart = dataLoadPhase * m_kVOutSize + m_kYstartInitOffset;

                // Read cascade input. Do nothing if cascade input not present.
                acc = readCascade<TT_DATA, TT_COEFF>(inInterface, acc);
                // Init Vector operation. VMUL if cascade not present, otherwise VMAC
                if (m_kFirLenCeilCols != 0) {
                    acc = initMacSrSym<TT_DATA, TT_COEFF>(inInterface, acc, sbuff, xstart, ystart, coe0, 0);
                }

// The following loop is unrolled because this allows compile-time rather than run-time calculation
// of some of the variables within the loop hence increasing throughput.
#pragma unroll(GUARD_ZERO((m_kFirLenCeilCols / (m_kColumns) - 1)))
                // Operations loop. Op indicates the data index.
                for (int op = m_kColumns; op < m_kFirLenCeilCols; op += m_kColumns) {
                    // MAC operation.
                    acc = macSrSym(acc, sbuff.val, dataLoadPhase * m_kVOutSize + (op + m_kSBuffXOffset),
                                   dataLoadPhase * m_kVOutSize + m_kYstartInitOffset - (op % m_kDataRegVsize),
                                   op >= m_kCoeffRegVsize ? coe1.val : coe0.val, (op % m_kCoeffRegVsize));
                }

                // Center tap vector operation.
                acc = macSrSymCT<TP_FIR_RANGE_LEN % (kSymmetryFactor * m_kColumns)>(
                    acc, sbuff.val, (dataLoadPhase * m_kVOutSize + m_kFirLenCeilCols + m_kSBuffXOffset),
                    m_kFirLenCeilCols >= m_kCoeffRegVsize ? coe1.val : coe0.val,
                    (m_kFirLenCeilCols % m_kCoeffRegVsize));

                // Write cascade. Do nothing if cascade not present.
                writeCascade<TT_DATA, TT_COEFF>(outInterface, acc);

                outVal = shiftAndSaturate(acc, TP_SHIFT);
                writeWindow<TT_DATA, TT_COEFF, TP_NUM_OUTPUTS>(outInterface, outVal);
            }
        }
};

// 2buff architecture.
// Used when data samples required for FIR calculation do not fit fully into 1024-bit input vector register.
// Architecture is characterized by the usage of 2 512-bit buffers, one for forward data and one for reverse data.
//----------------------------------------------------------------------------------------------------------------------
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
                              TP_USE_COEFF_RELOAD,
                              TP_NUM_OUTPUTS>::filterKernel2buff(T_inputIF<TP_CASC_IN, TT_DATA> inInterface,
                                                                 T_outputIF<TP_CASC_OUT, TT_DATA> outInterface) {
    set_rnd(TP_RND);
    set_sat();

    T_buff_256b<TT_COEFF>* restrict coeff = (T_buff_256b<TT_COEFF>*)m_internalTaps;
    T_buff_256b<TT_COEFF> coe0; // register for coeff values.
    T_buff_512b<TT_DATA> sbuff = null_buff_512b<TT_DATA>();
    // T_buff_512b<TT_DATA>            sbuff;
    T_buff_512b<TT_DATA> tbuff = null_buff_512b<TT_DATA>();
    // T_buff_512b<TT_DATA>            tbuff;
    T_accSym<TT_DATA, TT_COEFF> acc;
    T_outValSym<TT_DATA, TT_COEFF> outVal;
    T_buff_256b<TT_DATA> readDataS;
    T_buff_256b<TT_DATA> readDataT;
    T_buff_128b<TT_DATA> readDataV;
    T_buff_128b<TT_DATA> readDataU;
    unsigned int sDataLoaded, sDataNeeded, sNumDataLoads, sVDataLoads = 0;
    unsigned int tDataLoaded, tDataNeeded, tNumDataLoads, tVDataLoads = 0;
    unsigned int sDataBuffSwap = 0;

    input_window<TT_DATA> temp_w;
    input_window<TT_DATA>* restrict inWindow = inInterface.inWindow;
    input_window<TT_DATA>* restrict cpWindow;
    cpWindow = &temp_w;
    window_copy(cpWindow, inWindow);

    // Move data pointer away from data consumed by previous cascades
    // Move only by  multiples of 128bit.
    window_incr(inWindow, m_kSDataLoadInitOffset); // Cascade phase remainder goes to m_kSBuffXOffset
    if (fnBufferUpdateScheme<TT_DATA, TT_COEFF>() == kPreLoadUsing128) {
        window_incr(cpWindow,
                    m_kTDataLoadInitOffset + m_kDataLoadVsize / 2); // Cascade phase remainder goes to m_kTBuffXOffset
    } else {
        window_incr(cpWindow, m_kTDataLoadInitOffset); // Cascade phase remainder goes to m_kTBuffXOffset
    }

    for (unsigned i = 0; i < m_kLsize; i++)
        // Allow optimizations in the kernel compilation for this loop
        chess_prepare_for_pipelining chess_loop_range(m_kLsize, m_kLsize) {
            coeff = ((T_buff_256b<TT_COEFF>*)m_internalTaps);
            coe0 = *coeff++;

            sNumDataLoads = 0;
            tNumDataLoads = 0;
            sDataLoaded = m_kSInitialLoads * m_kDataLoadVsize;
            tDataLoaded = m_kTInitialLoads * m_kDataLoadVsize;
            sVDataLoads = 0;
            tVDataLoads = 0;

            // Preamble, calculate and load data from window into register
            sDataNeeded = m_kSBuffXOffset + m_kVOutSize + m_kColumns - 1;
            tDataNeeded = m_kTBuffXOffset + m_kVOutSize + m_kColumns - 1;
            if (fnBufferUpdateScheme<TT_DATA, TT_COEFF>() == kPreLoadUsing128) {
#pragma unroll(2 * m_kSInitialLoads)
                for (int initLoads = 0; initLoads < 2 * m_kSInitialLoads; ++initLoads) {
                    readDataV = window_readincr_128b<TT_DATA>(inWindow);
                    sbuff.val = upd_v(sbuff.val, sVDataLoads, readDataV.val); // Update Sbuff
                    sVDataLoads++;
                }
#pragma unroll(2 * m_kTInitialLoads)
                for (int initLoads = 0; initLoads < 2 * m_kTInitialLoads; ++initLoads) {
                    readDataU = window_readdecr_128b<TT_DATA>(cpWindow);
                    tbuff.val =
                        upd_v(tbuff.val, (2 * m_kDataLoadsInReg - 1 - tVDataLoads), readDataU.val); // Update Sbuff
                    tVDataLoads++;
                }
            } else {
#pragma unroll(m_kSInitialLoads)
                for (int initLoads = 0; initLoads < m_kSInitialLoads; ++initLoads) {
                    readDataS = window_readincr_256b<TT_DATA>(inWindow);
                    sbuff.val = upd_w(sbuff.val, sNumDataLoads % m_kDataLoadsInReg, readDataS.val);
                    sNumDataLoads++;
                }
#pragma unroll(m_kTInitialLoads)
                for (int initLoads = 0; initLoads < m_kTInitialLoads; ++initLoads) {
                    readDataT = window_readdecr_256b<TT_DATA>(cpWindow);
                    tbuff.val =
                        upd_w(tbuff.val, (m_kDataLoadsInReg - 1 - tNumDataLoads) % m_kDataLoadsInReg, readDataT.val);
                    tNumDataLoads++;
                }
            }

            // Read cascade input. Do nothing if cascade input not present.
            acc = readCascade<TT_DATA, TT_COEFF>(inInterface, acc);
            // Init Vector operation. VMUL if cascade not present, otherwise VMAC
            if (m_kFirLenCeilCols != 0) {
                acc = initMacSrSym<TT_DATA, TT_COEFF>(inInterface, acc, sbuff, m_kSBuffXOffset, tbuff, m_kTStartOffset,
                                                      coe0, 0);
            }

// The following loop is unrolled because this allows compile-time rather than run-time calculation
// of some of the variables within the loop hence increasing throughput.
#pragma unroll(GUARD_ZERO((m_kFirLenCeilCols / (m_kColumns) - 1)))
            // Operations loop. Op indicates the data index.
            for (int op = m_kColumns; op < m_kFirLenCeilCols; op += m_kColumns) {
                sDataNeeded += m_kColumns;
                tDataNeeded += m_kColumns;
                // indices track the amount of data loaded into registers and consumed
                // from those registers so that the need to load more can be determined.
                // sbuff is for forward direction data.
                // tbuff is for reverse direction data.

                if (fnBufferUpdateScheme<TT_DATA, TT_COEFF>() == kPreLoadUsing256) {
                    // kPreLoadUsing256 - uses upd_w - 256bit loads and  schedules load early, to avoid memory
                    // conflicts.
                    // To be used with data/coeff combo utilizing single column MUL/MAC intrinsics, i.e. coeff type
                    // greater than int16
                    // update sbuff with data read from memory bank.
                    if (sDataNeeded > sDataLoaded) {
                        readDataS = window_readincr_256b<TT_DATA>(inWindow);
                        sbuff.val = upd_w(sbuff.val, sNumDataLoads % m_kDataLoadsInReg, readDataS.val); // Update Sbuff
                        sDataLoaded += m_kDataLoadVsize;
                        sNumDataLoads++;
                    } else {
                        // update tbuff with data read from memory bank when not updating sbuff.
                        if (tDataNeeded + m_kColumns > tDataLoaded) {
                            readDataT = window_readdecr_256b<TT_DATA>(cpWindow);
                            tbuff.val = upd_w(tbuff.val, (m_kDataLoadsInReg - 1 - (tNumDataLoads % m_kDataLoadsInReg)),
                                              readDataT.val); // Update Tbuff
                            tDataLoaded += m_kDataLoadVsize;
                            tNumDataLoads++;
                        }
                    }
                } else if (fnBufferUpdateScheme<TT_DATA, TT_COEFF>() == kPreLoadUsing128) {
                    // kPreLoadUsing128 - uses upd_v to update xbuff - 128bit loads and schedules load early, to avoid
                    // memory conflicts.
                    // To be used with data/coeff combo utilizing multi column MUL/MAC intrinsics, i.e. coeff type equal
                    // to int16
                    if (sDataNeeded > sDataLoaded) {
                        readDataV = window_readincr_128b<TT_DATA>(inWindow);
                        sbuff.val = upd_v(sbuff.val, (kUpdWToUpdVRatio * sNumDataLoads + sVDataLoads) %
                                                         (kUpdWToUpdVRatio * m_kDataLoadsInReg),
                                          readDataV.val); // Update Sbuff
                        sDataLoaded += m_kDataLoadVsize / kUpdWToUpdVRatio;
                        sVDataLoads++;
                    }
                    // update tbuff with data read from memory bank.
                    if (tDataNeeded > tDataLoaded) {
                        readDataU = window_readdecr_128b<TT_DATA>(cpWindow);
                        tbuff.val =
                            upd_v(tbuff.val, (2 * m_kDataLoadsInReg - 1 - (tVDataLoads % (2 * m_kDataLoadsInReg))),
                                  readDataU.val); // Update Tbuff
                        tDataLoaded += m_kDataLoadVsize / kUpdWToUpdVRatio;
                        tVDataLoads++;
                    }

                } else { // (fnBufferUpdateScheme<TT_DATA, TT_COEFF>() == kNoPreLoad)
                    // kNoPreLoad - default, uses upd_w - 256bit loads, no load sequence optimization.

                    // update sbuff with data read from memory bank.
                    if (sDataNeeded > sDataLoaded) {
                        readDataS = window_readincr_256b<TT_DATA>(inWindow);
                        sbuff.val = upd_w(sbuff.val, sNumDataLoads % m_kDataLoadsInReg, readDataS.val); // Update Sbuff
                        sDataLoaded += m_kDataLoadVsize;
                        sNumDataLoads++;
                    }
                    // update tbuff with data read from memory bank.
                    if (tDataNeeded > tDataLoaded) {
                        readDataT = window_readdecr_256b<TT_DATA>(cpWindow);
                        tbuff.val = upd_w(tbuff.val, (m_kDataLoadsInReg - 1 - (tNumDataLoads % m_kDataLoadsInReg)),
                                          readDataT.val); // Update Tbuff
                        tDataLoaded += m_kDataLoadVsize;
                        tNumDataLoads++;
                    }
                }

                // The tracking of when to load a new splice of coefficients is simpler since it always starts at 0.
                if (op % m_kCoeffRegVsize == 0) {
                    coe0 = *coeff++;
                }

                int xstart = (op + m_kSBuffXOffset);
                int ystart = m_kTStartOffset - (op % m_kDataRegVsize);

                // MAC operation.
                acc = macSrSym(acc, sbuff.val, xstart, tbuff.val, ystart, coe0.val, (op % m_kCoeffRegVsize));
            }

            if (sDataNeeded + fnCTColumnsLeft(TP_FIR_RANGE_LEN, m_kColumns) > sDataLoaded) {
                if (fnBufferUpdateScheme<TT_DATA, TT_COEFF>() == kPreLoadUsing128) {
                    readDataV = window_readincr_128b<TT_DATA>(inWindow);
                    sbuff.val = upd_v(sbuff.val, (kUpdWToUpdVRatio * sNumDataLoads + sVDataLoads) %
                                                     (kUpdWToUpdVRatio * m_kDataLoadsInReg),
                                      readDataV.val); // Update Sbuff
                    sVDataLoads++;
                } else {
                    readDataS = window_readincr_256b<TT_DATA>(inWindow);
                    sbuff.val =
                        upd_w(sbuff.val, sNumDataLoads % m_kDataLoadsInReg, readDataS.val); // 00++|____|____|____
                    sNumDataLoads++;
                }
            }
            if (tDataNeeded + (TP_FIR_RANGE_LEN % (2 * m_kColumns)) / 2 > tDataLoaded) {
                if (fnBufferUpdateScheme<TT_DATA, TT_COEFF>() == kPreLoadUsing128) {
                    readDataU = window_readdecr_128b<TT_DATA>(cpWindow);
                    tbuff.val = upd_v(tbuff.val, (2 * m_kDataLoadsInReg - 1 - (tVDataLoads % (2 * m_kDataLoadsInReg))),
                                      readDataU.val); // Update Tbuff
                    tVDataLoads++;
                } else {
                    readDataT = window_readdecr_256b<TT_DATA>(cpWindow);
                    tbuff.val = upd_w(tbuff.val, (m_kDataLoadsInReg - 1 - tNumDataLoads) % m_kDataLoadsInReg,
                                      readDataT.val); // 00++|____|____|____
                    tNumDataLoads++;
                }
            }

            if ((TP_FIR_RANGE_LEN % (kSymmetryFactor * m_kColumns) != 0) || (TP_KERNEL_POSITION + 1 == TP_CASC_LEN)) {
                // Read coeffs for center tap operation.
                if (m_kFirLenCeilCols % m_kCoeffRegVsize == 0) {
                    coe0 = *coeff++;
                }
            }
            // Center tap vector operation.
            acc = macSrSymCT<TP_FIR_RANGE_LEN % (kSymmetryFactor * m_kColumns)>(
                acc, sbuff.val, (m_kFirLenCeilCols + m_kSBuffXOffset), tbuff.val,
                ((m_kDataRegVsize + m_kTStartOffset - (m_kFirLenCeilCols) % m_kDataRegVsize) % m_kDataRegVsize),
                coe0.val, (m_kFirLenCeilCols % m_kCoeffRegVsize),
                (TP_KERNEL_POSITION + 1 == TP_CASC_LEN) ? sDataBuffSwap : 0);

            // Write cascade. Do nothing if cascade not present.
            writeCascade<TT_DATA, TT_COEFF>(outInterface, acc);

            outVal = shiftAndSaturate(acc, TP_SHIFT);
            writeWindow<TT_DATA, TT_COEFF, TP_NUM_OUTPUTS>(outInterface, outVal);
            window_decr(inWindow,
                        (m_kDataLoadVsize * (kUpdWToUpdVRatio * sNumDataLoads + sVDataLoads) / kUpdWToUpdVRatio -
                         m_kVOutSize)); // return read pointer to start of next chunk of window.
            window_incr(cpWindow,
                        (m_kDataLoadVsize * (kUpdWToUpdVRatio * tNumDataLoads + tVDataLoads) / kUpdWToUpdVRatio +
                         m_kVOutSize)); // back to m_kTDataLoadInitOffset and then move up by m_kVOutSize to point to
                                        // start of next chunk of window.
        }
};

// FIR filter function overloaded with cascade interface variations
// This is the default specialization of the main class used  when there is only one kernel for the whole filter. Static
// coefficients, single output
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
          unsigned int TP_USE_COEFF_RELOAD,
          unsigned int TP_NUM_OUTPUTS>
void fir_sr_sym<TT_DATA,
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
                TP_NUM_OUTPUTS>::filter(input_window<TT_DATA>* inWindow, output_window<TT_DATA>* outWindow) {
    T_inputIF<CASC_IN_FALSE, TT_DATA> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outWindow = outWindow;
    this->filterKernel(inInterface, outInterface);
};

// This is a specialization of the main class for when there is only one kernel for the whole filter. Static
// coefficients, dual output
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN>
void fir_sr_sym<TT_DATA,
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

// FIR filter function overloaded with cascade interface variations
// This is a specialization of the main class for when there is only one kernel for the whole filter. Reloadable
// coefficients, single output
//-----------------------------------------------------------------------------------------------------

template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_sr_sym<TT_DATA,
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
                1>::filter(input_window<TT_DATA>* inWindow,
                           output_window<TT_DATA>* outWindow,
                           const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / kSymmetryFactor]) {
    T_inputIF<CASC_IN_FALSE, TT_DATA> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outWindow = outWindow;
    this->filterKernel(inInterface, outInterface, inTaps);
};

// This is a specialization of the main class for when there is only one kernel for the whole filter. Reloadable
// coefficients, dual output
//-----------------------------------------------------------------------------------------------------

template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_sr_sym<TT_DATA,
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
                2>::filter(input_window<TT_DATA>* inWindow,
                           output_window<TT_DATA>* outWindow,
                           output_window<TT_DATA>* outWindow2,
                           const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / kSymmetryFactor]) {
    T_inputIF<CASC_IN_FALSE, TT_DATA> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outWindow = outWindow;
    outInterface.outWindow2 = outWindow2;
    this->filterKernel(inInterface, outInterface, inTaps);
};

// FIR filter function overloaded with cascade interface variations
// This is a specialization of the main class for the final kernel in a cascade chain. Static coefficients, single
// output
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_sr_sym<TT_DATA,
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

// This is a specialization of the main class for the final kernel in a cascade chain. Static coefficients, dual output
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_sr_sym<TT_DATA,
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

// FIR filter function overloaded with cascade interface variations
// This is a specialization of the main class for the first kernel in a cascade chain. Static coefficients
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_sr_sym<TT_DATA,
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

// FIR filter function overloaded with cascade interface variations
// This is a specialization of the main class for any kernel within a cascade chain, but neither first nor last. Static
// coefficients
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_sr_sym<TT_DATA,
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

// This is a specialization of the main class for the final kernel in a cascade chain. Reloadable coefficients, single
// output
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_sr_sym<TT_DATA,
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

// This is a specialization of the main class for the final kernel in a cascade chain. Reloadable coefficients, dual
// output
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_sr_sym<TT_DATA,
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

// FIR filter function overloaded with cascade interface variations
// This is a specialization of the main class for the first kernel in a cascade chain. Reloadable coefficients
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_sr_sym<TT_DATA,
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
                1>::filter(input_window<TT_DATA>* inWindow,
                           output_stream_cacc48* outCascade,
                           const TT_COEFF (&inTaps)[(TP_FIR_LEN + 1) / kSymmetryFactor],
                           output_window<TT_DATA>* broadcastWindow) {
    T_inputIF<CASC_IN_FALSE, TT_DATA> inInterface;
    T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface;
    inInterface.inWindow = inWindow;
    outInterface.outCascade = outCascade;
    outInterface.broadcastWindow = broadcastWindow;
    this->filterKernel(inInterface, outInterface, inTaps);
};

// FIR filter function overloaded with cascade interface variations
// This is a specialization of the main class for any kernel within a cascade chain, but neither first nor last.
// Reloadable coefficients
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_FIR_RANGE_LEN,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void fir_sr_sym<TT_DATA,
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
