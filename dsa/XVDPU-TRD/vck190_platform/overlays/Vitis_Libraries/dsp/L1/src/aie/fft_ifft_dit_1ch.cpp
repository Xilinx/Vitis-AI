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
FFT/IFFT DIT single channel kernal code.
This file captures the body of run-time code for the kernal class.

Coding conventions
  TT_      template type suffix
  TP_      template parameter suffix
*/

#include <adf.h>
#include <stdio.h>

using namespace std;

#define __NEW_WINDOW_H__ 1
#define __AIEARCH__ 1
#define __AIENGINE__ 1
// if we use 1kb registers -> aie api uses 2x512b registers for 1024b so we need this for QoR
#define __AIE_API_USE_NATIVE_1024B_VECTOR__
#include "aie_api/aie_adf.hpp"

#include "fft_com_inc.h"
#include "fft_ifft_dit_1ch.hpp"
#include "kernel_api_utils.hpp"
#include "fft_ifft_dit_1ch_utils.hpp"
#include "fft_twiddle_lut_dit.h"
#include "fft_twiddle_lut_dit_cfloat.h"

namespace xf {
namespace dsp {
namespace aie {
namespace fft {
namespace dit_1ch {

// Stockham stages kernel common - this function is defered to the stockham object of the kernel class because it
// requires no specialization
// whereas the kernel class itself does, so this is just a way to avoid code duplication
template <typename TT_DATA,
          typename TT_OUT_DATA,
          typename TT_TWIDDLE,
          typename TT_INTERNAL_DATA,
          unsigned int TP_POINT_SIZE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void stockhamStages<TT_DATA,
                                TT_OUT_DATA,
                                TT_TWIDDLE,
                                TT_INTERNAL_DATA,
                                TP_POINT_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::stagePreamble(void* tw_table,
                                                                void* tmp1_buf,
                                                                void* tmp2_buf,
                                                                input_window<TT_DATA>* __restrict inputx,
                                                                output_window<TT_OUT_DATA>* __restrict outputy) {
    typedef typename std::conditional<std::is_same<TT_DATA, cint16>::value, cint32_t, TT_DATA>::type T_internalDataType;
    constexpr int minPtSizePwr = 4;
    constexpr int maxPtSizePwr = fnPointSizePower<TP_POINT_SIZE>();
    constexpr int kOpsInWindow = TP_WINDOW_VSIZE / TP_POINT_SIZE;

    TT_DATA* xbuff = (TT_DATA*)inputx->ptr;
    TT_OUT_DATA* obuff = (TT_OUT_DATA*)outputy->ptr;
    bool inv;
    if
        constexpr(TP_FFT_NIFFT == 1) { inv = false; }
    else {
        inv = true;
    }

    // This code would be moved to the constructor preferably, but contains intrinsics, so cannot. Also, it may be
    // necessary if 2 kernels share a processor
    set_rnd(rnd_pos_inf); // Match the twiddle round mode of Matlab.
    set_sat();            // do saturate.

    if
        constexpr(TP_DYN_PT_SIZE == 1) {
            TT_INTERNAL_DATA* myTmp2_buf = (TT_INTERNAL_DATA*)tmp2_buf; // when dynamic, xbuff header alters start of
                                                                        // data and tmp2_buf is a pointer to xbuff so
                                                                        // has to alter too.
            T_buff_256b<TT_DATA> header;
            TT_DATA headerVal;
            TT_OUT_DATA headerOpVal;
            T_buff_256b<TT_OUT_DATA> headerOp;
            //  ::aie::vector<TT_DATA,32/sizeof(TT_DATA)> headerRawVal;
            T_buff_256b<TT_OUT_DATA> blankOp;
            using in256VectorType = ::aie::vector<TT_DATA, 256 / 8 / sizeof(TT_DATA)>;
            using outVectorType =
                ::aie::vector<TT_OUT_DATA, 256 / 8 / sizeof(TT_DATA)>; // has to be same number of elements as input.
            in256VectorType* inPtr;
            outVectorType* outPtr;
            in256VectorType in256;
            outVectorType outVector;
            int ptSizePwr;
            ::aie::accum<cacc48, 256 / 8 / sizeof(TT_DATA)> cacc384;

            blankOp.val = ::aie::zeros<TT_OUT_DATA, 32 / sizeof(TT_OUT_DATA)>();
            headerOp.val = ::aie::zeros<TT_OUT_DATA, 32 / sizeof(TT_OUT_DATA)>();
            header = window_readincr_256b(inputx);
            xbuff = (TT_DATA*)inputx->ptr;
            if
                constexpr(!fnUsePingPongIntBuffer<TT_DATA>()) myTmp2_buf = (TT_INTERNAL_DATA*)xbuff;
            else
                myTmp2_buf = (TT_INTERNAL_DATA*)tmp2_buf;
            headerVal = header.val.get(0);
            headerOpVal.real = headerVal.real; // copy/cast header to output one field at a time
            headerOpVal.imag = headerVal.imag;
            headerOp.val.set(headerOpVal, 0);
            inv = headerVal.real == 0 ? true : false;
            headerVal = header.val.get(1);
            headerOpVal.real = headerVal.real;
            headerOpVal.imag = headerVal.imag;
            headerOp.val.set(headerOpVal, 1);
            ptSizePwr = (int)headerVal.real;
            if ((ptSizePwr >= minPtSizePwr) && (ptSizePwr <= maxPtSizePwr)) {
                window_write(outputy, headerOp.val);
                window_incr(outputy, 32 / sizeof(TT_OUT_DATA));
                obuff = (TT_OUT_DATA*)outputy->ptr;
                if (TP_START_RANK >=
                    ptSizePwr) { // i.e. kernels earlier in the chain have already performed the FFT for this size
                    // copy input window to output window in 256bit chunks
                    inPtr = (in256VectorType*)inputx->ptr;
                    outPtr = (outVectorType*)outputy->ptr;
                    if
                        constexpr(std::is_same<TT_DATA, cfloat>::value) {
                            for (int i = 0; i < TP_WINDOW_VSIZE / (32 / sizeof(TT_DATA)); i++) {
                                *outPtr++ = *inPtr++;
                            }
                        }
                    else {
                        for (int i = 0; i < TP_WINDOW_VSIZE / (32 / sizeof(TT_DATA)); i++) {
                            in256 = *inPtr++;
                            cacc384.from_vector(in256, 0);
                            outVector = cacc384.template to_vector<TT_OUT_DATA>(0);
                            *outPtr++ = outVector;
                        }
                    }

                } else {
                    for (int iter = 0; iter < kOpsInWindow; iter++) {
                        if
                            constexpr(!fnUsePingPongIntBuffer<TT_DATA>()) myTmp2_buf = (TT_INTERNAL_DATA*)xbuff;
                        else
                            myTmp2_buf = (TT_INTERNAL_DATA*)tmp2_buf;
                        calc(xbuff, (TT_TWIDDLE**)tw_table, (T_internalDataType*)tmp1_buf,
                             (T_internalDataType*)myTmp2_buf, obuff, ptSizePwr, inv); // dynamic variant of calc
                        obuff += TP_POINT_SIZE;
                        xbuff += TP_POINT_SIZE;
                    }
                }
            } else {
                headerOpVal = unitVector<TT_OUT_DATA>();
                headerOp.val.set(headerOpVal, 3);
                window_write(outputy, headerOp.val);
                window_incr(outputy, 32 / sizeof(TT_OUT_DATA));
            }
        }
    else { // else for TP_DYN_PT_SIZE == 1
        for (int iter = 0; iter < kOpsInWindow; iter++) chess_prepare_for_pipelining chess_loop_range(kOpsInWindow, ) {
                calc(xbuff, (TT_TWIDDLE**)tw_table, (T_internalDataType*)tmp1_buf, (T_internalDataType*)tmp2_buf,
                     obuff);
                obuff += TP_POINT_SIZE;
                xbuff += TP_POINT_SIZE;
            }
    }
};
// stockhamStages calc body. Static TP_POINT_SIZE variant
// This is a helper class which allows the many variants of the kernel class to call a single body of code for the fft
// stages.
template <typename TT_DATA,
          typename TT_OUT_DATA,
          typename TT_TWIDDLE,
          typename TT_INTERNAL_DATA,
          unsigned int TP_POINT_SIZE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void stockhamStages<TT_DATA,
                                TT_OUT_DATA,
                                TT_TWIDDLE,
                                TT_INTERNAL_DATA,
                                TP_POINT_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::calc(TT_DATA* __restrict xbuff,
                                                       TT_TWIDDLE** tw_table,
                                                       TT_INTERNAL_DATA* tmp1_buf,
                                                       TT_INTERNAL_DATA* tmp2_buf,
                                                       TT_OUT_DATA* __restrict obuff) {
    constexpr int kPointSizePower = fnPointSizePower<TP_POINT_SIZE>();
    bool inv = TP_FFT_NIFFT == 1 ? false : true;

    // This code should be moved to the constructor preferably
    set_rnd(rnd_pos_inf); // Match the twiddle round mode of Matlab.
    set_sat();            // do saturate.

    TT_DATA* inptr;
    TT_INTERNAL_DATA* outptr; // outptr is only used for multi-kernel operation where the kernel output (TT_OUT_DATA) is
                              // used. However, T_INTERNAL DATA has to be used since the output of a stage isn't
                              // necessarily TT_OUT_DATA.
    TT_INTERNAL_DATA* my_tmp2_buf = fnUsePingPongIntBuffer<TT_DATA>() ? tmp2_buf : (TT_INTERNAL_DATA*)xbuff;
    TT_INTERNAL_DATA* tmp_bufs[2] = {tmp1_buf, my_tmp2_buf}; // tmp2 is actually xbuff reused.
    unsigned int pingPong = 1;                               // use tmp_buf1 as initial output
    int tw_index = 0;
    int rank = 0;
    int r = 0; // r is an indication to the stage of rank.

    if
        constexpr(std::is_same<TT_DATA, cfloat>::value) {
            //-----------------------------------------------------------------------------
            // cfloat handling

            r = TP_POINT_SIZE >> 1;

// internal stages including input stage
#pragma unroll(kIntR2Stages)
            for (int stage = 0; stage < kIntR2Stages; ++stage) {
                if (stage >= TP_START_RANK && stage + 1 <= TP_END_RANK) { // i.e is this rank in the scope of the
                                                                          // kernel?
                    outptr = (stage == TP_END_RANK - 1) ? (TT_INTERNAL_DATA*)obuff
                                                        : (TT_INTERNAL_DATA*)tmp_bufs[1 - pingPong];
                    inptr = (stage == TP_START_RANK) ? (TT_INTERNAL_DATA*)xbuff : (TT_INTERNAL_DATA*)tmp_bufs[pingPong];
                    stage0_radix2_dit<cfloat, cfloat, cfloat>((cfloat*)inptr, (cfloat*)tw_table[tw_index],
                                                              TP_POINT_SIZE, r, 0, (cfloat*)outptr, inv);
                    pingPong = 1 - pingPong;
                }
                r = r >> 1; // divide by 2
                tw_index++;
            }

            // final 2 stages - these require a different granularity of sample interleaving (each) so are called stage
            // 1 and stage2.
            if
                constexpr(kIntR2Stages >= TP_START_RANK &&
                          kIntR2Stages < TP_END_RANK) { // i.e is this rank in the scope of the kernel?
                    outptr = (kIntR2Stages + 1 == TP_END_RANK) ? (TT_INTERNAL_DATA*)obuff
                                                               : (TT_INTERNAL_DATA*)tmp_bufs[1 - pingPong];
                    stage1_radix2_dit<cfloat, cfloat, cfloat>((cfloat*)tmp_bufs[pingPong], (cfloat*)tw_table[tw_index],
                                                              TP_POINT_SIZE, r, 0, (cfloat*)outptr, inv);
                    pingPong = 1 - pingPong;
                }
            r = r >> 1; // divide by 2
            tw_index++;

            if
                constexpr(kIntR2Stages + 2 == TP_END_RANK) { // i.e is this rank in the scope of the kernel?
                    stage2_radix2_dit<cfloat, cfloat, cfloat>((cfloat*)tmp_bufs[pingPong], (cfloat*)tw_table[tw_index],
                                                              TP_POINT_SIZE, r, TP_SHIFT, (cfloat*)obuff,
                                                              inv); // r is not used.
                }
        }
    else { // integer types can use radix 4 stages
        //-----------------------------------------------------------------------------
        // cint handling

        // input stage
        if
            constexpr(kOddPower == 1) {
                r = TP_POINT_SIZE >> 1; // divide by 2
                if
                    constexpr(TP_START_RANK == 0) {
                        outptr = (rank == TP_END_RANK - 1) ? (TT_INTERNAL_DATA*)obuff
                                                           : (TT_INTERNAL_DATA*)tmp_bufs[1 - pingPong];
                        stage0_radix2_dit<TT_DATA, TT_INTERNAL_DATA, TT_TWIDDLE>(xbuff, tw_table[0], TP_POINT_SIZE, r,
                                                                                 FFT_SHIFT15, outptr, inv);
                        pingPong = 1 - pingPong;
                    }
                tw_index = 1;
                rank++;
            }
        else {
            r = TP_POINT_SIZE >> 2; // divide by 4
            if
                constexpr(TP_START_RANK == 0) {
                    outptr = (rank == TP_END_RANK - 2) ? (TT_INTERNAL_DATA*)obuff
                                                       : (TT_INTERNAL_DATA*)tmp_bufs[1 - pingPong];
                    stage0_radix4_dit<TT_DATA, TT_INTERNAL_DATA, TT_TWIDDLE>(
                        xbuff, tw_table[0], tw_table[1], TP_POINT_SIZE, r, FFT_SHIFT15, outptr, inv);
                    pingPong = 1 - pingPong;
                }
            tw_index = 2;
            rank += 2;
        }

// internal stages
// Note that the loop ensures only intermediate ranks are handled.
// the Internal if appears to support the final rank being done here, but this is prevented by the loop.
#pragma unroll(GUARD_ZERO((kPointSizePower - 2 - 2 + kOddPower) / 2))
        for (int stage = rank; stage < kPointSizePower - 2; stage = stage + 2) {
            r = r >> 2;
            if ((rank >= TP_START_RANK) && (rank + 2 <= TP_END_RANK)) {
                outptr =
                    (rank + 2 == TP_END_RANK) ? (TT_INTERNAL_DATA*)obuff : (TT_INTERNAL_DATA*)tmp_bufs[1 - pingPong];
                stage0_radix4_dit<TT_INTERNAL_DATA, TT_INTERNAL_DATA, TT_TWIDDLE>(
                    tmp_bufs[pingPong], tw_table[tw_index], tw_table[tw_index + 1], TP_POINT_SIZE, r, FFT_SHIFT15,
                    outptr, inv);
                pingPong = 1 - pingPong;
            }
            rank += 2;
            tw_index += 2;
        }

        // output stage
        if
            constexpr(kPointSizePower == TP_END_RANK) {
                stage1_radix4_dit<TT_INTERNAL_DATA, TT_OUT_DATA, TT_TWIDDLE>(tmp_bufs[pingPong], tw_table[tw_index],
                                                                             tw_table[tw_index + 1], TP_POINT_SIZE,
                                                                             FFT_SHIFT15 + TP_SHIFT, obuff, inv);
            }
    }
};

// stockhamStages calc body. Dynamic variant.
template <typename TT_DATA,
          typename TT_OUT_DATA,
          typename TT_TWIDDLE,
          typename TT_INTERNAL_DATA,
          unsigned int TP_POINT_SIZE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void stockhamStages<TT_DATA,
                                TT_OUT_DATA,
                                TT_TWIDDLE,
                                TT_INTERNAL_DATA,
                                TP_POINT_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::calc(TT_DATA* __restrict xbuff,
                                                       TT_TWIDDLE** tw_table,
                                                       TT_INTERNAL_DATA* tmp1_buf,
                                                       TT_INTERNAL_DATA* tmp2_buf,
                                                       TT_OUT_DATA* __restrict obuff,
                                                       int ptSizePwr,
                                                       bool inv) {
    // This code should be moved to the constructor preferably
    set_rnd(rnd_pos_inf); // Match the twiddle round mode of Matlab.
    set_sat();            // do saturate.

    TT_DATA* inptr;
    TT_INTERNAL_DATA* outptr; // outptr is only used for multi-kernel operation where the kernel output (TT_OUT_DATA) is
                              // used. However, T_INTERNAL DATA has to be used since the output of a stage isn't
                              // necessarily TT_OUT_DATA.
    TT_INTERNAL_DATA* my_tmp2_buf = fnUsePingPongIntBuffer<TT_DATA>() ? tmp2_buf : (TT_INTERNAL_DATA*)xbuff;
    TT_INTERNAL_DATA* tmp_bufs[2] = {tmp1_buf, my_tmp2_buf}; // tmp2 is actually xbuff reused.
    unsigned int pingPong = 1; // use tmp_buf1 as input or tmp_buf2? Initially tmp2 is input since this is xbuff
    int tw_index = 0;
    int rank = 0;
    int r = 0; // r is an indication to the stage of rank.
    unsigned int intR2Stages;
    unsigned int ptSize = 1 << ptSizePwr;
    unsigned int myStart = TP_START_RANK;
    unsigned int myEnd = TP_END_RANK;

    if
        constexpr(std::is_same<TT_DATA, cfloat>::value) {
            //-----------------------------------------------------------------------------
            // cfloat handling

            r = 1 << (ptSizePwr - 1);
            intR2Stages = ptSizePwr - 2;

            // internal stages including input stage
            for (int stage = 0; stage < intR2Stages; ++stage) {
                if (rank >= TP_START_RANK && rank + 1 <= TP_END_RANK) { // i.e is this rank in the scope of the kernel?
                    outptr = (rank + 1 == TP_END_RANK) ? (TT_INTERNAL_DATA*)obuff
                                                       : (TT_INTERNAL_DATA*)tmp_bufs[1 - pingPong];
                    inptr = (rank == TP_START_RANK) ? (cfloat*)xbuff : (cfloat*)tmp_bufs[pingPong];
                    stage0_radix2_dit<cfloat, cfloat, cfloat>((cfloat*)inptr, (cfloat*)tw_table[tw_index], ptSize, r, 0,
                                                              (cfloat*)outptr, inv);
                    pingPong = 1 - pingPong;
                }
                rank++;
                r = r >> 1; // divide by 2
                tw_index++;
            }
            // final 2 stages - these require a different granularity of sample interleaving (each) so are called stage
            // 1 and stage2.
            if (rank >= TP_START_RANK && rank < TP_END_RANK) { // i.e is this rank in the scope of the kernel?
                outptr =
                    (rank + 1 == TP_END_RANK) ? (TT_INTERNAL_DATA*)obuff : (TT_INTERNAL_DATA*)tmp_bufs[1 - pingPong];
                stage1_radix2_dit<cfloat, cfloat, cfloat>((cfloat*)tmp_bufs[pingPong], (cfloat*)tw_table[tw_index],
                                                          ptSize, r, 0, (cfloat*)outptr, inv);
                pingPong = 1 - pingPong;
            }
            rank++;
            r = r >> 1; // divide by 2
            tw_index++;

            if (rank >= TP_START_RANK && rank + 1 <= TP_END_RANK) { // i.e is this rank in the scope of the kernel?
                ptSize = 1 << ptSizePwr;
                stage2_radix2_dit<cfloat, cfloat, cfloat>((cfloat*)tmp_bufs[pingPong], (cfloat*)tw_table[tw_index],
                                                          ptSize, r, TP_SHIFT, (cfloat*)obuff, inv); // r is not used.
            }
        }
    else { // integer types can use radix 4 stages
        //------------------------------------------------------------------------------------------------
        // cint handling, dynamic variant
        ptSize = ((unsigned int)1 << ptSizePwr); // without this, x86 zeros ptSize.

        // input stage. This cannot be rolled into the stage loop because the first stage has TT_DATA not
        // TT_INTERNAL_DATA as input
        if ((ptSizePwr & 1) == 1) { // odd point size power
            r = 1 << (ptSizePwr - 1);
            if
                constexpr(TP_START_RANK == 0) {
                    outptr =
                        (rank + 3 > TP_END_RANK) ? (TT_INTERNAL_DATA*)obuff : (TT_INTERNAL_DATA*)tmp_bufs[1 - pingPong];
                    ptSize = ((unsigned int)1 << ptSizePwr); // without this, x86 zeros ptSize.
                    stage0_radix2_dit<TT_DATA, TT_INTERNAL_DATA, TT_TWIDDLE>(xbuff, tw_table[0], ptSize, r, FFT_SHIFT15,
                                                                             outptr, inv);
                    pingPong = 1 - pingPong;
                }
            tw_index = 1;
            rank++;
        } else {
            r = 1 << (ptSizePwr - 2);
            if
                constexpr(TP_START_RANK == 0) {
                    outptr =
                        (rank + 4 > TP_END_RANK) ? (TT_INTERNAL_DATA*)obuff : (TT_INTERNAL_DATA*)tmp_bufs[1 - pingPong];
                    ptSize = ((unsigned int)1 << ptSizePwr); // without this, x86 zeros ptSize.
                    stage0_radix4_dit<TT_DATA, TT_INTERNAL_DATA, TT_TWIDDLE>(xbuff, tw_table[0], tw_table[1], ptSize, r,
                                                                             FFT_SHIFT15, outptr, inv);
                    pingPong = 1 - pingPong;
                }
            tw_index = 2;
            rank += 2;
        }

        // internal stages
        // Note that the loop ensures only intermediate ranks are handled.
        // the Internal if appears to support the final rank being done here, but this is prevented by the loop.
        for (int stage = rank; stage < ptSizePwr - 2; stage = stage + 2) {
            r = r >> 2;
            if ((rank + 2 > TP_START_RANK) && (rank + 1 < TP_END_RANK)) {
                outptr =
                    (rank + 4 > TP_END_RANK) ? (TT_INTERNAL_DATA*)obuff : (TT_INTERNAL_DATA*)tmp_bufs[1 - pingPong];
                stage0_radix4_dit<TT_INTERNAL_DATA, TT_INTERNAL_DATA, TT_TWIDDLE>(
                    tmp_bufs[pingPong], tw_table[tw_index], tw_table[tw_index + 1], ptSize, r, FFT_SHIFT15, outptr,
                    inv);
                pingPong = 1 - pingPong;
            }
            rank += 2;
            tw_index += 2;
        }

        // output stage. Regardless of kernel splits, the output of this stage goes to the output buffer, so no need for
        // the outptr switch
        if ((rank + 2 == ptSizePwr) && (rank + 2 > TP_START_RANK) && (rank + 1 < TP_END_RANK)) {
            stage1_radix4_dit<TT_INTERNAL_DATA, TT_OUT_DATA, TT_TWIDDLE>(tmp_bufs[pingPong], tw_table[tw_index],
                                                                         tw_table[tw_index + 1], ptSize,
                                                                         FFT_SHIFT15 + TP_SHIFT, obuff, inv);
        }
    }

    if ((1 << ptSizePwr) < TP_POINT_SIZE) {
        using outVectType = ::aie::vector<TT_OUT_DATA, 32 / sizeof(TT_OUT_DATA)>;
        outVectType zerosOut = ::aie::zeros<TT_OUT_DATA, 32 / sizeof(TT_OUT_DATA)>();
        outVectType* outFillPtr = (outVectType*)obuff;
        outFillPtr += (1 << ptSizePwr) * sizeof(TT_OUT_DATA) / 32;
        int fillCycles = (TP_POINT_SIZE - (1 << ptSizePwr)) * sizeof(TT_OUT_DATA) / 32;
        for (int i = 0; i < fillCycles; i++) {
            *outFillPtr++ = zerosOut;
        }
    }
};

// FFT/iFFT DIT single channel function - base of specialization .
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_OUT_DATA,
          typename TT_TWIDDLE,
          unsigned int TP_POINT_SIZE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelFFTClass<TT_DATA,
                                TT_OUT_DATA,
                                TT_TWIDDLE,
                                TP_POINT_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::kernelFFT(input_window<TT_DATA>* __restrict inputx,
                                                            output_window<TT_OUT_DATA>* __restrict outputy){};

template <typename TT_DATA,
          typename TT_OUT_DATA,
          typename TT_TWIDDLE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelFFTClass<TT_DATA,
                                TT_OUT_DATA,
                                TT_TWIDDLE,
                                FFT4096_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::kernelFFT(input_window<TT_DATA>* __restrict inputx,
                                                            output_window<TT_OUT_DATA>* __restrict outputy) {
    typedef typename std::conditional<std::is_same<TT_DATA, cint16>::value, cint32_t, TT_DATA>::type T_internalDataType;
    TT_DATA* xbuff = (TT_DATA*)inputx->ptr;
    T_internalDataType* tmp1_buf = (T_internalDataType*)ktmp1_buf;
    T_internalDataType* tmp2_buf =
        fnUsePingPongIntBuffer<TT_DATA>() ? (T_internalDataType*)ktmp2_buf : (T_internalDataType*)xbuff;

    stages.stagePreamble(tw_table, tmp1_buf, tmp2_buf, inputx, outputy);
};

template <typename TT_OUT_DATA,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelFFTClass<cint16,
                                TT_OUT_DATA,
                                cint16,
                                FFT4096_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::kernelFFT(input_window<cint16>* __restrict inputx,
                                                            output_window<TT_OUT_DATA>* __restrict outputy) {
    typedef cint16 TT_DATA;
    typedef cint32_t T_internalDataType;
    cint16* xbuff = (cint16*)inputx->ptr;
    TT_OUT_DATA* obuff = (TT_OUT_DATA*)outputy->ptr;
    T_internalDataType* tmp1_buf = (T_internalDataType*)ktmp1_buf;
    T_internalDataType* tmp2_buf = (T_internalDataType*)ktmp2_buf;

    stages.stagePreamble(tw_table, tmp1_buf, tmp2_buf, inputx, outputy);
};

template <typename TT_DATA,
          typename TT_OUT_DATA,
          typename TT_TWIDDLE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelFFTClass<TT_DATA,
                                TT_OUT_DATA,
                                TT_TWIDDLE,
                                FFT2048_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::kernelFFT(input_window<TT_DATA>* __restrict inputx,
                                                            output_window<TT_OUT_DATA>* __restrict outputy) {
    typedef typename std::conditional<std::is_same<TT_DATA, cint16>::value, cint32_t, TT_DATA>::type T_internalDataType;
    TT_DATA* xbuff = (TT_DATA*)inputx->ptr;
    T_internalDataType* tmp1_buf = (T_internalDataType*)ktmp1_buf;
    T_internalDataType* tmp2_buf =
        fnUsePingPongIntBuffer<TT_DATA>() ? (T_internalDataType*)ktmp2_buf : (T_internalDataType*)xbuff;

    stages.stagePreamble(tw_table, tmp1_buf, tmp2_buf, inputx, outputy);
};

template <typename TT_OUT_DATA,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelFFTClass<cint16,
                                TT_OUT_DATA,
                                cint16,
                                FFT2048_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::kernelFFT(input_window<cint16>* __restrict inputx,
                                                            output_window<TT_OUT_DATA>* __restrict outputy) {
    typedef cint16 TT_DATA;
    typedef cint32_t T_internalDataType;
    cint16* xbuff = (cint16*)inputx->ptr;
    TT_OUT_DATA* obuff = (TT_OUT_DATA*)outputy->ptr;
    T_internalDataType* tmp1_buf = (T_internalDataType*)ktmp1_buf;
    T_internalDataType* tmp2_buf = (T_internalDataType*)ktmp2_buf;

    stages.stagePreamble(tw_table, tmp1_buf, tmp2_buf, inputx, outputy);
};

template <typename TT_DATA,
          typename TT_OUT_DATA,
          typename TT_TWIDDLE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelFFTClass<TT_DATA,
                                TT_OUT_DATA,
                                TT_TWIDDLE,
                                FFT1024_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::kernelFFT(input_window<TT_DATA>* __restrict inputx,
                                                            output_window<TT_OUT_DATA>* __restrict outputy) {
    typedef typename std::conditional<std::is_same<TT_DATA, cint16>::value, cint32_t, TT_DATA>::type T_internalDataType;
    TT_DATA* xbuff = (TT_DATA*)inputx->ptr;
    T_internalDataType* tmp1_buf = (T_internalDataType*)ktmp1_buf;
    T_internalDataType* tmp2_buf =
        fnUsePingPongIntBuffer<TT_DATA>() ? (T_internalDataType*)ktmp2_buf : (T_internalDataType*)xbuff;

    stages.stagePreamble(tw_table, tmp1_buf, tmp2_buf, inputx, outputy);
};

template <typename TT_OUT_DATA,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelFFTClass<cint16,
                                TT_OUT_DATA,
                                cint16,
                                FFT1024_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::kernelFFT(input_window<cint16>* __restrict inputx,
                                                            output_window<TT_OUT_DATA>* __restrict outputy) {
    typedef cint16 TT_DATA;
    typedef cint32_t T_internalDataType;
    cint16* xbuff = (cint16*)inputx->ptr;
    TT_OUT_DATA* obuff = (TT_OUT_DATA*)outputy->ptr;
    T_internalDataType* tmp1_buf = (T_internalDataType*)ktmp1_buf;
    T_internalDataType* tmp2_buf = (T_internalDataType*)ktmp2_buf;

    stages.stagePreamble(tw_table, tmp1_buf, tmp2_buf, inputx, outputy);
};

template <typename TT_DATA,
          typename TT_OUT_DATA,
          typename TT_TWIDDLE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelFFTClass<TT_DATA,
                                TT_OUT_DATA,
                                TT_TWIDDLE,
                                FFT512_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::kernelFFT(input_window<TT_DATA>* __restrict inputx,
                                                            output_window<TT_OUT_DATA>* __restrict outputy) {
    typedef typename std::conditional<std::is_same<TT_DATA, cint16>::value, cint32_t, TT_DATA>::type T_internalDataType;
    TT_DATA* xbuff = (TT_DATA*)inputx->ptr;
    T_internalDataType* tmp1_buf = (T_internalDataType*)ktmp1_buf;
    T_internalDataType* tmp2_buf =
        fnUsePingPongIntBuffer<TT_DATA>() ? (T_internalDataType*)ktmp2_buf : (T_internalDataType*)xbuff;

    stages.stagePreamble(tw_table, tmp1_buf, tmp2_buf, inputx, outputy);
};

template <typename TT_OUT_DATA,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelFFTClass<cint16,
                                TT_OUT_DATA,
                                cint16,
                                FFT512_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::kernelFFT(input_window<cint16>* __restrict inputx,
                                                            output_window<TT_OUT_DATA>* __restrict outputy) {
    typedef cint16 TT_DATA;
    typedef cint32_t T_internalDataType;
    cint16* xbuff = (cint16*)inputx->ptr;
    TT_OUT_DATA* obuff = (TT_OUT_DATA*)outputy->ptr;
    T_internalDataType* tmp1_buf = (T_internalDataType*)ktmp1_buf;
    T_internalDataType* tmp2_buf = (T_internalDataType*)ktmp2_buf;

    stages.stagePreamble(tw_table, tmp1_buf, tmp2_buf, inputx, outputy);
};

template <typename TT_DATA,
          typename TT_OUT_DATA,
          typename TT_TWIDDLE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelFFTClass<TT_DATA,
                                TT_OUT_DATA,
                                TT_TWIDDLE,
                                FFT256_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::kernelFFT(input_window<TT_DATA>* __restrict inputx,
                                                            output_window<TT_OUT_DATA>* __restrict outputy) {
    typedef typename std::conditional<std::is_same<TT_DATA, cint16>::value, cint32_t, TT_DATA>::type T_internalDataType;
    TT_DATA* xbuff = (TT_DATA*)inputx->ptr;
    T_internalDataType* tmp1_buf = (T_internalDataType*)ktmp1_buf;
    T_internalDataType* tmp2_buf =
        fnUsePingPongIntBuffer<TT_DATA>() ? (T_internalDataType*)ktmp2_buf : (T_internalDataType*)xbuff;

    stages.stagePreamble(tw_table, tmp1_buf, tmp2_buf, inputx, outputy);
};

template <typename TT_OUT_DATA,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelFFTClass<cint16,
                                TT_OUT_DATA,
                                cint16,
                                FFT256_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::kernelFFT(input_window<cint16>* __restrict inputx,
                                                            output_window<TT_OUT_DATA>* __restrict outputy) {
    typedef cint16 TT_DATA;

    typedef cint32_t T_internalDataType;
    cint16* xbuff = (cint16*)inputx->ptr;
    TT_OUT_DATA* obuff = (TT_OUT_DATA*)outputy->ptr;
    T_internalDataType* tmp1_buf = (T_internalDataType*)ktmp1_buf;
    T_internalDataType* tmp2_buf = (T_internalDataType*)ktmp2_buf;

    stages.stagePreamble(tw_table, tmp1_buf, tmp2_buf, inputx, outputy);
};

template <typename TT_DATA,
          typename TT_OUT_DATA,
          typename TT_TWIDDLE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelFFTClass<TT_DATA,
                                TT_OUT_DATA,
                                TT_TWIDDLE,
                                FFT128_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::kernelFFT(input_window<TT_DATA>* __restrict inputx,
                                                            output_window<TT_OUT_DATA>* __restrict outputy) {
    typedef typename std::conditional<std::is_same<TT_DATA, cint16>::value, cint32_t, TT_DATA>::type T_internalDataType;
    TT_DATA* xbuff = (TT_DATA*)inputx->ptr;
    T_internalDataType* tmp1_buf = (T_internalDataType*)ktmp1_buf;
    T_internalDataType* tmp2_buf =
        fnUsePingPongIntBuffer<TT_DATA>() ? (T_internalDataType*)ktmp2_buf : (T_internalDataType*)xbuff;

    stages.stagePreamble(tw_table, tmp1_buf, tmp2_buf, inputx, outputy);
};

template <typename TT_OUT_DATA,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelFFTClass<cint16,
                                TT_OUT_DATA,
                                cint16,
                                FFT128_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::kernelFFT(input_window<cint16>* __restrict inputx,
                                                            output_window<TT_OUT_DATA>* __restrict outputy) {
    typedef cint16 TT_DATA;
    typedef cint32_t T_internalDataType;
    cint16* xbuff = (cint16*)inputx->ptr;
    TT_OUT_DATA* obuff = (TT_OUT_DATA*)outputy->ptr;
    T_internalDataType* tmp1_buf = (T_internalDataType*)ktmp1_buf;
    T_internalDataType* tmp2_buf = (T_internalDataType*)ktmp2_buf;

    stages.stagePreamble(tw_table, tmp1_buf, tmp2_buf, inputx, outputy);
};

template <typename TT_DATA,
          typename TT_OUT_DATA,
          typename TT_TWIDDLE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelFFTClass<TT_DATA,
                                TT_OUT_DATA,
                                TT_TWIDDLE,
                                FFT64_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::kernelFFT(input_window<TT_DATA>* __restrict inputx,
                                                            output_window<TT_OUT_DATA>* __restrict outputy) {
    typedef typename std::conditional<std::is_same<TT_DATA, cint16>::value, cint32_t, TT_DATA>::type T_internalDataType;
    TT_DATA* xbuff = (TT_DATA*)inputx->ptr;
    T_internalDataType* tmp1_buf = (T_internalDataType*)ktmp1_buf;
    T_internalDataType* tmp2_buf =
        fnUsePingPongIntBuffer<TT_DATA>() ? (T_internalDataType*)ktmp2_buf : (T_internalDataType*)xbuff;

    stages.stagePreamble(tw_table, tmp1_buf, tmp2_buf, inputx, outputy);
};

template <typename TT_OUT_DATA,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelFFTClass<cint16,
                                TT_OUT_DATA,
                                cint16,
                                FFT64_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::kernelFFT(input_window<cint16>* __restrict inputx,
                                                            output_window<TT_OUT_DATA>* __restrict outputy) {
    typedef cint16 TT_DATA;
    typedef cint32_t T_internalDataType;
    cint16* xbuff = (cint16*)inputx->ptr;
    TT_OUT_DATA* obuff = (TT_OUT_DATA*)outputy->ptr;
    T_internalDataType* tmp1_buf = (T_internalDataType*)ktmp1_buf;
    T_internalDataType* tmp2_buf = (T_internalDataType*)ktmp2_buf;

    stages.stagePreamble(tw_table, tmp1_buf, tmp2_buf, inputx, outputy);
};

template <typename TT_DATA,
          typename TT_OUT_DATA,
          typename TT_TWIDDLE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelFFTClass<TT_DATA,
                                TT_OUT_DATA,
                                TT_TWIDDLE,
                                FFT32_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::kernelFFT(input_window<TT_DATA>* __restrict inputx,
                                                            output_window<TT_OUT_DATA>* __restrict outputy) {
    typedef typename std::conditional<std::is_same<TT_DATA, cint16>::value, cint32_t, TT_DATA>::type T_internalDataType;
    TT_DATA* xbuff = (TT_DATA*)inputx->ptr;
    T_internalDataType* tmp1_buf = (T_internalDataType*)ktmp1_buf;
    T_internalDataType* tmp2_buf =
        fnUsePingPongIntBuffer<TT_DATA>() ? (T_internalDataType*)ktmp2_buf : (T_internalDataType*)xbuff;

    stages.stagePreamble(tw_table, tmp1_buf, tmp2_buf, inputx, outputy);
};

template <typename TT_OUT_DATA,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelFFTClass<cint16,
                                TT_OUT_DATA,
                                cint16,
                                FFT32_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::kernelFFT(input_window<cint16>* __restrict inputx,
                                                            output_window<TT_OUT_DATA>* __restrict outputy) {
    typedef cint16 TT_DATA;
    typedef cint32_t T_internalDataType;
    cint16* xbuff = (cint16*)inputx->ptr;
    TT_OUT_DATA* obuff = (TT_OUT_DATA*)outputy->ptr;
    T_internalDataType* tmp1_buf = (T_internalDataType*)ktmp1_buf;
    T_internalDataType* tmp2_buf = (T_internalDataType*)ktmp2_buf;

    stages.stagePreamble(tw_table, tmp1_buf, tmp2_buf, inputx, outputy);
};

template <typename TT_DATA,
          typename TT_OUT_DATA,
          typename TT_TWIDDLE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelFFTClass<TT_DATA,
                                TT_OUT_DATA,
                                TT_TWIDDLE,
                                FFT16_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::kernelFFT(input_window<TT_DATA>* __restrict inputx,
                                                            output_window<TT_OUT_DATA>* __restrict outputy) {
    const unsigned int TP_POINT_SIZE = FFT16_SIZE;

    typedef typename std::conditional<std::is_same<TT_DATA, cint16>::value, cint32_t, TT_DATA>::type T_internalDataType;
    TT_DATA* xbuff = (TT_DATA*)inputx->ptr;
    TT_OUT_DATA* obuff = (TT_OUT_DATA*)outputy->ptr;
    T_internalDataType* tmp1_buf = (T_internalDataType*)ktmp1_buf;
    // The following break with the pattern for other point sizes is a workaround for a compiler issue.
    T_internalDataType chess_storage(% chess_alignof(v4cfloat)) tmp2_buf[FFT16_SIZE]; // must be 256 bit aligned
    T_internalDataType chess_storage(% chess_alignof(v4cfloat)) tmp3_buf[FFT16_SIZE]; // must be 256 bit aligned

    bool inv = TP_FFT_NIFFT == 1 ? false : true;

    // This code should be moved to the constructor preferably
    set_rnd(rnd_pos_inf); // Match the twiddle round mode of Matlab.
    set_sat();            // do saturate.
    for (int iter = 0; iter < TP_WINDOW_VSIZE / TP_POINT_SIZE; iter++)
        chess_prepare_for_pipelining chess_loop_range(TP_WINDOW_VSIZE / TP_POINT_SIZE, ) {
            if
                constexpr(std::is_same<TT_DATA, cfloat>::value) {
                    stage0_radix2_dit<cfloat, cfloat, cfloat>((cfloat*)xbuff, (cfloat*)tw1, FFT16_SIZE, FFT_8, 0,
                                                              (cfloat*)tmp1_buf, inv);
                    stage0_radix2_dit<cfloat, cfloat, cfloat>((cfloat*)tmp1_buf, (cfloat*)tw2, FFT16_SIZE, FFT_4, 0,
                                                              (cfloat*)tmp2_buf, inv);
                    stage1_radix2_dit<cfloat, cfloat, cfloat>((cfloat*)tmp2_buf, (cfloat*)tw4, FFT16_SIZE, FFT_2, 0,
                                                              (cfloat*)tmp3_buf, inv);
                    stage2_radix2_dit<cfloat, cfloat, cfloat>((cfloat*)tmp3_buf, (cfloat*)tw8, FFT16_SIZE, FFT_1,
                                                              TP_SHIFT, (cfloat*)obuff, inv); // r is not used.
                }
            else {
                stages.calc(xbuff, tw_table, tmp1_buf, tmp2_buf, obuff);
            }
            xbuff += TP_POINT_SIZE;
            obuff += TP_POINT_SIZE;
        }
};

template <typename TT_OUT_DATA,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
INLINE_DECL void kernelFFTClass<cint16,
                                TT_OUT_DATA,
                                cint16,
                                FFT16_SIZE,
                                TP_FFT_NIFFT,
                                TP_SHIFT,
                                TP_START_RANK,
                                TP_END_RANK,
                                TP_DYN_PT_SIZE,
                                TP_WINDOW_VSIZE>::kernelFFT(input_window<cint16>* __restrict inputx,
                                                            output_window<TT_OUT_DATA>* __restrict outputy) {
    const unsigned int TP_POINT_SIZE = FFT16_SIZE;
    typedef cint16 TT_DATA;

    typedef cint32_t T_internalDataType;
    cint16* xbuff = (cint16*)inputx->ptr;
    TT_OUT_DATA* obuff = (TT_OUT_DATA*)outputy->ptr;
    T_internalDataType* tmp1_buf = (T_internalDataType*)ktmp1_buf;
    T_internalDataType* tmp2_buf = (T_internalDataType*)ktmp2_buf;
    bool inv;
    if
        constexpr(TP_FFT_NIFFT == 1) { inv = false; }
    else {
        inv = true;
    }

    // This code should be moved to the constructor preferably
    set_rnd(rnd_pos_inf); // Match the twiddle round mode of Matlab.
    set_sat();            // do saturate.
    for (int iter = 0; iter < TP_WINDOW_VSIZE / TP_POINT_SIZE; iter++)
        chess_prepare_for_pipelining chess_loop_range(TP_WINDOW_VSIZE / TP_POINT_SIZE, ) {
            stages.calc(xbuff, tw_table, tmp1_buf, tmp2_buf, obuff);
            xbuff += TP_POINT_SIZE;
            obuff += TP_POINT_SIZE;
        }
};

//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          typename TT_OUT_DATA,
          typename TT_TWIDDLE,
          unsigned int TP_POINT_SIZE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_START_RANK,
          unsigned int TP_END_RANK,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
__attribute__((noinline)) void
fft_ifft_dit_1ch<TT_DATA,
                 TT_OUT_DATA,
                 TT_TWIDDLE,
                 TP_POINT_SIZE,
                 TP_FFT_NIFFT,
                 TP_SHIFT,
                 TP_START_RANK,
                 TP_END_RANK,
                 TP_DYN_PT_SIZE,
                 TP_WINDOW_VSIZE>::fftMain(input_window<TT_DATA>* __restrict inWindow,
                                           output_window<TT_OUT_DATA>* __restrict outWindow) {
    m_fftKernel.kernelFFT(inWindow, outWindow);
};
}
}
}
}
}
