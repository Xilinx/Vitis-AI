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
Widget API cast kernal code.
This file captures the body of run-time code for the kernel class.

Coding conventions
  TT_      template type suffix
  TP_      template parameter suffix
*/

#pragma once
#include <adf.h>

#define __AIEARCH__ 1
#define __AIENGINE__ 1
#define __AIE_API_USE_NATIVE_1024B_VECTOR__
#include "aie_api/aie_adf.hpp"
#include "widget_api_cast.hpp"
#include "widget_api_cast_utils.hpp"

#include "kernel_api_utils.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace widget {
namespace api_cast {
template <typename TT_DATA,
          unsigned int TP_IN_API,
          unsigned int TP_OUT_API,
          unsigned int TP_NUM_INPUTS,
          unsigned int TP_WINDOW_VSIZE,
          unsigned int TP_NUM_OUTPUT_CLONES>
INLINE_DECL void
kernelClass<TT_DATA, TP_IN_API, TP_OUT_API, TP_NUM_INPUTS, TP_WINDOW_VSIZE, TP_NUM_OUTPUT_CLONES>::kernelClassMain(
    T_inputIF<TT_DATA, TP_IN_API> inInterface, T_outputIF<TT_DATA, TP_OUT_API> outInterface) {
    T_buff_256b<TT_DATA> readVal;
    constexpr int kLsize = TP_WINDOW_VSIZE * sizeof(TT_DATA) / 32;
    for (int i = 0; i < kLsize; i++) chess_prepare_for_pipelining chess_loop_range(kLsize, ) {
            readVal = window_readincr_256b(inInterface.inWindow0);
            window_writeincr(outInterface.outWindow0, readVal.val);
            if
                constexpr(TP_NUM_OUTPUT_CLONES >= 2) window_writeincr(outInterface.outWindow1, readVal.val);
            if
                constexpr(TP_NUM_OUTPUT_CLONES >= 3) window_writeincr(outInterface.outWindow2, readVal.val);
        }
};

template <typename TT_DATA, unsigned int TP_NUM_INPUTS, unsigned int TP_WINDOW_VSIZE, unsigned int TP_NUM_OUTPUT_CLONES>
INLINE_DECL void
kernelClass<TT_DATA, kStreamAPI, kWindowAPI, TP_NUM_INPUTS, TP_WINDOW_VSIZE, TP_NUM_OUTPUT_CLONES>::kernelClassMain(
    T_inputIF<TT_DATA, kStreamAPI> inInterface, T_outputIF<TT_DATA, kWindowAPI> outInterface) {
    T_buff_128b<TT_DATA> readVal1;
    T_buff_128b<TT_DATA> readVal2;
    T_buff_256b<TT_DATA> writeVal;
    if
        constexpr(TP_NUM_INPUTS == 1) {
            constexpr int kLsize = TP_WINDOW_VSIZE * sizeof(TT_DATA) / 16;
            for (int i = 0; i < kLsize; i++) chess_prepare_for_pipelining chess_loop_range(kLsize, ) {
                    readVal1 = stream_readincr_128b(inInterface.inStream0);
                    window_writeincr(outInterface.outWindow0, readVal1.val);
                    if
                        constexpr(TP_NUM_OUTPUT_CLONES >= 2) window_writeincr(outInterface.outWindow1, readVal1.val);
                    if
                        constexpr(TP_NUM_OUTPUT_CLONES >= 3) window_writeincr(outInterface.outWindow2, readVal1.val);
                    if
                        constexpr(TP_NUM_OUTPUT_CLONES >= 4) window_writeincr(outInterface.outWindow3, readVal1.val);
                }
        }
    else { // TP_NUM_INPUTS==2
        constexpr int kLsize = TP_WINDOW_VSIZE * sizeof(TT_DATA) / 32;
        for (int i = 0; i < kLsize; i++) chess_prepare_for_pipelining chess_loop_range(kLsize, ) {
                readVal1 = stream_readincr_128b(inInterface.inStream0);
                readVal2 = stream_readincr_128b(inInterface.inStream1);
                writeVal.val.insert(0, readVal1.val);
                writeVal.val.insert(1, readVal2.val);
                window_writeincr(outInterface.outWindow0, writeVal.val);
                if
                    constexpr(TP_NUM_OUTPUT_CLONES >= 2) window_writeincr(outInterface.outWindow1, writeVal.val);
                if
                    constexpr(TP_NUM_OUTPUT_CLONES >= 3) window_writeincr(outInterface.outWindow2, writeVal.val);
                if
                    constexpr(TP_NUM_OUTPUT_CLONES >= 4) window_writeincr(outInterface.outWindow3, writeVal.val);
            }
        if (TP_WINDOW_VSIZE * sizeof(TT_DATA) / 16 % 2 == 1) { // odd number of 128b chunks
            readVal1 = stream_readincr_128b(inInterface.inStream0);
            window_writeincr(outInterface.outWindow0, readVal1.val);
            if
                constexpr(TP_NUM_OUTPUT_CLONES >= 2) window_writeincr(outInterface.outWindow1, readVal1.val);
            if
                constexpr(TP_NUM_OUTPUT_CLONES >= 3) window_writeincr(outInterface.outWindow2, readVal1.val);
            if
                constexpr(TP_NUM_OUTPUT_CLONES >= 4) window_writeincr(outInterface.outWindow3, readVal1.val);
        }
    }
};

template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE, unsigned int TP_NUM_OUTPUT_CLONES>
INLINE_DECL void
kernelClass<TT_DATA, kWindowAPI, kStreamAPI, 1, TP_WINDOW_VSIZE, TP_NUM_OUTPUT_CLONES>::kernelClassMain(
    T_inputIF<TT_DATA, kWindowAPI> inInterface, T_outputIF<TT_DATA, kStreamAPI> outInterface) {
    T_buff_128b<TT_DATA> readVal;
    T_buff_128b<TT_DATA> writeVal;
    constexpr unsigned int kSamplesIn128b = 16 / sizeof(TT_DATA);
    if
        constexpr(TP_NUM_OUTPUT_CLONES == 1) {
            constexpr int kLsize = TP_WINDOW_VSIZE / kSamplesIn128b;
            for (int i = 0; i < kLsize; i++) chess_prepare_for_pipelining chess_loop_range(kLsize, ) {
                    readVal = window_readincr_128b(inInterface.inWindow0);
                    writeincr(outInterface.outStream0, readVal.val);
                }
        }
    else { // TP_NUM_OUTPUT_CLONES==2
        T_buff_256b<TT_DATA> readVal;
        T_buff_128b<TT_DATA> writeVal;
        constexpr int kLsize = TP_WINDOW_VSIZE / (kSamplesIn128b * 2);
        for (int i = 0; i < kLsize; i++) chess_prepare_for_pipelining chess_loop_range(kLsize, ) {
                readVal = window_readincr_256b(inInterface.inWindow0);
                writeVal.val = readVal.val.template extract<kSamplesIn128b>(0);
                writeincr(outInterface.outStream0, writeVal.val);
                writeVal.val = readVal.val.template extract<kSamplesIn128b>(1);
                writeincr(outInterface.outStream1, writeVal.val);
            }
        if (TP_WINDOW_VSIZE / kSamplesIn128b % 2 == 1) { // odd number of 128b chunks
            readVal = window_readincr_256b(inInterface.inWindow0);
            writeVal.val = readVal.val.template extract<kSamplesIn128b>(0);
            writeincr(outInterface.outStream0, writeVal.val);
        }
    }
};

//-------------------------------------------------------------------------------------------------------
// This is the base specialization of the main class for when there is only one window in and out
template <typename TT_DATA,
          unsigned int TP_IN_API,
          unsigned int TP_OUT_API,
          unsigned int TP_NUM_INPUTS,
          unsigned int TP_WINDOW_VSIZE,
          unsigned int TP_NUM_OUTPUT_CLONES>
__attribute__((noinline)) void
    widget_api_cast<TT_DATA, TP_IN_API, TP_OUT_API, TP_NUM_INPUTS, TP_WINDOW_VSIZE, TP_NUM_OUTPUT_CLONES>::
        transferData // transferData is QoR hook
    (input_window<TT_DATA>* __restrict inWindow0, output_window<TT_DATA>* __restrict outWindow0) {
    T_inputIF<TT_DATA, TP_IN_API> inInterface;
    T_outputIF<TT_DATA, TP_OUT_API> outInterface;
    inInterface.inWindow0 = inWindow0;
    outInterface.outWindow0 = outWindow0;
    this->kernelClassMain(inInterface, outInterface);
};

// window API in and out, 2 outputs
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
__attribute__((noinline)) void widget_api_cast<TT_DATA, kWindowAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 2>::transferData(
    input_window<TT_DATA>* __restrict inWindow0,
    output_window<TT_DATA>* __restrict outWindow0,
    output_window<TT_DATA>* __restrict outWindow1) {
    constexpr unsigned int TP_NUM_OUTPUTS = 3;
    T_inputIF<TT_DATA, kWindowAPI> inInterface;
    T_outputIF<TT_DATA, kWindowAPI> outInterface;
    inInterface.inWindow0 = inWindow0;
    outInterface.outWindow0 = outWindow0;
    outInterface.outWindow1 = outWindow1;
    this->kernelClassMain(inInterface, outInterface);
};

// window API in and out, 3 outputs
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
__attribute__((noinline)) void widget_api_cast<TT_DATA, kWindowAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 3>::transferData(
    input_window<TT_DATA>* __restrict inWindow0,
    output_window<TT_DATA>* __restrict outWindow0,
    output_window<TT_DATA>* __restrict outWindow1,
    output_window<TT_DATA>* __restrict outWindow2) {
    constexpr unsigned int TP_NUM_OUTPUTS = 3;
    T_inputIF<TT_DATA, kWindowAPI> inInterface;
    T_outputIF<TT_DATA, kWindowAPI> outInterface;
    inInterface.inWindow0 = inWindow0;
    outInterface.outWindow0 = outWindow0;
    outInterface.outWindow1 = outWindow1;
    outInterface.outWindow2 = outWindow2;
    this->kernelClassMain(inInterface, outInterface);
};

// stream API in and out, 1 outputs
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
__attribute__((noinline)) void widget_api_cast<TT_DATA, kStreamAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 1>::transferData(
    input_stream<TT_DATA>* __restrict inStream0, output_window<TT_DATA>* __restrict outWindow0) {
    constexpr unsigned int TP_NUM_OUTPUTS = 1;
    T_inputIF<TT_DATA, kStreamAPI> inInterface;
    T_outputIF<TT_DATA, kWindowAPI> outInterface;
    inInterface.inStream0 = inStream0;
    outInterface.outWindow0 = outWindow0;
    this->kernelClassMain(inInterface, outInterface);
};

// stream API in and out, 2 outputs
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
__attribute__((noinline)) void widget_api_cast<TT_DATA, kStreamAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 2>::transferData(
    input_stream<TT_DATA>* __restrict inStream0,
    output_window<TT_DATA>* __restrict outWindow0,
    output_window<TT_DATA>* __restrict outWindow1) {
    constexpr unsigned int TP_NUM_OUTPUTS = 2;
    T_inputIF<TT_DATA, kStreamAPI> inInterface;
    T_outputIF<TT_DATA, kWindowAPI> outInterface;
    inInterface.inStream0 = inStream0;
    outInterface.outWindow0 = outWindow0;
    outInterface.outWindow1 = outWindow1;
    this->kernelClassMain(inInterface, outInterface);
};

// stream API in and out, 3 outputs
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
__attribute__((noinline)) void widget_api_cast<TT_DATA, kStreamAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 3>::transferData(
    input_stream<TT_DATA>* __restrict inStream0,
    output_window<TT_DATA>* __restrict outWindow0,
    output_window<TT_DATA>* __restrict outWindow1,
    output_window<TT_DATA>* __restrict outWindow2) {
    constexpr unsigned int TP_NUM_OUTPUTS = 3;
    T_inputIF<TT_DATA, kStreamAPI> inInterface;
    T_outputIF<TT_DATA, kWindowAPI> outInterface;
    inInterface.inStream0 = inStream0;
    outInterface.outWindow0 = outWindow0;
    outInterface.outWindow1 = outWindow1;
    outInterface.outWindow2 = outWindow2;
    this->kernelClassMain(inInterface, outInterface);
};

// stream API in and out, 4 outputs
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
__attribute__((noinline)) void widget_api_cast<TT_DATA, kStreamAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 4>::transferData(
    input_stream<TT_DATA>* __restrict inStream0,
    output_window<TT_DATA>* __restrict outWindow0,
    output_window<TT_DATA>* __restrict outWindow1,
    output_window<TT_DATA>* __restrict outWindow2,
    output_window<TT_DATA>* __restrict outWindow3) {
    constexpr unsigned int TP_NUM_OUTPUTS = 4;
    T_inputIF<TT_DATA, kStreamAPI> inInterface;
    T_outputIF<TT_DATA, kWindowAPI> outInterface;
    inInterface.inStream0 = inStream0;
    outInterface.outWindow0 = outWindow0;
    outInterface.outWindow1 = outWindow1;
    outInterface.outWindow2 = outWindow2;
    outInterface.outWindow3 = outWindow3;
    this->kernelClassMain(inInterface, outInterface);
};

// dual stream in
// stream to window, 2 in 1 out
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
__attribute__((noinline)) void widget_api_cast<TT_DATA, kStreamAPI, kWindowAPI, 2, TP_WINDOW_VSIZE, 1>::transferData(
    input_stream<TT_DATA>* __restrict inStream0,
    input_stream<TT_DATA>* __restrict inStream1,
    output_window<TT_DATA>* __restrict outWindow0) {
    constexpr unsigned int TP_NUM_OUTPUTS = 1;
    T_inputIF<TT_DATA, kStreamAPI> inInterface;
    T_outputIF<TT_DATA, kWindowAPI> outInterface;
    inInterface.inStream0 = inStream0;
    inInterface.inStream1 = inStream1;
    outInterface.outWindow0 = outWindow0;
    this->kernelClassMain(inInterface, outInterface);
};

// stream to window, 2 in 2 out
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
__attribute__((noinline)) void widget_api_cast<TT_DATA, kStreamAPI, kWindowAPI, 2, TP_WINDOW_VSIZE, 2>::transferData(
    input_stream<TT_DATA>* __restrict inStream0,
    input_stream<TT_DATA>* __restrict inStream1,
    output_window<TT_DATA>* __restrict outWindow0,
    output_window<TT_DATA>* __restrict outWindow1) {
    constexpr unsigned int TP_NUM_OUTPUTS = 2;
    T_inputIF<TT_DATA, kStreamAPI> inInterface;
    T_outputIF<TT_DATA, kWindowAPI> outInterface;
    inInterface.inStream0 = inStream0;
    inInterface.inStream1 = inStream1;
    outInterface.outWindow0 = outWindow0;
    outInterface.outWindow1 = outWindow1;
    this->kernelClassMain(inInterface, outInterface);
};

// stream to window, 2 in 3 out
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
__attribute__((noinline)) void widget_api_cast<TT_DATA, kStreamAPI, kWindowAPI, 2, TP_WINDOW_VSIZE, 3>::transferData(
    input_stream<TT_DATA>* __restrict inStream0,
    input_stream<TT_DATA>* __restrict inStream1,
    output_window<TT_DATA>* __restrict outWindow0,
    output_window<TT_DATA>* __restrict outWindow1,
    output_window<TT_DATA>* __restrict outWindow2) {
    constexpr unsigned int TP_NUM_OUTPUTS = 3;
    T_inputIF<TT_DATA, kStreamAPI> inInterface;
    T_outputIF<TT_DATA, kWindowAPI> outInterface;
    inInterface.inStream0 = inStream0;
    inInterface.inStream1 = inStream1;
    outInterface.outWindow0 = outWindow0;
    outInterface.outWindow1 = outWindow1;
    outInterface.outWindow2 = outWindow2;
    this->kernelClassMain(inInterface, outInterface);
};

// stream to window, 2 in 3 out
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
__attribute__((noinline)) void widget_api_cast<TT_DATA, kStreamAPI, kWindowAPI, 2, TP_WINDOW_VSIZE, 4>::transferData(
    input_stream<TT_DATA>* __restrict inStream0,
    input_stream<TT_DATA>* __restrict inStream1,
    output_window<TT_DATA>* __restrict outWindow0,
    output_window<TT_DATA>* __restrict outWindow1,
    output_window<TT_DATA>* __restrict outWindow2,
    output_window<TT_DATA>* __restrict outWindow3) {
    constexpr unsigned int TP_NUM_OUTPUTS = 4;
    T_inputIF<TT_DATA, kStreamAPI> inInterface;
    T_outputIF<TT_DATA, kWindowAPI> outInterface;
    inInterface.inStream0 = inStream0;
    inInterface.inStream1 = inStream1;
    outInterface.outWindow0 = outWindow0;
    outInterface.outWindow1 = outWindow1;
    outInterface.outWindow2 = outWindow2;
    outInterface.outWindow3 = outWindow3;
    this->kernelClassMain(inInterface, outInterface);
};

// Window to Stream
// window to stream, 1 to 1
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
__attribute__((noinline)) void widget_api_cast<TT_DATA, kWindowAPI, kStreamAPI, 1, TP_WINDOW_VSIZE, 1>::transferData(
    input_window<TT_DATA>* __restrict inWindow0, output_stream<TT_DATA>* __restrict outStream0) {
    T_inputIF<TT_DATA, kWindowAPI> inInterface;
    T_outputIF<TT_DATA, kStreamAPI> outInterface;
    inInterface.inWindow0 = inWindow0;
    outInterface.outStream0 = outStream0;
    this->kernelClassMain(inInterface, outInterface);
};

// window to stream, 1 to 2
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
__attribute__((noinline)) void widget_api_cast<TT_DATA, kWindowAPI, kStreamAPI, 1, TP_WINDOW_VSIZE, 2>::transferData(
    input_window<TT_DATA>* __restrict inWindow0,
    output_stream<TT_DATA>* __restrict outStream0,
    output_stream<TT_DATA>* __restrict outStream1) {
    T_inputIF<TT_DATA, kWindowAPI> inInterface;
    T_outputIF<TT_DATA, kStreamAPI> outInterface;
    inInterface.inWindow0 = inWindow0;
    outInterface.outStream0 = outStream0;
    outInterface.outStream1 = outStream1;
    this->kernelClassMain(inInterface, outInterface);
};
}
}
}
}
}
