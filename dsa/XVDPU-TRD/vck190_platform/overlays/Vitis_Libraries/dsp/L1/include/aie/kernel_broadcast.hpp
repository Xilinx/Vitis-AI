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
#pragma once

#ifndef _DSPLIB_KERNEL_BROADCAST_HPP_
#define _DSPLIB_KERNEL_BROADCAST_HPP_

// This file holds sets of templated types and overloaded (or template specialized) functions
// for use by multiple kernels.
// Functions in this file as a rule use intrinsics from a single set. For instance, a set
// may contain a MAC with pre-add which uses a single 1024 bit buffer for both forward
// and reverse data. In cases where a library element has to use an intrinsic which differs
// by more than the types used for some combinations of library element parameter types
// then the set of templatized functions will be particular to that library element and should
// therefore be in <library_element>_utils.hpp

#include <stdio.h>
#include <adf.h>
#include "fir_utils.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace fir {

// To optimize performance, 256-bit vectors are copied, so storage element must be padded to 256-bits.
template <typename TT_DATA, unsigned int TP_INPUT_WINDOW_VSIZE, bool TP_CASC_IN>
inline void windowBroadcast(input_window<TT_DATA>* inWindow, output_stream_cacc48* outCascade) {
    using buff_type = typename T_buff_256b<int32>::v_type;
    buff_type buff256;
    buff_type* restrict inWindowPtr = (buff_type*)inWindow->head;
    // const int samplesPer256Buff = sizeof(buff_type)/sizeof(TT_DATA);
    const int samplesPer256Buff = 256 / 8 / sizeof(TT_DATA);
    static_assert(TP_INPUT_WINDOW_VSIZE % samplesPer256Buff == 0, "Error: Window size must be a multiple of 256-bits");

    for (int i = 0; i < TP_INPUT_WINDOW_VSIZE; i += samplesPer256Buff) {
        // copy 256-bit vector at a time
        buff256 = *inWindowPtr++;
        put_mcd(buff256);
    }
}

// To optimize performance, 256-bit vectors are copied, so storage element must be padded to 256-bits.
template <typename TT_DATA, unsigned int TP_INPUT_WINDOW_VSIZE, bool TP_CASC_IN>
inline void windowBroadcast(input_stream_cacc48* inCascade,
                            output_stream_cacc48* outCascade,
                            input_window<TT_DATA>* outWindow) {
    using buff_type = typename T_buff_256b<int32>::v_type;
    buff_type buff256;
    // buff_type* restrict inWindowPtr   = (buff_type*) inWindow->head;
    buff_type* restrict outWindowPtr = (buff_type*)outWindow->head;
    // const int samplesPer256Buff = sizeof(buff_type)/sizeof(TT_DATA);
    const int samplesPer256Buff = 256 / 8 / sizeof(TT_DATA);
    static_assert(TP_INPUT_WINDOW_VSIZE % samplesPer256Buff == 0, "Error: Window size must be a multiple of 256-bits");

    for (int i = 0; i < TP_INPUT_WINDOW_VSIZE; i += samplesPer256Buff) {
        // copy 256-bit vector at a time
        buff256 = get_scd_v8int32();
        put_mcd(buff256);
        *outWindowPtr++ = buff256;
    }
}

// To optimize performance, 256-bit vectors are copied, so storage element must be padded to 256-bits.
template <typename TT_DATA, unsigned int TP_INPUT_WINDOW_VSIZE, bool TP_CASC_IN>
inline void windowBroadcast(input_stream_cacc48* inCascade, input_window<TT_DATA>* outWindow) {
    using buff_type = typename T_buff_256b<int32>::v_type;
    buff_type buff256;
    buff_type* restrict outWindowPtr = (buff_type*)outWindow->head;
    // const int samplesPer256Buff = sizeof(buff_type)/sizeof(TT_DATA);
    const int samplesPer256Buff = 256 / 8 / sizeof(TT_DATA);
    static_assert(TP_INPUT_WINDOW_VSIZE % samplesPer256Buff == 0, "Error: Window size must be a multiple of 256-bits");

    for (int i = 0; i < TP_INPUT_WINDOW_VSIZE; i += samplesPer256Buff) {
        // copy 256-bit vector at a time
        buff256 = get_scd_v8int32();
        *outWindowPtr++ = buff256;
    }
}

// Window Lock Acquire. Overloaded with IO interface.
template <typename TT_DATA, unsigned int TP_DUAL_IP = 0>
inline void windowAcquire(T_inputIF<CASC_IN_FALSE, TT_DATA, TP_DUAL_IP> inInterface) {
    // Do nothing.
    // When Cascade is not present, Window is a sync connection
    // No need for internal syncronization mechanisms.
}
// Window Lock Acquire. Overloaded with IO interface.
template <typename TT_DATA, unsigned int TP_DUAL_IP = 0>
inline void windowAcquire(T_inputIF<CASC_IN_TRUE, TT_DATA, TP_DUAL_IP> inInterface) {
    // acquire a lock on async input window
    // window_acquire(inInterface.inWindow);
    inInterface.inWindow->ptr = inInterface.inWindow->head;
}

// Window Lock Release. Overloaded with IO interface.
template <typename TT_DATA, unsigned int TP_DUAL_IP = 0>
inline void windowRelease(T_inputIF<CASC_IN_FALSE, TT_DATA, TP_DUAL_IP> inInterface) {
    // Do nothing.
    // When Cascade is not present, Window is a sync connection
    // No need for internal syncronization mechanisms.
}

// Window Lock Release. Overloaded with IO interface.
template <typename TT_DATA, unsigned int TP_DUAL_IP = 0>
inline void windowRelease(T_inputIF<CASC_IN_TRUE, TT_DATA, TP_DUAL_IP> inInterface) {
    // acquire a lock on async broadcast window
    // window_release(inInterface.inWindow);
}

// Wrapper. Overloaded with IO interface.
template <typename TT_DATA, unsigned int TP_INPUT_WINDOW_VSIZE, unsigned int TP_DUAL_IP = 0>
inline void windowBroadcast(T_inputIF<CASC_IN_FALSE, TT_DATA, TP_DUAL_IP> inInterface,
                            T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface) {
    // Do nothing.
}

// Wrapper. Overloaded with IO interface.
template <typename TT_DATA, unsigned int TP_INPUT_WINDOW_VSIZE, unsigned int TP_DUAL_IP = 0>
inline void windowBroadcast(T_inputIF<CASC_IN_FALSE, TT_DATA, TP_DUAL_IP> inInterface,
                            T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface) {
    windowBroadcast<TT_DATA, TP_INPUT_WINDOW_VSIZE, CASC_IN_FALSE>(inInterface.inWindow, outInterface.outCascade);
    // chess_memory_fence();
}

// Wrapper. Overloaded with IO interface.
template <typename TT_DATA, unsigned int TP_INPUT_WINDOW_VSIZE, unsigned int TP_DUAL_IP = 0>
inline void windowBroadcast(T_inputIF<CASC_IN_TRUE, TT_DATA, TP_DUAL_IP> inInterface,
                            T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface) {
    windowBroadcast<TT_DATA, TP_INPUT_WINDOW_VSIZE, CASC_IN_TRUE>(inInterface.inCascade, outInterface.outCascade,
                                                                  inInterface.inWindow);
}

// Wrapper. Overloaded with IO interface.
template <typename TT_DATA, unsigned int TP_INPUT_WINDOW_VSIZE, unsigned int TP_DUAL_IP = 0>
inline void windowBroadcast(T_inputIF<CASC_IN_TRUE, TT_DATA, TP_DUAL_IP> inInterface,
                            T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface) {
    windowBroadcast<TT_DATA, TP_INPUT_WINDOW_VSIZE, CASC_IN_FALSE>(inInterface.inCascade, inInterface.inWindow);
}
}
}
}
} // namespaces
#endif // _DSPLIB_KERNEL_BROADCAST_HPP_
