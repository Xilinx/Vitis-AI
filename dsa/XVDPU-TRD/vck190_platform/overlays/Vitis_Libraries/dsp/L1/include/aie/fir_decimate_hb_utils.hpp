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
#ifndef _DSPLIB_FIR_DECIMATE_HB_UTILS_HPP_
#define _DSPLIB_FIR_DECIMATE_HB_UTILS_HPP_

/*
Half band decimation FIR Utilities.
This file contains sets of overloaded, templatized and specialized templatized functions for use
by the main kernel class and run-time function. These functions are separate from the traits file
because they are purely for kernel use, not graph level compilation.
*/

/*
Note regarding Input Vector register size selection.
The type of the data register is not a constant 1024b or 512b (1buff or 2 buff architecture)
because for 2 integer types the number of samples required for an operation exceeds the capacity
of the 512b buffer used by the symmetrical mul/mac commands. The workaround is to use 2 standard mul/macs
using a 1024b.
For float types, there are no mul/macs with preadd (symmetrical) so the same approach as above is used.
*/

#include <stdio.h>
#include <adf.h>
#include "fir_decimate_hb.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace decimate_hb {
//---------------------------------------------------------------------------------------------------
// Type definitions

// Input Vector Register type
template <typename T_D, typename T_C, eArchType TP_ARCH>
struct T_buff_FirDecHb {};
template <>
struct T_buff_FirDecHb<int16, int16, kArch1Buff> {
    v64int16 val = null_v64int16();
};
template <>
struct T_buff_FirDecHb<cint16, int16, kArch1Buff> {
    v32cint16 val = null_v32cint16();
};
template <>
struct T_buff_FirDecHb<cint16, cint16, kArch1Buff> {
    v32cint16 val = null_v32cint16();
};
template <>
struct T_buff_FirDecHb<int32, int16, kArch1Buff> {
    v32int32 val = null_v32int32();
};
template <>
struct T_buff_FirDecHb<int32, int32, kArch1Buff> {
    v32int32 val = null_v32int32();
};
template <>
struct T_buff_FirDecHb<cint32, int16, kArch1Buff> {
    v16cint32 val = null_v16cint32();
};
template <>
struct T_buff_FirDecHb<cint32, cint16, kArch1Buff> {
    v16cint32 val = null_v16cint32();
};
template <>
struct T_buff_FirDecHb<cint32, int32, kArch1Buff> {
    v16cint32 val = null_v16cint32();
};
template <>
struct T_buff_FirDecHb<cint32, cint32, kArch1Buff> {
    v16cint32 val = null_v16cint32();
};
template <>
struct T_buff_FirDecHb<float, float, kArch1Buff> {
    v32float val = null_v32float();
};
template <>
struct T_buff_FirDecHb<cfloat, float, kArch1Buff> {
    v16cfloat val = null_v16cfloat();
};
template <>
struct T_buff_FirDecHb<cfloat, cfloat, kArch1Buff> {
    v16cfloat val = null_v16cfloat();
};
template <>
struct T_buff_FirDecHb<int16, int16, kArch2Buff> {
    v32int16 val = null_v32int16();
};
template <>
struct T_buff_FirDecHb<cint16, int16, kArch2Buff> {
    v16cint16 val = null_v16cint16();
};
template <>
struct T_buff_FirDecHb<cint16, cint16, kArch2Buff> {
    v16cint16 val = null_v16cint16();
};
template <>
struct T_buff_FirDecHb<int32, int16, kArch2Buff> {
    v32int32 val = null_v32int32();
}; // See note regarding Input Vector register size selection.
template <>
struct T_buff_FirDecHb<int32, int32, kArch2Buff> {
    v16int32 val = null_v16int32();
};
template <>
struct T_buff_FirDecHb<cint32, int16, kArch2Buff> {
    v16cint32 val = null_v16cint32();
}; // See note regarding Input Vector register size selection.
template <>
struct T_buff_FirDecHb<cint32, cint16, kArch2Buff> {
    v8cint32 val = null_v8cint32();
};
template <>
struct T_buff_FirDecHb<cint32, int32, kArch2Buff> {
    v8cint32 val = null_v8cint32();
};
template <>
struct T_buff_FirDecHb<cint32, cint32, kArch2Buff> {
    v8cint32 val = null_v8cint32();
};
template <>
struct T_buff_FirDecHb<float, float, kArch2Buff> {
    v32float val = null_v32float();
};
template <>
struct T_buff_FirDecHb<cfloat, float, kArch2Buff> {
    v16cfloat val = null_v16cfloat();
};
template <>
struct T_buff_FirDecHb<cfloat, cfloat, kArch2Buff> {
    v16cfloat val = null_v16cfloat();
};

// Generic 768-bit wide accumulator type
template <typename T_D, typename T_C>
struct T_acc768 {};
template <>
struct T_acc768<int16, int16> {
    v16acc48 val = null_v16acc48();
};
template <>
struct T_acc768<cint16, int16> {
    v8cacc48 val = null_v8cacc48();
};
template <>
struct T_acc768<cint16, cint16> {
    v8cacc48 val = null_v8cacc48();
};
template <>
struct T_acc768<int32, int16> {
    v8acc80 val = null_v8acc80();
};
template <>
struct T_acc768<int32, int32> {
    v8acc80 val = null_v8acc80();
};
template <>
struct T_acc768<cint32, int16> {
    v4cacc80 val = null_v4cacc80();
};
template <>
struct T_acc768<cint32, cint16> {
    v4cacc80 val = null_v4cacc80();
};
template <>
struct T_acc768<cint32, int32> {
    v4cacc80 val = null_v4cacc80();
};
template <>
struct T_acc768<cint32, cint32> {
    v4cacc80 val = null_v4cacc80();
};
template <>
struct T_acc768<float, float> {
    v8float val = null_v8float();
};
template <>
struct T_acc768<cfloat, float> {
    v4cfloat val = null_v4cfloat();
};
template <>
struct T_acc768<cfloat, cfloat> {
    v4cfloat val = null_v4cfloat();
};

// Hald band decimation specific accumulator type
template <typename T_D, typename T_C, eArchType TP_ARCH>
struct T_accFirDecHb {};
template <>
struct T_accFirDecHb<int16, int16, kArch1Buff> {
    v16acc48 val = null_v16acc48();
};
template <>
struct T_accFirDecHb<cint16, int16, kArch1Buff> {
    v4cacc48 val = null_v4cacc48();
    v4cacc48 uval = null_v4cacc48();
};
template <>
struct T_accFirDecHb<cint16, cint16, kArch1Buff> {
    v4cacc48 val = null_v4cacc48();
    v4cacc48 uval = null_v4cacc48();
};
template <>
struct T_accFirDecHb<int32, int16, kArch1Buff> {
    v8acc80 val = null_v8acc80();
};
template <>
struct T_accFirDecHb<int32, int32, kArch1Buff> {
    v4acc80 val = null_v4acc80();
    v4acc80 uval = null_v4acc80();
};
template <>
struct T_accFirDecHb<cint32, int16, kArch1Buff> {
    v4cacc80 val = null_v4cacc80();
};
template <>
struct T_accFirDecHb<cint32, int32, kArch1Buff> {
    v4cacc80 val = null_v4cacc80();
};
template <>
struct T_accFirDecHb<cint32, cint16, kArch1Buff> {
    v4cacc80 val = null_v4cacc80();
};
template <>
struct T_accFirDecHb<cint32, cint32, kArch1Buff> {
    v2cacc80 val = null_v2cacc80();
    v2cacc80 uval = null_v2cacc80();
};
template <>
struct T_accFirDecHb<float, float, kArch1Buff> {
    v8float val = null_v8float();
};
template <>
struct T_accFirDecHb<cfloat, float, kArch1Buff> {
    v4cfloat val = null_v4cfloat();
};
template <>
struct T_accFirDecHb<cfloat, cfloat, kArch1Buff> {
    v4cfloat val = null_v4cfloat();
};
template <>
struct T_accFirDecHb<int16, int16, kArch2Buff> {
    v8acc48 val = null_v8acc48();
};
template <>
struct T_accFirDecHb<cint16, int16, kArch2Buff> {
    v4cacc48 val = null_v4cacc48();
};
template <>
struct T_accFirDecHb<cint16, cint16, kArch2Buff> {
    v4cacc48 val = null_v4cacc48();
};
template <>
struct T_accFirDecHb<int32, int16, kArch2Buff> {
    v8acc80 val = null_v8acc80();
};
template <>
struct T_accFirDecHb<int32, int32, kArch2Buff> {
    v4acc80 val = null_v4acc80();
};
template <>
struct T_accFirDecHb<cint32, int16, kArch2Buff> {
    v4cacc80 val = null_v4cacc80();
};
template <>
struct T_accFirDecHb<cint32, int32, kArch2Buff> {
    v4cacc80 val = null_v4cacc80();
};
template <>
struct T_accFirDecHb<cint32, cint16, kArch2Buff> {
    v4cacc80 val = null_v4cacc80();
};
template <>
struct T_accFirDecHb<cint32, cint32, kArch2Buff> {
    v2cacc80 val = null_v2cacc80();
    v2cacc80 uval = null_v2cacc80();
};
template <>
struct T_accFirDecHb<float, float, kArch2Buff> {
    v8float val = null_v8float();
};
template <>
struct T_accFirDecHb<cfloat, float, kArch2Buff> {
    v4cfloat val = null_v4cfloat();
};
template <>
struct T_accFirDecHb<cfloat, cfloat, kArch2Buff> {
    v4cfloat val = null_v4cfloat();
};

// Generic 128-bit wide output vector type
template <typename T_D>
struct T_outVal128 {};
template <>
struct T_outVal128<int16> {
    v8int16 val;
};
template <>
struct T_outVal128<cint16> {
    v4cint16 val;
};
template <>
struct T_outVal128<int32> {
    v4int32 val;
};
template <>
struct T_outVal128<cint32> {
    v2cint32 val;
};
template <>
struct T_outVal128<float> {
    v4float val;
};
template <>
struct T_outVal128<cfloat> {
    v2cfloat val;
};

// Hald band decimation specific output vector type
template <typename T_D, typename T_C, eArchType TP_ARCH>
struct T_outValFiRDecHb {};
template <>
struct T_outValFiRDecHb<int16, int16, kArch1Buff> {
    v16int16 val;
};
template <>
struct T_outValFiRDecHb<cint16, int16, kArch1Buff> {
    v4cint16 val;
}; // 4 lanes
template <>
struct T_outValFiRDecHb<cint16, cint16, kArch1Buff> {
    v8cint16 val;
};
template <>
struct T_outValFiRDecHb<int32, int16, kArch1Buff> {
    v8int32 val;
};
template <>
struct T_outValFiRDecHb<int32, int32, kArch1Buff> {
    v8int32 val;
};
template <>
struct T_outValFiRDecHb<cint32, int16, kArch1Buff> {
    v4cint32 val;
};
template <>
struct T_outValFiRDecHb<cint32, int32, kArch1Buff> {
    v4cint32 val;
};
template <>
struct T_outValFiRDecHb<cint32, cint16, kArch1Buff> {
    v4cint32 val;
};
template <>
struct T_outValFiRDecHb<cint32, cint32, kArch1Buff> {
    v4cint32 val;
};
template <>
struct T_outValFiRDecHb<float, float, kArch1Buff> {
    v8float val;
};
template <>
struct T_outValFiRDecHb<cfloat, float, kArch1Buff> {
    v4cfloat val;
};
template <>
struct T_outValFiRDecHb<cfloat, cfloat, kArch1Buff> {
    v4cfloat val;
};
template <>
struct T_outValFiRDecHb<int16, int16, kArch2Buff> {
    v16int16 val;
};
template <>
struct T_outValFiRDecHb<cint16, int16, kArch2Buff> {
    v4cint16 val;
};
template <>
struct T_outValFiRDecHb<cint16, cint16, kArch2Buff> {
    v4cint16 val;
};
template <>
struct T_outValFiRDecHb<int32, int16, kArch2Buff> {
    v8int32 val;
};
template <>
struct T_outValFiRDecHb<int32, int32, kArch2Buff> {
    v4int32 val;
};
template <>
struct T_outValFiRDecHb<cint32, int16, kArch2Buff> {
    v4cint32 val;
};
template <>
struct T_outValFiRDecHb<cint32, int32, kArch2Buff> {
    v4cint32 val;
};
template <>
struct T_outValFiRDecHb<cint32, cint16, kArch2Buff> {
    v4cint32 val;
};
template <>
struct T_outValFiRDecHb<cint32, cint32, kArch2Buff> {
    v4cint32 val;
};
template <>
struct T_outValFiRDecHb<float, float, kArch2Buff> {
    v8float val;
};
template <>
struct T_outValFiRDecHb<cfloat, float, kArch2Buff> {
    v4cfloat val;
};
template <>
struct T_outValFiRDecHb<cfloat, cfloat, kArch2Buff> {
    v4cfloat val;
};

//---------------------------------------------------------------------------------------------------
// Functions
template <typename T_D, typename T_C, eArchType TP_ARCH, unsigned int T_SIZE>
inline void fnLoadXIpData(T_buff_FirDecHb<T_D, T_C, TP_ARCH>& buff, unsigned int splice, input_window<T_D>* inWindow) {
    if
        constexpr(T_SIZE == 256) {
            T_buff_256b<T_D> readData;
            const short kSpliceRange = 4;
            // Read the next slice THEN increment
            readData = window_readincr_256b<T_D>(inWindow);
            buff.val = upd_w(buff.val, splice % kSpliceRange, readData.val);
        }
    else {
        T_buff_128b<T_D> readData;
        const short kSpliceRange = 8;
        readData = window_readincr_128b<T_D>(inWindow);
        buff.val = upd_v(buff.val, splice % kSpliceRange, readData.val);
    }
    // return retVal;
};

// Function to load reverse direction data
template <typename T_D, typename T_C, eArchType TP_ARCH, unsigned int T_SIZE>
inline void fnLoadYIpData(T_buff_FirDecHb<T_D, T_C, TP_ARCH>& buff, unsigned int splice, input_window<T_D>* inWindow) {
    if
        constexpr(T_SIZE == 256) {
            T_buff_256b<T_D> readData;
            const short kSpliceRange = 4;
            // Read the next slice (in forward direction) THEN decrement
            readData = window_readdecr_256b<T_D>(inWindow);
            buff.val = upd_w(buff.val, splice % kSpliceRange, readData.val);
        }
    else {
        T_buff_128b<T_D> readData;
        const short kSpliceRange = 8;
        readData = window_readdecr_128b<T_D>(inWindow);
        buff.val = upd_v(buff.val, splice % kSpliceRange, readData.val);
    }
    // return retVal;
};

// Set of functions to write out data. This can be from 2 concatenated accumulators in one pass,
// one accumulator in one pass, or two concatenated accumulators from 2 passes
template <typename T_D, typename T_C, eArchType TP_ARCH>
T_outValFiRDecHb<T_D, T_C, TP_ARCH> firDecHbWriteOut(T_accFirDecHb<T_D, T_C, TP_ARCH>* acc, int shift) {
    T_outValFiRDecHb<T_D, T_C, TP_ARCH> outVal;
    return outVal; // this mutes a warning;
    // Never used default case
}
template <>
T_outValFiRDecHb<int16, int16, kArch1Buff> firDecHbWriteOut<int16, int16, kArch1Buff>(
    T_accFirDecHb<int16, int16, kArch1Buff>* acc, int shift) {
    T_acc768<int16, int16> accConcat;
    T_outValFiRDecHb<int16, int16, kArch1Buff> retVal;

    accConcat.val = acc[0].val;
    retVal.val = srs(accConcat.val, shift);
    return retVal;
}

template <>
T_outValFiRDecHb<cint16, int16, kArch1Buff> firDecHbWriteOut<cint16, int16, kArch1Buff>(
    T_accFirDecHb<cint16, int16, kArch1Buff>* acc, int shift) {
    T_acc768<cint16, int16> accConcat;
    T_outValFiRDecHb<cint16, int16, kArch1Buff> retVal;

    // accConcat.val = concat(acc[0].val, acc[0].uval);
    retVal.val = srs(acc[0].val, shift);
    return retVal;
}

template <>
T_outValFiRDecHb<cint16, cint16, kArch1Buff> firDecHbWriteOut<cint16, cint16, kArch1Buff>(
    T_accFirDecHb<cint16, cint16, kArch1Buff>* acc, int shift) {
    T_acc768<cint16, cint16> accConcat;
    T_outValFiRDecHb<cint16, cint16, kArch1Buff> retVal;

    accConcat.val = concat(acc[0].val, acc[0].uval);
    retVal.val = srs(accConcat.val, shift);
    return retVal;
}

template <>
T_outValFiRDecHb<int32, int16, kArch1Buff> firDecHbWriteOut<int32, int16, kArch1Buff>(
    T_accFirDecHb<int32, int16, kArch1Buff>* acc, int shift) {
    T_acc768<int32, int16> accConcat;
    T_outValFiRDecHb<int32, int16, kArch1Buff> retVal;

    accConcat.val = acc[0].val;
    retVal.val = srs(accConcat.val, shift);
    return retVal;
}

template <>
T_outValFiRDecHb<int32, int32, kArch1Buff> firDecHbWriteOut<int32, int32, kArch1Buff>(
    T_accFirDecHb<int32, int32, kArch1Buff>* acc, int shift) {
    T_acc768<int32, int32> accConcat;
    T_outValFiRDecHb<int32, int32, kArch1Buff> retVal;

    accConcat.val = concat(acc[0].val, acc[0].uval);
    retVal.val = srs(accConcat.val, shift);
    return retVal;
}

template <>
T_outValFiRDecHb<cint32, int16, kArch1Buff> firDecHbWriteOut<cint32, int16, kArch1Buff>(
    T_accFirDecHb<cint32, int16, kArch1Buff>* acc, int shift) {
    T_acc768<cint32, int16> accConcat;
    T_outValFiRDecHb<cint32, int16, kArch1Buff> retVal;

    accConcat.val = acc[0].val;
    retVal.val = srs(accConcat.val, shift);
    return retVal;
}

template <>
T_outValFiRDecHb<cint32, cint16, kArch1Buff> firDecHbWriteOut<cint32, cint16, kArch1Buff>(
    T_accFirDecHb<cint32, cint16, kArch1Buff>* acc, int shift) {
    T_acc768<cint32, cint16> accConcat;
    T_outValFiRDecHb<cint32, cint16, kArch1Buff> retVal;

    accConcat.val = acc[0].val;
    retVal.val = srs(accConcat.val, shift);
    return retVal;
}

template <>
T_outValFiRDecHb<cint32, int32, kArch1Buff> firDecHbWriteOut<cint32, int32, kArch1Buff>(
    T_accFirDecHb<cint32, int32, kArch1Buff>* acc, int shift) {
    T_acc768<cint32, int32> accConcat;
    T_outValFiRDecHb<cint32, int32, kArch1Buff> retVal;

    accConcat.val = acc[0].val;
    retVal.val = srs(accConcat.val, shift);
    return retVal;
}

template <>
T_outValFiRDecHb<cint32, cint32, kArch1Buff> firDecHbWriteOut<cint32, cint32, kArch1Buff>(
    T_accFirDecHb<cint32, cint32, kArch1Buff>* acc, int shift) {
    v4cacc80 accConcat;
    T_outValFiRDecHb<cint32, cint32, kArch1Buff> retVal;

    accConcat = concat(acc[0].val, acc[0].uval);
    retVal.val = srs(accConcat, shift);
    return retVal;
}

template <>
T_outValFiRDecHb<float, float, kArch1Buff> firDecHbWriteOut<float, float, kArch1Buff>(
    T_accFirDecHb<float, float, kArch1Buff>* acc, int shift) {
    T_acc768<float, float> accConcat;
    T_outValFiRDecHb<float, float, kArch1Buff> retVal;

    accConcat.val = acc[0].val;
    retVal.val = accConcat.val;
    return retVal;
}

template <>
T_outValFiRDecHb<cfloat, float, kArch1Buff> firDecHbWriteOut<cfloat, float, kArch1Buff>(
    T_accFirDecHb<cfloat, float, kArch1Buff>* acc, int shift) {
    T_acc768<cfloat, float> accConcat;
    T_outValFiRDecHb<cfloat, float, kArch1Buff> retVal;

    accConcat.val = acc[0].val;
    retVal.val = accConcat.val;
    return retVal;
}

template <>
T_outValFiRDecHb<cfloat, cfloat, kArch1Buff> firDecHbWriteOut<cfloat, cfloat, kArch1Buff>(
    T_accFirDecHb<cfloat, cfloat, kArch1Buff>* acc, int shift) {
    T_acc768<cfloat, cfloat> accConcat;
    T_outValFiRDecHb<cfloat, cfloat, kArch1Buff> retVal;

    accConcat.val = acc[0].val;
    retVal.val = accConcat.val;
    return retVal;
}

template <>
T_outValFiRDecHb<int16, int16, kArch2Buff> firDecHbWriteOut<int16, int16, kArch2Buff>(
    T_accFirDecHb<int16, int16, kArch2Buff>* acc, int shift) {
    T_acc768<int16, int16> accConcat;
    T_outValFiRDecHb<int16, int16, kArch2Buff> retVal;

    accConcat.val = concat(acc[0].val, acc[1].val);
    retVal.val = srs(accConcat.val, shift);
    return retVal;
}

template <>
T_outValFiRDecHb<cint16, int16, kArch2Buff> firDecHbWriteOut<cint16, int16, kArch2Buff>(
    T_accFirDecHb<cint16, int16, kArch2Buff>* acc, int shift) {
    T_outValFiRDecHb<cint16, int16, kArch2Buff> retVal;

    retVal.val = srs(acc[0].val, shift);
    return retVal;
}

template <>
T_outValFiRDecHb<cint16, cint16, kArch2Buff> firDecHbWriteOut<cint16, cint16, kArch2Buff>(
    T_accFirDecHb<cint16, cint16, kArch2Buff>* acc, int shift) {
    T_outValFiRDecHb<cint16, cint16, kArch2Buff> retVal;

    retVal.val = srs(acc[0].val, shift);
    return retVal;
}

template <>
T_outValFiRDecHb<int32, int16, kArch2Buff> firDecHbWriteOut<int32, int16, kArch2Buff>(
    T_accFirDecHb<int32, int16, kArch2Buff>* acc, int shift) {
    T_acc768<int32, int16> accConcat;
    T_outValFiRDecHb<int32, int16, kArch2Buff> retVal;

    accConcat.val = acc[0].val;
    retVal.val = srs(accConcat.val, shift);
    return retVal;
}

template <>
T_outValFiRDecHb<int32, int32, kArch2Buff> firDecHbWriteOut<int32, int32, kArch2Buff>(
    T_accFirDecHb<int32, int32, kArch2Buff>* acc, int shift) {
    T_outValFiRDecHb<int32, int32, kArch2Buff> retVal;

    retVal.val = srs(acc[0].val, shift);
    return retVal;
}

template <>
T_outValFiRDecHb<cint32, int16, kArch2Buff> firDecHbWriteOut<cint32, int16, kArch2Buff>(
    T_accFirDecHb<cint32, int16, kArch2Buff>* acc, int shift) {
    T_acc768<cint32, int16> accConcat;
    T_outValFiRDecHb<cint32, int16, kArch2Buff> retVal;

    accConcat.val = acc[0].val;
    retVal.val = srs(accConcat.val, shift);
    return retVal;
}

template <>
T_outValFiRDecHb<cint32, cint16, kArch2Buff> firDecHbWriteOut<cint32, cint16, kArch2Buff>(
    T_accFirDecHb<cint32, cint16, kArch2Buff>* acc, int shift) {
    T_acc768<cint32, cint16> accConcat;
    T_outValFiRDecHb<cint32, cint16, kArch2Buff> retVal;

    accConcat.val = acc[0].val;
    retVal.val = srs(accConcat.val, shift);
    return retVal;
}

template <>
T_outValFiRDecHb<cint32, int32, kArch2Buff> firDecHbWriteOut<cint32, int32, kArch2Buff>(
    T_accFirDecHb<cint32, int32, kArch2Buff>* acc, int shift) {
    T_acc768<cint32, int32> accConcat;
    T_outValFiRDecHb<cint32, int32, kArch2Buff> retVal;

    accConcat.val = acc[0].val;
    retVal.val = srs(accConcat.val, shift);
    return retVal;
}

template <>
T_outValFiRDecHb<cint32, cint32, kArch2Buff> firDecHbWriteOut<cint32, cint32, kArch2Buff>(
    T_accFirDecHb<cint32, cint32, kArch2Buff>* acc, int shift) {
    T_acc768<cint32, cint32> accConcat;
    T_outValFiRDecHb<cint32, cint32, kArch2Buff> retVal;

    accConcat.val = concat(acc[0].val, acc[0].uval);
    retVal.val = srs(accConcat.val, shift);
    return retVal;
}

template <>
T_outValFiRDecHb<float, float, kArch2Buff> firDecHbWriteOut<float, float, kArch2Buff>(
    T_accFirDecHb<float, float, kArch2Buff>* acc, int shift) {
    T_acc768<float, float> accConcat;
    T_outValFiRDecHb<float, float, kArch2Buff> retVal;

    accConcat.val = acc[0].val;
    retVal.val = accConcat.val;
    return retVal;
}

template <>
T_outValFiRDecHb<cfloat, float, kArch2Buff> firDecHbWriteOut<cfloat, float, kArch2Buff>(
    T_accFirDecHb<cfloat, float, kArch2Buff>* acc, int shift) {
    T_acc768<cfloat, float> accConcat;
    T_outValFiRDecHb<cfloat, float, kArch2Buff> retVal;

    accConcat.val = acc[0].val;
    retVal.val = accConcat.val;
    return retVal;
}

template <>
T_outValFiRDecHb<cfloat, cfloat, kArch2Buff> firDecHbWriteOut<cfloat, cfloat, kArch2Buff>(
    T_accFirDecHb<cfloat, cfloat, kArch2Buff>* acc, int shift) {
    T_acc768<cfloat, cfloat> accConcat;
    T_outValFiRDecHb<cfloat, cfloat, kArch2Buff> retVal;

    accConcat.val = acc[0].val;
    retVal.val = accConcat.val;
    return retVal;
}

// Symmetric mul/mac for use in halfband decimator (fixed decimation factor of 2).
// Note that these functions are grouped by combination of type rather than by function as there
// is more commonality between functions for a given type then between same functions for different types.
//-----------------------------------------------------------------------------------------------------

template <typename T_D, typename T_C, eArchType TP_ARCH>
inline T_accFirDecHb<T_D, T_C, TP_ARCH> firDecHbMulSym(T_buff_FirDecHb<T_D, T_C, TP_ARCH> xbuff,
                                                       unsigned int xstart,
                                                       T_buff_FirDecHb<T_D, T_C, TP_ARCH> ybuff,
                                                       unsigned int ystart,
                                                       T_buff_256b<T_C> zbuff,
                                                       unsigned int zstart) {
    T_accFirDecHb<T_D, T_C, TP_ARCH> retVal;
    return retVal; // mute warning
};
template <typename T_D, typename T_C, eArchType TP_ARCH>
inline T_accFirDecHb<T_D, T_C, TP_ARCH> firDecHbMulSym1buff(T_buff_FirDecHb<T_D, T_C, TP_ARCH> xbuff,
                                                            unsigned int xstart,
                                                            unsigned int ystart,
                                                            T_buff_256b<T_C> zbuff,
                                                            unsigned int zstart) {
    T_accFirDecHb<T_D, T_C, TP_ARCH> retVal;
    return retVal; // mute warning
};
template <typename T_D, typename T_C, eArchType TP_ARCH>
inline T_accFirDecHb<T_D, T_C, TP_ARCH> firDecHbMulSymCt(T_buff_FirDecHb<T_D, T_C, TP_ARCH> xbuff,
                                                         unsigned int xstart,
                                                         T_buff_FirDecHb<T_D, T_C, TP_ARCH> ybuff,
                                                         unsigned int ystart,
                                                         unsigned int ct,
                                                         T_buff_256b<T_C> zbuff,
                                                         unsigned int zstart) {
    T_accFirDecHb<T_D, T_C, TP_ARCH> retVal;
    return retVal; // mute warning
};
template <typename T_D, typename T_C, eArchType TP_ARCH>
inline T_accFirDecHb<T_D, T_C, TP_ARCH> firDecHbMulSymCt1buff(T_buff_FirDecHb<T_D, T_C, TP_ARCH> xbuff,
                                                              unsigned int xstart,
                                                              unsigned int ystart,
                                                              unsigned int ct,
                                                              T_buff_256b<T_C> zbuff,
                                                              unsigned int zstart) {
    T_accFirDecHb<T_D, T_C, TP_ARCH> retVal;
    return retVal; // mute warning
};
template <typename T_D, typename T_C, eArchType TP_ARCH>
inline T_accFirDecHb<T_D, T_C, TP_ARCH> firDecHbMacSym(T_accFirDecHb<T_D, T_C, TP_ARCH> acc,
                                                       T_buff_FirDecHb<T_D, T_C, TP_ARCH> xbuff,
                                                       unsigned int xstart,
                                                       T_buff_FirDecHb<T_D, T_C, TP_ARCH> ybuff,
                                                       unsigned int ystart,
                                                       T_buff_256b<T_C> zbuff,
                                                       unsigned int zstart) {
    T_accFirDecHb<T_D, T_C, TP_ARCH> retVal;
    return retVal; // mute warning
};
template <typename T_D, typename T_C, eArchType TP_ARCH>
inline T_accFirDecHb<T_D, T_C, TP_ARCH> firDecHbMacSym1buff(T_accFirDecHb<T_D, T_C, TP_ARCH> acc,
                                                            T_buff_FirDecHb<T_D, T_C, TP_ARCH> xbuff,
                                                            unsigned int xstart,
                                                            unsigned int ystart,
                                                            T_buff_256b<T_C> zbuff,
                                                            unsigned int zstart) {
    T_accFirDecHb<T_D, T_C, TP_ARCH> retVal;
    return retVal; // mute warning
};
template <typename T_D, typename T_C, eArchType TP_ARCH>
inline T_accFirDecHb<T_D, T_C, TP_ARCH> firDecHbMacSymCt(T_accFirDecHb<T_D, T_C, TP_ARCH> acc,
                                                         T_buff_FirDecHb<T_D, T_C, TP_ARCH> xbuff,
                                                         unsigned int xstart,
                                                         T_buff_FirDecHb<T_D, T_C, TP_ARCH> ybuff,
                                                         unsigned int ystart,
                                                         unsigned int ct,
                                                         T_buff_256b<T_C> zbuff,
                                                         unsigned int zstart) {
    T_accFirDecHb<T_D, T_C, TP_ARCH> retVal;
    return retVal; // mute warning
};
template <typename T_D, typename T_C, eArchType TP_ARCH>
inline T_accFirDecHb<T_D, T_C, TP_ARCH> firDecHbMacSymCt1buff(T_accFirDecHb<T_D, T_C, TP_ARCH> acc,
                                                              T_buff_FirDecHb<T_D, T_C, TP_ARCH> xbuff,
                                                              unsigned int xstart,
                                                              unsigned int ystart,
                                                              unsigned int ct,
                                                              T_buff_256b<T_C> zbuff,
                                                              unsigned int zstart) {
    T_accFirDecHb<T_D, T_C, TP_ARCH> retVal;
    return retVal; // mute warning
};

// DATA = int16,  COEFF = int16>
/* This code is currently commented out because it transpires that int16/int16 requires an xsquare value in the intinsic
for
mul and mac which is out of range due to the fact that the data step is 2 rather than 1 for single rate FIRs.
A different approach will be required for TT_DATA = int16 with TT_COEFF = int16.
template<> inline T_accFirDecHb<int16,  int16, kArch2Buff>  firDecHbMulSym<int16,   int16, kArch2Buff>
(T_buff_FirDecHb<int16,  int16, kArch2Buff> xbuff, unsigned int xstart,
                                                                 T_buff_FirDecHb<int16,  int16, kArch2Buff> ybuff,
unsigned int ystart,
                                                                 T_buff_256b<int16>  zbuff, unsigned int zstart)
{
  T_accFirDecHb<int16,  int16, kArch2Buff> retVal;
  const unsigned int xoffsets      = 0x06040200;
  const unsigned int xstep         = 2;
  const unsigned int xsquare       = 0x4220;
  const unsigned int ysquare       = 0x2402;
  const unsigned int zoffsets      = 0x00000000;
  const unsigned int zstep         = 1;
  const unsigned int kLanes        = 8;
  const unsigned int kDecFactor    = 2;

  retVal.val   = mul8_sym( xbuff.val, xstart,                   xoffsets, xstep, xsquare,
                           ybuff.val, ystart,                                    ysquare,
                           zbuff.val, zstart, zoffsets, zstep);
  return retVal;
}
template<> inline T_accFirDecHb<int16,  int16, kArch1Buff>  firDecHbMulSym1buff<int16,  int16, kArch1Buff>
(T_buff_FirDecHb<int16,  int16, kArch1Buff> xbuff, unsigned int xstart,
                                                                                                                            unsigned int ystart,
                                                                        T_buff_256b<int16> zbuff, unsigned int zstart)
{
  T_accFirDecHb<int16,  int16, kArch1Buff> retVal;
  const unsigned int xoffsets      = 0x06040200;
  const unsigned int xoffsets_hi   = 0x0E0C0A08;
  const unsigned int xsquare       = 0x2110;
  const unsigned int xstep         = 2;
  const unsigned int ysquare       = 0x2402;
  const unsigned int zoffsets      = 0x00000000;
  const unsigned int zoffsets_hi   = 0x00000000;
  const unsigned int zstep         = 1;
  const unsigned int kLanes        = 16;
  const unsigned int kDecFactor    = 2;

  retVal.val   = mul16_sym( xbuff.val, xstart, xoffsets, xoffsets_hi, xsquare,
                                       ystart,                        ysquare,
                            zbuff.val, zstart, zoffsets, zoffsets_hi, zstep);
  return retVal;
}
template<> inline T_accFirDecHb<int16,  int16, kArch2Buff>  firDecHbMulSymCt<int16,  int16, kArch2Buff>
(T_buff_FirDecHb<int16,  int16, kArch2Buff> xbuff, unsigned int xstart,
                                                   T_buff_FirDecHb<int16,  int16, kArch2Buff> ybuff, unsigned int
ystart,
                                                   unsigned int ct,
                                                   T_buff_256b<int16> zbuff, unsigned int zstart)
{
  T_accFirDecHb<int16,  int16, kArch2Buff> retVal;
  const unsigned int xoffsets      = 0x06040200;
  const unsigned int xstep         = 2;
  const unsigned int xsquare       = 0x4220;
  const unsigned int ysquare       = 0x2402;
  const unsigned int zoffsets      = 0x00000000;
  const unsigned int zstep         = 1;
  const unsigned int kLanes        = 8;
  const unsigned int kCols         = 4;
  const unsigned int kDecFactor    = 2;

  T_buff_256b<int16> zbuff2;  //= zbuff;
  unsigned int ystartmod = ystart-(kCols-1)*xstep;
  unsigned int zstartmod = zstart+(kCols-1);
  int zstepmod = -(int)zstep;

  zbuff2.val = upd_elem(zbuff.val, zstart-1, 0); //zero the centre tap

  retVal.val   = mul8( xbuff.val,  xstart, xoffsets, xstep, xsquare,
                       zbuff.val,  zstart, zoffsets, zstep);
  retVal.val   = mac8( retVal.val,
                       xbuff.val,  ystartmod, xoffsets, xstep, ysquare,
                       zbuff2.val, zstartmod, zoffsets, zstepmod);
  return retVal;
}
template<> inline T_accFirDecHb<int16,  int16, kArch1Buff>  firDecHbMulSymCt1buff<int16,  int16, kArch1Buff>
(T_buff_FirDecHb<int16,  int16, kArch1Buff> xbuff, unsigned int xstart,
                                                         unsigned int ystart,
                                                         unsigned int ct,
                                                         T_buff_256b<int16> zbuff, unsigned int zstart)
{
  T_accFirDecHb<int16,  int16, kArch1Buff> retVal;
  const unsigned int xoffsets      = 0x06040200;
  const unsigned int xoffsets_hi   = 0x0E0C0A08;
  const unsigned int xsquare       = 0x4220;
  const unsigned int xstep         = 2;
  const unsigned int ysquare       = 0x2402;
  const unsigned int zoffsets      = 0x00000000;
  const unsigned int zoffsets_hi   = 0x00000000;
  const unsigned int zstep         = 1;
  const unsigned int kLanes        = 16;
  const unsigned int kCols         = 2;
  const unsigned int kDecFactor    = 2;

  T_buff_256b<int16> zbuff2;  //= zbuff;
  unsigned int ystartmod = ystart-(kCols-1)*xstep;
  unsigned int zstartmod = zstart+(kCols-1);
  int zstepmod = -(int)zstep;

  zbuff2.val = upd_elem(zbuff.val, zstart-1, 0); //zero the centre tap

  retVal.val   = mul16( xbuff.val, xstart, xoffsets, xoffsets_hi, xsquare,
                        zbuff.val, zstart, zoffsets, zoffsets_hi, zstep);
  retVal.val   = mac16( retVal.val,
                        xbuff.val, xstart, xoffsets, xoffsets_hi, xstep,
                        zbuff.val, zstart, zoffsets, zoffsets_hi, zstep);
  return retVal;
}
template<> inline T_accFirDecHb<int16,  int16, kArch2Buff> firDecHbMacSym<int16,  int16, kArch2Buff>
(T_accFirDecHb<int16,  int16, kArch2Buff> acc,
                                                 T_buff_FirDecHb<int16,  int16, kArch2Buff> xbuff, unsigned int xstart,
                                                 T_buff_FirDecHb<int16,  int16, kArch2Buff> ybuff, unsigned int ystart,
                                                 T_buff_256b<int16> zbuff, unsigned int zstart)
{
  T_accFirDecHb<int16,  int16, kArch2Buff> retVal;
  const unsigned int xoffsets      = 0x06040200;
  const unsigned int xstep         = 2;
  const unsigned int xsquare       = 0x4220;
  const unsigned int ysquare       = 0x2402;
  const unsigned int zoffsets      = 0x00000000;
  const unsigned int zstep         = 1;
  const unsigned int kLanes        = 8;
  const unsigned int kDecFactor    = 2;

  retVal.val   = mac8_sym( acc.val,
                           xbuff.val, xstart,                   xoffsets, xstep, xsquare,
                           ybuff.val, ystart,                                    ysquare,
                           zbuff.val, zstart, zoffsets, zstep);
  return retVal;
}
template<> inline T_accFirDecHb<int16,  int16, kArch1Buff> firDecHbMacSym1buff<int16,  int16, kArch1Buff>
(T_accFirDecHb<int16,  int16, kArch1Buff> acc,
                                                      T_buff_FirDecHb<int16,  int16, kArch1Buff>  xbuff, unsigned int
xstart,
                                                      unsigned int ystart,
                                                      T_buff_256b<int16> zbuff, unsigned int zstart)
{
  T_accFirDecHb<int16,  int16, kArch1Buff> retVal;
  const unsigned int xoffsets      = 0x06040200;
  const unsigned int xoffsets_hi   = 0x0E0C0A08;
  const unsigned int xsquare       = 0x4220;
  const unsigned int xstep         = 2;
  const unsigned int ysquare       = 0x2402;
  const unsigned int zoffsets      = 0x00000000;
  const unsigned int zoffsets_hi   = 0x00000000;
  const unsigned int zstep         = 1;
  const unsigned int kLanes        = 16;
  const unsigned int kDecFactor    = 2;

  retVal.val   = mac16_sym( acc.val,
                            xbuff.val, xstart, xoffsets, xoffsets_hi, xsquare,
                                       ystart,                        ysquare,
                            zbuff.val, zstart, zoffsets, zoffsets_hi, zstep);
  return retVal;
}
template<> inline T_accFirDecHb<int16,  int16, kArch2Buff> firDecHbMacSymCt<int16,  int16, kArch2Buff>
(T_accFirDecHb<int16,  int16, kArch2Buff> acc,
                                                   T_buff_FirDecHb<int16,  int16, kArch2Buff> xbuff, unsigned int
xstart,
                                                   T_buff_FirDecHb<int16,  int16, kArch2Buff> ybuff, unsigned int
ystart,
                                                   unsigned int ct,
                                                   T_buff_256b<int16>  zbuff, unsigned int zstart)
{
  T_accFirDecHb<int16,  int16, kArch2Buff> retVal;
  const unsigned int xoffsets      = 0x06040200;
  const unsigned int xstep         = 2;
  const unsigned int xsquare       = 0x4220;
  const unsigned int ysquare       = 0x2402;
  const unsigned int zoffsets      = 0x00000000;
  const unsigned int zstep         = 1;
  const unsigned int kLanes        = 8;
  const unsigned int kCols         = 4;
  const unsigned int kDecFactor    = 2;

  T_buff_256b<int16> zbuff2;  //= zbuff;
  unsigned int ystartmod = ystart-(kCols-1)*xstep;
  unsigned int zstartmod = zstart+(kCols-1);
  int zstepmod = -(int)zstep;

  zbuff2.val = upd_elem(zbuff.val, zstart-1, 0); //zero the centre tap

  retVal.val   = mac8( acc.val,
                       xbuff.val,  xstart, xoffsets, xstep, xsquare,
                       zbuff.val,  zstart, zoffsets, zstep);
  retVal.val   = mac8( retVal.val,
                       xbuff.val,  ystartmod, xoffsets, xstep, ysquare,
                       zbuff2.val, zstartmod, zoffsets, zstepmod);
  return retVal;
}
template<> inline T_accFirDecHb<int16,  int16, kArch1Buff> firDecHbMacSymCt1buff<int16,  int16, kArch1Buff>
(T_accFirDecHb<int16,  int16, kArch1Buff> acc,
                                                        T_buff_FirDecHb<int16,  int16, kArch1Buff> xbuff, unsigned int
xstart,
                                                        unsigned int ystart,
                                                        unsigned int ct,
                                                        T_buff_256b<int16>  zbuff, unsigned int zstart)
{
  T_accFirDecHb<int16,  int16, kArch1Buff> retVal;
  const unsigned int xoffsets      = 0x06040200;
  const unsigned int xoffsets_hi   = 0x0E0C0A08;
  const unsigned int xsquare       = 0x4220;
  const unsigned int xstep         = 2;
  const unsigned int ysquare       = 0x2402;
  const unsigned int zoffsets      = 0x00000000;
  const unsigned int zoffsets_hi   = 0x00000000;
  const unsigned int zstep         = 1;
  const unsigned int kLanes        = 4;
  const unsigned int kCols         = 2;
  const unsigned int kDecFactor    = 2;

  T_buff_256b<int16> zbuff2;  //= zbuff;
  unsigned int ystartmod = ystart-(kCols-1)*xstep;
  unsigned int zstartmod = zstart+(kCols-1);
  int zstepmod = -(int)zstep;

  zbuff2.val = upd_elem(zbuff.val, zstart-1, 0); //zero the centre tap

  retVal.val  = mac16( acc.val,
                       xbuff.val, xstart, xoffsets, xoffsets_hi, xsquare,
                       zbuff.val, zstart, zoffsets, zoffsets_hi, zstep);
  retVal.val  = mac16( retVal.val,
                       xbuff.val, xstart, xoffsets, xoffsets_hi, xstep,
                       zbuff.val, zstart, zoffsets, zoffsets_hi, zstep);
  return retVal;
}
*/
// DATA = cint16,  COEFF = int16>
template <>
inline T_accFirDecHb<cint16, int16, kArch2Buff> firDecHbMulSym<cint16, int16, kArch2Buff>(
    T_buff_FirDecHb<cint16, int16, kArch2Buff> xbuff,
    unsigned int xstart,
    T_buff_FirDecHb<cint16, int16, kArch2Buff> ybuff,
    unsigned int ystart,
    T_buff_256b<int16> zbuff,
    unsigned int zstart) {
    T_accFirDecHb<cint16, int16, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = mul4_sym(xbuff.val, xstart, xoffsets, xstep, ybuff.val, ystart, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<cint16, int16, kArch1Buff> firDecHbMulSym1buff<cint16, int16, kArch1Buff>(
    T_buff_FirDecHb<cint16, int16, kArch1Buff> xbuff,
    unsigned int xstart,
    unsigned int ystart,
    T_buff_256b<int16> zbuff,
    unsigned int zstart) {
    T_accFirDecHb<cint16, int16, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = mul4_sym(xbuff.val, xstart, xoffsets, xstep, ystart, zbuff.val, zstart, zoffsets, zstep);
    retVal.uval = mul4_sym(xbuff.val, xstart + kDecFactor * kLanes, xoffsets, xstep, ystart + kDecFactor * kLanes,
                           zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<cint16, int16, kArch2Buff> firDecHbMulSymCt<cint16, int16, kArch2Buff>(
    T_buff_FirDecHb<cint16, int16, kArch2Buff> xbuff,
    unsigned int xstart,
    T_buff_FirDecHb<cint16, int16, kArch2Buff> ybuff,
    unsigned int ystart,
    unsigned int ct,
    T_buff_256b<int16> zbuff,
    unsigned int zstart) {
    T_accFirDecHb<cint16, int16, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val =
        mul4_sym_ct(xbuff.val, xstart, xoffsets, xstep, ybuff.val, ystart, ct, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<cint16, int16, kArch1Buff> firDecHbMulSymCt1buff<cint16, int16, kArch1Buff>(
    T_buff_FirDecHb<cint16, int16, kArch1Buff> xbuff,
    unsigned int xstart,
    unsigned int ystart,
    unsigned int ct,
    T_buff_256b<int16> zbuff,
    unsigned int zstart) {
    T_accFirDecHb<cint16, int16, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = mul4_sym_ct(xbuff.val, xstart, xoffsets, xstep, ystart, ct, zbuff.val, zstart, zoffsets, zstep);
    retVal.uval = mul4_sym_ct(xbuff.val, xstart + kDecFactor * kLanes, xoffsets, xstep, ystart + kDecFactor * kLanes,
                              ct, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}

// template<> inline T_accFirDecHb<cint16,  int16, kArchHighDF> firDecHbMacSym<cint16,  int16, kArch2Buff>
// (T_accFirDecHb<cint16,  int16, kArch2Buff> acc,
//                                                  T_buff_FirDecHb<cint16,  int16, kArch2Buff> xbuff, unsigned int
//                                                  xstart,
//                                                  T_buff_FirDecHb<cint16,  int16, kArch2Buff> ybuff, unsigned int
//                                                  ystart,
//                                                  T_buff_256b<int16> zbuff, unsigned int zstart)
// {
//   T_accFirDecHb<cint16,  int16, kArchHighDF> retVal;
//   T_accFirDecHb<cint16,  int16, kArchHighDF> tmp;
//   const unsigned int sel           = 0xFF00;
//   const unsigned int xoffsets      = 0x00003210;
//   const unsigned int xoffsets_hi   = 0xA9870000;
//   const unsigned int yoffsets      = 0x00003210;
//   const unsigned int yoffsets_hi   = 0xA9870000;
//   const unsigned int xstart        = 0;
//   const unsigned int ystart        = 14;

//   tmp.val   = select16(sel, xbuff.val, xstart, xoffsets, xoffsets_hi,
//                                        ystart, yoffsets, yoffsets_hi);

//   const unsigned int xstep         = 1;
//   const unsigned int xoffsets      = 0xC840;
//   const unsigned int zoffsets      = 0x00000000;
//   const unsigned int zstep         = 1;
//   const unsigned int kLanes        = 4;
//   const unsigned int kDecFactor    = 2;

//   retVal.val   = mac4_sym(acc.val,  xbuff.val, xstart, xstep,
//                                     ybuff.val, ystart,
//                                     zbuff.val, zstart, zoffsets, zstep);
//   return retVal;
// }

template <>
inline T_accFirDecHb<cint16, int16, kArch2Buff> firDecHbMacSym<cint16, int16, kArch2Buff>(
    T_accFirDecHb<cint16, int16, kArch2Buff> acc,
    T_buff_FirDecHb<cint16, int16, kArch2Buff> xbuff,
    unsigned int xstart,
    T_buff_FirDecHb<cint16, int16, kArch2Buff> ybuff,
    unsigned int ystart,
    T_buff_256b<int16> zbuff,
    unsigned int zstart) {
    T_accFirDecHb<cint16, int16, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val =
        mac4_sym(acc.val, xbuff.val, xstart, xoffsets, xstep, ybuff.val, ystart, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<cint16, int16, kArch1Buff> firDecHbMacSym1buff<cint16, int16, kArch1Buff>(
    T_accFirDecHb<cint16, int16, kArch1Buff> acc,
    T_buff_FirDecHb<cint16, int16, kArch1Buff> xbuff,
    unsigned int xstart,
    unsigned int ystart,
    T_buff_256b<int16> zbuff,
    unsigned int zstart) {
    T_accFirDecHb<cint16, int16, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = mac4_sym(acc.val, xbuff.val, xstart, xoffsets, xstep, ystart, zbuff.val, zstart, zoffsets, zstep);
    // retVal.uval  = mac4_sym(acc.uval, xbuff.val, xstart+kDecFactor*kLanes, xoffsets, xstep,
    //                                          ystart+kDecFactor*kLanes,
    //                                   zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<cint16, int16, kArch2Buff> firDecHbMacSymCt<cint16, int16, kArch2Buff>(
    T_accFirDecHb<cint16, int16, kArch2Buff> acc,
    T_buff_FirDecHb<cint16, int16, kArch2Buff> xbuff,
    unsigned int xstart,
    T_buff_FirDecHb<cint16, int16, kArch2Buff> ybuff,
    unsigned int ystart,
    unsigned int ct,
    T_buff_256b<int16> zbuff,
    unsigned int zstart) {
    T_accFirDecHb<cint16, int16, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = mac4_sym_ct(acc.val, xbuff.val, xstart, xoffsets, xstep, ybuff.val, ystart, ct, zbuff.val, zstart,
                             zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<cint16, int16, kArch1Buff> firDecHbMacSymCt1buff<cint16, int16, kArch1Buff>(
    T_accFirDecHb<cint16, int16, kArch1Buff> acc,
    T_buff_FirDecHb<cint16, int16, kArch1Buff> xbuff,
    unsigned int xstart,
    unsigned int ystart,
    unsigned int ct,
    T_buff_256b<int16> zbuff,
    unsigned int zstart) {
    T_accFirDecHb<cint16, int16, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val =
        mac4_sym_ct(acc.val, xbuff.val, xstart, xoffsets, xstep, ystart, ct, zbuff.val, zstart, zoffsets, zstep);
    // retVal.uval  = mac4_sym_ct(acc.uval, xbuff.val, xstart+kDecFactor*kLanes, xoffsets, xstep,
    //                                             ystart+kDecFactor*kLanes, ct,
    //                                      zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = cint16,  COEFF = cint16>
template <>
inline T_accFirDecHb<cint16, cint16, kArch2Buff> firDecHbMulSym(T_buff_FirDecHb<cint16, cint16, kArch2Buff> xbuff,
                                                                unsigned int xstart,
                                                                T_buff_FirDecHb<cint16, cint16, kArch2Buff> ybuff,
                                                                unsigned int ystart,
                                                                T_buff_256b<cint16> zbuff,
                                                                unsigned int zstart) {
    T_accFirDecHb<cint16, cint16, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = mul4_sym(xbuff.val, xstart, xoffsets, xstep, ybuff.val, ystart, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<cint16, cint16, kArch1Buff> firDecHbMulSym1buff(T_buff_FirDecHb<cint16, cint16, kArch1Buff> xbuff,
                                                                     unsigned int xstart,
                                                                     unsigned int ystart,
                                                                     T_buff_256b<cint16> zbuff,
                                                                     unsigned int zstart) {
    T_accFirDecHb<cint16, cint16, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = mul4_sym(xbuff.val, xstart, xoffsets, xstep, ystart, zbuff.val, zstart, zoffsets, zstep);
    retVal.uval = mul4_sym(xbuff.val, xstart + kDecFactor * kLanes, xoffsets, xstep, ystart + kDecFactor * kLanes,
                           zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<cint16, cint16, kArch2Buff> firDecHbMulSymCt(T_buff_FirDecHb<cint16, cint16, kArch2Buff> xbuff,
                                                                  unsigned int xstart,
                                                                  T_buff_FirDecHb<cint16, cint16, kArch2Buff> ybuff,
                                                                  unsigned int ystart,
                                                                  unsigned int ct,
                                                                  T_buff_256b<cint16> zbuff,
                                                                  unsigned int zstart) {
    T_accFirDecHb<cint16, cint16, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = mul4_sym_ct(xbuff.val, xstart, xoffsets, ybuff.val, ystart, ct, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<cint16, cint16, kArch1Buff> firDecHbMulSymCt1buff(
    T_buff_FirDecHb<cint16, cint16, kArch1Buff> xbuff,
    unsigned int xstart,
    unsigned int ystart,
    unsigned int ct,
    T_buff_256b<cint16> zbuff,
    unsigned int zstart) {
    T_accFirDecHb<cint16, cint16, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = mul4_sym_ct(xbuff.val, xstart, xoffsets, ystart, ct, zbuff.val, zstart, zoffsets, zstep);
    retVal.uval = mul4_sym_ct(xbuff.val, xstart + kDecFactor * kLanes, xoffsets, ystart + kDecFactor * kLanes, ct,
                              zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<cint16, cint16, kArch2Buff> firDecHbMacSym(T_accFirDecHb<cint16, cint16, kArch2Buff> acc,
                                                                T_buff_FirDecHb<cint16, cint16, kArch2Buff> xbuff,
                                                                unsigned int xstart,
                                                                T_buff_FirDecHb<cint16, cint16, kArch2Buff> ybuff,
                                                                unsigned int ystart,
                                                                T_buff_256b<cint16> zbuff,
                                                                unsigned int zstart) {
    T_accFirDecHb<cint16, cint16, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val =
        mac4_sym(acc.val, xbuff.val, xstart, xoffsets, xstep, ybuff.val, ystart, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<cint16, cint16, kArch1Buff> firDecHbMacSym1buff(T_accFirDecHb<cint16, cint16, kArch1Buff> acc,
                                                                     T_buff_FirDecHb<cint16, cint16, kArch1Buff> xbuff,
                                                                     unsigned int xstart,
                                                                     unsigned int ystart,
                                                                     T_buff_256b<cint16> zbuff,
                                                                     unsigned int zstart) {
    T_accFirDecHb<cint16, cint16, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = mac4_sym(acc.val, xbuff.val, xstart, xoffsets, xstep, ystart, zbuff.val, zstart, zoffsets, zstep);
    retVal.uval = mac4_sym(acc.uval, xbuff.val, xstart + kDecFactor * kLanes, xoffsets, xstep,
                           ystart + kDecFactor * kLanes, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<cint16, cint16, kArch2Buff> firDecHbMacSymCt(T_accFirDecHb<cint16, cint16, kArch2Buff> acc,
                                                                  T_buff_FirDecHb<cint16, cint16, kArch2Buff> xbuff,
                                                                  unsigned int xstart,
                                                                  T_buff_FirDecHb<cint16, cint16, kArch2Buff> ybuff,
                                                                  unsigned int ystart,
                                                                  unsigned int ct,
                                                                  T_buff_256b<cint16> zbuff,
                                                                  unsigned int zstart) {
    T_accFirDecHb<cint16, cint16, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val =
        mac4_sym_ct(acc.val, xbuff.val, xstart, xoffsets, ybuff.val, ystart, ct, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<cint16, cint16, kArch1Buff> firDecHbMacSymCt1buff(
    T_accFirDecHb<cint16, cint16, kArch1Buff> acc,
    T_buff_FirDecHb<cint16, cint16, kArch1Buff> xbuff,
    unsigned int xstart,
    unsigned int ystart,
    unsigned int ct,
    T_buff_256b<cint16> zbuff,
    unsigned int zstart) {
    T_accFirDecHb<cint16, cint16, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = mac4_sym_ct(acc.val, xbuff.val, xstart, xoffsets, ystart, ct, zbuff.val, zstart, zoffsets, zstep);
    retVal.uval = mac4_sym_ct(acc.uval, xbuff.val, xstart + kDecFactor * kLanes, xoffsets, ystart + kDecFactor * kLanes,
                              ct, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = int32,  COEFF = int16>
template <>
inline T_accFirDecHb<int32, int16, kArch2Buff> firDecHbMulSym(T_buff_FirDecHb<int32, int16, kArch2Buff> xbuff,
                                                              unsigned int xstart,
                                                              T_buff_FirDecHb<int32, int16, kArch2Buff> ybuff,
                                                              unsigned int ystart,
                                                              T_buff_256b<int16> zbuff,
                                                              unsigned int zstart) {
    T_accFirDecHb<int32, int16, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 8;
    const unsigned int kCols = 2;
    const unsigned int kDecFactor = 2;

    unsigned int ystartmod = ystart - (kCols - 1) * xstep;
    unsigned int zstartmod = zstart + (kCols - 1);
    int zstepmod = -(int)zstep;

    retVal.val = lmul8(xbuff.val, xstart, xoffsets, xstep, zbuff.val, zstart, zoffsets, zstep);
    retVal.val = lmac8(retVal.val, ybuff.val, ystartmod, xoffsets, xstep, zbuff.val, zstartmod, zoffsets, zstepmod);
    return retVal;
}
template <>
inline T_accFirDecHb<int32, int16, kArch1Buff> firDecHbMulSym1buff(T_buff_FirDecHb<int32, int16, kArch1Buff> xbuff,
                                                                   unsigned int xstart,
                                                                   unsigned int ystart,
                                                                   T_buff_256b<int16> zbuff,
                                                                   unsigned int zstart) {
    T_accFirDecHb<int32, int16, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 8;
    const unsigned int kDecFactor = 2;

    retVal.val = lmul8_sym(xbuff.val, xstart, xoffsets, xstep, ystart, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<int32, int16, kArch2Buff> firDecHbMulSymCt(T_buff_FirDecHb<int32, int16, kArch2Buff> xbuff,
                                                                unsigned int xstart,
                                                                T_buff_FirDecHb<int32, int16, kArch2Buff> ybuff,
                                                                unsigned int ystart,
                                                                unsigned int ct,
                                                                T_buff_256b<int16> zbuff,
                                                                unsigned int zstart) {
    T_accFirDecHb<int32, int16, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 8;
    const unsigned int kCols = 2;
    const unsigned int kDecFactor = 2;

    T_buff_256b<int16> zbuff2; //= zbuff;
    unsigned int ystartmod = ystart - (kCols - 1) * xstep;
    unsigned int zstartmod = zstart + (kCols - 1);
    int zstepmod = -(int)zstep;

    zbuff2.val = upd_elem(zbuff.val, zstart - 1, 0); // zero the centre tap

    // These two commands spoof a lmul_sym_ct command which cannot be used directly because it uses 512b buffers
    // which cannot accommodate the range of data required. Hence it is split into an asym mul (including the centre
    // tap) and an asym mul which must not include the center tap. The center tap is zeroed directly.
    retVal.val = lmul8(xbuff.val, xstart, xoffsets, xstep, zbuff.val, zstart, zoffsets, zstep);
    retVal.val = lmac8(retVal.val, ybuff.val, ystartmod, xoffsets, xstep, //-1? y has to work backwards.
                       zbuff.val, zstartmod, zoffsets, zstepmod);
    return retVal;
}
template <>
inline T_accFirDecHb<int32, int16, kArch1Buff> firDecHbMulSymCt1buff(T_buff_FirDecHb<int32, int16, kArch1Buff> xbuff,
                                                                     unsigned int xstart,
                                                                     unsigned int ystart,
                                                                     unsigned int ct,
                                                                     T_buff_256b<int16> zbuff,
                                                                     unsigned int zstart) {
    T_accFirDecHb<int32, int16, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 1; // exception to decimation factor of 2 since this is the centre tap term
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 8;
    const unsigned int kDecFactor = 2;

    retVal.val = lmul8_sym_ct(xbuff.val, xstart, xoffsets, ystart, ct, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<int32, int16, kArch2Buff> firDecHbMacSym(T_accFirDecHb<int32, int16, kArch2Buff> acc,
                                                              T_buff_FirDecHb<int32, int16, kArch2Buff> xbuff,
                                                              unsigned int xstart,
                                                              T_buff_FirDecHb<int32, int16, kArch2Buff> ybuff,
                                                              unsigned int ystart,
                                                              T_buff_256b<int16> zbuff,
                                                              unsigned int zstart) {
    T_accFirDecHb<int32, int16, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2; // decimation factor
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 8;
    const unsigned int kCols = 2;
    const unsigned int kDecFactor = 2; // does not apply to z because coeffs are condensed in the constructor
    unsigned int ystartmod = ystart - (kCols - 1) * xstep;
    unsigned int zstartmod = zstart + (kCols - 1);
    int zstepmod = -(int)zstep;

    retVal.val = lmac8(acc.val, xbuff.val, xstart, xoffsets, xstep, zbuff.val, zstart, zoffsets, zstep);
    retVal.val = lmac8(retVal.val, ybuff.val, ystartmod, xoffsets, xstep, zbuff.val, zstartmod, zoffsets, zstepmod);
    return retVal;
}
template <>
inline T_accFirDecHb<int32, int16, kArch1Buff> firDecHbMacSym1buff(T_accFirDecHb<int32, int16, kArch1Buff> acc,
                                                                   T_buff_FirDecHb<int32, int16, kArch1Buff> xbuff,
                                                                   unsigned int xstart,
                                                                   unsigned int ystart,
                                                                   T_buff_256b<int16> zbuff,
                                                                   unsigned int zstart) {
    T_accFirDecHb<int32, int16, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 8;
    const unsigned int kDecFactor = 2;

    retVal.val = lmac8_sym(acc.val, xbuff.val, xstart, xoffsets, xstep, ystart, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<int32, int16, kArch2Buff> firDecHbMacSymCt(T_accFirDecHb<int32, int16, kArch2Buff> acc,
                                                                T_buff_FirDecHb<int32, int16, kArch2Buff> xbuff,
                                                                unsigned int xstart,
                                                                T_buff_FirDecHb<int32, int16, kArch2Buff> ybuff,
                                                                unsigned int ystart,
                                                                unsigned int ct,
                                                                T_buff_256b<int16> zbuff,
                                                                unsigned int zstart) {
    T_accFirDecHb<int32, int16, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 1; // exception to decimation factor, because the centre tap is in 2nd column
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 8;
    const unsigned int kCols = 2;
    const unsigned int kDecFactor = 2;

    T_buff_256b<int16> zbuff2; //= zbuff;
    unsigned int ystartmod = ystart - (kCols - 1) * xstep;
    unsigned int zstartmod = zstart + (kCols - 1);
    int zstepmod = -(int)zstep;

    zbuff2.val = upd_elem(zbuff.val, zstartmod, 0); // zero the centre tap. 16 is z vsize

    // These two commands spoof a lmul_sym_ct command which cannot be used directly because it uses 512b buffers
    // which cannot accommodate the range of data required. Hence it is split into an asym mul (including the centre
    // tap) and an asym mul which must not include the center tap. The center tap is zeroed directly.
    retVal.val = lmac8(acc.val, xbuff.val, xstart, xoffsets, xstep, zbuff.val, zstart, zoffsets, zstep);
    retVal.val = lmac8(retVal.val, ybuff.val, ystartmod, xoffsets, xstep, zbuff2.val, zstartmod, zoffsets, zstepmod);
    return retVal;
}
template <>
inline T_accFirDecHb<int32, int16, kArch1Buff> firDecHbMacSymCt1buff(T_accFirDecHb<int32, int16, kArch1Buff> acc,
                                                                     T_buff_FirDecHb<int32, int16, kArch1Buff> xbuff,
                                                                     unsigned int xstart,
                                                                     unsigned int ystart,
                                                                     unsigned int ct,
                                                                     T_buff_256b<int16> zbuff,
                                                                     unsigned int zstart) {
    T_accFirDecHb<int32, int16, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 8;
    const unsigned int kDecFactor = 2;

    retVal.val = lmac8_sym_ct(acc.val, xbuff.val, xstart, xoffsets, ystart, ct, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = int32,  COEFF = int32>
template <>
inline T_accFirDecHb<int32, int32, kArch2Buff> firDecHbMulSym(T_buff_FirDecHb<int32, int32, kArch2Buff> xbuff,
                                                              unsigned int xstart,
                                                              T_buff_FirDecHb<int32, int32, kArch2Buff> ybuff,
                                                              unsigned int ystart,
                                                              T_buff_256b<int32> zbuff,
                                                              unsigned int zstart) {
    T_accFirDecHb<int32, int32, kArch2Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmul4_sym(xbuff.val, xstart, xoffsets, xstep, ybuff.val, ystart, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<int32, int32, kArch1Buff> firDecHbMulSym1buff(T_buff_FirDecHb<int32, int32, kArch1Buff> xbuff,
                                                                   unsigned int xstart,
                                                                   unsigned int ystart,
                                                                   T_buff_256b<int32> zbuff,
                                                                   unsigned int zstart) {
    T_accFirDecHb<int32, int32, kArch1Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmul4_sym(xbuff.val, xstart, xoffsets, xstep, ystart, zbuff.val, zstart, zoffsets, zstep);
    retVal.uval = lmul4_sym(xbuff.val, xstart + kDecFactor * kLanes, xoffsets, xstep, ystart + kDecFactor * kLanes,
                            zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<int32, int32, kArch2Buff> firDecHbMulSymCt(T_buff_FirDecHb<int32, int32, kArch2Buff> xbuff,
                                                                unsigned int xstart,
                                                                T_buff_FirDecHb<int32, int32, kArch2Buff> ybuff,
                                                                unsigned int ystart,
                                                                unsigned int ct,
                                                                T_buff_256b<int32> zbuff,
                                                                unsigned int zstart) {
    T_accFirDecHb<int32, int32, kArch2Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmul4_sym_ct(xbuff.val, xstart, xoffsets, ybuff.val, ystart, ct, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<int32, int32, kArch1Buff> firDecHbMulSymCt1buff(T_buff_FirDecHb<int32, int32, kArch1Buff> xbuff,
                                                                     unsigned int xstart,
                                                                     unsigned int ystart,
                                                                     unsigned int ct,
                                                                     T_buff_256b<int32> zbuff,
                                                                     unsigned int zstart) {
    T_accFirDecHb<int32, int32, kArch1Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmul4_sym_ct(xbuff.val, xstart, xoffsets, ystart, ct, zbuff.val, zstart, zoffsets, zstep);
    retVal.uval = lmul4_sym_ct(xbuff.val, xstart + kDecFactor * kLanes, xoffsets, ystart + kDecFactor * kLanes, ct,
                               zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<int32, int32, kArch2Buff> firDecHbMacSym(T_accFirDecHb<int32, int32, kArch2Buff> acc,
                                                              T_buff_FirDecHb<int32, int32, kArch2Buff> xbuff,
                                                              unsigned int xstart,
                                                              T_buff_FirDecHb<int32, int32, kArch2Buff> ybuff,
                                                              unsigned int ystart,
                                                              T_buff_256b<int32> zbuff,
                                                              unsigned int zstart) {
    T_accFirDecHb<int32, int32, kArch2Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val =
        lmac4_sym(acc.val, xbuff.val, xstart, xoffsets, xstep, ybuff.val, ystart, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<int32, int32, kArch1Buff> firDecHbMacSym1buff(T_accFirDecHb<int32, int32, kArch1Buff> acc,
                                                                   T_buff_FirDecHb<int32, int32, kArch1Buff> xbuff,
                                                                   unsigned int xstart,
                                                                   unsigned int ystart,
                                                                   T_buff_256b<int32> zbuff,
                                                                   unsigned int zstart) {
    T_accFirDecHb<int32, int32, kArch1Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmac4_sym(acc.val, xbuff.val, xstart, xoffsets, xstep, ystart, zbuff.val, zstart, zoffsets, zstep);
    retVal.uval = lmac4_sym(acc.uval, xbuff.val, xstart + kDecFactor * kLanes, xoffsets, xstep,
                            ystart + kDecFactor * kLanes, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<int32, int32, kArch2Buff> firDecHbMacSymCt(T_accFirDecHb<int32, int32, kArch2Buff> acc,
                                                                T_buff_FirDecHb<int32, int32, kArch2Buff> xbuff,
                                                                unsigned int xstart,
                                                                T_buff_FirDecHb<int32, int32, kArch2Buff> ybuff,
                                                                unsigned int ystart,
                                                                unsigned int ct,
                                                                T_buff_256b<int32> zbuff,
                                                                unsigned int zstart) {
    T_accFirDecHb<int32, int32, kArch2Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val =
        lmac4_sym_ct(acc.val, xbuff.val, xstart, xoffsets, ybuff.val, ystart, ct, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<int32, int32, kArch1Buff> firDecHbMacSymCt1buff(T_accFirDecHb<int32, int32, kArch1Buff> acc,
                                                                     T_buff_FirDecHb<int32, int32, kArch1Buff> xbuff,
                                                                     unsigned int xstart,
                                                                     unsigned int ystart,
                                                                     unsigned int ct,
                                                                     T_buff_256b<int32> zbuff,
                                                                     unsigned int zstart) {
    T_accFirDecHb<int32, int32, kArch1Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmac4_sym_ct(acc.val, xbuff.val, xstart, xoffsets, ystart, ct, zbuff.val, zstart, zoffsets, zstep);
    retVal.uval = lmac4_sym_ct(acc.uval, xbuff.val, xstart + kDecFactor * kLanes, xoffsets,
                               ystart + kDecFactor * kLanes, ct, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = cint32,  COEFF = int16>
template <>
inline T_accFirDecHb<cint32, int16, kArch2Buff> firDecHbMulSym(T_buff_FirDecHb<cint32, int16, kArch2Buff> xbuff,
                                                               unsigned int xstart,
                                                               T_buff_FirDecHb<cint32, int16, kArch2Buff> ybuff,
                                                               unsigned int ystart,
                                                               T_buff_256b<int16> zbuff,
                                                               unsigned int zstart) {
    T_accFirDecHb<cint32, int16, kArch2Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kCols = 2;
    const unsigned int kDecFactor = 2;

    unsigned int ystartmod = ystart - (kCols - 1) * xstep;
    unsigned int zstartmod = zstart + (kCols - 1);
    int zstepmod = -(int)zstep;

    retVal.val = lmul4(xbuff.val, xstart, xoffsets, xstep, zbuff.val, zstart, zoffsets, zstep);
    retVal.val = lmac4(retVal.val, ybuff.val, ystartmod, xoffsets, xstep, zbuff.val, zstartmod, zoffsets, zstepmod);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, int16, kArch1Buff> firDecHbMulSym1buff(T_buff_FirDecHb<cint32, int16, kArch1Buff> xbuff,
                                                                    unsigned int xstart,
                                                                    unsigned int ystart,
                                                                    T_buff_256b<int16> zbuff,
                                                                    unsigned int zstart) {
    T_accFirDecHb<cint32, int16, kArch1Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmul4_sym(xbuff.val, xstart, xoffsets, xstep, ystart, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, int16, kArch2Buff> firDecHbMulSymCt(T_buff_FirDecHb<cint32, int16, kArch2Buff> xbuff,
                                                                 unsigned int xstart,
                                                                 T_buff_FirDecHb<cint32, int16, kArch2Buff> ybuff,
                                                                 unsigned int ystart,
                                                                 unsigned int ct,
                                                                 T_buff_256b<int16> zbuff,
                                                                 unsigned int zstart) {
    T_accFirDecHb<cint32, int16, kArch2Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 1; // exception to decimation factor of 2 because this is the centre tap term
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kCols = 2;
    const unsigned int kDecFactor = 2;

    T_buff_256b<int16> zbuff2; //= zbuff;
    unsigned int ystartmod = ystart - (kCols - 1) * xstep;
    unsigned int zstartmod = zstart + (kCols - 1);
    int zstepmod = -(int)zstep;

    zbuff2.val = upd_elem(zbuff.val, zstart - 1, 0); // zero the centre tap

    // These two commands spoof a lmul_sym_ct command which cannot be used directly because it uses 512b buffers
    // which cannot accommodate the range of data required. Hence it is split into an asym mul (including the centre
    // tap) and an asym mul which must not include the center tap. The center tap is zeroed directly.
    retVal.val = lmul4(xbuff.val, xstart, xoffsets, xstep, zbuff.val, zstart, zoffsets, zstep);
    retVal.val = lmac4(retVal.val, ybuff.val, ystartmod, xoffsets, xstep, //-1? y has to work backwards.
                       zbuff.val, zstartmod, zoffsets, zstepmod);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, int16, kArch1Buff> firDecHbMulSymCt1buff(T_buff_FirDecHb<cint32, int16, kArch1Buff> xbuff,
                                                                      unsigned int xstart,
                                                                      unsigned int ystart,
                                                                      unsigned int ct,
                                                                      T_buff_256b<int16> zbuff,
                                                                      unsigned int zstart) {
    T_accFirDecHb<cint32, int16, kArch1Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmul4_sym_ct(xbuff.val, xstart, xoffsets, ystart, ct, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, int16, kArch2Buff> firDecHbMacSym(T_accFirDecHb<cint32, int16, kArch2Buff> acc,
                                                               T_buff_FirDecHb<cint32, int16, kArch2Buff> xbuff,
                                                               unsigned int xstart,
                                                               T_buff_FirDecHb<cint32, int16, kArch2Buff> ybuff,
                                                               unsigned int ystart,
                                                               T_buff_256b<int16> zbuff,
                                                               unsigned int zstart) {
    T_accFirDecHb<cint32, int16, kArch2Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kCols = 2;
    const unsigned int kDecFactor = 2;

    unsigned int ystartmod = ystart - (kCols - 1) * xstep;
    unsigned int zstartmod = zstart + (kCols - 1);
    int zstepmod = -(int)zstep;

    retVal.val = lmac4(acc.val, xbuff.val, xstart, xoffsets, xstep, zbuff.val, zstart, zoffsets, zstep);
    retVal.val = lmac4(retVal.val, ybuff.val, ystartmod, xoffsets, xstep, zbuff.val, zstartmod, zoffsets, zstepmod);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, int16, kArch1Buff> firDecHbMacSym1buff(T_accFirDecHb<cint32, int16, kArch1Buff> acc,
                                                                    T_buff_FirDecHb<cint32, int16, kArch1Buff> xbuff,
                                                                    unsigned int xstart,
                                                                    unsigned int ystart,
                                                                    T_buff_256b<int16> zbuff,
                                                                    unsigned int zstart) {
    T_accFirDecHb<cint32, int16, kArch1Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmac4_sym(acc.val, xbuff.val, xstart, xoffsets, xstep, ystart, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, int16, kArch2Buff> firDecHbMacSymCt(T_accFirDecHb<cint32, int16, kArch2Buff> acc,
                                                                 T_buff_FirDecHb<cint32, int16, kArch2Buff> xbuff,
                                                                 unsigned int xstart,
                                                                 T_buff_FirDecHb<cint32, int16, kArch2Buff> ybuff,
                                                                 unsigned int ystart,
                                                                 unsigned int ct,
                                                                 T_buff_256b<int16> zbuff,
                                                                 unsigned int zstart) {
    T_accFirDecHb<cint32, int16, kArch2Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kCols = 2;
    const unsigned int kDecFactor = 2;

    T_buff_256b<int16> zbuff2; //= zbuff;
    unsigned int ystartmod = ystart - (kCols - 1) * xstep;
    unsigned int zstartmod = zstart + (kCols - 1);
    int zstepmod = -(int)zstep;

    zbuff2.val = upd_elem(zbuff.val, zstartmod, 0); // zero the centre tap. 16 is z vsize

    // These two commands spoof a lmul_sym_ct command which cannot be used directly because it uses 512b buffers
    // which cannot accommodate the range of data required. Hence it is split into an asym mul (including the centre
    // tap) and an asym mul which must not include the center tap. The center tap is zeroed directly.
    retVal.val = lmac4(acc.val, xbuff.val, xstart, xoffsets, xstep, zbuff.val, zstart, zoffsets, zstep);
    retVal.val = lmac4(retVal.val, ybuff.val, ystartmod, xoffsets, xstep, zbuff2.val, zstartmod, zoffsets, zstepmod);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, int16, kArch1Buff> firDecHbMacSymCt1buff(T_accFirDecHb<cint32, int16, kArch1Buff> acc,
                                                                      T_buff_FirDecHb<cint32, int16, kArch1Buff> xbuff,
                                                                      unsigned int xstart,
                                                                      unsigned int ystart,
                                                                      unsigned int ct,
                                                                      T_buff_256b<int16> zbuff,
                                                                      unsigned int zstart) {
    T_accFirDecHb<cint32, int16, kArch1Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmac4_sym_ct(acc.val, xbuff.val, xstart, xoffsets, ystart, ct, zbuff.val, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = cint32,  COEFF = cint16>
template <>
inline T_accFirDecHb<cint32, cint16, kArch2Buff> firDecHbMulSym(T_buff_FirDecHb<cint32, cint16, kArch2Buff> xbuff,
                                                                unsigned int xstart,
                                                                T_buff_FirDecHb<cint32, cint16, kArch2Buff> ybuff,
                                                                unsigned int ystart,
                                                                T_buff_256b<cint16> zbuff,
                                                                unsigned int zstart) {
    T_accFirDecHb<cint32, cint16, kArch2Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmul4_sym(xbuff.val, xstart, xoffsets, ybuff.val, ystart, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, cint16, kArch1Buff> firDecHbMulSym1buff(T_buff_FirDecHb<cint32, cint16, kArch1Buff> xbuff,
                                                                     unsigned int xstart,
                                                                     unsigned int ystart,
                                                                     T_buff_256b<cint16> zbuff,
                                                                     unsigned int zstart) {
    T_accFirDecHb<cint32, cint16, kArch1Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmul4_sym(xbuff.val, xstart, xoffsets, ystart, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, cint16, kArch2Buff> firDecHbMulSymCt(T_buff_FirDecHb<cint32, cint16, kArch2Buff> xbuff,
                                                                  unsigned int xstart,
                                                                  T_buff_FirDecHb<cint32, cint16, kArch2Buff> ybuff,
                                                                  unsigned int ystart,
                                                                  unsigned int ct,
                                                                  T_buff_256b<cint16> zbuff,
                                                                  unsigned int zstart) {
    T_accFirDecHb<cint32, cint16, kArch2Buff> retVal;
    // This would only be called for a fir of length 1
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, cint16, kArch1Buff> firDecHbMulSymCt1buff(
    T_buff_FirDecHb<cint32, cint16, kArch1Buff> xbuff,
    unsigned int xstart,
    unsigned int ystart,
    unsigned int ct,
    T_buff_256b<cint16> zbuff,
    unsigned int zstart) {
    T_accFirDecHb<cint32, cint16, kArch1Buff> retVal;
    // This would only be called for a fir of length 1
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, cint16, kArch2Buff> firDecHbMacSym(T_accFirDecHb<cint32, cint16, kArch2Buff> acc,
                                                                T_buff_FirDecHb<cint32, cint16, kArch2Buff> xbuff,
                                                                unsigned int xstart,
                                                                T_buff_FirDecHb<cint32, cint16, kArch2Buff> ybuff,
                                                                unsigned int ystart,
                                                                T_buff_256b<cint16> zbuff,
                                                                unsigned int zstart) {
    T_accFirDecHb<cint32, cint16, kArch2Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmac4_sym(acc.val, xbuff.val, xstart, xoffsets, ybuff.val, ystart, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, cint16, kArch1Buff> firDecHbMacSym1buff(T_accFirDecHb<cint32, cint16, kArch1Buff> acc,
                                                                     T_buff_FirDecHb<cint32, cint16, kArch1Buff> xbuff,
                                                                     unsigned int xstart,
                                                                     unsigned int ystart,
                                                                     T_buff_256b<cint16> zbuff,
                                                                     unsigned int zstart) {
    T_accFirDecHb<cint32, cint16, kArch1Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmac4_sym(acc.val, xbuff.val, xstart, xoffsets, ystart, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, cint16, kArch2Buff> firDecHbMacSymCt(T_accFirDecHb<cint32, cint16, kArch2Buff> acc,
                                                                  T_buff_FirDecHb<cint32, cint16, kArch2Buff> xbuff,
                                                                  unsigned int xstart,
                                                                  T_buff_FirDecHb<cint32, cint16, kArch2Buff> ybuff,
                                                                  unsigned int ystart,
                                                                  unsigned int ct,
                                                                  T_buff_256b<cint16> zbuff,
                                                                  unsigned int zstart) {
    T_accFirDecHb<cint32, cint16, kArch2Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmac4(acc.val, xbuff.val, xstart, xoffsets,
                       //         ybuff, ystart,                   ct,
                       zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, cint16, kArch1Buff> firDecHbMacSymCt1buff(
    T_accFirDecHb<cint32, cint16, kArch1Buff> acc,
    T_buff_FirDecHb<cint32, cint16, kArch1Buff> xbuff,
    unsigned int xstart,
    unsigned int ystart,
    unsigned int ct,
    T_buff_256b<cint16> zbuff,
    unsigned int zstart) {
    T_accFirDecHb<cint32, cint16, kArch1Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    // with one column, the centre tap is just a mac
    retVal.val = lmac4(acc.val, xbuff.val, xstart, xoffsets,
                       // ystart,                   ct,
                       zbuff.val, zstart, zoffsets);
    return retVal;
}

// DATA = cint32,  COEFF = int32>
template <>
inline T_accFirDecHb<cint32, int32, kArch2Buff> firDecHbMulSym(T_buff_FirDecHb<cint32, int32, kArch2Buff> xbuff,
                                                               unsigned int xstart,
                                                               T_buff_FirDecHb<cint32, int32, kArch2Buff> ybuff,
                                                               unsigned int ystart,
                                                               T_buff_256b<int32> zbuff,
                                                               unsigned int zstart) {
    T_accFirDecHb<cint32, int32, kArch2Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmul4_sym(xbuff.val, xstart, xoffsets, ybuff.val, ystart, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, int32, kArch1Buff> firDecHbMulSym1buff(T_buff_FirDecHb<cint32, int32, kArch1Buff> xbuff,
                                                                    unsigned int xstart,
                                                                    unsigned int ystart,
                                                                    T_buff_256b<int32> zbuff,
                                                                    unsigned int zstart) {
    T_accFirDecHb<cint32, int32, kArch1Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmul4_sym(xbuff.val, xstart, xoffsets, ystart, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, int32, kArch2Buff> firDecHbMulSymCt(T_buff_FirDecHb<cint32, int32, kArch2Buff> xbuff,
                                                                 unsigned int xstart,
                                                                 T_buff_FirDecHb<cint32, int32, kArch2Buff> ybuff,
                                                                 unsigned int ystart,
                                                                 unsigned int ct,
                                                                 T_buff_256b<int32> zbuff,
                                                                 unsigned int zstart) {
    T_accFirDecHb<cint32, int32, kArch2Buff> retVal;
    // this would only be used for a fir of length 1!
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, int32, kArch1Buff> firDecHbMulSymCt1buff(T_buff_FirDecHb<cint32, int32, kArch1Buff> xbuff,
                                                                      unsigned int xstart,
                                                                      unsigned int ystart,
                                                                      unsigned int ct,
                                                                      T_buff_256b<int32> zbuff,
                                                                      unsigned int zstart) {
    T_accFirDecHb<cint32, int32, kArch1Buff> retVal;
    // this would only be used for a fir of length 1!
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, int32, kArch2Buff> firDecHbMacSym(T_accFirDecHb<cint32, int32, kArch2Buff> acc,
                                                               T_buff_FirDecHb<cint32, int32, kArch2Buff> xbuff,
                                                               unsigned int xstart,
                                                               T_buff_FirDecHb<cint32, int32, kArch2Buff> ybuff,
                                                               unsigned int ystart,
                                                               T_buff_256b<int32> zbuff,
                                                               unsigned int zstart) {
    T_accFirDecHb<cint32, int32, kArch2Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmac4_sym(acc.val, xbuff.val, xstart, xoffsets, ybuff.val, ystart, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, int32, kArch1Buff> firDecHbMacSym1buff(T_accFirDecHb<cint32, int32, kArch1Buff> acc,
                                                                    T_buff_FirDecHb<cint32, int32, kArch1Buff> xbuff,
                                                                    unsigned int xstart,
                                                                    unsigned int ystart,
                                                                    T_buff_256b<int32> zbuff,
                                                                    unsigned int zstart) {
    T_accFirDecHb<cint32, int32, kArch1Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmac4_sym(acc.val, xbuff.val, xstart, xoffsets, ystart, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, int32, kArch2Buff> firDecHbMacSymCt(T_accFirDecHb<cint32, int32, kArch2Buff> acc,
                                                                 T_buff_FirDecHb<cint32, int32, kArch2Buff> xbuff,
                                                                 unsigned int xstart,
                                                                 T_buff_FirDecHb<cint32, int32, kArch2Buff> ybuff,
                                                                 unsigned int ystart,
                                                                 unsigned int ct,
                                                                 T_buff_256b<int32> zbuff,
                                                                 unsigned int zstart) {
    T_accFirDecHb<cint32, int32, kArch2Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmac4(acc.val, xbuff.val, xstart, xoffsets,
                       // ybuff, ystart,
                       zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, int32, kArch1Buff> firDecHbMacSymCt1buff(T_accFirDecHb<cint32, int32, kArch1Buff> acc,
                                                                      T_buff_FirDecHb<cint32, int32, kArch1Buff> xbuff,
                                                                      unsigned int xstart,
                                                                      unsigned int ystart,
                                                                      unsigned int ct,
                                                                      T_buff_256b<int32> zbuff,
                                                                      unsigned int zstart) {
    T_accFirDecHb<cint32, int32, kArch1Buff> retVal;
    const unsigned int xoffsets = 0x6420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = lmac4(acc.val, xbuff.val, xstart, xoffsets,
                       //       ystart,
                       zbuff.val, zstart, zoffsets);
    return retVal;
}

// DATA = cint32,  COEFF = cint32>
template <>
inline T_accFirDecHb<cint32, cint32, kArch2Buff> firDecHbMulSym(T_buff_FirDecHb<cint32, cint32, kArch2Buff> xbuff,
                                                                unsigned int xstart,
                                                                T_buff_FirDecHb<cint32, cint32, kArch2Buff> ybuff,
                                                                unsigned int ystart,
                                                                T_buff_256b<cint32> zbuff,
                                                                unsigned int zstart) {
    T_accFirDecHb<cint32, cint32, kArch2Buff> retVal;
    const unsigned int xoffsets = 0x20;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 2;
    const unsigned int kDecFactor = 2;

    retVal.val = lmul2_sym(xbuff.val, xstart, xoffsets, ybuff.val, ystart, zbuff.val, zstart, zoffsets);
    retVal.uval = lmul2_sym(xbuff.val, xstart + kDecFactor * kLanes, xoffsets, ybuff.val, ystart + kDecFactor * kLanes,
                            zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, cint32, kArch1Buff> firDecHbMulSym1buff(T_buff_FirDecHb<cint32, cint32, kArch1Buff> xbuff,
                                                                     unsigned int xstart,
                                                                     unsigned int ystart,
                                                                     T_buff_256b<cint32> zbuff,
                                                                     unsigned int zstart) {
    T_accFirDecHb<cint32, cint32, kArch1Buff> retVal;
    const unsigned int xoffsets = 0x20;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 2;
    const unsigned int kDecFactor = 2;

    retVal.val = lmul2_sym(xbuff.val, xstart, xoffsets, ystart, zbuff.val, zstart, zoffsets);
    retVal.uval = lmul2_sym(xbuff.val, xstart + kDecFactor * kLanes, xoffsets, ystart + kDecFactor * kLanes, zbuff.val,
                            zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, cint32, kArch2Buff> firDecHbMulSymCt(T_buff_FirDecHb<cint32, cint32, kArch2Buff> xbuff,
                                                                  unsigned int xstart,
                                                                  T_buff_FirDecHb<cint32, cint32, kArch2Buff> ybuff,
                                                                  unsigned int ystart,
                                                                  unsigned int ct,
                                                                  T_buff_256b<cint32> zbuff,
                                                                  unsigned int zstart) {
    T_accFirDecHb<cint32, cint32, kArch2Buff> retVal;
    // This would only be called for a fir of length 1
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, cint32, kArch1Buff> firDecHbMulSymCt1buff(
    T_buff_FirDecHb<cint32, cint32, kArch1Buff> xbuff,
    unsigned int xstart,
    unsigned int ystart,
    unsigned int ct,
    T_buff_256b<cint32> zbuff,
    unsigned int zstart) {
    T_accFirDecHb<cint32, cint32, kArch1Buff> retVal;
    // This would only be called for a fir of length 1
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, cint32, kArch2Buff> firDecHbMacSym(T_accFirDecHb<cint32, cint32, kArch2Buff> acc,
                                                                T_buff_FirDecHb<cint32, cint32, kArch2Buff> xbuff,
                                                                unsigned int xstart,
                                                                T_buff_FirDecHb<cint32, cint32, kArch2Buff> ybuff,
                                                                unsigned int ystart,
                                                                T_buff_256b<cint32> zbuff,
                                                                unsigned int zstart) {
    T_accFirDecHb<cint32, cint32, kArch2Buff> retVal;
    const unsigned int xoffsets = 0x20;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 2;
    const unsigned int kDecFactor = 2;

    retVal.val = lmac2_sym(acc.val, xbuff.val, xstart, xoffsets, ybuff.val, ystart, zbuff.val, zstart, zoffsets);
    retVal.uval = lmac2_sym(acc.uval, xbuff.val, xstart + kDecFactor * kLanes, xoffsets, ybuff.val,
                            ystart + kDecFactor * kLanes, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, cint32, kArch1Buff> firDecHbMacSym1buff(T_accFirDecHb<cint32, cint32, kArch1Buff> acc,
                                                                     T_buff_FirDecHb<cint32, cint32, kArch1Buff> xbuff,
                                                                     unsigned int xstart,
                                                                     unsigned int ystart,
                                                                     T_buff_256b<cint32> zbuff,
                                                                     unsigned int zstart) {
    T_accFirDecHb<cint32, cint32, kArch1Buff> retVal;
    const unsigned int xoffsets = 0x20;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 2;
    const unsigned int kDecFactor = 2;

    retVal.val = lmac2_sym(acc.val, xbuff.val, xstart, xoffsets, ystart, zbuff.val, zstart, zoffsets);
    retVal.uval = lmac2_sym(acc.uval, xbuff.val, xstart + kDecFactor * kLanes, xoffsets, ystart + kDecFactor * kLanes,
                            zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, cint32, kArch2Buff> firDecHbMacSymCt(T_accFirDecHb<cint32, cint32, kArch2Buff> acc,
                                                                  T_buff_FirDecHb<cint32, cint32, kArch2Buff> xbuff,
                                                                  unsigned int xstart,
                                                                  T_buff_FirDecHb<cint32, cint32, kArch2Buff> ybuff,
                                                                  unsigned int ystart,
                                                                  unsigned int ct,
                                                                  T_buff_256b<cint32> zbuff,
                                                                  unsigned int zstart) {
    T_accFirDecHb<cint32, cint32, kArch2Buff> retVal;
    const unsigned int xoffsets = 0x20;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 2;
    const unsigned int kDecFactor = 2;

    retVal.val = lmac2(acc.val, xbuff.val, xstart, xoffsets,
                       // ybuff, ystart,                   ct,
                       zbuff.val, zstart, zoffsets);
    retVal.uval = lmac2(acc.uval, xbuff.val, xstart + kDecFactor * kLanes, xoffsets,
                        // ybuff, ystart+kDecFactor*kLanes, ct,
                        zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cint32, cint32, kArch1Buff> firDecHbMacSymCt1buff(
    T_accFirDecHb<cint32, cint32, kArch1Buff> acc,
    T_buff_FirDecHb<cint32, cint32, kArch1Buff> xbuff,
    unsigned int xstart,
    unsigned int ystart,
    unsigned int ct,
    T_buff_256b<cint32> zbuff,
    unsigned int zstart) {
    T_accFirDecHb<cint32, cint32, kArch1Buff> retVal;
    const unsigned int xoffsets = 0x20;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 2;
    const unsigned int kDecFactor = 2;

    retVal.val = lmac2(acc.val, xbuff.val, xstart, xoffsets,
                       // ystart,                   ct,
                       zbuff.val, zstart, zoffsets);
    retVal.uval = lmac2(acc.uval, xbuff.val, xstart + kDecFactor * kLanes, xoffsets,
                        // ystart+kDecFactor*kLanes, ct,
                        zbuff.val, zstart, zoffsets);
    return retVal;
}

// DATA = float,  COEFF = float>
template <>
inline T_accFirDecHb<float, float, kArch2Buff> firDecHbMulSym(T_buff_FirDecHb<float, float, kArch2Buff> xbuff,
                                                              unsigned int xstart,
                                                              T_buff_FirDecHb<float, float, kArch2Buff> ybuff,
                                                              unsigned int ystart,
                                                              T_buff_256b<float> zbuff,
                                                              unsigned int zstart) {
    T_accFirDecHb<float, float, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 8;
    const unsigned int kDecFactor = 2;

    // since there is no fpmul with preadd, simply perform the 2 sides as assymmetric muls.
    retVal.val = fpmul(xbuff.val, xstart, xoffsets, zbuff.val, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, ybuff.val, ystart, xoffsets, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<float, float, kArch1Buff> firDecHbMulSym1buff(T_buff_FirDecHb<float, float, kArch1Buff> xbuff,
                                                                   unsigned int xstart,
                                                                   unsigned int ystart,
                                                                   T_buff_256b<float> zbuff,
                                                                   unsigned int zstart) {
    T_accFirDecHb<float, float, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 8;
    const unsigned int kDecFactor = 2;

    retVal.val = fpmul(xbuff.val, xstart, xoffsets, zbuff.val, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, xbuff.val, ystart, xoffsets, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<float, float, kArch2Buff> firDecHbMulSymCt(T_buff_FirDecHb<float, float, kArch2Buff> xbuff,
                                                                unsigned int xstart,
                                                                T_buff_FirDecHb<float, float, kArch2Buff> ybuff,
                                                                unsigned int ystart,
                                                                unsigned int ct,
                                                                T_buff_256b<float> zbuff,
                                                                unsigned int zstart) {
    T_accFirDecHb<float, float, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 8;
    const unsigned int kDecFactor = 2;

    // This would be used only for a fir of length 1.

    return retVal;
}
template <>
inline T_accFirDecHb<float, float, kArch1Buff> firDecHbMulSymCt1buff(T_buff_FirDecHb<float, float, kArch1Buff> xbuff,
                                                                     unsigned int xstart,
                                                                     unsigned int ystart,
                                                                     unsigned int ct,
                                                                     T_buff_256b<float> zbuff,
                                                                     unsigned int zstart) {
    T_accFirDecHb<float, float, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 8;
    const unsigned int kDecFactor = 2;

    // This would be used only for a FIR of length 1.

    return retVal;
}
template <>
inline T_accFirDecHb<float, float, kArch2Buff> firDecHbMacSym(T_accFirDecHb<float, float, kArch2Buff> acc,
                                                              T_buff_FirDecHb<float, float, kArch2Buff> xbuff,
                                                              unsigned int xstart,
                                                              T_buff_FirDecHb<float, float, kArch2Buff> ybuff,
                                                              unsigned int ystart,
                                                              T_buff_256b<float> zbuff,
                                                              unsigned int zstart) {
    T_accFirDecHb<float, float, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 8;
    const unsigned int kDecFactor = 2;

    retVal.val = fpmac(acc.val, xbuff.val, xstart, xoffsets, zbuff.val, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, ybuff.val, ystart, xoffsets, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<float, float, kArch1Buff> firDecHbMacSym1buff(T_accFirDecHb<float, float, kArch1Buff> acc,
                                                                   T_buff_FirDecHb<float, float, kArch1Buff> xbuff,
                                                                   unsigned int xstart,
                                                                   unsigned int ystart,
                                                                   T_buff_256b<float> zbuff,
                                                                   unsigned int zstart) {
    T_accFirDecHb<float, float, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 8;
    const unsigned int kDecFactor = 2;

    retVal.val = fpmac(acc.val, xbuff.val, xstart, xoffsets, zbuff.val, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, xbuff.val, ystart, xoffsets, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<float, float, kArch2Buff> firDecHbMacSymCt(T_accFirDecHb<float, float, kArch2Buff> acc,
                                                                T_buff_FirDecHb<float, float, kArch2Buff> xbuff,
                                                                unsigned int xstart,
                                                                T_buff_FirDecHb<float, float, kArch2Buff> ybuff,
                                                                unsigned int ystart,
                                                                unsigned int ct,
                                                                T_buff_256b<float> zbuff,
                                                                unsigned int zstart) {
    T_accFirDecHb<float, float, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 8;
    const unsigned int kDecFactor = 2;

    retVal.val = fpmac(acc.val, xbuff.val, xstart, xoffsets, zbuff.val, zstart, zoffsets);
    // the centre tap requires no symmetric pair.
    return retVal;
}
template <>
inline T_accFirDecHb<float, float, kArch1Buff> firDecHbMacSymCt1buff(T_accFirDecHb<float, float, kArch1Buff> acc,
                                                                     T_buff_FirDecHb<float, float, kArch1Buff> xbuff,
                                                                     unsigned int xstart,
                                                                     unsigned int ystart,
                                                                     unsigned int ct,
                                                                     T_buff_256b<float> zbuff,
                                                                     unsigned int zstart) {
    T_accFirDecHb<float, float, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 8;
    const unsigned int kDecFactor = 2;

    retVal.val = fpmac(acc.val, xbuff.val, xstart, xoffsets, zbuff.val, zstart, zoffsets);
    // The centre tap requires no symmetric pair term.
    return retVal;
}

// DATA = cfloat,  COEFF = float>
template <>
inline T_accFirDecHb<cfloat, float, kArch2Buff> firDecHbMulSym(T_buff_FirDecHb<cfloat, float, kArch2Buff> xbuff,
                                                               unsigned int xstart,
                                                               T_buff_FirDecHb<cfloat, float, kArch2Buff> ybuff,
                                                               unsigned int ystart,
                                                               T_buff_256b<float> zbuff,
                                                               unsigned int zstart) {
    T_accFirDecHb<cfloat, float, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    // since there is no fpmul with preadd, simply perform the 2 sides as assymmetric muls.
    retVal.val = fpmul(xbuff.val, xstart, xoffsets, zbuff.val, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, ybuff.val, ystart, xoffsets, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cfloat, float, kArch1Buff> firDecHbMulSym1buff(T_buff_FirDecHb<cfloat, float, kArch1Buff> xbuff,
                                                                    unsigned int xstart,
                                                                    unsigned int ystart,
                                                                    T_buff_256b<float> zbuff,
                                                                    unsigned int zstart) {
    T_accFirDecHb<cfloat, float, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = fpmul(xbuff.val, xstart, xoffsets, zbuff.val, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, xbuff.val, ystart, xoffsets, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cfloat, float, kArch2Buff> firDecHbMulSymCt(T_buff_FirDecHb<cfloat, float, kArch2Buff> xbuff,
                                                                 unsigned int xstart,
                                                                 T_buff_FirDecHb<cfloat, float, kArch2Buff> ybuff,
                                                                 unsigned int ystart,
                                                                 unsigned int ct,
                                                                 T_buff_256b<float> zbuff,
                                                                 unsigned int zstart) {
    T_accFirDecHb<cfloat, float, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    // This would be used only for a fir of length 1.

    return retVal;
}
template <>
inline T_accFirDecHb<cfloat, float, kArch1Buff> firDecHbMulSymCt1buff(T_buff_FirDecHb<cfloat, float, kArch1Buff> xbuff,
                                                                      unsigned int xstart,
                                                                      unsigned int ystart,
                                                                      unsigned int ct,
                                                                      T_buff_256b<float> zbuff,
                                                                      unsigned int zstart) {
    T_accFirDecHb<cfloat, float, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    // This would be used only for a FIR of length 1.

    return retVal;
}
template <>
inline T_accFirDecHb<cfloat, float, kArch2Buff> firDecHbMacSym(T_accFirDecHb<cfloat, float, kArch2Buff> acc,
                                                               T_buff_FirDecHb<cfloat, float, kArch2Buff> xbuff,
                                                               unsigned int xstart,
                                                               T_buff_FirDecHb<cfloat, float, kArch2Buff> ybuff,
                                                               unsigned int ystart,
                                                               T_buff_256b<float> zbuff,
                                                               unsigned int zstart) {
    T_accFirDecHb<cfloat, float, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = fpmac(acc.val, xbuff.val, xstart, xoffsets, zbuff.val, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, ybuff.val, ystart, xoffsets, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cfloat, float, kArch1Buff> firDecHbMacSym1buff(T_accFirDecHb<cfloat, float, kArch1Buff> acc,
                                                                    T_buff_FirDecHb<cfloat, float, kArch1Buff> xbuff,
                                                                    unsigned int xstart,
                                                                    unsigned int ystart,
                                                                    T_buff_256b<float> zbuff,
                                                                    unsigned int zstart) {
    T_accFirDecHb<cfloat, float, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = fpmac(acc.val, xbuff.val, xstart, xoffsets, zbuff.val, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, xbuff.val, ystart, xoffsets, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cfloat, float, kArch2Buff> firDecHbMacSymCt(T_accFirDecHb<cfloat, float, kArch2Buff> acc,
                                                                 T_buff_FirDecHb<cfloat, float, kArch2Buff> xbuff,
                                                                 unsigned int xstart,
                                                                 T_buff_FirDecHb<cfloat, float, kArch2Buff> ybuff,
                                                                 unsigned int ystart,
                                                                 unsigned int ct,
                                                                 T_buff_256b<float> zbuff,
                                                                 unsigned int zstart) {
    T_accFirDecHb<cfloat, float, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = fpmac(acc.val, xbuff.val, xstart, xoffsets, zbuff.val, zstart, zoffsets);
    // the centre tap requires no symmetric pair.
    return retVal;
}
template <>
inline T_accFirDecHb<cfloat, float, kArch1Buff> firDecHbMacSymCt1buff(T_accFirDecHb<cfloat, float, kArch1Buff> acc,
                                                                      T_buff_FirDecHb<cfloat, float, kArch1Buff> xbuff,
                                                                      unsigned int xstart,
                                                                      unsigned int ystart,
                                                                      unsigned int ct,
                                                                      T_buff_256b<float> zbuff,
                                                                      unsigned int zstart) {
    T_accFirDecHb<cfloat, float, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = fpmac(acc.val, xbuff.val, xstart, xoffsets, zbuff.val, zstart, zoffsets);
    // The centre tap requires no symmetric pair term.
    return retVal;
}

// DATA = cfloat,  COEFF = cfloat>
template <>
inline T_accFirDecHb<cfloat, cfloat, kArch2Buff> firDecHbMulSym(T_buff_FirDecHb<cfloat, cfloat, kArch2Buff> xbuff,
                                                                unsigned int xstart,
                                                                T_buff_FirDecHb<cfloat, cfloat, kArch2Buff> ybuff,
                                                                unsigned int ystart,
                                                                T_buff_256b<cfloat> zbuff,
                                                                unsigned int zstart) {
    T_accFirDecHb<cfloat, cfloat, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    // since there is no fpmul with preadd, simply perform the 2 sides as assymmetric muls.
    retVal.val = fpmul(xbuff.val, xstart, xoffsets, zbuff.val, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, ybuff.val, ystart, xoffsets, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cfloat, cfloat, kArch1Buff> firDecHbMulSym1buff(T_buff_FirDecHb<cfloat, cfloat, kArch1Buff> xbuff,
                                                                     unsigned int xstart,
                                                                     unsigned int ystart,
                                                                     T_buff_256b<cfloat> zbuff,
                                                                     unsigned int zstart) {
    T_accFirDecHb<cfloat, cfloat, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = fpmul(xbuff.val, xstart, xoffsets, zbuff.val, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, xbuff.val, ystart, xoffsets, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cfloat, cfloat, kArch2Buff> firDecHbMulSymCt(T_buff_FirDecHb<cfloat, cfloat, kArch2Buff> xbuff,
                                                                  unsigned int xstart,
                                                                  T_buff_FirDecHb<cfloat, cfloat, kArch2Buff> ybuff,
                                                                  unsigned int ystart,
                                                                  unsigned int ct,
                                                                  T_buff_256b<cfloat> zbuff,
                                                                  unsigned int zstart) {
    T_accFirDecHb<cfloat, cfloat, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    // This would be used only for a fir of length 1.

    return retVal;
}
template <>
inline T_accFirDecHb<cfloat, cfloat, kArch1Buff> firDecHbMulSymCt1buff(
    T_buff_FirDecHb<cfloat, cfloat, kArch1Buff> xbuff,
    unsigned int xstart,
    unsigned int ystart,
    unsigned int ct,
    T_buff_256b<cfloat> zbuff,
    unsigned int zstart) {
    T_accFirDecHb<cfloat, cfloat, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    // This would be used only for a FIR of length 1.

    return retVal;
}
template <>
inline T_accFirDecHb<cfloat, cfloat, kArch2Buff> firDecHbMacSym(T_accFirDecHb<cfloat, cfloat, kArch2Buff> acc,
                                                                T_buff_FirDecHb<cfloat, cfloat, kArch2Buff> xbuff,
                                                                unsigned int xstart,
                                                                T_buff_FirDecHb<cfloat, cfloat, kArch2Buff> ybuff,
                                                                unsigned int ystart,
                                                                T_buff_256b<cfloat> zbuff,
                                                                unsigned int zstart) {
    T_accFirDecHb<cfloat, cfloat, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = fpmac(acc.val, xbuff.val, xstart, xoffsets, zbuff.val, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, ybuff.val, ystart, xoffsets, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cfloat, cfloat, kArch1Buff> firDecHbMacSym1buff(T_accFirDecHb<cfloat, cfloat, kArch1Buff> acc,
                                                                     T_buff_FirDecHb<cfloat, cfloat, kArch1Buff> xbuff,
                                                                     unsigned int xstart,
                                                                     unsigned int ystart,
                                                                     T_buff_256b<cfloat> zbuff,
                                                                     unsigned int zstart) {
    T_accFirDecHb<cfloat, cfloat, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = fpmac(acc.val, xbuff.val, xstart, xoffsets, zbuff.val, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, xbuff.val, ystart, xoffsets, zbuff.val, zstart, zoffsets);
    return retVal;
}
template <>
inline T_accFirDecHb<cfloat, cfloat, kArch2Buff> firDecHbMacSymCt(T_accFirDecHb<cfloat, cfloat, kArch2Buff> acc,
                                                                  T_buff_FirDecHb<cfloat, cfloat, kArch2Buff> xbuff,
                                                                  unsigned int xstart,
                                                                  T_buff_FirDecHb<cfloat, cfloat, kArch2Buff> ybuff,
                                                                  unsigned int ystart,
                                                                  unsigned int ct,
                                                                  T_buff_256b<cfloat> zbuff,
                                                                  unsigned int zstart) {
    T_accFirDecHb<cfloat, cfloat, kArch2Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = fpmac(acc.val, xbuff.val, xstart, xoffsets, zbuff.val, zstart, zoffsets);
    // the centre tap requires no symmetric pair.
    return retVal;
}
template <>
inline T_accFirDecHb<cfloat, cfloat, kArch1Buff> firDecHbMacSymCt1buff(
    T_accFirDecHb<cfloat, cfloat, kArch1Buff> acc,
    T_buff_FirDecHb<cfloat, cfloat, kArch1Buff> xbuff,
    unsigned int xstart,
    unsigned int ystart,
    unsigned int ct,
    T_buff_256b<cfloat> zbuff,
    unsigned int zstart) {
    T_accFirDecHb<cfloat, cfloat, kArch1Buff> retVal;
    const unsigned int xoffsets = 0xECA86420;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    const unsigned int kDecFactor = 2;

    retVal.val = fpmac(acc.val, xbuff.val, xstart, xoffsets, zbuff.val, zstart, zoffsets);
    // The centre tap requires no symmetric pair term.
    return retVal;
}

#define CASC_IN_TRUE true
#define CASC_IN_FALSE false
#define CASC_OUT_TRUE true
#define CASC_OUT_FALSE false

// Overloaded function to write to cascade output.
template <typename TT_DATA, typename TT_COEFF, eArchType TP_ARCH>
inline void writeCascade(T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface,
                         T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> acc) {
    put_mcd(ext_lo(acc.val));
    put_mcd(ext_hi(acc.val));
}
// specialized functions for kArch1Buff 2 x 384-bit accs
template <>
inline void writeCascade(T_outputIF<CASC_OUT_TRUE, cint16> outInterface, T_accFirDecHb<cint16, int16, kArch1Buff> acc) {
    put_mcd(acc.val);
    put_mcd(acc.uval);
}
template <>
inline void writeCascade(T_outputIF<CASC_OUT_TRUE, cint16> outInterface,
                         T_accFirDecHb<cint16, cint16, kArch1Buff> acc) {
    put_mcd(acc.val);
    put_mcd(acc.uval);
}
template <>
inline void writeCascade(T_outputIF<CASC_OUT_TRUE, int32> outInterface, T_accFirDecHb<int32, int32, kArch1Buff> acc) {
    put_mcd(acc.val);
    put_mcd(acc.uval);
}
template <>
inline void writeCascade(T_outputIF<CASC_OUT_TRUE, cint32> outInterface,
                         T_accFirDecHb<cint32, cint32, kArch1Buff> acc) {
    put_mcd(acc.val);
    put_mcd(acc.uval);
}
template <>
inline void writeCascade(T_outputIF<CASC_OUT_TRUE, float> outInterface, T_accFirDecHb<float, float, kArch1Buff> acc) {
    put_mcd(ups(as_v8int32(acc.val), 0));
}
template <>
inline void writeCascade(T_outputIF<CASC_OUT_TRUE, cfloat> outInterface, T_accFirDecHb<cfloat, float, kArch1Buff> acc) {
    put_mcd((ups(as_v4cint32(acc.val), 0)));
}
template <>
inline void writeCascade(T_outputIF<CASC_OUT_TRUE, cfloat> outInterface,
                         T_accFirDecHb<cfloat, cfloat, kArch1Buff> acc) {
    put_mcd((ups(as_v4cint32(acc.val), 0)));
}
// specialized functions for kArch2Buff 384-bit accs
template <>
inline void writeCascade(T_outputIF<CASC_OUT_TRUE, int16> outInterface, T_accFirDecHb<int16, int16, kArch2Buff> acc) {
    put_mcd(acc.val);
}
template <>
inline void writeCascade(T_outputIF<CASC_OUT_TRUE, cint16> outInterface, T_accFirDecHb<cint16, int16, kArch2Buff> acc) {
    put_mcd(acc.val);
}
template <>
inline void writeCascade(T_outputIF<CASC_OUT_TRUE, cint16> outInterface,
                         T_accFirDecHb<cint16, cint16, kArch2Buff> acc) {
    put_mcd(acc.val);
}
template <>
inline void writeCascade(T_outputIF<CASC_OUT_TRUE, int32> outInterface, T_accFirDecHb<int32, int32, kArch2Buff> acc) {
    put_mcd(acc.val);
}
// specialized function for kArch2Buff 2 x 384-bit accs
template <>
inline void writeCascade(T_outputIF<CASC_OUT_TRUE, cint32> outInterface,
                         T_accFirDecHb<cint32, cint32, kArch2Buff> acc) {
    put_mcd(acc.val);
    put_mcd(acc.uval);
}
template <>
inline void writeCascade(T_outputIF<CASC_OUT_TRUE, float> outInterface, T_accFirDecHb<float, float, kArch2Buff> acc) {
    put_mcd(ups(as_v8int32(acc.val), 0));
}
template <>
inline void writeCascade(T_outputIF<CASC_OUT_TRUE, cfloat> outInterface, T_accFirDecHb<cfloat, float, kArch2Buff> acc) {
    put_mcd((ups(as_v4cint32(acc.val), 0)));
}
template <>
inline void writeCascade(T_outputIF<CASC_OUT_TRUE, cfloat> outInterface,
                         T_accFirDecHb<cfloat, cfloat, kArch2Buff> acc) {
    put_mcd((ups(as_v4cint32(acc.val), 0)));
}

// Overloaded function to skip writing to cascade output.
template <typename TT_DATA, typename TT_COEFF, eArchType TP_ARCH>
inline void writeCascade(T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface,
                         T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> acc) {
    // Do nothing
}

// Overloaded function to write to window output.
template <typename TT_DATA, typename TT_COEFF, eArchType TP_ARCH, unsigned int TP_NUM_OUTPUTS>
inline void writeWindow(T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface,
                        T_outValFiRDecHb<TT_DATA, TT_COEFF, TP_ARCH> outVal) {
    // Do nothing
}
template <typename TT_DATA, typename TT_COEFF, eArchType TP_ARCH, unsigned int TP_NUM_OUTPUTS>
inline void writeWindow(T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface,
                        T_outValFiRDecHb<TT_DATA, TT_COEFF, TP_ARCH> outVal) {
    window_writeincr(outInterface.outWindow, outVal.val);
    if
        constexpr(TP_NUM_OUTPUTS == 2) { window_writeincr(outInterface.outWindow2, outVal.val); }
}

// Initial MUL operation for 2buff arch. Take inputIF as an argument to ease overloading.
template <typename TT_DATA, typename TT_COEFF, eArchType TP_ARCH, unsigned int TP_DUAL_IP>
inline T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> initMacDecHb(T_inputIF<CASC_IN_FALSE, TT_DATA, TP_DUAL_IP> inInterface,
                                                              T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> acc,
                                                              T_buff_FirDecHb<TT_DATA, TT_COEFF, TP_ARCH> xbuff,
                                                              unsigned int xstart,
                                                              T_buff_FirDecHb<TT_DATA, TT_COEFF, TP_ARCH> ybuff,
                                                              unsigned int ystart,
                                                              T_buff_256b<TT_COEFF> zbuff,
                                                              unsigned int zstart) {
    return firDecHbMulSym(xbuff, xstart, ybuff, ystart, zbuff, zstart);
};

// Initial MAC operation for 2buff arch. Take inputIF as an argument to ease overloading.
template <typename TT_DATA, typename TT_COEFF, eArchType TP_ARCH, unsigned int TP_DUAL_IP>
inline T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> initMacDecHb(T_inputIF<CASC_IN_TRUE, TT_DATA, TP_DUAL_IP> inInterface,
                                                              T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> acc,
                                                              T_buff_FirDecHb<TT_DATA, TT_COEFF, TP_ARCH> xbuff,
                                                              unsigned int xstart,
                                                              T_buff_FirDecHb<TT_DATA, TT_COEFF, TP_ARCH> ybuff,
                                                              unsigned int ystart,
                                                              T_buff_256b<TT_COEFF> zbuff,
                                                              unsigned int zstart) {
    return firDecHbMacSym(acc, xbuff, xstart, ybuff, ystart, zbuff, zstart);
};

// Initial MAC operation for 1buff arch. Take inputIF as an argument to ease overloading.
template <typename TT_DATA, typename TT_COEFF, eArchType TP_ARCH, unsigned int TP_DUAL_IP>
inline T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> initMacDecHbCt(
    T_inputIF<CASC_IN_FALSE, TT_DATA, TP_DUAL_IP> inInterface,
    T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> acc,
    T_buff_FirDecHb<TT_DATA, TT_COEFF, TP_ARCH> xbuff,
    unsigned int xstart,
    T_buff_FirDecHb<TT_DATA, TT_COEFF, TP_ARCH> ybuff,
    unsigned int ystart,
    unsigned int ct,
    T_buff_256b<TT_COEFF> zbuff,
    unsigned int zstart) {
    return firDecHbMulSymCt(xbuff, xstart, ybuff, ystart, ct, zbuff, zstart);
};

// Initial MAC operation for 1buff arch. Take inputIF as an argument to ease overloading.
template <typename TT_DATA, typename TT_COEFF, eArchType TP_ARCH, unsigned int TP_DUAL_IP>
inline T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> initMacDecHbCt(
    T_inputIF<CASC_IN_TRUE, TT_DATA, TP_DUAL_IP> inInterface,
    T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> acc,
    T_buff_FirDecHb<TT_DATA, TT_COEFF, TP_ARCH> xbuff,
    unsigned int xstart,
    T_buff_FirDecHb<TT_DATA, TT_COEFF, TP_ARCH> ybuff,
    unsigned int ystart,
    unsigned int ct,
    T_buff_256b<TT_COEFF> zbuff,
    unsigned int zstart) {
    return firDecHbMacSymCt(acc, xbuff, xstart, ybuff, ystart, ct, zbuff, zstart);
};

// Initial MAC operation for 1buff arch. Take inputIF as an argument to ease overloading.
template <typename TT_DATA, typename TT_COEFF, eArchType TP_ARCH, unsigned int TP_DUAL_IP>
inline T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> initMacDecHb(T_inputIF<CASC_IN_FALSE, TT_DATA, TP_DUAL_IP> inInterface,
                                                              T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> acc,
                                                              T_buff_FirDecHb<TT_DATA, TT_COEFF, TP_ARCH> xbuff,
                                                              unsigned int xstart,
                                                              unsigned int ystart,
                                                              T_buff_256b<TT_COEFF> zbuff,
                                                              unsigned int zstart) {
    return firDecHbMulSym1buff(xbuff, xstart, ystart, zbuff, zstart);
};

// Initial MAC operation for 1buff arch. Take inputIF as an argument to ease overloading.
template <typename TT_DATA, typename TT_COEFF, eArchType TP_ARCH, unsigned int TP_DUAL_IP>
inline T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> initMacDecHb(T_inputIF<CASC_IN_TRUE, TT_DATA, TP_DUAL_IP> inInterface,
                                                              T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> acc,
                                                              T_buff_FirDecHb<TT_DATA, TT_COEFF, TP_ARCH> xbuff,
                                                              unsigned int xstart,
                                                              unsigned int ystart,
                                                              T_buff_256b<TT_COEFF> zbuff,
                                                              unsigned int zstart) {
    return firDecHbMacSym1buff(acc, xbuff, xstart, ystart, zbuff, zstart);
};

// Initial MAC operation for 1buff arch. Take inputIF as an argument to ease overloading.
template <typename TT_DATA, typename TT_COEFF, eArchType TP_ARCH, unsigned int TP_DUAL_IP>
inline T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> initMacDecHbCt(
    T_inputIF<CASC_IN_FALSE, TT_DATA, TP_DUAL_IP> inInterface,
    T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> acc,
    T_buff_FirDecHb<TT_DATA, TT_COEFF, TP_ARCH> xbuff,
    unsigned int xstart,
    unsigned int ystart,
    unsigned int ct,
    T_buff_256b<TT_COEFF> zbuff,
    unsigned int zstart) {
    return firDecHbMulSymCt1buff(xbuff, xstart, ystart, ct, zbuff, zstart);
};

// Initial MAC operation for 1buff arch. Take inputIF as an argument to ease overloading.
template <typename TT_DATA, typename TT_COEFF, eArchType TP_ARCH, unsigned int TP_DUAL_IP>
inline T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> initMacDecHbCt(
    T_inputIF<CASC_IN_TRUE, TT_DATA, TP_DUAL_IP> inInterface,
    T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> acc,
    T_buff_FirDecHb<TT_DATA, TT_COEFF, TP_ARCH> xbuff,
    unsigned int xstart,
    unsigned int ystart,
    unsigned int ct,
    T_buff_256b<TT_COEFF> zbuff,
    unsigned int zstart) {
    return firDecHbMacSymCt1buff(acc, xbuff, xstart, ystart, ct, zbuff, zstart);
};

// Overloaded function to read from cascade input.
template <typename TT_DATA, typename TT_COEFF, eArchType TP_ARCH, unsigned int TP_DUAL_IP>
inline T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> readCascade(T_inputIF<false, TT_DATA, TP_DUAL_IP> inInterface,
                                                             T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> acc) {
    // Do nothing
    return acc;
};

// Overloaded function to read from cascade input.
template <typename TT_DATA, typename TT_COEFF, eArchType TP_ARCH, unsigned int TP_DUAL_IP>
inline T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> readCascade(T_inputIF<true, TT_DATA, TP_DUAL_IP> inInterface,
                                                             T_accFirDecHb<TT_DATA, TT_COEFF, TP_ARCH> acc) {
    // Call readCascade(window). All cases covered below.
    return readCascade(inInterface.inCascade, acc);
};

// Overloaded readCascade, taking cascade IF as an input
inline T_accFirDecHb<int16, int16, kArch1Buff> readCascade(input_stream_cacc48* inCascade,
                                                           T_accFirDecHb<int16, int16, kArch1Buff> acc) {
    T_accFirDecHb<int16, int16, kArch1Buff> ret;
    ret.val = upd_lo(acc.val, get_scd());
    ret.val = upd_hi(ret.val, get_scd());
    return ret;
};
inline T_accFirDecHb<cint16, int16, kArch1Buff> readCascade(input_stream_cacc48* inCascade,
                                                            T_accFirDecHb<cint16, int16, kArch1Buff> acc) {
    T_accFirDecHb<cint16, int16, kArch1Buff> ret;
    ret.val = getc_scd();
    ret.uval = getc_scd();
    return ret;
};
inline T_accFirDecHb<cint16, cint16, kArch1Buff> readCascade(input_stream_cacc48* inCascade,
                                                             T_accFirDecHb<cint16, cint16, kArch1Buff> acc) {
    T_accFirDecHb<cint16, cint16, kArch1Buff> ret;
    ret.val = getc_scd();
    ret.uval = getc_scd();
    return ret;
};
inline T_accFirDecHb<int32, int16, kArch1Buff> readCascade(input_stream_cacc48* inCascade,
                                                           T_accFirDecHb<int32, int16, kArch1Buff> acc) {
    T_accFirDecHb<int32, int16, kArch1Buff> ret;
    ret.val = upd_lo(acc.val, getl_scd());
    ret.val = upd_hi(ret.val, getl_scd());
    return ret;
};
inline T_accFirDecHb<int32, int32, kArch1Buff> readCascade(input_stream_cacc48* inCascade,
                                                           T_accFirDecHb<int32, int32, kArch1Buff> acc) {
    T_accFirDecHb<int32, int32, kArch1Buff> ret;
    ret.val = getl_scd();
    ret.uval = getl_scd();
    return ret;
};
inline T_accFirDecHb<cint32, int16, kArch1Buff> readCascade(input_stream_cacc48* inCascade,
                                                            T_accFirDecHb<cint32, int16, kArch1Buff> acc) {
    T_accFirDecHb<cint32, int16, kArch1Buff> ret;
    ret.val = upd_lo(acc.val, getlc_scd());
    ret.val = upd_hi(ret.val, getlc_scd());
    return ret;
};
inline T_accFirDecHb<cint32, int32, kArch1Buff> readCascade(input_stream_cacc48* inCascade,
                                                            T_accFirDecHb<cint32, int32, kArch1Buff> acc) {
    T_accFirDecHb<cint32, int32, kArch1Buff> ret;
    ret.val = upd_lo(acc.val, getlc_scd());
    ret.val = upd_hi(ret.val, getlc_scd());
    return ret;
};
inline T_accFirDecHb<cint32, cint16, kArch1Buff> readCascade(input_stream_cacc48* inCascade,
                                                             T_accFirDecHb<cint32, cint16, kArch1Buff> acc) {
    T_accFirDecHb<cint32, cint16, kArch1Buff> ret;
    ret.val = upd_lo(acc.val, getlc_scd());
    ret.val = upd_hi(ret.val, getlc_scd());
    return ret;
};
inline T_accFirDecHb<cint32, cint32, kArch1Buff> readCascade(input_stream_cacc48* inCascade,
                                                             T_accFirDecHb<cint32, cint32, kArch1Buff> acc) {
    T_accFirDecHb<cint32, cint32, kArch1Buff> ret;
    ret.val = getlc_scd();
    ret.uval = getlc_scd();
    return ret;
};
inline T_accFirDecHb<float, float, kArch1Buff> readCascade(input_stream_cacc48* inCascade,
                                                           T_accFirDecHb<float, float, kArch1Buff> acc) {
    T_accFirDecHb<float, float, kArch1Buff> ret;
    ret.val = as_v8float(lsrs(get_scd(), 0));
    return ret;
};
inline T_accFirDecHb<cfloat, float, kArch1Buff> readCascade(input_stream_cacc48* inCascade,
                                                            T_accFirDecHb<cfloat, float, kArch1Buff> acc) {
    T_accFirDecHb<cfloat, float, kArch1Buff> ret;
    ret.val = as_v4cfloat(lsrs(getc_scd(), 0));
    return ret;
};
inline T_accFirDecHb<cfloat, cfloat, kArch1Buff> readCascade(input_stream_cacc48* inCascade,
                                                             T_accFirDecHb<cfloat, cfloat, kArch1Buff> acc) {
    T_accFirDecHb<cfloat, cfloat, kArch1Buff> ret;
    ret.val = as_v4cfloat(lsrs(getc_scd(), 0));
    return ret;
};

// Overloaded readCascade, taking cascade IF as an input
inline T_accFirDecHb<int16, int16, kArch2Buff> readCascade(input_stream_cacc48* inCascade,
                                                           T_accFirDecHb<int16, int16, kArch2Buff> acc) {
    T_accFirDecHb<int16, int16, kArch2Buff> ret;
    ret.val = get_scd();
    return ret;
};
inline T_accFirDecHb<cint16, int16, kArch2Buff> readCascade(input_stream_cacc48* inCascade,
                                                            T_accFirDecHb<cint16, int16, kArch2Buff> acc) {
    T_accFirDecHb<cint16, int16, kArch2Buff> ret;
    ret.val = getc_scd();
    return ret;
};
inline T_accFirDecHb<cint16, cint16, kArch2Buff> readCascade(input_stream_cacc48* inCascade,
                                                             T_accFirDecHb<cint16, cint16, kArch2Buff> acc) {
    T_accFirDecHb<cint16, cint16, kArch2Buff> ret;
    ret.val = getc_scd();
    return ret;
};
inline T_accFirDecHb<int32, int16, kArch2Buff> readCascade(input_stream_cacc48* inCascade,
                                                           T_accFirDecHb<int32, int16, kArch2Buff> acc) {
    T_accFirDecHb<int32, int16, kArch2Buff> ret;
    ret.val = upd_lo(acc.val, getl_scd());
    ret.val = upd_hi(ret.val, getl_scd());
    return ret;
};
inline T_accFirDecHb<int32, int32, kArch2Buff> readCascade(input_stream_cacc48* inCascade,
                                                           T_accFirDecHb<int32, int32, kArch2Buff> acc) {
    T_accFirDecHb<int32, int32, kArch2Buff> ret;
    ret.val = getl_scd();
    return ret;
};
inline T_accFirDecHb<cint32, int16, kArch2Buff> readCascade(input_stream_cacc48* inCascade,
                                                            T_accFirDecHb<cint32, int16, kArch2Buff> acc) {
    T_accFirDecHb<cint32, int16, kArch2Buff> ret;
    ret.val = upd_lo(acc.val, getlc_scd());
    ret.val = upd_hi(ret.val, getlc_scd());
    return ret;
};
inline T_accFirDecHb<cint32, int32, kArch2Buff> readCascade(input_stream_cacc48* inCascade,
                                                            T_accFirDecHb<cint32, int32, kArch2Buff> acc) {
    T_accFirDecHb<cint32, int32, kArch2Buff> ret;
    ret.val = upd_lo(acc.val, getlc_scd());
    ret.val = upd_hi(ret.val, getlc_scd());
    return ret;
};
inline T_accFirDecHb<cint32, cint16, kArch2Buff> readCascade(input_stream_cacc48* inCascade,
                                                             T_accFirDecHb<cint32, cint16, kArch2Buff> acc) {
    T_accFirDecHb<cint32, cint16, kArch2Buff> ret;
    ret.val = upd_lo(acc.val, getlc_scd());
    ret.val = upd_hi(ret.val, getlc_scd());
    return ret;
};
inline T_accFirDecHb<cint32, cint32, kArch2Buff> readCascade(input_stream_cacc48* inCascade,
                                                             T_accFirDecHb<cint32, cint32, kArch2Buff> acc) {
    T_accFirDecHb<cint32, cint32, kArch2Buff> ret;
    ret.val = getlc_scd();
    ret.uval = getlc_scd();
    return ret;
};
inline T_accFirDecHb<float, float, kArch2Buff> readCascade(input_stream_cacc48* inCascade,
                                                           T_accFirDecHb<float, float, kArch2Buff> acc) {
    T_accFirDecHb<float, float, kArch2Buff> ret;
    ret.val = as_v8float(lsrs(get_scd(), 0));
    return ret;
};
inline T_accFirDecHb<cfloat, float, kArch2Buff> readCascade(input_stream_cacc48* inCascade,
                                                            T_accFirDecHb<cfloat, float, kArch2Buff> acc) {
    T_accFirDecHb<cfloat, float, kArch2Buff> ret;
    ret.val = as_v4cfloat(lsrs(getc_scd(), 0));
    return ret;
};
inline T_accFirDecHb<cfloat, cfloat, kArch2Buff> readCascade(input_stream_cacc48* inCascade,
                                                             T_accFirDecHb<cfloat, cfloat, kArch2Buff> acc) {
    T_accFirDecHb<cfloat, cfloat, kArch2Buff> ret;
    ret.val = as_v4cfloat(lsrs(getc_scd(), 0));
    return ret;
};
}
}
}
}
}
#endif // _DSPLIB_FIR_DECIMATE_HB_UTILS_HPP_
