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
#ifndef _DSPLIB_FIR_SR_SYM_UTILS_HPP_
#define _DSPLIB_FIR_SR_SYM_UTILS_HPP_

/*
Single Rate Symmetrical FIR Utilities
This file contains sets of overloaded, templatized and specialized templatized functions for use
by the main kernel class and run-time function. These functions are separate from the traits file
because they are purely for kernel use, not graph level compilation.
*/

#include <stdio.h>
#include <adf.h>

namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace sr_sym {
// Constants used to specialize optimizations associated with the center-tap (CT) operation
#define K_CT_OP_WITH_1_SAMPLE 1
#define K_CT_OP_WITH_2_SAMPLES 2
#define K_CT_OP_WITH_3_SAMPLES 3
#define K_CT_OP_WITH_4_SAMPLES 4
#define K_CT_OP_WITH_5_SAMPLES 5
#define K_CT_OP_WITH_6_SAMPLES 6
#define K_CT_OP_WITH_7_SAMPLES 7

template <typename T_D, typename T_C>
struct T_accSym : T_acc<T_D, T_C> {
    using T_acc<T_D, T_C>::operator=;
};
template <>
struct T_accSym<int16, int16> : T_acc<int16, int16> {
    using T_acc<int16, int16>::operator=;
};
template <>
struct T_accSym<cint16, int16> : T_acc384<cint16, int16> {
    using T_acc384<cint16, int16>::operator=;
}; // narrow - 384-bit acc
template <>
struct T_accSym<cint16, cint16> : T_acc<cint16, cint16> {
    using T_acc<cint16, cint16>::operator=;
};
template <>
struct T_accSym<int32, int16> : T_acc<int32, int16> {
    using T_acc<int32, int16>::operator=;
};
template <>
struct T_accSym<int32, int32> : T_acc<int32, int32> {
    using T_acc<int32, int32>::operator=;
};
template <>
struct T_accSym<cint32, int16> : T_acc<cint32, int16> {
    using T_acc<cint32, int16>::operator=;
};
template <>
struct T_accSym<cint32, int32> : T_acc<cint32, int32> {
    using T_acc<cint32, int32>::operator=;
};
template <>
struct T_accSym<cint32, cint16> : T_acc<cint32, cint16> {
    using T_acc<cint32, cint16>::operator=;
};
template <>
struct T_accSym<cint32, cint32> : T_acc<cint32, cint32> {
    using T_acc<cint32, cint32>::operator=;
};
template <>
struct T_accSym<float, float> : T_acc<float, float> {
    using T_acc<float, float>::operator=;
};
template <>
struct T_accSym<cfloat, float> : T_acc<cfloat, float> {
    using T_acc<cfloat, float>::operator=;
};
template <>
struct T_accSym<cfloat, cfloat> : T_acc<cfloat, cfloat> {
    using T_acc<cfloat, cfloat>::operator=;
};

template <typename T_D, typename T_C>
struct T_outValSym : T_outVal<T_D, T_C> {
    using T_outVal<T_D, T_C>::operator=;
};
template <>
struct T_outValSym<int16, int16> : T_outVal<int16, int16> {
    using T_outVal<int16, int16>::operator=;
};
template <>
struct T_outValSym<cint16, int16> : T_outVal384<cint16, int16> {
    using T_outVal384<cint16, int16>::operator=;
}; // narrow - 384-bit acc
template <>
struct T_outValSym<cint16, cint16> : T_outVal<cint16, cint16> {
    using T_outVal<cint16, cint16>::operator=;
};
template <>
struct T_outValSym<int32, int16> : T_outVal<int32, int16> {
    using T_outVal<int32, int16>::operator=;
};
template <>
struct T_outValSym<int32, int32> : T_outVal<int32, int32> {
    using T_outVal<int32, int32>::operator=;
};
template <>
struct T_outValSym<cint32, int16> : T_outVal<cint32, int16> {
    using T_outVal<cint32, int16>::operator=;
};
template <>
struct T_outValSym<cint32, int32> : T_outVal<cint32, int32> {
    using T_outVal<cint32, int32>::operator=;
};
template <>
struct T_outValSym<cint32, cint16> : T_outVal<cint32, cint16> {
    using T_outVal<cint32, cint16>::operator=;
};
template <>
struct T_outValSym<cint32, cint32> : T_outVal<cint32, cint32> {
    using T_outVal<cint32, cint32>::operator=;
};
template <>
struct T_outValSym<float, float> : T_outVal<float, float> {
    using T_outVal<float, float>::operator=;
};
template <>
struct T_outValSym<cfloat, float> : T_outVal<cfloat, float> {
    using T_outVal<cfloat, float>::operator=;
};
template <>
struct T_outValSym<cfloat, cfloat> : T_outVal<cfloat, cfloat> {
    using T_outVal<cfloat, cfloat>::operator=;
};

// ---------------------------------------------------- 2 BUFF ---------------------------------------------------- //

// overloaded mul/macSrSym calls.
// second set of intrinsics, to cover 2 smaller xbuff and ybuff buffers
//-----------------------------------------------------------------------------------------------------
// DATA = int16, COEFF = int16
inline T_accSym<int16, int16> mulSrSym(
    v32int16 xbuff, unsigned int xstart, v32int16 ybuff, unsigned int ystart, v16int16 zbuff, unsigned int zstart) {
    T_accSym<int16, int16> retVal;
    const unsigned int xoffsets = 0x03020100;
    const unsigned int xoffsets_hi = 0x07060504;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zoffsets_hi = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int xstartmod = (xstart & 1) == 0 ? xstart : xstart - 1;
    const unsigned int xsquare = (xstart & 1) == 0 ? 0x2110 : 0x3221;
    const unsigned int ystartmod = (ystart & 1) == 0 ? ystart - 2 : ystart - 1;
    const unsigned int ysquare = (ystart & 1) == 0 ? 0x2312 : 0x1201;

    retVal.val = mul16_sym(xbuff, xstartmod, xoffsets, xoffsets_hi, xsquare, ybuff, ystartmod, ysquare, zbuff, zstart,
                           zoffsets, zoffsets_hi, zstep);
    return retVal;
}

inline T_accSym<int16, int16> macSrSym(T_accSym<int16, int16> acc,
                                       v32int16 xbuff,
                                       unsigned int xstart,
                                       v32int16 ybuff,
                                       unsigned int ystart,
                                       v16int16 zbuff,
                                       unsigned int zstart) {
    T_accSym<int16, int16> retVal;
    const unsigned int xoffsets = 0x03020100;
    const unsigned int xoffsets_hi = 0x07060504;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zoffsets_hi = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int xstartmod = (xstart & 1) == 0 ? xstart : xstart - 1;
    const unsigned int xsquare = (xstart & 1) == 0 ? 0x2110 : 0x3221;
    const unsigned int ystartmod = (ystart & 1) == 0 ? ystart - 2 : ystart - 1;
    const unsigned int ysquare = (ystart & 1) == 0 ? 0x2312 : 0x1201;

    retVal.val = mac16_sym(acc.val, xbuff, xstartmod, xoffsets, xoffsets_hi, xsquare, ybuff, ystartmod, ysquare, zbuff,
                           zstart, zoffsets, zoffsets_hi, zstep);
    return retVal;
}

// DATA = cint16, COEFF = int16
inline T_accSym<cint16, int16> mulSrSym(
    v16cint16 xbuff, unsigned int xstart, v16cint16 ybuff, unsigned int ystart, v16int16 zbuff, unsigned int zstart) {
    T_accSym<cint16, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;

    retVal.val = mul4_sym(xbuff, xstart, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSym<cint16, int16> macSrSym(T_accSym<cint16, int16> acc,
                                        v16cint16 xbuff,
                                        unsigned int xstart,
                                        v16cint16 ybuff,
                                        unsigned int ystart,
                                        v16int16 zbuff,
                                        unsigned int zstart) {
    T_accSym<cint16, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;

    retVal.val = mac4_sym(acc.val, xbuff, xstart, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = cint16, COEFF = cint16
inline T_accSym<cint16, cint16> mulSrSym(
    v16cint16 xbuff, unsigned int xstart, v16cint16 ybuff, unsigned int ystart, v8cint16 zbuff, unsigned int zstart) {
    T_accSym<cint16, cint16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;

    retVal.val = mul8_sym(xbuff, xstart, xoffsets, ybuff, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSym<cint16, cint16> macSrSym(T_accSym<cint16, cint16> acc,
                                         v16cint16 xbuff,
                                         unsigned int xstart,
                                         v16cint16 ybuff,
                                         unsigned int ystart,
                                         v8cint16 zbuff,
                                         unsigned int zstart) {
    T_accSym<cint16, cint16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;

    retVal.val = mac8_sym(acc.val, xbuff, xstart, xoffsets, ybuff, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = int32, COEFF = int16
inline T_accSym<int32, int16> mulSrSym(
    v16int32 xbuff, unsigned int xstart, v16int32 ybuff, unsigned int ystart, v16int16 zbuff, unsigned int zstart) {
    T_accSym<int32, int16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int xstep = 1;
    const unsigned int zstep = 1;

    retVal.val = lmul8_sym(xbuff, xstart, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSym<int32, int16> macSrSym(T_accSym<int32, int16> acc,
                                       v16int32 xbuff,
                                       unsigned int xstart,
                                       v16int32 ybuff,
                                       unsigned int ystart,
                                       v16int16 zbuff,
                                       unsigned int zstart) {
    T_accSym<int32, int16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int xstep = 1;
    const unsigned int zstep = 1;

    retVal.val = lmac8_sym(acc.val, xbuff, xstart, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = int32,  COEFF = int32>
inline T_accSym<int32, int32> mulSrSym(
    v16int32 xbuff, unsigned int xstart, v16int32 ybuff, unsigned int ystart, v8int32 zbuff, unsigned int zstart) {
    T_accSym<int32, int32> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;

    retVal.val = lmul8_sym(xbuff, xstart, xoffsets, ybuff, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSym<int32, int32> macSrSym(T_accSym<int32, int32> acc,
                                       v16int32 xbuff,
                                       unsigned int xstart,
                                       v16int32 ybuff,
                                       unsigned int ystart,
                                       v8int32 zbuff,
                                       unsigned int zstart) {
    T_accSym<int32, int32> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;

    retVal.val = lmac8_sym(acc.val, xbuff, xstart, xoffsets, ybuff, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cint32, COEFF =  int16>
inline T_accSym<cint32, int16> mulSrSym(
    v8cint32 xbuff, unsigned int xstart, v8cint32 ybuff, unsigned int ystart, v16int16 zbuff, unsigned int zstart) {
    T_accSym<cint32, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int xstep = 1;
    const unsigned int zstep = 1;

    retVal.val = lmul4_sym(xbuff, xstart, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSym<cint32, int16> macSrSym(T_accSym<cint32, int16> acc,
                                        v8cint32 xbuff,
                                        unsigned int xstart,
                                        v8cint32 ybuff,
                                        unsigned int ystart,
                                        v16int16 zbuff,
                                        unsigned int zstart) {
    T_accSym<cint32, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int xstep = 1;
    const unsigned int zstep = 1;

    retVal.val = lmac4_sym(acc.val, xbuff, xstart, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = cint32, COEFF =  int32>
inline T_accSym<cint32, int32> mulSrSym(
    v8cint32 xbuff, unsigned int xstart, v8cint32 ybuff, unsigned int ystart, v8int32 zbuff, unsigned int zstart) {
    T_accSym<cint32, int32> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = lmul4_sym(xbuff, xstart, xoffsets, ybuff, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSym<cint32, int32> macSrSym(T_accSym<cint32, int32> acc,
                                        v8cint32 xbuff,
                                        unsigned int xstart,
                                        v8cint32 ybuff,
                                        unsigned int ystart,
                                        v8int32 zbuff,
                                        unsigned int zstart) {
    T_accSym<cint32, int32> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = lmac4_sym(acc.val, xbuff, xstart, xoffsets, ybuff, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cint32, COEFF =  cint16>
inline T_accSym<cint32, cint16> mulSrSym(
    v8cint32 xbuff, unsigned int xstart, v8cint32 ybuff, unsigned int ystart, v8cint16 zbuff, unsigned int zstart) {
    T_accSym<cint32, cint16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = lmul4_sym(xbuff, xstart, xoffsets, ybuff, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSym<cint32, cint16> macSrSym(T_accSym<cint32, cint16> acc,
                                         v8cint32 xbuff,
                                         unsigned int xstart,
                                         v8cint32 ybuff,
                                         unsigned int ystart,
                                         v8cint16 zbuff,
                                         unsigned int zstart) {
    T_accSym<cint32, cint16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = lmac4_sym(acc.val, xbuff, xstart, xoffsets, ybuff, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cint32, COEFF =  cint32>
inline T_accSym<cint32, cint32> mulSrSym(v8cint32 xbuff,
                                         const unsigned int xstart,
                                         v8cint32 ybuff,
                                         unsigned int ystart,
                                         v4cint32 zbuff,
                                         unsigned int zstart) {
    T_accSym<cint32, cint32> retVal;
    const unsigned int xoffsets = 0x10;
    const unsigned int zoffsets = 0x0000;

    retVal.val = lmul2_sym(xbuff, xstart, xoffsets, ybuff, ystart, zbuff, zstart, zoffsets);
    retVal.uval = lmul2_sym(xbuff, xstart + 2, xoffsets, ybuff, ystart + 2, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSym<cint32, cint32> macSrSym(T_accSym<cint32, cint32> acc,
                                         v8cint32 xbuff,
                                         unsigned int xstart,
                                         v8cint32 ybuff,
                                         unsigned int ystart,
                                         v4cint32 zbuff,
                                         unsigned int zstart) {
    T_accSym<cint32, cint32> retVal;
    const unsigned int xoffsets = 0x10;
    const unsigned int zoffsets = 0x0000;

    retVal.val = lmac2_sym(acc.val, xbuff, xstart, xoffsets, ybuff, ystart, zbuff, zstart, zoffsets);
    retVal.uval = lmac2_sym(acc.uval, xbuff, xstart + 2, xoffsets, ybuff, ystart + 2, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = float,  COEFF = float>
inline T_accSym<float, float> mulSrSym(
    v16float xbuff, unsigned int xstart, v16float ybuff, unsigned int ystart, v8float zbuff, unsigned int zstart) {
    T_accSym<float, float> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;

    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, ybuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSym<float, float> macSrSym(T_accSym<float, float> acc,
                                       v16float xbuff,
                                       unsigned int xstart,
                                       v16float ybuff,
                                       unsigned int ystart,
                                       v8float zbuff,
                                       unsigned int zstart) {
    T_accSym<float, float> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;

    retVal.val = fpmac(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, ybuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cfloat, COEFF =  float>
inline T_accSym<cfloat, float> mulSrSym(
    v8cfloat xbuff, unsigned int xstart, v8cfloat ybuff, unsigned int ystart, v8float zbuff, unsigned int zstart) {
    T_accSym<cfloat, float> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, ybuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSym<cfloat, float> macSrSym(T_accSym<cfloat, float> acc,
                                        v8cfloat xbuff,
                                        unsigned int xstart,
                                        v8cfloat ybuff,
                                        unsigned int ystart,
                                        v8float zbuff,
                                        unsigned int zstart) {
    T_accSym<cfloat, float> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = fpmac(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, ybuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cfloat, COEFF =  cfloat>
inline T_accSym<cfloat, cfloat> mulSrSym(
    v8cfloat xbuff, unsigned int xstart, v8cfloat ybuff, unsigned int ystart, v4cfloat zbuff, unsigned int zstart) {
    T_accSym<cfloat, cfloat> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, ybuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSym<cfloat, cfloat> macSrSym(T_accSym<cfloat, cfloat> acc,
                                         v8cfloat xbuff,
                                         unsigned int xstart,
                                         v8cfloat ybuff,
                                         unsigned int ystart,
                                         v4cfloat zbuff,
                                         unsigned int zstart) {
    T_accSym<cfloat, cfloat> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = fpmac(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, ybuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// ---------------------------------------------------- 1 BUFF ---------------------------------------------------- //

// overloaded mul/macSrSym calls.
// second set of intrinsics, to cover 2 smaller xbuff and ybuff buffers
//-----------------------------------------------------------------------------------------------------
// DATA = int16, COEFF = int16
inline T_accSym<int16, int16> mulSrSym(
    v64int16 xbuff, unsigned int xstart, unsigned int ystart, v16int16 zbuff, unsigned int zstart) {
    T_accSym<int16, int16> retVal;
    const unsigned int xoffsets = 0x03020100;
    const unsigned int xoffsets_hi = 0x07060504;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zoffsets_hi = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int xstartmod = (xstart & 1) == 0 ? xstart : xstart - 1;
    const unsigned int xsquare = (xstart & 1) == 0 ? 0x2110 : 0x3221;
    const unsigned int ystartmod = (ystart & 1) == 0 ? ystart - 2 : ystart - 1;
    const unsigned int ysquare = (ystart & 1) == 0 ? 0x2312 : 0x1201;

    retVal.val = mul16_sym(xbuff, xstartmod, xoffsets, xoffsets_hi, xsquare, ystartmod, ysquare, zbuff, zstart,
                           zoffsets, zoffsets_hi, zstep);
    return retVal;
}

inline T_accSym<int16, int16> macSrSym(T_accSym<int16, int16> acc,
                                       v64int16 xbuff,
                                       unsigned int xstart,
                                       unsigned int ystart,
                                       v16int16 zbuff,
                                       unsigned int zstart) {
    T_accSym<int16, int16> retVal;
    const unsigned int xoffsets = 0x03020100;
    const unsigned int xoffsets_hi = 0x07060504;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zoffsets_hi = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int xstartmod = (xstart & 1) == 0 ? xstart : xstart - 1;
    const unsigned int xsquare = (xstart & 1) == 0 ? 0x2110 : 0x3221;
    const unsigned int ystartmod = (ystart & 1) == 0 ? ystart - 2 : ystart - 1;
    const unsigned int ysquare = (ystart & 1) == 0 ? 0x2312 : 0x1201;

    retVal.val = mac16_sym(acc.val, xbuff, xstartmod, xoffsets, xoffsets_hi, xsquare, ystartmod, ysquare, zbuff, zstart,
                           zoffsets, zoffsets_hi, zstep);
    return retVal;
}

// DATA = cint16, COEFF = int16
inline T_accSym<cint16, int16> mulSrSym(
    v32cint16 xbuff, unsigned int xstart, unsigned int ystart, v16int16 zbuff, unsigned int zstart) {
    T_accSym<cint16, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;

    retVal.val = mul4_sym(xbuff, xstart, xoffsets, xstep, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSym<cint16, int16> macSrSym(T_accSym<cint16, int16> acc,
                                        v32cint16 xbuff,
                                        unsigned int xstart,
                                        unsigned int ystart,
                                        v16int16 zbuff,
                                        unsigned int zstart) {
    T_accSym<cint16, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;

    retVal.val = mac4_sym(acc.val, xbuff, xstart, xoffsets, xstep, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = cint16, COEFF = cint16
inline T_accSym<cint16, cint16> mulSrSym(
    v32cint16 xbuff, unsigned int xstart, unsigned int ystart, v8cint16 zbuff, unsigned int zstart) {
    T_accSym<cint16, cint16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;

    retVal.val = mul8_sym(xbuff, xstart, xoffsets, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSym<cint16, cint16> macSrSym(T_accSym<cint16, cint16> acc,
                                         v32cint16 xbuff,
                                         unsigned int xstart,
                                         unsigned int ystart,
                                         v8cint16 zbuff,
                                         unsigned int zstart) {
    T_accSym<cint16, cint16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;

    retVal.val = mac8_sym(acc.val, xbuff, xstart, xoffsets, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = int32, COEFF = int16
inline T_accSym<int32, int16> mulSrSym(
    v32int32 xbuff, unsigned int xstart, unsigned int ystart, v16int16 zbuff, unsigned int zstart) {
    T_accSym<int32, int16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int xstep = 1;
    const unsigned int zstep = 1;

    retVal.val = lmul8_sym(xbuff, xstart, xoffsets, xstep, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSym<int32, int16> macSrSym(T_accSym<int32, int16> acc,
                                       v32int32 xbuff,
                                       unsigned int xstart,
                                       unsigned int ystart,
                                       v16int16 zbuff,
                                       unsigned int zstart) {
    T_accSym<int32, int16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int xstep = 1;
    const unsigned int zstep = 1;

    retVal.val = lmac8_sym(acc.val, xbuff, xstart, xoffsets, xstep, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = int32,  COEFF = int32>
inline T_accSym<int32, int32> mulSrSym(
    v32int32 xbuff, unsigned int xstart, unsigned int ystart, v8int32 zbuff, unsigned int zstart) {
    T_accSym<int32, int32> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;

    retVal.val = lmul8_sym(xbuff, xstart, xoffsets, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSym<int32, int32> macSrSym(T_accSym<int32, int32> acc,
                                       v32int32 xbuff,
                                       unsigned int xstart,
                                       unsigned int ystart,
                                       v8int32 zbuff,
                                       unsigned int zstart) {
    T_accSym<int32, int32> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;

    retVal.val = lmac8_sym(acc.val, xbuff, xstart, xoffsets, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cint32, COEFF =  int16>
inline T_accSym<cint32, int16> mulSrSym(
    v16cint32 xbuff, unsigned int xstart, unsigned int ystart, v16int16 zbuff, unsigned int zstart) {
    T_accSym<cint32, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int xstep = 1;
    const unsigned int zstep = 1;

    retVal.val = lmul4_sym(xbuff, xstart, xoffsets, xstep, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSym<cint32, int16> macSrSym(T_accSym<cint32, int16> acc,
                                        v16cint32 xbuff,
                                        unsigned int xstart,
                                        unsigned int ystart,
                                        v16int16 zbuff,
                                        unsigned int zstart) {
    T_accSym<cint32, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int xstep = 1;
    const unsigned int zstep = 1;

    retVal.val = lmac4_sym(acc.val, xbuff, xstart, xoffsets, xstep, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = cint32, COEFF =  int32>
inline T_accSym<cint32, int32> mulSrSym(
    v16cint32 xbuff, unsigned int xstart, unsigned int ystart, v8int32 zbuff, unsigned int zstart) {
    T_accSym<cint32, int32> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = lmul4_sym(xbuff, xstart, xoffsets, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSym<cint32, int32> macSrSym(T_accSym<cint32, int32> acc,
                                        v16cint32 xbuff,
                                        unsigned int xstart,
                                        unsigned int ystart,
                                        v8int32 zbuff,
                                        unsigned int zstart) {
    T_accSym<cint32, int32> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = lmac4_sym(acc.val, xbuff, xstart, xoffsets, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cint32, COEFF =  cint16>
inline T_accSym<cint32, cint16> mulSrSym(
    v16cint32 xbuff, unsigned int xstart, unsigned int ystart, v8cint16 zbuff, unsigned int zstart) {
    T_accSym<cint32, cint16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = lmul4_sym(xbuff, xstart, xoffsets, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSym<cint32, cint16> macSrSym(T_accSym<cint32, cint16> acc,
                                         v16cint32 xbuff,
                                         unsigned int xstart,
                                         unsigned int ystart,
                                         v8cint16 zbuff,
                                         unsigned int zstart) {
    T_accSym<cint32, cint16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = lmac4_sym(acc.val, xbuff, xstart, xoffsets, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cint32, COEFF =  cint32>
inline T_accSym<cint32, cint32> mulSrSym(
    v16cint32 xbuff, const unsigned int xstart, unsigned int ystart, v4cint32 zbuff, unsigned int zstart) {
    T_accSym<cint32, cint32> retVal;
    const unsigned int xoffsets = 0x10;
    const unsigned int zoffsets = 0x0000;

    retVal.val = lmul2_sym(xbuff, xstart, xoffsets, ystart, zbuff, zstart, zoffsets);
    retVal.uval = lmul2_sym(xbuff, xstart + 2, xoffsets, ystart + 2, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSym<cint32, cint32> macSrSym(T_accSym<cint32, cint32> acc,
                                         v16cint32 xbuff,
                                         unsigned int xstart,
                                         unsigned int ystart,
                                         v4cint32 zbuff,
                                         unsigned int zstart) {
    T_accSym<cint32, cint32> retVal;
    const unsigned int xoffsets = 0x10;
    const unsigned int zoffsets = 0x0000;

    retVal.val = lmac2_sym(acc.val, xbuff, xstart, xoffsets, ystart, zbuff, zstart, zoffsets);
    retVal.uval = lmac2_sym(acc.uval, xbuff, xstart + 2, xoffsets, ystart + 2, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = float,  COEFF = float>
inline T_accSym<float, float> mulSrSym(
    v32float xbuff, unsigned int xstart, unsigned int ystart, v8float zbuff, unsigned int zstart) {
    T_accSym<float, float> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;

    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, xbuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSym<float, float> macSrSym(T_accSym<float, float> acc,
                                       v32float xbuff,
                                       unsigned int xstart,
                                       unsigned int ystart,
                                       v8float zbuff,
                                       unsigned int zstart) {
    T_accSym<float, float> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;

    retVal.val = fpmac(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, xbuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cfloat, COEFF =  float>
inline T_accSym<cfloat, float> mulSrSym(
    v16cfloat xbuff, unsigned int xstart, unsigned int ystart, v8float zbuff, unsigned int zstart) {
    T_accSym<cfloat, float> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, xbuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSym<cfloat, float> macSrSym(T_accSym<cfloat, float> acc,
                                        v16cfloat xbuff,
                                        unsigned int xstart,
                                        unsigned int ystart,
                                        v8float zbuff,
                                        unsigned int zstart) {
    T_accSym<cfloat, float> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = fpmac(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, xbuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cfloat, COEFF =  cfloat>
inline T_accSym<cfloat, cfloat> mulSrSym(
    v16cfloat xbuff, unsigned int xstart, unsigned int ystart, v4cfloat zbuff, unsigned int zstart) {
    T_accSym<cfloat, cfloat> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, xbuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSym<cfloat, cfloat> macSrSym(T_accSym<cfloat, cfloat> acc,
                                         v16cfloat xbuff,
                                         unsigned int xstart,
                                         unsigned int ystart,
                                         v4cfloat zbuff,
                                         unsigned int zstart) {
    T_accSym<cfloat, cfloat> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = fpmac(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, xbuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// ---------------------------------------------------- 2 BUFF ---------------------------------------------------- //

// overloaded mul/macSrSymCT calls.
// second set of intrinsics, to cover 2 smaller xbuff and ybuff buffers
//-----------------------------------------------------------------------------------------------------
// DATA = int16, COEFF = int16
template <unsigned int T_Variant = 0>
inline T_accSym<int16, int16> macSrSymCT(T_accSym<int16, int16> acc,
                                         v32int16 xbuff,
                                         unsigned int xstart,
                                         v32int16 ybuff,
                                         unsigned int ystart,
                                         v16int16 zbuff,
                                         unsigned int zstart,
                                         unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<cint16, int16> macSrSymCT(T_accSym<cint16, int16> acc,
                                          v16cint16 xbuff,
                                          unsigned int xstart,
                                          v16cint16 ybuff,
                                          unsigned int ystart,
                                          v16int16 zbuff,
                                          unsigned int zstart,
                                          unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<cint16, cint16> macSrSymCT(T_accSym<cint16, cint16> acc,
                                           v16cint16 xbuff,
                                           unsigned int xstart,
                                           v16cint16 ybuff,
                                           unsigned int ystart,
                                           v8cint16 zbuff,
                                           unsigned int zstart,
                                           unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<int32, int16> macSrSymCT(T_accSym<int32, int16> acc,
                                         v16int32 xbuff,
                                         unsigned int xstart,
                                         v16int32 ybuff,
                                         unsigned int ystart,
                                         v16int16 zbuff,
                                         unsigned int zstart,
                                         unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<int32, int32> macSrSymCT(T_accSym<int32, int32> acc,
                                         v16int32 xbuff,
                                         unsigned int xstart,
                                         v16int32 ybuff,
                                         unsigned int ystart,
                                         v8int32 zbuff,
                                         unsigned int zstart,
                                         unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<cint32, int16> macSrSymCT(T_accSym<cint32, int16> acc,
                                          v8cint32 xbuff,
                                          unsigned int xstart,
                                          v8cint32 ybuff,
                                          unsigned int ystart,
                                          v16int16 zbuff,
                                          unsigned int zstart,
                                          unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<cint32, int32> macSrSymCT(T_accSym<cint32, int32> acc,
                                          v8cint32 xbuff,
                                          unsigned int xstart,
                                          v8cint32 ybuff,
                                          unsigned int ystart,
                                          v8int32 zbuff,
                                          unsigned int zstart,
                                          unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<cint32, cint16> macSrSymCT(T_accSym<cint32, cint16> acc,
                                           v8cint32 xbuff,
                                           unsigned int xstart,
                                           v8cint32 ybuff,
                                           unsigned int ystart,
                                           v8cint16 zbuff,
                                           unsigned int zstart,
                                           unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<cint32, cint32> macSrSymCT(T_accSym<cint32, cint32> acc,
                                           v8cint32 xbuff,
                                           unsigned int xstart,
                                           v8cint32 ybuff,
                                           unsigned int ystart,
                                           v4cint32 zbuff,
                                           unsigned int zstart,
                                           unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<float, float> macSrSymCT(T_accSym<float, float> acc,
                                         v16float xbuff,
                                         unsigned int xstart,
                                         v16float ybuff,
                                         unsigned int ystart,
                                         v8float zbuff,
                                         unsigned int zstart,
                                         unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<cfloat, float> macSrSymCT(T_accSym<cfloat, float> acc,
                                          v8cfloat xbuff,
                                          unsigned int xstart,
                                          v8cfloat ybuff,
                                          unsigned int ystart,
                                          v8float zbuff,
                                          unsigned int zstart,
                                          unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<cfloat, cfloat> macSrSymCT(T_accSym<cfloat, cfloat> acc,
                                           v8cfloat xbuff,
                                           unsigned int xstart,
                                           v8cfloat ybuff,
                                           unsigned int ystart,
                                           v4cfloat zbuff,
                                           unsigned int zstart,
                                           unsigned int xbuffSwap = 0) {
    return acc;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = int16, COEFF = int16
// ---------------------------------------------------------------------------------------------------------------------
// Variant K_CT_OP_WITH_1_SAMPLE
template <>
inline T_accSym<int16, int16> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<int16, int16> acc,
                                                                v32int16 xbuff,
                                                                unsigned int xstart,
                                                                v32int16 ybuff,
                                                                unsigned int ystart,
                                                                v16int16 zbuff,
                                                                unsigned int zstart,
                                                                unsigned int xbuffSwap) {
    T_accSym<int16, int16> retVal;
    const unsigned int xoffsets = 0x03020100;
    const unsigned int xoffsets_hi = 0x07060504;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zoffsets_hi = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int xstartmod = (xstart & 1) == 0 ? xstart : xstart - 1;
    const unsigned int xsquare = (xstart & 1) == 0 ? 0x2110 : 0x3221;

    retVal.val =
        mac16(acc.val, xbuff, xstart, xoffsets, xoffsets_hi, xsquare, zbuff, zstart, zoffsets, zoffsets_hi, zstep);
    return retVal;
}

// Variant K_CT_OP_WITH_2_SAMPLES
template <>
inline T_accSym<int16, int16> macSrSymCT<K_CT_OP_WITH_2_SAMPLES>(T_accSym<int16, int16> acc,
                                                                 v32int16 xbuff,
                                                                 unsigned int xstart,
                                                                 v32int16 ybuff,
                                                                 unsigned int ystart,
                                                                 v16int16 zbuff,
                                                                 unsigned int zstart,
                                                                 unsigned int xbuffSwap) {
    T_accSym<int16, int16> retVal;
    const unsigned int xoffsets = 0x03020100;
    const unsigned int xoffsets_hi = 0x07060504;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zoffsets_hi = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int xstartmod = (xstart & 1) == 0 ? xstart : xstart - 1;
    const unsigned int xsquare = (xstart & 1) == 0 ? 0x2110 : 0x3221;
    const unsigned int ystartmod =
        (ystart & 1) == 0 ? ystart - 2 : ystart - 1; // no need to cater for second column here
    const unsigned int ysquare = (ystart & 1) == 0 ? 0x2312 : 0x1201;
    const unsigned int xbuffSize = 32;
    const unsigned int xbuffHalfSize = 16;
    const unsigned int xbuffQuarterSize = 8;
    const unsigned int xlanes = 16;
    // Swap for FIR len = xbuffSize*n-2.
    const bool xstartAtBoundary =
        ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize - xbuffQuarterSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v32int16 xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;

    retVal.val = mac16_sym(acc.val, xbuffInt, xstartYAdj, xoffsets, xoffsets_hi, xsquare, ybuff, ystartmod, ysquare,
                           zbuff, zstart, zoffsets, zoffsets_hi, zstep);
    return retVal;
}

// Variant K_CT_OP_WITH_3_SAMPLES
template <>
inline T_accSym<int16, int16> macSrSymCT<K_CT_OP_WITH_3_SAMPLES>(T_accSym<int16, int16> acc,
                                                                 v32int16 xbuff,
                                                                 unsigned int xstart,
                                                                 v32int16 ybuff,
                                                                 unsigned int ystart,
                                                                 v16int16 zbuff,
                                                                 unsigned int zstart,
                                                                 unsigned int xbuffSwap) {
    // no mac16_sym_ct...
    // perform a mac16 on xbuff, followed by mac16 on ybuff with prefabricated coeff register.
    // On top of that, swap xbuff for ybuff for FIR lengths: 32n + 3.
    T_accSym<int16, int16> retVal;
    const unsigned int xoffsets = 0x03020100;
    const unsigned int xoffsets_hi = 0x07060504;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zoffsets_hi = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int xstartmod = (xstart & 1) == 0 ? xstart : xstart - 1;
    const unsigned int xsquare = (xstart & 1) == 0 ? 0x2110 : 0x3221;
    const unsigned int ystartmod = (ystart & 1) == 0 ? ystart - 2 : ystart - 1;
    const unsigned int ysquare = (ystart & 1) == 0 ? 0x4332 : 0x3221;
    const unsigned int xbuffSize = 32;
    const unsigned int xbuffHalfSize = 16;
    const unsigned int xbuffQuarterSize = 8;
    const unsigned int xlanes = 16;
    // Swap for FIR len = xbuffSize*n-1.
    const bool xstartAtBoundary =
        ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize - xbuffQuarterSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v32int16 xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;

    // clear center tap coeff for ybuff operation,
    const v16int16 zbuffInt = (upd_elem(zbuff, zstart + zstep, 0));

    retVal.val = mac16(acc.val, xbuffInt, xstartYAdj, xoffsets, xoffsets_hi, xsquare, zbuff, zstart, zoffsets,
                       zoffsets_hi, zstep);
    retVal.val = mac16(retVal.val, ybuff, ystartmod, xoffsets, xoffsets_hi, ysquare, zbuffInt, zstart, zoffsets,
                       zoffsets_hi, zstep);

    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = cint16, COEFF = int16
// ---------------------------------------------------------------------------------------------------------------------
// Variant K_CT_OP_WITH_1_SAMPLE
template <>
inline T_accSym<cint16, int16> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<cint16, int16> acc,
                                                                 v16cint16 xbuff,
                                                                 unsigned int xstart,
                                                                 v16cint16 ybuff,
                                                                 unsigned int ystart,
                                                                 v16int16 zbuff,
                                                                 unsigned int zstart,
                                                                 unsigned int xbuffSwap) {
    T_accSym<cint16, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int xbuffSize = 16;
    const unsigned int xbuffHalfSize = 8;
    const unsigned int xlanes = 8;
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && (xstart % xbuffHalfSize) == 1) ? (xstart + xbuffHalfSize) : (xstart);
    const v16cint16 xbuffInt = ((xbuffSwap == 1) && (xstart % xbuffHalfSize) == 1) ? ybuff : xbuff;
    // Swap for FIR len = xbuffSize*n-1.
    retVal.val = mac4(acc.val, xbuffInt, xstartYAdj, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// Variant K_CT_OP_WITH_2_SAMPLES
template <>
inline T_accSym<cint16, int16> macSrSymCT<K_CT_OP_WITH_2_SAMPLES>(T_accSym<cint16, int16> acc,
                                                                  v16cint16 xbuff,
                                                                  unsigned int xstart,
                                                                  v16cint16 ybuff,
                                                                  unsigned int ystart,
                                                                  v16int16 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    T_accSym<cint16, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int xbuffSize = 16;
    const unsigned int xbuffHalfSize = 8;
    const unsigned int xbuffQuarterSize = 4;
    const unsigned int xlanes = 8;
    const bool xstartAtBoundary =
        ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize - xbuffQuarterSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v16cint16 xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;

    // Swap for FIR len = xbuffSize*n-2.
    retVal.val =
        mac4_sym(acc.val, xbuffInt, xstartYAdj, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// Variant K_CT_OP_WITH_3_SAMPLES
template <>
inline T_accSym<cint16, int16> macSrSymCT<K_CT_OP_WITH_3_SAMPLES>(T_accSym<cint16, int16> acc,
                                                                  v16cint16 xbuff,
                                                                  unsigned int xstart,
                                                                  v16cint16 ybuff,
                                                                  unsigned int ystart,
                                                                  v16int16 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    // xbuff has enough data for 1 column, but not enough data for second column.
    // ybuff has the data needed for both columns, so swap xbuff and ybuff and save on reloading xbuff.
    T_accSym<cint16, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int mtap = 1; // TODO
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int xbuffSize = 16;
    const unsigned int xbuffHalfSize = 8;
    const unsigned int xlanes = 8;
    const unsigned int xbuffQuarterSize = 4;
    const bool xstartAtBoundary =
        ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize - xbuffQuarterSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v16cint16 xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;
    v16int16 zbuffInt =
        (upd_elem(zbuff, zstart + (3 * zstep), ext_elem(zbuff, zstart + (1 * zstep)))); // rewrite ct coeff
    zbuffInt = (upd_elem(zbuffInt, zstart + (1 * zstep), 0));                           // clear 3rd column
    zbuffInt = (upd_elem(zbuffInt, zstart + (2 * zstep), 0));                           // clear 3rd column
    // Swap for FIR len = xbuffSize*n-1.
    retVal.val = mac4_sym_ct(acc.val, xbuffInt, xstartYAdj, xoffsets, xstep, ybuff, ystart, mtap, zbuffInt, zstart,
                             zoffsets, zstep);
    return retVal;
}

// Variant K_CT_OP_WITH_2_SAMPLES
template <>
inline T_accSym<cint16, int16> macSrSymCT<K_CT_OP_WITH_4_SAMPLES>(T_accSym<cint16, int16> acc,
                                                                  v16cint16 xbuff,
                                                                  unsigned int xstart,
                                                                  v16cint16 ybuff,
                                                                  unsigned int ystart,
                                                                  v16int16 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    T_accSym<cint16, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int xbuffSize = 16;
    const unsigned int xbuffHalfSize = 8;
    const unsigned int xbuffQuarterSize = 4;
    const unsigned int xlanes = 8;
    const bool xstartAtBoundary =
        ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize - xbuffQuarterSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v16cint16 xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;

    // Swap for FIR len = xbuffSize*n-2.
    retVal.val =
        mac4_sym(acc.val, xbuffInt, xstartYAdj, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// Variant K_CT_OP_WITH_3_SAMPLES
template <>
inline T_accSym<cint16, int16> macSrSymCT<K_CT_OP_WITH_5_SAMPLES>(T_accSym<cint16, int16> acc,
                                                                  v16cint16 xbuff,
                                                                  unsigned int xstart,
                                                                  v16cint16 ybuff,
                                                                  unsigned int ystart,
                                                                  v16int16 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    // xbuff has enough data for 1 column, but not enough data for second column.
    // ybuff has the data needed for both columns, so swap xbuff and ybuff and save on reloading xbuff.
    T_accSym<cint16, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int mtap = 2; // TODO
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int xbuffSize = 16;
    const unsigned int xbuffHalfSize = 8;
    const unsigned int xlanes = 8;
    const unsigned int xbuffQuarterSize = 4;
    const bool xstartAtBoundary =
        ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize - xbuffQuarterSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v16cint16 xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;
    v16int16 zbuffInt =
        (upd_elem(zbuff, zstart + (3 * zstep), ext_elem(zbuff, zstart + (2 * zstep)))); // rewrite ct coeff
    zbuffInt = (upd_elem(zbuffInt, zstart + (2 * zstep), 0));                           // clear 3rd column
    // Swap for FIR len = xbuffSize*n-1.
    retVal.val = mac4_sym_ct(acc.val, xbuffInt, xstartYAdj, xoffsets, xstep, ybuff, ystart, mtap, zbuffInt, zstart,
                             zoffsets, zstep);
    return retVal;
}

// Variant K_CT_OP_WITH_2_SAMPLES
template <>
inline T_accSym<cint16, int16> macSrSymCT<K_CT_OP_WITH_6_SAMPLES>(T_accSym<cint16, int16> acc,
                                                                  v16cint16 xbuff,
                                                                  unsigned int xstart,
                                                                  v16cint16 ybuff,
                                                                  unsigned int ystart,
                                                                  v16int16 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    T_accSym<cint16, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int xbuffSize = 16;
    const unsigned int xbuffHalfSize = 8;
    const unsigned int xbuffQuarterSize = 4;
    const unsigned int xlanes = 8;
    const bool xstartAtBoundary =
        ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize - xbuffQuarterSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v16cint16 xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;

    // Swap for FIR len = xbuffSize*n-2.
    retVal.val =
        mac4_sym(acc.val, xbuffInt, xstartYAdj, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// Variant K_CT_OP_WITH_3_SAMPLES
template <>
inline T_accSym<cint16, int16> macSrSymCT<K_CT_OP_WITH_7_SAMPLES>(T_accSym<cint16, int16> acc,
                                                                  v16cint16 xbuff,
                                                                  unsigned int xstart,
                                                                  v16cint16 ybuff,
                                                                  unsigned int ystart,
                                                                  v16int16 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    // xbuff has enough data for 1 column, but not enough data for second column.
    // ybuff has the data needed for both columns, so swap xbuff and ybuff and save on reloading xbuff.
    T_accSym<cint16, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int mtap = 3; // TODO
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int xbuffSize = 16;
    const unsigned int xbuffHalfSize = 8;
    const unsigned int xlanes = 8;
    const unsigned int xbuffQuarterSize = 4;
    const bool xstartAtBoundary =
        ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize - xbuffQuarterSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v16cint16 xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;
    // Swap for FIR len = xbuffSize*n-1.
    retVal.val = mac4_sym_ct(acc.val, xbuffInt, xstartYAdj, xoffsets, xstep, ybuff, ystart, mtap, zbuff, zstart,
                             zoffsets, zstep);
    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = cint16, COEFF = cint16
// ---------------------------------------------------------------------------------------------------------------------
// Variant K_CT_OP_WITH_1_SAMPLE
template <>
inline T_accSym<cint16, cint16> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<cint16, cint16> acc,
                                                                  v16cint16 xbuff,
                                                                  unsigned int xstart,
                                                                  v16cint16 ybuff,
                                                                  unsigned int ystart,
                                                                  v8cint16 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    T_accSym<cint16, cint16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int xbuffSize = 16;
    const unsigned int xbuffHalfSize = 8;
    const unsigned int xlanes = 8;
    const bool xstartAtBoundary = ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v16cint16 xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;
    // Swap for FIR len = xbuffSize*n-1.
    retVal.val = mac8(acc.val, xbuffInt, xstartYAdj, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = int32, COEFF = int16
// ---------------------------------------------------------------------------------------------------------------------
// Variant K_CT_OP_WITH_1_SAMPLE
template <>
inline T_accSym<int32, int16> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<int32, int16> acc,
                                                                v16int32 xbuff,
                                                                unsigned int xstart,
                                                                v16int32 ybuff,
                                                                unsigned int ystart,
                                                                v16int16 zbuff,
                                                                unsigned int zstart,
                                                                unsigned int xbuffSwap) {
    T_accSym<int32, int16> retVal;

    const unsigned int xoffsets = 0x76543210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int xbuffSize = 16;
    const unsigned int xbuffHalfSize = 8;
    const unsigned int xlanes = 8;
    const unsigned int xbuffQuarterSize = 4;
    const bool xstartAtBoundary =
        ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize - xbuffQuarterSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v16int32 xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;

    retVal.val = lmac8(acc.val, xbuffInt, xstartYAdj, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}
// Variant K_CT_OP_WITH_2_SAMPLES
template <>
inline T_accSym<int32, int16> macSrSymCT<K_CT_OP_WITH_2_SAMPLES>(T_accSym<int32, int16> acc,
                                                                 v16int32 xbuff,
                                                                 unsigned int xstart,
                                                                 v16int32 ybuff,
                                                                 unsigned int ystart,
                                                                 v16int16 zbuff,
                                                                 unsigned int zstart,
                                                                 unsigned int xbuffSwap) {
    T_accSym<int32, int16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int xstep = 1;
    const unsigned int zstep = 1;
    const unsigned int xbuffSize = 16;
    const unsigned int xbuffHalfSize = 8;
    const unsigned int xlanes = 8;
    const unsigned int xbuffQuarterSize = 4;
    const bool xstartAtBoundary =
        ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize - xbuffQuarterSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v16int32 xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;
    // Swap for FIR len = xbuffSize*n-2.
    retVal.val =
        lmac8_sym(acc.val, xbuffInt, xstartYAdj, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}
// Variant K_CT_OP_WITH_3_SAMPLES
template <>
inline T_accSym<int32, int16> macSrSymCT<K_CT_OP_WITH_3_SAMPLES>(T_accSym<int32, int16> acc,
                                                                 v16int32 xbuff,
                                                                 unsigned int xstart,
                                                                 v16int32 ybuff,
                                                                 unsigned int ystart,
                                                                 v16int16 zbuff,
                                                                 unsigned int zstart,
                                                                 unsigned int xbuffSwap) {
    T_accSym<int32, int16> retVal;

    const unsigned int xoffsets = 0x76543210;
    const unsigned int xstep = 1;
    const int mtap = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int xbuffSize = 16;
    const unsigned int xbuffHalfSize = 8;
    const unsigned int xlanes = 8;
    const unsigned int xbuffQuarterSize = 4;
    const bool xstartAtBoundary =
        ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize - xbuffQuarterSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v16int32 xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;
    // Swap for FIR len = xbuffSize*n-1.
    retVal.val =
        lmac8_sym_ct(acc.val, xbuffInt, xstartYAdj, xoffsets, ybuff, ystart, mtap, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = int32,  COEFF = int32>
// ---------------------------------------------------------------------------------------------------------------------
// Variant K_CT_OP_WITH_1_SAMPLE
template <>
inline T_accSym<int32, int32> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<int32, int32> acc,
                                                                v16int32 xbuff,
                                                                unsigned int xstart,
                                                                v16int32 ybuff,
                                                                unsigned int ystart,
                                                                v8int32 zbuff,
                                                                unsigned int zstart,
                                                                unsigned int xbuffSwap) {
    T_accSym<int32, int32> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int xbuffSize = 16;
    const unsigned int xbuffHalfSize = 8;
    const unsigned int xlanes = 8;
    const bool xstartAtBoundary = ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v16int32 xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;
    // Swap for FIR len = xbuffSize*n-1.
    retVal.val = lmac8(acc.val, xbuffInt, xstartYAdj, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = cint32, COEFF =  int16>
// ---------------------------------------------------------------------------------------------------------------------
// Variant K_CT_OP_WITH_1_SAMPLE
template <>
inline T_accSym<cint32, int16> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<cint32, int16> acc,
                                                                 v8cint32 xbuff,
                                                                 unsigned int xstart,
                                                                 v8cint32 ybuff,
                                                                 unsigned int ystart,
                                                                 v16int16 zbuff,
                                                                 unsigned int zstart,
                                                                 unsigned int xbuffSwap) {
    T_accSym<cint32, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int xstep = 1;
    const unsigned int zstep = 1;
    const unsigned int xbuffSize = 8;
    const unsigned int xbuffHalfSize = 4;
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && (xstart % xbuffSize) == 1) ? (xstart + xbuffHalfSize) : (xstart);
    const v8cint32 xbuffInt = ((xbuffSwap == 1) && (xstart % xbuffSize) == 1) ? ybuff : xbuff;

    retVal.val = lmac4(acc.val, xbuffInt, xstartYAdj, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}
// Variant K_CT_OP_WITH_2_SAMPLES
template <>
inline T_accSym<cint32, int16> macSrSymCT<K_CT_OP_WITH_2_SAMPLES>(T_accSym<cint32, int16> acc,
                                                                  v8cint32 xbuff,
                                                                  unsigned int xstart,
                                                                  v8cint32 ybuff,
                                                                  unsigned int ystart,
                                                                  v16int16 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    T_accSym<cint32, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int xstep = 1;
    const unsigned int zstep = 1;
    const unsigned int xbuffSize = 8;
    const unsigned int xbuffHalfSize = 4;
    const unsigned int xlanes = 4;
    const bool xstartAtBoundary = ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v8cint32 xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;

    retVal.val =
        lmac4_sym(acc.val, xbuffInt, xstartYAdj, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}
// Variant K_CT_OP_WITH_3_SAMPLES
template <>
inline T_accSym<cint32, int16> macSrSymCT<K_CT_OP_WITH_3_SAMPLES>(T_accSym<cint32, int16> acc,
                                                                  v8cint32 xbuff,
                                                                  unsigned int xstart,
                                                                  v8cint32 ybuff,
                                                                  unsigned int ystart,
                                                                  v16int16 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    T_accSym<cint32, int16> retVal;

    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const int mtap = 1;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int xbuffSize = 8;
    const unsigned int xbuffHalfSize = 4;
    const unsigned int xlanes = 4;
    const bool xstartAtBoundary = ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v8cint32 xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;

    retVal.val =
        lmac4_sym_ct(acc.val, xbuffInt, xstartYAdj, xoffsets, ybuff, ystart, mtap, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = cint32, COEFF =  int32>
// ---------------------------------------------------------------------------------------------------------------------
template <>
inline T_accSym<cint32, int32> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<cint32, int32> acc,
                                                                 v8cint32 xbuff,
                                                                 unsigned int xstart,
                                                                 v8cint32 ybuff,
                                                                 unsigned int ystart,
                                                                 v8int32 zbuff,
                                                                 unsigned int zstart,
                                                                 unsigned int xbuffSwap) {
    T_accSym<cint32, int32> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int xbuffSize = 8;
    const unsigned int xbuffHalfSize = 4;
    const unsigned int xlanes = 4;
    const bool xstartAtBoundary = ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v8cint32 xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;

    retVal.val = lmac4(acc.val, xbuffInt, xstartYAdj, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = cint32, COEFF =  cint16>
// ---------------------------------------------------------------------------------------------------------------------
template <>
inline T_accSym<cint32, cint16> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<cint32, cint16> acc,
                                                                  v8cint32 xbuff,
                                                                  unsigned int xstart,
                                                                  v8cint32 ybuff,
                                                                  unsigned int ystart,
                                                                  v8cint16 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    T_accSym<cint32, cint16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int xbuffSize = 8;
    const unsigned int xbuffHalfSize = 4;
    const unsigned int xlanes = 4;
    const bool xstartAtBoundary = ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v8cint32 xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;

    retVal.val = lmac4(acc.val, xbuffInt, xstartYAdj, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = cint32, COEFF =  cint32>
// ---------------------------------------------------------------------------------------------------------------------
template <>
inline T_accSym<cint32, cint32> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<cint32, cint32> acc,
                                                                  v8cint32 xbuff,
                                                                  unsigned int xstart,
                                                                  v8cint32 ybuff,
                                                                  unsigned int ystart,
                                                                  v4cint32 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    T_accSym<cint32, cint32> retVal;
    const unsigned int xoffsets = 0x10;
    const unsigned int zoffsets = 0x0000;
    const unsigned int xbuffSize = 8;
    const unsigned int xbuffHalfSize = 4;
    const unsigned int xlanes = 4;
    const bool xstartAtBoundary = ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v8cint32 xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;

    retVal.val = lmac2(acc.val, xbuffInt, xstartYAdj, xoffsets, zbuff, zstart, zoffsets);
    retVal.uval = lmac2(acc.uval, xbuffInt, xstartYAdj + 2, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = float,  COEFF = float>
// ---------------------------------------------------------------------------------------------------------------------
template <>
inline T_accSym<float, float> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<float, float> acc,
                                                                v16float xbuff,
                                                                unsigned int xstart,
                                                                v16float ybuff,
                                                                unsigned int ystart,
                                                                v8float zbuff,
                                                                unsigned int zstart,
                                                                unsigned int xbuffSwap) {
    T_accSym<float, float> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int xbuffSize = 16;
    const unsigned int xbuffHalfSize = 8;
    const unsigned int xlanes = 8;
    const bool xstartAtBoundary = ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v16float xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;

    retVal.val = fpmac(acc.val, xbuffInt, xstartYAdj, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = cfloat, COEFF =  float>
// ---------------------------------------------------------------------------------------------------------------------
template <>
inline T_accSym<cfloat, float> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<cfloat, float> acc,
                                                                 v8cfloat xbuff,
                                                                 unsigned int xstart,
                                                                 v8cfloat ybuff,
                                                                 unsigned int ystart,
                                                                 v8float zbuff,
                                                                 unsigned int zstart,
                                                                 unsigned int xbuffSwap) {
    T_accSym<cfloat, float> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int xbuffSize = 8;
    const unsigned int xbuffHalfSize = 4;
    const unsigned int xlanes = 4;
    const bool xstartAtBoundary = ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v8cfloat xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;

    retVal.val = fpmac(acc.val, xbuffInt, xstartYAdj, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = cfloat, COEFF =  cfloat>
// ---------------------------------------------------------------------------------------------------------------------
template <>
inline T_accSym<cfloat, cfloat> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<cfloat, cfloat> acc,
                                                                  v8cfloat xbuff,
                                                                  unsigned int xstart,
                                                                  v8cfloat ybuff,
                                                                  unsigned int ystart,
                                                                  v4cfloat zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    T_accSym<cfloat, cfloat> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int xbuffSize = 8;
    const unsigned int xbuffHalfSize = 4;
    const unsigned int xlanes = 4;
    const bool xstartAtBoundary = ((xstart % xbuffSize) <= 1) || ((xstart % xbuffSize + xlanes) >= xbuffSize);
    const bool xstartAddOffset = ((xstart % xbuffSize) / xbuffHalfSize) != ((ystart % xbuffSize) / xbuffHalfSize);
    const unsigned int xstartYAdj =
        ((xbuffSwap == 1) && xstartAtBoundary && xstartAddOffset) ? (xstart + xbuffHalfSize) : (xstart);
    const v8cfloat xbuffInt = ((xbuffSwap == 1) && xstartAtBoundary) ? ybuff : xbuff;

    retVal.val = fpmac(acc.val, xbuffInt, xstartYAdj, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}
// ---------------------------------------------------- 1 BUFF ---------------------------------------------------- //

// overloaded mul/macSrSymCT calls.
// second set of intrinsics, to cover 2 smaller xbuff and ybuff buffers
//-----------------------------------------------------------------------------------------------------
// DATA = int16, COEFF = int16
template <unsigned int T_Variant = 0>
inline T_accSym<int16, int16> macSrSymCT(T_accSym<int16, int16> acc,
                                         v64int16 xbuff,
                                         unsigned int xstart,
                                         v16int16 zbuff,
                                         unsigned int zstart,
                                         unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<cint16, int16> macSrSymCT(T_accSym<cint16, int16> acc,
                                          v32cint16 xbuff,
                                          unsigned int xstart,
                                          v16int16 zbuff,
                                          unsigned int zstart,
                                          unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<cint16, cint16> macSrSymCT(T_accSym<cint16, cint16> acc,
                                           v32cint16 xbuff,
                                           unsigned int xstart,
                                           v8cint16 zbuff,
                                           unsigned int zstart,
                                           unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<int32, int16> macSrSymCT(T_accSym<int32, int16> acc,
                                         v32int32 xbuff,
                                         unsigned int xstart,
                                         v16int16 zbuff,
                                         unsigned int zstart,
                                         unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<int32, int32> macSrSymCT(T_accSym<int32, int32> acc,
                                         v32int32 xbuff,
                                         unsigned int xstart,
                                         v8int32 zbuff,
                                         unsigned int zstart,
                                         unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<cint32, int16> macSrSymCT(T_accSym<cint32, int16> acc,
                                          v16cint32 xbuff,
                                          unsigned int xstart,
                                          v16int16 zbuff,
                                          unsigned int zstart,
                                          unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<cint32, int32> macSrSymCT(T_accSym<cint32, int32> acc,
                                          v16cint32 xbuff,
                                          unsigned int xstart,
                                          v8int32 zbuff,
                                          unsigned int zstart,
                                          unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<cint32, cint16> macSrSymCT(T_accSym<cint32, cint16> acc,
                                           v16cint32 xbuff,
                                           unsigned int xstart,
                                           v8cint16 zbuff,
                                           unsigned int zstart,
                                           unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<cint32, cint32> macSrSymCT(T_accSym<cint32, cint32> acc,
                                           v16cint32 xbuff,
                                           unsigned int xstart,
                                           v4cint32 zbuff,
                                           unsigned int zstart,
                                           unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<float, float> macSrSymCT(T_accSym<float, float> acc,
                                         v32float xbuff,
                                         unsigned int xstart,
                                         v8float zbuff,
                                         unsigned int zstart,
                                         unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<cfloat, float> macSrSymCT(T_accSym<cfloat, float> acc,
                                          v16cfloat xbuff,
                                          unsigned int xstart,
                                          v8float zbuff,
                                          unsigned int zstart,
                                          unsigned int xbuffSwap = 0) {
    return acc;
}

template <unsigned int T_Variant = 0>
inline T_accSym<cfloat, cfloat> macSrSymCT(T_accSym<cfloat, cfloat> acc,
                                           v16cfloat xbuff,
                                           unsigned int xstart,
                                           v4cfloat zbuff,
                                           unsigned int zstart,
                                           unsigned int xbuffSwap = 0) {
    return acc;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = int16, COEFF = int16
// ---------------------------------------------------------------------------------------------------------------------
// Variant K_CT_OP_WITH_1_SAMPLE
template <>
inline T_accSym<int16, int16> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<int16, int16> acc,
                                                                v64int16 xbuff,
                                                                unsigned int xstart,
                                                                v16int16 zbuff,
                                                                unsigned int zstart,
                                                                unsigned int xbuffSwap) {
    T_accSym<int16, int16> retVal;
    const unsigned int xoffsets = 0x03020100;
    const unsigned int xoffsets_hi = 0x07060504;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zoffsets_hi = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int xstartmod = (xstart & 1) == 0 ? xstart : xstart - 1;
    const unsigned int xsquare = (xstart & 1) == 0 ? 0x2110 : 0x3221;

    retVal.val =
        mac16(acc.val, xbuff, xstart, xoffsets, xoffsets_hi, xsquare, zbuff, zstart, zoffsets, zoffsets_hi, zstep);
    return retVal;
}

// Variant K_CT_OP_WITH_2_SAMPLES
template <>
inline T_accSym<int16, int16> macSrSymCT<K_CT_OP_WITH_2_SAMPLES>(T_accSym<int16, int16> acc,
                                                                 v64int16 xbuff,
                                                                 unsigned int xstart,
                                                                 v16int16 zbuff,
                                                                 unsigned int zstart,
                                                                 unsigned int xbuffSwap) {
    T_accSym<int16, int16> retVal;
    const unsigned int xoffsets = 0x03020100;
    const unsigned int xoffsets_hi = 0x07060504;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zoffsets_hi = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int xstartmod = (xstart & 1) == 0 ? xstart : xstart - 1;
    const unsigned int xsquare = (xstart & 1) == 0 ? 0x2110 : 0x3221;
    const unsigned int ystartmod = (xstart & 1) == 0 ? xstart : xstart - 1;
    const unsigned int ysquare = (xstart & 1) == 0 ? 0x3221 : 0x4332;

    retVal.val = mac16_sym(acc.val, xbuff, xstartmod, xoffsets, xoffsets_hi, xsquare, ystartmod, ysquare, zbuff, zstart,
                           zoffsets, zoffsets_hi, zstep);
    return retVal;
}

// Variant K_CT_OP_WITH_3_SAMPLES
template <>
inline T_accSym<int16, int16> macSrSymCT<K_CT_OP_WITH_3_SAMPLES>(T_accSym<int16, int16> acc,
                                                                 v64int16 xbuff,
                                                                 unsigned int xstart,
                                                                 v16int16 zbuff,
                                                                 unsigned int zstart,
                                                                 unsigned int xbuffSwap) {
    // no mac16_sym_ct...
    // perform a mac16 on xbuff, followed by mac16 on ybuff with prefabricated coeff register.
    // On top of that, swap xbuff for ybuff for FIR lengths: 32n + 3.
    T_accSym<int16, int16> retVal;
    const unsigned int xoffsets = 0x03020100;
    const unsigned int xoffsets_hi = 0x07060504;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zoffsets_hi = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int xstartmod = (xstart & 1) == 0 ? xstart : xstart - 1;
    const unsigned int xsquare = (xstart & 1) == 0 ? 0x2110 : 0x3221;
    const unsigned int ystartmod = (xstart & 1) == 0 ? xstart + 2 : xstart + 1;
    const unsigned int ysquare = (xstart & 1) == 0 ? 0x2110 : 0x2110;

    // clear center tap coeff for ybuff operation,
    const v16int16 zbuffInt = (upd_elem(zbuff, zstart + zstep, 0));

    retVal.val =
        mac16(acc.val, xbuff, xstart, xoffsets, xoffsets_hi, xsquare, zbuff, zstart, zoffsets, zoffsets_hi, zstep);
    retVal.val = mac16(retVal.val, xbuff, ystartmod, xoffsets, xoffsets_hi, ysquare, zbuffInt, zstart, zoffsets,
                       zoffsets_hi, zstep);

    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = cint16, COEFF = int16
// ---------------------------------------------------------------------------------------------------------------------
// Variant K_CT_OP_WITH_1_SAMPLE
template <>
inline T_accSym<cint16, int16> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<cint16, int16> acc,
                                                                 v32cint16 xbuff,
                                                                 unsigned int xstart,
                                                                 v16int16 zbuff,
                                                                 unsigned int zstart,
                                                                 unsigned int xbuffSwap) {
    T_accSym<cint16, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    retVal.val = mac4(acc.val, xbuff, xstart, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// Variant K_CT_OP_WITH_2_SAMPLES
template <>
inline T_accSym<cint16, int16> macSrSymCT<K_CT_OP_WITH_2_SAMPLES>(T_accSym<cint16, int16> acc,
                                                                  v32cint16 xbuff,
                                                                  unsigned int xstart,
                                                                  v16int16 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    T_accSym<cint16, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    retVal.val = mac4_sym(acc.val, xbuff, xstart, xoffsets, xstep, xstart + 1, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// Variant K_CT_OP_WITH_3_SAMPLES
template <>
inline T_accSym<cint16, int16> macSrSymCT<K_CT_OP_WITH_3_SAMPLES>(T_accSym<cint16, int16> acc,
                                                                  v32cint16 xbuff,
                                                                  unsigned int xstart,
                                                                  v16int16 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    // xbuff has enough data for 1 column, but not enough data for second column.
    // ybuff has the data needed for both columns, so swap xbuff and ybuff and save on reloading xbuff.
    T_accSym<cint16, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int mtap = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    v16int16 zbuffInt =
        (upd_elem(zbuff, zstart + (3 * zstep), ext_elem(zbuff, zstart + (1 * zstep)))); // rewrite ct coeff
    zbuffInt = (upd_elem(zbuffInt, zstart + (1 * zstep), 0));                           // clear 3rd column
    zbuffInt = (upd_elem(zbuffInt, zstart + (2 * zstep), 0));                           // clear 3rd column
    retVal.val =
        mac4_sym_ct(acc.val, xbuff, xstart, xoffsets, xstep, xstart + 2, mtap, zbuffInt, zstart, zoffsets, zstep);
    return retVal;
}
// Variant K_CT_OP_WITH_2_SAMPLES
template <>
inline T_accSym<cint16, int16> macSrSymCT<K_CT_OP_WITH_4_SAMPLES>(T_accSym<cint16, int16> acc,
                                                                  v32cint16 xbuff,
                                                                  unsigned int xstart,
                                                                  v16int16 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    T_accSym<cint16, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    retVal.val = mac4_sym(acc.val, xbuff, xstart, xoffsets, xstep, xstart + 3, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// Variant K_CT_OP_WITH_3_SAMPLES
template <>
inline T_accSym<cint16, int16> macSrSymCT<K_CT_OP_WITH_5_SAMPLES>(T_accSym<cint16, int16> acc,
                                                                  v32cint16 xbuff,
                                                                  unsigned int xstart,
                                                                  v16int16 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    // xbuff has enough data for 1 column, but not enough data for second column.
    // ybuff has the data needed for both columns, so swap xbuff and ybuff and save on reloading xbuff.
    T_accSym<cint16, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int mtap = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    v16int16 zbuffInt =
        (upd_elem(zbuff, zstart + (3 * zstep), ext_elem(zbuff, zstart + (2 * zstep)))); // rewrite ct coeff
    zbuffInt = (upd_elem(zbuffInt, zstart + (2 * zstep), 0));                           // clear 3rd column
    retVal.val =
        mac4_sym_ct(acc.val, xbuff, xstart, xoffsets, xstep, xstart + 4, mtap, zbuffInt, zstart, zoffsets, zstep);
    return retVal;
}
// Variant K_CT_OP_WITH_2_SAMPLES
template <>
inline T_accSym<cint16, int16> macSrSymCT<K_CT_OP_WITH_6_SAMPLES>(T_accSym<cint16, int16> acc,
                                                                  v32cint16 xbuff,
                                                                  unsigned int xstart,
                                                                  v16int16 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    T_accSym<cint16, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    retVal.val = mac4_sym(acc.val, xbuff, xstart, xoffsets, xstep, xstart + 5, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// Variant K_CT_OP_WITH_3_SAMPLES
template <>
inline T_accSym<cint16, int16> macSrSymCT<K_CT_OP_WITH_7_SAMPLES>(T_accSym<cint16, int16> acc,
                                                                  v32cint16 xbuff,
                                                                  unsigned int xstart,
                                                                  v16int16 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    // xbuff has enough data for 1 column, but not enough data for second column.
    // ybuff has the data needed for both columns, so swap xbuff and ybuff and save on reloading xbuff.
    T_accSym<cint16, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int mtap = 3;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    retVal.val = mac4_sym_ct(acc.val, xbuff, xstart, xoffsets, xstep, xstart + 6, mtap, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = cint16, COEFF = cint16
// ---------------------------------------------------------------------------------------------------------------------
// Variant K_CT_OP_WITH_1_SAMPLE
template <>
inline T_accSym<cint16, cint16> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<cint16, cint16> acc,
                                                                  v32cint16 xbuff,
                                                                  unsigned int xstart,
                                                                  v8cint16 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    T_accSym<cint16, cint16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;
    retVal.val = mac8(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = int32, COEFF = int16
// ---------------------------------------------------------------------------------------------------------------------
// Variant K_CT_OP_WITH_1_SAMPLE
template <>
inline T_accSym<int32, int16> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<int32, int16> acc,
                                                                v32int32 xbuff,
                                                                unsigned int xstart,
                                                                v16int16 zbuff,
                                                                unsigned int zstart,
                                                                unsigned int xbuffSwap) {
    T_accSym<int32, int16> retVal;

    const unsigned int xoffsets = 0x76543210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;

    retVal.val = lmac8(acc.val, xbuff, xstart, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}
// Variant K_CT_OP_WITH_2_SAMPLES
template <>
inline T_accSym<int32, int16> macSrSymCT<K_CT_OP_WITH_2_SAMPLES>(T_accSym<int32, int16> acc,
                                                                 v32int32 xbuff,
                                                                 unsigned int xstart,
                                                                 v16int16 zbuff,
                                                                 unsigned int zstart,
                                                                 unsigned int xbuffSwap) {
    T_accSym<int32, int16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int xstep = 1;
    const unsigned int zstep = 1;

    retVal.val = lmac8_sym(acc.val, xbuff, xstart, xoffsets, xstep, xstart + 1, zbuff, zstart, zoffsets, zstep);
    return retVal;
}
// Variant K_CT_OP_WITH_3_SAMPLES
template <>
inline T_accSym<int32, int16> macSrSymCT<K_CT_OP_WITH_3_SAMPLES>(T_accSym<int32, int16> acc,
                                                                 v32int32 xbuff,
                                                                 unsigned int xstart,
                                                                 v16int16 zbuff,
                                                                 unsigned int zstart,
                                                                 unsigned int xbuffSwap) {
    T_accSym<int32, int16> retVal;

    const unsigned int xoffsets = 0x76543210;
    const unsigned int xstep = 1;
    const int mtap = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;

    retVal.val = lmac8_sym_ct(acc.val, xbuff, xstart, xoffsets, xstart + 2, mtap, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = int32,  COEFF = int32>
// ---------------------------------------------------------------------------------------------------------------------
// Variant K_CT_OP_WITH_1_SAMPLE
template <>
inline T_accSym<int32, int32> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<int32, int32> acc,
                                                                v32int32 xbuff,
                                                                unsigned int xstart,
                                                                v8int32 zbuff,
                                                                unsigned int zstart,
                                                                unsigned int xbuffSwap) {
    T_accSym<int32, int32> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;

    retVal.val = lmac8(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = cint32, COEFF =  int16>
// ---------------------------------------------------------------------------------------------------------------------
// Variant K_CT_OP_WITH_1_SAMPLE
template <>
inline T_accSym<cint32, int16> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<cint32, int16> acc,
                                                                 v16cint32 xbuff,
                                                                 unsigned int xstart,
                                                                 v16int16 zbuff,
                                                                 unsigned int zstart,
                                                                 unsigned int xbuffSwap) {
    T_accSym<cint32, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int xstep = 1;
    const unsigned int zstep = 1;

    retVal.val = lmac4(acc.val, xbuff, xstart, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}
// Variant K_CT_OP_WITH_2_SAMPLES
template <>
inline T_accSym<cint32, int16> macSrSymCT<K_CT_OP_WITH_2_SAMPLES>(T_accSym<cint32, int16> acc,
                                                                  v16cint32 xbuff,
                                                                  unsigned int xstart,
                                                                  v16int16 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    T_accSym<cint32, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int xstep = 1;
    const unsigned int zstep = 1;

    retVal.val = lmac4_sym(acc.val, xbuff, xstart, xoffsets, xstep, xstart + 1, zbuff, zstart, zoffsets, zstep);
    return retVal;
}
// Variant K_CT_OP_WITH_3_SAMPLES
template <>
inline T_accSym<cint32, int16> macSrSymCT<K_CT_OP_WITH_3_SAMPLES>(T_accSym<cint32, int16> acc,
                                                                  v16cint32 xbuff,
                                                                  unsigned int xstart,
                                                                  v16int16 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    T_accSym<cint32, int16> retVal;

    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const int mtap = 1;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;

    retVal.val = lmac4_sym_ct(acc.val, xbuff, xstart, xoffsets, xstart + 2, mtap, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = cint32, COEFF =  int32>
// ---------------------------------------------------------------------------------------------------------------------
template <>
inline T_accSym<cint32, int32> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<cint32, int32> acc,
                                                                 v16cint32 xbuff,
                                                                 unsigned int xstart,
                                                                 v8int32 zbuff,
                                                                 unsigned int zstart,
                                                                 unsigned int xbuffSwap) {
    T_accSym<cint32, int32> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = lmac4(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = cint32, COEFF =  cint16>
// ---------------------------------------------------------------------------------------------------------------------
template <>
inline T_accSym<cint32, cint16> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<cint32, cint16> acc,
                                                                  v16cint32 xbuff,
                                                                  unsigned int xstart,
                                                                  v8cint16 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    T_accSym<cint32, cint16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = lmac4(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = cint32, COEFF =  cint32>
// ---------------------------------------------------------------------------------------------------------------------
template <>
inline T_accSym<cint32, cint32> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<cint32, cint32> acc,
                                                                  v16cint32 xbuff,
                                                                  unsigned int xstart,
                                                                  v4cint32 zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    T_accSym<cint32, cint32> retVal;
    const unsigned int xoffsets = 0x10;
    const unsigned int zoffsets = 0x0000;

    retVal.val = lmac2(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.uval = lmac2(acc.uval, xbuff, xstart + 2, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = float,  COEFF = float>
// ---------------------------------------------------------------------------------------------------------------------
template <>
inline T_accSym<float, float> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<float, float> acc,
                                                                v32float xbuff,
                                                                unsigned int xstart,
                                                                v8float zbuff,
                                                                unsigned int zstart,
                                                                unsigned int xbuffSwap) {
    T_accSym<float, float> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;

    retVal.val = fpmac(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = cfloat, COEFF =  float>
// ---------------------------------------------------------------------------------------------------------------------
template <>
inline T_accSym<cfloat, float> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<cfloat, float> acc,
                                                                 v16cfloat xbuff,
                                                                 unsigned int xstart,
                                                                 v8float zbuff,
                                                                 unsigned int zstart,
                                                                 unsigned int xbuffSwap) {
    T_accSym<cfloat, float> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = fpmac(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// ---------------------------------------------------------------------------------------------------------------------
// DATA = cfloat, COEFF =  cfloat>
// ---------------------------------------------------------------------------------------------------------------------
template <>
inline T_accSym<cfloat, cfloat> macSrSymCT<K_CT_OP_WITH_1_SAMPLE>(T_accSym<cfloat, cfloat> acc,
                                                                  v16cfloat xbuff,
                                                                  unsigned int xstart,
                                                                  v4cfloat zbuff,
                                                                  unsigned int zstart,
                                                                  unsigned int xbuffSwap) {
    T_accSym<cfloat, cfloat> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = fpmac(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// ---------------------------------------------------- 2 BUFF ---------------------------------------------------- //

// Initial MAC/MUL operation. Take inputIF as an argument to ease overloading.
template <typename TT_DATA, typename TT_COEFF>
inline T_accSym<TT_DATA, TT_COEFF> initMacSrSym(T_inputIF<CASC_IN_FALSE, TT_DATA> inInterface,
                                                T_accSym<TT_DATA, TT_COEFF> acc,
                                                T_buff_512b<TT_DATA> xbuff,
                                                unsigned int xstart,
                                                T_buff_512b<TT_DATA> ybuff,
                                                unsigned int ystart,
                                                T_buff_256b<TT_COEFF> zbuff,
                                                unsigned int zstart) {
    return mulSrSym(xbuff.val, xstart, ybuff.val, ystart, zbuff.val, zstart);
};

template <typename TT_DATA, typename TT_COEFF>
inline T_accSym<TT_DATA, TT_COEFF> initMacSrSym(T_inputIF<CASC_IN_TRUE, TT_DATA> inInterface,
                                                T_accSym<TT_DATA, TT_COEFF> acc,
                                                T_buff_512b<TT_DATA> xbuff,
                                                unsigned int xstart,
                                                T_buff_512b<TT_DATA> ybuff,
                                                unsigned int ystart,
                                                T_buff_256b<TT_COEFF> zbuff,
                                                unsigned int zstart) {
    return macSrSym(acc, xbuff.val, xstart, ybuff.val, ystart, zbuff.val, zstart);
};

// ---------------------------------------------------- 1 BUFF ---------------------------------------------------- //

// Initial MAC/MUL operation. Take inputIF as an argument to ease overloading.
template <typename TT_DATA, typename TT_COEFF>
inline T_accSym<TT_DATA, TT_COEFF> initMacSrSym(T_inputIF<CASC_IN_FALSE, TT_DATA> inInterface,
                                                T_accSym<TT_DATA, TT_COEFF> acc,
                                                T_buff_1024b<TT_DATA> xbuff,
                                                unsigned int xstart,
                                                unsigned int ystart,
                                                T_buff_256b<TT_COEFF> zbuff,
                                                unsigned int zstart) {
    return mulSrSym(xbuff.val, xstart, ystart, zbuff.val, zstart);
};

template <typename TT_DATA, typename TT_COEFF>
inline T_accSym<TT_DATA, TT_COEFF> initMacSrSym(T_inputIF<CASC_IN_TRUE, TT_DATA> inInterface,
                                                T_accSym<TT_DATA, TT_COEFF> acc,
                                                T_buff_1024b<TT_DATA> xbuff,
                                                unsigned int xstart,
                                                unsigned int ystart,
                                                T_buff_256b<TT_COEFF> zbuff,
                                                unsigned int zstart) {
    return macSrSym(acc, xbuff.val, xstart, ystart, zbuff.val, zstart);
};

inline constexpr unsigned int fnCTColumnsLeft(unsigned int firLen, unsigned int columns) {
    // Returns number of columns left to process in the centre tap operation.
    return (firLen % (2 * columns) + 1) / 2;
};
}
}
}
}
}

#endif // _DSPLIB_FIR_SR_SYM_UTILS_HPP_
