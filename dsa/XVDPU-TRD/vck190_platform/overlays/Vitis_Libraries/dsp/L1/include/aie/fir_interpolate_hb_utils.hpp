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
#ifndef _DSPLIB_FIR_INTERPOLATE_HB_UTILS_HPP_
#define _DSPLIB_FIR_INTERPOLATE_HB_UTILS_HPP_

namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace interpolate_hb {
/*
Halfband interpolating FIR Utilities
This file contains sets of overloaded, templatized and specialized templatized functions for use
by the main kernel class and run-time function. These functions are separate from the traits file
because they are purely for kernel use, not graph level compilation.
*/

// Specialised type for final accumulator. Concat of two polyphases
template <typename T_D, typename T_C>
struct T_accIntHb : T_acc<T_D, T_C> {
    using T_acc<T_D, T_C>::operator=;
};
// template<> struct T_accIntHb<int16, int16>   {v16acc48 val = null_v16acc48(); v16acc48 uval = null_v16acc48();};
// template<> struct T_accIntHb<cint16, int16>  {v8cacc48 val = null_v8cacc48(); v8cacc48 uval = null_v8cacc48();};
// template<> struct T_accIntHb<cint16, cint16> {v8cacc48 val = null_v8cacc48(); v8cacc48 uval = null_v8cacc48();};
// template<> struct T_accIntHb<int32, int16>   {v8acc80  val = null_v8acc80 (); v8acc80  uval = null_v8acc80 ();};
// template<> struct T_accIntHb<int32, int32>   {v8acc80  val = null_v8acc80 (); v8acc80  uval = null_v8acc80 ();};
// template<> struct T_accIntHb<cint32, int16>  {v4cacc80 val = null_v4cacc80(); v4cacc80 uval = null_v4cacc80();};
// template<> struct T_accIntHb<cint32, int32>  {v4cacc80 val = null_v4cacc80(); v4cacc80 uval = null_v4cacc80();};
// template<> struct T_accIntHb<cint32, cint16> {v4cacc80 val = null_v4cacc80(); v4cacc80 uval = null_v4cacc80();};
// template<> struct T_accIntHb<cint32, cint32> {v2cacc80 val = null_v2cacc80(); v2cacc80 uval = null_v2cacc80();};
// template<> struct T_accIntHb<float,  float>  {v8float  val = null_v8float (); v8float  uval = null_v8float ();};
// template<> struct T_accIntHb<cfloat, float>  {v4cfloat val = null_v4cfloat(); v4cfloat uval = null_v4cfloat();};
// template<> struct T_accIntHb<cfloat, cfloat> {v4cfloat val = null_v4cfloat(); v4cfloat uval = null_v4cfloat();};
// Smaller type for each polyphase
template <typename T_D, typename T_C, unsigned int T_UCT = 0>
struct T_accSymIntHb : T_acc384<T_D, T_C> {
    using T_acc384<T_D, T_C>::operator=;
};
template <>
struct T_accSymIntHb<int16, int16, 1> : T_acc<int16, int16> {
    using T_acc<int16, int16>::operator=;
};
template <>
struct T_accSymIntHb<cint16, int16, 1> : T_acc<cint16, int16> {
    using T_acc<cint16, int16>::operator=;
};
template <>
struct T_accSymIntHb<cint16, cint16, 1> : T_acc<cint16, cint16> {
    using T_acc<cint16, cint16>::operator=;
};
// template<> struct T_accSymIntHb<int32, int16>   {v8acc80  val = null_v8acc80 (); v8acc80  uval = null_v8acc80 ();};
// template<> struct T_accSymIntHb<int32, int32>   {v4acc80  val = null_v4acc80 (); v4acc80  uval = null_v4acc80 ();};
// template<> struct T_accSymIntHb<cint32, int16>  {v4cacc80 val = null_v4cacc80(); v4cacc80 uval = null_v4cacc80();};
// template<> struct T_accSymIntHb<cint32, int32>  {v4cacc80 val = null_v4cacc80(); v4cacc80 uval = null_v4cacc80();};
// template<> struct T_accSymIntHb<cint32, cint16> {v4cacc80 val = null_v4cacc80(); v4cacc80 uval = null_v4cacc80();};
// template<> struct T_accSymIntHb<cint32, cint32> {v2cacc80 val = null_v2cacc80(); v2cacc80 uval = null_v2cacc80();};
// template<> struct T_accSymIntHb<float,  float>  {v8float  val = null_v8float (); v8float  uval = null_v8float ();};
// template<> struct T_accSymIntHb<cfloat, float>  {v4cfloat val = null_v4cfloat(); v4cfloat uval = null_v4cfloat();};
// template<> struct T_accSymIntHb<cfloat, cfloat> {v4cfloat val = null_v4cfloat(); v4cfloat uval = null_v4cfloat();};
// Final output value type after shift and rounding
template <typename T_D, typename T_C>
struct T_outValIntHb : T_outVal<T_D, T_C> {
    using T_outVal<T_D, T_C>::operator=;
};
// template<> struct T_outValIntHb<int16, int16>   {v16int16 val;};//v16int16 outVal;};
// template<> struct T_outValIntHb<cint16, int16>  {v8cint16 val;};//v8cint16 outVal;};
// template<> struct T_outValIntHb<cint16, cint16> {v8cint16 val;};//v8cint16 outVal;};
// template<> struct T_outValIntHb<int32, int16>   {v8int32  val;};//v8int32  outVal;};
// template<> struct T_outValIntHb<int32, int32>   {v8int32  val;};//v8int32  outVal;};
// template<> struct T_outValIntHb<cint32, int16>  {v4cint32 val;};//v4cint32 outVal;};
// template<> struct T_outValIntHb<cint32, int32>  {v4cint32 val;};//v4cint32 outVal;};
// template<> struct T_outValIntHb<cint32, cint16> {v4cint32 val;};//v4cint32 outVal;};
// template<> struct T_outValIntHb<cint32, cint32> {v4cint32 val;};//v4cint32 outVal;};
// template<> struct T_outValIntHb<float, float>   {v8float  val;};//v8float  outVal;};
// template<> struct T_outValIntHb<cfloat, float>  {v4cfloat val;};//v4cfloat outVal;};
// template<> struct T_outValIntHb<cfloat, cfloat> {v4cfloat val;};//v4cfloat outVal;};

// Symmetric mul/mac for use in halfband interpolator.
//-----------------------------------------------------------------------------------------------------
// DATA = int16, COEFF = int16
inline T_accSymIntHb<int16, int16> mulSym2buffIntHb(
    v32int16 xbuff, unsigned int xstart, v32int16 ybuff, unsigned int ystart, v16int16 zbuff, unsigned int zstart) {
    T_accSymIntHb<int16, int16> retVal;
    const unsigned int xoffsets = 0x03020100;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 8;
    unsigned int xstartmod = (xstart & 1) == 0 ? xstart : xstart - 1;
    unsigned int xsquare = (xstart & 1) == 0 ? 0x2110 : 0x3221;
    unsigned int ystartmod = (ystart & 1) == 0 ? ystart - 2 : ystart - 1;
    unsigned int ysquare = (ystart & 1) == 0 ? 0x2312 : 0x1201;

    retVal.val =
        mul8_sym(xbuff, xstartmod, xoffsets, xstep, xsquare, ybuff, ystartmod, ysquare, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<int16, int16> macSym2buffIntHb(T_accSymIntHb<int16, int16> acc,
                                                    v32int16 xbuff,
                                                    unsigned int xstart,
                                                    v32int16 ybuff,
                                                    unsigned int ystart,
                                                    v16int16 zbuff,
                                                    unsigned int zstart) {
    T_accSymIntHb<int16, int16> retVal;
    const unsigned int xoffsets = 0x03020100;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 8;
    unsigned int xstartmod = (xstart & 1) == 0 ? xstart : xstart - 1;
    unsigned int xsquare = (xstart & 1) == 0 ? 0x2110 : 0x3221;
    unsigned int ystartmod = (ystart & 1) == 0 ? ystart - 2 : ystart - 1;
    unsigned int ysquare = (ystart & 1) == 0 ? 0x2312 : 0x1201;

    retVal.val = mac8_sym(acc.val, xbuff, xstartmod, xoffsets, xstep, xsquare, ybuff, ystartmod, ysquare, zbuff, zstart,
                          zoffsets, zstep);
    return retVal;
}

// DATA = cint16, COEFF = int16
inline T_accSymIntHb<cint16, int16> mulSym2buffIntHb(
    v16cint16 xbuff, unsigned int xstart, v16cint16 ybuff, unsigned int ystart, v16int16 zbuff, unsigned int zstart) {
    T_accSymIntHb<cint16, int16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;

    retVal.val = mul4_sym(xbuff, xstart, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<cint16, int16> macSym2buffIntHb(T_accSymIntHb<cint16, int16> acc,
                                                     v16cint16 xbuff,
                                                     unsigned int xstart,
                                                     v16cint16 ybuff,
                                                     unsigned int ystart,
                                                     v16int16 zbuff,
                                                     unsigned int zstart) {
    T_accSymIntHb<cint16, int16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;

    retVal.val = mac4_sym(acc.val, xbuff, xstart, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = cint16, COEFF = cint16
inline T_accSymIntHb<cint16, cint16> mulSym2buffIntHb(
    v16cint16 xbuff, unsigned int xstart, v16cint16 ybuff, unsigned int ystart, v8cint16 zbuff, unsigned int zstart) {
    T_accSymIntHb<cint16, cint16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;

    retVal.val = mul4_sym(xbuff, xstart, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<cint16, cint16> macSym2buffIntHb(T_accSymIntHb<cint16, cint16> acc,
                                                      v16cint16 xbuff,
                                                      unsigned int xstart,
                                                      v16cint16 ybuff,
                                                      unsigned int ystart,
                                                      v8cint16 zbuff,
                                                      unsigned int zstart) {
    T_accSymIntHb<cint16, cint16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;

    retVal.val = mac4_sym(acc.val, xbuff, xstart, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = int32, COEFF = int16
inline T_accSymIntHb<int32, int16> mulSym2buffIntHb(
    v16int32 xbuff, unsigned int xstart, v16int32 ybuff, unsigned int ystart, v16int16 zbuff, unsigned int zstart) {
    T_accSymIntHb<int32, int16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;

    retVal.val = lmul8_sym(xbuff, xstart, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<int32, int16> macSym2buffIntHb(T_accSymIntHb<int32, int16> acc,
                                                    v16int32 xbuff,
                                                    unsigned int xstart,
                                                    v16int32 ybuff,
                                                    unsigned int ystart,
                                                    v16int16 zbuff,
                                                    unsigned int zstart) {
    T_accSymIntHb<int32, int16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;

    retVal.val = lmac8_sym(acc.val, xbuff, xstart, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = int32,  COEFF = int32>
inline T_accSymIntHb<int32, int32> mulSym2buffIntHb(
    v16int32 xbuff, unsigned int xstart, v16int32 ybuff, unsigned int ystart, v8int32 zbuff, unsigned int zstart) {
    T_accSymIntHb<int32, int32> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;

    retVal.val = lmul4_sym(xbuff, xstart, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    // retVal.uval = lmul4_sym( xbuff, xstart+kLanes, xoffsets, xstep, ybuff, ystart+kLanes, zbuff, zstart, zoffsets,
    // zstep);
    return retVal;
}

inline T_accSymIntHb<int32, int32> macSym2buffIntHb(T_accSymIntHb<int32, int32> acc,
                                                    v16int32 xbuff,
                                                    unsigned int xstart,
                                                    v16int32 ybuff,
                                                    unsigned int ystart,
                                                    v8int32 zbuff,
                                                    unsigned int zstart) {
    T_accSymIntHb<int32, int32> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;

    retVal.val = lmac4_sym(acc.val, xbuff, xstart, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    // retVal.uval = lmac4_sym(acc.uval, xbuff, xstart+kLanes, xoffsets, xstep, ybuff, ystart+kLanes, zbuff, zstart,
    // zoffsets, zstep);
    return retVal;
}

// DATA = cint32,  COEFF = int16>
inline T_accSymIntHb<cint32, int16> mulSym2buffIntHb(
    v8cint32 xbuff, unsigned int xstart, v8cint32 ybuff, unsigned int ystart, v16int16 zbuff, unsigned int zstart) {
    T_accSymIntHb<cint32, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;

    retVal.val = lmul4_sym(xbuff, xstart, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<cint32, int16> macSym2buffIntHb(T_accSymIntHb<cint32, int16> acc,
                                                     v8cint32 xbuff,
                                                     unsigned int xstart,
                                                     v8cint32 ybuff,
                                                     unsigned int ystart,
                                                     v16int16 zbuff,
                                                     unsigned int zstart) {
    T_accSymIntHb<cint32, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;

    retVal.val = lmac4_sym(acc.val, xbuff, xstart, xoffsets, xstep, ybuff, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = cint32,  COEFF = cint16>
inline T_accSymIntHb<cint32, cint16> mulSym2buffIntHb(
    v8cint32 xbuff, unsigned int xstart, v8cint32 ybuff, unsigned int ystart, v8cint16 zbuff, unsigned int zstart) {
    T_accSymIntHb<cint32, cint16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;

    retVal.val = lmul4_sym(xbuff, xstart, xoffsets, ybuff, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<cint32, cint16> macSym2buffIntHb(T_accSymIntHb<cint32, cint16> acc,
                                                      v8cint32 xbuff,
                                                      unsigned int xstart,
                                                      v8cint32 ybuff,
                                                      unsigned int ystart,
                                                      v8cint16 zbuff,
                                                      unsigned int zstart) {
    T_accSymIntHb<cint32, cint16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;

    retVal.val = lmac4_sym(acc.val, xbuff, xstart, xoffsets, ybuff, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cint32,  COEFF = int32>
inline T_accSymIntHb<cint32, int32> mulSym2buffIntHb(
    v8cint32 xbuff, unsigned int xstart, v8cint32 ybuff, unsigned int ystart, v8int32 zbuff, unsigned int zstart) {
    T_accSymIntHb<cint32, int32> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = lmul4_sym(xbuff, xstart, xoffsets, ybuff, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<cint32, int32> macSym2buffIntHb(T_accSymIntHb<cint32, int32> acc,
                                                     v8cint32 xbuff,
                                                     unsigned int xstart,
                                                     v8cint32 ybuff,
                                                     unsigned int ystart,
                                                     v8int32 zbuff,
                                                     unsigned int zstart) {
    T_accSymIntHb<cint32, int32> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = lmac4_sym(acc.val, xbuff, xstart, xoffsets, ybuff, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cint32,  COEFF = cint32>
inline T_accSymIntHb<cint32, cint32> mulSym2buffIntHb(
    v8cint32 xbuff, unsigned int xstart, v8cint32 ybuff, unsigned int ystart, v4cint32 zbuff, unsigned int zstart) {
    T_accSymIntHb<cint32, cint32> retVal;
    const unsigned int xoffsets = 0x10;
    const unsigned int zoffsets = 0x00;
    const unsigned int kLanes = 2;

    retVal.val = lmul2_sym(xbuff, xstart, xoffsets, ybuff, ystart, zbuff, zstart, zoffsets);
    // retVal.uval = lmul2_sym( xbuff, xstart+kLanes, xoffsets, ybuff, ystart+kLanes, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<cint32, cint32> macSym2buffIntHb(T_accSymIntHb<cint32, cint32> acc,
                                                      v8cint32 xbuff,
                                                      unsigned int xstart,
                                                      v8cint32 ybuff,
                                                      unsigned int ystart,
                                                      v4cint32 zbuff,
                                                      unsigned int zstart) {
    T_accSymIntHb<cint32, cint32> retVal;
    const unsigned int xoffsets = 0x10;
    const unsigned int zoffsets = 0x00;
    const unsigned int kLanes = 2;

    retVal.val = lmac2_sym(acc.val, xbuff, xstart, xoffsets, ybuff, ystart, zbuff, zstart, zoffsets);
    // retVal.uval = lmac2_sym(acc.uval, xbuff, xstart+kLanes, xoffsets, ybuff, ystart+kLanes, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = float,  COEFF = float>
inline T_accSymIntHb<float, float> mulSym2buffIntHb(
    v16float xbuff, unsigned int xstart, v16float ybuff, unsigned int ystart, v8float zbuff, unsigned int zstart) {
    T_accSymIntHb<float, float> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int kLanes = 8;

    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, ybuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<float, float> macSym2buffIntHb(T_accSymIntHb<float, float> acc,
                                                    v16float xbuff,
                                                    unsigned int xstart,
                                                    v16float ybuff,
                                                    unsigned int ystart,
                                                    v8float zbuff,
                                                    unsigned int zstart) {
    T_accSymIntHb<float, float> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int kLanes = 8;

    retVal.val = fpmac(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, ybuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cfloat,  COEFF = float>
inline T_accSymIntHb<cfloat, float> mulSym2buffIntHb(
    v8cfloat xbuff, unsigned int xstart, v8cfloat ybuff, unsigned int ystart, v8float zbuff, unsigned int zstart) {
    T_accSymIntHb<cfloat, float> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int kLanes = 4;

    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, ybuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<cfloat, float> macSym2buffIntHb(T_accSymIntHb<cfloat, float> acc,
                                                     v8cfloat xbuff,
                                                     unsigned int xstart,
                                                     v8cfloat ybuff,
                                                     unsigned int ystart,
                                                     v8float zbuff,
                                                     unsigned int zstart) {
    T_accSymIntHb<cfloat, float> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int kLanes = 4;

    retVal.val = fpmac(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, ybuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cfloat,  COEFF = cfloat>
inline T_accSymIntHb<cfloat, cfloat> mulSym2buffIntHb(
    v8cfloat xbuff, unsigned int xstart, v8cfloat ybuff, unsigned int ystart, v4cfloat zbuff, unsigned int zstart) {
    T_accSymIntHb<cfloat, cfloat> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int kLanes = 4;

    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, ybuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<cfloat, cfloat> macSym2buffIntHb(T_accSymIntHb<cfloat, cfloat> acc,
                                                      v8cfloat xbuff,
                                                      unsigned int xstart,
                                                      v8cfloat ybuff,
                                                      unsigned int ystart,
                                                      v4cfloat zbuff,
                                                      unsigned int zstart) {
    T_accSymIntHb<cfloat, cfloat> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int kLanes = 4;

    retVal.val = fpmac(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, ybuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// Centre tap mul for use in halfband
//-----------------------------------------------------------------------------------------------------
inline T_accSymIntHb<int16, int16> mulCentreTap2buffIntHb(v32int16 xbuff, unsigned int xstart, v16int16 zbuff) {
    T_accSymIntHb<int16, int16> retVal;
    const unsigned int xoffsets = 0x03020100;
    const unsigned int xstep = 0;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstart = 0;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 8;
    unsigned int xstartmod = (xstart & 1) == 0 ? xstart : xstart - 1;
    unsigned int xsquare = (xstart & 1) == 0 ? 0x2110 : 0x3221;
    retVal.val = mul8(xbuff, xstartmod, xoffsets, xstep, xsquare, zbuff, zstart, zoffsets, zstep);
    // retVal.uval = mul8(xbuff, xstartmod+kLanes, xoffsets, xstep, xsquare, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<cint16, int16> mulCentreTap2buffIntHb(v16cint16 xbuff, unsigned int xstart, v16int16 zbuff) {
    T_accSymIntHb<cint16, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 0;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstart = 0;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    retVal.val = mul4(xbuff, xstart, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    // retVal.uval = mul4(xbuff, xstart+kLanes, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<cint16, cint16> mulCentreTap2buffIntHb(v16cint16 xbuff, unsigned int xstart, v8cint16 zbuff) {
    T_accSymIntHb<cint16, cint16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 0;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstart = 0;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    retVal.val = mul4(xbuff, xstart, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    // retVal.uval = mul4(xbuff, xstart+kLanes, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<int32, int16> mulCentreTap2buffIntHb(v16int32 xbuff, unsigned int xstart, v16int16 zbuff) {
    T_accSymIntHb<int32, int16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int xstep = 0;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstart = 0;
    const unsigned int zstep = 1;
    retVal.val = lmul8(xbuff, xstart, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<int32, int32> mulCentreTap2buffIntHb(v16int32 xbuff, unsigned int xstart, v8int32 zbuff) {
    T_accSymIntHb<int32, int32> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 0;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstart = 0;
    const unsigned int zstep = 1;
    const unsigned int kLanes = 4;
    retVal.val = lmul4(xbuff, xstart, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    // retVal.uval = lmul4(xbuff, xstart+kLanes, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<cint32, int16> mulCentreTap2buffIntHb(v8cint32 xbuff, unsigned int xstart, v16int16 zbuff) {
    T_accSymIntHb<cint32, int16> retVal;
    retVal.val = null_v4cacc80();
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 0;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstart = 0;
    const unsigned int zstep = 1;
    retVal.val = lmac4(retVal.val, xbuff, xstart, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<cint32, cint16> mulCentreTap2buffIntHb(v8cint32 xbuff, unsigned int xstart, v8cint16 zbuff) {
    T_accSymIntHb<cint32, cint16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 0;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstart = 0;
    const unsigned int zstep = 1;
    retVal.val = lmul4(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<cint32, int32> mulCentreTap2buffIntHb(v8cint32 xbuff, unsigned int xstart, v8int32 zbuff) {
    T_accSymIntHb<cint32, int32> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstart = 0;
    retVal.val = lmul4(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<cint32, cint32> mulCentreTap2buffIntHb(v8cint32 xbuff, unsigned int xstart, v4cint32 zbuff) {
    T_accSymIntHb<cint32, cint32> retVal;
    const unsigned int xoffsets = 0x10;
    const unsigned int zoffsets = 0x00;
    const unsigned int zstart = 0;
    retVal.val = lmul2(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    // retVal.uval = lmul2(xbuff, xstart+2, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<float, float> mulCentreTap2buffIntHb(v16float xbuff, unsigned int xstart, v8float zbuff) {
    T_accSymIntHb<float, float> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstart = 0;
    const unsigned int kLanes = 8;

    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<cfloat, float> mulCentreTap2buffIntHb(v8cfloat xbuff, unsigned int xstart, v8float zbuff) {
    T_accSymIntHb<cfloat, float> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstart = 0;
    const unsigned int kLanes = 4;

    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<cfloat, cfloat> mulCentreTap2buffIntHb(v8cfloat xbuff, unsigned int xstart, v4cfloat zbuff) {
    T_accSymIntHb<cfloat, cfloat> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstart = 0;
    const unsigned int kLanes = 4;

    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// Halfband Output. This function takes the two polyphase lane sets, interleaves them and outputs them
// however for some types, the lane vector is too large for the interleave intrinsic, so they have to be split and for
// the splices to be interleaved.
// int16/int16
inline void writeOutputIntHb(output_window<int16>* outWindow,
                             const T_accSymIntHb<int16, int16> accLow,
                             const T_accSymIntHb<int16, int16> accHigh,
                             const int shift) {
    T_outValIntHb<int16, int16> outVal;
    T_accIntHb<int16, int16> acc;
    acc.val = concat(accLow.val, accHigh.val);
    outVal.val = srs_ilv(acc.val, shift);
    window_writeincr(outWindow, outVal.val);
}

// cint16/int16
inline void writeOutputIntHb(output_window<cint16>* outWindow,
                             const T_accSymIntHb<cint16, int16> accLow,
                             const T_accSymIntHb<cint16, int16> accHigh,
                             const int shift) {
    T_outValIntHb<cint16, int16> outVal;
    T_accIntHb<cint16, int16> acc;
    acc.val = concat(accLow.val, accHigh.val);
    outVal.val = srs_ilv(acc.val, shift);
    window_writeincr(outWindow, outVal.val);
}

// cint16/cint16
inline void writeOutputIntHb(output_window<cint16>* outWindow,
                             const T_accSymIntHb<cint16, cint16> accLow,
                             const T_accSymIntHb<cint16, cint16> accHigh,
                             const int shift) {
    T_outValIntHb<cint16, cint16> outVal;
    T_accIntHb<cint16, cint16> acc;
    acc.val = concat(accLow.val, accHigh.val);
    outVal.val = srs_ilv(acc.val, shift);
    window_writeincr(outWindow, outVal.val);
}

// int32/int16
inline void writeOutputIntHb(output_window<int32>* outWindow,
                             const T_accSymIntHb<int32, int16> accLow,
                             const T_accSymIntHb<int32, int16> accHigh,
                             const int shift) {
    T_outValIntHb<int32, int16> outVal;
    T_accIntHb<int32, int16> acc;
    acc.val = concat(ext_lo(accLow.val), ext_lo(accHigh.val));
    outVal.val = srs_ilv(acc.val, shift);
    window_writeincr(outWindow, outVal.val);
    acc.val = concat(ext_hi(accLow.val), ext_hi(accHigh.val));
    outVal.val = srs_ilv(acc.val, shift);
    window_writeincr(outWindow, outVal.val);
}

// int32/int32
inline void writeOutputIntHb(output_window<int32>* outWindow,
                             const T_accSymIntHb<int32, int32> accLow,
                             const T_accSymIntHb<int32, int32> accHigh,
                             const int shift) {
    T_outValIntHb<int32, int32> outVal;
    T_accIntHb<int32, int32> acc;
    acc.val = concat(accLow.val, accHigh.val);
    outVal.val = srs_ilv(acc.val, shift);
    window_writeincr(outWindow, outVal.val);
}

// cint32/int16
inline void writeOutputIntHb(output_window<cint32>* outWindow,
                             const T_accSymIntHb<cint32, int16> accLow,
                             const T_accSymIntHb<cint32, int16> accHigh,
                             const int shift) {
    T_outValIntHb<cint32, int16> outVal;
    T_accIntHb<cint32, int16> acc;
    acc.val = concat(ext_lo(accLow.val), ext_lo(accHigh.val));
    outVal.val = srs_ilv(acc.val, shift);
    window_writeincr(outWindow, outVal.val);
    acc.val = concat(ext_hi(accLow.val), ext_hi(accHigh.val));
    outVal.val = srs_ilv(acc.val, shift);
    window_writeincr(outWindow, outVal.val);
}

// cint32/cint16
inline void writeOutputIntHb(output_window<cint32>* outWindow,
                             const T_accSymIntHb<cint32, cint16> accLow,
                             const T_accSymIntHb<cint32, cint16> accHigh,
                             const int shift) {
    T_outValIntHb<cint32, cint16> outVal;
    T_accIntHb<cint32, cint16> acc;
    acc.val = concat(ext_lo(accLow.val), ext_lo(accHigh.val));
    outVal.val = srs_ilv(acc.val, shift);
    window_writeincr(outWindow, outVal.val);
    acc.val = concat(ext_hi(accLow.val), ext_hi(accHigh.val));
    outVal.val = srs_ilv(acc.val, shift);
    window_writeincr(outWindow, outVal.val);
}

// cint32/int32
inline void writeOutputIntHb(output_window<cint32>* outWindow,
                             const T_accSymIntHb<cint32, int32> accLow,
                             const T_accSymIntHb<cint32, int32> accHigh,
                             const int shift) {
    T_outValIntHb<cint32, int32> outVal;
    T_accIntHb<cint32, int32> acc;
    acc.val = concat(ext_lo(accLow.val), ext_lo(accHigh.val));
    outVal.val = srs_ilv(acc.val, shift);
    window_writeincr(outWindow, outVal.val);
    acc.val = concat(ext_hi(accLow.val), ext_hi(accHigh.val));
    outVal.val = srs_ilv(acc.val, shift);
    window_writeincr(outWindow, outVal.val);
}

// cint32/cint32
inline void writeOutputIntHb(output_window<cint32>* outWindow,
                             const T_accSymIntHb<cint32, cint32> accLow,
                             const T_accSymIntHb<cint32, cint32> accHigh,
                             const int shift) {
    // T_outValIntHb<cint32, cint32> outVal;
    // v4cacc80 acc;
    // acc = concat(accLow.val, accHigh.val);
    // outVal.val = srs_ilv(acc, shift);
    // window_writeincr(outWindow, outVal.val);
    // acc = concat(accLow.uval, accHigh.uval);
    // outVal.val = srs_ilv(acc, shift);
    // window_writeincr(outWindow, outVal.val);

    // 2 lane low phase + 2 lane high phase
    window_writeincr(outWindow, srs_ilv(concat((accLow.val), (accHigh.val)), shift));
}

// float/float
inline void writeOutputIntHb(output_window<float>* outWindow,
                             const T_accSymIntHb<float, float> accLow,
                             const T_accSymIntHb<float, float> accHigh,
                             const int shift) {
    T_outValIntHb<float, float> outVal;
    v8acc80 acc1, acc2, concatacc;
    v4acc80 acc3, acc4;
    acc1 = lups(as_v8int32(accLow.val), 0);
    acc2 = lups(as_v8int32(accHigh.val), 0);
    acc3 = ext_lo(acc1);
    acc4 = ext_lo(acc2);
    concatacc = concat(acc3, acc4);
    outVal.val = as_v8float(srs_ilv(concatacc, 0));
    window_writeincr(outWindow, outVal.val);
    acc3 = ext_hi(acc1);
    acc4 = ext_hi(acc2);
    concatacc = concat(acc3, acc4);
    outVal.val = as_v8float(srs_ilv(concatacc, 0));
    window_writeincr(outWindow, outVal.val);
}

// cfloat/float
inline void writeOutputIntHb(output_window<cfloat>* outWindow,
                             const T_accSymIntHb<cfloat, float> accLow,
                             const T_accSymIntHb<cfloat, float> accHigh,
                             const int shift) {
    T_outValIntHb<cfloat, float> outVal;
    v4cacc80 acc1, acc2, concatacc;
    v2cacc80 acc3, acc4;
    acc1 = lups(as_v4cint32(accLow.val), 0);
    acc2 = lups(as_v4cint32(accHigh.val), 0);
    acc3 = ext_lo(acc1);
    acc4 = ext_lo(acc2);
    concatacc = concat(acc3, acc4);
    outVal.val = as_v4cfloat(srs_ilv(concatacc, 0));
    window_writeincr(outWindow, outVal.val);
    acc3 = ext_hi(acc1);
    acc4 = ext_hi(acc2);
    concatacc = concat(acc3, acc4);
    outVal.val = as_v4cfloat(srs_ilv(concatacc, 0));
    window_writeincr(outWindow, outVal.val);
}

// cfloat/cfloat
inline void writeOutputIntHb(output_window<cfloat>* outWindow,
                             const T_accSymIntHb<cfloat, cfloat> accLow,
                             const T_accSymIntHb<cfloat, cfloat> accHigh,
                             const int shift) {
    T_outValIntHb<cfloat, cfloat> outVal;
    v4cacc80 acc1, acc2, concatacc;
    v2cacc80 acc3, acc4;
    acc1 = lups(as_v4cint32(accLow.val), 0);
    acc2 = lups(as_v4cint32(accHigh.val), 0);
    acc3 = ext_lo(acc1);
    acc4 = ext_lo(acc2);
    concatacc = concat(acc3, acc4);
    outVal.val = as_v4cfloat(srs_ilv(concatacc, 0));
    window_writeincr(outWindow, outVal.val);
    acc3 = ext_hi(acc1);
    acc4 = ext_hi(acc2);
    concatacc = concat(acc3, acc4);
    outVal.val = as_v4cfloat(srs_ilv(concatacc, 0));
    window_writeincr(outWindow, outVal.val);
}

// Symmetric mul/mac for use in 1buff
//-----------------------------------------------------------------------------------------------------
// DATA = int16, COEFF = int16
inline T_accSymIntHb<int16, int16> mulSym1buffIntHb(
    v64int16 xbuff, unsigned int xstart, unsigned int ystart, v16int16 zbuff, unsigned int zstart) {
    T_accSymIntHb<int16, int16> retVal;
    const unsigned int xoffsets = 0x03020100;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    unsigned int xstartmod = (xstart & 1) == 0 ? xstart : xstart - 1;
    unsigned int xsquare = (xstart & 1) == 0 ? 0x2110 : 0x3221;
    unsigned int ystartmod = (ystart & 1) == 0 ? ystart - 2 : ystart - 1;
    unsigned int ysquare = (ystart & 1) == 0 ? 0x2312 : 0x1201;

    retVal.val =
        mul8_sym(xbuff, xstartmod, xoffsets, xstep, xsquare, ystartmod, ysquare, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<int16, int16> macSym1buffIntHb(T_accSymIntHb<int16, int16> acc,
                                                    v64int16 xbuff,
                                                    unsigned int xstart,
                                                    unsigned int ystart,
                                                    v16int16 zbuff,
                                                    unsigned int zstart) {
    T_accSymIntHb<int16, int16> retVal;
    const unsigned int xoffsets = 0x03020100;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;
    unsigned int xstartmod = (xstart & 1) == 0 ? xstart : xstart - 1;
    unsigned int xsquare = (xstart & 1) == 0 ? 0x2110 : 0x3221;
    unsigned int ystartmod = (ystart & 1) == 0 ? ystart - 2 : ystart - 1;
    unsigned int ysquare = (ystart & 1) == 0 ? 0x2312 : 0x1201;

    retVal.val = mac8_sym(acc.val, xbuff, xstartmod, xoffsets, xstep, xsquare, ystartmod, ysquare, zbuff, zstart,
                          zoffsets, zstep);
    return retVal;
}

// DATA = cint16, COEFF = int16
inline T_accSymIntHb<cint16, int16> mulSym1buffIntHb(
    v32cint16 xbuff, unsigned int xstart, unsigned int ystart, v16int16 zbuff, unsigned int zstart) {
    T_accSymIntHb<cint16, int16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;

    retVal.val = mul4_sym(xbuff, xstart, xoffsets, xstep, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<cint16, int16> macSym1buffIntHb(T_accSymIntHb<cint16, int16> acc,
                                                     v32cint16 xbuff,
                                                     unsigned int xstart,
                                                     unsigned int ystart,
                                                     v16int16 zbuff,
                                                     unsigned int zstart) {
    T_accSymIntHb<cint16, int16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;

    retVal.val = mac4_sym(acc.val, xbuff, xstart, xoffsets, xstep, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = cint16, COEFF = cint16
inline T_accSymIntHb<cint16, cint16> mulSym1buffIntHb(
    v32cint16 xbuff, unsigned int xstart, unsigned int ystart, v8cint16 zbuff, unsigned int zstart) {
    T_accSymIntHb<cint16, cint16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;

    retVal.val = mul4_sym(xbuff, xstart, xoffsets, xstep, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<cint16, cint16> macSym1buffIntHb(T_accSymIntHb<cint16, cint16> acc,
                                                      v32cint16 xbuff,
                                                      unsigned int xstart,
                                                      unsigned int ystart,
                                                      v8cint16 zbuff,
                                                      unsigned int zstart) {
    T_accSymIntHb<cint16, cint16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;

    retVal.val = mac4_sym(acc.val, xbuff, xstart, xoffsets, xstep, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = int32, COEFF = int16
inline T_accSymIntHb<int32, int16> mulSym1buffIntHb(
    v32int32 xbuff, unsigned int xstart, unsigned int ystart, v16int16 zbuff, unsigned int zstart) {
    T_accSymIntHb<int32, int16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;

    retVal.val = lmul8_sym(xbuff, xstart, xoffsets, xstep, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<int32, int16> macSym1buffIntHb(T_accSymIntHb<int32, int16> acc,
                                                    v32int32 xbuff,
                                                    unsigned int xstart,
                                                    unsigned int ystart,
                                                    v16int16 zbuff,
                                                    unsigned int zstart) {
    T_accSymIntHb<int32, int16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstep = 1;

    retVal.val = lmac8_sym(acc.val, xbuff, xstart, xoffsets, xstep, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = int32,  COEFF = int32>
inline T_accSymIntHb<int32, int32> mulSym1buffIntHb(
    v32int32 xbuff, unsigned int xstart, unsigned int ystart, v8int32 zbuff, unsigned int zstart) {
    T_accSymIntHb<int32, int32> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;

    retVal.val = lmul4_sym(xbuff, xstart, xoffsets, xstep, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<int32, int32> macSym1buffIntHb(T_accSymIntHb<int32, int32> acc,
                                                    v32int32 xbuff,
                                                    unsigned int xstart,
                                                    unsigned int ystart,
                                                    v8int32 zbuff,
                                                    unsigned int zstart) {
    T_accSymIntHb<int32, int32> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;

    retVal.val = lmac4_sym(acc.val, xbuff, xstart, xoffsets, xstep, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = cint32,  COEFF = int16>
inline T_accSymIntHb<cint32, int16> mulSym1buffIntHb(
    v16cint32 xbuff, unsigned int xstart, unsigned int ystart, v16int16 zbuff, unsigned int zstart) {
    T_accSymIntHb<cint32, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;

    retVal.val = lmul4_sym(xbuff, xstart, xoffsets, xstep, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<cint32, int16> macSym1buffIntHb(T_accSymIntHb<cint32, int16> acc,
                                                     v16cint32 xbuff,
                                                     unsigned int xstart,
                                                     unsigned int ystart,
                                                     v16int16 zbuff,
                                                     unsigned int zstart) {
    T_accSymIntHb<cint32, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;

    retVal.val = lmac4_sym(acc.val, xbuff, xstart, xoffsets, xstep, ystart, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = cint32,  COEFF = cint16>
inline T_accSymIntHb<cint32, cint16> mulSym1buffIntHb(
    v16cint32 xbuff, unsigned int xstart, unsigned int ystart, v8cint16 zbuff, unsigned int zstart) {
    T_accSymIntHb<cint32, cint16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;

    retVal.val = lmul4_sym(xbuff, xstart, xoffsets, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<cint32, cint16> macSym1buffIntHb(T_accSymIntHb<cint32, cint16> acc,
                                                      v16cint32 xbuff,
                                                      unsigned int xstart,
                                                      unsigned int ystart,
                                                      v8cint16 zbuff,
                                                      unsigned int zstart) {
    T_accSymIntHb<cint32, cint16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstep = 1;

    retVal.val = lmac4_sym(acc.val, xbuff, xstart, xoffsets, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cint32,  COEFF = int32>
inline T_accSymIntHb<cint32, int32> mulSym1buffIntHb(
    v16cint32 xbuff, unsigned int xstart, unsigned int ystart, v8int32 zbuff, unsigned int zstart) {
    T_accSymIntHb<cint32, int32> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = lmul4_sym(xbuff, xstart, xoffsets, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<cint32, int32> macSym1buffIntHb(T_accSymIntHb<cint32, int32> acc,
                                                     v16cint32 xbuff,
                                                     unsigned int xstart,
                                                     unsigned int ystart,
                                                     v8int32 zbuff,
                                                     unsigned int zstart) {
    T_accSymIntHb<cint32, int32> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = lmac4_sym(acc.val, xbuff, xstart, xoffsets, ystart, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cint32,  COEFF = cint32>
inline T_accSymIntHb<cint32, cint32> mulSym1buffIntHb(
    v16cint32 xbuff, unsigned int xstart, unsigned int ystart, v4cint32 zbuff, unsigned int zstart) {
    T_accSymIntHb<cint32, cint32> retVal;
    const unsigned int xoffsets = 0x10;
    const unsigned int zoffsets = 0x00;

    retVal.val = lmul2_sym(xbuff, xstart, xoffsets, ystart, zbuff, zstart, zoffsets);
    // retVal.uval = lmul2_sym( xbuff, xstart+2, xoffsets, ystart+2, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<cint32, cint32> macSym1buffIntHb(T_accSymIntHb<cint32, cint32> acc,
                                                      v16cint32 xbuff,
                                                      unsigned int xstart,
                                                      unsigned int ystart,
                                                      v4cint32 zbuff,
                                                      unsigned int zstart) {
    T_accSymIntHb<cint32, cint32> retVal;
    const unsigned int xoffsets = 0x10;
    const unsigned int zoffsets = 0x00;

    retVal.val = lmac2_sym(acc.val, xbuff, xstart, xoffsets, ystart, zbuff, zstart, zoffsets);
    // retVal.uval = lmac2_sym(acc.uval, xbuff, xstart+2, xoffsets, ystart+2, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = float,  COEFF = float>
inline T_accSymIntHb<float, float> mulSym1buffIntHb(
    v32float xbuff, unsigned int xstart, unsigned int ystart, v8float zbuff, unsigned int zstart) {
    T_accSymIntHb<float, float> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;

    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, xbuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<float, float> macSym1buffIntHb(T_accSymIntHb<float, float> acc,
                                                    v32float xbuff,
                                                    unsigned int xstart,
                                                    unsigned int ystart,
                                                    v8float zbuff,
                                                    unsigned int zstart) {
    T_accSymIntHb<float, float> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;

    retVal.val = fpmac(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, xbuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cfloat,  COEFF = float>
inline T_accSymIntHb<cfloat, float> mulSym1buffIntHb(
    v16cfloat xbuff, unsigned int xstart, unsigned int ystart, v8float zbuff, unsigned int zstart) {
    T_accSymIntHb<cfloat, float> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, xbuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<cfloat, float> macSym1buffIntHb(T_accSymIntHb<cfloat, float> acc,
                                                     v16cfloat xbuff,
                                                     unsigned int xstart,
                                                     unsigned int ystart,
                                                     v8float zbuff,
                                                     unsigned int zstart) {
    T_accSymIntHb<cfloat, float> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = fpmac(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, xbuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cfloat,  COEFF = cfloat>
inline T_accSymIntHb<cfloat, cfloat> mulSym1buffIntHb(
    v16cfloat xbuff, unsigned int xstart, unsigned int ystart, v4cfloat zbuff, unsigned int zstart) {
    T_accSymIntHb<cfloat, cfloat> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, xbuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<cfloat, cfloat> macSym1buffIntHb(T_accSymIntHb<cfloat, cfloat> acc,
                                                      v16cfloat xbuff,
                                                      unsigned int xstart,
                                                      unsigned int ystart,
                                                      v4cfloat zbuff,
                                                      unsigned int zstart) {
    T_accSymIntHb<cfloat, cfloat> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;

    retVal.val = fpmac(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.val = fpmac(retVal.val, xbuff, ystart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// Centre tap mul for 1buff architecture using 1024b buffer
//-----------------------------------------------------------------------------------------------------
inline T_accSymIntHb<int16, int16> mulCentreTap1buffIntHb(v64int16 xbuff, unsigned int xstart, v16int16 zbuff) {
    T_accSymIntHb<int16, int16> retVal;
    const unsigned int xoffsets = 0x03020100;
    const unsigned int xstep = 0;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstart = 0;
    const unsigned int zstep = 1;
    unsigned int xstartmod = (xstart & 1) == 0 ? xstart : xstart - 1;
    unsigned int xsquare = (xstart & 1) == 0 ? 0x2110 : 0x3221;
    retVal.val = mul8(xbuff, xstartmod, xoffsets, xstep, xsquare, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<cint16, int16> mulCentreTap1buffIntHb(v32cint16 xbuff, unsigned int xstart, v16int16 zbuff) {
    T_accSymIntHb<cint16, int16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 0;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstart = 0;
    const unsigned int zstep = 1;
    retVal.val = mul4(xbuff, xstart, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<cint16, cint16> mulCentreTap1buffIntHb(v32cint16 xbuff, unsigned int xstart, v8cint16 zbuff) {
    T_accSymIntHb<cint16, cint16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 0;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstart = 0;
    const unsigned int zstep = 1;
    retVal.val = mul4(xbuff, xstart, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<int32, int16> mulCentreTap1buffIntHb(v32int32 xbuff, unsigned int xstart, v16int16 zbuff) {
    T_accSymIntHb<int32, int16> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int xstep = 0;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstart = 0;
    const unsigned int zstep = 1;
    retVal.val = lmul8(xbuff, xstart, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<int32, int32> mulCentreTap1buffIntHb(v32int32 xbuff, unsigned int xstart, v8int32 zbuff) {
    T_accSymIntHb<int32, int32> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 0;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstart = 0;
    const unsigned int zstep = 1;
    retVal.val = lmul4(xbuff, xstart, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<cint32, int16> mulCentreTap1buffIntHb(v16cint32 xbuff, unsigned int xstart, v16int16 zbuff) {
    T_accSymIntHb<cint32, int16> retVal;
    retVal.val = null_v4cacc80();
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 0;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstart = 0;
    const unsigned int zstep = 1;
    retVal.val = lmac4(retVal.val, xbuff, xstart, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accSymIntHb<cint32, cint16> mulCentreTap1buffIntHb(v16cint32 xbuff, unsigned int xstart, v8cint16 zbuff) {
    T_accSymIntHb<cint32, cint16> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int xstep = 0;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstart = 0;
    const unsigned int zstep = 1;
    retVal.val = lmul4(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<cint32, int32> mulCentreTap1buffIntHb(v16cint32 xbuff, unsigned int xstart, v8int32 zbuff) {
    T_accSymIntHb<cint32, int32> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstart = 0;
    retVal.val = lmul4(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<cint32, cint32> mulCentreTap1buffIntHb(v16cint32 xbuff, unsigned int xstart, v4cint32 zbuff) {
    T_accSymIntHb<cint32, cint32> retVal;
    const unsigned int xoffsets = 0x10;
    const unsigned int zoffsets = 0x00;
    const unsigned int zstart = 0;
    retVal.val = lmul2(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    // retVal.uval = lmul2(xbuff, xstart+2, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<float, float> mulCentreTap1buffIntHb(v32float xbuff, unsigned int xstart, v8float zbuff) {
    T_accSymIntHb<float, float> retVal;
    const unsigned int xoffsets = 0x76543210;
    const unsigned int zoffsets = 0x00000000;
    const unsigned int zstart = 0;
    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<cfloat, float> mulCentreTap1buffIntHb(v16cfloat xbuff, unsigned int xstart, v8float zbuff) {
    T_accSymIntHb<cfloat, float> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstart = 0;
    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accSymIntHb<cfloat, cfloat> mulCentreTap1buffIntHb(v16cfloat xbuff, unsigned int xstart, v4cfloat zbuff) {
    T_accSymIntHb<cfloat, cfloat> retVal;
    const unsigned int xoffsets = 0x3210;
    const unsigned int zoffsets = 0x0000;
    const unsigned int zstart = 0;
    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

//////////////////////////////////////////////////////////////// API

// // Specialised type for final accumulator. Concat of two polyphases
// template<typename T_D, typename T_C> struct T_accIntHb  : T_acc<T_D, T_C>
//  {using T_acc<T_D, T_C>  ::operator=;};

// // Smaller type for each polyphase
// template<typename T_D, typename T_C> struct T_accSymIntHb  : T_acc<T_D, T_C>
// {using T_acc<T_D, T_C>  ::operator=;};
// // Final output value type after shift and rounding
// template<typename T_D, typename T_C> struct T_outValIntHb  : T_outVal<T_D, T_C>
// {using T_outVal<T_D, T_C>  ::operator=;};

// template<typename TT_DATA, typename TT_COEFF> inline constexpr unsigned int fnAccSizeIntHb()
// {
//   return fnAccSize<TT_DATA, TT_COEFF>();
// };

// Halfband Output. This function takes the two polyphase lane sets, interleaves them and outputs them
// however for some types, the lane vector is too large for the interleave intrinsic, so they have to be split and for
// the splices to be interleaved.
template <typename TT_DATA, typename TT_COEFF, unsigned int TP_UPSHIFT_CT>
inline void writeOutputIntHb(output_window<TT_DATA>* outWindow,
                             const T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> accHigh,
                             const int shift) {
    using acc_type = typename T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT>::v_type;
    acc_type chess_storage(bm0) tmp;
    tmp = accHigh.val;
    T_outValIntHb<TT_DATA, TT_COEFF> outVal;
    outVal.val = tmp.template to_vector_zip<TT_DATA>(shift);
    window_writeincr(outWindow, outVal.val);
}

template <typename TT_DATA, typename TT_COEFF, unsigned int TP_FIR_LEN>
inline int upshiftPos() {
    int retVal;
    retVal = TP_FIR_LEN;
    return retVal;
}

// Overloaded function to write to window output.
template <typename TT_DATA, typename TT_COEFF, unsigned int TP_NUM_OUTPUTS, unsigned int TP_UPSHIFT_CT = 0>
inline void writeWindow(T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface,
                        T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> accHP,
                        T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> accLP,
                        unsigned int shift) {
    // Do nothing
}
template <typename TT_DATA, typename TT_COEFF, unsigned int TP_NUM_OUTPUTS, unsigned int TP_UPSHIFT_CT = 0>
inline void writeWindow(T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface,
                        T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> accHP,
                        T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> accLP,
                        unsigned int shift) {
    if
        constexpr(TP_UPSHIFT_CT == 0) {
            writeOutputIntHb(outInterface.outWindow, accHP, accLP, shift);
            if
                constexpr(TP_NUM_OUTPUTS == 2) { writeOutputIntHb(outInterface.outWindow2, accHP, accLP, shift); }
        }
    else {
        writeOutputIntHb(outInterface.outWindow, accHP, shift);
        if
            constexpr(TP_NUM_OUTPUTS == 2) { writeOutputIntHb(outInterface.outWindow2, accHP, shift); }
    }
}

// template for mulSlidingSym1buffIntHb - uses ::aie::api HLI
template <typename TT_DATA, typename TT_COEFF, unsigned int TP_FIR_LEN, unsigned int TP_UPSHIFT_CT>
inline T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> mulSlidingSym1buffIntHb(
    T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> acc,
    T_buff_1024b<TT_DATA> xbuff,
    unsigned int xstart, // no ystart - API calculates ystart based on Points size
    T_buff_256b<TT_COEFF> zbuff,
    unsigned int zstart,
    unsigned int cfShift) {
    constexpr unsigned int Lanes = 2 * fnNumSymLanesIntHb<TT_DATA, TT_COEFF>();
    constexpr unsigned int Points = (TP_FIR_LEN + 1) / 2;
    constexpr unsigned int CoeffStep = 1;
    constexpr unsigned int DataStep = 1;
    using acc_type = typename T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT>::v_type;
    acc_type chess_storage(bm0) tmp;

    // #define _DSPLIB_FIR_INT_HB_UTILS_DEBUG_

    T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> retVal;
    tmp = ::aie::sliding_mul_sym_uct<Lanes, Points, CoeffStep, DataStep>(zbuff.val, zstart, xbuff.val, xstart, cfShift);
    retVal.val = tmp;
    return retVal;
}

// template for macSlidingSym1buffIntHb - uses ::aie::api HLI
template <typename TT_DATA, typename TT_COEFF, unsigned int TP_FIR_LEN, unsigned int TP_UPSHIFT_CT>
inline T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> macSlidingSym1buffIntHb(
    T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> acc,
    T_buff_1024b<TT_DATA> xbuff,
    unsigned int xstart, // no ystart - API calculates ystart based on Points size
    T_buff_256b<TT_COEFF> zbuff,
    unsigned int zstart,
    unsigned int cfShift) {
    constexpr unsigned int Lanes = 2 * fnNumSymLanesIntHb<TT_DATA, TT_COEFF>();
    constexpr unsigned int Points = (TP_FIR_LEN + 1) / 2;
    constexpr unsigned int CoeffStep = 1;
    constexpr unsigned int DataStep = 1;
    using acc_type = typename T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT>::v_type;
    acc_type chess_storage(bm0) tmp;

    T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> retVal;
    tmp = ::aie::sliding_mac_sym_uct<Lanes, Points, CoeffStep, DataStep>(acc.val, zbuff.val, zstart, xbuff.val, xstart,
                                                                         cfShift);
    retVal.val = tmp;
    return retVal;
}

// template for mulSlidingSym2buffIntHb - uses ::aie::api HLI
template <typename TT_DATA, typename TT_COEFF, unsigned int TP_FIR_LEN, unsigned int TP_UPSHIFT_CT>
inline T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> mulSlidingSym2buffIntHb(
    T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> acc,
    T_buff_512b<TT_DATA> xbuff,
    unsigned int xstart,
    T_buff_512b<TT_DATA> ybuff,
    unsigned int ystart,
    T_buff_256b<TT_COEFF> zbuff,
    unsigned int zstart,
    unsigned int cfShift) {
    constexpr unsigned int Lanes = 2 * fnNumSymLanesIntHb<TT_DATA, TT_COEFF>();
    // adjust Point size to where the center tap is in relation to number of columns in low-level intrinsic
    constexpr unsigned int Points = 2 * (((TP_FIR_LEN + 1) / 4 - 1) % fnNumSymColsIntHb<TT_DATA, TT_COEFF>() + 1);
    constexpr unsigned int CoeffStep = 1;
    constexpr unsigned int DataStep = 1;
    using acc_type = typename T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT>::v_type;
    acc_type chess_storage(bm0) tmp;

    T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> retVal;
    tmp = ::aie::sliding_mul_sym_uct<Lanes, Points, CoeffStep, DataStep>(zbuff.val, zstart, xbuff.val, xstart,
                                                                         ybuff.val, ystart, cfShift);
    retVal.val = tmp;
    return retVal;
}

// template for macSlidingSym2buffIntHb - uses ::aie::api HLI
template <typename TT_DATA, typename TT_COEFF, unsigned int TP_FIR_LEN, unsigned int TP_UPSHIFT_CT>
inline T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> macSlidingSym2buffIntHb(
    T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> acc,
    T_buff_512b<TT_DATA> xbuff,
    unsigned int xstart,
    T_buff_512b<TT_DATA> ybuff,
    unsigned int ystart,
    T_buff_256b<TT_COEFF> zbuff,
    unsigned int zstart,
    unsigned int cfShift) {
    constexpr unsigned int Lanes = 2 * fnNumSymLanesIntHb<TT_DATA, TT_COEFF>();
    // adjust Point size to where the center tap is in relation to number of columns in low-level intrinsic
    constexpr unsigned int Points = 2 * (((TP_FIR_LEN + 1) / 4 - 1) % fnNumSymColsIntHb<TT_DATA, TT_COEFF>() + 1);
    constexpr unsigned int CoeffStep = 1;
    constexpr unsigned int DataStep = 1;
    using acc_type = typename T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT>::v_type;
    acc_type chess_storage(bm0) tmp;

    T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> retVal;
    tmp = ::aie::sliding_mac_sym_uct<Lanes, Points, CoeffStep, DataStep>(acc.val, zbuff.val, zstart, xbuff.val, xstart,
                                                                         ybuff.val, ystart, cfShift);
    retVal.val = tmp;
    return retVal;
}

// template for macSlidingSym2buffIntHb - uses ::aie::api HLI
template <typename TT_DATA, typename TT_COEFF, unsigned int TP_FIR_LEN, unsigned int TP_UPSHIFT_CT>
inline T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> macSlidingSym2buffIntHb(
    T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> acc,
    T_buff_512b<TT_DATA> xbuff,
    unsigned int xstart,
    T_buff_512b<TT_DATA> ybuff,
    unsigned int ystart,
    T_buff_256b<TT_COEFF> zbuff,
    unsigned int zstart) {
    constexpr unsigned int Lanes = fnNumSymLanesIntHb<TT_DATA, TT_COEFF>();
    // adjust Point size to where the center tap is in relation to number of columns in low-level intrinsic
    constexpr unsigned int Points = 2 * (((TP_FIR_LEN + 1) / 4 - 1) % fnNumSymColsIntHb<TT_DATA, TT_COEFF>() + 1);
    constexpr unsigned int CoeffStep = 1;
    constexpr unsigned int DataStep = 1;
    using acc_type = typename T_accSymIntHb<TT_DATA, TT_COEFF, 0>::v_type;
    // acc_type chess_storage(aml0) tmp;  // lower part of bm0, used in UCT
    acc_type tmp; // lower part of bm0, used in UCT

    T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> retVal = acc;
    tmp = ::aie::sliding_mac_sym<Lanes, Points, CoeffStep, DataStep>(acc.val.template extract<Lanes>(0), zbuff.val,
                                                                     zstart, xbuff.val, xstart, ybuff.val, ystart);
    retVal.val = retVal.val.insert(0, tmp);
    return retVal;
}

// MAC operation for 1buff arch. Template function which also hides the struct contents.
template <typename TT_DATA, typename TT_COEFF, unsigned int TP_UPSHIFT_CT = 0>
inline T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> macSym1buffIntHb(
    T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> acc,
    T_buff_1024b<TT_DATA> xbuff,
    unsigned int xstart,
    unsigned int ystart,
    T_buff_256b<TT_COEFF> zbuff,
    unsigned int zstart) {
    if
        constexpr(TP_UPSHIFT_CT == 0) {
            // Call overloaded function which uses native vectors
            return macSym1buffIntHb(acc, xbuff.val, xstart, ystart, zbuff.val, zstart);
        }
    else {
        // Do nothing, API _uct call already instantiated all the required calls.
        return acc;
    }
};

// MAC operation for 2buff arch. Template function which also hides the struct contents.
template <typename TT_DATA, typename TT_COEFF, unsigned int TP_FIR_LEN, unsigned int TP_UPSHIFT_CT>
inline T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> macSym2buffIntHb(
    T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> acc,
    T_buff_512b<TT_DATA> xbuff,
    unsigned int xstart,
    T_buff_512b<TT_DATA> ybuff,
    unsigned int ystart,
    T_buff_256b<TT_COEFF> zbuff,
    unsigned int zstart) {
    if
        constexpr(TP_UPSHIFT_CT == 0) {
            // Call overloaded low level function which uses native vectors
            return macSym2buffIntHb(acc, xbuff.val, xstart, ybuff.val, ystart, zbuff.val, zstart);
        }
    else {
        // Call API _uct .
        return macSlidingSym2buffIntHb<TT_DATA, TT_COEFF, TP_FIR_LEN, TP_UPSHIFT_CT>(acc, xbuff, xstart, ybuff, ystart,
                                                                                     zbuff, zstart);
    }
};

// MAC operation for 2buff arch. Template function which also hides the struct contents.
template <typename TT_DATA, typename TT_COEFF, unsigned int TP_FIR_LEN, unsigned int TP_UPSHIFT_CT>
inline T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> macSym2buffIntHb(
    T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> acc,
    T_buff_512b<TT_DATA> xbuff,
    unsigned int xstart,
    T_buff_512b<TT_DATA> ybuff,
    unsigned int ystart,
    T_buff_256b<TT_COEFF> zbuff,
    unsigned int zstart,
    unsigned int ctShift) {
    if
        constexpr(TP_UPSHIFT_CT == 0) {
            // Call overloaded low level function which uses native vectors
            return macSym2buffIntHb(acc, xbuff.val, xstart, ybuff.val, ystart, zbuff.val, zstart);
        }
    else {
        // Call API _uct .
        return macSlidingSym2buffIntHb<TT_DATA, TT_COEFF, TP_FIR_LEN, TP_UPSHIFT_CT>(acc, xbuff, xstart, ybuff, ystart,
                                                                                     zbuff, zstart, ctShift);
    }
};

// MAC operation for Low Polyphase 1buff arch. Template function which also hides the struct contents.
template <typename TT_DATA, typename TT_COEFF, unsigned int TP_UPSHIFT_CT>
inline T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> mulCentreTap2buffIntHb(T_buff_512b<TT_DATA> xbuff,
                                                                              unsigned int xstart,
                                                                              T_buff_256b<TT_COEFF> zbuff) {
    if
        constexpr(TP_UPSHIFT_CT == 0) {
            // Call overloaded low level function which uses native vectors
            return mulCentreTap2buffIntHb(xbuff.val, xstart, zbuff.val);
        }
    else {
        // Call API  .
        T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> retVal;
        return retVal; // return undef vector. Nothing to do here.
    }
};

// Initial MUL operation for 1buff arch. Take inputIF as an argument to ease overloading.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_DUAL_IP,
          unsigned int TP_FIR_LEN,
          unsigned int TP_UPSHIFT_CT>
inline T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> initMacIntHb(
    T_inputIF<CASC_IN_FALSE, TT_DATA, TP_DUAL_IP> inInterface,
    T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> acc,
    T_buff_1024b<TT_DATA> xbuff,
    unsigned int xstart,
    unsigned int ystart,
    T_buff_256b<TT_COEFF> zbuff,
    unsigned int zstart,
    unsigned int ctShift) {
    if
        constexpr(TP_UPSHIFT_CT == 0) {
            // Call overloaded low level function which uses native vectors
            return mulSym1buffIntHb(xbuff.val, xstart, ystart, zbuff.val, zstart);
            // return macSym1buffIntHb(acc, xbuff, xstart, ystart, zbuff, zstart);
        }
    else {
        // Call API  .
        return mulSlidingSym1buffIntHb<TT_DATA, TT_COEFF, TP_FIR_LEN, TP_UPSHIFT_CT>(acc, xbuff, xstart, zbuff, zstart,
                                                                                     ctShift);
    }
};

// Initial MAC operation for 1buff arch. Take inputIF as an argument to ease overloading.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_DUAL_IP,
          unsigned int TP_FIR_LEN,
          unsigned int TP_UPSHIFT_CT>
inline T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> initMacIntHb(
    T_inputIF<CASC_IN_TRUE, TT_DATA, TP_DUAL_IP> inInterface,
    T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> acc,
    T_buff_1024b<TT_DATA> xbuff,
    unsigned int xstart,
    unsigned int ystart,
    T_buff_256b<TT_COEFF> zbuff,
    unsigned int zstart,
    unsigned int ctShift) {
    if
        constexpr(TP_UPSHIFT_CT == 0) {
            // Call overloaded low level function which uses native vectors
            return macSym1buffIntHb(acc, xbuff, xstart, ystart, zbuff, zstart);
        }
    else {
        // Call API  .
        return macSlidingSym1buffIntHb<TT_DATA, TT_COEFF, TP_FIR_LEN, TP_UPSHIFT_CT>(acc, xbuff, xstart, zbuff, zstart,
                                                                                     ctShift);
    }
};
// MAC operation for Low Polyphase 1buff arch. Template function which also hides the struct contents.
template <typename TT_DATA, typename TT_COEFF, unsigned int TP_UPSHIFT_CT>
inline T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> mulCentreTap1buffIntHb(T_buff_1024b<TT_DATA> xbuff,
                                                                              unsigned int xstart,
                                                                              T_buff_256b<TT_COEFF> zbuff) {
    if
        constexpr(TP_UPSHIFT_CT == 0) {
            // Call overloaded low level function which uses native vectors
            return mulCentreTap1buffIntHb(xbuff.val, xstart, zbuff.val);
        }
    else {
        T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> retVal;
        return retVal; // return undef vector. Nothing to do here.
    }
};

// ----------------------------
// Initial MUL operation for 2buff arch. Take inputIF as an argument to ease overloading.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_DUAL_IP,
          unsigned int TP_FIR_LEN,
          unsigned int TP_UPSHIFT_CT>
inline T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> initMacIntHb(
    T_inputIF<CASC_IN_FALSE, TT_DATA, TP_DUAL_IP> inInterface,
    T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> acc,
    T_buff_512b<TT_DATA> xbuff,
    unsigned int xstart,
    T_buff_512b<TT_DATA> ybuff,
    unsigned int ystart,
    T_buff_256b<TT_COEFF> zbuff,
    unsigned int zstart,
    unsigned int ctShift) {
    if
        constexpr(TP_UPSHIFT_CT == 0) {
            // Call overloaded low level function which uses native vectors
            return mulSym2buffIntHb(xbuff.val, xstart, ybuff.val, ystart, zbuff.val, zstart);
        }
    else {
        return mulSlidingSym2buffIntHb<TT_DATA, TT_COEFF, TP_FIR_LEN, TP_UPSHIFT_CT>(acc, xbuff, xstart, ybuff, ystart,
                                                                                     zbuff, zstart, ctShift);
    }
};
// Initial MAC operation for 2buff arch. Take inputIF as an argument to ease overloading.
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_DUAL_IP,
          unsigned int TP_FIR_LEN,
          unsigned int TP_UPSHIFT_CT>
inline T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> initMacIntHb(
    T_inputIF<CASC_IN_TRUE, TT_DATA, TP_DUAL_IP> inInterface,
    T_accSymIntHb<TT_DATA, TT_COEFF, TP_UPSHIFT_CT> acc,
    T_buff_512b<TT_DATA> xbuff,
    unsigned int xstart,
    T_buff_512b<TT_DATA> ybuff,
    unsigned int ystart,
    T_buff_256b<TT_COEFF> zbuff,
    unsigned int zstart,
    unsigned int ctShift) {
    if
        constexpr(TP_UPSHIFT_CT == 0) {
            // Call overloaded low level function which uses native vectors
            return macSym2buffIntHb(acc, xbuff.val, xstart, ybuff.val, ystart, zbuff.val, zstart);
        }
    else {
        return macSlidingSym2buffIntHb<TT_DATA, TT_COEFF, TP_FIR_LEN, TP_UPSHIFT_CT>(acc, xbuff, xstart, ybuff, ystart,
                                                                                     zbuff, zstart, ctShift);
    }
};
}
}
}
}
}
#endif // _DSPLIB_FIR_INTERPOLATE_HB_UTILS_HPP_
