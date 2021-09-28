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
#ifndef _DSPLIB_FIR_INTERPOLATE_ASYM_UTILS_HPP_
#define _DSPLIB_FIR_INTERPOLATE_ASYM_UTILS_HPP_

namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace interpolate_asym {
/*
Asymmetrical Interpolation FIR Utilities
This file contains sets of overloaded, templatized and specialized templatized functions for use
by the main kernel class and run-time function. These functions are separate from the traits file
because they are purely for kernel use, not graph level compilation.
*/

// Templated struct to hold the accumulator type appropriate
// to the selected data and coefficient types;
// derived from base T_ACC struct - which requires bringing operator= into scope.
template <typename T_D, typename T_C>
struct T_accIntAsym : T_acc<T_D, T_C> {
    using T_acc<T_D, T_C>::operator=;
};
// template<> struct T_accIntAsym<int16, int16>   : T_acc<int16, int16>       {using T_acc<int16, int16>
// ::operator=;};// {v16acc48 val = null_v16acc48(); v16acc48 uval = null_v16acc48();};
// template<> struct T_accIntAsym<cint16, int16>  : T_acc<cint16, int16>      {using T_acc<cint16, int16>
// ::operator=;};// {v8cacc48 val = null_v8cacc48(); v8cacc48 uval = null_v8cacc48();};
// template<> struct T_accIntAsym<cint16, cint16> : T_acc<cint16, cint16>     {using T_acc<cint16,
// cint16>::operator=;};// {v8cacc48 val = null_v8cacc48(); v8cacc48 uval = null_v8cacc48();};
// template<> struct T_accIntAsym<int32, int16>   : T_acc<int32, int16>       {using T_acc<int32, int16>
// ::operator=;};// {v8acc80  val = null_v8acc80 (); v8acc80  uval = null_v8acc80 ();};
// template<> struct T_accIntAsym<int32, int32>   : T_acc<int32, int32>       {using T_acc<int32, int32>
// ::operator=;};// {v8acc80  val = null_v8acc80 (); v8acc80  uval = null_v8acc80 ();};
// template<> struct T_accIntAsym<cint32, int16>  : T_acc<cint32, int16>      {using T_acc<cint32, int16>
// ::operator=;};// {v4cacc80 val = null_v4cacc80(); v4cacc80 uval = null_v4cacc80();};
// template<> struct T_accIntAsym<cint32, int32>  : T_acc<cint32, int32>      {using T_acc<cint32, int32>
// ::operator=;};// {v4cacc80 val = null_v4cacc80(); v4cacc80 uval = null_v4cacc80();};
// template<> struct T_accIntAsym<cint32, cint16> : T_acc<cint32, cint16>     {using T_acc<cint32,
// cint16>::operator=;};// {v4cacc80 val = null_v4cacc80(); v4cacc80 uval = null_v4cacc80();};
// template<> struct T_accIntAsym<cint32, cint32> : T_acc<cint32, cint32>     {using T_acc<cint32,
// cint32>::operator=;};// {v2cacc80 val = null_v2cacc80(); v2cacc80 uval = null_v2cacc80();};
// template<> struct T_accIntAsym<float,  float>  : T_acc<float,  float>      {using T_acc<float,  float>
// ::operator=;};// {v8float  val = null_v8float (); v8float  uval = null_v8float ();};
// template<> struct T_accIntAsym<cfloat, float>  : T_acc<cfloat, float>      {using T_acc<cfloat, float>
// ::operator=;};// {v4cfloat val = null_v4cfloat(); v4cfloat uval = null_v4cfloat();};
// template<> struct T_accIntAsym<cfloat, cfloat> : T_acc<cfloat, cfloat>     {using T_acc<cfloat,
// cfloat>::operator=;};// {v4cfloat val = null_v4cfloat(); v4cfloat uval = null_v4cfloat();};

// templated struct to hold the appropriate vector type for write to the class output window.
template <typename T_D, typename T_C>
struct T_outValIntAsym : T_outVal<T_D, T_C> {
    using T_outVal<T_D, T_C>::operator=;
};
// template<> struct T_outValIntAsym<int16, int16>   : T_outVal<int16, int16>   {using T_outVal<int16, int16>
// ::operator=;};  //{v16int16 val;};//v16int16 outVal;};
// template<> struct T_outValIntAsym<cint16, int16>  : T_outVal<cint16, int16>  {using T_outVal<cint16, int16>
// ::operator=;};  //{v8cint16 val;};//v8cint16 outVal;};
// template<> struct T_outValIntAsym<cint16, cint16> : T_outVal<cint16, cint16> {using T_outVal<cint16,
// cint16>::operator=;};  //{v8cint16 val;};//v8cint16 outVal;};
// template<> struct T_outValIntAsym<int32, int16>   : T_outVal<int32, int16>   {using T_outVal<int32, int16>
// ::operator=;};  //{v8int32  val;};//v8int32  outVal;};
// template<> struct T_outValIntAsym<int32, int32>   : T_outVal<int32, int32>   {using T_outVal<int32, int32>
// ::operator=;};  //{v8int32  val;};//v8int32  outVal;};
// template<> struct T_outValIntAsym<cint32, int16>  : T_outVal<cint32, int16>  {using T_outVal<cint32, int16>
// ::operator=;};  //{v4cint32 val;};//v4cint32 outVal;};
// template<> struct T_outValIntAsym<cint32, int32>  : T_outVal<cint32, int32>  {using T_outVal<cint32, int32>
// ::operator=;};  //{v4cint32 val;};//v4cint32 outVal;};
// template<> struct T_outValIntAsym<cint32, cint16> : T_outVal<cint32, cint16> {using T_outVal<cint32,
// cint16>::operator=;};  //{v4cint32 val;};//v4cint32 outVal;};
// template<> struct T_outValIntAsym<cint32, cint32> : T_outVal<cint32, cint32> {using T_outVal<cint32,
// cint32>::operator=;};  //{v4cint32 val;};//v4cint32 outVal;};
// template<> struct T_outValIntAsym<float, float>   : T_outVal<float,  float>  {using T_outVal<float,  float>
// ::operator=;};  //{v8float  val;};//v8float  outVal;};
// template<> struct T_outValIntAsym<cfloat, float>  : T_outVal<cfloat, float>  {using T_outVal<cfloat, float>
// ::operator=;};  //{v4cfloat val;};//v4cfloat outVal;};
// template<> struct T_outValIntAsym<cfloat, cfloat> : T_outVal<cfloat, cfloat> {using T_outVal<cfloat,
// cfloat>::operator=;};  //{v4cfloat val;};//v4cfloat outVal;};

// Overloaded mul/mac calls for asymmetric interpolation
//-----------------------------------------------------------------------------------------------------
// DATA = int16, COEFF = int16
inline T_accIntAsym<int16, int16> mulIntAsym(v64int16 xbuff,
                                             v16int16 zbuff,
                                             unsigned int interpolateFactor,
                                             unsigned int lanes,
                                             int64 xoffsetslut,
                                             int xstartlut) {
    T_accIntAsym<int16, int16> retVal;
    const unsigned int xsquare = 0x1010;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x03020100;
    const unsigned int zoffsets_hi = 0x07060504;
    const unsigned int zstep = 1;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int xoffsets_hi = (int32)(xoffsetslut >> 32);
    unsigned int zstart = 0;

    retVal.val = mul16(xbuff, xstart, xoffsets, xoffsets_hi, xsquare, zbuff, zstart, zoffsets, zoffsets_hi, zstep);
    return retVal;
}

inline T_accIntAsym<int16, int16> macIntAsym(T_accIntAsym<int16, int16> acc,
                                             v64int16 xbuff,
                                             v16int16 zbuff,
                                             unsigned int interpolateFactor,
                                             unsigned int lanes,
                                             int64 xoffsetslut,
                                             int xstartlut) {
    T_accIntAsym<int16, int16> retVal;
    const unsigned int xsquare = 0x1010;
    const unsigned int xstep = 2;
    const unsigned int zoffsets = 0x03020100;
    ;
    const unsigned int zoffsets_hi = 0x07060504;
    const unsigned int zstep = 1;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int xoffsets_hi = (int32)(xoffsetslut >> 32);
    unsigned int zstart = 0;

    retVal.val =
        mac16(acc.val, xbuff, xstart, xoffsets, xoffsets_hi, xsquare, zbuff, zstart, zoffsets, zoffsets_hi, zstep);
    return retVal;
}

// DATA = cint16, COEFF = int16
inline T_accIntAsym<cint16, int16> mulIntAsym(v32cint16 xbuff,
                                              v16int16 zbuff,
                                              unsigned int interpolateFactor,
                                              unsigned int lanes,
                                              int64 xoffsetslut,
                                              int xstartlut) {
    T_accIntAsym<cint16, int16> retVal;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x76543210;
    const unsigned int zstep = lanes;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int zstart = 0;

    retVal.val = mul8(xbuff, xstart, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accIntAsym<cint16, int16> macIntAsym(T_accIntAsym<cint16, int16> acc,
                                              v32cint16 xbuff,
                                              v16int16 zbuff,
                                              unsigned int interpolateFactor,
                                              unsigned int lanes,
                                              int64 xoffsetslut,
                                              int xstartlut) {
    T_accIntAsym<cint16, int16> retVal;
    const unsigned int xstep = 1;
    const unsigned int zoffsets = 0x76543210;
    const unsigned int zstep = lanes;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int zstart = 0;

    retVal.val = mac8(acc.val, xbuff, xstart, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = cint16, COEFF = cint16
inline T_accIntAsym<cint16, cint16> mulIntAsym(v32cint16 xbuff,
                                               v8cint16 zbuff,
                                               unsigned int interpolateFactor,
                                               unsigned int lanes,
                                               int64 xoffsetslut,
                                               int xstartlut) {
    T_accIntAsym<cint16, cint16> retVal;
    const unsigned int zoffsets = 0x76543210;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int zstart = 0;

    retVal.val = mul8(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accIntAsym<cint16, cint16> macIntAsym(T_accIntAsym<cint16, cint16> acc,
                                               v32cint16 xbuff,
                                               v8cint16 zbuff,
                                               unsigned int interpolateFactor,
                                               unsigned int lanes,
                                               int64 xoffsetslut,
                                               int xstartlut) {
    T_accIntAsym<cint16, cint16> retVal;
    const unsigned int zoffsets = 0x76543210;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int zstart = 0;

    retVal.val = mac8(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = int32, COEFF = int16
inline T_accIntAsym<int32, int16> mulIntAsym(v32int32 xbuff,
                                             v16int16 zbuff,
                                             unsigned int interpolateFactor,
                                             unsigned int lanes,
                                             int64 xoffsetslut,
                                             int xstartlut) {
    T_accIntAsym<int32, int16> retVal;
    const unsigned int kDRegLen = 8;
    const unsigned int kBitsInNibble = 4;
    const unsigned int zoffsets = 0x76543210;
    const unsigned int xstep = 1;
    const unsigned int zstep = lanes;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int zstart = 0;

    retVal.val = lmul8(xbuff, xstart, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accIntAsym<int32, int16> macIntAsym(T_accIntAsym<int32, int16> acc,
                                             v32int32 xbuff,
                                             v16int16 zbuff,
                                             unsigned int interpolateFactor,
                                             unsigned int lanes,
                                             int64 xoffsetslut,
                                             int xstartlut) {
    T_accIntAsym<int32, int16> retVal;
    const unsigned int kDRegLen = 8;
    const unsigned int kBitsInNibble = 4;
    const unsigned int zoffsets = 0x76543210;
    const unsigned int xstep = 1;
    const unsigned int zstep = lanes;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int zstart = 0;

    retVal.val = lmac8(acc.val, xbuff, xstart, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = int32,  COEFF = int32>
inline T_accIntAsym<int32, int32> mulIntAsym(v32int32 xbuff,
                                             v8int32 zbuff,
                                             unsigned int interpolateFactor,
                                             unsigned int lanes,
                                             int64 xoffsetslut,
                                             int xstartlut) {
    T_accIntAsym<int32, int32> retVal;
    const unsigned int kDRegLen = 8;
    const unsigned int kBitsInNibble = 4;
    const unsigned int zoffsets = 0x76543210; // was = 0x00000000;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int zstart = 0;

    retVal.val = lmul8(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accIntAsym<int32, int32> macIntAsym(T_accIntAsym<int32, int32> acc,
                                             v32int32 xbuff,
                                             v8int32 zbuff,
                                             unsigned int interpolateFactor,
                                             unsigned int lanes,
                                             int64 xoffsetslut,
                                             int xstartlut) {
    T_accIntAsym<int32, int32> retVal;
    const unsigned int kDRegLen = 8;
    const unsigned int kBitsInNibble = 4;
    const unsigned int zoffsets = 0x76543210; // 0x00000000;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int zstart = 0;

    retVal.val = lmac8(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cint32, COEFF =  int16>
inline T_accIntAsym<cint32, int16> mulIntAsym(v16cint32 xbuff,
                                              v16int16 zbuff,
                                              unsigned int interpolateFactor,
                                              unsigned int lanes,
                                              int64 xoffsetslut,
                                              int xstartlut) {
    T_accIntAsym<cint32, int16> retVal;
    const unsigned int zoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zstep = lanes;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int zstart = 0;

    retVal.val = lmul4(xbuff, xstart, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

inline T_accIntAsym<cint32, int16> macIntAsym(T_accIntAsym<cint32, int16> acc,
                                              v16cint32 xbuff,
                                              v16int16 zbuff,
                                              unsigned int interpolateFactor,
                                              unsigned int lanes,
                                              int64 xoffsetslut,
                                              int xstartlut) {
    T_accIntAsym<cint32, int16> retVal;
    const unsigned int zoffsets = 0x3210;
    const unsigned int xstep = 1;
    const unsigned int zstep = lanes;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int zstart = 0;

    retVal.val = lmac4(acc.val, xbuff, xstart, xoffsets, xstep, zbuff, zstart, zoffsets, zstep);
    return retVal;
}

// DATA = cint32, COEFF =  int32>
inline T_accIntAsym<cint32, int32> mulIntAsym(v16cint32 xbuff,
                                              v8int32 zbuff,
                                              unsigned int interpolateFactor,
                                              unsigned int lanes,
                                              int64 xoffsetslut,
                                              int xstartlut) {
    T_accIntAsym<cint32, int32> retVal;
    const unsigned int zoffsets = 0x3210;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int zstart = 0;

    retVal.val = lmul4(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accIntAsym<cint32, int32> macIntAsym(T_accIntAsym<cint32, int32> acc,
                                              v16cint32 xbuff,
                                              v8int32 zbuff,
                                              unsigned int interpolateFactor,
                                              unsigned int lanes,
                                              int64 xoffsetslut,
                                              int xstartlut) {
    T_accIntAsym<cint32, int32> retVal;
    const unsigned int zoffsets = 0x3210;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int zstart = 0;

    retVal.val = lmac4(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cint32, COEFF =  cint16>
inline T_accIntAsym<cint32, cint16> mulIntAsym(v16cint32 xbuff,
                                               v8cint16 zbuff,
                                               unsigned int interpolateFactor,
                                               unsigned int lanes,
                                               int64 xoffsetslut,
                                               int xstartlut) {
    T_accIntAsym<cint32, cint16> retVal;
    const unsigned int zoffsets = 0x3210;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int zstart = 0;

    retVal.val = lmul4(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accIntAsym<cint32, cint16> macIntAsym(T_accIntAsym<cint32, cint16> acc,
                                               v16cint32 xbuff,
                                               v8cint16 zbuff,
                                               unsigned int interpolateFactor,
                                               unsigned int lanes,
                                               int64 xoffsetslut,
                                               int xstartlut) {
    T_accIntAsym<cint32, cint16> retVal;
    const unsigned int zoffsets = 0x3210;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int zstart = 0;

    retVal.val = lmac4(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cint32, COEFF =  cint32>
inline T_accIntAsym<cint32, cint32> mulIntAsym(v16cint32 xbuff,
                                               const v4cint32 zbuff,
                                               unsigned int interpolateFactor,
                                               unsigned int lanes,
                                               int64 xoffsetslut,
                                               int xstartlut) {
    T_accIntAsym<cint32, cint32> retVal;
    const unsigned int zoffsets = 0x10;
    const unsigned int zoffsets2 = 0x32;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int xoffsets2 = (int32)xoffsetslut >> 8;
    unsigned int zstart = 0;

    retVal.val = lmul2(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.uval = lmul2(xbuff, xstart, xoffsets2, zbuff, zstart, zoffsets2);
    return retVal;
}

inline T_accIntAsym<cint32, cint32> macIntAsym(T_accIntAsym<cint32, cint32> acc,
                                               v16cint32 xbuff,
                                               v4cint32 zbuff,
                                               unsigned int interpolateFactor,
                                               unsigned int lanes,
                                               int64 xoffsetslut,
                                               int xstartlut) {
    T_accIntAsym<cint32, cint32> retVal;
    const unsigned int zoffsets = 0x10;
    const unsigned int zoffsets2 = 0x32;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int xoffsets2 = (int32)xoffsetslut >> 8;
    unsigned int zstart = 0;

    retVal.val = lmac2(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    retVal.uval = lmac2(acc.uval, xbuff, xstart, xoffsets2, zbuff, zstart, zoffsets2);
    return retVal;
}

// DATA = float,  COEFF = float>
inline T_accIntAsym<float, float> mulIntAsym(v32float xbuff,
                                             v8float zbuff,
                                             unsigned int interpolateFactor,
                                             unsigned int lanes,
                                             int64 xoffsetslut,
                                             int xstartlut) {
    T_accIntAsym<float, float> retVal;
    const unsigned int zoffsets = 0x76543210;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int zstart = 0;

    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accIntAsym<float, float> macIntAsym(T_accIntAsym<float, float> acc,
                                             v32float xbuff,
                                             v8float zbuff,
                                             unsigned int interpolateFactor,
                                             unsigned int lanes,
                                             int64 xoffsetslut,
                                             int xstartlut) {
    T_accIntAsym<float, float> retVal;
    const unsigned int zoffsets = 0x76543210;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int zstart = 0;

    retVal.val = fpmac(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cfloat, COEFF =  float>
inline T_accIntAsym<cfloat, float> mulIntAsym(v16cfloat xbuff,
                                              v8float zbuff,
                                              unsigned int interpolateFactor,
                                              unsigned int lanes,
                                              int64 xoffsetslut,
                                              int xstartlut) {
    T_accIntAsym<cfloat, float> retVal;
    const unsigned int zoffsets = 0x3210;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int zstart = 0;

    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accIntAsym<cfloat, float> macIntAsym(T_accIntAsym<cfloat, float> acc,
                                              v16cfloat xbuff,
                                              v8float zbuff,
                                              unsigned int interpolateFactor,
                                              unsigned int lanes,
                                              int64 xoffsetslut,
                                              int xstartlut) {
    T_accIntAsym<cfloat, float> retVal;
    const unsigned int zoffsets = 0x3210;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int zstart = 0;

    retVal.val = fpmac(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

// DATA = cfloat, COEFF =  cfloat>
inline T_accIntAsym<cfloat, cfloat> mulIntAsym(v16cfloat xbuff,
                                               v4cfloat zbuff,
                                               unsigned int interpolateFactor,
                                               unsigned int lanes,
                                               int64 xoffsetslut,
                                               int xstartlut) {
    T_accIntAsym<cfloat, cfloat> retVal;
    const unsigned int zoffsets = 0x3210;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int zstart = 0;

    retVal.val = fpmul(xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

inline T_accIntAsym<cfloat, cfloat> macIntAsym(T_accIntAsym<cfloat, cfloat> acc,
                                               v16cfloat xbuff,
                                               v4cfloat zbuff,
                                               unsigned int interpolateFactor,
                                               unsigned int lanes,
                                               int64 xoffsetslut,
                                               int xstartlut) {
    T_accIntAsym<cfloat, cfloat> retVal;
    const unsigned int zoffsets = 0x3210;
    unsigned int xstart = xstartlut;
    unsigned int xoffsets = (int32)xoffsetslut;
    unsigned int zstart = 0;

    retVal.val = fpmac(acc.val, xbuff, xstart, xoffsets, zbuff, zstart, zoffsets);
    return retVal;
}

template <typename TT_DATA, typename TT_COEFF>
inline T_outVal<TT_DATA, TT_COEFF> shiftAndSaturateIntAsym(const T_acc<TT_DATA, TT_COEFF> acc, const int shift) {
    return shiftAndSaturate(acc, shift);
}

// Overloaded mul call for asymmetric interpolation
template <typename TT_DATA, typename TT_COEFF>
inline T_accIntAsym<TT_DATA, TT_COEFF> mulIntAsym(T_buff_1024b<TT_DATA> xbuff,
                                                  T_buff_256b<TT_COEFF> zbuff,
                                                  unsigned int interpolateFactor,
                                                  unsigned int lanes,
                                                  int64 xoffsetslut,
                                                  int xstart) {
    return mulIntAsym(xbuff.val, zbuff.val, interpolateFactor, lanes, xoffsetslut, xstart);
};
// (T_accIntAsym<int16, int16> acc, v64int16 xbuff, v16int16 zbuff,
// unsigned int interpolateFactor, unsigned int lanes, int64 xoffsetslut, int xstartlut)
// Overloaded mac call for asymmetric interpolation
template <typename TT_DATA, typename TT_COEFF>
inline T_accIntAsym<TT_DATA, TT_COEFF> macIntAsym(T_accIntAsym<TT_DATA, TT_COEFF> acc,
                                                  T_buff_1024b<TT_DATA> xbuff,
                                                  T_buff_256b<TT_COEFF> zbuff,
                                                  unsigned int interpolateFactor,
                                                  unsigned int lanes,
                                                  int64 xoffsetslut,
                                                  int xstart) {
    return macIntAsym(acc, xbuff.val, zbuff.val, interpolateFactor, lanes, xoffsetslut, xstart);
};

// Initial MAC/MUL operation. Take inputIF as an argument to ease overloading.
template <typename TT_DATA, typename TT_COEFF>
inline T_accIntAsym<TT_DATA, TT_COEFF> initMacIntAsym(T_inputIF<CASC_IN_FALSE, TT_DATA> inInterface,
                                                      T_accIntAsym<TT_DATA, TT_COEFF> acc,
                                                      T_buff_1024b<TT_DATA> xbuff,
                                                      T_buff_256b<TT_COEFF> zbuff,
                                                      unsigned int interpolateFactor,
                                                      unsigned int lanes,
                                                      int64 xoffsetslut,
                                                      int xstart) {
    return mulIntAsym(xbuff, zbuff, interpolateFactor, lanes, xoffsetslut, xstart);
};

template <typename TT_DATA, typename TT_COEFF>
inline T_accIntAsym<TT_DATA, TT_COEFF> initMacIntAsym(T_inputIF<CASC_IN_TRUE, TT_DATA> inInterface,
                                                      T_accIntAsym<TT_DATA, TT_COEFF> acc,
                                                      T_buff_1024b<TT_DATA> xbuff,
                                                      T_buff_256b<TT_COEFF> zbuff,
                                                      unsigned int interpolateFactor,
                                                      unsigned int lanes,
                                                      int64 xoffsetslut,
                                                      int xstart) {
    return macIntAsym(acc, xbuff, zbuff, interpolateFactor, lanes, xoffsetslut, xstart);
};
}
}
}
}
} // namespaces

#endif // _DSPLIB_FIR_INTERPOLATE_HB_UTILS_HPP_
