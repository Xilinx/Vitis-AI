/*
 * Copyright 2019 Xilinx, Inc.
 *
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

// This code is derived from OpenCV:
// opencv/modules/core/include/opencv2/core/types_c.h

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/*
 * HLS Video Types Header File
 */

#ifndef ___XF__VIDEO_TYPES__
#define ___XF__VIDEO_TYPES__
#include <hls_math.h>
//#define __INTERVAL(LB,x,RB) ( ( (x) > RB ) ? RB : ( (x) < LB ) ? LB : (x) )

//#define XF_CN_MAX     512
//#define XF_CN_SHIFT   11
//#define XF_DEPTH_MAX  (1 << HLS_CN_SHIFT)
//
#define XF_8U 0
#define XF_8S 1
#define XF_16U 2
#define XF_16S 3
#define XF_32S 4
#define XF_32F 5
#define XF_64F 6
#define XF_USRTYPE1 7
#define XF_10U 8
#define XF_10S 9
#define XF_12U 10
#define XF_12S 11
//
//#define XF_MAT_DEPTH_MASK       (XF_DEPTH_MAX - 1)
//#define XF_MAT_DEPTH(flags)     ((flags) & XF_MAT_DEPTH_MASK)
//
//#define XF_MAKETYPE(depth,cn) (XF_MAT_DEPTH(depth) + (((cn)-1) << XF_CN_SHIFT))
//#define XF_MAKE_TYPE XF_MAKETYPE
//
//#define HLS_8UC1 HLS_MAKETYPE(HLS_8U,1)
//#define HLS_8UC2 HLS_MAKETYPE(HLS_8U,2)
//#define HLS_8UC3 HLS_MAKETYPE(HLS_8U,3)
//#define HLS_8UC4 HLS_MAKETYPE(HLS_8U,4)
//
//#define HLS_8SC1 HLS_MAKETYPE(HLS_8S,1)
//#define HLS_8SC2 HLS_MAKETYPE(HLS_8S,2)
//#define HLS_8SC3 HLS_MAKETYPE(HLS_8S,3)
//#define HLS_8SC4 HLS_MAKETYPE(HLS_8S,4)
//
//#define HLS_10UC1 HLS_MAKETYPE(HLS_10U,1)
//#define HLS_10UC2 HLS_MAKETYPE(HLS_10U,2)
//#define HLS_10UC3 HLS_MAKETYPE(HLS_10U,3)
//#define HLS_10UC4 HLS_MAKETYPE(HLS_10U,4)
//
//#define HLS_10SC1 HLS_MAKETYPE(HLS_10S,1)
//#define HLS_10SC2 HLS_MAKETYPE(HLS_10S,2)
//#define HLS_10SC3 HLS_MAKETYPE(HLS_10S,3)
//#define HLS_10SC4 HLS_MAKETYPE(HLS_10S,4)
//
//#define HLS_12UC1 HLS_MAKETYPE(HLS_12U,1)
//#define HLS_12UC2 HLS_MAKETYPE(HLS_12U,2)
//#define HLS_12UC3 HLS_MAKETYPE(HLS_12U,3)
//#define HLS_12UC4 HLS_MAKETYPE(HLS_12U,4)
//
//#define HLS_12SC1 HLS_MAKETYPE(HLS_12S,1)
//#define HLS_12SC2 HLS_MAKETYPE(HLS_12S,2)
//#define HLS_12SC3 HLS_MAKETYPE(HLS_12S,3)
//#define HLS_12SC4 HLS_MAKETYPE(HLS_12S,4)
//
//#define HLS_16UC1 HLS_MAKETYPE(HLS_16U,1)
//#define HLS_16UC2 HLS_MAKETYPE(HLS_16U,2)
//#define HLS_16UC3 HLS_MAKETYPE(HLS_16U,3)
//#define HLS_16UC4 HLS_MAKETYPE(HLS_16U,4)
//
//#define HLS_16SC1 HLS_MAKETYPE(HLS_16S,1)
//#define HLS_16SC2 HLS_MAKETYPE(HLS_16S,2)
//#define HLS_16SC3 HLS_MAKETYPE(HLS_16S,3)
//#define HLS_16SC4 HLS_MAKETYPE(HLS_16S,4)
//
//#define HLS_32SC1 HLS_MAKETYPE(HLS_32S,1)
//#define HLS_32SC2 HLS_MAKETYPE(HLS_32S,2)
//#define HLS_32SC3 HLS_MAKETYPE(HLS_32S,3)
//#define HLS_32SC4 HLS_MAKETYPE(HLS_32S,4)
//
//#define HLS_32FC1 HLS_MAKETYPE(HLS_32F,1)
//#define HLS_32FC2 HLS_MAKETYPE(HLS_32F,2)
//#define HLS_32FC3 HLS_MAKETYPE(HLS_32F,3)
//#define HLS_32FC4 HLS_MAKETYPE(HLS_32F,4)
//
//#define HLS_64FC1 HLS_MAKETYPE(HLS_64F,1)
//#define HLS_64FC2 HLS_MAKETYPE(HLS_64F,2)
//#define HLS_64FC3 HLS_MAKETYPE(HLS_64F,3)
//#define HLS_64FC4 HLS_MAKETYPE(HLS_64F,4)
//
//#define HLS_SC(BITDEPTH,CN)  HLS_MAKETYPE(BITDEPTH+12,CN)
//
//#define HLS_MAT_CN_MASK          ((HLS_CN_MAX - 1) << HLS_CN_SHIFT)
//#define HLS_MAT_CN(flags)        ((((flags) & HLS_MAT_CN_MASK) >> HLS_CN_SHIFT) + 1)
//#define HLS_MAT_TYPE_MASK        (HLS_DEPTH_MAX*HLS_CN_MAX - 1)
//#define HLS_MAT_TYPE(flags)      ((flags) & HLS_MAT_TYPE_MASK)
//
//#define HLS_ARE_TYPES_EQ(type1, type2) \
//    (((type1 ^ type2) & HLS_MAT_TYPE_MASK) == 0)
//
//#define HLS_ARE_SIZES_EQ(mat1, mat2) \
//    ((mat1).rows == (mat2).rows && (mat1).cols == (mat2).cols)

template <int T>
struct Type {
    typedef ap_int<T - 12> name;
    static const int bitdepth = T - 12;
};
template <>
struct Type<XF_8U> {
    typedef unsigned char name;
    static const int bitdepth = 8;
};
template <>
struct Type<XF_8S> {
    typedef char name;
    static const int bitdepth = 8;
};
template <>
struct Type<XF_10U> {
    typedef ap_uint<10> name;
    static const int bitdepth = 10;
};
template <>
struct Type<XF_10S> {
    typedef ap_int<10> name;
    static const int bitdepth = 10;
};
template <>
struct Type<XF_12U> {
    typedef ap_uint<12> name;
    static const int bitdepth = 12;
};
template <>
struct Type<XF_12S> {
    typedef ap_int<12> name;
    static const int bitdepth = 12;
};
template <>
struct Type<XF_16U> {
    typedef unsigned short name;
    static const int bitdepth = 16;
};
template <>
struct Type<XF_16S> {
    typedef short name;
    static const int bitdepth = 16;
};
template <>
struct Type<XF_32S> {
    typedef int name;
    static const int bitdepth = 32;
};
template <>
struct Type<XF_32F> {
    typedef float name;
    static const int bitdepth = 32;
};
template <>
struct Type<XF_64F> {
    typedef double name;
    static const int bitdepth = 64;
};

template <typename PIXEL_T>
struct pixel_op_type {
    typedef PIXEL_T T;
};
template <>
struct pixel_op_type<unsigned char> {
    typedef ap_uint<8> T;
};
template <>
struct pixel_op_type<char> {
    typedef ap_int<8> T;
};
template <>
struct pixel_op_type<unsigned short> {
    typedef ap_uint<16> T;
};
template <>
struct pixel_op_type<short> {
    typedef ap_int<16> T;
};
template <>
struct pixel_op_type<unsigned int> {
    typedef ap_uint<32> T;
};
template <>
struct pixel_op_type<int> {
    typedef ap_int<32> T;
};
template <int W>
struct pixel_op_type<ap_int<W> > {
    typedef ap_int<W> T;
};
template <int W>
struct pixel_op_type<ap_uint<W> > {
    typedef ap_uint<W> T;
};

#define HLS_TNAME(flags) typename Type<HLS_MAT_DEPTH(flags)>::name

#define HLS_TBITDEPTH(flags) Type<HLS_MAT_DEPTH(flags)>::bitdepth

#define XF_8U_MIN 0
#define XF_8U_MAX 255
#define XF_8S_MIN -127
#define XF_8S_MAX 127
#define XF_10U_MIN 0
#define XF_10U_MAX 1023
#define XF_10S_MIN -511
#define XF_10S_MAX 511
#define XF_12U_MIN 0
#define XF_12U_MAX 4095
#define XF_12S_MIN -2047
#define XF_12S_MAX 2047
#define XF_16U_MIN 0
#define XF_16U_MAX 65535
#define XF_16S_MIN -32767
#define XF_16S_MAX 32767
#define XF_32S_MIN -2147483647
#define XF_32S_MAX 2147483647

template <typename T>
struct Name {
    static const int _min = XF_32S_MIN;
    static const int _max = XF_32S_MAX;
};
template <>
struct Name<unsigned char> {
    static const int _min = XF_8U_MIN;
    static const int _max = XF_8U_MAX;
};
template <>
struct Name<char> {
    static const int _min = XF_8S_MIN;
    static const int _max = XF_8S_MAX;
};
template <>
struct Name<unsigned short> {
    static const int _min = XF_16U_MIN;
    static const int _max = XF_16U_MAX;
};
template <>
struct Name<short> {
    static const int _min = XF_16S_MIN;
    static const int _max = XF_16S_MAX;
};
template <>
struct Name<int> {
    static const int _min = XF_32S_MIN;
    static const int _max = XF_32S_MAX;
};

template <typename T>
unsigned char Convert2uchar(T v) {
    unsigned char result = XF_8U_MIN;
    if (v >= XF_8U_MAX) {
        result = XF_8U_MAX;
    } else if (v >= XF_8U_MIN && v < XF_8U_MAX) {
        ap_fixed<9, 9, AP_RND> temp = v;
        result = temp;
    }
    return result;
}
template <typename T>
char Convert2char(T v) {
    char result = XF_8S_MIN;
    if (v >= XF_8S_MAX) {
        result = XF_8S_MAX;
    } else if (v >= XF_8S_MIN && v < XF_8S_MAX) {
        ap_fixed<9, 9, AP_RND> temp = v;
        result = temp;
    }
    return result;
}
template <typename T>
unsigned short Convert2ushort(T v) {
    unsigned short result = XF_16U_MIN;
    if (v >= XF_16U_MAX) {
        result = XF_16U_MAX;
    } else if (v >= XF_16U_MIN && v < XF_16U_MAX) {
        ap_fixed<17, 17, AP_RND> temp = v;
        result = temp;
    }
    return result;
}
template <typename T>
short Convert2short(T v) {
    short result = XF_16S_MIN;
    if (v >= XF_16S_MAX) {
        result = XF_16S_MAX;
    } else if (v >= XF_16S_MIN && v < XF_16S_MAX) {
        ap_fixed<17, 17, AP_RND> temp = v;
        result = temp;
    }
    return result;
}
template <typename T>
int Convert2int(T v) {
    int result = XF_32S_MIN;
    if (v >= XF_32S_MAX) {
        result = XF_32S_MAX;
    } else if (v >= XF_32S_MIN && v < XF_32S_MAX) {
        ap_fixed<32, 32, AP_RND> temp = v;
        result = temp;
    }
    return result;
}
// The type is redefined, in previous versions it was ap_int<12>
typedef ap_uint<32> XF_SIZE_T;
typedef ap_uint<5> XF_CHANNEL_T;

namespace xf {
namespace cv {

/* sr_cast: saturate and round cast: T1 -> T2 */

template <typename T2>
class sr_cast_class {};

template <>
class sr_cast_class<float> {
   public:
    template <typename T1>
    inline float operator()(T1 v) {
        return v;
    }
    inline float operator()(double v) { return HLS_FPO_DTOF(v); }
};

template <>
class sr_cast_class<double> {
   public:
    template <typename T1>
    inline double operator()(T1 v) {
        return v;
    }
    inline double operator()(float v) { return HLS_FPO_FTOD(v); }
};

template <int N2>
class sr_cast_class<ap_int<N2> > {
   public:
    template <int N1>
    inline ap_int<N2> operator()(ap_int<N1> v) {
        return ap_fixed<N2, N2, AP_TRN, AP_SAT>(v);
    }
    template <int N1>
    inline ap_int<N2> operator()(ap_uint<N1> v) {
        return ap_fixed<N2, N2, AP_TRN, AP_SAT>(v);
    }
    template <int W, int I, ap_q_mode _AP_Q, ap_o_mode _AP_O>
    inline ap_int<N2> operator()(ap_fixed<W, I, _AP_Q, _AP_O> v) {
        return ap_fixed<N2, N2, AP_RND, AP_SAT>(v);
    }
    template <int W, int I, ap_q_mode _AP_Q, ap_o_mode _AP_O>
    inline ap_int<N2> operator()(ap_ufixed<W, I, _AP_Q, _AP_O> v) {
        return ap_fixed<N2, N2, AP_RND, AP_SAT>(v);
    }
    template <int W, int I, ap_q_mode _AP_Q, ap_o_mode _AP_O, int N>
    inline ap_int<N2> operator()(ap_fixed_base<W, I, true, _AP_Q, _AP_O, N> v) {
        return ap_fixed<N2, N2, AP_RND, AP_SAT>(v);
    }
    template <int W, int I, ap_q_mode _AP_Q, ap_o_mode _AP_O, int N>
    inline ap_int<N2> operator()(ap_fixed_base<W, I, false, _AP_Q, _AP_O, N> v) {
        return ap_fixed<N2, N2, AP_RND, AP_SAT>(v);
    }
    inline ap_int<N2> operator()(float v) {
        if (::hls::__isnan(v))
            return 0;
        else
            return ap_fixed<N2, N2, AP_RND, AP_SAT>(v);
    }
    inline ap_int<N2> operator()(double v) {
        if (::hls::__isnan(v))
            return 0;
        else
            return ap_fixed<N2, N2, AP_RND, AP_SAT>(v);
    }
    inline ap_int<N2> operator()(unsigned char v) { return operator()(ap_uint<8>(v)); }
    inline ap_int<N2> operator()(char v) { return operator()(ap_int<8>(v)); }
    inline ap_int<N2> operator()(unsigned short v) { return operator()(ap_uint<16>(v)); }
    inline ap_int<N2> operator()(short v) { return operator()(ap_int<16>(v)); }
    inline ap_int<N2> operator()(unsigned int v) { return operator()(ap_uint<32>(v)); }
    inline ap_int<N2> operator()(int v) { return operator()(ap_int<32>(v)); }
    inline ap_int<N2> operator()(unsigned long long v) { return operator()(ap_uint<64>(v)); }
    inline ap_int<N2> operator()(long long v) { return operator()(ap_int<64>(v)); }
};

template <int N2>
class sr_cast_class<ap_uint<N2> > {
   public:
    template <int N1>
    inline ap_uint<N2> operator()(ap_int<N1> v) {
        return ap_ufixed<N2, N2, AP_TRN, AP_SAT>(v);
    }
    template <int N1>
    inline ap_uint<N2> operator()(ap_uint<N1> v) {
        return ap_ufixed<N2, N2, AP_TRN, AP_SAT>(v);
    }
    template <int W, int I, ap_q_mode _AP_Q, ap_o_mode _AP_O>
    inline ap_uint<N2> operator()(ap_fixed<W, I, _AP_Q, _AP_O> v) {
        return ap_ufixed<N2, N2, AP_RND, AP_SAT>(v);
    }
    template <int W, int I, ap_q_mode _AP_Q, ap_o_mode _AP_O>
    inline ap_uint<N2> operator()(ap_ufixed<W, I, _AP_Q, _AP_O> v) {
        return ap_ufixed<N2, N2, AP_RND, AP_SAT>(v);
    }
    template <int W, int I, ap_q_mode _AP_Q, ap_o_mode _AP_O, int N>
    inline ap_uint<N2> operator()(ap_fixed_base<W, I, true, _AP_Q, _AP_O, N> v) {
        return ap_ufixed<N2, N2, AP_RND, AP_SAT>(v);
    }
    template <int W, int I, ap_q_mode _AP_Q, ap_o_mode _AP_O, int N>
    inline ap_uint<N2> operator()(ap_fixed_base<W, I, false, _AP_Q, _AP_O, N> v) {
        return ap_ufixed<N2, N2, AP_RND, AP_SAT>(v);
    }
    inline ap_uint<N2> operator()(float v) {
        if (::hls::__isnan(v))
            return 0;
        else
            return ap_ufixed<N2, N2, AP_RND, AP_SAT>(v);
    }
    inline ap_uint<N2> operator()(double v) {
        if (::hls::__isnan(v))
            return 0;
        else
            return ap_ufixed<N2, N2, AP_RND, AP_SAT>(v);
    }
    inline ap_uint<N2> operator()(unsigned char v) { return operator()(ap_uint<8>(v)); }
    inline ap_uint<N2> operator()(char v) { return operator()(ap_int<8>(v)); }
    inline ap_uint<N2> operator()(unsigned short v) { return operator()(ap_uint<16>(v)); }
    inline ap_uint<N2> operator()(short v) { return operator()(ap_int<16>(v)); }
    inline ap_uint<N2> operator()(unsigned int v) { return operator()(ap_uint<32>(v)); }
    inline ap_uint<N2> operator()(int v) { return operator()(ap_int<32>(v)); }
    inline ap_uint<N2> operator()(unsigned long long v) { return operator()(ap_uint<64>(v)); }
    inline ap_uint<N2> operator()(long long v) { return operator()(ap_int<64>(v)); }
};

template <>
class sr_cast_class<unsigned char> : public sr_cast_class<ap_uint<8> > {
   public:
    using sr_cast_class<ap_uint<8> >::operator();
};

template <>
class sr_cast_class<char> : public sr_cast_class<ap_int<8> > {
   public:
    using sr_cast_class<ap_int<8> >::operator();
};

template <>
class sr_cast_class<unsigned short> : public sr_cast_class<ap_uint<16> > {
   public:
    using sr_cast_class<ap_uint<16> >::operator();
};

template <>
class sr_cast_class<short> : public sr_cast_class<ap_int<16> > {
   public:
    using sr_cast_class<ap_int<16> >::operator();
};

template <>
class sr_cast_class<unsigned int> : public sr_cast_class<ap_uint<32> > {
   public:
    using sr_cast_class<ap_uint<32> >::operator();
};

template <>
class sr_cast_class<int> : public sr_cast_class<ap_int<32> > {
   public:
    using sr_cast_class<ap_int<32> >::operator();
};

template <>
class sr_cast_class<unsigned long long> : public sr_cast_class<ap_uint<64> > {
   public:
    using sr_cast_class<ap_uint<64> >::operator();
};

template <>
class sr_cast_class<long long> : public sr_cast_class<ap_int<64> > {
   public:
    using sr_cast_class<ap_int<64> >::operator();
};

template <typename T2, typename T1>
inline T2 sr_cast(T1 v) {
    ::xf::cv::sr_cast_class<T2> V;
    return V(v);
}

} // namespace cv
} // namespace xf

#endif
