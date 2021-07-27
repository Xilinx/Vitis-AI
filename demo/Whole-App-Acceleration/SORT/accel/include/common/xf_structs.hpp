/*
 * Copyright 2020 Xilinx, Inc.
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

#ifndef _XF_STRUCTS_H_
#define _XF_STRUCTS_H_

#ifndef __cplusplus
#error C++ is needed to use this file!
#endif

#ifndef __SYNTHESIS__
#include <iostream>
#endif
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "xf_types.hpp"
#include <assert.h>
#include <math.h>
#include <stdio.h>

namespace xf {
namespace cv {

template <typename T>
T float2ap_uint(float val) {
    T* val_out = (T*)(&val);
    return *val_out;
}

template <typename T>
float ap_uint2float(T val) {
    float* val_out = (float*)(&val);
    return *val_out;
}

//----------------------------------------------------------------------------------------------------//
//  LOCAL STEREO BLOCK MATCHING UTILITY
//----------------------------------------------------------------------------------------------------//
template <int WSIZE, int NDISP, int NDISP_UNIT>
class xFSBMState {
   public:
    // pre-filtering (normalization of input images)
    int preFilterType; // =HLS_STEREO_BM_XSOBEL_TEST
    int preFilterSize; // averaging window size: ~5x5..21x21
    int preFilterCap;  // the output of pre-filtering is clipped by
                       // [-preFilterCap,preFilterCap]

    // correspondence using Sum of Absolute Difference (SAD)
    int SADWindowSize;       // ~5x5..21x21 // defined in macro
    int minDisparity;        // minimum disparity (can be negative)
    int numberOfDisparities; // maximum disparity - minimum disparity (> 0)

    // post-filtering
    int textureThreshold; // the disparity is only computed for pixels

    // with textured enough neighborhood
    int uniquenessRatio; // accept the computed disparity d* only if
    // SAD(d) >= SAD(d*)*(1 + uniquenessRatio/100.)
    // for any d != d*+/-1 within the search range.

    // int speckleWindowSize; // disparity variation window
    // int speckleRange;      // acceptable range of variation in window

    int ndisp_unit;
    int sweepFactor;
    int remainder;

    xFSBMState() {
        preFilterType = XF_STEREO_PREFILTER_SOBEL_TYPE; // Default Sobel filter
        preFilterSize = WSIZE;
        preFilterCap = 31;
        SADWindowSize = WSIZE;
        minDisparity = 0;
        numberOfDisparities = NDISP;
        textureThreshold = 10;
        uniquenessRatio = 15;
        sweepFactor = (NDISP / NDISP_UNIT) + ((NDISP % NDISP_UNIT) != 0);
        ndisp_unit = NDISP_UNIT;
        remainder = NDISP_UNIT * sweepFactor - NDISP;
    }
};
//----------------------------------------------------------------------------------------------------//

//----------------------------------------------------------------------------------------------------//
// Template class of Point_
//----------------------------------------------------------------------------------------------------//
template <typename T>
class Point_ {
   public:
    Point_();
    Point_(T _x, T _y);
    Point_(const Point_& pt);
    ~Point_();

    T x, y;
};

/* Member functions of Point_ class */
template <typename T>
inline Point_<T>::Point_() {}
template <typename T>
inline Point_<T>::Point_(T _x, T _y) : x(_x), y(_y) {}
template <typename T>
inline Point_<T>::Point_(const Point_<T>& pt) : x(pt.x), y(pt.y) {}
template <typename T>
inline Point_<T>::~Point_() {}

typedef Point_<int> Point;
//----------------------------------------------------------------------------------------------------//

//----------------------------------------------------------------------------------------------------//
// Template class of Size_
//----------------------------------------------------------------------------------------------------//
template <typename T>
class Size_ {
   public:
    Size_();
    Size_(T _width, T _height);
    Size_(const Size_<T>& sz);
    Size_(const Point_<T>& pt);
    T area();
    ~Size_();

    T width, height;
};

/* Member functions of Size_ class */
template <typename T>
inline Size_<T>::Size_() {}
template <typename T>
inline Size_<T>::Size_(T _width, T _height) : width(_width), height(_height) {}
template <typename T>
inline Size_<T>::Size_(const Size_<T>& sz) : width(sz.width), height(sz.height) {}
template <typename T>
inline Size_<T>::Size_(const Point_<T>& pt) : width(pt.x), height(pt.y) {}
template <typename T>
inline T Size_<T>::area() {
    return width * height;
}
template <typename T>
inline Size_<T>::~Size_() {}

typedef Size_<int> Size;
//----------------------------------------------------------------------------------------------------//

//----------------------------------------------------------------------------------------------------//
// Template class of Rect_
//----------------------------------------------------------------------------------------------------//
template <typename T>
class Rect_ {
   public:
    Rect_();
    Rect_(T _x, T _y, T _width, T _height);
    Rect_(const Rect_& rect);
    Rect_(const Point_<T>& pt, const Size_<T>& sz);
    T area();
    Size_<T> size();
    Point_<T> tl(); // top-left point(inside);
    Point_<T> tr(); // top-right point(outside);
    Point_<T> bl(); // bottom-left point(outside);
    Point_<T> br(); // bottom-right point(outside);
    bool bContains(const Point_<T>& pt);
    ~Rect_();

    T x, y, width, height;
};

/* Member functions of Rect_ class */
template <typename T>
inline Rect_<T>::Rect_() {}
template <typename T>
inline Rect_<T>::Rect_(T _x, T _y, T _width, T _height) : x(_x), y(_y), width(_width), height(_height) {}
template <typename T>
inline Rect_<T>::Rect_(const Rect_<T>& rect) : x(rect.x), y(rect.y), width(rect.width), height(rect.height) {}
template <typename T>
inline Rect_<T>::Rect_(const Point_<T>& pt, const Size_<T>& sz)
    : x(pt.x), y(pt.y), width(sz.width), height(sz.height) {}
template <typename T>
inline T Rect_<T>::area() {
    return width * height;
}
template <typename T>
inline Size_<T> Rect_<T>::size() {
    return Size_<T>(width, height);
}
template <typename T>
inline Point_<T> Rect_<T>::tl() {
    return Point_<T>(x, y);
}
template <typename T>
inline Point_<T> Rect_<T>::tr() {
    return Point_<T>(x + width, y);
}
template <typename T>
inline Point_<T> Rect_<T>::bl() {
    return Point_<T>(x, y + height);
}
template <typename T>
inline Point_<T> Rect_<T>::br() {
    return Point_<T>(x + width, y + height);
}
template <typename T>
inline bool Rect_<T>::bContains(const Point_<T>& pt) {
    return (pt.x >= x && pt.x < x + width && pt.y >= y && pt.y < y + height);
}
template <typename T>
inline Rect_<T>::~Rect_() {}

typedef Rect_<int> Rect;
//----------------------------------------------------------------------------------------------------//

//----------------------------------------------------------------------------------------------------//
// Template class of Scalar
//----------------------------------------------------------------------------------------------------//
template <int N, typename T>
class Scalar {
   public:
    Scalar() {
// clang-format off
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=val dim=1 complete
        // clang-format on
        assert(N > 0);
    }
    Scalar(T v0) {
// clang-format off
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=val dim=1 complete
        // clang-format on
        assert(N >= 1 && "Scalar must have enough channels for constructor.");
        val[0] = v0;
    }
    Scalar(T v0, T v1) {
// clang-format off
#pragma HLS ARRAY_PARTITION variable=val dim=1 complete
#pragma HLS INLINE
        // clang-format on
        assert(N >= 2 && "Scalar must have enough channels for constructor.");
        val[0] = v0;
        val[1] = v1;
    }
    Scalar(T v0, T v1, T v2) {
// clang-format off
#pragma HLS ARRAY_PARTITION variable=val dim=1 complete
#pragma HLS INLINE
        // clang-format on
        assert(N >= 3 && "Scalar must have enough channels for constructor.");
        val[0] = v0;
        val[1] = v1;
        val[2] = v2;
    }
    Scalar(T v0, T v1, T v2, T v3) {
// clang-format off
#pragma HLS ARRAY_PARTITION variable=val dim=1 complete
#pragma HLS INLINE
        // clang-format on
        assert(N >= 4 && "Scalar must have enough channels for constructor.");
        val[0] = v0;
        val[1] = v1;
        val[2] = v2;
        val[3] = v3;
    }

    void operator=(T value);
    Scalar<N, T> operator+(T value);
    Scalar<N, T> operator+(Scalar<N, T> s);
    Scalar<N, T> operator-(T value);
    Scalar<N, T> operator-(Scalar<N, T> s);
    Scalar<N, T> operator*(T value);
    Scalar<N, T> operator*(Scalar<N, T> s);
    Scalar<N, T> operator/(T value);
    Scalar<N, T> operator/(Scalar<N, T> s);

    T val[N];
};

template <int N, typename T>
void Scalar<N, T>::operator=(T value) {
// clang-format off
#pragma HLS inline
    // clang-format on
    for (int k = 0; k < N; k++) {
// clang-format off
#pragma HLS unroll
        // clang-format on
        val[k] = value;
    }
}

template <int N, typename T>
Scalar<N, T> Scalar<N, T>::operator+(T value) {
// clang-format off
#pragma HLS inline
    // clang-format on
    Scalar<N, T> res;
    for (int k = 0; k < N; k++) {
// clang-format off
#pragma HLS unroll
        // clang-format on
        res.val[k] = val[k] + value;
    }
    return res;
}

template <int N, typename T>
Scalar<N, T> Scalar<N, T>::operator+(Scalar<N, T> s) {
// clang-format off
#pragma HLS inline
    // clang-format on
    Scalar<N, T> res;
    for (int k = 0; k < N; k++) {
// clang-format off
#pragma HLS unroll
        // clang-format on
        res.val[k] = val[k] + s.val[k];
    }
    return res;
}

template <int N, typename T>
Scalar<N, T> Scalar<N, T>::operator-(T value) {
// clang-format off
#pragma HLS inline
    // clang-format on
    Scalar<N, T> res;
    for (int k = 0; k < N; k++) {
// clang-format off
#pragma HLS unroll
        // clang-format on
        res.val[k] = val[k] - value;
    }
    return res;
}

template <int N, typename T>
Scalar<N, T> Scalar<N, T>::operator-(Scalar<N, T> s) {
// clang-format off
#pragma HLS inline
    // clang-format on
    Scalar<N, T> res;
    for (int k = 0; k < N; k++) {
// clang-format off
#pragma HLS unroll
        // clang-format on
        res.val[k] = val[k] - s.val[k];
    }
    return res;
}

template <int N, typename T>
Scalar<N, T> Scalar<N, T>::operator*(T value) {
// clang-format off
#pragma HLS inline
    // clang-format on
    Scalar<N, T> res;
    for (int k = 0; k < N; k++) {
// clang-format off
#pragma HLS unroll
        // clang-format on
        res.val[k] = val[k] * value;
    }
    return res;
}

template <int N, typename T>
Scalar<N, T> Scalar<N, T>::operator*(Scalar<N, T> s) {
// clang-format off
#pragma HLS inline
    // clang-format on
    Scalar<N, T> res;
    for (int k = 0; k < N; k++) {
// clang-format off
#pragma HLS unroll
        // clang-format on
        res.val[k] = val[k] * s.val[k];
    }
    return res;
}

template <int N, typename T>
Scalar<N, T> Scalar<N, T>::operator/(T value) {
// clang-format off
#pragma HLS inline
    // clang-format on
    Scalar<N, T> res;
    for (int k = 0; k < N; k++) {
// clang-format off
#pragma HLS unroll
        // clang-format on
        res.val[k] = val[k] / value;
    }
    return res;
}

template <int N, typename T>
Scalar<N, T> Scalar<N, T>::operator/(Scalar<N, T> s) {
// clang-format off
#pragma HLS inline
    // clang-format on
    Scalar<N, T> res;
    for (int k = 0; k < N; k++) {
// clang-format off
#pragma HLS unroll
        // clang-format on
        res.val[k] = val[k] / s.val[k];
    }
    return res;
}
//----------------------------------------------------------------------------------------------------//

//----------------------------------------------------------------------------------------------------//
// Template class of Mat
//----------------------------------------------------------------------------------------------------//
template <int T, int ROWS, int COLS, int NPC, int XFCVDEPTH = 2>
class Mat {
   public:
    unsigned char allocatedFlag; // flag to mark memory allocation in this class
    int rows, cols, size;        // actual image size
    //	int cols_align_npc;						// cols
    // multiple
    // of
    // NPC

    typedef XF_TNAME(T, NPC) DATATYPE;

#if defined(__SYNTHESIS__) && !defined(__SDA_MEM_MAP__)
    hls::stream<DATATYPE, XFCVDEPTH> data;
#else
    DATATYPE* data;
#endif

    Mat(); // default constructor
    Mat(Size _sz);
    Mat(int _rows, int _cols);
    Mat(int _size, int _rows, int _cols);
    Mat(int _rows, int _cols, void* _data);
    Mat(const Mat&); // copy constructor

    ~Mat();

    Mat& operator=(const Mat&); // Assignment operator
    //  XF_TNAME(T, XF_NPPC1) operator() (unsigned int r, unsigned int c);
    //  XF_CTUNAME(T, NPC) operator() (unsigned int r, unsigned int c, unsigned
    //  int ch);
    XF_TNAME(T, NPC) read(int index);
    float read_float(int index);
    void write(int index, XF_TNAME(T, NPC) val);
    void write_float(int index, float val);

    void init(int _rows, int _cols, bool allocate = true);
    void copyTo(void* fromData);
    unsigned char* copyFrom();

    const int type() const;
    const int depth() const;
    const int channels() const;

    template <int DST_T>
    void convertTo(Mat<DST_T, ROWS, COLS, NPC, XFCVDEPTH>& dst, int otype, double alpha = 1, double beta = 0);
};

template <int T, int ROWS, int COLS, int NPC, int XFCVDEPTH>
const int Mat<T, ROWS, COLS, NPC, XFCVDEPTH>::type() const {
// clang-format off
#pragma HLS inline
    // clang-format on
    return (T);
}

template <int T, int ROWS, int COLS, int NPC, int XFCVDEPTH>
const int Mat<T, ROWS, COLS, NPC, XFCVDEPTH>::depth() const {
// clang-format off
#pragma HLS inline
    // clang-format on
    return XF_DTPIXELDEPTH(T, NPC);
}
template <int T, int ROWS, int COLS, int NPC, int XFCVDEPTH>
const int Mat<T, ROWS, COLS, NPC, XFCVDEPTH>::channels() const {
// clang-format off
#pragma HLS inline
    // clang-format on
    return XF_CHANNELS(T, NPC);
}

template <int T, int ROWS, int COLS, int NPPC, int XFCVDEPTH>
inline void Mat<T, ROWS, COLS, NPPC, XFCVDEPTH>::init(int _rows, int _cols, bool allocate) {
// clang-format off
#pragma HLS inline
    // clang-format on

    assert((_rows > 0) && (_rows <= ROWS) && (_cols > 0) && (_cols <= COLS) &&
           "The number of rows and columns must be less than the template "
           "arguments.");

    rows = _rows;
    cols = _cols;
    size = _rows * ((_cols + NPPC - 1) >> XF_BITSHIFT(NPPC));

    if (allocate) {
#ifndef __SYNTHESIS__
        data = (DATATYPE*)malloc(size * sizeof(DATATYPE));

        if (data == NULL) {
            fprintf(stderr, "\nFailed to allocate memory\n");
        } else {
            allocatedFlag = 1;
        }
#endif
    }
}

/*
template <int T, int ROWS, int COLS, int NPC>
inline XF_TNAME(T, XF_NPPC1) Mat<T, ROWS, COLS, NPC>::operator() (unsigned int
r, unsigned int c) {

    XF_TNAME(T, XF_NPPC1) pix_val;

    unsigned int mask      = 0xFFFFFFFF;
    unsigned int npc_c     = c >> (XF_BITSHIFT(NPC));
    unsigned int npc_cols  = cols >> (XF_BITSHIFT(NPC));
    unsigned int pix_idx   = c & ~(mask << (XF_BITSHIFT(NPC)));

#ifdef __SDSVHLS__
    unsigned int bit_width= XF_DTPIXELDEPTH(T, NPC);

    pix_val = data[(r*npc_cols) + npc_c].range((pix_idx+1)*bit_width-1,
pix_idx*bit_width);
#else
    unsigned int pix_depth = XF_DTPIXELDEPTH(T, NPC) / XF_CHANNELS(T,NPC);

    for (int ch=0; ch < XF_CHANNELS(T,NPC); ch++) {
       pix_val.range((ch+1)*pix_depth-1, ch*pix_depth) = data[r*npc_cols +
npc_c].chnl[ch].range((pix_idx+1)*pix_depth-1, pix_idx*pix_depth);
    }
#endif

    return pix_val;
}

template <int T, int ROWS, int COLS, int NPC>
inline XF_CTUNAME(T, NPC) Mat<T, ROWS, COLS, NPC>::operator() (unsigned int r,
unsigned int c, unsigned int ch) {

    XF_CTUNAME(T, NPC) pix_val;

    unsigned int mask      = 0xFFFFFFFF;
    unsigned int npc_c     = c >> (XF_BITSHIFT(NPC));
    unsigned int npc_cols  = cols >> (XF_BITSHIFT(NPC));
    unsigned int pix_idx   = c & ~(mask << (XF_BITSHIFT(NPC)));

#ifdef __SDSVHLS__
    unsigned int bit_width = XF_DTPIXELDEPTH(T, NPC);
    unsigned int pix_depth = XF_DTPIXELDEPTH(T, NPC) / XF_CHANNELS(T, NPC);

    pix_val = data[(r*npc_cols) + npc_c].range(pix_idx*bit_width +
(ch+1)*pix_depth - 1, pix_idx*bit_width +
ch*pix_depth); #else unsigned int pix_depth = XF_DTPIXELDEPTH(T, NPC) /
XF_CHANNELS(T, NPC);

    pix_val = data[r*npc_cols + npc_c].chnl[ch].range((pix_idx+1)*pix_depth-1,
pix_idx*pix_depth);
#endif

    return pix_val;
}
 */
/*Copy constructor definition*/
template <int T, int ROWS, int COLS, int NPC, int XFCVDEPTH>
inline Mat<T, ROWS, COLS, NPC, XFCVDEPTH>::Mat(const Mat& src) {
    init(src.rows, src.cols);

//	for(int i =0; i< (rows*(cols>>(XF_BITSHIFT(NPC))));++i){
#ifndef __SYNTHESIS__
    for (int i = 0; i < (rows * ((cols + NPC - 1) >> XF_BITSHIFT(NPC))); ++i) {
        data[i] = src.data[i];
    }
#else
//  for (int i = 0; i < (rows * ((cols + NPC - 1) >> XF_BITSHIFT(NPC))); ++i) {
//      data[i] = src.read(i);
//  }
#endif
}

/*Assignment operator definition*/
template <int T, int ROWS, int COLS, int NPC, int XFCVDEPTH>
inline Mat<T, ROWS, COLS, NPC, XFCVDEPTH>& Mat<T, ROWS, COLS, NPC, XFCVDEPTH>::operator=(const Mat& src) {
    if (this == &src) {
        return *this; // For self-assignment cases
    }

    // Cleaning up old data memory if any
    if (data != NULL) {
#ifndef __SYNTHESIS__
        free(data);
#endif
    }

    allocatedFlag = 0;

    init(src.rows, src.cols);

    //	for(int i =0; i< (rows*(cols>>(XF_BITSHIFT(NPC))));++i){
    for (int i = 0; i < (rows * ((cols + NPC - 1) >> XF_BITSHIFT(NPC))); ++i) {
        data[i] = src.data[i];
    }

    return *this;
}

template <int T, int ROWS, int COLS, int NPPC, int XFCVDEPTH>
inline Mat<T, ROWS, COLS, NPPC, XFCVDEPTH>::Mat() {
// clang-format off
#pragma HLS inline
    // clang-format on

    init(ROWS, COLS);
}

template <int T, int ROWS, int COLS, int NPPC, int XFCVDEPTH>
inline Mat<T, ROWS, COLS, NPPC, XFCVDEPTH>::Mat(int _rows, int _cols, void* _data) {
// clang-format off
#pragma HLS inline
// clang-format on

#if defined __SDA_MEM_MAP__ //(__SDSCC__)  && defined (__SYNTHESIS__)
    init(_rows, _cols, false);
    data = (DATATYPE*)_data;
#else
    init(_rows, _cols);
    copyTo(_data);
#endif
}

template <int T, int ROWS, int COLS, int NPPC, int XFCVDEPTH>
inline Mat<T, ROWS, COLS, NPPC, XFCVDEPTH>::Mat(int _rows, int _cols) {
// clang-format off
#pragma HLS inline
    // clang-format on

    init(_rows, _cols);
}

template <int T, int ROWS, int COLS, int NPPC, int XFCVDEPTH>
inline Mat<T, ROWS, COLS, NPPC, XFCVDEPTH>::Mat(Size _sz) {
// clang-format off
#pragma HLS inline
    // clang-format on

    init(_sz.height, _sz.width);
}

template <int T, int ROWS, int COLS, int NPPC, int XFCVDEPTH>
inline XF_TNAME(T, NPPC) Mat<T, ROWS, COLS, NPPC, XFCVDEPTH>::read(int index) {
#if defined(__SYNTHESIS__) && !defined(__SDA_MEM_MAP__)
    return data.read();
#else
    return data[index];
#endif
}

template <int T, int ROWS, int COLS, int NPPC, int XFCVDEPTH>
inline float Mat<T, ROWS, COLS, NPPC, XFCVDEPTH>::read_float(int index) {
    union int2float {
        unsigned I;
        float F;
    };
    int2float val;
    val.I = read(index).to_uint();
    return val.F;
}

template <int T, int ROWS, int COLS, int NPPC, int XFCVDEPTH>
inline void Mat<T, ROWS, COLS, NPPC, XFCVDEPTH>::write(int index, XF_TNAME(T, NPPC) val) {
#if defined(__SYNTHESIS__) && !defined(__SDA_MEM_MAP__)
    data.write(val);
#else
    data[index] = val;
#endif
}

template <int T, int ROWS, int COLS, int NPPC, int XFCVDEPTH>
inline void Mat<T, ROWS, COLS, NPPC, XFCVDEPTH>::write_float(int index, float float_val) {
    float val = float_val;
    ap_uint<32>* val_out = (ap_uint<32>*)(&val);
    write(index, *val_out);
}

// template <int T, int ROWS, int COLS, int NPPC>
// inline void Mat<T, ROWS, COLS, NPPC>::copyTo(void* _input) {
//    #pragma HLS inline
//
//    XF_PTSNAME(T, NPPC)* input = (XF_PTSNAME(T, NPPC)*)_input;
//    XF_CTUNAME(T, NPPC) in_val;
//
//    //	int packcols  = cols >> XF_BITSHIFT(NPPC); 		//Total
//    columns
//    after
//    considering parallelism
//    int pixdepth = XF_PIXELWIDTH(T, NPPC);          // Total bits that make up
//    the pixel
//    int bitdepth = pixdepth / XF_CHANNELS(T, NPPC); // Total bits that make up
//    each channel of the pixel
//    int nppc = XF_NPIXPERCYCLE(NPPC);
//
//    /*	for (int r=0; r < rows; r++) {
//            for (int c=0; c < packcols; c++) {
//                    for (int p=0; p < nppc; p++) {
//                            for (int ch=0; ch < XF_CHANNELS(T,NPPC); ch++) {
//
//                                    if (T == XF_32FC1) {
//                                            in_val = float2ap_uint<
//                                            ap_uint<32> >(input[XF_CHANNELS(T,
// NPPC)*((r*packcols + c)*nppc + p) + ch]); } else { in_val =
// input[XF_CHANNELS(T, NPPC)*((r*packcols + c)*nppc + p) +
// ch];
//                                    }
//#ifdef __SDSVHLS__
//                                    data[r*packcols +
//                                    c].range((p*pixdepth)+(ch+1)*bitdepth-1,
//                                    (p*pixdepth)+ch*bitdepth)
//= in_val; #else data[r*packcols + c].chnl[p][ch] = in_val; #endif
//                            }
//                    }
//            }
//    }
//     */
//    int ncpr = ((cols + NPPC - 1) >> XF_BITSHIFT(NPPC));
//    for (int r = 0; r < rows; r++) {
//        for (int c = 0; c < ncpr; c++) {
//            for (int p = 0; p < nppc; p++) {
//                for (int ch = 0; ch < XF_CHANNELS(T, NPPC); ch++) {
//                    if ((c << XF_BITSHIFT(NPPC) + p) < cols) {
//                        if (T == XF_32FC1) {
//                            in_val = float2ap_uint<ap_uint<32> >(
//                                input[XF_CHANNELS(T, NPPC) * ((r * cols + c) *
//                                nppc + p) + ch]);
//                        } else {
//                            in_val = input[XF_CHANNELS(T, NPPC) * ((r * cols +
//                            c) * nppc + p) + ch];
//                        }
//                    } else {
//                        in_val = 0;
//                    }
//
//#ifdef __SDSVHLS__
//                    data[r * ncpr + c].range((p * pixdepth) + (ch + 1) *
//                    bitdepth - 1, (p * pixdepth) + ch * bitdepth)
//                    =
//                        in_val;
//#else
//                    data[r * ncpr + c].chnl[p][ch] = in_val;
//#endif
//                }
//            }
//        } // c
//    }     // r
//}

template <int T, int ROWS, int COLS, int NPPC, int XFCVDEPTH>
inline void Mat<T, ROWS, COLS, NPPC, XFCVDEPTH>::copyTo(void* _input) {
// clang-format off
#pragma HLS inline
    // clang-format on

    XF_PTSNAME(T, NPPC)* input = (XF_PTSNAME(T, NPPC)*)_input;
    XF_CTUNAME(T, NPPC) in_val;

    int packcols = cols >> XF_BITSHIFT(NPPC);       // Total columns after considering parallelism
    int pixdepth = XF_PIXELWIDTH(T, NPPC);          // Total bits that make up the pixel
    int bitdepth = pixdepth / XF_CHANNELS(T, NPPC); // Total bits that make up each channel of the pixel
    int nppc = XF_NPIXPERCYCLE(NPPC);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < packcols; c++) {
            for (int p = 0; p < nppc; p++) {
                for (int ch = 0; ch < XF_CHANNELS(T, NPPC); ch++) {
                    if (T == XF_32FC1) {
                        in_val = float2ap_uint<ap_uint<32> >(
                            input[XF_CHANNELS(T, NPPC) * ((r * packcols + c) * nppc + p) + ch]);
                    } else {
                        in_val = input[XF_CHANNELS(T, NPPC) * ((r * packcols + c) * nppc + p) + ch];
                    }

                    data[r * packcols + c].range((p * pixdepth) + (ch + 1) * bitdepth - 1,
                                                 (p * pixdepth) + ch * bitdepth) = in_val;
                }
            }
        }
    }
}

template <int T, int ROWS, int COLS, int NPPC, int XFCVDEPTH>
inline unsigned char* Mat<T, ROWS, COLS, NPPC, XFCVDEPTH>::copyFrom() {
// clang-format off
#pragma HLS inline
    // clang-format on

    //	int packcols  = cols >> XF_BITSHIFT(NPPC); //Total columns after
    // considering parallelism
    int pixdepth = XF_PIXELWIDTH(T, NPPC);          // Total bits that make up the pixel
    int bitdepth = pixdepth / XF_CHANNELS(T, NPPC); // Total bits that make up each channel of the pixel
    int nppc = XF_NPIXPERCYCLE(NPPC);

    int cv_nbytes = bitdepth / 8;

    unsigned char* value =
        (unsigned char*)malloc(rows * cols * (XF_CHANNELS(T, NPPC)) * (sizeof(unsigned char)) * cv_nbytes);

    int xf_npc_idx = 0;
    int diff_ptr = 0;
    int xf_ptr = 0;
    int cv_ptr = 0;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            for (int ch = 0; ch < XF_CHANNELS(T, NPPC); ch++) {
                for (int b = 0; b < cv_nbytes; ++b) {
                    value[cv_ptr++] = data[xf_ptr].range((xf_npc_idx * pixdepth) + (ch * bitdepth) + (b + 1) * 8 - 1,
                                                         (xf_npc_idx * pixdepth) + (ch * bitdepth) + b * 8);
                }
            }
            if (xf_npc_idx == nppc - 1) {
                xf_npc_idx = 0;
                xf_ptr++;
            } else {
                xf_npc_idx++;
            }
        }
    }

    return (unsigned char*)value;
}

/* Member functions of Mat class */
template <int T, int ROWS, int COLS, int NPPC, int XFCVDEPTH>
template <int DST_T>
inline void Mat<T, ROWS, COLS, NPPC, XFCVDEPTH>::convertTo(Mat<DST_T, ROWS, COLS, NPPC, XFCVDEPTH>& dst,
                                                           int otype,
                                                           double alpha,
                                                           double beta) {
    assert((XF_CHANNELS(T, NPPC) == 1) && "Multi-channel images not supported");

    XF_TNAME(T, NPPC) tmp_in_pix;
    XF_TNAME(DST_T, NPPC) tmp_out_pix;

    XF_CTUNAME(T, NPPC) in_pix;
    XF_CTUNAME(DST_T, NPPC) out_pix;

    int min, max;

    if (DST_T == XF_8UC1) {
        min = 0;
        max = 255;
    } else if (DST_T == XF_16UC1) {
        min = 0;
        max = 65535;
    } else if (DST_T == XF_16SC1) {
        min = -32768;
        max = 32767;
    } else if (DST_T == XF_32SC1) {
        min = -2147483648;
        max = 2147483647;
    } else {
        assert(1 &&
               "Output image type not supoorted. XF_8UC1, XF_16UC1, XF_16SC1 "
               "and XF_32SC1 are valid");
    }

#define __SATCAST(X) (X >= max ? max : (X < 0 ? 0 : lround(X)))

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j<cols>> (XF_BITSHIFT(NPPC)); j++) {
            int IN_STEP = XF_PIXELDEPTH(XF_DEPTH(T, NPPC));
            int OUT_STEP = XF_PIXELDEPTH(XF_DEPTH(DST_T, NPPC));
            int in_shift = 0;
            int out_shift = 0;

            for (int k = 0; k < (1 << (XF_BITSHIFT(NPPC))); k++) {
#ifdef __SDSVHLS__
                in_pix = data[(i * cols >> (XF_BITSHIFT(NPPC))) + j].range(in_shift + IN_STEP - 1, in_shift);
#else
                in_pix = data[(i * cols >> (XF_BITSHIFT(NPPC))) + j].chnl[k][0];
#endif

                if (otype == XF_CONVERT_16U_TO_8U || otype == XF_CONVERT_16S_TO_8U || otype == XF_CONVERT_32S_TO_8U ||
                    otype == XF_CONVERT_32S_TO_16U || otype == XF_CONVERT_32S_TO_16S) {
                    float tmp = (float)(in_pix * alpha + beta);
                    in_pix = __SATCAST(tmp);

                    if (in_pix < min) in_pix = min;
                    if (in_pix > max) in_pix = max;

                    tmp_out_pix.range(out_shift + OUT_STEP - 1, out_shift) = in_pix;
                } else {
                    if ((((XF_PTNAME(XF_DEPTH(DST_T, NPPC)))in_pix * alpha) + beta) > max) {
                        tmp_out_pix.range(out_shift + OUT_STEP - 1, out_shift) = max;

                    } else if ((((XF_PTNAME(XF_DEPTH(DST_T, NPPC)))in_pix * alpha) + beta) < min) {
                        tmp_out_pix.range(out_shift + OUT_STEP - 1, out_shift) = min;

                    } else {
                        tmp_out_pix.range(out_shift + OUT_STEP - 1, out_shift) = __SATCAST(in_pix * alpha + beta);
                    }
                }

                out_pix = tmp_out_pix.range(out_shift + OUT_STEP - 1, out_shift);
#ifdef __SDSVHLS__
                dst.data[(i * cols >> (XF_BITSHIFT(NPPC))) + j].range(out_shift + OUT_STEP - 1, out_shift) = out_pix;
#else
                dst.data[(i * cols >> (XF_BITSHIFT(NPPC))) + j].chnl[k][0] = out_pix;
#endif

                in_shift = in_shift + IN_STEP;
                out_shift = out_shift + OUT_STEP;
            }
        }
    }
}

template <int SRC_T, int ROWS, int COLS, int NPC, int XFCVDEPTH>
Mat<SRC_T, ROWS, COLS, NPC, XFCVDEPTH>::~Mat() {
#ifndef __SYNTHESIS__
    if (data != NULL && allocatedFlag == 1) {
        free(data);
    }
#endif
}
//----------------------------------------------------------------------------------------------------//

// Template metaprogramming implementation of floor log2 [[
template <int N>
struct log2 {
   public:
    static constexpr int fvalue = 1 + (log2<N / 2>::fvalue);                   // floor value
    static constexpr int cvalue = (N > (1 << fvalue)) ? (fvalue + 1) : fvalue; // ceiling value
};

template <>
struct log2<1> {
   public:
    static constexpr int fvalue = 0;
    static constexpr int cvalue = 0;
};
//]]

/*
 * The purpose of this class is to provide an interator over ap_uint<PTR_WIDTH>
 * object array such
 * that at each iter step one can get data bits corresponding to data to be
 * processed per cycle.
 *
 */
template <int PTR_WIDTH, int T, int ROWS, int COLS, int NPC, int XFCVDEPTH = 2>
class MMIter : public Mat<T, ROWS, COLS, NPC, XFCVDEPTH> {
   public:
    using Mat<T, ROWS, COLS, NPC, XFCVDEPTH>::data;
    using Mat<T, ROWS, COLS, NPC, XFCVDEPTH>::rows;
    using Mat<T, ROWS, COLS, NPC, XFCVDEPTH>::cols;

    static constexpr int XF_BITS_PER_CLOCK = XF_PIXELWIDTH(T, NPC) * XF_NPIXPERCYCLE(NPC);
    static constexpr int COLS_BOUND_PER_NPC = ((COLS + (XF_NPIXPERCYCLE(NPC) - 1)) >> XF_BITSHIFT(NPC));
    static constexpr int LOOPBOUND = ROWS * COLS_BOUND_PER_NPC;
    static constexpr int ADDRBOUND = ((ROWS * COLS * XF_PIXELWIDTH(T, NPC)) + (PTR_WIDTH - 1)) >>
                                     (log2<PTR_WIDTH>::cvalue);
    static constexpr int LAST_BLK_PXL_WIDTH =
        ((COLS >> XF_BITSHIFT(NPC)) == COLS_BOUND_PER_NPC)
            ? XF_BITS_PER_CLOCK
            : XF_PIXELWIDTH(T, NPC) * (COLS - ((COLS >> XF_BITSHIFT(NPC)) << XF_BITSHIFT(NPC)));

    static int cols_npc_aligned(int cols) { return ((cols + (XF_NPIXPERCYCLE(NPC) - 1)) >> XF_BITSHIFT(NPC)); }

    static int last_blk_pxl_width(int cols, int cols_bound_per_npc) {
        return ((cols >> XF_BITSHIFT(NPC)) == cols_bound_per_npc)
                   ? XF_BITS_PER_CLOCK
                   : XF_PIXELWIDTH(T, NPC) * (cols - ((cols >> XF_BITSHIFT(NPC)) << XF_BITSHIFT(NPC)));
    }

    static int loopbound(int rows, int cols) { return rows * cols_npc_aligned(cols); }

    int loopbound() { return rows * cols_npc_aligned(cols); }

    static int addrbound(int rows, int cols) {
        return ((rows * cols * XF_PIXELWIDTH(T, NPC)) + (PTR_WIDTH - 1)) >> (log2<PTR_WIDTH>::cvalue);
    }

    MMIter() : Mat<T, ROWS, COLS, NPC, XFCVDEPTH>() {}

    MMIter(int _rows, int _cols) : Mat<T, ROWS, COLS, NPC, XFCVDEPTH>(_rows, _cols) {}
};

template <int PTR_WIDTH, int T, int ROWS, int COLS, int NPC, int XFCVDEPTH = 2>
class MMIterIn : public MMIter<PTR_WIDTH, T, ROWS, COLS, NPC, XFCVDEPTH> {
   public:
    using Mat<T, ROWS, COLS, NPC, XFCVDEPTH>::data;
    using Mat<T, ROWS, COLS, NPC, XFCVDEPTH>::rows;
    using Mat<T, ROWS, COLS, NPC, XFCVDEPTH>::cols;

    using MMIter<PTR_WIDTH, T, ROWS, COLS, NPC, XFCVDEPTH>::XF_BITS_PER_CLOCK;
    using MMIter<PTR_WIDTH, T, ROWS, COLS, NPC, XFCVDEPTH>::ADDRBOUND;
    using MMIter<PTR_WIDTH, T, ROWS, COLS, NPC, XFCVDEPTH>::COLS_BOUND_PER_NPC;
    using MMIter<PTR_WIDTH, T, ROWS, COLS, NPC, XFCVDEPTH>::LAST_BLK_PXL_WIDTH;
    using MMIter<PTR_WIDTH, T, ROWS, COLS, NPC, XFCVDEPTH>::LOOPBOUND;

   private:
    static void Axi2AxiStream(ap_uint<PTR_WIDTH>* din,
                              hls::stream<ap_uint<PTR_WIDTH> >& dout,
                              ap_uint<log2<ADDRBOUND>::cvalue + 1>& addrbound) {
        ap_uint<log2<ADDRBOUND>::cvalue + 1> i;
    MMIterInLoop1:
        for (i = 0; i < addrbound; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=ADDRBOUND
#pragma HLS PIPELINE
            // clang-format on
            dout.write(din[i]);
        }
    }

    static void Axi2AxiStream(hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >& din,
                              hls::stream<ap_uint<PTR_WIDTH> >& dout,
                              ap_uint<log2<ADDRBOUND>::cvalue + 1>& addrbound) {
        ap_uint<log2<ADDRBOUND>::cvalue + 1> i;
    MMIterInLoop3:
        for (i = 0; i < addrbound; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=ADDRBOUND
#pragma HLS PIPELINE
            // clang-format on
            ap_axiu<PTR_WIDTH, 0, 0, 0> v = din.read();
            dout.write(v.data);
        }
    }

    static void AxiStream2MatStream(hls::stream<ap_uint<PTR_WIDTH> >& din,
                                    hls::stream<ap_uint<XF_BITS_PER_CLOCK>, XFCVDEPTH>& dout,
                                    int rows,
                                    int cols_bound_per_npc,
                                    int last_blk_width,
                                    int stride = -1) {
        int stride_bound_per_npc, strideBased_last_blk_width;

        if (stride == -1) {
            stride_bound_per_npc = cols_bound_per_npc;
            strideBased_last_blk_width = last_blk_width;
        } else {
            stride_bound_per_npc = MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::cols_npc_aligned(stride);
            strideBased_last_blk_width =
                MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::last_blk_pxl_width(stride, stride_bound_per_npc);
        }

        int rd_cnt = 0;

        int rem = 0;
        ap_uint<PTR_WIDTH> val = 0;
        int i;
        int j = 0;
        int bound = rows * stride_bound_per_npc;
    MMIterInLoopRow:
        for (i = 0; i < bound; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=LOOPBOUND
#pragma HLS PIPELINE
            // clang-format on
            bool bLast = (j == (stride_bound_per_npc - 1));
            int xf_bits_per_clock = bLast ? strideBased_last_blk_width : XF_BITS_PER_CLOCK;
            int ptr_width_minus = bLast ? (PTR_WIDTH - strideBased_last_blk_width) : (PTR_WIDTH - XF_BITS_PER_CLOCK);
            int ptr_width_plus = bLast ? (PTR_WIDTH + strideBased_last_blk_width) : (PTR_WIDTH + XF_BITS_PER_CLOCK);

            ap_uint<XF_BITS_PER_CLOCK> localbuffer = 0;
            if (rem < xf_bits_per_clock) {
                if (rem != 0) {
                    localbuffer.range(rem - 1, 0) = val.range(PTR_WIDTH - 1, (PTR_WIDTH - rem));
                }
                val = din.read();
                rd_cnt++;
                localbuffer.range((xf_bits_per_clock - 1), rem) = val.range(((xf_bits_per_clock - 1) - rem), 0);
                rem = ptr_width_minus + rem;
            } else {
                localbuffer = val.range(((ptr_width_plus - 1) - rem), (PTR_WIDTH - rem));
                rem = rem - xf_bits_per_clock;
            }

            bool bLast_width = (j == (cols_bound_per_npc - 1));
            ap_uint<XF_BITS_PER_CLOCK> localbuffer2 = 0;
            if (bLast_width == 0)
                localbuffer2 = localbuffer;
            else
                localbuffer2 = localbuffer.range(last_blk_width - 1, 0);

            if (j < cols_bound_per_npc) dout.write(localbuffer2);
            j = (bLast) ? 0 : (j + 1);
        }
    }

    static void MatStream2Mat(hls::stream<ap_uint<XF_BITS_PER_CLOCK>, XFCVDEPTH>& din,
                              ap_uint<XF_BITS_PER_CLOCK>* dout,
                              int rows,
                              int cols_bound_per_npc) {
        int i;
        int bound = rows * cols_bound_per_npc;
    MMIterInLoop2:
        for (i = 0; i < bound; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=LOOPBOUND
#pragma HLS PIPELINE
            // clang-format on
            dout[i] = din.read();
        }
    }

    static void AxiStream2Mat(hls::stream<ap_uint<PTR_WIDTH> >& din,
                              hls::stream<ap_uint<XF_BITS_PER_CLOCK>, XFCVDEPTH>& dout,
                              int rows = ROWS,
                              int cols = COLS) {
// clang-format off
#pragma HLS DATAFLOW
        // clang-format on
        int cols_bound_per_npc = MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::cols_npc_aligned(cols);
        int last_blk_width = MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::last_blk_pxl_width(cols, cols_bound_per_npc);
        AxiStream2MatStream(din, dout, rows, cols_bound_per_npc, last_blk_width);
    }

    static void AxiStream2Mat(hls::stream<ap_uint<PTR_WIDTH> >& din,
                              ap_uint<XF_BITS_PER_CLOCK>* dout,
                              int rows = ROWS,
                              int cols = COLS,
                              int stride = -1) {
// clang-format off
#pragma HLS DATAFLOW
        // clang-format on
        hls::stream<ap_uint<XF_BITS_PER_CLOCK>, XFCVDEPTH> ldata;
        int cols_bound_per_npc = MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::cols_npc_aligned(cols);
        int last_blk_width = MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::last_blk_pxl_width(cols, cols_bound_per_npc);
        AxiStream2MatStream(din, ldata, rows, cols_bound_per_npc, last_blk_width, stride);
        MatStream2Mat(ldata, dout, rows, cols_bound_per_npc);
    }

    static void Axi2Mat(ap_uint<PTR_WIDTH>* din,
                        hls::stream<ap_uint<XF_BITS_PER_CLOCK>, XFCVDEPTH>& dout,
                        int rows = ROWS,
                        int cols = COLS,
                        int stride = -1) {
// clang-format off
#pragma HLS DATAFLOW
        // clang-format on
        hls::stream<ap_uint<PTR_WIDTH> > ldata;
        int cols_bound_per_npc = MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::cols_npc_aligned(cols);
        int last_blk_width = MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::last_blk_pxl_width(cols, cols_bound_per_npc);

        int cols_tmp;
        if (stride == -1)
            cols_tmp = cols;
        else
            cols_tmp = stride;

        ap_uint<log2<ADDRBOUND>::cvalue + 1> axibound =
            MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::addrbound(rows, cols_tmp);
        Axi2AxiStream(din, ldata, axibound);
        AxiStream2MatStream(ldata, dout, rows, cols_bound_per_npc, last_blk_width, stride);
    }

    static void Axi2Mat(hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >& din,
                        hls::stream<ap_uint<XF_BITS_PER_CLOCK>, XFCVDEPTH>& dout,
                        int rows = ROWS,
                        int cols = COLS,
                        int stride = -1) {
// clang-format off
#pragma HLS DATAFLOW
        // clang-format on
        hls::stream<ap_uint<PTR_WIDTH> > ldata;
        int cols_bound_per_npc = MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::cols_npc_aligned(cols);
        int last_blk_width = MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::last_blk_pxl_width(cols, cols_bound_per_npc);

        int cols_tmp;
        if (stride == -1)
            cols_tmp = cols;
        else
            cols_tmp = stride;

        ap_uint<log2<ADDRBOUND>::cvalue + 1> axibound =
            MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::addrbound(rows, cols_tmp);
        Axi2AxiStream(din, ldata, axibound);
        AxiStream2MatStream(ldata, dout, rows, cols_bound_per_npc, last_blk_width);
    }

    static void Axi2Mat(
        ap_uint<PTR_WIDTH>* din, ap_uint<XF_BITS_PER_CLOCK>* dout, int rows = ROWS, int cols = COLS, int stride = -1) {
// clang-format off
#pragma HLS DATAFLOW
        // clang-format on
        hls::stream<ap_uint<PTR_WIDTH> > ldata;

        int cols_tmp;
        if (stride == -1)
            cols_tmp = cols;
        else
            cols_tmp = stride;

        ap_uint<log2<ADDRBOUND>::cvalue + 1> axibound =
            MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::addrbound(rows, cols_tmp);
        Axi2AxiStream(din, ldata, axibound);
        AxiStream2Mat(ldata, dout, rows, cols, stride);
    }

    static void Axi2Mat(hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >& din,
                        ap_uint<XF_BITS_PER_CLOCK>* dout,
                        int rows = ROWS,
                        int cols = COLS) {
// clang-format off
#pragma HLS DATAFLOW
        // clang-format on
        hls::stream<ap_uint<PTR_WIDTH> > ldata;
        ap_uint<log2<ADDRBOUND>::cvalue + 1> axibound = MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::addrbound(rows, cols);
        Axi2AxiStream(din, ldata, axibound);
        AxiStream2Mat(ldata, dout, rows, cols);
    }

   public:
    MMIterIn(ap_uint<PTR_WIDTH>* d) : MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>() { Axi2Mat(d, data); }

    MMIterIn(ap_uint<PTR_WIDTH>* d, int _rows, int _cols) : MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>(_rows, _cols) {
        Axi2Mat(d, data, rows, cols);
    }

    MMIterIn(hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >& d) : MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>() {
        Axi2Mat(d, data);
    }

    MMIterIn(hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >& d, int _rows, int _cols)
        : MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>(_rows, _cols) {
        Axi2Mat(d, data, rows, cols);
    }

    MMIterIn(hls::stream<ap_uint<PTR_WIDTH> >& d) : MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>() { AxiStream2Mat(d, data); }

    MMIterIn(hls::stream<ap_uint<PTR_WIDTH> >& d, int _rows, int _cols)
        : MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>(_rows, _cols) {
        AxiStream2Mat(d, data, rows, cols);
    }

    inline static ap_uint<XF_BITS_PER_CLOCK> read(hls::stream<ap_uint<XF_BITS_PER_CLOCK> >& din, int index) {
        return din.read();
    }

    inline static ap_uint<XF_BITS_PER_CLOCK> read(ap_uint<XF_BITS_PER_CLOCK>* din, int index) { return din[index]; }

    ap_uint<XF_BITS_PER_CLOCK> read(int index) { return read(data, index); }

    static void Array2xfMat(ap_uint<PTR_WIDTH>* srcPtr,
                            xf::cv::Mat<T, ROWS, COLS, NPC, XFCVDEPTH>& dstMat,
                            int stride = -1) {
        Axi2Mat(srcPtr, dstMat.data, dstMat.rows, dstMat.cols, stride);
    }

    static void axiStrm2xfMat(hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >& srcPtr,
                              xf::cv::Mat<T, ROWS, COLS, NPC, XFCVDEPTH>& dstMat) {
        Axi2Mat(srcPtr, dstMat.data, dstMat.rows, dstMat.cols);
    }
};

template <int PTR_WIDTH, int T, int ROWS, int COLS, int NPC, int FILLZERO = 1, int XFCVDEPTH = 2>
class MMIterOut : public MMIter<PTR_WIDTH, T, ROWS, COLS, NPC> {
   public:
    using Mat<T, ROWS, COLS, NPC, XFCVDEPTH>::data;
    using Mat<T, ROWS, COLS, NPC, XFCVDEPTH>::rows;
    using Mat<T, ROWS, COLS, NPC, XFCVDEPTH>::cols;

    using MMIter<PTR_WIDTH, T, ROWS, COLS, NPC, XFCVDEPTH>::XF_BITS_PER_CLOCK;
    using MMIter<PTR_WIDTH, T, ROWS, COLS, NPC, XFCVDEPTH>::ADDRBOUND;
    using MMIter<PTR_WIDTH, T, ROWS, COLS, NPC, XFCVDEPTH>::COLS_BOUND_PER_NPC;
    using MMIter<PTR_WIDTH, T, ROWS, COLS, NPC, XFCVDEPTH>::LAST_BLK_PXL_WIDTH;
    using MMIter<PTR_WIDTH, T, ROWS, COLS, NPC, XFCVDEPTH>::LOOPBOUND;

   private:
    static void Mat2MatStream(ap_uint<XF_BITS_PER_CLOCK>* din,
                              hls::stream<ap_uint<XF_BITS_PER_CLOCK>, XFCVDEPTH>& dout,
                              int rows,
                              int cols_bound_per_npc) {
        int i;
        int bound = rows * cols_bound_per_npc;
    MMIterOutLoop1:
        for (i = 0; i < bound; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=LOOPBOUND
#pragma HLS PIPELINE
            // clang-format on
            dout.write(din[i]);
        }
    }

    static void MatStream2AxiStream(hls::stream<ap_uint<XF_BITS_PER_CLOCK>, XFCVDEPTH>& din,
                                    hls::stream<ap_uint<PTR_WIDTH> >& dout,
                                    int rows,
                                    int cols_bound_per_npc,
                                    int last_blk_width,
                                    int stride = -1) {
        int strideBased_cols_bound_per_npc;
        if (stride == -1 || FILLZERO == 0) {
            strideBased_cols_bound_per_npc = cols_bound_per_npc;
        } else {
            strideBased_cols_bound_per_npc = MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::cols_npc_aligned(stride);
        }

        ap_uint<log2<PTR_WIDTH>::cvalue + 1> filled = 0; // valid bits remaining in current buffer
        ap_uint<PTR_WIDTH> localbuffer = 0;
        int i;
        int j;

    MMIterOutRow:
        for (i = 0; i < rows; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        MMIterOutCol:
            for (j = 0; j < strideBased_cols_bound_per_npc; j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=COLS_BOUND_PER_NPC
#pragma HLS PIPELINE
                // clang-format on

                bool bLast = (j == (cols_bound_per_npc - 1));
                int xf_bits_per_clock = bLast ? last_blk_width : XF_BITS_PER_CLOCK;
                ap_uint<PTR_WIDTH> val;
                if (j < cols_bound_per_npc || FILLZERO == 0)
                    val = din.read();
                else
                    val = 0;

                ap_uint<PTR_WIDTH> tempval = (val << filled);
                localbuffer = (localbuffer | tempval);
                ap_uint<log2<PTR_WIDTH>::cvalue + 1> filled_next;

                if (filled < (PTR_WIDTH - xf_bits_per_clock)) {
                    filled_next = filled + xf_bits_per_clock;
                } else if (j == cols_bound_per_npc - 1 && FILLZERO == 0) {
                    dout.write(localbuffer);
                    localbuffer = 0;
                    filled_next = 0;
                } else {
                    dout.write(localbuffer);
                    localbuffer = (val >> (PTR_WIDTH - filled));
                    filled_next = filled + (xf_bits_per_clock - PTR_WIDTH);
                }

                filled = filled_next;
            }
        }

        if (filled != 0) {
            dout.write(localbuffer);
        }
    }

    static void Mat2AxiStream(hls::stream<ap_uint<XF_BITS_PER_CLOCK>, XFCVDEPTH>& din,
                              hls::stream<ap_uint<PTR_WIDTH> >& dout,
                              int rows = ROWS,
                              int cols = COLS,
                              int stride = -1) {
// clang-format off
#pragma HLS DATAFLOW
        // clang-format on
        int cols_bound_per_npc = MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::cols_npc_aligned(cols);
        int last_blk_width = MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::last_blk_pxl_width(cols, cols_bound_per_npc);
        MatStream2AxiStream(din, dout, rows, cols_bound_per_npc, last_blk_width, stride);
    }

    static void Mat2AxiStream(ap_uint<XF_BITS_PER_CLOCK>* din,
                              hls::stream<ap_uint<PTR_WIDTH> >& dout,
                              int rows = ROWS,
                              int cols = COLS,
                              int stride = -1) {
// clang-format off
#pragma HLS DATAFLOW
        // clang-format on
        hls::stream<ap_uint<XF_BITS_PER_CLOCK>, XFCVDEPTH> ldata;
        int cols_bound_per_npc = MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::cols_npc_aligned(cols);
        int last_blk_width = MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::last_blk_pxl_width(cols, cols_bound_per_npc);
        Mat2MatStream(din, ldata, rows, cols_bound_per_npc);
        MatStream2AxiStream(ldata, dout, rows, cols_bound_per_npc, last_blk_width, stride);
    }

    static void AxiStream2Axi(hls::stream<ap_uint<PTR_WIDTH> >& din,
                              ap_uint<PTR_WIDTH>* dout,
                              ap_uint<log2<ADDRBOUND>::cvalue + 1>& addrbound) {
        ap_uint<log2<ADDRBOUND>::cvalue + 1> i;

    MMIterOutLoop2:
        for (i = 0; i < addrbound; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=ADDRBOUND
#pragma HLS PIPELINE
            // clang-format on
            dout[i] = din.read();
        }
    }

    static void AxiStream2Axi(hls::stream<ap_uint<PTR_WIDTH> >& din,
                              hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >& dout,
                              ap_uint<log2<ADDRBOUND>::cvalue + 1>& addrbound) {
        ap_uint<log2<ADDRBOUND>::cvalue + 1> i;

    MMIterOutLoop3:
        for (i = 0; i < addrbound; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=ADDRBOUND
#pragma HLS PIPELINE
            // clang-format on
            ap_axiu<PTR_WIDTH, 0, 0, 0> v;
            v.data = din.read();
            dout.write(v);
        }
    }

    static void Mat2Axi(hls::stream<ap_uint<XF_BITS_PER_CLOCK>, XFCVDEPTH>& din,
                        ap_uint<PTR_WIDTH>* dout,
                        int rows = ROWS,
                        int cols = COLS,
                        int stride = -1) {
// clang-format off
#pragma HLS DATAFLOW
        // clang-format on

        int cols_tmp;
        if (stride == -1)
            cols_tmp = cols;
        else
            cols_tmp = stride;

        hls::stream<ap_uint<PTR_WIDTH> > ldata;
        ap_uint<log2<ADDRBOUND>::cvalue + 1> axibound =
            MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::addrbound(rows, cols_tmp);

        Mat2AxiStream(din, ldata, rows, cols, stride);
        if (FILLZERO == 1)
            AxiStream2Axi(ldata, dout, axibound);
        else {
            ap_uint<log2<ADDRBOUND>::cvalue + 1> axibound_1row =
                MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::addrbound(1, cols);
            ap_uint<log2<ADDRBOUND>::cvalue + 1> offset_1row =
                MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::addrbound(1, stride);
            for (int rowIdx = 0, offset = 0; rowIdx < rows; rowIdx++, offset += offset_1row) {
                AxiStream2Axi(ldata, dout + offset, axibound_1row);
            }
        }
    }

    static void Mat2Axi(
        ap_uint<XF_BITS_PER_CLOCK>* din, ap_uint<PTR_WIDTH>* dout, int rows = ROWS, int cols = COLS, int stride = -1) {
// clang-format off
#pragma HLS DATAFLOW
        // clang-format on

        int cols_tmp;
        if (stride == -1)
            cols_tmp = cols;
        else
            cols_tmp = stride;

        hls::stream<ap_uint<PTR_WIDTH> > ldata;
        ap_uint<log2<ADDRBOUND>::cvalue + 1> axibound =
            MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::addrbound(rows, cols_tmp);

        Mat2AxiStream(din, ldata, rows, cols, stride);
        if (FILLZERO == 1)
            AxiStream2Axi(ldata, dout, axibound);
        else {
            ap_uint<log2<ADDRBOUND>::cvalue + 1> axibound_1row =
                MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::addrbound(1, cols);
            ap_uint<log2<ADDRBOUND>::cvalue + 1> offset_1row =
                MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::addrbound(1, stride);

            for (int rowIdx = 0, offset = 0; rowIdx < rows; rowIdx++, offset += offset_1row) {
                AxiStream2Axi(ldata, dout + offset, axibound_1row);
            }
        }
    }

    static void Mat2Axi(hls::stream<ap_uint<XF_BITS_PER_CLOCK>, XFCVDEPTH>& din,
                        hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >& dout,
                        int rows = ROWS,
                        int cols = COLS,
                        int stride = -1) {
// clang-format off
#pragma HLS DATAFLOW
        // clang-format on

        int cols_tmp;
        if (stride == -1)
            cols_tmp = cols;
        else
            cols_tmp = stride;

        hls::stream<ap_uint<PTR_WIDTH> > ldata;
        ap_uint<log2<ADDRBOUND>::cvalue + 1> axibound =
            MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::addrbound(rows, cols_tmp);

        Mat2AxiStream(din, ldata, rows, cols, stride);
        if (FILLZERO == 1)
            AxiStream2Axi(ldata, dout, axibound);
        else {
            ap_uint<log2<ADDRBOUND>::cvalue + 1> axibound_1row =
                MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::addrbound(1, cols);
            ap_uint<log2<ADDRBOUND>::cvalue + 1> offset_1row =
                MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::addrbound(1, stride);
            for (int rowIdx = 0, offset = 0; rowIdx < rows; rowIdx++, offset += offset_1row) {
                AxiStream2Axi(ldata, dout + offset, axibound_1row);
            }
        }
    }

    static void Mat2Axi(ap_uint<XF_BITS_PER_CLOCK>* din,
                        hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >& dout,
                        int rows = ROWS,
                        int cols = COLS,
                        int stride = -1) {
// clang-format off
#pragma HLS DATAFLOW
        // clang-format on

        int cols_tmp;
        if (stride == -1)
            cols_tmp = cols;
        else
            cols_tmp = stride;

        hls::stream<ap_uint<PTR_WIDTH> > ldata;
        ap_uint<log2<ADDRBOUND>::cvalue + 1> axibound =
            MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::addrbound(rows, cols_tmp);
        Mat2AxiStream(din, ldata, rows, cols, stride);

        if (FILLZERO == 1)
            AxiStream2Axi(ldata, dout, axibound);
        else {
            ap_uint<log2<ADDRBOUND>::cvalue + 1> axibound_1row =
                MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::addrbound(1, cols);
            ap_uint<log2<ADDRBOUND>::cvalue + 1> offset_1row =
                MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>::addrbound(1, stride);
            for (int rowIdx = 0, offset = 0; rowIdx < rows; rowIdx++, offset += offset_1row) {
                AxiStream2Axi(ldata, dout + offset, axibound_1row);
            }
        }
    }

   public:
    MMIterOut() : MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>() {}

    MMIterOut(int _rows, int _cols) : MMIter<PTR_WIDTH, T, ROWS, COLS, NPC>(_rows, _cols) {}

    inline static void write(hls::stream<ap_uint<XF_BITS_PER_CLOCK> >& dout,
                             ap_uint<XF_BITS_PER_CLOCK>& val,
                             int index) {
        dout.write(val);
    }

    inline static void write(ap_uint<XF_BITS_PER_CLOCK>* dout, ap_uint<XF_BITS_PER_CLOCK>& val, int index) {
        dout[index] = val;
    }

    void write(ap_uint<XF_BITS_PER_CLOCK>& val, int index) { write(data, val, index); }

    void transfer(ap_uint<PTR_WIDTH>* dout) { Mat2Axi(data, dout); }

    void transfer(ap_uint<PTR_WIDTH>* dout, int rows, int cols) { Mat2Axi(data, dout, rows, cols); }

    void transfer(hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >& dout) { Mat2Axi(data, dout); }

    void transfer(hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >& dout, int rows, int cols) {
        Mat2Axi(data, dout, rows, cols);
    }

    void transfer(hls::stream<ap_uint<PTR_WIDTH> >& dout) { Mat2AxiStream(data, dout); }

    void transfer(hls::stream<ap_uint<PTR_WIDTH> >& dout, int rows, int cols) { Mat2AxiStream(data, dout, rows, cols); }

    static void xfMat2Array(xf::cv::Mat<T, ROWS, COLS, NPC, XFCVDEPTH>& srcMat,
                            ap_uint<PTR_WIDTH>* dstPtr,
                            int stride = -1) {
        Mat2Axi(srcMat.data, dstPtr, srcMat.rows, srcMat.cols, stride);
    }

    static void xfMat2axiStrm(xf::cv::Mat<T, ROWS, COLS, NPC, XFCVDEPTH>& srcMat,
                              hls::stream<ap_uint<PTR_WIDTH> >& dstPtr) {
        Mat2Axi(srcMat.data, dstPtr, srcMat.rows, srcMat.cols);
    }
};

} // namespace cv
} // namespace xf

#endif // _XF_STRUCTS_H_
