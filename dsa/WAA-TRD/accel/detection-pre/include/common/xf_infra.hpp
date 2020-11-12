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

#ifndef _XF_INFRA_H_
#define _XF_INFRA_H_

#ifndef __cplusplus
#error C++ is needed to use this file!
#endif

#include <stdio.h>
#include <assert.h>
#include "xf_types.hpp"
#include "hls_stream.h"
#include "xf_axi_sdata.hpp"
#include "common/xf_axi_io.hpp"

namespace xf {
namespace cv {
/* reading data from scalar and write into img*/
template <int SRC_T, int ROWS, int COLS, int NPC>
void write(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& img,
           xf::cv::Scalar<XF_CHANNELS(SRC_T, NPC), XF_TNAME(SRC_T, NPC)> s,
           int ind) {
// clang-format off
    #pragma HLS inline
    // clang-format on

    img.write(ind, s.val[0]);
}
/* reading data from scalar and write into img*/
template <int SRC_T, int ROWS, int COLS, int NPC>
void fetchingmatdata(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& img,
                     xf::cv::Scalar<XF_CHANNELS(SRC_T, NPC), XF_TNAME(SRC_T, NPC)> s,
                     int val) {
// clang-format off
    #pragma HLS inline
    // clang-format on
    write(img, s, val);
}
/* reading data from img and writing onto scalar variable*/
template <int SRC_T, int ROWS, int COLS, int NPC>
xf::cv::Scalar<XF_CHANNELS(SRC_T, NPC), XF_TNAME(SRC_T, NPC)> read(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& img,
                                                                   int index) {
// clang-format off
    #pragma HLS inline
    // clang-format on

    xf::cv::Scalar<XF_CHANNELS(SRC_T, NPC), XF_TNAME(SRC_T, NPC)> scl;
    scl.val[0] = img.read(index);

    return scl;
}
/* reading data from img and writing onto scalar variable*/
template <int SRC_T, int ROWS, int COLS, int NPC>
void fillingdata(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& img,
                 xf::cv::Scalar<XF_CHANNELS(SRC_T, NPC), XF_TNAME(SRC_T, NPC)>& s,
                 int index) {
// clang-format off
    #pragma HLS inline
    // clang-format on

    s = read(img, index);
}

#define HLS_CN_MAX 512
#define HLS_CN_SHIFT 11
#define HLS_DEPTH_MAX (1 << HLS_CN_SHIFT)

#define HLS_MAT_CN_MASK ((HLS_CN_MAX - 1) << HLS_CN_SHIFT)
#define HLS_MAT_CN(flags) ((((flags)&HLS_MAT_CN_MASK) >> HLS_CN_SHIFT) + 1)
#define HLS_MAT_TYPE_MASK (HLS_DEPTH_MAX * HLS_CN_MAX - 1)
#define HLS_MAT_TYPE(flags) ((flags)&HLS_MAT_TYPE_MASK)

#define ERROR_IO_EOL_EARLY (1 << 0)
#define ERROR_IO_EOL_LATE (1 << 1)
#define ERROR_IO_SOF_EARLY (1 << 0)
#define ERROR_IO_SOF_LATE (1 << 1)

/*
Unpack a AXI video stream into a xf::cv::Mat<> object
 *input: AXI_video_strm
 *output: img
 */

template <int W, int T, int ROWS, int COLS, int NPC>
int AXIvideo2xfMat(hls::stream<ap_axiu<W, 1, 1, 1> >& AXI_video_strm, xf::cv::Mat<T, ROWS, COLS, NPC>& img) {
// clang-format off
    #pragma HLS inline
    // clang-format on

    int res = 0, val = 0, depth;
    ap_axiu<W, 1, 1, 1> axi;
    xf::cv::Scalar<XF_CHANNELS(T, NPC), XF_TNAME(T, NPC)> pix;
    depth = XF_WORDDEPTH(XF_WORDWIDTH(T, NPC));
    //    HLS_SIZE_T rows = img.rows;
    //    HLS_SIZE_T cols = img.cols;
    int rows = img.rows;
    int cols = img.cols;
    assert(rows <= ROWS);
    assert(cols <= COLS);
    bool sof = 0;
loop_wait_for_start:
    while (!sof) { // checking starting of frame
                   // clang-format off
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount avg=0 max=0
        // clang-format on
        AXI_video_strm >> axi;
        sof = axi.user.to_int();
    }
loop_height:
    for (int i = 0; i < rows; i++) {
        bool eol = 0;
    loop_width:
        for (int j = 0; j < (cols / NPC); j++) {
// clang-format off
            #pragma HLS loop_flatten off
            #pragma HLS pipeline II=1
            // clang-format on
            if (sof || eol) {
                sof = 0;
                eol = axi.last.to_int();
            } else {
                AXI_video_strm >> axi; // If we didn't reach EOL, then read the next pixel

                eol = axi.last.to_int();
                bool user = axi.user.to_int();
                if (user) {
                    res |= ERROR_IO_SOF_EARLY;
                }
            }
            if (eol && (j != cols - 1)) { // checking end of each row
                res |= ERROR_IO_EOL_EARLY;
            }
            // All channels are merged in cvMat2AXIVideoxf function
            xf::cv::AXIGetBitFields(axi, 0, depth, pix.val[0]);

            fetchingmatdata<T, ROWS, COLS, NPC>(img, pix, val);
            val++;
        }
    loop_wait_for_eol:
        while (!eol) {
// clang-format off
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount avg=0 max=0
            // clang-format on
            // Keep reading until we get to EOL
            AXI_video_strm >> axi;
            eol = axi.last.to_int();
            res |= ERROR_IO_EOL_LATE;
        }
    }
    return res;
}

// Pack the data of a xf::cv::Mat<> object into an AXI Video stream
/*
 *  input: img
 *  output: AXI_video_stream
 */
template <int W, int T, int ROWS, int COLS, int NPC>
int xfMat2AXIvideo(xf::cv::Mat<T, ROWS, COLS, NPC>& img, hls::stream<ap_axiu<W, 1, 1, 1> >& AXI_video_strm) {
// clang-format off
    #pragma HLS inline
    // clang-format on
    int res = 0, index = 0, depth;
    xf::cv::Scalar<XF_CHANNELS(T, NPC), XF_TNAME(T, NPC)> pix;
    ap_axiu<W, 1, 1, 1> axi;
    depth = XF_WORDDEPTH(XF_WORDWIDTH(T, NPC)); // 8;// HLS_TBITDEPTH(T);

    // std::cout << W << " " << depth << " " << HLS_MAT_CN(T) << "\n";
    assert(W >= depth * HLS_MAT_CN(T) &&
           "Bit-Width of AXI stream must be greater than the total number of bits in a pixel");
    //    HLS_SIZE_T rows = img.rows;
    //    HLS_SIZE_T cols = img.cols;
    int rows = img.rows;
    int cols = img.cols;
    assert(rows <= ROWS);
    assert(cols <= COLS);
    bool sof = 1;
loop_height:
    for (int i = 0; i < rows; i++) {
    loop_width:
        for (int j = 0; j < (cols / NPC); j++) {
// clang-format off
            #pragma HLS loop_flatten off
            #pragma HLS pipeline II=1
            // clang-format on
            if (sof) { // checking the start of frame
                axi.user = 1;
                sof = 0;

            } else {
                axi.user = 0;
            }
            if (j == (cols - 1)) { // enabling the last signat at end each row
                axi.last = 1;
            } else {
                axi.last = 0;
            }
            fillingdata<T, ROWS, COLS, NPC>(img, pix, index); // reading data from img writing into scalar pix
            index++;
            axi.data = -1;

            xf::cv::AXISetBitFields(axi, 0, depth, pix.val[0]); // assigning the pix value to AXI data structure
            axi.keep = -1;
            AXI_video_strm << axi; // writing axi data into AXI stream
        }
    }
    return res;
}
} // namespace cv
} // namespace xf

#endif
