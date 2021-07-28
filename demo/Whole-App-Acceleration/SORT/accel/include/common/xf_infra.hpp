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
#include "ap_axi_sdata.h"
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

template <int W, int TYPE, int ROWS, int COLS, int NPPC>
int AXIvideo2xfMat(hls::stream<ap_axiu<W, 1, 1, 1> >& AXI_video_strm, xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& img) {
    ap_axiu<W, 1, 1, 1> axi;
    int res = 0;

    const int m_pix_width = XF_PIXELWIDTH(TYPE, NPPC) * XF_NPIXPERCYCLE(NPPC);

    int rows = img.rows;
    int cols = img.cols >> XF_BITSHIFT(NPPC);
    int idx = 0;

    assert(img.rows <= ROWS);
    assert(img.cols <= COLS);

    bool start = false;
    bool last = false;

loop_start_hunt:
    while (!start) {
// clang-format off
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount avg=1 max=1
        // clang-format on

        AXI_video_strm >> axi;
        start = axi.user.to_bool();
    }

loop_row_axi2mat:
    for (int i = 0; i < rows; i++) {
        last = false;

    loop_col_zxi2mat:
        for (int j = 0; j < cols; j++) {
// clang-format off
#pragma HLS loop_flatten off
#pragma HLS pipeline II=1
            // clang-format on

            if (start || last) {
                start = false;
            } else {
                AXI_video_strm >> axi;

                bool user = axi.user.to_int();
                if (user) {
                    res |= ERROR_IO_SOF_EARLY;
                }
            }
            if (last && (j != img.cols - 1)) { // checking end of each row
                res |= ERROR_IO_EOL_EARLY;
            }

            last = axi.last.to_bool();

            img.write(idx++, axi.data(m_pix_width - 1, 0));
        }

    loop_last_hunt:
        while (!last) {
// clang-format off
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount avg=1 max=1
            // clang-format on

            AXI_video_strm >> axi;
            last = axi.last.to_bool();
            res |= ERROR_IO_EOL_LATE;
        }
    }

    return res;
}

// Pack the data of a xf::cv::Mat<> object into an AXI Video stream
/*
 *  input: img
 *  output: AXI_video_strm
 */
template <int W, int TYPE, int ROWS, int COLS, int NPPC>
int xfMat2AXIvideo(xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& img, hls::stream<ap_axiu<W, 1, 1, 1> >& AXI_video_strm) {
    ap_axiu<W, 1, 1, 1> axi;
    int res = 0;

    int rows = img.rows;
    int cols = img.cols >> XF_BITSHIFT(NPPC);
    int idx = 0;

    assert(img.rows <= ROWS);
    assert(img.cols <= COLS);

    const int m_pix_width = XF_PIXELWIDTH(TYPE, NPPC) * XF_NPIXPERCYCLE(NPPC);

    bool sof = true; // Indicates start of frame

loop_row_mat2axi:
    for (int i = 0; i < rows; i++) {
    loop_col_mat2axi:
        for (int j = 0; j < cols; j++) {
// clang-format off
#pragma HLS loop_flatten off
#pragma HLS pipeline II = 1
            // clang-format on

            if (sof) {
                axi.user = 1;
            } else {
                axi.user = 0;
            }

            if (j == cols - 1) {
                axi.last = 1;
            } else {
                axi.last = 0;
            }

            axi.data = 0;
            axi.data(m_pix_width - 1, 0) = img.read(idx++);

            axi.keep = -1;
            AXI_video_strm << axi;

            sof = false;
        }
    }

    return res;
}

} // namespace cv
} // namespace xf

#endif
