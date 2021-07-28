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

//#include "opencv_core_types.h"

#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
/* From/To AXI video stream */
#include "ap_axi_sdata.h"
#include "utils/x_hls_utils.h"

#include <assert.h>

namespace xf {
namespace cv {

template <int W, int NPC>
void xfAXISetBitFields(ap_uint<W>& pix, IplImage* img, int row, int col) {
// clang-format off
    #pragma HLS inline
    // clang-format on
    int start = 0;
    ap_uint<W> pack;
    CvScalar cv_pix;
    fp_struct<float> temp;

    //    int depth_8s=IPL_DEPTH_8S;
    //    int depth_16s=IPL_DEPTH_16S;
    //    int depth_16u=IPL_DEPTH_16U;
    //    int depth_32s=IPL_DEPTH_32S;
    //    int depth_32f=IPL_DEPTH_32F;
    //    int IPL_DEPTH_64F=IPL_DEPTH_64F;
    int pix_depth = (img->depth & 0xff);

    for (int n_pix = 0; n_pix < NPC; n_pix++) {
        cv_pix = cvGet2D(img, row, col + n_pix);

        for (int k = 0; k < img->nChannels; k++) {
            switch (img->depth) {
                case IPL_DEPTH_8S: {
                    pack(start + pix_depth - 1, start) = (char)cv_pix.val[k];
                } break;
                case IPL_DEPTH_16U: {
                    pack(start + pix_depth - 1, start) = (unsigned short)cv_pix.val[k];
                } break;
                case IPL_DEPTH_16S: {
                    pack(start + pix_depth - 1, start) = (short)cv_pix.val[k];
                } break;
                case IPL_DEPTH_32S: {
                    pack(start + pix_depth - 1, start) = (int)cv_pix.val[k];
                } break;
                case IPL_DEPTH_32F: {
                    fp_struct<float> temp((float)cv_pix.val[k]);
                    pack(start + pix_depth - 1, start) = temp.data();
                } break;
                case IPL_DEPTH_64F: {
                    pack(start + pix_depth - 1, start) = (double)cv_pix.val[k];
                } break;
                default: { // Includes IPL_DEPTH_8U
                    pack(start + pix_depth - 1, start) = (unsigned char)cv_pix.val[k];
                } break;
            }

            start = start + pix_depth;
        }
    }
    pix = pack;
}
template <int W, int NPC>
void xfAXISetBitFields(ap_axiu<W, 1, 1, 1>& axi, IplImage* img, int row, int col) {
// clang-format off
    #pragma HLS inline
    // clang-format on
    xfAXISetBitFields<W, NPC>(axi.data, img, row, col);
}
template <int W, int NPC>
void IplImage2AXIvideoxf(IplImage* img, hls::stream<ap_axiu<W, 1, 1, 1> >& AXI_video_strm) {
    int i, j, p = 0;

    CvScalar cv_pix;
    ap_axiu<W, 1, 1, 1> axi;
    int depth = (img->depth & 0xff);
    assert(img && img->imageData && (W >= depth * img->nChannels) &&
           "Image must be valid and have width less than the width of the stream.");

    int channels = img->nChannels;

    for (i = 0; i < img->height; i++) {
        for (j = 0; j < (img->width); j = j + NPC) {
            if ((i == 0) && (j == 0)) {
                axi.user = 1;
            } else {
                axi.user = 0;
            }
            if (j == (img->width - NPC)) {
                axi.last = 1;
            } else {
                axi.last = 0;
            }
            axi.data = -1;

            xfAXISetBitFields<W, NPC>(axi, img, i, j);
            axi.keep = -1;
            AXI_video_strm << axi;
        }
    }
}

template <int NPC, int W>
void cvMat2AXIvideoxf(::cv::Mat& cv_mat, hls::stream<ap_axiu<W, 1, 1, 1> >& AXI_video_strm) {
    IplImage img = cv_mat;
    IplImage2AXIvideoxf<W, NPC>(&img, AXI_video_strm);
}
template <int NPC, int W>
void xfAXIGetBitFields(ap_uint<W> pix, IplImage* img, int row, int col) {
// clang-format off
    #pragma HLS inline
    // clang-format on
    CvScalar cv_pix;
    int depth = (img->depth & 0xff);
    int start = 0;
    //	  const int depth_8u=IPL_DEPTH_8U;
    //	  const int depth_8s=IPL_DEPTH_8S;
    //	  const int depth_16s=IPL_DEPTH_16S;
    //	  const int depth_16u=IPL_DEPTH_16U;
    //	  const int depth_32s=IPL_DEPTH_32S;
    //	  const int depth_32f=IPL_DEPTH_32F;
    //	  const int depth_64f=IPL_DEPTH_64F;

    for (int n_pix = 0; n_pix < NPC; n_pix++) {
        for (int k = 0; k < img->nChannels; k++) {
            switch (img->depth) {
                case IPL_DEPTH_8U: {
                    unsigned char temp = pix.range(start + depth - 1, start);
                    cv_pix.val[k] = temp;
                } break;
                case IPL_DEPTH_8S: {
                    char temp = pix.range(start + depth - 1, start);
                    cv_pix.val[k] = temp;
                } break;
                case IPL_DEPTH_16U: {
                    unsigned short temp = pix.range(start + depth - 1, start);
                    cv_pix.val[k] = temp;
                } break;
                case IPL_DEPTH_16S: {
                    short temp = pix.range(start + depth - 1, start);
                    cv_pix.val[k] = temp;
                } break;
                case IPL_DEPTH_32S: {
                    int temp = pix.range(start + depth - 1, start);
                    cv_pix.val[k] = temp;
                } break;
                case IPL_DEPTH_32F: {
                    float temp = pix.range(start + depth - 1, start);
                    cv_pix.val[k] = temp;
                } break;
                case IPL_DEPTH_64F: {
                    double temp;
                    cv_pix.val[k] = temp;
                } break;
                default: {
                    unsigned char temp = pix.range(start + depth - 1, start);
                    cv_pix.val[k] = temp;
                } break;
            }

            start = start + depth;
        }
        cvSet2D(img, row, col + n_pix, cv_pix);
    }
}
template <int NPC, int W>
void xfAXIGetBitFields(ap_axiu<W, 1, 1, 1> axi, IplImage* img, int row, int col) {
// clang-format off
    #pragma HLS inline
    // clang-format on
    xfAXIGetBitFields<NPC, W>(axi.data, img, row, col);
}
template <int W, int NPC>
void AXIvideo2IplImagexf(hls::stream<ap_axiu<W, 1, 1, 1> >& AXI_video_strm, IplImage* img) {
    int i, j;
    ap_axiu<W, 1, 1, 1> axi;
    CvScalar cv_pix;
    int depth = (img->depth & 0xff);
    bool sof = 0;
    assert(img && img->imageData && (W >= depth * img->nChannels) &&
           "Image must be valid and have width less than the width of the stream.");

    for (i = 0; i < img->height; i++) {
        for (j = 0; j < (img->width); j += NPC) {
            AXI_video_strm >> axi;
            if ((i == 0) && (j == 0)) {
                if (axi.user.to_int() == 1) {
                    sof = 1;
                } else {
                    j--;
                }
            }
            if (sof) {
                xfAXIGetBitFields<NPC, W>(axi, img, i, j);
            }
        }
    }
}

template <int NPC, int W>
void AXIvideo2cvMatxf(hls::stream<ap_axiu<W, 1, 1, 1> >& AXI_video_strm, ::cv::Mat& cv_mat) {
    IplImage img = cv_mat;
    AXIvideo2IplImagexf<W, NPC>(AXI_video_strm, &img);
}
} // namespace cv
} // namespace xf
