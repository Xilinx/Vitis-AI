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

#ifndef _XF_CUSTOM_CONVOLUTION_HPP_
#define _XF_CUSTOM_CONVOLUTION_HPP_

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "hls_stream.h"
typedef unsigned char uchar;

namespace xf {
namespace cv {

/****************************************************************************************
 * xFApplyCustomFilter: Applies the user defined kernel to the input image.
 *
 * _lbuf	   ->  Buffer containing the input image data
 * _kernel	   ->  Kernel provided by the user of type 16S
 * shift	   ->  Fixed point format of the filter co-efficients for unity
 *gain filter
 ****************************************************************************************/
template <int DEPTH_SRC,
          int DEPTH_DST,
          int filter_height,
          int filter_width,
          int NPC,
          int PLANES,
          int buf_width,
          typename buf_type>
XF_PTNAME(DEPTH_DST)
xFApplyCustomFilter(buf_type _lbuf[][buf_width], short int _kernel[][filter_width], int ind, unsigned char shift) {
// clang-format off
    #pragma HLS INLINE off
    // clang-format on
    XF_PTNAME(DEPTH_DST) res = 0;
    ap_int32_t tmp_res[PLANES];
    ap_int24_t conv_val[filter_height][filter_width][PLANES];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=conv_val complete dim=0
    // clang-format on

    ap_int32_t row_sum[filter_height][PLANES], fix_res = 0, tmp_row_sum = 0;
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=row_sum complete dim=1
    // clang-format on

    XF_PTNAME(DEPTH_DST) arr_ind = ind;

// performing kernel operation and storing in the temporary buffer
filterLoopI:
    for (uchar i = 0; i < filter_height; i++) {
// clang-format off
#pragma HLS UNROLL
        // clang-format on
        arr_ind = ind;

    filterLoopJ:
        for (uchar j = 0; j < filter_width; j++) {
// clang-format off
#pragma HLS UNROLL
        // clang-format on
        planes_loop1:
            for (uchar k = 0; k < PLANES; k++) {
// clang-format off
#pragma HLS UNROLL
                // clang-format on
                conv_val[i][j][k] = ((_lbuf[i][arr_ind].range((k * 8) + 7, k * 8)) * _kernel[i][j]);
            }
            arr_ind++;
        }
    }

// accumulating the row sum values of the temporary buffer
planes_add_row:
    for (uchar p = 0; p < PLANES; p++) {
// clang-format off
#pragma HLS UNROLL
    // clang-format on
    addFilterLoopI:
        for (uchar i = 0; i < filter_height; i++) {
// clang-format off
#pragma HLS UNROLL
            // clang-format on
            tmp_row_sum = 0;

        addFilterLoopJ:
            for (uchar j = 0; j < filter_width; j++) {
// clang-format off
#pragma HLS UNROLL
                // clang-format on
                tmp_row_sum += conv_val[i][j][p];
            }
            row_sum[i][p] = tmp_row_sum;
        }
    }

// adding the row_sum buffer elements and storing in the result
add_row_col_plane_loop:
    for (uchar p = 0; p < PLANES; p++) {
// clang-format off
#pragma HLS UNROLL
        // clang-format on
        fix_res = 0;
    resultFilterLoopI:
        for (uchar i = 0; i < filter_height; i++) {
// clang-format off
#pragma HLS UNROLL
            // clang-format on
            fix_res += row_sum[i][p];
        }

        // converting the input type from Q1.shift
        tmp_res[p] = (fix_res >> shift);
    }

    // overflow handling depending upon the input type
    if ((DEPTH_DST == XF_8UP) || (DEPTH_DST == XF_24UP)) {
    planes_loop_out8:
        for (uchar p = 0; p < PLANES; p++) {
// clang-format off
#pragma HLS UNROLL
            // clang-format on
            if (tmp_res[p] > 255) {
                res.range((p * 8) + 7, p * 8) = 255;
            } else if (tmp_res[p] < 0) {
                res.range((p * 8) + 7, p * 8) = 0;
            } else {
                res.range((p * 8) + 7, p * 8) = tmp_res[p];
            }
        }
    } else if ((DEPTH_DST == XF_16SP) || (DEPTH_DST == XF_48SP)) {
    planes_loop_out16:
        for (uchar p = 0; p < PLANES; p++) {
// clang-format off
#pragma HLS UNROLL
            // clang-format on
            int tmp_val = (int)tmp_res[p];
            if (tmp_val > ((1 << (16 - 1)) - 1)) {
                res.range((p * 16) + 15, p * 16) = ((1 << (16 - 1)) - 1);
            } else if (tmp_val < -(1 << (16 - 1))) {
                res.range((p * 16) + 15, p * 16) = -(1 << (16 - 1));
            } else {
                res.range((p * 16) + 15, p * 16) = (short)tmp_val;
            }
        }
    }
    return res;
}

/****************************************************************************************
 * xFComputeCustomFilter : Applies the mask and Computes the filter value for
 *NPC
 * 					number of times.
 *
 * _lbuf	   ->  Buffer containing the input image data
 * _kernel	   ->  Kernel provided by the user of type 16S
 * _mask_value ->  The output buffer containing ouput image data
 * shift	   ->  Fixed point format of the filter co-efficients for unity
 *gain filter
 ****************************************************************************************/
template <int filter_height, int filter_width, int buf_width, int NPC, int DEPTH_SRC, int DEPTH_DST, int PLANES>
void xFComputeCustomFilter(XF_PTNAME(DEPTH_SRC) _lbuf[][buf_width],
                           short int _kernel[][filter_width],
                           XF_PTNAME(DEPTH_DST) * _mask_value,
                           unsigned char shift) {
// clang-format off
    #pragma HLS inline
// clang-format on
// computes the filter operation depending upon the mode of parallelism
computeFilterLoop:
    for (ap_uint<5> j = 0; j < XF_NPIXPERCYCLE(NPC); j++) {
// clang-format off
        #pragma HLS UNROLL
        // clang-format on
        _mask_value[j] = xFApplyCustomFilter<DEPTH_SRC, DEPTH_DST, filter_height, filter_width, NPC, PLANES>(
            _lbuf, _kernel, j, shift);
    }
}

template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int TC,
          int FW,
          int filter_height,
          int filter_width,
          int F_COUNT,
          int PLANES>
void Convolution_Process(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src,
                         xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst,
                         XF_SNAME(WORDWIDTH_SRC) buf[filter_height][COLS >> XF_BITSHIFT(NPC)],
                         XF_PTNAME(DEPTH_SRC) lbuf[filter_height][XF_NPIXPERCYCLE(NPC) + filter_width - 1],
                         XF_SNAME(WORDWIDTH_SRC) tmp_buf[filter_height],
                         XF_PTNAME(DEPTH_DST) mask_value[XF_NPIXPERCYCLE(NPC)],
                         short int _filter[][filter_width],
                         uint16_t image_width,
                         uchar row_ind,
                         unsigned char shift,
                         XF_SNAME(WORDWIDTH_DST) & P0,
                         unsigned char index[filter_height],
                         ap_uint<13> col_factor,
                         uchar filter_width_factor,
                         unsigned short image_height,
                         ap_uint<13> row,
                         int& rd_ind,
                         int& wr_ind) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    uchar step = XF_PIXELDEPTH(DEPTH_DST);
    unsigned short max_loop = XF_WORDDEPTH(WORDWIDTH_DST);
mainColLoop:
    for (ap_uint<13> col = 0; col < (image_width); col++) // Width of the image
    {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
        #pragma HLS PIPELINE II=1
        // clang-format on

        // reading the data from the stream to the input buffer

        if (row < image_height) {
            buf[row_ind][col] = _src.read(rd_ind);
            rd_ind++;
        } else {
            buf[row_ind][col] = 0;
        }

    // loading the data from the input buffer to the temporary buffer
    fillTempBuffer_1:
        for (uchar l = 0; l < filter_height; l++) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on
            tmp_buf[l] = buf[index[l]][col];
        }

    // extracting the pixels from the temporary buffer to the line buffer
    extractPixelsLoop_1:
        for (uchar l = 0; l < filter_height; l++) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on
            xfExtractPixels<NPC, WORDWIDTH_SRC, DEPTH_SRC>(&lbuf[l][(filter_width - 1)], tmp_buf[l], 0);
        }

        // computing the mask value
        xFComputeCustomFilter<filter_height, filter_width, (XF_NPIXPERCYCLE(NPC) + filter_width - 1), NPC, DEPTH_SRC,
                              DEPTH_DST, PLANES>(lbuf, _filter, mask_value, shift);

        // left column border condition
        if (col <= col_factor) {
            ap_uint<13> ind = filter_width_factor;
            ap_uint<13> range_step = 0;

            if ((XF_NPIXPERCYCLE(NPC) - filter_width_factor) >= 0) {
            packMaskToTempRes_1:
                for (uchar l = 0; l < (XF_NPIXPERCYCLE(NPC) - FW); l++) {
// clang-format off
                    #pragma HLS LOOP_TRIPCOUNT min=F_COUNT max=F_COUNT
                    #pragma HLS UNROLL
                    // clang-format on
                    P0.range((range_step + (step - 1)), range_step) = mask_value[ind++];
                    range_step += step;
                }
            } else {
                filter_width_factor -= XF_NPIXPERCYCLE(NPC);
            }
        }

        // packing the data from the mask value to the temporary result P0 and
        // pushing data into stream
        else {
            ap_uint<10> max_range_step = max_loop - (filter_width_factor * step);

        packMaskToTempRes_2:
            for (uchar l = 0; l < FW; l++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=FW max=FW
                #pragma HLS UNROLL
                // clang-format on
                P0.range((max_range_step + (step - 1)), (max_range_step)) = mask_value[l];
                max_range_step += step;
            }

            // writing the temporary result into the stream
            _dst.write(wr_ind, P0);
            wr_ind++;

            ap_uint<13> ind = filter_width_factor;
            ap_uint<13> range_step = 0;

        packMaskToTempRes_3:
            for (ap_uint<13> l = 0; l < (XF_NPIXPERCYCLE(NPC) - FW); l++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=F_COUNT max=F_COUNT
                #pragma HLS UNROLL
                // clang-format on
                P0.range((range_step + (step - 1)), range_step) = mask_value[ind++];
                range_step += step;
            }
        }

    // re-initializing the line buffers
    copyEndPixelsI_1:
        for (uchar i = 0; i < filter_height; i++) {
// clang-format off
            #pragma HLS UNROLL
        // clang-format on
        copyEndPixelsJ_1:
            for (uchar l = 0; l < (filter_width - 1); l++) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on
                lbuf[i][l] = lbuf[i][XF_NPIXPERCYCLE(NPC) + l];
            }
        }
    } //  end of main column loop*/
}

/************************************************************************************
 * xFCustomConvKernel : Convolutes the input filter over the input image and
 *writes
 * 					onto the output image.
 *
 * _src		->  Input image of type 8U
 * _filter	->  Kernel provided by the user of type 16S
 * _dst		->  Output image after applying the filter operation, of type
 *8U or 16S
 * shift	->  Fixed point format of the filter co-efficients for unity
 *gain
 *filter
 ************************************************************************************/
template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int COLS_COUNT,
          int filter_height,
          int filter_width,
          int F_COUNT,
          int FW,
          int COL_FACTOR_COUNT,
          int PLANES>
void xFCustomConvolutionKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src,
                               short int _filter[][filter_width],
                               xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst,
                               unsigned char shift,
                               unsigned short img_width,
                               unsigned short img_height) {
    uchar step = XF_PIXELDEPTH(DEPTH_DST);
    unsigned short max_loop = XF_WORDDEPTH(WORDWIDTH_DST);
    uchar buf_size = (XF_NPIXPERCYCLE(NPC) + filter_width - 1);

    uchar row_ind = 0, row_ptr = 0;
    unsigned char index[filter_height];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=index complete dim=1
    // clang-format on

    XF_SNAME(WORDWIDTH_DST) P0;
    XF_SNAME(WORDWIDTH_SRC) buf[filter_height][COLS >> XF_BITSHIFT(NPC)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=buf complete dim=1
    // clang-format on

    XF_PTNAME(DEPTH_SRC)
    lbuf[filter_height][XF_NPIXPERCYCLE(NPC) + filter_width - 1];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=lbuf complete dim=0
    // clang-format on

    XF_SNAME(WORDWIDTH_SRC) tmp_buf[filter_height];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=tmp_buf complete dim=1
    // clang-format on

    XF_PTNAME(DEPTH_DST) mask_value[XF_NPIXPERCYCLE(NPC)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=mask_value complete dim=1
    // clang-format on

    XF_PTNAME(DEPTH_DST) col_border_mask[(filter_width >> 1)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=col_border_mask complete dim=1
    // clang-format on

    ap_uint<13> col_factor = 0;
    uchar filter_width_factor = (filter_width >> 1);
    int rd_ind = 0, wr_ind = 0;

// setting the column factor depending upon the filter dimensions
colFactorLoop:
    for (uchar f = (filter_width >> 1); f > (XF_NPIXPERCYCLE(NPC)); f = (f - XF_NPIXPERCYCLE(NPC))) {
        col_factor++;
    }

// initializing the first two rows to zeros
fillBufZerosI:
    for (uchar i = 0; i < (filter_height >> 1); i++) {
// clang-format off
        #pragma HLS UNROLL
    // clang-format on
    fillBufZerosJ:
        for (ap_uint<13> j = 0; j < (img_width); j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=COLS_COUNT max=COLS_COUNT
            #pragma HLS UNROLL
            // clang-format on
            buf[row_ind][j] = 0;
        }
        row_ind++;
    }

// reading the first two rows from the input stream
readTopBorderI:
    for (uchar i = 0; i < (filter_height >> 1); i++) {
// clang-format off
        #pragma HLS UNROLL
    // clang-format on
    readTopBorderJ:
        for (ap_uint<13> j = 0; j < (img_width); j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=COLS_COUNT max=COLS_COUNT
            #pragma HLS PIPELINE
            // clang-format on
            buf[row_ind][j] = _src.read(rd_ind);
            rd_ind++;
        }
        row_ind++;
    }

// row loop from 1 to the end of the image
mainRowLoop:
    for (ap_uint<13> row = (filter_height >> 1); row < (img_height + ((filter_height >> 1))); row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        // clang-format on

        row_ptr = row_ind + 1;

    // index calculation
    settingIndex_1:
        for (int l = 0; l < filter_height; l++) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on
            if (row_ptr >= filter_height) row_ptr = 0;

            index[l] = row_ptr++;
        }

    // initializing the line buffer to zero
    fillingLineBufferZerosI_1:
        for (uchar i = 0; i < filter_height; i++) {
// clang-format off
            #pragma HLS UNROLL
        // clang-format on
        fillingLineBufferZerosJ_1:
            for (uchar j = 0; j < (filter_width - 1); j++) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on
                lbuf[i][j] = 0;
            }
        }
        // initializing the temporary result value to zero
        P0 = 0;

        Convolution_Process<SRC_T, DST_T, ROWS, COLS, DEPTH_SRC, DEPTH_DST, NPC, WORDWIDTH_SRC, WORDWIDTH_DST,
                            COLS_COUNT, FW, filter_height, filter_width, F_COUNT, PLANES>(
            _src, _dst, buf, lbuf, tmp_buf, mask_value, _filter, img_width, row_ind, shift, P0, index, col_factor,
            filter_width_factor, img_height, row, rd_ind, wr_ind);

    /////////  Column right border  /////////

    // initializing the line buffers to zero
    fillingLineBufferZerosI_2:
        for (uchar i = 0; i < filter_height; i++) {
// clang-format off
            #pragma HLS UNROLL
        // clang-format on
        fillingLineBufferZerosJ_2:
            for (ap_uint<13> l = (filter_width - 1); l < buf_size; l++) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on
                lbuf[i][l] = 0;
            }
        }

        // applying the filter and computing the mask_value
        if ((filter_width >> 1) > 0) {
        getMaskValue_1:
            for (uchar i = 0; i < (filter_width >> 1); i++) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on
                col_border_mask[i] =
                    xFApplyCustomFilter<DEPTH_SRC, DEPTH_DST, filter_height, filter_width, NPC, PLANES>(lbuf, _filter,
                                                                                                        i, shift);
            }
        }

        int max_range_step = max_loop - (FW * step);

    packMaskToTempRes_4:
        for (uchar l = 0; l < FW; l++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=FW max=FW
            #pragma HLS UNROLL
            // clang-format on
            P0.range((max_range_step + step - 1), (max_range_step)) = col_border_mask[l];
            max_range_step += step;
        }

        // writing the temporary result into the stream
        _dst.write(wr_ind, P0);
        wr_ind++;

    colFactorLoopBorder:
        for (ap_uint<13> c = 0; c < col_factor; c++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=COL_FACTOR_COUNT max=COL_FACTOR_COUNT
            // clang-format on

            max_range_step = 0;
        widthFactorLoopBorder:
            for (int l = FW; l < (XF_NPIXPERCYCLE(NPC) + FW); l++) {
                P0.range((max_range_step + (step - 1)), (max_range_step)) = col_border_mask[l];
                max_range_step += step;
            }
            _dst.write(wr_ind, P0);
            wr_ind++;
        }

        // incrementing the row_ind for each iteration of row
        row_ind++;
        if (row_ind == filter_height) {
            row_ind = 0;
        }
    } // end of main row loop
} // end of xFCustomConvKernel

template <int DEPTH_SRC, int DEPTH_DST, int F_HEIGHT, int F_WIDTH, int PLANES>
void xFApplyFilter2D(XF_PTNAME(DEPTH_SRC) _kernel_pixel[F_HEIGHT][F_WIDTH],
                     short int _kernel_filter[F_HEIGHT][F_WIDTH],
                     XF_PTNAME(DEPTH_DST) & out,
                     unsigned char shift) {
// clang-format off
    #pragma HLS INLINE off
    // clang-format on

    ap_int<32> sum = 0, in_step = 0, out_step = 0, p = 0;
    ap_int<32> temp = 0;
    ap_int<32> tmp_sum = 0;
FILTER_LOOP_HEIGHT:
    ap_uint<24> bgr_val;
    if ((DEPTH_DST == XF_8UP) || (DEPTH_DST == XF_24UP)) {
        in_step = 8;
        out_step = 8;
    } else {
        in_step = 8;
        out_step = 16;
    }
    for (ap_uint<8> c = 0, k = 0; c < PLANES; c++, k += out_step) {
        sum = 0;
        temp = 0;
        tmp_sum = 0;
        for (ap_int<8> m = 0; m < F_HEIGHT; m++) {
        FILTER_LOOP_WIDTH:
            for (ap_int<8> n = 0; n < F_WIDTH; n++) {
                XF_PTNAME(DEPTH_SRC)
                src_v = _kernel_pixel[F_HEIGHT - m - 1][F_WIDTH - 1 - n];

                short int filter_v = _kernel_filter[m][n];
                temp = src_v.range(p + (in_step - 1), p) * filter_v;
                sum = sum + temp;
            }
        }
        p = p + 8;
        tmp_sum = sum >> shift;

        if ((DEPTH_DST == XF_8UP) || (DEPTH_DST == XF_24UP)) {
            if (tmp_sum > ((1 << (8)) - 1)) {
                out.range(k + 7, k) = ((1 << (8)) - 1);
            } else if (tmp_sum < 0) {
                out.range(k + 7, k) = 0;
            } else {
                out.range(k + 7, k) = tmp_sum;
            }
        } else if ((DEPTH_DST == XF_16SP) || (DEPTH_DST == XF_48SP)) {
            if (tmp_sum > ((1 << (16 - 1)) - 1)) {
                out.range(k + 15, k) = ((1 << (16 - 1)) - 1);
            } else if (tmp_sum < -(1 << (16 - 1))) {
                out.range(k + 15, k) = -(1 << (16 - 1));
            } else {
                out.range(k + 15, k) = tmp_sum;
            }
        }
    }
}
static int borderInterpolate(int p, int len, int borderType) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    if (p >= 0 && p < len)
        return p;
    else
        p = -1;
    return p;
}

template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int TC,
          int K_HEIGHT,
          int K_WIDTH,
          int PLANES>
static void xFFilter2Dkernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                             xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_mat,
                             short int _filter_kernel[K_HEIGHT][K_WIDTH],
                             unsigned char shift,
                             uint16_t rows,
                             uint16_t cols)

{
    XF_SNAME(WORDWIDTH_SRC) fillvalue = 0;
// clang-format off
    #pragma HLS INLINE off
    // clang-format on

    // The main processing window
    XF_PTNAME(DEPTH_SRC) src_kernel_win[K_HEIGHT][K_WIDTH];
    // The main line buffer
    XF_SNAME(WORDWIDTH_SRC) k_buf[K_HEIGHT][COLS >> XF_BITSHIFT(NPC)];
    // A small buffer keeping a few pixels from the line
    // buffer, so that we can complete right borders correctly.
    XF_SNAME(WORDWIDTH_SRC) right_border_buf[K_HEIGHT][K_WIDTH];
    // Temporary storage for reading from the line buffers.
    XF_SNAME(WORDWIDTH_SRC) col_buf[K_HEIGHT];
#ifndef __SYNTHESIS__
    assert(rows >= 8);
    assert(cols >= 8);
    assert(rows <= ROWS);
    assert(cols <= COLS);
#endif
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=col_buf complete dim=0
    #pragma HLS ARRAY_PARTITION variable=_filter_kernel complete dim=0
    #pragma HLS ARRAY_PARTITION variable=src_kernel_win complete dim=0
    #pragma HLS ARRAY_PARTITION variable=k_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=right_border_buf complete dim=0
    // clang-format on

    int heightloop = rows + K_HEIGHT - 1 + K_HEIGHT;
    int widthloop = cols + K_WIDTH - 1; // one pixel overlap, so it should minus one
    /*ap_uint<13> i,j;
    ap_uint<13> anchorx=K_WIDTH/2,anchory=K_HEIGHT/2;
    ap_uint<13> ImagLocx=0,ImagLocy =0;*/

    uint16_t i, j;
    int rd_ind = 0, wr_ind = 0;
    uint16_t anchorx = K_WIDTH >> 1, anchory = K_HEIGHT >> 1;
    int16_t ImagLocx = 0, ImagLocy = 0;

ROW_LOOP:
    for (i = 0; i < heightloop; i++) {
    COL_LOOP:
        for (j = 0; j < widthloop; j++) {
// This DEPENDENCE pragma is necessary because the border mode handling is not
// affine.
// clang-format off
            #pragma HLS DEPENDENCE array inter false
            #pragma HLS LOOP_FLATTEN OFF
            #pragma HLS PIPELINE
            // clang-format on

            // fill data x,y are the coordinate in the image, it could be negative.
            // For example (-1,-1) represents the
            // interpolation pixel.
            ImagLocx = j - anchorx;
            ImagLocy = i - K_HEIGHT - anchory;
            int16_t x = borderInterpolate(ImagLocx, cols, 0);

            // column left shift
            for (ap_int<8> row = 0; row < K_HEIGHT; row++)
                for (ap_int<8> col = K_WIDTH - 1; col >= 1; col--)
                    src_kernel_win[row][col] = src_kernel_win[row][col - 1];

            for (ap_int<8> buf_row = 0; buf_row < K_HEIGHT; buf_row++) {
// Fetch the column from the line buffer to shift into the window.
#ifndef __SYNTHESIS__
                assert((x < COLS));
#endif
                col_buf[buf_row] = ((x < 0)) ? fillvalue : k_buf[buf_row][x];
            }

            if ((ImagLocy < (-anchory)) || (ImagLocy >= K_HEIGHT - 1 && ImagLocy < rows - 1)) {
                // Advance load and body process
                if (ImagLocx >= 0 && ImagLocx < cols) {
                    XF_SNAME(WORDWIDTH_SRC)
                    Toppixel = col_buf[K_HEIGHT - 1]; // k_buf[k](K_HEIGHT-1,ImagLocx);
                    src_kernel_win[K_HEIGHT - 1][0] = Toppixel;
                    if (ImagLocx >= cols - K_WIDTH) {
                        right_border_buf[0][ImagLocx - (cols - K_WIDTH)] = Toppixel;
                    }
                    for (ap_int<8> buf_row = K_HEIGHT - 1; buf_row >= 1; buf_row--) {
                        XF_SNAME(WORDWIDTH_SRC)
                        temp = col_buf[buf_row - 1]; // k_buf[k](buf_row-1,ImagLocx);
                        src_kernel_win[buf_row - 1][0] = temp;
                        k_buf[buf_row][x] = temp;
                        if (ImagLocx >= cols - K_WIDTH) {
                            right_border_buf[K_HEIGHT - buf_row][ImagLocx - (cols - K_WIDTH)] = temp;
                        }
                    }
                    XF_SNAME(WORDWIDTH_SRC) temp = 0;
                    temp = (_src_mat.read(rd_ind));
                    rd_ind++;

                    k_buf[0][x] = temp;
                } else if (ImagLocx < 0) {
                    for (int buf_row = 0; buf_row < K_HEIGHT; buf_row++) {
                        src_kernel_win[buf_row][0] = fillvalue;
                    }
                } else if (ImagLocx >= cols) {
                    for (int buf_row = 0; buf_row < K_HEIGHT; buf_row++) {
                        src_kernel_win[buf_row][0] = fillvalue;
                    }
                }
            } else if (ImagLocy >= 0) { //   && ImagLocy < K_HEIGHT-1) ||
                // (ImagLocy >= rows-1      && ImagLocy < heightloop)) {
                // top extend pixel bottom keep the buffer 0 with the data rows-1
                // content.
                int ref = K_HEIGHT - 1;
                if (ImagLocy >= rows - 1) ref = rows - 1;
                int y = ImagLocy;
                for (int buf_row = 0; buf_row < K_HEIGHT; buf_row++) {
                    int t = borderInterpolate(y, rows, 0);
                    int locy = ref - t;
#ifndef __SYNTHESIS__
                    assert(t < 0 || (locy >= 0 && locy < K_HEIGHT));
#endif
                    if (y >= rows)
                        src_kernel_win[buf_row][0] = fillvalue;
                    else if (y < 0)
                        src_kernel_win[buf_row][0] = fillvalue;
                    else
                        src_kernel_win[buf_row][0] = col_buf[locy];
                    y--;
                }
            }

            // figure out the output image pixel value
            if (i >= (K_HEIGHT + K_HEIGHT - 1) && j >= (K_WIDTH - 1)) {
                XF_PTNAME(DEPTH_DST) temp;
                xFApplyFilter2D<DEPTH_SRC, DEPTH_DST, K_HEIGHT, K_WIDTH, PLANES>(src_kernel_win, _filter_kernel, temp,
                                                                                 shift);
                XF_SNAME(WORDWIDTH_DST) temp1 = temp;
                _dst_mat.write(wr_ind, temp1);
                wr_ind++;
            }
        }
    }
}

template <int BORDER_TYPE, int FILTER_WIDTH, int FILTER_HEIGHT, int SRC_T, int DST_T, int ROWS, int COLS, int NPC>
void filter2D(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
              xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_mat,
              short int filter[FILTER_HEIGHT * FILTER_WIDTH],
              unsigned char _shift) {
// clang-format off
    #pragma HLS INLINE OFF
// clang-format on
#ifndef __SYNTHESIS__
    assert(((_src_mat.rows <= ROWS) && (_src_mat.cols <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
    unsigned short img_width = _src_mat.cols >> XF_BITSHIFT(NPC);
    unsigned short img_height = _src_mat.rows;

    short int lfilter[FILTER_HEIGHT][FILTER_WIDTH];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=lfilter complete dim=0
    // clang-format on
    for (unsigned char i = 0; i < FILTER_HEIGHT; i++) {
        for (unsigned char j = 0; j < FILTER_WIDTH; j++) {
            lfilter[i][j] = filter[i * FILTER_WIDTH + j];
        }
    }

    if (NPC == XF_NPPC8) {
        xFCustomConvolutionKernel<SRC_T, DST_T, ROWS, COLS, XF_DEPTH(SRC_T, NPC), XF_DEPTH(DST_T, NPC), NPC,
                                  XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(DST_T, NPC), (COLS >> XF_BITSHIFT(NPC)),
                                  FILTER_HEIGHT, FILTER_WIDTH,
                                  (XF_NPIXPERCYCLE(NPC) - ((FILTER_WIDTH >> 1) % XF_NPIXPERCYCLE(NPC))),
                                  ((FILTER_WIDTH >> 1) % XF_NPIXPERCYCLE(NPC)),
                                  (((FILTER_WIDTH >> 1) - 1) >> XF_BITSHIFT(NPC)), XF_CHANNELS(SRC_T, NPC)>(
            _src_mat, lfilter, _dst_mat, _shift, img_width, img_height);

    }

    else if (NPC == XF_NPPC1) {
        xFFilter2Dkernel<SRC_T, DST_T, ROWS, COLS, XF_DEPTH(SRC_T, NPC), XF_DEPTH(DST_T, NPC), NPC,
                         XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(DST_T, NPC), COLS, FILTER_HEIGHT, FILTER_WIDTH,
                         XF_CHANNELS(SRC_T, NPC)>(_src_mat, _dst_mat, lfilter, _shift, img_height, img_width);
    }
}
} // namespace cv
} // namespace xf
#endif // _XF_CUSTOM_CONVOLUTION_HPP_
