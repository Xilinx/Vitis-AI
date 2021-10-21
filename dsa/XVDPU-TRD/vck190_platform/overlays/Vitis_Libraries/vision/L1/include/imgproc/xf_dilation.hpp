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

#ifndef _XF_DILATION_
#define _XF_DILATION_

#include "ap_int.h"
#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"

namespace xf {
namespace cv {
////////////////////////	Core Processing		//////////////////////////////
template <int PLANES, int DEPTH, int K_ROWS, int K_COLS>
void dilate_function_apply(XF_PTUNAME(DEPTH) * OutputValue,
                           XF_PTUNAME(DEPTH) src_buf[K_ROWS][K_COLS],
                           unsigned char kernel[K_ROWS][K_COLS]) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    XF_PTUNAME(DEPTH) packed_val = 0, depth = XF_PIXELWIDTH(DEPTH, XF_NPPC1) / PLANES;

Apply_dilate_Loop:
    for (ap_uint<5> c = 0, k = 0; c < PLANES; c++, k += depth) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=PLANES
        #pragma HLS UNROLL
        // clang-format on
        XF_PTNAME(DEPTH) max = 0;
        for (ap_uint<13> k_rows = 0; k_rows < K_ROWS; k_rows++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=K_ROWS
            #pragma HLS UNROLL
            // clang-format on
            for (ap_uint<13> k_cols = 0; k_cols < K_COLS; k_cols++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=1 max=K_COLS
                #pragma HLS UNROLL
                // clang-format on
                if (kernel[k_rows][k_cols]) {
                    if (src_buf[k_rows][k_cols].range(k + (depth - 1), k) > max) {
                        max = src_buf[k_rows][k_cols].range(k + (depth - 1), k);
                    }
                }
            }
        }
        packed_val.range(k + (depth - 1), k) = max;
    }

    *OutputValue = packed_val;
    return;
}

template <int ROWS, int COLS, int PLANES, int TYPE, int NPC, int WORDWIDTH, int TC, int K_ROWS, int K_COLS>
void Process_function_d(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _src_mat,
                        unsigned char kernel[K_ROWS][K_COLS],
                        xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _out_mat,
                        XF_TNAME(TYPE, NPC) buf[K_ROWS][(COLS >> XF_BITSHIFT(NPC))],
                        XF_PTUNAME(TYPE) src_buf[K_ROWS][XF_NPIXPERCYCLE(NPC) + (K_COLS - 1)],
                        XF_TNAME(TYPE, NPC) & P0,
                        uint16_t img_width,
                        uint16_t img_height,
                        uint16_t& shift_x,
                        ap_uint<13> row_ind[K_ROWS],
                        ap_uint<13> row,
                        int& rd_ind,
                        int& wr_ind) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    XF_TNAME(TYPE, NPC) buf_cop[K_ROWS];
    XF_PTUNAME(TYPE) OutputValue;
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=buf_cop complete dim=1
    // clang-format on

    uint16_t npc = XF_NPIXPERCYCLE(NPC);
    uint16_t col_loop_var = (((K_COLS >> 1) + (npc - 1)) / npc);

    ///////////////////////////////	Initializing src_buf buffer to zero	///////////////////////
    for (int k_row = 0; k_row < K_ROWS; k_row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=K_ROWS
        #pragma HLS unroll
        // clang-format on
        for (int k_col = 0; k_col < (npc + K_COLS - 1); k_col++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=K_COLS
            #pragma HLS unroll
            // clang-format on
            src_buf[k_row][k_col] = 0;
        }
    }

Col_Loop:
    for (ap_uint<13> col = 0; col < ((img_width) >> XF_BITSHIFT(NPC)) + col_loop_var;
         col++) // Image width should be multiple of NPC
    {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=TC
        #pragma HLS pipeline
        #pragma HLS LOOP_FLATTEN OFF
        // clang-format on

        if (col < (img_width >> XF_BITSHIFT(NPC))) {
            if (row < img_height) {
                buf[row_ind[K_ROWS - 1]][col] = _src_mat.read(rd_ind); // Read data
                rd_ind++;
            } else {
                buf[row_ind[K_ROWS - 1]][col] = 0;
            }
        }

        for (int idx = 0; idx < K_ROWS; idx++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=K_ROWS max=K_ROWS
            #pragma HLS UNROLL
            // clang-format on
            if (col < (img_width >> XF_BITSHIFT(NPC)))
                buf_cop[idx] = buf[(row_ind[idx])][col];
            else
                buf_cop[idx] = 0; // buf_cop[idx];
        }

        XF_PTUNAME(TYPE) src_buf_temp_copy[K_ROWS][XF_NPIXPERCYCLE(NPC)];
        XF_PTUNAME(TYPE) src_buf_temp_copy_extract[XF_NPIXPERCYCLE(NPC)];

        /////////////////////	Extracting 8 pixels from packed data ////////////////////////
        for (int extract_px = 0; extract_px < K_ROWS; extract_px++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=K_ROWS
            #pragma HLS unroll
            // clang-format on

            xfExtractPixels<NPC, XF_WORDWIDTH(TYPE, NPC), XF_DEPTH(TYPE, NPC)>(src_buf_temp_copy_extract,
                                                                               buf_cop[extract_px], 0);
            for (int ext_copy = 0; ext_copy < npc; ext_copy++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                src_buf[extract_px][(K_COLS - 1) + ext_copy] = src_buf_temp_copy_extract[ext_copy];
            }
        }
        ///////////////////////  Actual Process function cal	///////////////////////////////////

        XF_PTUNAME(TYPE) src_buf_temp_med_apply[K_ROWS][K_COLS];
        XF_PTUNAME(TYPE) OutputValues[XF_NPIXPERCYCLE(NPC)];

        for (int iter = 0; iter < npc; iter++) {
// clang-format off
            #pragma HLS pipeline
            // clang-format on
            for (int copyi = 0; copyi < K_ROWS; copyi++) {
                for (int copyj = 0; copyj < K_COLS; copyj++) {
                    src_buf_temp_med_apply[copyi][copyj] = src_buf[copyi][copyj + iter];
                }
            }

            dilate_function_apply<PLANES, TYPE, K_ROWS, K_COLS>(&OutputValue, src_buf_temp_med_apply,
                                                                kernel); // processing 8-pixels
            OutputValues[iter] = OutputValue;
        }

        //////////////////////	writing Processed pixel onto DDR //////////////////////////////////////////
        int start_write = (((K_COLS >> 1) + (npc - 1)) / npc);
        if (NPC == XF_NPPC8) {
            if (col == 0) {
                shift_x = 0;
                P0 = 0;
                xfPackPixels<NPC, XF_WORDWIDTH(TYPE, NPC), XF_DEPTH(TYPE, NPC)>(&OutputValues[0], P0, (K_COLS >> 1),
                                                                                (npc - (K_COLS >> 1)), shift_x);

            } else {
                xfPackPixels<NPC, XF_WORDWIDTH(TYPE, NPC), XF_DEPTH(TYPE, NPC)>(&OutputValues[0], P0, 0, (K_COLS >> 1),
                                                                                shift_x);
                _out_mat.write(wr_ind, P0);
                shift_x = 0;
                P0 = 0;
                xfPackPixels<NPC, XF_WORDWIDTH(TYPE, NPC), XF_DEPTH(TYPE, NPC)>(&OutputValues[0], P0, (K_COLS >> 1),
                                                                                (npc - (K_COLS >> 1)), shift_x);

                wr_ind++;
            }
        }
        if (NPC == XF_NPPC1) {
            if (col >= start_write) {
                _out_mat.write(wr_ind, OutputValue);
                wr_ind++;
            }
        }

        ////////////////////	Shifting the buffers in 8-pixel mode////////////////////////

        for (ap_uint<13> extract_px = 0; extract_px < K_ROWS; extract_px++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=K_ROWS max=K_ROWS
            // clang-format on
            for (ap_uint<13> col_warp = 0; col_warp < (K_COLS - 1); col_warp++) {
// clang-format off
                #pragma HLS UNROLL
                #pragma HLS LOOP_TRIPCOUNT min=K_COLS max=K_COLS
                // clang-format on

                src_buf[extract_px][col_warp] =
                    src_buf[extract_px][col_warp + npc]; // Shifting the 2 colums of src_buf to process next pixel
            }
        }

    } // Col_Loop

} //	end of processDilate

template <int ROWS, int COLS, int PLANES, int TYPE, int NPC, int WORDWIDTH, int TC, int K_ROWS, int K_COLS>
void xfdilate(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _src,
              xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _dst,
              uint16_t img_height,
              uint16_t img_width,
              unsigned char kernel[K_ROWS][K_COLS]) {
    ap_uint<13> row_ind[K_ROWS];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=row_ind complete dim=1
    // clang-format on

    uint16_t shift_x = 0;
    ap_uint<13> row, col;

    XF_PTUNAME(TYPE) src_buf[K_ROWS][XF_NPIXPERCYCLE(NPC) + (K_COLS - 1)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=src_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf complete dim=2
    // clang-format on

    XF_TNAME(TYPE, NPC) P0;

    XF_TNAME(TYPE, NPC) buf[K_ROWS][(COLS >> XF_BITSHIFT(NPC))];
// clang-format off
    #pragma HLS RESOURCE variable=buf core=RAM_S2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=buf complete dim=1
    // clang-format on

    // initializing row index

    for (int init_row_ind = 0; init_row_ind < K_ROWS; init_row_ind++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=K_ROWS
        // clang-format on
        row_ind[init_row_ind] = init_row_ind;
    }

    int rd_ind = 0;

// Reading the first row of image and initializing the buf
read_lines:
    for (ap_uint<13> i_row = (K_ROWS >> 1); i_row < (K_ROWS - 1); i_row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=K_ROWS
        // clang-format on
        for (ap_uint<13> i_col = 0; i_col<img_width>> XF_BITSHIFT(NPC); i_col++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=TC/NPC
            #pragma HLS pipeline
            #pragma HLS LOOP_FLATTEN OFF
            // clang-format on
            buf[i_row][i_col] = _src.read(rd_ind); // reading the rows of image
            rd_ind++;
        }
    }

// takes care of top borders ( intializing them with first row of image)
init_boarder:
    for (ap_uint<13> i_row = 0; i_row<K_ROWS>> 1; i_row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=K_ROWS
        // clang-format on
        for (ap_uint<13> i_col = 0; i_col<img_width>> XF_BITSHIFT(NPC); i_col++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=TC/NPC
            #pragma HLS UNROLL
            // clang-format on
            buf[i_row][i_col] = 0; // buf[K_ROWS>>1][i_col];
        }
    }

    int wr_ind = 0;
Row_Loop:
    for (row = (K_ROWS >> 1); row < img_height + (K_ROWS >> 1); row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on

        P0 = 0;
        Process_function_d<ROWS, COLS, PLANES, TYPE, NPC, WORDWIDTH, TC, K_ROWS, K_COLS>(
            _src, kernel, _dst, buf, src_buf, P0, img_width, img_height, shift_x, row_ind, row, rd_ind, wr_ind);

        // update indices (by cyclic shifting )
        ap_uint<13> zero_ind = row_ind[0];
        for (ap_uint<13> init_row_ind = 0; init_row_ind < K_ROWS - 1; init_row_ind++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=K_ROWS
            #pragma HLS UNROLL
            // clang-format on
            row_ind[init_row_ind] = row_ind[init_row_ind + 1];
        }
        row_ind[K_ROWS - 1] = zero_ind;
    } // Row_Loop
}

template <int BORDER_TYPE,
          int TYPE,
          int ROWS,
          int COLS,
          int K_SHAPE,
          int K_ROWS,
          int K_COLS,
          int ITERATIONS,
          int NPC = 1>
void dilate(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _src,
            xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _dst,
            unsigned char _kernel[K_ROWS * K_COLS]) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    unsigned short imgheight = _src.rows;
    unsigned short imgwidth = _src.cols;
#ifndef __SYNTHESIS__
    assert(BORDER_TYPE == XF_BORDER_CONSTANT && "Only XF_BORDER_CONSTANT is supported");
    assert(((imgheight <= ROWS) && (imgwidth <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
    //    FILE *fpx = fopen("xf_in.txt","w");
    //
    //           for(int i=0; i< 128;++i){
    //                          for(int j=0; j< (128>>3);++j){
    //                                         for(int m=0; m< XF_NPPC8; ++m){
    //                                                        for(int k=0; k< 1; ++k){
    //                                                                       fprintf(fpx,"%d\n",(unsigned char)
    //                                                                       _src.data[i*(128>>3)+j].chnl[m][k]);
    //                                                        }
    //                                         }
    //                          }
    //           }
    //
    //           fclose(fpx);

    if (K_SHAPE == XF_SHAPE_RECT) // iterations >1 is not supported for ELLIPSE and  CROSS
    {
#define NEW_K_ROWS (K_ROWS + ((ITERATIONS - 1) * (K_ROWS - 1)))
#define NEW_K_COLS (K_COLS + ((ITERATIONS - 1) * (K_COLS - 1)))

        unsigned char kernel_new[NEW_K_ROWS][NEW_K_COLS];
// clang-format off
        #pragma HLS array_partition variable=kernel_new dim=0
        // clang-format on
        for (unsigned char i = 0; i < NEW_K_ROWS; i++) {
            for (unsigned char j = 0; j < NEW_K_COLS; j++) {
                kernel_new[i][j] = 1; // _kernel[i*NEW_K_COLS+j];
            }
        }

        xfdilate<ROWS, COLS, XF_CHANNELS(TYPE, NPC), TYPE, NPC, 0, (COLS >> XF_BITSHIFT(NPC)) + (NEW_K_ROWS >> 1),
                 NEW_K_ROWS, NEW_K_COLS>(_src, _dst, imgheight, imgwidth, kernel_new);
    } else {
        unsigned char kernel[K_ROWS][K_COLS];
// clang-format off
        #pragma HLS array_partition variable=kernel complete dim=0
        // clang-format on
        for (unsigned char i = 0; i < K_ROWS; i++) {
            for (unsigned char j = 0; j < K_COLS; j++) {
                kernel[i][j] = _kernel[i * K_COLS + j];
            }
        }
        xfdilate<ROWS, COLS, XF_CHANNELS(TYPE, NPC), TYPE, NPC, 0, (COLS >> XF_BITSHIFT(NPC)) + (K_COLS >> 1), K_ROWS,
                 K_COLS>(_src, _dst, imgheight, imgwidth, kernel);
    }

    //         FILE *fp1 = fopen("xf_out.txt","w");
    //
    //               for(int i=0; i< 128;++i){
    //                              for(int j=0; j< (128>>3);++j){
    //                                             for(int m=0; m< XF_NPPC8; ++m){
    //                                                            for(int k=0; k< 1; ++k){
    //                                                                           fprintf(fp1, "%d\n",(unsigned char)
    //                                                                           _dst.data[i*(128>>3)+j].chnl[m][k]);
    //                                                                           //printf("%d ",
    //                                                                           imgInput.data[i*in_rgba.cols+j].chnl[m][k]);
    //                                                            }
    //                                             }
    //                              }
    //               }
    //
    //               fclose(fp1);

    return;
} // dilate
} // namespace cv
} // namespace xf
#endif
