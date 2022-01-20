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

#ifndef __XF_PYR_DENSE_OPTICAL_FLOW_MEDIAN_BLUR__
#define __XF_PYR_DENSE_OPTICAL_FLOW_MEDIAN_BLUR__
template <int NPC, int DEPTH, int WIN_SZ, int WIN_SZ_SQ, int FLOW_WIDTH, int FLOW_INT>
void auMedianProc(ap_fixed<FLOW_WIDTH, FLOW_INT> OutputValues[1],
                  ap_fixed<FLOW_WIDTH, FLOW_INT> src_buf[WIN_SZ][1 + (WIN_SZ - 1)],
                  ap_uint<8> win_size) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    ap_fixed<FLOW_WIDTH, FLOW_INT> array[WIN_SZ_SQ];
// #pragma HLS RESOURCE variable=array core=DSP48
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=array complete dim=1
    // clang-format on

    int array_ptr = 0;
// OutputValues[0] = src_buf[WIN_SZ>>1][WIN_SZ>>1];
// return;
Compute_Grad_Loop:
    for (int copy_arr = 0; copy_arr < WIN_SZ; copy_arr++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
        #pragma HLS UNROLL
        // clang-format on
        for (int copy_in = 0; copy_in < WIN_SZ; copy_in++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
            #pragma HLS UNROLL
            // clang-format on
            array[array_ptr] = src_buf[copy_arr][copy_in];
            array_ptr++;
        }
    }
// OutputValues[0] = array[(WIN_SZ_SQ)>>1];
// return;

auApplyMaskLoop:
    for (int16_t j = 0; j <= WIN_SZ_SQ - 1; j++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
        // clang-format on
        int16_t tmp = j & 0x0001;
        if (tmp == 0) {
        auSortLoop1:
            for (int i = 0; i <= ((WIN_SZ_SQ >> 1) - 1); i++) // even sort
            {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                #pragma HLS unroll
                // clang-format on
                int c = (i * 2);
                int c1 = (c + 1);

                if (array[c] < array[c1]) {
                    ap_fixed<FLOW_WIDTH, FLOW_INT> temp = array[c];
                    array[c] = array[c1];
                    array[c1] = temp;
                }
            }
        }

        else {
        auSortLoop2:
            for (int i = 0; i <= ((WIN_SZ_SQ >> 1) - 1); i++) // odd sort WINDOW_SIZE_H>>1 -1
            {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                #pragma HLS unroll
                // clang-format on
                int c = (i * 2);
                int c1 = (c + 1);
                int c2 = (c + 2);
                if (array[c1] < array[c2]) {
                    ap_fixed<FLOW_WIDTH, FLOW_INT> temp = array[c1];
                    array[c1] = array[c2];
                    array[c2] = temp;
                }
            }
        }
    }

    // OutputValues[0] = auapplymedian3x3<DEPTH, WIN_SZ>(array, WIN_SZ);
    OutputValues[0] = array[(WIN_SZ_SQ) >> 1];
    return;
}

template <int ROWS,
          int COLS,
          int DEPTH,
          int NPC,
          int WORDWIDTH,
          int TC,
          int WIN_SZ,
          int WIN_SZ_SQ,
          int FLOW_WIDTH,
          int FLOW_INT>
void ProcessMedian3x3(hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& _src_mat,
                      hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& _out_mat,
                      hls::stream<bool>& flag,
                      ap_fixed<FLOW_WIDTH, FLOW_INT> buf[WIN_SZ][(COLS >> NPC)],
                      ap_fixed<FLOW_WIDTH, FLOW_INT> src_buf[WIN_SZ][1 + (WIN_SZ - 1)],
                      ap_fixed<FLOW_WIDTH, FLOW_INT> OutputValues[1],
                      ap_fixed<FLOW_WIDTH, FLOW_INT>& P0,
                      uint16_t img_width,
                      uint16_t img_height,
                      uint16_t& shift_x,
                      ap_uint<13> row_ind[WIN_SZ],
                      ap_uint<13> row,
                      ap_uint<8> win_size) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    ap_fixed<FLOW_WIDTH, FLOW_INT> buf_cop[WIN_SZ];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=buf_cop complete dim=1
    // clang-format on

    uint16_t npc = 1;
Col_Loop:
    for (ap_uint<16> col = 0; col < img_width + (WIN_SZ >> 1); col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=TC
        #pragma HLS pipeline
        #pragma HLS LOOP_FLATTEN OFF
        // clang-format on

        if (row < img_height && col < img_width)
            buf[row_ind[win_size - 1]][col] = _src_mat.read(); // Read data
        else
            buf[row_ind[win_size - 1]][col] = 0;

        for (int copy_buf_var = 0; copy_buf_var < WIN_SZ; copy_buf_var++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
            #pragma HLS UNROLL
            // clang-format on
            if ((row > (img_height - 1)) && (copy_buf_var > (win_size - 1 - (row - (img_height - 1))))) {
                buf_cop[copy_buf_var] = buf[(row_ind[win_size - 1 - (row - (img_height - 1))])][col];
            } else {
                buf_cop[copy_buf_var] = buf[(row_ind[copy_buf_var])][col];
            }
        }

        // if(NPC == AU_NPPC8)
        // {
        // for(int extract_px=0;extract_px<win_size;extract_px++)
        // {
        // #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
        // auExtractPixels<NPC, WORDWIDTH, DEPTH>(&src_buf[extract_px][win_size-1], buf_cop[extract_px], 0);
        // }
        // }
        // else
        {
            for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                #pragma HLS UNROLL
                // clang-format on
                if (col < img_width) {
                    src_buf[extract_px][win_size - 1] = buf_cop[extract_px];
                } else {
                    src_buf[extract_px][win_size - 1] = src_buf[extract_px][win_size - 2];
                }
            }
        }

        auMedianProc<NPC, DEPTH, WIN_SZ, WIN_SZ_SQ, FLOW_WIDTH, FLOW_INT>(OutputValues, src_buf, win_size);
        if (col >= (win_size >> 1)) {
            // auPackPixels<NPC, WORDWIDTH, DEPTH>(&OutputValues[0], P0, 0, 1, shift_x);
            // shift_x = 0;
            // P0 = 0;
            // auPackPixels<NPC, WORDWIDTH, DEPTH>(&OutputValues[0], P0, 1, (npc-1), shift_x);
            if (flag.read()) {
                _out_mat.write(OutputValues[0]);
            } else {
                _out_mat.write(OutputValues[0]); // can use the disable medianblur filter flag at a later point
            }
        }

        for (int wrap_buf = 0; wrap_buf < WIN_SZ; wrap_buf++) {
// clang-format off
            #pragma HLS UNROLL
            #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
            // clang-format on
            for (int col_warp = 0; col_warp < WIN_SZ - 1; col_warp++) {
// clang-format off
                #pragma HLS UNROLL
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                // clang-format on
                if ((col >= (img_width - 1) - (win_size >> 1)) && (wrap_buf >= win_size >> 1)) {
                    src_buf[wrap_buf][col_warp] = src_buf[win_size - 1][col_warp];
                }
                if (col == 0) {
                    src_buf[wrap_buf][col_warp] = src_buf[wrap_buf][win_size - 1];
                } else {
                    src_buf[wrap_buf][col_warp] = src_buf[wrap_buf][col_warp + 1];
                }
            }
        }
    } // Col_Loop
}

template <int ROWS,
          int COLS,
          int DEPTH,
          int NPC,
          int WORDWIDTH,
          int TC,
          int WIN_SZ,
          int WIN_SZ_SQ,
          int FLOW_WIDTH,
          int FLOW_INT,
          bool USE_URAM>
void auMedian3x3(hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& _src_mat,
                 hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& _out_mat,
                 hls::stream<bool>& flag,
                 ap_uint<8> win_size,
                 uint16_t img_height,
                 uint16_t img_width) {
    ap_uint<13> row_ind[WIN_SZ];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=row_ind complete dim=1
    // clang-format on

    ap_uint<8> buf_size = 1 + (WIN_SZ - 1);
    uint16_t shift_x = 0;
    ap_uint<16> row, col;

    ap_fixed<FLOW_WIDTH, FLOW_INT> OutputValues[1];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=OutputValues complete dim=1
    // clang-format on

    ap_fixed<FLOW_WIDTH, FLOW_INT> src_buf[WIN_SZ][1 + (WIN_SZ - 1)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=src_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf complete dim=2
    // clang-format on
    // src_buf1 et al merged
    ap_fixed<FLOW_WIDTH, FLOW_INT> P0;

    ap_fixed<FLOW_WIDTH, FLOW_INT> buf[WIN_SZ][(COLS >> NPC)];

    if (USE_URAM) {
// clang-format off
        #pragma HLS ARRAY_RESHAPE variable=buf complete dim=1
        #pragma HLS RESOURCE variable=buf core=RAM_S2P_URAM
        // clang-format on
    } else {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=buf complete dim=1
        #pragma HLS RESOURCE variable=buf core=RAM_S2P_BRAM
        // clang-format on
    }
    // initializing row index

    for (int init_row_ind = 0; init_row_ind < win_size; init_row_ind++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
        // clang-format on
        row_ind[init_row_ind] = init_row_ind;
    }

read_lines:
    for (int init_buf = row_ind[win_size >> 1]; init_buf < row_ind[win_size - 1]; init_buf++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
        // clang-format on
        for (col = 0; col < img_width; col++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            #pragma HLS pipeline
            #pragma HLS LOOP_FLATTEN OFF
            // clang-format on
            buf[init_buf][col] = _src_mat.read();
        }
    }

    // takes care of top borders
    for (col = 0; col < img_width; col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=TC
        // clang-format on
        for (int init_buf = 0; init_buf<WIN_SZ>> 1; init_buf++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
            #pragma HLS UNROLL
            // clang-format on
            buf[init_buf][col] = buf[row_ind[win_size >> 1]][col];
        }
    }

Row_Loop:
    for (row = (win_size >> 1); row < img_height + (win_size >> 1); row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on

        // //initialize buffers to be sent for sorting
        // for(int init_src=0;init_src<(win_size>>1);init_src++)
        // {
        // for(int init_src1=0;init_src1<win_size;init_src1++)
        // {
        // #pragma HLS UNROLL
        // src_buf[init_src1][init_src] =  buf[row_ind[init_src1]][0];
        // }
        // }
        P0 = 0;
        ProcessMedian3x3<ROWS, COLS, DEPTH, NPC, WORDWIDTH, TC, WIN_SZ, WIN_SZ_SQ, FLOW_WIDTH, FLOW_INT>(
            _src_mat, _out_mat, flag, buf, src_buf, OutputValues, P0, img_width, img_height, shift_x, row_ind, row,
            win_size);

        // update indices
        ap_uint<13> zero_ind = row_ind[0];
        for (int init_row_ind = 0; init_row_ind < WIN_SZ - 1; init_row_ind++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
            #pragma HLS UNROLL
            // clang-format on
            row_ind[init_row_ind] = row_ind[init_row_ind + 1];
        }
        row_ind[win_size - 1] = zero_ind;

    } // Row_Loop
}

template <int ROWS,
          int COLS,
          int DEPTH,
          int NPC,
          int WORDWIDTH,
          int PIPELINEFLAG,
          int WIN_SZ,
          int WIN_SZ_SQ,
          int FLOW_WIDTH,
          int FLOW_INT,
          bool USE_URAM>
void auMedianBlur(hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& _src,
                  hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& _dst,
                  hls::stream<bool>& flag,
                  ap_uint<8> win_size,
                  int _border_type,
                  uint16_t imgheight,
                  uint16_t imgwidth) {
// clang-format off
    #pragma HLS inline off
// clang-format on

// #pragma HLS license key=IPAUVIZ_CV_BASIC
// assert(_border_type == AU_BORDER_CONSTANT && "Only AU_BORDER_CONSTANT is supported");

#ifndef __SYNTHESIS__
    assert(((imgheight <= ROWS) && (imgwidth <= COLS)) && "ROWS and COLS should be greater than input image");

    assert((win_size <= WIN_SZ) && "win_size must not be greater than WIN_SZ");
#endif

    imgwidth = imgwidth >> NPC;

    auMedian3x3<ROWS, COLS, DEPTH, NPC, WORDWIDTH, (COLS >> NPC) + (WIN_SZ >> 1), WIN_SZ, WIN_SZ_SQ, FLOW_WIDTH,
                FLOW_INT, USE_URAM>(_src, _dst, flag, WIN_SZ, imgheight, imgwidth);
}
#endif
