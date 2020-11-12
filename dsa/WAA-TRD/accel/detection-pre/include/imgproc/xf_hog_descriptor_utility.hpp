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

#ifndef _XF_HOG_DESCRIPTOR_UTILITY_
#define _XF_HOG_DESCRIPTOR_UTILITY_

/*******************************************************************************************
 * 								          _HOG_ABS
 *******************************************************************************************
 *  Functional macro to find the absolute of a number
 *******************************************************************************************/
#define __HOG_ABS(X) \
    if (X < 0) X = -(X);

/*******************************************************************************************
 * 									  xFIdentifySignBits
 *******************************************************************************************
 *  Identifies the number of sign bits in the input value
 *******************************************************************************************/
static char xFIdentifySignBits(ap_uint<24> in_val) {
    ap_uint<1> flag = 0;
    char counter = 0;
    ap_uint<1> signbit = in_val.range(23, 23);

signBitsLoop:
    for (ap_uint<5> i = 0; i < 24; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=24 max=24
        #pragma HLS PIPELINE II=1
        // clang-format on

        if (flag == 0) {
            bool bit_val = in_val.range((23 - counter), (23 - counter));

            if (bit_val == signbit)
                counter++;
            else
                flag = 1;
        }
    }
    return counter;
}

/*******************************************************************************************
 * 										xFInverse24Kernel
 *******************************************************************************************
 *   Performs inverse 24 bit operation for the input value and returns the output and the
 *   format of the output
 *******************************************************************************************/
static unsigned int xFInverse24Kernel(ap_uint<24> in_val, int m, char s, char* n) {
    ap_uint<24> catch_val = in_val << s;

    int m1 = (m - s);

    unsigned short in_val_shifted = catch_val.range(23, 8);

    char n2;
    unsigned int out_val = xf::cv::Inverse(in_val_shifted, m1, &n2);

    unsigned int _out = out_val;
    *n = n2;

    char tmp_n = n2 - s;
    if (tmp_n < 0) {
        char tmp_n_2 = -tmp_n;
        tmp_n = 0;
        _out = out_val << tmp_n_2;
        *n = n2 + tmp_n_2; // keeping the fractional part more than 16 bits
    }

    return _out;
}

/*******************************************************************************************
 *	 									xFInverse24
 *******************************************************************************************
 *   Acts as a wrapper function for inverse 24 bit function
 *******************************************************************************************/
static unsigned int xFInverse24(ap_uint<24> in_val, int m, char* n) {
    unsigned int out_val;

    char s = xFIdentifySignBits(in_val); // find the upper sign bits (i.e) the unused bits
    out_val = xFInverse24Kernel(in_val, m, s, n);

    return out_val;
}
/******************************** End Of Inverse Wrapper ***********************************/

/*************************************************************************************
 * xFHOGReadFromStream: This function read from the stream and writes into the
 * 					output stream
 *
 * Input:
 * ------
 * in_stream: Stream containing the input data
 *
 * Output:
 * -------
 * out_stream: Stream to which the data is pushed
 *
 *************************************************************************************/
template <int ROWS, int COLS, int NOS, typename INPUT_TYPE, typename OUTPUT_TYPE>
void xFHOGReadFromStreamKernel(hls::stream<INPUT_TYPE>& in_stream,
                               hls::stream<OUTPUT_TYPE> out_stream[NOS],
                               uint16_t height,
                               uint16_t width) {
    ap_uint<32> input_data, i, j;
    ap_uint<5> upper_limit, lower_limit;
    ap_uint<3> k;

row_loop:
    for (i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on

    col_loop:
        for (j = 0; j < width; j++) {
// clang-format off
            #pragma HLS PIPELINE
            #pragma HLS LOOP_FLATTEN off
            #pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
            // clang-format on

            // reading the data from the stream
            input_data = in_stream.read();

            upper_limit = 7, lower_limit = 0;
            uchar_t in_data[3];

        no_of_channel_loop:
            for (k = 0; k < NOS; k++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=NOS max=NOS
                #pragma HLS UNROLL
                // clang-format on

                out_stream[k].write(input_data.range(upper_limit, lower_limit));
                in_data[k] = input_data.range(upper_limit, lower_limit);
                upper_limit += 8;
                lower_limit += 8;
            }
        }
    }
}

/*************************************************************************************
 * xFHOGReadFromStream: wrapper function for Read stream function
 *
 * Input:
 * ------
 * in_stream: Stream containing the input data
 *
 * Output:
 * -------
 * out_stream: Stream to which the data is pushed
 *
 *************************************************************************************/
template <int ROWS, int COLS, int NOS, typename INPUT_TYPE, typename OUTPUT_TYPE>
void xFHOGReadFromStream(hls::stream<INPUT_TYPE>& in_stream,
                         hls::stream<OUTPUT_TYPE> out_stream[NOS],
                         uint16_t height,
                         uint16_t width) {
    xFHOGReadFromStreamKernel<ROWS, COLS, NOS>(in_stream, out_stream, height, width);
}

/*********************************************************************************
 *  xFWriteHOGDescKernelRB: Write function for HoG repeated block configuration
 *
 *  Input:
 *  ------
 *  _block_strm: Block stream from HoG descriptor function, containing the
 *  	descriptor data
 *
 *  output:
 *  -------
 *  _desc_strm: Output descriptor data stream to the host
 *
 ********************************************************************************/
template <int NOB,
          int WIN_STRIDE,
          int CELL_HEIGHT,
          int CELL_WIDTH,
          int NOVBPW,
          int NOHBPW,
          int NOVW,
          int NOHW,
          int NOVB,
          int NOHB,
          int ROWS,
          int COLS,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int loop_count,
          bool USE_URAM>
void xFWriteHOGDescKernelRB(hls::stream<XF_SNAME(WORDWIDTH_SRC)>& _block_strm,
                            hls::stream<XF_SNAME(WORDWIDTH_DST)>& _desc_strm,
                            uint16_t novw,
                            uint16_t nohw,
                            uint16_t novb,
                            uint16_t nohb) {
    // feature buffer to hold the block data
    XF_SNAME(WORDWIDTH_SRC) feature_buf[NOVBPW][NOHB], block_data_1, block_data_2;
    if (USE_URAM) {
// clang-format off
        #pragma HLS RESOURCE variable=feature_buf core=RAM_1P_URAM
        // clang-format on
    }

    // indexes for accessing the feature buffer
    static ap_uint16_t row_idx = 0;
    static ap_uint16_t row_idx_buf[NOVBPW];

    ap_uint<8> step = XF_WORDDEPTH(WORDWIDTH_DST);
    uint16_t offset = 0;

    // loop indexes
    ap_uint16_t i, j;
    ap_uint16_t x, y, h;

// Initial filling of the BRAM (feature buffer)
loop_vert_blocks_per_win_1:
    for (i = 0; i < NOVBPW; i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN
    // clang-format on

    loop_horiz_blocks_per_win_1:
        for (j = 0; j < nohb; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=NOHB max=NOHB
            #pragma HLS pipeline
            // clang-format on

            block_data_1 = _block_strm.read();
            feature_buf[i][j] = block_data_1;
        }
    }

// Vertical window loop
main_vert_win:
    for (i = 0; i < novw; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=NOVW max=NOVW
        #pragma HLS LOOP_FLATTEN off
        // clang-format on
        ap_uint16_t row_ptr = row_idx;

    // Setting the row index for circular buffer organization
    settingIndex_1:
        for (j = 0; j < NOVBPW; j++) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on

            if (row_ptr >= NOVBPW) row_ptr = 0;

            row_idx_buf[j] = row_ptr++;
        }

    // horizontal Window loop
    main_horiz_win:
        for (j = 0; j < nohw; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=NOHW max=NOHW
            #pragma HLS LOOP_FLATTEN
        // clang-format on
        // vertical block loop
        main_vert_blocks_per_win_2:
            for (x = 0; x < NOVBPW; x++) {
            main_horiz_blocks_per_win_2:
                for (y = 0; y < NOHBPW; y++) {
// clang-format off
                    #pragma HLS PIPELINE
                    // clang-format on
                    block_data_2 = feature_buf[row_idx_buf[x]][j + y];

                    offset = 0;

                write_block_loop:
                    for (h = 0; h < loop_count; h++) {
                        _desc_strm.write(block_data_2.range(offset + (step - 1), offset));
                        offset += step;
                    }
                }
            }

            // replacing the non-repeating blocks with new blocks
            /*								ARRAY filling position
             ------------------------------------------------------------------------------------
             |///|||||||||||		  -   -   - |
             |
             ------------------------------------------------------------------------------------
             |	 |||||||||||		  -   -   - |
             |
             ------------------------------------------------------------------------------------
             |	 |||||||||||		  -   -   - |
             |
             ------------------------------------------------------------------------------------
             |	 |||||||||||		  -   -   - |
             |
             ------------------------------------------------------------------------------------

             |///|  -> Replaced block in the feature buffer

             |||||  -> Window to be processed in the next iteration                 			 */

            if (i != (novw - 1)) {
                block_data_1 = _block_strm.read();
                feature_buf[row_idx_buf[0]][j] = block_data_1;
            }
        }

        /*								ARRAY filling position
         ------------------------------------------------------------------------------------
         |		|		|		  -   -   -
         |////////|
         ------------------------------------------------------------------------------------
         |		|		|		  -   -   -
         |		|
         ------------------------------------------------------------------------------------
         |		|		|		  -   -   -
         |		|
         ------------------------------------------------------------------------------------
         |		|		|		  -   -   -
         |		|
         ------------------------------------------------------------------------------------
         */

        // filling the newer last block's data
        if (i != (novw - 1)) {
            for (j = 0; j < (NOHBPW - 1); j++) {
// clang-format off
                #pragma HLS pipeline
                // clang-format on
                block_data_1 = _block_strm.read();
                feature_buf[row_idx_buf[0]][j + nohw] = block_data_1;
            }
        }
        row_idx++;

        // resetting the row index
        if (row_idx == NOVBPW) {
            row_idx = 0;
        }
    }
    row_idx = 0;
    for (ap_uint<16> i = 0; i < NOVBPW; i++) {
        row_idx_buf[i] = 0;
    }
}

/******************************************************************************************************
 * xFWriteHOGDescRB: Top function for HOG
 ******************************************************************************************************
 *
 * Input:
 * ------
 * _block_stream: Stream containing the input desc data
 *
 * Output:
 * -------
 * _desc_stream: Output descriptor stream
 *
 ******************************************************************************************************/
template <int WIN_HEIGHT,
          int WIN_WIDTH,
          int WIN_STRIDE,
          int CELL_HEIGHT,
          int CELL_WIDTH,
          int NOB,
          int ROWS,
          int COLS,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          bool USE_URAM>
void xFWriteHOGDescRB(hls::stream<XF_SNAME(WORDWIDTH_SRC)>& _block_strm,
                      hls::stream<XF_SNAME(WORDWIDTH_DST)>& _desc_strm,
                      uint16_t _height,
                      uint16_t _width) {
    uint16_t novw = (((_height - WIN_HEIGHT) / WIN_STRIDE) + 1);
    uint16_t nohw = (((_width - WIN_WIDTH) / WIN_STRIDE) + 1);
    uint16_t novb = ((_height / CELL_HEIGHT) - 1);
    uint16_t nohb = ((_width / CELL_WIDTH) - 1);
    xFWriteHOGDescKernelRB<NOB, WIN_STRIDE, CELL_HEIGHT, CELL_WIDTH, ((WIN_HEIGHT / CELL_HEIGHT) - 1),
                           ((WIN_WIDTH / CELL_WIDTH) - 1), (((ROWS - WIN_HEIGHT) / WIN_STRIDE) + 1),
                           (((COLS - WIN_WIDTH) / WIN_STRIDE) + 1), ((ROWS / CELL_HEIGHT) - 1),
                           ((COLS / CELL_WIDTH) - 1), ROWS, COLS, DEPTH_SRC, DEPTH_DST, NPC, WORDWIDTH_SRC,
                           WORDWIDTH_DST, (XF_WORDDEPTH(WORDWIDTH_SRC) / XF_WORDDEPTH(WORDWIDTH_DST)), USE_URAM>(
        _block_strm, _desc_strm, novw, nohw, novb, nohb);
}

/*******************************************************************************************
 * 							 			xFWriteHOGDescNRB
 *******************************************************************************************
 *   This function reads the data form the _block_mat and memcopies into the DDR
 *******************************************************************************************/
template <int ROWS,
          int COLS,
          int DEPTH,
          int NPC,
          int WORDWIDTH,
          typename OUTPUT_TYPE,
          int NOVB,
          int NOHB,
          int step_val,
          int loop_count,
          int word_size,
          int TC>
void xFWriteHOGDescKernelNRB(hls::stream<XF_SNAME(WORDWIDTH)>& _block_strm,
                             hls::stream<OUTPUT_TYPE>& _desc_strm,
                             uint16_t novb,
                             uint16_t nohb) {
    XF_SNAME(WORDWIDTH) block_data;
    OUTPUT_TYPE block_descriptor;
    uint32_t offset = 0;
    uchar_t step = word_size << 3;
    int k = 0;
    int i;
    ap_uint<8> j;

write_loop_1:
    for (i = 0; i < (novb * nohb); i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
        #pragma HLS PIPELINE
        // clang-format on

        block_data = _block_strm.read();
        offset = 0;

    write_loop_2:
        for (j = 0; j < loop_count; j++) {
            block_descriptor = block_data.range((offset + (step - 1)), offset);
            _desc_strm.write(block_descriptor);
            offset += step;
        }
    }
}

/*******************************************************************************************
 * 							 			xFWriteHOGDescNRB
 *******************************************************************************************
 *   Acts as a wrapper function for Descriptor write kernel function
 *******************************************************************************************/
template <int BLOCK_HEIGHT,
          int BLOCK_WIDTH,
          int CELL_HEIGHT,
          int CELL_WIDTH,
          int NOB,
          int HOG_TYPE,
          int ROWS,
          int COLS,
          int DEPTH,
          int NPC,
          int WORDWIDTH,
          typename OUTPUT_TYPE>
void xFWriteHOGDescNRB(hls::stream<XF_SNAME(WORDWIDTH)>& _block_strm,
                       hls::stream<OUTPUT_TYPE>& _desc_strm,
                       uint16_t height,
                       uint16_t width) {
    int novb = ((height / CELL_HEIGHT) - ((BLOCK_HEIGHT / CELL_HEIGHT) - 1));
    int nohb = ((width / CELL_WIDTH) - ((BLOCK_WIDTH / CELL_WIDTH) - 1));
    if (HOG_TYPE == XF_SHOG) {
        xFWriteHOGDescKernelNRB<ROWS, COLS, DEPTH, NPC, WORDWIDTH, OUTPUT_TYPE,
                                ((ROWS / CELL_HEIGHT) - ((BLOCK_HEIGHT / CELL_HEIGHT) - 1)),
                                ((COLS / CELL_WIDTH) - ((BLOCK_WIDTH / CELL_WIDTH) - 1)), (sizeof(OUTPUT_TYPE) << 3),
                                ((NOB * sizeof(XF_PTNAME(DEPTH))) / sizeof(OUTPUT_TYPE)), sizeof(OUTPUT_TYPE),
                                ((ROWS / CELL_HEIGHT) - ((BLOCK_HEIGHT / CELL_HEIGHT) - 1)) *
                                    ((COLS / CELL_WIDTH) - ((BLOCK_WIDTH / CELL_WIDTH) - 1))>(_block_strm, _desc_strm,
                                                                                              novb, nohb);
    } else if (HOG_TYPE == XF_DHOG) {
        xFWriteHOGDescKernelNRB<
            ROWS, COLS, DEPTH, NPC, WORDWIDTH, OUTPUT_TYPE, ((ROWS / CELL_HEIGHT) - ((BLOCK_HEIGHT / CELL_HEIGHT) - 1)),
            ((COLS / CELL_WIDTH) - ((BLOCK_WIDTH / CELL_WIDTH) - 1)), (sizeof(OUTPUT_TYPE) << 3),
            ((NOB * (BLOCK_WIDTH / CELL_WIDTH) * (BLOCK_HEIGHT / CELL_HEIGHT) * sizeof(XF_PTNAME(DEPTH))) /
             sizeof(OUTPUT_TYPE)),
            sizeof(OUTPUT_TYPE), ((ROWS / CELL_HEIGHT) - ((BLOCK_HEIGHT / CELL_HEIGHT) - 1)) *
                                     ((COLS / CELL_WIDTH) - ((BLOCK_WIDTH / CELL_WIDTH) - 1))>(_block_strm, _desc_strm,
                                                                                               novb, nohb);
    } else {
        assert(((HOG_TYPE == XF_SHOG) || (HOG_TYPE == XF_DHOG)) && "HOG_TYPE must be either XF_SHOG or XF_DHOG");
    }
}

/*******************************************************************************************
 *		   						xFWriteHOGDescKernelNRB2
 *******************************************************************************************
 *   Alternative method for writing the descriptors to the host
 *******************************************************************************************/
/*template<int ROWS, int COLS, int DEPTH, int NPC, int WORDWIDTH, typename OUTPUT_TYPE,
int NOB, int NOVB, int NOHB, int step_val, int loop_count, int word_size>
void xFWriteHOGDescKernelNRB2( auviz::Mat<ROWS,COLS,DEPTH,NPC,WORDWIDTH>& _block_mat,
                OUTPUT_TYPE* _out_ptr)
{
        uint16_t mem_offset = 0, counter = 0,
                        pack_offset = 0, pack_step = 144;

        ap_uint<576> block_data;

        write_loop_1:
        for(uint16_t i = 0; i < (NOVB*NOHB); i++)
        {
// clang-format off
#pragma HLS PIPELINE
// clang-format on

                block_data.range(pack_offset+(pack_step-1),pack_offset) = _block_mat.read();

                pack_offset += pack_step;
                counter++;

                if(counter == 4)
                {
                        uchar_t step = step_val; uint16_t offset = 0;
                        OUTPUT_TYPE out_data[1];

                        write_loop_2:
                        for(uint16_t j = 0; j < NOB; j++)
                        {
                                out_data[0] = block_data.range((offset+(step-1)),offset);
                                memcpy(_out_ptr+mem_offset,out_data,word_size);
                                offset += step;
                                mem_offset++;
                        }
                        pack_offset = 0;
                        counter = 0;
                }
        }

        for(uchar_t i = 0; i < (4-counter); i++)
        {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=3
#pragma HLS PIPELINE
// clang-format on

                block_data.range(pack_offset+(pack_step-1),pack_offset) = 0;
                pack_offset += pack_step;
        }

        uchar_t step = step_val; uint16_t offset = 0;
        OUTPUT_TYPE out_data[1];

        write_loop_3:
        for(uint16_t j = 0; j < NOB; j++)
        {
// clang-format off
#pragma HLS PIPELINE
// clang-format on

                out_data[0] = block_data.range((offset+(step-1)),offset);
                memcpy(_out_ptr+mem_offset,out_data,word_size);
                offset += step;
                mem_offset++;
        }
}*/

/*****************************************************************************
 *                          xFDHOGwriteDescRB2
 *****************************************************************************
 *  This function finds the various repetitions of the blocks and writes the
 *  block data to the memory.
 *
 *  block: contains the normalized block data (I)
 *  _out_ptr : output descriptor memory location (O)
 *  bi & bj: indexes to find the memory offset (I)
 *****************************************************************************/
/*template<int block_size, int NOHW, int NOVW, int NOHCPB, int NOVCPB,
int NOVBPW, int NOHBPW, typename block_type, typename OUT_TYPE>

void xFDHOGwriteDescRB2(block_type* block, uint16_t bi, uint16_t bj,
                hls::stream<OUT_TYPE> &_out_ptr)
{
        int16_t k = (bi - (NOVBPW-1)), k_limit = (bi + 1), p_tmp = (bi + 1);
        int16_t l = (bj - (NOHBPW-1)), l_limit = (bj + 1), q_tmp = (bj + 1);
        OUT_TYPE temp;
        if(k < 0)	k = 0;
        if(l < 0)	l = 0;

        if(k_limit > NOVW)	  k_limit = NOVW;
        if(l_limit > NOHW)	  l_limit = NOHW;

        uint16_t word_size = block_size * sizeof(block_type);
        uint16_t win_size = ((NOVBPW * NOHBPW) * block_size);
        ap_uint<13> high=31,low=0;
        for(uint16_t m = k; m < k_limit; m++)
        {
// clang-format off
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_TRIPCOUNT min=15 max=15
// clang-format on

                for(uint16_t n = l; n < l_limit; n++)
                {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=7 max=7
#pragma HLS pipeline
// clang-format on

                        uint16_t p = ((p_tmp-1) - m);
                        uint16_t q = ((q_tmp-1) - n);

                        int offset = ((m * NOHW + n) * win_size +
                                        (p * NOHBPW + q) * block_size);
                        high=31,low=0;

                        for(uint16_t k=0;k<18;k++)
                        {
                                temp.range(high,low)=block[k];
                                high=high+32;
                                low=low+32;
                        }
                        _out_ptr.write(temp);
                }
        }
}*/

/*******************************************************************************************
 * 							 		 xFHOGDuplicateKernel
 *******************************************************************************************
 *   Duplicates the input stream into two copies
 *******************************************************************************************/
template <int ROWS, int COLS, int DEPTH, int NPC, int WORDWIDTH, int TC>
void xFHOGDuplicateKernel(hls::stream<XF_SNAME(WORDWIDTH)>& _src_strm,
                          hls::stream<XF_SNAME(WORDWIDTH)>& _dst1_strm,
                          hls::stream<XF_SNAME(WORDWIDTH)>& _dst2_strm,
                          uint16_t height,
                          uint16_t width) {
    ap_uint<16> i, j;
Row_Loop:
    for (i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        #pragma HLS LOOP_FLATTEN OFF
    // clang-format on

    Col_Loop:
        for (j = 0; j < width; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            #pragma HLS PIPELINE
            // clang-format on
            XF_SNAME(WORDWIDTH) tmp_src;
            tmp_src = _src_strm.read();
            _dst1_strm.write(tmp_src);
            _dst2_strm.write(tmp_src);
        }
    }
}

/*******************************************************************************************
 * 							 		  xFHOGDuplicate
 *******************************************************************************************
 *   Wrapper function for the Duplicate kernel function
 *******************************************************************************************/
template <int ROWS, int COLS, int DEPTH, int NPC, int WORDWIDTH>
void xFHOGDuplicate(hls::stream<XF_SNAME(WORDWIDTH)>& _src_strm,
                    hls::stream<XF_SNAME(WORDWIDTH)>& _dst1_strm,
                    hls::stream<XF_SNAME(WORDWIDTH)>& _dst2_strm,
                    uint16_t height,
                    uint16_t width) {
    xFHOGDuplicateKernel<ROWS, COLS, DEPTH, NPC, WORDWIDTH, (COLS >> XF_BITSHIFT(NPC))>(_src_strm, _dst1_strm,
                                                                                        _dst2_strm, height, width);
}

/*******************************************************************************************
 * 							 				xFSqrtHOG
 *******************************************************************************************
 *   Performs the square root operation. Input must be in hls_fixed point unsigned type and
 *   the output will be in ap_unsigned type
 *******************************************************************************************/
#define Wg 1
template <int OBN, int W, int I, ap_q_mode _AP_Q, ap_o_mode _AP_O>
ap_uint<OBN> xFSqrtHOG(ap_ufixed<W, I, _AP_Q, _AP_O> x) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    assert(I >= 0 && "Number of integer bits for sqrt() must be greater than zero");
    assert(W >= I && "Number of integer bits for sqrt() must be less than or equal to total width");
    ap_ufixed<W + Wg, I> factor = 0;
    bool offset;

    // Since input bits are handled in pairs, the
    // start condition for even and odd integer widths
    // are handled slightly differently.
    if (I % 2 == 0)
        offset = 1;
    else
        offset = 0;

    factor[W + Wg - 1 - offset] = 1;
    ap_ufixed<W + Wg + 1, I + 1> result = 0;
    ap_ufixed<W + Wg + 2, I + 2> x2 = x;
    for (uchar_t i = W + Wg - offset; i > (I - 1) / 2; i -= 1) {
// clang-format off
        #pragma HLS PIPELINE
        // clang-format on

        ap_ufixed<W + 2 + Wg, I + 2> t = (result << 1) + factor;
        ap_ufixed<W + Wg, I> thisfactor = 0;
        if (x2 >= t) {
            x2 -= t;
            thisfactor = factor;
        }
        result = result + thisfactor;
        factor >>= 1;
        x2 <<= 1;
    }

    return result >> ((I - 1) >> 1);
}

#endif // _XF_HOG_DESCRIPTOR_UTILITY_
