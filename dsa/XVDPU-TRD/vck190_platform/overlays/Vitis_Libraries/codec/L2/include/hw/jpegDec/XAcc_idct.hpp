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

/**
 * @file XAcc_idct.cpp
 * @brief mcu_reorder template iq+idct+izigzag function implementation.
 *
 * This file is part of HLS algorithm library.
 */

#ifndef _XACC_IDCT_HPP_
#define _XACC_IDCT_HPP_

#include "XAcc_jpegdecoder.hpp"

namespace xf {
namespace codec {
namespace details {
// ------------------------------------------------------------
enum {
    w1 = 2841, // 2048*sqrt(2)*cos(1*pi/16)
    w2 = 2676, // 2048*sqrt(2)*cos(2*pi/16)
    w3 = 2408, // 2048*sqrt(2)*cos(3*pi/16)
    w5 = 1609, // 2048*sqrt(2)*cos(5*pi/16)
    w6 = 1108, // 2048*sqrt(2)*cos(6*pi/16)
    w7 = 565,  // 2048*sqrt(2)*cos(7*pi/16)

    w1pw7 = w1 + w7,
    w1mw7 = w1 - w7,
    w2pw6 = w2 + w6,
    w2mw6 = w2 - w6,
    w3pw5 = w3 + w5,
    w3mw5 = w3 - w5,

    r2 = 181 // 256/sqrt(2)
};
// typedef ap_int<25> idct25_t;
typedef int idct25_t;
typedef ap_int<24> idctm_t;

// ------------------------------------------------------------
/**
 * @brief Level 1 : decode all mcu with burst read data from DDR
 *
 * @tparam _WAxi size of data path in dataflow region, in bit.
 *         when _WAxi is 16, the decoder could decode one symbol per cycle in about 99% cases.
 *         when _WAxi is 8 , the decoder could decode one symbol per cycle in about 80% cases, but use less resource.
 *
 * @param ptr the pointer to DDR.
 * @param sz the total bytes to be read from DDR.
 * @param c the column to be read from AXI in the case when AXI_WIDTH > 8*sizeof(char)
 * @param dht_tbl1/dht_tbl2 the segment data of Define huffman table marker.
 * @param hls_cmp the shift register organized by the index of each color component.
 * @param hls_mbs the number of blocks in mcu for each component.
 * @param q_tables the quent table of huffman.
 * @param img_info include hls_cs_cmpc/hls_mbs/hls_mcuh/hls_mcuc is just for csim tests.
 * @param bas_info the basic infomation for the image.
 * @param yuv_mcu_pointer pointer to the hls_mcuc*{hls_mbs[0~2]*{Y/U/V}}
 */
template <int _WAxi>
void decoder_jpg_full_top(ap_uint<_WAxi>* ptr,
                          const int sz,
                          const int c,
                          const uint16_t dht_tbl1[2][2][1 << DHT1],
                          //
                          uint8_t ac_value_buckets[2][165],
                          HCODE_T ac_huff_start_code[2][AC_N],
                          int16_t ac_huff_start_addr[2][16],
                          //
                          uint8_t dc_value_buckets[2][12],
                          HCODE_T dc_huff_start_code[2][DC_N],
                          int16_t dc_huff_start_addr[2][16],
                          // image info
                          ap_uint<12> hls_cmp,
                          const uint8_t hls_mbs[MAX_NUM_COLOR],
                          const uint8_t q_tables[2][8][8],
                          const img_info img_info,
                          const bas_info bas_info,
                          // ouput
                          int& rtn2,
                          uint32_t& rst_cnt,
                          ap_uint<64>* yuv_mcu_pointer);
}

namespace details {
// ------------------------------------------------------------
inline void hls_idct_h(uint32_t hls_mcuc,
                       ap_uint<3> mcu_cmp,
                       COLOR_FORMAT fmt,
                       hls::stream<idct_in_t> str_rast8[8],
                       const uint8_t q[2][8][8],
                       hls::stream<idctm_t> strm_intermed[8][8]) {
    bool ignore_dc = true;
#pragma HLS ARRAY_PARTITION variable = str_rast8 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = q complete dim = 2

// Horizontal 1-D IDCT.
H_LOOP:
    for (int mcu = 0; mcu < hls_mcuc; mcu++) {
        for (int cmp = 0; cmp < mcu_cmp; cmp++) {
            for (int y = 0; y < 8; ++y) {
#pragma HLS DEPENDENCE array inter false
#pragma HLS PIPELINE II = 1
                idct_in_t c0 = str_rast8[0].read();
// ap_uint<5> idx = idx_coef_row.read();

#ifndef __SYNTHESIS__
                if (c0 >= 900) {
                    //_XF_IMAGE_PRINT
                    printf("Brightest Pixel Search: %d \n ", (int)c0);
                }
#endif

                bool idx_q; // = (idx[1]) ? 0 : 1; // Y ? 0: 1
                if (fmt == C420) {
                    idx_q = (cmp < 4) ? 0 : 1;
                } else if (fmt == C422) {
                    idx_q = (cmp < 2) ? 0 : 1;
                } else {
                    idx_q = (cmp < 1) ? 0 : 1;
                }

                //			idct25_t x0 = c0 * q[idx_q][y][0];//
                //					 x0 = (x0<<11) + 128;
                //			idct25_t x1 = str_rast8[4].read() * q[idx_q][y][4];
                //					 x1 = x1 <<11;
                idct25_t x0 = ((c0 * q[idx_q][y][0]) << 11); //+ 128;
                idct25_t x1 = (str_rast8[4].read() * q[idx_q][y][4]) << 11;
                idct25_t x2 = str_rast8[6].read() * q[idx_q][y][6];
                idct25_t x3 = str_rast8[2].read() * q[idx_q][y][2];
                idct25_t x4 = str_rast8[1].read() * q[idx_q][y][1];
                idct25_t x5 = str_rast8[7].read() * q[idx_q][y][7];
                idct25_t x6 = str_rast8[5].read() * q[idx_q][y][5];
                idct25_t x7 = str_rast8[3].read() * q[idx_q][y][3];

                // Prescale.

                // Stage 1.
                idct25_t x8 = w7 * (x4 + x5);
                x4 = x8 + w1mw7 * x4;
                x5 = x8 - w1pw7 * x5;
                x8 = w3 * (x6 + x7);
                x6 = x8 - w3mw5 * x6;
                x7 = x8 - w3pw5 * x7;

                // Stage 2.
                x8 = x0 + x1;
                x0 -= x1;
                x1 = w6 * (x3 + x2);
                x2 = x1 - w2pw6 * x2;
                x3 = x1 + w2mw6 * x3;
                x1 = x4 + x6;
                x4 -= x6;
                x6 = x5 + x7;
                x5 -= x7;

                // Stage 3.
                x7 = x8 + x3;
                x8 -= x3;
                x3 = x0 + x2;
                x0 -= x2;
                x2 = (r2 * (x4 + x5) + 128) >> 8;
                x4 = (r2 * (x4 - x5) + 128) >> 8;

                // Stage 4.

                strm_intermed[y][0].write((x7 + x1) >> 8);
                strm_intermed[y][1].write((x3 + x2) >> 8);
                strm_intermed[y][2].write((x0 + x4) >> 8);
                strm_intermed[y][3].write((x8 + x6) >> 8);
                strm_intermed[y][4].write((x8 - x6) >> 8);
                strm_intermed[y][5].write((x0 - x4) >> 8);
                strm_intermed[y][6].write((x3 - x2) >> 8);
                strm_intermed[y][7].write((x7 - x1) >> 8);
            }
        }
    }
}

// ------------------------------------------------------------
inline void hls_idct_v(uint32_t hls_mcuc,
                       ap_uint<3> mcu_cmp,
                       hls::stream<idctm_t> strm_intermed[8][8],
                       // output
                       hls::stream<idct_out_t> strm_out[8]) {
// Vertical 1-D IDCT.
V_LOOP:
    for (int mcu = 0; mcu < hls_mcuc; mcu++) {
        for (int cmp = 0; cmp < mcu_cmp; cmp++) {
            for (int32_t x = 0; x < 8; ++x) {
#pragma HLS PIPELINE
                // Similar to the horizontal 1-D IDCT case, if all the AC components are zero, then the IDCT is trivial.
                // However, after performing the horizontal 1-D IDCT, there are typically non-zero AC components, so
                // we do not bother to check for the all-zero case.

                // Prescale.
                idct25_t y0 = (strm_intermed[0][x].read() << 8) + 8192;
                idct25_t y1 = strm_intermed[4][x].read() << 8;
                idct25_t y2 = strm_intermed[6][x].read();
                idct25_t y3 = strm_intermed[2][x].read();
                idct25_t y4 = strm_intermed[1][x].read();
                idct25_t y5 = strm_intermed[7][x].read();
                idct25_t y6 = strm_intermed[5][x].read();
                idct25_t y7 = strm_intermed[3][x].read();

                // Stage 1.
                idct25_t y8 = w7 * (y4 + y5) + 4;
                y4 = (y8 + w1mw7 * y4) >> 3;
                y5 = (y8 - w1pw7 * y5) >> 3;
                y8 = w3 * (y6 + y7) + 4;
                y6 = (y8 - w3mw5 * y6) >> 3;
                y7 = (y8 - w3pw5 * y7) >> 3;

                // Stage 2.
                y8 = y0 + y1;
                y0 -= y1;
                y1 = w6 * (y3 + y2) + 4;
                y2 = (y1 - w2pw6 * y2) >> 3;
                y3 = (y1 + w2mw6 * y3) >> 3;
                y1 = y4 + y6;
                y4 -= y6;
                y6 = y5 + y7;
                y5 -= y7;

                // Stage 3.
                y7 = y8 + y3;
                y8 -= y3;
                y3 = y0 + y2;
                y0 -= y2;
                y2 = (r2 * (y4 + y5) + 128) >> 8;
                y4 = (r2 * (y4 - y5) + 128) >> 8;

                // Stage 4.
                // shift to save the one decimal place
                // hard code because of the algorithm
                const char shift_bits = 14;
                ap_int<27 - shift_bits> tmp[8];

                tmp[0] = (y7 + y1) >> shift_bits;
                tmp[1] = (y3 + y2) >> shift_bits;
                tmp[2] = (y0 + y4) >> shift_bits;
                tmp[3] = (y8 + y6) >> shift_bits;
                tmp[4] = (y8 - y6) >> shift_bits;
                tmp[5] = (y0 - y4) >> shift_bits;
                tmp[6] = (y3 - y2) >> shift_bits;
                tmp[7] = (y7 - y1) >> shift_bits;

                for (int j = 0; j < 8; j++) {
#pragma HLS UNROLL
                    idct_out_t cut = 0;
                    if ((tmp[j] + 128) > 255) {
                        cut = 255;
                    } else if ((tmp[j] + 128) < 0) {
                        cut = 0;
                    } else {
                        cut = (tmp[j] + 128);
                    }
                    strm_out[j].write(cut); //+128
                }
            }
        }
    }
}

// ------------------------------------------------------------
inline void cache_mcu(COLOR_FORMAT fmt,
                      ap_uint<3> mcu_cmp,
                      hls::stream<ap_uint<24> >& block_strm,
                      idct_in_t coef_mcu[6][64]) {
    // idx =(0~7)<<2+(0~3) for row is (0~7), and the ex. cmp is (YYYYUV=2,2,3,3,1,0) when 420

    int i_blk = 0;
    int cmp = 0;
    ap_uint<5> idx = 0;
    int32_t dpos[3] = {0, 0, 0}; // Y,U,V
#pragma HLS ARRAY_PARTITION variable = dpos complete

    for (int c = 0; c < mcu_cmp; c++) {
#pragma HLS PIPELINE II = 1
        for (int i = 0; i < 64; i++) {
            coef_mcu[c][i] = 0;
        }
    }

    // write one 8x8
    while (i_blk < mcu_cmp) {
#pragma HLS PIPELINE II = 1
        ap_uint<24> block_coeff = block_strm.read();
        bool is_endblock = block_coeff[22];
        uint8_t bpos = block_coeff(21, 16);
        int16_t tmp = block_coeff(15, 0);      // attention: the sign
        ap_int<11> block = block_coeff(10, 0); // attention: the sign

        if (fmt == C420 && cmp == 0) {
            //////////////////////////
            bool isY2 = (dpos[0] & 2);

            coef_mcu[i_blk][bpos] = block;
            if (isY2)
                idx = 3;
            else
                idx = 2;
        } else {
            coef_mcu[i_blk][bpos] = block;
            idx = 2 - cmp;
        }

        // loop YUV
        if (is_endblock) {
            if (fmt == C444) {
                dpos[cmp]++;
                cmp = (cmp == 2) ? 0 : cmp + 1;
            } else if (fmt == C422) {
                if (cmp == 0) {
                    if ((dpos[0] & 1) == 1) {
                        cmp = 1;
                    }
                    dpos[0]++;
                } else if (cmp == 1) {
                    cmp = 2;
                    dpos[1]++;
                } else { // cmp==2
                    cmp = 0;
                    dpos[2]++;
                }
            } else if (fmt == C420) {
                if (cmp == 0) {
                    if ((dpos[0] & 3) == 3) {
                        cmp = 1;
                    }
                    dpos[0]++;
                } else if (cmp == 1) {
                    cmp = 2;
                    dpos[1]++;
                } else { // cmp==2
                    cmp = 0;
                    dpos[2]++;
                }
            } else { // C400
                cmp = 0;
                dpos[0]++; // II=2
            }

            i_blk++;
        } // end one block
    }     // end while
}
// ------------------------------------------------------------

// ------------------------------------------------------------
// zigzag to raster
// cmp=6, 4 ,3
// cnt is the count of block in one mcu
//
inline void array_to_raster( // uint32_t hls_mcuc,
    ap_uint<3> mcu_cmp,
    idct_in_t coef_mcu[6][64],
    hls::stream<idct_in_t> strm_out[8]) {
    int cnt = 0;
    int mbs = 0;
    ap_uint<5> idx;
    // loop in the mcu, number of block will be 3, 4, or 6
    while (mbs < mcu_cmp) {
#pragma HLS DEPENDENCE array inter false
#pragma HLS PIPELINE II = 1

        if (cnt == 0) {
            // idx = idx_cmp.read();
            strm_out[0].write(coef_mcu[mbs][0]);
            strm_out[1].write(coef_mcu[mbs][1]);
            strm_out[2].write(coef_mcu[mbs][5]);
            strm_out[3].write(coef_mcu[mbs][6]);
            strm_out[4].write(coef_mcu[mbs][14]);
            strm_out[5].write(coef_mcu[mbs][15]);
            strm_out[6].write(coef_mcu[mbs][27]);
            strm_out[7].write(coef_mcu[mbs][28]);
        } else if (cnt == 1) {
            strm_out[0].write(coef_mcu[mbs][2]);
            strm_out[1].write(coef_mcu[mbs][4]);
            strm_out[2].write(coef_mcu[mbs][7]);
            strm_out[3].write(coef_mcu[mbs][13]);
            strm_out[4].write(coef_mcu[mbs][16]);
            strm_out[5].write(coef_mcu[mbs][26]);
            strm_out[6].write(coef_mcu[mbs][29]);
            strm_out[7].write(coef_mcu[mbs][42]);
        } else if (cnt == 2) {
            strm_out[0].write(coef_mcu[mbs][3]);
            strm_out[1].write(coef_mcu[mbs][8]);
            strm_out[2].write(coef_mcu[mbs][12]);
            strm_out[3].write(coef_mcu[mbs][17]);
            strm_out[4].write(coef_mcu[mbs][25]);
            strm_out[5].write(coef_mcu[mbs][30]);
            strm_out[6].write(coef_mcu[mbs][41]);
            strm_out[7].write(coef_mcu[mbs][43]);
        } else if (cnt == 3) {
            strm_out[0].write(coef_mcu[mbs][9]);
            strm_out[1].write(coef_mcu[mbs][11]);
            strm_out[2].write(coef_mcu[mbs][18]);
            strm_out[3].write(coef_mcu[mbs][24]);
            strm_out[4].write(coef_mcu[mbs][31]);
            strm_out[5].write(coef_mcu[mbs][40]);
            strm_out[6].write(coef_mcu[mbs][44]);
            strm_out[7].write(coef_mcu[mbs][53]);
        } else if (cnt == 4) {
            strm_out[0].write(coef_mcu[mbs][10]);
            strm_out[1].write(coef_mcu[mbs][19]);
            strm_out[2].write(coef_mcu[mbs][23]);
            strm_out[3].write(coef_mcu[mbs][32]);
            strm_out[4].write(coef_mcu[mbs][39]);
            strm_out[5].write(coef_mcu[mbs][45]);
            strm_out[6].write(coef_mcu[mbs][52]);
            strm_out[7].write(coef_mcu[mbs][54]);
        } else if (cnt == 5) {
            strm_out[0].write(coef_mcu[mbs][20]);
            strm_out[1].write(coef_mcu[mbs][22]);
            strm_out[2].write(coef_mcu[mbs][33]);
            strm_out[3].write(coef_mcu[mbs][38]);
            strm_out[4].write(coef_mcu[mbs][46]);
            strm_out[5].write(coef_mcu[mbs][51]);
            strm_out[6].write(coef_mcu[mbs][55]);
            strm_out[7].write(coef_mcu[mbs][60]);
        } else if (cnt == 6) {
            strm_out[0].write(coef_mcu[mbs][21]);
            strm_out[1].write(coef_mcu[mbs][34]);
            strm_out[2].write(coef_mcu[mbs][37]);
            strm_out[3].write(coef_mcu[mbs][47]);
            strm_out[4].write(coef_mcu[mbs][50]);
            strm_out[5].write(coef_mcu[mbs][56]);
            strm_out[6].write(coef_mcu[mbs][59]);
            strm_out[7].write(coef_mcu[mbs][61]);
        } else if (cnt == 7) {
            strm_out[0].write(coef_mcu[mbs][35]);
            strm_out[1].write(coef_mcu[mbs][36]);
            strm_out[2].write(coef_mcu[mbs][48]);
            strm_out[3].write(coef_mcu[mbs][49]);
            strm_out[4].write(coef_mcu[mbs][57]);
            strm_out[5].write(coef_mcu[mbs][58]);
            strm_out[6].write(coef_mcu[mbs][62]);
            strm_out[7].write(coef_mcu[mbs][63]);
        }

        if (cnt == 7) {
            mbs++;
            cnt = 0;
        } else {
            cnt++;
        }
    }
}

// ------------------------------------------------------------
inline void jpeg_zigzag_to_raster(COLOR_FORMAT fmt,
                                  ap_uint<3> mcu_cmp,
                                  hls::stream<ap_uint<24> >& block_strm,
                                  hls::stream<idct_in_t> str_rast_x8[8]) {
#pragma HLS inline
#pragma HLS DATAFLOW

    idct_in_t coef_mcu[6][64];
#pragma HLS ARRAY_PARTITION variable = coef_mcu dim = 2

    cache_mcu(fmt, mcu_cmp, block_strm, coef_mcu);
    array_to_raster(mcu_cmp, coef_mcu, str_rast_x8);
}

// ------------------------------------------------------------
template <typename IDCT_OUT_T>
void mcu_reorder(hls::stream<ap_uint<24> >& block_strm,
                 const uint8_t q_tables[2][8][8],
                 const bas_info bas_info,
                 // ouput
                 hls::stream<IDCT_OUT_T> strm_iDCT_x8[8]) {
    // for select the quant table
    // for 444 YUV_idx=2,1,0,2,1,0
    // for 422 YUV_idx=2,2,1,0,2,2,1,0
    // for 420 YUV_idx=2,2,3,3,1,0,2,2,3,3,1,0
    ap_uint<5> idx;
    int test = 0;

#ifndef __SYNTHESIS__
    fprintf(stderr, "mcu_reorder start\n");
#endif

    COLOR_FORMAT fmt = bas_info.format;
    uint32_t hls_mcuc = bas_info.hls_mcuc;
    ap_uint<3> mcu_cmp = bas_info.mcu_cmp;

#pragma HLS DATAFLOW

    // clang-format off
	    hls::stream< idct_in_t >  str_rast_x8[8];
#pragma HLS STREAM 			      	variable=str_rast_x8 depth=128
#pragma HLS ARRAY_PARTITION 		variable=str_rast_x8 complete dim=0
		hls::stream<idctm_t> strm_intermed[8][8];
#pragma HLS STREAM 				  variable=strm_intermed depth=16
#pragma HLS ARRAY_PARTITION 	  variable=strm_intermed complete dim=0
        idct_in_t coef_mcu[6][64];
#pragma HLS ARRAY_PARTITION variable = coef_mcu dim = 2
    // clang-format on

    for (int mcu = 0; mcu < hls_mcuc; mcu++) {
#pragma HLS DATAFLOW
        // jpeg_zigzag_to_raster(fmt, mcu_cmp, block_strm, str_rast_x8);
        cache_mcu(fmt, mcu_cmp, block_strm, coef_mcu);
        array_to_raster(mcu_cmp, coef_mcu, str_rast_x8);
    }

    hls_idct_h(hls_mcuc, mcu_cmp, fmt, str_rast_x8, q_tables, strm_intermed);

    hls_idct_v(hls_mcuc, mcu_cmp, strm_intermed, strm_iDCT_x8);
}

// ------------------------------------------------------------
/**
 * @brief the template of stream width of _WAxi burst out.
 *
 * @tparam _WAxi   width of axi port.
 *
 * @param wbuf AXI master port to write to, ex. 64 bits.
 * @param strm_iDCT_x8 stream width is 8 bits
 */
template <int _WAxi>
void burstWrite(ap_uint<_WAxi>* yuv_mcu_pointer, hls::stream<idct_out_t> strm_iDCT_x8[8], const uint32_t all_blocks) {
    // write each burst to axi
    int cnt = 0;
    ap_uint<8 * 8 * sizeof(idct_out_t)> tmp;
// assert(_WAxi == 8*8*sizeof(idct_out_t));

DOING_BURST:
    for (int i = 0; i < 8 * all_blocks; i++) { // write one block of Y or U or V
#pragma HLS PIPELINE II = 1
        for (int j = 0; j < 8; j++) {
#pragma HLS UNROLL
            tmp(8 * (j + 1) - 1, 8 * j) = strm_iDCT_x8[j].read();
        }
        yuv_mcu_pointer[cnt] = tmp;
        cnt++;
    }
}

// ------------------------------------------------------------
// copy image infos from struct to pointer for axi master
template <int _WAxi>
void copyInfo(const img_info img_info,
              const cmp_info cmp_info[MAX_NUM_COLOR],
              const bas_info bas_info,
              const int rtn,
              const bool rtn2,
              ap_uint<_WAxi> infos[1024]) {
    *(infos + 0) = img_info.hls_cs_cmpc;
    *(infos + 1) = img_info.hls_mcuc;
    *(infos + 2) = img_info.hls_mcuh;
    *(infos + 3) = img_info.hls_mcuv;
    *(infos + 4) = rtn;
    *(infos + 5) = (ap_uint<_WAxi>)rtn2;

    *(infos + 10) = bas_info.all_blocks;
    for (int i = 0; i < MAX_NUM_COLOR; i++) {
#pragma HLS PIPELINE II = 1
        *(infos + 11 + i) = bas_info.axi_height[i];
    }
    for (int i = 0; i < 4; i++) {
#pragma HLS PIPELINE II = 1
        *(infos + 14 + i) = bas_info.axi_map_row2cmp[i];
    }
    *(infos + 18) = bas_info.axi_mcuv;
    *(infos + 19) = bas_info.axi_num_cmp;
    *(infos + 20) = bas_info.axi_num_cmp_mcu;
    for (int i = 0; i < MAX_NUM_COLOR; i++) {
#pragma HLS PIPELINE II = 1
        *(infos + 21 + i) = bas_info.axi_width[i];
    }
    *(infos + 24) = bas_info.format;
    for (int i = 0; i < MAX_NUM_COLOR; i++) {
#pragma HLS PIPELINE II = 1
        *(infos + 25 + i) = bas_info.hls_mbs[i];
    }
    *(infos + 28) = bas_info.hls_mcuc;
    for (int c = 0; c < MAX_NUM_COLOR; c++) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
#pragma HLS PIPELINE II = 1
                *(infos + 29 + c * 64 + i * 8 + j) = bas_info.idct_q_table_x[c][i][j];
            }
        }
    }
    for (int c = 0; c < MAX_NUM_COLOR; c++) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
#pragma HLS PIPELINE II = 1
                *(infos + 221 + c * 64 + i * 8 + j) = bas_info.idct_q_table_y[c][i][j];
            }
        }
    }
    *(infos + 413) = bas_info.mcu_cmp;
    for (int c = 0; c < MAX_NUM_COLOR; c++) {
        for (int i = 0; i < 64; i++) {
#pragma HLS PIPELINE II = 1
            *(infos + 414 + c * 64 + i) = bas_info.min_nois_thld_x[c][i];
        }
    }
    for (int c = 0; c < MAX_NUM_COLOR; c++) {
        for (int i = 0; i < 64; i++) {
#pragma HLS PIPELINE II = 1
            *(infos + 606 + c * 64 + i) = bas_info.min_nois_thld_y[c][i];
        }
    }
    for (int c = 0; c < MAX_NUM_COLOR; c++) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
#pragma HLS PIPELINE II = 1
                *(infos + 798 + c * 64 + i * 8 + j) = bas_info.q_tables[c][i][j];
            }
        }
    }
    for (int c = 0; c < MAX_NUM_COLOR; c++) {
#pragma HLS PIPELINE II = 1
        *(infos + 990 + c * 6) = cmp_info[c].bc;
        *(infos + 991 + c * 6) = cmp_info[c].bch;
        *(infos + 992 + c * 6) = cmp_info[c].bcv;
        *(infos + 993 + c * 6) = cmp_info[c].mbs;
        *(infos + 994 + c * 6) = cmp_info[c].sfh;
        *(infos + 995 + c * 6) = cmp_info[c].sfv;
    }
}

// Alloc buf
// ------------------------------------------------------------
template <int _WAxi>
void decoder_jpg_full_top(ap_uint<_WAxi>* ptr,
                          const int sz,
                          const int c,
                          const uint16_t dht_tbl1[2][2][1 << DHT1],
                          //
                          uint8_t ac_value_buckets[2][165],
                          HCODE_T ac_huff_start_code[2][AC_N],
                          int16_t ac_huff_start_addr[2][16],
                          //
                          uint8_t dc_value_buckets[2][12],
                          HCODE_T dc_huff_start_code[2][DC_N],
                          int16_t dc_huff_start_addr[2][16],
                          // image info
                          ap_uint<12> hls_cmp,
                          const uint8_t hls_mbs[MAX_NUM_COLOR],
                          const uint8_t q_tables[2][8][8],
                          const img_info img_info,
                          const bas_info bas_info,
                          // ouput
                          int& rtn2,
                          uint32_t& rst_cnt,
                          ap_uint<64>* yuv_mcu_pointer) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    // clang-format off
    _XF_IMAGE_PRINT(" ************* start decode %d mcus in FPGA  *************\n", (int)img_info.hls_mcuc);
    _XF_IMAGE_PRINT(
    				"  hls_cs_cmpc=%d, hls_mbs[0]=%d, hls_mbs[1]=%d, hls_mbs[2]=%d, \n",
						img_info.hls_cs_cmpc, hls_mbs[0], hls_mbs[1], hls_mbs[2]);


     hls::stream<CHType> image_strm;
#pragma HLS RESOURCE variable = image_strm core = FIFO_LUTRAM
#pragma HLS STREAM   variable = image_strm depth = 32
     hls::stream<bool>   eof_strm;
#pragma HLS RESOURCE variable = eof_strm core = FIFO_LUTRAM
#pragma HLS STREAM   variable = eof_strm depth = 32

    // clang-format on

    // in case AXI_WIDTH=16 cut down resource
    ap_uint<1> column = c;
    xf::common::utils_hw::axi_to_char_stream<BURST_LENTH, _WAxi, CHType>(ptr, image_strm, eof_strm, sz, (int)column);

    // clang-format off
    hls::stream<ap_uint<24> > block_strm;
#pragma HLS STREAM   variable = block_strm depth = 1024
#pragma HLS BIND_STORAGE variable = block_strm type = ram_2p impl = bram
    // clang-format on

    xf::codec::details::mcu_decoder(image_strm, eof_strm, dht_tbl1, // dht_tbl3, dht_tbl4,
                                    ac_value_buckets, ac_huff_start_code, ac_huff_start_addr, dc_value_buckets,
                                    dc_huff_start_code, dc_huff_start_addr, hls_cmp, img_info.hls_cs_cmpc, hls_mbs,
                                    img_info.hls_mcuh, img_info.hls_mcuc, rtn2, rst_cnt, block_strm);

    hls::stream<idct_out_t> strm_iDCT_x8[8];
#pragma HLS stream variable = strm_iDCT_x8 depth = 256
#pragma HLS ARRAY_PARTITION variable = strm_iDCT_x8 complete dim = 0
#pragma HLS BIND_STORAGE variable = strm_iDCT_x8 type = ram_2p impl = bram

    // Functions to reorder the mcu block from only effective coefficient to real coefficient
    // Then zigzag to raster scan, iQ,iDCT and shift form (-128~127) to (0~255)
    //----------------------------------------------------------
    xf::codec::details::mcu_reorder(block_strm, q_tables, bas_info, strm_iDCT_x8); // idx_coef,

    burstWrite<64>(yuv_mcu_pointer, strm_iDCT_x8, bas_info.all_blocks);
}

} // namespace details

// ------------------------------------------------------------
/**
 * @brief Level 2 : kernel implement for jfif parser + huffman decoder + iQ_iDCT
 *
 * @tparam CH_W size of data path in dataflow region, in bit.
 *         when CH_W is 16, the decoder could decode one symbol per cycle in about 99% cases.
 *         when CH_W is 8 , the decoder could decode one symbol per cycle in about 80% cases, but use less resource.
 *
 * @param jpeg_pointer the input jpeg to be read from DDR.
 * @param size the total bytes to be read from DDR.
 * @param yuv_mcu_pointer the output yuv to DDR in mcu order.
 * @param info information of the image, maybe use in the recovery image.
 */
// a.input the jpg 420/422/444 baseline file
// b.output the as the 8x8 's Column scan order YUV (0~255), like [Y*allpixels,U*0.5*allpixels, V*0.5*allpixels], and
// image infos
// c.Fault tolerance: If the picture's format is incorrect, error codes will directly end the kernel
// and wait for the input of the next image. Error codes cloud help to position at which decoding stage does the error
// occur
// d.performance: input throughput: 150MB/s~300MB/s(1symbol/clk), output 1~1.6GB/s (max 8B/clk),
// frequency 250MHz for kernel, for only huffman core 286MHz by vivado 2018.3

inline void kernelJpegDecoderTop(ap_uint<AXI_WIDTH>* jpeg_pointer,
                                 const int size,
                                 ap_uint<64>* yuv_mcu_pointer,
                                 ap_uint<32>* infos) {
    // clang-format off

	//for jfif parser
    int r = 0, c = 0;
    int left = 0;
    ap_uint<12> hls_cmp;
    uint8_t hls_mbs[MAX_NUM_COLOR] = {0};
#pragma HLS ARRAY_PARTITION variable = hls_mbs  complete

    //for reset of the decoder
    uint32_t rst_cnt;
    int rtn;
    int rtn2;

    //tables
	uint8_t 						   q_tables[2][8][8];
#pragma HLS ARRAY_PARTITION variable = q_tables  dim = 3
    uint16_t 					       dht_tbl1[2][2][1 << DHT1];
#pragma HLS ARRAY_PARTITION variable = dht_tbl1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = dht_tbl1 complete dim = 2

	uint8_t ac_value_buckets  [2][ 165 ];//every val relative with the huffman codes
	HCODE_T ac_huff_start_code[2][AC_N]; // the huff_start_code<65535
	int16_t ac_huff_start_addr[2][16];   // the addr of the val huff_start_addr<256
	uint8_t dc_value_buckets  [2][ 12 ]; //every val relative with the huffman codes
	HCODE_T dc_huff_start_code[2][DC_N]; // the huff_start_code<65535
	int16_t dc_huff_start_addr[2][16];   // the addr of the val huff_start_addr<256
//#pragma HLS RESOURCE 		variable = ac_huff_start_code core = RAM_2P_LUTRAM
#pragma HLS ARRAY_PARTITION variable = ac_huff_start_code complete dim = 2
//#pragma HLS RESOURCE 		variable = dc_huff_start_code core = RAM_2P_LUTRAM
#pragma HLS ARRAY_PARTITION variable = dc_huff_start_code complete dim = 2

#pragma HLS RESOURCE variable = ac_huff_start_addr  core = RAM_2P_LUTRAM
#pragma HLS RESOURCE variable = ac_value_buckets 	core = RAM_2P_LUTRAM
#pragma HLS RESOURCE variable = dc_huff_start_addr  core = RAM_2P_LUTRAM
#pragma HLS RESOURCE variable = dc_value_buckets 	core = RAM_2P_LUTRAM

	// image infos for multi-applications
    xf::codec::img_info img_info;
    xf::codec::cmp_info cmp_info[MAX_NUM_COLOR];
    xf::codec::bas_info bas_info;
    img_info.hls_cs_cmpc = 0;//init

    // Functions to parser the header before the data burst load from DDR
    //----------------------------------------------------------
    xf::codec::details::parser_jpg_top(jpeg_pointer, size, r, c, dht_tbl1,
    								   ac_value_buckets, ac_huff_start_code, ac_huff_start_addr,
                                       dc_value_buckets, dc_huff_start_code, dc_huff_start_addr,
									   hls_cmp, left,
									   hls_mbs, q_tables, rtn,
									   img_info, cmp_info, bas_info);

    ap_uint<AXI_WIDTH>* ptr = jpeg_pointer + r;

    // Functions to decode the huffman code to non(Inverse quantization+IDCT) block coefficient
    //----------------------------------------------------------
    xf::codec::details::decoder_jpg_full_top(ptr, left, c, dht_tbl1,
											 ac_value_buckets, ac_huff_start_code, ac_huff_start_addr,
											 dc_value_buckets, dc_huff_start_code, dc_huff_start_addr,
											 hls_cmp, hls_mbs, q_tables, img_info, bas_info,
											 rtn2, rst_cnt, yuv_mcu_pointer);
											 //strm_iDCT_x8);//idx_coef,

    // Functions to copy image infos from struct to pointer for axi master
    //----------------------------------------------------------
    xf::codec::details::copyInfo(img_info, cmp_info, bas_info, rtn, rtn2, infos);

// clang-format on
#ifndef __SYNTHESIS__
    if (rtn || (rtn2)) {
        fprintf(stderr, "Warning: parser the bad case input file! \n");
    }
#endif
}

} // namespace codec
} // namespace xf
#endif
