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
 * @file XAcc_jpegdecoder.hpp
 * @brief mcu_decoder template function API.
 *
 * This file is part of HLS algorithm library.
 */

#ifndef __cplusplus
#error " XAcc_jpegdecoder hls::stream<> interface, and thus requires C++"
#endif

#ifndef _XACC_JPEGDECODER_HPP_
#define _XACC_JPEGDECODER_HPP_

#include <ap_int.h>
#include <stdint.h>
#include <hls_stream.h>
#include "axi_to_stream.hpp"

#ifndef __SYNTHESIS__
// For debug
#include <bitset>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cstdio>
#endif

#define _XF_IMAGE_VOID_CAST static_cast<void>
// XXX toggle here to debug this file
//#ifndef __SYNTHESIS__
#if 0
#define _XF_IMAGE_PRINT(msg...) \
    do {                        \
        printf(msg);            \
    } while (0)
#else
#define _XF_IMAGE_PRINT(msg...) (_XF_IMAGE_VOID_CAST(0))

#endif

// ------------------------------------------------------------
#define DHT1 (9)        // the number of leading bits of huffman codes
#define DHT2 (10)       // the number of tail bits of huffman codes
#define DHT_S 16 - DHT1 // the exponent of the address weight, weight = 2^DHT_S
#define SCALE1 (1 << DHT1)
#define SCALE2 (1 << DHT2)
#define DHT_M (1 << DHT2)
#define MAX_NUM_COLOR (3)     // the max number of cmp for this decoder, current is 3
#define MAX_DEC_PIX (5000000) // the max bytes of input jpg, 1M is enough for 800*800 co-sim
#define CMPhuff (2)           // the max number of huffman tables for all cmp, current is 2
#define DC_N 3                // >12(bit) - DHT1 -1
#define AC_N 6                // >16(bit) - DHT1 -1
// ------------------------------------------------------------
#define BURST_LENTH (128)
#define CH_W (16)
#if (CH_W == 32)
typedef uint32_t CHType; // channel data type
#else
typedef uint16_t CHType; // channel data type
#endif
typedef ap_uint<14> HCODE_T;
// ------------------------------------------------------------
// tmp vecter for the max of image'block of all cmps, to decode the hq.jpg need 1036800
// to decode 420 800*800 need 50*50*4 =10000
// to decode 444 800*800 need 100*100*4 =40000
#define MAXCMP_BC (1036800)

namespace xf {
namespace codec {
// ------------------------------------------------------------
#if 1
#define AXI_WIDTH (16)
//
enum COLOR_FORMAT { C400 = 0, C420, C422, C444 };
//
const static uint8_t hls_jpeg_zigzag_to_raster[64] = {0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
                                                      12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
                                                      35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
                                                      58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};
//
const short hls_icos_base_8192_scaled[64] = {
    8192,   8192,  8192,   8192,  8192, 8192,  8192,   8192,   11363, 9633,  6436,   2260,  -2260,
    -6436,  -9633, -11363, 10703, 4433, -4433, -10703, -10703, -4433, 4433,  10703,  9633,  -2260,
    -11363, -6436, 6436,   11363, 2260, -9633, 8192,   -8192,  -8192, 8192,  8192,   -8192, -8192,
    8192,   6436,  -11363, 2260,  9633, -9633, -2260,  11363,  -6436, 4433,  -10703, 10703, -4433,
    -4433,  10703, -10703, 4433,  2260, -6436, 9633,   -11363, 11363, -9633, 6436,   -2260,
};
//
struct decOutput {
    COLOR_FORMAT format;
    uint16_t axi_width[MAX_NUM_COLOR];
    uint16_t axi_height[MAX_NUM_COLOR];
    uint8_t axi_map_row2cmp[4];
    uint8_t min_nois_thld_x[MAX_NUM_COLOR][64];
    uint8_t min_nois_thld_y[MAX_NUM_COLOR][64];
    uint8_t q_tables[MAX_NUM_COLOR][8][8];
    int32_t idct_q_table_x[MAX_NUM_COLOR][8][8];
    int32_t idct_q_table_y[MAX_NUM_COLOR][8][8];
    int32_t idct_q_table_l[MAX_NUM_COLOR][8][8]; // todo

    uint16_t axi_mcuv;
    uint8_t axi_num_cmp_mcu;
    uint8_t axi_num_cmp;
};
#endif
// ------------------------------------------------------------

} // namespace codec
} // namespace xf

namespace xf {
namespace codec {

// ------------------------------------------------------------
struct hls_huff_DHT {
    unsigned short tbl1[2][CMPhuff][1 << DHT1];
    unsigned short tbl2[2][CMPhuff][1 << DHT2];
};
// ------------------------------------------------------------
struct hls_huff_segment {
    unsigned char size[16]; // the number of the i+1 bits huffman codes
    unsigned char val[256];
};
// ------------------------------------------------------------
struct sos_data {
    uint8_t bits;
    uint8_t garbage_bits;
    CHType data;
    bool rst; // todo
    bool end_sos;
};
// ------------------------------------------------------------
struct img_info {
    uint8_t hls_cs_cmpc;
    uint32_t hls_mcuc; // the total mcu
    uint16_t hls_mcuh; // the horizontal mcu
    uint16_t hls_mcuv;
};
// ------------------------------------------------------------
struct hls_compInfo {
    int sfv; // sample factor vertical
    int sfh; // sample factor horizontal
    int mbs; // blocks in mcu
    int bcv; // block count vertical (interleaved)
    int bch; // block count horizontal (interleaved)
    int bc;  // block count (all) (interleaved)
};
} // namespace codec
} // namespace xf

// ------------------------------------------------------------
// top
void top_mcu_decoder(
    // input
    hls::stream<CHType>& image_strm,
    hls::stream<bool>& eof_strm,
    const uint16_t dht_tbl1[2][2][1 << DHT1],
    // const uint16_t dht_tbl2[2][2][1 << DHT2],
    //
    const uint8_t ac_val[2][165],
    const HCODE_T ac_huff_start_code[2][AC_N],
    const int16_t ac_huff_start_addr[2][16],
    //
    const uint8_t dc_val[2][12],
    const HCODE_T dc_huff_start_code[2][DC_N],
    const int16_t dc_huff_start_addr[2][16],
    //
    ap_uint<12> hls_cmp,

    // image info
    const uint8_t hls_cs_cmpc, // component count in current scan
    const uint8_t hls_mbs[MAX_NUM_COLOR],
    const uint16_t hls_mcuh, // the horizontal mcu
    const uint32_t hls_mcuc, // the total mcu

    // output
    bool& rtn2,
    uint32_t& rst_cnt,
    hls::stream<ap_uint<24> >& block_strm);

// ------------------------------------------------------------

namespace xf {
namespace codec {
namespace details {
// ------------------------------------------------------------
/**
 * @brief Level 1 : decode all mcu
 *
 * @tparam CH_W size of data path in dataflow region, in bit.
 *         when CH_W is 16, the decoder could decode one symbol per cycle in about 99% cases.
 *         when CH_W is 8 , the decoder could decode one symbol per cycle in about 80% cases, but use less resource.
 *
 * @param image_strm the stream of compressed data after SOS marker.
 * @param eof_strm the stream of end flag for image_strm, synchronous signal using false and an addtional true in the
 * end.
 * @param dht_tbl1/dht_tbl2 the segment data of Define huffman table marker.
 * @param hls_cmp the shift register organized by the index of each color component.
 * @param image info include hls_cs_cmpc/hls_mbs/hls_mcuh/hls_mcuc is just for csim tests.
 * @param block_strm the stream of coefficients in block,23:is_rst, 22:is_endblock,21~16:bpos,15~0:block val
 */
void mcu_decoder(
    // input
    hls::stream<CHType>& image_strm,
    hls::stream<bool>& eof_strm,
    const uint16_t dht_tbl1[2][2][1 << DHT1],
    const uint16_t dht_tbl2[2][2][1 << DHT2],
    ap_uint<12> hls_cmp,

    // image info
    const uint8_t hls_cs_cmpc, // component count in current scan
    const uint8_t hls_mbs[MAX_NUM_COLOR],
    const uint16_t hls_mcuh, // the horizontal mcu
    const uint32_t hls_mcuc, // the total mcu

    // output
    bool& rtn2,
    uint32_t& rst_cnt,
    hls::stream<ap_uint<24> >& block_strm);
// ------------------------------------------------------------
/**
 * @brief convert strm to Aligned_block
 *
 * @param block_strm the stream of coefficients in block,23:is_rst, 22:is_endblock,21~16:bpos,15~0:block val
 * @param hls_block the maxsize block, will be copy to the aligned_block
 * @param hls_sfv the sample factor vertical for each component
 * @param hls_sfh the sample factor horizontal for each component
 * @param hls_mbs the blocks in mcu for each component.
 * @param hls_bch the max block count horizontal (interleaved)
 * @param hls_bc  the max block count (all) (interleaved)
 * @param sta     the status of the process, 0: keep doing, 1: reset decoder(todo), 2: decode done
 */
void hls_next_mcupos2(hls::stream<ap_uint<24> >& block_strm,
                      int16_t hls_block[MAX_NUM_COLOR * MAXCMP_BC * 64],
                      int hls_sfv[4],
                      int hls_sfh[4],
                      const uint8_t hls_mbs[4],
                      int hls_bch,
                      int hls_bc,
                      int32_t hls_mcuc,
                      uint8_t hls_cs_cmpc,
                      bool rtn2,
                      int& sta);

// ------------------------------------------------------------

} // namespace details
} // namespace codec
} // namespace xf
#endif
