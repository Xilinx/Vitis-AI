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
 * @file XAcc_jfifparser.hpp
 * @brief parser_jpg_top template function API.
 *
 * This file is part of HLS algorithm library.
 */

#ifndef _XACC_JFIFPARSER_HPP_
#define _XACC_JFIFPARSER_HPP_

#ifndef __cplusplus
#error "XF Image Library only works with C++."
#endif

#include "XAcc_jpegdecoder.hpp"

namespace xf {
namespace codec {
namespace details {
// ------------------------------------------------------------
/*
 * @brief Level 1 : parser the jfif register for the jepg decoder
 *
 * @tparam CH_W size of data path in dataflow region, in bit.
 *         when CH_W is 16, the decoder could decode one symbol per cycle in about 99% cases.
 *         when CH_W is 8 , the decoder could decode one symbol per cycle in about 80% cases, but use less resource.
 *
 * @param datatoDDR the pointer to DDR.
 * @param size the total bytes to be read from DDR.
 * @param r the index of vector to be read from AXI in all cases
 * @param c the column to be read from AXI in the case when AXI_WIDTH > 8*sizeof(char)
 * @param dht_tbl1/dht_tbl2 the segment data of Define huffman table marker.
 * @param hls_cmp the shift register organized by the index of each color component.
 * @param hls_mbs the number of blocks in mcu for each component.
 * @param left the number of bytes to be read from DDR after parser.
 * @param image info include hls_cs_cmpc/hls_mbs/hls_mcuh/hls_mcuc is just for csim tests.
 * @param hls_compinfo image information may be used to generate the decOutput.
 * @param rtn return flag.
 * @param pout information used by next module.
 */
void parser_jpg_top(ap_uint<AXI_WIDTH>* datatoDDR,
                    const int size,
                    int& r,
                    int& c,
                    uint16_t dht_tbl1[2][2][1 << DHT1],
                    // uint16_t dht_tbl2[2][2][1 << DHT2],
                    //
                    uint8_t ac_val[2][165],
                    HCODE_T ac_huff_start_code[2][AC_N],
                    int16_t ac_huff_start_addr[2][16],
                    //
                    uint8_t dc_val[2][12],
                    HCODE_T dc_huff_start_code[2][DC_N],
                    int16_t dc_huff_start_addr[2][16],
                    //
                    ap_uint<12>& hls_cmp,
                    int& left,
                    // image info
                    img_info& img_info,
                    uint8_t hls_mbs[MAX_NUM_COLOR],
                    hls_compInfo hls_compinfo[MAX_NUM_COLOR],
                    int& rtn,
                    decOutput* pout);

// ------------------------------------------------------------
/*
 * @brief Level 1 : decode all mcu with burst read data from DDR
 *
 * @tparam CH_W size of data path in dataflow region, in bit.
 *         when CH_W is 16, the decoder could decode one symbol per cycle in about 99% cases.
 *         when CH_W is 8 , the decoder could decode one symbol per cycle in about 80% cases, but use less resource.
 *
 * @param ptr the pointer to DDR.
 * @param sz the total bytes to be read from DDR.
 * @param c the column to be read from AXI in the case when AXI_WIDTH > 8*sizeof(char)
 * @param dht_tbl1/dht_tbl2 the segment data of Define huffman table marker.
 * @param hls_cmp the shift register organized by the index of each color component.
 * @param hls_mbs the number of blocks in mcu for each component.
 * @param image info include hls_cs_cmpc/hls_mbs/hls_mcuh/hls_mcuc is just for csim tests.
 * @param block_strm the stream of coefficients in block,23:is_rst, 22:is_endblock,21~16:bpos,15~0:block val
 */
void decoder_jpg_top(ap_uint<AXI_WIDTH>* ptr,
                     const int sz,
                     const int c,
                     const uint16_t dht_tbl1[2][2][1 << DHT1],
                     // const uint16_t dht_tbl2[2][2][1 << DHT2],
                     //
                     uint8_t ac_val[2][165],
                     HCODE_T ac_huff_start_code[2][AC_N],
                     int16_t ac_huff_start_addr[2][16],
                     //
                     uint8_t dc_val[2][12],
                     HCODE_T dc_huff_start_code[2][DC_N],
                     int16_t dc_huff_start_addr[2][16],
                     //
                     ap_uint<12> hls_cmp,
                     const uint8_t hls_mbs[MAX_NUM_COLOR],
                     const img_info img_info,

                     bool& rtn2,
                     uint32_t& rst_cnt,
                     hls::stream<ap_uint<24> >& block_strm);

} // namespace details

// ------------------------------------------------------------
/**
 * @brief Level 2 : kernel for jfif parser + huffman decoder
 *
 * @tparam CH_W size of data path in dataflow region, in bit.
 *         when CH_W is 16, the decoder could decode one symbol per cycle in about 99% cases.
 *         when CH_W is 8 , the decoder could decode one symbol per cycle in about 80% cases, but use less resource.
 *
 * @param datatoDDR the pointer to DDR.
 * @param size the total bytes to be read from DDR.
 * @param img_info information to recovery the image.
 * @param hls_compInfo the component info used to generate the decOutput.
 * @param block_strm the stream of coefficients in block,23:is_rst, 22:is_endblock,21~16:bpos,15~0:block val
 * @param rtn the flag of the jfif parser succeed
 * @param rtn2 the flag of the decode succeed
 */
void kernelParserDecoderTop(ap_uint<AXI_WIDTH>* datatoDDR,
                            const int size,
                            xf::codec::img_info& img_info,
                            xf::codec::hls_compInfo hls_cmpnfo[MAX_NUM_COLOR],
                            hls::stream<ap_uint<24> >& block_strm,
                            int& rtn,
                            bool& rtn2,
                            xf::codec::decOutput* pout);

} // namespace codec
} // namespace xf
void kernel_parser_decoder(ap_uint<AXI_WIDTH>* datatoDDR,
                           const int size,
                           xf::codec::img_info& img_info,
                           xf::codec::hls_compInfo hls_cmpnfo[MAX_NUM_COLOR],
                           hls::stream<ap_uint<24> >& block_strm,
                           int& rtn,
                           bool& rtn2,
                           xf::codec::decOutput* pout);

#endif
