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
 * @file build_table_encode_histo.hpp
 */

#ifndef _XF_CODEC_BUILD_TABLE_ENCODE_HISTO_HPP_
#define _XF_CODEC_BUILD_TABLE_ENCODE_HISTO_HPP_

#include "kernel3/build_cluster.hpp"
#include "kernel3/kernel3_common.hpp"

// Static Huffman code for encoding logcounts. The last symbol is used as RLE
// sequence.
static const uint8_t hls_kLogCountBitLengths[hls_ANS_LOG_TAB_SIZE + 2] = {
    5, 4, 4, 4, 3, 3, 2, 3, 3, 6, 7, 7,
};
static const uint16_t hls_kLogCountSymbols[hls_ANS_LOG_TAB_SIZE + 2] = {
    15, 3, 11, 7, 2, 6, 0, 1, 5, 31, 63, 127,
};

void XAcc_BuildAndStore_top(uint32_t histogram[MAX_ALPHABET_SIZE],
                            const uint16_t alphabet_size,
                            hls_ANSEncSymbolInfo ans_table[MAX_ALPHABET_SIZE],
                            int* pos, // tmp cache int64
                            hls::stream<uint8_t>& strm_histo,
                            uint8_t& tail_bits);

void XAcc_EncodeHistogramsFast_top(const bool is_dc,
                                   uint8_t dc_context_map[MAX_NUM_COLOR],
                                   hls::stream<ap_uint<13> >& strm_token_addr,
                                   hls::stream<bool>& strm_e_addr,
                                   hls_ANSEncSymbolInfo hls_codes[hls_kNumStaticContexts][hls_alphabet_size],
                                   hist_t hls_histograms[hls_NumHistograms],
                                   int pos,
                                   int& len_histo,
                                   hls::stream<uint8_t>& strm_histo_byte,
                                   hls::stream<bool>& strm_histo_e);

void ADD_FP_strm(const int num_in,
                 hls::stream<float>& strm_in, // max 256 input
                 hls::stream<float>& strm_sum);

void build_historgram(hls::stream<ap_uint<13> >& strm_token_addr,
                      hls::stream<bool>& strm_e_addr,
                      hist_t total[hls_kMinClustersForHistogramRemap],
                      hist_t hls_histograms[hls_NumHistograms]);

void build_historgram_syn(hls::stream<ap_uint<13> >& strm_token_addr,
                          hls::stream<bool>& strm_e_addr,
                          hist_t total[hls_kMinClustersForHistogramRemap],
                          hist_t hls_histograms[hls_NumHistograms],
                          hist_t hls_histograms2[hls_kNumStaticContexts][MAX_ALPHABET_SIZE]);

void hls_EncodeContextMap(const uint8_t context_map[MAX_NUM_COLOR],
                          const int num_histograms,
                          int& num,
                          hls::stream<nbits_t>& strm_nbits,
                          hls::stream<uint16_t>& strm_bits);

void hls_build_and_encode_top(const bool is_dc,
                              uint32_t histogram[hls_kNumStaticContexts][MAX_ALPHABET_SIZE],
                              const uint16_t alphabet_size,
                              const int cluster_size,
                              const uint16_t alphabet_size_dc[MAX_NUM_COLOR],
                              hls_ANSEncSymbolInfo ans_table[hls_kNumStaticContexts][MAX_ALPHABET_SIZE],
                              int& pos, // tmp cache int64
                              const bool do_encode,
                              hls::stream<uint8_t>& strm_histo,
                              hls::stream<bool>& strm_histo_e,
                              uint8_t& tail_bits);

void hls_encode_histo_context(const bool is_dc,
                              const hist_t hls_histograms2[hls_kNumStaticContexts][MAX_ALPHABET_SIZE],
                              hls_ANSEncSymbolInfo hls_codes[hls_kNumStaticContexts][MAX_ALPHABET_SIZE],
                              uint8_t dc_context_map[MAX_NUM_COLOR],
                              int& pos,
                              hls::stream<uint8_t>& strm_histo_byte,
                              hls::stream<bool>& strm_histo_e);

#endif
