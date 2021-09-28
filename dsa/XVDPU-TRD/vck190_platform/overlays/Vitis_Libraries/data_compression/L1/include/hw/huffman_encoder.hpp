/*
 * (c) Copyright 2019-2021 Xilinx, Inc. All rights reserved.
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
 *
 */
#ifndef _XFCOMPRESSION_HUFFMAN_ENCODER_HPP_
#define _XFCOMPRESSION_HUFFMAN_ENCODER_HPP_

/**
 * @file huffman_encoder.hpp
 * @brief Header for module used in ZLIB huffman kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "compress_utils.hpp"
#include <ap_int.h>
#include "zlib_specs.hpp"
#include "zlib_tables.hpp"

// LZ specific Defines
#define d_code(dist, dist_code) ((dist) < 256 ? dist_code[dist] : dist_code[256 + ((dist) >> 7)])

namespace xf {
namespace compression {
namespace details {

void bitPackingStatic(hls::stream<DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> >& inStream,
                      hls::stream<IntVectorStream_dt<8, 2> >& outStream) {
    DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> inValue;
    IntVectorStream_dt<8, 2> outValue;

    while (true) { // iterate over blocks
        ap_uint<32> localBits = 0;
        uint8_t localBits_idx = 0;
        bool last_block = true;

    bitpack:
        for (inValue = inStream.read(); inValue.strobe != 0; inValue = inStream.read()) {
#pragma HLS PIPELINE II = 1
            uint16_t val = inValue.data[0].code;
            uint8_t size = inValue.data[0].bitlen;
            localBits.range(size + localBits_idx - 1, localBits_idx) = val;
            localBits_idx += size;

            if (localBits_idx >= 16) {
                uint16_t pack_byte = localBits;
                localBits >>= 16;
                localBits_idx -= 16;

                outValue.strobe = 2;
                outValue.data[0] = (uint8_t)pack_byte;
                outValue.data[1] = (uint8_t)(pack_byte >> 8);
                outStream << outValue;
            }
            last_block = false;
        }

        // end of file
        if (last_block) {
            outValue.strobe = 0;
            outStream << outValue;
            break;
        }

        int leftBits = localBits_idx % 8;
        int val = 0;
        ap_uint<64> packedBits = 0;

        if (leftBits) {
            // 3 bit header
            localBits_idx += 3;

            leftBits = localBits_idx % 8;

            if (leftBits != 0) {
                val = 8 - leftBits;
                localBits_idx += val;
            }

            // Zero bytes and complement as per type0 z_sync_flush
            uint64_t val = 0xffff0000;
            packedBits = localBits | (val << localBits_idx);
            localBits_idx += 32;

        } else {
            // Zero bytes and complement as per type0 z_sync_flush
            uint64_t val = 0xffff000000;
            packedBits = localBits | (val << localBits_idx);
            localBits_idx += 40;
        }
        for (uint32_t i = 0; i < localBits_idx; i += 16) {
            uint16_t pack_byte = packedBits;
            packedBits >>= 16;

            outValue.data[0] = pack_byte;
            if (localBits_idx - i < 16) {
                outValue.strobe = 1;
            } else {
                outValue.strobe = 2;
                outValue.data[1] = (uint8_t)(pack_byte >> 8);
            }
            outStream << outValue;
        }

        // end of block
        outValue.strobe = 0;
        outValue.data[0] = 0;
        outStream << outValue;
    }
}

void bitPackingStream(hls::stream<DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> >& inStream,
                      hls::stream<IntVectorStream_dt<8, 2> >& outStream) {
    DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> inValue;
    IntVectorStream_dt<8, 2> outValue;
    int blk_n = 0;
    while (true) { // iterate over blocks
        ap_uint<32> localBits = 0;
        uint8_t localBits_idx = 0;
        bool last_block = true;

    bitpack:
        for (inValue = inStream.read(); inValue.strobe != 0; inValue = inStream.read()) {
#pragma HLS PIPELINE II = 1
            uint16_t val = inValue.data[0].code;
            uint8_t size = inValue.data[0].bitlen;
            localBits.range(size + localBits_idx - 1, localBits_idx) = val;
            localBits_idx += size;

            if (localBits_idx >= 16) {
                uint16_t pack_byte = localBits;
                localBits >>= 16;
                localBits_idx -= 16;

                outValue.strobe = 2;
                outValue.data[0] = (uint8_t)pack_byte;
                outValue.data[1] = (uint8_t)(pack_byte >> 8);
                outStream << outValue;
            }
            last_block = false;
        }

        // end of file
        if (last_block) {
            outValue.strobe = 0;
            outStream << outValue;
            break;
        }

        int leftBits = localBits_idx % 8;
        int val = 0;
        ap_uint<64> packedBits = 0;

        if (leftBits) {
            // 3 bit header
            localBits_idx += 3;

            leftBits = localBits_idx % 8;

            if (leftBits != 0) {
                val = 8 - leftBits;
                localBits_idx += val;
            }

            // Zero bytes and complement as per type0 z_sync_flush
            uint64_t val = 0xffff0000;
            packedBits = localBits | (val << localBits_idx);
            localBits_idx += 32;

        } else {
            // Zero bytes and complement as per type0 z_sync_flush
            uint64_t val = 0xffff000000;
            packedBits = localBits | (val << localBits_idx);
            localBits_idx += 40;
        }
        for (uint32_t i = 0; i < localBits_idx; i += 16) {
#pragma HLS PIPELINE II = 1
            uint16_t pack_byte = packedBits;
            packedBits >>= 16;

            outValue.data[0] = pack_byte;
            if (localBits_idx - i < 9) {
                outValue.strobe = 1;
            } else {
                outValue.strobe = 2;
                outValue.data[1] = (uint8_t)(pack_byte >> 8);
            }
            outStream << outValue;
        }
        // end of block
        outValue.strobe = 0;
        outValue.data[0] = 0;
        outStream << outValue;
    }
}

/**
 * @brief This module does updates lz77 byte data to 32bit data.
 * @param inStream input packet of 8bit contains either literal or marker or match length or distance
 * information.
 * @param outStream output 32bit compressed data
 */
void huffmanProcessingUnit(hls::stream<IntVectorStream_dt<9, 1> >& inStream,
                           hls::stream<IntVectorStream_dt<32, 1> >& outStream) {
    enum HuffEncoderStates { READ_LITERAL, READ_MATCH_LEN, READ_OFFSET0, READ_OFFSET1 };
    IntVectorStream_dt<32, 1> outValue;
    outValue.strobe = 0;
    bool done = false;
    int blk_n = 0;
    while (true) { // iterate over multiple blocks in a file
        enum HuffEncoderStates next_state = READ_LITERAL;

        IntVectorStream_dt<9, 1> nextValue = inStream.read();
        // end of file
        if (nextValue.strobe == 0) {
            outValue.strobe = 0;
            outStream << outValue;
            break;
        }
        outValue.data[0] = 0;
        outValue.strobe = 1;

        done = false;
    hf_proc_static_main:
        while (!done) { // iterate over data in block
#pragma HLS PIPELINE II = 1
            bool outFlag = false;
            ap_uint<9> inValue = nextValue.data[0];
            bool tokenFlag = (inValue.range(8, 8) == 1);
            nextValue = inStream.read();

            if (next_state == READ_LITERAL) {
                if (tokenFlag) {
                    outValue.data[0].range(7, 0) = 0;
                    outValue.data[0].range(15, 8) = inValue.range(7, 0);
                    outFlag = false;
                    next_state = READ_OFFSET0;
                } else {
                    outValue.data[0].range(7, 0) = inValue;
                    outFlag = true;
                    next_state = READ_LITERAL;
                }
            } else if (next_state == READ_OFFSET0) {
                outFlag = false;
                outValue.data[0].range(23, 16) = inValue;
                outValue.strobe++;
                next_state = READ_OFFSET1;
            } else if (next_state == READ_OFFSET1) {
                outFlag = true;
                outValue.data[0].range(31, 24) = inValue;
                outValue.strobe++;
                next_state = READ_LITERAL;
            }

            if (outFlag) {
                outStream << outValue;
                outValue.data[0] = 0;
                outFlag = false;
            }
            // exit condition
            if (nextValue.strobe == 0) {
                done = true;
            }
        }
        // end of block
        outValue.strobe = 0;
        outStream << outValue;
    }
}

} // end details

void huffmanEncoderStatic(hls::stream<IntVectorStream_dt<32, 1> >& inStream,
                          hls::stream<DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> >& outStream) {
    uint8_t extra_lbits[c_length_codes] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                           2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};

    uint8_t extra_dbits[c_distance_codes] = {0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
                                             6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

    enum huffmanEncoderState { WRITE_TOKEN, ML_DIST_REP, LIT_REP, SEND_OUTPUT, ML_EXTRA, DIST_REP, DIST_EXTRA, DONE };
    bool done = false;
    DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> outValue;

    while (true) { // iterate over blocks
        enum huffmanEncoderState next_state = WRITE_TOKEN;
        uint8_t tCh = 0;
        uint8_t tLen = 0;
        uint16_t tOffset = 0;

        uint16_t lcode = 0;
        uint16_t dcode = 0;
        uint8_t lextra = 0;
        uint8_t dextra = 0;

        auto nextEncodedValue = inStream.read();
        if (nextEncodedValue.strobe == 0) {
            outValue.strobe = 0;
            outStream << outValue;
            break;
        }

        outValue.strobe = 1;

        // ZLIB Header for Static huffmanTree
        outValue.data[0].code = 2;
        outValue.data[0].bitlen = 3;
        outStream << outValue;
        done = false;
    huffman_loop:
        while (true) { // iterate over data in a block
#pragma HLS LOOP_TRIPCOUNT min = 1048576 max = 1048576
#pragma HLS PIPELINE II = 1
            if (next_state == WRITE_TOKEN) {
                auto curEncodedValue = nextEncodedValue.data[0];
                // exit if already done
                if (done) break;
                // read next value if not done
                nextEncodedValue = inStream.read();
                if (nextEncodedValue.strobe == 0) {
                    done = true;
                }

                tCh = curEncodedValue.range(7, 0);
                tLen = curEncodedValue.range(15, 8);
                tOffset = curEncodedValue.range(31, 16);
                dcode = d_code(tOffset, dist_code);
                lcode = length_code[tLen];

                if (tLen > 0) {
                    next_state = ML_DIST_REP;
                } else {
                    outValue.data[0].code = lit_code_fixed[tCh];
                    outValue.data[0].bitlen = lit_blen_fixed[tCh];
                    outStream << outValue;

                    next_state = WRITE_TOKEN;
                }
            } else if (next_state == ML_DIST_REP) {
                uint16_t code_s = lit_code_fixed[lcode + c_literals + 1];
                uint8_t bitlen = lit_blen_fixed[lcode + c_literals + 1];
                lextra = extra_lbits[lcode];

                outValue.data[0].code = code_s;
                outValue.data[0].bitlen = bitlen;
                outStream << outValue;

                if (lextra)
                    next_state = ML_EXTRA;
                else
                    next_state = DIST_REP;

            } else if (next_state == DIST_REP) {
                uint16_t code_s = dist_codes_fixed[dcode];
                uint8_t bitlen = dist_blen_fixed[dcode];
                dextra = extra_dbits[dcode];

                outValue.data[0].code = code_s;
                outValue.data[0].bitlen = bitlen;
                outStream << outValue;

                if (dextra)
                    next_state = DIST_EXTRA;
                else
                    next_state = WRITE_TOKEN;

            } else if (next_state == ML_EXTRA) {
                tLen -= base_length[lcode];
                outValue.data[0].code = tLen;
                outValue.data[0].bitlen = lextra;
                outStream << outValue;

                next_state = DIST_REP;

            } else if (next_state == DIST_EXTRA) {
                tOffset -= base_dist[dcode];
                outValue.data[0].code = tOffset;
                outValue.data[0].bitlen = dextra;
                outStream << outValue;

                next_state = WRITE_TOKEN;
            }
        }
        // End of block as per GZip standard
        outValue.data[0].code = lit_code_fixed[256];
        outValue.data[0].bitlen = lit_blen_fixed[256];
        outStream << outValue;
        // indicate end of data in stream for this block
        outValue.strobe = 0;
        outStream << outValue;
    }
}
/**
 * @brief This module does zlib/gzip dynamic huffman encoding
 * @param inStream input packet of 32bit size which contains either literal or match length and distance
 * information.
 *          Example: [Literal (1 Byte) | ML (1 Byte) | DIST (2 Bytes)]
 * @param outStream output bit encoded LZ77 compressed data
 * @param hufCodeInStream Huffman Codes input
 * @param hufCodeOutStream HuffmanCodes output
 */
void huffmanEncoderStream(hls::stream<IntVectorStream_dt<32, 1> >& inStream,
                          hls::stream<DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> >& hufCodeInStream,
                          hls::stream<DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> >& hufCodeOutStream) {
    enum huffmanEncoderState { WRITE_TOKEN, ML_DIST_REP, LIT_REP, SEND_OUTPUT, ML_EXTRA, DIST_REP, DIST_EXTRA, DONE };
    ap_uint<4> extra_lbits[c_length_codes] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                              2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};

    ap_uint<4> extra_dbits[c_distance_codes] = {0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
                                                6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
    const uint16_t c_ltree_size = 286;
    const uint8_t c_dtree_size = 30;

    bool done = false;
    DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> outValue;

    while (true) {
        HuffmanCode_dt<c_maxBits> litlnTree[c_ltree_size];
        HuffmanCode_dt<c_maxBits> distTree[c_dtree_size];
#pragma HLS aggregate variable = litlnTree
#pragma HLS aggregate variable = distTree

        // check if any tree data present
        auto hfcIn = hufCodeInStream.read();
        if (hfcIn.strobe == 0) {
            inStream.read(); // read zero strobe from input
            break;
        }
    init_litlen_tree_table:
        for (ap_uint<9> i = 0; i < c_ltree_size - 1; ++i) {
#pragma HLS PIPELINE II = 1
            litlnTree[i].code = hfcIn.data[0].code;
            litlnTree[i].bitlen = hfcIn.data[0].bitlen;
            hfcIn = hufCodeInStream.read();
        }
        litlnTree[c_ltree_size - 1].code = hfcIn.data[0].code;
        litlnTree[c_ltree_size - 1].bitlen = hfcIn.data[0].bitlen;

    init_dist_tree_table:
        for (ap_uint<5> i = 0; i < c_dtree_size; ++i) {
#pragma HLS PIPELINE II = 1
            hfcIn = hufCodeInStream.read();
            distTree[i].code = hfcIn.data[0].code;
            distTree[i].bitlen = hfcIn.data[0].bitlen;
        }

    // Passthrough the Preamble Data coming from Treegen to BitPacker
    preamble_pass_through:
        for (hfcIn = hufCodeInStream.read(); hfcIn.data[0].bitlen != 0; hfcIn = hufCodeInStream.read()) {
#pragma HLS PIPELINE II = 1
            hufCodeOutStream << hfcIn;
        }

        enum huffmanEncoderState next_state = WRITE_TOKEN;
        uint8_t tCh = 0;
        uint8_t tLen = 0;
        uint16_t tOffset = 0;

        uint16_t lcode = 0;
        uint16_t dcode = 0;
        uint8_t lextra = 0;
        uint8_t dextra = 0;

        outValue.strobe = 1;
        done = false;
        bool flag_out = false;
        // read ahead
        auto nextEncodedValue = inStream.read();

    huffman_loop:
        while (true) { // iterate over data in a block
#pragma HLS LOOP_TRIPCOUNT min = 1048576 max = 1048576
#pragma HLS PIPELINE II = 1
            if (next_state == WRITE_TOKEN) {
                auto curEncodedValue = nextEncodedValue.data[0];
                // exit if already done
                if (done) break;
                // read next value if not done
                nextEncodedValue = inStream.read();
                if (nextEncodedValue.strobe == 0) {
                    done = true;
                }

                tCh = curEncodedValue.range(7, 0);
                tLen = curEncodedValue.range(15, 8);
                tOffset = curEncodedValue.range(31, 16);
                dcode = d_code(tOffset, dist_code);
                lcode = length_code[tLen];

                if (tLen > 0) {
                    next_state = ML_DIST_REP;
                } else {
                    outValue.data[0].code = litlnTree[tCh].code;
                    outValue.data[0].bitlen = litlnTree[tCh].bitlen;
                    hufCodeOutStream << outValue;
                    next_state = WRITE_TOKEN;
                }
            } else if (next_state == ML_DIST_REP) {
                outValue.data[0].code = litlnTree[lcode + c_literals + 1].code;
                outValue.data[0].bitlen = litlnTree[lcode + c_literals + 1].bitlen;
                lextra = extra_lbits[lcode];
                hufCodeOutStream << outValue;

                if (lextra)
                    next_state = ML_EXTRA;
                else
                    next_state = DIST_REP;

            } else if (next_state == DIST_REP) {
                outValue.data[0].code = distTree[dcode].code;
                outValue.data[0].bitlen = distTree[dcode].bitlen;
                dextra = extra_dbits[dcode];
                hufCodeOutStream << outValue;

                if (dextra)
                    next_state = DIST_EXTRA;
                else
                    next_state = WRITE_TOKEN;

            } else if (next_state == ML_EXTRA) {
                tLen -= base_length[lcode];
                outValue.data[0].code = tLen;
                outValue.data[0].bitlen = lextra;
                hufCodeOutStream << outValue;

                next_state = DIST_REP;

            } else if (next_state == DIST_EXTRA) {
                tOffset -= base_dist[dcode];
                outValue.data[0].code = tOffset;
                outValue.data[0].bitlen = dextra;
                hufCodeOutStream << outValue;

                next_state = WRITE_TOKEN;
            }
        }
        // End of block as per GZip standard
        outValue.data[0].code = litlnTree[256].code;
        outValue.data[0].bitlen = litlnTree[256].bitlen;
        hufCodeOutStream << outValue;
        // indicate end of data in stream for this block
        outValue.strobe = 0;
        hufCodeOutStream << outValue;
    }
    // end of data
    outValue.strobe = 0;
    hufCodeOutStream << outValue;
}
}
}
#endif // _XFCOMPRESSION_HUFFMAN_ENCODER_HPP_
