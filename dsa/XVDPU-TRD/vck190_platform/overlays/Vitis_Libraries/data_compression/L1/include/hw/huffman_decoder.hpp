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
#ifndef _XFCOMPRESSION_HUFFMAN_DECODER_HPP_
#define _XFCOMPRESSION_HUFFMAN_DECODER_HPP_
/**
 * @file huffman_decoder.hpp
 * @brief Header for module used in ZLIB decompress kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

namespace xf {
namespace compression {

enum eHuffmanType { FIXED = 0, DYNAMIC, FULL };
typedef ap_uint<32> bitBufferTypeLL;
typedef ap_uint<48> bitBufferType;

namespace details {

enum blockStatus { PENDING = 0, FINISH = 1 };
void loadBitStream(bitBufferType& bitbuffer,
                   ap_uint<6>& bits_cntr,
                   hls::stream<ap_uint<16> >& inStream,
                   hls::stream<bool>& inEos,
                   bool& done) {
#pragma HLS INLINE off
    while (bits_cntr < 32 && (done == false)) {
    loadBitStream:
        uint16_t tmp_dt = (uint16_t)inStream.read();
        bitbuffer += (bitBufferType)(tmp_dt) << bits_cntr;
        done = inEos.read();
        bits_cntr += 16;
    }
}

void loadBitStreamLL(bitBufferTypeLL& bitbuffer,
                     ap_uint<6>& bits_cntr,
                     hls::stream<ap_uint<16> >& inStream,
                     hls::stream<bool>& inEos,
                     bool& done) {
#pragma HLS INLINE off
    if (bits_cntr < 16 && (done == false)) {
    loadBitStream:
        uint16_t tmp_dt = (uint16_t)inStream.read();
        bitbuffer += (bitBufferTypeLL)(tmp_dt) << bits_cntr;
        done = inEos.read();
        bits_cntr += 16;
    }
}

void discardBitStream(bitBufferType& bitbuffer, ap_uint<6>& bits_cntr, ap_uint<6> n_bits) {
#pragma HLS INLINE off
    bitbuffer >>= n_bits;
    bits_cntr -= n_bits;
}

void discardBitStreamLL(bitBufferTypeLL& bitbuffer, ap_uint<6>& bits_cntr, ap_uint<6> n_bits) {
#pragma HLS INLINE off
    bitbuffer >>= n_bits;
    bits_cntr -= n_bits;
}

template <typename T>
T reg(T d) {
#pragma HLS PIPELINE II = 1
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS INLINE off
    return d;
}

inline uint8_t huffmanBytegenStatic(bitBufferType& _bitbuffer,
                                    ap_uint<6>& bits_cntr,
                                    hls::stream<ap_uint<17> >& outStream,
                                    hls::stream<bool>& inEos,
                                    hls::stream<ap_uint<16> >& inStream,
                                    const uint8_t* array_codes_op,
                                    const uint8_t* array_codes_bits,
                                    const uint16_t* array_codes_val,
                                    bool& done) {
#pragma HLS INLINE
    uint16_t used = 512;
    uint16_t lit_mask = 511;
    uint16_t dist_mask = 31;
    bitBufferType bitbuffer = details::reg<bitBufferType>(_bitbuffer);
    ap_uint<10> lidx = bitbuffer & lit_mask;
    uint8_t current_op = details::reg<uint8_t>(array_codes_op[lidx]);
    uint8_t current_bits = details::reg<uint8_t>(array_codes_bits[lidx]);
    uint16_t current_val = details::reg<uint16_t>(array_codes_val[lidx]);
    bool is_length = true;
    ap_uint<17> tmpVal;
    uint8_t ret = 0;

    bool huffDone = false;
ByteGenStatic:
    for (; !huffDone;) {
#pragma HLS PIPELINE II = 1
        ap_uint<4> len1 = current_bits;
        ap_uint<4> len2 = 0;
        ap_uint<4> ml_op = current_op;
        uint64_t bitbuffer1 = bitbuffer >> current_bits;
        ap_uint<9> bitbuffer3 = bitbuffer >> current_bits;
        uint64_t bitbuffer2 = bitbuffer >> (current_bits + ml_op);
        bits_cntr -= current_bits;

        if (current_op == 0) {
            tmpVal.range(8, 1) = (uint8_t)(current_val);
            tmpVal.range(16, 9) = 0XFF;
            tmpVal.range(0, 0) = 0;
            outStream << tmpVal;
            lidx = bitbuffer3;
            is_length = true;
        } else if (current_op & 16) {
            uint16_t len = (uint16_t)(current_val);
            len += (uint16_t)bitbuffer1 & ((1 << ml_op) - 1);
            len2 = ml_op;
            bits_cntr -= ml_op;
            tmpVal.range(16, 1) = len;
            tmpVal.range(0, 0) = 0;
            outStream << tmpVal;
            uint16_t array_offset = (is_length) ? used : 0;
            ap_uint<9> mask = (is_length) ? dist_mask : lit_mask;
            lidx = array_offset + (bitbuffer2 & mask);
            is_length = !(is_length);
        } else if (current_op & 32) {
            if (is_length) {
                ret = blockStatus::PENDING;
            } else {
                ret = blockStatus::FINISH;
            }
            huffDone = true;
        }

        if ((done == true) && (bits_cntr < 32)) {
            ret = blockStatus::FINISH;
            huffDone = true;
        }
        if ((bits_cntr < 32) && (done == false)) {
            uint16_t inValue = inStream.read();
            done = inEos.read();
            bitbuffer = (bitbuffer >> (len1 + len2)) | (bitBufferType)(inValue) << bits_cntr;
            bits_cntr += (uint8_t)16;
        } else {
            bitbuffer >>= (len1 + len2);
        }
        current_op = array_codes_op[lidx];
        current_bits = array_codes_bits[lidx];
        current_val = array_codes_val[lidx];
    }
    _bitbuffer = bitbuffer;
    return ret;
}

uint8_t huffmanBytegen(bitBufferType& _bitbuffer,
                       ap_uint<6>& bits_cntr,
                       hls::stream<ap_uint<17> >& outStream,
                       hls::stream<bool>& inEos,
                       hls::stream<ap_uint<16> >& inStream,
                       const uint32_t* array_codes,
                       const uint32_t* array_codes_extra,
                       const uint32_t* array_codes_dist,
                       const uint32_t* array_codes_dist_extra,
                       bool& done) {
#pragma HLS INLINE
    uint16_t lit_mask = 511; // Adjusted according to 8 bit
    uint16_t dist_mask = 511;
    bitBufferType bitbuffer = details::reg<bitBufferType>(_bitbuffer);
    //    bitBufferType bitbuffer = _bitbuffer;
    ap_uint<9> lidx = bitbuffer;
    ap_uint<9> lidx1;
    ap_uint<32> current_array_val = details::reg<ap_uint<32> >(array_codes[lidx]);
    uint8_t current_op = current_array_val.range(31, 24);
    uint8_t current_bits = current_array_val.range(23, 16);
    uint16_t current_val = current_array_val.range(15, 0);
    bool is_length = true;
    ap_uint<32> tmpVal;
    uint8_t ret = 0;
    bool dist_extra = false;
    bool len_extra = false;

    bool huffDone = false;
ByteGen:
    for (; !huffDone;) {
#pragma HLS PIPELINE II = 1
        ap_uint<4> len1 = current_bits;
        ap_uint<4> len2 = 0;
        ap_uint<4> ml_op = current_op;
        uint8_t current_op1 = (current_op == 0 || current_op >= 64) ? 1 : current_op;
        ap_uint<64> bitbuffer1 = bitbuffer >> current_bits;
        ap_uint<9> bitbuffer3 = bitbuffer >> current_bits;
        lidx1 = bitbuffer1.range(current_op1 - 1, 0) + current_val;
        ap_uint<9> bitbuffer2 = bitbuffer.range(current_bits + ml_op + 8, current_bits + ml_op);
        bits_cntr -= current_bits;
        dist_extra = false;
        len_extra = false;

        if (current_op == 0) {
            tmpVal.range(8, 1) = (uint8_t)(current_val);
            tmpVal.range(16, 9) = 0xFF;
            tmpVal.range(0, 0) = 0;
            // std::cout << (char)(current_val);
            outStream << tmpVal;
            lidx = bitbuffer3;
            is_length = true;
        } else if (current_op & 16) {
            uint16_t len = (uint16_t)(current_val);
            len += (uint16_t)bitbuffer1 & ((1 << ml_op) - 1);
            len2 = ml_op;
            bits_cntr -= ml_op;
            tmpVal.range(0, 0) = 0;
            tmpVal.range(16, 1) = len;
            outStream << tmpVal;
            lidx = bitbuffer2;
            is_length = !(is_length);
        } else if ((current_op & 64) == 0) {
            if (is_length) {
                len_extra = true;
            } else {
                dist_extra = true;
            }
        } else if (current_op & 32) {
            if (is_length) {
                ret = blockStatus::PENDING;
            } else {
                ret = blockStatus::FINISH;
            }
            huffDone = true;
        }
        if ((done == true) && (bits_cntr < 32)) {
            huffDone = true;
            ret = blockStatus::FINISH;
        }
        if (bits_cntr < 32 && (done == false)) {
            uint16_t inValue = inStream.read();
            done = inEos.read();
            bitbuffer = (bitbuffer >> (len1 + len2)) | (bitBufferType)(inValue) << bits_cntr;
            bits_cntr += (uint8_t)16;
        } else {
            bitbuffer >>= (len1 + len2);
        }
        if (len_extra) {
            ap_uint<32> val = array_codes_extra[lidx1];
            current_op = val.range(31, 24);
            current_bits = val.range(23, 16);
            current_val = val.range(15, 0);
        } else if (dist_extra) {
            ap_uint<32> val = array_codes_dist_extra[lidx1];
            current_op = val.range(31, 24);
            current_bits = val.range(23, 16);
            current_val = val.range(15, 0);
        } else if (is_length) {
            ap_uint<32> val = array_codes[lidx];
            current_op = val.range(31, 24);
            current_bits = val.range(23, 16);
            current_val = val.range(15, 0);
        } else {
            ap_uint<32> val = array_codes_dist[lidx];
            current_op = val.range(31, 24);
            current_bits = val.range(23, 16);
            current_val = val.range(15, 0);
        }
    }
    _bitbuffer = bitbuffer;
    return ret;
}

uint8_t huffmanBytegenLL(bitBufferTypeLL& _bitbuffer,
                         ap_uint<6>& bits_cntr,
                         hls::stream<ap_uint<16> >& outStream,
                         hls::stream<bool>& inEos,
                         hls::stream<ap_uint<16> >& inStream,
                         ap_uint<16> (&codeOffsets)[2][15],
                         ap_uint<9> (&bl1Codes)[2][2],
                         ap_uint<9> (&bl2Codes)[2][4],
                         ap_uint<9> (&bl3Codes)[2][8],
                         ap_uint<9> (&bl4Codes)[2][16],
                         ap_uint<9> (&bl5Codes)[2][32],
                         ap_uint<9> (&bl6Codes)[2][64],
                         ap_uint<9> (&bl7Codes)[2][128],
                         ap_uint<9> (&bl8Codes)[2][256],
                         ap_uint<9> (&bl9Codes)[2][256],
                         ap_uint<9> (&bl10Codes)[2][256],
                         ap_uint<9> (&bl11Codes)[2][256],
                         ap_uint<9> (&bl12Codes)[2][256],
                         ap_uint<9> (&bl13Codes)[2][256],
                         ap_uint<9> (&bl14Codes)[2][256],
                         ap_uint<9> (&bl15Codes)[2][256],
                         bool& done,
                         uint8_t ignoreValue) {
    ap_uint<15> validCodeOffset[2];
    ap_uint<4> current_bits[2];
#pragma HLS ARRAY_PARTITION variable = current_bits complete dim = 0
    ap_uint<9> symbol[2][16];
#pragma HLS ARRAY_PARTITION variable = symbol complete dim = 0
    ap_uint<9> lsymbol[2];
    ap_uint<16> tmpVal;
    bitBufferTypeLL lBitBuffer[4];
#pragma HLS ARRAY_PARTITION variable = lBitBuffer complete dim = 0
    ap_uint<6> lbitsCntr[2], ebitsCntr[2];
    bool isDistance = false;
    bool isExtra = false;
    bool huffDone = false;
    uint16_t lval, dval;
    ap_uint<5> val0, val1;
    ap_uint<4> lextra, dextra;
    uint8_t ret = 0;
    bool outputReady = false;

    const ap_uint<4> lext[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2,  2,
                                 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 13, 10};

    const ap_uint<4> dext[32] = {0, 0, 0, 0, 1, 1, 2,  2,  3,  3,  4,  4,  5,  5,  6, 6,
                                 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 0, 0};
ByteGen:
    while (!huffDone && !done) {
#pragma HLS PIPELINE II = 1
    comparator:
        for (ap_uint<2> j = 0; j < 2; j++) {
#pragma HLS UNROLL
            //#pragma HLS LOOP_FLATTEN
            for (ap_uint<5> i = 0; i < 15; i++) {
#pragma HLS UNROLL
                validCodeOffset[j].range(i, i) = (_bitbuffer.range(0, i) >= codeOffsets[j][i]) ? 1 : 0;
            }
        }
        current_bits[0] = ap_uint<6>(32) - ap_uint<6>(__builtin_clz((unsigned int)(validCodeOffset[0])));
        current_bits[1] = ap_uint<6>(32) - ap_uint<6>(__builtin_clz((unsigned int)(validCodeOffset[1])));
        for (ap_uint<2> i = 0; i < 2; i++) {
#pragma HLS UNROLL
            symbol[i][1] = bl1Codes[i][_bitbuffer.range(0, 0)];
            symbol[i][2] = bl2Codes[i][_bitbuffer.range(0, 1)];
            symbol[i][3] = bl3Codes[i][_bitbuffer.range(0, 2)];
            symbol[i][4] = bl4Codes[i][_bitbuffer.range(0, 3)];
            symbol[i][5] = bl5Codes[i][_bitbuffer.range(0, 4)];
            symbol[i][6] = bl6Codes[i][_bitbuffer.range(0, 5)];
            symbol[i][7] = bl7Codes[i][_bitbuffer.range(0, 6)];
            symbol[i][8] = bl8Codes[i][_bitbuffer.range(0, 7)];
            symbol[i][9] = bl9Codes[i][ap_uint<8>(_bitbuffer.range(0, 8))];
            symbol[i][10] = bl10Codes[i][ap_uint<8>(_bitbuffer.range(0, 9))];
            symbol[i][11] = bl11Codes[i][ap_uint<8>(_bitbuffer.range(0, 10))];
            symbol[i][12] = bl12Codes[i][ap_uint<8>(_bitbuffer.range(0, 11))];
            symbol[i][13] = bl13Codes[i][ap_uint<8>(_bitbuffer.range(0, 12))];
            symbol[i][14] = bl14Codes[i][ap_uint<8>(_bitbuffer.range(0, 13))];
            symbol[i][15] = bl15Codes[i][ap_uint<8>(_bitbuffer.range(0, 14))];
        }

        lbitsCntr[1] = bits_cntr - current_bits[1];
        lbitsCntr[0] = bits_cntr - current_bits[0];
        lsymbol[0] = symbol[0][current_bits[0]];
        lsymbol[1] = symbol[1][current_bits[1]];
        lBitBuffer[0] = _bitbuffer >> current_bits[0];
        lBitBuffer[1] = _bitbuffer >> current_bits[1];

        lextra = lext[val0]; // previous symbol
        lval = (uint16_t)_bitbuffer & ((1 << lextra) - 1);
        ebitsCntr[0] = bits_cntr - lextra;
        lBitBuffer[2] = _bitbuffer >> lextra;

        dextra = dext[val1];
        dval = (uint16_t)_bitbuffer & ((1 << dextra) - 1);
        ebitsCntr[1] = bits_cntr - dextra;
        lBitBuffer[3] = _bitbuffer >> dextra;

        if (isExtra && isDistance) { // extra bits ml/distance
            isExtra = false;
            bits_cntr = ebitsCntr[0];
            tmpVal.range(15, 0) = lval;
            _bitbuffer = lBitBuffer[2];
        } else if (isExtra && !isDistance) {
            isExtra = false;
            bits_cntr = ebitsCntr[1];
            tmpVal.range(15, 0) = dval;
            _bitbuffer = lBitBuffer[3];
        } else if ((isDistance == false) && lsymbol[0] < 256) { // literal
            tmpVal.range(7, 0) = (uint8_t)(lsymbol[0]);
            tmpVal.range(15, 8) = 0xF0;
            bits_cntr = lbitsCntr[0];
            isExtra = false;
            _bitbuffer = lBitBuffer[0];
        } else if (lsymbol[0] == 256 && (isDistance == false)) {
            huffDone = true;
            ret = blockStatus::FINISH;
            bits_cntr = lbitsCntr[0];
            _bitbuffer = lBitBuffer[0];
            tmpVal.range(15, 8) = ignoreValue; // invalid Value
        } else if (isDistance == false) {      // match length
            isDistance = true;
            val0 = lsymbol[0];
            isExtra = val0 < 9 ? false : true;
            tmpVal.range(15, 0) = val0;
            bits_cntr = lbitsCntr[0];
            _bitbuffer = lBitBuffer[0];
        } else { // distance
            isDistance = false;
            val1 = lsymbol[1];
            isExtra = val1 < 4 ? false : true;
            tmpVal.range(15, 0) = val1;
            bits_cntr = lbitsCntr[1];
            _bitbuffer = lBitBuffer[1];
        }

        outStream << tmpVal;

        if (!(bits_cntr & 0xF0)) {
            uint16_t inValue = inStream.read();
            done = inEos.read();
            _bitbuffer |= (bitBufferTypeLL)(inValue) << bits_cntr;
            bits_cntr += (ap_uint<6>)16;
        }
    }
    return ret;
}

void byteGen(bitBufferTypeLL& _bitbuffer,
             ap_uint<6>& bits_cntr,
             ap_uint<16>* codeOffsets,
             ap_uint<9>* bl1Codes,
             ap_uint<9>* bl2Codes,
             ap_uint<9>* bl3Codes,
             ap_uint<9>* bl4Codes,
             ap_uint<9>* bl5Codes,
             ap_uint<9>* bl6Codes,
             ap_uint<9>* bl7Codes,
             uint16_t* lens,
             hls::stream<bool>& inEos,
             hls::stream<ap_uint<16> >& inStream,
             ap_uint<9> nlen,
             ap_uint<9> ndist,
             bool& done) {
    loadBitStreamLL(_bitbuffer, bits_cntr, inStream, inEos, done);
    uint8_t copy = 0;
    uint8_t extra_copy = 0;
    uint8_t nNum = 0;
    ap_uint<7> validCodeOffset;
    ap_uint<5> symbol[8];
#pragma HLS ARRAY_PARTITION variable = symbol dim = 1 complete
    uint16_t len = 0;
    uint16_t dynamic_curInSize = 0;
    bitBufferTypeLL bitbuffer[2];
    bool isExtra = false;
    ap_uint<3> extra = 0;
bytegen:
    while ((dynamic_curInSize < nlen + ndist) || (copy != 0)) {
#pragma HLS PIPELINE II = 1
        validCodeOffset.range(0, 0) = (_bitbuffer.range(0, 0) >= codeOffsets[0]) ? 1 : 0;
        validCodeOffset.range(1, 1) = (_bitbuffer.range(0, 1) >= codeOffsets[1]) ? 1 : 0;
        validCodeOffset.range(2, 2) = (_bitbuffer.range(0, 2) >= codeOffsets[2]) ? 1 : 0;
        validCodeOffset.range(3, 3) = (_bitbuffer.range(0, 3) >= codeOffsets[3]) ? 1 : 0;
        validCodeOffset.range(4, 4) = (_bitbuffer.range(0, 4) >= codeOffsets[4]) ? 1 : 0;
        validCodeOffset.range(5, 5) = (_bitbuffer.range(0, 5) >= codeOffsets[5]) ? 1 : 0;
        validCodeOffset.range(6, 6) = (_bitbuffer.range(0, 6) >= codeOffsets[6]) ? 1 : 0;
        ap_uint<4> current_bits = 32 - __builtin_clz((unsigned int)(validCodeOffset));
        symbol[1] = bl1Codes[_bitbuffer.range(0, 0)];
        symbol[2] = bl2Codes[_bitbuffer.range(0, 1)];
        symbol[3] = bl3Codes[_bitbuffer.range(0, 2)];
        symbol[4] = bl4Codes[_bitbuffer.range(0, 3)];
        symbol[5] = bl5Codes[_bitbuffer.range(0, 4)];
        symbol[6] = bl6Codes[_bitbuffer.range(0, 5)];
        symbol[7] = bl7Codes[_bitbuffer.range(0, 6)];

        auto current_val = symbol[current_bits];
        bitbuffer[0] = _bitbuffer >> current_bits;
        bitbuffer[1] = _bitbuffer >> extra;
        extra_copy = _bitbuffer & ((1 << extra) - 1);

        if (isExtra) {
            isExtra = false;
            copy += extra_copy;
            _bitbuffer = bitbuffer[1];
            bits_cntr -= extra;
        } else if (copy != 0) {
        } else if (current_val < 16) {
            _bitbuffer = bitbuffer[0];
            bits_cntr -= current_bits;
            len = current_val;
            copy = 1;
        } else if (current_val == 16) {
            copy = 3;                  // use 2 bits
            _bitbuffer = bitbuffer[0]; // dump 2 bits
            bits_cntr -= current_bits; // update bits_cntr
            extra = 2;
            isExtra = true;
        } else if (current_val == 17) {
            len = 0;
            copy = 3; // use 3 bits
            _bitbuffer = bitbuffer[0];
            bits_cntr -= current_bits;
            extra = 3;
            isExtra = true;
        } else {
            len = 0;
            copy = 11; // use 7 bits
            _bitbuffer = bitbuffer[0];
            bits_cntr -= current_bits;
            extra = 7;
            isExtra = true;
        }

        if (copy != 0) {
            lens[dynamic_curInSize++] = len;
            copy -= 1;
        }
        if (bits_cntr < 16 && !done) {
            uint16_t tmp_data = inStream.read();
            _bitbuffer += (bitBufferTypeLL)(tmp_data) << bits_cntr;
            done = inEos.read();
            bits_cntr += 16;
        }
    }
}

void code_generator_array_dyn(
    uint8_t curr_table, uint16_t* lens, ap_uint<9> codes, uint32_t* table, uint32_t* table_extra, uint32_t bits) {
/**
 * @brief This module regenerates the code values based on bit length
 * information present in block preamble. Output generated by this module
 * presents operation, bits and value for each literal, match length and
 * distance.
 *
 * @param curr_table input current module to process i.e., literal or
 * distance table etc
 * @param lens input bit length information
 * @param codes input number of codes
 * @param table_op output operation per active symbol (literal or distance)
 * @param table_bits output bits to process per symbol (literal or distance)
 * @param table_val output value per symbol (literal or distance)
 * @param bits represents the start of the table
 * @param used presents next valid entry in table
 */
#pragma HLS INLINE REGION
    uint16_t sym = 0;
    uint16_t min, max = 0;
    uint16_t extra_idx = 0;
    uint32_t root = bits;
    uint16_t curr;
    uint16_t drop;
    uint16_t huff = 0;
    uint16_t incr;
    int16_t fill;
    uint16_t low;
    uint16_t mask;

    const uint16_t c_maxbits = 15;
    uint8_t code_data_op = 0;
    uint8_t code_data_bits = 0;
    uint16_t code_data_val = 0;

    uint8_t* nptr_op;
    uint8_t* nptr_bits;
    uint16_t* nptr_val;
    uint32_t* nptr;
    uint32_t* nptr_extra;

    const uint16_t* base;
    const uint16_t* extra;
    uint16_t match;
    uint16_t count[c_maxbits + 1] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#pragma HLS ARRAY_PARTITION variable = count

    uint16_t offs[c_maxbits + 1];
#pragma HLS ARRAY_PARTITION variable = offs

    uint16_t codeBuffer[512];
#pragma HLS DEPENDENCE false inter variable = codeBuffer

    const uint16_t lbase[32] = {3,  4,  5,  6,  7,  8,  9,  10,  11,  13,  15,  17,  19,  23, 27, 31,
                                35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 0,  0,  0};
    const uint16_t lext[32] = {16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18,  18,
                               19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 16, 77, 202, 0};
    const uint16_t dbase[32] = {1,    2,    3,    4,    5,    7,     9,     13,    17,  25,   33,
                                49,   65,   97,   129,  193,  257,   385,   513,   769, 1025, 1537,
                                2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 0,   0};
    const uint16_t dext[32] = {16, 16, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22,
                               23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 64, 64};
cnt_lens:
    for (ap_uint<9> i = 0; i < codes; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS UNROLL FACTOR = 2
        uint16_t val = lens[i];
        if (val > max) {
            max = val;
        }
        count[val]++;
    }

min_loop:
    for (min = 1; min < max; min++) {
#pragma HLS PIPELINE II = 1
        if (count[min] != 0) break;
    }

    int left = 1;

    offs[1] = 0;
offs_loop:
    for (uint16_t i = 1; i < c_maxbits; i++)
#pragma HLS PIPELINE II = 1
        offs[i + 1] = offs[i] + count[i];

codes_loop:
    for (ap_uint<9> i = 0; i < codes; i++) {
#pragma HLS DEPENDENCE false inter variable = codeBuffer
#pragma HLS PIPELINE II = 1
#pragma HLS UNROLL FACTOR = 2
        if (lens[i] != 0) codeBuffer[offs[lens[i]]++] = (uint16_t)i;
    }

    switch (curr_table) {
        case 1:
            base = extra = codeBuffer;
            match = 20;
            break;
        case 2:
            base = lbase;
            extra = lext;
            match = 257;
            break;
        case 3:
            base = dbase;
            extra = dext;
            match = 0;
    }

    uint16_t len = min;

    nptr = table;
    nptr_extra = table_extra;

    curr = root;
    drop = 0;
    low = (uint32_t)(-1);
    mask = (1 << root) - 1;
    bool is_extra = false;

code_gen:
    for (;;) {
        code_data_bits = (uint8_t)(len - drop);

        if (codeBuffer[sym] + 1 < match) {
            code_data_op = (uint8_t)0;
            code_data_val = codeBuffer[sym];
        } else if (codeBuffer[sym] >= match) {
            code_data_op = (uint8_t)(extra[codeBuffer[sym] - match]);
            code_data_val = base[codeBuffer[sym] - match];
        } else {
            code_data_op = (uint8_t)(96);
            code_data_val = 0;
        }

        incr = 1 << (len - drop);
        fill = 1 << curr;
        min = fill;

        uint32_t code_val = ((uint32_t)code_data_op << 24) | ((uint32_t)code_data_bits << 16) | code_data_val;
        uint16_t fill_itr = fill / incr;
        uint16_t fill_idx = huff >> drop;
    fill:
        for (uint16_t i = 0; i < fill_itr; i++) {
#pragma HLS DEPENDENCE false inter variable = nptr
#pragma HLS DEPENDENCE false inter variable = nptr_extra
#pragma HLS PIPELINE II = 1
#pragma HLS UNROLL FACTOR = 2
            if (is_extra) {
                nptr_extra[fill_idx] = code_val;
            } else {
                nptr[fill_idx] = code_val;
            }

            fill_idx += incr;
        }

        fill = 0;

        incr = 1 << (len - 1);

        while (huff & incr) incr >>= 1;

        if (incr != 0) {
            huff &= incr - 1;
            huff += incr;
        } else
            huff = 0;

        sym++;

        if (--(count[len]) == 0) {
            if (len == max) break;
            len = lens[codeBuffer[sym]];
        }

        if (len > root && (huff & mask) != low) {
            if (drop == 0) {
                drop = root;
                min = 0;
                is_extra = true;
            }

            extra_idx += min;
            nptr_extra += min;
            curr = len - drop;
            left = (int)(1 << curr);

            uint16_t sum = curr + drop;
        left:
            for (int i = curr; i + drop < max; i++, curr++) {
#pragma HLS PIPELINE II = 1
#pragma HLS UNROLL FACTOR = 2
                left -= count[sum];
                if (left <= 0) break;
                left <<= 1;
                sum++;
            }

            low = huff & mask;
            table[low] = ((uint32_t)curr << 24) | ((uint32_t)root << 16) | extra_idx;
        }
    }
}

void code_generator_array_dyn_new(uint8_t curr_table,
                                  uint16_t* lens,
                                  ap_uint<9> codes,
                                  ap_uint<16>* codeOffsets,
                                  ap_uint<9>* bl1Codes,
                                  ap_uint<9>* bl2Codes,
                                  ap_uint<9>* bl3Codes,
                                  ap_uint<9>* bl4Codes,
                                  ap_uint<9>* bl5Codes,
                                  ap_uint<9>* bl6Codes,
                                  ap_uint<9>* bl7Codes,
                                  ap_uint<9>* bl8Codes,
                                  ap_uint<9>* bl9Codes,
                                  ap_uint<9>* bl10Codes,
                                  ap_uint<9>* bl11Codes,
                                  ap_uint<9>* bl12Codes,
                                  ap_uint<9>* bl13Codes,
                                  ap_uint<9>* bl14Codes,
                                  ap_uint<9>* bl15Codes) {
    uint16_t min = 15;
    uint16_t max = 0;

    const uint16_t c_maxbits = 15;

    ap_uint<9> count[c_maxbits + 1] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#pragma HLS ARRAY_PARTITION variable = count

cnt_lens:
    for (ap_uint<9> i = 0; i < codes; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<5> val = lens[i];
        count[val]++;
    }

    count[0] = 0;
    ap_uint<15> firstCode[15];
#pragma HLS ARRAY_PARTITION variable = firstCode
    ap_uint<16> code = 0;
firstCode:
    for (uint16_t i = 1; i <= c_maxbits; i++) {
#pragma HLS PIPELINE II = 1
        code = (code + count[i - 1]) << 1;
        codeOffsets[i - 1] = code;
        firstCode[i - 1] = code;
    }

    uint16_t blen = 0;
CodeGen:
    for (uint16_t i = 0; i < codes; i++) {
#pragma HLS PIPELINE II = 1
        blen = lens[i];
        if (blen != 0) {
            switch (blen) {
                case 1:
                    bl1Codes[firstCode[0]] = i;
                    break;
                case 2:
                    bl2Codes[firstCode[1]] = i;
                    break;
                case 3:
                    bl3Codes[firstCode[2]] = i;
                    break;
                case 4:
                    bl4Codes[firstCode[3]] = i;
                    break;
                case 5:
                    bl5Codes[firstCode[4]] = i;
                    break;
                case 6:
                    bl6Codes[firstCode[5]] = i;
                    break;
                case 7:
                    bl7Codes[firstCode[6]] = i;
                    break;
                case 8:
                    bl8Codes[firstCode[7]] = i;
                    break;
                case 9:
                    bl9Codes[ap_uint<8>(firstCode[8])] = i;
                    break;
                case 10:
                    bl10Codes[ap_uint<8>(firstCode[9])] = i;
                    break;
                case 11:
                    bl11Codes[ap_uint<8>(firstCode[10])] = i;
                    break;
                case 12:
                    bl12Codes[ap_uint<8>(firstCode[11])] = i;
                    break;
                case 13:
                    bl13Codes[ap_uint<8>(firstCode[12])] = i;
                    break;
                case 14:
                    bl14Codes[ap_uint<8>(firstCode[13])] = i;
                    break;
                case 15:
                    bl15Codes[ap_uint<8>(firstCode[14])] = i;
                    break;
            }
            firstCode[blen - 1]++;
        }
    }
}
}

/**
 * @brief This module is ZLIB/GZIP Fixed, Dynamic and Stored block supported
 * decoder. It takes ZLIB/GZIP Huffman encoded data as input and generates
 * decoded data in LZ77 format (Literal, Length, Offset).
 *
 * @tparam DECODER Fixed, Full, Dynamic huffman block support
 * @tparam ByteGenLoopII core bytegenerator loop initiation interval
 * @tparam USE_GZIP switch that controls GZIP/ZLIB header processing
 *
 *
 * @param inStream input bit packed data
 * @param outStream output lz77 compressed output in the form of 32bit packets
 * (Literals, Match Length, Distances)
 * @param input_size input data size
 */
template <eHuffmanType DECODER = FULL>
void huffmanDecoderLL(hls::stream<ap_uint<16> >& inStream,
                      hls::stream<bool>& inEos,
                      hls::stream<ap_uint<16> >& outStream) {
    bitBufferTypeLL bitbuffer = 0;
    ap_uint<6> bits_cntr = 0;
    bool isMultipleFiles = false;
    bool done = false;
    details::loadBitStreamLL(bitbuffer, bits_cntr, inStream, inEos, done);
huffmanDecoder_label0:
    while (done == false) {
        uint8_t current_op = 0;
        uint8_t current_bits = 0;
        uint16_t current_val = 0;
        ap_uint<32> current_table_val;

        uint8_t len = 0;

        const ap_uint<5> order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

        bool dynamic_last = 0;
        ap_uint<9> dynamic_nlen = 0;
        ap_uint<9> dynamic_ndist = 0;
        ap_uint<5> dynamic_ncode = 0;
        ap_uint<9> dynamic_curInSize = 0;
        uint16_t dynamic_lens[512];

        uint8_t copy = 0;

        bool blocks_processed = false;

        const uint16_t c_tcodesize = 2048;

        uint32_t array_codes[512];
        uint32_t array_codes_extra[512];
        uint32_t array_codes_dist[512];
        uint32_t array_codes_dist_extra[512];

        bool isGzip = false;
        // New huffman code
        ap_uint<16> codeOffsets[2][15];
#pragma HLS ARRAY_PARTITION variable = codeOffsets dim = 1 complete
        ap_uint<9> bl1Code[2][2];
        ap_uint<9> bl2Code[2][4];
        ap_uint<9> bl3Code[2][8];
        ap_uint<9> bl4Code[2][16];
        ap_uint<9> bl5Code[2][32];
        ap_uint<9> bl6Code[2][64];
        ap_uint<9> bl7Code[2][128];
        ap_uint<9> bl8Code[2][256];
        ap_uint<9> bl9Code[2][256];
        ap_uint<9> bl10Code[2][256];
        ap_uint<9> bl11Code[2][256];
        ap_uint<9> bl12Code[2][256];
        ap_uint<9> bl13Code[2][256];
        ap_uint<9> bl14Code[2][256];
        ap_uint<9> bl15Code[2][256];
#pragma HLS BIND_STORAGE variable = bl1Code type = ram_1p impl = lutram
#pragma HLS ARRAY_PARTITION variable = bl1Code complete dim = 1
#pragma HLS BIND_STORAGE variable = bl2Code type = ram_1p impl = lutram
#pragma HLS ARRAY_PARTITION variable = bl2Code complete dim = 1
#pragma HLS BIND_STORAGE variable = bl3Code type = ram_1p impl = lutram
#pragma HLS ARRAY_PARTITION variable = bl3Code complete dim = 1
#pragma HLS BIND_STORAGE variable = bl4Code type = ram_1p impl = lutram
#pragma HLS ARRAY_PARTITION variable = bl4Code complete dim = 1
#pragma HLS BIND_STORAGE variable = bl5Code type = ram_1p impl = lutram
#pragma HLS ARRAY_PARTITION variable = bl5Code complete dim = 1
#pragma HLS BIND_STORAGE variable = bl6Code type = ram_1p impl = lutram
#pragma HLS ARRAY_PARTITION variable = bl6Code complete dim = 1
#pragma HLS BIND_STORAGE variable = bl7Code type = ram_1p impl = lutram
#pragma HLS ARRAY_PARTITION variable = bl7Code complete dim = 1
#pragma HLS BIND_STORAGE variable = bl8Code type = ram_1p impl = lutram
#pragma HLS ARRAY_PARTITION variable = bl8Code complete dim = 1
#pragma HLS BIND_STORAGE variable = bl9Code type = ram_1p impl = lutram
#pragma HLS ARRAY_PARTITION variable = bl9Code complete dim = 1
#pragma HLS BIND_STORAGE variable = bl10Code type = ram_1p impl = lutram
#pragma HLS ARRAY_PARTITION variable = bl10Code complete dim = 1
#pragma HLS BIND_STORAGE variable = bl11Code type = ram_1p impl = lutram
#pragma HLS ARRAY_PARTITION variable = bl11Code complete dim = 1
#pragma HLS BIND_STORAGE variable = bl12Code type = ram_1p impl = lutram
#pragma HLS ARRAY_PARTITION variable = bl12Code complete dim = 1
#pragma HLS BIND_STORAGE variable = bl13Code type = ram_1p impl = lutram
#pragma HLS ARRAY_PARTITION variable = bl13Code complete dim = 1
#pragma HLS BIND_STORAGE variable = bl14Code type = ram_1p impl = lutram
#pragma HLS ARRAY_PARTITION variable = bl14Code complete dim = 1
#pragma HLS BIND_STORAGE variable = bl15Code type = ram_1p impl = lutram
#pragma HLS ARRAY_PARTITION variable = bl15Code complete dim = 1
        const bool include_fixed_block = (DECODER == FIXED || DECODER == FULL);
        const bool include_dynamic_block = (DECODER == DYNAMIC || DECODER == FULL);
        bool skip_fname = false;
        details::loadBitStreamLL(bitbuffer, bits_cntr, inStream, inEos, done);
        uint16_t magic_number = bitbuffer & 0xFFFF;
        details::discardBitStreamLL(bitbuffer, bits_cntr, (ap_uint<6>)16);
        if (magic_number == 0x8b1f) {
            // GZIP Header Processing
            // Deflate mode & file name flag
            isGzip = true;
            isMultipleFiles = false;
            details::loadBitStreamLL(bitbuffer, bits_cntr, inStream, inEos, done);
            uint16_t lcl_tmp = bitbuffer & 0xFFFF;
            details::discardBitStreamLL(bitbuffer, bits_cntr, (ap_uint<6>)16);

            // Check for fnam content
            skip_fname = (lcl_tmp >> 8) ? true : false;
            details::loadBitStreamLL(bitbuffer, bits_cntr, inStream, inEos, done);
            details::discardBitStreamLL(bitbuffer, bits_cntr, (ap_uint<6>)16);

            // MTIME - must
            // XFL (2 for high compress, 4 fast)
            // OS code (3Unix, 0Fat)
            details::loadBitStreamLL(bitbuffer, bits_cntr, inStream, inEos, done);
            details::discardBitStreamLL(bitbuffer, bits_cntr, (ap_uint<6>)16);
            details::loadBitStreamLL(bitbuffer, bits_cntr, inStream, inEos, done);

            // MTIME - must
            // XFL (2 for high compress, 4 fast)
            // OS code (3Unix, 0Fat)
            details::discardBitStreamLL(bitbuffer, bits_cntr, (ap_uint<6>)16);
            // If FLG is set to zero by using -n
            if (skip_fname) {
            // Read file name
            read_fname:
                do {
#pragma HLS PIPELINE II = 1
                    if (bits_cntr < 16 && (done == false)) {
                        uint16_t tmp_data = inStream.read();
                        bitbuffer += (bitBufferTypeLL)(tmp_data << bits_cntr);
                        done = inEos.read();
                        bits_cntr += 16;
                    }
                    lcl_tmp = bitbuffer & 0xFF;
                    details::discardBitStreamLL(bitbuffer, bits_cntr, (ap_uint<6>)8);
                } while (lcl_tmp != 0);
            }
        } else if ((magic_number & 0x00FF) != 0x0078) {
            blocks_processed = true;
        }

        if (isMultipleFiles) blocks_processed = true;
        while (!blocks_processed && (done == false)) {
            // one block per iteration
            // check if the following block is stored block or compressed block
            isMultipleFiles = true;
            details::loadBitStreamLL(bitbuffer, bits_cntr, inStream, inEos, done);
            // read the last bit in bitbuffer to check if this is last block
            dynamic_last = bitbuffer & 1;
            bitbuffer >>= 1; // dump the bit read
            ap_uint<2> cb_type = (uint8_t)(bitbuffer)&3;
            bitbuffer >>= 2;
            bits_cntr -= 3; // previously dumped 1 bit + current dumped 2 bits

            if (cb_type == 0) { // stored block
                bitbuffer >>= bits_cntr & 7;
                bits_cntr -= bits_cntr & 7;

                details::loadBitStreamLL(bitbuffer, bits_cntr, inStream, inEos, done);
                uint16_t store_length = bitbuffer & 0xffff;
                details::discardBitStreamLL(bitbuffer, bits_cntr, (ap_uint<6>)16);
                details::loadBitStreamLL(bitbuffer, bits_cntr, inStream, inEos, done);
                details::discardBitStreamLL(bitbuffer, bits_cntr, (ap_uint<6>)16);

                if (DECODER == FULL) {
                    ap_uint<16> tmpVal = 0;
                strd_blk_cpy:
                    for (uint16_t i = 0; i < store_length; i++) {
#pragma HLS PIPELINE II = 1
                        if (bits_cntr < 8 && (done == false)) {
                            uint16_t tmp_dt = (uint16_t)inStream.read();
                            bitbuffer += (bitBufferTypeLL)(tmp_dt) << bits_cntr;
                            done = inEos.read();
                            bits_cntr += 16;
                        }
                        tmpVal.range(7, 0) = bitbuffer;
                        tmpVal.range(15, 8) = 0xF0;
                        outStream << tmpVal;
                        details::discardBitStreamLL(bitbuffer, bits_cntr, (ap_uint<6>)8);
                    }
                }

            } else if (cb_type == 2) {       // dynamic huffman compressed block
                if (include_dynamic_block) { // compile if decoder should be dynamic/full
                                             // Read 14 bits HLIT(5-bits), HDIST(5-bits) and HCLEN(4-bits)
                    details::loadBitStreamLL(bitbuffer, bits_cntr, inStream, inEos, done);
                    dynamic_nlen = (bitbuffer & ((1 << 5) - 1)) + 257; // Max 288
                    details::discardBitStreamLL(bitbuffer, bits_cntr, (ap_uint<6>)5);

                    dynamic_ndist = (bitbuffer & ((1 << 5) - 1)) + 1; // Max 30
                    details::discardBitStreamLL(bitbuffer, bits_cntr, (ap_uint<6>)5);

                    dynamic_ncode = (bitbuffer & ((1 << 4) - 1)) + 4; // Max 19
                    details::discardBitStreamLL(bitbuffer, bits_cntr, (ap_uint<6>)4);

                    dynamic_curInSize = 0;

                dyn_len_bits:
                    while (dynamic_curInSize < dynamic_ncode) {
#pragma HLS PIPELINE II = 1
                        if ((bits_cntr < 16) && (done == false)) {
                            uint16_t tmp_data = inStream.read();
                            bitbuffer += (bitBufferTypeLL)(tmp_data << bits_cntr);
                            done = inEos.read();
                            bits_cntr += 16;
                        }
                        dynamic_lens[order[dynamic_curInSize++]] = (uint16_t)(bitbuffer & ((1 << 3) - 1));
                        details::discardBitStreamLL(bitbuffer, bits_cntr, (ap_uint<6>)3);
                    }

                    while (dynamic_curInSize < 19) dynamic_lens[order[dynamic_curInSize++]] = 0;
                    details::code_generator_array_dyn_new(1, dynamic_lens, 19, codeOffsets[0], bl1Code[0], bl2Code[0],
                                                          bl3Code[0], bl4Code[0], bl5Code[0], bl6Code[0], bl7Code[0],
                                                          bl8Code[0], bl9Code[0], bl10Code[0], bl11Code[0], bl12Code[0],
                                                          bl13Code[0], bl14Code[0], bl15Code[0]);

                    details::byteGen(bitbuffer, bits_cntr, codeOffsets[0], bl1Code[0], bl2Code[0], bl3Code[0],
                                     bl4Code[0], bl5Code[0], bl6Code[0], bl7Code[0], dynamic_lens, inEos, inStream,
                                     dynamic_nlen, dynamic_ndist, done);

                    details::code_generator_array_dyn_new(2, dynamic_lens, dynamic_nlen, codeOffsets[0], bl1Code[0],
                                                          bl2Code[0], bl3Code[0], bl4Code[0], bl5Code[0], bl6Code[0],
                                                          bl7Code[0], bl8Code[0], bl9Code[0], bl10Code[0], bl11Code[0],
                                                          bl12Code[0], bl13Code[0], bl14Code[0], bl15Code[0]);

                    details::code_generator_array_dyn_new(
                        3, dynamic_lens + dynamic_nlen, dynamic_ndist, codeOffsets[1], bl1Code[1], bl2Code[1],
                        bl3Code[1], bl4Code[1], bl5Code[1], bl6Code[1], bl7Code[1], bl8Code[1], bl9Code[1], bl10Code[1],
                        bl11Code[1], bl12Code[1], bl13Code[1], bl14Code[1], bl15Code[1]);
                    // BYTEGEN dynamic state
                    // ********************************
                    //  Create Packets Below
                    //  [LIT|ML|DIST|DIST] --> 32 Bit
                    //  Read data from inStream - 8bits
                    //  at a time. Decode the literals,
                    //  ML, Distances based on tables
                    // ********************************

                    // Read from inStream
                    details::loadBitStreamLL(bitbuffer, bits_cntr, inStream, inEos, done);
                    uint8_t ignoreValue = (dynamic_last) ? 0xFD : 0XFE;
                    uint8_t ret = details::huffmanBytegenLL(bitbuffer, bits_cntr, outStream, inEos, inStream,
                                                            codeOffsets, bl1Code, bl2Code, bl3Code, bl4Code, bl5Code,
                                                            bl6Code, bl7Code, bl8Code, bl9Code, bl10Code, bl11Code,
                                                            bl12Code, bl13Code, bl14Code, bl15Code, done, ignoreValue);

                } else {
                    blocks_processed = true;
                }
            } else if (cb_type == 1) {     // fixed huffman compressed block
                if (include_fixed_block) { // compile if decoder should be fixed/full
                    // ********************************

                    details::loadBitStreamLL(bitbuffer, bits_cntr, inStream, inEos, done);

                    // BitCodes
                    ap_uint<15> bit7 = 0;
                    ap_uint<15> bit8 = 48;
                    ap_uint<15> bit9 = 400;
                    for (ap_uint<9> i = 0; i < 288; i++) {
#pragma HLS PIPELINE II = 1
                        // codeoffsets
                        if (i == 4) {
                            codeOffsets[0][i] = -1;
                            codeOffsets[1][i] = 0;
                        } else if (i == 6) {
                            codeOffsets[0][i] = 0;
                            codeOffsets[1][i] = -1;
                        } else if (i == 7) {
                            codeOffsets[0][i] = 48;
                            codeOffsets[1][i] = -1;
                        } else if (i == 8) {
                            codeOffsets[0][i] = 400;
                            codeOffsets[1][i] = -1;
                        } else if (i < 15) {
                            codeOffsets[0][i] = -1;
                            codeOffsets[1][i] = -1;
                        }

                        // bitcodes
                        if (i < 32) { // fill distance codes
                            bl5Code[1][i] = i;
                        }
                        if (i < 144) { // literal upto 144
                            bl8Code[0][bit8] = i;
                            bit8 += 1;
                        } else if (i < 256) { // literal upto 256
                            bl9Code[0][bit9.range(7, 0)] = i;
                            bit9 += 1;
                        } else if (i < 280) {
                            bl7Code[0][bit7] = i;
                            bit7 += 1;
                        } else {
                            bl8Code[0][bit8] = i;
                            bit8 += 1;
                        }
                    }
                    uint8_t ignoreValue = (dynamic_last) ? 0xFD : 0XFE;
                    uint8_t ret = details::huffmanBytegenLL(bitbuffer, bits_cntr, outStream, inEos, inStream,
                                                            codeOffsets, bl1Code, bl2Code, bl3Code, bl4Code, bl5Code,
                                                            bl6Code, bl7Code, bl8Code, bl9Code, bl10Code, bl11Code,
                                                            bl12Code, bl13Code, bl14Code, bl15Code, done, ignoreValue);
                } else {
                    blocks_processed = true;
                }
            } else {
                blocks_processed = true;
            }
            if (dynamic_last) blocks_processed = true;
        } // While end
        // Checksum 4Bytes
        if (isGzip) {
            details::loadBitStreamLL(bitbuffer, bits_cntr, inStream, inEos, done);
            details::discardBitStreamLL(bitbuffer, bits_cntr, (ap_uint<6>)16);
            details::loadBitStreamLL(bitbuffer, bits_cntr, inStream, inEos, done);
            details::discardBitStreamLL(bitbuffer, bits_cntr, (ap_uint<6>)16);

            details::loadBitStreamLL(bitbuffer, bits_cntr, inStream, inEos, done);
            details::discardBitStreamLL(bitbuffer, bits_cntr, (ap_uint<6>)16);
            details::loadBitStreamLL(bitbuffer, bits_cntr, inStream, inEos, done);
            details::discardBitStreamLL(bitbuffer, bits_cntr, (ap_uint<6>)16);
            ap_uint<6> leftOverBits = bits_cntr % 8;
            details::loadBitStreamLL(bitbuffer, bits_cntr, inStream, inEos, done);
            details::discardBitStreamLL(bitbuffer, bits_cntr, (ap_uint<6>)leftOverBits);
        } else {
        consumeLeftOverData:
            while (done == false) {
                inStream.read();
                done = inEos.read();
            }
        }
    }
    outStream << 0xFFFF; // Adding Dummy Data for last end of stream case
}

/**
 * @brief This module is ZLIB/GZIP Fixed, Dynamic and Stored block supported
 * decoder. It takes ZLIB/GZIP Huffman encoded data as input and generates
 * decoded data in LZ77 format (Literal, Length, Offset).
 *
 * @tparam DECODER Fixed, Full, Dynamic huffman block support
 * @tparam ByteGenLoopII core bytegenerator loop initiation interval
 * @tparam USE_GZIP switch that controls GZIP/ZLIB header processing
 *
 *
 * @param inStream input bit packed data
 * @param outStream output lz77 compressed output in the form of 32bit packets
 * (Literals, Match Length, Distances)
 * @param input_size input data size
 */
template <eHuffmanType DECODER = FULL>
void huffmanDecoder(hls::stream<ap_uint<16> >& inStream,
                    hls::stream<bool>& inEos,
                    hls::stream<ap_uint<17> >& outStream) {
    bitBufferType bitbuffer = 0;
    ap_uint<6> bits_cntr = 0;
    bool isMultipleFiles = false;
    bool done = false;
    details::loadBitStream(bitbuffer, bits_cntr, inStream, inEos, done);
    while (done == false) {
        uint8_t current_op = 0;
        uint8_t current_bits = 0;
        uint16_t current_val = 0;
        ap_uint<32> current_table_val;

        uint8_t len = 0;

        const ap_uint<5> order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

        bool dynamic_last = 0;
        ap_uint<9> dynamic_nlen = 0;
        ap_uint<9> dynamic_ndist = 0;
        ap_uint<5> dynamic_ncode = 0;
        ap_uint<9> dynamic_curInSize = 0;
        uint16_t dynamic_lens[512];

        uint8_t copy = 0;

        bool blocks_processed = false;

        const uint16_t c_tcodesize = 2048;

        uint32_t array_codes[512];
        uint32_t array_codes_extra[512];
        uint32_t array_codes_dist[512];
        uint32_t array_codes_dist_extra[512];

        bool isGzip = false;

        const bool include_fixed_block = (DECODER == FIXED || DECODER == FULL);
        const bool include_dynamic_block = (DECODER == DYNAMIC || DECODER == FULL);
        bool skip_fname = false;
        details::loadBitStream(bitbuffer, bits_cntr, inStream, inEos, done);
        uint16_t magic_number = bitbuffer & 0xFFFF;
        details::discardBitStream(bitbuffer, bits_cntr, (ap_uint<6>)16);
        if (magic_number == 0x8b1f) {
            // GZIP Header Processing
            // Deflate mode & file name flag
            isGzip = true;
            isMultipleFiles = false;
            details::loadBitStream(bitbuffer, bits_cntr, inStream, inEos, done);
            uint16_t lcl_tmp = bitbuffer & 0xFFFF;
            details::discardBitStream(bitbuffer, bits_cntr, (ap_uint<6>)16);

            // Check for fnam content
            skip_fname = (lcl_tmp >> 8) ? true : false;
            details::loadBitStream(bitbuffer, bits_cntr, inStream, inEos, done);

            // MTIME - must
            // XFL (2 for high compress, 4 fast)
            // OS code (3Unix, 0Fat)
            details::discardBitStream(bitbuffer, bits_cntr, (ap_uint<6>)32);
            details::loadBitStream(bitbuffer, bits_cntr, inStream, inEos, done);

            // MTIME - must
            // XFL (2 for high compress, 4 fast)
            // OS code (3Unix, 0Fat)
            details::discardBitStream(bitbuffer, bits_cntr, (ap_uint<6>)16);
            // If FLG is set to zero by using -n
            if (skip_fname) {
            // Read file name
            read_fname:
                do {
#pragma HLS PIPELINE II = 1
                    if (bits_cntr < 16 && (done == false)) {
                        uint16_t tmp_data = inStream.read();
                        bitbuffer += (bitBufferType)(tmp_data << bits_cntr);
                        done = inEos.read();
                        bits_cntr += 16;
                    }
                    lcl_tmp = bitbuffer & 0xFF;
                    details::discardBitStream(bitbuffer, bits_cntr, (ap_uint<6>)8);
                } while (lcl_tmp != 0);
            }
        } else if ((magic_number & 0x00FF) != 0x0078) {
            blocks_processed = true;
        }

        if (isMultipleFiles) blocks_processed = true;
        while (!blocks_processed && (done == false)) {
            // one block per iteration
            // check if the following block is stored block or compressed block
            isMultipleFiles = true;
            details::loadBitStream(bitbuffer, bits_cntr, inStream, inEos, done);
            // read the last bit in bitbuffer to check if this is last block
            dynamic_last = bitbuffer & 1;
            bitbuffer >>= 1; // dump the bit read
            ap_uint<2> cb_type = (uint8_t)(bitbuffer)&3;
            bitbuffer >>= 2;
            bits_cntr -= 3; // previously dumped 1 bit + current dumped 2 bits

            if (cb_type == 0) { // stored block
                bitbuffer >>= bits_cntr & 7;
                bits_cntr -= bits_cntr & 7;

                details::loadBitStream(bitbuffer, bits_cntr, inStream, inEos, done);
                uint16_t store_length = bitbuffer & 0xffff;
                details::discardBitStream(bitbuffer, bits_cntr, (ap_uint<6>)32);

                if (DECODER == FULL) {
                    ap_uint<17> tmpVal = 0;
                strd_blk_cpy:
                    for (uint16_t i = 0; i < store_length; i++) {
#pragma HLS PIPELINE II = 1
                        if (bits_cntr < 8 && (done == false)) {
                            uint16_t tmp_dt = (uint16_t)inStream.read();
                            bitbuffer += (bitBufferType)(tmp_dt) << bits_cntr;
                            done = inEos.read();
                            bits_cntr += 16;
                        }
                        tmpVal.range(8, 1) = bitbuffer;
                        tmpVal.range(16, 9) = 0xFF;
                        tmpVal.range(0, 0) = 0;
                        outStream << tmpVal;
                        details::discardBitStream(bitbuffer, bits_cntr, (ap_uint<6>)8);
                    }
                }

            } else if (cb_type == 2) {       // dynamic huffman compressed block
                if (include_dynamic_block) { // compile if decoder should be dynamic/full
                                             // Read 14 bits HLIT(5-bits), HDIST(5-bits) and HCLEN(4-bits)
                    details::loadBitStream(bitbuffer, bits_cntr, inStream, inEos, done);
                    dynamic_nlen = (bitbuffer & ((1 << 5) - 1)) + 257; // Max 288
                    details::discardBitStream(bitbuffer, bits_cntr, (ap_uint<6>)5);

                    dynamic_ndist = (bitbuffer & ((1 << 5) - 1)) + 1; // Max 30
                    details::discardBitStream(bitbuffer, bits_cntr, (ap_uint<6>)5);

                    dynamic_ncode = (bitbuffer & ((1 << 4) - 1)) + 4; // Max 19
                    details::discardBitStream(bitbuffer, bits_cntr, (ap_uint<6>)4);

                    dynamic_curInSize = 0;

                dyn_len_bits:
                    while (dynamic_curInSize < dynamic_ncode) {
#pragma HLS PIPELINE II = 1
                        if ((bits_cntr < 16) && (done == false)) {
                            uint16_t tmp_data = inStream.read();
                            bitbuffer += (bitBufferType)(tmp_data << bits_cntr);
                            done = inEos.read();
                            bits_cntr += 16;
                        }
                        dynamic_lens[order[dynamic_curInSize++]] = (uint16_t)(bitbuffer & ((1 << 3) - 1));
                        details::discardBitStream(bitbuffer, bits_cntr, (ap_uint<6>)3);
                    }

                    while (dynamic_curInSize < 19) dynamic_lens[order[dynamic_curInSize++]] = 0;
                    details::code_generator_array_dyn(1, dynamic_lens, 19, array_codes, array_codes_extra, 7);

                    dynamic_curInSize = 0;
                    uint32_t dlenb_mask = ((1 << 7) - 1);

                    details::loadBitStream(bitbuffer, bits_cntr, inStream, inEos, done);
                    current_table_val = array_codes[(bitbuffer & dlenb_mask)];
                // Figure out codes for LIT/ML and DIST
                bitlen_gen:
                    while (dynamic_curInSize < dynamic_nlen + dynamic_ndist || (copy != 0)) {
#pragma HLS PIPELINE II = 1
                        //#pragma HLS dependence variable = dynamic_lens inter false

                        current_bits = current_table_val.range(23, 16);
                        current_op = current_table_val.range(24, 31);
                        current_val = current_table_val.range(15, 0);

                        if (current_val < 16 && (copy == 0)) {
                            bitbuffer >>= current_bits;
                            bits_cntr -= current_bits;
                            len = current_val;
                            copy = 1;
                        } else if (current_val == 16 && (copy == 0)) {
                            bitbuffer >>= current_bits;

                            if (dynamic_curInSize == 0) blocks_processed = true;

                            // len = dynamic_lens[dynamic_curInSize - 1];
                            copy = 3 + (bitbuffer & 3);      // use 2 bits
                            bitbuffer >>= 2;                 // dump 2 bits
                            bits_cntr -= (current_bits + 2); // update bits_cntr
                        } else if (current_val == 17 && (copy == 0)) {
                            bitbuffer >>= current_bits;
                            len = 0;
                            copy = 3 + (bitbuffer & 7); // use 3 bits
                            bitbuffer >>= 3;
                            bits_cntr -= (current_bits + 3);
                        } else if (copy == 0) {
                            bitbuffer >>= current_bits;
                            len = 0;
                            copy = 11 + (bitbuffer & ((1 << 7) - 1)); // use 7 bits
                            bitbuffer >>= 7;
                            bits_cntr -= (current_bits + 7);
                        }

                        // std::cout << "lens[" << dynamic_curInSize <<"] = " << (uint16_t)len << std::endl;
                        dynamic_lens[dynamic_curInSize++] = (uint16_t)len;
                        copy -= 1;
                        current_table_val = details::reg<ap_uint<32> >(array_codes[(bitbuffer & dlenb_mask)]);
                        if ((bits_cntr < 32) && (done == false)) {
                            uint16_t tmp_data = inStream.read();
                            bitbuffer += (bitBufferType)(tmp_data) << bits_cntr;
                            done = inEos.read();
                            bits_cntr += 16;
                        }
                    } // End of while
                    details::code_generator_array_dyn(2, dynamic_lens, dynamic_nlen, array_codes, array_codes_extra, 9);

                    details::code_generator_array_dyn(3, dynamic_lens + dynamic_nlen, dynamic_ndist, array_codes_dist,
                                                      array_codes_dist_extra, 9);
                    // BYTEGEN dynamic state
                    // ********************************
                    //  Create Packets Below
                    //  [LIT|ML|DIST|DIST] --> 32 Bit
                    //  Read data from inStream - 8bits
                    //  at a time. Decode the literals,
                    //  ML, Distances based on tables
                    // ********************************

                    // Read from inStream
                    details::loadBitStream(bitbuffer, bits_cntr, inStream, inEos, done);

                    uint8_t ret =
                        details::huffmanBytegen(bitbuffer, bits_cntr, outStream, inEos, inStream, array_codes,
                                                array_codes_extra, array_codes_dist, array_codes_dist_extra, done);

                    if (ret == details::blockStatus::FINISH) blocks_processed = true;

                } else {
                    blocks_processed = true;
                }
            } else if (cb_type == 1) {     // fixed huffman compressed block
                if (include_fixed_block) { // compile if decoder should be fixed/full
#include "fixed_codes.hpp"
                    // ********************************
                    //  Create Packets Below
                    //  [LIT|ML|DIST|DIST] --> 32 Bit
                    //  Read data from inStream - 8bits
                    //  at a time. Decode the literals,
                    //  ML, Distances based on tables
                    // ********************************
                    // Read from inStream
                    details::loadBitStream(bitbuffer, bits_cntr, inStream, inEos, done);

                    // ByteGeneration module
                    uint8_t ret =
                        details::huffmanBytegenStatic(bitbuffer, bits_cntr, outStream, inEos, inStream, fixed_litml_op,
                                                      fixed_litml_bits, fixed_litml_val, done);

                    if (ret == details::blockStatus::FINISH) blocks_processed = true;

                } else {
                    blocks_processed = true;
                }
            } else {
                blocks_processed = true;
            }
            if (dynamic_last) blocks_processed = true;
        } // While end
        // Checksum 4Bytes
        if (isGzip) {
            details::loadBitStream(bitbuffer, bits_cntr, inStream, inEos, done);
            ap_uint<6> leftOverBits = 32;
            details::discardBitStream(bitbuffer, bits_cntr, (ap_uint<6>)leftOverBits);

            details::loadBitStream(bitbuffer, bits_cntr, inStream, inEos, done);
            leftOverBits = 32 + (bits_cntr % 8);
            details::discardBitStream(bitbuffer, bits_cntr, (ap_uint<6>)leftOverBits);
        } else {
        consumeLeftOverData:
            while (done == false) {
                inStream.read();
                done = inEos.read();
            }
        }
    }
    outStream << 1; // Adding Dummy Data for last end of stream case
}

} // Compression
} // XF
#endif // _XFCOMPRESSION_HUFFMAN_DECODER_HPP_
