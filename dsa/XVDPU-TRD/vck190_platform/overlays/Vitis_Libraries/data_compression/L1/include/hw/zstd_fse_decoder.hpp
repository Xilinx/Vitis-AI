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

#ifndef _XFCOMPRESSION_ZSTD_FSE_DECODER_HPP_
#define _XFCOMPRESSION_ZSTD_FSE_DECODER_HPP_

/**
 * @file zstd_fse_decoder.hpp
 * @brief Header for modules used in ZSTD decompress kernel. This file contains
 * modules used for FSE and Huffman bitstream decoding and table generation.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include <stdint.h>
#include "hls_stream.h"
#include <ap_int.h>

#include "zstd_specs.hpp"

namespace xf {
namespace compression {
namespace details {

template <uint8_t PARALLEL_BYTE>
inline void sendData(hls::stream<ap_uint<8 * PARALLEL_BYTE> >& inStream,
                     ap_uint<8 * PARALLEL_BYTE * 2>& accRegister,
                     uint8_t& bytesInAcc,
                     hls::stream<ap_uint<8 * PARALLEL_BYTE> >& outStream,
                     uint64_t size2Write) {
#pragma HLS INLINE
    const uint16_t c_streamWidth = 8 * PARALLEL_BYTE;
    const uint16_t c_accRegWidthx3 = c_streamWidth * 3;
    // transfer data module
    ap_uint<c_accRegWidthx3> wbuf = accRegister;
    uint8_t bytesWritten = 0;
    uint8_t updBInAcc = bytesInAcc;

    uint8_t bitsInAcc = bytesInAcc * 8;
// write block data
send_data_main_loop:
    for (int i = 0; i < size2Write; i += PARALLEL_BYTE) {
#pragma HLS PIPELINE II = 1
        if (i < (const int)(size2Write - bytesInAcc)) {
            wbuf.range((bitsInAcc + c_streamWidth - 1), bitsInAcc) = inStream.read();
            updBInAcc += PARALLEL_BYTE;
        }
        ap_uint<c_streamWidth> tmpV = wbuf;
        outStream << tmpV;

        if (i > (const int)(size2Write - PARALLEL_BYTE)) {
            bytesWritten = size2Write - i;
            break;
        }
        wbuf >>= c_streamWidth;
        updBInAcc -= PARALLEL_BYTE;
    }
    accRegister = (wbuf >> (bytesWritten * 8));
    bytesInAcc = updBInAcc - bytesWritten;
}

template <uint8_t PARALLEL_BYTE>
int generateFSETable(hls::stream<uint32_t>& fseTableStream,
                     hls::stream<ap_uint<8 * PARALLEL_BYTE> >& inStream,
                     ap_uint<8 * PARALLEL_BYTE * 2>& fseAcc,
                     uint8_t& bytesInAcc,
                     uint8_t& tableLog,
                     uint16_t maxChar,
                     xfSymbolCompMode_t fseMode,
                     xfSymbolCompMode_t prevFseMode,
                     const int16_t* defDistTable,
                     int16_t* prevDistribution) {
    const uint16_t c_streamWidth = 8 * PARALLEL_BYTE;
    const uint16_t c_accRegWidth = c_streamWidth * 2;
    uint32_t bytesAvailable = bytesInAcc;
    uint8_t bitsAvailable = bytesInAcc * 8;
    uint32_t bitsConsumed = 0;
    int totalBytesConsumed = 0;
    uint32_t totalBitsConsumed = 0;
    int remaining = 0;
    int threshold = 0;
    bool previous0 = false;
    uint16_t charnum = 0;
    int16_t normalizedCounter[64];
    // initialize normalized counter
    for (int i = 0; i < 64; ++i) normalizedCounter[i] = 0;

    if (fseMode == FSE_COMPRESSED_MODE) {
        // read PARALLEL_BYTE bytes
        if (bytesAvailable < PARALLEL_BYTE) {
            fseAcc.range((bitsAvailable + c_streamWidth - 1), bitsAvailable) = inStream.read();
            bytesAvailable += PARALLEL_BYTE;
            bitsAvailable += c_streamWidth;
        }
        uint8_t accuracyLog = (fseAcc & 0xF) + 5;
        tableLog = accuracyLog;
        fseAcc >>= 4;
        bitsAvailable -= 4;
        totalBitsConsumed = 4;

        remaining = (1 << accuracyLog) + 1;
        threshold = 1 << accuracyLog;
        accuracyLog += 1;
    fsegenNormDistTable:
        while ((remaining > 1) & (charnum <= maxChar)) {
#pragma HLS PIPELINE II = 1
            bitsConsumed = 0;
            // read PARALLEL_BYTE bytes
            if (bytesAvailable < PARALLEL_BYTE - 1) {
                fseAcc.range((bitsAvailable + c_streamWidth - 1), bitsAvailable) = inStream.read();
                bytesAvailable += PARALLEL_BYTE;
                bitsAvailable += c_streamWidth;
            }
            if (previous0) {
                unsigned n0 = 0;
                if ((fseAcc & 0xFFFF) == 0xFFFF) {
                    n0 += 24;
                    bitsConsumed += 16;
                } else if ((fseAcc & 3) == 3) {
                    n0 += 3;
                    bitsConsumed += 2;
                } else {
                    n0 += fseAcc & 3;
                    bitsConsumed += 2;
                    previous0 = false;
                }
                charnum += n0;
            } else {
                int16_t max = (2 * threshold - 1) - remaining;
                int count;
                uint8_t c1 = 0;
                if ((fseAcc & (threshold - 1)) < max) {
                    c1 = 1;
                    count = fseAcc & (threshold - 1);
                    bitsConsumed += accuracyLog - 1;
                } else {
                    c1 = 2;
                    count = fseAcc & (2 * threshold - 1);
                    if (count >= threshold) count -= max;
                    bitsConsumed += accuracyLog;
                }

                --count;                                     /* extra accuracy */
                remaining -= ((count < 0) ? -count : count); /* -1 means +1 */
                normalizedCounter[charnum++] = count;
                previous0 = ((count == 0) ? 1 : 0);
            genDTableIntLoop:
                while (remaining < threshold) {
                    --accuracyLog;
                    threshold >>= 1;
                }
            }
            fseAcc >>= bitsConsumed;
            bitsAvailable -= bitsConsumed;
            totalBitsConsumed += bitsConsumed;

            bytesAvailable = bitsAvailable >> 3;
        }
        totalBytesConsumed += (totalBitsConsumed + 7) >> 3;
        bytesInAcc = bytesAvailable;
        // clear accumulator, so that it is byte aligned
        fseAcc >>= (bitsAvailable & 7);
        // copy to prevDistribution table
        for (int i = 0; i < 64; ++i) prevDistribution[i] = normalizedCounter[i];
    } else if (fseMode == PREDEFINED_MODE) { /*TODO: use fixed table directly*/
        for (int i = 0; i < maxChar + 1; ++i) {
            normalizedCounter[i] = defDistTable[i];
            prevDistribution[i] = defDistTable[i];
        }
        charnum = maxChar + 1;
    } else if (fseMode == RLE_MODE) {
        // next state for entire table is 0
        // accuracy log is 0
        // read PARALLEL_BYTE bytes
        if (bytesAvailable < 1) {
            fseAcc.range((bitsAvailable + c_streamWidth - 1), bitsAvailable) = inStream.read();
            bytesAvailable += PARALLEL_BYTE;
            bitsAvailable += c_streamWidth;
        }
        uint8_t symbol = (uint8_t)fseAcc;
        fseAcc >>= 8;
        bytesInAcc = bytesAvailable - 1;

        fseTableStream << 1; // tableSize
        fseTableStream << symbol;
        tableLog = 0;

        return 1;
    } else if (fseMode == REPEAT_MODE) {
        // check if previous mode was RLE
        fseTableStream << 0xFFFFFFFF; // all 1's to indicate use previous table
        return 0;
    } else {
        // else -> Error: invalid fse mode
        return 0;
    }

    // Table Generation
    const uint32_t tableSize = 1 << tableLog;
    uint32_t highThreshold = tableSize - 1;
    uint16_t symbolNext[64];
    int16_t symTable[513];
    // initialize table
    for (uint32_t i = 0; i < tableSize; ++i) symTable[i] = 0;

fse_gen_next_symbol_table:
    for (uint16_t i = 0; i < charnum; i++) {
#pragma HLS PIPELINE II = 1
        if (normalizedCounter[i] == -1) {
            symTable[highThreshold] = i; // symbol, assign lower 8-bits
            --highThreshold;
            symbolNext[i] = 1;
        } else {
            symbolNext[i] = normalizedCounter[i];
        }
    }

    uint32_t mask = tableSize - 1;
    const uint32_t step = (tableSize >> 1) + (tableSize >> 3) + 3;
    uint32_t pos = 0;

fse_gen_symbol_table_outer:
    for (uint16_t i = 0; i < charnum; ++i) {
    fse_gen_symbol_table:
        for (int j = 0; j < normalizedCounter[i];) {
#pragma HLS PIPELINE II = 1
            if (pos > highThreshold) {
                pos = (pos + step) & mask;
            } else {
                symTable[pos] = i;
                pos = (pos + step) & mask;
                ++j;
            }
        }
    }
    // FSE table
    fseTableStream << tableSize;
gen_fse_final_table:
    for (uint32_t i = 0; i < tableSize; i++) {
#pragma HLS PIPELINE II = 1
        uint8_t symbol = (uint8_t)symTable[i];
        uint16_t nextState = symbolNext[symbol]++;
        uint16_t nBits = (uint8_t)(tableLog - (31 - __builtin_clz(nextState)));
        uint32_t tblVal = ((uint32_t)((nextState << nBits) - tableSize) << 16) + ((uint32_t)nBits << 8) + symbol;
        fseTableStream << tblVal;
    }

    return totalBytesConsumed;
}

template <uint8_t PARALLEL_BYTE, uint8_t BLOCK_SIZE_KB, uint8_t BSWIDTH>
void fseStreamStates(uint32_t* litFSETable,
                     uint32_t* oftFSETable,
                     uint32_t* mlnFSETable,
                     ap_uint<BSWIDTH>* bitStream,
                     int byteIndx,
                     uint8_t lastByteValidBits,
                     uint32_t seqCnt,
                     uint8_t* accuracyLog,
                     hls::stream<uint8_t>& symbolStream,
                     hls::stream<ap_uint<32> >& bsWordStream,
                     hls::stream<ap_uint<5> >& extraBitStream) {
    // fetch fse states from fse sequence bitstream and stream out for further processing
    const uint16_t c_streamWidth = 8 * PARALLEL_BYTE;
    const uint16_t c_accRegWidth = c_streamWidth * 2;
    const uint8_t c_bsBytes = BSWIDTH / 8;

    uint16_t fseStateLL, fseStateOF, fseStateML; // literal_length, offset, match_length states
    FseBSState bsStateLL, bsStateOF, bsStateML;  // offset, match_length and literal_length
    ap_uint<c_accRegWidth> acchbs;
    uint8_t bitsInAcc = lastByteValidBits;
    uint8_t bitsToRead = 0;
    uint8_t bytesToRead = 0;

    acchbs.range(BSWIDTH - 1, 0) = bitStream[byteIndx--];
    uint8_t byte_0 = acchbs.range(bitsInAcc - 1, bitsInAcc - 8);
// find valid last bit, bitstream read in reverse order
fsedseq_skip_zero:
    for (uint8_t i = 7; i >= 0; --i) {
#pragma HLS UNROLL
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 7
        if ((byte_0 & (1 << i)) > 0) {
            --bitsInAcc;
            break;
        }
        --bitsInAcc; // discount higher bits which are zero
    }

fsedseq_fill_acc:
    while ((bitsInAcc + BSWIDTH < c_accRegWidth) && (byteIndx > -1)) {
#pragma HLS PIPELINE II = 1
        acchbs <<= BSWIDTH;
        acchbs.range(BSWIDTH - 1, 0) = bitStream[byteIndx--];
        bitsInAcc += BSWIDTH;
    }
    // Read literal_length, offset and match_length states from input stream
    // get *accuracyLog bits from higher position in accuracyLog, mask out higher scrap bits
    uint64_t mskLL = ((1 << accuracyLog[0]) - 1);
    fseStateLL = ((acchbs >> (bitsInAcc - accuracyLog[0])) & mskLL);
    uint64_t mskOF = ((1 << accuracyLog[1]) - 1);
    fseStateOF = ((acchbs >> (bitsInAcc - (accuracyLog[0] + accuracyLog[1]))) & mskOF);
    uint64_t mskML = ((1 << accuracyLog[2]) - 1);
    fseStateML = ((acchbs >> (bitsInAcc - (accuracyLog[0] + accuracyLog[1] + accuracyLog[2]))) & mskML);

    bitsInAcc -= (accuracyLog[0] + accuracyLog[1] + accuracyLog[2]);

    enum FSEDState_t { LITLEN = 0, MATLEN, OFFSET, NEXTSTATE };
    FSEDState_t smState = OFFSET;
    uint8_t bitCntLML, bitCntLMO;
    uint32_t bswLL, bswML, bswOF;

decode_sequence_bitStream:
    while (seqCnt) {
#pragma HLS PIPELINE II = 1
        // read data to bitstream if necessary
        if (bitsInAcc < c_streamWidth && byteIndx > -1) {
            auto tmp = bitStream[byteIndx--];
            acchbs = (acchbs << BSWIDTH) + tmp;
            bitsInAcc += BSWIDTH;
        }
        uint32_t stateVal;
        uint8_t extraBits;
        // get state values and stream
        // stream fse metadata to decoding unit
        if (smState == LITLEN) {
            bsWordStream << bswOF;

            stateVal = litFSETable[fseStateLL]; // literal_length
            bsStateLL.symbol = stateVal & 0x000000FF;
            bsStateLL.nextState = (stateVal >> 16) & 0x0000FFFF;
            bsStateLL.bitCount = (stateVal >> 8) & 0x000000FF;

            extraBits = c_extraBitsLL[bsStateLL.symbol]; // max 16-bits
            bitsInAcc -= extraBits;
            symbolStream << bsStateLL.symbol;
            bswLL = acchbs >> bitsInAcc;
            extraBitStream << extraBits;

            bitCntLML += bsStateLL.bitCount;
            bitCntLMO += bsStateLL.bitCount;
            smState = NEXTSTATE;
        } else if (smState == MATLEN) {
            stateVal = mlnFSETable[fseStateML]; // match_length
            bsStateML.symbol = stateVal & 0x000000FF;
            bsStateML.nextState = (stateVal >> 16) & 0x0000FFFF;
            bsStateML.bitCount = (stateVal >> 8) & 0x000000FF;

            extraBits = c_extraBitsML[bsStateML.symbol]; // max 16-bits
            bitsInAcc -= extraBits;
            symbolStream << bsStateML.symbol;
            bswML = acchbs >> bitsInAcc;
            extraBitStream << extraBits;

            bitCntLML = bsStateML.bitCount;
            bitCntLMO += bsStateML.bitCount;
            smState = LITLEN;
        } else if (smState == OFFSET) {
            // get the state codes for offset, match_length and literal_length
            stateVal = oftFSETable[fseStateOF]; // offset
            bsStateOF.symbol = stateVal & 0x000000FF;
            bsStateOF.nextState = (stateVal >> 16) & 0x0000FFFF;
            bsStateOF.bitCount = (stateVal >> 8) & 0x000000FF;

            extraBits = bsStateOF.symbol; // also represents extra bits to be read, max 31-bits
            bitsInAcc -= extraBits;
            symbolStream << bsStateOF.symbol;
            bswOF = acchbs >> bitsInAcc;

            bsWordStream << bswLL;
            bitCntLMO = bsStateOF.bitCount;
            smState = MATLEN;
        } else {
            bsWordStream << bswML;
            // update state for next sequence
            // read bits to get states for literal_length, match_length, offset
            // accumulator must contain these many bits can be max 26-bits
            mskLL = (((uint64_t)1 << bsStateLL.bitCount) - 1);
            fseStateLL = ((acchbs >> (bitsInAcc - bsStateLL.bitCount)) & mskLL);
            fseStateLL += bsStateLL.nextState;

            mskML = (((uint64_t)1 << bsStateML.bitCount) - 1);
            fseStateML = ((acchbs >> (bitsInAcc - bitCntLML)) & mskML);
            fseStateML += bsStateML.nextState;

            mskOF = (((uint64_t)1 << bsStateOF.bitCount) - 1);
            fseStateOF = ((acchbs >> (bitsInAcc - bitCntLMO)) & mskOF);
            fseStateOF += bsStateOF.nextState;

            bitsInAcc -= bitCntLMO;

            --seqCnt;
            smState = OFFSET;
        }
    }
    bsWordStream << bswLL;
}

template <int LMO_WIDTH>
void fseDecodeStates(hls::stream<uint8_t>& symbolStream,
                     hls::stream<ap_uint<32> >& bsWordStream,
                     hls::stream<ap_uint<5> >& extraBitStream,
                     uint32_t seqCnt,
                     uint32_t litCnt,
                     ap_uint<LMO_WIDTH>* prevOffsets,
                     hls::stream<ap_uint<LMO_WIDTH> >& litLenStream,
                     hls::stream<ap_uint<LMO_WIDTH> >& offsetStream,
                     hls::stream<ap_uint<LMO_WIDTH> >& matLenStream,
                     hls::stream<bool>& litlenValidStream) {
    // calculate literal length, match length and offset values, stream them for sequence execution
    enum FSEDecode_t { LITLEN = 0, MATLEN, OFFSET_CALC, OFFSET_WNU };
    FSEDecode_t sqdState = OFFSET_CALC;
    ap_uint<LMO_WIDTH> offsetVal;
    ap_uint<LMO_WIDTH> litLenCode;
    uint8_t ofi;
    bool checkLL = false;
    bsWordStream.read(); // dump first word
decode_sequence_codes:
    while (seqCnt) {
#pragma HLS PIPELINE II = 1
        if (sqdState == OFFSET_CALC) {
            // calculate offset and set prev offsets
            auto symbol = symbolStream.read();
            auto bsWord = bsWordStream.read();
            auto extBit = symbol;
            uint32_t extVal = bsWord & (((uint64_t)1 << extBit) - 1);
            offsetVal = (1 << symbol) + extVal;

            ofi = 3;
            if (offsetVal > 3) {
                offsetVal -= 3;
                checkLL = false;
            } else {
                checkLL = true;
            }
            sqdState = MATLEN;
        } else if (sqdState == MATLEN) {
            // calculate match length
            auto symbol = symbolStream.read();
            auto bsWord = bsWordStream.read();
            auto extBit = extraBitStream.read();
            uint16_t extVal = (bsWord & (((uint64_t)1 << extBit) - 1));
            ap_uint<LMO_WIDTH> matchLenCode = c_baseML[symbol] + extVal;
            matLenStream << matchLenCode;

            sqdState = LITLEN;
        } else if (sqdState == LITLEN) {
            // calculate literal length
            auto symbol = symbolStream.read();
            auto bsWord = bsWordStream.read();
            auto extBit = extraBitStream.read();
            uint16_t extVal = (bsWord & (((uint64_t)1 << extBit) - 1));
            litLenCode = c_baseLL[symbol] + extVal;
            litlenValidStream << 1;
            litLenStream << litLenCode;
            litCnt -= litLenCode;

            // update offset as per literal length
            if (checkLL) {
                // repeat offsets 1 - 3
                if (litLenCode == 0) {
                    if (offsetVal == 3) {
                        offsetVal = prevOffsets[0] - 1;
                        ofi = 2;
                    } else {
                        ofi = offsetVal;
                        offsetVal = prevOffsets[offsetVal];
                    }
                } else {
                    ofi = offsetVal - 1;
                    offsetVal = prevOffsets[offsetVal - 1];
                }
            }
            checkLL = false;
            sqdState = OFFSET_WNU;
        } else {
            // OFFSET_WNU: write offset and update previous offsets
            offsetStream << offsetVal;
            // shift previous offsets
            auto prev1 = prevOffsets[1];
            if (ofi > 1) {
                prevOffsets[2] = prev1;
            }
            if (ofi > 0) {
                prevOffsets[1] = prevOffsets[0];
                prevOffsets[0] = offsetVal;
            }

            sqdState = OFFSET_CALC;
            --seqCnt;
        }
    }
    if (litCnt > 0) {
        litlenValidStream << 1;
        litLenStream << litCnt;
        matLenStream << 0;
        offsetStream << 0;
    }
}

template <uint8_t PARALLEL_BYTE, uint8_t BLOCK_SIZE_KB, uint8_t BSWIDTH, uint8_t LMO_WIDTH>
void decodeSeqCore(uint32_t* litFSETable,
                   uint32_t* oftFSETable,
                   uint32_t* mlnFSETable,
                   ap_uint<BSWIDTH>* bitStream,
                   int bsIndx,
                   uint8_t lastByteValidBits,
                   uint32_t seqCnt,
                   uint32_t litCnt,
                   uint8_t* accuracyLog,
                   ap_uint<LMO_WIDTH>* prevOffsets,
                   hls::stream<ap_uint<LMO_WIDTH> >& litLenStream,
                   hls::stream<ap_uint<LMO_WIDTH> >& offsetStream,
                   hls::stream<ap_uint<LMO_WIDTH> >& matLenStream,
                   hls::stream<bool>& litlenValidStream) {
    // core module for decoding fse sequences
    const uint8_t c_intlStreamDepth = 16;
    // Internal streams
    hls::stream<uint8_t> symbolStream("symbolStream");
    hls::stream<ap_uint<32> > bsWordStream("bsWordStream");
    hls::stream<ap_uint<5> > extraBitStream("extraBitStream");

#pragma HLS STREAM variable = symbolStream depth = c_intlStreamDepth
#pragma HLS STREAM variable = bsWordStream depth = c_intlStreamDepth
#pragma HLS STREAM variable = extraBitStream depth = c_intlStreamDepth

#pragma HLS dataflow

    fseStreamStates<PARALLEL_BYTE, BLOCK_SIZE_KB, BSWIDTH>(litFSETable, oftFSETable, mlnFSETable, bitStream, bsIndx,
                                                           lastByteValidBits, seqCnt, accuracyLog, symbolStream,
                                                           bsWordStream, extraBitStream);

    fseDecodeStates<LMO_WIDTH>(symbolStream, bsWordStream, extraBitStream, seqCnt, litCnt, prevOffsets, litLenStream,
                               offsetStream, matLenStream, litlenValidStream);
}

template <uint8_t PARALLEL_BYTE, uint8_t BLOCK_SIZE_KB, uint8_t LMO_WIDTH>
inline void fseDecode(ap_uint<64> accword,
                      uint8_t bytesInAcc,
                      hls::stream<ap_uint<8 * PARALLEL_BYTE> >& inStream,
                      uint32_t* litFSETable,
                      uint32_t* oftFSETable,
                      uint32_t* mlnFSETable,
                      uint32_t seqCnt,
                      uint32_t litCount,
                      uint32_t remBlockSize,
                      uint8_t* accuracyLog,
                      ap_uint<LMO_WIDTH>* prevOffsets,
                      hls::stream<ap_uint<LMO_WIDTH> >& litLenStream,
                      hls::stream<ap_uint<LMO_WIDTH> >& offsetStream,
                      hls::stream<ap_uint<LMO_WIDTH> >& matLenStream,
                      hls::stream<bool>& litlenValidStream) {
    // decode fse encoded bitstream using prebuilt fse tables
    const uint16_t c_streamWidth = 8 * PARALLEL_BYTE;
    const uint16_t c_accRegWidth = c_streamWidth * 2;
    const uint16_t c_BSWidth = c_streamWidth;
    const uint8_t c_bsBytes = c_BSWidth / 8;

    ap_uint<c_BSWidth> bitStream[(BLOCK_SIZE_KB * 1024) / c_bsBytes];
#pragma HLS BIND_STORAGE variable = bitStream type = ram_t2p impl = uram

    uint8_t bitsInAcc = bytesInAcc * 8;

    // copy data from bitstream to buffer
    ap_uint<c_accRegWidth> bsbuff = accword;
    int bsIdx = 0;
    uint8_t bytesWritten = c_bsBytes;
    uint8_t updBInAcc = bytesInAcc;
// write block data
fsedseq_fill_bitstream:
    for (int i = 0; i < remBlockSize; i += c_bsBytes) { // TODO: biggest culprit as of now
#pragma HLS pipeline II = 1
        if (i < (const int)(remBlockSize - bytesInAcc)) {
            if (bytesInAcc < c_bsBytes) {
                bsbuff.range(bitsInAcc + c_streamWidth - 1, bitsInAcc) = inStream.read();
                updBInAcc += PARALLEL_BYTE;
            }
        }
        ap_uint<c_BSWidth> tmpv = bsbuff.range(c_BSWidth - 1, 0);
        bitStream[bsIdx++] = tmpv;

        if (i > (const int)(remBlockSize - c_bsBytes)) {
            bytesWritten = (remBlockSize - i);
            bsbuff >>= (bytesWritten * 8);
            updBInAcc -= bytesWritten;
        } else {
            bsbuff >>= c_BSWidth;
            updBInAcc -= c_bsBytes;
        }
    }

    decodeSeqCore<PARALLEL_BYTE, BLOCK_SIZE_KB, c_BSWidth, LMO_WIDTH>(
        litFSETable, oftFSETable, mlnFSETable, bitStream, bsIdx - 1, 8 * bytesWritten, seqCnt, litCount, accuracyLog,
        prevOffsets, litLenStream, offsetStream, matLenStream, litlenValidStream);
}

template <uint8_t PARALLEL_BYTE>
void fseDecodeHuffWeight(hls::stream<ap_uint<8 * PARALLEL_BYTE> >& inStream,
                         uint32_t remSize,
                         ap_uint<8 * PARALLEL_BYTE * 2>& accHuff,
                         uint8_t& bytesInAcc,
                         uint8_t accuracyLog,
                         uint32_t* fseTable,
                         uint8_t* weights,
                         uint16_t& weightCnt,
                         uint8_t& huffDecoderTableLog) {
    //#pragma HLS INLINE
    const uint16_t c_streamWidth = 8 * PARALLEL_BYTE;
    uint8_t bitsInAcc = bytesInAcc * 8;

    uint8_t bitStream[128];
    //#pragma HLS BIND_STORAGE variable = bitStream type = ram_t2p impl = bram
    int bsByteIndx = remSize - 1;

    // copy data from bitstream to buffer
    uint32_t itrCnt = 1 + ((remSize - 1) / PARALLEL_BYTE);
    uint32_t k = 0;
fseDecHF_read_input:
    for (uint32_t i = 0; i < itrCnt; ++i) {
        accHuff.range(bitsInAcc + c_streamWidth - 1, bitsInAcc) = inStream.read();
        bitsInAcc += c_streamWidth;
        for (uint8_t j = 0; j < PARALLEL_BYTE && k < remSize; ++j, ++k) {
#pragma HLS PIPELINE II = 1
            bitStream[k] = accHuff.range(7, 0);
            accHuff >>= 8;
            bitsInAcc -= 8;
        }
    }
    bytesInAcc = bitsInAcc >> 3;

    // decode FSE bitStream using fse table to get huffman weights
    // skip initial 0 bits and single 1 bit
    uint32_t accState;
    uint32_t codeIdx = 0;
    uint8_t bitsToRead = 0;
    bitsInAcc = 0;
    int32_t rembits = remSize * 8;

    uint8_t fseState[2];
    FseBSState bsState[2];
    // find beginning of the stream
    accState = bitStream[bsByteIndx--];
    bitsInAcc = 8;
    for (uint8_t i = 7; i >= 0; --i) {
#pragma HLS UNROLL
        if ((accState & (1 << i)) > 0) {
            --bitsInAcc;
            break;
        }
        --bitsInAcc; // discount higher bits which are zero
    }
    rembits -= (8 - bitsInAcc);

    // Read bits needed for first two states
    bitsToRead = accuracyLog * 2;
    if (bitsToRead > bitsInAcc) {
        uint8_t bytesToRead = 1 + ((bitsToRead - bitsInAcc - 1) / 8);
        for (uint8_t i = 0; i < bytesToRead; ++i) {
            uint8_t tmp = bitStream[bsByteIndx--];
            accState <<= 8;
            accState += tmp;
            bitsInAcc += 8;
        }
    }
    // Read initial state1 and state2
    // get readBits bits from higher position in accuracyLog, mask out higher scrap bits
    // read state 1
    bitsInAcc -= accuracyLog;
    uint64_t msk = ((1 << accuracyLog) - 1);
    fseState[0] = ((accState >> bitsInAcc) & msk);
    // read state 2
    bitsInAcc -= accuracyLog;
    msk = ((1 << accuracyLog) - 1);
    fseState[1] = ((accState >> bitsInAcc) & msk);
    rembits -= (accuracyLog * 2);

    bool stateIdx = 0; // 0 for even, 1 for odd
    bool overflow = false;
    uint32_t totalWeights = 0;
fse_decode_huff_weights:
    while (1) {
#pragma HLS PIPELINE II = 1
        // get the weight, bitCount and nextState
        uint32_t stateVal = fseTable[fseState[stateIdx]];
        uint8_t cw = (uint8_t)(stateVal & 0xFF);
        weights[codeIdx++] = cw;
        totalWeights += (1 << cw) >> 1;

        // overflow
        if (rembits < 0) break;

        // get other values
        bsState[stateIdx].nextState = (stateVal >> 16) & 0x0000FFFF;
        bsState[stateIdx].bitCount = (stateVal >> 8) & 0x000000FF;
        uint8_t bitsToRead = bsState[stateIdx].bitCount;
        if (bitsToRead > bitsInAcc) {
            uint8_t tmp = 0;
            if (bsByteIndx > -1) {
                // max 1 read is required, since accuracy log <= 6
                tmp = bitStream[bsByteIndx--];
            }
            accState <<= 8;
            accState += tmp;
            bitsInAcc += 8;
        }
        // get next fse state
        bitsInAcc -= bitsToRead;
        uint8_t msk = ((1 << bitsToRead) - 1);
        fseState[stateIdx] = ((accState >> bitsInAcc) & msk);
        fseState[stateIdx] += bsState[stateIdx].nextState;
        rembits -= bitsToRead;

        // switch state flow
        stateIdx = (stateIdx + 1) & 1; // 0 if 1, 1 if 0
    }
    huffDecoderTableLog = 1 + (31 - __builtin_clz(totalWeights));
    // add last weight
    uint16_t lw = (1 << huffDecoderTableLog) - totalWeights;
    weights[codeIdx++] = 1 + (31 - __builtin_clz(lw));
    weightCnt = codeIdx;
}

template <int MAX_CODELEN>
void huffGenLookupTable(uint8_t* weights, HuffmanTable* huffTable, uint8_t accuracyLog, uint16_t weightCnt) {
    // create huffman lookup table
    // regenerate huffman tree using literal bit-lengths
    typedef ap_uint<MAX_CODELEN + 1> LCL_Code_t;
    LCL_Code_t first_codeword[MAX_CODELEN + 1];
    ap_uint<32> bitlen_histogram[MAX_CODELEN + 1];
    ap_uint<4> bitlens[256];
#pragma HLS ARRAY_PARTITION variable = first_codeword complete
#pragma HLS ARRAY_PARTITION variable = bitlen_histogram complete

    uint16_t codes[256];
// initialize first_codeword and bitlength histogram
hflkpt_init_blen_hist:
    for (uint8_t i = 0; i < MAX_CODELEN + 1; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = MAX_CODELEN max = MAX_CODELEN
        bitlen_histogram[i] = 0;
    }
// read bit-lengths
hflkpt_fill_blen_hist:
    for (uint16_t i = 0; i < weightCnt; ++i) {
#pragma HLS PIPELINE II = 1
        // convert weight to bitlen
        uint8_t cblen = weights[i];
        bitlen_histogram[cblen]++;
        if (cblen > 0) cblen = (accuracyLog + 1 - cblen);
        bitlens[i] = cblen;
    }

    // generate first codes
    first_codeword[0] = bitlen_histogram[0];

    uint16_t nextCode = 0;
hflkpt_initial_codegen:
    for (uint8_t i = 1; i < accuracyLog + 1; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 8
        uint16_t cur = nextCode;
        nextCode += (bitlen_histogram[i] << (i - 1));
        first_codeword[i] = cur;
    }

hflkpt_codegen_outer:
    for (int i = 0; i < weightCnt; ++i) {
        uint32_t hfw = weights[i];
        const uint32_t len = (1 << hfw) >> 1;
        const auto fcw = first_codeword[hfw];
    hflkpt_codegen:
        for (uint16_t u = fcw; u < fcw + len; ++u) {
#pragma HLS PIPELINE II = 1
            huffTable[u].symbol = i;
            huffTable[u].bitlen = bitlens[i];
        }
        first_codeword[hfw] = fcw + len;
    }
}

template <int MAX_CODELEN>
void code_generator(uint8_t* weights,
                    ap_uint<MAX_CODELEN + 1>* codeOffsets,
                    ap_uint<8>* bl1Codes,
                    ap_uint<8>* bl2Codes,
                    ap_uint<8>* bl3Codes,
                    ap_uint<8>* bl4Codes,
                    ap_uint<8>* bl5Codes,
                    ap_uint<8>* bl6Codes,
                    ap_uint<8>* bl7Codes,
                    ap_uint<8>* bl8Codes,
                    ap_uint<8>* bl9Codes,
                    ap_uint<8>* bl10Codes,
                    ap_uint<8>* bl11Codes,
                    uint8_t accuracyLog,
                    uint16_t weightCnt) {
    // regenerate huffman tree using literal bit-lengths
    typedef ap_uint<MAX_CODELEN + 1> LCL_Code_t;
    LCL_Code_t first_codeword[MAX_CODELEN + 1];
    ap_uint<32> bitlen_histogram[MAX_CODELEN + 1];
    ap_uint<4> bitlens[256];
#pragma HLS ARRAY_PARTITION variable = first_codeword complete
#pragma HLS ARRAY_PARTITION variable = bitlen_histogram complete

    uint16_t codes[256];
// initialize first_codeword and bitlength histogram
hflkpt_init_blen_hist:
    for (uint8_t i = 0; i < MAX_CODELEN + 1; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = MAX_CODELEN max = MAX_CODELEN
        bitlen_histogram[i] = 0;
    }
// read bit-lengths
hflkpt_fill_blen_hist:
    for (uint16_t i = 0; i < weightCnt; ++i) {
#pragma HLS PIPELINE II = 1
        // convert weight to bitlen
        uint8_t cblen = weights[i];
        if (cblen > 0) cblen = (accuracyLog + 1 - cblen);
        bitlens[i] = cblen;
        bitlen_histogram[cblen]++;
    }

    // generate first codes
    first_codeword[0] = 0;
    codeOffsets[0] = 0;

    uint16_t nextCode = 0;
hflkpt_initial_codegen:
    for (int8_t i = accuracyLog - 1; i >= 0; --i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 11
        uint16_t cur = nextCode;
        nextCode += (bitlen_histogram[i + 1]);
        nextCode >>= 1;
        first_codeword[i] = cur;
        codeOffsets[i] = cur;
    }

    uint16_t blen = 0;
CodeGen:
    for (uint16_t i = 0; i < weightCnt; i++) {
#pragma HLS PIPELINE II = 1
        blen = bitlens[i];
        if (blen != 0) {
            switch (blen) {
                case 1:
                    bl1Codes[first_codeword[0]] = i;
                    break;
                case 2:
                    bl2Codes[first_codeword[1]] = i;
                    break;
                case 3:
                    bl3Codes[first_codeword[2]] = i;
                    break;
                case 4:
                    bl4Codes[first_codeword[3]] = i;
                    break;
                case 5:
                    bl5Codes[first_codeword[4]] = i;
                    break;
                case 6:
                    bl6Codes[first_codeword[5]] = i;
                    break;
                case 7:
                    bl7Codes[first_codeword[6]] = i;
                    break;
                case 8:
                    bl8Codes[first_codeword[7]] = i;
                    break;
                case 9:
                    bl9Codes[ap_uint<8>(first_codeword[8])] = i;
                    break;
                case 10:
                    bl10Codes[ap_uint<8>(first_codeword[9])] = i;
                    break;
                case 11:
                    bl11Codes[ap_uint<8>(first_codeword[10])] = i;
                    break;
                default:
                    assert(0);
                    break;
            }
            first_codeword[blen - 1]++;
        }
    }
}

template <uint8_t PARALLEL_BYTE>
void huffDecodeLiteralsSeq(hls::stream<ap_uint<8 * PARALLEL_BYTE> >& inStream,
                           bool quadStream,
                           ap_uint<8 * PARALLEL_BYTE * 2> accHuff,
                           uint8_t bytesInAcc,
                           uint32_t remSize,
                           uint32_t regeneratedSize,
                           uint8_t accuracyLog,
                           uint16_t weightCnt,
                           uint8_t* weights,
                           hls::stream<ap_uint<8 * PARALLEL_BYTE> >& literalStream) {
    const uint16_t c_streamWidth = 8 * PARALLEL_BYTE;
    const uint16_t c_BSWidth = 16;
    const uint16_t c_accRegWidth = c_streamWidth * 2;
    const uint16_t c_accRegWidthx3 = c_streamWidth * 3;
    const uint16_t c_maxCodeLen = 11;
    // huffman lookup table
    HuffmanTable huffTable[2048];
#pragma HLS BIND_STORAGE variable = huffTable type = ram_t2p impl = bram

    ap_uint<c_BSWidth> bitStream[16 * 1024];
#pragma HLS BIND_STORAGE variable = bitStream type = ram_t2p impl = bram
    uint16_t decSize[4];
    uint16_t cmpSize[4];
#pragma HLS ARRAY_PARTITION variable = cmpSize complete
#pragma HLS ARRAY_PARTITION variable = decSize complete
    uint8_t streamCnt = 1;

    // get stream sizes if 4 streams are present
    if (quadStream) {
        streamCnt = 4;
        // Jump table is 6 bytes long
        // read from input if needed
        if (bytesInAcc < PARALLEL_BYTE) {
            accHuff.range(((PARALLEL_BYTE + bytesInAcc) * 8) - 1, bytesInAcc * 8) = inStream.read();
            bytesInAcc += PARALLEL_BYTE;
        }
        // use 4 bytes
        // get decompressed size
        uint32_t dcmpSize = (regeneratedSize + 3) / 4;
        decSize[0] = decSize[1] = decSize[2] = dcmpSize;
        decSize[3] = regeneratedSize - (dcmpSize * 3);

        // get compressed size
        cmpSize[0] = accHuff;
        accHuff >>= 16;
        cmpSize[1] = accHuff;
        accHuff >>= 16;
        bytesInAcc -= 4;
        // read from input if needed
        if (bytesInAcc < 2) {
            accHuff.range(((PARALLEL_BYTE + bytesInAcc) * 8) - 1, bytesInAcc * 8) = inStream.read();
            bytesInAcc += PARALLEL_BYTE;
        }
        cmpSize[2] = accHuff;
        accHuff >>= 16;
        bytesInAcc -= 2;

        cmpSize[3] = remSize - (6 + cmpSize[0] + cmpSize[1] + cmpSize[2]);
    } else {
        decSize[0] = regeneratedSize;
        cmpSize[0] = remSize;
    }
    // generate huffman lookup table
    huffGenLookupTable<c_maxCodeLen>(weights, huffTable, accuracyLog, weightCnt);

    // decode bitstreams
    ap_uint<(8 * PARALLEL_BYTE)> outBuffer;
    uint8_t obbytes = 0;
    ap_uint<c_accRegWidth> bsbuff = accHuff;
    uint8_t bitsInAcc = bytesInAcc * 8;

    ap_uint<c_streamWidth> bsacc[c_maxCodeLen + 1];
#pragma HLS ARRAY_PARTITION variable = bsacc complete

decode_huff_bitstream_outer:
    for (uint8_t si = 0; si < streamCnt; ++si) {
        // copy data from bitstream to buffer
        uint32_t bsIdx = 0;
        uint8_t bitsWritten = c_BSWidth;
        const int bsPB = c_BSWidth / 8;
        int sIdx = 0;
    // write block data
    hufdlit_fill_bitstream:
        for (int i = 0; i < cmpSize[si]; i += bsPB) {
#pragma HLS PIPELINE II = 1
            if (i + bytesInAcc < cmpSize[si] && bytesInAcc < bsPB) {
                bsbuff.range(((bytesInAcc + PARALLEL_BYTE) * 8) - 1, bytesInAcc * 8) = inStream.read();
                bitsInAcc += c_streamWidth;
            }

            bitStream[bsIdx++] = bsbuff.range(c_BSWidth - 1, 0);

            if (i + bsPB > cmpSize[si]) bitsWritten = 8 * (cmpSize[si] - i);
            bsbuff >>= bitsWritten;
            bitsInAcc -= bitsWritten;
            bytesInAcc = bitsInAcc >> 3;
        }

        // generate decompressed bytes from huffman encoded stream
        ap_uint<c_streamWidth> acchbs = 0;
        uint8_t bitcnt = 0;
        int byteIndx = bsIdx - 1;
        uint32_t outBytes = 0;

        acchbs.range(c_BSWidth - 1, 0) = bitStream[byteIndx--];
        bitcnt = bitsWritten;
        uint8_t byte_0 = acchbs.range(bitcnt - 1, bitcnt - 8);
    // find valid last bit, bitstream read in reverse order
    hufdlit_skip_zero:
        for (uint8_t i = 7; i >= 0; --i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 7
            if ((byte_0 & (1 << i)) > 0) {
                --bitcnt;
                break;
            }
            --bitcnt; // discount higher bits which are zero
        }
        // shift to higher end
        acchbs <<= (c_streamWidth - bitcnt);
        const int msbBitCnt = c_streamWidth - c_BSWidth;
        uint8_t shiftCnt = c_streamWidth - accuracyLog;
        uint8_t sym, blen = 0;
    // decode huffman bitstream
    huf_dec_bitstream:
        while (outBytes < decSize[si]) {
#pragma HLS PIPELINE II = 1
            // fill the acchbs in reverse
            if (bitcnt < 16 && byteIndx > -1) {
                uint32_t tmp = bitStream[byteIndx--];
                acchbs += tmp << (msbBitCnt - bitcnt);
                bitcnt += c_BSWidth;
            }

            uint16_t code = acchbs >> shiftCnt;

            sym = huffTable[code].symbol;
            blen = huffTable[code].bitlen;

        hfdbs_shift_acc:
            for (int s = 1; s < c_maxCodeLen + 1; ++s) {
#pragma HLS UNROLL
#pragma HLS LOOP_TRIPCOUNT min = c_maxCodeLen max = c_maxCodeLen
                bsacc[s] = acchbs << s;
            }
            bitcnt -= blen;
            acchbs = bsacc[blen];

            // write the literal to output stream
            outBuffer.range(((obbytes + 1) * 8) - 1, obbytes * 8) = sym;
            ++obbytes;
            if (obbytes == PARALLEL_BYTE) {
                literalStream << outBuffer;
                obbytes = 0;
                outBuffer = 0;
            }
            ++outBytes;
        }
    }
    if (obbytes > 0) {
        literalStream << outBuffer;
    }
}

template <uint8_t PARALLEL_BYTE, int BS_WIDTH>
void hfdDataFeader(hls::stream<ap_uint<8 * PARALLEL_BYTE> >& inStream,
                   uint8_t streamCnt,
                   uint16_t* cmpSize,
                   ap_uint<8 * PARALLEL_BYTE * 2> accHuff,
                   uint8_t bytesInAcc,
                   hls::stream<ap_uint<BS_WIDTH> >& huffBitStream,
                   hls::stream<ap_uint<8> >& validBitCntStream) {
    const uint16_t c_streamWidth = 8 * PARALLEL_BYTE;
    const uint16_t c_BSWidth = BS_WIDTH;
    const uint16_t c_accRegWidth = c_streamWidth * 2;
    const int c_bsPB = c_BSWidth / 8;
    const int c_bsUpperLim = (((32 / c_bsPB) / 2) * 1024);

    ap_uint<c_BSWidth> bitStream[c_bsUpperLim];
#pragma HLS BIND_STORAGE variable = bitStream type = ram_t2p impl = bram

    // internal registers
    ap_uint<c_accRegWidth> bsbuff = accHuff; // must not contain more than 3 bytes
    uint8_t bitsInAcc = bytesInAcc * 8;

    int wIdx = 0;
    int rIdx = 0;
    int fmInc = 1; // can be +1 or -1
    int smInc = 1;
    uint8_t bitsWritten = c_BSWidth;
    int streamRBgn[4];     // starting index for BRAM read
    int streamRLim[4 + 1]; // point/index till which the BRAM can be read, 1 extra buffer entry
#pragma HLS ARRAY_PARTITION variable = streamRBgn complete
#pragma HLS ARRAY_PARTITION variable = streamRLim complete
    uint8_t wsi = 0, rsi = 0;
    int inIdx = 0;
    // modes
    bool fetchMode = 1;
    bool streamMode = 0;
    bool done = 0;
    // initialize
    streamRLim[wsi] = 0; // initial stream will stream from higher to lower address
hfdl_dataStreamer:
    while (!done) {
#pragma HLS PIPELINE II = 1
        // stream data, bitStream buffer width is equal to inStream width for simplicity
        if (fetchMode) {
            // fill bitstream in direction specified by increment variable
            if (inIdx + bytesInAcc < cmpSize[wsi] && bytesInAcc < c_bsPB) {
                bsbuff.range(bitsInAcc + c_streamWidth - 1, bitsInAcc) = inStream.read();
                bitsInAcc += c_streamWidth;
            }
            bitStream[wIdx] = bsbuff.range(c_BSWidth - 1, 0);
            if (inIdx + c_bsPB >= cmpSize[wsi]) {
                auto bw = 8 * (cmpSize[wsi] - inIdx);
                bitsWritten = (bw == 0) ? bitsWritten : bw;
                validBitCntStream << bitsWritten;
                bsbuff >>= bitsWritten;
                bitsInAcc -= bitsWritten;
                bytesInAcc = bitsInAcc >> 3;

                // update fetch mode state
                if (streamMode == 0) {
                    streamMode = 1;
                    rIdx = wIdx;
                }
                inIdx = 0;            // just an index, not directional
                fmInc = (~fmInc) + 1; // flip 1 and -1
                streamRBgn[wsi] = wIdx;
                ++wsi;
                if (wsi & 1) {
                    streamRLim[wsi] = c_bsUpperLim - 1;
                    wIdx = c_bsUpperLim - 1;
                } else {
                    streamRLim[wsi] = 0;
                    wIdx = 0;
                }
                // post increment checks
                if ((wsi == streamCnt) || (wsi - rsi > 1)) fetchMode = 0;
                // reset default value
                bitsWritten = c_BSWidth;
                continue;
            }
            bsbuff >>= bitsWritten;
            bitsInAcc -= bitsWritten;
            bytesInAcc = bitsInAcc >> 3;

            inIdx += c_bsPB;
            wIdx += fmInc;
        }
        if (streamMode) {
            // write data to output stream
            uint32_t tmp = bitStream[rIdx];
            huffBitStream << tmp;
            // update stream mode state
            if (rIdx == streamRLim[rsi]) {
                ++rsi;
                rIdx = streamRBgn[rsi];
                smInc = (~smInc) + 1; // flip 1 and -1
                // no need to check if fetchMode == 0
                if (wsi < streamCnt) fetchMode = 1;
                // either previous streamMode ended quicker than next fetchMode or streamCnt reached
                if (wsi == rsi) streamMode = 0;
            } else {
                rIdx -= smInc;
            }
        }
        // end condition
        if (!(fetchMode | streamMode)) done = 1;
    }
}

template <uint8_t PARALLEL_BYTE, int MAX_CODELEN, int BS_WIDTH>
void hfdGetCodesStreamLiterals(uint16_t* cmpSize,
                               uint16_t* decSize,
                               uint8_t accuracyLog,
                               uint8_t streamCnt,
                               uint16_t weightCnt,
                               uint8_t* weights,
                               hls::stream<ap_uint<BS_WIDTH> >& huffBitStream,
                               hls::stream<ap_uint<8> >& validBitCntStream,
                               hls::stream<ap_uint<9> >& literalStream) {
    const uint16_t c_HBFSize = 32;
    const uint16_t c_BSWidth = BS_WIDTH;
    const int c_bsPB = c_BSWidth / 8;

    ap_uint<11> validCodeOffset;
    ap_uint<4> bitLen;
    ap_uint<8> symbol[MAX_CODELEN + 1];
#pragma HLS ARRAY_PARTITION variable = symbol complete dim = 0

    // New huffman code
    ap_uint<MAX_CODELEN + 1> codeOffsets[MAX_CODELEN + 1];
#pragma HLS ARRAY_PARTITION variable = codeOffsets dim = 1 complete

    ap_uint<8> bl1Code[2];
    ap_uint<8> bl2Code[4];
    ap_uint<8> bl3Code[8];
    ap_uint<8> bl4Code[16];
    ap_uint<8> bl5Code[32];
    ap_uint<8> bl6Code[64];
    ap_uint<8> bl7Code[128];
    ap_uint<8> bl8Code[256];
    ap_uint<8> bl9Code[256];
    ap_uint<8> bl10Code[256];
    ap_uint<8> bl11Code[256];
#pragma HLS BIND_STORAGE variable = bl1Code type = ram_1p impl = lutram
#pragma HLS BIND_STORAGE variable = bl2Code type = ram_1p impl = lutram
#pragma HLS BIND_STORAGE variable = bl3Code type = ram_1p impl = lutram
#pragma HLS BIND_STORAGE variable = bl4Code type = ram_1p impl = lutram
#pragma HLS BIND_STORAGE variable = bl5Code type = ram_1p impl = lutram
#pragma HLS BIND_STORAGE variable = bl6Code type = ram_1p impl = lutram
#pragma HLS BIND_STORAGE variable = bl7Code type = ram_1p impl = lutram
#pragma HLS BIND_STORAGE variable = bl8Code type = ram_1p impl = lutram
#pragma HLS BIND_STORAGE variable = bl9Code type = ram_1p impl = lutram
#pragma HLS BIND_STORAGE variable = bl10Code type = ram_1p impl = lutram
#pragma HLS BIND_STORAGE variable = bl11Code type = ram_1p impl = lutram

    // generate codes
    code_generator<MAX_CODELEN>(weights, codeOffsets, bl1Code, bl2Code, bl3Code, bl4Code, bl5Code, bl6Code, bl7Code,
                                bl8Code, bl9Code, bl10Code, bl11Code, accuracyLog, weightCnt);

    ap_uint<9> outBuffer;
    uint8_t obbytes = 0;
decode_huff_bitstream_outer:
    for (uint8_t si = 0; si < streamCnt; ++si) {
        // generate decompressed bytes from huffman encoded stream
        ap_uint<c_HBFSize> acchbs = 0;
        uint8_t bitcnt = 0;
        uint32_t outBytes = 0;

        bitcnt = validBitCntStream.read();
        acchbs.range(c_BSWidth - 1, 0) = huffBitStream.read();
        uint8_t byte_0 = acchbs.range(bitcnt - 1, bitcnt - 8);
    // find valid last bit, bitstream read in reverse order
    hufdlit_skip_zero:
        for (uint8_t i = 7; i >= 0; --i) {
#pragma HLS UNROLL
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 7
            if ((byte_0 & (1 << i)) > 0) {
                --bitcnt;
                break;
            }
            --bitcnt; // discount higher bits which are zero
        }
        // shift to higher end
        acchbs <<= (c_HBFSize - bitcnt);
        uint16_t byteIndx = c_bsPB; // just to verify with the compressed size
        const int msbBitCnt = c_HBFSize - c_BSWidth;

        if (bitcnt < 16 && byteIndx < cmpSize[si]) {
            uint32_t tmp = huffBitStream.read();
            acchbs += tmp << (msbBitCnt - bitcnt);
            bitcnt += c_BSWidth;
            byteIndx += c_bsPB;
        }

    // decode huffman bitstreams
    huf_dec_bitstream:
        while (true) {
#pragma HLS PIPELINE II = 1

            for (uint8_t i = 0; i < MAX_CODELEN; ++i) {
#pragma HLS UNROLL
                validCodeOffset.range(i, i) = (acchbs.range(31, 31 - i) >= codeOffsets[i]) ? 1 : 0;
            }

            bitLen = 1 + ap_uint<6>(__builtin_ctz((unsigned int)(validCodeOffset)));

            symbol[1] = bl1Code[acchbs.range(31, 31)];
            symbol[2] = bl2Code[acchbs.range(31, 30)];
            symbol[3] = bl3Code[acchbs.range(31, 29)];
            symbol[4] = bl4Code[acchbs.range(31, 28)];
            symbol[5] = bl5Code[acchbs.range(31, 27)];
            symbol[6] = bl6Code[acchbs.range(31, 26)];
            symbol[7] = bl7Code[acchbs.range(31, 25)];
            symbol[8] = bl8Code[acchbs.range(31, 24)];
            symbol[9] = bl9Code[ap_uint<8>(acchbs.range(31, 23))];
            symbol[10] = bl10Code[ap_uint<8>(acchbs.range(31, 22))];
            symbol[11] = bl11Code[ap_uint<8>(acchbs.range(31, 21))];

            bitcnt -= bitLen;
            acchbs <<= bitLen;

            // write the literal to output stream
            outBuffer.range(7, 0) = symbol[bitLen];
            outBuffer.range(8, 8) = true;
            literalStream << outBuffer;

            ++outBytes;
            if (outBytes == decSize[si]) break;

            if (bitcnt < 16 && byteIndx < cmpSize[si]) {
                uint32_t tmp = huffBitStream.read();
                acchbs += tmp << (msbBitCnt - bitcnt);
                bitcnt += c_BSWidth;
                byteIndx += c_bsPB;
            }
        }
    }

    outBuffer = 0;
    literalStream << outBuffer;
}

template <uint8_t PARALLEL_BYTE>
void writeAccLiteralData(hls::stream<ap_uint<9> >& byteStream,
                         hls::stream<ap_uint<8 * PARALLEL_BYTE> >& literalStream) {
    ap_uint<8 * PARALLEL_BYTE> outBuffer;
    ap_uint<4> writeIdx = 0;

    for (ap_uint<9> inData = byteStream.read(); inData.range(8, 8) == 1; inData = byteStream.read()) {
        outBuffer.range((writeIdx + 1) * 8 - 1, writeIdx * 8) = inData.range(7, 0);
        writeIdx++;
        if (writeIdx == PARALLEL_BYTE) {
            literalStream << outBuffer;
            writeIdx = 0;
        }
    }

    if (writeIdx) {
        literalStream << outBuffer;
    }
}

template <uint8_t PARALLEL_BYTE, int MAX_CODELEN>
void huffDecodeLitInternal(hls::stream<ap_uint<8 * PARALLEL_BYTE> >& inStream,
                           ap_uint<8 * PARALLEL_BYTE * 2> accHuff,
                           uint8_t bytesInAcc,
                           uint16_t* cmpSize,
                           uint16_t* decSize,
                           uint8_t accuracyLog,
                           uint8_t streamCnt,
                           uint16_t weightCnt,
                           uint8_t* weights,
                           hls::stream<ap_uint<8 * PARALLEL_BYTE> >& literalStream) {
    const uint16_t c_BSWidth = 16;
    const uint16_t c_huffBSStreamDepth = 1024;
    const uint8_t c_vbitCntStreamDepth = 8;
    // internal streams
    hls::stream<ap_uint<c_BSWidth> > huffBitStream("huffBitStream");
    hls::stream<ap_uint<8> > validBitCntStream("validBitCntStream");
    hls::stream<ap_uint<9> > byteLiteralStream("byteLiteralStream");
#pragma HLS STREAM variable = huffBitStream depth = c_huffBSStreamDepth
#pragma HLS STREAM variable = validBitCntStream depth = c_vbitCntStreamDepth
#pragma HLS STREAM variable = byteLiteralStream depth = 4

#pragma HLS DATAFLOW

    hfdDataFeader<PARALLEL_BYTE, c_BSWidth>(inStream, streamCnt, cmpSize, accHuff, bytesInAcc, huffBitStream,
                                            validBitCntStream);

    hfdGetCodesStreamLiterals<PARALLEL_BYTE, MAX_CODELEN, c_BSWidth>(cmpSize, decSize, accuracyLog, streamCnt,
                                                                     weightCnt, weights, huffBitStream,
                                                                     validBitCntStream, byteLiteralStream);
    writeAccLiteralData<PARALLEL_BYTE>(byteLiteralStream, literalStream);
}

template <uint8_t PARALLEL_BYTE>
void huffDecodeLiterals(hls::stream<ap_uint<8 * PARALLEL_BYTE> >& inStream,
                        bool quadStream,
                        ap_uint<8 * PARALLEL_BYTE * 2> accHuff,
                        uint8_t bytesInAcc,
                        uint32_t remSize,
                        uint32_t regeneratedSize,
                        uint8_t accuracyLog,
                        uint16_t weightCnt,
                        uint8_t* weights,
                        hls::stream<ap_uint<8 * PARALLEL_BYTE> >& literalStream) {
    const uint16_t c_streamWidth = 8 * PARALLEL_BYTE;
    const uint16_t c_accRegWidth = c_streamWidth * 2;
    const uint16_t c_accRegWidthx3 = c_streamWidth * 3;
    const uint16_t c_maxCodeLen = 11;
    uint8_t streamCnt = 1;
    uint16_t decSize[4];
    uint16_t cmpSize[4];
#pragma HLS ARRAY_PARTITION variable = decSize complete
#pragma HLS ARRAY_PARTITION variable = cmpSize complete
    // get stream sizes if 4 streams are present
    if (quadStream) {
        streamCnt = 4;
        // Jump table is 6 bytes long
        // read from input if needed
        if (bytesInAcc < PARALLEL_BYTE) {
            accHuff.range(((PARALLEL_BYTE + bytesInAcc) * 8) - 1, bytesInAcc * 8) = inStream.read();
            bytesInAcc += PARALLEL_BYTE;
        }
        // use 4 bytes
        // get decompressed size
        uint32_t dcmpSize = (regeneratedSize + 3) / 4;
        decSize[0] = decSize[1] = decSize[2] = dcmpSize;
        decSize[3] = regeneratedSize - (dcmpSize * 3);

        // get compressed size
        cmpSize[0] = accHuff;
        accHuff >>= 16;
        cmpSize[1] = accHuff;
        accHuff >>= 16;
        bytesInAcc -= 4;
        // read from input if needed
        if (bytesInAcc < 2) {
            accHuff.range(((PARALLEL_BYTE + bytesInAcc) * 8) - 1, bytesInAcc * 8) = inStream.read();
            bytesInAcc += PARALLEL_BYTE;
        }
        cmpSize[2] = accHuff;
        accHuff >>= 16;
        bytesInAcc -= 2;

        remSize -= 6;
        cmpSize[3] = remSize - (cmpSize[0] + cmpSize[1] + cmpSize[2]);
    } else {
        decSize[0] = regeneratedSize;
        cmpSize[0] = remSize;
    }
    // parallel huffman decoding
    huffDecodeLitInternal<PARALLEL_BYTE, c_maxCodeLen>(inStream, accHuff, bytesInAcc, cmpSize, decSize, accuracyLog,
                                                       streamCnt, weightCnt, weights, literalStream);
}

} // details
} // compression
} // xf

#endif // _XFCOMPRESSION_ZSTD_FSE_DECODER_HPP_
