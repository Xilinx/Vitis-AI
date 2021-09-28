/*
 * (c) Copyright 2021 Xilinx, Inc. All rights reserved.
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
#ifndef _XFCOMPRESSION_ZSTD_ENCODERS_HPP_
#define _XFCOMPRESSION_ZSTD_ENCODERS_HPP_

/**
 * @file zstd_encoders.hpp
 * @brief Header for modules used in ZSTD compression kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include <ap_int.h>
#include <stdint.h>

#include "zstd_specs.hpp"
#include "compress_utils.hpp"

namespace xf {
namespace compression {
namespace details {

template <int MAX_BITS>
void zstdHuffBitPacker(hls::stream<DSVectorStream_dt<HuffmanCode_dt<MAX_BITS>, 1> >& hfEncodedStream,
                       hls::stream<IntVectorStream_dt<8, 2> >& hfBitStream) {
    // pack huffman codes from multiple input code streams
    bool done = false;
    IntVectorStream_dt<8, 2> outVal;
    ap_uint<32> outReg = 0;
    while (!done) {
        done = true;
        outVal.strobe = 2;
        uint8_t validBits = 0;
        uint16_t outCnt = 0;
    hf_bitPacking:
        for (auto inVal = hfEncodedStream.read(); inVal.strobe > 0; inVal = hfEncodedStream.read()) {
#pragma HLS PIPELINE II = 1
            done = false;
            outReg.range((uint8_t)(inVal.data[0].bitlen) + validBits - 1, validBits) = inVal.data[0].code;
            validBits += (uint8_t)(inVal.data[0].bitlen);
            if (validBits > 15) {
                validBits -= 16;
                outVal.data[0] = outReg.range(7, 0);
                outVal.data[1] = outReg.range(15, 8);
                // outVal.data[2] = outReg.range(23, 16);
                // outVal.data[3] = outReg.range(31, 24);
                outReg >>= 16;
                hfBitStream << outVal;
                outCnt += 2;
            }
        }
        if (outCnt) {
            // mark end by adding 1-bit "1"
            outReg.range(validBits, validBits) = 1;
            ++validBits;
        }
        // write remaining bits
        if (validBits) {
            outVal.strobe = (validBits + 7) >> 3;
            outVal.data[0] = outReg.range(7, 0);
            outVal.data[1] = outReg.range(15, 8);
            // outVal.data[2] = outReg.range(23, 16);
            // outVal.data[3] = outReg.range(31, 24);
            hfBitStream << outVal;
            outCnt += outVal.strobe;
        }
        // printf("bitpacker written bytes: %u\n", outCnt);
        outVal.strobe = 0;
        hfBitStream << outVal;
    }
}

template <int MAX_BITS, int INSTANCES = 1>
void zstdHuffmanEncoder(hls::stream<IntVectorStream_dt<8, 1> >& inValStream,
                        hls::stream<bool>& rleFlagStream,
                        hls::stream<DSVectorStream_dt<HuffmanCode_dt<MAX_BITS>, 1> >& hfCodeStream,
                        hls::stream<DSVectorStream_dt<HuffmanCode_dt<MAX_BITS>, 1> >& hfEncodedStream,
                        hls::stream<ap_uint<16> >& hfLitMetaStream) {
    // read huffman table
    HuffmanCode_dt<MAX_BITS> hfcTable[256]; // use LUTs for implementation as registers
#pragma HLS ARRAY_PARTITION variable = hfcTable complete
    DSVectorStream_dt<HuffmanCode_dt<MAX_BITS>, 1> outVal;
    bool done = false;

    while (true) {
        ap_uint<16> streamSizes[4] = {0, 0, 0, 0};
#pragma HLS ARRAY_PARTITION variable = streamSizes complete
        uint16_t hIdx = 0;
        // pre-read value to check strobe
        auto inVal = inValStream.read();
        if (inVal.strobe == 0) break;
        // Check if this literal block is RLE type
        bool isRle = rleFlagStream.read();
    // read this table only once
    read_hfc_table:
        for (auto hfCode = hfCodeStream.read(); hfCode.strobe > 0; hfCode = hfCodeStream.read()) {
#pragma HLS PIPELINE II = 1
            hfcTable[hIdx++] = hfCode.data[0];
        }
        uint8_t streamCnt = inVal.data[0];
        hfLitMetaStream << ((uint16_t)streamCnt & 0x000F);
    // read the stream sizes

    get_lit_streams_size:
        for (uint8_t si = 0; si < streamCnt; ++si) {
#pragma HLS PIPELINE II = 2
            // read present stream size
            inVal = inValStream.read();
            streamSizes[si].range(7, 0) = inVal.data[0];
            inVal = inValStream.read();
            streamSizes[si].range(15, 8) = inVal.data[0];
        }
    // Parallel read 8 * INSTANCES bits of input
    // Parallel encode to INSTANCES output streams
    encode_lit_streams:
        for (uint8_t si = 0; si < streamCnt; ++si) {
            auto streamSize = streamSizes[si];
            uint16_t streamCmpSize = 0;
            uint8_t cmpBits = 0;
            // Since for RLE type literals, only one stream is present
            // in that stream, last literal must not be encoded, comes first in reversed stream
            if (isRle) {
                --streamSize;
                inValStream.read();
            }
            outVal.strobe = 1;
        huff_encode:
            for (uint16_t i = 0; i < (uint16_t)streamSize; ++i) {
#pragma HLS PIPELINE II = 1
                inVal = inValStream.read();
                outVal.data[0] = hfcTable[inVal.data[0]];
                hfEncodedStream << outVal;
                cmpBits += outVal.data[0].bitlen;
                if (cmpBits > 7) {
                    streamCmpSize += cmpBits >> 3;
                    cmpBits &= 7;
                }
            }
            // cmpBits cannot be greater than 7, 1 extra bit indicating end of stream
            if (cmpBits + 1) {
                ++streamCmpSize;
            }
            hfLitMetaStream << streamCmpSize;
            // end of sub-stream
            outVal.strobe = 0;
            hfEncodedStream << outVal;
        }
    }
    // end of file
    hfCodeStream.read();
    outVal.strobe = 0;
    hfEncodedStream << outVal;
}

inline uint8_t getOptimalTableLog(const uint8_t maxLog, uint16_t symbolSize, uint16_t currentMaxCode) {
#pragma HLS INLINE off
    auto maxBitsNeeded = bitsUsed31(symbolSize - 1) - 2;
    ap_uint<10> tableLog = maxLog;
    // fin table log
    uint8_t minBitsSrc = bitsUsed31(symbolSize) + 1;
    uint8_t minBitsSymbols = bitsUsed31(currentMaxCode) + 2;
    uint8_t minBitsNeeded = minBitsSrc < minBitsSymbols ? minBitsSrc : minBitsSymbols;

    if (maxBitsNeeded < tableLog) tableLog = maxBitsNeeded; // Accuracy can be reduced
    if (minBitsNeeded > tableLog) tableLog = minBitsNeeded; // Need a minimum to safely represent all symbol values
    if (tableLog < c_fseMinTableLog) tableLog = c_fseMinTableLog;
    if (tableLog > maxLog) tableLog = maxLog; // will be changed to c_fseMaxTableLog
    return (uint8_t)tableLog;
}

template <int MAX_FREQ_DWIDTH = 17>
void normalizeFreq(
    ap_uint<MAX_FREQ_DWIDTH>* inFreq,
    uint16_t symbolSize, // total : maximum value of symbols (max literal value, number of LL/ML/OF code values: seqcnt)
    uint16_t curMaxCodeVal, // maxSymbolValue : maximum code value (max bitlen value for literal huffman codes, max
                            // LL/ML/OF code value)
    uint8_t tableLog,
    int16_t* normTable) {
#pragma HLS INLINE off
    // core module to generate normalized table
    uint64_t rtbXvStepTable[8];
    uint8_t scale = 62 - tableLog;
    ap_uint<64> step = ((ap_uint<64>)1 << 62) / symbolSize; /* <-- one division here ! */
    uint64_t vStep = (uint64_t)1 << (scale - 20);
    int stillToDistribute = 1 << tableLog;
    uint32_t largest = 0;
    int16_t largestP = 0;
    uint32_t lowThreshold = (uint32_t)(symbolSize >> tableLog);
init_rtbTableSteps:
    for (int i = 0; i < 8; ++i) {
#pragma HLS PIPELINE off
        rtbXvStepTable[i] = vStep * c_rtbTable[i];
    }

norm_count:
    for (uint16_t s = 0; s <= curMaxCodeVal; ++s) {
#pragma HLS PIPELINE II = 1
        // if (inFreq[s] == symbolSize) return 0;   /* rle special case */
        auto freq = inFreq[s];
        int16_t proba = (int16_t)((freq * step) >> scale);
        ap_uint<1> incrProba = (((freq * step) - ((uint64_t)proba << scale)) > rtbXvStepTable[proba]);
        int16_t val = 0;
        int16_t valDecrement = 0;
        if (freq == 0) {
            val = 0;
            valDecrement = 0;
        } else if (freq <= lowThreshold) {
            val = -1;
            valDecrement = 1;
        } else {
            if (proba < 8) {
                // uint64_t restToBeat = vStep * c_rtbTable[proba];
                proba += incrProba;
            }
            if (proba > largestP) {
                largestP = proba;
                largest = s;
            }
            val = proba;
            valDecrement = proba;
        }
        normTable[s] = val;
        stillToDistribute -= valDecrement;
    }
    // assert(-stillToDistribute < (normTable[largest] >> 1));
    // corner case, need another normalization method
    // FSE_normalizeM2(normTable, tableLog, inFreq, symbolSize, curMaxCodeVal);
    normTable[largest] += (short)stillToDistribute;
}

template <int MAX_FREQ_DWIDTH = 17>
void normalizedTableGen(hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> >& inFreqStream,
                        hls::stream<IntVectorStream_dt<16, 1> > normTableStream[2]) {
    // generate normalized counter to be used for fse table generation
    /*
     * > Read the frequencies
     * > Get max frequency value, maxCode and sequence count
     * > Calculate optimal table log
     * > Calculate normalized distribution table
     */
    const uint8_t c_maxTableLog[4] = {c_fseMaxTableLogLL, c_fseMaxTableLogOF, c_fseMaxTableLogHF, c_fseMaxTableLogML};
    const uint8_t c_hfIdx = 2;
    IntVectorStream_dt<16, 1> outVal;
    ap_uint<MAX_FREQ_DWIDTH> inFreq[64]; // using max possible size for array
    int16_t normTable[64];
    uint16_t seqCnt = 0;
norm_tbl_outer_loop:
    while (true) {
        bool noSeq = true;
        uint8_t lastSeq[3] = {0, 0, 0}; // ll, ml, of
        uint8_t lsk = 0;
        // first value is sequence count
        auto inVal = inFreqStream.read();
        if (inVal.strobe == 0) break;
        seqCnt = inVal.data[0];
        noSeq = (seqCnt == 0);
    // last sequence
    read_last_seq:
        for (uint8_t i = 0; i < 3; ++i) {
#pragma HLS PIPELINE II = 1
            inVal = inFreqStream.read();
            lastSeq[i] = inVal.data[0];
        }
    calc_norm_outer:
        for (uint8_t k = 0; k < 4; ++k) {
        init_norm_table:
            for (uint8_t i = 0; i < 64; ++i) {
#pragma HLS PIPELINE off
                normTable[i] = 0;
            }
            uint16_t maxSymbol;
            ap_uint<MAX_FREQ_DWIDTH> maxFreq = 0;
            uint16_t symCnt = seqCnt;
            volatile uint16_t symCnt_1 = seqCnt;
            // read max literal value, before weight freq data
            if (c_hfIdx == k) {
                inVal = inFreqStream.read();
                symCnt = inVal.data[0];
                symCnt_1 = symCnt;
            }
            // number of frequencies to read
            inVal = inFreqStream.read();
            uint16_t freqCnt = inVal.data[0];
        // read input frequencies
        read_in_freq:
            for (uint16_t i = 0; i < freqCnt; ++i) {
#pragma HLS PIPELINE II = 1
                inVal = inFreqStream.read();
                inFreq[i] = inVal.data[0];
                if (inVal.data[0] > 0) maxSymbol = i;
                if (inVal.data[0] > maxFreq) maxFreq = inVal.data[0];
            }
            uint8_t tableLog = 0;
            if (noSeq == false || c_hfIdx == k) {
                // get optimal table log
                tableLog = getOptimalTableLog(c_maxTableLog[k], symCnt, maxSymbol);
                if (c_hfIdx != k) {
                    if (inFreq[lastSeq[lsk]] > 1) {
                        inFreq[lastSeq[lsk]]--;
                        symCnt_1 -= 1;
                    }
                    ++lsk;
                }
                // generate normalized distribution table
                normalizeFreq<MAX_FREQ_DWIDTH>(inFreq, symCnt_1, maxSymbol, tableLog, normTable);
            }
            // write normalized table to output
            outVal.strobe = 1;
            // write tableLog, max val at the end of table log
            normTable[63] = maxSymbol;
            normTable[62] = tableLog;
            normTable[61] = (int16_t)(c_hfIdx == k); // To be used later for optimization
        write_norm_table:
            for (uint16_t i = 0; i < 64; ++i) {
#pragma HLS PIPELINE II = 1
                outVal.data[0] = normTable[i];
                normTableStream[0] << outVal;
                normTableStream[1] << outVal;
            }
        }
    }
    outVal.strobe = 0;
    normTableStream[0] << outVal;
    normTableStream[1] << outVal;
}

void fseHeaderGen(hls::stream<IntVectorStream_dt<16, 1> >& normTableStream,
                  hls::stream<IntVectorStream_dt<8, 2> >& fseHeaderStream) {
    // generate normalized counter to be used for fse table generation
    int16_t normTable[64];
    IntVectorStream_dt<8, 2> outVal;

fse_header_gen_outer:
    while (true) {
        auto inVal = normTableStream.read();
        if (inVal.strobe == 0) break;
        normTable[0] = inVal.data[0];
    read_norm_table:
        for (uint8_t i = 1; i < 64; ++i) {
#pragma HLS PIPELINE II = 1
            inVal = normTableStream.read();
            normTable[i] = inVal.data[0];
        }
        uint16_t maxCharSize = normTable[63] + 1;
        uint8_t tableLog = normTable[62];
        outVal.strobe = 2;

        if (tableLog > 0) {
            uint16_t tableSize = 1 << tableLog;
            ap_uint<32> bitStream = 0;
            int8_t bitCount = 0;
            uint16_t symbol = 0;
            uint8_t outCount = 0;

            /* Table Size */
            bitStream = (tableLog - c_fseMinTableLog);
            bitCount = 4;

            /* Init */
            int remaining = tableSize + 1; /* +1 for extra accuracy */
            uint16_t threshold = tableSize;
            uint8_t nbBits = tableLog + 1;
            uint16_t start = symbol;
            enum ReadNCountStates { PREV_IS_ZERO, REM_LT_THR, NON_ZERO_CNT };
            ReadNCountStates fsmState = NON_ZERO_CNT;
            ReadNCountStates fsmNextState = NON_ZERO_CNT;
            bool previousIs0 = false;
            bool skipZeroFreq = true;
        gen_fse_header_bitstream:
            while ((symbol < maxCharSize) && (remaining > 1)) {
#pragma HLS PIPELINE II = 1
                if (fsmState == PREV_IS_ZERO) {
                    if (skipZeroFreq) {
                        if (symbol < maxCharSize && !normTable[symbol]) {
                            ++symbol;
                        } else {
                            skipZeroFreq = false;
                        }
                    } else {
                        if (symbol >= start + 24) {
                            start += 24;
                            // bitStream += 0xFFFF << bitCount;
                            bitStream.range(15 + bitCount, bitCount) = 0xFFFF;
                            bitCount += 16;
                        } else if (symbol >= start + 3) {
                            start += 3;
                            // bitStream += 3 << bitCount;
                            bitStream.range(1 + bitCount, bitCount) = 3;
                            bitCount += 2;
                        } else {
                            fsmState = NON_ZERO_CNT;
                            // bitStream += (uint64_t)(symbol - start) << bitCount;
                            bitStream.range(1 + bitCount, bitCount) = symbol - start;
                            bitCount += 2;
                        }
                    }
                } else if (fsmState == REM_LT_THR) {
                    --nbBits;
                    threshold >>= 1;
                    if (remaining > threshold - 1) {
                        fsmState = fsmNextState;
                    }
                } else {
                    int16_t count = normTable[symbol++];
                    int max = (2 * threshold) - (1 + remaining);
                    remaining -= (count < 0) ? -count : count;
                    ++count;
                    if (count >= threshold) count += max;
                    // bitStream += count << bitCount;
                    bitStream.range(nbBits + bitCount - 1, bitCount) = count;
                    bitCount += nbBits;
                    bitCount -= (uint8_t)(count < max);
                    previousIs0 = (count == 1);
                    start = symbol;      // set starting symbol for PREV_IS_ZERO state
                    skipZeroFreq = true; // enable skipping of zero norm values for PREV_IS_ZERO state
                    fsmNextState = (previousIs0 ? PREV_IS_ZERO : NON_ZERO_CNT);
                    fsmState = ((remaining < threshold) ? REM_LT_THR : fsmNextState);
                }
                // write output bitstream 16-bits at a time
                if (bitCount > 15) {
                    outVal.data[0] = bitStream.range(7, 0);
                    outVal.data[1] = bitStream.range(15, 8);
                    // outVal.data[2] = bitStream.range(23, 16);
                    // outVal.data[3] = bitStream.range(31, 24);
                    fseHeaderStream << outVal;
                    bitStream >>= 16;
                    bitCount -= 16;
                    outCount += 2;
                }
            }
            if (bitCount) {
                outVal.strobe = (uint8_t)((bitCount + 7) / 8);
                outVal.data[0] = bitStream.range(7, 0);
                outVal.data[1] = bitStream.range(15, 8);
                // outVal.data[2] = bitStream.range(23, 16);
                // outVal.data[3] = bitStream.range(31, 24);
                fseHeaderStream << outVal;
                outCount += outVal.strobe;
            }
            outVal.strobe = 0;
            fseHeaderStream << outVal;
        }
    }
    outVal.strobe = 0;
    fseHeaderStream << outVal;
}

void fseEncodingTableGen(hls::stream<IntVectorStream_dt<16, 1> >& normTableStream,
                         hls::stream<IntVectorStream_dt<36, 1> >& fseTableStream) {
    // generate normalized counter to be used for fse table generation
    const uint8_t c_hfIdx = 2; // index of data for literal bitlens
    int16_t normTable[64];
    uint8_t symTable[512];
    uint16_t tableU16[512];
    uint32_t intm[257];
    IntVectorStream_dt<36, 1> outVal;
    uint8_t cIdx = 0;
    while (true) {
        // read normalized counter
        auto inVal = normTableStream.read();
        if (inVal.strobe == 0) break;
        normTable[0] = inVal.data[0];
    fetg_read_norm_tbl:
        for (uint8_t i = 1; i < 64; ++i) {
#pragma HLS PIPELINE II = 1
            inVal = normTableStream.read();
            normTable[i] = inVal.data[0];
        }
        uint16_t maxSymbol = normTable[63];
        uint8_t tableLog = normTable[62];
        outVal.strobe = 1;
        // send tableLog and maxSymbol
        outVal.data[0].range(7, 0) = tableLog;
        outVal.data[0].range(35, 8) = maxSymbol;
        fseTableStream << outVal;

        if (tableLog > 0) {
            uint16_t tableSize = 1 << tableLog;
            uint32_t tableMask = tableSize - 1;
            const uint32_t step = (tableSize >> 1) + (tableSize >> 3) + 3;
            uint32_t highThreshold = tableSize - 1;

            intm[0] = 0;
            uint32_t ivSp = intm[0];
        fse_gen_symbol_start_pos:
            for (uint32_t s = 1; s <= maxSymbol + 1; ++s) {
#pragma HLS PIPELINE II = 1
                auto nvSp = normTable[s - 1];
                int16_t ivInc = 1;
                if (nvSp == -1) {
                    symTable[highThreshold] = s - 1;
                    --highThreshold;
                } else {
                    ivInc = nvSp;
                }
                intm[s] = ivSp + ivInc;
                ivSp += ivInc;
            }
            intm[maxSymbol + 1] = tableSize + 1;

            // spread symbols
            uint16_t pos = 0;
        fse_spread_symbols_outer:
            for (uint32_t s = 0; s <= maxSymbol; ++s) {
            fse_spread_symbols:
                for (int16_t n = 0; n < normTable[s];) {
#pragma HLS PIPELINE II = 1
                    if (pos > highThreshold) {
                        pos = (pos + step) & tableMask;
                    } else {
                        symTable[pos] = s;
                        pos = (pos + step) & tableMask;
                        ++n;
                    }
                }
            }
        // tableU16[-2] = tableLog;
        // tableU16[-1] = maxSymbol;
        build_table:
            for (uint16_t u = 0; u < tableSize; ++u) {
#pragma HLS PIPELINE II = 1
                auto s = symTable[u];
                tableU16[intm[s]++] = tableSize + u;
            }

        // send state table, tableSize on reader side can be calculated using tableLog
        send_state_table:
            for (uint16_t i = 0; i < tableSize; ++i) {
#pragma HLS PIPELINE II = 1
                outVal.data[0] = tableU16[i];
                fseTableStream << outVal;
            }

            // printf("Find state and bit count table\n");
            uint16_t total = 0;
        build_sym_transformation_table:
            for (uint16_t s = 0; s <= maxSymbol; ++s) {
#pragma HLS PIPELINE II = 1
                int nv = normTable[s];
                uint8_t sym = 0;
                uint32_t nBits = 0;
                int16_t findState = 0;
                if (nv == 0) {
                    nBits = ((tableLog + 1) << 16) - (1 << tableLog);
                } else if (nv == 1 || nv == -1) {
                    nBits = (tableLog << 16) - (1 << tableLog);
                    findState = total - 1;
                    ++total;
                } else {
                    uint8_t maxBitsOut = tableLog - bitsUsed31(nv - 1);
                    uint32_t minStatePlus = (uint32_t)nv << maxBitsOut;
                    nBits = (maxBitsOut << 16) - minStatePlus;
                    findState = total - nv;
                    total += nv;
                }
                outVal.data[0].range(19, 0) = nBits;
                outVal.data[0].range(35, 20) = findState;
                fseTableStream << outVal;
            }
        }
        outVal.strobe = 0;
        fseTableStream << outVal;
        cIdx++;
        if (cIdx == 4) cIdx = 0;
    }
    outVal.strobe = 0;
    fseTableStream << outVal;
}

void separateLitSeqTableStreams(hls::stream<IntVectorStream_dt<36, 1> >& fseTableStream,
                                hls::stream<IntVectorStream_dt<36, 1> >& fseLitTableStream,
                                hls::stream<IntVectorStream_dt<36, 1> >& fseSeqTableStream) {
    const uint8_t c_hfIdx = 2; // index of data for literal bitlens
    uint8_t cIdx = 0;
    IntVectorStream_dt<36, 1> fseTableVal;
sep_lit_seq_fset_outer:
    while (true) {
        fseTableVal = fseTableStream.read();
        if (fseTableVal.strobe == 0) break;
    sep_lit_seq_fseTableStreams:
        while (fseTableVal.strobe > 0) {
#pragma HLS PIPELINE II = 1
            if (cIdx == c_hfIdx) {
                fseLitTableStream << fseTableVal;
            } else {
                fseSeqTableStream << fseTableVal;
            }
            fseTableVal = fseTableStream.read();
        }
        // write strobe 0 value
        if (cIdx == c_hfIdx) {
            fseLitTableStream << fseTableVal;
        } else {
            fseSeqTableStream << fseTableVal;
        }
        ++cIdx;
        if (cIdx == 4) cIdx = 0;
    }
    fseTableVal.strobe = 0;
    fseLitTableStream << fseTableVal;
    fseSeqTableStream << fseTableVal;
}

template <int MAX_FREQ_DWIDTH = 17>
void fseTableGen(hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> >& inFreqStream,
                 hls::stream<IntVectorStream_dt<8, 2> >& fseHeaderStream,
                 hls::stream<IntVectorStream_dt<36, 1> >& fseLitTableStream,
                 hls::stream<IntVectorStream_dt<36, 1> >& fseSeqTableStream) {
    // internal streams
    hls::stream<IntVectorStream_dt<16, 1> > normTableStream[2];
    hls::stream<IntVectorStream_dt<36, 1> > fseTableStream("fseTableStream");
#pragma HLS STREAM variable = normTableStream depth = 128
#pragma HLS STREAM variable = fseTableStream depth = 16

#pragma HLS DATAFLOW
    // generate normalized counter table
    normalizedTableGen<MAX_FREQ_DWIDTH>(inFreqStream, normTableStream);
    // generate FSE header
    fseHeaderGen(normTableStream[0], fseHeaderStream);
    // generate FSE encoding tables
    fseEncodingTableGen(normTableStream[1], fseTableStream);
    // separate lit-seq fse table streams
    separateLitSeqTableStreams(fseTableStream, fseLitTableStream, fseSeqTableStream);
}

inline bool readFseTable(hls::stream<IntVectorStream_dt<36, 1> >& fseTableStream,
                         ap_uint<36>* fseStateBitsTable,
                         uint16_t* fseNextStateTable,
                         uint8_t& tableLog,
                         uint16_t& maxFreqLL) {
    // read FSE table values from input table stream
    auto fseVal = fseTableStream.read();
    if (fseVal.strobe == 0) return true;
    tableLog = fseVal.data[0].range(7, 0);
    maxFreqLL = fseVal.data[0].range(23, 8);

    uint16_t tableSize = (1 << tableLog);
    uint16_t fIdx = 0;
read_fse_tables:
    for (fseVal = fseTableStream.read(); fseVal.strobe > 0 && tableLog > 0; fseVal = fseTableStream.read()) {
#pragma HLS PIPELINE II = 1
        if (fIdx < tableSize) {
            fseNextStateTable[fIdx] = (int16_t)fseVal.data[0].range(15, 0);
        } else {
            fseStateBitsTable[fIdx - tableSize] = fseVal.data[0];
        }
        ++fIdx;
    }
    return false;
}

template <int BITSTREAM_DWIDTH = 32>
inline void fseEncodeSymbol(uint8_t symbol,
                            ap_uint<36>* fseStateBitsTable,
                            uint16_t* fseNextStateTable,
                            uint16_t& fseState,
                            ap_uint<BITSTREAM_DWIDTH>& bitstream,
                            uint8_t& bitCount,
                            bool isInit) {
#pragma HLS INLINE
    // encode a symbol using fse table
    // This version of function is best for literal header encoding
    constexpr uint32_t c_oneBy15Lsh = ((uint32_t)1 << 15);
    ap_uint<36> stateBitVal = fseStateBitsTable[symbol];
    uint32_t deltaBits = (uint32_t)(stateBitVal.range(19, 0));
    int16_t findState = (int16_t)(stateBitVal.range(35, 20));
    uint32_t nbBits;
    uint16_t stVal;
    if (isInit) {
        nbBits = (deltaBits + c_oneBy15Lsh) >> 16;
        stVal = (nbBits << 16) - deltaBits;
    } else {
        nbBits = (uint32_t)(deltaBits + fseState) >> 16;
        stVal = fseState;
        // write bits to bitstream
        bitstream |= ((ap_uint<BITSTREAM_DWIDTH>)(stVal & c_bitMask[nbBits]) << bitCount);
        bitCount += nbBits;
    }
    // store current state
    uint32_t nxIdx = (stVal >> nbBits) + findState;
    fseState = fseNextStateTable[nxIdx];
}

template <int OUT_DWIDTH = 16, int IS_INIT = 0>
inline void fseEncodeSymbol(uint8_t symbol,
                            ap_uint<36>* fseStateBitsTable,
                            uint16_t* fseNextStateTable,
                            uint16_t& fseState,
                            ap_uint<OUT_DWIDTH>& outWord,
                            ap_uint<5>& bitCount) {
#pragma HLS INLINE
    // encode a symbol using fse table
    // This version of function is best for sequence encoding
    constexpr uint32_t c_oneBy15Lsh = ((uint32_t)1 << 15);
    ap_uint<36> stateBitVal = fseStateBitsTable[symbol];
    uint32_t deltaBits = (uint32_t)(stateBitVal.range(19, 0));
    int16_t findState = (int16_t)(stateBitVal.range(35, 20));
    uint32_t nbBits;
    uint16_t stVal;
    if (IS_INIT) {
        nbBits = (deltaBits + c_oneBy15Lsh) >> 16;
        stVal = (nbBits << 16) - deltaBits;
    } else {
        nbBits = (uint32_t)(deltaBits + fseState) >> 16;
        stVal = fseState;
        // write bits to bitstream
        outWord = (ap_uint<OUT_DWIDTH>)(stVal & c_bitMask[nbBits]);
        bitCount = nbBits;
    }
    // store current state
    uint32_t nxIdx = (stVal >> nbBits) + findState;
    fseState = fseNextStateTable[nxIdx];
}

void fseEncodeLitHeader(hls::stream<IntVectorStream_dt<4, 1> >& hufWeightStream,
                        hls::stream<IntVectorStream_dt<36, 1> >& fseLitTableStream,
                        hls::stream<IntVectorStream_dt<8, 2> >& encodedOutStream) {
    // fse encoding of huffman header for encoded literals
    IntVectorStream_dt<8, 2> outVal;
    ap_uint<36> fseStateBitsTable[256];
    uint16_t fseNextStateTable[256];
    ap_uint<4> hufWeights[256];

fse_lit_encode_outer:
    while (true) {
        uint8_t tableLog;
        uint16_t maxSymbol;
        uint16_t maxFreq;
        // read fse table
        uint16_t fIdx = 0;
        // read FSE encoding tables
        bool done = readFseTable(fseLitTableStream, fseStateBitsTable, fseNextStateTable, tableLog, maxFreq);
        if (done) break;
    // read details::c_maxLitV + 1 data from weight stream
    read_hf_weights:
        for (auto inWeight = hufWeightStream.read(); inWeight.strobe > 0; inWeight = hufWeightStream.read()) {
#pragma HLS PIPELINE II = 1
            hufWeights[fIdx] = inWeight.data[0];
            if (inWeight.data[0] > 0) maxSymbol = fIdx;
            ++fIdx;
        }
        uint16_t preStateVal[2];
        bool isInit[2] = {true, true};
        bool stateIdx = 0; // 0 for even, 1 for odd
        uint8_t bitCount = 0;
        ap_uint<32> bitstream = 0;
        outVal.strobe = 2;
        int outCnt = 0;
        // encode weights stored in reverse order
        stateIdx = maxSymbol & 1;
    fse_lit_encode:
        for (int16_t w = maxSymbol - 1; w > -1; --w) { // TODO: Fast forward to 2 symbols per clock cycle
#pragma HLS PIPELINE II = 1
            uint8_t symbol = hufWeights[w];
            fseEncodeSymbol<32>(symbol, fseStateBitsTable, fseNextStateTable, preStateVal[stateIdx], bitstream,
                                bitCount, isInit[stateIdx]);
            isInit[stateIdx] = false;
            // write bitstream to output
            if (bitCount > 15) {
                // write to output stream
                outVal.data[0] = bitstream.range(7, 0);
                outVal.data[1] = bitstream.range(15, 8);
                encodedOutStream << outVal;
                bitstream >>= 16;
                bitCount -= 16;
                outCnt += 2;
            }
            // switch state flow
            stateIdx = (stateIdx + 1) & 1; // 0 if 1, 1 if 0
        }
        // encode last two
        bitstream |= ((ap_uint<32>)(preStateVal[0] & c_bitMask[tableLog]) << bitCount);
        bitCount += tableLog;
        bitstream |= ((ap_uint<32>)(preStateVal[1] & c_bitMask[tableLog]) << bitCount);
        bitCount += tableLog;
        // mark end by adding 1-bit "1"
        bitstream |= (ap_uint<32>)1 << bitCount;
        ++bitCount;
        // max remaining biCount can be 15 + (2 * 6) + 1= 28 bits => 4 bytes
        int8_t remBytes = (int8_t)((bitCount + 7) / 8);
    // write bitstream to output
    write_rem_bytes:
        while (remBytes > 0) {
#pragma HLS PIPELINE II = 1
            // write to output stream
            outVal.data[0] = bitstream.range(7, 0);
            outVal.data[1] = bitstream.range(15, 8);
            outVal.strobe = ((remBytes > 1) ? 2 : 1);
            encodedOutStream << outVal;
            bitstream >>= 16;
            remBytes -= 2;
            outCnt += outVal.strobe;
        }
        outVal.strobe = 0;
        encodedOutStream << outVal;
    }
    // dump strobe 0 data
    hufWeightStream.read();
    // write end of block
    outVal.strobe = 0;
    encodedOutStream << outVal;
}

template <int MAX_FREQ_DWIDTH>
void fseGetSeqCodes(hls::stream<DSVectorStream_dt<Sequence_dt<MAX_FREQ_DWIDTH>, 1> >& inSeqStream,
                    hls::stream<DSVectorStream_dt<Sequence_dt<6>, 1> >& seqCodeStream,
                    hls::stream<bool>& noSeqFlagStream,
                    hls::stream<ap_uint<33> >& extCodewordStream,
                    hls::stream<ap_uint<8> >& extBitlenStream) {
    // get sequence, code and code bit-lengths
    DSVectorStream_dt<Sequence_dt<6>, 1> seqCode;
    ap_uint<33> extCodeword;
    ap_uint<8> extBitlen;
fse_get_seq_codes_main:
    while (true) {
        auto nextSeq = inSeqStream.read();
        if (nextSeq.strobe == 0) break;
        seqCode.strobe = 1;
        // check for noSeq condition
        if (nextSeq.data[0].litlen == 0 && nextSeq.data[0].matlen == 0 && nextSeq.data[0].offset == 0) {
            // read strobe zero value, since no sequence is present
            nextSeq = inSeqStream.read();
            noSeqFlagStream << 1;
        } else {
            noSeqFlagStream << 0;
        // Send sequence codes and extra bit-lengths with extra codewords
        fetch_sequence_codes:
            while (nextSeq.strobe > 0) {
#pragma HLS PIPELINE II = 1
                auto inSeq = nextSeq;
                // Read next sequence
                nextSeq = inSeqStream.read();
                // process current sequence
                seqCode.data[0].litlen = getLLCode(inSeq.data[0].litlen);
                seqCode.data[0].matlen = getMLCode(inSeq.data[0].matlen);
                seqCode.data[0].offset = bitsUsed31(inSeq.data[0].offset);
                // get bits for adding to bitstream
                uint8_t llBits = c_extraBitsLL[seqCode.data[0].litlen];
                uint8_t mlBits = c_extraBitsML[seqCode.data[0].matlen];
                uint8_t ofBits = seqCode.data[0].offset;
                // get masked extra bit values
                ap_uint<33> excLL = inSeq.data[0].litlen & c_bitMask[llBits];
                ap_uint<33> excML = inSeq.data[0].matlen & c_bitMask[mlBits];
                ap_uint<33> excOF = inSeq.data[0].offset & c_bitMask[ofBits];
                // get combined extra codeword
                extCodeword = (excOF << (mlBits + llBits)) + (excML << llBits) + excLL;
                extBitlen = ofBits + mlBits + llBits;
                // write information to next units
                seqCodeStream << seqCode;
                extCodewordStream << extCodeword;
                extBitlenStream << extBitlen;
            }
            // End of block in case of valid sequence block
            seqCode.strobe = 0;
            seqCodeStream << seqCode;
        }
    }
}

void fseEncodeSeqCodes(hls::stream<IntVectorStream_dt<36, 1> >& fseSeqTableStream,
                       hls::stream<DSVectorStream_dt<Sequence_dt<6>, 1> >& seqCodeStream,
                       hls::stream<bool>& noSeqFlagStream,
                       hls::stream<IntVectorStream_dt<28, 1> >& seqFseWordStream,
                       hls::stream<ap_uint<5> >& wordBitlenStream) {
    // Encode sequence codes
    // Internal tables
    ap_uint<36> fseStateBitsTableLL[512];
    uint16_t fseNextStateTableLL[512];
    ap_uint<36> fseStateBitsTableML[512];
    uint16_t fseNextStateTableML[512];
    ap_uint<36> fseStateBitsTableOF[256];
    uint16_t fseNextStateTableOF[256];
    // out word having D-WIDTH = (9 (max tableLog) * 3) + 1(end bit)
    IntVectorStream_dt<28, 1> fseOutWord;
fse_encode_seq_main:
    while (true) {
        uint8_t tableLogLL, tableLogML, tableLogOF;
        uint16_t maxSymbolLL, maxSymbolML, maxSymbolOF;
        uint16_t maxFreqLL, maxFreqML, maxFreqOF;
        // read initial value to check OES
        // read FSE encoding tables for litlen, matlen, offset
        bool noData = readFseTable(fseSeqTableStream, fseStateBitsTableLL, fseNextStateTableLL, tableLogLL, maxFreqLL);
        if (noData) break;
        readFseTable(fseSeqTableStream, fseStateBitsTableOF, fseNextStateTableOF, tableLogOF, maxFreqOF);
        readFseTable(fseSeqTableStream, fseStateBitsTableML, fseNextStateTableML, tableLogML, maxFreqML);
        // Check for no sequence condition
        auto noSeq = noSeqFlagStream.read();
        if (noSeq) continue;
        // read and fse encode sequence codes
        uint16_t llPrevStateVal, mlPrevStateVal, ofPrevStateVal;
        ap_uint<9> outWordLL, outWordML, outWordOF;
        ap_uint<5> bitsLL, bitsML, bitsOF;
        // Initialise fse states for first sequence set
        auto seqCode = seqCodeStream.read();
        uint8_t llCode = (uint8_t)seqCode.data[0].litlen;
        uint8_t ofCode = (uint8_t)seqCode.data[0].offset;
        uint8_t mlCode = (uint8_t)seqCode.data[0].matlen;
        // Initialization does not write any output
        fseEncodeSymbol<9, 1>(ofCode, fseStateBitsTableOF, fseNextStateTableOF, ofPrevStateVal, outWordOF, bitsOF);
        fseEncodeSymbol<9, 1>(mlCode, fseStateBitsTableML, fseNextStateTableML, mlPrevStateVal, outWordML, bitsML);
        fseEncodeSymbol<9, 1>(llCode, fseStateBitsTableLL, fseNextStateTableLL, llPrevStateVal, outWordLL, bitsLL);
        uint8_t tableLogSum = tableLogOF + tableLogML + tableLogLL;
        ap_uint<28> endMark = (1 << tableLogSum);
        ap_uint<5> fseBitCnt = 0;
        fseOutWord.strobe = 1;
    fse_encode_seq_codes:
        for (seqCode = seqCodeStream.read(); seqCode.strobe > 0; seqCode = seqCodeStream.read()) {
#pragma HLS PIPELINE II = 1
            uint8_t ofCode = (uint8_t)seqCode.data[0].offset;
            uint8_t mlCode = (uint8_t)seqCode.data[0].matlen;
            uint8_t llCode = (uint8_t)seqCode.data[0].litlen;
            fseEncodeSymbol<9, 0>(ofCode, fseStateBitsTableOF, fseNextStateTableOF, ofPrevStateVal, outWordOF, bitsOF);
            fseEncodeSymbol<9, 0>(mlCode, fseStateBitsTableML, fseNextStateTableML, mlPrevStateVal, outWordML, bitsML);
            fseEncodeSymbol<9, 0>(llCode, fseStateBitsTableLL, fseNextStateTableLL, llPrevStateVal, outWordLL, bitsLL);
            // Prepare output
            fseOutWord.data[0] =
                ((ap_uint<28>)outWordLL << (bitsOF + bitsML)) + ((ap_uint<28>)outWordML << bitsOF) + outWordOF;
            fseBitCnt = bitsOF + bitsML + bitsLL;
            // Write output to stream
            seqFseWordStream << fseOutWord;
            wordBitlenStream << fseBitCnt;
        }
        // encode last sequence states
        outWordML = mlPrevStateVal & c_bitMask[tableLogML];
        outWordOF = ofPrevStateVal & c_bitMask[tableLogOF];
        outWordLL = llPrevStateVal & c_bitMask[tableLogLL];
        // prepare last output
        fseOutWord.data[0] = ((ap_uint<28>)outWordLL << (tableLogOF + tableLogML)) +
                             ((ap_uint<18>)outWordOF << tableLogML) + outWordML + endMark;
        fseBitCnt = 1 + tableLogSum;
        // write last valid output for this block
        seqFseWordStream << fseOutWord;
        wordBitlenStream << fseBitCnt;
        // End of block
        fseOutWord.strobe = 0;
        seqFseWordStream << fseOutWord;
    }
    // End of all data
    fseOutWord.strobe = 0;
    seqFseWordStream << fseOutWord;
}

template <int MAX_FREQ_DWIDTH>
void seqFseBitPacker(hls::stream<IntVectorStream_dt<28, 1> >& seqFseWordStream,
                     hls::stream<ap_uint<5> >& fseWordBitlenStream,
                     hls::stream<ap_uint<33> >& extCodewordStream,
                     hls::stream<ap_uint<8> >& extBitlenStream,
                     hls::stream<IntVectorStream_dt<8, 6> >& encodedOutStream,
                     hls::stream<ap_uint<MAX_FREQ_DWIDTH> >& seqEncSizeStream) {
    // generate fse bitstream for sequences
    IntVectorStream_dt<8, 6> outVal;

seq_fse_bitPack_outer:
    while (true) {
        auto seqFseWord = seqFseWordStream.read();
        if (seqFseWord.strobe == 0) break;
        // local buffer
        ap_uint<96> bitstream = 0;
        int8_t bitCount = 0;
        ap_uint<MAX_FREQ_DWIDTH> outCnt = 0;
        // 4 bytes in an output word
        outVal.strobe = 6;
    // pack fse bitstream
    seq_fse_bit_packing:
        for (; seqFseWord.strobe > 0; seqFseWord = seqFseWordStream.read()) {
#pragma HLS PIPELINE II = 1
            // add extra bit word and then fse word
            // Read input
            auto extWord = (ap_uint<96>)extCodewordStream.read();
            auto extBlen = extBitlenStream.read();
            auto fseWord = (ap_uint<96>)seqFseWord.data[0];
            auto fseBlen = fseWordBitlenStream.read();

            bitstream += ((ap_uint<96>)fseWord << (extBlen + bitCount)) + ((ap_uint<96>)extWord << bitCount);
            bitCount += (extBlen + fseBlen);
            // push bitstream
            if (bitCount > 47) {
                // write to output stream
                outVal.data[0] = bitstream.range(7, 0);
                outVal.data[1] = bitstream.range(15, 8);
                outVal.data[2] = bitstream.range(23, 16);
                outVal.data[3] = bitstream.range(31, 24);
                outVal.data[4] = bitstream.range(39, 32);
                outVal.data[5] = bitstream.range(47, 40);
                encodedOutStream << outVal;
                bitstream >>= 48;
                bitCount -= 48;
                outCnt += 6;
            }
        }
        // write remaining bitstream
        if (bitCount) {
            // write to output stream
            outVal.data[0] = bitstream.range(7, 0);
            outVal.data[1] = bitstream.range(15, 8);
            outVal.data[2] = bitstream.range(23, 16);
            outVal.data[3] = bitstream.range(31, 24);
            outVal.data[4] = bitstream.range(39, 32);
            outVal.data[5] = bitstream.range(47, 40);
            outVal.strobe = (uint8_t)((bitCount + 7) / 8);
            encodedOutStream << outVal;
            outCnt += outVal.strobe;
        }
        // send size of encoded sequence bitstream
        seqEncSizeStream << outCnt;
        // end of block
        outVal.strobe = 0;
        encodedOutStream << outVal;
    }
    // end of all data
    outVal.strobe = 0;
    encodedOutStream << outVal;
}

template <int MAX_FREQ_DWIDTH>
void fseEncodeSequences(hls::stream<DSVectorStream_dt<Sequence_dt<MAX_FREQ_DWIDTH>, 1> >& inSeqStream,
                        hls::stream<IntVectorStream_dt<36, 1> >& fseSeqTableStream,
                        hls::stream<IntVectorStream_dt<8, 6> >& encodedOutStream,
                        hls::stream<ap_uint<MAX_FREQ_DWIDTH> >& seqEncSizeStream) {
    // internal streams
    hls::stream<DSVectorStream_dt<Sequence_dt<6>, 1> > seqCodeStream("seqCodeStream");
    hls::stream<bool> noSeqFlagStream("noSeqFlagStream");
    hls::stream<ap_uint<33> > extCodewordStream("extCodewordStream");
    hls::stream<ap_uint<8> > extBitlenStream("extBitlenStream");
    hls::stream<IntVectorStream_dt<28, 1> > seqFseWordStream("seqFseWordStream");
    hls::stream<ap_uint<5> > wordBitlenStream("wordBitlenStream");

#pragma HLS STREAM variable = seqCodeStream depth = 4
#pragma HLS STREAM variable = noSeqFlagStream depth = 4
#pragma HLS STREAM variable = extCodewordStream depth = 16
#pragma HLS STREAM variable = extBitlenStream depth = 16
#pragma HLS STREAM variable = seqFseWordStream depth = 4
#pragma HLS STREAM variable = wordBitlenStream depth = 4

#pragma HLS dataflow

    fseGetSeqCodes<MAX_FREQ_DWIDTH>(inSeqStream, seqCodeStream, noSeqFlagStream, extCodewordStream, extBitlenStream);

    fseEncodeSeqCodes(fseSeqTableStream, seqCodeStream, noSeqFlagStream, seqFseWordStream, wordBitlenStream);

    seqFseBitPacker<MAX_FREQ_DWIDTH>(seqFseWordStream, wordBitlenStream, extCodewordStream, extBitlenStream,
                                     encodedOutStream, seqEncSizeStream);
}

} // details
} // compression
} // xf
#endif
