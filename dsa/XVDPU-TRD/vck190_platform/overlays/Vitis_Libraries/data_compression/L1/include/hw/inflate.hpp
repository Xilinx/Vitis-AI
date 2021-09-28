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
#ifndef _XFCOMPRESSION_INFLATE_HPP_
#define _XFCOMPRESSION_INFLATE_HPP_

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "huffman_decoder.hpp"
#include "lz_decompress.hpp"
#include "stream_upsizer.hpp"
#include "stream_downsizer.hpp"
#include "mm2s.hpp"
#include "s2mm.hpp"

#include <ap_int.h>
#include <assert.h>
#include <stdint.h>

namespace xf {
namespace compression {

namespace details {

template <int PARALLEL_BYTES>
void lzLiteralUpsizer(hls::stream<ap_uint<10> >& inStream, hls::stream<ap_uint<PARALLEL_BYTES * 8> >& litStream) {
    const uint8_t c_parallelBit = PARALLEL_BYTES * 8;
    const uint8_t c_maxLitLen = 128;
    ap_uint<c_parallelBit> outBuffer;
    ap_uint<4> idx = 0;

    ap_uint<2> status = 0;
    ap_uint<10> val;
    bool done = false;
lzliteralUpsizer:
    while (status != 2) {
#pragma HLS PIPELINE II = 1
        status = 0;
        val = inStream.read();
        status = val.range(1, 0);
        outBuffer.range((idx + 1) * 8 - 1, idx * 8) = val.range(9, 2);
        idx++;

        if ((status & 1) || (idx == 8)) {
            if (status != 3) {
                litStream << outBuffer;
            }
            idx = 0;
        }
    }
    if (idx > 1) {
        litStream << outBuffer;
        idx = 0;
    }
}

template <int PARALLEL_BYTES>
void lzLiteralUpsizerLL(hls::stream<ap_uint<10> >& inStream, hls::stream<ap_uint<PARALLEL_BYTES * 8> >& litStream) {
    const uint8_t c_parallelBit = PARALLEL_BYTES * 8;
    const uint8_t c_maxLitLen = 128;
    ap_uint<c_parallelBit> outBuffer;
    ap_uint<4> idx = 0;

    ap_uint<2> status = 0;
    ap_uint<10> val;
    bool done = false;
lzliteralUpsizer:
    while (status != 2) {
#pragma HLS PIPELINE II = 1
        status = 0;
        val = inStream.read();
        status = val.range(1, 0);
        outBuffer.range((idx + 1) * 8 - 1, idx * 8) = val.range(9, 2);
        idx++;

        if ((status & 1) || (idx == 8)) {
            if (idx > 1) {
                litStream << outBuffer;
            }
            idx = 0;
        }
    }
    if (idx > 1) {
        litStream << outBuffer;
        idx = 0;
    }
}

template <class SIZE_DT = uint8_t>
void lzProcessingUnit(hls::stream<ap_uint<17> >& inStream,
                      hls::stream<SIZE_DT>& litLenStream,
                      hls::stream<SIZE_DT>& matchLenStream,
                      hls::stream<ap_uint<16> >& offsetStream,
                      hls::stream<ap_uint<10> >& outStream) {
    ap_uint<17> inValue, nextValue;
    const int c_maxLitLen = 128;
    uint16_t offset = 0;
    uint16_t matchLen = 0;
    uint8_t litLen = 0;
    uint8_t outLitLen = 0;
    ap_uint<10> lit = 0;

    nextValue = inStream.read();
    bool eosFlag = nextValue.range(0, 0);
    bool lastLiteral = false;
    bool isLiteral = true;
lzProcessing:
    for (; eosFlag == false;) {
#pragma HLS PIPELINE II = 1
        inValue = nextValue;
        nextValue = inStream.read();
        eosFlag = nextValue.range(0, 0);

        bool outFlag, outStreamFlag;
        if (inValue.range(16, 9) == 0xFF && isLiteral) {
            outStreamFlag = true;
            outLitLen = litLen + 1;
            if (litLen == c_maxLitLen - 1) {
                outFlag = true;
                matchLen = 0;
                offset = 1; // dummy value
                litLen = 0;
            } else {
                outFlag = false;
                litLen++;
            }
        } else {
            if (isLiteral) {
                matchLen = inValue.range(16, 1);
                isLiteral = false;
                outFlag = false;
                outStreamFlag = false;
            } else {
                offset = inValue.range(16, 1);
                isLiteral = true;
                outFlag = true;
                outLitLen = litLen;
                litLen = 0;
                outStreamFlag = false;
            }
        }
        if (outStreamFlag) {
            lit.range(9, 2) = inValue.range(8, 1);
            if (nextValue.range(16, 9) == 0xFF) {
                lit.range(1, 0) = 0;
            } else {
                lit.range(1, 0) = 1;
            }
            lastLiteral = true;
            outStream << lit;
        } else if (lastLiteral) {
            outStream << 3;
            lastLiteral = false;
        }

        if (outFlag) {
            litLenStream << outLitLen;
            offsetStream << offset;
            matchLenStream << matchLen;
        }
    }

    if (litLen) {
        litLenStream << litLen;
        offsetStream << 0;
        matchLenStream << 0;
    }

    // Terminate condition
    outStream << 2;
    offsetStream << 0;
    matchLenStream << 0;
    litLenStream << 0;
}

template <class SIZE_DT = uint8_t>
void lzProcessingUnitLL(hls::stream<ap_uint<16> >& inStream,
                        hls::stream<SIZE_DT>& litLenStream,
                        hls::stream<SIZE_DT>& matchLenStream,
                        hls::stream<ap_uint<16> >& offsetStream,
                        hls::stream<ap_uint<10> >& outStream) {
    ap_uint<16> inValue, nextValue;
    const int c_maxLitLen = 128;
    uint16_t offset = 0;
    uint16_t matchLen = 0;
    uint8_t litLen = 0;
    uint8_t outLitLen = 0;
    ap_uint<10> lit = 0;
    const uint16_t lbase[32] = {0,  3,  4,  5,  6,  7,  8,  9,  10,  11,  13,  15,  17,  19,  23, 27,
                                31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 0,  0};

    const uint16_t dbase[32] = {1,    2,    3,    4,    5,    7,     9,     13,    17,  25,   33,
                                49,   65,   97,   129,  193,  257,   385,   513,   769, 1025, 1537,
                                2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 0,   0};
    nextValue = inStream.read();
    bool eosFlag = (nextValue == 0xFFFF) ? true : false;
    bool lastLiteral = false;
    bool isLitLength = true;
    bool isExtra = false;
    bool dummyValue = false;
lzProcessing:
    for (; eosFlag == false;) {
#pragma HLS PIPELINE II = 1
        inValue = nextValue;
        nextValue = inStream.read();
        eosFlag = (nextValue == 0xFFFF);

        bool outFlag, outStreamFlag;
        if ((inValue.range(15, 8) == 0xFE) || (inValue.range(15, 8) == 0xFD)) {
            // ignore invalid byte
            outFlag = false;
            outStreamFlag = false;
        } else if (inValue.range(15, 8) == 0xF0) {
            outStreamFlag = true;
            outLitLen = litLen + 1;
            if (litLen == c_maxLitLen - 1) {
                outFlag = true;
                matchLen = 0;
                offset = 1; // dummy value
                litLen = 0;
            } else {
                outFlag = false;
                matchLen = 0;
                offset = 1; // dummy value
                litLen++;
            }
        } else if (isExtra && isLitLength) { // matchLen Extra
            matchLen += inValue.range(15, 0);
            isExtra = false;
            isLitLength = false;
            outStreamFlag = true;

        } else if (isExtra && !isLitLength) { // offset Extra
            offset += inValue.range(15, 0);
            isExtra = false;
            isLitLength = true;
            outFlag = true;
            outStreamFlag = true;
        } else if (isLitLength) {
            auto val = inValue.range(4, 0);
            matchLen = lbase[val];
            if (val < 9) {
                isExtra = false;
                isLitLength = false;
            } else {
                isExtra = true;
                isLitLength = true;
            }
            outFlag = false;
            outStreamFlag = true;
            dummyValue = true;
        } else {
            auto val = inValue.range(4, 0);
            offset = dbase[val];
            if (val < 4) {
                isExtra = false;
                isLitLength = true;
                outFlag = true;

            } else {
                isExtra = true;
                isLitLength = false;
                outFlag = false;
            }

            outLitLen = litLen;
            litLen = 0;
            outStreamFlag = true;
        }

        if (outStreamFlag) {
            lit.range(9, 2) = inValue.range(7, 0);
            if ((inValue.range(15, 8) == 0xF0)) {
                lit.range(1, 0) = 0;
            } else {
                lit.range(1, 0) = 1;
            }
            outStream << lit;
        }

        if (outFlag) {
            litLenStream << outLitLen;
            offsetStream << offset;
            matchLenStream << matchLen;
        }
    }

    if (litLen) {
        litLenStream << litLen;
        offsetStream << 0;
        matchLenStream << 0;
        outStream << 3;
    }

    // Terminate condition
    outStream << 2;
    offsetStream << 0;
    matchLenStream << 0;
    litLenStream << 0;
}

template <int STREAM_WIDTH>
void kStreamReadZlibDecomp(hls::stream<ap_axiu<STREAM_WIDTH, 0, 0, 0> >& in,
                           hls::stream<ap_uint<STREAM_WIDTH> >& out,
                           hls::stream<bool>& outEos) {
    /**
     * @brief kStreamReadZlibDecomp Read 16-bit wide data from internal streams output by compression modules
     *                              and write to output axi stream.
     *
     * @param inKStream     input kernel stream
     * @param readStream    internal stream to be read for processing
     * @param input_size    input data size
     *
     */
    bool last = false;
    while (last == false) {
#pragma HLS PIPELINE II = 1
        ap_axiu<STREAM_WIDTH, 0, 0, 0> tmp = in.read();
        out << tmp.data;
        last = tmp.last;
        outEos << 0;
    }
    out << 0;
    outEos << 1; // Terminate condition
}

template <int STREAM_WIDTH>
void kStreamWriteZlibDecomp(hls::stream<ap_axiu<STREAM_WIDTH, 0, 0, 0> >& outKStream,
                            hls::stream<ap_uint<STREAM_WIDTH + (STREAM_WIDTH / 8)> >& outDataStream) {
    /**
     * @brief kStreamWriteZlibDecomp Read 16-bit wide data from internal streams output by compression modules
     *                                and write to output axi stream.
     *
     * @param outKStream    output kernel stream
     * @param outDataStream output data stream from internal modules
     *
     */
    ap_uint<STREAM_WIDTH / 8> strb = 0;
    ap_uint<STREAM_WIDTH> data;
    ap_uint<STREAM_WIDTH + (STREAM_WIDTH / 8)> tmp;
    ap_axiu<STREAM_WIDTH, 0, 0, 0> t1;

    tmp = outDataStream.read();
    strb = tmp.range((STREAM_WIDTH / 8) - 1, 0);
    t1.data = tmp.range(STREAM_WIDTH + (STREAM_WIDTH / 8) - 1, STREAM_WIDTH / 8);
    t1.strb = strb;
    t1.keep = strb;
    t1.last = 0;
    if (strb == 0) {
        t1.last = 1;
        outKStream << t1;
    }
    while (strb != 0) {
#pragma HLS PIPELINE II = 1
        tmp = outDataStream.read();
        strb = tmp.range((STREAM_WIDTH / 8) - 1, 0);
        if (strb == 0) {
            t1.last = 1;
        }
        outKStream << t1;
        t1.data = tmp.range(STREAM_WIDTH + (STREAM_WIDTH / 8) - 1, STREAM_WIDTH / 8);
        t1.strb = strb;
        t1.keep = strb;
        t1.last = 0;
    }
}

template <int DECODER, int PARALLEL_BYTES, bool LOW_LATENCY = false, int HISTORY_SIZE = (32 * 1024)>
void inflateMultiByteCore(hls::stream<ap_uint<16> >& inStream,
                          hls::stream<bool>& inEos,
                          hls::stream<ap_uint<(PARALLEL_BYTES * 8) + PARALLEL_BYTES> >& outStream) {
    const int c_parallelBit = PARALLEL_BYTES * 8;
    const eHuffmanType c_decoderType = (eHuffmanType)DECODER;

    hls::stream<ap_uint<17> > bitunpackstream("bitUnPackStream");
    hls::stream<ap_uint<16> > bitunpackstreamLL("bitUnPackStreamLL");
    hls::stream<ap_uint<c_parallelBit> > litStream("litStream");
    hls::stream<ap_uint<9> > matchLenStream("matchLenStream");
    hls::stream<ap_uint<9> > litLenStream("litLenStream");
    hls::stream<ap_uint<16> > offsetStream("offsetStream");
    hls::stream<ap_uint<10> > lzProcOutStream("lzProcOutStream");

#pragma HLS STREAM variable = litStream depth = 32
#pragma HLS STREAM variable = lzProcOutStream depth = 4
#pragma HLS STREAM variable = bitunpackstream depth = 1024
#pragma HLS STREAM variable = litLenStream depth = 16
#pragma HLS STREAM variable = matchLenStream depth = 16
#pragma HLS STREAM variable = offsetStream depth = 16

#pragma HLS BIND_STORAGE variable = litStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = bitunpackstream type = FIFO impl = BRAM
#pragma HLS BIND_STORAGE variable = lzProcOutStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = litLenStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = matchLenStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = offsetStream type = FIFO impl = SRL

#pragma HLS dataflow

    if (LOW_LATENCY) {
        xf::compression::huffmanDecoderLL<c_decoderType>(inStream, inEos, bitunpackstreamLL);

        xf::compression::details::lzProcessingUnitLL<ap_uint<9> >(bitunpackstreamLL, litLenStream, matchLenStream,
                                                                  offsetStream, lzProcOutStream);

        xf::compression::details::lzLiteralUpsizerLL<PARALLEL_BYTES>(lzProcOutStream, litStream);
    } else {
        xf::compression::huffmanDecoder<c_decoderType>(inStream, inEos, bitunpackstream);

        xf::compression::details::lzProcessingUnit<ap_uint<9> >(bitunpackstream, litLenStream, matchLenStream,
                                                                offsetStream, lzProcOutStream);

        xf::compression::details::lzLiteralUpsizer<PARALLEL_BYTES>(lzProcOutStream, litStream);
    }
    xf::compression::lzMultiByteDecompress<PARALLEL_BYTES, HISTORY_SIZE, ap_uint<9> >(
        litLenStream, litStream, offsetStream, matchLenStream, outStream);
}

} // namespace details

template <int DECODER, int PARALLEL_BYTES, bool LOW_LATENCY = false, int HISTORY_SIZE = (32 * 1024)>
void inflateMultiByte(hls::stream<ap_axiu<16, 0, 0, 0> >& inaxistream,
                      hls::stream<ap_axiu<PARALLEL_BYTES * 8, 0, 0, 0> >& outaxistream) {
    const int c_parallelBit = PARALLEL_BYTES * 8;

    hls::stream<ap_uint<16> > axi2HlsStrm("axi2HlsStrm");
    hls::stream<bool> axi2HlsEos("axi2HlsEos");
    hls::stream<ap_uint<c_parallelBit + PARALLEL_BYTES> > inflateOut("inflateOut");
    hls::stream<uint64_t> outSizeStream("outSizeStream");

#pragma HLS STREAM variable = axi2HlsStrm depth = 32
#pragma HLS STREAM variable = axi2HlsEos depth = 32
#pragma HLS STREAM variable = inflateOut depth = 32
//#pragma HLS STREAM variable = inflateOut depth = 4096

#pragma HLS BIND_STORAGE variable = axi2HlsStrm type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = axi2HlsEos type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = inflateOut type = fifo impl = SRL

#pragma HLS dataflow
    details::kStreamReadZlibDecomp<16>(inaxistream, axi2HlsStrm, axi2HlsEos);

    details::inflateMultiByteCore<DECODER, PARALLEL_BYTES, LOW_LATENCY, HISTORY_SIZE>(axi2HlsStrm, axi2HlsEos,
                                                                                      inflateOut);

    details::kStreamWriteZlibDecomp<c_parallelBit>(outaxistream, inflateOut);
}

} // namespace compression
} // namespace xf
#endif // _XFCOMPRESSION_INFLATE_HPP_
