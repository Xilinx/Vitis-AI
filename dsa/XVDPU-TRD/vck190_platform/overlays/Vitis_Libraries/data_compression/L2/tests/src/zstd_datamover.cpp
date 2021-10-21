/*
 * (c) Copyright 2019 Xilinx, Inc. All rights reserved.
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

/**
 * @file zstd_datamover.cpp
 * @brief Source file for data mover kernel which streams data to decompression
 * streaming kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "zstd_datamover.hpp"

const int c_gMemDWidth = 512;

template <uint16_t STREAMDWIDTH>
void streamDataDm2k(hls::stream<ap_uint<STREAMDWIDTH> >& in,
                    hls::stream<ap_axiu<STREAMDWIDTH, 0, 0, 0> >& inStream_dm,
                    uint32_t inputSize) {
    /**
     * @brief Write N-bit wide data of given size from hls stream to kernel axi
     * stream.
     *        N is passed as template parameter.
     *
     * @tparam STREAMDWIDTH stream data width
     *
     * @param in            input hls stream
     * @param inStream_dm   output kernel stream
     * @param inputSize     size of data in to be transferred
     *
     */
    constexpr uint8_t c_streamBytes = (STREAMDWIDTH / 8);
    uint8_t lastValid = inputSize % c_streamBytes;
    ap_axiu<STREAMDWIDTH, 0, 0, 0> dataIn;
    // read data from input hls to input stream for decompression kernel
    uint32_t itrLim = 1 + (inputSize - 1) / c_streamBytes;
    for (uint32_t i = 0; i < itrLim - 1; ++i) {
#pragma HLS PIPELINE II = 1
        auto temp = in.read();
        dataIn.data = temp; // kernel to kernel data transfer
        dataIn.keep = -1;
        dataIn.last = false;
        inStream_dm << dataIn;
    }
    // write last word
    auto temp = in.read();
    dataIn.data = temp;
    dataIn.keep = (1 << c_streamBytes) - 1;
    dataIn.last = true;
    inStream_dm << dataIn;
}

void __xf_zstd_datamover(xf::compression::uintMemWidth_t* in,
                         xf::compression::uintMemWidth_t* out,
                         uint32_t inputSize,
                         uint32_t* outputSize,
                         hls::stream<ap_axiu<STREAM_IN_DWIDTH, 0, 0, 0> >& origStream,
                         hls::stream<ap_axiu<STREAM_OUT_DWIDTH, 0, 0, 0> >& destStream) {
    constexpr int c_gmemParallelBytes = STREAM_OUT_DWIDTH / 8;

    hls::stream<xf::compression::uintMemWidth_t> instream512("inputStream");
    hls::stream<ap_uint<STREAM_IN_DWIDTH> > outdownstream("outDownStream");
    hls::stream<ap_uint<STREAM_OUT_DWIDTH> > decompoutstream("decompoutstream");
    hls::stream<bool> decompressedStreamEoS("decompressedStreamEoS");
    hls::stream<xf::compression::uintMemWidth_t> outstream512("outputStream");
    hls::stream<bool> outStreamEoS("outStreamEoS");
    hls::stream<uint32_t> decompressSizeStream("decompressSizeStream");

#pragma HLS STREAM variable = outdownstream depth = 32
#pragma HLS STREAM variable = decompoutstream depth = 32
#pragma HLS STREAM variable = decompressedStreamEoS depth = 32
#pragma HLS STREAM variable = decompressSizeStream depth = 32
#pragma HLS STREAM variable = instream512 depth = 32
#pragma HLS STREAM variable = outstream512 depth = 32
#pragma HLS STREAM variable = outStreamEoS depth = 32

#pragma HLS dataflow
    xf::compression::details::mm2sSimple<c_gMemDWidth>(in, instream512, inputSize);
    xf::compression::details::streamDownsizer<uint32_t, c_gMemDWidth, STREAM_IN_DWIDTH>(instream512, outdownstream,
                                                                                        inputSize);

    streamDataDm2k<STREAM_IN_DWIDTH>(outdownstream, origStream, inputSize);
    xf::compression::details::streamDataK2dmMultiByteStrobe<c_gmemParallelBytes>(decompoutstream, decompressedStreamEoS,
                                                                                 decompressSizeStream, destStream);

    xf::compression::details::upsizerEos<STREAM_OUT_DWIDTH, c_gMemDWidth>(decompoutstream, decompressedStreamEoS,
                                                                          outstream512, outStreamEoS);
    xf::compression::details::s2mmEosSimple<c_gMemDWidth, 16>(out, outstream512, outStreamEoS, decompressSizeStream,
                                                              outputSize);
}

extern "C" {
void xilZstdDataMover(xf::compression::uintMemWidth_t* in,
                      xf::compression::uintMemWidth_t* out,
                      uint32_t inputSize,
                      uint32_t* outputSize,
                      hls::stream<ap_axiu<STREAM_IN_DWIDTH, 0, 0, 0> >& origStream,
                      hls::stream<ap_axiu<STREAM_OUT_DWIDTH, 0, 0, 0> >& destStream) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = outputSize offset = slave bundle = gmem
#pragma HLS interface axis port = origStream
#pragma HLS interface axis port = destStream
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = inputSize bundle = control
#pragma HLS INTERFACE s_axilite port = outputSize bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    // Transfer Data to and from compression kernels
    __xf_zstd_datamover(in, out, inputSize, outputSize, origStream, destStream);
}
}
