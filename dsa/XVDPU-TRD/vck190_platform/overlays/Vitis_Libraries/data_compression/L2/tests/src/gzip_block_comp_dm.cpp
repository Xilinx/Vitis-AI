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

/**
 * @file gzip_block_comp_dm.cpp
 * @brief Source file for data mover kernel which streams data to decompression
 * streaming kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "gzip_block_comp_dm.hpp"

const int c_gmemParallelBytes = GMEM_OUT_DWIDTH / 8;
const int c_gMemDWidth = 512;

template <uint16_t STREAMDWIDTH>
void streamDataDm2kSize(hls::stream<ap_uint<STREAMDWIDTH> >& in,
                        hls::stream<ap_axiu<STREAMDWIDTH, 0, 0, 0> >& inStream_dm,
                        hls::stream<ap_axiu<32, 0, 0, 0> >& inStreamSize_dm,
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
     * @param inStreamSize_dm   output data size kernel stream
     * @param inputSize     size of data in to be transferred
     *
     */
    ap_axiu<32, 0, 0, 0> isize;
    isize.data = inputSize;
    isize.last = false;
    inStreamSize_dm << isize;
    isize.data = 0;
    isize.last = true;
    inStreamSize_dm << isize;

    // read data from input hls to input stream for decompression kernel
    uint32_t itrLim = 1 + (inputSize - 1) / (STREAMDWIDTH / 8);
    for (uint32_t i = 0; i < itrLim; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<STREAMDWIDTH> temp = in.read();
        ap_axiu<STREAMDWIDTH, 0, 0, 0> dataIn;
        dataIn.data = temp; // kernel to kernel data transfer
        inStream_dm << dataIn;
    }
}

void __xf_comp_datamover(xf::compression::uintMemWidth_t* in,
                         xf::compression::uintMemWidth_t* out,
                         uint32_t input_size,
                         uint32_t* outputSize,
                         hls::stream<ap_axiu<GMEM_IN_DWIDTH, 0, 0, 0> >& instream_orig,
                         hls::stream<ap_axiu<32, 0, 0, 0> >& instream_size,
                         hls::stream<ap_axiu<GMEM_OUT_DWIDTH, 0, 0, 0> >& outstream_dest) {
    hls::stream<xf::compression::uintMemWidth_t> instream512("inputStream");
    hls::stream<ap_uint<GMEM_IN_DWIDTH> > outdownstream("outDownStream");
    hls::stream<ap_uint<GMEM_OUT_DWIDTH> > decompoutstream("decompoutstream");
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
    xf::compression::details::mm2sSimple<c_gMemDWidth>(in, instream512, input_size);
    xf::compression::details::streamDownsizer<uint32_t, c_gMemDWidth, GMEM_IN_DWIDTH>(instream512, outdownstream,
                                                                                      input_size);

    streamDataDm2kSize<GMEM_IN_DWIDTH>(outdownstream, instream_orig, instream_size, input_size);
    xf::compression::details::streamDataK2dmMultiByte<c_gmemParallelBytes>(decompoutstream, decompressedStreamEoS,
                                                                           decompressSizeStream, outstream_dest);

    xf::compression::details::upsizerEos<GMEM_OUT_DWIDTH, c_gMemDWidth>(decompoutstream, decompressedStreamEoS,
                                                                        outstream512, outStreamEoS);
    xf::compression::details::s2mmEosSimple<c_gMemDWidth, 16>(out, outstream512, outStreamEoS, decompressSizeStream,
                                                              outputSize);
}

extern "C" {
void xilCompDatamover(xf::compression::uintMemWidth_t* in,
                      xf::compression::uintMemWidth_t* out,
                      uint32_t inputSize,
                      uint32_t* outputSize,
                      hls::stream<ap_axiu<GMEM_IN_DWIDTH, 0, 0, 0> >& instream_orig,
                      hls::stream<ap_axiu<32, 0, 0, 0> >& instream_size,
                      hls::stream<ap_axiu<GMEM_OUT_DWIDTH, 0, 0, 0> >& outstream_dest) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = outputSize offset = slave bundle = gmem
#pragma HLS interface axis port = instream_orig
#pragma HLS interface axis port = instream_size
#pragma HLS interface axis port = outstream_dest
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = input_size bundle = control
#pragma HLS INTERFACE s_axilite port = outputSize bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    // Transfer Data to and from compression kernels
    __xf_comp_datamover(in, out, inputSize, outputSize, instream_orig, instream_size, outstream_dest);
}
}
