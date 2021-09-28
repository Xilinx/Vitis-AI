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
 * @file xil_block_decomp_datamover_kernel.cpp
 * @brief Source file for data mover kernel which streams data to decompression
 * streaming kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "block_decomp_dm.hpp"

const int kGMemBurstSize = 16;
const int kGMemDWidth = 512;
void __xf_decomp_datamover(xf::compression::uintMemWidth_t* in,
                           xf::compression::uintMemWidth_t* out,
                           uint32_t decompressed_size,
                           uint32_t input_size,
                           hls::stream<ap_axiu<8, 0, 0, 0> >& instream_orig,
                           hls::stream<ap_axiu<8, 0, 0, 0> >& outstream_dest) {
    hls::stream<xf::compression::uintMemWidth_t> instream512("inputStream");
    hls::stream<ap_uint<8> > outdownstream("outDownStream");
    hls::stream<ap_uint<8> > decompoutstream("decompoutstream");
    hls::stream<xf::compression::uintMemWidth_t> outstream512("outputStream");

#pragma HLS STREAM variable = outdownstream depth = 2
#pragma HLS STREAM variable = decompoutstream depth = 2
#pragma HLS STREAM variable = instream512 depth = 32
#pragma HLS STREAM variable = outstream512 depth = 32

#pragma HLS dataflow
    xf::compression::details::mm2sSimple<kGMemDWidth>(in, instream512, input_size);
    xf::compression::details::streamDownsizer<uint32_t, kGMemDWidth, 8>(instream512, outdownstream, input_size);

    xf::compression::details::streamDataDm2k<8>(outdownstream, instream_orig, input_size);
    xf::compression::details::streamDataK2dmFixedSize(decompoutstream, outstream_dest, decompressed_size);

    xf::compression::details::streamUpsizer<uint32_t, 8, kGMemDWidth>(decompoutstream, outstream512, decompressed_size);
    xf::compression::details::s2mmSimple<kGMemDWidth>(out, outstream512, decompressed_size);
}

extern "C" {
void xilDecompDatamover(xf::compression::uintMemWidth_t* in,
                        xf::compression::uintMemWidth_t* out,
                        uint32_t decompressed_size,
                        uint32_t input_size,
                        hls::stream<ap_axiu<8, 0, 0, 0> >& instream_orig,
                        hls::stream<ap_axiu<8, 0, 0, 0> >& outstream_dest) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem
#pragma HLS interface axis port = instream_orig
#pragma HLS interface axis port = outstream_dest
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = decompressed_size bundle = control
#pragma HLS INTERFACE s_axilite port = input_size bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    // Transfer Data to and from compression kernels
    __xf_decomp_datamover(in, out, decompressed_size, input_size, instream_orig, outstream_dest);
}
}
