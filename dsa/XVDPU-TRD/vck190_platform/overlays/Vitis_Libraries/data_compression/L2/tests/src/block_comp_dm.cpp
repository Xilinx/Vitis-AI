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
 * @file xil_block_comp_datamover_kernel.cpp
 * @brief Source file for data mover kernel which streams data to compression
 * streaming kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "block_comp_dm.hpp"
const int kGMemBurstSize = 16;
const int kGMemDWidth = 512;

void __xf_comp_datamover(xf::compression::uintMemWidth_t* in,
                         xf::compression::uintMemWidth_t* out,
                         uint32_t* compressed_size,
                         uint32_t input_size,
                         hls::stream<ap_axiu<8, 0, 0, 0> >& instream_orig,
                         hls::stream<ap_axiu<8, 0, 0, 0> >& outstream_dest) {
    hls::stream<xf::compression::uintMemWidth_t> instream512("inputStream");
    hls::stream<ap_uint<8> > outdownstream("outDownStream");
    hls::stream<ap_uint<8> > compoutstream("compoutstream");
    hls::stream<bool> lz4OutEos("lz4OutEos");
    hls::stream<xf::compression::uintMemWidth_t> outstream512("outputStream");
    hls::stream<bool> outstream512_eos("outputStreamSize");

#pragma HLS STREAM variable = outdownstream depth = 2
#pragma HLS STREAM variable = compoutstream depth = 2
// compoutstream pragma is experimental
#pragma HLS STREAM variable = instream512 depth = 8
#pragma HLS STREAM variable = outstream512 depth = 8
#pragma HLS STREAM variable = outstream512_eos depth = 8
#pragma HLS STREAM variable = lz4OutEos depth = 8

#pragma HLS BIND_STORAGE variable = instream512 type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = outstream512 type = FIFO impl = SRL

    hls::stream<uint32_t> compSizeVal;
#pragma HLS STREAM variable = compSizeVal depth = 2

#pragma HLS dataflow
    xf::compression::details::mm2sSimple<kGMemDWidth>(in, instream512, input_size);
    xf::compression::details::streamDownsizer<uint32_t, kGMemDWidth, 8>(instream512, outdownstream, input_size);

    xf::compression::details::streamDataDm2k<8>(outdownstream, instream_orig, input_size);
    xf::compression::details::streamDataK2dm(compoutstream, lz4OutEos, compSizeVal, outstream_dest);

    xf::compression::details::upsizerEos<8, kGMemDWidth>(compoutstream, lz4OutEos, outstream512, outstream512_eos);
    xf::compression::details::s2mmEosSimple<kGMemDWidth, 1>(out, outstream512, outstream512_eos, compSizeVal,
                                                            compressed_size);
}

extern "C" {
/**
 * @brief Data mover kernel top function for block based compression algorithms.
 * It reads
 *        data from memory and streams it to block compression kernel.
 *
 * @param in input stream
 * @param out output stream
 * @param compressed_size decompressed size output
 * @param input_size input size (block size or less)
 * @param instream_orig input axi kernel stream (written by this kernel)
 * @param outstream_dest output axi kernel stream (read by this kernel)
 *
 */
void xilCompDatamover(xf::compression::uintMemWidth_t* in,
                      xf::compression::uintMemWidth_t* out,
                      uint32_t* compressed_size,
                      uint32_t input_size,
                      hls::stream<ap_axiu<8, 0, 0, 0> >& instream_orig,
                      hls::stream<ap_axiu<8, 0, 0, 0> >& outstream_dest) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = compressed_size offset = slave bundle = gmem
#pragma HLS interface axis port = instream_orig
#pragma HLS interface axis port = outstream_dest
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = compressed_size bundle = control
#pragma HLS INTERFACE s_axilite port = input_size bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    // Transfer Data to and from compression kernels
    __xf_comp_datamover(in, out, compressed_size, input_size, instream_orig, outstream_dest);
}
}
