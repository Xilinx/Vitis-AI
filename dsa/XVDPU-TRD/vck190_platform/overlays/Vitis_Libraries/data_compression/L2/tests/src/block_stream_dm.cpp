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
 * @file block_stream_dm.cpp
 * @brief Source file for data mover kernel which streams data to the streaming
 * kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "block_stream_dm.hpp"

const int factor = GMEM_DWIDTH / 8;

// Free running Process
void dataWrapper(hls::stream<ap_uint<GMEM_DWIDTH> >& mm2sStream,
                 hls::stream<uint32_t>& sizeStream,
                 hls::stream<ap_uint<GMEM_DWIDTH> >& outStream,
                 hls::stream<bool>& outEos,
                 hls::stream<uint32_t>& outSizeStream,
                 hls::stream<ap_axiu<GMEM_DWIDTH, 0, 0, 0> >& instream_orig,
                 hls::stream<ap_axiu<GMEM_DWIDTH, 0, 0, 0> >& outstream_dest) {
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS dataflow
    // HLS 2 AXI
    xf::compression::details::streamDm2k<GMEM_DWIDTH, uint32_t, 32>(mm2sStream, sizeStream, instream_orig);

    // AXI 2 HLS
    xf::compression::details::streamK2Dm<factor, uint32_t, 32>(outStream, outEos, outSizeStream, outstream_dest);
}

// Top Function
extern "C" {
void xilDataMover(uintDataWidth* in,
                  uintDataWidth* out,
                  uint32_t input_size,
                  uint32_t* compressed_size,
                  hls::stream<ap_axiu<GMEM_DWIDTH, 0, 0, 0> >& instream_orig,
                  hls::stream<ap_axiu<GMEM_DWIDTH, 0, 0, 0> >& outstream_dest) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem max_write_burst_length = 64
#pragma HLS INTERFACE m_axi port = compressed_size offset = slave bundle = gmem
#pragma HLS interface axis port = instream_orig
#pragma HLS interface axis port = outstream_dest
#pragma HLS INTERFACE s_axilite port = input_size
#pragma HLS INTERFACE ap_ctrl_chain port = return

    // Internal Streams
    hls::stream<ap_uint<GMEM_DWIDTH> > mm2sStream;
    hls::stream<uint32_t> sizeStream;
    hls::stream<uint32_t> sizeStreamV;
    hls::stream<ap_uint<GMEM_DWIDTH> > outStream;
    hls::stream<bool> outEos;
    hls::stream<uint32_t> outSizeStream;

    // Initialize Size Stream
    uint32_t tmp = input_size;
    sizeStreamV.write(tmp);

#pragma HLS STREAM variable = mm2sStream depth = 32
#pragma HLS STREAM variable = sizeStream depth = 32
#pragma HLS STREAM variable = sizeStreamV depth = 32
#pragma HLS STREAM variable = outStream depth = 32
#pragma HLS STREAM variable = outEos depth = 32
#pragma HLS STREAM variable = outSizeStream depth = 32

#pragma HLS BIND_STORAGE variable = mm2sStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = sizeStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = sizeStreamV type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = outStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = outEos type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = outSizeStream type = FIFO impl = SRL

#pragma HLS dataflow
    // Memory Read to HLS Streams
    xf::compression::details::mm2sSimple<GMEM_DWIDTH, GMEM_BURST_SIZE>(in, mm2sStream, sizeStream, sizeStreamV);

    // Frer running process to convert HLS to AXI streams
    dataWrapper(mm2sStream, sizeStream, outStream, outEos, outSizeStream, instream_orig, outstream_dest);

    // HLS Streams to Memory Write
    xf::compression::details::s2mmEosSimple<GMEM_DWIDTH, GMEM_BURST_SIZE, uint32_t>(out, outStream, outEos,
                                                                                    outSizeStream, compressed_size);
}
}
