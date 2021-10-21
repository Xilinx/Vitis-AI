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
 * @file zlib_dm_wr.cpp
 * @brief Source for data writer kernel for streaming data to zlib decompression
 * streaming kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "gzip_dm_mm2s.hpp"

template <int STREAMDWIDTH>
void streamDataDm2kSync(hls::stream<ap_uint<STREAMDWIDTH> >& in,
                        hls::stream<ap_axiu<STREAMDWIDTH, 0, 0, 0> >& inStream_dm,
                        uint32_t inputSize,
                        uint32_t last) {
    // read data from input hls to input stream for decompression kernel
    auto swidth = STREAMDWIDTH / 8;
    uint32_t itrLim = 1 + (inputSize - 1) / swidth;
    uint8_t strb = (1 << (inputSize % swidth)) - 1;
streamDataDm2kSync:
    for (uint32_t i = 0; i < itrLim; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<STREAMDWIDTH> temp = in.read();
        ap_axiu<STREAMDWIDTH, 0, 0, 0> dataIn;
        dataIn.data = temp; // kernel to kernel data transfer
        dataIn.last = 0;
        dataIn.strb = -1;
        if (i == itrLim - 1) {
            dataIn.last = (ap_uint<1>)last;
            dataIn.strb = strb;
        }
        inStream_dm.write(dataIn);
    }
}

extern "C" {
void xilGzipMM2S(uintMemWidth_t* in,
                 uint32_t inputSize,
                 uint32_t last,
                 hls::stream<ap_axiu<c_inStreamDwidth, 0, 0, 0> >& outStream) {
    const int c_gmem0_width = c_inStreamDwidth;
#pragma HLS INTERFACE m_axi port = in max_widen_bitwidth = c_gmem0_width offset = slave bundle = \
    gmem0 max_read_burst_length = 64 max_write_burst_length = 2 num_read_outstanding = 8 num_write_outstanding = 1
#pragma HLS interface axis port = outStream
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = inputSize bundle = control
#pragma HLS INTERFACE s_axilite port = last bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS_INTERFACE ap_ctrl_chain port = return bundle = control

    hls::stream<uintMemWidth_t> inHlsStream("inputStream");
    hls::stream<ap_uint<c_inStreamDwidth> > outdownstream("outDownStream");
#pragma HLS STREAM variable = inHlsStream depth = 512
#pragma HLS STREAM variable = outdownstream depth = 4
#pragma HLS BIND_STORAGE variable = outdownstream type = FIFO impl = SRL

#pragma HLS dataflow
    xf::compression::details::mm2sSimple<MULTIPLE_BYTES * 8>(in, inHlsStream, inputSize);

    xf::compression::details::streamDownsizer<uint32_t, MULTIPLE_BYTES * 8, c_inStreamDwidth>(inHlsStream,
                                                                                              outdownstream, inputSize);

    streamDataDm2kSync<c_inStreamDwidth>(outdownstream, outStream, inputSize, last);
}
}
