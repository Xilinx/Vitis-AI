/*
 * Copyright 2019 Xilinx, Inc.
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
 */

#include "crc32_kernel.hpp"

void readDataM2S(int n, ap_uint<512>* inData, hls::stream<ap_uint<512> >& outDataStrm) {
    for (int i = 0; i < (n + K - 1) / K; i++) {
#pragma HLS pipeline ii = 1
        outDataStrm.write(inData[i]);
    }
}

void splitStrm(int n, hls::stream<ap_uint<512> >& inStrm, hls::stream<ap_uint<W * 8> >& outStrm) {
    ap_uint<512> tmp;
    for (int i = 0; i < n; i++) {
#pragma HLS pipeline ii = 1
        int offset = i % K;
        if (offset == 0) {
            tmp = inStrm.read();
        }
        outStrm.write(tmp(offset * (8 * W) + 8 * W - 1, offset * (8 * W)));
    }
}

void readLenM2S(int n,
                ap_uint<32>* inData,
                ap_uint<32>* inData2,
                hls::stream<ap_uint<32> >& outDataStrm,
                hls::stream<ap_uint<32> >& outData2Strm,
                hls::stream<bool>& outEndStrm) {
    for (int i = 0; i < n; i++) {
#pragma HLS pipeline ii = 1
        outDataStrm.write(inData[i]);
        outData2Strm.write(inData2[i]);
        outEndStrm.write(0);
    }
    outEndStrm.write(1);
}

void writeS2M(int n, hls::stream<ap_uint<32> >& inStrm, hls::stream<bool>& endStrm, ap_uint<32>* crc32Result) {
    for (int i = 0; i < n; i++) {
#pragma HLS pipeline ii = 1
        endStrm.read();
        crc32Result[i] = inStrm.read();
    }
    endStrm.read();
}

extern "C" void CRC32Kernel(
    int num, int size, ap_uint<32>* len, ap_uint<32>* crcInit, ap_uint<512>* inData, ap_uint<32>* crc32Result) {
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 16 bundle = gmem0 port = len
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 16 bundle = gmem1 port = crcInit
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 16 bundle = gmem2 port = inData
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 32 num_read_outstanding = \
    1 max_write_burst_length = 16 max_read_burst_length = 2 bundle = gmem3 port = crc32Result
#pragma HLS INTERFACE s_axilite port = num bundle = control
#pragma HLS INTERFACE s_axilite port = size bundle = control
#pragma HLS INTERFACE s_axilite port = len bundle = control
#pragma HLS INTERFACE s_axilite port = crcInit bundle = control
#pragma HLS INTERFACE s_axilite port = inData bundle = control
#pragma HLS INTERFACE s_axilite port = crc32Result bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS dataflow
    hls::stream<ap_uint<512> > data512Strm("data512Strm");
    hls::stream<ap_uint<8 * W> > dataStrm("dataStrm");
    hls::stream<ap_uint<32> > lenStrm;
    hls::stream<ap_uint<32> > crcInitStrm;
    hls::stream<bool> endStrm;

    hls::stream<ap_uint<32> > crc32Strm;
    hls::stream<bool> crc32EndStrm;

#pragma HLS stream variable = data512Strm depth = 64
#pragma HLS stream variable = dataStrm depth = 16
#pragma HLS stream variable = lenStrm depth = 16
#pragma HLS stream variable = crcInitStrm depth = 16
#pragma HLS stream variable = endStrm depth = 16
#pragma HLS stream variable = crc32Strm depth = 16
#pragma HLS stream variable = crc32EndStrm depth = 16
    readLenM2S(num, len, crcInit, lenStrm, crcInitStrm, endStrm);
    readDataM2S(size, inData, data512Strm);
    splitStrm(size, data512Strm, dataStrm);
    xf::security::crc32<W>(crcInitStrm, dataStrm, lenStrm, endStrm, crc32Strm, crc32EndStrm);
    writeS2M(num, crc32Strm, crc32EndStrm, crc32Result);
}
