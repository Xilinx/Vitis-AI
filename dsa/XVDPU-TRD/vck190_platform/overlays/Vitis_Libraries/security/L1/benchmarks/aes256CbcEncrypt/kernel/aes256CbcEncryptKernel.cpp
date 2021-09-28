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

/**
 *
 * @file aes256CbcEncryptKernel.cpp
 * @brief kernel code of Cipher Block Chaining (CBC) block cipher mode of operation.
 * This file is part of Vitis Security Library.
 *
 * @detail Containing scan, distribute, encrypt, merge, and write-out functions.
 *
 */

#include <ap_int.h>
#include <hls_stream.h>
#include "xf_security/cbc.hpp"
#include "xf_security/msgpack.hpp"

#ifndef __SYNTHESIS
#include <iostream>
#endif

void cbcWrapper(hls::stream<ap_uint<128> >& plaintext,
                hls::stream<bool>& plaintext_e,
                hls::stream<ap_uint<256> >& key,
                hls::stream<ap_uint<128> >& iv,
                hls::stream<ap_uint<128> >& res,
                hls::stream<bool>& endRes,
                ap_uint<64> msgNum) {
    for (ap_uint<64> i = 0; i < msgNum; i++) {
        xf::security::aes256CbcEncrypt(plaintext, plaintext_e, key, iv, res, endRes);
    }
}

void wrapper(ap_uint<128>* input, ap_uint<128>* output, ap_uint<64> msg_num, ap_uint<64> row_num) {
#pragma HLS dataflow

    hls::stream<ap_uint<128> > textStrm;
#pragma HLS stream variable = textStrm depth = 128
#pragma HLS resource variable = textStrm core = FIFO_LUTRAM
    hls::stream<bool> endTextStrm;
#pragma HLS stream variable = endTextStrm depth = 128
#pragma HLS resource variable = endTextStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<256> > keyStrm;
#pragma HLS stream variable = keyStrm depth = 4
#pragma HLS resource variable = keyStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<128> > ivStrm;
#pragma HLS stream variable = ivStrm depth = 4
#pragma HLS resource variable = ivStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<64> > lenStrm;
#pragma HLS stream variable = lenStrm depth = 4
#pragma HLS resource variable = lenStrm core = FIFO_LUTRAM

    xf::security::internal::aesCbcPack<256> packer;
    packer.scanPack(input, msg_num, row_num, textStrm, endTextStrm, keyStrm, ivStrm, lenStrm);

    hls::stream<ap_uint<128> > resStrm;
#pragma HLS stream variable = resStrm depth = 128
#pragma HLS resource variable = resStrm core = FIFO_LUTRAM

    hls::stream<bool> endResStrm;
#pragma HLS stream variable = endResStrm depth = 128
#pragma HLS resource variable = endResStrm core = FIFO_LUTRAM

    cbcWrapper(textStrm, endTextStrm, keyStrm, ivStrm, resStrm, endResStrm, msg_num);

    packer.writeOutMsgPack(output, msg_num, resStrm, endResStrm, lenStrm);
}

// @brief top of kernel
extern "C" void aes256CbcEncryptKernel(ap_uint<128>* inputData, ap_uint<128>* outputData) {
// clang-format off
#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 16 num_read_outstanding = 16 \
	max_write_burst_length = 64 max_read_burst_length = 64 \
	bundle = gmem0_0 port = inputData

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 16 num_read_outstanding = 16 \
	max_write_burst_length = 64 max_read_burst_length = 64 \
	bundle = gmem0_1 port = outputData
// clang-format on

#pragma HLS INTERFACE s_axilite port = inputData bundle = control
#pragma HLS INTERFACE s_axilite port = outputData bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    ap_uint<128> tmp = inputData[0];
    ap_uint<64> msg_num = tmp.range(63, 0);
    ap_uint<64> row_num = tmp.range(127, 64);
    wrapper(inputData, outputData, msg_num, row_num);

} // end aes256CbcDecryptKernel
