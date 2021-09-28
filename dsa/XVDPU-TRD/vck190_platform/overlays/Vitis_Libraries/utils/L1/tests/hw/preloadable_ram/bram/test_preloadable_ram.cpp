
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

#include <iostream>
#include <stdlib.h>

#include "xf_datamover/preloadable_ram.hpp"

#define SIZE 1025
#define AXI_WIDTH 64
#define RAM_DEPTH 1024

void dut(hls::stream<xf::datamover::ConstData::type>& is,
         hls::stream<ap_axiu<AXI_WIDTH, 0, 0, 0> >& os,
         hls::stream<ap_axiu<AXI_WIDTH, 0, 0, 0> >& cs,
         hls::stream<xf::datamover::CheckResult::type>& rs,
         const uint64_t sz) {
#pragma HLS INTERFACE axis port = os
#pragma HLS INTERFACE axis port = cs
    xf::datamover::PreloadableBram<AXI_WIDTH, RAM_DEPTH> pbram;
    pbram.preload(is, sz);
    pbram.toStream(os, sz);
    pbram.checkStream(cs, rs, sz);
}

int main() {
    std::cout << "Testing PreloadableBram with " << SIZE << " bytes of input..." << std::endl;
    xf::datamover::ConstData::type golden[AXI_WIDTH / xf::datamover::ConstData::Port_Width * RAM_DEPTH];
    hls::stream<xf::datamover::ConstData::type> is;
    hls::stream<ap_axiu<AXI_WIDTH, 0, 0, 0> > cs;
    hls::stream<ap_axiu<AXI_WIDTH, 0, 0, 0> > os;
    hls::stream<xf::datamover::CheckResult::type> rs;
    // initializing random seed
    unsigned int seed = 12;
    srand(seed);
    const int bytePerInData = xf::datamover::ConstData::Port_Width / 8;
    const int inBlks = SIZE / bytePerInData + ((SIZE % bytePerInData) > 0);
    // preparing input buffer
    for (int i = 0; i < inBlks; i++) {
        golden[i] = rand();
        is.write(golden[i]);
    }
    const int bytePerCmpData = AXI_WIDTH / 8;
    const int cmpBlks = SIZE / bytePerCmpData + ((SIZE % bytePerCmpData) > 0);
    // preparing compare stream
    for (int i = 0; i < cmpBlks; i++) {
        ap_axiu<AXI_WIDTH, 0, 0, 0> tmp;
        tmp.data = *((ap_uint<AXI_WIDTH>*)golden + i);
        tmp.keep = -1;
        tmp.last = 0;
        cs.write(tmp);
    }
    // call FPGA
    dut(is, os, cs, rs, SIZE);
    // check direct output result
    int error_out = 0;
    const int bytePerOutData = AXI_WIDTH / 8;
    const int outBlks = SIZE / bytePerOutData;
    for (int i = 0; i < outBlks; i++) {
        ap_axiu<AXI_WIDTH, 0, 0, 0> tmp = os.read();
        if (tmp.data != *((ap_uint<AXI_WIDTH>*)golden + i)) {
            error_out++;
            std::cout << std::hex << "out = " << tmp.data << "    golden = " << *((ap_uint<AXI_WIDTH>*)golden + i)
                      << std::endl;
        }
    }
    int leftBytes = SIZE % bytePerOutData;
    if (leftBytes > 0) {
        ap_axiu<AXI_WIDTH, 0, 0, 0> tmp = os.read();
        ap_uint<AXI_WIDTH> cmp_out = tmp.data;
        ap_uint<AXI_WIDTH> cmp_golden = *((ap_uint<AXI_WIDTH>*)golden + outBlks);
        if (cmp_out.range(leftBytes * 8 - 1, 0) != cmp_golden.range(leftBytes * 8 - 1, 0)) {
            error_out++;
            std::cout << std::hex << "out = " << tmp.data << "    golden = " << *((ap_uint<AXI_WIDTH>*)golden + outBlks)
                      << std::endl;
        }
    }
    if (error_out) {
        std::cout << "FAIL: Direct output stream is not equal to the input one." << std::endl;
    } else {
        std::cout << "PASS: Direct output stream verified." << std::endl;
    }

    // check compare result
    int error_cmp = 0;
    xf::datamover::CheckResult::type r = rs.read();
    if (r != 1) {
        error_cmp++;
        std::cout << "FAIL: Comparing result incorrect." << std::endl;
    }

    return error_out + error_cmp;
}
