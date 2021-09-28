
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

#include "xf_datamover/types.hpp"
#include "xf_datamover/check_stream_with_master.hpp"

#define SIZE 1025
#define MASTER_WIDTH 32
#define AXI_WIDTH 32

void dut(hls::stream<ap_axiu<AXI_WIDTH, 0, 0, 0> >& in_s,
         ap_uint<MASTER_WIDTH> mm[SIZE],
         hls::stream<xf::datamover::CheckResult::type>& rs,
         uint64_t sz) {
#pragma HLS interface axis port = in_s
    xf::datamover::checkStreamWithMaster(in_s, mm, rs, sz);
}

int main() {
    std::cout << "Testing checkStreamWithMaster with " << SIZE << " bytes of input..." << std::endl;
    const int bytePerData = MASTER_WIDTH / 8;
    const int nBlks = SIZE / bytePerData + ((SIZE % bytePerData) > 0);
    std::cout << "Master port width is " << MASTER_WIDTH << std::endl;
    std::cout << "Totally " << nBlks << " data blocks" << std::endl;
    ap_uint<MASTER_WIDTH> mm[nBlks];
    hls::stream<ap_axiu<AXI_WIDTH, 0, 0, 0> > is;
    hls::stream<xf::datamover::CheckResult::type> rs;
    // initializing random seed
    unsigned int seed = 12;
    srand(seed);
    // preparing input buffer & stream
    for (int i = 0; i < nBlks; i++) {
        mm[i] = rand();
        ap_axiu<AXI_WIDTH, 0, 0, 0> tmp;
        tmp.data = mm[i];
        tmp.keep = -1;
        tmp.last = 0;
        is.write(tmp);
    }
    // call FPGA
    dut(is, mm, rs, SIZE);
    // check result
    xf::datamover::CheckResult::type out = rs.read();
    if (out != 1) {
        std::cout << "FAIL: Comparing result unmatched." << std::endl;
        return 1;
    } else {
        std::cout << "PASS: Comparing result matched." << std::endl;
        return 0;
    }
}
