
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

#include "xf_datamover/load_master_to_stream.hpp"

#define SIZE 1025
#define MASTER_WIDTH 32
#define AXI_WIDTH 32

void dut(ap_uint<MASTER_WIDTH> mm[SIZE], hls::stream<ap_axiu<AXI_WIDTH, 0, 0, 0> >& s, uint64_t sz) {
#pragma HLS INTERFACE axis port = s
    xf::datamover::loadMasterToStream(mm, s, sz);
}

int main() {
    std::cout << "Testing loadMasterToStream with " << SIZE << " bytes of input..." << std::endl;
    const int bytePerData = MASTER_WIDTH / 8;
    const int nBlks = SIZE / bytePerData + ((SIZE % bytePerData) > 0);
    std::cout << "Master port width is " << MASTER_WIDTH << std::endl;
    std::cout << "Totally " << nBlks << " data blocks" << std::endl;
    ap_uint<MASTER_WIDTH> mm[nBlks];
    hls::stream<ap_axiu<AXI_WIDTH, 0, 0, 0> > s;
    // initializing random seed
    unsigned int seed = 12;
    srand(seed);
    // preparing input buffer
    for (int i = 0; i < nBlks; i++) {
        mm[i] = rand();
    }
    // call FPGA
    dut(mm, s, SIZE);
    int nerror = 0;
    int ncorrect = 0;
    // check result
    for (int i = 0; i < nBlks; i++) {
        ap_axiu<AXI_WIDTH, 0, 0, 0> out = s.read();
        if (out.data != mm[i]) {
            nerror++;
            std::cout << std::hex << "out = " << out.data << "    golden = " << mm[i] << std::endl;
        } else {
            ncorrect++;
        }
    }
    if (nerror) {
        std::cout << "FAIL: " << nerror << " errors in " << nBlks << " inputs." << std::endl;
    } else {
        std::cout << "PASS: " << ncorrect << " input blocks verified." << std::endl;
    }

    return nerror;
}
