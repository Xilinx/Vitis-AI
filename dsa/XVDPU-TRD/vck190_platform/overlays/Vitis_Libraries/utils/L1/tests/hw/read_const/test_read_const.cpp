
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

#include "xf_datamover/read_const.hpp"

#define SIZE 1025

void dut(xf::datamover::ConstData::type mm[SIZE], hls::stream<xf::datamover::ConstData::type>& cs, uint64_t sz) {
    xf::datamover::readConst(mm, cs, sz);
}

int main() {
    std::cout << "Testing readConst with " << SIZE << " bytes of input..." << std::endl;
    const int bytePerData = xf::datamover::ConstData::Port_Width / 8;
    const int nBlks = SIZE / bytePerData + ((SIZE % bytePerData) > 0);
    std::cout << "AXI port with is " << xf::datamover::ConstData::Port_Width << std::endl;
    std::cout << "Totally " << nBlks << " data blocks" << std::endl;
    xf::datamover::ConstData::type mm[nBlks];
    hls::stream<xf::datamover::ConstData::type> cs;
    // initializing random seed
    unsigned int seed = 12;
    srand(seed);
    // preparing input buffer
    for (int i = 0; i < nBlks; i++) {
        mm[i] = rand();
    }
    // call FPGA
    dut(mm, cs, SIZE);
    int nerror = 0;
    int ncorrect = 0;
    // check result
    for (int i = 0; i < nBlks; i++) {
        xf::datamover::ConstData::type out = cs.read();
        if (out != mm[i]) {
            nerror++;
            std::cout << std::hex << "out = " << out << "    golden = " << mm[i] << std::endl;
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
