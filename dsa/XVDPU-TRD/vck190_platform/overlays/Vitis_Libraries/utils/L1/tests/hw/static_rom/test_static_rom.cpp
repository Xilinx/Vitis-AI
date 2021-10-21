
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

#include "xf_datamover/static_rom.hpp"

#define NUM 17
#define ROM_WIDTH 64
#define ROM_DEPTH 1024

void dut(hls::stream<ap_axiu<ROM_WIDTH, 0, 0, 0> >& os,
         hls::stream<ap_axiu<ROM_WIDTH, 0, 0, 0> >& cs,
         hls::stream<xf::datamover::CheckResult::type>& rs,
         const uint64_t sz) {
#pragma HLS INTERFACE axis port = os
#pragma HLS INTERFACE axis port = cs
    const ap_uint<ROM_WIDTH> in[] = {
#include "init_file.inc"
    };
#pragma HLS resource variable = in core = ROM_2P
    xf::datamover::StaticRom<ROM_WIDTH, ROM_DEPTH> srom;
    srom.data = in;
    srom.toStream(os, sz);
    srom.checkStream(cs, rs, sz);
}

int main() {
    std::cout << "Testing StaticRom with " << NUM << " inputs..." << std::endl;
    ap_uint<ROM_WIDTH> golden[] = {
#include "init_file.inc"
    };
    hls::stream<ap_axiu<ROM_WIDTH, 0, 0, 0> > os;
    hls::stream<ap_axiu<ROM_WIDTH, 0, 0, 0> > cs;
    hls::stream<xf::datamover::CheckResult::type> rs;
    // preparing input buffer
    for (int i = 0; i < NUM; i++) {
        golden[i] = i;
    }
    // preparing compare stream
    for (int i = 0; i < NUM; i++) {
        ap_axiu<ROM_WIDTH, 0, 0, 0> tmp;
        tmp.data = golden[i];
        tmp.keep = -1;
        tmp.last = 0;
        cs.write(tmp);
    }
    // call FPGA
    dut(os, cs, rs, NUM * ROM_WIDTH / 8);
    // check direct output result
    int error_out = 0;
    for (int i = 0; i < NUM; i++) {
        ap_axiu<ROM_WIDTH, 0, 0, 0> tmp = os.read();
        if (tmp.data != golden[i]) {
            error_out++;
            std::cout << std::hex << "out = " << tmp.data << "    golden = " << golden[i] << std::endl;
        }
    }
    if (error_out) {
        std::cout << "FAIL: Direct output stream is not equal to the one in ROM." << std::endl;
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
