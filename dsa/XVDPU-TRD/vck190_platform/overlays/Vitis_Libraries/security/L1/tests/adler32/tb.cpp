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

#include <vector>
#include <math.h>
#include "dut.hpp"
#include <iostream>
#include <fstream>

int main() {
    int nerr = 0;
    uint32_t golden = 0xeb66ed50;
    auto file = std::string("test.dat");

    std::ifstream ifs(file, std::ios::binary);
    if (!ifs) return 1;

    uint32_t size;
    ifs.seekg(0, std::ios::end);
    size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::vector<ap_uint<W * 8> > in(size);
    ifs.read(reinterpret_cast<char*>(in.data()), size);

    hls::stream<ap_uint<32> > adlerStrm;
    hls::stream<ap_uint<W * 8> > inStrm;
    hls::stream<ap_uint<32> > inLenStrm;
    hls::stream<bool> endInLenStrm;
    hls::stream<ap_uint<32> > outStrm;
    hls::stream<bool> endOutStrm;

    for (int t = 0; t < 1; t++) {
        adlerStrm.write(ap_uint<32>(1));
        for (int i = 0; i < (size + W - 1) / W; i++) {
            inStrm.write(in[i]);
        }
        inLenStrm.write(size);
        endInLenStrm.write(false);
    }
    endInLenStrm.write(true);

    dut(adlerStrm, inStrm, inLenStrm, endInLenStrm, outStrm, endOutStrm);

    for (int t = 0; t < 1; t++) {
        ap_uint<32> adler_out = outStrm.read();
        endOutStrm.read();
        if (golden != adler_out) {
            std::cout << std::hex << "adler_out=" << adler_out << ",golden=" << golden << std::endl;
            nerr = 1;
        }
    }
    endOutStrm.read();

    return nerr;
}
