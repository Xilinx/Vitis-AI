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
    auto file = std::string("test.dat");
    ap_uint<32> golden = 0xff7e73d8;

    std::ifstream ifs(file, std::ios::binary);
    if (!ifs) return 1;

    uint32_t size;
    ifs.seekg(0, std::ios::end);
    size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::vector<ap_uint<W * 8> > in(size);
    ifs.read(reinterpret_cast<char*>(in.data()), size);

    hls::stream<ap_uint<32> > crcInitStrm;
    hls::stream<ap_uint<W * 8> > inStrm;
    hls::stream<ap_uint<5> > inPackLenStrm;
    hls::stream<bool> endInPackLenStrm;
    hls::stream<ap_uint<32> > outStrm;
    hls::stream<bool> endOutStrm;

    for (int t = 0; t < 1; t++) {
        crcInitStrm.write(~0);

        for (int i = 0; i < (size + W - 1) / W; i++) {
            inStrm.write(in[i]);
            if ((i * W + W) <= size) {
                inPackLenStrm.write(ap_uint<5>(W));
            } else {
                inPackLenStrm.write(ap_uint<5>(size % W));
            }
        }
        inPackLenStrm.write(ap_uint<5>(0));
        endInPackLenStrm.write(false);
    }
    endInPackLenStrm.write(true);

    dut(crcInitStrm, inStrm, inPackLenStrm, endInPackLenStrm, outStrm, endOutStrm);

    for (int t = 0; t < 1; t++) {
        ap_uint<32> crc_out = outStrm.read();
        endOutStrm.read();
        if (golden != crc_out) {
            std::cout << std::hex << "crc_out=" << crc_out << ",golden=" << golden << std::endl;
            nerr = 1;
        }
    }
    endOutStrm.read();
    return nerr;
}
