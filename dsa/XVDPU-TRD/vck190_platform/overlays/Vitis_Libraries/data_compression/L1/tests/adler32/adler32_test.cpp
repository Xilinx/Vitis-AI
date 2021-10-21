/*
 * Copyright 2019-2021 Xilinx, Inc.
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

#include "hls_stream.h"
#include <ap_int.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include <vector>
#include <math.h>

#include "xf_security/adler32.hpp"
#include "zlib.h"
#define W 1
#define HOST_BUFFER_SIZE (2 * 1024 * 1024)

// DUT
void hls_adler32(hls::stream<ap_uint<32> >& adlerStrm,
                 hls::stream<ap_uint<W * 8> >& inStrm,
                 hls::stream<ap_uint<32> >& inLenStrm,
                 hls::stream<bool>& endInLenStrm,
                 hls::stream<ap_uint<32> >& outStrm,
                 hls::stream<bool>& endOutStrm) {
    xf::security::adler32<W>(adlerStrm, inStrm, inLenStrm, endInLenStrm, outStrm, endOutStrm);
}

// Testbench
int main(int argc, char* argv[]) {
    int nerr = 0;

    std::ifstream ifs;

    ifs.open(argv[1], std::ofstream::binary | std::ofstream::in);
    if (!ifs.is_open()) {
        std::cout << "Cannot open the input file!!" << std::endl;
        exit(0);
    }

    ifs.seekg(0, std::ios::end);
    uint32_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::vector<ap_uint<W * 8> > in(size);
    ifs.read(reinterpret_cast<char*>(in.data()), size);

    unsigned long adlerTmp = 1;
    auto golden = adler32(adlerTmp, reinterpret_cast<const unsigned char*>(in.data()), size);

    hls::stream<ap_uint<32> > adlerStrm;
    hls::stream<ap_uint<W * 8> > inStrm;
    hls::stream<ap_uint<32> > inLenStrm;
    hls::stream<bool> endInLenStrm;
    hls::stream<ap_uint<32> > outStrm;
    hls::stream<bool> endOutStrm;

    // Calculating chunks of file
    uint32_t no_blks = (size - 1) / HOST_BUFFER_SIZE + 1;
    uint32_t readSize = 0;
    ap_uint<32> adlerData = 1;
    int offset = 0;

    for (int t = 0; t < no_blks; t++) {
        uint32_t inSize = HOST_BUFFER_SIZE;
        if (readSize + inSize > size) inSize = size - readSize;
        readSize += inSize;
        adlerStrm.write(ap_uint<32>(adlerData));
        for (int i = 0; i < (inSize - 1) / W + 1; i++) {
            inStrm.write(in[i + offset]);
        }
        offset += (inSize - 1) / W + 1;
        inLenStrm.write(inSize);
        endInLenStrm.write(false);
        endInLenStrm.write(true);

        hls_adler32(adlerStrm, inStrm, inLenStrm, endInLenStrm, outStrm, endOutStrm);
        adlerData = outStrm.read();
        endOutStrm.read();
        endOutStrm.read();
    }

    if (golden != adlerData) {
        std::cout << std::hex << "adler_out=" << adlerData << ",golden=" << golden << std::endl;
        nerr = 1;
    } else
        std::cout << "Test Passed" << std::endl;

    return nerr;
}
