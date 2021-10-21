/*
 * Copyright 2021 Xilinx, Inc.
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
#include <stdlib.h>
#include <string>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <iostream>
#include <fstream>

#include "xf_security/crc32.hpp"
#include "zlib.h"
auto constexpr W = 8;
auto constexpr HOST_BUFFER_SIZE = 2 * 1024 * 1024;

void hls_crc32(hls::stream<ap_uint<32> >& inCrcStrm,
               hls::stream<ap_uint<8 * W> >& inStrm,
               hls::stream<ap_uint<5> >& inLenStrm,
               hls::stream<bool>& endInStrm,
               hls::stream<ap_uint<32> >& outStrm,
               hls::stream<bool>& endOutStrm) {
    xf::security::crc32<W>(inCrcStrm, inStrm, inLenStrm, endInStrm, outStrm, endOutStrm);
}

int main(int argc, char* argv[]) {
    int nerr = 0;

    std::ifstream ifs(argv[1], std::ios::binary);
    if (!ifs) return 1;

    uint32_t size;
    ifs.seekg(0, std::ios::end);
    size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::vector<ap_uint<W * 8> > in(size);
    ifs.read(reinterpret_cast<char*>(in.data()), size);

    unsigned long crctmp = 0;
    auto golden = crc32(crctmp, reinterpret_cast<const unsigned char*>(in.data()), size);

    hls::stream<ap_uint<W * 8> > inStrm;
    hls::stream<ap_uint<5> > inLenStrm;
    hls::stream<ap_uint<32> > inCrcStrm;
    hls::stream<bool> endInStrm;
    hls::stream<ap_uint<32> > outStrm;
    hls::stream<bool> endOutStrm;
    uint32_t readSize = 0;
    ap_uint<32> crcData = 0;
    int offset = 0;

    uint32_t no_blks = (size - 1) / HOST_BUFFER_SIZE + 1;

    for (int t = 0; t < no_blks; t++) {
        uint32_t inSize = HOST_BUFFER_SIZE;
        if (readSize + inSize > size) inSize = size - readSize;
        readSize += inSize;

        for (int i = 0; i < (inSize + W - 1) / W; i++) {
            inStrm.write(in[i + offset]);
            if ((i * W + W) <= inSize) {
                inLenStrm.write(ap_uint<5>(W));
            } else {
                inLenStrm.write(ap_uint<5>(size % W));
            }
        }

        offset += inSize;

        inCrcStrm.write(ap_uint<32>(~crcData));
        inLenStrm.write(ap_uint<5>(0));
        endInStrm.write(false);
        endInStrm.write(true);

        hls_crc32(inCrcStrm, inStrm, inLenStrm, endInStrm, outStrm, endOutStrm);
        crcData = outStrm.read();
        endOutStrm.read();
        endOutStrm.read();
    }

    if (golden != crcData) {
        std::cout << std::hex << "crc_out=" << crcData << ",golden=" << golden << std::endl;
        nerr = 1;
    } else
        std::cout << "Test Passed" << std::endl;

    return nerr;
}
