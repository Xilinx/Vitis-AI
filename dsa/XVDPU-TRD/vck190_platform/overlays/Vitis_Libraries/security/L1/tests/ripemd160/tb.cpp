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
#include <string>
#include <ap_int.h>
#include <iostream>
#include <fstream>

#include "xf_security/ripemd.hpp"
void dut(hls::stream<ap_uint<32> >& inStrm,
         hls::stream<ap_uint<64> >& inLenStrm,
         hls::stream<bool>& endInLenStrm,
         hls::stream<ap_uint<160> >& outStrm,
         hls::stream<bool>& endOutStrm) {
    xf::security::ripemd160(inStrm, inLenStrm, endInLenStrm, outStrm, endOutStrm);
}

#ifndef __SYNTHESIS__
int main() {
    const int caseNum = 8;
    std::vector<std::string> testVectors;
    testVectors.resize(caseNum);
    testVectors[0] = "";
    testVectors[1] = "a";
    testVectors[2] = "abc";
    testVectors[3] = "message digest";
    testVectors[4] = "abcdefghijklmnopqrstuvwxyz";
    testVectors[5] = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    testVectors[6] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    testVectors[7] = "12345678901234567890123456789012345678901234567890123456789012345678901234567890";

    ap_uint<160> golden[8];
    golden[0] = ap_uint<160>("0x318D25B248F5E87E9708286154FCE9C5A585119C");
    golden[1] = ap_uint<160>("0xFE7F465A83DCF4E67B34AEDAE93E6B252D9DDC0B");
    golden[2] = ap_uint<160>("0xFC0B5AF187B0C6988E4A049B7A985DE0F708B28E");
    golden[3] = ap_uint<160>("0x365F5921FA5FA823B181B872E5FAD249EF89065D");
    golden[4] = ap_uint<160>("0xBC8D70B365289D5BEBDCBB561B2C699C10271CF7");
    golden[5] = ap_uint<160>("0x2BEB62DA9AF4DC276CA005E4880C9C4A3853A012");
    golden[6] = ap_uint<160>("0x89511FB2793071A5873AED86026416316E0BE2B0");
    golden[7] = ap_uint<160>("0xFB6B3263BF82AB3C32D3DBF4394B3D57452E759B");

    hls::stream<ap_uint<32> > inStrm;
    hls::stream<ap_uint<64> > inLenStrm;
    hls::stream<bool> endLenStrm;
    hls::stream<ap_uint<160> > outStrm;
    hls::stream<bool> endOutStrm;

    for (int i = 0; i < caseNum; i++) {
        ap_uint<64> len = testVectors[i].size();
        inLenStrm.write(len);
        endLenStrm.write(false);
        for (int j = 0; j < len; j += 4) {
            int left;
            if ((j + 4) < len) {
                left = 4;
            } else {
                left = len - j;
            }
            ap_uint<32> data = 0;
            for (int k = 0; k < left; k++) {
                data.range(k * 8 + 7, k * 8) = testVectors[i][j + k];
            }
            inStrm.write(data);
        }
    }
    endLenStrm.write(true);

    dut(inStrm, inLenStrm, endLenStrm, outStrm, endOutStrm);

    int nerr = 0;
    int idx = 0;
    while (!endOutStrm.read()) {
        ap_uint<160> res = outStrm.read();
        if (res != golden[idx]) {
            std::cout << "result does not match golden" << std::endl;
            std::cout << "golden:" << std::hex << golden[idx] << std::endl;
            std::cout << "result:" << std::hex << res << std::endl;
            nerr++;
        }
        idx++;
    }

    return nerr;
}
#endif
