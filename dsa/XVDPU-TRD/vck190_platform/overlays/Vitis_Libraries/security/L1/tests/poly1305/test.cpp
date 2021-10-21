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

#include <ap_int.h>
#include <hls_stream.h>
#include "xf_security/poly1305.hpp"
#if !defined(__SYNTHESIS__)
#include <iostream>
#endif

void poly1305Top(
    // stream in
    hls::stream<ap_uint<256> >& keyStrm,
    hls::stream<ap_uint<128> >& payloadStrm,
    hls::stream<ap_uint<64> >& lenPldStrm,
    hls::stream<bool>& endLenStrm,
    // stream out
    hls::stream<ap_uint<128> >& tagStrm) {
    xf::security::poly1305(keyStrm, payloadStrm, lenPldStrm, endLenStrm, tagStrm);
}
int testPoly1305() {
    ap_uint<128> golden;
    golden.range(127, 64) = 0xa927010caf8b2bc2;
    golden.range(63, 0) = 0xc6365130c11d06a8;

    hls::stream<ap_uint<256> > keyStrm;
    hls::stream<ap_uint<128> > payloadStrm;
    hls::stream<ap_uint<64> > lenPldStrm;
    hls::stream<bool> endLenStrm;
    hls::stream<ap_uint<128> > tagStrm;
    unsigned char k0[] = {0x85, 0xd6, 0xbe, 0x78, 0x57, 0x55, 0x6d, 0x33, 0x7f, 0x44, 0x52,
                          0xfe, 0x42, 0xd5, 0x06, 0xa8, 0x01, 0x03, 0x80, 0x8a, 0xfb, 0x0d,
                          0xb2, 0xfd, 0x4a, 0xbf, 0xf6, 0xaf, 0x41, 0x49, 0xf5, 0x1b};
    unsigned char m0[] = {0x43, 0x72, 0x79, 0x70, 0x74, 0x6f, 0x67, 0x72, 0x61, 0x70, 0x68, 0x69,
                          0x63, 0x20, 0x46, 0x6f, 0x72, 0x75, 0x6d, 0x20, 0x52, 0x65, 0x73, 0x65,
                          0x61, 0x72, 0x63, 0x68, 0x20, 0x47, 0x72, 0x6f, 0x75, 0x70};
    ap_uint<256> apK0;
    for (int i = 0; i < 32; i++) {
        apK0.range(7 + i * 8, i * 8) = k0[i];
    }
#if !defined(__SYNTHESIS__)
    std::cout << "apK0 = " << std::hex << apK0 << std::endl;
    std::cout << "message size: " << sizeof(m0) / sizeof(unsigned char) << std::endl;
#endif
    int len0 = 34;
    lenPldStrm.write(len0);
    endLenStrm.write(true);
    ap_uint<128> pl0 = 0;
    for (int i = 0; i < len0 / 16.0; i++) {
        for (int j = 0; j < 16; j++) {
            if (j + i * 16 < len0)
                pl0.range(7 + j * 8, j * 8) = m0[j + i * 16];
            else
                pl0.range(7 + j * 8, j * 8) = 0;
        }
        // std::cout << "pl0=" << std::hex << pl0 << std::endl;
        payloadStrm.write(pl0);
    }
    ap_uint<128> payload1;
    ap_uint<64> lenPld1;
    keyStrm.write(apK0);
    endLenStrm.write(false);
    poly1305Top(keyStrm, payloadStrm, lenPldStrm, endLenStrm, tagStrm);
    ap_uint<128> tag;
    tag = tagStrm.read();
#if !defined(__SYNTHESIS__)
    std::cout << std::hex << tag << std::endl;
//    std::cout << std::hex << golden << std::endl;
#endif
    if (tag == golden)
        return 0;
    else
        return 1;
}

int main(int argc, char* argv[]) {
    return testPoly1305();
}
