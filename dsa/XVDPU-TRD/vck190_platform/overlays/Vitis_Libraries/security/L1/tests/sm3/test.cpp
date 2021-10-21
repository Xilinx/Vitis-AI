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

#include "xf_security/sm234.hpp"
#ifndef __SYNTHESIS__
#include <iostream>
#endif

void test(hls::stream<ap_uint<64> >& msgStrm,
          hls::stream<ap_uint<64> >& lenStrm,
          hls::stream<bool>& endLenStrm,
          hls::stream<ap_uint<256> >& hashStrm,
          hls::stream<bool>& endHashStrm) {
    xf::security::sm3(msgStrm, lenStrm, endLenStrm, hashStrm, endHashStrm);
}

#ifndef __SYNTHESIS__
int main() {
    // Test vector is from Chapter A.2 of GMT 0003.2-2012
    hls::stream<ap_uint<64> > msgStrm;
    hls::stream<ap_uint<64> > lenStrm;
    hls::stream<bool> endLenStrm;
    hls::stream<ap_uint<256> > hashStrm;
    hls::stream<bool> endHashStrm;

    ap_uint<64> msg = 0;
    msg = "0x6463626164636261";
    msgStrm.write(msg);
    msgStrm.write(msg);
    msgStrm.write(msg);
    msgStrm.write(msg);
    msgStrm.write(msg);
    msgStrm.write(msg);
    msgStrm.write(msg);
    msgStrm.write(msg);
    lenStrm.write(512);
    endLenStrm.write(false);
    endLenStrm.write(true);

    test(msgStrm, lenStrm, endLenStrm, hashStrm, endHashStrm);

    endHashStrm.read();
    endHashStrm.read();

    ap_uint<256> res = hashStrm.read();
    ap_uint<256> gold = ap_uint<256>("0x32570C9CA3CB3D2965577E38E570DB6F4D5A8EC189486038A1B87522F99FBEDE");
    std::cout << std::hex << res << std::endl;
    if (res != gold) {
        return 1;
    } else {
        return 0;
    }
}
#endif
