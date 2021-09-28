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

#include "xf_security/ecdsa.hpp"
#ifndef __SYNTHESIS__
#include <iostream>
#endif

void test(ap_uint<256> hash, ap_uint<256> k, ap_uint<256> privateKey, ap_uint<256>& r, ap_uint<256>& s, bool& kValid) {
    xf::security::ecdsaSecp256k1<256> processor;
    processor.init();
    kValid = processor.sign(hash, k, privateKey, r, s);
}

#ifndef __SYNTHESIS__
int main() {
    ap_uint<256> m = ap_uint<256>("0x4b688df40bcedbe641ddb16ff0a1842d9c67ea1c3bf63f3e0471baa664531d1a");
    ap_uint<256> privateKey = ap_uint<256>("0xebb2c082fd7727890a28ac82f6bdf97bad8de9f5d7c9028692de1a255cad3e0f");
    ap_uint<256> k = ap_uint<256>("0x49a0d7b786ec9cde0d0721d72804befd06571c974b191efb42ecf322ba9ddd9a");
    ap_uint<256> gold_r = ap_uint<256>("0x241097efbf8b63bf145c8961dbdf10c310efbb3b2676bbc0f8b08505c9e2f795");
    ap_uint<256> gold_s = ap_uint<256>("0x021006b7838609339e8b415a7f9acb1b661828131aef1ecbc7955dfb01f3ca0e");

    ap_uint<256> r, s;
    bool kValid;

    test(m, k, privateKey, r, s, kValid);

    if (gold_r != r || gold_s != s) {
        std::cout << std::hex << "r:" << r << std::endl;
        std::cout << std::hex << "s:" << s << std::endl;
        return 1;
    } else {
        return 0;
    }
}
#endif
