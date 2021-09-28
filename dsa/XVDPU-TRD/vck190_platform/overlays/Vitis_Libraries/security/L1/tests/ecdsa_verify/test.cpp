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

void test(ap_uint<256> hash, ap_uint<256> Qx, ap_uint<256> Qy, ap_uint<256> r, ap_uint<256> s, bool& ifValid) {
    xf::security::ecdsaSecp256k1<256> processor;
    processor.init();
    ifValid = processor.verify(r, s, hash, Qx, Qy);
}

int main() {
    ap_uint<256> m = ap_uint<256>("0x4b688df40bcedbe641ddb16ff0a1842d9c67ea1c3bf63f3e0471baa664531d1a");
    ap_uint<256> r = ap_uint<256>("0x241097efbf8b63bf145c8961dbdf10c310efbb3b2676bbc0f8b08505c9e2f795");
    ap_uint<256> s = ap_uint<256>("0x021006b7838609339e8b415a7f9acb1b661828131aef1ecbc7955dfb01f3ca0e");
    ap_uint<256> Qx = ap_uint<256>("0x779dd197a5df977ed2cf6cb31d82d43328b790dc6b3b7d4437a427bd5847dfcd");
    ap_uint<256> Qy = ap_uint<256>("0xe94b724a555b6d017bb7607c3e3281daf5b1699d6ef4124975c9237b917d426f");

    bool ifValid;

    test(m, Qx, Qy, r, s, ifValid);

    if (ifValid != true) {
        return 1;
    } else {
        return 0;
    }
}
