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

void test(ap_uint<256> p,
          ap_uint<256> a,
          ap_uint<256> b,
          ap_uint<256> Gx,
          ap_uint<256> Gy,
          ap_uint<256> n,
          ap_uint<256> r,
          ap_uint<256> s,
          ap_uint<256> hashZaM,
          ap_uint<256> Px,
          ap_uint<256> Py,
          bool& valid) {
    xf::security::sm2<256> processor;
    processor.init(a, b, p, Gx, Gy, n);
    valid = processor.verify(r, s, hashZaM, Px, Py);
}

int main() {
    // Test vector is from Chapter A.2 of GMT 0003.2-2012
    ap_uint<256> p = ap_uint<256>("0x8542D69E4C044F18E8B92435BF6FF7DE457283915C45517D722EDB8B08F1DFC3");
    ap_uint<256> a = ap_uint<256>("0x787968B4FA32C3FD2417842E73BBFEFF2F3C848B6831D7E0EC65228B3937E498");
    ap_uint<256> b = ap_uint<256>("0x63E4C6D3B23B0C849CF84241484BFE48F61D59A5B16BA06E6E12D1DA27C5249A");
    ap_uint<256> Gx = ap_uint<256>("0x421DEBD61B62EAB6746434EBC3CC315E32220B3BADD50BDC4C4E6C147FEDD43D");
    ap_uint<256> Gy = ap_uint<256>("0x0680512BCBB42C07D47349D2153B70C4E5D7FDFCBFA36EA1A85841B9E46E09A2");
    ap_uint<256> n = ap_uint<256>("0x8542D69E4C044F18E8B92435BF6FF7DD297720630485628D5AE74EE7C32E79B7");

    ap_uint<256> hashZaM = ap_uint<256>("0xB524F552CD82B8B028476E005C377FB19A87E6FC682D48BB5D42E3D9B9EFFE76");
    ap_uint<256> r = ap_uint<256>("0x40F1EC59F793D9F49E09DCEF49130D4194F79FB1EED2CAA55BACDB49C4E755D1");
    ap_uint<256> s = ap_uint<256>("0x6FC6DAC32C5D5CF10C77DFB20F7C2EB667A457872FB09EC56327A67EC7DEEBE7");
    ap_uint<256> Px = ap_uint<256>("0x0AE4C7798AA0F119471BEE11825BE46202BB79E2A5844495E97C04FF4DF2548A");
    ap_uint<256> Py = ap_uint<256>("0x7C0240F88F1CD4E16352A73C17B7F16F07353E53A176D684A9FE0C6BB798E857");
    bool valid;

    test(p, a, b, Gx, Gy, n, r, s, hashZaM, Px, Py, valid);

    if (!valid) {
        return 1;
    } else {
        return 0;
    }
}
