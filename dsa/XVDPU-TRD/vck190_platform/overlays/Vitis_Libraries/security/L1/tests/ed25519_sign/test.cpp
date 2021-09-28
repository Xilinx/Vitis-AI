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

#include "xf_security/eddsa.hpp"

void test(hls::stream<ap_uint<64> >& msgStrm,
          hls::stream<ap_uint<128> >& lenStrm,
          hls::stream<bool>& endLenStrm,
          hls::stream<ap_uint<512> >& signatureStrm,
          hls::stream<bool>& endSignatureStrm,
          ap_uint<256> privateKey) {
    xf::security::eddsaEd25519 processor;
    ap_uint<256> publicKey;
    ap_uint<512> privateKeyHash;
    processor.generatePublicKey(privateKey, publicKey, privateKeyHash);
    processor.sign(msgStrm, lenStrm, endLenStrm, publicKey, privateKeyHash, signatureStrm, endSignatureStrm);
}

int main() {
    xf::security::eddsaEd25519 processor;

    // Test vector is Test 2 from Chapter 7.1 of RFC 8032.
    ap_uint<256> tmp = ap_uint<256>("0x4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb");
    ap_uint<256> privateKey;
    for (int i = 0; i < 32; i++) {
        int j = 31 - i;
        privateKey.range(i * 8 + 7, i * 8) = tmp.range(j * 8 + 7, j * 8);
    }

    //
    hls::stream<ap_uint<64> > msgStrm;
    hls::stream<ap_uint<128> > lenStrm;
    hls::stream<bool> endLenStrm;
    hls::stream<ap_uint<512> > signatureStrm;
    hls::stream<bool> endSignatureStrm;

    lenStrm.write(1);
    ap_uint<64> msg = ap_uint<64>("0x72");
    msgStrm.write(msg);
    msgStrm.write(msg);
    endLenStrm.write(false);
    endLenStrm.write(true);

    test(msgStrm, lenStrm, endLenStrm, signatureStrm, endSignatureStrm, privateKey);
    endSignatureStrm.read();
    ap_uint<512> sig = signatureStrm.read();
    endSignatureStrm.read();

    ap_uint<512> golden = ap_uint<512>(
        "0x0CBB1216290DB0EE2A30B4AE2E7B388C1DF1D013368F456E99153EE4C15A08DA69DBEB232276B38F3F5016547BB2A24025645F0B820E"
        "72B8CAD4F0A909A092");

    if (sig == golden) {
        return 0;
    } else {
        return 1;
    }
}
