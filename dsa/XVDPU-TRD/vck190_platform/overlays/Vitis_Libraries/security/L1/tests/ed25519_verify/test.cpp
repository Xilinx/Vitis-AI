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
          hls::stream<ap_uint<512> >& signatureStrm,
          hls::stream<bool>& endSignatureStrm,
          hls::stream<ap_uint<256> >& publicKeyStrm,
          hls::stream<bool>& ifValidStrm,
          hls::stream<bool>& endIfValidStrm) {
    xf::security::eddsaEd25519 processor;
    processor.verify(msgStrm, lenStrm, signatureStrm, endSignatureStrm, publicKeyStrm, ifValidStrm, endIfValidStrm);
}

int main() {
    xf::security::eddsaEd25519 processor;

    // Test vector is Test 2 from Chapter 7.1 of RFC 8032.
    hls::stream<ap_uint<64> > msgStrm;
    hls::stream<ap_uint<128> > lenStrm;
    hls::stream<ap_uint<512> > signatureStrm;
    hls::stream<bool> endSignatureStrm("endSignatureStrm");
    hls::stream<ap_uint<256> > publicKeyStrm;
    hls::stream<bool> ifValidStrm("ifValidStrm");
    hls::stream<bool> endIfValidStrm("endIfValidStrm");

    lenStrm.write(1);
    ap_uint<64> msg = ap_uint<64>("0x72");
    msgStrm.write(msg);
    ap_uint<512> sig = ap_uint<512>(
        "0x0CBB1216290DB0EE2A30B4AE2E7B388C1DF1D013368F456E99153EE4C15A08DA69DBEB232276B38F3F5016547BB2A24025645F0B820E"
        "72B8CAD4F0A909A092");
    signatureStrm.write(sig);
    endSignatureStrm.write(false);
    endSignatureStrm.write(true);
    ap_uint<256> publicKey = ap_uint<256>("0x0C66F42AF155CDC08C96C42ECF2C989CBC7E1B4DA70AB7925A8943E8C317403D");
    publicKeyStrm.write(publicKey);

    test(msgStrm, lenStrm, signatureStrm, endSignatureStrm, publicKeyStrm, ifValidStrm, endIfValidStrm);

    bool valid = ifValidStrm.read();
    endIfValidStrm.read();
    endIfValidStrm.read();

    if (valid) {
        return 0;
    } else {
        return 1;
    }
}
