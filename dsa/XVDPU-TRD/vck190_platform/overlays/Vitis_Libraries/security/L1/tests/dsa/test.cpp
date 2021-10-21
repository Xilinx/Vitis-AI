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

#include <hls_stream.h>
#define AP_INT_MAX_W 4097
#include <ap_int.h>

#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
using namespace std;
#include <sstream>
#include <string>
#include <vector>

#include "xf_security/dsa.hpp"

void test(ap_uint<1024> p,
          ap_uint<160> q,
          ap_uint<1024> g,
          ap_uint<160> x,
          ap_uint<1024> y,
          ap_uint<160> k,
          ap_uint<256> digest,
          ap_uint<160>& r,
          ap_uint<160>& s,
          bool& result) {
    xf::security::dsa<1024, 160, 256> processor;
    // signing process
    processor.updateSigningParam(p, q, g, x);
    processor.sign(digest, k, r, s);
    // verify process
    processor.updateVerifyingParam(p, q, g, y);
    result = processor.verify(digest, r, s);
}

#ifndef __SYNTHESIS__
int main() {
    ap_uint<1024> p = ap_uint<1024>(
        "0xcba13e533637c37c0e80d9fcd052c1e41a88ac325c4ebe13b7170088d54eef4881f3d35eae47c210385a8485d2423a64da3ffda63a26"
        "f92cf5a304f39260384a9b7759d8ac1adc81d3f8bfc5e6cb10efb4e0f75867f4e848d1a338586dd0648feeb163647ffe7176174370540e"
        "e8a8f588da8cc143d939f70b114a7f981b8483");
    ap_uint<160> q = ap_uint<160>("0x95031b8aa71f29d525b773ef8b7c6701ad8a5d99");
    ap_uint<1024> g = ap_uint<1024>(
        "0x45bcaa443d4cd1602d27aaf84126edc73bd773de6ece15e97e7fef46f13072b7adcaf7b0053cf4706944df8c4568f26c997ee7753000"
        "fbe477a37766a4e970ff40008eb900b9de4b5f9ae06e06db6106e78711f3a67feca74dd5bddcdf675ae4014ee9489a42917fbee3bb9f2a"
        "24df67512c1c35c97bfbf2308eaacd28368c5c");
    ap_uint<160> x = ap_uint<160>("0x2eac4f4196fedb3e651b3b00040184cfd6da2ab4");
    ap_uint<1024> y = ap_uint<1024>(
        "0x4cd6178637d0f0de1488515c3b12e203a3c0ca652f2fe30d088dc7278a87affa634a727a721932d671994a958a0f89223c286c3a9b10"
        "a96560542e2626b72e0cd28e5133fb57dc238b7fab2de2a49863ecf998751861ae668bf7cad136e6933f57dfdba544e3147ce0e7370fa6"
        "e8ff1de690c51b4aeedf0485183889205591e8");

    ap_uint<256> digest = ap_uint<256>("0x32aaa5938f00b165e144058d5c2190baf4693c0c2a7db4e59a77ec14a70a6e50");

    ap_uint<160> k = ap_uint<160>("0x85976c5610a74959531040a5512b347eac587e48");

    ap_uint<160> gld_r = ap_uint<160>("0x76683a085d6742eadf95a61af75f881276cfd26a");
    ap_uint<160> gld_s = ap_uint<160>("0x3b9da7f9926eaaad0bebd4845c67fcdb64d12453");

    ap_uint<160> r = 0;
    ap_uint<160> s = 0;
    bool res = true;

    test(p, q, g, x, y, k, digest, r, s, res);

    if (gld_r != r) {
        res = false;
        std::cout << "r of signing is not correct" << std::endl;
    }
    if (gld_s != s) {
        res = false;
        std::cout << "s of signing is not correct" << std::endl;
    }

    if (res) {
        return 0;
    } else {
        return 1;
    }
}

#endif
