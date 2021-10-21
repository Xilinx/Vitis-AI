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
#include <hls_stream.h>
#include "xf_security/vdf.hpp"
#if !defined(__SYNTHESIS__)
#include <iostream>
#endif
#define N 16
#define W1 (8 * 128)
#define W2 32

bool vdfVerify(ap_uint<W1> g,
               ap_uint<W1> l,
               ap_uint<W1> m,
               ap_uint<W1> r2Mod,
               ap_uint<W1> tMod,
               ap_uint<W2> t,
               ap_uint<W1>& y,
               ap_uint<W1>& pi) {
    return xf::security::verifyPietrzak<W1, W2>(g, m, r2Mod, t, y);
}

int main(int argc, char* argv[]) {
    const int T = 256;
    ap_uint<W1> l = ap_uint<W1>("0x6694ceb5");
    ap_uint<W1> g = ap_uint<W1>("0x18f204fe6846aeb6f58174d57a3372363c0d9fcfaa3dc18b1eff7e89bf767863");
    ap_uint<W1> m = ap_uint<W1>("0x2ee5e7f0e757f2ed656347023038c4a92698d7f77fa267fb0783d94faaa55dc5");
    ap_uint<W1> y = ap_uint<W1>("0xa441d092a775c816c8121d2ba83b024750c4aa17133022663b224dfcee5e291");
    ap_uint<W1> pi = ap_uint<W1>("0x23202eb644107a4baf67548a76387f07f956ba21bcce919d1ccf5d827d6437eb");
    ap_uint<W2> t = T;

    ap_uint<W1* 2 + 1> tmp = 1;
    tmp <<= W1 * 2;
    ap_uint<W1> r_mod = tmp % m;

    ap_uint<T + 1> t2 = (ap_uint<T + 1>)1 << t;
    ap_uint<W1> t_mod = t2 % l;

    bool re = vdfVerify(g, l, m, r_mod, t_mod, t, y, pi);
    return re;
}
