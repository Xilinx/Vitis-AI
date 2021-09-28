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

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <stdlib.h>
#include <stdint.h>

#include "xf_database/hash_lookup3.hpp"

#define TOTAL_NUM 1000

void lookup3_dut(hls::stream<ap_uint<64> >& key_strm,
                 hls::stream<bool>& ein,
                 hls::stream<ap_uint<32> >& out_strm,
                 hls::stream<bool>& eout) {
    xf::database::hashLookup3(key_strm, ein, out_strm, eout);
}

uint32_t hashlookup3_x86_32(const void* key, int length, uint32_t seed);
uint64_t hashlookup3_x86_64(const void* key, int length, uint32_t seed1, uint32_t seed2);

int main(int argc, const char* argv[]) {
    hls::stream<ap_uint<64> > key_strm("key_strm");
    hls::stream<ap_uint<32> > hash_strm("hash_strm");
    hls::stream<bool> ein, eout;

    uint32_t h_ref[TOTAL_NUM];
    int nerror = 0;

    srand(11);

    // 1. random the distinct keys
    for (int i = 0; i < TOTAL_NUM; i++) {
        uint64_t k = rand();
        h_ref[i] = hashlookup3_x86_32(&k, 64 / 8, 0x3D83917A);
        key_strm.write(ap_uint<64>(k));
        ein.write(false);
    }
    ein.write(true);

    lookup3_dut(key_strm, ein, hash_strm, eout);

    int cnt = 0;
    while (!eout.read()) {
        ap_uint<32> h = hash_strm.read();
        if (h != h_ref[cnt]) {
            nerror++;
            if (nerror < 10) {
                std::cout << std::hex << "dut:" << h << ", ref:" << h_ref[cnt] << std::dec << std::endl;
            }
        }
        ++cnt;
    }
    if (cnt != TOTAL_NUM) {
        nerror++;
        std::cout << TOTAL_NUM << " keys in, " << cnt << " hash values out, mismatch!" << std::endl;
    }

    if (nerror) {
        std::cout << "\nFAIL: " << nerror << " errors found.\n";
    } else {
        std::cout << "\nPASS: no error found.\n";
    }
    return nerror;
}

#define rot(x, k) (((x) << (k)) | ((x) >> (32 - (k))))

uint32_t hashlookup3_x86_32(const void* key, int length, uint32_t seed) {
    uint32_t a, b, c; /* internal state */

    a = b = c = 0xdeadbeef + ((uint32_t)length) + seed;

    const uint32_t* k = (const uint32_t*)key; /* read 32-bit chunks */

    /*------ all but last block: aligned reads and affect 32 bits of (a,b,c) */
    while (length > 12) {
        a += k[0];
        b += k[1];
        c += k[2];
        a -= c;
        a ^= rot(c, 4);
        c += b;

        b -= a;
        b ^= rot(a, 6);
        a += c;

        c -= b;
        c ^= rot(b, 8);
        b += a;

        a -= c;
        a ^= rot(c, 16);
        c += b;

        b -= a;
        b ^= rot(a, 19);
        a += c;

        c -= b;
        c ^= rot(b, 4);
        b += a;

        length -= 12;
        k += 3;
    }

    const uint8_t* k8 = (const uint8_t*)k;
    switch (length) {
        case 12:
            c += k[2];
            b += k[1];
            a += k[0];
            break;
        case 11:
            c += ((uint32_t)k8[10]) << 16; /* fall through */
        case 10:
            c += ((uint32_t)k8[9]) << 8; /* fall through */
        case 9:
            c += k8[8]; /* fall through */
        case 8:
            b += k[1];
            a += k[0];
            break;
        case 7:
            b += ((uint32_t)k8[6]) << 16; /* fall through */
        case 6:
            b += ((uint32_t)k8[5]) << 8; /* fall through */
        case 5:
            b += k8[4]; /* fall through */
        case 4:
            a += k[0];
            break;
        case 3:
            a += ((uint32_t)k8[2]) << 16; /* fall through */
        case 2:
            a += ((uint32_t)k8[1]) << 8; /* fall through */
        case 1:
            a += k8[0];
            break;
        case 0:
            return c;
    }

    c ^= b;
    c -= rot(b, 14);

    a ^= c;
    a -= rot(c, 11);

    b ^= a;
    b -= rot(a, 25);

    c ^= b;
    c -= rot(b, 16);

    a ^= c;
    a -= rot(c, 4);

    b ^= a;
    b -= rot(a, 14);

    c ^= b;
    c -= rot(b, 24);

    return c;
}

uint64_t hashlookup3_x86_64(const void* key, int length, uint32_t seed1, uint32_t seed2) {
    uint32_t a, b, c; /* internal state */

    a = b = c = 0xdeadbeef + ((uint32_t)length) + seed1;
    c += seed2;

    const uint32_t* k = (const uint32_t*)key; /* read 32-bit chunks */

    /*------ all but last block: aligned reads and affect 32 bits of (a,b,c) */
    while (length > 12) {
        a += k[0];
        b += k[1];
        c += k[2];
        a -= c;
        a ^= rot(c, 4);
        c += b;

        b -= a;
        b ^= rot(a, 6);
        a += c;

        c -= b;
        c ^= rot(b, 8);
        b += a;

        a -= c;
        a ^= rot(c, 16);
        c += b;

        b -= a;
        b ^= rot(a, 19);
        a += c;

        c -= b;
        c ^= rot(b, 4);
        b += a;

        length -= 12;
        k += 3;
    }

    const uint8_t* k8 = (const uint8_t*)k;
    switch (length) {
        case 12:
            c += k[2];
            b += k[1];
            a += k[0];
            break;
        case 11:
            c += ((uint32_t)k8[10]) << 16; /* fall through */
        case 10:
            c += ((uint32_t)k8[9]) << 8; /* fall through */
        case 9:
            c += k8[8]; /* fall through */
        case 8:
            b += k[1];
            a += k[0];
            break;
        case 7:
            b += ((uint32_t)k8[6]) << 16; /* fall through */
        case 6:
            b += ((uint32_t)k8[5]) << 8; /* fall through */
        case 5:
            b += k8[4]; /* fall through */
        case 4:
            a += k[0];
            break;
        case 3:
            a += ((uint32_t)k8[2]) << 16; /* fall through */
        case 2:
            a += ((uint32_t)k8[1]) << 8; /* fall through */
        case 1:
            a += k8[0];
            break;
        case 0:
            return (((uint64_t)c << 32) | b);
    }

    c ^= b;
    c -= rot(b, 14);

    a ^= c;
    a -= rot(c, 11);

    b ^= a;
    b -= rot(a, 25);

    c ^= b;
    c -= rot(b, 16);

    a ^= c;
    a -= rot(c, 4);

    b ^= a;
    b -= rot(a, 14);

    c ^= b;
    c -= rot(b, 24);

    return (((uint64_t)c << 32) | b);
}
