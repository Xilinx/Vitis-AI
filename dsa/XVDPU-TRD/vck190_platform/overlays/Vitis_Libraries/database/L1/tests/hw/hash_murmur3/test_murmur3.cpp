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
#include <sstream>
#include <string>
#include <vector>

#define AP_INT_MAX_W 4096
#include "xf_database/hash_murmur3.hpp"

#define IN_W 512
#define OUT_W 32

void hashmurmur3_W512_H32_strm(hls::stream<ap_uint<IN_W> >& key_strm, hls::stream<ap_uint<OUT_W> >& out_strm) {
#pragma HLS PIPELINE II = 1
    xf::database::hashMurmur3(key_strm, out_strm);
}

uint32_t hashmurmur3_x86_32(const void* key, int len, uint32_t seed);

struct Test {
    std::string key;
    Test(const char* k) : key(k) {}
};

int main(int argc, const char* argv[]) {
    std::vector<Test> tests;

    tests.push_back(
        Test("This is the time for all good men to come to the aid of their "
             "country..."));
    /* test the https */
    tests.push_back(
        Test("https://blog.csdn.net/hengyunabc/article/details/5914934-test our "
             "murmur3"));
    tests.push_back(Test("http://burtleburtle.net/bob/hash/doobs.html                      "));
    tests.push_back(Test("http://confluence.xilinx.com/display/XSW/Perforce+Getting+Started"));
    tests.push_back(
        Test("http://confluence.xilinx.com/display/XSW/"
             "Setting+up+arcanist+and+upload+patch+to+phabricator"));
    /* test the same word dif space , https*/
    tests.push_back(Test("http://burtleburtle.net/bob/c/lookup3.c                          "));
    tests.push_back(Test(" http://burtleburtle.net/bob/c/lookup3.c                         "));
    tests.push_back(Test("http://burtleburtle.net/bob/c/lookup3.c"));
    tests.push_back(Test("https://burtleburtle.net/bob/c/lookup3.c                         "));

    int nerror = 0;
    hls::stream<ap_uint<IN_W> > key_strm("key_strm");
    hls::stream<ap_uint<OUT_W> > hash_strm("hash_strm");

    // loop 140*8 test
    for (int t = 0; t < 140; t++) {
        for (std::vector<Test>::const_iterator test = tests.begin(); test != tests.end(); ++test) {
            // prepare input
            ap_uint<IN_W> key;
            for (std::string::size_type i = 0; i < IN_W / 8; ++i) {
                key(8 * i + 7, 8 * i) = ((unsigned char)((*test).key[i]));
            }
            key_strm.write(key);
            uint8_t* q = (uint8_t*)(*test).key.data();

            // call module
            uint32_t ret_c = hashmurmur3_x86_32(q, IN_W / 8, 13);
            hashmurmur3_W512_H32_strm(key_strm, hash_strm);

            uint32_t ret_strm = (unsigned)hash_strm.read();
            std::cout << "\nkey: \"" << (*test).key << "\"\nmurmur3_c: " << std::hex << ret_c << "\n";
            std::cout << "murmur3_hash: " << std::hex << ret_strm << "\n";
            if (ret_strm != ret_c) ++nerror;
        }
    }

    if (nerror) {
        std::cout << "\nFAIL: " << nerror << " errors found.\n";
    } else {
        std::cout << "\nPASS: no error found.\n";
    }
    return nerror;
}

#define FORCE_INLINE inline __attribute__((always_inline))

inline uint32_t rotl32(uint32_t x, int8_t r) {
    return (x << r) | (x >> (32 - r));
}

#define ROTL32(x, y) rotl32(x, y)
#define ROTL64(x, y) rotl64(x, y)

#define BIG_CONSTANT(x) (x##LLU)

//-----------------------------------------------------------------------------
// Block read - if your platform needs to do endian-swapping or can only
// handle aligned reads, do the conversion here
FORCE_INLINE uint32_t getblock32(const uint32_t* p, int i) {
    return p[i];
}
//-----------------------------------------------------------------------------
// Finalization mix - force all bits of a hash block to avalanche

FORCE_INLINE uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    return h;
}
uint32_t hashmurmur3_x86_32(const void* key, int len, uint32_t seed) {
    const uint8_t* data = (const uint8_t*)key;
    const int nblocks = len / 4;
    uint32_t out;

    uint32_t h1 = seed;

    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    //----------
    // body

    const uint32_t* blocks = (const uint32_t*)(data + nblocks * 4);

    for (int i = -nblocks; i; i++) {
        uint32_t k1 = getblock32(blocks, i);

        k1 *= c1;
        k1 = ROTL32(k1, 15);
        k1 *= c2;

        h1 ^= k1;
        h1 = ROTL32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64;
    }

    //----------
    // tail

    const uint8_t* tail = (const uint8_t*)(data + nblocks * 4);

    uint32_t k1 = 0;

    switch (len & 3) {
        case 3:
            k1 ^= tail[2] << 16;
            break;
        case 2:
            k1 ^= tail[1] << 8;
            break;
        case 1:
            k1 ^= tail[0];
            k1 *= c1;
            k1 = ROTL32(k1, 15);
            k1 *= c2;
            h1 ^= k1;
            break;
        default:; // do nothing
    };

    //----------
    // finalization
    h1 ^= len;

    h1 = fmix32(h1);
    return h1; // 20180416
}
