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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <ap_int.h>

#include "xf_security/sha224_256.hpp"

// For verification
#include <openssl/sha.h>

#define NUM_TESTS 1000

#define M_SZ 8

#define _STR_CON(a, b) a##b
#define STR_CON(a, b) _STR_CON(a, b)

void DUT_FUNC( //
    hls::stream<ap_uint<8 * M_SZ> >& msg_strm,
    hls::stream<ap_uint<64> >& len_strm,
    hls::stream<bool>& end_len_strm,
    //
    hls::stream<ap_uint<224> >& hash_strm,
    hls::stream<bool>& end_hash_strm) {
    xf::security::sha224(msg_strm, len_strm, end_len_strm, hash_strm, end_hash_strm);
}

struct Test {
    std::string msg;
    unsigned char h224[28];
    Test(const char* m, const void* h) : msg(m) { memcpy(h224, h, 28); }
};

std::string hash2str(char* h, int len) {
    std::ostringstream oss;
    std::string retstr;

    // check output
    oss.str("");
    oss << std::hex;
    for (int i = 0; i < len; ++i) {
        oss << std::setw(2) << std::setfill('0') << (unsigned)h[i];
    }
    retstr = oss.str();
    return retstr;
}

int main(int argc, const char* argv[]) {
    const char message[] =
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopqabcdabcd"
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopqabcdabc";

    std::vector<Test> tests;
    /* these values can be generated with
     *   echo -n "abc" | sha256sum,
     * where -n prevents echo to add \n after abc.
     */

    for (int i = 0; i < NUM_TESTS; ++i) {
        int len = i % 128;
        char m[128];
        if (len != 0) memcpy(m, message, len);
        m[len] = 0;
        unsigned char h[28];
        SHA224((const unsigned char*)message, len, (unsigned char*)h);
        tests.push_back(Test(m, h));
    }

    int nerror = 0;
    int ncorrect = 0;
    hls::stream<ap_uint<8 * M_SZ> > msg_strm("msg_strm");
    hls::stream<ap_uint<64> > len_strm("len_strm");
    hls::stream<bool> end_len_strm("end_len_strm");
    hls::stream<ap_uint<224> > hash_strm("hash_strm");
    hls::stream<bool> end_hash_strm("end_hash_strm");

    for (std::vector<Test>::const_iterator test = tests.begin(); test != tests.end(); ++test) {
        // std::cout << "\nmessage: \"" << (*test).msg << "\"\n";

        // prepare input
        ap_uint<8 * M_SZ> m;
        int n = 0;
        int cnt = 0;
        for (std::string::size_type i = 0; i < (*test).msg.length(); ++i) {
            if (n == 0) {
                m = 0;
            }
            m.range(7 + 8 * n, 8 * n) = (unsigned)((*test).msg[i]);
            ++n;
            if (n == M_SZ) {
                msg_strm.write(m);
                ++cnt;
                n = 0;
            }
        }
        if (n != 0) {
            msg_strm.write(m);
            ++cnt;
        }
        len_strm.write((unsigned long long)((*test).msg.length()));
        end_len_strm.write(false);

#ifdef DEBUG_VERBOSE
        std::cout << "\nmessage: \"" << (*test).msg << "\"(" << (*test).msg.length() << ") (" << cnt << " words)\n";
#endif
    }
    end_len_strm.write(true);

    std::cout << "\n" << NUM_TESTS << " inputs ready...\n";

    // call module
    DUT_FUNC(msg_strm, len_strm, end_len_strm, hash_strm, end_hash_strm);

    // check result
    for (std::vector<Test>::const_iterator test = tests.begin(); test != tests.end(); ++test) {
#ifdef DEBUG_VERBOSE
        std::cout << "\nmessage: \"" << (*test).msg << "\"(" << (*test).msg.length() << ")\n";
#endif

        ap_uint<224> h224 = hash_strm.read();
        bool x = end_hash_strm.read();

        unsigned char h[28];
        for (int i = 0; i < 28; ++i) {
            h[i] = (unsigned char)(h224.range(7 + 8 * i, 8 * i).to_int() & 0xff);
        }

#ifdef DEBUG_VERBOSE
        std::cout << "return: " << hash2str((char*)h, 28) << "\n";
#endif

        if (memcmp((*test).h224, h, 28)) {
            ++nerror;
#ifndef DEBUG_VERBOSE
            std::cout << "\nmessage: \"" << (*test).msg << "\"(" << (*test).msg.length() << ")\n";
            std::cout << "return: " << hash2str((char*)h, 28) << "\n";
#endif
            std::cout << "golden: " << hash2str((char*)(*test).h224, 28) << "\n";
        } else {
            ++ncorrect;
        }
    }
    bool x = end_hash_strm.read();

    if (nerror) {
        std::cout << "\nFAIL: " << nerror << " errors found.\n";
    } else {
        std::cout << "\nPASS: " << ncorrect << " inputs verified, no error found.\n";
    }
    return nerror;
}
