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

#include "test.hpp"

#include <hls_stream.h>
#include <ap_int.h>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

// number of times to perform the test in different message length
#define NUM_TESTS 2
// the size of each message word in byte
#define MSG_SIZE 8
// the maximum size of the digest in byte
#define DIG_SIZE 64

// table to save each message and its hash value
struct Test {
    string msg;
    unsigned int msg_len;
    string key;
    ap_uint<8> key_len;
    unsigned char hash[DIG_SIZE];
    ap_uint<8> hash_len;
    Test(const char* m, unsigned int m_len, const char* k, unsigned int k_len, const void* h, unsigned int h_len)
        : msg(m), msg_len(m_len), key(k), key_len(k_len), hash_len(h_len) {
        memcpy(hash, h, DIG_SIZE);
    }
};

// print hash value
string hash2str(unsigned char* h, int len) {
    ostringstream oss;
    string retstr;

    // check output
    oss.str("");
    oss << hex;
    for (int i = 0; i < len; i++) {
        oss << setw(2) << setfill('0') << (unsigned)h[i];
    }
    retstr = oss.str();
    return retstr;
}

int main() {
    std::cout << "************************************" << std::endl;
    std::cout << "   Testing BLAKE2B on HLS project   " << std::endl;
    std::cout << "************************************" << std::endl;

    // the original message to be digested
    const char message[] =
        "abcdefghijklmnopqrstuvwxyz"
        "abcdefghijklmnopqrstuvwxyz"
        "abcdefghijklmnopqrstuvwxyz"
        "abcdefghijklmnopqrstuvwxyz"
        "abcdefghijklmnopqrstuvwxyz"
        "abcdefghijklmnopqrstuvwxyz"
        "abcdefghijklmnopqrstuvwxyz"
        "abcdefghijklmnopqrstuvwxyz"
        "abcdefghijklmnopqrstuvwxyz"
        "abcdefghijklmnopqrstuvwxyz"
        "abcdefghijklmnopqrstuvwxyz";

    // the optional key
    const char key[] =
        "abcdefghijklmnopqrstuvwxyz"
        "abcdefghijklmnopqrstuvwxyz"
        "abcdefghijklmnopqrstuvwxyz";

    // open file
    FILE* fp = fopen("testcases.dat", "r");

    vector<Test> tests;

    // generate golden
    for (ap_uint<8> key_len = 0; key_len <= 64; key_len++) {
        for (ap_uint<8> h_len = 1; h_len <= 64; h_len++) {
            for (unsigned int i = 0; i < NUM_TESTS; i++) {
                unsigned int msg_len = i % 256;
                char m[256];
                char k[65];
                if (msg_len != 0) {
                    memcpy(m, message, msg_len);
                }
                if (key_len != 0) {
                    memcpy(k, key, key_len);
                }
                m[msg_len] = 0;
                k[key_len] = 0;
                unsigned char h[DIG_SIZE];
                // read golden hash values from testcases.dat
                fread(h, sizeof(unsigned char), (unsigned int)h_len, fp);
                tests.push_back(Test(m, msg_len, k, key_len, h, h_len));
            }
        }
    }
    // close file
    fclose(fp);

    unsigned int nerror = 0;
    unsigned int ncorrect = 0;

    hls::stream<ap_uint<8 * MSG_SIZE> > msg_strm("msg_strm");
    hls::stream<ap_uint<6> > msg_len_strm("msg_len_strm");
    hls::stream<ap_uint<512> > key_strm("key_strm");
    hls::stream<ap_uint<8> > key_len_strm("key_len_strm");
    hls::stream<ap_uint<8> > out_len_strm("out_len_strm");
    hls::stream<ap_uint<8 * DIG_SIZE> > digest_strm("digest_strm");
    hls::stream<bool> end_digest_strm("end_digest_strm");

    // generate input message words
    for (vector<Test>::const_iterator test = tests.begin(); test != tests.end(); test++) {
        ap_uint<8 * MSG_SIZE> msg;
        ap_uint<8 * MSG_SIZE> key;
        ap_uint<8> key_len = (*test).key_len;
        ap_uint<8> h_len = (*test).hash_len;
        unsigned int n = 0;
        // write msg stream word by word
        if ((*test).msg.length() > 0) {
            for (string::size_type i = 0; i < (*test).msg.length(); i++) {
                if (n == 0) {
                    msg = 0;
                }
                msg.range(7 + 8 * n, 8 * n) = (unsigned)((*test).msg[i]);
                n++;
                if (n == MSG_SIZE) {
                    msg_strm.write(msg);

                    ap_uint<6> msg_len = n;
                    if ((i + 1) == (*test).msg.length()) {
                        msg_len[4] = 1;
                    }
                    msg_len_strm.write(msg_len);

                    n = 0;
                }
            }
            // deal with the condition that we didn't hit a boundary of the last word
            if (n != 0) {
                msg_strm.write(msg);
                ap_uint<6> msg_len = n;
                msg_len[4] = 1;
                msg_len_strm.write(msg_len);
            }
        } else {
            msg_strm.write(0);
            msg_len_strm.write(ap_uint<6>(1 << 4));
        }
        // write key stream word by word
        ap_uint<512> tmp_k = 0;
        for (string::size_type i = 0; i < (*test).key.length(); i++) {
            tmp_k.range(i * 8 + 7, i * 8) = (unsigned)((*test).key[i]);
        }
        key_strm.write(tmp_k);

        // inform the prmitive how many bytes do we have in this message
        key_len_strm.write((unsigned long long)((*test).key.length()));
        out_len_strm.write(h_len);
    }
    msg_len_strm.write(ap_uint<6>(1 << 5));

    // call fpga module
    test(msg_strm, msg_len_strm, key_strm, key_len_strm, out_len_strm, digest_strm, end_digest_strm);

    // check result
    for (vector<Test>::const_iterator test = tests.begin(); test != tests.end(); test++) {
        ap_uint<8 * DIG_SIZE> digest = digest_strm.read();
        bool x = end_digest_strm.read();

        unsigned char hash[DIG_SIZE];
        for (unsigned int i = 0; i < DIG_SIZE; i++) {
            hash[i] = (unsigned char)(digest.range(7 + 8 * i, 8 * i).to_int() & 0xff);
        }

        if (memcmp((*test).hash, hash, (*test).hash_len)) {
            ++nerror;
            std::cout << "fpga   : " << hash2str((unsigned char*)hash, (*test).hash_len) << std::endl;
            std::cout << "golden : " << hash2str((unsigned char*)(*test).hash, (*test).hash_len) << std::endl;
        } else {
            ++ncorrect;
        }
    }

    bool x = end_digest_strm.read();

    if (nerror) {
        std::cout << "FAIL: " << dec << nerror << " errors found." << std::endl;
    } else {
        std::cout << "PASS: " << dec << ncorrect << " inputs verified, no error found." << std::endl;
    }

    return nerror;
}
