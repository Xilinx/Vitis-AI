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
using namespace std;
#include <sstream>
#include <string>
#include <vector>

#include <openssl/rc4.h>
#include <openssl/evp.h>

// number of times to perform the test in different text and length
// XXX notice that the datain char array should be long enough
#define NUM_TESTS 300
// cipherkey size in byte
#define KEY_SIZE 16

// print result
std::string printr(unsigned char* result, unsigned int len) {
    ostringstream oss;
    string retstr;

    // check output
    oss.str("");
    oss << hex;
    for (unsigned int i = 0; i < len; i++) {
        oss << setw(2) << setfill('0') << (unsigned)result[i];
    }
    retstr = oss.str();
    return retstr;
}

// table to save each input data and its result
struct Test {
    string data;
    unsigned char* result;
    unsigned int length;
    Test(const char* d, const char* r, unsigned int len) : data(d), length(len) {
        result = (unsigned char*)malloc(len);
        memcpy(result, r, len);
    }
};

int main() {
    std::cout << "********************************" << std::endl;
    std::cout << "   Testing RC4 on HLS project   " << std::endl;
    std::cout << "********************************" << std::endl;

    // input data
    const char datain[] =
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
        "abcdefghijklmnopqrstuvwxyz"
        "abcdefghijklmnopqrstuvwxyz";

    // cipher key
    const unsigned char key[] =
        "abcdefghijklmnopqrstuvwxyz"
        "abcdefghijklmnopqrstuvwxyz";

    // initialization vector
    const unsigned char ivec[] = "abcdefghijklmnopqrstuvwxyz";

    vector<Test> tests;

    // generate golden
    for (unsigned int i = 1; i < NUM_TESTS + 1; i++) {
        // ouput length of the result
        int outlen = 0;
        // input data length must be a multiple of 16
        unsigned int inlen = (i % 256);
        // output result buffer
        unsigned char dout[2 * inlen];

        char din[256];
        if (inlen != 0) {
            memcpy(din, datain + i, inlen);
        }
        din[inlen] = 0;
        // call OpenSSL API to get the golden
        EVP_CIPHER_CTX* ctx;
        ctx = EVP_CIPHER_CTX_new();
        EVP_CipherInit_ex(ctx, EVP_rc4(), NULL, NULL, NULL, 1);
        EVP_CIPHER_CTX_set_key_length(ctx, KEY_SIZE);
        EVP_CipherInit_ex(ctx, NULL, NULL, key, ivec, 1);
        for (unsigned int i = 0; i < inlen; i++) {
            EVP_CipherUpdate(ctx, dout + i, &outlen, (const unsigned char*)din + i, 1);
        }
        /*
for (unsigned int i = 0; i < inlen / 16; i++) {
        cout << "EVP_golden[" << dec << i << "] : " << printr((unsigned char*)dout + i * 16, 16) << endl;
}
        */
        tests.push_back(Test(datain + i, (const char*)dout, inlen));
    }

    unsigned int nerror = 0;
    unsigned int ncorrect = 0;

    hls::stream<ap_uint<8> > plaintextStrm("plaintextStrm");
    hls::stream<bool> endPlaintextStrm("endPlaintextStrm");
    hls::stream<ap_uint<8> > cipherkeyStrm("cipherkeyStrm");
    hls::stream<bool> endCipherkeyStrm("endCipherkeyStrm");
    hls::stream<ap_uint<8> > ciphertextStrm("ciphertextStrm");
    hls::stream<bool> endCiphertextStrm("endCiphertextStrm");

    for (vector<Test>::const_iterator singletest = tests.begin(); singletest != tests.end(); singletest++) {
        // generate cipherkey
        for (unsigned int i = 0; i < KEY_SIZE; i++) {
            cipherkeyStrm.write(key[i]);
            endCipherkeyStrm.write(false);
        }
        endCipherkeyStrm.write(true);

        unsigned int n = 0;
        // generate input data stream
        for (unsigned int i = 0; i < (*singletest).length; i++) {
            plaintextStrm.write((*singletest).data[i]);
            endPlaintextStrm.write(false);
        }
        endPlaintextStrm.write(true);

        // call fpga module
        test(cipherkeyStrm, endCipherkeyStrm, plaintextStrm, endPlaintextStrm, ciphertextStrm, endCiphertextStrm);

        // check result
        ap_uint<8> ciphertext[256];
        bool end = endCiphertextStrm.read();
        bool checked = true;
        int index = 0;
        while (!end) {
            ciphertext[index] = ciphertextStrm.read();
            if ((*singletest).result[index] != ciphertext[index]) {
                checked = false;
            }
            end = endCiphertextStrm.read();
            index++;
        }

        if (!checked) {
            ++nerror;
        } else {
            ++ncorrect;
        }
    }

    if (nerror) {
        cout << "FAIL: " << dec << nerror << " errors found." << endl;
    } else {
        cout << "PASS: " << dec << ncorrect << " inputs verified, no error found." << endl;
    }

    return nerror;
}
