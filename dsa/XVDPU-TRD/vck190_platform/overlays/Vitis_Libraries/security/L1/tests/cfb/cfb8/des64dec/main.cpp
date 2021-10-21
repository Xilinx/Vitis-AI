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

#include <openssl/aes.h>
#include <openssl/evp.h>

// number of times to perform the test in different text and length
// XXX notice that the datain char array should be long enough
#define NUM_TESTS 30
// cipherkey size in byte
#define KEY_SIZE 8
// cipher block size in byte
#define BLK_SIZE 8

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
    std::cout << "***********************************************" << std::endl;
    std::cout << "   Testing CFB8 mode with DES on HLS project   " << std::endl;
    std::cout << "***********************************************" << std::endl;

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
        int outlen1 = 0;
        int outlen2 = 0;
        // input data length must be a multiple of 16
        unsigned int inlen = (i % 16) * BLK_SIZE;
        // output result buffer
        unsigned char dout[2 * inlen];

        char din[256];
        if (inlen != 0) {
            memcpy(din, datain + i, inlen);
        }
        din[inlen] = 0;
        cout << "inlen = " << dec << inlen << endl;
        // call OpenSSL API to get the golden
        EVP_CIPHER_CTX ctx;
        EVP_DecryptInit(&ctx, EVP_des_cfb8(), key, ivec);
        EVP_DecryptUpdate(&ctx, dout, &outlen1, (const unsigned char*)din, inlen);
        EVP_DecryptFinal(&ctx, dout + outlen1, &outlen2);
        cout << "outlen1 = " << dec << outlen1 << endl;
        cout << "outlen2 = " << dec << outlen2 << endl;
        /*
        for (unsigned int i = 0; i < inlen / BLK_SIZE + 1; i++) {
                cout << "EVP_golden[" << dec << i << "] : " << printr((unsigned char*)dout + i * BLK_SIZE, BLK_SIZE) <<
        endl;
        }
        */
        tests.push_back(Test(datain + i, (const char*)dout, inlen));
    }

    unsigned int nerror = 0;
    unsigned int ncorrect = 0;

    hls::stream<ap_uint<64> > ciphertextStrm("ciphertextStrm");
    hls::stream<bool> endCiphertextStrm("endCiphertextStrm");
    hls::stream<ap_uint<8 * KEY_SIZE> > cipherkeyStrm("cipherkeyStrm");
    hls::stream<ap_uint<64> > IVStrm("IVStrm");
    hls::stream<ap_uint<64> > plaintextStrm("plaintextStrm");
    hls::stream<bool> endPlaintextStrm("endPlaintextStrm");

    for (vector<Test>::const_iterator singletest = tests.begin(); singletest != tests.end(); singletest++) {
        // generate cipherkey
        ap_uint<8 * KEY_SIZE> keyReg;
        for (unsigned int i = 0; i < KEY_SIZE; i++) {
            keyReg.range(i * 8 + 7, i * 8) = key[i];
        }
        cipherkeyStrm.write(keyReg);

        // generate initialization vector
        ap_uint<64> IVReg;
        for (unsigned int i = 0; i < 8; i++) {
            IVReg.range(i * 8 + 7, i * 8) = ivec[i];
        }
        IVStrm.write(IVReg);

        ap_uint<8 * BLK_SIZE> datainReg;
        unsigned int n = 0;
        // generate input data stream
        for (unsigned int i = 0; i < (*singletest).length; i++) {
            if (n == 0) {
                datainReg = 0;
            }
            datainReg.range(7 + 8 * n, 8 * n) = (unsigned char)((*singletest).data[i]);
            n++;
            if (n == BLK_SIZE) {
                ciphertextStrm.write(datainReg);
                endCiphertextStrm.write(false);
                n = 0;
            }
        }
        // deal with the condition that we didn't hit a boundary of the last block
        if (n != 0) {
            datainReg.range(7 + 8 * n, 8 * n) = 0x01UL;
            ciphertextStrm.write(datainReg);
            endCiphertextStrm.write(false);
        }
        endCiphertextStrm.write(true);

        // call fpga module
        test(ciphertextStrm, endCiphertextStrm, cipherkeyStrm, IVStrm, plaintextStrm, endPlaintextStrm);

        // check result
        ap_uint<8 * BLK_SIZE> plaintext;
        ap_uint<8 * BLK_SIZE> golden;
        bool end = endPlaintextStrm.read();
        bool checked = true;
        int index = 0;
        while (!end) {
            plaintext = plaintextStrm.read();
            for (unsigned int i = 0; i < BLK_SIZE; i++) {
                golden.range(7 + 8 * i, 8 * i) = (unsigned char)((*singletest).result[i + index]);
            }
            if (golden != plaintext) {
                checked = false;
            }
            end = endPlaintextStrm.read();
            index += BLK_SIZE;
        }

        if (!checked) {
            ++nerror;
            cout << "fpga   : " << hex << plaintext << endl;
            cout << "golden : " << hex << golden << endl;
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
