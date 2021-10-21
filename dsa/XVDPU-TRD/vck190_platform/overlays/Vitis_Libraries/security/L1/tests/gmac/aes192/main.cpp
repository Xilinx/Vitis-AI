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
#define NUM_TESTS 300
// cipherkey size in byte
#define KEY_SIZE 24
// cipher block size in byte
#define BLK_SIZE 16

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
    unsigned char* data;
    unsigned int length;
    unsigned char* tag;
    Test(const char* d, unsigned int len, const char* t) : length(len) {
        data = (unsigned char*)malloc(len);
        memcpy(data, d, len);
        tag = (unsigned char*)malloc(BLK_SIZE);
        memcpy(tag, t, BLK_SIZE);
    }
};

int main() {
    std::cout << "**********************************************" << std::endl;
    std::cout << "   Testing GMAC with AES-192 on HLS project   " << std::endl;
    std::cout << "**********************************************" << std::endl;

    // additional authenticated data
    const unsigned char aad[] =
        /*
         {0xD6, 0x09, 0xB1, 0xF0, 0x56, 0x63, 0x7A, 0x0D,
         0x46, 0xDF, 0x99, 0x8D, 0x88, 0xE5, 0x22, 0x2A,
         0xB2, 0xC2, 0x84, 0x65, 0x12, 0x15, 0x35, 0x24,
         0xC0, 0x89, 0x5E, 0x81, 0x08, 0x00, 0x0F, 0x10,
         0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
         0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20,
         0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
         0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30,
         0x31, 0x32, 0x33, 0x34, 0x00, 0x01};
         */
        /*
{0xE2, 0x01, 0x06, 0xD7, 0xCD, 0x0D, 0xF0, 0x76,
 0x1E, 0x8D, 0xCD, 0x3D, 0x88, 0xE5, 0x40, 0x00,
 0x76, 0xD4, 0x57, 0xED, 0x08, 0x00, 0x0F, 0x10,
 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20,
 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30,
 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38,
 0x39, 0x3A, 0x00, 0x03};
 */
        /*
{0x84, 0xC5, 0xD5, 0x13, 0xD2, 0xAA, 0xF6, 0xE5,
 0xBB, 0xD2, 0x72, 0x77, 0x88, 0xE5, 0x23, 0x00,
 0x89, 0x32, 0xD6, 0x12, 0x7C, 0xFD, 0xE9, 0xF9,
 0xE3, 0x37, 0x24, 0xC6, 0x08, 0x00, 0x0F, 0x10,
 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20,
 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30,
 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38,
 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F, 0x00,
 0x05};
 */
        /*
{0x68, 0xF2, 0xE7, 0x76, 0x96, 0xCE, 0x7A, 0xE8,
 0xE2, 0xCA, 0x4E, 0xC5, 0x88, 0xE5, 0x41, 0x00,
 0x2E, 0x58, 0x49, 0x5C, 0x08, 0x00, 0x0F, 0x10,
 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20,
 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30,
 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38,
 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F, 0x40,
 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x00, 0x07};
 */

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
        "abcdefghijklmnopqrstuvwxyz"
        "abcdefghijklmnopqrstuvwxyz"
        "abcdefghijklmnopqrstuvwxyz"
        "abcdefghijklmnopqrstuvwxyz"
        "abcdefghijklmnopqrstuvwxyz";

    // cipher key
    const unsigned char key[] = /*
                                                                {0xE3, 0xC0, 0x8A, 0x8F, 0x06, 0xC6, 0xE3, 0xAD,
                                                                 0x95, 0xA7, 0x05, 0x57, 0xB2, 0x3F, 0x75, 0x48,
                                                                 0x3C, 0xE3, 0x30, 0x21, 0xA9, 0xC7, 0x2B, 0x70,
                                                                 0x25, 0x66, 0x62, 0x04, 0xC6, 0x9C, 0x0B, 0x72};
                                                                 */
        /*
        {0x69, 0x1D, 0x3E, 0xE9, 0x09, 0xD7, 0xF5, 0x41,
         0x67, 0xFD, 0x1C, 0xA0, 0xB5, 0xD7, 0x69, 0x08,
         0x1F, 0x2B, 0xDE, 0x1A, 0xEE, 0x65, 0x5F, 0xDB,
         0xAB, 0x80, 0xBD, 0x52, 0x95, 0xAE, 0x6B, 0xE7};
         */
        /*
        {0x83, 0xC0, 0x93, 0xB5, 0x8D, 0xE7, 0xFF, 0xE1,
         0xC0, 0xDA, 0x92, 0x6A, 0xC4, 0x3F, 0xB3, 0x60,
         0x9A, 0xC1, 0xC8, 0x0F, 0xEE, 0x1B, 0x62, 0x44,
         0x97, 0xEF, 0x94, 0x2E, 0x2F, 0x79, 0xA8, 0x23};
         */
        {0x4C, 0x97, 0x3D, 0xBC, 0x73, 0x64, 0x62, 0x16, 0x74, 0xF8, 0xB5, 0xB8, 0x9E, 0x5C, 0x15, 0x51,
         0x1F, 0xCE, 0xD9, 0x21, 0x64, 0x90, 0xFB, 0x1C, 0x1A, 0x2C, 0xAA, 0x0F, 0xFE, 0x04, 0x07, 0xE5};

    // initialization vector
    const unsigned char iv[] =
        /*
        {0x12, 0x15, 0x35, 0x24, 0xC0, 0x89, 0x5E, 0x81, 0xB2, 0xC2, 0x84, 0x65};
        */
        /*
        {0xF0, 0x76, 0x1E, 0x8D, 0xCD, 0x3D, 0x00, 0x01, 0x76, 0xD4, 0x57, 0xED};
        */
        /*
        {0x7C, 0xFD, 0xE9, 0xF9, 0xE3, 0x37, 0x24, 0xC6, 0x89, 0x32, 0xD6, 0x12};
        */
        {0x7A, 0xE8, 0xE2, 0xCA, 0x4E, 0xC5, 0x00, 0x01, 0x2E, 0x58, 0x49, 0x5C};
    int iv_len = 12;

    vector<Test> tests;

    // generate golden
    for (unsigned int i = 0; i < NUM_TESTS; i++) {
        // length of the additional authenticated data
        int aad_len = i % 256;
        // int aad_len = 87;
        int unused;

        // output result buffer
        unsigned char tag[BLK_SIZE];

        char din[300];
        memcpy(din, aad + i, aad_len);

        // call OpenSSL API to get the golden
        EVP_CIPHER_CTX* ctx;
        ctx = EVP_CIPHER_CTX_new();
        EVP_EncryptInit_ex(ctx, EVP_aes_192_gcm(), NULL, NULL, NULL);
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, iv_len, NULL);
        EVP_EncryptInit_ex(ctx, NULL, NULL, key, iv);
        EVP_EncryptUpdate(ctx, NULL, &unused, aad + i, aad_len);
        EVP_EncryptFinal_ex(ctx, NULL, &unused);
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, BLK_SIZE, tag);
        EVP_CIPHER_CTX_free(ctx);
        /*
cout << "tag_golden : " << printr((unsigned char*)tag, BLK_SIZE) << endl;
        */
        tests.push_back(Test((const char*)aad + i, aad_len, (const char*)tag));
    }

    unsigned int nerror = 0;
    unsigned int ncorrect = 0;

    hls::stream<ap_uint<128> > AADStrm("AADStrm");
    hls::stream<ap_uint<64> > lenAADStrm("lenAADStrm");
    hls::stream<ap_uint<8 * KEY_SIZE> > cipherkeyStrm("cipherkeyStrm");
    hls::stream<ap_uint<96> > IVStrm("IVStrm");
    hls::stream<ap_uint<128> > tagStrm("tagStrm");

    for (vector<Test>::const_iterator singletest = tests.begin(); singletest != tests.end(); singletest++) {
        // generate cipherkey
        ap_uint<8 * KEY_SIZE> keyReg;
        for (unsigned int i = 0; i < KEY_SIZE; i++) {
            keyReg.range(i * 8 + 7, i * 8) = key[i];
        }
        cipherkeyStrm.write(keyReg);

        // generate initialization vector
        ap_uint<128> IVReg;
        for (unsigned int i = 0; i < 12; i++) {
            IVReg.range(i * 8 + 7, i * 8) = iv[i];
        }
        IVStrm.write(IVReg);

        ap_uint<128> AADReg;
        unsigned int n = 0;
        // generate additional authenticated data stream
        for (unsigned int i = 0; i < (*singletest).length; i++) {
            if (n == 0) {
                AADReg = 0;
            }
            AADReg.range(7 + 8 * n, 8 * n) = (unsigned char)((*singletest).data[i]);
            n++;
            if (n == BLK_SIZE) {
                AADStrm.write(AADReg);
                n = 0;
            }
        }
        // deal with the condition that we didn't hit a boundary of the last block
        if (n != 0) {
            AADStrm.write(AADReg);
        }
        lenAADStrm.write((*singletest).length * 8);

        // call fpga module
        test(AADStrm, lenAADStrm, cipherkeyStrm, IVStrm, tagStrm);

        // check result
        ap_uint<8 * BLK_SIZE> tag;
        ap_uint<8 * BLK_SIZE> tag_golden;
        bool checked = true;
        tag = tagStrm.read();
        for (unsigned int i = 0; i < BLK_SIZE; i++) {
            tag_golden.range(7 + 8 * i, 8 * i) = (unsigned char)((*singletest).tag[i]);
        }
        if (tag_golden != tag) {
            checked = false;
        }

        if (!checked) {
            ++nerror;
            cout << "fpga_tag   : " << hex << tag << endl;
            cout << "golden_tag : " << hex << tag_golden << endl;
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
