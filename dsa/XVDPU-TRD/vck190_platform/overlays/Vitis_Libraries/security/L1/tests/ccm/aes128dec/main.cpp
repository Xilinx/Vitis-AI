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
// XXX notice that the plaintext char array should be long enough
#define NUM_TESTS 300
// cipherkey size in byte
#define KEY_SIZE 16
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
    unsigned char* result;
    unsigned int length;
    unsigned char* tag;
    Test(const char* d, const char* r, unsigned int len, const char* t) : length(len) {
        data = (unsigned char*)malloc(len);
        memcpy(data, d, len);
        result = (unsigned char*)malloc(len);
        memcpy(result, r, len);
        tag = (unsigned char*)malloc(TAG_SIZE);
        memcpy(tag, t, TAG_SIZE);
    }
};

int main() {
    std::cout << "**************************************************" << std::endl;
    std::cout << "   Testing CCM mode with AES-128 on HLS project   " << std::endl;
    std::cout << "**************************************************" << std::endl;

    // plaintext
    const unsigned char
        plaintext[] = /*{0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                                                       0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
                                                       0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
                                                       0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
                                                       0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
                                                       0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f,
                                                       0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57,
                                                       0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f};*/
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
    const unsigned char key[] = /*{0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
                                                             0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f};*/
        "abcdefghijklmnopqrstuvwxyz"
        "abcdefghijklmnopqrstuvwxyz";

    // initialization vector
    const unsigned char iv[] = /*{0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                                                            0x18, 0x19, 0x1a, 0x1b};*/
        "0000000000000000";

    // additional authenticated data
    const unsigned char aad[] = /*{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                                             0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
                                                             0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                                                             0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
                                                             0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                                                             0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
                                                             0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x27,
                                                             0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,};*/
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
    int aad_len = 64;

    vector<Test> tests;

    // generate golden
    for (unsigned int i = 0; i < NUM_TESTS; i++) {
        // ouput length of the result
        int len = 0;
        int ciphertext_len = 0;
        // input data length must be a multiple of 16
        unsigned int plaintext_len = (i % 16) * BLK_SIZE;
        // output result buffer
        unsigned char ciphertext[2 * plaintext_len];
        unsigned char tag[TAG_SIZE];

        char din[300];
        if (plaintext_len != 0) {
            memcpy(din, plaintext + i, plaintext_len);
        }
        din[plaintext_len] = 0;
        // call OpenSSL API to get the golden
        EVP_CIPHER_CTX* ctx;
        ctx = EVP_CIPHER_CTX_new();
        EVP_EncryptInit_ex(ctx, EVP_aes_128_ccm(), NULL, NULL, NULL);
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_CCM_SET_IVLEN, N_SIZE, NULL);
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_CCM_SET_TAG, TAG_SIZE, NULL);
        EVP_EncryptInit_ex(ctx, NULL, NULL, key, iv);
        EVP_EncryptUpdate(ctx, NULL, &len, NULL, plaintext_len);
        EVP_EncryptUpdate(ctx, NULL, &len, aad, aad_len);
        EVP_EncryptUpdate(ctx, ciphertext, &len, (const unsigned char*)din, plaintext_len);
        ciphertext_len = len;
        EVP_EncryptFinal_ex(ctx, ciphertext + len, &len);
        ciphertext_len += len;
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_CCM_GET_TAG, TAG_SIZE, tag);
        EVP_CIPHER_CTX_free(ctx);
        /*
        if (ciphertext_len < BLK_SIZE) {
                cout << "EVP_golden : " << printr((unsigned char*)ciphertext, ciphertext_len) << endl;
        } else {
                unsigned int index = 0;
                for (unsigned int i = 0; i < ciphertext_len / BLK_SIZE; i++) {
                        cout << "EVP_golden[" << dec << i << "] : " << printr((unsigned char*)ciphertext + i * BLK_SIZE,
        BLK_SIZE) << endl;
                        index++;
                }
                if (ciphertext_len % BLK_SIZE > 0) {
                        cout << "EVP_golden[" << dec << index << "] : " << printr((unsigned char*)ciphertext + index *
        BLK_SIZE, ciphertext_len % BLK_SIZE) << endl;
                }
        }
        cout << "tag_golden : " << printr((unsigned char*)tag, TAG_SIZE) << endl;
        */
        tests.push_back(Test((const char*)plaintext + i, (const char*)ciphertext, plaintext_len, (const char*)tag));
    }

    unsigned int nerror = 0;
    unsigned int ncorrect = 0;

    hls::stream<ap_uint<128> > payloadStrm("payloadStrm");
    hls::stream<ap_uint<8 * KEY_SIZE> > cipherkeyStrm("cipherkeyStrm");
    hls::stream<ap_uint<8 * N_SIZE> > nonceStrm("nonceStrm");
    hls::stream<ap_uint<128> > ADStrm("ADStrm");
    hls::stream<ap_uint<64> > lenADStrm("lenADStrm");
    hls::stream<ap_uint<64> > lenPldStrm("lenPldStrm");
    hls::stream<bool> endLenStrm("endLenStrm");
    hls::stream<ap_uint<128> > cipherStrm("cipherStrm");
    hls::stream<ap_uint<64> > lenCphStrm("lenCphStrm");
    hls::stream<ap_uint<8 * TAG_SIZE> > tagStrm("tagStrm");
    hls::stream<bool> endTagStrm("endTagStrm");

    for (vector<Test>::const_iterator singletest = tests.begin(); singletest != tests.end(); singletest++) {
        // generate cipherkey
        ap_uint<8 * KEY_SIZE> keyReg;
        for (unsigned int i = 0; i < KEY_SIZE; i++) {
            keyReg.range(i * 8 + 7, i * 8) = key[i];
        }
        cipherkeyStrm.write(keyReg);

        // generate nonce
        ap_uint<8 * N_SIZE> nonceReg = 0;
        for (unsigned int i = 0; i < N_SIZE; i++) {
            nonceReg.range(i * 8 + 7, i * 8) = iv[i];
        }
        nonceStrm.write(nonceReg);

        // generate associated data
        ap_uint<8 * BLK_SIZE> ADReg = 0;
        unsigned int index = 0;
        for (unsigned int j = 0; j < aad_len / BLK_SIZE; j++) {
            for (unsigned int i = 0; i < BLK_SIZE; i++) {
                ADReg.range(i * 8 + 7, i * 8) = aad[i + index];
            }
            index += BLK_SIZE;
            ADStrm.write(ADReg);
        }
        if (aad_len % BLK_SIZE > 0) {
            ADReg = 0;
            for (unsigned int i = 0; i < aad_len % BLK_SIZE; i++) {
                ADReg.range(i * 8 + 7, i * 8) = aad[i + index];
            }
            ADStrm.write(ADReg);
        }

        // write length streams
        lenPldStrm.write((*singletest).length);
        lenADStrm.write(aad_len);
        endLenStrm.write(false);

        ap_uint<8 * BLK_SIZE> payloadReg;
        unsigned int n = 0;
        // generate input data stream
        for (unsigned int i = 0; i < (*singletest).length; i++) {
            if (n == 0) {
                payloadReg = 0;
            }
            payloadReg.range(7 + 8 * n, 8 * n) = (unsigned char)((*singletest).result[i]);
            n++;
            if (n == BLK_SIZE) {
                payloadStrm.write(payloadReg);
                n = 0;
            }
        }
        // deal with the condition that we didn't hit a boundary of the last block
        if (n != 0) {
            payloadStrm.write(payloadReg);
        }
    }
    endLenStrm.write(true);

    cout << "\n" << NUM_TESTS << " inputs ready..." << endl;

    // call fpga module
    test(payloadStrm, cipherkeyStrm, nonceStrm, ADStrm, lenADStrm, lenPldStrm, endLenStrm, cipherStrm, lenCphStrm,
         tagStrm, endTagStrm);

    // check result
    ap_uint<8 * BLK_SIZE> cipher;
    ap_uint<8 * BLK_SIZE> cipher_golden;
    ap_uint<8 * TAG_SIZE> tag;
    ap_uint<8 * TAG_SIZE> tag_golden;
    ap_uint<64> lenCipher;
    for (std::vector<Test>::const_iterator singletest = tests.begin(); singletest != tests.end(); ++singletest) {
        bool x = endTagStrm.read();
        lenCipher = lenCphStrm.read();

        bool checked = true;
        int index = 0;
        // generate input data stream
        for (unsigned int i = 0; i < (*singletest).length / 16; i++) {
            cipher = cipherStrm.read();
            for (unsigned int j = 0; j < BLK_SIZE; j++) {
                cipher_golden.range(7 + 8 * j, 8 * j) = (unsigned char)((*singletest).data[j + index]);
            }
            index += BLK_SIZE;
            if (cipher != cipher_golden) {
                checked = false;
                cout << "fpga_ciphertext   : " << hex << cipher << endl;
                cout << "golden_ciphertext : " << hex << cipher_golden << endl;
            }
        }
        if ((*singletest).length % 16 > 0) {
            cipher = cipherStrm.read();
            for (unsigned int j = 0; j < (*singletest).length % 16; j++) {
                cipher_golden.range(7 + 8 * j, 8 * j) = (unsigned char)((*singletest).data[j + index]);
            }
            if (cipher.range((*singletest).length % 16 * 8 - 1, 0) !=
                cipher_golden((*singletest).length % 16 * 8 - 1, 0)) {
                checked = false;
                cout << "fpga_ciphertext   : " << hex << cipher << endl;
                cout << "golden_ciphertext : " << hex << cipher_golden << endl;
            }
        }

        tag = tagStrm.read();
        for (unsigned int i = 0; i < TAG_SIZE; i++) {
            tag_golden.range(7 + 8 * i, 8 * i) = (unsigned char)((*singletest).tag[i]);
        }
        if (tag_golden != tag) {
            checked = false;
            cout << "fpga_tag   : " << hex << tag << endl;
            cout << "golden_tag : " << hex << tag_golden << endl;
        }

        if (!checked) {
            ++nerror;
        } else {
            ++ncorrect;
        }
    }
    bool x = endTagStrm.read();

    if (nerror) {
        cout << "FAIL: " << dec << nerror << " errors found." << endl;
    } else {
        cout << "PASS: " << dec << ncorrect << " inputs verified, no error found." << endl;
    }

    return nerror;
}
