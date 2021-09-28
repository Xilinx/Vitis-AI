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

#include <ap_int.h>
#include <hls_stream.h>
#define LEN 1023
#define KEY_LEN 3
#define XF_SECURITY_DECRYPT_DEBUG 1
#include "xf_security/chacha20.hpp"
#include "test_vector.hpp"
#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
#include <iostream>
#endif
void chacha20Core(hls::stream<ap_uint<256> >& keyStrm,
                  hls::stream<ap_uint<128> >& counterStrm,
                  hls::stream<ap_uint<512> >& plainStream,
                  hls::stream<bool>& ePlainStream,
                  hls::stream<ap_uint<512> >& cipherStream,
                  hls::stream<bool>& eCipherStream) {
    xf::security::chacha20(keyStrm, counterStrm, plainStream, ePlainStream, cipherStream, eCipherStream);
}

void print(ap_uint<8> arr[], int len) {
#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
    for (int i = 0; i < len; i++) {
        std::cout << std::hex << arr[i];
    }
    std::cout << std::endl;
#endif
}

struct tv {
    unsigned char* key;
    int counter;
    unsigned char* nonce;
    int len;
    unsigned char* plain;
    unsigned char* cipher;
    tv(unsigned char* k, int c, unsigned char* n, int l, unsigned char* p, unsigned char* cph)
        : key(k), counter(c), nonce(n), len(l), plain(p), cipher(cph) {}
};

bool test() {
    int err = 0;
    // for encryption
    hls::stream<ap_uint<256> > key1Strm;
    hls::stream<bool> eKey1Strm;
    hls::stream<ap_uint<128> > counter1Strm;
    hls::stream<ap_uint<512> > plainTextStrm;
    hls::stream<bool> ePlainTextStrm;
    hls::stream<ap_uint<512> > cipherStream;
    hls::stream<bool> eCipherStream;
    // for decryption
    hls::stream<ap_uint<256> > key2Strm;
    hls::stream<bool> eKey2Strm;
    hls::stream<ap_uint<128> > counter2Strm;
    hls::stream<ap_uint<512> > cipher2Strm;
    hls::stream<bool> eCipher2Strm;
    hls::stream<ap_uint<512> > deText;
    hls::stream<bool> eDetext;

    tv tsuit1(key1, counter1, nonce1, sizeof(plaintext1) / sizeof(unsigned char), plaintext1, ciphertext1);
    tv tsuit2(key2, counter2, nonce2, sizeof(plaintext2) / sizeof(unsigned char), plaintext2, ciphertext2);
    tv tsuit3(key3, counter3, nonce3, sizeof(plaintext3) / sizeof(unsigned char), plaintext3, ciphertext3);
    // choose one as test vector
    tv tsuit = tsuit2;
    // key to stream
    ap_uint<256> k = 0;
    for (int i = 0; i < 32; ++i) {
        k.range(i * 8 + 7, i * 8) = tsuit.key[i];
    }
    key1Strm.write(k);
    key2Strm.write(k);
    // combine counter and nonce
    ap_uint<128> counterNonce = 0;
    counterNonce.range(31, 0) = tsuit.counter;
    int count[4] = {0};
    for (int i = 4; i < 16; ++i) {
        counterNonce.range(i * 8 + 7, i * 8) = tsuit.nonce[i - 4];
    }
    counter1Strm.write(counterNonce);
    counter2Strm.write(counterNonce);
    // plain text to stream
    ap_uint<512> cbk = 0;
    int ii = -1;
    for (int i = 0; i < tsuit.len; i++) {
        ii = i % 64;
        if (ii == 0) cbk = 0;
        cbk.range((ii + 1) * 8 - 1, ii * 8) = tsuit.plain[i];
        if (ii == 63) {
#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
            std::cout << "text:" << cbk << std::endl;
#endif
            plainTextStrm.write(cbk);
            ePlainTextStrm.write(false);
        }
    }
    if (ii != -1 && ii != 63) {
#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
        std::cout << "text:" << cbk << std::endl;
#endif
        plainTextStrm.write(cbk);
        ePlainTextStrm.write(false);
    }
    ePlainTextStrm.write(true);
    // encrpt plain text  bychacha20
    chacha20Core(key1Strm, counter1Strm, plainTextStrm, ePlainTextStrm, cipherStream, eCipherStream);

    int cnt = 0;
#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
    std::cout << "cipher text:" << std::endl;
#endif
    while (!eCipherStream.read()) {
        ap_uint<512> cc = cipherStream.read();
        for (int i = 0; i < 64 && cnt < tsuit.len; i++) {
            ap_uint<8> c = cc.range(i * 8 + 7, i * 8);
#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
            std::cout << std::hex << c;
#endif
            // check cipher text
            if (c != ((short)(tsuit.cipher[cnt++]) & 0xff)) {
                err = 1;
            }
        }
        cipher2Strm.write(cc);
        eCipher2Strm.write(false);
    }
    eCipher2Strm.write(true);
#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
    std::cout << std::endl;
    std::cout << std::dec << "cipher len:" << cnt << std::endl;
#endif
    // decrypt cipher text
    xf::security::chacha20(key2Strm, counter2Strm, cipher2Strm, eCipher2Strm, deText, eDetext);

#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
    std::cout << std::hex << "recover text:" << std::endl;
#endif
    int ct = 0;
    while (!eDetext.read()) {
        ap_uint<512> msg = deText.read();
#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
        std::cout << std::hex << msg;
#endif
        for (int i = 0; i < 64 && ct < tsuit.len; ++i) {
            ap_uint<8> c = msg.range(i * 8 + 7, i * 8);
            // check decrypted text
            if (c != ((short)(tsuit.plain[ct++]) & 0xff)) {
                err = 1;
            }
        }
    }
    if (ct != tsuit.len) err = 1;

#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
    std::cout << std::endl;
// std::cout<<std::endl<<"plain:"<<std::endl;
// print(tsuit.plain,tsuit.len);
#endif
    return err;
}

int main(int argc, char** argv) {
    return test();
}
