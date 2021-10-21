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
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <openssl/des.h>
#include <openssl/rand.h>
#include "test.hpp"
#include "xf_security/des.hpp"

void cBlock2AP(DES_cblock block, ap_uint<64>& ap) {
    for (int i = 0; i < 8; ++i) {
        ap(i * 8 + 7, i * 8) = block[i];
    }
}

void ap2CBlock(ap_uint<64> ap, DES_cblock& block) {
    for (int i = 0; i < 8; ++i) {
        block[i] = (char)ap(i * 8 + 7, i * 8);
    }
}

void printCBlock(DES_cblock block) {
    std::cout << "[";
    for (int i = 0; i < 8; ++i) {
        std::cout << block[i];
    }
    std::cout << "]" << std::endl;
}

int main() {
    const int W = 8;

    ap_uint<8> input[8] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF};
    ap_uint<64>* a = (ap_uint<64>*)input;

    std::cout << "Plaintext\n";
    xf::security::internal::print<64>(*a);

    DES_cblock data;
    ap2CBlock(*a, data);

    ap_uint<64> dataAP;
    cBlock2AP(data, dataAP);

    unsigned char in[W], out[W], back[W];
    unsigned char* e = out;

    memset(in, 0, sizeof(in));
    memset(out, 0, sizeof(out));
    memset(back, 0, sizeof(back));

    ap_uint<8> keyArr[8] = {0x13, 0x34, 0x57, 0x79, 0x9B, 0xBC, 0xDF, 0xF1};
    ap_uint<64>* keyAP = (ap_uint<64>*)keyArr;
    std::cout << "Key\n";
    xf::security::internal::print<64>(*keyAP);

    DES_cblock key;
    DES_key_schedule keysched;

    // DES_random_key(&key);
    ap2CBlock(*keyAP, key);
    ap_uint<64> keyCp;
    cBlock2AP(key, keyCp);
    DES_set_key((C_Block*)key, &keysched);

    DES_ecb_encrypt((C_Block*)data, (C_Block*)out, &keysched, DES_ENCRYPT);

    ap_uint<64> cipherAP;
    cBlock2AP(out, cipherAP);
    std::cout << "Ciphertext\n";
    xf::security::internal::print<64>(cipherAP);

    DES_ecb_encrypt((C_Block*)out, (C_Block*)back, &keysched, DES_DECRYPT);

    ap_uint<64> cipher;
    testEnc(*a, *keyAP, cipher);
    std::cout << "Implementation ciphertext\n";
    xf::security::internal::print<64>(cipher);

    ap_uint<64> blockText;
    testDec(cipher, *keyAP, blockText);
    std::cout << "Block text\n";
    xf::security::internal::print<64>(blockText);

    std::cout << "\nTest DES ";
    if (cipherAP == cipher && blockText == *a) {
        std::cout << "PASS\n" << std::endl;
        return 0;
    } else {
        std::cout << "FAIL\n" << std::endl;
        return 1;
    }
}
