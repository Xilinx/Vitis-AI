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
    // unsigned char arr[8] = block;
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

    ap_uint<8> input[8] = {0x54, 0x68, 0x65, 0x20, 0x71, 0x75, 0x66, 0x63};
    ap_uint<64>* a = (ap_uint<64>*)input;

    std::cout << "Plaintext\n";
    xf::security::internal::print<64>(*a);

    ap_uint<8> keyArr[8] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF};
    ap_uint<64>* keyAP = (ap_uint<64>*)keyArr;

    ap_uint<8> keyArr2[8] = {0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01};
    ap_uint<64>* keyAP2 = (ap_uint<64>*)keyArr2;
    ap_uint<8> keyArr3[8] = {0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x23};
    ap_uint<64>* keyAP3 = (ap_uint<64>*)keyArr3;
    ap_uint<8> keyArr5[8] = {};

    ap_uint<64> cipherDat;
    // xf::security::des3Encrypt(*a, *keyAP, *keyAP2, *keyAP3, cipherDat);
    testEnc(*a, *keyAP, *keyAP2, *keyAP3, cipherDat);
    std::cout << "Cipher text for 3DES\n";
    xf::security::internal::print<64>(cipherDat);

    ap_uint<8> golden[8] = {0xA8, 0x26, 0xFD, 0x8C, 0xE5, 0x3B, 0x85, 0x5F};
    ap_uint<64>* goldenCipher = (ap_uint<64>*)golden;
    std::cout << "Golden\n";
    xf::security::internal::print<64>(*goldenCipher);

    ap_uint<64> blockDat;
    // xf::security::des3Decrypt(cipherDat, *keyAP, *keyAP2, *keyAP3, blockDat);
    testDec(cipherDat, *keyAP, *keyAP2, *keyAP3, blockDat);
    std::cout << "Decrypted text for 3DES\n";
    xf::security::internal::print<64>(blockDat);

    std::cout << "\nTest 3DES ";
    if (cipherDat == *goldenCipher && blockDat == *a) {
        std::cout << "PASS\n" << std::endl;
        return 0;
    } else {
        std::cout << "FAIL\n" << std::endl;
        return 1;
    }
}
