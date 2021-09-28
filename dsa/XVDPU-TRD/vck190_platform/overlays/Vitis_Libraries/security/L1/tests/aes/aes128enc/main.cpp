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
#include "test.hpp"
#include "xf_security/aes.hpp"
#include "xf_security/des.hpp"

int main() {
    const int W = 8;

    // ap_uint<8> keyArr[16] = {
    //    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c
    //};
    ap_uint<8> keyArr[16] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                             0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};

    ap_uint<128>* key = (ap_uint<128>*)keyArr;

    // std::cout << "Key ";

    // ap_uint<8> inputArr[16] = {
    //    0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34
    //};
    //
    ap_uint<8> inputArr[16] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                               0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
    ap_uint<128>* input = (ap_uint<128>*)inputArr;

    // std::cout << "Plain text ";

    ap_uint<128> cipher;
    testEnc(*input, *key, cipher);
    // std::cout << "Cipher text ";

    ap_uint<8> goldenCipherArr[16] = {
        // 0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb, 0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b, 0x32
        0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a};

    ap_uint<128>* golden = (ap_uint<128>*)goldenCipherArr;

    if (cipher == *golden) {
        std::cout << "\nAES128 test PASS\n" << std::endl;
        return 0;
    } else {
        std::cout << "\nAES128 test FAIL\n" << std::endl;
        return 1;
    }
}
