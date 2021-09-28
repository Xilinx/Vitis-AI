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

    ap_uint<8> keyArr[24] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,
                             0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17};
    ap_uint<192>* key = (ap_uint<192>*)keyArr;

    ap_uint<8> inputArr[16] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                               0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
    ap_uint<128>* input = (ap_uint<128>*)inputArr;

    ap_uint<128> cipher;
    testEnc(*input, *key, cipher);

    ap_uint<8> goldenCipherArr[16] = {0xdd, 0xa9, 0x7c, 0xa4, 0x86, 0x4c, 0xdf, 0xe0,
                                      0x6e, 0xaf, 0x70, 0xa0, 0xec, 0x0d, 0x71, 0x91};
    ap_uint<128>* golden = (ap_uint<128>*)goldenCipherArr;

    if (cipher == *golden) {
        std::cout << "\nAES192 test PASS\n" << std::endl;
        return 0;
    } else {
        std::cout << "\nAES192 test FAIL\n" << std::endl;
        return 1;
    }
}
