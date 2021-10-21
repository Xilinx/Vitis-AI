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

#include <iostream>
#include "test.hpp"
#include <string>

int main() {
    std::cout << std::hex;

    ap_uint<2048> modulus = ap_uint<2048>(
        "0x9d41cd0d38339220ebd110e8c31feb279c5fae3c23090045a0886301588d4c8114fa5cdde708ea77ba0f527e6f6ea8f5634acf517f04"
        "ca6399e188d5c2d7f03cc90e04dbf7d5d0056ee1b14b8baaf90ef78f5142ddce9ba2eff84c0295f656c29aecaae80ddd5c7127ddc60215"
        "9458f272316100f726a71362516223f26ddeafa425d3eb2c7f61de7e8586e77d475037563425d931885f03693618bb885ab9b58de74f60"
        "4a86f28e494dcd819bd8c0bb42f699596969b84f680819e4c9fc0ba687558775f770a302d5b266905defe47bc53c98ce261523b49db624"
        "1567f4b48c661482ef9c453750c6d420a0b1a3bd4d3d05b060c026ce8efd9bb9456dfe2f5d");

    ap_uint<2048> message = 0;
    // unsigned char resm[256];
    unsigned char rawm[256] =
        "RSA TEST FILE : "
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuv"
        "wxyz1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh"
        "ijklmnopqrstuvwxyz.";
    for (int i = 0; i < 256; i++) {
        message.range(i * 8 + 7, i * 8) = (unsigned int)rawm[255 - i];
    }
    message.range(7, 0) = ap_uint<8>("0x0a");

    ap_uint<20> exponent = ap_uint<2048>("0x10001");

    ap_uint<2048> golden = ap_uint<2048>(
        "0x7E85E87850F80E298487446A567585F7FF39C7DD1E1BAB4303B0000CF581494182D9FB50B27946DD555921727DEA816F41D5750B57BF"
        "BD130CD4D93BCF81B070AE55B4A8B06D44668E01B1B2FF2B123AE2FB2DB7BE7F4C8158695C4567E9A741F3F8BD7658345185FC78B90F12"
        "3FF17311534630391A78340D0E8A9C8BC501E28E2EA0AAE5B0941C6D3480389784CBFE3A55B3D05943B657BAD7616423F30808E0312A72"
        "F89056F66BFE1FFC5F54119A88C6553B41075F0869D228EC6CC3C09738406AAEA6530659C6573B44FDA1CFE20EC9FDEBA7B4B52456B9BB"
        "D3762929248C11B6006CDAEA05800B902B42FDF8FD5B2F7D3B5F23F2A28E50B05B26E4A778");

    // get test result
    ap_uint<2048> result;

    rsa_test(message, modulus, exponent, result);

    if (result != golden) {
        std::cout << "Not Match !!!" << std::endl;
        return 1;
    }

    std::cout << std::endl;
    return 0;
}
