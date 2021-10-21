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

#include "xf_security/sm234.hpp"
#ifndef __SYNTHESIS__
#include <iostream>
#endif

void test(ap_uint<128> key, ap_uint<128> ciphertext, ap_uint<128>& plaintext) {
    xf::security::sm4 processor;
    processor.keyExtension(key);
    processor.decrypt(ciphertext, plaintext);
}

#ifndef __SYNTHESIS__
int main() {
    ap_uint<128> key = ap_uint<128>("0x76543210fedcba9889abcdef01234567");
    ap_uint<128> golden = ap_uint<128>("0x76543210fedcba9889abcdef01234567");
    ap_uint<128> res;
    ap_uint<128> ciphertext = ap_uint<128>("0x536E424686B3E94FD206965E681EDF34");

    test(key, ciphertext, res);

    if (res != golden) {
        std::cout << std::hex << res << std::endl;
        std::cout << std::hex << golden << std::endl;
        return 1;
    } else {
        return 0;
    }
}
#endif
