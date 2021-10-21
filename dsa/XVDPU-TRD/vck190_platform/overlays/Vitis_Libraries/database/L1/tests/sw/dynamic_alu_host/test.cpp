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

#include "xf_database/dynamic_alu_host.hpp"
#include <iostream>

#define test_number 20

typedef int32_t constant_t1;
typedef int32_t constant_t2;
typedef int32_t constant_t3;
typedef int32_t constant_t4;

int main(int argc, const char* argv[]) {
    char const* formula[test_number];

    // TPC-H
    formula[0] = "strm1*strm2";
    formula[1] = "strm1*c1";
    formula[2] = "strm1*(-strm2+c2)";
    formula[3] = "strm1*(-strm2+c2)-strm3*strm4";
    formula[4] = "strm1*(-strm2+c2)*(strm3+c3)";
    formula[5] = "strm1==strm2 ? strm3:c4";
    formula[6] = "((strm1==c1) || (strm2==c2)) ? c3:c4";
    formula[7] = "((strm1!=c1) && (strm2!=c2)) ? c3:c4";

    // one strm
    formula[8] = "strm1+c1";
    formula[9] = "strm1>c1";
    formula[10] = "strm1&&c1";

    // two strm
    formula[11] = "strm1+strm2";
    formula[12] = "strm1>strm2";
    formula[13] = "strm1==strm2";
    formula[14] = "(strm1+c1)*(strm2+c2)";

    // three strm
    formula[15] = "strm1+strm2+strm3";
    formula[16] = "strm1*strm2*strm3";
    formula[17] = "(strm1+c1)*(strm2+c2)*(strm3+c3)";

    // four strm
    formula[18] = "(strm1+strm2)+(strm3+strm4)";
    formula[19] = "((strm1+c1)*(strm2+c2))*((strm3+c3)*(strm4+c4))";

    constant_t1 c1 = rand();
    constant_t2 c2 = rand();
    constant_t3 c3 = rand();
    constant_t4 c4 = rand();

    ap_uint<289> OP;

    bool test_pass = 1;

    for (int i = 0; i < test_number; i++) {
        std::cout << "\n"
                  << "Formula:" << formula[i] << std::endl;

        test_pass &= xf::database::dynamicALUOPCompiler<constant_t1, constant_t2, constant_t3, constant_t4>(
            (char*)formula[i], c1, c2, c3, c4, OP);

        if (test_pass)
            std::cout << "OP: " << std::hex << OP << std::endl;
        else {
            std::cout << "OP: Generate failed" << std::endl;
            break;
        }
    }

    if (test_pass)
        std::cout << "\n"
                  << "TEST PASS!" << std::endl;
    else
        std::cout << "\n"
                  << "TEST FAILED!" << std::endl;

    return 0;
}
