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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <vector> // std::vector

#include "xf_database/dynamic_eval.hpp"

typedef int32_t stream_t1;
typedef int32_t stream_t2;
typedef int32_t stream_t3;
typedef int32_t stream_t4;

typedef int32_t constant_t1;
typedef int32_t constant_t2;
typedef int32_t constant_t3;
typedef int32_t constant_t4;

typedef int32_t stream_out_t;

#define TestNumber 100
#define Test_Operator 0x033388CA8

/// TPC-H test operators.
enum ALU_OPERATOR {
    /// strm1*strm2
    TPC_H_op0 = 0x039900409,
    /// strm1*constant1
    TPC_H_op1 = 0x074900909,
    /// strm1*(constant2-strm2)
    TPC_H_op2 = 0x035200409,
    /// strm1*(constant2-strm2)*(constant3+strm3)
    TPC_H_op3 = 0x009299441,
    /// strm1*(cosntant2-strm2)-strm3*strm4
    TPC_H_op4 = 0x019200494,
    /// strm1==constant1 ? strm3:constant4
    TPC_H_op5 = 0x0199982A8,
    ///(strm1==constant1 || strm2==constant2) ? constant3:constant4
    TPC_H_op6 = 0x032288DA8,
    ///(strm1!=constant1 && strm2!=constant2) ? constant3:cosntant4
    TPC_H_op7 = 0x033388CA8
};

struct constant {
    constant_t1 c1;
    constant_t2 c2;
    constant_t3 c3;
    constant_t4 c4;
};

void generate_test_data(uint64_t Number,
                        std::vector<stream_t1>& testVector1,
                        std::vector<stream_t2>& testVector2,
                        std::vector<stream_t3>& testVector3,
                        std::vector<stream_t4>& testVector4,
                        constant& constant_inst) {
    srand(1);

    for (int i = 0; i < Number; i++) {
        testVector1.push_back(rand());
        testVector2.push_back(rand());
        testVector3.push_back(rand());
        testVector4.push_back(rand());
    }

    constant_inst.c1 = rand();
    constant_inst.c2 = rand();
    constant_inst.c3 = rand();
    constant_inst.c4 = rand();

    std::cout << " random test data generated! " << std::endl;
}

template <typename Strm_Type1,
          typename Strm_Type2,
          typename Strm_Type3,
          typename Strm_Type4,

          typename Constant_Type1,
          typename Constant_Type2,
          typename Constant_Type3,
          typename Constant_Type4,

          typename Strm_Out_Type>
void reference_dynamic_ALU(hls::stream<Strm_Type1>& strm_in1,
                           hls::stream<Strm_Type2>& strm_in2,
                           hls::stream<Strm_Type3>& strm_in3,
                           hls::stream<Strm_Type4>& strm_in4,

                           Constant_Type1 c1,
                           Constant_Type2 c2,
                           Constant_Type3 c3,
                           Constant_Type4 c4,

                           ap_uint<32> OP,

                           hls::stream<Strm_Out_Type>& strm_out) {
    Strm_Type1 in1;
    Strm_Type2 in2;
    Strm_Type3 in3;
    Strm_Type4 in4;

    Strm_Out_Type result;

    for (int i = 0; i < TestNumber; i++) {
        in1 = strm_in1.read();
        in2 = strm_in2.read();
        in3 = strm_in3.read();
        in4 = strm_in4.read();

        switch (OP) {
            case 0x039900409:
                result = in1 * in2;
                break;
            case 0x074000909:
                result = in1 * c1;
                break;
            case 0x035200409:
                result = in1 * (c2 - in2);
                break;
            case 0x009299441:
                result = in1 * (c2 - in2) * (in3 + c3);
                break;
            case 0x019200494:
                result = in1 * (c2 - in2) - in3 * in4;
                break;
            case 0x0199982A8:
                result = in1 == c1 ? in3 : c4;
                break;
            case 0x032288DA8:
                result = (in1 == c1 || in2 == c2) ? c3 : c4;
                break;
            case 0x033388CA8:
                result = (in1 != c1 && in2 != c2) ? c3 : c4;
                break;
        }

        strm_out.write(result);
    }
}

void hls_db_dynamic_ALU_function(ap_uint<289> OP,
                                 hls::stream<stream_t1>& strm_in1,
                                 hls::stream<stream_t2>& strm_in2,
                                 hls::stream<stream_t3>& strm_in3,
                                 hls::stream<stream_t4>& strm_in4,
                                 hls::stream<bool>& strm_in_end,
                                 hls::stream<stream_out_t>& strm_out,
                                 hls::stream<bool>& strm_out_end) {
    xf::database::dynamicEval<stream_t1, stream_t2, stream_t3, stream_t4,

                              constant_t1, constant_t2, constant_t3, constant_t4,

                              stream_out_t>(OP, strm_in1, strm_in2, strm_in3, strm_in4, strm_in_end, strm_out,
                                            strm_out_end);
}

int main() {
    std::vector<stream_t1> testVector1;
    std::vector<stream_t2> testVector2;
    std::vector<stream_t3> testVector3;
    std::vector<stream_t4> testVector4;

    hls::stream<stream_t1> strm_in1("strm_in1");
    hls::stream<stream_t2> strm_in2("strm_in2");
    hls::stream<stream_t3> strm_in3("strm_in3");
    hls::stream<stream_t4> strm_in4("strm_in4");
    hls::stream<bool> strm_in_end("strm_in_end");

    constant constant_inst;
    constant_t1 c1;
    constant_t2 c2;
    constant_t3 c3;
    constant_t4 c4;

    ap_uint<289> Operation;
    ap_uint<33> Operator = Test_Operator;
    ap_uint<256> Operand;

    hls::stream<stream_out_t> strm_out("strm_out");
    hls::stream<bool> strm_out_end("strm_out_end");

    hls::stream<stream_t1> ref_strm_in1("ref_strm_in1");
    hls::stream<stream_t2> ref_strm_in2("ref_strm_in2");
    hls::stream<stream_t3> ref_strm_in3("ref_strm_in3");
    hls::stream<stream_t4> ref_strm_in4("ref_strm_in4");
    hls::stream<stream_out_t> ref_strm_out("ref_strm_out");

    int nerror = 0;

    // generate test data
    generate_test_data(TestNumber, testVector1, testVector2, testVector3, testVector4, constant_inst);

    // prepare input data
    std::cout << "testVector List:" << std::endl;

    c1 = constant_inst.c1;
    c2 = constant_inst.c2;
    c3 = constant_inst.c3;
    c4 = constant_inst.c4;

    for (std::string::size_type i = 0; i < TestNumber; i++) {
        // print vector value
        std::cout << "Index=" << i << ' ';
        std::cout << "Stream1=" << testVector1[i] << ' ';
        std::cout << "c1=" << c1 << ' ';
        std::cout << "Stream2=" << testVector2[i] << ' ';
        std::cout << "c2=" << c2 << ' ';
        std::cout << "Stream3=" << testVector3[i] << ' ';
        std::cout << "c3=" << c3 << ' ';
        std::cout << "Stream4=" << testVector4[i] << ' ';
        std::cout << "c4=" << c4 << std::endl;

        // write stream
        strm_in1.write(testVector1[i]); // write test data to strm_in1
        strm_in2.write(testVector2[i]); // write test data to strm_in2
        strm_in3.write(testVector3[i]); // write test data to strm_in3
        strm_in4.write(testVector4[i]); // write test data to strm_in4
        strm_in_end.write(false);       // write data to end_flag_strm

        ref_strm_in1.write(testVector1[i]); // write test data to ref_strm_in1
        ref_strm_in2.write(testVector2[i]); // write test data to ref_strm_in2
        ref_strm_in3.write(testVector3[i]); // write test data to ref_strm_in3
        ref_strm_in4.write(testVector4[i]); // write test data to ref_strm_in4
    }
    strm_in_end.write(true);

    Operand(255, 192) = c1;
    Operand(191, 128) = c2;
    Operand(127, 64) = c3;
    Operand(63, 0) = c4;

    Operation(288, 256) = Operator;
    Operation(255, 0) = Operand;

    std::cout << "Operator=" << std::hex << Operator << ' ';
    std::cout << "Operand=" << std::hex << Operand << std::endl;
    std::cout << "Operation=" << std::hex << Operation << std::endl;

    // call dynamic_ALU function
    hls_db_dynamic_ALU_function(Operation, strm_in1, strm_in2, strm_in3, strm_in4, strm_in_end, strm_out, strm_out_end);

    // run reference dynamic_ALU
    reference_dynamic_ALU<stream_t1, stream_t2, stream_t3, stream_t4, constant_t1, constant_t2, constant_t3,
                          constant_t4, stream_out_t>(ref_strm_in1, ref_strm_in2, ref_strm_in3, ref_strm_in4, c1, c2, c3,
                                                     c4, Operator, ref_strm_out);

    //===== check if the output flag e_out_strm is correct or not =====
    for (int i = 0; i < (TestNumber); i++) {
        bool e = strm_out_end.read();
        if (e) {
            std::cout << "\n the output flag is incorrect" << std::endl;
            nerror++;
        }
    }
    // read out the last flag that e should =1
    bool e = strm_out_end.read();
    if (!e) {
        std::cout << "\n the last output flag is incorrect" << std::endl;
        nerror++;
    }

    stream_out_t out, ref_out;
    for (int i = 0; i < (TestNumber); i++) {
        // compare the key
        out = strm_out.read();
        ref_out = ref_strm_out.read();
        bool cmp_key = (out == ref_out) ? 1 : 0;
        if (!cmp_key) {
            nerror++;
            std::cout << "Index=" << i << ' ' << "stream_out=" << out << ' ' << "ref_strm_out=" << ref_out;
            std::cout << "\n the result is incorrect" << std::endl;
        }
    }

    // print result
    if (nerror) {
        std::cout << "\nFAIL: nerror= " << nerror << " errors found.\n";
    } else {
        std::cout << "\nPASS: no error found.\n";
    }
    return 0;
}
