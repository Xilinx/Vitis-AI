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

#include <vector> // std::vector
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <algorithm>

#include "xf_database/merge_sort.hpp"

typedef uint32_t DATA_TYPE;
typedef uint32_t KEY_TYPE;

enum SortSign { SORT_ASCENDING = 1, SORT_DESCENDING = 0 };

#define TestNumber1 1024
#define TestNumber2 277
#define OP SORT_ASCENDING

template <typename Data_Type, typename Key_Type>
void generate_test_data(uint64_t Number1,
                        std::vector<KEY_TYPE>& testVector1,
                        uint64_t Number2,
                        std::vector<KEY_TYPE>& testVector2,
                        bool sign) {
    std::vector<KEY_TYPE> RandomVector1;
    std::vector<KEY_TYPE> RandomVector2;

    srand(1);
    for (int i = 0; i < Number1; i++) {
        RandomVector1.push_back(rand()); // generate random key value
    }

    for (int i = 0; i < Number2; i++) {
        RandomVector2.push_back(rand()); // generate random key value
    }

    std::sort(&RandomVector1[0], &RandomVector1[Number1]);
    std::sort(&RandomVector2[0], &RandomVector2[Number2]);

    if (sign) {
        for (int i = 0; i < Number1; i++) {
            testVector1.push_back(RandomVector1[i]); // generate ascending key value1
        }

        for (int i = 0; i < Number2; i++) {
            testVector2.push_back(RandomVector2[i]); // generate descending key value1
        }
    } else {
        for (int i = Number1 - 1; i >= 0; i--) {
            testVector1.push_back(RandomVector1[i]); // generate ascending key value2
        }

        for (int i = Number2 - 1; i >= 0; i--) {
            testVector2.push_back(RandomVector2[i]); // generate descending key value2
        }
    }

    std::cout << " random test data generated! " << std::endl;
}

template <typename Data_Type, typename Key_Type>
void reference_sort(hls::stream<Key_Type>& ref_in_strm1,
                    uint64_t Sort_Number1,
                    hls::stream<Key_Type>& ref_in_strm2,
                    uint64_t Sort_Number2,
                    hls::stream<Key_Type>& ref_out_strm,
                    bool sign) {
    Key_Type key_strm[Sort_Number1 + Sort_Number2];

    for (int i = 0; i < Sort_Number1; i++) {
        key_strm[i] = ref_in_strm1.read();
    }
    for (int i = Sort_Number1; i < Sort_Number1 + Sort_Number2; i++) {
        key_strm[i] = ref_in_strm2.read();
    }

    std::sort(&key_strm[0], &key_strm[Sort_Number1 + Sort_Number2]);

    if (sign) {
        for (int i = 0; i < Sort_Number1 + Sort_Number2; i++) {
            ref_out_strm.write(key_strm[i]);
        }
    } else {
        for (int i = Sort_Number1 + Sort_Number2 - 1; i >= 0; i--) {
            ref_out_strm.write(key_strm[i]);
        }
    }
}

void hls_db_merge_sort_function(hls::stream<DATA_TYPE>& left_din_strm,
                                hls::stream<KEY_TYPE>& left_kin_strm,
                                hls::stream<bool>& left_strm_in_end,

                                hls::stream<DATA_TYPE>& right_din_strm,
                                hls::stream<KEY_TYPE>& right_kin_strm,
                                hls::stream<bool>& right_strm_in_end,

                                hls::stream<DATA_TYPE>& dout_strm,
                                hls::stream<KEY_TYPE>& kout_strm,
                                hls::stream<bool>& strm_out_end,

                                bool sign) {
    xf::database::mergeSort<DATA_TYPE, KEY_TYPE>(left_din_strm, left_kin_strm, left_strm_in_end,

                                                 right_din_strm, right_kin_strm, right_strm_in_end,

                                                 dout_strm, kout_strm, strm_out_end, sign);
}

int main() {
    int i;

    /// sort_strm_test

    std::vector<KEY_TYPE> testVector1;
    std::vector<KEY_TYPE> testVector2;

    hls::stream<DATA_TYPE> left_din_strm("left_din_strm");
    hls::stream<KEY_TYPE> left_kin_strm("left_kin_strm");
    hls::stream<bool> left_strm_in_end("left_strm_in_end");

    hls::stream<DATA_TYPE> right_din_strm("right_din_strm");
    hls::stream<KEY_TYPE> right_kin_strm("right_kin_strm");
    hls::stream<bool> right_strm_in_end("right_strm_in_end");

    hls::stream<DATA_TYPE> dout_strm("dout_strm");
    hls::stream<KEY_TYPE> kout_strm("kout_strm");
    hls::stream<bool> strm_out_end("strm_out_end");

    hls::stream<KEY_TYPE> ref_in_strm1("ref_kin_strm1");
    hls::stream<KEY_TYPE> ref_in_strm2("ref_kin_strm2");
    hls::stream<KEY_TYPE> ref_out_strm("ref_kout_strm");

    int nerror = 0;

    // generate test data
    generate_test_data<DATA_TYPE, KEY_TYPE>(TestNumber1, testVector1, TestNumber2, testVector2, OP);

    // prepare input data
    std::cout << "testVector List:" << std::endl;
    for (std::string::size_type i = 0; i < TestNumber1; i++) {
        // print vector value
        std::cout << "Index=" << i << ' ';
        std::cout << "Left Stream=" << testVector1[i] << std::endl;

        // left
        left_din_strm.write(testVector1[i]); // write test data to din_strm
        left_kin_strm.write(testVector1[i]); // write test data to kin_strm
        ref_in_strm1.write(testVector1[i]);  // write test data to ref_in_strm
        left_strm_in_end.write(false);       // write data to end_flag_strm
    }
    left_strm_in_end.write(true);

    std::cout << std::endl;

    for (std::string::size_type i = 0; i < TestNumber2; i++) {
        // print vector value
        std::cout << "Index=" << i << ' ';
        std::cout << "Right Stream=" << testVector2[i] << std::endl;

        // right
        right_din_strm.write(testVector2[i]); // write test data to din_strm
        right_kin_strm.write(testVector2[i]); // write test data to kin_strm
        ref_in_strm2.write(testVector2[i]);   // write test data to ref_in_strm
        right_strm_in_end.write(false);       // write data to end_flag_strm
    }
    right_strm_in_end.write(true);

    std::cout << std::endl;

    // call merge_sort function
    hls_db_merge_sort_function(left_din_strm, left_kin_strm, left_strm_in_end,

                               right_din_strm, right_kin_strm, right_strm_in_end,

                               dout_strm, kout_strm, strm_out_end,

                               OP);

    // run reference sort
    reference_sort<DATA_TYPE, KEY_TYPE>(ref_in_strm1, TestNumber1, ref_in_strm2, TestNumber2, ref_out_strm, OP);

    //===== check if the output flag e_out_strm is correct or not =====
    for (int i = 0; i < (TestNumber1 + TestNumber2); i++) {
        bool e = strm_out_end.read();
        if (e) {
            std::cout << "\nthe output flag is incorrect" << std::endl;
            nerror++;
        }
    }
    // read out the last flag that e should =1
    bool e = strm_out_end.read();
    if (!e) {
        std::cout << "\nthe last output flag is incorrect" << std::endl;
        nerror++;
    }

    KEY_TYPE key, ref_key;
    DATA_TYPE data, ref_data;
    for (int i = 0; i < (TestNumber1 + TestNumber2); i++) {
        // compare the key
        key = kout_strm.read();
        data = dout_strm.read();
        ref_key = ref_out_strm.read();
        ref_data = ref_key;
        bool cmp_key = (key == ref_key) ? 1 : 0;
        bool cmp_data = (data == ref_data) ? 1 : 0;
        if (!cmp_key || !cmp_data) {
            nerror++;
            std::cout << "Index=" << i << ' ' << "key=" << key << ' ' << "reference key=" << ref_key << ' '
                      << "data=" << data << ' ' << "reference data=" << ref_data;
            std::cout << "\nthe sort key is incorrect" << std::endl;
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
