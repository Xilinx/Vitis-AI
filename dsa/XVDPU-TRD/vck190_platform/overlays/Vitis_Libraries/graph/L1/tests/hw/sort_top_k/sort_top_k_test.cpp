
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
#include <stdlib.h>
#include <algorithm>
#include <stdint.h>

#include "similarity/sort_top_k.hpp"

typedef uint32_t KEY_TYPE;
typedef uint32_t DATA_TYPE;

/// @brief basic sort operators
enum SortOrder { SORT_ASCENDING = 1, SORT_DESCENDING = 0 };

#define MaxSortNumber 128
#define TestNumber 1024
#define OP SORT_ASCENDING
#define TopK 100

template <typename Key_Type>
void generate_test_data(uint64_t Number, std::vector<KEY_TYPE>& testVector) {
    srand(1);
    for (int i = 0; i < Number; i++) {
        testVector.push_back(rand()); // generate random key value
    }
    std::cout << " random test data generated! " << std::endl;
}

template <typename Data_Type, typename Key_Type>
void reference_sort(hls::stream<Key_Type>& ref_in_strm,
                    hls::stream<Key_Type>& ref_out_strm,
                    uint64_t Test_Number,
                    uint64_t Max_Sort_Number,
                    bool sign) {
    Key_Type key_strm[TestNumber];

    // determine sort number for std::sort
    for (int i = 0; i < TestNumber; i++) {
        key_strm[i] = ref_in_strm.read();
    }

    std::sort(&key_strm[0], &key_strm[TestNumber]);

    if (!sign) {
        for (int i = 0; i < TopK; i++) {
            ref_out_strm.write(key_strm[i]);
        }
    } else {
        for (int i = TestNumber - 1; i >= TestNumber - TopK; i--) {
            ref_out_strm.write(key_strm[i]);
        }
    }
}

void hls_sort_top_k_function(hls::stream<DATA_TYPE>& din_strm,
                             hls::stream<KEY_TYPE>& kin_strm,
                             hls::stream<bool>& strm_in_end,
                             hls::stream<DATA_TYPE>& dout_strm,
                             hls::stream<KEY_TYPE>& kout_strm,
                             hls::stream<bool>& strm_out_end,
                             int k,
                             bool sign) {
    xf::graph::sortTopK<KEY_TYPE, DATA_TYPE, MaxSortNumber>(din_strm, kin_strm, strm_in_end, dout_strm, kout_strm,
                                                            strm_out_end, k, sign);
}

int main() {
    int i;

    /// sort_strm_test

    std::vector<KEY_TYPE> testVector;
    hls::stream<bool> din_strm_end("din_strm_end");
    hls::stream<bool> dout_strm_end("dout_strm_end");
    hls::stream<DATA_TYPE> din_strm("din_strm");
    hls::stream<DATA_TYPE> dout_strm("dout_strm");
    hls::stream<KEY_TYPE> kin_strm("kin_strm");
    hls::stream<KEY_TYPE> kout_strm("kout_strm");

    hls::stream<KEY_TYPE> ref_kin_strm("ref_kin_strm");
    hls::stream<KEY_TYPE> ref_kout_strm("ref_kout_strm");
    hls::stream<DATA_TYPE> ref_din_strm("ref_din_strm");
    hls::stream<DATA_TYPE> ref_dout_strm("ref_dout_strm");

    int nerror = 0;

    // generate test data
    generate_test_data<KEY_TYPE>(TestNumber, testVector);

    // prepare input data
    std::cout << "testVector List:" << std::endl;
    for (std::string::size_type i = 0; i < TestNumber; i++) {
        // print vector value
        std::cout << "Index=" << i << ' ';
        std::cout << "Vector=" << testVector[i] << std::endl;

        kin_strm.write(testVector[i]);     // write data to kin_strm
        din_strm.write(testVector[i]);     // write data to din_strm
        ref_kin_strm.write(testVector[i]); // write data to ref_kin_strm
        din_strm_end.write(false);         // write data to end_flag_strm
    }
    din_strm_end.write(true);

    // call insert_sort function

    hls_sort_top_k_function(din_strm, kin_strm, din_strm_end, dout_strm, kout_strm, dout_strm_end, TopK, OP);

    // run reference sort
    reference_sort<DATA_TYPE, KEY_TYPE>(ref_kin_strm, ref_kout_strm, TestNumber, MaxSortNumber, OP);

    //===== check if the output flag e_out_strm is correct or not =====
    for (int i = 0; i < TopK; i++) {
        bool e = dout_strm_end.read();
        if (e) {
            std::cout << "\nthe output flag is incorrect" << std::endl;
            nerror++;
        }
    }
    // read out the last flag that e should =1
    bool e = dout_strm_end.read();
    if (!e) {
        std::cout << "\nthe last output flag is incorrect" << std::endl;
        nerror++;
    }

    KEY_TYPE key, ref_key;
    DATA_TYPE data, ref_data;
    for (int i = 0; i < TopK; i++) {
        // compare the key
        key = kout_strm.read();
        data = dout_strm.read();
        ref_key = ref_kout_strm.read();
        ref_data = ref_key;
        // std::cout << "Index=" << i <<' '<< "key=" << key <<' '<< "reference key=" << ref_key<< std::endl;
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
