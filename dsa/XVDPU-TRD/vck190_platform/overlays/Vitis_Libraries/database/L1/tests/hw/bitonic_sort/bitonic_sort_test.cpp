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

#include <vector>
#include <iostream>
#include <stdint.h>
#include <algorithm>

#include "xf_database/bitonic_sort.hpp"

typedef uint32_t KEY_TYPE;

#define SortNumber 128
#define BitonicSortNumber 32
#define OP xf::database::SORT_ASCENDING

template <typename Key_Type>
void generate_test_data(uint64_t Number, std::vector<KEY_TYPE>& testVector) {
    srand(1);
    for (int i = 0; i < Number; i++) {
        testVector.push_back(rand()); // generate random key value
    }
    std::cout << " random test data generated! " << std::endl;
}

template <typename Key_Type, uint64_t Sort_Number>
void reference_sort(hls::stream<Key_Type>& ref_in_strm, hls::stream<Key_Type>& ref_out_strm) {
    Key_Type key_strm[BitonicSortNumber];

    for (int j = 0; j < Sort_Number / BitonicSortNumber; j++) {
        for (int i = 0; i < BitonicSortNumber; i++) {
            key_strm[i] = ref_in_strm.read();
        }

        std::sort(&key_strm[0], &key_strm[BitonicSortNumber]);

        for (int i = 0; i < BitonicSortNumber; i++) {
            ref_out_strm.write(key_strm[i]);
        }
    }
}

void hls_db_bitonic_sort_function(hls::stream<KEY_TYPE>& kin_strm,
                                  hls::stream<bool>& kin_strm_end,
                                  hls::stream<KEY_TYPE>& kout_strm,
                                  hls::stream<bool>& kout_strm_end,
                                  bool sign) {
    xf::database::bitonicSort<KEY_TYPE, BitonicSortNumber>(kin_strm, kin_strm_end, kout_strm, kout_strm_end, sign);
}

int main() {
    int i;

    /// sort_strm_test

    std::vector<KEY_TYPE> testVector;
    hls::stream<bool> din_strm_end("din_strm_end");
    hls::stream<bool> dout_strm_end("dout_strm_end");
    hls::stream<KEY_TYPE> kin_strm("kin_strm");
    hls::stream<KEY_TYPE> kout_strm("kout_strm");

    hls::stream<KEY_TYPE> ref_in_strm("ref_kin_strm");
    hls::stream<KEY_TYPE> ref_out_strm("ref_kout_strm");

    int nerror = 0;

    // generate test data
    generate_test_data<KEY_TYPE>(SortNumber, testVector);

    // prepare input data
    std::cout << "testVector List:" << std::endl;
    for (std::string::size_type i = 0; i < SortNumber; i++) {
        // print vector value
        std::cout << "Index=" << i << ' ';
        std::cout << "Vector=" << testVector[i] << std::endl;

        kin_strm.write(testVector[i]);    // write data to kin_strm
        ref_in_strm.write(testVector[i]); // write data to ref_in_strm
        din_strm_end.write(false);        // write data to end_flag_strm
    }
    din_strm_end.write(true);

    // call aggregate function
    hls_db_bitonic_sort_function(kin_strm, din_strm_end, kout_strm, dout_strm_end, OP);

    // run reference sort
    reference_sort<KEY_TYPE, SortNumber>(ref_in_strm, ref_out_strm);

    //===== check if the output flag e_out_strm is correct or not =====
    for (int i = 0; i < SortNumber; i++) {
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
    for (int i = 0; i < SortNumber; i++) {
        // compare the key
        key = kout_strm.read();
        ref_key = ref_out_strm.read();

        bool cmp_key = (key == ref_key) ? 1 : 0;
        if (!cmp_key) {
            nerror++;
            std::cout << "Index=" << i << ' ' << "key=" << key << ' ' << "reference key=" << ref_key;
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
