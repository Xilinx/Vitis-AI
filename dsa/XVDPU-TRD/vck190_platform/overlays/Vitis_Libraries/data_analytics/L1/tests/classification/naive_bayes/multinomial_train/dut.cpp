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

#include "dut.hpp"
#include "xf_data_analytics/classification/naive_bayes.hpp"

void multinomialNB(const int num_of_class,
                   const int num_of_terms,
                   hls::stream<ap_uint<64> > i_data_strm_array[PU],
                   hls::stream<bool> i_e_strm_array[PU],

                   hls::stream<int>& o_terms_strm,
                   hls::stream<ap_uint<64> > mnnb_d0_strm[PU],
                   hls::stream<ap_uint<64> > mnnb_d1_strm[PU]) {
    // core function
    xf::data_analytics::classification::naiveBayesTrain<32, WL>(
        num_of_class, num_of_terms, i_data_strm_array, i_e_strm_array, o_terms_strm, mnnb_d0_strm, mnnb_d1_strm);
}
