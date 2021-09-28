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
                   const int num_of_term,
                   hls::stream<ap_uint<64> >& i_model_theta_strm,
                   hls::stream<ap_uint<64> >& i_model_prior_strm,

                   hls::stream<ap_uint<32> > i_sample_data_strm[CH_NM],
                   hls::stream<bool>& i_sample_e_strm,

                   hls::stream<ap_uint<10> >& o_class_strm,
                   hls::stream<bool>& o_e_strm) {
    // core function
    xf::data_analytics::classification::naiveBayesPredict<CH_NM, GRP_NM>(num_of_class, num_of_term, i_model_theta_strm,
                                                                         i_model_prior_strm, i_sample_data_strm,
                                                                         i_sample_e_strm, o_class_strm, o_e_strm);
}
