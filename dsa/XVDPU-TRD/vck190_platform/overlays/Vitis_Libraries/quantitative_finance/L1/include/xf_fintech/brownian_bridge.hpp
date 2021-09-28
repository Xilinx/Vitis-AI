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

#ifndef _XF_BROWNIAN_BRIDGE_H_
#define _XF_BROWNIAN_BRIDGE_H_
//#include <cmath>

#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"
#include "xf_fintech/rng.hpp"

namespace xf {

namespace fintech {

/**
 * @brief Brownian bridge transformation using inverse simulation.
 *
 * @tparam DT data type supported include float and double.
 * @tparam SZ maximum length of input sequence, maximum is 1024.
 */

template <typename DT, int SZ>
class BrownianBridge {
   private:
    // data width W  = log(SZ), current maximum SZ is 1024.
    static const int W = 10;

    // loop body of transform
    inline void trans_body(ap_uint<W> idx, double inputVal, double* result, double* result_dup) {
#pragma HLS inline
        ap_uint<W> j = left_index[idx];
        ap_uint<W> k = right_index[idx];
        ap_uint<W> l = bridge_index[idx];
        // TODO: check if j-1 and k are previous l
        DT lf_w = 0;
        // DT rg_w = 0;
        DT std_dev = 0;
        DT pre_1 = 0;
        DT pre_2 = 0;
        if (j != 0) {
            pre_1 = result[j - 1];
            pre_2 = result_dup[k];
        } else {
            pre_1 = 0;
            pre_2 = result[k];
        }
        std_dev = stdDev[idx];
        lf_w = left_weight[idx];
        // rg_w = 1 - lf_w;//right_weight[i];
        DT r_1 = internal::FPTwoSub(pre_1, pre_2);
        DT r_2 = std_dev * inputVal;
        DT r_3 = lf_w * r_1;
        DT r_4 = internal::FPTwoAdd(r_2, pre_2);
        DT res = internal::FPTwoAdd(r_4, r_3);
        // DT res = lf_w * (pre_1 - pre_2) + pre_2 + std_dev * inputVal;
        result[l] = res;
        result_dup[l] = res;

#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
        std::cout << "i =" << idx << ", out_id =" << l << ", pre_id_1 =" << j - 1 << ", pre_id_2 =" << k << std::endl;
#endif
#endif
    }

   public:
    /**
     * @brief  Default constructor
     *
     */
    BrownianBridge() {
#pragma HLS inline
    }

    /**
     * @brief  Transform in_strm to out_strm using brownian bridge transformation
     *
     * @param in_strm stream containing input sequence
     * @param out_strm stream containing output sequence which applys to brownian bridge disribution
     *
     */
    void transform(hls::stream<DT>& in_strm, hls::stream<DT>& out_strm) {
        // Result
        DT result[SZ];

        // duplicate the result buffer to solve memory port limitation.
        DT result_dup[SZ];

        DT inputVal = in_strm.read();
        DT last_d = stdDev[0] * inputVal;
        result[size - 1] = last_d; // stdDev[0] * inputVal;
        result_dup[size - 1] = last_d;

        ap_uint<W> loop_nm[8];
#pragma HLS array_partition variable = loop_nm dim = 0

        // cut the input into several parts. result in each part have no dependence.
        // the length of each part is 1, 2, 4, 8, 16, 32, size-64
        for (int i = 0; i < 7; ++i) {
#pragma HLS unroll
            ap_uint<W> batch = 1 << i;
            loop_nm[i] = MIN(size, batch);
        }
        loop_nm[7] = size;
        for (int lp = 1; lp < 8; ++lp) {
#pragma HLS loop_tripcount min = 8 max = 8
            ap_uint<W> start_id = loop_nm[lp - 1];
            ap_uint<W> end_id = loop_nm[lp];
            for (ap_uint<W> i = start_id; i < end_id; ++i) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 8 max = 8
#pragma HLS dependence variable = result inter false
#pragma HLS dependence variable = result_dup inter false
                DT inputVal = in_strm.read();
                trans_body(i, inputVal, result, result_dup);
                // std::cout<<"i ="<<i<<", out_id ="<<l<<", pre_1 ="<<pre_1<<", pre_2
                // ="<<pre_2<<std::endl;
            }
        }

        // re-order the output sequence.
        for (int i = 0; i < size; ++i) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 128 max = 128
            DT o;
            if (i == 0)
                o = result[i];
            else
                o = internal::FPTwoSub(result[i], result[i - 1]);
            out_strm.write(o);
        }
    }

    /**
     * @brief Initialize weights, indices, standard deviations used in
     * transformation
     *
     * @param size_in the lenght of sequence to be transformed.
     */

    void initialize(ap_uint<W> size_in) {
#ifndef __SYNTHESIS__
        assert(SZ <= (1 << W));
        assert(size_in <= SZ);
#endif
        size = size_in;
        ap_uint<SZ> map = 0;

        bridge_index[0] = size - 1;
        map[size - 1] = true;

        // find the position of left, right and middle
        for (int j = 0, i = 1; i < size; ++i) {
#pragma HLS loop_tripcount min = 128 max = 128
#ifndef __SYNTHESIS__
            int cnt = 0;
#endif
            while (map[j]) {
#pragma HLS loop_tripcount min = 10 max = 10
#pragma HLS pipeline II = 1
                ++j;
#ifndef __SYNTHESIS__
                cnt++;
#endif
            }

            int k = j;

            while (!map[k]) {
#pragma HLS loop_tripcount min = 10 max = 10
#pragma HLS pipeline II = 1
                ++k;
#ifndef __SYNTHESIS__
                cnt++;
// std::cout<<"loop_latency ="<<cnt<<std::endl;
#endif
            }

            int l = j + ((k - 1 - j) >> 1);

            // map[l] = i;
            map[l] = true;

            bridge_index[i] = l;
            left_index[i] = j;
            right_index[i] = k;

            j = k + 1;

            if (j >= size) {
                j = 0; //  wrap around
            }
        }

        left_weight[0] = 0.0;

        stdDev[0] = hls::sqrt(size);

        // calculate the weight for left, right,
        for (int i = 1; i < size; ++i) {
#pragma HLS loop_tripcount min = 128 max = 128
#pragma HLS pipeline II = 1

            ap_uint<W> l = bridge_index[i]; // bd_strm.read();
            ap_uint<W> j = left_index[i];   // lf_strm.read();
            ap_uint<W> k = right_index[i];  // rg_strm.read();

#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
            std::cout << "l = " << l << ", j=" << j << ", k=" << k << std::endl;
#endif
#endif
            DT div_1 = k - l;
            int lf, rg;
            if (j != 0) {
                // left_weight[i] = (times[k]-times[l])/(times[k]-times[j-1]);
                lf = k - j + 1;
                // left_weight[i] = (DT)(k - l)/(DT)(k - j  + 1);
                // stdDev[i] = hls::sqrt(((DT)(l - j + 1)*(k - l))/(DT)(k - j +1));
                rg = l - j + 1;
            } else {
                lf = k + 1;
                // left_weight[i]  = (times[k]-times[l])/times[k];
                // left_weight[i]  = (DT)(k - l)/(DT)(k+1);
                // stdDev[i] = hls::sqrt((DT)((l + 1)* (k - l))/(DT)(k+1));
                // stdDev[i] = std::sqrt((times[l] * (times[k] - times[l]))/times[k]);
                rg = l + 1;
            }
            DT div_2 = lf;
            DT rg_dis = rg;
            DT div = div_1 / div_2;
            left_weight[i] = div;
            stdDev[i] = hls::sqrt(rg_dis * div);
        }
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
        for (int i = 0; i < size; ++i) {
            std::cout << "i = " << i << ", left_weight=" << left_weight[i] << ", stdDev=" << stdDev[i] << std::endl;
            std::cout << "i = " << i << ", bridgeIndex_=" << bridge_index[i] << ", leftIndex_=" << left_index[i]
                      << ", rightIndex_=" << right_index[i] << std::endl;
        }
#endif
#endif
    }

   private:
    // int size;
    ap_uint<W> size;
    DT left_weight[SZ];
    DT stdDev[SZ];
    ap_uint<W> left_index[SZ];
    ap_uint<W> right_index[SZ];
    ap_uint<W> bridge_index[SZ];
};

} // end of namespace fintech

} // end of namespace xf

#endif
