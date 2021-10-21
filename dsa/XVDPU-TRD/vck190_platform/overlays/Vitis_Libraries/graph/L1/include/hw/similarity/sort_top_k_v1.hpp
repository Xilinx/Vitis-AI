/*
 * Copyright 2020 Xilinx, Inc.
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

#ifndef __XF_GRAPH_SORTTOPK_HPP_
#define __XF_GRAPH_SORTTOPK_HPP_

#include <stdint.h>
#include <ap_int.h>
#include <hls_stream.h>

#define DEBUG_SORT true

namespace xf {
namespace graph {
namespace internal {
namespace sort_top_k {

template <typename Data_Type, typename Key_Type, int max_sort_number>
void bubble_sort_top(hls::stream<Data_Type>& din_strm,
                     hls::stream<Key_Type>& kin_strm,
                     hls::stream<bool>& strm_in_end,
                     hls::stream<Data_Type>& dout_strm,
                     hls::stream<Key_Type>& kout_strm,
                     hls::stream<bool>& strm_out_end,
                     int k,
                     bool sign) {
    bool end;
    Key_Type in_temp, out_temp;
    Data_Type in_dtemp, out_dtemp;
    ap_uint<16> insert_id = 0;

    Key_Type array_temp[max_sort_number];
#pragma HLS ARRAY_PARTITION variable = array_temp complete
    Data_Type array_dtemp[max_sort_number];
#pragma HLS ARRAY_PARTITION variable = array_dtemp complete
    Key_Type bubble_temp[max_sort_number];
#pragma HLS ARRAY_PARTITION variable = bubble_temp complete
    Data_Type bubble_dtemp[max_sort_number];
#pragma HLS ARRAY_PARTITION variable = bubble_dtemp complete
    bool comparative_sign[max_sort_number];
#pragma HLS ARRAY_PARTITION variable = comparative_sign complete

initial:
    for (int i = 0; i < max_sort_number; i++) {
#pragma HLS UNROLL
        array_temp[i] = 0;
        array_dtemp[i] = 0;
        bubble_temp[i] = 0;
        bubble_dtemp[i] = 0;
    }

    uint16_t residual_count = k;
    end = strm_in_end.read();

sort_loop:
    while (!end || residual_count) {
#pragma HLS PIPELINE II = 1

        if (!end) {
            in_temp = kin_strm.read();
            in_dtemp = din_strm.read();
            end = strm_in_end.read();
        } else {
            in_temp = 0;
            in_dtemp = 0;
            residual_count--;
        }

#ifndef __SYNTHESIS__
#ifdef DEBUG_SORT
        std::cout << std::endl;
        std::cout << "end=" << end << " key=" << in_temp << " data=" << in_dtemp << " residual=" << residual_count
                  << std::endl;
#endif
#endif

        // initialize comparative sign
        for (int i = 0; i < max_sort_number - 1; i++) {
            if ((i < insert_id) && (array_temp[i + 1] < bubble_temp[i]))
                comparative_sign[i] = true;
            else
                comparative_sign[i] = false;
        }

    // insert for intermediate elements
    right_shift:
        for (int i = max_sort_number - 2; i >= 0; i--) {
#pragma HLS UNROLL
            if (sign && comparative_sign[i]) {
                bubble_temp[i + 1] = array_temp[i + 1];
                array_temp[i + 1] = bubble_temp[i];

                bubble_dtemp[i + 1] = array_dtemp[i + 1];
                array_dtemp[i + 1] = bubble_dtemp[i];
            } else {
                bubble_temp[i + 1] = bubble_temp[i];

                bubble_dtemp[i + 1] = bubble_dtemp[i];
            }
        }

        if (insert_id == 0) {
            array_temp[0] = in_temp;

            array_dtemp[0] = in_dtemp;
        } else if (sign && (array_temp[0] < in_temp)) {
            bubble_temp[0] = array_temp[0];
            array_temp[0] = in_temp;

            bubble_dtemp[0] = array_dtemp[0];
            array_dtemp[0] = in_dtemp;
        } else {
            bubble_temp[0] = in_temp;

            bubble_dtemp[0] = in_dtemp;
        }

#ifndef __SYNTHESIS__
#ifdef DEBUG_SORT
        std::cout << "key array: ";
        for (int i = 0; i < max_sort_number; i++) {
            std::cout << array_temp[i] << ", ";
        }
        std::cout << std::endl;

        std::cout << "bubble array: ";
        for (int i = 0; i < max_sort_number; i++) {
            std::cout << bubble_temp[i] << ", ";
        }
        std::cout << std::endl;

        std::cout << "data array: ";
        for (int i = 0; i < max_sort_number; i++) {
            std::cout << array_dtemp[i] << ", ";
        }
        std::cout << std::endl;

        std::cout << std::endl;
#endif
#endif

        if (insert_id < max_sort_number) insert_id++;
        ;
    }

    residual_count = k;
output_loop:
    while (residual_count) {
#pragma HLS PIPELINE II = 1

        out_temp = array_temp[0];
        out_dtemp = array_dtemp[0];
        for (int i = 0; i < max_sort_number - 1; i++) {
#pragma HLS UNROLL
            array_temp[i] = array_temp[i + 1];
            array_dtemp[i] = array_dtemp[i + 1];
        }
        dout_strm.write(out_dtemp);
        kout_strm.write(out_temp);
        strm_out_end.write(0);

        residual_count--;
    }
    strm_out_end.write(1);
}

} // namespace sort_top_k
} // namespace internal
} // namespace graph
} // namespace xf

namespace xf {
namespace graph {

/**
 * @brief sort top k function.
 *
 * @tparam KEY_TYPE the input and output key type
 * @tparam DATA_TYPE the input and output data type
 * @tparam MAX_SORT_NUMBER the max number of the sequence can be sorted
 *
 * @param dinStrm input data stream
 * @param kinStrm input key stream
 * @param endInStrm end flag stream for input
 * @param doutStrm output data stream
 * @param koutStrm output key stream
 * @param endOutStrm end flag stream for output
 * @param number of top K
 * @param order 1:sort ascending 0:sort descending
 */
template <typename KEY_TYPE, typename DATA_TYPE, int MAX_SORT_NUMBER>
void sortTopK(hls::stream<DATA_TYPE>& dinStrm,
              hls::stream<KEY_TYPE>& kinStrm,
              hls::stream<bool>& endInStrm,
              hls::stream<DATA_TYPE>& doutStrm,
              hls::stream<KEY_TYPE>& koutStrm,
              hls::stream<bool>& endOutStrm,
              int k,
              bool order) {
#pragma HLS INLINE

    internal::sort_top_k::bubble_sort_top<DATA_TYPE, KEY_TYPE, MAX_SORT_NUMBER>(dinStrm, kinStrm, endInStrm, doutStrm,
                                                                                koutStrm, endOutStrm, k, order);
}

} // namespace database
} // namespace xf

#endif
