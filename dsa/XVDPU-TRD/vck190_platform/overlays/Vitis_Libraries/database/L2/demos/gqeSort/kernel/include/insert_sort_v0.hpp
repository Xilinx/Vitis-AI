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
#include <ap_int.h>
#include <hls_stream.h>
#include "merge_sort_v0.hpp"

namespace xf {
namespace database {
namespace details {

template <typename Data_Type, typename Key_Type, int max_sort_number>
void insert_sort_top(hls::stream<Pair<Data_Type, Key_Type> >& packInStrm,
                     hls::stream<Pair<Data_Type, Key_Type> >& packOutStrm,
                     bool sign) {
    bool end;
    Key_Type in_temp, out_temp;
    Data_Type in_dtemp, out_dtemp;
    bool array_full = 0;

    Key_Type array_temp[max_sort_number];
#pragma HLS ARRAY_PARTITION variable = array_temp complete
    Data_Type array_dtemp[max_sort_number];
#pragma HLS ARRAY_PARTITION variable = array_dtemp complete

    bool comparative_sign[max_sort_number];
#pragma HLS ARRAY_PARTITION variable = comparative_sign complete
    comparative_sign[0] = 0;

    ap_uint<16> inserting_id = 1;
    ap_uint<16> residual_count = max_sort_number + 1;
    Pair<Data_Type, Key_Type> packIn = packInStrm.read();
    end = packIn.end();
    Pair<Data_Type, Key_Type> packOut;

insert_loop:
    while (!end || residual_count) {
#pragma HLS PIPELINE

        // read input strm
        if (!end) {
            in_temp = packIn.key();
            in_dtemp = packIn.data();
            packIn = packInStrm.read();
            end = packIn.end();
        } else {
        }

    // initialize sign
    initial_sign_loop:
        for (int i = 1; i < max_sort_number; i++) {
#pragma HLS UNROLL
            if (i < inserting_id) {
                if (array_temp[i - 1] < in_temp) {
                    comparative_sign[i] = sign;
                } else {
                    comparative_sign[i] = !sign;
                }
            } else {
                comparative_sign[i] = 1;
            }
        }

        // manage the last element
        out_temp = array_temp[max_sort_number - 1];
        out_dtemp = array_dtemp[max_sort_number - 1];
        if (comparative_sign[max_sort_number - 1] == 0) {
            array_temp[max_sort_number - 1] = in_temp;
            array_dtemp[max_sort_number - 1] = in_dtemp;
        }

    // right shift && insert for intermediate elements
    right_shift_insert_loop:
        for (int i = max_sort_number - 2; i >= 0; i--) {
#pragma HLS UNROLL
            if (comparative_sign[i] == 0 && comparative_sign[i + 1] == 0) {
            } else if (comparative_sign[i] == 0 && comparative_sign[i + 1] == 1) {
                array_temp[i + 1] = array_temp[i];
                array_dtemp[i + 1] = array_dtemp[i];
                array_temp[i] = in_temp;
                array_dtemp[i] = in_dtemp;
            } else if (comparative_sign[i] == 1 && comparative_sign[i + 1] == 1) {
                array_temp[i + 1] = array_temp[i];
                array_dtemp[i + 1] = array_dtemp[i];
            }
        }

        // write output strm
        if (array_full) {
            packOut.key(out_temp);
            packOut.data(out_dtemp);
            packOut.end(0);
            packOutStrm.write(packOut);
        }

        // update loop parameters
        if (end) {
            inserting_id = 1;
            residual_count--;
            array_full = 1;
        } else {
            if (inserting_id == max_sort_number) {
                inserting_id = 1;
                array_full = 1;
            } else {
                inserting_id++;
            }
        }
    }
    packOut.key(0);
    packOut.data(0);
    packOut.end(1);
    packOutStrm.write(packOut);
}

/**
 * @brief Insert sort top function.
 *
 * @tparam DATA_TYPE the input and output data type
 * @tparam KEY_TYPE the input and output key type
 * @tparam MAX_SORT_NUMBER the max number of the sequence can be sorted
 *
 * @param packInStrm input packed data / key / end stream
 * @param packOutStrm output sorted packed data / key / end stream
 * @param order 1:sort ascending 0:sort descending
 */
template <typename DATA_TYPE, typename KEY_TYPE, int MAX_SORT_NUMBER>
void insertSort(hls::stream<Pair<DATA_TYPE, KEY_TYPE> >& packInStrm,
                hls::stream<Pair<DATA_TYPE, KEY_TYPE> >& packOutStrm,
                bool order) {
#pragma HLS PIPELINE
#pragma HLS INLINE

    insert_sort_top<DATA_TYPE, KEY_TYPE, MAX_SORT_NUMBER>(packInStrm, packOutStrm, order);
}

} // namespace details
} // namespace database
} // namespace xf
