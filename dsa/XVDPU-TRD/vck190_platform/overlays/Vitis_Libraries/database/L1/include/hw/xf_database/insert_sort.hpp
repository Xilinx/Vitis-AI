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
#ifndef _XF_DATABASE_INSERT_SORT_HPP_
#define _XF_DATABASE_INSERT_SORT_HPP_

#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

namespace xf {
namespace database {
namespace details {

template <typename Data_Type, typename Key_Type, int max_sort_number>
void insert_sort_top(hls::stream<Data_Type>& din_strm,
                     hls::stream<Key_Type>& kin_strm,
                     hls::stream<bool>& strm_in_end,
                     hls::stream<Data_Type>& dout_strm,
                     hls::stream<Key_Type>& kout_strm,
                     hls::stream<bool>& strm_out_end,
                     bool sign) {
#pragma HLS INLINE

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

    uint16_t inserting_id = 1;
    uint16_t residual_count = max_sort_number + 1;
    end = strm_in_end.read();

insert_loop:
    while (!end || residual_count) {
#pragma HLS PIPELINE ii = 1

        // read input strm
        if (!end) {
            in_temp = kin_strm.read();
            in_dtemp = din_strm.read();
            end = strm_in_end.read();
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
        } else {
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
            } else {
            }
        }

        // write output strm
        if (array_full) {
            kout_strm.write(out_temp);
            dout_strm.write(out_dtemp);
            strm_out_end.write(0);
        } else {
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
    strm_out_end.write(1);
}

template <typename Key_Type, int max_sort_number>
void insert_sort_top(hls::stream<Key_Type>& kin_strm,
                     hls::stream<bool>& strm_in_end,
                     hls::stream<Key_Type>& kout_strm,
                     hls::stream<bool>& strm_out_end,
                     bool sign) {
#pragma HLS INLINE
    bool end;
    Key_Type in_temp, out_temp;
    bool array_full = 0;

    Key_Type array_temp[max_sort_number];
#pragma HLS ARRAY_PARTITION variable = array_temp complete

    bool comparative_sign[max_sort_number];
#pragma HLS ARRAY_PARTITION variable = comparative_sign complete
    comparative_sign[0] = 0;

    uint16_t inserting_id = 1;
    uint16_t residual_count = 1;
    // uint16_t residual_count = max_sort_number + 1;
    uint16_t begin = 0;
    end = strm_in_end.read();

insert_loop:
    while (!end || residual_count) {
#pragma HLS PIPELINE ii = 1
#pragma HLS loop_tripcount max = 20 min = 20
        // read input strm
        if (!end) {
            in_temp = kin_strm.read();
            end = strm_in_end.read();
            if (begin < max_sort_number) {
                residual_count++;
                begin++;
            }
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
        out_temp = array_temp[begin - 1];
        if (comparative_sign[max_sort_number - 1] == 0) {
            array_temp[max_sort_number - 1] = in_temp;
        } else {
        }

    // right shift && insert for intermediate elements
    right_shift_insert_loop:
        for (int i = max_sort_number - 2; i >= 0; i--) {
#pragma HLS UNROLL
            if (comparative_sign[i] == 0 && comparative_sign[i + 1] == 0) {
            } else if (comparative_sign[i] == 0 && comparative_sign[i + 1] == 1) {
                array_temp[i + 1] = array_temp[i];
                array_temp[i] = in_temp;
            } else if (comparative_sign[i] == 1 && comparative_sign[i + 1] == 1) {
                array_temp[i + 1] = array_temp[i];
            } else {
            }
        }

        // write output strm
        if (array_full) {
            kout_strm.write(out_temp);
            strm_out_end.write(0);
        } else {
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
    strm_out_end.write(1);
}

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {

/**
 * @brief Insert sort top function.
 *
 * @tparam KEY_TYPE the input and output key type
 * @tparam MAX_SORT_NUMBER the max number of the sequence can be sorted
 *
 * @param kinStrm input key stream
 * @param endInStrm end flag stream for input
 * @param koutStrm output key stream
 * @param endOutStrm end flag stream for output
 * @param order 1:sort ascending 0:sort descending
 */
template <typename KEY_TYPE, int MAX_SORT_NUMBER>
void insertSort(hls::stream<KEY_TYPE>& kinStrm,
                hls::stream<bool>& endInStrm,
                hls::stream<KEY_TYPE>& koutStrm,
                hls::stream<bool>& endOutStrm,
                bool order) {
    details::insert_sort_top<KEY_TYPE, MAX_SORT_NUMBER>(kinStrm, endInStrm, koutStrm, endOutStrm, order);
}

/**
 * @brief Insert sort top function.
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
 * @param order 1:sort ascending 0:sort descending
 */
template <typename KEY_TYPE, typename DATA_TYPE, int MAX_SORT_NUMBER>
void insertSort(hls::stream<DATA_TYPE>& dinStrm,
                hls::stream<KEY_TYPE>& kinStrm,
                hls::stream<bool>& endInStrm,
                hls::stream<DATA_TYPE>& doutStrm,
                hls::stream<KEY_TYPE>& koutStrm,
                hls::stream<bool>& endOutStrm,
                bool order) {
    details::insert_sort_top<DATA_TYPE, KEY_TYPE, MAX_SORT_NUMBER>(dinStrm, kinStrm, endInStrm, doutStrm, koutStrm,
                                                                   endOutStrm, order);
}

} // namespace database
} // namespace xf

#endif
