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

/**
 * @file nested_loop_join.hpp
 * @brief nested_loop_join function
 *
 * Limitation:
 * 1. One of the input table should be small, otherwise this function will need
 * numerous resource
 *
 */

#ifndef XF_DATABASE_NESTED_LOOP_JOIN_H
#define XF_DATABASE_NESTED_LOOP_JOIN_H

#ifndef __cplusplus
#error "Vitis Database Library only works with C++."
#endif

#include <hls_stream.h>

namespace xf {
namespace database {
/**
 * @brief Nested loop join function
 *
 * @tparam KEY_T the type of the key of left table
 * @tparam LEFT_FIELD_T the type of the field of left table
 * @tparam RIGHT_FIELD_T the type of the field of right table
 *
 * @param strm_in_left_key the key stream of the left input table
 * @param strm_in_left_field the field stream of the left input table
 * @param strm_in_left_e the end flag stream to mark the end of left input table
 *
 * @param strm_in_right_key  the key stream of the right input table
 * @param strm_in_right_field the field stream of the right input table
 * @param strm_in_right_e the end flag stream to mark the end of right input
 * table
 *
 * @param strm_out_left_key the output key stream of left table
 * @param strm_out_left_field the output field stream of left table
 * @param strm_out_right_key the output key stream of right table
 * @param strm_out_right_field the output field stream of right
 * @param strm_out_e the end flag stream to mark the end of out table
 */

template <int CMP_NUM, typename KEY_T, typename LEFT_FIELD_T, typename RIGHT_FIELD_T>
inline void nestedLoopJoin(hls::stream<KEY_T>& strm_in_left_key,
                           hls::stream<LEFT_FIELD_T>& strm_in_left_field,
                           hls::stream<bool>& strm_in_left_e,

                           hls::stream<KEY_T>& strm_in_right_key,
                           hls::stream<RIGHT_FIELD_T>& strm_in_right_field,
                           hls::stream<bool>& strm_in_right_e,

                           hls::stream<KEY_T> strm_out_left_key[CMP_NUM],
                           hls::stream<LEFT_FIELD_T> strm_out_left_field[CMP_NUM],

                           hls::stream<KEY_T> strm_out_right_key[CMP_NUM],
                           hls::stream<RIGHT_FIELD_T> strm_out_right_field[CMP_NUM],

                           hls::stream<bool> strm_out_e[CMP_NUM]) {
    int i, j;
    bool left_e, right_e;

    KEY_T right_key;
    RIGHT_FIELD_T right_field;

    KEY_T key_array[CMP_NUM];
#pragma HLS ARRAY_PARTITION variable = key_array complete dim = 1

    LEFT_FIELD_T field_array[CMP_NUM];
#pragma HLS ARRAY_PARTITION variable = field_array complete dim = 1

    bool valid[CMP_NUM];
#pragma HLS ARRAY_PARTITION variable = valid complete dim = 1

    // check if the left table is an empty table
    left_e = strm_in_left_e.read();

    // check if the right table is an empty table
    right_e = strm_in_right_e.read();

    if (!right_e && !left_e) {
    INIT_VALID:
        for (i = 0; i < CMP_NUM; i++) { // initialize the valid
#pragma HLS unroll
            valid[i] = false;
        }

        i = 0;
    READ_LEFT:
        while (i < CMP_NUM && !left_e) { // shift register to read in left table
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1 max = 1000
        READ_LEFT_SHIFT_REG_0:
            for (j = CMP_NUM - 1; j > 0; j--) {
#pragma HLS unroll
                key_array[j] = key_array[j - 1];
                field_array[j] = field_array[j - 1];
                valid[j] = valid[j - 1];
            }
            key_array[0] = strm_in_left_key.read();
            field_array[0] = strm_in_left_field.read();
            valid[0] = true;

            //				key_array[i] = strm_in_left_key.read();
            //				field_array[i] =
            // strm_in_left_field.read();
            //				valid[i] = true;

            left_e = strm_in_left_e.read();
            i++;
        }

    NORMAL_NESTED_LOOP:
        while (!right_e) { // When right table is empty, end the loop
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 40000 max = 40000

            right_e = strm_in_right_e.read();
            right_key = strm_in_right_key.read();
            right_field = strm_in_right_field.read();

            for (i = 0; i < CMP_NUM; i++) {
#pragma HLS unroll
                if (valid[i] && right_key == key_array[i]) {
                    strm_out_left_key[i].write(key_array[i]);
                    strm_out_left_field[i].write(field_array[i]);
                    strm_out_right_key[i].write(right_key);
                    strm_out_right_field[i].write(right_field);
                    strm_out_e[i].write(false);
                }
            }
        }
    } else { // if one of the table is empty, read out the other one.
        if (left_e) {
        LEFT_TABLE_IS_EMPTY_LOOP:
            while (!right_e) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 0 max = 40000
                right_e = strm_in_right_e.read();
                right_key = strm_in_right_key.read();
                right_field = strm_in_right_field.read();
            }

        } else if (right_e) {
        RIGHT_TABLE_IS_EMPTY_LOOP:
            while (!left_e) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 0 max = 40000
                left_e = strm_in_left_e.read();
                strm_in_left_key.read();
                strm_in_left_field.read();
            }
        }
    }

    for (i = 0; i < CMP_NUM; i++) {
#pragma HLS unroll
        strm_out_e[i].write(true);
    }

} // nested_loop_join

} // namespace database
} // namespace xf
#endif // XF_DATABASE_NESTED_LOOP_JOIN_H
