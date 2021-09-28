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
 * @file merge_join.hpp
 * @brief merge join function for sorted table without duplicated key
 *
 * Limitation:
 * 1. left table should not contain duplicated keys.
 *
 */

#ifndef XF_DATABASE_MERGE_JOIN_H
#define XF_DATABASE_MERGE_JOIN_H

#ifndef __cplusplus
#error "Vitis Database Library only works with C++."
#endif

#include <hls_stream.h>

namespace xf {
namespace database {

/**
 * @brief merge join function for sorted tables without duplicated key in the
 * left table
 *
 * @tparam KEY_T the type of the key of left table
 * @tparam LEFT_FIELD_T the type of the field of left table
 * @tparam RIGHT_FIELD_T the type of the field of right table
 *
 * @param isascend the flag to show if the input tables are ascend or descend
 * tables
 *
 * @param left_strm_in_key the key stream of the left input table
 * @param left_strm_in_field the field stream of the left input table
 * @param left_e_strm the end flag stream to mark the end of left input table
 *
 * @param right_strm_in_key  the key stream of the right input table
 * @param right_strm_in_field the field stream of the right input table
 * @param right_e_strm the end flag stream to mark the end of right input table
 *
 * @param left_strm_out_key the output key stream of left table
 * @param left_strm_out_field the output field stream of left table
 * @param right_strm_out_key the output key stream of right table
 * @param right_strm_out_field the output field stream of right
 * @param out_e_strm the end flag stream to mark the end of out table
 *
 */

template <typename KEY_T, typename LEFT_FIELD_T, typename RIGHT_FIELD_T>
inline void mergeJoin(bool isascend,
                      hls::stream<KEY_T>& left_strm_in_key,
                      hls::stream<LEFT_FIELD_T>& left_strm_in_field,
                      hls::stream<bool>& left_e_strm,

                      hls::stream<KEY_T>& right_strm_in_key,
                      hls::stream<RIGHT_FIELD_T>& right_strm_in_field,
                      hls::stream<bool>& right_e_strm,

                      hls::stream<KEY_T>& left_strm_out_key,
                      hls::stream<LEFT_FIELD_T>& left_strm_out_field,

                      hls::stream<KEY_T>& right_strm_out_key,
                      hls::stream<RIGHT_FIELD_T>& right_strm_out_field,

                      hls::stream<bool>& out_e_strm) {
    bool left_e;
    KEY_T left_key;
    LEFT_FIELD_T left_field;

    bool right_e;
    KEY_T right_key;
    RIGHT_FIELD_T right_field;

    // check if the left table is an empty table
    left_e = left_e_strm.read();
    if (!left_e) {
        left_key = left_strm_in_key.read();
        left_field = left_strm_in_field.read();
    }

    // check if the right table is an empty table
    right_e = right_e_strm.read();
    if (!right_e) {
        right_key = right_strm_in_key.read();
        right_field = right_strm_in_field.read();
    }

BOTH_TABLE_HAVE_VALUES_LOOP:
    // Pull the right and left streams until one of them is empty.
    while (!left_e && !right_e) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 0 max = 50000000
        // if the left key and the right key are the same, add them to the result
        // table.
        // Ascend and descend input tables do not matter.
        if (left_key == right_key) {
            left_strm_out_key.write(left_key);
            left_strm_out_field.write(left_field);
            right_strm_out_key.write(right_key);
            right_strm_out_field.write(right_field);

            out_e_strm.write(false);

            // Pull the right stream and hold the left stream.
            // If the next right key equals the current right key, as we only pull the
            // right stream,
            // we still have the chance to compare the next right key with the current
            // left key in
            // the next trip of the loop.
            right_e = right_e_strm.read();
            if (!right_e) {
                right_key = right_strm_in_key.read();
                right_field = right_strm_in_field.read();
            }

            // Which stream to pull:
            // For ascend tables, pull the stream with a smaller key
            // For descend tables, pull the stream with a larger key
        } else if ((isascend && left_key < right_key) || (!isascend && left_key > right_key)) {
            left_e = left_e_strm.read();
            if (!left_e) {
                left_key = left_strm_in_key.read();
                left_field = left_strm_in_field.read();
            }
        } else if ((isascend && left_key > right_key) || (!isascend && left_key < right_key)) {
            right_e = right_e_strm.read();
            if (!right_e) {
                right_key = right_strm_in_key.read();
                right_field = right_strm_in_field.read();
            }
        }
    }

    // if left table is empty, pull all the rest of right table.
    if (left_e) {
    LEFT_TABLE_IS_EMPTY_LOOP:
        while (!right_e) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 0 max = 50000000
            right_e = right_e_strm.read();
            if (!right_e) {
                right_key = right_strm_in_key.read();
                right_field = right_strm_in_field.read();
            }
        }
    }

    // if the right table is empty, pull all the rest of left table
    else if (right_e) {
    RIGHT_TABLE_IS_EMPTY_LOOP:
        while (!left_e) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 0 max = 50000000
            left_e = left_e_strm.read();
            if (!left_e) {
                left_key = left_strm_in_key.read();
                left_field = left_strm_in_field.read();
            }
        }
    }

    // Add end of table flag to the result table.
    out_e_strm.write(true);
} // merge_join

} // namespace database
} // namespace xf

#endif // XF_DATABASE_MERGE_JOIN_H
