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
 * @file merge_left_join.hpp
 * @brief The implementation of merge left join template
 *
 * Limitation:
 * 1. left table should not contain duplicated keys.
 *
 *
 */

#ifndef XF_DATABASE_MERGE_LEFT_JOIN_H
#define XF_DATABASE_MERGE_LEFT_JOIN_H

#ifndef __cplusplus
#error "Vitis Database Library only works with C++."
#endif

#include <hls_stream.h>

namespace xf {
namespace database {

/**
 *
 * @brief merge left join function for sorted table, left table should not have
 * duplicated keys.
 * @tparam KEY_T the type of the key
 * @tparam LEFT_FIELD_T the type of the field of left table
 * @tparam RIGHT_FIELD_T the type of the field of right table
 *
 * @param isascend flag to show if the input tables are ascend tables
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
 * @param isnull_strm the isnull stream to show if the result right table is
 * null.
 *
 */

template <typename KEY_T, typename LEFT_FIELD_T, typename RIGHT_FIELD_T>
inline void mergeLeftJoin(bool isascend,

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

                          hls::stream<bool>& out_e_strm,
                          hls::stream<bool>& isnull_strm) {
    bool left_e;
    KEY_T left_key;
    LEFT_FIELD_T left_field;

    bool right_e;
    KEY_T right_key;
    RIGHT_FIELD_T right_field;

    bool has_left_key = false;
    // has_left_key is the flag to record if the left row is pushed into the
    // result stream
    // when push the left row into the result stream, set the has_left_key to true
    // when pull the left stream, set the has_left_key to false

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

BOTH_TABLES_HAVE_VALUES_LOOP:
    while (!left_e && !right_e) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 0 max = 50000000
        // if the right key equals the left key, add them to the result table
        // Ascend and descend dose not matter.
        if (left_key == right_key) {
            left_strm_out_key.write(left_key);
            left_strm_out_field.write(left_field);
            right_strm_out_key.write(right_key);
            right_strm_out_field.write(right_field);

            isnull_strm.write(false); // right table has the key in left table, so isnull is false
            out_e_strm.write(false);
            has_left_key = true; // we add the left row in the result so set the
                                 // has_left_key flag to true

            // Pull only the right stream, if the next right key equals to the current
            // right key,
            // we still have chance to compare the current left key with the next
            // right key
            // in the next trip of the loop.
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
                // The situation not to insert left row to the result table with right
                // row is null:
                // Left key equals to a previous right key, this is marked by the
                // has_left_key flag.
                if (!has_left_key) {
                    left_strm_out_key.write(left_key);
                    left_strm_out_field.write(left_field);
                    right_strm_out_key.write(right_key);
                    right_strm_out_field.write(right_field);

                    isnull_strm.write(true); // cannot find left key in the right table,
                                             // so isnull is true
                    out_e_strm.write(false);
                }

                left_key = left_strm_in_key.read();
                left_field = left_strm_in_field.read();
                has_left_key = false; // pull the left stream and reset the has_left_key flag.
            }
        } else if ((isascend && left_key > right_key) || (!isascend && left_key < right_key)) {
            // no need to push the right table to the result if we cannot find right
            // key in the left table
            right_e = right_e_strm.read();
            if (!right_e) {
                right_key = right_strm_in_key.read();
                right_field = right_strm_in_field.read();
            }
        }
    }

    // if the left table is empty, pull all the rest of the right table, do not
    // add to the result
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

    // if the right table is empty, pull all the rest of left table and add to the
    // result
    else if (right_e) {
        // For this situation:
        // The left key equals to the last right key
        if (!has_left_key) {
            left_strm_out_key.write(left_key);
            left_strm_out_field.write(left_field);
            right_strm_out_key.write(right_key);
            right_strm_out_field.write(right_field);

            isnull_strm.write(true);
            out_e_strm.write(false);
        }

    RIGHT_TABLE_IS_EMPTY_LOOP:
        while (!left_e) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 0 max = 50000000
            left_e = left_e_strm.read();

            if (!left_e) {
                left_key = left_strm_in_key.read();
                left_field = left_strm_in_field.read();

                left_strm_out_key.write(left_key);
                left_strm_out_field.write(left_field);
                right_strm_out_key.write(right_key);
                right_strm_out_field.write(right_field);

                isnull_strm.write(true);
                out_e_strm.write(false);
            }
        }
    }

    out_e_strm.write(true);
} // merge_left_join

} // namespace database
} // namespace xf

#endif // XF_DATABASE_MERGE_LEFT_JOIN_H
