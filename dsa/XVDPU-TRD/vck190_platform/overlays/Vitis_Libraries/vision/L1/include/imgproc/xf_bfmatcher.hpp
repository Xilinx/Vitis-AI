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

#ifndef __XF_VITIS_BFMATCHER_HPP__
#define __XF_VITIS_BFMATCHER_HPP__

#include "ap_int.h"
#include "common/xf_utility.hpp"

namespace xf {
namespace cv {

// Some macros related to template (for easiness of coding)
const int DESC_SIZE = 256;
const ap_int<16> POOR_MATCH = -1;

using DESC_TYPE = ap_uint<DESC_SIZE>;
using DIST_TYPE = ap_uint<9>;
using MATCH_TYPE = ap_int<16>;

#define _GENERIC_BF_TPLT template <int PU = 1, int MAX_KEYPOINTS = 10000>
#define _GENERIC_BF GenericBF<PU, MAX_KEYPOINTS>

// ======================================================================================
// Top bfMatcher API
// --------------------------------------------------------------------------------------
// Template Args:-
// ......................................................................................

_GENERIC_BF_TPLT class GenericBF {
   public:
    DESC_TYPE stack_t[MAX_KEYPOINTS];

    void copyTrainingSet(DESC_TYPE desc_train[MAX_KEYPOINTS], ap_uint<32> keypoints_t) {
    READ_LOOP_DESC_TRAIN:
        for (ap_uint<32> i = 0; i < keypoints_t; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=MAX_KEYPOINTS max=MAX_KEYPOINTS
#pragma HLS PIPELINE II=1
            // clang-format on
            stack_t[i] = desc_train[i];
        }
    }

    DIST_TYPE computeHamming(DESC_TYPE desc_val_q, DESC_TYPE desc_val_t) {
// clang-format off
#pragma HLS INLINE OFF
        // clang-format on

        DIST_TYPE dist = 0;
        for (int i = 0; i < DESC_SIZE; i++) {
// clang-format off
#pragma HLS UNROLL
            // clang-format on
            dist += desc_val_q.range(i, i) == desc_val_t.range(i, i) ? 0 : 1;
        }
        return dist;
    }

    void process(DESC_TYPE data_q,
                 DESC_TYPE data_t,
                 int idx_t,
                 DIST_TYPE& dist_min_1,
                 DIST_TYPE& dist_min_2,
                 MATCH_TYPE& match_idx_1) {
// clang-format off
#pragma HLS INLINE
        // clang-format on

        DIST_TYPE local_dist = computeHamming(data_q, data_t);
        if (local_dist < dist_min_1) {
            dist_min_2 = dist_min_1;
            dist_min_1 = local_dist;
            match_idx_1 = idx_t;
        } else if (local_dist < dist_min_2) {
            dist_min_2 = local_dist;
        }
    }
};

_GENERIC_BF_TPLT
void bfMatcher(DESC_TYPE desc_list_q[MAX_KEYPOINTS],
               DESC_TYPE desc_list_t[MAX_KEYPOINTS],
               MATCH_TYPE match_list[MAX_KEYPOINTS],
               ap_uint<32> num_keypoints_q,
               ap_uint<32> num_keypoints_t,
               float ratio_thresh) {
// clang-format off
#pragma HLS INLINE OFF
// clang-format on

#ifndef __SYNTHESIS__
    assert((num_keypoints_q <= MAX_KEYPOINTS) &&
           "Number of keypoints in the descriptor query set must be less than the MAX_KEYPOINTS parameter");
    assert((num_keypoints_t <= MAX_KEYPOINTS) &&
           "Number of keypoints in the descriptor training set must be less than the MAX_KEYPOINTS parameter");
#endif

    const int MAX_KEYPOINTS_PU = MAX_KEYPOINTS / PU;
    int proc_loop_kp1 = (num_keypoints_q / PU);

    MATCH_TYPE match_1[PU];
// clang-format off
#pragma HLS ARRAY_PARTITION variable=match_1 complete dim=1
    // clang-format on

    DIST_TYPE dist_min_1[PU], dist_min_2[PU];
// clang-format off
#pragma HLS ARRAY_PARTITION variable=dist_min_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=dist_min_2 complete dim=1
    // clang-format on

    _GENERIC_BF bf;

    bf.copyTrainingSet(desc_list_t, num_keypoints_t);

PROCESS_LOOP_DESC_1:
    for (int i = 0; i < proc_loop_kp1; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=MAX_KEYPOINTS_PU max=MAX_KEYPOINTS_PU
        // clang-format on
        DESC_TYPE query_set[PU];
// clang-format off
#pragma HLS ARRAY_PARTITION variable=query_set complete dim=1
    // clang-format on

    INIT_LOOP_1:
        for (int init_idx = 0; init_idx < PU; init_idx++) {
// clang-format off
#pragma HLS PIPELINE II=1
            // clang-format on
            query_set[init_idx] = desc_list_q[i * PU + init_idx];
            dist_min_1[init_idx] = 511;
            dist_min_2[init_idx] = 511;
            match_1[init_idx] = 0;
        }

    PROCESS_LOOP_DESC_2:
        for (int j = 0; j < num_keypoints_t; j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=MAX_KEYPOINTS max=MAX_KEYPOINTS
#pragma	HLS PIPELINE II=1
            // clang-format on
            DESC_TYPE train_data = bf.stack_t[j];
        PROCESS_LOOP_DESC_3:
            for (int k = 0; k < PU; k++) {
// clang-format off
#pragma HLS UNROLL
                // clang-format on
                bf.process(query_set[k], train_data, j, dist_min_1[k], dist_min_2[k], match_1[k]);
            }
        }

    WRITE_LOOP_1:
        for (int w_pu_idx = 0; w_pu_idx < PU; w_pu_idx++) {
// clang-format off
#pragma HLS PIPELINE II=1
            // clang-format on
            // Lowe's ratio test
            if ((float)dist_min_1[w_pu_idx] < (ratio_thresh * (float)dist_min_2[w_pu_idx]))
                match_list[i * PU + w_pu_idx] = match_1[w_pu_idx];
            else
                match_list[i * PU + w_pu_idx] = POOR_MATCH;
        }
    }

    const int REM_KEYPOINTS = PU - 1;
    int proc_loop_kp1_trunc = (num_keypoints_q / PU) * PU;
    int rem_extra = num_keypoints_q - proc_loop_kp1_trunc;

PROCESS_LOOP_EXTRA:
    for (int rem = 0; rem < rem_extra; rem++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=REM_KEYPOINTS max=REM_KEYPOINTS
        // clang-format on
        dist_min_1[0] = 511;
        dist_min_2[0] = 511;
    PROCESS_LOOP_REM_DESC_2:
        for (int j = 0; j < num_keypoints_t; j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=MAX_KEYPOINTS max=MAX_KEYPOINTS
#pragma	HLS PIPELINE II=1
            // clang-format on

            bf.process(desc_list_q[proc_loop_kp1_trunc + rem], bf.stack_t[j], j, dist_min_1[0], dist_min_2[0],
                       match_1[0]);
        }

        if ((float)dist_min_1[0] < (ratio_thresh * (float)dist_min_2[0]))
            match_list[proc_loop_kp1_trunc + rem] = match_1[0];
        else
            match_list[proc_loop_kp1_trunc + rem] = POOR_MATCH;
    }

    return;
}
// ======================================================================================

// Some clean up for macros used
#undef DESC_SIZE
#undef DESC_TYPE
} // end of cv
} // end of xf

#endif // end of __XF_VITIS_BFMATCHER_HPP__
