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
 * @file time_grid.hpp
 * @brief This file include the class TimeGrid
 *
 */

#ifndef __XF_FINTECH_TIMEGRID_HPP_
#define __XF_FINTECH_TIMEGRID_HPP_

#include "hls_math.h"
#include "ap_int.h"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {

namespace fintech {

namespace internal {

/**
 * @brief TimeGrid
 *
 * @tparam DT date type supported include float and double.
 * @tparam LEN data length
 *
 */

template <typename DT, int LEN>
class TimeGrid {
   public:
    // default constructor
    TimeGrid() {
#pragma HLS inline
    }

    void calcuGrid(int size,
                   DT* init_time,
                   DT dtMax,
                   DT* time,
                   DT* dtime,
                   int& exerciseEndCnt,
                   int* exerciseCnt,
                   int& fixedEndCnt,
                   int& floatingEndCnt,
                   int* fixedResetCnt,
                   int* floatingResetCnt,
                   int& endCnt) {
#pragma HLS inline
        time[0] = 0.0;
        int i;
        int j;
        int steps[LEN];
#pragma HLS resource variable = steps core = RAM_1P_LUTRAM
        int exercise_cnt_tmp = 0;
        int floating_reset_cnt_tmp = 0;
        int fixed_reset_cnt_tmp = 0;
        steps[0] = 0;
    loop_timegrid_1:
        for (i = 0; i < size - 1; i++) {
#pragma HLS loop_tripcount min = 10 max = 10
            DT tmp_dt = init_time[i + 1] - init_time[i];
            int step = hls::round(tmp_dt / dtMax);
            DT dt = tmp_dt / step;
            dtime[steps[i]] = dt;
        loop_timegrid_2:
            for (j = 1; j < step; j++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 5 max = 5
                time[steps[i] + j] = init_time[i] + dt * j;
                dtime[steps[i] + j] = dt;
            }
            time[steps[i] + j] = init_time[i + 1];
            steps[i + 1] = steps[i] + step;
            endCnt = steps[i + 1];
            if (exerciseCnt[exercise_cnt_tmp] == i) {
                exerciseCnt[exercise_cnt_tmp++] = steps[i + 1];
            }
            if (fixedResetCnt[fixed_reset_cnt_tmp] == i) {
                fixedResetCnt[fixed_reset_cnt_tmp++] = steps[i + 1];
            }
            if (floatingResetCnt[floating_reset_cnt_tmp] == i) {
                floatingResetCnt[floating_reset_cnt_tmp++] = steps[i + 1];
            }
        }
        exerciseEndCnt = exercise_cnt_tmp - 1;
        fixedEndCnt = fixed_reset_cnt_tmp - 1;
        floatingEndCnt = floating_reset_cnt_tmp - 1;
    }

    void calcuGrid(int size,
                   DT* init_time,
                   DT dtMax,
                   DT* time,
                   DT* dtime,
                   int& fixedEndCnt,
                   int& floatingEndCnt,
                   int* fixedResetCnt,
                   int* floatingResetCnt,
                   int& endCnt) {
#pragma HLS inline
        time[0] = 0.0;
        int i;
        int j;
        int steps[LEN];
#pragma HLS resource variable = steps core = RAM_1P_LUTRAM
        int floating_reset_cnt_tmp = 0;
        int fixed_reset_cnt_tmp = 0;
        steps[0] = 0;
    loop_timegrid_1:
        for (i = 0; i < size - 1; i++) {
#pragma HLS loop_tripcount min = 10 max = 10
            DT tmp_dt = init_time[i + 1] - init_time[i];
            int step = hls::round(tmp_dt / dtMax);
            DT dt = tmp_dt / step;
            dtime[steps[i]] = dt;
        loop_timegrid_2:
            for (j = 1; j < step; j++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 5 max = 5
                time[steps[i] + j] = init_time[i] + dt * j;
                dtime[steps[i] + j] = dt;
            }
            time[steps[i] + j] = init_time[i + 1];
            steps[i + 1] = steps[i] + step;
            endCnt = steps[i + 1];
            if (fixedResetCnt[fixed_reset_cnt_tmp] == i) {
                fixedResetCnt[fixed_reset_cnt_tmp++] = steps[i + 1];
            }
            if (floatingResetCnt[floating_reset_cnt_tmp] == i) {
                floatingResetCnt[floating_reset_cnt_tmp++] = steps[i + 1];
            }
        }
        fixedEndCnt = fixed_reset_cnt_tmp - 1;
        floatingEndCnt = floating_reset_cnt_tmp - 1;
    }

    void calcuGrid(int size,
                   DT* init_time,
                   DT dtMax,
                   DT* time,
                   DT* dtime,
                   int& floatingEndCnt,
                   int* floatingResetCnt,
                   int& endCnt) {
#pragma HLS inline
        time[0] = 0.0;
        int i;
        int j;
        int steps[LEN];
#pragma HLS resource variable = steps core = RAM_1P_LUTRAM
        int floating_reset_cnt_tmp = 0;
        steps[0] = 0;
    loop_timegrid_1:
        for (i = 0; i < size - 1; i++) {
#pragma HLS loop_tripcount min = 10 max = 10
            DT tmp_dt = init_time[i + 1] - init_time[i];
            int step = hls::round(tmp_dt / dtMax);
            DT dt = tmp_dt / step;
            dtime[steps[i]] = dt;
        loop_timegrid_2:
            for (j = 1; j < step; j++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 5 max = 5
                time[steps[i] + j] = init_time[i] + dt * j;
                dtime[steps[i] + j] = dt;
            }
            time[steps[i] + j] = init_time[i + 1];
            steps[i + 1] = steps[i] + step;
            endCnt = steps[i + 1];
            if (floatingResetCnt[floating_reset_cnt_tmp] == i) {
                floatingResetCnt[floating_reset_cnt_tmp++] = steps[i + 1];
            }
        }
        floatingEndCnt = floating_reset_cnt_tmp - 1;
    }

}; // class

}; // internal
}; // fintech
}; // xf

#endif
