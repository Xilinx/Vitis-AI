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
 * @file trinomial_tree.hpp
 * @brief This file include the class TrinomialTree
 *
 */

#ifndef __XF_FINTECH_TRINOMIAL_HPP_
#define __XF_FINTECH_TRINOMIAL_HPP_

#include "hls_math.h"
#include "ap_int.h"
#include <iostream>
using namespace std;

namespace xf {

namespace fintech {

#define Sqrt3 1.732050807568877

/**
 * @brief TrinomialTree is a lattice based trinomial tree structure
 *
 * @tparam DT date type supported include float and double.
 * @tparam Process stochastic process.
 * @tparam LEN maximum length of timestep
 *
 */

template <typename DT, typename Process, int LEN>
class TrinomialTree {
   private:
    DT x0;
    DT dx_next;
    DT dx_now;
    int j_min_next;
    int j_min_now;
    int j_max_now;
    DT arr_v2[LEN];
    DT v2;
    DT arr_v[LEN];
    DT v;
    DT m;
    DT dx[LEN];
    // Maximum value of j per timepoints
    int j_max[LEN];
    // Minimum value of j per timepoints
    int j_min[LEN];

    Process process;

    // calculate x
    DT calcuX(int j, DT dx, DT x0) {
#pragma HLS inline
        return j * dx + x0;
    }

   public:
    // default constructor
    TrinomialTree() {
#pragma HLS inline
#pragma HLS resource variable = dx core = RAM_2P_BRAM
#pragma HLS resource variable = j_min core = RAM_2P_BRAM
#pragma HLS resource variable = j_max core = RAM_2P_BRAM
#pragma HLS resource variable = arr_v2 core = RAM_1P_BRAM
#pragma HLS resource variable = arr_v core = RAM_1P_BRAM
    }

    /**
     * @brief initialize parameters.
     *
     * @param processParam parameters of stochastic process.
     * @param endCnt end counter of timepoints
     * @param x0_ initial underlying
     */
    void initialization(DT processParam[4], unsigned endCnt, DT x0_) {
#pragma HLS pipeline
        process.init(processParam[0], processParam[1], processParam[2], processParam[3]);
        // parameter initialize
        x0 = x0_;
        dx[0] = 0.0;
        dx_next = 0.0;
        j_min_next = 0;
        j_max[0] = 0;
        j_min[0] = 0;
        int i;
    loop_triTree:
        for (i = 1; i < endCnt; i++) {
#pragma HLS loop_tripcount min = 50 max = 50
            j_max[i] = 1 - LEN;
            j_min[i] = LEN - 1;
        }
    }

    /**
     * @brief update parameters from array to calculate probability for timepoint=i
     *
     * @param i timepoint counter
     * @param t time for timepoint=i
     * @param dt the difference time between timepoint=i+1 and timepoint=i
     */
    void dxUpdateNoCalcu(int i, DT t, DT dt) {
#pragma HLS inline
        // v2 = process.variance(t,0.0,dt);
        v2 = arr_v2[i];
        // v = hls::sqrt(v2);
        v = arr_v[i];
        dx_next = dx[i + 1];
        dx_now = dx[i];
        j_min_now = j_min[i];
        j_min_next = j_min[i + 1];
    }

    /**
     * @brief calculate to get parameters to calculate probability for timepoint=i
     *
     * @param i timepoint counter
     * @param t time for timepoint=i
     * @param dt the difference time between timepoint=i+1 and timepoint=i
     */
    void dxUpdate(int i, DT t, DT dt) {
#pragma HLS inline
        dx_now = dx_next;
        // dx_now = dx[i];
        v2 = process.variance(t, 0.0, dt);
        arr_v2[i] = v2;
#ifndef __SYNTHESIS__
        v = std::sqrt(v2);
#else
        v = hls::sqrt(v2);
#endif
        arr_v[i] = v;
        dx_next = v * Sqrt3;
        dx[i + 1] = dx_next;
        j_min_now = j_min_next;
        // j_min_now = j_min[i];
        j_max_now = j_max[i];
        DT x_min = calcuX(j_min_now, dx_now, x0);
        DT x_max = calcuX(j_max_now, dx_now, x0);
        DT m_min = process.expectation(t, x_min, dt);
        DT m_max = process.expectation(t, x_max, dt);
#ifndef __SYNTHESIS__
        int k_min = std::round((m_min - x0) / dx_next) - 1;
        int k_max = std::round((m_max - x0) / dx_next) + 1;
#else
        int k_min = hls::round((m_min - x0) / dx_next) - 1;
        int k_max = hls::round((m_max - x0) / dx_next) + 1;
#endif
        j_min[i + 1] = k_min;
        j_max[i + 1] = k_max;
        j_min_next = k_min;
    }

    /**
     * @brief calculate probability and descendant index for timepoint=i and node=j
     *
     * @param j node index
     * @param t time for timepoint=i
     * @param dt the difference time between timepoint=i+1 and timepoint=i
     * @param probs return the calculated probability
     * @return return the calculated descendant index
     */
    int calculateProbability(int j, DT t, DT dt, DT* probs) {
#pragma HLS inline
        DT x = calcuX(j + j_min_now, dx_now, x0);
        m = process.expectation(t, x, dt);
#ifndef __SYNTHESIS__
        int k = std::round((m - x0) / dx_next); // floor:+0.5
#else
        int k = hls::round((m - x0) / dx_next); // floor:+0.5
#endif
        DT e = m - x0 - k * dx_next;
        DT e2 = e * e / v2;
        DT e3 = e * Sqrt3 / v;
        DT temp = 1.0 + e2;
        probs[0] = (temp - e3) / 6.0;
        probs[2] = (temp + e3) / 6.0;
        probs[1] = 1 - probs[0] - probs[2];
        return k - j_min_next - 1; // descendant index
    }

    /**
     * @brief Branch size per timepoints
     *
     * @param i timepoint counter
     * @return the branch size at timepoint is i
     */
    int size(int i) {
#pragma HLS inline
        return j_max[i] - j_min[i] + 1;
    }

    /**
     * @brief underlying
     *
     * @param index node index for branch
     * @return underlying
     */
    DT underlying(int index) {
#pragma HLS inline
        int j = index + j_min_now;
        return calcuX(j, dx_now, x0);
    }

    /**
     * @brief get dx
     *
     * @param i timepoint counter
     * @return dx at timepoint is i
     */
    DT getDx(int i) { return dx[i]; }

}; // class

#undef Sqrt3

}; // fintech

}; // xf
#endif
