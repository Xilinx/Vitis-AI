
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
 * @file tree_instrument.hpp
 * @brief This file include class TreeInstrument
 *
 */

#ifndef __XF_FINTECH_TREEINSTRUMENT_HPP_
#define __XF_FINTECH_TREEINSTRUMENT_HPP_

#include "hls_math.h"
#include "ap_int.h"
#include "utils.hpp"
#include "trinomial_tree.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
using namespace std;
#endif

namespace xf {

namespace fintech {

namespace internal {

/**
 * @brief TreeInstrument swaption, swap, cap floor, callable bond
 *
 * @tparam DT date type supported include float and double.
 * @tparam IT 0: swaption, 1: swap, 2: cap floor, 3: callable bond
 * @param  LEN2 maximum length of node of tree, which affects the latency and resources utilization.
 *
 */
template <typename DT, int IT, int LEN2>
class TreeInstrument {
   public:
    /**
         * @brief default constructor
         */
    TreeInstrument() {
#pragma HLS inline
    }
    /**
     * @brief reset reset values
     *
     * @param size size of used values
     * @param values process values to calulate NPV
     *
     */
    void reset(unsigned size, DT values[][LEN2]);

    /**
     * @brief stepback calculate values of the current timepoint based on the values of the next timepoint for 1D
     * framework.
     *
     * @param j node of tree structure
     * @param index node of tree structure
     * @param discount price of the current timepoint relative to the next timepoint.
     * @param probs probability of node of tree structure
     * @param values1 process values to calulate NPV
     * @param values2 process values to calulate NPV
     *
     */
    void stepback(unsigned j, unsigned index, DT discount, DT* probs, DT values1[][LEN2], DT values2[][LEN2]);
    /**
     * @brief stepback calculate values of the current timepoint based on the values of the next timepoint for 2D
     * framework.
     *
     * @param j node of tree structure
     * @param index node of tree structure
     * @param modulo node length of the next timepoint
     * @param discount price of the current timepoint relative to the next timepoint.
     * @param probs probability of node of tree structure
     * @param values1 process values to calulate NPV
     * @param values2 process values to calulate NPV
     *
     */
    void stepback(unsigned j,
                  unsigned* index,
                  unsigned modulo,
                  DT discount,
                  DT probs[2][3],
                  DT values1[][LEN2],
                  DT values2[][LEN2]);

    /**
     * @brief adjustValues Adjust value based on conditions
     *
     * @param i the current timepoint counter
     * @param size node size of tree structure of the current timepoint
     * @param t the current timepoint
     * @param values1 process values to calulate NPV
     * @param values2 process values to calulate NPV
     *
     */
    void adjustValues(int i, int size, DT t, DT values1[][LEN2], DT values2[][LEN2]);

}; // class TreeInstrument

template <typename DT, int LEN2>
class TreeInstrument<DT, 0, LEN2> { // swaption
   private:
    static const int L = 4;
    int type;
    DT nominal, accruedSpread, fixedCoupon;
    int floating_cnt, fixed_cnt, exercise_cnt;
    int floating_data[20], fixed_data[20], exercise_data[20];
    DT m[3][3];

   public:
    TreeInstrument() {
#pragma HLS inline
    }
    void initialize(int typeIn,
                    DT nominalIn,
                    DT accruedSpreadIn,
                    DT fixedCouponIn,
                    int floatingEndCnt,
                    int fixedEndCnt,
                    int exerciseEndCnt,
                    int* floatingCnt,
                    int* fixedCnt,
                    int* exerciseCnt) {
#pragma HLS inline
        type = typeIn;
        nominal = nominalIn;
        accruedSpread = accruedSpreadIn;
        fixedCoupon = fixedCouponIn;

        floating_cnt = floatingEndCnt;
        fixed_cnt = fixedEndCnt;
        exercise_cnt = exerciseEndCnt;

        for (int i = 0; i <= floatingEndCnt; i++)
#pragma HLS loop_tripcount min = 10 max = 10
            floating_data[i] = floatingCnt[i];

        for (int i = 0; i <= fixedEndCnt; i++)
#pragma HLS loop_tripcount min = 10 max = 10
            fixed_data[i] = fixedCnt[i];

        for (int i = 0; i <= exerciseEndCnt; i++)
#pragma HLS loop_tripcount min = 10 max = 10
            exercise_data[i] = exerciseCnt[i];
    }

    void initialize(int typeIn,
                    DT rho,
                    DT nominalIn,
                    DT accruedSpreadIn,
                    DT fixedCouponIn,
                    int floatingEndCnt,
                    int fixedEndCnt,
                    int exerciseEndCnt,
                    int* floatingCnt,
                    int* fixedCnt,
                    int* exerciseCnt) {
#pragma HLS inline
        type = typeIn;
        nominal = nominalIn;
        accruedSpread = accruedSpreadIn;
        fixedCoupon = fixedCouponIn;

        floating_cnt = floatingEndCnt;
        fixed_cnt = fixedEndCnt;
        exercise_cnt = exerciseEndCnt;

        for (int i = 0; i <= floatingEndCnt; i++)
#pragma HLS loop_tripcount min = 10 max = 10
            floating_data[i] = floatingCnt[i];

        for (int i = 0; i <= fixedEndCnt; i++)
#pragma HLS loop_tripcount min = 10 max = 10
            fixed_data[i] = fixedCnt[i];

        for (int i = 0; i <= exerciseEndCnt; i++)
#pragma HLS loop_tripcount min = 10 max = 10
            exercise_data[i] = exerciseCnt[i];

        m[0][0] = -1.0 / 36.0 * rho;
        m[0][1] = -4.0 / 36.0 * rho;
        m[0][2] = 5.0 / 36.0 * rho;
        m[1][0] = -4.0 / 36.0 * rho;
        m[1][1] = 8.0 / 36.0 * rho;
        m[1][2] = -4.0 / 36.0 * rho;
        m[2][0] = 5.0 / 36.0 * rho;
        m[2][1] = -4.0 / 36.0 * rho;
        m[2][2] = -1.0 / 36.0 * rho;
    }

    void reset(unsigned size, DT values[L][LEN2]) {
#pragma HLS inline
        for (int i = 0; i < size; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 100 max = 100
            values[0][i] = 0.0; // swpation
            values[1][i] = 0.0; // swap
            values[2][i] = 1.0; // fixed
            values[3][i] = 1.0; // floating
        }
    }

    void reset(unsigned size, xf_2D_array<DT, 4, LEN2>& values) {
#pragma HLS inline
        for (int i = 0; i < size; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 100 max = 100
            values.write(0.0, 0, i);
            values.write(0.0, 1, i);
            values.write(1.0, 2, i);
            values.write(1.0, 3, i);
        }
    }

    void stepback(unsigned j, unsigned index, DT discount, DT* probs, DT values1[L][LEN2], DT values2[L][LEN2]) {
#pragma HLS inline
        for (int k1 = 0; k1 < L; k1++) {
            DT v[3];
            for (int k2 = 0; k2 < 3; k2++) {
                v[k2] = probs[k2] * values1[k1][index + k2];
            }
            values2[k1][j] = (v[0] + v[1] + v[2]) * discount;
        }
    }

    void stepback(unsigned j,
                  unsigned* index,
                  unsigned modulo,
                  DT discount,
                  DT probs[2][3],
                  xf_2D_array<DT, 4, LEN2>& values1,
                  xf_2D_array<DT, 4, LEN2>& values2) {
        //#pragma HLS inline
        DT tmp[L] = {0.0};
        for (int k1 = 0; k1 < L; k1++) {
            for (int branch2 = 0; branch2 < 3; branch2++) {
                DT prob[3];
                for (int branch1 = 0; branch1 < 3; branch1++) {
                    prob[branch1] = probs[0][branch1] * probs[1][branch2] + m[branch1][branch2];
                }
                int k = index[0] + (index[1] + branch2) * modulo;
                tmp[k1] += prob[0] * values1.read(k1, k) + prob[1] * values1.read(k1, k + 1) +
                           prob[2] * values1.read(k1, k + 2);
            }
            values2.write(tmp[k1] * discount, k1, j);
        }
    }

    void adjustValues(int i, int size, DT t, DT values1[L][LEN2], DT values2[L][LEN2]) {
#pragma HLS inline
        DT coupon_floating;
        DT coupon_fixed;
        DT swap_value;
        int temp_floating_cnt, temp_fixed_cnt, temp_swaption_cnt;
        for (int j = 0; j < size; j++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 100 max = 100
            // floating coupon swap
            if (floating_data[floating_cnt] == i && floating_cnt >= 0) {
                coupon_floating = nominal * (1.0 - values2[3][j]) + accruedSpread * values2[3][j];
                temp_floating_cnt = floating_cnt - 1;
                values1[3][j] = 1.0;
            } else {
                coupon_floating = 0.0;
                temp_floating_cnt = floating_cnt;
                values1[3][j] = values2[3][j];
            }

            // fixed coupon swap
            if (fixed_data[fixed_cnt] == i && fixed_cnt >= 0) {
                coupon_fixed = fixedCoupon * values2[2][j];
                temp_fixed_cnt = fixed_cnt - 1;
                values1[2][j] = 1.0;
            } else {
                coupon_fixed = 0.0;
                temp_fixed_cnt = fixed_cnt;
                values1[2][j] = values2[2][j];
            }

            if (type)
                swap_value = values2[1][j] - coupon_floating + coupon_fixed; // Receiver
            else
                swap_value = values2[1][j] + coupon_floating - coupon_fixed; // Payer

            values1[1][j] = swap_value;

            // exercise
            if (exercise_data[exercise_cnt] == i && exercise_cnt >= 0) {
                if (values2[0][j] <= swap_value) {
                    values1[0][j] = swap_value;
                } else {
                    values1[0][j] = values2[0][j];
                }
                temp_swaption_cnt = exercise_cnt - 1;
            } else {
                values1[0][j] = values2[0][j];
                temp_swaption_cnt = exercise_cnt;
            }
        }
        floating_cnt = temp_floating_cnt;
        fixed_cnt = temp_fixed_cnt;
        exercise_cnt = temp_swaption_cnt;
    }

    void adjustValues(int i, int size, DT t, xf_2D_array<DT, 4, LEN2>& values1, xf_2D_array<DT, 4, LEN2>& values2) {
#pragma HLS inline
        DT coupon_floating;
        DT coupon_fixed;
        DT swap_value;
        int temp_floating_cnt, temp_fixed_cnt, temp_swaption_cnt;
        for (int j = 0; j < size; j++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 100 max = 100
            // floating coupon swap
            if (floating_data[floating_cnt] == i && floating_cnt >= 0) {
                coupon_floating = nominal * (1.0 - values2.read(3, j)) + accruedSpread * values2.read(3, j);
                temp_floating_cnt = floating_cnt - 1;
                values1.write(1.0, 3, j);
            } else {
                coupon_floating = 0.0;
                temp_floating_cnt = floating_cnt;
                values1.write(values2.read(3, j), 3, j);
            }

            // fixed coupon swap
            if (fixed_data[fixed_cnt] == i && fixed_cnt >= 0) {
                coupon_fixed = fixedCoupon * values2.read(2, j);
                temp_fixed_cnt = fixed_cnt - 1;
                values1.write(1.0, 2, j);
            } else {
                coupon_fixed = 0.0;
                temp_fixed_cnt = fixed_cnt;
                values1.write(values2.read(2, j), 2, j);
            }

            if (type)
                swap_value = values2.read(1, j) - coupon_floating + coupon_fixed; // Receiver
            else
                swap_value = values2.read(1, j) + coupon_floating - coupon_fixed; // Payer

            values1.write(swap_value, 1, j);

            // exercise
            if (exercise_data[exercise_cnt] == i && exercise_cnt >= 0) {
                if (values2.read(0, j) <= swap_value) {
                    values1.write(swap_value, 0, j);
                } else {
                    values1.write(values2.read(0, j), 0, j);
                }
                temp_swaption_cnt = exercise_cnt - 1;
            } else {
                values1.write(values2.read(0, j), 0, j);
                temp_swaption_cnt = exercise_cnt;
            }
        }
        floating_cnt = temp_floating_cnt;
        fixed_cnt = temp_fixed_cnt;
        exercise_cnt = temp_swaption_cnt;
    }
}; // class Tree Swaption Instrument

template <typename DT, int LEN2>
class TreeInstrument<DT, 1, LEN2> { // swap
   private:
    static const int L = 3;
    int type;
    DT nominal, accruedSpread, fixedCoupon;
    int floating_cnt, fixed_cnt;
    int floating_data[20], fixed_data[20];
    DT m[3][3];

   public:
    TreeInstrument() {
#pragma HLS inline
    }

    void initialize(int typeIn,
                    DT nominalIn,
                    DT accruedSpreadIn,
                    DT fixedCouponIn,
                    int floatingEndCnt,
                    int fixedEndCnt,
                    int* floatingCnt,
                    int* fixedCnt) {
#pragma HLS inline
        type = typeIn;
        nominal = nominalIn;
        accruedSpread = accruedSpreadIn;
        fixedCoupon = fixedCouponIn;

        floating_cnt = floatingEndCnt;
        fixed_cnt = fixedEndCnt;

        for (int i = 0; i <= floatingEndCnt; i++) floating_data[i] = floatingCnt[i];

        for (int i = 0; i <= fixedEndCnt; i++) fixed_data[i] = fixedCnt[i];
    }

    void initialize(int typeIn,
                    DT rho,
                    DT nominalIn,
                    DT accruedSpreadIn,
                    DT fixedCouponIn,
                    int floatingEndCnt,
                    int fixedEndCnt,
                    int* floatingCnt,
                    int* fixedCnt) {
#pragma HLS inline
        type = typeIn;
        nominal = nominalIn;
        accruedSpread = accruedSpreadIn;
        fixedCoupon = fixedCouponIn;

        floating_cnt = floatingEndCnt;
        fixed_cnt = fixedEndCnt;

        for (int i = 0; i <= floatingEndCnt; i++) floating_data[i] = floatingCnt[i];

        for (int i = 0; i <= fixedEndCnt; i++) fixed_data[i] = fixedCnt[i];

        m[0][0] = -1.0 / 36.0 * rho;
        m[0][1] = -4.0 / 36.0 * rho;
        m[0][2] = 5.0 / 36.0 * rho;
        m[1][0] = -4.0 / 36.0 * rho;
        m[1][1] = 8.0 / 36.0 * rho;
        m[1][2] = -4.0 / 36.0 * rho;
        m[2][0] = 5.0 / 36.0 * rho;
        m[2][1] = -4.0 / 36.0 * rho;
        m[2][2] = -1.0 / 36.0 * rho;
    }

    void reset(unsigned size, DT values[L][LEN2]) {
#pragma HLS inline
        for (int i = 0; i < size; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 100 max = 100
            values[0][i] = 0.0; // swpation
            values[1][i] = 1.0; // swap
            values[2][i] = 1.0; // fixed
        }
    }

    void reset(unsigned size, xf_2D_array<DT, 4, LEN2>& values) {
#pragma HLS inline
        for (int i = 0; i < size; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 100 max = 100
            values.write(0.0, 0, i);
            values.write(1.0, 1, i);
            values.write(1.0, 2, i);
        }
    }

    void stepback(unsigned j, unsigned index, DT discount, DT* probs, DT values1[L][LEN2], DT values2[L][LEN2]) {
#pragma HLS inline
        for (int k1 = 0; k1 < L; k1++) {
            DT v[3];
            for (int k2 = 0; k2 < 3; k2++) {
                v[k2] = probs[k2] * values1[k1][index + k2];
            }
            values2[k1][j] = (v[0] + v[1] + v[2]) * discount;
        }
    }

    void stepback(unsigned j,
                  unsigned* index,
                  unsigned modulo,
                  DT discount,
                  DT probs[2][3],
                  xf_2D_array<DT, 4, LEN2>& values1,
                  xf_2D_array<DT, 4, LEN2>& values2) {
        //#pragma HLS inline
        DT tmp[L] = {0.0};
        for (int k1 = 0; k1 < L; k1++) {
            for (int branch2 = 0; branch2 < 3; branch2++) {
                DT prob[3];
                for (int branch1 = 0; branch1 < 3; branch1++) {
                    prob[branch1] = probs[0][branch1] * probs[1][branch2] + m[branch1][branch2];
                }
                int k = index[0] + (index[1] + branch2) * modulo;
                tmp[k1] += prob[0] * values1.read(k1, k) + prob[1] * values1.read(k1, k + 1) +
                           prob[2] * values1.read(k1, k + 2);
            }
            values2.write(tmp[k1] * discount, k1, j);
        }
    }

    void adjustValues(int i, int size, DT t, DT values1[L][LEN2], DT values2[L][LEN2]) {
#pragma HLS inline
        DT coupon_floating;
        DT coupon_fixed;
        DT swap_value;
        int temp_floating_cnt, temp_fixed_cnt;
        for (int j = 0; j < size; j++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 100 max = 100
            // floating coupon swap
            if (floating_data[floating_cnt] == i && floating_cnt >= 0) {
                coupon_floating = nominal * (1.0 - values2[2][j]) + accruedSpread * values2[2][j];
                temp_floating_cnt = floating_cnt - 1;
                values1[2][j] = 1.0;
            } else {
                coupon_floating = 0.0;
                temp_floating_cnt = floating_cnt;
                values1[2][j] = values2[2][j];
            }

            // fixed coupon swap
            if (fixed_data[fixed_cnt] == i && fixed_cnt >= 0) {
                coupon_fixed = fixedCoupon * values2[1][j];
                temp_fixed_cnt = fixed_cnt - 1;
                values1[1][j] = 1.0;
            } else {
                coupon_fixed = 0.0;
                temp_fixed_cnt = fixed_cnt;
                values1[1][j] = values2[1][j];
            }

            if (type)
                swap_value = values2[0][j] - coupon_floating + coupon_fixed; // Receiver
            else
                swap_value = values2[0][j] + coupon_floating - coupon_fixed; // Payer

            values1[0][j] = swap_value;
        }
        floating_cnt = temp_floating_cnt;
        fixed_cnt = temp_fixed_cnt;
    }

    void adjustValues(int i, int size, DT t, xf_2D_array<DT, 4, LEN2>& values1, xf_2D_array<DT, 4, LEN2>& values2) {
#pragma HLS inline
        DT coupon_floating;
        DT coupon_fixed;
        DT swap_value;
        int temp_floating_cnt, temp_fixed_cnt;
        for (int j = 0; j < size; j++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 100 max = 100
            // floating coupon swap
            if (floating_data[floating_cnt] == i && floating_cnt >= 0) {
                coupon_floating = nominal * (1.0 - values2.read(2, j)) + accruedSpread * values2.read(2, j);
                temp_floating_cnt = floating_cnt - 1;
                values1.write(1.0, 2, j);
            } else {
                coupon_floating = 0.0;
                temp_floating_cnt = floating_cnt;
                values1.write(values2.read(2, j), 2, j);
            }

            // fixed coupon swap
            if (fixed_data[fixed_cnt] == i && fixed_cnt >= 0) {
                coupon_fixed = fixedCoupon * values2.read(1, j);
                temp_fixed_cnt = fixed_cnt - 1;
                values1.write(1.0, 1, j);
            } else {
                coupon_fixed = 0.0;
                temp_fixed_cnt = fixed_cnt;
                values1.write(values2.read(1, j), 1, j);
            }

            if (type)
                swap_value = values2.read(0, j) - coupon_floating + coupon_fixed; // Receiver
            else
                swap_value = values2.read(0, j) + coupon_floating - coupon_fixed; // Payer

            values1.write(swap_value, 0, j);
        }
        floating_cnt = temp_floating_cnt;
        fixed_cnt = temp_fixed_cnt;
    }

}; // class Tree Swap Instrument

template <typename DT, int LEN2>
class TreeInstrument<DT, 2, LEN2> { // capfloor
   private:
    static const int L = 2;
    int type;
    DT nominal;
    DT cfRate[2];
    int floating_cnt;
    DT lastTime;
    int floating_data[20];
    DT m[3][3];

   public:
    TreeInstrument() {
#pragma HLS inline
    }

    void initialize(int typeIn, DT lastTimeIn, DT nominalIn, DT* cfRateIn, int floatingEndCnt, int* floatingCnt) {
#pragma HLS inline
        type = typeIn;
        lastTime = lastTimeIn;
        nominal = nominalIn;
        cfRate[0] = cfRateIn[0];
        cfRate[1] = cfRateIn[1];

        floating_cnt = floatingEndCnt;

        for (int i = 0; i <= floatingEndCnt; i++) floating_data[i] = floatingCnt[i];
    }

    void initialize(
        int typeIn, DT lastTimeIn, DT rho, DT nominalIn, DT* cfRateIn, int floatingEndCnt, int* floatingCnt) {
#pragma HLS inline
        type = typeIn;
        lastTime = lastTimeIn;
        nominal = nominalIn;
        cfRate[0] = cfRateIn[0];
        cfRate[1] = cfRateIn[1];

        floating_cnt = floatingEndCnt;

        for (int i = 0; i <= floatingEndCnt; i++) floating_data[i] = floatingCnt[i];

        m[0][0] = -1.0 / 36.0 * rho;
        m[0][1] = -4.0 / 36.0 * rho;
        m[0][2] = 5.0 / 36.0 * rho;
        m[1][0] = -4.0 / 36.0 * rho;
        m[1][1] = 8.0 / 36.0 * rho;
        m[1][2] = -4.0 / 36.0 * rho;
        m[2][0] = 5.0 / 36.0 * rho;
        m[2][1] = -4.0 / 36.0 * rho;
        m[2][2] = -1.0 / 36.0 * rho;
    }

    void reset(unsigned size, DT values[L][LEN2]) {
#pragma HLS inline
        for (int i = 0; i < size; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 100 max = 100
            values[0][i] = 0.0;
            values[1][i] = 1.0;
        }
    }

    void reset(unsigned size, xf_2D_array<DT, 4, LEN2>& values) {
#pragma HLS inline
        for (int i = 0; i < size; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 100 max = 100
            values.write(0.0, 0, i);
            values.write(1.0, 1, i);
        }
    }

    void stepback(unsigned j, unsigned index, DT discount, DT* probs, DT values1[L][LEN2], DT values2[L][LEN2]) {
#pragma HLS inline
        for (int k1 = 0; k1 < L; k1++) {
            DT v[3];
            for (int k2 = 0; k2 < 3; k2++) {
                v[k2] = probs[k2] * values1[k1][index + k2];
            }
            values2[k1][j] = (v[0] + v[1] + v[2]) * discount;
        }
    }

    void stepback(unsigned j,
                  unsigned* index,
                  unsigned modulo,
                  DT discount,
                  DT probs[2][3],
                  xf_2D_array<DT, 4, LEN2>& values1,
                  xf_2D_array<DT, 4, LEN2>& values2) {
        //#pragma HLS inline
        DT tmp[L] = {0.0};
        for (int k1 = 0; k1 < L; k1++) {
            for (int branch2 = 0; branch2 < 3; branch2++) {
                DT prob[3];
                for (int branch1 = 0; branch1 < 3; branch1++) {
                    prob[branch1] = probs[0][branch1] * probs[1][branch2] + m[branch1][branch2];
                }
                int k = index[0] + (index[1] + branch2) * modulo;
                tmp[k1] += prob[0] * values1.read(k1, k) + prob[1] * values1.read(k1, k + 1) +
                           prob[2] * values1.read(k1, k + 2);
            }
            values2.write(tmp[k1] * discount, k1, j);
        }
    }

    void adjustValues(int i, int size, DT t, DT values1[L][LEN2], DT values2[L][LEN2]) {
#pragma HLS inline
        DT coupon1 = 0.0;
        DT coupon2 = 0.0;
        DT swap_value;
        int temp_floating_cnt, temp_fixed_cnt;
        DT temp_lastTime;
        for (int j = 0; j < size; j++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 100 max = 100
            DT temp_value = values2[1][j];
            if (floating_data[floating_cnt] == i && floating_cnt >= 0) {
                DT accrual = 1.0 + cfRate[0] * (lastTime - t);
                DT accrual2 = 1.0 + cfRate[1] * (lastTime - t);
                DT strike = 1.0 / accrual - temp_value;
                DT strike2 = 1.0 / accrual2 - temp_value;
                if (type == 0) {
                    if (strike < 0.0)
                        coupon1 = 0.0;
                    else
                        coupon1 = nominal * accrual * strike;
                } else if (type == 1) {
                    if (strike < 0.0)
                        coupon1 = 0.0;
                    else
                        coupon1 = nominal * accrual * strike;
                    if (strike2 > 0.0)
                        coupon2 = 0.0;
                    else
                        coupon2 = nominal * accrual2 * strike2;
                } else if (type == 2) {
                    if (strike2 > 0.0)
                        coupon2 = 0.0;
                    else
                        coupon2 = nominal * accrual2 * (-strike2);
                }
                temp_lastTime = t;
                temp_floating_cnt = floating_cnt - 1;
                values1[1][j] = 1.0;
            } else {
                coupon1 = 0.0;
                temp_floating_cnt = floating_cnt;
                temp_lastTime = lastTime;
                values1[1][j] = values2[1][j];
            }

            swap_value = values2[0][j] + coupon1 + coupon2; // Payer

            values1[0][j] = swap_value;
        }
        floating_cnt = temp_floating_cnt;
        lastTime = temp_lastTime;
    }

    void adjustValues(int i, int size, DT t, xf_2D_array<DT, 4, LEN2>& values1, xf_2D_array<DT, 4, LEN2>& values2) {
#pragma HLS inline
        DT coupon1 = 0.0;
        DT coupon2 = 0.0;
        DT swap_value;
        int temp_floating_cnt, temp_fixed_cnt;
        DT temp_lastTime;
        for (int j = 0; j < size; j++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 100 max = 100
            DT temp_value = values2.read(1, j);
            if (floating_data[floating_cnt] == i && floating_cnt >= 0) {
                DT accrual = 1.0 + cfRate[0] * (lastTime - t);
                DT accrual2 = 1.0 + cfRate[1] * (lastTime - t);
                DT strike = 1.0 / accrual - temp_value;
                DT strike2 = 1.0 / accrual2 - temp_value;
                if (type == 0) {
                    if (strike < 0.0)
                        coupon1 = 0.0;
                    else
                        coupon1 = nominal * accrual * strike;
                } else if (type == 1) {
                    if (strike < 0.0)
                        coupon1 = 0.0;
                    else
                        coupon1 = nominal * accrual * strike;
                    if (strike2 > 0.0)
                        coupon2 = 0.0;
                    else
                        coupon2 = nominal * accrual2 * strike2;
                } else if (type == 2) {
                    if (strike2 > 0.0)
                        coupon2 = 0.0;
                    else
                        coupon2 = nominal * accrual2 * (-strike2);
                }
                temp_lastTime = t;
                temp_floating_cnt = floating_cnt - 1;
                values1.write(1.0, 1, j);
            } else {
                coupon1 = 0.0;
                temp_floating_cnt = floating_cnt;
                temp_lastTime = lastTime;
                values1.write(values2.read(1, j), 1, j);
            }

            swap_value = values2.read(0, j) + coupon1 + coupon2; // Payer

            values1.write(swap_value, 0, j);
        }
        floating_cnt = temp_floating_cnt;
        lastTime = temp_lastTime;
    }

}; // class Tree CapFloor Instrument

template <typename DT, int LEN2>
class TreeInstrument<DT, 3, LEN2> { // callable
   private:
    static const int L = 1;
    int type;
    DT nominal, fixedCoupon;
    int floating_cnt, fixed_cnt;
    int floating_data[20], fixed_data[20];
    DT m[3][3];

   public:
    TreeInstrument() {
#pragma HLS inline
    }

    void initialize(int typeIn,
                    DT nominalIn,
                    DT fixedCouponIn,
                    int floatingEndCnt,
                    int fixedEndCnt,
                    int* floatingCnt,
                    int* fixedCnt) {
#pragma HLS inline
        type = typeIn;
        nominal = nominalIn;
        fixedCoupon = fixedCouponIn;

        floating_cnt = floatingEndCnt;
        fixed_cnt = fixedEndCnt;

        for (int i = 0; i <= floatingEndCnt; i++) floating_data[i] = floatingCnt[i];

        for (int i = 0; i <= fixedEndCnt; i++) fixed_data[i] = fixedCnt[i];
    }

    void initialize(int typeIn,
                    DT rho,
                    DT nominalIn,
                    DT fixedCouponIn,
                    int floatingEndCnt,
                    int fixedEndCnt,
                    int* floatingCnt,
                    int* fixedCnt) {
#pragma HLS inline
        type = typeIn;
        nominal = nominalIn;
        fixedCoupon = fixedCouponIn;

        floating_cnt = floatingEndCnt;
        fixed_cnt = fixedEndCnt;

        for (int i = 0; i <= floatingEndCnt; i++) floating_data[i] = floatingCnt[i];

        for (int i = 0; i <= fixedEndCnt; i++) fixed_data[i] = fixedCnt[i];

        m[0][0] = -1.0 / 36.0 * rho;
        m[0][1] = -4.0 / 36.0 * rho;
        m[0][2] = 5.0 / 36.0 * rho;
        m[1][0] = -4.0 / 36.0 * rho;
        m[1][1] = 8.0 / 36.0 * rho;
        m[1][2] = -4.0 / 36.0 * rho;
        m[2][0] = 5.0 / 36.0 * rho;
        m[2][1] = -4.0 / 36.0 * rho;
        m[2][2] = -1.0 / 36.0 * rho;
    }

    void reset(unsigned size, DT values[L][LEN2]) {
#pragma HLS inline
        for (int i = 0; i < size; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 100 max = 100
            values[0][i] = nominal + fixedCoupon;
        }
    }

    void reset(unsigned size, xf_2D_array<DT, 4, LEN2>& values) {
#pragma HLS inline
        for (int i = 0; i < size; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 100 max = 100
            values.write(nominal + fixedCoupon, 0, i);
        }
    }

    void stepback(unsigned j, unsigned index, DT discount, DT* probs, DT values1[L][LEN2], DT values2[L][LEN2]) {
#pragma HLS inline
        for (int k1 = 0; k1 < L; k1++) {
            DT v[3];
            for (int k2 = 0; k2 < 3; k2++) {
                v[k2] = probs[k2] * values1[k1][index + k2];
            }
            values2[k1][j] = (v[0] + v[1] + v[2]) * discount;
        }
    }

    void stepback(unsigned j,
                  unsigned* index,
                  unsigned modulo,
                  DT discount,
                  DT probs[2][3],
                  xf_2D_array<DT, 4, LEN2>& values1,
                  xf_2D_array<DT, 4, LEN2>& values2) {
        //#pragma HLS inline
        DT tmp[L] = {0.0};
        for (int k1 = 0; k1 < L; k1++) {
            for (int branch2 = 0; branch2 < 3; branch2++) {
                DT prob[3];
                for (int branch1 = 0; branch1 < 3; branch1++) {
                    prob[branch1] = probs[0][branch1] * probs[1][branch2] + m[branch1][branch2];
                }
                int k = index[0] + (index[1] + branch2) * modulo;
                tmp[k1] += prob[0] * values1.read(k1, k) + prob[1] * values1.read(k1, k + 1) +
                           prob[2] * values1.read(k1, k + 2);
            }
            values2.write(tmp[k1] * discount, k1, j);
        }
    }

    void adjustValues(int i, int size, DT t, DT values1[L][LEN2], DT values2[L][LEN2]) {
#pragma HLS inline
        DT coupon;
        int temp_floating_cnt, temp_fixed_cnt;
        for (int j = 0; j < size; j++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 100 max = 100

            coupon = values2[0][j];
            if (floating_data[floating_cnt] == i && floating_cnt >= 0) {
                if (type == 0) {
                    if (coupon > nominal) coupon = nominal;
                } else {
                    if (coupon < nominal) coupon = nominal;
                }
                temp_floating_cnt = floating_cnt - 1;
            } else {
                temp_floating_cnt = floating_cnt;
            }

            if (fixed_data[fixed_cnt] == i && fixed_cnt >= 0) {
                coupon += fixedCoupon;
                temp_fixed_cnt = fixed_cnt - 1;
            } else {
                temp_fixed_cnt = fixed_cnt;
            }

            values1[0][j] = coupon;
        }
        floating_cnt = temp_floating_cnt;
        fixed_cnt = temp_fixed_cnt;
    }

    void adjustValues(int i, int size, DT t, xf_2D_array<DT, 4, LEN2>& values1, xf_2D_array<DT, 4, LEN2>& values2) {
#pragma HLS inline
        DT coupon;
        int temp_floating_cnt, temp_fixed_cnt;
        for (int j = 0; j < size; j++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 100 max = 100

            coupon = values2.read(0, j);
            if (floating_data[floating_cnt] == i && floating_cnt >= 0) {
                if (type == 0) {
                    if (coupon > nominal) coupon = nominal;
                } else {
                    if (coupon < nominal) coupon = nominal;
                }
                temp_floating_cnt = floating_cnt - 1;
            } else {
                temp_floating_cnt = floating_cnt;
            }

            if (fixed_data[fixed_cnt] == i && fixed_cnt >= 0) {
                coupon += fixedCoupon;
                temp_fixed_cnt = fixed_cnt - 1;
            } else {
                temp_fixed_cnt = fixed_cnt;
            }

            values1.write(coupon, 0, j);
        }
        floating_cnt = temp_floating_cnt;
        fixed_cnt = temp_fixed_cnt;
    }

}; // class Tree Callable Fixed Rate Bond Instrument

} // internal
} // fintech
} // xf
#endif
