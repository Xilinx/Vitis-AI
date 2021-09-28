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
 * @file ecir_model.hpp
 * @brief This file include the ECIRModel
 *
 */

#ifndef __XF_FINTECH_ECIRMODEL_HPP_
#define __XF_FINTECH_ECIRMODEL_HPP_

#include "hls_math.h"
#include "ap_int.h"

#ifndef __SYNTHESIS__
#include "iostream"
using namespace std;
#endif

namespace xf {

namespace fintech {

/**
 * @brief Extended Cox-Ingersoll-Ross  model
 *
 * @tparam DT data type supported include float and double
 * @tparam Tree class TrinomialTree
 * @tparam LEN2 maximum length of node of tree, which affects the latency and resources utilization.
 *
 */
template <typename DT, typename Tree, int LEN2>
class ECIRModel {
   private:
    // spreads on interest rates
    DT spread_;
    DT rate_;

    // internal process
    DT iterRate(int size, DT* statePrices, DT discountBond, DT x0, DT dx, DT dt, DT r) {
#pragma HLS inline
        DT values16[16] = {0.0};
    loop_calcuRate:
        for (int j = 0; j < size; j++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 100 max = 100
            DT x_now = x0 + dx * j;
            values16[j % 16] -= statePrices[j] * discount(0.0, dt, &x_now, r);
        }

        DT values8[8];
    loop_add8:
        for (int k = 0; k < 8; k++) {
#pragma HLS unroll factor = 8
            values8[k] = values16[k] + values16[k + 8];
        }

        DT values4[4];
    loop_add4:
        for (int i = 0; i < 4; i++) {
#pragma HLS unroll factor = 4
            values4[i] = values8[i] + values8[i + 4];
        }

        DT values2_0 = values4[0] + values4[2];
        DT values2_1 = values4[1] + values4[3];
        DT value = values2_0 + values2_1;
        DT rate = discountBond + value;
        return rate;
    }

    // internal process
    void initRate(int i,
                  int size,
                  DT* values3x16,
                  DT tmp_values1[3][LEN2],
                  DT tmp_values2[3][LEN2],
                  DT* statePrices,
                  DT discountBond,
                  DT x0,
                  DT dx,
                  DT dt,
                  DT r[3]) {
#pragma HLS inline
    loop_initRate:
        for (int j = 0; j < size; j++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 100 max = 100
            DT x_now = x0 + dx * j;
            DT state_price;
            if (i % 2) {
                state_price = tmp_values2[0][j] + tmp_values2[1][j] + tmp_values2[2][j];
                tmp_values1[0][j] = 0.0;
                tmp_values1[1][j] = 0.0;
                tmp_values1[2][j] = 0.0;
            } else {
                state_price = tmp_values1[0][j] + tmp_values1[1][j] + tmp_values1[2][j];
                tmp_values2[0][j] = 0.0;
                tmp_values2[1][j] = 0.0;
                tmp_values2[2][j] = 0.0;
            }
            statePrices[j] = state_price;
            for (int k = 0; k < 3; k++) {
#pragma HLS unroll factor = 3
                values3x16[k * 16 + j % 16] -= state_price * discount(0.0, dt, &x_now, r[k]);
            }
        }

        DT values8[3][8];
        DT values4[3][4];
        DT values2_0[3], values2_1[3];
        for (int k = 0; k < 3; k++) {
#pragma HLS unroll factor = 3
        // r[k] = addArray16To1<DT>(values16[k]);
        loop_add8:
            for (int i = 0; i < 8; i++) {
#pragma HLS unroll factor = 8
                values8[k][i] = values3x16[k * 16 + i] + values3x16[k * 16 + i + 8];
            }

        loop_add4:
            for (int i = 0; i < 4; i++) {
#pragma HLS unroll factor = 4
                values4[k][i] = values8[k][i] + values8[k][i + 4];
            }

            values2_0[k] = values4[k][0] + values4[k][2];
            values2_1[k] = values4[k][1] + values4[k][3];
            r[k] = values2_0[k] + values2_1[k] + discountBond;
        }
    }

   public:
    /**
     * @brief constructor
     */
    ECIRModel() {
#pragma HLS inline
    }

    /**
     * @brief initialize parameters
     *
     * @param r floating benchmark annual interest rate
     * @param spread spreads on interest rates
     */
    void initialization(DT r, DT spread) {
#pragma HLS inline
        spread_ = spread;
        rate_ = r;
    }

    /**
     * @brief calculate the discount after time dt
     *
     * @param t the current timepoint
     * @param dt the difference between the next timepoint and the current timepoint
     * @param x underlying
     * @param r shortrate
     * @return discount
     */
    DT discount(DT t, DT dt, DT* x, DT r) {
#pragma HLS inline
        // DT rate = (hls::exp(r + *x)+spread_)*dt;
        DT rate = ((*x) * (*x) + spread_ + r) * dt;
        return hls::exp(-rate);
    }

    /**
     * @brief calcutate short-rate of dt at t for TreeEngine
     *
     * @param tree class TrinomialTree
     * @param endCnt end counter of timepoints
     * @param time array timepoints
     * @param dtime array the difference between the next timepoint and the current timepoint
     * @param tmp_values1 process values
     * @param tmp_values2 process values
     * @param statePrices state prices
     * @param rates array short-rates
     */
    void treeShortRate(Tree& tree,
                       int endCnt,
                       DT* time,
                       DT* dtime,
                       DT tmp_values1[3][LEN2],
                       DT tmp_values2[3][LEN2],
                       DT* statePrices,
                       DT* rates) {
#pragma HLS inline
        int size = 0;
        DT probs[3];
    loop_compute_init:
        for (int j = 0; j < endCnt * 2; j++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 100 max = 100
            tmp_values1[0][j] = 0.0;
            tmp_values1[1][j] = 0.0;
            tmp_values1[2][j] = 0.0;
            tmp_values2[0][j] = 0.0;
            tmp_values2[1][j] = 0.0;
            tmp_values2[2][j] = 0.0;
        }

        DT values16[48];
        DT rate_last = 1.0;
#pragma HLS array_partition variable = values16 block factor = 3 dim = 1
#pragma HLS resource variable = values16 core = RAM_2P_LUTRAM
    loop_init_values16:
        for (int j = 0; j < 16; j++) {
#pragma HLS pipeline
            values16[j] = 0.0;
            values16[16 + j] = 0.0;
            values16[32 + j] = 0.0;
        }

        DT t, dt;
    loop_compute_LEN:
        for (int i = 0; i < endCnt - 1; i++) {
#pragma HLS loop_tripcount min = 50 max = 50
            if (size == 0) {
                tmp_values1[0][0] = 1.0;
            } else {
                // compute state prices
                DT tmp1 = 0.0, tmp2 = 0.0, tmp3 = 0.0;
                int index_d = -1;
                int flag = 0;
            loop_compute_branch:
                for (int j = 0; j < size; j++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 100 max = 100
#pragma HLS dependence variable = tmp_values1 inter false
#pragma HLS dependence variable = tmp_values2 inter false
                    values16[j % 16] = 0.0;
                    values16[16 + j % 16] = 0.0;
                    values16[32 + j % 16] = 0.0;
                    int index = tree.calculateProbability(j, t, dt, probs);
                    DT x = tree.underlying(j);
                    DT state_price_disc = discount(t, dt, &x, rate_last) * statePrices[j];
                    DT price_tmp1 = state_price_disc * probs[0];
                    DT price_tmp2 = state_price_disc * probs[1];
                    DT price_tmp3 = state_price_disc * probs[2];
                    if (flag == 0) {
                        flag = 1;
                    } else if (index == index_d) {
                        if (i % 2) {
                            tmp_values2[0][index_d] += tmp1 + price_tmp1;
                            tmp_values2[1][index_d + 1] += tmp2 + price_tmp2;
                            tmp_values2[2][index_d + 2] += tmp3 + price_tmp3;
                        } else {
                            tmp_values1[0][index_d] += tmp1 + price_tmp1;
                            tmp_values1[1][index_d + 1] += tmp2 + price_tmp2;
                            tmp_values1[2][index_d + 2] += tmp3 + price_tmp3;
                        }
                        flag = 0;
                    } else {
                        if (i % 2) {
                            tmp_values2[0][index_d] += tmp1 + 0.0;
                            tmp_values2[1][index_d + 1] += tmp2 + 0.0;
                            tmp_values2[2][index_d + 2] += tmp3 + 0.0;
                        } else {
                            tmp_values1[0][index_d] += tmp1 + 0.0;
                            tmp_values1[1][index_d + 1] += tmp2 + 0.0;
                            tmp_values1[2][index_d + 2] += tmp3 + 0.0;
                        }
                    }
                    index_d = index;
                    tmp1 = price_tmp1;
                    tmp2 = price_tmp2;
                    tmp3 = price_tmp3;
                }
                if (flag != 0) {
                    if (i % 2) {
                        tmp_values2[0][index_d] += tmp1;
                        tmp_values2[1][index_d + 1] += tmp2;
                        tmp_values2[2][index_d + 2] += tmp3;
                    } else {
                        tmp_values1[0][index_d] += tmp1;
                        tmp_values1[1][index_d + 1] += tmp2;
                        tmp_values1[2][index_d + 2] += tmp3;
                    }
                }
            }

            // update
            t = time[i];
            dt = dtime[i];
            tree.dxUpdate(i, t, dt);
            size = tree.size(i);
            DT x = tree.underlying(0);
            DT dx = tree.getDx(i);

            DT xMax = 100.0, xMin = -100.0;
            // DT xMax=50.0,xMin=-50.0;
            DT froot, fxMax, fxMin;
            DT root;
            if (i == 0)
                root = 1;
            else
                root = rates[i - 1];
            DT discountBond = 1 / hls::exp(rate_ * (t + dt));
            DT rate3[3] = {root, xMax, xMin};
            initRate(i, size, values16, tmp_values1, tmp_values2, statePrices, discountBond, x, dx, dt, rate3);
            froot = rate3[0];
            fxMax = rate3[1];
            fxMin = rate3[2];
            if (froot > 0) {
                xMax = xMin;
                fxMax = fxMin;
            } else {
                xMin = xMax;
                fxMin = fxMax;
            }

            DT d, e;
            if ((froot > 0.0 && fxMax > 0.0) || (froot < 0.0 && fxMax < 0.0)) {
                xMax = xMin;
                fxMax = fxMin;
                d = root - xMin;
                e = root - xMin;
            } else {
                d = root - xMax;
                e = root - xMax;
            }

            if (hls::fabs(fxMax) < hls::fabs(froot)) {
                xMin = root;
                root = xMax;
                xMax = xMin;
                fxMin = froot;
                froot = fxMax;
                fxMax = fxMin;
            }
            DT xAccl = 4.4408920985006262e-16 * hls::fabs(root) + 0.5e-7;
            DT xMid = (xMax - root) / 2.0;
            int ii = 0;
        loop_rateModel:
            for (; hls::fabs(xMid) > xAccl;) { // for(int k=begin;k<=1000;k++){
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 10 max = 10
                if (hls::fabs(e) >= xAccl && hls::fabs(fxMin) > hls::fabs(froot)) {
                    DT s = froot / fxMin;
                    DT p, q, r;
                    if (xMin == xMax) {
                        p = 2.0 * xMid * s;
                        q = 1.0 - s;
                    } else {
                        q = fxMin / fxMax;
                        r = froot / fxMax;
                        p = s * (2.0 * xMid * q * (q - r) - (root - xMin) * (r - 1.0));
                        q = (q - 1.0) * (r - 1.0) * (s - 1.0);
                    }

                    if (p > 0.0) q = -q;
                    p = hls::fabs(p);
                    DT min1 = 3.0 * xMid * q - hls::fabs(xAccl * q);
                    DT min2 = hls::fabs(e * q);
                    if (2.0 * p < (min1 < min2 ? min1 : min2)) {
                        e = d;
                        d = p / q;
                    } else {
                        d = xMid;
                        e = d;
                    }
                } else {
                    d = xMid;
                    e = d;
                }

                xMin = root;
                fxMin = froot;
                if (hls::fabs(d) > xAccl)
                    root += d;
                else if (xMid >= 0.0)
                    root += hls::fabs(xAccl);
                else
                    root -= std::fabs(xAccl);
                froot = iterRate(size, statePrices, discountBond, x, dx, dt, root);

                if ((froot > 0.0 && fxMax > 0.0) || (froot < 0.0 && fxMax < 0.0)) {
                    xMax = xMin;
                    fxMax = fxMin;
                    d = root - xMin;
                    e = d;
                }

                if (hls::fabs(fxMax) < hls::fabs(froot)) {
                    xMin = root;
                    root = xMax;
                    xMax = xMin;
                    fxMin = froot;
                    froot = fxMax;
                    fxMax = fxMin;
                }
                xAccl = 4.4408920985006262e-16 * hls::fabs(root) + 0.5e-7;
                xMid = (xMax - root) / 2.0;
            }
            rate_last = root;
            rates[i] = root;
        }
    }

}; // class
} // fintech
} // xf

#endif
