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
 * @file  fdmmesher.hpp
 * @brief header file for mesher construction.
 *
 * This file is part of XF Fintech 1.0 Library.
 */

#ifndef XF_FINTECH_FDMMESHER_HPP
#define XF_FINTECH_FDMMESHER_HPP

#include "ap_int.h"
#include "xf_fintech/ornstein_uhlenbeck_process.hpp"

namespace xf {
namespace fintech {

/**
 * @brief One-dimensional simple FDM mesher object working on an index.
 *
 * @tparam DT The data type, which decides the precision of the mesher, and the default data type is double.
 * @tparam _sizeMax The maximum number of the coordinates supported in the implementation
 *
 */
template <typename DT, unsigned int _sizeMax>
class Fdm1dMesher {
   public:
    /**
         * @brief constructor
         */
    Fdm1dMesher() {
#pragma HLS inline
    }

    /**
     * @brief Calulate the mesher using Ornstein-Uhlenbeck process.
     * The implementation is modified for minimum resource utilizations.
     *
     * @param process The initialized Ornstein-Uhlenbeck process.
     * @param maturity The maturity of the swaption in years.
     * @param eps The default Epsilon should be 1.0e-5.
     * @param _locations The result of coordinates.
     * @param size The actual size of the coordinates.
     *
     */
    void init(
        // inputs
        const OrnsteinUhlenbeckProcess<DT>& process,
        DT maturity,
        DT eps,
        unsigned int size,
        // ouput
        DT _locations[_sizeMax]) {
        DT x0 = process._x0;
        DT p = eps;

        DT deps;
#pragma HLS resource variable = deps core = DMul_meddsp
        deps = 2 * p;

        DT numerator;
#pragma HLS resource variable = numerator core = DAddSub_nodsp
        numerator = 1 - deps;

        DT denominator;
#pragma HLS resource variable = denominator core = DAddSub_nodsp
        denominator = size - 1;

        DT dp = numerator / denominator;

    LOOP_LOCATIONS:
        for (unsigned int i = 0; i < size; i++) {
#pragma HLS loop_tripcount min = 100 max = 100 avg = 100
#pragma HLS pipeline
            if (0 == i) {
                DT evoTmp1 = process.evolve(/*0, x0,*/ maturity, inverseCumulativeNormalAcklamAreaOpt<DT>(eps));
                _locations[i] = x0 < evoTmp1 ? x0 : evoTmp1;
            } else if (size - 1 == i) {
                DT evoTmp2 = process.evolve(/*0, x0,*/ maturity, inverseCumulativeNormalAcklamAreaOpt<DT>(1 - eps));
                _locations[i] = x0 > evoTmp2 ? x0 : evoTmp2;
            } else {
#pragma HLS resource variable = p core = DAddSub_nodsp
                p += dp;
                _locations[i] = process.evolve(/*0, x0,*/ maturity, inverseCumulativeNormalAcklamAreaOpt<DT>(p));
            }
        }
    }

    /**
     * @brief Inverse CumulativeNormal using Acklam's approximation to transform uniform random number to normal random
     * number.
     * As this process will only be executed for once in the pricing engine, so it is optimized for minimum resource
     * utilization while having a reasonable latency.
     *
     * Reference: Acklam's approximation: by Peter J. Acklam, University of Oslo, Statistics Division.
     *
     * @tparam mType data type.
     * @param input input uniform random number
     * @return normal random number
     */
    template <typename mType>
    mType inverseCumulativeNormalAcklamAreaOpt(mType input) {
        const mType a1 = -3.969683028665376e+01;
        const mType a2 = 2.209460984245205e+02;
        const mType a3 = -2.759285104469687e+02;
        const mType a4 = 1.383577518672690e+02;
        const mType a5 = -3.066479806614716e+01;
        const mType a6 = 2.506628277459239e+00;
        const mType b1 = -5.447609879822406e+01;
        const mType b2 = 1.615858368580409e+02;
        const mType b3 = -1.556989798598866e+02;
        const mType b4 = 6.680131188771972e+01;
        const mType b5 = -1.328068155288572e+01;
        const mType c1 = -7.784894002430293e-03;
        const mType c2 = -3.223964580411365e-01;
        const mType c3 = -2.400758277161838e+00;
        const mType c4 = -2.549732539343734e+00;
        const mType c5 = 4.374664141464968e+00;
        const mType c6 = 2.938163982698783e+00;
        const mType d1 = 7.784695709041462e-03;
        const mType d2 = 3.224671290700398e-01;
        const mType d3 = 2.445134137142996e+00;
        const mType d4 = 3.754408661907416e+00;
        const mType x_low = 0.02425;
        const mType x_high = 1.0 - x_low;

        mType standard_value, z, r, f1, f1_1, f2, tmp;
        mType p1, p2, p3, p4, p5, p6;
        mType q1, q2, q3, q4, q5;
        ap_uint<1> not_tail;
        ap_uint<1> upper_tail;

        mType t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;
        mType t11, t12, t13, t14, t15, t16, t17, t18, t19;

#pragma HLS allocation operation instances = dmul limit = 1

        if (input < x_low || x_high < input) {
            if (input < x_low) {
                tmp = input;
                upper_tail = 0;
            } else {
#pragma HLS resource variable = tmp core = DAddSub_nodsp
                tmp = mType(1.0) - input;
                upper_tail = 1;
            }
#pragma HLS resource variable = t1 core = DLog_meddsp
#ifndef __SYNTHESIS__
            t1 = std::log(tmp);
#else
            t1 = hls::log(tmp);
#endif
#pragma HLS resource variable = t2 core = DMul_meddsp
            t2 = t1 * 2;
            t3 = -t2;
#ifndef __SYNTHESIS__
            z = std::sqrt(t3);
#else
            z = hls::sqrt(t3);
#endif
            r = z;
            p1 = c1;
            p2 = c2;
            p3 = c3;
            p4 = c4;
            p5 = c5;
            p6 = c6;
            q1 = d1;
            q2 = d2;
            q3 = d3;
            q4 = d4;
            not_tail = 0;
        } else {
#pragma HLS resource variable = z core = DAddSub_nodsp
            z = input - mType(0.5);
#pragma HLS resource variable = r core = DMul_meddsp
            r = z * z;
            p1 = a1;
            p2 = a2;
            p3 = a3;
            p4 = a4;
            p5 = a5;
            p6 = a6;
            q1 = b1;
            q2 = b2;
            q3 = b3;
            q4 = b4;
            q5 = b5;
            not_tail = 1;
        }

#pragma HLS resource variable = t4 core = DMul_meddsp
        t4 = p1 * r;
#pragma HLS resource variable = t5 core = DAddSub_nodsp
        t5 = t4 + p2;
#pragma HLS resource variable = t6 core = DMul_meddsp
        t6 = t5 * r;
#pragma HLS resource variable = t7 core = DAddSub_nodsp
        t7 = t6 + p3;
#pragma HLS resource variable = t8 core = DMul_meddsp
        t8 = t7 * r;
#pragma HLS resource variable = t9 core = DAddSub_nodsp
        t9 = t8 + p4;
#pragma HLS resource variable = t10 core = DMul_meddsp
        t10 = t9 * r;
#pragma HLS resource variable = t11 core = DAddSub_nodsp
        t11 = t10 + p5;
#pragma HLS resource variable = t12 core = DMul_meddsp
        t12 = t11 * r;
#pragma HLS resource variable = f1_1 core = DAddSub_nodsp
        f1_1 = t12 + p6;
        if (not_tail) {
#pragma HLS resource variable = f1 core = DMul_meddsp
            f1 = f1_1 * z;
        } else {
            f1 = f1_1;
        }

#pragma HLS resource variable = t13 core = DMul_meddsp
        t13 = q1 * r;
#pragma HLS resource variable = t14 core = DAddSub_nodsp
        t14 = t13 + q2;
#pragma HLS resource variable = t15 core = DMul_meddsp
        t15 = t14 * r;
#pragma HLS resource variable = t16 core = DAddSub_nodsp
        t16 = t15 + q3;
#pragma HLS resource variable = t17 core = DMul_meddsp
        t17 = t16 * r;
#pragma HLS resource variable = f2 core = DAddSub_nodsp
        f2 = t17 + q4;
        if (not_tail) {
#pragma HLS resource variable = t18 core = DMul_meddsp
            t18 = f2 * r;
#pragma HLS resource variable = f2 core = DAddSub_nodsp
            f2 = t18 + q5;
        }
#pragma HLS resource variable = t19 core = DMul_meddsp
        t19 = f2 * r;
#pragma HLS resource variable = f2 core = DAddSub_nodsp
        f2 = t19 + 1;

        standard_value = f1 / f2;
        if ((!not_tail) && (upper_tail)) {
            standard_value = -standard_value;
        }
        return standard_value;
    }
};

} // fintech
} // xf

#endif // XF_FINTECH_FDMMESHER_H
