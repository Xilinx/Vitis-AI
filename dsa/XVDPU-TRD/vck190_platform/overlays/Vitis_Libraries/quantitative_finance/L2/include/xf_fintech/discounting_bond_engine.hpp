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
 * @file discounting_bond_engine.hpp
 *
 * @brief the file include the class DiscountingBondEngine.
 */
#ifndef _XF_FINTECH_DISCOUNT_BOND_ENGINE_H_
#define _XF_FINTECH_DISCOUNT_BOND_ENGINE_H_

#include <ap_fixed.h>
#include <ap_int.h>
#include "xf_fintech/linear_interpolation.hpp"

#include "hls_math.h"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {

namespace fintech {

/**
 * @brief DiscountingBondEngine Discounting Bond Engine
 *
 * @tparam DT data type supported include float and double.
 * @tparam LEN maximum length of array
 *
 */
template <typename DT, int LEN>
class DiscountingBondEngine {
   private:
    DT len;
    DT time[LEN];
    DT disc[LEN]; // ln(disc)

   public:
    /**
     * @brief default constructor
     */
    DiscountingBondEngine() {
#pragma HLS inline
#pragma HLS resource variable = time core = RAM_2P_LUTRAM
#pragma HLS resource variable = disc core = RAM_2P_LUTRAM
    }

    /**
     * @brief init initialize array and parameter
     *
     * @param size the actual size of array timeIn
     * @param timeIn array times, the difference between the maturity date and the reference date, unit is year.
     * @param discIn array logarithm of discount
     */
    void init(int size, DT* timeIn, DT* discIn) {
        len = size;
    loop_init:
        for (int i = 0; i < size; i++) {
#pragma HLS loop_tripcount max = 10 min = 10
#pragma HLS pipeline ii = 1
            time[i] = timeIn[i];
            disc[i] = discIn[i];
        }
    }

    /**
     * @brief calcuNPV calculate NPV function
     *
     * @param t the difference between the maturity date and the reference date, unit is year.
     * @param amount redemption price (face value)
     * @return return the NPV result
     */
    DT calcuNPV(DT t, DT amount) {
        DT discount = internal::linearInterpolation(t, len, time, disc);
#ifdef __SYNTHESIS__
        discount = hls::exp(discount);
#else
        discount = std::exp(discount);
#endif
        return amount * discount;
    }
};

} // fintech
} // xf
#endif
