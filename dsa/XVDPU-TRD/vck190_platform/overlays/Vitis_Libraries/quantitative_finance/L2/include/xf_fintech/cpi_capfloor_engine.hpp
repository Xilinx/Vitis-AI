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
 * @file cpi_capfloor_engine.hpp
 *
 *
 * @brief the file include the class CPICapFloorEngine that is CPI Cap/Floor Engine.
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
 * @brief CPICapFloorEngine CPI Cap/Floor Engine
 *
 * @tparam DT data type supported include float and double.
 * @tparam LEN maximum length of array
 *
 */
template <typename DT, int LEN>
class CPICapFloorEngine {
   private:
    int xLen;
    int yLen;
    DT time[LEN];
    DT rate[LEN];
    DT price[LEN * LEN];

   public:
    /**
     * @brief default constructor
     */
    CPICapFloorEngine() {
#pragma HLS inline
#pragma HLS resource variable = time core = RAM_2P_LUTRAM
#pragma HLS resource variable = rate core = RAM_2P_LUTRAM
#pragma HLS resource variable = price core = RAM_2P_LUTRAM
    }

    /**
     * @brief init initialize arrays and parameters
     *
     * @param xSize the actual size of array timeIn
     * @param ySize the actual size of array rateIn
     * @param timeIn array time, the difference between the maturity date and the reference date, unit is year.
     * @param rateIn array rate
     * @param priceIn array price
     */
    void init(int xSize, int ySize, DT* timeIn, DT* rateIn, DT* priceIn) {
        xLen = xSize;
        yLen = ySize;
        for (int i = 0; i < xSize; i++) {
#pragma HLS loop_tripcount max = 10 min = 10
#pragma HLS pipeline ii = 1
            time[i] = timeIn[i];
        }
        for (int i = 0; i < ySize; i++) {
#pragma HLS loop_tripcount max = 10 min = 10
#pragma HLS pipeline ii = 1
            rate[i] = rateIn[i];
        }

        for (int i = 0; i < ySize * xSize; i++) {
#pragma HLS loop_tripcount max = 100 min = 100
#pragma HLS pipeline ii = 1
            price[i] = priceIn[i];
        }
    }

    /**
     * @brief calcuNPV calculate NPV function
     *
     * @param t the difference between the maturity date and the reference date, unit is year.
     * @param r the strike rate
     * @return return the NPV result
     */
    DT calcuNPV(DT t, DT r) {
        DT cfPrice = internal::linearInterpolation2D<DT>(t, r, xLen, yLen, time, rate, price);
        return cfPrice;
    }
};

} // fintech
} // xf
#endif
