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

#ifndef _HJM_MODEL_HPP_
#define _HJM_MODEL_HPP_

#include "ap_int.h"

namespace xf {
namespace fintech {

/**
 * @brief Defines the Heath-Jarrow-Morton multi-factor model specific parameters.
 */
struct hjmModelParams {
    // Number of factors in HJM Model
    constexpr static unsigned N = 3;
    // Degrees of fitted volatilities to PCA results, the degree includes the 0th power (constant term)
    constexpr static unsigned int PD[N] = {1, 4, 4};
    // dt is the simulation step time, in years.
    constexpr static float dt = 0.01f;
    constexpr static float dtInv = 1 / dt;
    constexpr static float sqrtDt = 0.1f;
    // tau is difference between tenors, in years.
    constexpr static float tau = 0.5f;
    // How many copies of polyfitted coefficients we need
    constexpr static unsigned VOL_POLYFIT_FANOUT = 2;
};

/**
 * @brief Defines the constant terms of the Heath-Jarrow-Morton framework stochastic differential equation.
 * The data in this structure is meant to be set once and reused in every path generator
 */
template <typename DT, unsigned int MAX_TENORS>
struct hjmModelData : public hjmModelParams {
    DT vol1[MAX_TENORS];
    DT vol2[MAX_TENORS];
    DT vol3[MAX_TENORS];
    DT RnD[MAX_TENORS];
    DT m_initialFc[MAX_TENORS];

    ap_uint<16> tenors;

    hjmModelData() {}

    inline void initialise(const ap_uint<16> nTenors, DT* v1, DT* v2, DT* v3, DT* drift, DT* currentFc) {
#pragma HLS INLINE
        tenors = nTenors;

        for (unsigned i = 0; i < nTenors; i++) {
#pragma HLS PIPELINE
            vol1[i] = v1[i];
            vol2[i] = v2[i];
            vol3[i] = v3[i];
            RnD[i] = drift[i];
            m_initialFc[i] = currentFc[i];
        }
    }
};
}
}

#endif // _HJM_MODEL_HPP_