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

#ifndef _XF_FINTECH_M76_ENGINE_H_
#define _XF_FINTECH_M76_ENGINE_H_

#include "L2_utils.hpp"
#include "cf_bsm.hpp"
#include "m76_engine_defn.hpp"

namespace xf {
namespace fintech {
namespace internal {
/// @brief                  Summing loop for the individual BS solutions
/// @param[in]  in          An array of the individual solutions
/// @param[out] out         The sum
template <typename DT>
void sum(DT* out, DT* in) {
    DT tmp = 0;
sum_loop:
    for (int i = 0; i < MAX_N; i++) {
        tmp += in[i];
    }
    *out = tmp;
}

} // internal

/// @brief                  Engine for the Merton Jump Diffusion Model
/// @param[in]  p           A structure containing the jump diffusion parameters
/// @param[out] call_price  An array of BS solutions multiplied by the jump weighting
///                         Note that these must be subsequently summed to get the Jump Diffusion solution
template <typename DT>
void M76Engine(struct jump_diffusion_params<DT>* p, DT* call_price) {
    DT factorial = 1.0;
    DT X = 1;

N_loop:
    for (int n = 0; n < MAX_N; n++) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100 avg = 100
#pragma HLS PIPELINE II = 1 rewind
        DT lambda_primed_T = p->lambda * (p->kappa + 1) * p->T;
        DT sigma_n = xf::fintech::internal::SQRT((p->sigma * p->sigma) + (n * p->delta * p->delta / p->T));
        DT r_n = p->r - (p->lambda * p->kappa) + (n * xf::fintech::internal::LOG(p->kappa + 1) / p->T);

        if (n > 0) {
            X *= (lambda_primed_T / n);
        }

        DT bs_call;            // the Black Scholes call price
        unsigned int call = 1; // request a call price to cf_bsm
        // the greeks - not used here
        DT bs_delta;
        DT bs_gamma;
        DT bs_vega;
        DT bs_theta;
        DT bs_rho;
        cfBSMEngine<float>(p->S, sigma_n, r_n, p->T, p->K, 0, call, &bs_call, &bs_delta, &bs_gamma, &bs_vega, &bs_theta,
                           &bs_rho);
        *(call_price + n) = xf::fintech::internal::EXP(-lambda_primed_T) * X * bs_call;
    }
}

} // namespace fintech
} // namespace xf

#endif
