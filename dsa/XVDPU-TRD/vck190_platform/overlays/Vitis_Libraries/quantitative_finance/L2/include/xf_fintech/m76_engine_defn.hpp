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

#ifndef _XF_FINTECH_M76_ENGINE_DEFN_H_
#define _XF_FINTECH_M76_ENGINE_DEFN_H_

namespace xf {
namespace fintech {

#define MAX_N (100)

/// @tparam   DT     Data Type used for this structure
/// @param  S      current stock price
/// @param  K      strike price
/// @param  r      risk free interest rate
/// @param  sigma  diffusion volatility
/// @param  T      time to maturity (years)
/// @param  kappa  mean of jump
/// @param  lambda jump intensity
/// @param  delta  standard deviation of log-normal jump process
template <typename DT>
struct jump_diffusion_params {
    // diffusion parameters
    DT S;
    DT K;
    DT r;
    DT sigma;
    DT T;

    // jump parameters
    DT lambda;
    DT kappa;
    DT delta;
    // int padding[6]; // #pragma data pack requires structure size to be a power of 2
};
}
}

#endif
