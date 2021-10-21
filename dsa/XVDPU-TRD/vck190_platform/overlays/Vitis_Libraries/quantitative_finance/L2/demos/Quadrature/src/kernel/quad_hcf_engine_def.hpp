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

#ifndef _XF_FINTECH_QUAD_HCF_ENGINE_DEF_H_
#define _XF_FINTECH_QUAD_HCF_ENGINE_DEF_H_

#include "xf_fintech/L2_utils.hpp"

#define TEST_DT float
/// @param s    the stock price at t=0
/// @param k     the strike price
/// @param v    the stock price variance at t=0
/// @param t     the expiration time
/// @param r     the risk free interest rate
/// @param rho   the correlation of the 2 Weiner processes
/// @param vvol  the volatility of volatility (sigma)
/// @param vbar  the long term average variance (theta)
/// @param kappa the rate of reversion
/// @param padding the pragma data pack requires structure size to be a power
/// of 2
struct hcfEngineInputDataType {
    TEST_DT s;
    TEST_DT k;
    TEST_DT v;
    TEST_DT t;
    TEST_DT r;
    TEST_DT rho;
    TEST_DT vvol;
    TEST_DT vbar;
    TEST_DT kappa;
    TEST_DT tol;
    int padding[6];
};

namespace xf {
namespace fintech {
TEST_DT hcfEngine(struct hcfEngineInputDataType* input_data);
namespace internal {
TEST_DT integrateForPi1(struct hcfEngineInputDataType* in);
TEST_DT integrateForPi2(struct hcfEngineInputDataType* in);
struct complex_num<TEST_DT> charFunc(struct hcfEngineInputDataType* in, struct complex_num<TEST_DT> w);
} // namespace internal
} // namespace fintech
} // namespace xf

#define MAX_NUMBER_TESTS 4096

#endif
