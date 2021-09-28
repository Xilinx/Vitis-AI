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

#include "xf_fintech/L2_utils.hpp"
#include "quad_hcf_engine_def.hpp"

namespace xf {
namespace fintech {
namespace internal {

/// @brief function to calculate the characteristic function
/// @param[in] in A structure containing the kerenl model parameters
/// @param[in] w complex representation of w
/// @return the calculated characterisic function value
struct complex_num<TEST_DT> charFunc(struct hcfEngineInputDataType* in, struct complex_num<TEST_DT> w) {
    TEST_DT vv = in->vvol * in->vvol;
    TEST_DT gamma = vv / 2;
    struct complex_num<TEST_DT> i = cn_init((TEST_DT)0, (TEST_DT)1);

    struct complex_num<TEST_DT> alpha = cn_scalar_mul(cn_add(cn_mul(w, w), cn_mul(w, i)), (TEST_DT)-0.5);
    struct complex_num<TEST_DT> beta =
        cn_sub(cn_init(in->kappa, (TEST_DT)0), cn_mul(cn_scalar_mul(w, in->rho * in->vvol), i));
    struct complex_num<TEST_DT> h = cn_sqrt(cn_sub(cn_mul(beta, beta), (cn_scalar_mul(alpha, gamma*(TEST_DT)4))));
    struct complex_num<TEST_DT> r_plus = cn_div(cn_add(beta, h), cn_init(vv, (TEST_DT)0));
    struct complex_num<TEST_DT> r_minus = cn_div(cn_sub(beta, h), cn_init(vv, (TEST_DT)0));
    struct complex_num<TEST_DT> g = cn_div(r_minus, r_plus);

    struct complex_num<TEST_DT> exp_hT = cn_exp(cn_scalar_mul(h, -in->t));
    struct complex_num<TEST_DT> tmp = cn_sub(cn_init((TEST_DT)1, (TEST_DT)0), cn_mul(g, exp_hT));
    struct complex_num<TEST_DT> D = cn_mul(r_minus, cn_div(cn_sub(cn_init((TEST_DT)1, (TEST_DT)0), exp_hT), tmp));

    struct complex_num<TEST_DT> C = cn_div(tmp, cn_sub(cn_init((TEST_DT)1, (TEST_DT)0), g));
    C = cn_scalar_mul(cn_ln(C), (TEST_DT)2);
    C = cn_div(C, cn_init(vv, (TEST_DT)0));
    C = cn_mul(cn_init(in->kappa, (TEST_DT)0), cn_sub(cn_scalar_mul(r_minus, in->t), C));

    struct complex_num<TEST_DT> cf = cn_add(cn_scalar_mul(C, in->vbar), cn_scalar_mul(D, in->v));
    cf = cn_add(cf, cn_scalar_mul(cn_mul(i, w), LOG(in->s* EXP(in->r * in->t))));
    cf = cn_exp(cf);

    return cf;
}

} // namespace internal

#define PI (3.14159265359f)
/// @brief Engine for Hestion Closed Form Solution
/// @param[in] input_data A structure containing the kerenl model parameters
/// @return the calculated call value
TEST_DT hcfEngine(struct hcfEngineInputDataType* input_data) {
    TEST_DT pi1 = 0.5 + ((1 / PI) * internal::integrateForPi1(input_data));
    TEST_DT pi2 = 0.5 + ((1 / PI) * internal::integrateForPi2(input_data));
    return (input_data->s * pi1) - (internal::EXP(-(input_data->r * input_data->t)) * input_data->k * pi2);
}
#undef PI

} // namespace fintech
} // namespace xf
