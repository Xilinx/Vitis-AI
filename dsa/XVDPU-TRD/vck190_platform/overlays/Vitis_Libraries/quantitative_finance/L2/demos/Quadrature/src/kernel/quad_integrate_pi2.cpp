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
TEST_DT pi2Integrand(TEST_DT w, struct hcfEngineInputDataType* in);
#define MAX_ITERATIONS 10000
#define MAX_DEPTH 20
} // namespace internal
} // namespace fintech
} // namespace xf
#define XF_INTEGRAND_FN internal::pi2Integrand
#define XF_USER_DATA_TYPE struct hcfEngineInputDataType
#include "xf_fintech/quadrature.hpp"

namespace xf {
namespace fintech {
namespace internal {

extern struct complex_num<TEST_DT> charFunc(struct hcfEngineInputDataType* in, struct complex_num<TEST_DT> w);

/// @brief function to calculate the integrand for pi 2
/// @param[in] in A structure containing the kerenl model parameters
/// @param[in] w the variable of integration
/// @return the calculated integrand value
TEST_DT pi2Integrand(TEST_DT w, struct hcfEngineInputDataType* in) {
    struct complex_num<TEST_DT> cf1 = charFunc(in, cn_init(w, (TEST_DT)0));

    struct complex_num<TEST_DT> tmp = cn_div(cf1, cn_scalar_mul(cn_init((TEST_DT)0, (TEST_DT)1), w));
    return cn_real(cn_mul(cn_exp(cn_scalar_mul(cn_init((TEST_DT)0, (TEST_DT)-1), w * LOG(in->k))), tmp));
}

/// @brief integration function pi 2
/// @param[in] in A structure containing the kerenl model parameters
/// @return the calculated value
TEST_DT integrateForPi2(struct hcfEngineInputDataType* in) {
    TEST_DT res = 0;
    (void)xf::fintech::romberg_integrate((TEST_DT)1e-10, (TEST_DT)200, in->tol, (TEST_DT*)&res, in);
    return res;
}

} // namespace internal
} // namespace fintech
} // namespace xf
