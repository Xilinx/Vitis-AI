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

#ifndef _XF_FINTECH_M76_HOST_H_
#define _XF_FINTECH_M76_HOST_H_

#include <vector>
#include "m76.hpp"
#include "xcl2.hpp"

struct parsed_params {
    // diffusion parameters
    double S;     // current stock price
    double K;     // strike price
    double r;     // interest rate
    double sigma; // volatility
    double T;     // time to vest (years)

    // jump parameters
    double lambda;         // mean jump per unit time
    double kappa;          // Expected[Y-1] Y is the random variable
    double delta;          // delta^2 = variance of ln(Y)
    int N;                 // number of terms in the finite sum approximation
    int validation;        // internal validation use
    double expected_value; // value to compare calculated call price with
    int line_number;
};

void cpu_merton_jump_diffusion(std::vector<struct xf::fintech::jump_diffusion_params<TEST_DT>,
                                           aligned_allocator<struct xf::fintech::jump_diffusion_params<TEST_DT> > >& p,
                               int num_tests,
                               std::vector<TEST_DT, aligned_allocator<TEST_DT> >& res);
std::vector<struct parsed_params*>* parse_file(std::string file);

#endif
