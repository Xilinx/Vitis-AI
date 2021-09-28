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

#include <cmath>
#include "m76_host.hpp"

static TEST_DT norm_pdf(const TEST_DT& x) {
    return (1.0 / (pow(2 * M_PI, 0.5))) * exp(-0.5 * x * x);
}

static TEST_DT norm_cdf(const TEST_DT& x) {
    TEST_DT k = 1.0 / (1.0 + 0.2316419 * x);
    TEST_DT k_sum = k * (0.319381530 + k * (-0.356563782 + k * (1.781477937 + k * (-1.821255978 + 1.330274429 * k))));

    if (x >= 0.0) {
        return (1.0 - (1.0 / (pow(2 * M_PI, 0.5))) * exp(-0.5 * x * x) * k_sum);
    } else {
        return 1.0 - norm_cdf(-x);
    }
}

static TEST_DT d_j(
    const int& j, const TEST_DT& S, const TEST_DT& K, const TEST_DT& r, const TEST_DT& v, const TEST_DT& T) {
    return (log(S / K) + (r + (pow(-1, j - 1)) * 0.5 * v * v) * T) / (v * (pow(T, 0.5)));
}

static TEST_DT bs_call_price(const TEST_DT& S, const TEST_DT& K, const TEST_DT& r, const TEST_DT& v, const TEST_DT& T) {
    return S * norm_cdf(d_j(1, S, K, r, v, T)) - K * exp(-r * T) * norm_cdf(d_j(2, S, K, r, v, T));
}

// https://www.quantstart.com/articles/Jump-Diffusion-Models-for-European-Options-Pricing-in-C
// Calculate the Merton jump-diffusion price based on
// a finite sum approximation to the infinite series
// solution, making use of the BS call price.
static TEST_DT bs_jd_call_price(const TEST_DT S,
                                const TEST_DT K,
                                const TEST_DT r,
                                const TEST_DT sigma,
                                const TEST_DT T,
                                const TEST_DT kappa,
                                const TEST_DT lambda,
                                const TEST_DT nu) {
    TEST_DT price = 0.0; // Stores the final call price
    TEST_DT factorial = 1.0;

    // Pre-calculate as much as possible
    TEST_DT lambda_p = lambda * (kappa + 1);
    TEST_DT lambda_p_T = lambda_p * T;

    // Calculate the finite sum over N terms
    for (int n = 0; n < 100; n++) {
        TEST_DT sigma_n = sqrt(sigma * sigma + n * nu * nu / T);
        TEST_DT r_n = r - lambda * (kappa) + n * log(kappa + 1) / T;

        // Calculate n!
        if (n > 0) {
            factorial *= (lambda_p_T / n);
        }

        // Refine the jump price over the loop
        price += exp(-lambda_p_T) * factorial * bs_call_price(S, K, r_n, sigma_n, T);
    }
    return price;
}

void cpu_merton_jump_diffusion(std::vector<struct xf::fintech::jump_diffusion_params<TEST_DT>,
                                           aligned_allocator<struct xf::fintech::jump_diffusion_params<TEST_DT> > >& p,
                               int num_tests,
                               std::vector<TEST_DT, aligned_allocator<TEST_DT> >& res) {
    for (int i = 0; i < num_tests; i++) {
        res.at(i) = bs_jd_call_price(p.at(i).S, p.at(i).K, p.at(i).r, p.at(i).sigma, p.at(i).T, p.at(i).kappa,
                                     p.at(i).lambda, p.at(i).delta);
    }
}
