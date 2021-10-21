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

#include <complex>

#include "hcf.hpp"
#include "hcf_host.hpp"

const std::complex<TEST_DT> _i(0, 1);

std::complex<TEST_DT> charFunc(struct xf::fintech::hcfEngineInputDataType<TEST_DT>* p, std::complex<TEST_DT> w) {
    TEST_DT vv = p->vvol * p->vvol;
    TEST_DT gamma = vv / 2;
    std::complex<TEST_DT> alpha = (std::complex<TEST_DT>)-0.5 * ((w * w) + (w * _i));
    std::complex<TEST_DT> beta = p->kappa - (p->rho * p->vvol * w) * _i;
    std::complex<TEST_DT> h = sqrt((beta * beta) - ((std::complex<TEST_DT>)4 * alpha * gamma));
    std::complex<TEST_DT> r_plus = (beta + h) / vv;
    std::complex<TEST_DT> r_minus = (beta - h) / vv;
    std::complex<TEST_DT> g = r_minus / r_plus;
    std::complex<TEST_DT> D =
        r_minus * (((std::complex<TEST_DT>)1 - exp(-(h * p->T))) / ((std::complex<TEST_DT>)1 - (g * exp(-(h * p->T)))));

    std::complex<TEST_DT> C = (((std::complex<TEST_DT>)1 - (g * exp(-(h * p->T)))) / ((std::complex<TEST_DT>)1 - g));
    C = log(C) * (std::complex<TEST_DT>)2;
    C = C / vv;
    C = (r_minus * p->T) - C;
    C = C * p->kappa;

    std::complex<TEST_DT> cf = (C * p->vbar) + (D * p->v0);
    cf = cf + ((_i * w) * (std::complex<TEST_DT>)log(p->s0 * exp(p->r * p->T)));
    cf = exp(cf);
    return cf;
}

TEST_DT pi1Integrand(struct xf::fintech::hcfEngineInputDataType<TEST_DT>* p, TEST_DT w) {
    std::complex<TEST_DT> ww(w, -1);
    std::complex<TEST_DT> cf1 = charFunc(p, ww);

    ww = std::complex<TEST_DT>(0, -1);
    std::complex<TEST_DT> cf2 = charFunc(p, ww);

    std::complex<TEST_DT> tmp = ((_i * w) * cf2);
    return real(exp(_i * (w * (TEST_DT)-log(p->K))) * (cf1 / tmp));
}

TEST_DT pi2Integrand(struct xf::fintech::hcfEngineInputDataType<TEST_DT>* p, TEST_DT w) {
    std::complex<TEST_DT> ww = std::complex<TEST_DT>(w, 0);
    std::complex<TEST_DT> cf1 = charFunc(p, ww);

    return real(exp(_i * (w * (TEST_DT)-log(p->K))) * (cf1 / (_i * w)));
}

TEST_DT integrateForPi1(struct xf::fintech::hcfEngineInputDataType<TEST_DT>* p) {
    TEST_DT elem;
    TEST_DT sum = 0;
    TEST_DT f_n = 0;
    TEST_DT f_n_plus_1 = 0;
    TEST_DT w = 0;
    TEST_DT max = p->w_max / p->dw;
    int n;

    f_n = pi1Integrand(p, 1e-10);

    for (n = 1; n <= (int)max; n++) {
        w = n * p->dw;
        f_n_plus_1 = pi1Integrand(p, w);
        elem = p->dw * (f_n_plus_1 + f_n) / 2;
        sum += elem;
        f_n = f_n_plus_1;
    }
    return sum;
}

TEST_DT integrateForPi2(struct xf::fintech::hcfEngineInputDataType<TEST_DT>* p) {
    TEST_DT elem;
    TEST_DT sum = 0;
    TEST_DT f_n = 0;
    TEST_DT f_n_plus_1 = 0;
    TEST_DT w = 0;
    TEST_DT max = p->w_max / p->dw;
    int n;

    f_n = pi2Integrand(p, 1e-10);

    for (n = 1; n <= (int)max; n++) {
        w = n * p->dw;
        f_n_plus_1 = pi2Integrand(p, w);
        elem = p->dw * (f_n_plus_1 + f_n) / 2;
        sum += elem;
        f_n = f_n_plus_1;
    }
    return sum;
}

void call_price(std::vector<struct xf::fintech::hcfEngineInputDataType<TEST_DT>,
                            aligned_allocator<struct xf::fintech::hcfEngineInputDataType<TEST_DT> > >& p,
                int num_tests,
                TEST_DT* res) {
    for (int i = 0; i < num_tests; i++) {
        TEST_DT pi1 = 0.5 + ((1 / M_PI) * integrateForPi1(&p[i]));
        TEST_DT pi2 = 0.5 + ((1 / M_PI) * integrateForPi2(&p[i]));
        res[i] = (p[i].s0 * pi1) - (exp(-(p[i].r * p[i].T)) * p[i].K * pi2);
    }
}
