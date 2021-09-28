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
#include <iostream>
#include <cmath>

double my_sin(double x, void* p) {
    return sin(x);
}

#define XF_INTEGRAND_FN my_sin
#define XF_USER_DATA_TYPE void
#define MAX_DEPTH 20
#define MAX_ITERATIONS 10000
#include "quadrature.hpp"

int top_trap(double a, double b, double tol, double* res, int* iter) {
    return xf::fintech::trap_integrate(a, b, tol, res, NULL);
}

int top_simp(double a, double b, double tol, double* res, int* iter) {
    return xf::fintech::simp_integrate(a, b, tol, res, NULL);
}

int top_romberg(double a, double b, double tol, double* res, int* iter) {
    return xf::fintech::romberg_integrate(a, b, tol, res, NULL);
}

static int test_trapezoidal_rule(double a, double b, double tol, double exp) {
    double res = 0;
    int iter = 0;

    std::cout << "Integrating sin(x) from " << a << " to " << b;
    std::cout << " using adaptive trapezoidal rule" << std::endl;

    if (!top_trap(a, b, tol, &res, &iter)) {
        std::cout << "FAIL: integration function failed to calculate result" << std::endl;
        return 0;
    }
    double diff = std::fabs(res - exp);
    if (diff > tol) {
        std::cout << "FAIL: expected " << exp << " got " << res << " diff " << diff << std::endl;
        std::cout << "Iterations = " << iter << std::endl;
        return 0;
    }
    std::cout << "PASS: result = " << res << std::endl;
    std::cout << "Iterations = " << iter << std::endl;
    return 1;
}

static int test_simpson_rule(double a, double b, double tol, double exp) {
    double res = 0;
    int iter = 0;

    std::cout << "Integrating sin(x) from " << a << " to " << b;
    std::cout << " using adaptive simpson rule" << std::endl;

    if (!top_simp(a, b, tol, &res, &iter)) {
        std::cout << "FAIL: integration function failed to calculate result" << std::endl;
        return 0;
    }
    double diff = std::fabs(res - exp);
    if (diff > tol) {
        std::cout << "FAIL: expected " << exp << " got " << res << " diff " << diff << std::endl;
        std::cout << "Iterations = " << iter << std::endl;
        return 0;
    }
    std::cout << "PASS: result = " << res << std::endl;
    std::cout << "Iterations = " << iter << std::endl;
    return 1;
}

static int test_romberg_rule(double a, double b, double tol, double exp) {
    double res = 0;
    int iter = 0;

    std::cout << "Integrating sin(x) from " << a << " to " << b;
    std::cout << " using romberg rule" << std::endl;

    if (!top_romberg(a, b, tol, &res, &iter)) {
        std::cout << "FAIL: integration function failed to calculate result" << std::endl;
        return 0;
    }
    double diff = std::fabs(res - exp);
    if (diff > tol) {
        std::cout << "FAIL: expected " << exp << " got " << res << " diff " << diff << std::endl;
        std::cout << "Iterations = " << iter << std::endl;
        return 0;
    }
    std::cout << "PASS: result = " << res << std::endl;
    std::cout << "Iterations = " << iter << std::endl;
    return 1;
}

struct test_data_type {
    double a;
    double b;
    double exp;
};

/*
 * test integrates sin(x) from 0 to 1, 1 to 20, -0.3 to 0.2
 * and compares with expected values calculated using gsl QAG
 * to a tolerance of 0.0001
 */
int main() {
    double tol = 0.0001;
    double res;
    int ret;

    struct test_data_type test_data[] = {
        {0, 0.1, 0.004996},    {0, 1.57, 0.999204},  {0, 1, 0.459698},       {1, 20, 0.132220},
        {10.5, 20, -0.883619}, {19.8, 20, 0.173240}, {-0.3, 0.2, -0.024730}, {-1.3, -0.2, -0.712568},
    };

    for (int i = 0; i < sizeof(test_data) / sizeof(test_data_type); i++) {
        if (!test_trapezoidal_rule(test_data[i].a, test_data[i].b, tol, test_data[i].exp)) {
            return 1;
        }
    }

    for (int i = 0; i < sizeof(test_data) / sizeof(test_data_type); i++) {
        if (!test_simpson_rule(test_data[i].a, test_data[i].b, tol, test_data[i].exp)) {
            return 1;
        }
    }

    for (int i = 0; i < sizeof(test_data) / sizeof(test_data_type); i++) {
        if (!test_romberg_rule(test_data[i].a, test_data[i].b, tol, test_data[i].exp)) {
            return 1;
        }
    }

    std::cout << "TEST PASS" << std::endl;
    return 0;
};
