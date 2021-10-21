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

#ifndef _XF_FINTECH_HESTON_TYPES_H_
#define _XF_FINTECH_HESTON_TYPES_H_

#include <vector>

namespace xf {
namespace fintech {
namespace hestonfd {

typedef struct model_parameters_t {
    double K;
    double S;
    double V;
    double T;
    double kappa;
    double sig;
    double rho;
    double eta;
    double rd;
    double rf;
} model_parameters_t;

typedef struct solver_parameters_t {
    int scheme;
    double theta;
    int N;
    double dt;
    int m1;
    int m2;
    int gridType;
} solver_parameters_t;

void Diff(std::vector<double> inputArray, std::vector<double>* diffArray);
void Concatenate(std::vector<double> firstInputArray,
                 std::vector<double> secondInputArray,
                 std::vector<double>* concatenatedArray);

} // namespace hestonfd
} // namespace fintech
} // namespace xf

#endif //_XF_FINTECH_HESTON_TYPES_H_
