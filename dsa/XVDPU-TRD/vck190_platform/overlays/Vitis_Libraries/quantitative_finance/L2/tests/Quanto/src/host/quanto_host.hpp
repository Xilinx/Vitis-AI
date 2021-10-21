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

/**
 * @file gk_model.hpp
 * @brief Header file for Garman-Kohlhagen model
 */

#ifndef __XF_FINTECH_QUANTO_HOST_HPP_
#define __XF_FINTECH_QUANTO_HOST_HPP_

struct parsed_params {
    double s;
    double k;
    double v;
    double t;
    double rd;
    double rf;
    double q;
    double E;
    double fxv;
    double corr;
    double exp;
    int validation;
};

void quanto_model(double s,
                  double v,
                  double rd,
                  double t,
                  double k,
                  double rf,
                  double q,
                  double E,
                  double fxv,
                  double corr,
                  unsigned int call,
                  double& price,
                  double& delta,
                  double& gamma,
                  double& vega,
                  double& theta,
                  double& rho);

std::vector<struct parsed_params*>* parse_file(std::string file);

#endif
