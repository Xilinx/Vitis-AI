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

#ifndef __XF_FINTECH_GKMODEL_HPP_
#define __XF_FINTECH_GKMODEL_HPP_

struct parsed_params {
    double s;
    double k;
    double v;
    double t;
    double r_domestic;
    double r_foreign;
    int validation;
};

void gk_model(double s,
              double v,
              double r_domestic,
              double t,
              double k,
              double r_foreign,
              unsigned int call,
              double& price,
              double& delta,
              double& gamma,
              double& vega,
              double& theta,
              double& rho);

std::vector<struct parsed_params*>* parse_file(std::string file);

#endif
