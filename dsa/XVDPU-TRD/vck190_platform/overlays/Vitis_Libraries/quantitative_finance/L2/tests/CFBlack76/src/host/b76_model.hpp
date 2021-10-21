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
 * @file b76_model.hpp
 * @brief Header file for BSM model
 */

#ifndef __XF_FINTECH_BSMMODEL_HPP_
#define __XF_FINTECH_BSMMODEL_HPP_

double phi(double x);
double random_range(double range_min, double range_max);
void b76_model(double f,
               double v,
               double r,
               double t,
               double k,
               double q,
               unsigned int call,
               double& price,
               double& delta,
               double& gamma,
               double& vega,
               double& theta,
               double& rho);

#endif
