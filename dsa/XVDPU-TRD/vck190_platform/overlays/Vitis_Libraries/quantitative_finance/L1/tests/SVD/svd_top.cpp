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
#include "svd_top.hpp"
void svd_top(double dataA_reduced[4][4],
             double sigma[4][4],
             double dataU_reduced[4][4],
             double dataV_reduced[4][4],
             int diagSize1) {
    xf::fintech::svd<double, 4>(dataA_reduced, sigma, dataU_reduced, dataV_reduced);
}
