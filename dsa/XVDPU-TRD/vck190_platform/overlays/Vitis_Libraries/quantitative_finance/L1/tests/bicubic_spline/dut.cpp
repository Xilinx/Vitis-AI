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
 * @file dut.cpp
 *
 * @brief This file contains top function of test case.
 */

#include "dut.hpp"

DT dut(int n, DT xArr[N], DT yArr[N], DT fArr[N][N], DT x, DT y) {
    xf::fintech::BicubicSplineInterpolation<DT, N> spline;
    spline.init(n, xArr, yArr, fArr);
    return spline.calcu(x, y);
}
