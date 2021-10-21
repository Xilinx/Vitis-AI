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

#include "covariance_regularization.hpp"
#define DT double

void dut(int n, DT threshold, hls::stream<DT>& inStrm, hls::stream<DT>& outStrm) {
    xf::fintech::covReSoftThreshold<DT>(n, threshold, inStrm, outStrm);
}
