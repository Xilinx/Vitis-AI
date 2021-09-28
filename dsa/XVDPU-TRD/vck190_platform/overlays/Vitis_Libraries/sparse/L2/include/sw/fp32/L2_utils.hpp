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
 * @file L2_utils.hpp
 * @brief header file for common functions used in L2/L3 host code.
 *
 * This file is part of Vitis SPARSE Library.
 */
#ifndef XF_SPARSE_L2_UTILS_HPP
#define XF_SPARSE_L2_UTILS_HPP

#include <chrono>
#include <iostream>
using namespace std;

namespace xf {
namespace sparse {
typedef chrono::time_point<chrono::high_resolution_clock> TimePointType;

double showTimeData(string p_Task, TimePointType& t1, TimePointType& t2) {
    t2 = chrono::high_resolution_clock::now();
    chrono::duration<double> l_durationSec = t2 - t1;
    double l_mSecs = l_durationSec.count() * 1e3;
    cout << "  DATA: time " << p_Task << "  " << fixed << setprecision(6) << l_mSecs << " msec\n";
    return l_mSecs;
}

ifstream::pos_type getFileSize(string p_fileName) {
    ifstream l_f(p_fileName.c_str(), ifstream::ate | ifstream::binary);
    return l_f.tellg();
}
unsigned int alignedNum(unsigned int p_val, unsigned int p_blockSize) {
    unsigned int l_blocks = (p_val + p_blockSize - 1) / p_blockSize;
    return (l_blocks * p_blockSize);
}
unsigned int alignedBlock(unsigned int p_val, unsigned int p_blockSize) {
    unsigned int l_blocks = (p_val + p_blockSize - 1) / p_blockSize;
    return (l_blocks);
}

} // end namespace sparse
} // end namespace xf
#endif
