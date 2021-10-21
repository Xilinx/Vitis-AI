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
/** @file pentadiag_top.h
* @brief header file for top level wrapping function for HLS testbench.
*/
#ifndef _XF_FINTECH_PENTADIAGTOP_HPP_
#define _XF_FINTECH_PENTADIAGTOP_HPP_

#include <stdio.h>
/// \var typedef double TEST_DT
/// @brief Type definition for all data members used in solver
typedef double TEST_DT;
/// @brief Size of the input matrix
const unsigned P_SIZE = 32;
/// @brief function to calculate log2 to get number of steps
constexpr size_t mylog2(size_t n) {
    return ((n < 2) ? 0 : 1 + mylog2(n / 2));
}
const unsigned LOGN2 = mylog2(P_SIZE);
/// @brief Number of steps for the algorithm
const unsigned LOGN = (LOGN2 > 8) ? 8 : LOGN2;
#include "pentadiag_cr.hpp"

void pentadiag_top(TEST_DT a[P_SIZE],
                   TEST_DT b[P_SIZE],
                   TEST_DT c[P_SIZE],
                   TEST_DT d[P_SIZE],
                   TEST_DT e[P_SIZE],
                   TEST_DT v[P_SIZE],
                   TEST_DT u[P_SIZE]);

#endif
