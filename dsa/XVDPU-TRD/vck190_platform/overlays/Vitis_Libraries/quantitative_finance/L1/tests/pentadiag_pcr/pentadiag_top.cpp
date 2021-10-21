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
/** @file pentadiag_top.cpp
* @brief This file contain top level wrapping function for HLS testbench.
*/
#include "pentadiag_top.hpp"
#include <stdio.h>
/** @brief Top level wrapping function for HLS testbench.
*
* It calls function \a pentadiagCr to solve for u
* Structure of input matrix: \n
*  | c d e 0 0 | \n
*  | b c d e 0 | \n
*  | a b c d e | \n
*  | 0 a b c d | \n
*  | 0 0 a b c | \n
*@param[in] c - Main diagonal \n
*@param[in] b - First lower \n
*@param[in] a - Second lower \n
*@param[in] d - First upper \n
*@param[in] e - Second upper \n

*@param[in]     v - Right hand side vector of length n \n
*@param[out]    u - Vectors of unknows to solve for \n
*/
void pentadiag_top(TEST_DT a[P_SIZE],
                   TEST_DT b[P_SIZE],
                   TEST_DT c[P_SIZE],
                   TEST_DT d[P_SIZE],
                   TEST_DT e[P_SIZE],
                   TEST_DT v[P_SIZE],
                   TEST_DT u[P_SIZE]) {
    xf::fintech::pentadiagCr<TEST_DT, P_SIZE, LOGN>(a, b, c, d, e, v, u);
}
