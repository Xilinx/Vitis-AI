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

#ifndef _XF_FINTECH_LI_PRIVATE_HPP_
#define _XF_FINTECH_LI_PRIVATE_HPP_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

#include "xf_fintech_li.hpp"

#define XLNX_MIN_VALUES_REQUIRED_FOR_INTERPOLATION (2)

/*
 *
 * Grid
 *
 *
 */

void Xilinx_SetGridValueAt(double* pFxyGrid, int Size_Y, int iXCol, int iYRow, double Value);

double Xilinx_GetGridValueAt(double* pFxyGrid, int Size_Y, int iXCol, int iYRow);

/*
 *
 * Vector
 *
 *
 */

double Xilinx_VectorValueAtIndex(double* pVector, int iAny);

bool Xilinx_VectorInterpolationRangeFor(double* pVector,
                                        int AnySize,
                                        double TargetValue,
                                        double* pLowerValue,
                                        double* pUpperValue,
                                        int* piLower,
                                        int* piUpper);

bool Xilinx_VectorValueExists(double* pVector, int AnySize, double Target, int* piExists);

/*
 *
 * Interpolator
 *
 *
 */

double Xilinx_FindInterpolatedValue(
    double Z11, double Z12, double Z21, double Z22, double X1, double X2, double Xi, double Y1, double Y2, double Yj);

/*
 *
 * Rules
 *
 *
 */

bool Xilinx_Rule_SizeOk(int AnySize);

#ifdef __cplusplus
}
#endif

#endif /* _Xilinx_PRIVATE_ */
