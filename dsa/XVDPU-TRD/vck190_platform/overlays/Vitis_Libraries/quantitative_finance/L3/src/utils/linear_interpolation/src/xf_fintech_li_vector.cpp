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

#include "xf_fintech_li_private.hpp"

double Xilinx_VectorValueAtIndex(double* pVector, int iAny) {
    return pVector[iAny];
}

bool Xilinx_VectorInterpolationRangeFor(double* pVector,
                                        int AnySize,
                                        double TargetValue,
                                        double* pLowerValue,
                                        double* pUpperValue,
                                        int* piLower,
                                        int* piUpper) {
    bool Result = false;
    int i;
    double Value;

    *pLowerValue = Xilinx_VectorValueAtIndex(pVector, 0);

    for (i = 1; i < AnySize; i++) {
        Value = Xilinx_VectorValueAtIndex(pVector, i);
        if (Value >= TargetValue) {
            *piLower = i - 1;
            *piUpper = i;
            *pUpperValue = Value;
            Result = true;
            break;
        }
        *pLowerValue = Value;
    }

    return Result;
}

bool Xilinx_VectorValueExists(double* pVector, int AnySize, double Target, int* piExists) {
    bool Result = false;
    int i;

    for (i = 0; i < AnySize; i++) {
        if (Xilinx_VectorValueAtIndex(pVector, i) == Target) {
            *piExists = i;
            Result = true;
            break;
        }
    }

    return Result;
}
