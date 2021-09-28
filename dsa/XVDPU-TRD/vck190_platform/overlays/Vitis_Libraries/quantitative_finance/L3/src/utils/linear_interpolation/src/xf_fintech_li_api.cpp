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

bool Xilinx_Interpolate(double* pFxyGrid,
                        double* pX,
                        double* pY,
                        int Size_X,
                        int Size_Y,
                        double Target_X,
                        double Target_Y,
                        double* pAnswer) {
    bool Result = false;
    double Z11, Z12, Z21, Z22, X1, X2, Xi, Y1, Y2, Yj;
    int iTarget_X, iTarget_Y, iX1, iX2, iY1, iY2;

    Xi = Target_X;
    Yj = Target_Y;
    *pAnswer = 0;

    if (Xilinx_Rule_SizeOk(Size_X) && Xilinx_Rule_SizeOk(Size_Y)) {
        if (Xilinx_VectorValueExists(pX, Size_X, Xi, &iTarget_X) &&
            (Xilinx_VectorValueExists(pY, Size_Y, Yj, &iTarget_Y))) {
            *pAnswer = Xilinx_GetGridValueAt(pFxyGrid, Size_Y, iTarget_X, iTarget_Y);
            Result = true;
        } else {
            if (Xilinx_VectorInterpolationRangeFor(pX, Size_X, Xi, &X1, &X2, &iX1, &iX2)) {
                if (Xilinx_VectorInterpolationRangeFor(pY, Size_Y, Yj, &Y1, &Y2, &iY1, &iY2)) {
                    Z11 = Xilinx_GetGridValueAt(pFxyGrid, Size_X, iX1, iY1);
                    Z12 = Xilinx_GetGridValueAt(pFxyGrid, Size_X, iX1, iY2);
                    Z21 = Xilinx_GetGridValueAt(pFxyGrid, Size_X, iX2, iY1);
                    Z22 = Xilinx_GetGridValueAt(pFxyGrid, Size_X, iX2, iY2);

                    *pAnswer = Xilinx_FindInterpolatedValue(Z11, Z12, Z21, Z22, X1, X2, Xi, Y1, Y2, Yj);

                    Result = true;
                }
            }
        }
    }

    return Result;
}
