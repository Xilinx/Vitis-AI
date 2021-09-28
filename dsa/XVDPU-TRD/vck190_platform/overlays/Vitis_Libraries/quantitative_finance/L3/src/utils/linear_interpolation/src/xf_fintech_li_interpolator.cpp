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

double Xilinx_MakeDenominator(double X1, double X2, double Y1, double Y2) {
    return (X2 - X1) * (Y2 - Y1);
}

double Xilinx_MakeNumeratorElement(double Z, double X1, double X2, double Y1, double Y2) {
    return Z * ((X1 - X2) * (Y1 - Y2));
}

double Xilinx_MakeNumerator(
    double Z11, double Z12, double Z21, double Z22, double X1, double X2, double Xi, double Y1, double Y2, double Yj) {
    return Xilinx_MakeNumeratorElement(Z11, X2, Xi, Y2, Yj) + Xilinx_MakeNumeratorElement(Z12, X2, Xi, Yj, Y1) +
           Xilinx_MakeNumeratorElement(Z21, Xi, X1, Y2, Yj) + Xilinx_MakeNumeratorElement(Z22, Xi, X1, Yj, Y1);
}

double Xilinx_FindInterpolatedValue(
    double Z11, double Z12, double Z21, double Z22, double X1, double X2, double Xi, double Y1, double Y2, double Yj) {
    double Numerator = Xilinx_MakeNumerator(Z11, Z12, Z21, Z22, X1, X2, Xi, Y1, Y2, Yj);
    double Denominator = Xilinx_MakeDenominator(X1, X2, Y1, Y2);
    double Result = Numerator / Denominator;

    return Result;
}
