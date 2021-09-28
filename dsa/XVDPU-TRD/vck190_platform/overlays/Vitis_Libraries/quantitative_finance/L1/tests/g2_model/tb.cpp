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

#include <ap_int.h>
#include <math.h>
#include <stdio.h>
#include <fstream>
#include <iostream>

#define LEN 16
typedef double DT;

void dut(int endCnt,
         DT time[LEN],
         DT dtime[LEN],
         DT flatRate,
         DT spread,
         DT a,
         DT sigma,
         DT b,
         DT eta,
         DT rho,
         DT x0,
         DT* discount);

int main() {
    DT golden = 1.0977666441501908;
    DT err = 1e-8;

    int endCnt = 13;
    DT discount;
    DT time[LEN] = {0,
                    0.5,
                    1,
                    1.4958904109589042,
                    2,
                    2.4986301369863013,
                    3.0027397260273974,
                    3.4986301369863013,
                    4.0027397260273974,
                    4.4986301369863018,
                    5.0027397260273974,
                    5.4986301369863018,
                    6.0027397260273974};
    DT dtime[LEN] = {0.5,
                     0.5,
                     0.49589041095890418,
                     0.50410958904109582,
                     0.49863013698630132,
                     0.50410958904109604,
                     0.49589041095890396,
                     0.50410958904109604,
                     0.4958904109589044,
                     0.5041095890410956,
                     0.4958904109589044,
                     0.5041095890410956};
    DT flatRate = 0.04875825;
    DT spread = 0.0;
    /*DT a = 0.050055733653096922;
    DT sigma = 0.0094424342056787739;
    DT b = 0.050052910248222851;
    DT eta = 0.0094424313463861171;
    DT rho = -0.76300324120391616;*/
    DT a = 0.046773707045288972;
    DT sigma = 0.0088292650357800363;
    DT eta = 0.0088292220795323662;
    DT b = 0.046778630704504463;
    DT rho = -0.77927619585931251;

    DT x0 = 0.0;
    DT rates[LEN];
    dut(endCnt, time, dtime, flatRate, spread, a, sigma, b, eta, rho, x0, &discount);

    std::cout << "discount=" << discount << ",diff=" << discount - golden << std::endl;
    if (std::abs(discount - golden) < err)
        return 0;
    else
        return 1;
}
