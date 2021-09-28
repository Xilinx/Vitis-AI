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

#define LEN (12)
#define TEST_CASES (30)
typedef double DT;

DT dut(DT A[LEN], DT B[LEN], DT t, bool tc);

int main() {
    DT A[LEN] = {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0};

    DT B[LEN] = {0.002, 0.005, 0.007, 0.011, 0.015, 0.018, 0.020, 0.022, 0.025, 0.0288, 0.031, 0.034};

    DT GOLDEN_S[TEST_CASES] = {0.999500125,  0.9890602788, 0.9646402935, 0.9417645336, 0.9157608767, 0.8824969026,
                               0.8507803459, 0.8224345,    0.7966734015, 0.7727110867, 0.7497615922, 0.7271738391,
                               0.704836287,  0.6827722804, 0.6610051637, 0.6395582813, 0.6184549776, 0.597718597,
                               0.5773724838, 0.5574399826, 0.5379444376, 0.5188976309, 0.5002650948, 0.4820007993,
                               0.4640587142, 0.4463928096, 0.4289570553, 0.4117054213, 0.3945918775, 0.3775703938};

    DT GOLDEN_S1[TEST_CASES] = {-0.008041342642, -0.02435402438, -0.02586905605, -0.02259585226, -0.03038578518,
                                -0.03366390001,  -0.02990020736, -0.02692247826, -0.02473071271, -0.0233249107,
                                -0.02270507224,  -0.02246654335, -0.02220467006, -0.02191945237, -0.02161089028,
                                -0.0212789838,   -0.02092373291, -0.02054513762, -0.02014319794, -0.01971791385,
                                -0.01926928536,  -0.01883199971, -0.01844074413, -0.01809551861, -0.01779632317,
                                -0.01754315779,  -0.01733602248, -0.01717491724, -0.01705984207, -0.01699079697};

    int failCnt = 0;
    DT err = 1e-8;

    DT C[LEN] = {0};

    std::cout << std::setprecision(10) << std::endl;

    for (int i = 0; i < LEN; i++) {
        C[i] = std::exp(-B[i] * A[i]);
    }

    // test against python model results
    for (int t = 0; t < TEST_CASES; t++) {
        DT val = dut(A, C, t, 0);
        if (std::abs(val - GOLDEN_S[t]) > err) {
            std::cout << "Failure at t:" << t << " golden S:" << GOLDEN_S[t] << " val:" << val << std::endl;
            failCnt++;
        }

        DT valRate = dut(A, C, t, 1);
        if (std::abs(valRate - GOLDEN_S1[t]) > err) {
            std::cout << "Failure at t:" << t << " golden S1:" << GOLDEN_S1[t] << " val:" << valRate << std::endl;
            failCnt++;
        }
    }

    if (failCnt > 0) {
        return 1;
    }

    return 0;
}
