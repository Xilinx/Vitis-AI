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
#include <iostream>
#include "dut.hpp"

extern "C" void dut(double input[(2 * ASSETS + 1) * ASSETS],
                    double output[(2 * ASSETS - 1) * ASSETS],
                    int func,
                    int calls);

int main(int argc, char* argv[]) {
    bool run_csim = true;
    if (argc >= 2) {
        run_csim = std::stoi(argv[1]);
        if (run_csim) std::cout << "run csim for function verify\n";
    }
    double input[(2 * ASSETS + 1) * ASSETS] = {1,
                                               0.957482990199099,
                                               0.93205263221546,
                                               0.704604422145584,
                                               0.908537126629074,
                                               0.288489728550935,
                                               -0.347830942556597,
                                               0.322028310070462,
                                               -0.282880277688434,
                                               -0.0869605147949781,
                                               0.101447159548319,
                                               0.433358254023517,
                                               0.0585844440814827,
                                               0.670197643163892,
                                               0.136816081865735,
                                               -0.460468239377852,
                                               0.159762043117832,
                                               -0.263238481376653,
                                               0.201654994964664,
                                               -0.508648405149937,
                                               0.256091761917442,
                                               -0.215370101854328,
                                               0.419522360267681,
                                               -0.295569819487608,
                                               0.0791014322139116,
                                               -0.282045902360057,
                                               0.0521688184047328,
                                               0.166596923764531,
                                               -0.405180562577374,
                                               0.615197238544191,
                                               0.137133529673341,
                                               0.0941281685207789,
                                               -0.178220937616234,
                                               -0.0923108695861689,
                                               0.589952810041471,
                                               0.270260823126419,
                                               -0.221972738249483,
                                               -0.354542786827701,
                                               0.800962488960067,
                                               -0.318668342777602,
                                               -0.543253982706129,
                                               0.316608693055737,
                                               0.669447757145699,
                                               0.231483089318767,
                                               0.206191663082365,
                                               -0.257851643709044,
                                               0.391911075086652,
                                               0.209688545857994,
                                               0.4655696787689,
                                               0.19772072083053,
                                               0.0586506911295391,
                                               0.345363893613915,
                                               0.278205771260688,
                                               -0.345680044018216,
                                               -0.0311876913709608};

    double corrMatrix[(ASSETS * 2) * (ASSETS * 2)] = {
        1.000000,  0.288490,  0.101447,  -0.460468, 0.256092,  -0.282046, 0.137134,  0.270261,  -0.543254, -0.257852,
        0.288490,  1.000000,  -0.303776, 0.282093,  0.226849,  -0.287581, 0.089512,  0.168094,  -0.369258, 0.228760,
        0.101447,  -0.303776, 1.000000,  0.102699,  0.025013,  -0.199052, 0.386783,  0.149954,  -0.144014, -0.466737,
        -0.460468, 0.282093,  0.102699,  1.000000,  -0.229141, 0.423995,  0.236647,  -0.238266, -0.188927, 0.076722,
        0.256092,  0.226849,  0.025013,  -0.229141, 1.000000,  -0.390652, 0.135289,  -0.284505, 0.001458,  0.548820,
        -0.282046, -0.287581, -0.199052, 0.423995,  -0.390652, 1.000000,  -0.225096, -0.172568, 0.086160,  -0.035896,
        0.137134,  0.089512,  0.386783,  0.236647,  0.135289,  -0.225096, 1.000000,  0.102372,  0.078166,  -0.368966,
        0.270261,  0.168094,  0.149954,  -0.238266, -0.284505, -0.172568, 0.102372,  1.000000,  0.171035,  -0.284791,
        -0.543254, -0.369258, -0.144014, -0.188927, 0.001458,  0.086160,  0.078166,  0.171035,  1.000000,  0.245637,
        -0.257852, 0.228760,  -0.466737, 0.076722,  0.548820,  -0.035896, -0.368966, -0.284791, 0.245637,  1.000000};

    /*
        double input[(2 * ASSETS + 1) * ASSETS] = {
            1, 1, 1, 1, 1,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 1,
            0, 0, 0, 1, 0,
            0, 0, 1, 0, 0,
            0, 1, 0, 0, 0,
            1, 0, 0, 0, 0
        };

        double corrMatrix[(ASSETS * 2) * (ASSETS * 2)] = {
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1
        };
    */
    int func = 2;
    int calls = 50;
    if (!run_csim) calls = 1;
    double output[ASSETS * (2 * ASSETS + 1)];

    dut(input, output, func, calls);

    if (func == 0) {
        for (int i = 0; i < ASSETS * 2; i++) {
            std::cout << i << "th random variate's mean of " << calls * BUFFDEPTH << " samples is " << output[i]
                      << std::endl;
            if (abs(output[i] - 0.0) > 0.1) {
                return -1;
            }
        }
    } else if (func == 1) {
        for (int i = 0; i < ASSETS * 2; i++) {
            std::cout << i << "th random variate's variance of " << calls * BUFFDEPTH << " samples is " << output[i]
                      << std::endl;
            if (abs(output[i] - 1.0) > 0.1) {
                return -1;
            }
        }
    } else if (func == 2) {
        std::cout << "input correlation(output corrlation)" << std::endl;
        int cnt = 0;
        for (int i = 1; i < ASSETS * 2; i++) {
            for (int j = 0; j < i; j++) {
                std::cout << corrMatrix[i * ASSETS * 2 + j] << "(" << output[cnt] << ") ";
                if (abs(corrMatrix[i * ASSETS * 2 + j] - output[cnt]) > 0.1) {
                    return -1;
                }
                cnt++;
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
