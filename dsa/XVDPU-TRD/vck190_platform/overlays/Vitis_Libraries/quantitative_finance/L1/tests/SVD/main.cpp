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
#include <iostream>
#include "jacobi_svd.hpp"
#define NA 4 // dataA N
#include "svd_top.hpp"
using namespace std;

void linearSolver(double** U_bdc, double* vS_bdc, double** V_bdc, double* B, double* aa) {
    double product[NA];
    for (int kk = 0; kk < NA; ++kk) {
        product[kk] = 0;
        for (int zz = 0; zz < NA; ++zz) {
            product[kk] += U_bdc[zz][kk] * B[zz];
        }
    }
    for (int j = 0; j < NA; ++j) {
        aa[j] = 0;
    }
    double u;
    for (int i = 0; i < NA; ++i) {
        if (vS_bdc[i] > 1e-10) {
            u = product[i] / vS_bdc[i];
            for (int j = 0; j < NA; ++j) {
                aa[j] += u * V_bdc[j][i];
            }
        }
    }
}

int main() {
    double dataB_reduced[NA];
    double dataA_reduced[NA][NA];
    double dataA_golden[NA][NA];
    double dataU_reduced[NA][NA];
    double dataV_reduced[NA][NA];
    double sigma[NA][NA];
    double sigma2[NA][NA];
    dataA_reduced[0][0] = 2545;
    dataA_reduced[0][1] = 2137.34902052323241150588728487;
    dataA_reduced[0][2] = 1821.87553334160179474565666169;
    dataA_reduced[0][3] = 16306.0391790706853498704731464;
    dataA_reduced[1][0] = 2137.34902052323241150588728487;
    dataA_reduced[1][1] = 1821.87553334160179474565666169;
    dataA_reduced[1][2] = 1573.78716625872380063810851425;
    dataA_reduced[1][3] = 12618.9394872652155754622071981;
    dataA_reduced[2][0] = 1821.87553334160179474565666169;
    dataA_reduced[2][1] = 1573.78716625872380063810851425;
    dataA_reduced[2][2] = 1375.76089061747416053549386561;
    dataA_reduced[2][3] = 9923.53468331521253276150673628;
    dataA_reduced[3][0] = 16306.0391790706853498704731464;
    dataA_reduced[3][1] = 12618.9394872652155754622071981;
    dataA_reduced[3][2] = 9923.53468331521253276150673628;
    dataA_reduced[3][3] = 147483.987672218354418873786926;
    dataB_reduced[0] = 16270.5645060545830347109586;
    dataB_reduced[1] = 12590.6436801607960660476237535;
    dataB_reduced[2] = 9900.72724799832030839752405882;
    dataB_reduced[3] = 147196.833035749499686062335968;

    for (int i = 0; i < NA; ++i) {
        for (int j = 0; j < NA; ++j) {
            dataA_golden[i][j] = dataA_reduced[i][j];
        }
    }

    svd_top(dataA_reduced, sigma, dataU_reduced, dataV_reduced, NA);

    double** dataU1;
    dataU1 = new double*[NA];
    for (int i = 0; i < NA; i++) {
        dataU1[i] = new double[NA];
    }
    double** dataV1;
    dataV1 = new double*[NA];
    for (int i = 0; i < NA; i++) {
        dataV1[i] = new double[NA];
    }

    double vSS[NA];

    for (int r = 0; r < NA; r++) {
        for (int j = 0; j < NA; j++) {
            dataU1[r][j] = dataU_reduced[r][j];
            dataV1[r][j] = dataV_reduced[r][j];
        }
        vSS[r] = sigma[r][r];
    }

    // solve Ax = B
    double output[NA];
    linearSolver(dataU1, vSS, dataV1, dataB_reduced, output);
    double Golden[4];
    Golden[0] = -0.180969384057199866866483262129;
    Golden[1] = -0.206203666499685800417296377418;
    Golden[2] = 0.391362006544230478510826287675;
    Golden[3] = 1.0093712978332951557547403354;
    double error = 0;
    for (int i = 0; i < NA; ++i) {
        error += abs(output[i] - Golden[i]);
    }

    // calculate U*Sigma*V
    for (int i = 0; i < NA; ++i) {
        for (int j = 0; j < NA; ++j) {
            dataU_reduced[i][j] = dataU_reduced[i][j] * sigma[j][j];
        }
    }
    double dataA_out[NA][NA];
    for (int i = 0; i < NA; ++i) {
        for (int j = 0; j < NA; ++j) {
            double tmpSum = 0;
            for (int k = 0; k < NA; ++k) {
                tmpSum += dataU_reduced[i][k] * dataV_reduced[j][k];
            }
            dataA_out[i][j] = tmpSum; // dataU_reduced[i][j] * dataV_reduced[i][j];
        }
    }

    double errA = 0;
    for (int i = 0; i < NA; i++) {
        for (int j = 0; j < NA; j++) {
            errA += (dataA_golden[i][j] - dataA_out[i][j]) * (dataA_golden[i][j] - dataA_out[i][j]);
        }
    }
    errA = std::sqrt(errA);

    if ((errA > 0.0001) || (error > 0.0001)) {
        std::cout << "result false" << std::endl;
        return -1;
    } else {
        std::cout << "result correct" << std::endl;
        return 0;
    }
}
