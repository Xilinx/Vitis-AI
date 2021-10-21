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

#include "xf_hpc_mlp.hpp"

#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <chrono>

#define IDX2R(i, j, ld) (((i) * (ld)) + (j))

typedef std::chrono::time_point<std::chrono::high_resolution_clock> TimePointType;

double showTimeData(string p_Task, TimePointType& t1, TimePointType& t2) {
    t2 = chrono::high_resolution_clock::now();
    chrono::duration<double> l_durationSec = t2 - t1;
    double l_timeMs = l_durationSec.count() * 1e3;
    cout << p_Task << "  " << fixed << setprecision(6) << l_timeMs << " msec\n";
    return l_timeMs;
}

// read to 4k aligned memory
void readMatBin(char* mat, unsigned int size, string dataDir, string name, unsigned int eleSize) {
    ifstream inFile;
    inFile.open(dataDir + name + ".bin", ifstream::binary);
    if (inFile.is_open()) {
        inFile.read((char*)mat, eleSize * size);
        inFile.close();
    } else {
        cerr << "Could not find " << (dataDir + name + ".bin") << endl;
        exit(1);
    }
}

bool compare(float* c, float* goldenC, int m, int n, int padded_n, float p_TolRel = 1e-3, float p_TolAbs = 1e-5) {
    bool l_check = true;
    int check_num = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            float l_ref = goldenC[IDX2R(row, col, n)];
            float l_result = c[IDX2R(row, col, padded_n)];
            float l_diffAbs = abs(l_ref - l_result);
            float l_diffRel = l_diffAbs;
            if (goldenC[IDX2R(row, col, n)] != 0) {
                l_diffRel /= abs(l_ref);
            }
            bool check = (l_diffRel <= p_TolRel) || (l_diffAbs <= p_TolAbs);
            if (!check) {
                cout << "row " << row << " col " << col << "\n";
                cout << "golden result " << setprecision(10) << goldenC[IDX2R(row, col, n)]
                     << " is not equal to fpga result " << setprecision(10) << c[IDX2R(row, col, padded_n)] << "\n";

                l_check = false;
                check_num++;
            }
        }
    }
    cout << "number of mismatch" << check_num << "\n";
    return l_check;
}

int getPaddedSize(int p_size, int p_minSize) {
    return p_size + p_minSize - 1 - (p_size - 1) % p_minSize;
}

float* fillMat(string dataDir, string name, int m, int n, int padded_m, int padded_n) {
    float* mat;
    float* padded_input;
    posix_memalign((void**)&mat, 4096, m * n * sizeof(float));
    posix_memalign((void**)&padded_input, 4096, padded_m * padded_n * sizeof(float));
    memset(padded_input, 0, padded_m * padded_n * sizeof(float));

    ifstream inFile;
    inFile.open(dataDir + name + ".bin", ifstream::binary);
    if (inFile.is_open()) {
        inFile.read((char*)mat, m * n * sizeof(float));
        inFile.close();
    } else {
        cerr << "Could not find " << (dataDir + name + ".bin") << endl;
        exit(1);
    }

    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            padded_input[IDX2R(row, col, padded_n)] = mat[IDX2R(row, col, n)];
        }
    }

    free(mat);

    return padded_input;
}

void sendInput(float* inputs[BLAS_numKernels], int input_size, int k0) {
    bool check = true;

    for (int CU = 0; CU < BLAS_numKernels; CU++) {
        // FPGA calls to create BO for inputs
        check = xfhpcMalloc(input_size, k0, sizeof(float), inputs[CU], k0, CU);
        if (!check) {
            cout << "Malloc memory for input failed. \n";
        }

        // FPGA calls to send inputs to device
        check = xfhpcSetMatrix(inputs[CU], CU);
        if (!check) {
            cout << "Set Matrix failed. \n";
        }
    }
}

void sendWeight(float* w0,
                float* w1,
                float* w2,
                float* bias0,
                float* bias1,
                float* bias2,
                int k0,
                int n0,
                int k1,
                int n1,
                int k2,
                int n2) {
    bool check = true;

    for (int j = 0; j < BLAS_numKernels; j++) {
        // FPGA calls to create BO for weights
        check = xfhpcMalloc(k0, n0, sizeof(float), w0, n0, j);
        check = xfhpcMalloc(k1, n1, sizeof(float), w1, n1, j);
        check = xfhpcMalloc(k2, n2, sizeof(float), w2, n2, j);

        // FPGA calls to send weights to device
        check = xfhpcSetMatrix(w0, j);
        check = xfhpcSetMatrix(w1, j);
        check = xfhpcSetMatrix(w2, j);

        // FPGA calls to create BO for bias
        check = xfhpcMalloc(1, n0, sizeof(float), bias0, n0, j);
        check = xfhpcMalloc(1, n1, sizeof(float), bias1, n1, j);
        check = xfhpcMalloc(1, n2, sizeof(float), bias2, n2, j);

        // FPGA calls to send bias to device
        check = xfhpcSetMatrix(bias0, j);
        check = xfhpcSetMatrix(bias1, j);
        check = xfhpcSetMatrix(bias2, j);
    }

    if (!check) {
        cout << "Send weights and bias failed. \n";
    }
}

float* executeInput(float* inputs[BLAS_numKernels],
                    float* w0,
                    float* w1,
                    float* w2,
                    float* bias0,
                    float* bias1,
                    float* bias2,
                    int k0,
                    int n0,
                    int k1,
                    int n1,
                    int k2,
                    int n2,
                    int batch_size,
                    double* executeTime,
                    double* totalSendTime,
                    double* totalGetTime) {
    bool check = true;

    int input_size = batch_size / BLAS_numKernels;

    // create empty host memory for results
    float *c0, *c1;
    vector<float*> c2;

    posix_memalign((void**)&c0, 4096, input_size * n0 * sizeof(float));
    posix_memalign((void**)&c1, 4096, input_size * n1 * sizeof(float));
    memset(c0, 0, input_size * n0 * sizeof(float));
    memset(c1, 0, input_size * n1 * sizeof(float));

    for (int j = 0; j < BLAS_numKernels; j++) {
        float* tmp_c;
        posix_memalign((void**)&tmp_c, 4096, input_size * n2 * sizeof(float));
        memset(tmp_c, 0, input_size * n2 * sizeof(float));
        c2.push_back(tmp_c);
    }

    for (int j = 0; j < BLAS_numKernels; j++) {
        // FPGA calls to create BO for results, since results don't have initial values, there is no need to send inputs
        // to device
        check = xfhpcMalloc(input_size, n0, sizeof(float), c0, n0, j);
        check = xfhpcMalloc(input_size, n1, sizeof(float), c1, n1, j);
        check = xfhpcMalloc(input_size, n2, sizeof(float), c2[j], n2, j);

        // FPGA calls to write instructions

        xfhpcFcn(input_size, n0, k0, 1, inputs[j], k0, w0, n0, 1, c0, n0, bias0, n0, 1, 0, 1, 0, j);
        xfhpcFcn(input_size, n1, k1, 1, c0, k1, w1, n1, 1, c1, n1, bias1, n1, 1, 0, 1, 0, j);
        xfhpcFcn(input_size, n2, k2, 1, c1, k2, w2, n2, 1, c2[j], n2, bias2, n2, 1, 0, 1, 1, j);
    }

    // FPGA call to run all CUs
    TimePointType l_tp_startrun_time = chrono::high_resolution_clock::now();
    TimePointType l_tp_execute_time;

    xfhpcExecuteAsync(BLAS_numKernels);

    *executeTime = *executeTime + showTimeData("xfhpcExecuteTime", l_tp_startrun_time, l_tp_execute_time);

    // allocate memory for reult
    float* result_c;
    posix_memalign((void**)&result_c, 4096, batch_size * n2 * sizeof(float));

    for (int j = 0; j < BLAS_numKernels; j++) {
        // FPGA call to get result in each CU from device
        TimePointType l_tp_startget_time = chrono::high_resolution_clock::now();
        TimePointType l_tp_get_time;

        // FPGA call to get result in each CU from device
        xfhpcGetByPointer(c2[j], j);
        *totalGetTime = *totalGetTime + showTimeData("xfhpcGetTime " + to_string(j), l_tp_startget_time, l_tp_get_time);

        // combine them into one result_c
        copy(c2[j], c2[j] + input_size * n2, result_c + j * input_size * n2);
    }

    // FPGA calls to free B0s in device
    for (int j = 0; j < BLAS_numKernels; j++) {
        xfhpcFree(c2[j], j);
        xfhpcFree(c0, j);
        xfhpcFree(c1, j);
        xfhpcFreeInstr(j);
        free(c2[j]);
    }

    free(c0);
    free(c1);

    return result_c;
}

int main(int argc, char** argv) {
    unsigned int l_argIdx = 1;
    string l_xclbinDir(argv[l_argIdx++]);
    string l_dataDir(argv[l_argIdx++]);
    l_dataDir = l_dataDir + "/";

    double totalSendTime = 0;
    double executeTime = 0;
    double totalGetTime = 0;

    int l_model = atoi(argv[l_argIdx++]);
    int l_batch_size = atoi(argv[l_argIdx++]);
    int number_of_inputs = atoi(argv[l_argIdx++]);

    int l_k0 = 356;
    int l_n0 = 30;
    int l_k1 = 30;
    int l_n1 = 20;
    int l_k2 = 20;
    int l_n2 = 3;

    int l_padded_batch_size = getPaddedSize(l_batch_size, BLAS_gemmMBlocks * BLAS_ddrWidth * BLAS_numKernels);
    int l_padded_k0 = getPaddedSize(l_k0, BLAS_gemmKBlocks * BLAS_ddrWidth);
    int l_padded_n0 = getPaddedSize(l_n0, BLAS_gemmNBlocks * BLAS_ddrWidth);
    int l_padded_k1 = getPaddedSize(l_k1, max(BLAS_gemmKBlocks, BLAS_gemmNBlocks) * BLAS_ddrWidth);
    int l_padded_n1 = getPaddedSize(l_n1, BLAS_gemmNBlocks * BLAS_ddrWidth);
    int l_padded_k2 = getPaddedSize(l_k2, max(BLAS_gemmKBlocks, BLAS_gemmNBlocks) * BLAS_ddrWidth);
    int l_padded_n2 = getPaddedSize(l_n2, BLAS_gemmNBlocks * BLAS_ddrWidth);

    // FPGA call to load xclbin and create device handle for each CU

    bool check = xfhpcCreate((l_xclbinDir + "/fcn.xclbin").c_str(), BLAS_numKernels);
    if (!check) {
        cout << "Create Handle failed. \n";
    }
    vector<float *> w0s, w1s, w2s;
    vector<float *> bias0s, bias1s, bias2s;

    for (int model = 0; model < l_model; model++) {
        // create host memory for weights and bias read from data files

        float* w0 = fillMat(l_dataDir, "matW1_" + to_string(model), l_k0, l_n0, l_padded_k0, l_padded_n0);
        float* w1 = fillMat(l_dataDir, "matW2_" + to_string(model), l_k1, l_n1, l_padded_k1, l_padded_n1);
        float* w2 = fillMat(l_dataDir, "matW3_" + to_string(model), l_k2, l_n2, l_padded_k2, l_padded_n2);

        float* bias0 = fillMat(l_dataDir, "matb1_" + to_string(model), 1, l_n0, 1, l_padded_n0);
        float* bias1 = fillMat(l_dataDir, "matb2_" + to_string(model), 1, l_n1, 1, l_padded_n1);
        float* bias2 = fillMat(l_dataDir, "matb3_" + to_string(model), 1, l_n2, 1, l_padded_n2);

        w0s.push_back(w0);
        w1s.push_back(w1);
        w2s.push_back(w2);

        bias0s.push_back(bias0);
        bias1s.push_back(bias1);
        bias2s.push_back(bias2);

        TimePointType l_tp_start_time = chrono::high_resolution_clock::now();
        TimePointType l_tp_send_time;

        // FPGA call to send weights and bias
        sendWeight(w0, w1, w2, bias0, bias1, bias2, l_padded_k0, l_padded_n0, l_padded_k1, l_padded_n1, l_padded_k2,
                   l_padded_n2);

        totalSendTime = totalSendTime +
                        showTimeData("xfhpcSendTime for model " + to_string(model), l_tp_start_time, l_tp_send_time);
    }

    // create host memory for input and read from data file

    for (int inp = 0; inp < number_of_inputs; inp++) {
        // create host memory for input read from data file

        float* input = fillMat(l_dataDir, "mat_input_" + to_string(inp) + "_" + to_string(l_batch_size), l_batch_size,
                               l_k0, l_padded_batch_size, l_padded_k0);

        int input_size = l_padded_batch_size / BLAS_numKernels;
        float* inputs[BLAS_numKernels];

        // split input into parts, each part for each CU
        for (int CU = 0; CU < BLAS_numKernels; CU++) {
            posix_memalign((void**)&inputs[CU], 4096, input_size * l_padded_k0 * sizeof(float));
            copy(input + CU * input_size * l_padded_k0, input + (CU + 1) * input_size * l_padded_k0, inputs[CU]);
        }

        TimePointType l_tp_startSend_time = chrono::high_resolution_clock::now();
        TimePointType l_tp_send_input_time;

        // FPGA calls to send input to each CU
        sendInput(inputs, input_size, l_padded_k0);

        totalSendTime =
            totalSendTime + showTimeData("xfhpcSendTime for input", l_tp_startSend_time, l_tp_send_input_time);

        for (int model = 0; model < l_model; model++) {
            // FPGA calls to get result_c (sizes are padded)

            float* result_c =
                executeInput(inputs, w0s[model], w1s[model], w2s[model], bias0s[model], bias1s[model], bias2s[model],
                             l_padded_k0, l_padded_n0, l_padded_k1, l_padded_n1, l_padded_k2, l_padded_n2,
                             l_padded_batch_size, &executeTime, &totalSendTime, &totalGetTime);

            // read golden reference and compare with result_c
            float* golden_c;
            posix_memalign((void**)&golden_c, 4096, l_batch_size * l_n2 * sizeof(float));
            readMatBin((char*)golden_c, l_batch_size * l_n2, l_dataDir,
                       "mat_sigmoid_output_input_" + to_string(inp) + "_model_" + to_string(model), sizeof(float));
            compare(result_c, golden_c, l_batch_size, l_n2, l_padded_n2);

            free(result_c);
            free(golden_c);
        }

        for (int CU = 0; CU < BLAS_numKernels; CU++) {
            xfhpcFree(inputs[CU]);
            free(inputs[CU]);
        }
        free(input);
    }

    double l_timeApiInMs = totalGetTime + totalSendTime + executeTime;

    cout << "DATA_CSV:,Traces,Inputs,L1,L2,Outputs,TimeApiSeconds,ExecutionSeconds,SendDataSeconds,"
            "GetOutputSeconds\n";
    cout << "DATA_CSV:," << l_batch_size << "," << l_padded_batch_size << "," << l_padded_n0 << "," << l_padded_n1
         << "," << l_padded_n2 << "," << l_timeApiInMs * 1e-3 << "," << executeTime * 1e-3 << ","
         << totalSendTime * 1e-3 << "," << totalGetTime * 1e-3 << "\n";

    for (int model = 0; model < l_model; model++) {
        // FPGA call to free device memory of weights and bias
        for (int j = 0; j < BLAS_numKernels; j++) {
            xfhpcFree(w0s[model], j);
            xfhpcFree(w1s[model], j);
            xfhpcFree(w2s[model], j);
            xfhpcFree(bias0s[model], j);
            xfhpcFree(bias1s[model], j);
            xfhpcFree(bias2s[model], j);
        }
        // free host memory of weights and bias
        free(w0s[model]);
        free(w1s[model]);
        free(w2s[model]);
        free(bias0s[model]);
        free(bias1s[model]);
        free(bias2s[model]);
    }
    // release device handle
    xfhpcDestroy(BLAS_numKernels);

    return 0;
}
