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
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "utils.cuh"
#include "readBin.cuh"

using namespace std;

template<typename T>
void transpose(T *A, T *t_A, int m, int n){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            t_A[j * m + i] = A[i * n + j];
        }
    }
}

int main(int argc, char** argv) {
    assert(argc >= 4);
    int arg = 0;
    int p_m = atoi(argv[++arg]);
    int p_n = atoi(argv[++arg]);
    string dataPath = argv[++arg];
    
    
    double *m_host_x, *m_host_b, *m_host_A, *m_host_r;
    
    double *m_device_x, *m_device_A, *m_device_r;

    double *m_host_tA = new double[p_m * p_n];
    readBin(dataPath + "A.mat", m_host_tA, p_m * p_n * sizeof(double));
    m_host_A = new double[p_m * p_n];
    transpose(m_host_tA, m_host_A, p_m, p_n);
    delete[] m_host_tA;
    
    m_host_x = new double[p_n];
    readBin(dataPath + "x.mat", m_host_x, p_n * sizeof(double));
    m_host_b = new double[p_m];
    readBin(dataPath + "b.mat", m_host_b, p_m * sizeof(double));
        
    m_host_r = new double[p_m]();

    cublasHandle_t m_handle;
    cublasStatus_t m_stat;
    
    
    m_stat = cublasCreate(&m_handle);
    if (m_stat != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS create handle failed\n");
    }
    
    
    cudaError_t cudaStat = cudaMalloc((void**)&m_device_A, p_m * p_n * sizeof(double));
    if (cudaStat != cudaSuccess) {
        printf("device memory allocation for matrix A failed\n");
    }
    
    cudaStat = cudaMalloc((void**)&m_device_x, p_n * sizeof(double));
    if (cudaStat != cudaSuccess) {
        printf("device memory allocation failed: %d\n", cudaStat);
    }
    
    cudaStat = cudaMalloc((void**)&m_device_r, p_m * sizeof(double));
    if (cudaStat != cudaSuccess) {
        printf("device memory allocation failed: %d\n", cudaStat);
    }
        
    m_stat = cublasSetMatrix(p_m, p_n, sizeof(double), m_host_A, p_m, m_device_A, p_m);
    if (m_stat != CUBLAS_STATUS_SUCCESS) {
        printf("Set matrix A failed: %d\n", m_stat);
    }
    
    m_stat = cublasSetVector(p_n, sizeof(double), m_host_x, 1, m_device_x, 1);
    if (m_stat != CUBLAS_STATUS_SUCCESS) {
        printf("Set vector x failed: %d\n", m_stat);
    }

    m_stat = cublasSetVector(p_m, sizeof(double), m_host_r, 1, m_device_r, 1);
    if (m_stat != CUBLAS_STATUS_SUCCESS) {
        printf("Set vector r failed: %d\n", m_stat);
    }

#ifdef BENCHMARK
    auto start = chrono::high_resolution_clock::now();
#endif
    const double ONE=1, ZERO=0;
    m_stat = cublasDgemv(m_handle, CUBLAS_OP_N, p_m, p_n, &ONE, m_device_A, p_m, m_device_x, 1, &ZERO,
                                 m_device_r, 1);
    if (m_stat != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS gemv failed\n");
    }
    m_stat = cublasGetVector(p_m, sizeof(double),m_device_r,1,m_host_r,1);
    
    if (m_stat !=CUBLAS_STATUS_SUCCESS) {
        printf("cublas get failed\n");
    }


#ifdef BENCHMARK
    auto stop = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = stop - start;
    double duration = elapsed.count();
    cout << "Execution time is " << duration << "s." << endl;
#endif

    
    int err = 0;
    compare(p_m, m_host_b, m_host_r, err);
    if (err == 0) {
        printf("Results verified.\n");
    } else {
        printf("There are in total %d error(s).\n", err);
    }

    cudaFree(m_device_A);
    cudaFree(m_device_x);
    cudaFree(m_device_r);
    delete[] m_host_A;
    delete[] m_host_x;
    delete[] m_host_r;
    delete[] m_host_b;

    return 0;
}
