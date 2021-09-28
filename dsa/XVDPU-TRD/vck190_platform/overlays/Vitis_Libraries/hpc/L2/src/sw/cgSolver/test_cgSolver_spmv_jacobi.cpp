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
#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <chrono>
#include <cassert>

// This file is required for OpenCL C++ wrapper APIs
#include "cgInstr.hpp"
#include "cgSolverKernel.hpp"
#include "blas/L1/tests/sw/include/utils.hpp"
#include "blas/L1/tests/sw/include/binFiles.hpp"
#include "sparse/L2/include/sw/fp64/spmvKernel.hpp"

using namespace std;

int main(int argc, char** argv) {
    if (argc < 6 || argc > 9) {
        cout << "Usage: " << argv[0] << " <XCLBIN File> <Max Iteration> <Tolerence> <signature_path> <vector_path> "
                                        "<mtx_name> [--debug] [device id]"
             << endl;
        return EXIT_FAILURE;
    }

    uint32_t l_idx = 1;

    string binaryFile = argv[l_idx++];

    int l_maxIter = atoi(argv[l_idx++]);
    CG_dataType l_tol = atof(argv[l_idx++]);
    string l_sigPath = argv[l_idx++];
    string l_vecPath = argv[l_idx++];
    string l_mtxName = argv[l_idx++];
    string l_sigFilePath = l_sigPath + "/" + l_mtxName;
    string l_vecFilePath = l_vecPath + "/" + l_mtxName;

    int l_instrSize = CG_instrBytes * (1 + l_maxIter);

    vector<uint32_t> l_info(6);
    readBin<uint32_t>(l_sigFilePath + "/info.dat", 6 * sizeof(uint32_t), l_info.data());
    unsigned int l_vecSize = (l_info[1] + CG_parEntries - 1) / CG_parEntries;
    l_vecSize = l_vecSize * CG_parEntries;

    // I/O Data Vectors
    host_buffer_t<uint8_t> h_instr(l_instrSize);
    host_buffer_t<uint8_t> h_nnzVal[CG_numChannels];
    host_buffer_t<uint8_t> h_parParam, h_rbParam;
    host_buffer_t<CG_dataType> h_diagA(l_vecSize);
    host_buffer_t<CG_dataType> h_x(l_vecSize);
    host_buffer_t<CG_dataType> h_b(l_vecSize);
    host_buffer_t<CG_dataType> h_pk(l_vecSize);
    host_buffer_t<CG_dataType> h_Apk(l_vecSize);
    host_buffer_t<CG_dataType> h_xk(l_vecSize);
    host_buffer_t<CG_dataType> h_rk(l_vecSize);
    host_buffer_t<CG_dataType> h_zk(l_vecSize);
    host_buffer_t<CG_dataType> h_jacobi(l_vecSize);

    for (int i = 0; i < l_vecSize; i++) {
        h_xk[i] = 0;
        h_Apk[i] = 0;
    }

    size_t l_bytes = getBinBytes(l_sigFilePath + "/parParam.dat");
    readBin<uint8_t, aligned_allocator<uint8_t> >(l_sigFilePath + "/parParam.dat", l_bytes, h_parParam);

    l_bytes = getBinBytes(l_sigFilePath + "/rbParam.dat");
    readBin<uint8_t, aligned_allocator<uint8_t> >(l_sigFilePath + "/rbParam.dat", l_bytes, h_rbParam);
    for (unsigned int i = 0; i < CG_numChannels; ++i) {
        string l_nnzFileName = l_sigFilePath + "/nnzVal_" + to_string(i) + ".dat";
        l_bytes = getBinBytes(l_nnzFileName);
        readBin<uint8_t, aligned_allocator<uint8_t> >(l_nnzFileName, l_bytes, h_nnzVal[i]);
    }

    readBin(l_vecFilePath + "/A_diag.mat", h_diagA.size() * sizeof(CG_dataType), h_diagA);
    readBin(l_vecFilePath + "/x.mat", h_x.size() * sizeof(CG_dataType), h_x);
    readBin(l_vecFilePath + "/b.mat", h_b.size() * sizeof(CG_dataType), h_b);

    CG_dataType l_dot = 0, l_rz = 0;
    for (int i = 0; i < l_vecSize; i++) {
        h_rk[i] = h_b[i];
        h_jacobi[i] = 1.0 / h_diagA[i];
        h_zk[i] = h_jacobi[i] * h_rk[i];
        l_dot += h_b[i] * h_b[i];
        l_rz += h_rk[i] * h_zk[i];
        h_pk[i] = h_zk[i];
    }

    xf::hpc::MemInstr<CG_instrBytes> l_memInstr;
    xf::hpc::cg::CGSolverInstr<CG_dataType> l_cgInstr;
    l_cgInstr.setMaxIter(l_maxIter);
    l_cgInstr.setTols(l_dot * l_tol * l_tol);
    l_cgInstr.setRes(l_dot);
    l_cgInstr.setRZ(l_rz);
    l_cgInstr.setVecSize(l_vecSize);
    l_cgInstr.store(h_instr.data(), l_memInstr);
    //     cout << "Square of the norm(b) is: " << l_dot << endl;

    int l_deviceId = 0;
    bool l_debug = false;
    if (argc > l_idx) {
        string l_option = argv[l_idx++];
        if (l_option == "--debug")
            l_debug = true;
        else
            l_deviceId = stoi(l_option);
    }

    if (argc > l_idx) l_deviceId = atoi(argv[l_idx++]);

    FPGA l_fpga(l_deviceId);
    l_fpga.xclbin(binaryFile);

    CGKernelControl l_kernelControl(&l_fpga);
    l_kernelControl.getCU("krnl_control");
    l_kernelControl.setMem(h_instr);

    KernelLoadNnz<CG_numChannels> l_kernelLoadAval(&l_fpga);
    l_kernelLoadAval.getCU("krnl_loadAval:{krnl_loadAval}");
    l_kernelLoadAval.setMem(h_nnzVal);

    KernelLoadCol l_kernelLoadPkApar(&l_fpga);
    l_kernelLoadPkApar.getCU("krnl_loadPkApar:{krnl_loadPkApar}");
    l_kernelLoadPkApar.setMem(h_parParam, reinterpret_cast<host_buffer_t<uint8_t>&>(h_pk));

    KernelLoadRbParam l_kernelLoadArbParam(&l_fpga);
    l_kernelLoadArbParam.getCU("krnl_loadArbParam:{krnl_loadArbParam}");
    l_kernelLoadArbParam.setMem(h_rbParam);

    CGKernelStoreApk<CG_dataType> l_kernelStoreApk(&l_fpga);
    l_kernelStoreApk.getCU("krnl_storeApk:{krnl_storeApk}");
    l_kernelStoreApk.setMem(h_pk, h_Apk);

    CGKernelUpdatePk<CG_dataType> l_kernelUpdatePk(&l_fpga);
    l_kernelUpdatePk.getCU("krnl_update_pk");
    l_kernelUpdatePk.setMem(h_pk, h_zk);

    CGKernelUpdateRkJacobi<CG_dataType> l_kernelUpdateRkJacobi(&l_fpga);
    l_kernelUpdateRkJacobi.getCU("krnl_update_rk_jacobi");
    l_kernelUpdateRkJacobi.setMem(h_rk, h_zk, h_jacobi, h_Apk);

    CGKernelUpdateXk<CG_dataType> l_kernelUpdateXk(&l_fpga);
    l_kernelUpdateXk.getCU("krnl_update_xk");
    l_kernelUpdateXk.setMem(h_xk, h_pk);

    vector<Kernel> l_kernels;
    l_kernels.push_back(l_kernelControl);
    l_kernels.push_back(l_kernelLoadArbParam);
    l_kernels.push_back(l_kernelLoadAval);
    l_kernels.push_back(l_kernelLoadPkApar);
    l_kernels.push_back(l_kernelStoreApk);
    l_kernels.push_back(l_kernelUpdatePk);
    l_kernels.push_back(l_kernelUpdateXk);
    l_kernels.push_back(l_kernelUpdateRkJacobi);

    double l_runTime = Kernel::run(l_kernels);

    l_kernelControl.getMem();
    l_kernelUpdateXk.getMem();
    l_kernelUpdateRkJacobi.getMem();

    int lastIter = 0;
    uint64_t finalClock = 0;
    for (int i = 0; i < l_maxIter; i++) {
        lastIter = i;
        l_cgInstr.load(h_instr.data() + (i + 1) * CG_instrBytes, l_memInstr);
        if (l_debug) {
            cout << l_cgInstr << endl;
        }
        if (l_cgInstr.getMaxIter() == 0) {
            break;
        }
        finalClock = l_cgInstr.getClock();
    }
    cout << "HW execution time is: " << finalClock * HW_CLK << "s." << endl;
    float l_padRatio = (float)(l_info[5]) / (float)(l_info[2]);

    if (l_debug) {
        writeBin(l_vecFilePath + "/rk.dat", h_rk.size() * sizeof(CG_dataType), h_rk);
        writeBin(l_vecFilePath + "/xk.dat", h_xk.size() * sizeof(CG_dataType), h_xk);
    }

    int err = 0;
    compare(h_x.size(), h_x.data(), h_xk.data(), err, l_debug);
    cout << "matrix name, original rows/cols, original NNZs, padded rows/cols, padded NNZs, padding ";
    cout << "ratio, num of iterations, total run time[sec], time[ms]/run, num_mismatches";
    cout << endl;
    cout << "DATA_CSV:," << l_mtxName << "," << l_info[0] << "," << l_info[2];
    cout << "," << l_info[4] << "," << l_info[5] << "," << l_padRatio;
    cout << "," << lastIter + 1 << "," << l_runTime << "," << (float)l_runTime * 1000 / (lastIter + 1);
    cout << "," << err << endl;
    if (err == 0) {
        cout << "Test pass!" << endl;
        return EXIT_SUCCESS;
    } else {
        cout << "Test failed! There are in total " << err << " mismatches in the solution." << endl;
        return EXIT_FAILURE;
    }
}
