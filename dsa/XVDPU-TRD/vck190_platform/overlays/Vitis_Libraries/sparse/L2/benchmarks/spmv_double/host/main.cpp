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
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <chrono>
#include <cassert>

// This file is required for OpenCL C++ wrapper APIs
#include "blas/L1/tests/sw/include/binFiles.hpp"
#include "blas/L1/tests/sw/include/utils.hpp"
#include "hpc/L2/include/sw/fpga.hpp"
#include "spmvKernel.hpp"

using namespace std;

int main(int argc, char** argv) {
    if (argc < 5 || argc > 7) {
        cout << "Usage: " << argv[0] << " <XCLBIN File> <sigature_path> <vector_path> <mtx_name> [numRuns] [device id]"
             << endl;
        return EXIT_FAILURE;
    }

    int l_idx = 1;

    string l_xclbinFile = argv[l_idx++];
    string l_sigPath = argv[l_idx++];
    string l_vecPath = argv[l_idx++];
    string l_mtxName = argv[l_idx++];
    unsigned int l_numRuns = 1;
    if (argc > l_idx) l_numRuns = atoi(argv[l_idx++]);

    string l_sigFilePath = l_sigPath + "/" + l_mtxName;
    string l_vecFilePath = l_vecPath + "/" + l_mtxName;

    string l_sigFileNames[SPARSE_hbmChannels + 2];
    string l_vecFileNames[3];

    for (unsigned int i = 0; i < SPARSE_hbmChannels; ++i) {
        l_sigFileNames[i] = l_sigFilePath + "/nnzVal_" + to_string(i) + ".dat";
    }
    l_sigFileNames[SPARSE_hbmChannels] = l_sigFilePath + "/parParam.dat";
    l_sigFileNames[SPARSE_hbmChannels + 1] = l_sigFilePath + "/rbParam.dat";

    l_vecFileNames[0] = l_vecFilePath + "/inVec.dat";
    l_vecFileNames[1] = l_vecFilePath + "/refVec.dat";
    l_vecFileNames[2] = l_vecFilePath + "/outVec.dat";

    // I/O Data Vectors
    host_buffer_t<uint8_t> l_nnzBuf[SPARSE_hbmChannels];
    host_buffer_t<uint8_t> l_paramBuf[2];
    host_buffer_t<uint8_t> l_vecBuf[2];
    host_buffer_t<uint8_t> l_outVecBuf;

    for (unsigned int i = 0; i < SPARSE_hbmChannels + 2; ++i) {
        size_t l_bytes = getBinBytes(l_sigFileNames[i]);
        if (i < SPARSE_hbmChannels) {
            readBin<uint8_t, aligned_allocator<uint8_t> >(l_sigFileNames[i], l_bytes, l_nnzBuf[i]);
        } else {
            readBin<uint8_t, aligned_allocator<uint8_t> >(l_sigFileNames[i], l_bytes,
                                                          l_paramBuf[i - SPARSE_hbmChannels]);
        }
    }
    for (unsigned int i = 0; i < 2; ++i) {
        size_t l_bytes = getBinBytes(l_vecFileNames[i]);
        readBin<uint8_t, aligned_allocator<uint8_t> >(l_vecFileNames[i], l_bytes, l_vecBuf[i]);
    }
    unsigned int l_yRows = l_vecBuf[1].size() / sizeof(SPARSE_dataType);
    size_t l_outVecBytes = l_yRows * sizeof(SPARSE_dataType);
    l_outVecBuf.resize(l_outVecBytes);

    int l_deviceId = 0;
    if (argc > l_idx) l_deviceId = atoi(argv[l_idx++]);

    cout << "matrix name, original rows, original cols, original NNZs, padded rows, padded cols, padded NNZs, padding "
            "ratio, num of runs, total run time[sec], time[ms]/run"
         << endl;

    FPGA l_fpga(l_deviceId);
    l_fpga.xclbin(l_xclbinFile);

    KernelLoadNnz<SPARSE_hbmChannels> l_kernelLoadNnz(&l_fpga);
    l_kernelLoadNnz.getCU("loadNnzKernel:{krnl_loadNnz}");
    l_kernelLoadNnz.setMem(l_numRuns, l_nnzBuf);

    KernelLoadCol l_kernelLoadParX(&l_fpga);
    l_kernelLoadParX.getCU("loadParXkernel:{krnl_loadParX}");
    l_kernelLoadParX.setMem(l_numRuns, l_paramBuf[0], l_vecBuf[0]);

    KernelLoadRbParam l_kernelLoadRbParam(&l_fpga);
    l_kernelLoadRbParam.getCU("loadRbParamKernel:{krnl_loadRbParam}");
    l_kernelLoadRbParam.setMem(l_numRuns, l_paramBuf[1]);

    KernelStoreY l_kernelStoreY(&l_fpga);
    l_kernelStoreY.getCU("storeYkernel:{krnl_storeY}");
    l_kernelStoreY.setArgs(l_numRuns, l_yRows, l_outVecBuf);

    vector<Kernel> l_kernels;
    l_kernels.push_back(l_kernelLoadRbParam);
    l_kernels.push_back(l_kernelLoadParX);
    l_kernels.push_back(l_kernelLoadNnz);
    l_kernels.push_back(l_kernelStoreY);

    double l_runTime = Kernel::run(l_kernels);

    l_kernelStoreY.getMem();

    writeBin<uint8_t>(l_vecFileNames[2], l_yRows * sizeof(SPARSE_dataType),
                      reinterpret_cast<uint8_t*>(&(l_outVecBuf[0])));
    vector<uint32_t> l_info(6);
    readBin<uint32_t>(l_sigFilePath + "/info.dat", 6 * sizeof(uint32_t), l_info.data());
    float l_padRatio = (float)(l_info[5]) / (float)(l_info[2]);
    cout << "DATA_CSV:," << l_mtxName << "," << l_info[0] << "," << l_info[1] << "," << l_info[2];
    cout << "," << l_info[3] << "," << l_info[4] << "," << l_info[5] << "," << l_padRatio;
    cout << "," << l_numRuns << "," << l_runTime << "," << (float)l_runTime * 1000 / l_numRuns;
    cout << endl;

    int l_errs = 0;
    compare<SPARSE_dataType>(l_yRows, reinterpret_cast<SPARSE_dataType*>(l_outVecBuf.data()),
                             reinterpret_cast<SPARSE_dataType*>(l_vecBuf[1].data()), l_errs, true);
    if (l_errs == 0) {
        cout << "INFO: Test pass!" << endl;
        return EXIT_SUCCESS;
    } else {
        cout << "ERROR: Test failed! Out of total " << l_yRows << " entries, there are " << l_errs << " mismatches."
             << endl;
        return EXIT_FAILURE;
    }
}
