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

/**
 * @file parallelCscmv.cpp
 * @brief main function for cscmv kernel host code
 *
 * This file is part of Vitis SPARSE Library.
 */
#include <cmath>
#include <cstdlib>
#include <string>
#include <iostream>
#include "L1_utils.hpp"
#include "L2_definitions.hpp"

// This extension file is required for stream APIs
#include "CL/cl_ext_xilinx.h"
// This file is required for OpenCL C++ wrapper APIs
#include "xcl2.hpp"

using namespace std;
using namespace xf::sparse;
int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "ERROR: passed " << argc << " arguments, expected at least 3 arguments." << endl;
        cout << "  Usage: cscmv.exe cscmv.xclbin kernel_config_binary_file [num_iterations]" << endl;
        return EXIT_FAILURE;
    }

    string l_xclbinFile = argv[1];
    string l_binFile = argv[2];
    unsigned int l_iterations = 1;
    if (argc == 4) {
        l_iterations = atoi(argv[3]);
    }

    // read and interpret kernel config file
    ifstream l_if(l_binFile.c_str(), ios::binary);
    if (!l_if.is_open()) {
        cout << "ERROR: failed to open file " << l_binFile << endl;
        return EXIT_FAILURE;
    }
    cout << "INFO: loading " << l_binFile << endl;
    ProgramType l_prog;
    RunConfigType l_runConfig;

    l_runConfig.readBinFile(l_if, l_prog);
    l_if.close();

    // OpenCL host code begins
    cl_int l_err;
    cl::Device l_device;
    cl::Context l_context;
    cl::CommandQueue l_cmdQueue;
    cl::Program l_clProgram;

    auto l_devices = xcl::get_xil_devices();
    l_device = l_devices[0];

    OCL_CHECK(l_err, l_context = cl::Context(l_device, NULL, NULL, NULL, &l_err));
    OCL_CHECK(l_err,
              l_cmdQueue = cl::CommandQueue(l_context, l_device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &l_err));
    cl::Program::Binaries l_bins = xcl::import_binary_file(l_xclbinFile);
    l_devices.resize(1);

    OCL_CHECK(l_err, l_clProgram = cl::Program(l_context, l_devices, l_bins, NULL, &l_err));

    cl::Kernel l_loadColKernel;
    cl::Kernel l_readWriteHbmKernel[SPARSE_hbmChannels / 8];

    OCL_CHECK(l_err, l_loadColKernel = cl::Kernel(l_clProgram, "loadColKernel:{loadColKernel_0}", &l_err));
    for (unsigned int i = 0; i < SPARSE_hbmChannels / 8; ++i) {
        string l_extName = "_" + to_string(i) + "}";
        string l_kName = "readWriteHbmKernel:{readWriteHbmKernel";
        l_kName += l_extName;
        OCL_CHECK(l_err, l_readWriteHbmKernel[i] = cl::Kernel(l_clProgram, l_kName.c_str(), &l_err));
    }
    // trigger all CUs to run kernel configs;
    TimePointType l_tp[10];
    unsigned int l_tpIdx = 0;
    double l_krnApiTime = 0;
    double l_totalTime = 0;

    void* l_loadColValPtr = l_runConfig.getColVecAddr();
    memcpy((reinterpret_cast<char*>(l_loadColValPtr)) + 4, &l_iterations, sizeof(unsigned int));
    unsigned long long l_loadColValSz = l_prog.getBufSz(l_loadColValPtr);
    cl::Buffer l_loadColValBuf;
    OCL_CHECK(l_err, l_loadColValBuf = cl::Buffer(l_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, l_loadColValSz,
                                                  l_loadColValPtr, &l_err));
    OCL_CHECK(l_err, l_err = l_loadColKernel.setArg(0, l_loadColValBuf));
    OCL_CHECK(l_err, l_err = l_cmdQueue.enqueueMigrateMemObjects({l_loadColValBuf}, 0));

    void* l_loadColPtrPtr = l_runConfig.getColPtrAddr();
    memcpy((reinterpret_cast<char*>(l_loadColPtrPtr)) + 4, &l_iterations, sizeof(unsigned int));
    unsigned long long l_loadColPtrSz = l_prog.getBufSz(l_loadColPtrPtr);
    cl::Buffer l_loadColPtrBuf;
    OCL_CHECK(l_err, l_loadColPtrBuf = cl::Buffer(l_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, l_loadColPtrSz,
                                                  l_loadColPtrPtr, &l_err));
    OCL_CHECK(l_err, l_err = l_loadColKernel.setArg(1, l_loadColPtrBuf));
    OCL_CHECK(l_err, l_err = l_cmdQueue.enqueueMigrateMemObjects({l_loadColPtrBuf}, 0));

    void* l_readHbmPtr[SPARSE_hbmChannels];
    unsigned long long l_readHbmSz[SPARSE_hbmChannels];
    cl::Buffer l_readHbmBuf[SPARSE_hbmChannels];
    void* l_rowResPtr[SPARSE_hbmChannels];
    unsigned long long l_rowResSz[SPARSE_hbmChannels];
    cl::Buffer l_rowResBuf[SPARSE_hbmChannels];
    for (unsigned int ch = 0; ch < SPARSE_hbmChannels; ++ch) {
        l_readHbmPtr[ch] = l_runConfig.getRdHbmAddr(ch);
        memcpy(reinterpret_cast<char*>(l_readHbmPtr[ch]) + 4, &l_iterations, sizeof(unsigned int));
        l_readHbmSz[ch] = l_prog.getBufSz(l_readHbmPtr[ch]);
        OCL_CHECK(l_err, l_readHbmBuf[ch] = cl::Buffer(l_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                       l_readHbmSz[ch], l_readHbmPtr[ch], &l_err));
        unsigned int l_argIdx = ch % 8;
        OCL_CHECK(l_err, l_err = l_readWriteHbmKernel[ch / 8].setArg(l_argIdx * 2, l_readHbmBuf[ch]));
        OCL_CHECK(l_err, l_err = l_cmdQueue.enqueueMigrateMemObjects({l_readHbmBuf[ch]}, 0));

        l_rowResPtr[ch] = l_runConfig.getWrHbmAddr(ch);
        l_rowResSz[ch] = l_prog.getBufSz(l_rowResPtr[ch]);
        if (l_rowResSz[ch] != 0) {
            OCL_CHECK(l_err, l_rowResBuf[ch] = cl::Buffer(l_context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                                          l_rowResSz[ch], l_rowResPtr[ch], &l_err));
            OCL_CHECK(l_err, l_err = l_readWriteHbmKernel[ch / 8].setArg(l_argIdx * 2 + 1, l_rowResBuf[ch]));
        } else {
            OCL_CHECK(l_err, l_err = l_readWriteHbmKernel[ch / 8].setArg(l_argIdx * 2 + 1, l_readHbmBuf[ch]));
        }
    }

    l_cmdQueue.finish();

    // finishing transfer data from host to device, kick off kernels
    OCL_CHECK(l_err, l_err = l_cmdQueue.enqueueTask(l_loadColKernel));
    for (unsigned int kr = 0; kr < SPARSE_hbmChannels / 8; ++kr) {
        OCL_CHECK(l_err, l_err = l_cmdQueue.enqueueTask(l_readWriteHbmKernel[kr]));
    }

    l_tp[l_tpIdx] = std::chrono::high_resolution_clock::now();
    l_cmdQueue.finish();
    l_krnApiTime += showTimeData("Kernel run time: ", l_tp[l_tpIdx], l_tp[l_tpIdx + 1]);
    l_totalTime += l_krnApiTime;
    l_tpIdx++;

    // read data back to host meory
    for (unsigned int ch = 0; ch < SPARSE_hbmChannels; ++ch) {
        if (l_rowResSz[ch] != 0) {
            OCL_CHECK(l_err,
                      l_err = l_cmdQueue.enqueueMigrateMemObjects({l_rowResBuf[ch]}, CL_MIGRATE_MEM_OBJECT_HOST));
        }
    }
    l_cmdQueue.finish();
    l_totalTime += showTimeData("Data read back time: ", l_tp[l_tpIdx], l_tp[l_tpIdx + 1]);
    l_tpIdx++;

    unsigned int l_matNnzs = l_runConfig.nnzs();
    unsigned int l_matRows = l_runConfig.rows();
    unsigned int l_matCols = l_runConfig.cols();
    double l_krnPerf = (l_matNnzs * 2) * l_iterations / l_krnApiTime / 1e6;
    // double l_krnApiPerf = (l_krnApiTime == 0)? 0:  l_matNnzs * 2 / l_krnApiTime / 1e6;
    double l_apiPerf = (l_totalTime == 0) ? 0 : l_matNnzs * 2 * l_iterations / l_totalTime / 1e6;
    size_t l_sepLoc = l_binFile.rfind('/', l_binFile.length());
    string l_matName = (l_sepLoc != string::npos) ? l_binFile.substr(l_sepLoc + 1, l_binFile.length() - l_sepLoc) : "";
    size_t l_extLoc = l_matName.find_last_of(".");
    l_matName = l_matName.substr(0, l_extLoc);
    cout << "DATA_CSV:, matrix, row, cols, nnzs, KernelRunTime(ms), TotalRunTime(ms),KernelPerf(GFlops/Sec), "
            "API_Perf(GFlops/Sec)"
         << endl;
    cout << "DATA_CSV:," << l_matName << "," << l_matRows << "," << l_matCols << "," << l_matNnzs << "," << l_krnApiTime
         << "," << l_totalTime << "," << l_krnPerf << "," << l_apiPerf << endl;

    // compare l_yOut against l_y

    unsigned int l_checkErrs = 0;
    l_checkErrs = l_runConfig.checkRowRes();
    if (l_checkErrs != 0) {
        cout << "ERROR: there are total " << l_checkErrs << " mismatches between outputs and golden references."
             << endl;
        return EXIT_FAILURE;
    } else {
        cout << "Test Pass!" << endl;
        return EXIT_SUCCESS;
    }
}
