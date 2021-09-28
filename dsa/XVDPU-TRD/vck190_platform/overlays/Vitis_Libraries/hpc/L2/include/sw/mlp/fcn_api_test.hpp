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
#ifndef XF_HPC_FCN_API_TEST_HPP
#define XF_HPC_FCN_API_TEST_HPP

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <chrono>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <stdio.h> // fgets for popen

#include "fpga.hpp"
#include "gen_bin.hpp"

float getBoardFreqMHz(std::string p_xclbinFile) {
    std::string l_freqCmd = "xclbinutil --info --input " + p_xclbinFile;
    ;
    float l_freq = -1;
    char l_lineBuf[256];
    std::shared_ptr<FILE> l_pipe(popen(l_freqCmd.c_str(), "r"), pclose);
    // if (!l_pipe) throw std::runtime_error("ERROR: popen(" + l_freqCmd + ") failed");
    if (!l_pipe) std::cout << ("ERROR: popen(" + l_freqCmd + ") failed");
    bool l_nextLine_isFreq = false;
    while (l_pipe && fgets(l_lineBuf, 256, l_pipe.get())) {
        std::string l_line(l_lineBuf);
        // std::cout << "DEBUG: read line " << l_line << std::endl;
        if (l_nextLine_isFreq) {
            std::string l_prefix, l_val, l_mhz;
            std::stringstream l_ss(l_line);
            l_ss >> l_prefix >> l_val >> l_mhz;
            l_freq = std::stof(l_val);
            assert(l_mhz == "MHz");
            break;
        } else if (l_line.find("Type:      DATA") != std::string::npos) {
            l_nextLine_isFreq = true;
        }
    }
    if (l_freq == -1) {
        l_freq = 200;
        std::cout << "INFO: Failed to get board frequency by xbutil. This is normal for cpu and hw emulation, using "
                     "200 MHz for reporting.\n";
    }
    return (l_freq);
}

double run_hw_test(std::string l_xclbinFile, ProgramType* l_program, unsigned int p_deviceId = 0) {
    xf::blas::MemDesc l_memDesc[BLAS_numKernels];

    for (int i = 0; i < BLAS_numKernels; ++i) {
        l_memDesc[i] = l_program[i].getMemDesc();
    }

    //############  Runtime reporting Infra  ############
    TimePointType l_tp[10];
    unsigned int l_tpIdx = 0;
    l_tp[l_tpIdx] = std::chrono::high_resolution_clock::now();

    //############  Run FPGA accelerator  ############
    // Init FPGA
    xf::blas::Fpga l_fpga(p_deviceId);

    if (l_fpga.loadXclbin(l_xclbinFile)) {
        std::cout << "INFO: created kernels" << std::endl;
    } else {
        std::cerr << "ERROR: failed to load " + l_xclbinFile + "\n";
        return EXIT_FAILURE;
    }
    showTimeData("loadXclbin", l_tp[l_tpIdx], l_tp[l_tpIdx + 1]);
    l_tpIdx++;

    for (unsigned int i = 0; i < BLAS_numKernels; ++i) {
        if (!l_fpga.createKernel(i, "fcnKernel")) {
            std::cerr << "ERROR: failed to create kernel " << i << std::endl;
        }
    }
    showTimeData("create kernels", l_tp[l_tpIdx], l_tp[l_tpIdx + 1]);
    l_tpIdx++;

    for (unsigned int i = 0; i < BLAS_numKernels; ++i) {
        if (!l_fpga.createBufferForKernel(i, l_memDesc[i])) {
            std::cerr << "ERROR: failed to create buffer for kernel " << i << std::endl;
        }
    }
    showTimeData("create buffers", l_tp[l_tpIdx], l_tp[l_tpIdx + 1]);
    l_tpIdx++;

    // Transfer data to FPGA
    for (unsigned int i = 0; i < BLAS_numKernels; ++i) {
        if (l_fpga.copyToKernel(i)) {
            std::cout << "INFO: transferred data to kernel " << i << std::endl;
        } else {
            std::cerr << "ERROR: failed to transfer data to kernel" << i << std::endl;
            return EXIT_FAILURE;
        }
    }
    showTimeData("copy to kernels", l_tp[l_tpIdx], l_tp[l_tpIdx + 1]);
    l_tpIdx++;

    // launch kernels
    for (unsigned int i = 0; i < BLAS_numKernels; ++i) {
        if (l_fpga.callKernel(i)) {
            std::cout << "INFO: Executed kernel " << i << std::endl;
        } else {
            std::cerr << "ERROR: failed to call kernel " << i << std::endl;
            return EXIT_FAILURE;
        }
    }
    showTimeData("call kernels", l_tp[l_tpIdx], l_tp[l_tpIdx + 1]);
    l_tpIdx++;
    l_fpga.finish();

    // Transfer data back to host
    for (unsigned int i = 0; i < BLAS_numKernels; ++i) {
        if (l_fpga.copyFromKernel(i)) {
            std::cout << "INFO: Transferred data from kernel" << i << std::endl;
        } else {
            std::cerr << "ERROR: failed to transfer data from kernel " << i << std::endl;
            return EXIT_FAILURE;
        }
    }
    l_fpga.finish();

    showTimeData("copyFromFpga", l_tp[l_tpIdx], l_tp[l_tpIdx + 1]);
    l_tpIdx++;
    showTimeData("total", l_tp[0], l_tp[l_tpIdx]);
    l_tpIdx++;
    double l_timeApiInMs = -1;
    showTimeData("subtotalFpga", l_tp[2], l_tp[l_tpIdx], &l_timeApiInMs);
    l_tpIdx++; // Host->DDR, kernel, DDR->host

    return l_timeApiInMs;
}

void compareMultiInstrs(float p_TolRel, float p_TolAbs, ProgramType& p_Program0, ProgramType& p_Program1) {
#if BLAS_runGemm == 1
    GenGemm l_gemm;
#endif
#if BLAS_runFcn == 1
    GenFcn l_fcn;
#endif
    bool l_isLastOp = false;
    bool l_compareOk = true;
    KargsType l_kargs0, l_kargs1;
    unsigned int l_pc = 0;

    do {
        KargsOpType l_op0 = l_kargs0.load(p_Program0.getBaseInstrAddr(), l_pc);
        KargsOpType l_op1 = l_kargs1.load(p_Program1.getBaseInstrAddr(), l_pc);
        if (l_op1 == KargsType::OpResult) {
            break;
        }
        assert(l_op0 == l_op1);
        switch (l_op0) {
#if BLAS_runGemm == 1
            case KargsType::OpGemm: {
                GemmArgsType l_gemmArgs = l_kargs0.getGemmArgs();
                bool l_opOk = l_gemm.compare(p_TolRel, p_TolAbs, p_Program0, p_Program1, l_gemmArgs);
                l_compareOk = l_compareOk && l_opOk;
                break;
            }
#endif
#if BLAS_runFcn == 1
            case KargsType::OpFcn: {
                FcnArgsType l_fcnArgs = l_kargs0.getFcnArgs();
                bool l_opOk = l_fcn.compare(p_TolRel, p_TolAbs, p_Program0, p_Program1, l_fcnArgs);
                l_compareOk = l_compareOk && l_opOk;
                break;
            }
#endif
            default:
                break;
        }
        l_pc += l_kargs0.getInstrWidth();
    } while (!l_isLastOp);

    if (!l_compareOk) {
        std::cout << "fail\n" << std::endl;
    } else {
        std::cout << "pass\n" << std::endl;
    }
}

#endif
