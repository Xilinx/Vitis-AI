/**********
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
 * **********/

#include "gen_gemm.hpp"
#include "api_test.hpp"

int main(int argc, char** argv) {
    //############  UI and GEMM problem size  ############
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  host.exe <path/blas.xclbin> M K N \n"
                  << "  Examples:\n"
                  << "    host.exe   out_hw/blas.xclbin\n"
                  << "    host.exe   out_hw/blas.xclbin  256 256 256\n";
        exit(2);
    }
    int l_argIdx = 1;
    std::string l_xclbinFile(argv[l_argIdx]);

    // Row major  C  M rows N cols  =  A  M rows K cols  *  B  K rows N cols
    //   MatType - tensor like type to allocate/store/align memory; you can use your own type instead
    //   Min size is the array edge (e.g., 32 on ku115), see GenGemm::check() for example of arg checking functions
    unsigned int l_ddrW = BLAS_ddrWidth;
    // the smallest matrices for flow testing
    unsigned int l_M = l_ddrW * BLAS_gemmMBlocks, l_K = l_ddrW * BLAS_gemmKBlocks, l_N = l_ddrW * BLAS_gemmNBlocks;
    if (argc > ++l_argIdx) {
        l_M = atoi(argv[l_argIdx]);
    }
    if (argc > ++l_argIdx) {
        l_K = atoi(argv[l_argIdx]);
    }
    if (argc > ++l_argIdx) {
        l_N = atoi(argv[l_argIdx]);
    }
    int32_t l_postScaleVal = 1, l_postScaleShift = 0;

    int32_t l_postScale = (l_postScaleVal << 8) | (l_postScaleShift & 0x000000ff);

    int l_deviceId = 0;
    if (argc > ++l_argIdx) {
        l_deviceId = atoi(argv[l_argIdx]);
    }
    //############  Client code - prepare the gemm problem input  ############
    GenGemm l_gemm;

    if (!l_gemm.check(l_M, l_K, l_N, l_K, l_N, l_N, l_N)) {
        return EXIT_FAILURE;
    }

    ProgramType l_program[BLAS_numKernels]; // Holds instructions and controls memory allocation
    ProgramType l_program_golden;

    std::string l_handleA[BLAS_numKernels];
    std::string l_handleB[BLAS_numKernels];
    std::string l_handleC[BLAS_numKernels];

    for (int i = 0; i < BLAS_numKernels; ++i) {
        l_handleA[i] = "A" + std::to_string(i);
        l_handleB[i] = "B" + std::to_string(i);
        l_handleC[i] = "C" + std::to_string(i);

        l_gemm.addInstr(l_program[i], l_M, l_K, l_N, l_K, l_N, l_N, l_N, l_postScale, l_handleA[i], l_handleB[i],
                        l_handleC[i], l_handleC[i], false);

        std::cout << "In kernel " << i << " ";
        std::cout << "Added instruction GEMM (" << l_M << "x" << l_K << " * " << l_K << "x" << l_N << ") \n";
    }
    l_gemm.addInstr(l_program_golden, l_M, l_K, l_N, l_K, l_N, l_N, l_N, l_postScale, l_handleA[0], l_handleB[0],
                    l_handleC[0], l_handleC[0], true);

    //############  Run FPGA accelerator  ############

    double l_timeApiInMs = run_hw_test(l_xclbinFile, l_program, l_deviceId);

    //############  Get the exact kernel time from HW cycle counters on the accelerator  ############
    float l_boardFreqMHz = getBoardFreqMHz(l_xclbinFile);
    unsigned long int l_Ops = 2ull * l_M * l_N * l_K + l_M * l_N * 3;
    unsigned long int l_Parallel_Ops = 2ull * l_M * l_N * l_K;
    KargsType l_kargsRes[BLAS_numKernels];
    KargsOpType l_op[BLAS_numKernels];
    xf::blas::InstrResArgs l_instrRes[BLAS_numKernels];
    unsigned long int l_cycleCount[BLAS_numKernels];
    unsigned long int l_maxCycleCount = 0;
    double l_timeKernelInMs[BLAS_numKernels];
    double l_maxTimeKernelInMs = 0;
    double l_perfKernelInTops[BLAS_numKernels];
    double l_totalPerfKernelInTops = 0;
    double l_perfApiInTops;
    double l_timeMsAt100pctEff;
    double l_effKernelPct;
    double l_effApiPct;

    for (int i = 0; i < BLAS_numKernels; ++i) {
        l_op[i] = l_kargsRes[i].load(l_program[i].getBaseResAddr(), 0);
        assert(l_op[i] == KargsType::OpResult);
        l_instrRes[i] = l_kargsRes[i].getInstrResArgs();
        l_cycleCount[i] = l_instrRes[i].getDuration();
        l_maxCycleCount = (l_cycleCount[i] > l_maxCycleCount) ? l_cycleCount[i] : l_maxCycleCount;
        l_timeKernelInMs[i] = l_cycleCount[i] / (l_boardFreqMHz * 1e6) * 1e3;
        l_maxTimeKernelInMs = (l_timeKernelInMs[i] > l_maxTimeKernelInMs) ? l_timeKernelInMs[i] : l_maxTimeKernelInMs;
        l_perfKernelInTops[i] = l_Ops / (l_timeKernelInMs[i] * 1e-3) / 1e12;
        l_totalPerfKernelInTops += l_perfKernelInTops[i];
    }
    l_perfApiInTops = (l_Ops * BLAS_numKernels) / (l_timeApiInMs * 1e-3) / 1e12;
    l_timeMsAt100pctEff = l_Parallel_Ops / 2 / BLAS_ddrWidth / BLAS_ddrWidth / (l_boardFreqMHz * 1e6) * 1e3;
    l_effKernelPct = (100 * l_timeMsAt100pctEff / l_maxTimeKernelInMs < 100)
                         ? (100 * l_timeMsAt100pctEff / l_maxTimeKernelInMs)
                         : 100;
    l_effApiPct = 100 * l_timeMsAt100pctEff / l_timeApiInMs;
    // Show time, Tops in csv format
    std::cout << std::string("DATA_CSV:,DdrWidth,Freq,M,K,N,") + "Ops,KernelCycles," + "TimeKernelMs,TimeApiMs," +
                     "EffKernelPct,EffApiPct," + "PerfKernelTops,PerfApiTops\n"
              << "DATA_CSV:," << BLAS_ddrWidth << "," << l_boardFreqMHz << "," << l_M << "," << l_K << "," << l_N << ","
              << l_Ops * BLAS_numKernels << "," << l_maxCycleCount << "," << l_maxTimeKernelInMs << "," << l_timeApiInMs
              << "," << l_effKernelPct << "," << l_effApiPct << "," << l_totalPerfKernelInTops << "," << l_perfApiInTops
              << std::endl;
    float l_TolRel = 1e-3, l_TolAbs = 1e-5;
    compareMultiInstrs(l_TolRel, l_TolAbs, l_program_golden, l_program[0]);
    return EXIT_SUCCESS;
}
