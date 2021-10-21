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
#include "cgSolverGemv.hpp"
#include "cgInstr.hpp"

using namespace std;

int main(int argc, char** argv) {
    if (argc < 6 || argc > 7) {
        cout << "Usage: " << argv[0]
             << " <XCLBIN File> <Max Iteration> <Tolerence> <Matrix Dim> <Matrix Path> [device id]" << endl;
        return EXIT_FAILURE;
    }

    uint32_t l_index = 1;

    string binaryFile = argv[l_index++];
    int l_maxIter = atoi(argv[l_index++]);
    CG_dataType l_tol = atof(argv[l_index++]);

    int vecSize = atoi(argv[l_index++]);
    assert(vecSize % CG_parEntries == 0);
    string filepath = argv[l_index++];
    int l_deviceId = 0;
    if (argc > l_index) l_deviceId = atoi(argv[l_index++]);
    FPGA fpga(l_deviceId);
    fpga.xclbin(binaryFile);

    CgSolverGemv<CG_dataType, CG_instrBytes, CG_parEntries, CG_numChannels, xf::hpc::MemInstr<CG_instrBytes>,
                 xf::hpc::cg::CGSolverInstr<CG_dataType> >
        solver(&fpga, l_maxIter, l_tol);
    solver.setA(filepath + "A.mat", vecSize);
    solver.setB(filepath + "b.mat");
    int lastIter = 0;
    uint64_t finalClock = 0;
    solver.solve(lastIter, finalClock);
    double l_runTime = finalClock * HW_CLK;
    cout << "The HW efficiency is: "
         << 100.0 * vecSize * vecSize * lastIter / CG_parEntries / CG_numChannels / finalClock << '%' << endl;

    int err = solver.verify(filepath + "x.mat");

    cout << "Rows/Cols\t No. iterations\t Total Execution time[sec]\t Time per Iter[ms]\t No. mismatches" << endl;
    cout << vecSize << '\t' << lastIter << '\t' << l_runTime << '\t' << (float)l_runTime * 1000 / lastIter << '\t'
         << err << endl;

    if (err == 0) {
        cout << "Solution verified." << endl;
        return EXIT_SUCCESS;
    } else {
        cout << "There are in total " << err << " mismatches in the solution." << endl;
        return EXIT_FAILURE;
    }
}
