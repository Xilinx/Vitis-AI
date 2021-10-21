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
#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "fd_bs_lv_util.hpp"
#include "xcl2.hpp"
#include "xf_utils_sw/logger.hpp"

using namespace fd;

// Temporary copy of this macro definition until new xcl2.hpp is used
#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

int main(int argc, char* argv[]) {
    // Sanity check the input parameters
    if (argc != 3) {
        std::cout << "ERROR: passed " << (argc - 1) << " arguments instead of 2, exiting" << std::endl;
        std::cout << "  Usage:" << std::endl;
        std::cout << "    fd_bs_lv_test.exe <path_to_xclbin/xclbin_file> <testcase>" << std::endl;
        std::cout << "  Example:" << std::endl;
        std::cout << "    fd_bs_lv_test.exe path-to-xclbin/fd_bs_lv_test.xclbin data/case0" << std::endl;
        return EXIT_FAILURE;
    }
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);

    // Parse arguments
    unsigned int argIdx = 1;
    string xclbin_file(argv[argIdx++]);
    string testcase_file(argv[argIdx++]);

    // Assemble directory name of reference data based on parameters
    std::stringstream sstm;
    sstm << testcase_file;
    string path = sstm.str();

    // Read testcase parameters
    testcaseParams params;
    std::cout << "Loading testcase..." << std::endl;
    FdBsLvUtil<FD_DATA_TYPE, FD_DATA_WORDS_IN_DDR> util;
    if (util.ReadTestcaseParameters(path + "/parameters.csv", params)) return EXIT_FAILURE;
    std::cout << "    N     = " << params.solverN << std::endl;
    std::cout << "    M     = " << params.solverM << std::endl;
    std::cout << "    theta = " << params.solverTheta << std::endl;
    std::cout << "    s     = " << params.modelS << std::endl;
    std::cout << "    k     = " << params.modelK << std::endl;

    // Shortnames for readability
    unsigned int N = params.solverN;
    unsigned int M = params.solverM;

    // Vectors for parameter storage.  These use an aligned allocator in order
    // to avoid an additional copy of the host memory into the device
    std::vector<FD_DATA_TYPE, aligned_allocator<FD_DATA_TYPE> > xGrid(N);
    std::vector<FD_DATA_TYPE, aligned_allocator<FD_DATA_TYPE> > tGrid(M);
    std::vector<FD_DATA_TYPE, aligned_allocator<FD_DATA_TYPE> > sigma(N * M);
    std::vector<FD_DATA_TYPE, aligned_allocator<FD_DATA_TYPE> > rate(M);
    std::vector<FD_DATA_TYPE, aligned_allocator<FD_DATA_TYPE> > initialCondition(N);
    std::vector<FD_DATA_TYPE, aligned_allocator<FD_DATA_TYPE> > solution(N);
    std::vector<FD_DATA_TYPE, aligned_allocator<FD_DATA_TYPE> > reference(N);

    // Check selected xclbin file matches testcase
    sstm.str("");
    sstm << "_N" << N << "_M" << M;
    if (xclbin_file.find(sstm.str()) == string::npos) {
        std::cout << "Specified xclbin has wrong size for testcase\n";
        return EXIT_FAILURE;
    }

    // Read precomputed array data and golden reference from .csv files
    std::cout << "Loading precomputed data from Python reference model..." << std::endl;
    if (util.ReadVector(path + "/xGrid.csv", xGrid, N)) return EXIT_FAILURE;
    if (util.ReadVector(path + "/tGrid.csv", tGrid, M)) return EXIT_FAILURE;
    if (util.ReadVector(path + "/sigma.csv", sigma, N * M)) return EXIT_FAILURE;
    if (util.ReadVector(path + "/rate.csv", rate, M)) return EXIT_FAILURE;
    if (util.ReadVector(path + "/initialCondition.csv", initialCondition, N)) return EXIT_FAILURE;
    if (util.ReadVector(path + "/reference.csv", reference, N)) return EXIT_FAILURE;

    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl_int err;

    cl::Context context(device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    logger.logCreateCommandQueue(err);

    // Load the binary file (using function from xcl2.cpp)
    cl::Program::Binaries bins = xcl::import_binary_file(xclbin_file);

    devices.resize(1);
    cl::Program program(context, devices, bins, NULL, &err);
    logger.logCreateProgram(err);
    cl::Kernel krnl_fd_bs_lv(program, "fd_bs_lv_kernel", &err);
    logger.logCreateKernel(err);

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication
    OCL_CHECK(err, cl::Buffer buffer_xGrid(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, N * sizeof(FD_DATA_TYPE),
                                           xGrid.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_tGrid(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, M * sizeof(FD_DATA_TYPE),
                                           tGrid.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_sigma(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                           N * M * sizeof(FD_DATA_TYPE), sigma.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_rate(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, M * sizeof(FD_DATA_TYPE),
                                          rate.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_initialCondition(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                      N * sizeof(FD_DATA_TYPE), initialCondition.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_solution(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                              N * sizeof(FD_DATA_TYPE), solution.data(), &err));

    // Set the arguments
    OCL_CHECK(err, err = krnl_fd_bs_lv.setArg(0, buffer_xGrid));
    OCL_CHECK(err, err = krnl_fd_bs_lv.setArg(1, buffer_tGrid));
    OCL_CHECK(err, err = krnl_fd_bs_lv.setArg(2, buffer_sigma));
    OCL_CHECK(err, err = krnl_fd_bs_lv.setArg(3, buffer_rate));
    OCL_CHECK(err, err = krnl_fd_bs_lv.setArg(4, buffer_initialCondition));
    OCL_CHECK(err, err = krnl_fd_bs_lv.setArg(5, (float)params.solverTheta));
    OCL_CHECK(err, err = krnl_fd_bs_lv.setArg(6, (FD_DATA_TYPE)params.boundaryLower));
    OCL_CHECK(err, err = krnl_fd_bs_lv.setArg(7, (FD_DATA_TYPE)params.boundaryUpper));
    OCL_CHECK(err, err = krnl_fd_bs_lv.setArg(8, M));
    OCL_CHECK(err, err = krnl_fd_bs_lv.setArg(9, buffer_solution));

    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_xGrid}, 0));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_tGrid}, 0));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_sigma}, 0));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_rate}, 0));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_initialCondition}, 0));

    // Launch the kernel, timing only the kernel run
    std::cout << "Launching kernel..." << std::endl;
    uint64_t nstimestart, nstimeend;
    cl::Event event;
    OCL_CHECK(err, err = q.enqueueTask(krnl_fd_bs_lv, NULL, &event));
    OCL_CHECK(err, err = q.finish());

    // Retrieve the solution data
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_solution}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q.finish());

    // Calculate elapsed time
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
    auto duration_nanosec = nstimeend - nstimestart;
    std::cout << "  Duration returned by profile API is " << (duration_nanosec * (1.0e-6)) << " ms **** " << std::endl;
    // OPENCL HOST CODE AREA END

    // Calculate absolute worst difference across whole grid compared to reference
    std::vector<FD_DATA_TYPE, aligned_allocator<FD_DATA_TYPE> > diff(N);
    FD_DATA_TYPE MaxDiff = util.CompareReference(solution, reference, N, diff);
    util.PrintVector("solution", solution, N);
    util.PrintVector("reference", reference, N);
    util.PrintVector("difference", diff, N);

    int ret = 0;
    if (std::abs(MaxDiff) > 0.003) {
        std::cout << "FAIL: MaxDiff = " << MaxDiff << std::endl;
        ret = 1;
    }

    ret ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);

    return ret;
}
