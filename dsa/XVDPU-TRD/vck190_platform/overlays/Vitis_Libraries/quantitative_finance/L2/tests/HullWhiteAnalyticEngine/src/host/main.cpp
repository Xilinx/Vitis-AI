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

#define TEST_KRNL_0
#define TEST_KRNL_1
#define TEST_KRNL_2

#ifndef HLS_TEST
#include "xcl2.hpp"
#endif
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include "ap_int.h"
#include "utils.hpp"
#include "hwa_engine_kernel.hpp"
#include "xf_utils_sw/logger.hpp"

#define XCL_BANK(n) (((unsigned int)(n)) | XCL_MEM_TOPOLOGY)

#define XCL_BANK0 XCL_BANK(0)
#define XCL_BANK1 XCL_BANK(1)
#define XCL_BANK2 XCL_BANK(2)
#define XCL_BANK3 XCL_BANK(3)
#define XCL_BANK4 XCL_BANK(4)
#define XCL_BANK5 XCL_BANK(5)
#define XCL_BANK6 XCL_BANK(6)
#define XCL_BANK7 XCL_BANK(7)
#define XCL_BANK8 XCL_BANK(8)
#define XCL_BANK9 XCL_BANK(9)
#define XCL_BANK10 XCL_BANK(10)
#define XCL_BANK11 XCL_BANK(11)
#define XCL_BANK12 XCL_BANK(12)
#define XCL_BANK13 XCL_BANK(13)
#define XCL_BANK14 XCL_BANK(14)
#define XCL_BANK15 XCL_BANK(15)
class ArgParser {
   public:
    ArgParser(int& argc, const char** argv) {
        for (int i = 1; i < argc; ++i) mTokens.push_back(std::string(argv[i]));
    }
    bool getCmdOption(const std::string option, std::string& value) const {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->mTokens.begin(), this->mTokens.end(), option);
        if (itr != this->mTokens.end() && ++itr != this->mTokens.end()) {
            value = *itr;
            return true;
        }
        return false;
    }

   private:
    std::vector<std::string> mTokens;
};

int main(int argc, const char* argv[]) {
    // tolerance
    TEST_DT tolerance = 1e-8;
    int failCnt = 0;

    std::cout << std::setprecision(10) << std::endl;
    std::cout << "\n----------------------HullWhite Analytic Engine (HWA)-----------------\n";
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    // cmd parser
    ArgParser parser(argc, argv);
    std::string xclbin_path;
#ifndef HLS_TEST
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }
    printf("xclbin Path=%s\n", xclbin_path.c_str());
#endif

    std::string mode_emu = "hw";
    if (std::getenv("XCL_EMULATION_MODE") != nullptr) {
        mode_emu = std::getenv("XCL_EMULATION_MODE");
    }
    std::cout << "Running in " << mode_emu << " mode" << std::endl;

    // list of test vectors, should be a mulitple of N_k0 etc...
    static const int maxOutputBuffer = 131072;
    static int maxk0 = maxOutputBuffer / N_k0;
    static int maxk1 = maxOutputBuffer / N_k1;
    static int maxk2 = maxOutputBuffer / N_k2;

    std::vector<int> testCases{16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    if (mode_emu == "hw_emu") {
        // limit test vectors for hardware emulation
        testCases.resize(1);
        maxk0 = 16;
        maxk1 = 16;
        maxk2 = 16;
    } else if (mode_emu == "sw_emu") {
        // limit test vectors for software emulation
        testCases.resize(5);
        maxk0 = 256;
        maxk1 = 256;
        maxk2 = 256;
    }

    // -------------setup k0 params---------------
    int err = 0;

    // Yield Curve Data
    TEST_DT rates[LEN] = {0.0020, 0.0050, 0.0070, 0.0110, 0.0150, 0.0180,
                          0.0200, 0.0220, 0.0250, 0.0288, 0.0310, 0.0340};
    TEST_DT times[LEN] = {0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 10.0, 20.0, 30.0};

    // Model Parameters
    TEST_DT a = 0.10;
    TEST_DT sigma = 0.01;

    // TestCases kernel 0 bond price
    // populate with random data
    // t between 0-15
    // T = t + 0-15
    TEST_DT t0[N_k0];
    TEST_DT T0[N_k0];
    for (int i = 0; i < N_k0; i++) {
        t0[i] = static_cast<TEST_DT>(rand()) / (static_cast<TEST_DT>(RAND_MAX / 15.0));
        T0[i] = t0[i] + 1.0 + static_cast<TEST_DT>(rand()) / (static_cast<TEST_DT>(RAND_MAX / 15.0));
    }

    // Testcases kernel 1 option price calls/puts
    TEST_DT t1[N_k1];
    TEST_DT T1[N_k1];
    TEST_DT S1[N_k1];
    TEST_DT K1[N_k1];
    int types1[N_k1];
    for (int i = 0; i < N_k1; i++) {
        // limit testcase for hardware emulation
        if (mode_emu == "hw_emu") {
            t1[i] = 0;
            T1[i] = 1;
            S1[i] = 2;
            types1[i] = 0;
            K1[i] = 4;
        } else {
            t1[i] = static_cast<TEST_DT>(rand()) / (static_cast<TEST_DT>(RAND_MAX / 10.0));
            T1[i] = t1[i] + 1.0 + static_cast<TEST_DT>(rand()) / (static_cast<TEST_DT>(RAND_MAX / 10.0));
            S1[i] = T1[i] + 1.0 + static_cast<TEST_DT>(rand()) / (static_cast<TEST_DT>(RAND_MAX / 10.0));
            // call or put
            types1[i] = rand() % 2;
            if (types1[i] == 0) {
                K1[i] = 0.01 + static_cast<TEST_DT>(rand()) / (static_cast<TEST_DT>(RAND_MAX / 10.0));
            } else {
                K1[i] = 0.01 + static_cast<TEST_DT>(rand()) / (static_cast<TEST_DT>(RAND_MAX / 1.0));
            }
        }
    }

    // test cases cap/floor
    TEST_DT startY2[N_k2];
    TEST_DT endY2[N_k2];
    int settleFreq2[N_k2];
    TEST_DT N2[N_k2]; // nominal
    TEST_DT X2[N_k2]; // rate
    int types2[N_k2]; // cap/floor

    for (int i = 0; i < N_k2; i++) {
        // limit testcase for hardware emulation
        if (mode_emu == "hw_emu") {
            startY2[i] = 1;
            endY2[i] = 2;
            settleFreq2[i] = 1;
            N2[i] = 100;
            X2[i] = 110;
            types2[i] = 1;
        } else {
            startY2[i] = rand() % 5;
            endY2[i] = startY2[i] + 1.0 + rand() % 10;
            settleFreq2[i] = 4;
            N2[i] = rand() % 100;
            X2[i] = static_cast<TEST_DT>(rand()) / (static_cast<TEST_DT>(RAND_MAX / 0.10));
            types2[i] = rand() % 2;
        }
    }

    // Allocate Memory in Host
    TEST_DT* outputP0[maxk0];
    for (int i = 0; i < maxk0; i++) {
        outputP0[i] = aligned_alloc<TEST_DT>(N_k0);
    }

    TEST_DT* outputP1[maxk1];
    for (int i = 0; i < maxk1; i++) {
        outputP1[i] = aligned_alloc<TEST_DT>(N_k1);
    }

    TEST_DT* outputP2[maxk2];
    for (int i = 0; i < maxk2; i++) {
        outputP2[i] = aligned_alloc<TEST_DT>(N_k2);
    }

    // do pre-process on CPU
    struct timeval start_time, end_time;

#ifndef HLS_TEST

    // kernel 0
    TEST_DT* t0_alloc = aligned_alloc<TEST_DT>(N_k0);
    TEST_DT* T0_alloc = aligned_alloc<TEST_DT>(N_k0);

    // kernel 1
    TEST_DT* t1_alloc = aligned_alloc<TEST_DT>(N_k1);
    TEST_DT* T1_alloc = aligned_alloc<TEST_DT>(N_k1);
    TEST_DT* S1_alloc = aligned_alloc<TEST_DT>(N_k1);
    TEST_DT* K1_alloc = aligned_alloc<TEST_DT>(N_k1);
    int* types1_alloc = aligned_alloc<int>(N_k1);

    // kernel 2
    TEST_DT* startY2_alloc = aligned_alloc<TEST_DT>(N_k2);
    TEST_DT* endY2_alloc = aligned_alloc<TEST_DT>(N_k2);
    int* settleFreq2_alloc = aligned_alloc<int>(N_k2);
    TEST_DT* N2_alloc = aligned_alloc<TEST_DT>(N_k2);
    TEST_DT* X2_alloc = aligned_alloc<TEST_DT>(N_k2);
    int* types2_alloc = aligned_alloc<int>(N_k2);

    // all kernels
    TEST_DT* rates_alloc = aligned_alloc<TEST_DT>(LEN);
    TEST_DT* times_alloc = aligned_alloc<TEST_DT>(LEN);

    // platform related operations
    cl_int cl_err;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Creating Context and Command Queue for selected Device
    cl::Context context(device, NULL, NULL, NULL, &cl_err);
    logger.logCreateContext(cl_err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &cl_err);
    logger.logCreateCommandQueue(cl_err);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("Found Device=%s\n", devName.c_str());

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &cl_err);
    logger.logCreateProgram(cl_err);
#ifdef TEST_KRNL_0
    cl::Kernel kernel_hwa_k0(program, "HWA_k0", &cl_err);
    logger.logCreateKernel(cl_err);
    std::cout << "hwa k0 kernel has been created" << std::endl;
#endif

#ifdef TEST_KRNL_1
    cl::Kernel kernel_hwa_k1(program, "HWA_k1", &cl_err);
    logger.logCreateKernel(cl_err);
    std::cout << "hwa k1 kernel has been created" << std::endl;
#endif

#ifdef TEST_KRNL_2
    cl::Kernel kernel_hwa_k2(program, "HWA_k2", &cl_err);
    logger.logCreateKernel(cl_err);
    std::cout << "hwa k2 kernel has been created" << std::endl;
#endif

    for (int i = 0; i < LEN; i++) {
        rates_alloc[i] = rates[i];
        times_alloc[i] = times[i];
    }

    // kernel 0
    for (int i = 0; i < N_k0; i++) {
        t0_alloc[i] = t0[i];
        T0_alloc[i] = T0[i];
    }

    // kernel 1
    for (int i = 0; i < N_k1; i++) {
        types1_alloc[i] = types1[i];
        t1_alloc[i] = t1[i];
        T1_alloc[i] = T1[i];
        S1_alloc[i] = S1[i];
        K1_alloc[i] = K1[i];
    }

    // kernel 2
    for (int i = 0; i < N_k2; i++) {
        startY2_alloc[i] = startY2[i];
        endY2_alloc[i] = endY2[i];
        settleFreq2_alloc[i] = settleFreq2[i];
        N2_alloc[i] = N2[i];
        X2_alloc[i] = X2[i];
        types2_alloc[i] = types2[i];
    }

    cl_mem_ext_ptr_t mextIn0[4];
    mextIn0[0] = {2, times_alloc, kernel_hwa_k0()};
    mextIn0[1] = {3, rates_alloc, kernel_hwa_k0()};
    mextIn0[2] = {4, t0_alloc, kernel_hwa_k0()};
    mextIn0[3] = {5, T0_alloc, kernel_hwa_k0()};

    cl_mem_ext_ptr_t mextIn1[7];
    mextIn1[0] = {2, times_alloc, kernel_hwa_k1()};
    mextIn1[1] = {3, rates_alloc, kernel_hwa_k1()};
    mextIn1[2] = {4, types1_alloc, kernel_hwa_k1()};
    mextIn1[3] = {5, t1_alloc, kernel_hwa_k1()};
    mextIn1[4] = {6, T1_alloc, kernel_hwa_k1()};
    mextIn1[5] = {7, S1_alloc, kernel_hwa_k1()};
    mextIn1[6] = {8, K1_alloc, kernel_hwa_k1()};

    cl_mem_ext_ptr_t mextIn2[8];
    mextIn2[0] = {2, times_alloc, kernel_hwa_k2()};
    mextIn2[1] = {3, rates_alloc, kernel_hwa_k2()};
    mextIn2[2] = {4, types2_alloc, kernel_hwa_k2()};
    mextIn2[3] = {5, startY2_alloc, kernel_hwa_k2()};
    mextIn2[4] = {6, endY2_alloc, kernel_hwa_k2()};
    mextIn2[5] = {7, settleFreq2_alloc, kernel_hwa_k2()};
    mextIn2[6] = {8, N2_alloc, kernel_hwa_k2()};
    mextIn2[7] = {9, X2_alloc, kernel_hwa_k2()};

    cl_mem_ext_ptr_t mextOut0[maxk0];
    for (int i = 0; i < maxk0; i++) {
        mextOut0[i] = {6, outputP0[i], kernel_hwa_k0()};
    }
    cl_mem_ext_ptr_t mextOut1[maxk1];
    for (int i = 0; i < maxk1; i++) {
        mextOut1[i] = {9, outputP1[i], kernel_hwa_k1()};
    }
    cl_mem_ext_ptr_t mextOut2[maxk2];
    for (int i = 0; i < maxk2; i++) {
        mextOut2[i] = {10, outputP2[i], kernel_hwa_k2()};
    }

#ifdef TEST_KRNL_0

    // ---------------------------------------------
    // kernel 0
    // ---------------------------------------------

    // create device buffer and map dev buf to host buf
    cl::Buffer outputBuf0[maxk0];
    for (int i = 0; i < maxk0; i++) {
        outputBuf0[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(TEST_DT) * N_k0, &mextOut0[i]);
    }

    cl::Buffer inputBuf0[4];
    inputBuf0[0] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              sizeof(TEST_DT) * LEN, &mextIn0[0]);
    inputBuf0[1] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              sizeof(TEST_DT) * LEN, &mextIn0[1]);
    inputBuf0[2] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              sizeof(TEST_DT) * N_k0, &mextIn0[2]);
    inputBuf0[3] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              sizeof(TEST_DT) * N_k0, &mextIn0[3]);

    std::vector<cl::Memory> obIn0;
    obIn0.push_back(inputBuf0[0]);
    obIn0.push_back(inputBuf0[1]);
    obIn0.push_back(inputBuf0[2]);
    obIn0.push_back(inputBuf0[3]);

    // launch kernel and calculate kernel execution time
    std::cout << "HWA Kernel 0 Execution" << std::endl;
    kernel_hwa_k0.setArg(0, a);
    kernel_hwa_k0.setArg(1, sigma);
    kernel_hwa_k0.setArg(2, inputBuf0[0]);
    kernel_hwa_k0.setArg(3, inputBuf0[1]);
    kernel_hwa_k0.setArg(4, inputBuf0[2]);
    kernel_hwa_k0.setArg(5, inputBuf0[3]);
    kernel_hwa_k0.setArg(6, outputBuf0[0]);
    q.enqueueMigrateMemObjects(obIn0, 0);

    for (std::vector<int>::iterator it = testCases.begin(); it != testCases.end(); ++it) {
        gettimeofday(&start_time, 0);

        // enqueue multiple kernels all using the same random inputs but outputting result to a different buffer each
        // time
        int loop_num = *it / N_k0;
        std::cout << "TC: " << *it << " " << std::endl;
        for (int i = 0; i < loop_num; ++i) {
            kernel_hwa_k0.setArg(6, outputBuf0[i]);
            q.enqueueTask(kernel_hwa_k0);
        }

        // wait for them all to complete
        q.finish();
        gettimeofday(&end_time, 0);
        std::cout << "HWA K0 FPGA: " << tvdiff(&start_time, &end_time) << "us" << std::endl;

        // output vector depedant on test case
        std::vector<cl::Memory> obOut0;
        for (int i = 0; i < loop_num; ++i) {
            obOut0.push_back(outputBuf0[i]);
        }
        q.enqueueMigrateMemObjects(obOut0, CL_MIGRATE_MEM_OBJECT_HOST);
        q.finish();

        // check accuracy compared to CPU
        TEST_DT cpuExpectedP0[N_k0];
        gettimeofday(&start_time, 0);
        for (int i = 0; i < loop_num; ++i) {
            HWA_CPU_k0(a, sigma, times, rates, LEN, t0, T0, cpuExpectedP0, N_k0);
            int offset = i * N_k0;
            for (int j = 0; j < N_k0; j++) {
                TEST_DT temp = *(outputP0[i] + j);
                if (std::abs(temp - cpuExpectedP0[j]) > tolerance) {
                    std::cout << "HWA K0 Failure at:" << offset + j << " Expected:" << cpuExpectedP0[j]
                              << " Calculated:" << temp << std::endl;
                    failCnt++;
                }
            }
        }
        gettimeofday(&end_time, 0);
        std::cout << "HWA K0 CPU:  " << tvdiff(&start_time, &end_time) << "us" << std::endl;
    }

#endif // TEST_KRNL_0

#ifdef TEST_KRNL_1

    // ---------------------------------------------
    // kernel 1
    // ---------------------------------------------

    // create device buffer and map dev buf to host buf
    cl::Buffer outputBuf1[maxk1];
    for (int i = 0; i < maxk1; i++) {
        outputBuf1[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(TEST_DT) * N_k1, &mextOut1[i]);
    }

    cl::Buffer inputBuf1[7];
    inputBuf1[0] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              sizeof(TEST_DT) * LEN, &mextIn1[0]);
    inputBuf1[1] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              sizeof(TEST_DT) * LEN, &mextIn1[1]);
    inputBuf1[2] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              sizeof(int) * N_k1, &mextIn1[2]);
    inputBuf1[3] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              sizeof(TEST_DT) * N_k1, &mextIn1[3]);
    inputBuf1[4] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              sizeof(TEST_DT) * N_k1, &mextIn1[4]);
    inputBuf1[5] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              sizeof(TEST_DT) * N_k1, &mextIn1[5]);
    inputBuf1[6] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              sizeof(TEST_DT) * N_k1, &mextIn1[6]);

    std::vector<cl::Memory> obIn1;
    obIn1.push_back(inputBuf1[0]);
    obIn1.push_back(inputBuf1[1]);
    obIn1.push_back(inputBuf1[2]);
    obIn1.push_back(inputBuf1[3]);
    obIn1.push_back(inputBuf1[4]);
    obIn1.push_back(inputBuf1[5]);
    obIn1.push_back(inputBuf1[6]);
    q.finish();

    // launch kernel and calculate kernel execution time
    std::cout << "HWA Kernel 1 Execution" << std::endl;

    kernel_hwa_k1.setArg(0, a);
    kernel_hwa_k1.setArg(1, sigma);
    kernel_hwa_k1.setArg(2, inputBuf1[0]);
    kernel_hwa_k1.setArg(3, inputBuf1[1]);
    kernel_hwa_k1.setArg(4, inputBuf1[2]);
    kernel_hwa_k1.setArg(5, inputBuf1[3]);
    kernel_hwa_k1.setArg(6, inputBuf1[4]);
    kernel_hwa_k1.setArg(7, inputBuf1[5]);
    kernel_hwa_k1.setArg(8, inputBuf1[6]);
    kernel_hwa_k1.setArg(9, outputBuf1[0]);
    q.enqueueMigrateMemObjects(obIn1, 0);

    for (std::vector<int>::iterator it = testCases.begin(); it != testCases.end(); ++it) {
        gettimeofday(&start_time, 0);

        // enqueue multiple kernels all using the same inputs but outputting result to a different buffer
        int loop_num = *it / N_k1;
        std::cout << "TC: " << *it << " " << std::endl;
        for (int i = 0; i < loop_num; ++i) {
            kernel_hwa_k1.setArg(9, outputBuf1[i]);
            q.enqueueTask(kernel_hwa_k1);
        }

        // wait for them all to complete
        q.finish();
        gettimeofday(&end_time, 0);
        std::cout << "HWA K1 FPGA: " << tvdiff(&start_time, &end_time) << "us" << std::endl;

        // output vector depedant on test case
        std::vector<cl::Memory> obOut1;
        for (int i = 0; i < loop_num; ++i) {
            obOut1.push_back(outputBuf1[i]);
        }
        q.enqueueMigrateMemObjects(obOut1, CL_MIGRATE_MEM_OBJECT_HOST);
        q.finish();

        // check accuracy compared to CPU
        TEST_DT cpuExpectedP1[N_k1];
        gettimeofday(&start_time, 0);
        for (int i = 0; i < loop_num; ++i) {
            HWA_CPU_k1(a, sigma, times, rates, LEN, types1, t1, T1, S1, K1, cpuExpectedP1, N_k1);
            int offset = i * N_k1;
            for (int j = 0; j < N_k1; j++) {
                TEST_DT temp = *(outputP1[i] + j);
                if (std::abs(temp - cpuExpectedP1[j]) > tolerance) {
                    std::cout << "HWA K1 Failure at:" << offset + j << " Expected:" << cpuExpectedP1[j]
                              << " Calculated:" << temp << std::endl;
                    failCnt++;
                }
            }
        }
        gettimeofday(&end_time, 0);
        std::cout << "HWA K1 CPU:  " << tvdiff(&start_time, &end_time) << "us" << std::endl;
    }

#endif // TEST_KRNL_1

#ifdef TEST_KRNL_2

    // ---------------------------------------------
    // kernel 2
    // ---------------------------------------------

    // create device buffer and map dev buf to host buf
    cl::Buffer outputBuf2[maxk2];
    for (int i = 0; i < maxk2; i++) {
        outputBuf2[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(TEST_DT) * N_k2, &mextOut2[i]);
    }

    cl::Buffer inputBuf2[8];
    inputBuf2[0] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              sizeof(TEST_DT) * LEN, &mextIn2[0]); // times
    inputBuf2[1] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              sizeof(TEST_DT) * LEN, &mextIn2[1]); // rates
    inputBuf2[2] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              sizeof(int) * N_k2, &mextIn2[2]); // types
    inputBuf2[3] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              sizeof(TEST_DT) * N_k2, &mextIn2[3]); // start year
    inputBuf2[4] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              sizeof(TEST_DT) * N_k2, &mextIn2[4]); // end year
    inputBuf2[5] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              sizeof(int) * N_k2, &mextIn2[5]); // settlement freq
    inputBuf2[6] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              sizeof(TEST_DT) * N_k2, &mextIn2[6]); // N
    inputBuf2[7] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                              sizeof(TEST_DT) * N_k2, &mextIn2[7]); // X

    std::vector<cl::Memory> obIn2;
    obIn2.push_back(inputBuf2[0]);
    obIn2.push_back(inputBuf2[1]);
    obIn2.push_back(inputBuf2[2]);
    obIn2.push_back(inputBuf2[3]);
    obIn2.push_back(inputBuf2[4]);
    obIn2.push_back(inputBuf2[5]);
    obIn2.push_back(inputBuf2[6]);
    obIn2.push_back(inputBuf2[7]);
    q.finish();

    // launch kernel and calculate kernel execution time
    std::cout << "HWA Kernel 2 Execution" << std::endl;
    kernel_hwa_k2.setArg(0, a);
    kernel_hwa_k2.setArg(1, sigma);
    kernel_hwa_k2.setArg(2, inputBuf2[0]);
    kernel_hwa_k2.setArg(3, inputBuf2[1]);
    kernel_hwa_k2.setArg(4, inputBuf2[2]);
    kernel_hwa_k2.setArg(5, inputBuf2[3]);
    kernel_hwa_k2.setArg(6, inputBuf2[4]);
    kernel_hwa_k2.setArg(7, inputBuf2[5]);
    kernel_hwa_k2.setArg(8, inputBuf2[6]);
    kernel_hwa_k2.setArg(9, inputBuf2[7]);
    kernel_hwa_k2.setArg(10, outputBuf2[0]);
    q.enqueueMigrateMemObjects(obIn2, 0);

    for (std::vector<int>::iterator it = testCases.begin(); it != testCases.end(); ++it) {
        gettimeofday(&start_time, 0);

        // enqueue multiple kernels all using the same inputs but outputting result to a different buffer
        int loop_num = *it / N_k2;
        std::cout << "TC: " << *it << std::endl;
        for (int i = 0; i < loop_num; ++i) {
            kernel_hwa_k2.setArg(10, outputBuf2[i]);
            q.enqueueTask(kernel_hwa_k2);
        }

        // wait for them all to complete
        q.finish();
        gettimeofday(&end_time, 0);
        std::cout << "HWA K2 FPGA: " << tvdiff(&start_time, &end_time) << "us" << std::endl;

        // output vector depedant on test case
        std::vector<cl::Memory> obOut2;
        for (int i = 0; i < loop_num; ++i) {
            obOut2.push_back(outputBuf2[i]);
        }
        q.enqueueMigrateMemObjects(obOut2, CL_MIGRATE_MEM_OBJECT_HOST);
        q.finish();

        // check accuracy compared to CPU
        TEST_DT cpuExpectedP2[N_k2];
        gettimeofday(&start_time, 0);
        for (int i = 0; i < loop_num; ++i) {
            HWA_CPU_k2(a, sigma, times, rates, LEN, types2, startY2, endY2, settleFreq2, N2, X2, cpuExpectedP2, N_k2);
            int offset = i * N_k2;
            for (int j = 0; j < N_k2; j++) {
                TEST_DT temp = *(outputP2[i] + j);
                if (std::abs(temp - cpuExpectedP2[j]) > tolerance) {
                    std::cout << "HWA K2 Failure at:" << offset + j << " Expected:" << cpuExpectedP2[j]
                              << " Calculated:" << temp << std::endl;
                    failCnt++;
                }
            }
        }
        gettimeofday(&end_time, 0);
        std::cout << "HWA K2 CPU:  " << tvdiff(&start_time, &end_time) << "us" << std::endl;
    }

#endif // TEST_KRNL_2

#else

    HWA_k0(a, sigma, times, rates, t0, T0, outputP0);
    HWA_k1(a, sigma, times, rates, types1, t1, T1, S1, K1, outputP1);
    HWA_k2(a, sigma, times, rates, types2, startY2, endY2, settleFreq2, N2, X2, outputP2);

#endif // HLS_TEST

    // ---------------------------------------------
    // Overall Pass/Fail
    // ---------------------------------------------

    // overall test result
    if (failCnt > 0) {
        err = -1;
    }
    err ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);

    return err;
}
