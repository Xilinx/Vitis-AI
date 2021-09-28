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

#ifndef HLS_TEST
#include "xcl2.hpp"
#endif
#include <cstring>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include "ap_int.h"
#include "utils.hpp"
#include "cds_engine_kernel.hpp"
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

extern TEST_DT cpu_cds_kernel(std::map<TEST_DT, TEST_DT> interestRateCurve,
                              std::map<TEST_DT, TEST_DT> hazardRateCurve,
                              TEST_DT notionalValue,
                              TEST_DT recoveryRate,
                              TEST_DT maturity,
                              int frequency);

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
    struct timeval start_time, end_time;
    TEST_DT tolerance = 1e-4;
    int failCnt = 0;
    int err = 0;

    std::cout << std::setprecision(10) << std::endl;
    std::cout << "\n----------------------Credit Default Swap Engine (CDS)-----------------\n";
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

    // inputs
    TEST_DT ratesIR[IRLEN] = {0.0300, 0.0335, 0.0366, 0.0394, 0.0418, 0.0439, 0.0458, 0.0475, 0.0490, 0.0503, 0.0514,
                              0.0524, 0.0533, 0.0541, 0.0548, 0.0554, 0.0559, 0.0564, 0.0568, 0.0572, 0.0575};

    TEST_DT timesIR[IRLEN] = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                              5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0};

    TEST_DT ratesHazard[HAZARDLEN] = {0.005, 0.01, 0.01, 0.015, 0.010, 0.010};
    TEST_DT timesHazard[HAZARDLEN] = {0.0, 0.5, 1.0, 2.0, 5.0, 10.0};

    // golden test vector & expected results obtained from Derivgaem DG300 XLS CDS Tab
    TEST_DT maturity[N] = {2.0, 3.0, 4.0, 5.55, 6.33, 7.27, 8.001, 9.999};
    int frequency[N] = {4, 12, 2, 1, 12, 4, 1, 12};
    TEST_DT recovery[N] = {0.15, 0.67, 0.22, 0.01, 0.80, 0.99, 0.001, 0.44};
    TEST_DT nominal[N] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    TEST_DT expectedCDSSpread[N] = {(106.23041583 / 10000), (38.58272908 / 10000), (89.16354729 / 10000),
                                    (101.74021757 / 10000), (21.79021152 / 10000), (1.04496915 / 10000),
                                    (94.82650838 / 10000),  (59.63129003 / 10000)};

    // outputs
    TEST_DT* outputP;
    outputP = aligned_alloc<TEST_DT>(N);

#ifndef HLS_TEST

    TEST_DT* nominal_alloc = aligned_alloc<TEST_DT>(N);
    TEST_DT* recovery_alloc = aligned_alloc<TEST_DT>(N);
    TEST_DT* maturity_alloc = aligned_alloc<TEST_DT>(N);
    int* frequency_alloc = aligned_alloc<int>(N);

    TEST_DT* ratesIR_alloc = aligned_alloc<TEST_DT>(IRLEN);
    TEST_DT* timesIR_alloc = aligned_alloc<TEST_DT>(IRLEN);
    TEST_DT* ratesHazard_alloc = aligned_alloc<TEST_DT>(HAZARDLEN);
    TEST_DT* timesHazard_alloc = aligned_alloc<TEST_DT>(HAZARDLEN);

    cl_int cl_err;
    // platform related operations
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Creating Context and Command Queue for selected Device
    cl::Context context(device, NULL, NULL, NULL, &cl_err);
    logger.logCreateContext(cl_err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &cl_err);
    logger.logCreateCommandQueue(cl_err);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("Found Device=%s\n", devName.c_str());

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &cl_err);
    logger.logCreateProgram(cl_err);
    cl::Kernel kernel_cds(program, "CDS_kernel", &cl_err);
    logger.logCreateKernel(cl_err);

    for (int i = 0; i < IRLEN; i++) {
        ratesIR_alloc[i] = ratesIR[i];
        timesIR_alloc[i] = timesIR[i];
    }

    for (int i = 0; i < HAZARDLEN; i++) {
        ratesHazard_alloc[i] = ratesHazard[i];
        timesHazard_alloc[i] = timesHazard[i];
    }

    for (int i = 0; i < N; i++) {
        nominal_alloc[i] = nominal[i];
        recovery_alloc[i] = recovery[i];
        maturity_alloc[i] = maturity[i];
        frequency_alloc[i] = frequency[i];
    }

    cl_mem_ext_ptr_t mextIn0[8];
    mextIn0[0] = {0, timesIR_alloc, kernel_cds()};
    mextIn0[1] = {1, ratesIR_alloc, kernel_cds()};
    mextIn0[2] = {2, timesHazard_alloc, kernel_cds()};
    mextIn0[3] = {3, ratesHazard_alloc, kernel_cds()};
    mextIn0[4] = {4, nominal_alloc, kernel_cds()};
    mextIn0[5] = {5, recovery_alloc, kernel_cds()};
    mextIn0[6] = {6, maturity_alloc, kernel_cds()};
    mextIn0[7] = {7, frequency_alloc, kernel_cds()};

    cl_mem_ext_ptr_t mextOut0;
    mextOut0 = {8, outputP, kernel_cds()};

    // create device buffer and map dev buf to host buf
    cl::Buffer outputBuf;
    outputBuf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                           sizeof(TEST_DT) * N, &mextOut0);

    cl::Buffer inputBuf0[8];
    inputBuf0[0] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                              sizeof(TEST_DT) * IRLEN, &mextIn0[0]);
    inputBuf0[1] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                              sizeof(TEST_DT) * IRLEN, &mextIn0[1]);
    inputBuf0[2] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                              sizeof(TEST_DT) * HAZARDLEN, &mextIn0[2]);
    inputBuf0[3] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                              sizeof(TEST_DT) * HAZARDLEN, &mextIn0[3]);

    inputBuf0[4] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                              sizeof(TEST_DT) * N, &mextIn0[4]);
    inputBuf0[5] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                              sizeof(TEST_DT) * N, &mextIn0[5]);
    inputBuf0[6] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                              sizeof(TEST_DT) * N, &mextIn0[6]);
    inputBuf0[7] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * N,
                              &mextIn0[7]);

    std::vector<cl::Memory> obIn0;
    obIn0.push_back(inputBuf0[0]);
    obIn0.push_back(inputBuf0[1]);
    obIn0.push_back(inputBuf0[2]);
    obIn0.push_back(inputBuf0[3]);
    obIn0.push_back(inputBuf0[4]);
    obIn0.push_back(inputBuf0[5]);
    obIn0.push_back(inputBuf0[6]);
    obIn0.push_back(inputBuf0[7]);

    // output vector depedant on test case
    std::vector<cl::Memory> obOut0;
    obOut0.push_back(outputBuf);

    // launch kernel and calculate kernel execution time
    std::cout << "CDS Kernel Execution" << std::endl;
    kernel_cds.setArg(0, inputBuf0[0]);
    kernel_cds.setArg(1, inputBuf0[1]);
    kernel_cds.setArg(2, inputBuf0[2]);
    kernel_cds.setArg(3, inputBuf0[3]);
    kernel_cds.setArg(4, inputBuf0[4]);
    kernel_cds.setArg(5, inputBuf0[5]);
    kernel_cds.setArg(6, inputBuf0[6]);
    kernel_cds.setArg(7, inputBuf0[7]);
    kernel_cds.setArg(8, outputBuf);
    q.enqueueMigrateMemObjects(obIn0, 0);

    // enqueue kernel
    cl::Event event;
    uint64_t nstimestart, nstimeend;
    gettimeofday(&start_time, 0);
    q.enqueueTask(kernel_cds, 0, &event);

    // wait for them all to complete
    q.finish();
    gettimeofday(&end_time, 0);
    std::cout << "CDS FPGA Execution Time: " << tvdiff(&start_time, &end_time) << "us" << std::endl;

    event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart);
    event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend);
    auto duration_nanosec = nstimeend - nstimestart;
    std::cout << "CDS Kernel Execution Time: " << (duration_nanosec / 1000) << " us" << std::endl;

    std::cout << "CDS FPGA Migration Complete" << std::endl;
    q.enqueueMigrateMemObjects(obOut0, CL_MIGRATE_MEM_OBJECT_HOST);
    q.finish();

    for (int i = 0; i < N; ++i) {
        TEST_DT temp = outputP[i];
        if (std::abs(temp - expectedCDSSpread[i]) > tolerance) {
            std::cout << "CDS Failure at:" << i << " Expected:" << expectedCDSSpread[i] << " Calculated:" << temp
                      << std::endl;
            failCnt++;
        } else {
            std::cout << i << " Expected:" << expectedCDSSpread[i] << " Calculated:" << temp << std::endl;
        }
    }

    // CPU
    std::cout << std::endl;
    std::cout << "CDS CPU Execution" << std::endl;

    std::map<TEST_DT, TEST_DT> interestRateCurve;
    for (int i = 0; i < IRLEN; ++i) {
        interestRateCurve.insert(std::pair<TEST_DT, TEST_DT>(timesIR[i], ratesIR[i]));
    }

    std::map<TEST_DT, TEST_DT> hazardRateCurve;
    for (int i = 0; i < HAZARDLEN; ++i) {
        hazardRateCurve.insert(std::pair<TEST_DT, TEST_DT>(timesHazard[i], ratesHazard[i]));
    }

    gettimeofday(&start_time, 0);
    for (int i = 0; i < N; ++i) {
        TEST_DT cpu_temp =
            cpu_cds_kernel(interestRateCurve, hazardRateCurve, nominal[i], recovery[i], maturity[i], frequency[i]);

        if (std::abs(cpu_temp - expectedCDSSpread[i]) > tolerance) {
            std::cout << "CPU CDS Failure at:" << i << " Expected:" << expectedCDSSpread[i]
                      << " Calculated:" << cpu_temp << std::endl;
            failCnt++;
        } else {
            std::cout << i << " Expected:" << expectedCDSSpread[i] << " Calculated:" << cpu_temp << std::endl;
        }
    }
    gettimeofday(&end_time, 0);
    std::cout << "CDS CPU Execution Time: " << tvdiff(&start_time, &end_time) << "us" << std::endl;

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
