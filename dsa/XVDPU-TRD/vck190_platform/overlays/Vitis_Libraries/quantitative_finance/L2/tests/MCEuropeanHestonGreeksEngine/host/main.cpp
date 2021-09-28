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
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include "ap_int.h"
#include "utils.hpp"
#include "mcengine_top.hpp"
#include "xf_utils_sw/logger.hpp"

#define LENGTH(a) (sizeof(a) / sizeof(a[0]))

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
    std::cout << "\n----------------------McEuropeanHestonGreeksEngine-----------------\n";
    // cmd parser
    ArgParser parser(argc, argv);
    std::string xclbin_path;
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }
    // Allocate Memory in Host Memory
    ap_uint<32>* seed = aligned_alloc<ap_uint<32> >(8 * 2);

    xf::common::utils_sw::Logger logger(std::cout, std::cerr);

    // -------------setup k0 params---------------
    int timeSteps = 100;
    TEST_DT requiredTolerance = 0.2;
    TEST_DT underlying = 1;
    TEST_DT riskFreeRate = 0.05;
    TEST_DT sigma = 0.500000502859;
    TEST_DT v0 = 0.497758237075;
    TEST_DT theta = 0.080904894692;
    TEST_DT kappa = 250.000000532040;
    TEST_DT rho = -0.000249561287;
    TEST_DT dividendYield = 0.0;
    TEST_DT strike = 1;
    unsigned int optionType = 0;
    TEST_DT timeLength = 1;
    TEST_DT* outputs = aligned_alloc<TEST_DT>(8);

    ap_uint<32> seed_2d[8][2];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 2; j++) {
            seed_2d[i][j] = i * 10000 + j * 227 + 1;
        }
    }

    for (unsigned int i = 0; i < 8; i++) {
        for (unsigned int j = 0; j < 2; j++) {
            seed[i * 2 + j] = seed_2d[i][j];
        }
    }

    std::string mode = {"hw"};
    unsigned int requiredSamples;
    // golden value from Heston Closed form in STAC-A2.
    TEST_DT golden[8] = {-0.047598475,    -0.108130457089, 0.506724428909,   1.0020988446,
                         -4.348313105e-6, 0.406696644,     -3.4746136286e-6, 0.00260451088};

    if (std::getenv("XCL_EMULATION_MODE") != nullptr) {
        mode = std::getenv("XCL_EMULATION_MODE");
    }
    TEST_DT max_tolerance = 0;
    if (mode.compare("hw_emu") == 0) {
        timeSteps = 3;
        requiredSamples = 1024;
        max_tolerance = 0.2;
    } else {
        requiredSamples = 4096;
        max_tolerance = 0.03;
    }
    unsigned int maxSamples = 1000000;

#ifndef HLS_TEST
    // do pre-process on CPU
    struct timeval start_time, end_time, test_time;
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

    // cl::Program::Binaries xclBins = xcl::import_binary_file("../xclbin/MCAE_u250_hw.xclbin");
    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &cl_err);
    logger.logCreateProgram(cl_err);
    cl::Kernel MCEHGEngine(program, "MCEHGEngine_k0", &cl_err);
    logger.logCreateKernel(cl_err);

    cl_mem_ext_ptr_t mext_o[2];
    mext_o[0] = {16, outputs, MCEHGEngine()};
    mext_o[1] = {14, seed, MCEHGEngine()};

    // create device buffer and map dev buf to host buf
    cl::Buffer output_buf;
    cl::Buffer seed_buf;
    output_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            sizeof(TEST_DT) * 8, &mext_o[0]);
    seed_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                          sizeof(ap_uint<32>) * (8 * 2), &mext_o[1]);

    std::vector<cl::Memory> ob_out;
    ob_out.push_back(output_buf);

    q.finish();
    // launch kernel and calculate kernel execution time
    std::cout << "kernel start------" << std::endl;
    gettimeofday(&start_time, 0);
    int loop_num = 1;
    for (int i = 0; i < loop_num; ++i) {
        MCEHGEngine.setArg(0, underlying);
        MCEHGEngine.setArg(1, riskFreeRate);
        MCEHGEngine.setArg(2, sigma);
        MCEHGEngine.setArg(3, v0);
        MCEHGEngine.setArg(4, theta);
        MCEHGEngine.setArg(5, kappa);
        MCEHGEngine.setArg(6, rho);
        MCEHGEngine.setArg(7, dividendYield);
        MCEHGEngine.setArg(8, optionType);
        MCEHGEngine.setArg(9, strike);
        MCEHGEngine.setArg(10, timeLength);
        MCEHGEngine.setArg(11, timeSteps);
        MCEHGEngine.setArg(12, requiredSamples);
        MCEHGEngine.setArg(13, maxSamples);
        MCEHGEngine.setArg(14, seed_buf);
        MCEHGEngine.setArg(15, requiredTolerance);
        MCEHGEngine.setArg(16, output_buf);

        q.enqueueTask(MCEHGEngine, nullptr, nullptr);
    }

    q.finish();
    gettimeofday(&end_time, 0);
    std::cout << "kernel end------" << std::endl;
    std::cout << "Execution time " << tvdiff(&start_time, &end_time) / loop_num << "us" << std::endl;
    q.enqueueMigrateMemObjects(ob_out, 1, nullptr, nullptr);
    q.finish();

#else
    MCEHGEngine_k0(underlying, riskFreeRate, sigma, v0, theta, kappa, rho, dividendYield, optionType, strike,
                   timeLength, timeSteps, requiredSamples, maxSamples, seed, requiredTolerance, outputs);
#endif

    int err = 0;
    for (int i = 0; i < 8; ++i) {
        TEST_DT er = std::fabs(golden[i] - outputs[i]); /// std::fabs(golden[i]);
        if (er > max_tolerance) {
            std::cout << "difference[" << i << "]: " << er << std::endl;
            err++;
            // return -1;
        }
    }
    std::cout << "theta: " << outputs[0] << ", rho: " << outputs[1] << ", delta: " << outputs[2]
              << ", gamma: " << outputs[3] << ", MV_kappa: " << outputs[4] << ", MV_theta: " << outputs[5]
              << ", MV_KHI: " << outputs[6] << ", MV_VO: " << outputs[7] << std::endl;
    err ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return err;
}
