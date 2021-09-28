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

struct BarrierOptionData {
    xf::fintech::enums::BarrierType barrierType;
    TEST_DT barrier;
    TEST_DT rebate;
    bool type;
    TEST_DT strike;
    TEST_DT s;      // spot
    TEST_DT q;      // dividend
    TEST_DT r;      // risk-free rate
    TEST_DT t;      // time to maturity
    TEST_DT v;      // volatility
    TEST_DT result; // result
    TEST_DT tol;    // tolerance
};

int main(int argc, const char* argv[]) {
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    // cmd parser
    ArgParser parser(argc, argv);
    std::string mode;
    std::string xclbin_path;
    std::string mode_emu = "hw";
#ifndef HLS_TEST
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }

    if (std::getenv("XCL_EMULATION_MODE") != nullptr) {
        mode_emu = std::getenv("XCL_EMULATION_MODE");
    }
    std::cout << "[INFO]Running in " << mode_emu << " mode" << std::endl;
#endif
    int nerr = 0;
    // Allocate Memory in Host Memory
    TEST_DT* outputs = aligned_alloc<TEST_DT>(1);
    unsigned int* seed = aligned_alloc<unsigned int>(1);

    // -------------setup k0 params---------------
    BarrierOptionData values[] = {
        // barrierType,    barrier, rebate,  type,  strike,   s,     q,     r,    t,   vol,  result,   tol
        {xf::fintech::enums::BarrierType::DownIn, 90, 0, 0, 100, 100, 0.02, 0.05, 1, 0.10, 0.07187, 0.02},
        {xf::fintech::enums::BarrierType::DownIn, 90, 0, 0, 100, 100, 0.02, 0.05, 1, 0.15, 0.60638, 0.02},
        {xf::fintech::enums::BarrierType::DownIn, 90, 0, 0, 100, 100, 0.02, 0.05, 1, 0.20, 1.64005, 0.02},
        {xf::fintech::enums::BarrierType::DownIn, 90, 0, 0, 100, 100, 0.02, 0.05, 1, 0.25, 2.98495, 0.02},
        {xf::fintech::enums::BarrierType::DownIn, 90, 0, 0, 100, 100, 0.02, 0.05, 1, 0.30, 4.50952, 0.02},

        {xf::fintech::enums::BarrierType::UpIn, 110, 0, 0, 100, 100, 0.02, 0.05, 1, 0.10, 4.79148, 0.02},
        {xf::fintech::enums::BarrierType::UpIn, 110, 0, 0, 100, 100, 0.02, 0.05, 1, 0.15, 7.08268, 0.02},
        {xf::fintech::enums::BarrierType::UpIn, 110, 0, 0, 100, 100, 0.02, 0.05, 1, 0.20, 9.11008, 0.02},
        {xf::fintech::enums::BarrierType::UpIn, 110, 0, 0, 100, 100, 0.02, 0.05, 1, 0.25, 11.0615, 0.02},
        {xf::fintech::enums::BarrierType::UpIn, 110, 0, 0, 100, 100, 0.02, 0.05, 1, 0.30, 12.9835, 0.02},

        {xf::fintech::enums::BarrierType::DownOut, 45, 0, 0, 50, 50, 0.00, 0.09531018, 1, 0.50, 5.477, 0.01}};

    seed[0] = 5;

    unsigned int maxSamples = 0;
    unsigned int requiredSamples = 131071;
    int test_nm = 11;
    if (mode_emu == "hw_emu") test_nm = 1;
    // do pre-process on CPU
    struct timeval start_time, end_time, test_time;
    // platform related operations
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Creating Context and Command Queue for selected Device
    cl_int cl_err;
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
    cl::Kernel kernel_Engine(program, "MCBarrierNoBiasEngine_k0", &cl_err);
    logger.logCreateKernel(cl_err);

    cl_mem_ext_ptr_t mext_o[2];
    mext_o[1] = {9, seed, kernel_Engine()};
    mext_o[0] = {10, outputs, kernel_Engine()};

    // create device buffer and map dev buf to host buf
    cl::Buffer output_buf;
    cl::Buffer seed_buf;
    output_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(TEST_DT),
                            &mext_o[0]);
    seed_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                          sizeof(unsigned int), &mext_o[1]);

    for (int i = 0; i < test_nm; ++i) {
        unsigned int timeSteps = 1;
        TEST_DT requiredTolerance = 0.02;

        TEST_DT underlying = values[i].s;
        TEST_DT riskFreeRate = values[i].r;
        TEST_DT dividendYield = values[i].q;
        TEST_DT volatility = values[i].v;

        TEST_DT strike = values[i].strike;
        int optionType = values[i].type;
        TEST_DT barrier = values[i].barrier;
        TEST_DT rebate = values[i].rebate;
        xf::fintech::enums::BarrierType barrierType = values[i].barrierType;

        TEST_DT timeLength = 1;

        TEST_DT expectedVal = values[i].result;

        std::vector<cl::Memory> ob_out;
        ob_out.push_back(output_buf);

        q.finish();
        // launch kernel and calculate kernel execution time
        std::cout << "kernel start------" << std::endl;
        gettimeofday(&start_time, 0);
        int j = 0;
        kernel_Engine.setArg(j++, underlying);
        kernel_Engine.setArg(j++, volatility);
        kernel_Engine.setArg(j++, dividendYield);
        kernel_Engine.setArg(j++, riskFreeRate);
        kernel_Engine.setArg(j++, timeLength);
        kernel_Engine.setArg(j++, barrier);
        kernel_Engine.setArg(j++, strike);
        kernel_Engine.setArg(j++, barrierType);
        kernel_Engine.setArg(j++, optionType);
        kernel_Engine.setArg(j++, seed_buf);
        kernel_Engine.setArg(j++, output_buf);
        kernel_Engine.setArg(j++, rebate);
        kernel_Engine.setArg(j++, requiredTolerance);
        kernel_Engine.setArg(j++, requiredSamples);
        kernel_Engine.setArg(j++, timeSteps);

        q.enqueueTask(kernel_Engine, nullptr, nullptr);

        q.finish();
        gettimeofday(&end_time, 0);
        std::cout << "kernel end------" << std::endl;
        std::cout << "Execution time " << tvdiff(&start_time, &end_time) << "us" << std::endl;
        q.enqueueMigrateMemObjects(ob_out, 1, nullptr, nullptr);
        q.finish();
        TEST_DT error = std::fabs(expectedVal - outputs[0]) / expectedVal;
        if (error > values[i].tol) {
            std::cout << "Output is wrong!" << std::endl;
            std::cout << "Acutal value: " << outputs[0] << ", Expected value: " << expectedVal
                      << ", Relative error: " << error << std::endl;
            nerr++;
            // return -1;
        }
    }
    nerr ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
         : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);

    return nerr;
}
