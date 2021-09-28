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

struct DigitalOptionData {
    bool optionType;
    TEST_DT strike;
    TEST_DT s;      // spot
    TEST_DT q;      // dividend
    TEST_DT r;      // risk-free rate
    TEST_DT t;      // time to maturity
    TEST_DT v;      // volatility
    TEST_DT result; // expected result
    TEST_DT tol;    // tolerance
    bool knockin;   // true if knock-in
};

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
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    // cmd parser
    ArgParser parser(argc, argv);
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
    // Allocate Memory in Host Memory
    TEST_DT* outputs = aligned_alloc<TEST_DT>(1);
    ap_uint<32>* seed = aligned_alloc<ap_uint<32> >(1);

    // -------------setup params---------------
    // bool run_csim = false;
    // if (argc >= 2) {
    //    run_csim = std::stoi(argv[1]);
    //    if (run_csim) std::cout << "run csim for function verify\n";
    //}

    DigitalOptionData values[] = {// type, strike,   spot,    q,    r,   t,  vol,   value, tol
                                  {1, 100.00, 105.00, 0.20, 0.10, 0.5, 0.20, 12.2715, 2e-2, true},
                                  {0, 100.00, 95.00, 0.20, 0.10, 0.5, 0.20, 8.9109, 2e-2, true}};

    unsigned int timeSteps = 45;
    TEST_DT cashPayoff = 15.0;
    TEST_DT timeLength = 0.5;
    unsigned int maxSamples = 0;
    unsigned int requiredSamples = 4096 * 4 - 1;
    int test_nm = 2;
    if (mode_emu == "hw_emu") test_nm = 1;
    if (mode_emu == "hw_emu") requiredSamples = 1024;
    seed[0] = 1;
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
    cl::Kernel kernel_Engine(program, "MCDigitalEngine_k", &cl_err);
    logger.logCreateKernel(cl_err);

    cl_mem_ext_ptr_t mext_o[2];
    mext_o[0] = {10, outputs, kernel_Engine()};
    mext_o[1] = {9, seed, kernel_Engine()};

    // create device buffer and map dev buf to host buf
    cl::Buffer output_buf;
    cl::Buffer seed_buf;
    output_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(TEST_DT),
                            &mext_o[0]);
    seed_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                          sizeof(unsigned int), &mext_o[1]);

    int nerr = 0;
    for (int i = 0; i < test_nm; ++i) {
        // testbench 1, put option
        TEST_DT requiredTolerance = 0.005;

        TEST_DT underlying = values[i].s;
        TEST_DT riskFreeRate = values[i].r;
        TEST_DT dividendYield = values[i].q;

        TEST_DT volatility = values[i].v;
        TEST_DT strike = values[i].strike;
        int optionType = values[i].optionType;

        TEST_DT expectedVal = values[i].result;
        TEST_DT maxMcErrorAllowed = values[i].tol;

        int exEarly = values[i].knockin;

        if (optionType)
            std::cout << "Put option:\n";
        else
            std::cout << "Call option:\n";
        std::cout << "   strike:              " << strike << "\n"
                  << "   spot value:          " << underlying << "\n"
                  << "   risk-free rate:      " << riskFreeRate << "\n"
                  << "   volatility:          " << volatility << "\n"
                  << "   dividend yield:      " << dividendYield << "\n"
                  << "   maturity:            " << timeLength << "\n"
                  << "   tolerance:           " << requiredTolerance << "\n"
                  << "   requaried samples:   " << requiredSamples << "\n"
                  << "   maximum samples:     " << maxSamples << "\n"
                  << "   timesteps:           " << timeSteps << "\n"
                  << "   fixed payoff:        " << cashPayoff << "\n"
                  << "   golden:              " << expectedVal << "\n";

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
        kernel_Engine.setArg(j++, strike);
        kernel_Engine.setArg(j++, cashPayoff);
        kernel_Engine.setArg(j++, optionType);
        kernel_Engine.setArg(j++, exEarly);
        kernel_Engine.setArg(j++, seed_buf);
        kernel_Engine.setArg(j++, output_buf);
        kernel_Engine.setArg(j++, requiredTolerance);
        kernel_Engine.setArg(j++, timeSteps);
        kernel_Engine.setArg(j++, requiredSamples);

        q.enqueueTask(kernel_Engine, nullptr, nullptr);

        q.finish();
        gettimeofday(&end_time, 0);
        std::cout << "kernel end------" << std::endl;
        std::cout << "Execution time " << tvdiff(&start_time, &end_time) << "us" << std::endl;
        q.enqueueMigrateMemObjects(ob_out, 1, nullptr, nullptr);
        q.finish();

        TEST_DT error = std::fabs(expectedVal - outputs[0]);
        if (error > maxMcErrorAllowed) {
            std::cout << "Output is wrong!" << std::endl;
            std::cout << "Acutal value: " << outputs[0] << ", Expected value: " << expectedVal << ", Error: " << error
                      << std::endl;
            nerr++;
            // return -1;
        }
    }
    nerr ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
         : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);

    return nerr;
}
