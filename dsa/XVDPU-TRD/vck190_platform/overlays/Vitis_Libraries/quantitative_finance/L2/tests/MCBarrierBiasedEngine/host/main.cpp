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
#include <iostream>
#include <math.h>
#include "kernel_mcbarrierbiasedengine.hpp"
#include "utils.hpp"
#include "xf_fintech/mc_engine.hpp"

#ifndef HLS_TEST
#include "xcl2.hpp"
#endif
#include "xf_utils_sw/logger.hpp"

#define KN 1
#define OUTDEP 1024

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

int printResult(float* out, float golden, float tol, int loopNum) {
    if (std::fabs(*out - golden) > tol) {
        std::cout << "Expected value: " << golden << std::endl;
        std::cout << "FPGA result:    " << *out << std::endl;
        return 1;
    }
    return 0;
}

int main(int argc, const char* argv[]) {
    std::cout << "\n----------------------MC(BarrierBias) Engine-----------------\n";
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    // cmd parser
    ArgParser parser(argc, argv);
    std::string xclbin_path;

    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }

    struct timeval st_time, end_time;
    DtUsed* out0_a = aligned_alloc<DtUsed>(OUTDEP);

    DtUsed* out0_b = aligned_alloc<DtUsed>(OUTDEP);

    // test data
    unsigned int loopNum = 1;
    unsigned int timeSteps = 1;
    DtUsed requiredTolerance = 0.005;

    DtUsed underlying = 100;
    DtUsed riskFreeRate = 0.05;
    DtUsed dividendYield = 0.02;

    DtUsed volatility = 0.10;
    DtUsed strike = 100;
    int optionType = 0;
    DtUsed barrier = 110;
    DtUsed rebate = 0.0;
    // xf::fintech::enums::BarrierType barrierType = xf::fintech::enums::BarrierType::UpIn;
    // ap_uint<2> barrierType = xf::fintech::enums::BarrierType::UpIn;
    DtUsed timeLength = 1;
    xf::fintech::enums::BarrierType barrierType = xf::fintech::enums::BarrierType::UpIn;
    DtUsed outputs[OUTDEP];

    // barrierType = xf::fintech::enums::BarrierType::UpIn;
    DtUsed expectedVal = 3.793058; // expect value from QuantLib not analytic results.4.791480;
    DtUsed maxErrorAllowed = 0.16; // 0.02;

    unsigned int maxSamples = 0;
    unsigned int requiredSamples = 4096;

    std::cout << "Call option:\n"
              << "   strike:              " << strike << "\n"
              << "   underlying:          " << underlying << "\n"
              << "   risk-free rate:      " << riskFreeRate << "\n"
              << "   volatility:          " << volatility << "\n"
              << "   dividend yield:      " << dividendYield << "\n"
              << "   maturity:            " << timeLength << "\n"
              << "   tolerance:           " << requiredTolerance << "\n"
              << "   requaried samples:   " << requiredSamples << "\n"
              << "   maximum samples:     " << maxSamples << "\n"
              << "   timesteps:           " << timeSteps << "\n"
              << "   barrier:             " << barrier << "\n"
              << "   rebate:              " << rebate << "\n"
              << "   barrier type:        " << barrierType << "\n"
              << "   golden:              " << expectedVal << "\n";

//
#ifdef HLS_TEST
    int num_rep = 1;
#else
    int num_rep = 1;
    std::string num_str;
    if (parser.getCmdOption("-rep", num_str)) {
        try {
            num_rep = std::stoi(num_str);
        } catch (...) {
            num_rep = 1;
        }
    }
    if (num_rep > 20) {
        num_rep = 20;
        std::cout << "WARNING: limited repeat to " << num_rep << " times\n.";
    }
    if (parser.getCmdOption("-p", num_str)) {
        try {
            requiredSamples = std::stoi(num_str);
        } catch (...) {
            std::cout << "Set sample count" << std::endl;
            requiredSamples = 48128;
        }
    }
#endif

#ifdef HLS_TEST
    McBarrierBiasedEngine_k(loopNum, underlying, volatility, dividendYield, riskFreeRate, timeLength, barrier, strike,
                            // barrierType,
                            optionType, outputs, rebate, requiredTolerance, requiredSamples, timeSteps);

    printResult(outputs, expectedVal, maxErrorAllowed, loopNum);
#endif
    cl_int cl_err;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device, NULL, NULL, NULL, &cl_err);
    logger.logCreateContext(cl_err);
#ifdef SW_EMU_TEST
    // hls::exp and hls::log have bug in multi-thread.
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE,
                       &cl_err); // | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
#else
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &cl_err);
#endif
    logger.logCreateCommandQueue(cl_err);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();

    std::cout << "Selected Device " << devName << "\n";

    cl::Program::Binaries xclbins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);

    cl::Program program(context, devices, xclbins, NULL, &cl_err);
    logger.logCreateProgram(cl_err);

    cl::Kernel kernel0[2];
    for (int i = 0; i < 2; ++i) {
        kernel0[i] = cl::Kernel(program, "McBarrierBiasedEngine_k", &cl_err);
    }
    logger.logCreateKernel(cl_err);

    cl_mem_ext_ptr_t mext_out_a;
    cl_mem_ext_ptr_t mext_out_b;
    mext_out_a = {9, out0_a, kernel0[0]()};
    mext_out_b = {9, out0_b, kernel0[1]()};

    cl::Buffer out_buff_a;
    cl::Buffer out_buff_b;

    out_buff_a = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            (size_t)(OUTDEP * sizeof(DtUsed)), &mext_out_a);
    out_buff_b = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            (size_t)(OUTDEP * sizeof(DtUsed)), &mext_out_b);

    std::vector<std::vector<cl::Event> > kernel_events(num_rep);
    std::vector<std::vector<cl::Event> > read_events(num_rep);
    for (int i = 0; i < num_rep; ++i) {
        kernel_events[i].resize(KN);
        read_events[i].resize(1);
    }
    int j = 0;
    kernel0[0].setArg(j++, loopNum);
    kernel0[0].setArg(j++, underlying);
    kernel0[0].setArg(j++, volatility);
    kernel0[0].setArg(j++, dividendYield);
    kernel0[0].setArg(j++, riskFreeRate);
    kernel0[0].setArg(j++, timeLength);
    kernel0[0].setArg(j++, barrier);
    kernel0[0].setArg(j++, strike);
    kernel0[0].setArg(j++, optionType);
    kernel0[0].setArg(j++, out_buff_a);
    kernel0[0].setArg(j++, rebate);
    kernel0[0].setArg(j++, requiredTolerance);
    kernel0[0].setArg(j++, requiredSamples);
    kernel0[0].setArg(j++, timeSteps);
    j = 0;
    kernel0[1].setArg(j++, loopNum);
    kernel0[1].setArg(j++, underlying);
    kernel0[1].setArg(j++, volatility);
    kernel0[1].setArg(j++, dividendYield);
    kernel0[1].setArg(j++, riskFreeRate);
    kernel0[1].setArg(j++, timeLength);
    kernel0[1].setArg(j++, barrier);
    kernel0[1].setArg(j++, strike);
    kernel0[1].setArg(j++, optionType);
    kernel0[1].setArg(j++, out_buff_b);
    kernel0[1].setArg(j++, rebate);
    kernel0[1].setArg(j++, requiredTolerance);
    kernel0[1].setArg(j++, requiredSamples);
    kernel0[1].setArg(j++, timeSteps);

    std::vector<cl::Memory> out_vec[2]; //{out_buff[0]};

    out_vec[0].push_back(out_buff_a);
    out_vec[1].push_back(out_buff_b);

    q.finish();
    gettimeofday(&st_time, 0);

    /*
    q.enqueueTask(kernel0[0], nullptr, &kernel_events[0][0]);
    q.finish();
    q.enqueueMigrateMemObjects(out_vec[0], CL_MIGRATE_MEM_OBJECT_HOST, &kernel_events[0], &read_events[0][0]);
    q.finish();
    */

    for (int i = 0; i < num_rep; ++i) {
        int use_a = i & 1;
        if (use_a) {
            if (i > 1) {
                q.enqueueTask(kernel0[0], &read_events[i - 2], &kernel_events[i][0]);
            } else {
                q.enqueueTask(kernel0[0], nullptr, &kernel_events[i][0]);
            }
        } else {
            if (i > 1) {
                q.enqueueTask(kernel0[1], &read_events[i - 2], &kernel_events[i][0]);
            } else {
                q.enqueueTask(kernel0[1], nullptr, &kernel_events[i][0]);
            }
        }
        if (use_a) {
            q.enqueueMigrateMemObjects(out_vec[0], CL_MIGRATE_MEM_OBJECT_HOST, &kernel_events[i], &read_events[i][0]);
        } else {
            q.enqueueMigrateMemObjects(out_vec[1], CL_MIGRATE_MEM_OBJECT_HOST, &kernel_events[i], &read_events[i][0]);
        }
    }

    q.flush();
    q.finish();

    gettimeofday(&end_time, 0);
    int exec_time = tvdiff(&st_time, &end_time);
    std::cout << "FPGA execution time of " << num_rep << " runs:" << exec_time / 1000 << " ms\n"
              << "Average executiom per run: " << exec_time / num_rep / 1000 << " ms\n";

    int err = 0;
    if (num_rep > 1) {
        err += printResult(out0_a, expectedVal, maxErrorAllowed, loopNum);
    }

    std::cout << "Execution time " << tvdiff(&st_time, &end_time) << std::endl;
    err += printResult(out0_b, expectedVal, maxErrorAllowed, loopNum);
    err ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return err;
}
