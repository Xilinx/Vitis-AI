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
#include "utils.hpp"
#ifndef HLS_TEST
#include "xcl2.hpp"
#endif
#define KN 1

#include <math.h>
#include "kernel_mceuropeanengine.hpp"
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
bool print_result(double* out1, double golden, double tol, int loop_nm) {
    bool passed = true;
    for (int i = 0; i < loop_nm; ++i) {
        if (std::fabs(out1[i] - golden) > tol) {
            std::cout << "loop_nm: " << loop_nm << ", Expected value: " << golden << ::std::endl;
            std::cout << "FPGA result: " << out1[0] << std::endl;
            passed = false;
        }
    }
    return passed;
}

int main(int argc, const char* argv[]) {
    std::cout << "\n----------------------MC(European) Engine-----------------\n";
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    // cmd parser
    ArgParser parser(argc, argv);
    std::string xclbin_path;

    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }

    int err = 0;
    struct timeval st_time, end_time;
    DtUsed* out0_a = aligned_alloc<DtUsed>(OUTDEP);

    DtUsed* out0_b = aligned_alloc<DtUsed>(OUTDEP);
    // test data
    unsigned int timeSteps = 12;
    DtUsed requiredTolerance = 0.02;
    DtUsed underlying = 36;
    DtUsed riskFreeRate = 0.06;
    DtUsed sigma = 0.001;
    DtUsed v0 = 0.04;
    DtUsed theta = 0.04;
    DtUsed kappa = 1.0;
    DtUsed rho = 0.0;
    DtUsed dividendYield = 0.0;
    DtUsed strike = 40;
    bool optionType = 1;
    DtUsed timeLength = 1;

    unsigned int requiredSamples = 48128; // 0;//1024;//0;

    unsigned int maxSamples = 0;
    //
    unsigned int loop_nm = 1; // 1000;
    DtUsed golden = 3.83;
    DtUsed tol = 0.02;
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
            requiredSamples = 48128;
        }
    }
    std::string mode_emu = "hw";
    if (std::getenv("XCL_EMULATION_MODE") != nullptr) {
        mode_emu = std::getenv("XCL_EMULATION_MODE");
    }
    std::cout << "[INFO]Running in " << mode_emu << " mode" << std::endl;
    if (mode_emu.compare("hw_emu") == 0) {
        requiredSamples = 1024;
        golden = 3.92513;
        tol = 1e-5;
    }
#endif

#ifdef HLS_TEST
    kernel_mc_0(loop_nm, underlying, riskFreeRate, sigma, v0, theta, kappa, rho, dividendYield, optionType, strike,
                timeLength, timeSteps, requiredSamples, maxSamples, requiredTolerance, out0_b);
    print_result(out0_b, golden, tol, 1);
#else

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
        kernel0[i] = cl::Kernel(program, "kernel_mc_0", &cl_err);
    }
    logger.logCreateKernel(cl_err);

    cl_mem_ext_ptr_t mext_out_a[KN];
    cl_mem_ext_ptr_t mext_out_b[KN];
    mext_out_a[0] = {16, out0_a, kernel0[0]()};
    mext_out_b[0] = {16, out0_b, kernel0[1]()};

    cl::Buffer out_buff_a[KN];
    cl::Buffer out_buff_b[KN];
    for (int i = 0; i < KN; i++) {
        out_buff_a[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   (size_t)(OUTDEP * sizeof(DtUsed)), &mext_out_a[i]);
        out_buff_b[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   (size_t)(OUTDEP * sizeof(DtUsed)), &mext_out_b[i]);
    }
    std::vector<std::vector<cl::Event> > kernel_events(num_rep);
    std::vector<std::vector<cl::Event> > read_events(num_rep);
    for (int i = 0; i < num_rep; ++i) {
        kernel_events[i].resize(KN);
        read_events[i].resize(1);
    }
    int j = 0;
    kernel0[0].setArg(j++, loop_nm);
    kernel0[0].setArg(j++, underlying);
    kernel0[0].setArg(j++, riskFreeRate);
    kernel0[0].setArg(j++, sigma);
    kernel0[0].setArg(j++, v0);
    kernel0[0].setArg(j++, theta);
    kernel0[0].setArg(j++, kappa);
    kernel0[0].setArg(j++, rho);
    kernel0[0].setArg(j++, dividendYield);
    kernel0[0].setArg(j++, optionType);
    kernel0[0].setArg(j++, strike);
    kernel0[0].setArg(j++, timeLength);
    kernel0[0].setArg(j++, timeSteps);
    kernel0[0].setArg(j++, requiredSamples);
    kernel0[0].setArg(j++, maxSamples);
    kernel0[0].setArg(j++, requiredTolerance);
    kernel0[0].setArg(j++, out_buff_a[0]);
    j = 0;
    kernel0[1].setArg(j++, loop_nm);
    kernel0[1].setArg(j++, underlying);
    kernel0[1].setArg(j++, riskFreeRate);
    kernel0[1].setArg(j++, sigma);
    kernel0[1].setArg(j++, v0);
    kernel0[1].setArg(j++, theta);
    kernel0[1].setArg(j++, kappa);
    kernel0[1].setArg(j++, rho);
    kernel0[1].setArg(j++, dividendYield);
    kernel0[1].setArg(j++, optionType);
    kernel0[1].setArg(j++, strike);
    kernel0[1].setArg(j++, timeLength);
    kernel0[1].setArg(j++, timeSteps);
    kernel0[1].setArg(j++, requiredSamples);
    kernel0[1].setArg(j++, maxSamples);
    kernel0[1].setArg(j++, requiredTolerance);
    kernel0[1].setArg(j++, out_buff_b[0]);

    std::vector<cl::Memory> out_vec[2]; //{out_buff[0]};
    for (int i = 0; i < KN; ++i) {
        out_vec[0].push_back(out_buff_a[i]);
        out_vec[1].push_back(out_buff_b[i]);
    }
    q.finish();
    gettimeofday(&st_time, 0);
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

    if (num_rep > 1) {
        bool passed = print_result(out0_a, golden, tol, loop_nm);
        if (!passed) {
            err++;
            // return -1;
        }
    }
    bool passed = print_result(out0_b, golden, tol, loop_nm);
    if (!passed) {
        err++;
        // return -1;
    }

    std::cout << "Execution time " << tvdiff(&st_time, &end_time) << std::endl;
#endif

    err ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return err;
}
