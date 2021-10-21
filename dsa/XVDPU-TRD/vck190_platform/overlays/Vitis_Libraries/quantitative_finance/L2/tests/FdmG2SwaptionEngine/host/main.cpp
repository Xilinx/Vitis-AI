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
#include "fdmg2_engine_kernel.hpp"
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

int main(int argc, const char* argv[]) {
    std::cout << "\n----------------------Fdm g2 Engine-----------------\n";
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);

    // cmd parser
    ArgParser parser(argc, argv);
    std::string xclbin_path;
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }
    // Allocate Memory in Host Memory
    double* output = aligned_alloc<double>(1);

    // -------------setup k0 params---------------

    int nerror = 0;

    double maturity = 5.0027397260273974;
    double epsilon = 1e-5;
    double nominal = 1000.0;
    unsigned int xGrid = 5;
    unsigned int yGrid = 5;

    // g2 model
    double r = 0.04875825;
    double fixedRate = 0.049995978501700372;
    double a = 0.050055733653096922, sigma = 0.0094424342056787739, b = 0.050052910248222851,
           eta = 0.0094424313463861171, rho = -0.76300324120391616;
    double t = 5.0027397260273974, T = 6.0027397260273974;
    double factor[2] = {-0.079888357349334832, -0.079888850463537983};

    // innervalue
    double Dates[EXSize + 2] = {
        37306, // year 0
        37671, // year 1
        38036, // year 2
        38402, // year 3
        38767, // year 4
        39132, // year 5
        39497  // year 6
    };
    double Dates_floating[6] = {37671, 38036, 38404, 38768, 39132, 39497};
    double theta = 0.78867513459481287;
    double mu = 0.5;
    double* stoppingTimes = aligned_alloc<double>(EXSize + 1);
    ;
    for (int i = 0; i <= EXSize; i++) {
        if (0 == i) {
            stoppingTimes[i] = 0.99 * (1.0 / 365.0);
        } else {
            stoppingTimes[i] = (Dates[i] - Dates[0]) / 365.0;
        }
    }

    // fixedAccrualTime
    double* fixedAccrualTime = aligned_alloc<double>(EXSize + 1);
    double* fixedAccralPeriod = aligned_alloc<double>(EXSize + 1);
    double* floatingAccrualTime = aligned_alloc<double>(EXSize + 1);
    double* floatingAccrualPeriod = aligned_alloc<double>(EXSize + 1);
    double* iborTime = aligned_alloc<double>(EXSize + 1);
    double* iborPeriod = aligned_alloc<double>(EXSize + 1);
    for (int i = 0; i <= EXSize; i++) {
        fixedAccrualTime[i] = (Dates[i + 1] - Dates[0]) / 365.0;
    }
    for (int i = 0; i <= EXSize; i++) {
        fixedAccralPeriod[i] = (Dates_floating[i] - Dates[0]) / 365.0;
    }
    // floatingAccrualTime
    for (int i = 0; i <= EXSize; i++) {
        floatingAccrualTime[i] = (Dates[i + 1] - Dates[0]) / 365.0;
    }
    // floatingAccrualPeriod
    for (int i = 0; i <= EXSize; i++) {
        floatingAccrualPeriod[i] = (Dates[i + 1] - Dates[0]) / 360.0;
    }
    // iborTime
    for (int i = 0; i <= EXSize; i++) {
        iborTime[i] = (Dates_floating[i] - Dates[0]) / 365.0;
    }
    // iborPeriod
    for (int i = 0; i <= EXSize; i++) {
        iborPeriod[i] = (Dates_floating[i] - Dates[0]) / 360.0;
    }

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
    cl::Kernel kernel_fdmg2Engine(program, "FDMG2_k0", &cl_err);
    logger.logCreateKernel(cl_err);

    cl_mem_ext_ptr_t mext_o[6];
    mext_o[0] = {19, output, kernel_fdmg2Engine()};
    mext_o[1] = {14, stoppingTimes, kernel_fdmg2Engine()};
    mext_o[2] = {15, fixedAccrualTime, kernel_fdmg2Engine()};
    mext_o[3] = {16, floatingAccrualPeriod, kernel_fdmg2Engine()};
    mext_o[4] = {17, iborTime, kernel_fdmg2Engine()};
    mext_o[5] = {18, iborPeriod, kernel_fdmg2Engine()};

    // create device buffer and map dev buf to host buf
    cl::Buffer output_buf;
    cl::Buffer stoppingTimes_buf, fixedAccrualTime_buf, floatingAccrualPeriod_buf, iborTime_buf, iborPeriod_buf;
    output_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double),
                            &mext_o[0]);
    stoppingTimes_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(double) * (EXSize + 1), &mext_o[1]);
    fixedAccrualTime_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                      sizeof(double) * (EXSize + 1), &mext_o[2]);
    floatingAccrualPeriod_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                           sizeof(double) * (EXSize + 1), &mext_o[3]);
    iborTime_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                              sizeof(double) * (EXSize + 1), &mext_o[4]);
    iborPeriod_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                sizeof(double) * (EXSize + 1), &mext_o[5]);

    std::vector<cl::Memory> ob_in;
    ob_in.push_back(stoppingTimes_buf);
    ob_in.push_back(fixedAccrualTime_buf);
    ob_in.push_back(floatingAccrualPeriod_buf);
    ob_in.push_back(iborTime_buf);
    ob_in.push_back(iborPeriod_buf);
    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, nullptr);
    std::vector<cl::Memory> ob_out;
    ob_out.push_back(output_buf);

    q.finish();
    // launch kernel and calculate kernel execution time
    std::cout << "kernel start------" << std::endl;
    gettimeofday(&start_time, 0);
    for (int i = 0; i < 1; ++i) {
        int j = 0;
        kernel_fdmg2Engine.setArg(j++, a);
        kernel_fdmg2Engine.setArg(j++, sigma);
        kernel_fdmg2Engine.setArg(j++, b);
        kernel_fdmg2Engine.setArg(j++, eta);
        kernel_fdmg2Engine.setArg(j++, rho);
        kernel_fdmg2Engine.setArg(j++, STEPS);
        kernel_fdmg2Engine.setArg(j++, xGrid);
        kernel_fdmg2Engine.setArg(j++, yGrid);
        kernel_fdmg2Engine.setArg(j++, epsilon);
        kernel_fdmg2Engine.setArg(j++, theta);
        kernel_fdmg2Engine.setArg(j++, mu);
        kernel_fdmg2Engine.setArg(j++, fixedRate);
        kernel_fdmg2Engine.setArg(j++, r);
        kernel_fdmg2Engine.setArg(j++, nominal);
        kernel_fdmg2Engine.setArg(j++, stoppingTimes_buf);
        kernel_fdmg2Engine.setArg(j++, fixedAccrualTime_buf);
        kernel_fdmg2Engine.setArg(j++, floatingAccrualPeriod_buf);
        kernel_fdmg2Engine.setArg(j++, iborTime_buf);
        kernel_fdmg2Engine.setArg(j++, iborPeriod_buf);
        kernel_fdmg2Engine.setArg(j++, output_buf);

        q.enqueueTask(kernel_fdmg2Engine, nullptr, nullptr);
    }

    q.finish();
    gettimeofday(&end_time, 0);
    std::cout << "kernel end------" << std::endl;
    std::cout << "Execution time " << tvdiff(&start_time, &end_time) << "us" << std::endl;
    q.enqueueMigrateMemObjects(ob_out, 1, nullptr, nullptr);
    q.finish();
#else
    FDMG2_k0(a, sigma, b, eta, rho, STEPS, xGrid, yGrid, epsilon, theta, mu, fixedRate, r, nominal, stoppingTimes,
             fixedAccrualTime, floatingAccrualPeriod, iborTime, iborPeriod, output);
#endif
    double out = output[0];
    double golden = 10.139327717152; // 229.843923204834;//14.149595735802;
    if (std::fabs(out - golden) > 1.0e-10) nerror++;
    std::cout << "NPV= " << std::setprecision(15) << out << " ,diff/NPV= " << (out - golden) / golden << std::endl;
    nerror ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
           : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return nerror;
}
