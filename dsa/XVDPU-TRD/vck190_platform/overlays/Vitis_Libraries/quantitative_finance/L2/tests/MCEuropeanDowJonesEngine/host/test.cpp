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
#include <math.h>
#include <iostream>

#include "kernel_mceuropeanengine.hpp"
#include "utils.hpp"
#include "xcl2.hpp"
#include "xf_utils_sw/logger.hpp"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define KN 1
#define NUM_ASSETS 30
#define DIA_DIVISOR 0.14748071991788

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
        for (int i = 1; i < argc; ++i) {
            mTokens.push_back(std::string(argv[i]));
        }
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
    std::cout << "\n----------------------MC(European) DIA "
                 "Engine----------------------\n";

    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    // input parameters (per asset)
    DtUsed underlying[NUM_ASSETS] = {
        163.69, // MMM
        117.38, // AXP
        182.06, // AAPL
        347.82, // BA
        122.38, // CAT
        116.65, // CVX
        54.00,  // CSCO
        50.64,  // KO
        135.32, // DIS
        49.53,  // DOW
        72.77,  // XOM
        187.86, // GS
        194.16, // HD
        131.49, // IBM
        44.27,  // INTC
        134.14, // JNJ
        109.09, // JPM
        199.87, // MCD
        81.94,  // MRK
        124.86, // MSFT
        82.31,  // NKE
        42.67,  // PFE
        106.31, // PG
        148.56, // TRV
        130.23, // UTX
        244.16, // UNH
        57.34,  // VZ
        163.21, // V
        103.97, // WMT
        50.81   // WBA
    };

    // common
    bool optionType = 1;
    DtUsed strike = 0.0;
    DtUsed riskFreeRate = 0.03;
    DtUsed volatility = 0.20;
    DtUsed dividendYield = 0.0;
    DtUsed timeLength = 1.0;
    DtUsed requiredTolerance = 0.02;
    unsigned int requiredSamples = 1024;
    unsigned int maxSamples = 0;
    unsigned int timeSteps = 1;
    unsigned int loop_nm = 1;
    DtUsed expectedDIA = 262.223;

    // outputs
    DtUsed optionValue[NUM_ASSETS] = {};
    DtUsed optionValueSum = 0;
    DtUsed optionValueDIA = 0;

    ArgParser parser(argc, argv);
    std::string xclbin_path;

    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }

    cl_int cl_err;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device, NULL, NULL, NULL, &cl_err);
    logger.logCreateContext(cl_err);
    std::vector<cl::Device> deviceList;
    deviceList.push_back(device);

#ifdef SW_EMU_TEST
    // hls::exp and hls::log have bug in multi-thread.
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &cl_err);
#else
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &cl_err);
#endif
    logger.logCreateCommandQueue(cl_err);

    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Selected Device " << devName << "\n";

    cl::Program::Binaries xclbins = xcl::import_binary_file(xclbin_path);
    cl::Program program(context, deviceList, xclbins, NULL, &cl_err);
    logger.logCreateProgram(cl_err);

    cl::Kernel kernel0;
    kernel0 = cl::Kernel(program, "kernel_mc_0", &cl_err);
    logger.logCreateKernel(cl_err);

    DtUsed* out0 = aligned_alloc<DtUsed>(OUTDEP);
    cl_mem_ext_ptr_t mext_out;
#ifndef USE_HBM
    mext_out = {XCL_MEM_DDR_BANK1, out0, 0};
#else
    mext_out = {XCL_BANK1, out0, 0};
#endif

    cl::Buffer out_buff;
    out_buff = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                          (size_t)(OUTDEP * sizeof(DtUsed)), &mext_out);

    std::cout << "COMMON" << std::endl
              << "  strike:           " << strike << std::endl
              << "  maturity:         " << timeLength << std::endl
              << "  tolerance:        " << requiredTolerance << std::endl
              << "  required samples: " << requiredSamples << std::endl
              << "  maximum samples:  " << maxSamples << std::endl
              << "  timesteps:        " << timeSteps << std::endl
              << std::endl;

    std::string mode_emu = "hw";
    if (std::getenv("XCL_EMULATION_MODE") != nullptr) {
        mode_emu = std::getenv("XCL_EMULATION_MODE");
    }
    std::cout << "[INFO]Running in " << mode_emu << " mode" << std::endl;
    int asset_nm = NUM_ASSETS;

    if (mode_emu.compare("hw_emu") == 0) {
        requiredSamples = 256;
        requiredTolerance = 0.05;
        asset_nm = 1;
        expectedDIA = 11.4478;
    }

    for (int i = 0; i < asset_nm; i++) {
        int j = 0;

        kernel0.setArg(j++, loop_nm);
        kernel0.setArg(j++, underlying[i]);
        kernel0.setArg(j++, volatility);
        kernel0.setArg(j++, dividendYield);
        kernel0.setArg(j++, riskFreeRate);
        kernel0.setArg(j++, timeLength);
        kernel0.setArg(j++, strike);
        kernel0.setArg(j++, optionType);
        kernel0.setArg(j++, out_buff);
        kernel0.setArg(j++, requiredTolerance);
        kernel0.setArg(j++, requiredSamples);
        kernel0.setArg(j++, timeSteps);
        kernel0.setArg(j++, maxSamples);

        q.enqueueTask(kernel0, nullptr, nullptr);

        q.flush();
        q.finish();

        std::vector<cl::Memory> out_vec[1];
        out_vec[0].push_back(out_buff);

        q.enqueueMigrateMemObjects(out_vec[0], CL_MIGRATE_MEM_OBJECT_HOST, nullptr, nullptr);

        q.flush();
        q.finish();

        optionValue[i] = out0[0];
        optionValueSum += optionValue[i];

        std::cout << "ASSET[" << i << "]:" << std::endl
                  << "  underlying:     " << underlying[i] << std::endl
                  << "  risk-free rate: " << riskFreeRate << std::endl
                  << "  volatility:     " << volatility << std::endl
                  << "  dividend yield: " << dividendYield << std::endl
                  << "  --              " << std::endl
                  << "  option value:   " << optionValue[i] << std::endl
                  << std::endl;
    }

    optionValueDIA = (optionValueSum / DIA_DIVISOR / 100);

    std::cout << "DIA:" << std::endl;
    std::cout << "  option value: " << optionValueDIA << std::endl << std::endl;

    std::cout << "strike \tcall \tput" << std::endl;
    std::cout << "------ \t---- \t---" << std::endl;
    for (strike = 250.0; strike <= 275.0; strike += 5.0) {
        DtUsed payoff_put, payoff_call;

        payoff_put = MAX((strike - optionValueDIA), 0);
        payoff_call = MAX((optionValueDIA - strike), 0);

        std::cout << strike << "\t" << payoff_call << "\t" << payoff_put << std::endl;
    }

    // quick fix to get pass/fail criteria
    int ret = 0;
    if (std::abs(optionValueDIA - expectedDIA) > 0.1) {
        ret = 1;
    }
    ret ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);

    return ret;
}
