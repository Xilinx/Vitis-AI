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
#include <iomanip>
#ifndef HLS_TEST
#include "xcl2.hpp"
#endif

#include <math.h>
#include "kernel_MCAsianAPEngine.hpp"
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

struct TestSuite {
    int fixings;
    double result;
};

int main(int argc, const char* argv[]) {
    std::cout << "\n----------------------MC(AsianAP) Engine-----------------\n";
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);

    // cmd parser
    ArgParser parser(argc, argv);
    std::string xclbin_path;

    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }

    std::string mode = "hw";
    if (std::getenv("XCL_EMULATION_MODE") != nullptr) {
        mode = std::getenv("XCL_EMULATION_MODE");
    }
    std::cout << "[INFO]Running in " << mode << " mode" << std::endl;

    struct timeval st_time, end_time;
    TEST_DT* outputs = aligned_alloc<TEST_DT>(1);

    // test data
    unsigned int timeSteps = 1;
    TEST_DT requiredTolerance = 0.02;
    TEST_DT underlying = 90;
    TEST_DT riskFreeRate = 0.025;
    TEST_DT volatility = 0.13;
    TEST_DT dividendYield = 0.06;
    TEST_DT strike = 87;
    int optionType = 1;
    TEST_DT timeLength = 11.0 / 12.0;

    unsigned int requiredSamples = 0;
    unsigned int maxSamples = 0;

    std::cout << "Call option:\n"
              << "   strike:              " << strike << "\n"
              << "   underlying:          " << underlying << "\n"
              << "   risk-free rate:      " << riskFreeRate << "\n"
              << "   volatility:          " << volatility << "\n"
              << "   dividend yield:      " << dividendYield << "\n"
              << "   maturity:            " << timeLength << "\n"
              << "   tolerance:           " << requiredTolerance << "\n"
              << "   requaried samples:   " << requiredSamples << "\n"
              << "   maximum samples:     " << maxSamples << "\n";

    // Test suite
    TestSuite tests[] = {{2, 1.3942835683},   {4, 1.5852442983},   {8, 1.66970673},     {12, 1.6980019214},
                         {26, 1.7255070456},  {52, 1.7401553533},  {100, 1.7478303712}, {250, 1.7490291943},
                         {500, 1.7515113291}, {1000, 1.7537344885}};

    int ret = 0;
#ifdef HLS_TEST
    kernel_MCAsianAP_0(underlying, volatility, dividendYield, riskFreeRate, timeLength, strike, optionType, outputs,
                       requiredTolerance, requiredSamples, timeSteps, maxSamples);
    std::cout << "output ====== " << outputs[0] << std::endl;
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

    cl::Kernel kernel_asianAP;
    kernel_asianAP = cl::Kernel(program, "kernel_MCAsianAP_0", &cl_err);
    logger.logCreateKernel(cl_err);

    cl_mem_ext_ptr_t mext_out;
    mext_out = {7, outputs, kernel_asianAP()};

    cl::Buffer out_buff;
    out_buff = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                          (size_t)(1 * sizeof(TEST_DT)), &mext_out);

    kernel_asianAP.setArg(0, underlying);
    kernel_asianAP.setArg(1, volatility);
    kernel_asianAP.setArg(2, dividendYield);
    kernel_asianAP.setArg(3, riskFreeRate);
    kernel_asianAP.setArg(4, timeLength);
    kernel_asianAP.setArg(5, strike);
    kernel_asianAP.setArg(6, optionType);
    kernel_asianAP.setArg(7, out_buff);
    kernel_asianAP.setArg(8, requiredTolerance);
    kernel_asianAP.setArg(9, requiredSamples);
    //    kernel_asianAP.setArg(10, timeSteps);
    kernel_asianAP.setArg(11, maxSamples);

    std::vector<cl::Memory> out_vec; //{out_buff[0]};
    out_vec.push_back(out_buff);

    int run_num = 10;
    if (mode == "hw_emu") {
        run_num = 1;
    }
    std::vector<std::vector<cl::Event> > kernel_events(run_num);
    std::vector<std::vector<cl::Event> > read_events(run_num);
    for (int i = 0; i < run_num; ++i) {
        kernel_events[i].resize(1);
        read_events[i].resize(1);
    }

    // save the output results
    TEST_DT result[run_num] = {0};

    q.finish();
    gettimeofday(&st_time, 0);

    // first run
    timeSteps = tests[0].fixings - 1;
    kernel_asianAP.setArg(10, timeSteps);
    q.enqueueTask(kernel_asianAP, nullptr, &kernel_events[0][0]);
    q.enqueueMigrateMemObjects(out_vec, CL_MIGRATE_MEM_OBJECT_HOST, &kernel_events[0], &read_events[0][0]);

    q.finish();
    result[0] = outputs[0];
    for (int i = 1; i < run_num; i++) {
        timeSteps = tests[i].fixings - 1;
        kernel_asianAP.setArg(10, timeSteps);
        q.enqueueTask(kernel_asianAP, &read_events[i - 1], &kernel_events[i][0]);
        q.enqueueMigrateMemObjects(out_vec, CL_MIGRATE_MEM_OBJECT_HOST, &kernel_events[i], &read_events[i][0]);
        q.finish();
        result[i] = outputs[0];
    }

    q.finish();
    gettimeofday(&end_time, 0);

    TEST_DT diff = 0;
    for (int i = 0; i < run_num; i++) {
        std::cout << "output[" << i << "] = " << std::setprecision(12) << result[i] << ",   "
                  << "golden[" << i << "] = " << tests[i].result << std::endl;
        diff = std::fabs(result[i] - tests[i].result);
        if (diff > requiredTolerance) {
            ret++;
        }
    }
    std::cout << "Execution time " << tvdiff(&st_time, &end_time) / 1000 << " us" << std::endl;
#endif

    ret ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return ret;
}
