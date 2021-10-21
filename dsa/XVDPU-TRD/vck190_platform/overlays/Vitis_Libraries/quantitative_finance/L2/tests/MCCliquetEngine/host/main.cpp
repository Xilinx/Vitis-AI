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
    int err = 0;
    // Allocate Memory in Host Memory
    TEST_DT* outputs = aligned_alloc<TEST_DT>(1);
    ap_uint<32>* seed = aligned_alloc<ap_uint<32> >(1);
    TEST_DT* resetDates2 = aligned_alloc<TEST_DT>(2048);

    // -------------setup params---------------
    bool types[] = {0, 1};
    TEST_DT moneyness[] = {0.9, 1.1};
    TEST_DT underlyings[] = {100.0};
    TEST_DT qRates[] = {0.04, 0.06};
    TEST_DT rRates[] = {0.01, 0.10};
    unsigned int lengths[] = {2, 4};
    TEST_DT vols[] = {0.10, 0.90};

    seed[0] = 42;

    TEST_DT requiredTolerance = 5.0e-3;
    TEST_DT results[] = {0.0873118,   0.278456, 0.118823,    0.282698, 0.0785721,   0.271838, 0.109742,    0.276142,
                         0.0923743,   0.219167, 0.109829,    0.224231, 0.0874834,   0.215801, 0.104954,    0.220852,
                         0.358523,    0.929269, 0.420919,    0.888853, 0.332381,    0.909461, 0.394953,    0.870093,
                         0.276555,    0.654187, 0.321037,    0.654232, 0.262017,    0.644186, 0.306878,    0.644418,
                         0.00189635,  0.205788, 0.00669363,  0.211512, 0.00134551,  0.20034,  0.00508764,  0.206047,
                         0.000403698, 0.138451, 0.00123045,  0.143428, 0.000304877, 0.135943, 0.000964477, 0.140878,
                         0.00566046,  0.614654, 0.0191094,   0.604084, 0.00401625,  0.59839,  0.0145244,   0.588483,
                         0.00118346,  0.412437, 0.00352143,  0.417644, 0.000894204, 0.404996, 0.00276098,  0.410248,
                         0.00322334,  0.194368, 0.000668712, 0.164544, 0.00429215,  0.197558, 0.000955765, 0.167355,
                         0.000482532, 0.127275, 0.000129675, 0.114532, 0.000624946, 0.128942, 0.00017447,  0.116072,
                         0.0096216,   0.580368, 0.00190901,  0.469844, 0.012812,    0.589891, 0.00272852,  0.477869,
                         0.00140876,  0.379041, 0.000370821, 0.333566, 0.00182411,  0.383993, 0.000498677, 0.338039,
                         0.115785,    0.319677, 0.0692058,   0.274024, 0.125042,    0.324037, 0.0769674,   0.277927,
                         0.107492,    0.24554,  0.0815662,   0.223764, 0.112427,    0.248065, 0.0862191,   0.226133,
                         0.443675,    1.05267,  0.279016,    0.863991, 0.471362,    1.06574,  0.301216,    0.875175,
                         0.321478,    0.732732, 0.238898,    0.653021, 0.336142,    0.740244, 0.252425,    0.659912};

    TEST_DT resetDates[2][1024];
    TEST_DT dt[2][6] = {
        {0.51111111111111107, 0.50555555555555554, 0.5083333333333333, 0.50277777777777777, 0.0027777777777777679, 0},
        {0.25555555555555554, 0.25555555555555554, 0.25, 0.25555555555555554, 0, 0}};
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 6; ++i) {
            if (i == 0) {
                resetDates[j][i] = dt[j][i];
                resetDates2[j * 1024 + i] = dt[j][i];
            } else {
                resetDates[j][i] = dt[j][i] + resetDates[j][i - 1];
                resetDates2[j * 1024 + i] = dt[j][i] + resetDates[j][i - 1];
            }
        }
    }
    int tp_nm = 1, mn_nm = 1, ln_nm = 1, fq_nm = 1, udly_num = 1, q_nm = 1, r_nm = 1, v_nm = 1;

    if (mode_emu != "hw_emu") {
        tp_nm = LENGTH(types);
        mn_nm = LENGTH(moneyness);
        ln_nm = LENGTH(lengths);
        fq_nm = 2;
        udly_num = LENGTH(underlyings);
        q_nm = LENGTH(qRates);
        r_nm = LENGTH(rRates);
        v_nm = LENGTH(vols);
    }
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

    // cl::Program::Binaries xclBins =
    // xcl::import_binary_file("../xclbin/MCAE_u250_hw.xclbin");
    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &cl_err);
    logger.logCreateProgram(cl_err);
    cl::Kernel kernel_Engine(program, "MCCliquetEngine_k", &cl_err);
    logger.logCreateKernel(cl_err);

    cl_mem_ext_ptr_t mext_in_0;
    cl_mem_ext_ptr_t mext_in_1[2];
    cl_mem_ext_ptr_t mext_in_2;
    mext_in_0 = {9, outputs, kernel_Engine()};
    mext_in_1[0] = {7, &resetDates2[0], kernel_Engine()};
    mext_in_1[1] = {7, &resetDates2[1024], kernel_Engine()};
    ;
    mext_in_2 = {8, seed, kernel_Engine()};
    ;

    // create device buffer and map dev buf to host buf
    cl::Buffer output_buf;
    cl::Buffer seed_buf, resetDates_buf[2];
    output_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(TEST_DT),
                            &mext_in_0);
    resetDates_buf[0] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(TEST_DT) * 1024, &mext_in_1[0]);
    resetDates_buf[1] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(TEST_DT) * 1024, &mext_in_1[1]);
    seed_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                          sizeof(unsigned int), &mext_in_2);
    int idx = 0;
    for (int i = 0; i < tp_nm; ++i) {
        for (int j = 0; j < mn_nm; ++j) {
            for (int k = 0; k < ln_nm; ++k) {
                for (int kk = 0; kk < fq_nm; ++kk) {
                    for (int l = 0; l < udly_num; ++l) {
                        for (int m = 0; m < q_nm; ++m) {
                            for (int n = 0; n < r_nm; ++n) {
                                for (int p = 0; p < v_nm; ++p) {
                                    TEST_DT underlying = underlyings[l];
                                    TEST_DT riskFreeRate = rRates[n];
                                    TEST_DT volatility = vols[p];
                                    TEST_DT dividendYield = qRates[m];
                                    TEST_DT strike = moneyness[j];
                                    int optionType = types[i];
                                    TEST_DT timeLength = 1;
                                    unsigned int timeSteps;
                                    if (k == 1 && kk == 0)
                                        timeSteps = lengths[k] + 1;
                                    else
                                        timeSteps = lengths[k];
                                    // TEST_DT outputs[1];
                                    // do pre-process on CPU

                                    TEST_DT golden = results[idx];
                                    TEST_DT maxError = 0.015;
                                    unsigned int requiredSamples = 0;

                                    std::cout << "Put option:\n"
                                              << "   strike:              " << strike << "\n"
                                              << "   underlying:          " << underlying << "\n"
                                              << "   risk-free rate:      " << riskFreeRate << "\n"
                                              << "   volatility:          " << volatility << "\n"
                                              << "   dividend yield:      " << dividendYield << "\n"
                                              << "   maturity:            " << timeLength << "\n"
                                              << "   tolerance:           " << requiredTolerance << "\n"
                                              << "   requaried samples:   " << requiredSamples << "\n"
                                              << "   timesteps:           " << timeSteps << "\n"
                                              << "   golden:              " << golden << "\n";

                                    std::vector<cl::Memory> ob_out;
                                    ob_out.push_back(output_buf);

                                    q.finish();
                                    // launch kernel and calculate kernel execution time
                                    std::cout << "kernel start------" << std::endl;
                                    gettimeofday(&start_time, 0);
                                    int a = 0;
                                    kernel_Engine.setArg(a++, underlying);
                                    kernel_Engine.setArg(a++, volatility);
                                    kernel_Engine.setArg(a++, dividendYield);
                                    kernel_Engine.setArg(a++, riskFreeRate);
                                    kernel_Engine.setArg(a++, timeLength);
                                    kernel_Engine.setArg(a++, strike);
                                    kernel_Engine.setArg(a++, optionType);
                                    kernel_Engine.setArg(a++, resetDates_buf[kk]);
                                    kernel_Engine.setArg(a++, seed_buf);
                                    kernel_Engine.setArg(a++, output_buf);
                                    kernel_Engine.setArg(a++, requiredTolerance);
                                    kernel_Engine.setArg(a++, timeSteps);
                                    kernel_Engine.setArg(a++, requiredSamples);

                                    q.enqueueTask(kernel_Engine, nullptr, nullptr);

                                    q.finish();
                                    gettimeofday(&end_time, 0);
                                    std::cout << "kernel end------" << std::endl;
                                    std::cout << "Execution time " << tvdiff(&start_time, &end_time) << "us"
                                              << std::endl;
                                    q.enqueueMigrateMemObjects(ob_out, 1, nullptr, nullptr);
                                    q.finish();
                                    // do post-process on CPU
                                    TEST_DT diff = std::fabs(outputs[0] - golden);
                                    if (diff > maxError) {
                                        std::cout << "Output is wrong!" << std::endl;
                                        std::cout << "Acutal value: " << outputs[0] << ", Expected value: " << golden
                                                  << std::endl;
                                        err++;
                                        // return -1;
                                    }
                                    std::cout << "Output: " << outputs[0] << std::endl;
                                    idx++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    err ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return err;
}
