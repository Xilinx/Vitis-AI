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
#include "tree_engine_kernel.hpp"
#include "xf_utils_sw/logger.hpp"

#define XCL_BANK(n) (((unsigned int)(n)) | XCL_MEM_TOPOLOGY)

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
    std::cout << "\n----------------------Tree Bermudan (Extended Cox-Ingersoll-Ross) Engine-----------------\n";
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    // cmd parser
    ArgParser parser(argc, argv);
    std::string xclbin_path;
#ifndef HLS_TEST
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }
#endif
    // Allocate Memory in Host Memory
    DT* initTime_alloc = aligned_alloc<DT>(LEN);
    int* exerciseCnt_alloc = aligned_alloc<int>(ExerciseLen);
    int* fixedCnt_alloc = aligned_alloc<int>(FixedLen);
    int* floatingCnt_alloc = aligned_alloc<int>(FloatingLen);
    DT* output = aligned_alloc<DT>(1);

    // -------------setup k0 params---------------
    int err = 0;
    DT minErr = 10e-10;
    int timestep = 10;
    cout << "timestep=" << timestep << endl;

    double golden;

    if (timestep == 10) golden = 14.709005576867522;
    if (timestep == 50) golden = 13.55176914814926;
    if (timestep == 100) golden = 13.21804760550833;
    if (timestep == 500) golden = 13.17092869790816;
    if (timestep == 1000) golden = 13.16387131687744;
    double fixedRate = 0.049995924285639641;
    double initTime[12] = {0,
                           1,
                           1.4958904109589042,
                           2,
                           2.4986301369863013,
                           3.0027397260273974,
                           3.4986301369863013,
                           4.0027397260273974,
                           4.4986301369863018,
                           5.0027397260273974,
                           5.4986301369863018,
                           6.0027397260273974};

    int initSize = 12;
    int exerciseCnt[5] = {0, 2, 4, 6, 8};
    int fixedCnt[5] = {0, 2, 4, 6, 8};
    int floatingCnt[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    for (int i = 0; i < initSize; i++) {
        initTime_alloc[i] = initTime[i];
    }

    for (int i = 0; i < ExerciseLen; i++) {
        exerciseCnt_alloc[i] = exerciseCnt[i];
    }

    for (int i = 0; i < FixedLen; i++) {
        fixedCnt_alloc[i] = fixedCnt[i];
    }

    for (int i = 0; i < FloatingLen; i++) {
        floatingCnt_alloc[i] = floatingCnt[i];
    }
#ifndef HLS_TEST
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
    cl::Kernel kernel_TreeBermudanEngine(program, "TREE_k0", &cl_err);
    logger.logCreateKernel(cl_err);

    cl_mem_ext_ptr_t mext_o[5];
    mext_o[0] = {8, output, kernel_TreeBermudanEngine()};
    mext_o[1] = {3, initTime_alloc, kernel_TreeBermudanEngine()};
    mext_o[2] = {5, exerciseCnt_alloc, kernel_TreeBermudanEngine()};
    mext_o[3] = {7, fixedCnt_alloc, kernel_TreeBermudanEngine()};
    mext_o[4] = {6, floatingCnt_alloc, kernel_TreeBermudanEngine()};

    // create device buffer and map dev buf to host buf
    cl::Buffer output_buf;
    cl::Buffer initTime_buf, exerciseCnt_buf, fixedCnt_buf, floatingCnt_buf;
    output_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(DT) * N,
                            &mext_o[0]);
    initTime_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                              sizeof(DT) * LEN, &mext_o[1]);
    exerciseCnt_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                 sizeof(int) * ExerciseLen, &mext_o[2]);
    fixedCnt_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                              sizeof(int) * FixedLen, &mext_o[3]);
    floatingCnt_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                 sizeof(int) * FloatingLen, &mext_o[4]);

    std::vector<cl::Memory> ob_out;
    ob_out.push_back(output_buf);

    q.finish();
    // launch kernel and calculate kernel execution time
    std::cout << "kernel start------" << std::endl;
    gettimeofday(&start_time, 0);
    int loop_num = 1;
    for (int i = 0; i < loop_num; ++i) {
        kernel_TreeBermudanEngine.setArg(0, 0);
        kernel_TreeBermudanEngine.setArg(1, fixedRate);
        kernel_TreeBermudanEngine.setArg(2, timestep);
        kernel_TreeBermudanEngine.setArg(3, initTime_buf);
        kernel_TreeBermudanEngine.setArg(4, initSize);
        kernel_TreeBermudanEngine.setArg(5, exerciseCnt_buf);
        kernel_TreeBermudanEngine.setArg(6, floatingCnt_buf);
        kernel_TreeBermudanEngine.setArg(7, fixedCnt_buf);
        kernel_TreeBermudanEngine.setArg(8, output_buf);

        q.enqueueTask(kernel_TreeBermudanEngine, nullptr, nullptr);
    }

    q.finish();
    gettimeofday(&end_time, 0);
    std::cout << "kernel end------" << std::endl;
    std::cout << "Execution time " << tvdiff(&start_time, &end_time) / loop_num << "us" << std::endl;
    q.enqueueMigrateMemObjects(ob_out, 1, nullptr, nullptr);
    q.finish();

#else
    TREE_k0(0, fixedRate, timestep, initTime, initSize, exerciseCnt, floatingCnt, fixedCnt, output);
#endif

    DT out = output[0];
    if (std::fabs(out - golden) > minErr) err++;
    std::cout << "NPV= " << out << " ,diff/NPV= " << (out - golden) / golden << std::endl;
    err ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return err;
}
