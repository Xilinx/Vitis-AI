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
    std::cout << "\n----------------------Tree Bermudan (CIR) Engine-----------------\n";
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    // cmd parser
    std::string run_mode = "hw";
#ifndef HLS_TEST
    ArgParser parser(argc, argv);
    std::string mode;
    std::string xclbin_path;
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }

    if (std::getenv("XCL_EMULATION_MODE") != nullptr) {
        run_mode = std::getenv("XCL_EMULATION_MODE");
    }
    std::cout << "[INFO]Running in " << run_mode << " mode" << std::endl;
#endif

    ScanInputParam0* inputParam0_alloc = aligned_alloc<ScanInputParam0>(1);
    ScanInputParam1* inputParam1_alloc = aligned_alloc<ScanInputParam1>(1);

    // -------------setup params---------------
    int err = 0;
    DT minErr = 10e-10;
    int timestep = 10;
    int len = K;
    if (run_mode == "hw_emu") {
        timestep = 10;
    }

    std::cout << "timestep=" << timestep << std::endl;

    double golden;

    if (timestep == 10) golden = 39.878441781617973;
    if (timestep == 50) golden = 40.56088931110556;
    if (timestep == 100) golden = 40.67732609528822;
    if (timestep == 500) golden = 40.6262816851493;
    if (timestep == 1000) golden = 40.60965878956618;

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

    for (int i = 0; i < 1; i++) {
        inputParam1_alloc[i].index = i;
        inputParam1_alloc[i].type = 0;
        inputParam1_alloc[i].fixedRate = fixedRate;
        inputParam1_alloc[i].timestep = timestep;
        inputParam1_alloc[i].initSize = initSize;
        inputParam1_alloc[i].a = 0.043389447297063261;
        inputParam1_alloc[i].sigma = 0.068963597413997324;
        inputParam1_alloc[i].flatRate = 0.04875825;
        inputParam0_alloc[i].x0 = 0.18580295883843218;
        inputParam0_alloc[i].nominal = 1000.0;
        inputParam0_alloc[i].spread = 0.0;
        for (int j = 0; j < initSize; j++) {
            inputParam0_alloc[i].initTime[j] = initTime[j];
        }
        for (int j = 0; j < ExerciseLen; j++) {
            inputParam1_alloc[i].exerciseCnt[j] = exerciseCnt[j];
        }
        for (int j = 0; j < FloatingLen; j++) {
            inputParam1_alloc[i].floatingCnt[j] = floatingCnt[j];
        }
        for (int j = 0; j < FixedLen; j++) {
            inputParam1_alloc[i].fixedCnt[j] = fixedCnt[j];
        }
    }

#ifndef HLS_TEST
    cl_int cl_err;
    // do pre-process on CPU
    struct timeval start_time, end_time, test_time;
    // platform related operations
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Creating Context and Command Queue for selected Device
    cl::Context context(device, NULL, NULL, NULL, &cl_err);
    logger.logCreateContext(cl_err);
#ifdef SW_EMU_TEST
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE,
                       &cl_err); // | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
#else
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &cl_err);
#endif
    logger.logCreateCommandQueue(cl_err);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("Found Device=%s\n", devName.c_str());

    // load xclbin
    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &cl_err);
    logger.logCreateProgram(cl_err);

    std::string krnl_name = "scanTreeKernel";
    cl_uint cu_number;
    {
        cl::Kernel k(program, krnl_name.c_str());
        k.getInfo(CL_KERNEL_COMPUTE_UNIT_COUNT, &cu_number);
    }

    std::vector<cl::Kernel> krnl_TreeEngine(cu_number);
    for (cl_uint i = 0; i < cu_number; ++i) {
        std::string krnl_full_name = krnl_name + ":{" + krnl_name + "_" + std::to_string(i + 1) + "}";
        krnl_TreeEngine[i] = cl::Kernel(program, krnl_full_name.c_str(), &cl_err);
        logger.logCreateKernel(cl_err);
    }

    std::cout << "kernel has been created" << std::endl;

    std::vector<cl_mem_ext_ptr_t> mext_in0(cu_number);
    std::vector<cl_mem_ext_ptr_t> mext_in1(cu_number);
    std::vector<cl_mem_ext_ptr_t> mext_out(cu_number);

    std::vector<DT*> output(cu_number);
    for (int i = 0; i < cu_number; i++) {
        output[i] = aligned_alloc<DT>(N * K);
    }

    for (int c = 0; c < cu_number; ++c) {
        mext_in0[c] = {1, inputParam0_alloc, krnl_TreeEngine[c]()};
        mext_in1[c] = {2, inputParam1_alloc, krnl_TreeEngine[c]()};
        mext_out[c] = {3, output[c], krnl_TreeEngine[c]()};
    }

    // create device buffer and map dev buf to host buf
    std::vector<cl::Buffer> output_buf(cu_number);
    std::vector<cl::Buffer> inputParam0_buf(cu_number);
    std::vector<cl::Buffer> inputParam1_buf(cu_number);
    for (int i = 0; i < cu_number; i++) {
        inputParam0_buf[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                        sizeof(ScanInputParam0), &mext_in0[i]);
        inputParam1_buf[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                        sizeof(ScanInputParam1), &mext_in1[i]);
        output_buf[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(DT) * N * K, &mext_out[i]);
    }

    std::vector<cl::Memory> ob_in;
    for (int i = 0; i < cu_number; i++) {
        ob_in.push_back(inputParam0_buf[i]);
        ob_in.push_back(inputParam1_buf[i]);
    }

    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, nullptr);

    std::vector<cl::Memory> ob_out;
    for (int i = 0; i < cu_number; i++) {
        ob_out.push_back(output_buf[i]);
    }

    q.finish();

    // launch kernel and calculate kernel execution time
    std::cout << "kernel start------" << std::endl;
    std::vector<cl::Event> events_kernel(4);
    gettimeofday(&start_time, 0);

    for (int c = 0; c < cu_number; ++c) {
        int j = 0;
        krnl_TreeEngine[c].setArg(j++, len);
        krnl_TreeEngine[c].setArg(j++, inputParam0_buf[c]);
        krnl_TreeEngine[c].setArg(j++, inputParam1_buf[c]);
        krnl_TreeEngine[c].setArg(j++, output_buf[c]);
    }

    for (int i = 0; i < cu_number; ++i) {
        q.enqueueTask(krnl_TreeEngine[i], nullptr, &events_kernel[i]);
    }

    q.finish();
    gettimeofday(&end_time, 0);
    std::cout << "kernel end------" << std::endl;

    unsigned long time_start, time_end;
    unsigned long time1, time2;
    for (int c = 0; c < cu_number; ++c) {
        events_kernel[c].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
        events_kernel[c].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
        printf("Kernel-%d Execution time %d ms\n", c, (time2 - time1) / 1000000.0);
    }

    std::cout << "FPGA Execution time " << tvdiff(&start_time, &end_time) / 1000.0 << "ms" << std::endl;
    q.enqueueMigrateMemObjects(ob_out, 1, nullptr, nullptr);
    q.finish();
#endif
    for (int i = 0; i < cu_number; i++) {
        for (int j = 0; j < len; j++) {
            DT out = output[i][j];
            if (std::fabs(out - golden) > minErr) {
                err++;
                std::cout << "[ERROR] Kernel-" << i + 1 << ": NPV[" << j << "]= " << std::setprecision(15) << out
                          << " ,diff/NPV= " << (out - golden) / golden << std::endl;
            }
        }
    }
    std::cout << "NPV[" << 0 << "]= " << std::setprecision(15) << output[0][0]
              << " ,diff/NPV= " << (output[0][0] - golden) / golden << std::endl;
    err ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return err;
}
