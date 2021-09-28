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
#include "discounting_bond_engine_kernel.hpp"
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
    std::cout << "\n----------------------Zero Coupon Dsicounting Bond Engine-----------------\n";
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    // cmd parser
    ArgParser parser(argc, argv);
    std::string xclbin_path;
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }
    // Allocate Memory in Host Memory
    double* times_alloc = aligned_alloc<double>(LEN);
    double* discs_alloc = aligned_alloc<double>(LEN);
    DT* output = aligned_alloc<DT>(1);

    // -------------setup k0 params---------------
    int err = 0;
    DT minErr = 10e-10;
    int size = 9;

    double golden = 100.92198481404495;

    double times[10] = {0,
                        0.24931506849315069,
                        0.49589041095890413,
                        1,
                        1.9506849315068493,
                        2.9506849315068493,
                        4.9506849315068493,
                        9.9068493150684933,
                        29.654794520547945};
    double discs[10] = {1,
                        0.9976122901461052,
                        0.9928609219461707,
                        0.98096919756719625,
                        0.95871411294025854,
                        0.93080892359482892,
                        0.86171326609710219,
                        0.6817644109906752,
                        0.26018386576831076};
    for (int i = 0; i < size; i++) {
        discs[i] = log(discs[i]);
    }

    DT amount = 116.92;
    DT t = 4.9068493150684933;

    for (int i = 0; i < size; i++) {
        times_alloc[i] = times[i];
        discs_alloc[i] = discs[i];
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
    cl::Kernel kernel_BondEngine(program, "BOND_k0", &cl_err);
    logger.logCreateKernel(cl_err);

    cl_mem_ext_ptr_t mext_o[3];
    mext_o[0] = {5, output, kernel_BondEngine()};
    mext_o[1] = {1, times_alloc, kernel_BondEngine()};
    mext_o[2] = {2, discs_alloc, kernel_BondEngine()};

    // create device buffer and map dev buf to host buf
    cl::Buffer output_buf;
    cl::Buffer times_buf, discs_buf;
    output_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(DT) * N,
                            &mext_o[0]);
    times_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(DT) * LEN,
                           &mext_o[1]);
    discs_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(DT) * LEN,
                           &mext_o[2]);

    std::vector<cl::Memory> ob_out;
    ob_out.push_back(output_buf);

    q.finish();
    // launch kernel and calculate kernel execution time
    std::cout << "kernel start------" << std::endl;
    gettimeofday(&start_time, 0);
    for (int i = 0; i < 1; ++i) {
        kernel_BondEngine.setArg(0, size);
        kernel_BondEngine.setArg(1, times_buf);
        kernel_BondEngine.setArg(2, discs_buf);
        kernel_BondEngine.setArg(3, amount);
        kernel_BondEngine.setArg(4, t);
        kernel_BondEngine.setArg(5, output_buf);

        q.enqueueTask(kernel_BondEngine, nullptr, nullptr);
    }

    q.finish();
    gettimeofday(&end_time, 0);
    std::cout << "kernel end------" << std::endl;
    std::cout << "Execution time " << tvdiff(&start_time, &end_time) << "us" << std::endl;
    q.enqueueMigrateMemObjects(ob_out, 1, nullptr, nullptr);
    q.finish();
#else
    BOND_k0(size, times, discs, amount, t, output);
#endif
    DT out = output[0];
    if (std::fabs(out - golden) > minErr) err++;
    std::cout << "NPV= " << out << " ,diff/NPV= " << (out - golden) / golden << std::endl;
    err ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);

    return err;
}
