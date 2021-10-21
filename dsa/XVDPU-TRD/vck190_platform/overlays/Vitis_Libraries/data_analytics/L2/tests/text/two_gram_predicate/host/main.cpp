/*
 * Copyright 2021 Xilinx, Inc.
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
#include "xcl2.hpp"
#include "utils.hpp"
#include <iostream>
#include "index.hpp"
#include "predicate_kernel.hpp"
#include "xf_utils_sw/logger.hpp"

int main(int argc, const char* argv[]) {
    std::cout << "\n---------------------2-Gram Predicate Flow-----------------\n";
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    // cmd parser
    ArgParser parser(argc, argv);

    std::string xclbin_path;
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return -1;
    }

    std::string inFile;
    if (!parser.getCmdOption("-in", inFile)) {
        std::cout << "ERROR: input file path is not set!\n";
        return -1;
    }

    std::string goldenFile;
    if (!parser.getCmdOption("-golden", goldenFile)) {
        std::cout << "ERROR: golden file path is not set!\n";
        return -1;
    }

    int nerr = 0;
    ConfigParam config;
    const int BS = 1024 * 1024 * 256; // Buffer Size
    const int RN = 1024 * 1024 * 64;  // Record Number
    const int TFLEN = 1024 * 1024 * 32;
    uint8_t* fields = aligned_alloc<uint8_t>(BS);
    uint32_t* offsets = aligned_alloc<uint32_t>(RN);
    double* idfValue = aligned_alloc<double>(4096);
    uint64_t* tfAddr = aligned_alloc<uint64_t>(4096);
    uint64_t* tfValue = aligned_alloc<uint64_t>(TFLEN);
    uint32_t* indexId = aligned_alloc<uint32_t>(RN);

    std::vector<std::string> vec_fields;
    readStringField(inFile, config.docSize, config.fldSize, fields, offsets, vec_fields);
    twoGramIndex(vec_fields, idfValue, tfAddr, tfValue);

    // do pre-process on CPU
    // platform related operations
    struct timeval tk1, tk2;
    gettimeofday(&tk1, 0);
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    cl_int cl_err;
    // Creating Context and Command Queue for selected Device
    cl::Context context(device, NULL, NULL, NULL, &cl_err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &cl_err);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("Found Device=%s\n", devName.c_str());

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &cl_err);
    cl::Kernel PKernel(program, "TGP_Kernel", &cl_err);
    std::cout << "kernel has been created" << std::endl;

    cl_mem_ext_ptr_t mext_o[7];
    mext_o[0] = {1, fields, PKernel()};
    mext_o[1] = {2, offsets, PKernel()};
    mext_o[2] = {3, idfValue, PKernel()};
    mext_o[3] = {4, tfAddr, PKernel()};
    mext_o[4] = {5, tfValue, PKernel()};
    mext_o[5] = {9, indexId, PKernel()};

    // create device buffer and map dev buf to host buf
    std::cout << "create device buffer\n";
    cl::Buffer buff[7];

    // input buffer
    buff[0] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(uint8_t) * BS,
                         &mext_o[0]);
    buff[1] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                         sizeof(uint32_t) * RN, &mext_o[1]);
    buff[2] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                         sizeof(double) * 4096, &mext_o[2]);
    buff[3] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                         sizeof(uint64_t) * 4096, &mext_o[3]);
    buff[4] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                         sizeof(uint64_t) * TFLEN, &mext_o[4]);
    // output buffer
    buff[5] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                         sizeof(uint32_t) * RN, &mext_o[5]);

    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;
    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(1);
    std::vector<cl::Event> events_read(1);

    // push buffer
    for (int i = 0; i < 5; i++) {
        ob_in.push_back(buff[i]);
    }
    ob_out.push_back(buff[5]);

    // launch kernel and calculate kernel execution time
    int j = 0;
    PKernel.setArg(j++, config);
    for (int i = 0; i < 5; i++) {
        PKernel.setArg(j++, buff[i]);
    }
    PKernel.setArg(j++, buff[4]);
    PKernel.setArg(j++, buff[4]);
    PKernel.setArg(j++, buff[4]);
    PKernel.setArg(j++, buff[5]);

    std::cout << "kernel start------" << std::endl;
    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &events_write[0]);
    q.enqueueTask(PKernel, &events_write, &events_kernel[0]);
    q.enqueueMigrateMemObjects(ob_out, 1, &events_kernel, &events_read[0]);
    q.finish();
    std::cout << "kernel end------" << std::endl;

    unsigned long time1, time2, total_time;
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    std::cout << "Write DDR Execution time " << (time2 - time1) / 1000000.0 << " ms" << std::endl;
    total_time = time2 - time1;
    events_kernel[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    events_kernel[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    std::cout << "Kernel Execution time " << (time2 - time1) / 1000000.0 << " ms" << std::endl;
    total_time += time2 - time1;
    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    std::cout << "Read DDR Execution time " << (time2 - time1) / 1000000.0 << " ms" << std::endl;
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    total_time = time2 - time1;
    std::cout << "FPGA Execution time " << total_time / 1000000.0 << " ms" << std::endl;
    gettimeofday(&tk2, 0);
    std::cout << "Kernel Execution time " << tvdiff(&tk1, &tk2) / 1000.0 << "ms" << std::endl;

    free(fields);
    free(offsets);
    free(idfValue);
    free(tfAddr);
    free(tfValue);

    nerr = checkResult(goldenFile, indexId);
    nerr ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
         : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return nerr;
}
