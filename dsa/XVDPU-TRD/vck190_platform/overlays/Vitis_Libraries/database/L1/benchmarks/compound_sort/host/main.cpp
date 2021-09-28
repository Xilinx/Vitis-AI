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
#include <vector>
#include <cstring>
#include <ap_int.h>
#include <xcl2.hpp>
#include "utils.hpp"
#include "kernel_sort.hpp"

#include "xf_utils_sw/logger.hpp"

int main(int argc, const char* argv[]) {
    std::cout << "\n-----------Sort Design---------------\n";

    using namespace xf::common::utils_sw;
    Logger logger(std::cout, std::cerr);

    int err = 0;
    int keyLength;

    std::string mode = "hw";
    ArgParser parser(argc, argv);
    std::string xclbin_path;
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }

    std::string sort_length;
    if (!parser.getCmdOption("-sl", sort_length)) {
        keyLength = LEN;
    } else {
        keyLength = std::stoi(sort_length);
    }
    // keyLength = 893;
    std::cout << "key length is " << keyLength << std::endl;

    if (std::getenv("XCL_EMULATION_MODE") != nullptr) {
        mode = std::getenv("XCL_EMULATION_MODE");
    }
    std::cout << "[INFO]Running in " << mode << " mode" << std::endl;

    if (keyLength > LEN) {
        std::cout << "[ERROR] Keys length is more than the max supported length!!!\n";
        return 1;
    }

    std::vector<KEY_TYPE> v(keyLength);
    for (unsigned i = 0; i < v.size(); i++) {
        v[i] = rand();
    }

    KEY_TYPE* inKey_alloc = aligned_alloc<KEY_TYPE>(LEN);
    KEY_TYPE* outKey_alloc = aligned_alloc<KEY_TYPE>(LEN);
    for (int i = 0; i < keyLength; i++) {
        inKey_alloc[i] = v[i];
    }

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    logger.logCreateCommandQueue(err);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("Found Device=%s\n", devName.c_str());
    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &err);
    logger.logCreateProgram(err);
    cl::Kernel kernel_SortKernel(program, "SortKernel", &err);
    logger.logCreateKernel(err);
    std::cout << "kernel has been created" << std::endl;

    cl_mem_ext_ptr_t mext_o[3];
    mext_o[0] = {2, inKey_alloc, kernel_SortKernel()};  // arg 2 of kernel
    mext_o[1] = {3, outKey_alloc, kernel_SortKernel()}; // arg 3 of kernel
    cl::Buffer inKey_buf, outKey_buf, outIndex_buf;
    inKey_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                           sizeof(KEY_TYPE) * LEN, &mext_o[0]);
    outKey_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            sizeof(KEY_TYPE) * LEN, &mext_o[1]);
    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;
    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(1);
    std::vector<cl::Event> events_read(1);

    ob_in.push_back(inKey_buf);
    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &events_write[0]);
    ob_out.push_back(outKey_buf);
    // q.finish();
    std::cout << "kernel start------" << std::endl;
    int j = 0;
    kernel_SortKernel.setArg(j++, 1);
    kernel_SortKernel.setArg(j++, keyLength);
    kernel_SortKernel.setArg(j++, inKey_buf);
    kernel_SortKernel.setArg(j++, outKey_buf);
    q.enqueueTask(kernel_SortKernel, &events_write, &events_kernel[0]);
    // q.finish();
    q.enqueueMigrateMemObjects(ob_out, 1, &events_kernel, &events_read[0]);
    q.finish();
    std::sort(v.begin(), v.end());
    for (int i = 0; i < keyLength; i++) {
        bool cmp_key = (outKey_alloc[i] == v[i]) ? 1 : 0;
        if (!cmp_key) {
            std::cout << "v[" << i << "]=" << v[i] << ",key[" << i << "]=" << outKey_alloc[i] << std::endl;
            std::cout << "\nthe sort key is incorrect" << std::endl;
            err++;
        }
    }

    unsigned long time1, time2, total_time;
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    std::cout << "Write DDR Execution time " << (time2 - time1) / 1000.0 << "us" << std::endl;
    total_time = time2 - time1;
    events_kernel[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    events_kernel[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    std::cout << "Kernel Execution time " << (time2 - time1) / 1000.0 << "us" << std::endl;
    total_time += time2 - time1;
    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    std::cout << "Read DDR Execution time " << (time2 - time1) / 1000.0 << "us" << std::endl;
    total_time += time2 - time1;
    std::cout << "Total Execution time " << total_time / 1000.0 << "us" << std::endl;

    err ? logger.error(Logger::Message::TEST_FAIL) : logger.info(Logger::Message::TEST_PASS);
    return err;
}
