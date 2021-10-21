/*
 * Copyright 2020 Xilinx, Inc.
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
#include "ap_int.h"
#include "triangle_count_kernel.hpp"
#include "utils.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <vector>
#include "xf_utils_sw/logger.hpp"

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

#define P2_32 4294967296
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
    std::cout << "\n---------------------Triangle Count-----------------\n";
    // cmd parser
    ArgParser parser(argc, argv);
    std::string xclbin_path;
#ifndef HLS_TEST
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }
#endif
    std::string filename;
    std::string filename1;
    if (!parser.getCmdOption("-o", filename)) { // offset
        std::cout << "ERROR: offset file path is not set!\n";
#ifdef HLS_TEST
        filename = "data/csr_offsets.txt";
#else
        return -1;
#endif
    }
    if (!parser.getCmdOption("-i", filename1)) { // row
        std::cout << "ERROR: row file path is not set!\n";
#ifdef HLS_TEST
        filename1 = "data/csr_columns.txt";
#else
        return -1;
#endif
    }
    // Allocate Memory in Host Memory
    int vertexNum;
    int edgeNum;
    int TC_golden = 11;
    uint64_t TC[N] = {0};

    DT* offsets = aligned_alloc<DT>(V);
    DT* rows = aligned_alloc<DT>(E);
    // -------------setup k0 params---------------
    int nerr = 0;

    std::fstream fin(filename.c_str(), std::ios::in);
    if (!fin) {
        std::cout << "Error : " << filename << " file doesn't exist !" << std::endl;
        exit(1);
    }
    DT tmp;
    char line[1024] = {0};
    int index = 0;
    int max_diff = 0;
    int id = 0;
    while (fin.getline(line, sizeof(line))) {
        std::stringstream data(line);
        data >> tmp;
        if (index == 0)
            vertexNum = tmp;
        else {
            offsets[index - 1] = tmp;
            if (index > 1 && (max_diff < offsets[index - 1] - offsets[index - 2])) {
                max_diff = offsets[index - 1] - offsets[index - 2];
                id = index - 1;
            }
        }
        index++;
    }
    std::cout << "vertex " << id << ",max_diff=" << max_diff << std::endl;
    if (max_diff > ML) {
        std::cout << "[ERROR] more than maximum setting storage space, if must, increase the parameter ML!\n";
        return -1;
    }

    std::fstream fin1(filename1.c_str(), std::ios::in);
    if (!fin1) {
        std::cout << "Error : " << filename1 << " file doesn't exist !" << std::endl;
        exit(1);
    }

    index = 0;
    while (fin1.getline(line, sizeof(line))) {
        std::stringstream data(line);
        data >> tmp;
        if (index == 0) {
            edgeNum = tmp;
        } else {
            rows[index - 1] = tmp;
        }
        index++;
    }
#ifndef HLS_TEST
    // do pre-process on CPU
    struct timeval start_time, end_time, test_time;
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);

    // platform related operations
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    cl_int err;
    // Creating Context and Command Queue for selected Device
    cl::Context context(device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("Found Device=%s\n", devName.c_str());

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &err);
    cl::Kernel TCkernel(program, "TC_kernel", &err);
    std::cout << "kernel has been created" << std::endl;

    cl_mem_ext_ptr_t mext_o[7];

    mext_o[0] = {2, offsets, TCkernel()};
    mext_o[1] = {3, rows, TCkernel()};
    mext_o[5] = {4, offsets, TCkernel()};
    mext_o[6] = {5, rows, TCkernel()};
    mext_o[2] = {6, NULL, TCkernel()};
    mext_o[3] = {7, rows, TCkernel()};
    mext_o[4] = {8, TC, TCkernel()};

    // create device buffer and map dev buf to host buf
    cl::Buffer offset1d_buf, row1d_buf, offset1_buf, row1_buf, offset2_buf, row2_buf, TC_buf;
    offset1_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(DT) * V,
                             &mext_o[0]);
    row1_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(DT) * E,
                          &mext_o[1]);
    offset1d_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(DT) * V,
                              &mext_o[5]);
    row1d_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(DT) * E,
                           &mext_o[6]);
    offset2_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, sizeof(DT) * V * 2, &mext_o[2]);
    row2_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(DT) * E,
                          &mext_o[3]);
    TC_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(uint64_t) * N,
                        &mext_o[4]);

    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_buf;
    std::vector<cl::Memory> ob_out;
    std::vector<cl::Event> events_write(2);
    std::vector<cl::Event> events_buf(1);
    std::vector<cl::Event> events_kernel(1);
    std::vector<cl::Event> events_read(1);

    ob_in.push_back(offset1_buf);
    ob_in.push_back(row1_buf);
    ob_in.push_back(offset1d_buf);
    ob_in.push_back(row1d_buf);
    ob_buf.push_back(offset2_buf);
    ob_in.push_back(row2_buf);
    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &events_write[0]);
    q.enqueueMigrateMemObjects(ob_buf, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, &events_write[1]);

    ob_out.push_back(TC_buf);

    // launch kernel and calculate kernel execution time
    std::cout << "kernel start------" << std::endl;
    gettimeofday(&start_time, 0);
    int j = 0;
    TCkernel.setArg(j++, vertexNum);
    TCkernel.setArg(j++, edgeNum);
    TCkernel.setArg(j++, offset1_buf);
    TCkernel.setArg(j++, row1_buf);
    TCkernel.setArg(j++, offset1d_buf);
    TCkernel.setArg(j++, row1d_buf);
    TCkernel.setArg(j++, offset2_buf);
    TCkernel.setArg(j++, row2_buf);
    TCkernel.setArg(j++, TC_buf);

    q.enqueueTask(TCkernel, &events_write, &events_kernel[0]);

    q.enqueueMigrateMemObjects(ob_out, 1, &events_kernel, &events_read[0]);
    q.finish();

    gettimeofday(&end_time, 0);
    std::cout << "kernel end------" << std::endl;
    std::cout << "Execution time " << tvdiff(&start_time, &end_time) / 1000.0 << "ms" << std::endl;

    cl_ulong ts, te;
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &ts);
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &te);
    float elapsed = ((float)te - (float)ts) / 1000000.0;
    logger.info(xf::common::utils_sw::Logger::Message::TIME_H2D_MS, elapsed);

    events_kernel[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &ts);
    events_kernel[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &te);
    elapsed = ((float)te - (float)ts) / 1000000.0;
    logger.info(xf::common::utils_sw::Logger::Message::TIME_KERNEL_MS, elapsed);

    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &ts);
    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &te);
    elapsed = ((float)te - (float)ts) / 1000000.0;
    logger.info(xf::common::utils_sw::Logger::Message::TIME_D2H_MS, elapsed);

    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &ts);
    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &te);
    elapsed = ((float)te - (float)ts) / 1000000.0;
    std::cout << "Info: Total Execution time " << elapsed << " ms" << std::endl;
#else
    DT* offsets2 = aligned_alloc<DT>(V * 2);
    TC_kernel(vertexNum, edgeNum, (uint512*)offsets, (uint512*)rows, (uint512*)offsets, (uint512*)rows,
              (uint512*)offsets2, (uint512*)rows, TC);
#endif
    uint64_t out = TC[0];
    if (TC_golden != out) {
        nerr++;
        logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL);
    } else {
        logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    }

    return nerr;
}
