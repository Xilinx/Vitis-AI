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
#include "utils.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <vector>
#include "label_propagation_top.hpp"
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
    std::cout << "\n---------------------Label Propagation-----------------\n";
    // cmd parser
    ArgParser parser(argc, argv);
    std::string xclbin_path;
#ifndef HLS_TEST
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }
#endif
    int iter = 1;
    std::string iteration;
    std::string filename_offsets;
    std::string filename_index;
    std::string filename2_offsets;
    std::string filename2_index;
    std::string filename_label;
    if (!parser.getCmdOption("-iter", iteration)) { // iteration
        iter = 1;
    } else {
        iter = std::stoi(iteration);
    }

    if (!parser.getCmdOption("-o", filename_offsets)) { // offset
        std::cout << "ERROR: [-o] file path is not set!\n";
        return -1;
    }
    if (!parser.getCmdOption("-i", filename_index)) { // index
        std::cout << "ERROR: [-i] file path is not set!\n";
        return -1;
    }

    if (!parser.getCmdOption("-label", filename_label)) {
        std::cout << "ERROR: [-label] file path is not set!\n";
        return -1;
    }

    // Allocate Memory in Host Memory
    int vertexNum;
    int edgeNum;

    DT* offsetsCSR = aligned_alloc<DT>(V * 16);
    DT* columnsCSR = aligned_alloc<DT>(E * 16);
    DT* offsetsCSC = aligned_alloc<DT>(V * 16);
    DT* rowsCSC = aligned_alloc<DT>(E * 16);

    DT* labelPing = aligned_alloc<DT>(V * 16);
    DT* labelPong = aligned_alloc<DT>(V * 16);
    DT* bufPing = aligned_alloc<DT>(V * 16);
    DT* bufPong = aligned_alloc<DT>(V * 16);
    DT* labelGolden = aligned_alloc<DT>(V * 16);

    DT* offsets = offsetsCSR;
    DT* rows = columnsCSR;
    // -------------setup k0 params---------------
    int nerr = 0;

    std::fstream fin(filename_offsets.c_str(), std::ios::in);
    if (!fin) {
        std::cout << "Error : " << filename_offsets << " file doesn't exist !" << std::endl;
        exit(1);
    }
    DT tmp;
    char line[1024] = {0};
    int index = 0;
    while (fin.getline(line, sizeof(line))) {
        std::stringstream data(line);
        data >> tmp;
        if (index == 0)
            vertexNum = tmp;
        else {
            offsets[index - 1] = tmp;
        }
        index++;
    }

    std::fstream fin1(filename_index.c_str(), std::ios::in);
    if (!fin1) {
        std::cout << "Error : " << filename_index << " file doesn't exist !" << std::endl;
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

    std::fstream fin_label(filename_label.c_str(), std::ios::in);
    if (!fin_label) {
        std::cout << "Error : " << filename_label << " file doesn't exist !" << std::endl;
        exit(1);
    }

    index = 0;
    while (fin_label.getline(line, sizeof(line))) {
        std::stringstream data(line);
        data >> tmp;
        data >> tmp;
        if (index > 0) {
            labelGolden[index - 1] = tmp;
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
    logger.logCreateCommandQueue(err);

    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("Found Device=%s\n", devName.c_str());

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &err);
    logger.logCreateProgram(err);

    cl::Kernel LPKernel(program, "LPKernel", &err);
    logger.logCreateKernel(err);
    std::cout << "kernel has been created" << std::endl;

    cl_mem_ext_ptr_t mext_o[8];

    mext_o[0] = {3, offsetsCSR, LPKernel()};
    mext_o[1] = {4, columnsCSR, LPKernel()};
    mext_o[2] = {5, offsetsCSC, LPKernel()};
    mext_o[3] = {6, rowsCSC, LPKernel()};
    mext_o[4] = {8, bufPing, LPKernel()};
    mext_o[5] = {9, bufPong, LPKernel()};
    mext_o[6] = {10, labelPing, LPKernel()};
    mext_o[7] = {11, labelPong, LPKernel()};

    // create device buffer and map dev buf to host buf
    cl::Buffer offsetsCSR_buf, columnsCSR_buf, offsetsCSC_buf, rowsCSC_buf, bufPing_buf, bufPong_buf, labelPing_buf,
        labelPong_buf;
    offsetsCSR_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                sizeof(DT) * V, &mext_o[0]);
    columnsCSR_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                sizeof(DT) * E, &mext_o[1]);
    offsetsCSC_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                sizeof(DT) * V, &mext_o[2]);
    rowsCSC_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(DT) * E,
                             &mext_o[3]);
    bufPing_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(DT) * V,
                             &mext_o[4]);
    bufPong_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(DT) * V,
                             &mext_o[5]);
    labelPing_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(DT) * V,
                               &mext_o[6]);
    labelPong_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(DT) * V,
                               &mext_o[7]);

    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_buf;
    std::vector<cl::Memory> ob_out;
    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(1);
    std::vector<cl::Event> events_read(1);

    ob_in.push_back(offsetsCSR_buf);
    ob_in.push_back(columnsCSR_buf);
    ob_out.push_back(offsetsCSC_buf);
    ob_out.push_back(rowsCSC_buf);
    ob_out.push_back(bufPing_buf);
    ob_out.push_back(bufPong_buf);
    ob_out.push_back(labelPing_buf);
    ob_out.push_back(labelPong_buf);
    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &events_write[0]);
    // q.enqueueMigrateMemObjects(ob_buf, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, &events_write[1]);

    // launch kernel and calculate kernel execution time
    std::cout << "kernel start------" << std::endl;
    std::cout << "vertexNum=" << vertexNum << "   edgeNum=" << edgeNum << std::endl;
    gettimeofday(&start_time, 0);
    int j = 0;
    LPKernel.setArg(j++, vertexNum);
    LPKernel.setArg(j++, edgeNum);
    LPKernel.setArg(j++, iter);
    LPKernel.setArg(j++, offsetsCSR_buf);
    LPKernel.setArg(j++, columnsCSR_buf);
    LPKernel.setArg(j++, offsetsCSC_buf);
    LPKernel.setArg(j++, rowsCSC_buf);
    LPKernel.setArg(j++, rowsCSC_buf);
    LPKernel.setArg(j++, bufPing_buf);
    LPKernel.setArg(j++, bufPong_buf);
    LPKernel.setArg(j++, labelPing_buf);
    LPKernel.setArg(j++, labelPong_buf);

    q.enqueueTask(LPKernel, &events_write, &events_kernel[0]);

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
    LPKernel(vertexNum, edgeNum, iter, (uint512*)offsetsCSR, (uint512*)columnsCSR, (uint512*)offsetsCSC,
             (uint512*)rowsCSC, (ap_uint<32>*)rowsCSC, (uint512*)bufPing, (uint512*)bufPong, (uint512*)labelPing,
             (uint512*)labelPong);
#endif
    DT* label;
    if (iter % 2)
        label = labelPong;
    else
        label = labelPing;

    for (int i = 0; i < vertexNum; i++)
        if (label[i] != labelGolden[i]) {
            std::cout << "degree[" << i << "]=" << offsetsCSR[i + 1] - offsetsCSR[i] + offsetsCSC[i + 1] - offsetsCSC[i]
                      << ",label[" << i << "]=" << label[i] << ", labelPing=" << labelPing[i]
                      << ", labelPong=" << labelPong[i] << ",golden[" << i << "]=" << labelGolden[i] << std::endl;
            nerr++;
        }

    if (!nerr) std::cout << "INFO: case pass!\n";
    nerr ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
         : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return nerr;
}
