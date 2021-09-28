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
#include "wcc_kernel.hpp"
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
    std::cout << "\n---------------------WCC Test----------------\n";
    // cmd parser
    ArgParser parser(argc, argv);
    std::string xclbin_path;
#ifndef HLS_TEST
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }
#endif
#ifndef HLS_TEST
    std::string offsetfile;
    std::string columnfile;
    std::string goldenfile;
    if (!parser.getCmdOption("-o", offsetfile)) { // offset
        std::cout << "ERROR: offsetfile is not set!\n";
        return -1;
    }
    if (!parser.getCmdOption("-c", columnfile)) { // column
        std::cout << "ERROR: columnfile is not set!\n";
        return -1;
    }
    if (!parser.getCmdOption("-g", goldenfile)) { // row
        std::cout << "ERROR: goldenfile is not set!\n";
        return -1;
    }
#else
    std::string offsetfile = "./data/test_offset.csr";
    std::string columnfile = "./data/test_column.csr";
    std::string goldenfile = "./data/test_golden.mtx";
#endif

    char line[1024] = {0};
    int index = 0;

    int numVertices;
    int numEdges;

    std::fstream offsetfstream(offsetfile.c_str(), std::ios::in);
    if (!offsetfstream) {
        std::cout << "Error : " << offsetfile << " file doesn't exist !" << std::endl;
        exit(1);
    }

    offsetfstream.getline(line, sizeof(line));
    std::stringstream numOdata(line);
    numOdata >> numVertices;

    ap_uint<32>* offset32 = aligned_alloc<ap_uint<32> >(numVertices + 1);
    while (offsetfstream.getline(line, sizeof(line))) {
        std::stringstream data(line);
        data >> offset32[index];
        index++;
    }

    std::fstream columnfstream(columnfile.c_str(), std::ios::in);
    if (!columnfstream) {
        std::cout << "Error : " << columnfile << " file doesn't exist !" << std::endl;
        exit(1);
    }

    index = 0;

    columnfstream.getline(line, sizeof(line));
    std::stringstream numCdata(line);
    numCdata >> numEdges;

    ap_uint<32>* column32 = aligned_alloc<ap_uint<32> >(numEdges);
    while (columnfstream.getline(line, sizeof(line))) {
        std::stringstream data(line);
        data >> column32[index];
        index++;
    }

    ap_uint<32>* column32G2 = aligned_alloc<ap_uint<32> >(numEdges);
    ap_uint<32>* offset32G2 = aligned_alloc<ap_uint<32> >(numVertices + 1);

    ap_uint<32>* offset32Tmp1 = aligned_alloc<ap_uint<32> >(numVertices + 1);
    ap_uint<32>* offset32Tmp2 = aligned_alloc<ap_uint<32> >(numVertices + 1);

    ap_uint<32>* queue = aligned_alloc<ap_uint<32> >(numVertices);

    ap_uint<32>* result32 = aligned_alloc<ap_uint<32> >(numVertices);

#ifndef HLS_TEST
    // do pre-process on CPU
    struct timeval start_time, end_time;

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
    printf("INFO: Found Device=%s\n", devName.c_str());

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &err);
    logger.logCreateProgram(err);
    cl::Kernel wcc(program, "wcc_kernel");
    logger.logCreateKernel(err);
    std::cout << "INFO: kernel has been created" << std::endl;

    cl_mem_ext_ptr_t mext_o[8];

    mext_o[0] = {2, column32, wcc()};
    mext_o[1] = {3, offset32, wcc()};
    mext_o[2] = {5, column32G2, wcc()};
    mext_o[3] = {6, offset32G2, wcc()};
    mext_o[4] = {7, offset32Tmp1, wcc()};
    mext_o[5] = {8, offset32Tmp2, wcc()};
    mext_o[6] = {10, queue, wcc()};
    mext_o[7] = {12, result32, wcc()};

    // create device buffer and map dev buf to host buf
    cl::Buffer column32G1_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                           sizeof(ap_uint<32>) * numEdges, &mext_o[0]);
    cl::Buffer offset32G1_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                           sizeof(ap_uint<32>) * (numVertices + 1), &mext_o[1]);
    cl::Buffer column32G2_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                           sizeof(ap_uint<32>) * numEdges, &mext_o[2]);
    cl::Buffer offset32G2_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                           sizeof(ap_uint<32>) * (numVertices + 1), &mext_o[3]);
    cl::Buffer offset32Tmp1_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                             sizeof(ap_uint<32>) * (numVertices + 1), &mext_o[4]);
    cl::Buffer offset32Tmp2_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                             sizeof(ap_uint<32>) * (numVertices + 1), &mext_o[5]);

    cl::Buffer queue_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                      sizeof(ap_uint<32>) * numVertices, &mext_o[6]);

    cl::Buffer result_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                       sizeof(ap_uint<32>) * numVertices, &mext_o[7]);

    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(1);
    std::vector<cl::Event> events_read(1);

    std::vector<cl::Memory> ob_in;
    ob_in.push_back(column32G1_buf);
    ob_in.push_back(offset32G1_buf);

    std::vector<cl::Memory> ob_out;
    ob_out.push_back(result_buf);

    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &events_write[0]);

    // launch kernel and calculate kernel execution time
    std::cout << "INFO: kernel start------" << std::endl;
    gettimeofday(&start_time, 0);
    int j = 0;
    wcc.setArg(j++, numEdges);
    wcc.setArg(j++, numVertices);
    wcc.setArg(j++, column32G1_buf);
    wcc.setArg(j++, offset32G1_buf);
    wcc.setArg(j++, column32G2_buf);
    wcc.setArg(j++, column32G2_buf);
    wcc.setArg(j++, offset32G2_buf);
    wcc.setArg(j++, offset32Tmp1_buf);
    wcc.setArg(j++, offset32Tmp2_buf);
    wcc.setArg(j++, queue_buf);
    wcc.setArg(j++, queue_buf);
    wcc.setArg(j++, result_buf);
    wcc.setArg(j++, result_buf);

    // q.enqueueTask(wcc, &events_write, &events_kernel[0]);
    q.enqueueTask(wcc, &events_write, &events_kernel[0]);

    q.enqueueMigrateMemObjects(ob_out, 1, &events_kernel, &events_read[0]);
    q.finish();

    gettimeofday(&end_time, 0);
    std::cout << "INFO: kernel end------" << std::endl;

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

    std::cout << "INFO: Execution time " << tvdiff(&start_time, &end_time) / 1000.0 << "ms" << std::endl;
#else
    wcc_kernel(numEdges, numVertices, (ap_uint<512>*)column32, (ap_uint<512>*)offset32, (ap_uint<512>*)column32G2,
               column32G2, (ap_uint<512>*)offset32G2, (ap_uint<512>*)offset32Tmp1, (ap_uint<512>*)offset32Tmp2,
               (ap_uint<512>*)queue, queue, (ap_uint<512>*)result32, result32);
#endif

    std::cout << "============================================================" << std::endl;

    std::vector<int> gold_result(numVertices, -1);

    std::fstream goldenfstream(goldenfile.c_str(), std::ios::in);
    if (!goldenfstream) {
        std::cout << "Error: " << goldenfile << " file doesn't exist !" << std::endl;
        exit(1);
    }
    index = 0;
    while (goldenfstream.getline(line, sizeof(line))) {
        std::stringstream data(line);
        std::string tmp[2];
        int tmpi[2];
        data >> tmp[0];
        data >> tmp[1];

        tmpi[0] = std::stoi(tmp[0]);

        if (index > 0) {
            tmpi[1] = std::stoi(tmp[1]);
            gold_result[tmpi[0] - 1] = tmpi[1];
        } else
            std::cout << "INFO: The number of components:" << tmpi[0] << std::endl;

        index++;
    }

    if (index - 1 != numVertices) {
        std::cout << "Warning: Some nodes are missing in the golden file, validation will skip them." << std::endl;
    }

    int errs = 0;
    for (int i = 0; i < numVertices; i++) {
        if (result32[i].to_int() != gold_result[i] && gold_result[i] != -1) {
            std::cout << "Mismatch-" << i + 1 << ":\tsw: " << gold_result[i] << " -> "
                      << "hw: " << result32[i] << std::endl;
            errs++;
        }
    }

    errs ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
         : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return errs;
}
