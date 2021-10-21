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
#include "bfs_kernel.hpp"
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
    std::cout << "\n---------------------BFS Traversal Test----------------\n";
    // cmd parser
    ArgParser parser(argc, argv);
    std::string xclbin_path;
#ifndef HLS_TEST
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }
#endif

    int srcNodeID = 0;
#ifndef HLS_TEST
    std::string offsetfile;
    std::string columnfile;
    std::string goldenfile;
    std::string nodeIDStr;
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
    if (!parser.getCmdOption("-i", nodeIDStr)) { // source node
        std::cout << "ERROR: source node is not set!\n";
        return -1;
    } else {
        srcNodeID = std::stoi(nodeIDStr);
        std::cout << "Source node ID:" << srcNodeID << std::endl;
    }
#else
    std::string offsetfile = "../data/test_offset.csr";
    std::string columnfile = "../data/test_column.csr";
    std::string goldenfile = "../data/test_golden.mtx";
    srcNodeID = 0;
#endif

    char line[1024] = {0};
    int index = 0;

    int numVertices;
    int maxVertexId;
    int numEdges;

    std::fstream offsetfstream(offsetfile.c_str(), std::ios::in);
    if (!offsetfstream) {
        std::cout << "Error : " << offsetfile << " file doesn't exist !" << std::endl;
        exit(1);
    }

    offsetfstream.getline(line, sizeof(line));
    std::stringstream numOdata(line);
    numOdata >> numVertices;
    numOdata >> maxVertexId;

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

    ap_uint<32>* queue = aligned_alloc<ap_uint<32> >(numVertices);

    ap_uint<32>* result32_dt = aligned_alloc<ap_uint<32> >(((numVertices + 15) / 16) * 16);
    ap_uint<32>* result32_ft = aligned_alloc<ap_uint<32> >(((numVertices + 15) / 16) * 16);
    ap_uint<32>* result32_pt = aligned_alloc<ap_uint<32> >(((numVertices + 15) / 16) * 16);
    ap_uint<32>* result32_lt = aligned_alloc<ap_uint<32> >(((numVertices + 15) / 16) * 16);

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
    printf("Found Device=%s\n", devName.c_str());

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &err);
    logger.logCreateProgram(err);

    cl::Kernel bfs(program, "bfs_kernel", &err);
    std::cout << "kernel has been created" << std::endl;

    cl_mem_ext_ptr_t mext_o[7];
    mext_o[0] = {2, column32, bfs()};
    mext_o[1] = {3, offset32, bfs()};
    mext_o[2] = {4, queue, bfs()};
    mext_o[3] = {6, result32_dt, bfs()};
    mext_o[4] = {8, result32_ft, bfs()};
    mext_o[5] = {9, result32_pt, bfs()};
    mext_o[6] = {10, result32_lt, bfs()};

    // create device buffer and map dev buf to host buf
    cl::Buffer column_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                       sizeof(ap_uint<32>) * numEdges, &mext_o[0]);
    cl::Buffer offset_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                       sizeof(ap_uint<32>) * (numVertices + 1), &mext_o[1]);

    cl::Buffer queue_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                      sizeof(ap_uint<32>) * numVertices, &mext_o[2]);

    cl::Buffer resultDT_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                         sizeof(ap_uint<32>) * ((numVertices + 15) / 16) * 16, &mext_o[3]);
    cl::Buffer resultFT_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                         sizeof(ap_uint<32>) * ((numVertices + 15) / 16) * 16, &mext_o[4]);
    cl::Buffer resultPT_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                         sizeof(ap_uint<32>) * ((numVertices + 15) / 16) * 16, &mext_o[5]);
    cl::Buffer resultLT_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                         sizeof(ap_uint<32>) * ((numVertices + 16) / 16) * 16, &mext_o[6]);

    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(1);
    std::vector<cl::Event> events_read(1);

    std::vector<cl::Memory> ob_in;
    ob_in.push_back(column_buf);
    ob_in.push_back(offset_buf);

    std::vector<cl::Memory> ob_out;
    ob_out.push_back(resultDT_buf);
    ob_out.push_back(resultFT_buf);
    ob_out.push_back(resultPT_buf);
    ob_out.push_back(resultLT_buf);

    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &events_write[0]);

    // launch kernel and calculate kernel execution time
    std::cout << "kernel start------" << std::endl;
    std::cout << "Input: numVertex=" << numVertices << ", numEdges=" << numEdges << std::endl;
    gettimeofday(&start_time, 0);
    int j = 0;
    bfs.setArg(j++, srcNodeID);
    bfs.setArg(j++, numVertices);
    bfs.setArg(j++, column_buf);
    bfs.setArg(j++, offset_buf);
    bfs.setArg(j++, queue_buf);
    bfs.setArg(j++, queue_buf);
    bfs.setArg(j++, resultDT_buf);
    bfs.setArg(j++, resultDT_buf);
    bfs.setArg(j++, resultFT_buf);
    bfs.setArg(j++, resultPT_buf);
    bfs.setArg(j++, resultLT_buf);

    // q.enqueueTask(bfs, &events_write, &events_kernel[0]);
    q.enqueueTask(bfs, &events_write, &events_kernel[0]);

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

#else
    bfs_kernel(srcNodeID, numVertices, (ap_uint<512>*)column32, (ap_uint<512>*)offset32, (ap_uint<512>*)queue, queue,
               (ap_uint<512>*)result32_dt, result32_dt, result32_ft, result32_pt, result32_lt);
#endif

    std::cout << "============================================================" << std::endl;
    int errs = 0;

    std::vector<std::vector<int> > gold_result(numVertices, std::vector<int>(4, -1));
    gold_result.resize(numVertices);

    std::fstream goldenfstream(goldenfile.c_str(), std::ios::in);
    if (!goldenfstream) {
        std::cout << "Error : " << goldenfile << " file doesn't exist !" << std::endl;
        exit(1);
    }
    index = 0;
    while (goldenfstream.getline(line, sizeof(line))) {
        std::stringstream data(line);
        for (int i = 0; i < 4; i++) {
            std::string tmp;
            data >> tmp;

            int tmpi = std::stoi(tmp);
            gold_result[index][i] = tmpi;
        }

        index++;
    }

    if (index != numVertices) {
        std::cout << "Error : Mismatch has been found in the golden file !" << std::endl;
        return -1;
    }

    for (int i = 0; i < numVertices; i++) {
        if ((result32_dt[i] != (ap_uint<32>)-1 && result32_dt[i].to_int() != gold_result[i][0]) ||
            result32_ft[i].to_int() != gold_result[i][1] || result32_pt[i].to_int() != gold_result[i][2] ||
            result32_lt[i].to_int() != gold_result[i][3]) {
            std::cout << "Mismatch-" << i << ":\tsw: " << gold_result[i][0] << " " << gold_result[i][1] << " "
                      << gold_result[i][2] << " " << gold_result[i][3] << " <-> "
                      << "hw: " << result32_dt[i] << " " << result32_ft[i] << " " << result32_pt[i] << " "
                      << result32_lt[i];

            std::cout << "\t\t\t***\t";
            if (result32_dt[i] != (ap_uint<32>)-1 && result32_dt[i].to_int() != gold_result[i][0]) std::cout << "D";
            if (result32_ft[i].to_int() != gold_result[i][1]) std::cout << "F";
            if (result32_pt[i].to_int() != gold_result[i][2]) std::cout << "P";
            if (result32_lt[i].to_int() != gold_result[i][3]) std::cout << "L";

            std::cout << std::endl;
            errs++;
        }
    }

    errs ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
         : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);

    return errs;
}
