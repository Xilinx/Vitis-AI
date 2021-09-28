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
#else
#include "twoHop_kernel.hpp"
#endif
#include "xf_utils_sw/logger.hpp"
#include "ap_int.h"
#include "utils.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <vector>
#include <stdlib.h>

#include <unordered_map>

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
    std::cout << "\n---------------------Two Hop-------------------\n";
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    cl_int fail;

    // cmd parser
    ArgParser parser(argc, argv);
    std::string xclbin_path;
#ifndef HLS_TEST
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }
#endif

    std::string offsetfile;
    std::string indexfile;
    std::string pairfile;
    std::string goldenfile;

    if (!parser.getCmdOption("--offset", offsetfile)) {
        std::cout << "ERROR: offset file path is not set!\n";
        return -1;
    }

    if (!parser.getCmdOption("--index", indexfile)) {
        std::cout << "ERROR: index file path is not set!\n";
        return -1;
    }

    if (!parser.getCmdOption("--pair", pairfile)) {
        std::cout << "ERROR: pair file path is not set!\n";
        return -1;
    }

    if (!parser.getCmdOption("--golden", goldenfile)) {
        std::cout << "ERROR: golden file path is not set!\n";
        return -1;
    }

    // -------------setup k0 params---------------
    int err = 0;

    char line[1024] = {0};
    int fileIdx = 0;

    int numVertices;
    int numEdges;
    int numPairs;

    std::fstream offsetfstream(offsetfile.c_str(), std::ios::in);
    if (!offsetfstream) {
        std::cout << "Error : " << offsetfile << " file doesn't exist !" << std::endl;
        exit(1);
    }

    offsetfstream.getline(line, sizeof(line));
    std::stringstream numOdata(line);
    numOdata >> numVertices;
    numOdata >> numVertices;

    unsigned* offset32 = aligned_alloc<unsigned>(numVertices + 1);
    while (offsetfstream.getline(line, sizeof(line))) {
        std::stringstream data(line);
        data >> offset32[fileIdx];
        fileIdx++;
    }
    offsetfstream.close();

    fileIdx = 0;
    std::fstream indexfstream(indexfile.c_str(), std::ios::in);
    if (!indexfstream) {
        std::cout << "Error : " << indexfile << " file doesn't exist !" << std::endl;
        exit(1);
    }

    indexfstream.getline(line, sizeof(line));
    std::stringstream numCdata(line);
    numCdata >> numEdges;

    unsigned* index32 = aligned_alloc<unsigned>(numEdges);
    while (indexfstream.getline(line, sizeof(line))) {
        std::stringstream data(line);
        data >> index32[fileIdx];
        float tmp;
        data >> tmp;
        fileIdx++;
    }
    indexfstream.close();

    fileIdx = 0;
    std::fstream pairfstream(pairfile.c_str(), std::ios::in);
    if (!pairfstream) {
        std::cout << "Error : " << pairfile << " file doesn't exist !" << std::endl;
        exit(1);
    }

    pairfstream.getline(line, sizeof(line));
    std::stringstream numPdata(line);
    numPdata >> numPairs;

    ap_uint<64>* pair = aligned_alloc<ap_uint<64> >(numPairs);

    while (pairfstream.getline(line, sizeof(line))) {
        std::stringstream data(line);
        ap_uint<64> tmp64;
        unsigned src;
        unsigned des;
        data >> src;
        tmp64.range(63, 32) = src - 1;
        data >> des;
        tmp64.range(31, 0) = des - 1;
        pair[fileIdx] = tmp64;
        fileIdx++;
    }
    pairfstream.close();

    unsigned* cnt_res = aligned_alloc<unsigned>(numPairs);

    // do pre-process on CPU
    struct timeval start_time, end_time;
    // platform related operations
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Creating Context and Command Queue for selected Device
    cl::Context context(device, NULL, NULL, NULL, &fail);
    logger.logCreateContext(fail);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &fail);
    logger.logCreateCommandQueue(fail);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("Found Device=%s\n", devName.c_str());

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    devices[0] = device;
    cl::Program program(context, devices, xclBins, NULL, &fail);
    logger.logCreateProgram(fail);
    cl::Kernel twoHop;
    twoHop = cl::Kernel(program, "twoHop_kernel", &fail);
    logger.logCreateKernel(fail);

    std::cout << "kernel has been created" << std::endl;

    std::vector<cl_mem_ext_ptr_t> mext_o(6);

    mext_o[0] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, pair, 0};
    mext_o[1] = {(unsigned int)(1) | XCL_MEM_TOPOLOGY, cnt_res, 0};
    mext_o[2] = {(unsigned int)(2) | XCL_MEM_TOPOLOGY, offset32, 0};
    mext_o[3] = {(unsigned int)(3) | XCL_MEM_TOPOLOGY, index32, 0};
    mext_o[4] = {(unsigned int)(4) | XCL_MEM_TOPOLOGY, offset32, 0};
    mext_o[5] = {(unsigned int)(5) | XCL_MEM_TOPOLOGY, index32, 0};

    // create device buffer and map dev buf to host buf
    cl::Buffer pair_buf, cnt_buf, offsetOneHop_buf, indexOneHop_buf, offsetTwoHop_buf, indexTwoHop_buf;

    pair_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                          sizeof(ap_uint<64>) * numPairs, &mext_o[0]);
    cnt_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                         sizeof(unsigned) * numPairs, &mext_o[1]);
    offsetOneHop_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(unsigned) * (numVertices + 1), &mext_o[2]);
    indexOneHop_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                 sizeof(unsigned) * (numEdges), &mext_o[3]);
    offsetTwoHop_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(unsigned) * (numVertices + 1), &mext_o[4]);
    indexTwoHop_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                 sizeof(unsigned) * (numEdges), &mext_o[5]);

    std::vector<cl::Memory> init;
    init.push_back(pair_buf);
    init.push_back(cnt_buf);
    init.push_back(offsetOneHop_buf);
    init.push_back(indexOneHop_buf);
    init.push_back(offsetTwoHop_buf);
    init.push_back(indexTwoHop_buf);
    q.enqueueMigrateMemObjects(init, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, nullptr);
    q.finish();

    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;
    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(1);
    std::vector<cl::Event> events_read(1);

    ob_in.push_back(pair_buf);
    ob_in.push_back(offsetOneHop_buf);
    ob_in.push_back(indexOneHop_buf);
    ob_in.push_back(offsetTwoHop_buf);
    ob_in.push_back(indexTwoHop_buf);
    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &events_write[0]);

    ob_out.push_back(cnt_buf);
    // launch kernel and calculate kernel execution time
    std::cout << "kernel start------" << std::endl;
    gettimeofday(&start_time, 0);
    int j = 0;
    twoHop.setArg(j++, numPairs);
    twoHop.setArg(j++, pair_buf);
    twoHop.setArg(j++, offsetOneHop_buf);
    twoHop.setArg(j++, indexOneHop_buf);
    twoHop.setArg(j++, offsetTwoHop_buf);
    twoHop.setArg(j++, indexTwoHop_buf);
    twoHop.setArg(j++, cnt_buf);

    q.enqueueTask(twoHop, &events_write, &events_kernel[0]);

    q.enqueueMigrateMemObjects(ob_out, 1, &events_kernel, &events_read[0]);
    q.finish();

    gettimeofday(&end_time, 0);
    std::cout << "kernel end------" << std::endl;
    /*    std::cout << "Execution time " << tvdiff(&start_time, &end_time) / 1000.0 << "ms" << std::endl;

        unsigned long time1, time2;
        events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
        events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
        std::cout << "Write DDR Execution time " << (time2 - time1) / 1000000.0 << "ms" << std::endl;
        events_kernel[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
        events_kernel[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
        std::cout << "kernel Execution time " << (time2 - time1) / 1000000.0 << "ms" << std::endl;
        events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
        events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
        std::cout << "Read DDR Execution time " << (time2 - time1) / 1000000.0 << "ms" << std::endl;*/
    std::cout << "============================================================" << std::endl;

    std::fstream goldenfstream(goldenfile.c_str(), std::ios::in);
    if (!goldenfstream) {
        std::cout << "Error : " << goldenfile << " file doesn't exist !" << std::endl;
        exit(1);
    }

    std::unordered_map<unsigned long, float> goldenHashMap;
    while (goldenfstream.getline(line, sizeof(line))) {
        std::string str(line);
        std::replace(str.begin(), str.end(), ',', ' ');
        std::stringstream data(str.c_str());
        unsigned long golden_src;
        unsigned long golden_des;
        unsigned golden_res;
        data >> golden_src;
        data >> golden_des;
        data >> golden_res;
        unsigned long tmp = 0UL | golden_src << 32UL | golden_des;
        goldenHashMap.insert(std::pair<unsigned long, unsigned>(tmp, golden_res));
    }
    goldenfstream.close();

    std::unordered_map<unsigned long, float> resHashMap;
    for (int i = 0; i < numPairs; i++) {
        unsigned long tmp_src = pair[i].range(63, 32) + 1;
        unsigned long tmp_des = pair[i].range(31, 0) + 1;
        unsigned long tmp_res = cnt_res[i];
        unsigned long tmp = 0UL | tmp_src << 32UL | tmp_des;
        resHashMap.insert(std::pair<unsigned long, unsigned>(tmp, tmp_res));
    }

    if (resHashMap.size() != goldenHashMap.size()) std::cout << "miss pairs!" << std::endl;
    for (auto it = resHashMap.begin(); it != resHashMap.end(); it++) {
        unsigned long tmp_src = (it->first) / (1UL << 32UL);
        unsigned long tmp_des = (it->first) % (1UL << 32UL);
        unsigned long tmp_res = it->second;
        auto got = goldenHashMap.find(it->first);
        if (got == goldenHashMap.end()) {
            std::cout << "ERROR: pair not found! cnt_src: " << tmp_src << " cnt_des: " << tmp_des
                      << " cnt_res: " << tmp_res << std::endl;
            err++;
        } else if (got->second != it->second) {
            std::cout << "ERROR: incorrect count! golden_src: " << (got->first) / (1UL << 32UL)
                      << " golden_des: " << (got->first) % (1UL << 32UL) << " golden_res: " << (got->second)
                      << " cnt_src: " << tmp_src << " cnt_des: " << tmp_des << " cnt_res: " << tmp_res << std::endl;
            err++;
        }
    }

    if (err) {
        logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL);
    } else {
        logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    }

    return err;
}
