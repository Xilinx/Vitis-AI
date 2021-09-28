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
#include "shortestPath_top.hpp"
#endif
#include "xf_utils_sw/logger.hpp"
#include "ap_int.h"
#include "utils.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <vector>
#include <limits>

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
    std::cout << "\n---------------------Shortest Path----------------\n";
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
    std::string columnfile;
    std::string goldenfile;
    int repInt = 1;
    if (!parser.getCmdOption("-o", offsetfile)) { // offset
        std::cout << "ERROR: offset file path is not set!\n";
        return -1;
    }
    if (!parser.getCmdOption("-c", columnfile)) { // column
        std::cout << "ERROR: row file path is not set!\n";
        return -1;
    }
    if (!parser.getCmdOption("-g", goldenfile)) { // golden
        std::cout << "ERROR: row file path is not set!\n";
        return -1;
    }

    // -------------setup k0 params---------------
    int err = 0;

    char line[1024] = {0};
    int index = 0;

    int numVertices;
    int numEdges;
    unsigned int sourceID = 30;

    std::fstream offsetfstream(offsetfile.c_str(), std::ios::in);
    if (!offsetfstream) {
        std::cout << "Error : " << offsetfile << " file doesn't exist !" << std::endl;
        exit(1);
    }

    offsetfstream.getline(line, sizeof(line));
    std::stringstream numOdata(line);
    numOdata >> numVertices;
    numOdata >> numVertices;

    ap_uint<32>* offset32 = aligned_alloc<ap_uint<32> >(numVertices + 1);
    while (offsetfstream.getline(line, sizeof(line))) {
        std::stringstream data(line);
        data >> offset32[index];
        index++;
    }

    ap_uint<512>* offset512 = reinterpret_cast<ap_uint<512>*>(offset32);
    int max = 0;
    int id = 0;
    for (int i = 0; i < numVertices; i++) {
        if (offset32[i + 1] - offset32[i] > max) {
            max = offset32[i + 1] - offset32[i];
            id = i;
        }
    }
    std::cout << "id: " << id << " max out: " << max << std::endl;
    sourceID = id;

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
    double* weight32 = aligned_alloc<double>(numEdges);
    while (columnfstream.getline(line, sizeof(line))) {
        std::stringstream data(line);
        data >> column32[index];
        data >> weight32[index];
        index++;
    }
    ap_uint<512>* column512 = reinterpret_cast<ap_uint<512>*>(column32);
    ap_uint<512>* weight512 = reinterpret_cast<ap_uint<512>*>(weight32);

    ap_uint<8>* info = aligned_alloc<ap_uint<8> >(4);
    memset(info, 0, 4 * sizeof(ap_uint<8>));
    double* result;
    result = aligned_alloc<double>(((numVertices + 511) / 512) * 512);

    ap_uint<32>* pred;
    pred = aligned_alloc<ap_uint<32> >(((numVertices + 1023) / 1024) * 1024);

    ap_uint<32>* ddrQue = aligned_alloc<ap_uint<32> >(10 * 300 * 4096);

    ap_uint<32>* config;
    config = aligned_alloc<ap_uint<32> >(6);
    config[0] = numVertices;
    union f_cast {
        double f;
        unsigned long int i;
    };
    f_cast tmp;
    tmp.f = std::numeric_limits<double>::infinity();
    ap_uint<64> infdouble = tmp.i;
    config[1] = infdouble.range(0, 31);
    config[2] = infdouble.range(63, 32);
    config[3] = 10 * 300 * 4096;
    ap_uint<32> cmd;
    cmd.set_bit(0, 1); // enable weight?
    cmd.set_bit(1, 1); // enable predecessor?
    cmd.set_bit(2, 0); // double or fixed? 0 for double, 1 for fixed
    config[4] = cmd;
    config[5] = sourceID;
#ifndef HLS_TEST
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
    cl::Program program(context, devices, xclBins, NULL, &fail);
    logger.logCreateProgram(fail);
    cl::Kernel shortestPath;
    shortestPath = cl::Kernel(program, "shortestPath_top", &fail);
    logger.logCreateKernel(fail);

    std::cout << "kernel has been created" << std::endl;

    std::vector<cl_mem_ext_ptr_t> mext_o(8);
#ifdef USE_HBM
    mext_o[0] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, offset512, 0};
    mext_o[1] = {(unsigned int)(2) | XCL_MEM_TOPOLOGY, column512, 0};
    mext_o[2] = {(unsigned int)(4) | XCL_MEM_TOPOLOGY, weight512, 0};
    mext_o[3] = {(unsigned int)(2) | XCL_MEM_TOPOLOGY, info, 0};
    mext_o[4] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, config, 0};
    mext_o[5] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, ddrQue, 0};
    mext_o[6] = {(unsigned int)(2) | XCL_MEM_TOPOLOGY, result, 0};
    mext_o[7] = {(unsigned int)(4) | XCL_MEM_TOPOLOGY, pred, 0};
#else
    mext_o[0] = {XCL_MEM_DDR_BANK0, offset512, 0};
    mext_o[1] = {XCL_MEM_DDR_BANK0, column512, 0};
    mext_o[2] = {XCL_MEM_DDR_BANK0, weight512, 0};
    mext_o[3] = {XCL_MEM_DDR_BANK0, info, 0};
    mext_o[4] = {XCL_MEM_DDR_BANK0, config, 0};
    mext_o[5] = {XCL_MEM_DDR_BANK0, ddrQue, 0};
    mext_o[6] = {XCL_MEM_DDR_BANK0, result, 0};
    mext_o[7] = {XCL_MEM_DDR_BANK0, pred, 0};
#endif
    // create device buffer and map dev buf to host buf
    cl::Buffer offset_buf, column_buf, weight_buf, info_buf, ddrQue_buf, result_buf, config_buf, pred_buf;
    offset_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            sizeof(ap_uint<32>) * (numVertices + 1), &mext_o[0]);
    column_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            sizeof(ap_uint<32>) * numEdges, &mext_o[1]);
    weight_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            sizeof(ap_uint<64>) * numEdges, &mext_o[2]);
    info_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                          sizeof(ap_uint<8>) * 4, &mext_o[3]);
    config_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            sizeof(ap_uint<32>) * 6, &mext_o[4]);
    ddrQue_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            sizeof(ap_uint<32>) * 10 * 300 * 4096, &mext_o[5]);
    result_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            sizeof(double) * ((numVertices + 511) / 512) * 512, &mext_o[6]);
    pred_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                          sizeof(ap_uint<32>) * ((numVertices + 1023) / 1024) * 1024, &mext_o[7]);

    std::vector<cl::Memory> init;
    init.push_back(config_buf);
    init.push_back(offset_buf);
    init.push_back(column_buf);
    init.push_back(weight_buf);
    init.push_back(ddrQue_buf);
    init.push_back(result_buf);
    init.push_back(pred_buf);
    init.push_back(info_buf);
    q.enqueueMigrateMemObjects(init, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, nullptr);
    q.finish();

    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;
    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(1);
    std::vector<cl::Event> events_read(1);

    ob_in.push_back(config_buf);
    ob_in.push_back(offset_buf);
    ob_in.push_back(column_buf);
    ob_in.push_back(weight_buf);
    ob_in.push_back(info_buf);
    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &events_write[0]);

    ob_out.push_back(result_buf);
    ob_out.push_back(pred_buf);
    ob_out.push_back(info_buf);
    //    q.finish();
    // launch kernel and calculate kernel execution time
    std::cout << "kernel start------" << std::endl;
    gettimeofday(&start_time, 0);
    int j = 0;
    shortestPath.setArg(j++, config_buf);
    shortestPath.setArg(j++, offset_buf);
    shortestPath.setArg(j++, column_buf);
    shortestPath.setArg(j++, weight_buf);
    shortestPath.setArg(j++, ddrQue_buf);
    shortestPath.setArg(j++, ddrQue_buf);
    shortestPath.setArg(j++, result_buf);
    shortestPath.setArg(j++, result_buf);
    shortestPath.setArg(j++, pred_buf);
    shortestPath.setArg(j++, pred_buf);
    shortestPath.setArg(j++, info_buf);

    q.enqueueTask(shortestPath, &events_write, &events_kernel[0]);

    q.enqueueMigrateMemObjects(ob_out, 1, &events_kernel, &events_read[0]);
    q.finish();

    gettimeofday(&end_time, 0);
    std::cout << "kernel end------" << std::endl;
//    std::cout << "Execution time " << tvdiff(&start_time, &end_time) / 1000.0 << "ms" << std::endl;

/*    unsigned long time1, time2, total_time;
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    std::cout << "Write DDR Execution time " << (time2 - time1) / 1000000.0 << "ms" << std::endl;
    total_time = time2 - time1;
    for (int i = 0; i < repInt; i++) {
        events_kernel[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
        events_kernel[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
        std::cout << "Kernel[" << i << "] Execution time " << (time2 - time1) / 1000000.0 << "ms" << std::endl;
        total_time += time2 - time1;
    }
    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    std::cout << "Read DDR Execution time " << (time2 - time1) / 1000000.0 << "ms" << std::endl;
    total_time += time2 - time1;
    std::cout << "Total Execution time " << total_time / 1000000.0 << "ms" << std::endl;*/
#else
    ap_uint<512>* result512 = reinterpret_cast<ap_uint<512>*>(result);
    ap_uint<64>* result64 = reinterpret_cast<ap_uint<64>*>(result);
    ap_uint<512>* ddrQue512 = reinterpret_cast<ap_uint<512>*>(ddrQue);
    ap_uint<512>* pred512 = reinterpret_cast<ap_uint<512>*>(pred);
    shortestPath_top(config, offset512, column512, weight512, ddrQue512, ddrQue, result512, result64, pred512, pred,
                     info);
#endif
    std::cout << "============================================================" << std::endl;

    if (info[0] != 0) {
        std::cout << "queue overflow" << std::endl;
        exit(1);
    }
    if (info[1] != 0) {
        std::cout << "table overflow" << std::endl;
        exit(1);
    }

    bool* connect;
    connect = aligned_alloc<bool>(((numVertices + 1023) / 1024) * 1024);
    for (int i = 0; i < numVertices; i++) {
        connect[i] = false;
    }

    std::fstream goldenfstream(goldenfile.c_str(), std::ios::in);
    if (!goldenfstream) {
        std::cout << "Err : " << goldenfile << " file doesn't exist !" << std::endl;
        exit(1);
    }
    goldenfstream.getline(line, sizeof(line));

    index = 0;
    while (goldenfstream.getline(line, sizeof(line))) {
        std::string str(line);
        std::replace(str.begin(), str.end(), ',', ' ');
        std::stringstream data(str.c_str());
        int vertex;
        double distance;
        int pred_golden;
        data >> vertex;
        data >> distance;
        data >> pred_golden;
        if (std::abs(result[vertex - 1] - distance) / distance > 0.00001) {
            std::cout << "Err distance: " << vertex - 1 << " " << distance << " " << result[vertex - 1] << std::endl;
            err++;
        }
        if (pred_golden - 1 != pred[vertex - 1]) {
            unsigned int tmp_fromID = pred[vertex - 1];
            unsigned int tmp_toID = vertex - 1;
            double tmp_distance = 0;
            int iter = 0;
            while ((tmp_fromID != sourceID || tmp_toID != sourceID) && iter < numVertices) {
                double tmp_weight = 0;
                int begin = offset32[tmp_fromID];
                int end = offset32[tmp_fromID + 1];
                for (int i = begin; i < end; i++) {
                    if (column32[i] == tmp_toID) {
                        tmp_weight = weight32[i];
                    }
                }
                tmp_distance = tmp_distance + tmp_weight;
                tmp_toID = tmp_fromID;
                tmp_fromID = pred[tmp_fromID];
                iter++;
            }
            if (std::abs(result[vertex - 1] - tmp_distance) / tmp_distance > 0.00001) {
                std::cout << "Err predecessor: " << vertex - 1 << std::endl;
                std::cout << "WRONG PATH is: " << std::endl;
                tmp_fromID = pred[vertex - 1];
                tmp_toID = vertex - 1;
                iter = 0;
                while ((tmp_fromID != sourceID || tmp_toID != sourceID) && iter < numVertices) {
                    std::cout << tmp_fromID << " ";
                    tmp_toID = tmp_fromID;
                    tmp_fromID = pred[tmp_fromID];
                    iter++;
                }
                std::cout << std::endl;
                err++;
            }
        }
        connect[vertex - 1] = true;
    }

    for (int i = 0; i < numVertices; i++) {
        if (connect[i] == false && result[i] != std::numeric_limits<double>::infinity()) {
            std::cout << "Err distance: " << i << " " << std::numeric_limits<double>::infinity() << " " << result[i]
                      << std::endl;
            err++;
        }
        if (connect[i] == false && pred[i] != std::numeric_limits<unsigned int>::max()) {
            std::cout << "Err predecessor: " << i << " not connected " << pred[i] << std::endl;
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
