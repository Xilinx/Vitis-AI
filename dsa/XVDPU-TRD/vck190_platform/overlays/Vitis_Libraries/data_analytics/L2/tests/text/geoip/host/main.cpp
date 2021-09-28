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
#include "utils.hpp"
#include <cstring>
#include <sstream>
#include <iostream>
#include <sys/time.h>
#include "geoip_sw.hpp"
#include "xf_utils_sw/logger.hpp"

extern "C" void GeoIP_kernel(
    int ipNum, uint32* ip, uint64* netHigh16, uint512* netLow21, uint512* net2Low21, uint32* netID);

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
    std::cout << "\n---------------------Geo IP-----------------\n";
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
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
    if (!parser.getCmdOption("-csv", filename)) { // offset
        std::cout << "ERROR: csv file path is not set!\n";
#ifdef HLS_TEST
        filename = "test.csv";
// filename = "GeoLite2-City-Blocks-IPv4.csv";
#else
        return -1;
#endif
    }
    int ipNum = NIP;
    std::string iter;

    if (parser.getCmdOption("-iter", iter)) {
        ipNum = NIP * std::stoi(iter);
    }

    int nerr = 0;
    unsigned int* ip = aligned_alloc<unsigned int>(ipNum);
    unsigned int* id = aligned_alloc<unsigned int>(ipNum);

    unsigned int ipUnit[17] = {0x101,      0x1000400,  0x1014009,  0x01019405, 0x279C454F, 0x28400000,
                               0x284280FF, 0x384080FF, 0x384280FF, 0x28620615, 0x28631001, 0x6144130E,
                               0x4A8FDDEE, 0x77AC6088, 0xD839A288, 0xD839A4FF};

    unsigned int idGolden[16] = {0xFFFFFFFF, 0x3,     0x70,    0x093,    0x5330a, 0x533e0, 0x53481, 0x81E3D,
                                 0x81E3D,    0x53bc9, 0x53bc9, 0x197827, 1000480, 1999998, 2999998, 3000001};

    std::cout << "ipNum=" << ipNum << ", NIP=" << NIP << ", TH1=" << TH1 << ", TH2=" << TH2 << ", Bank1=" << Bank1
              << ", Bank2=" << Bank2 << std::endl;

    // randomly generate the test IP data
    for (int i = 0; i < ipNum; i++) {
        unsigned int R = rand() % 256;
        R = (R << 8);
        R += rand() % 256;
        R = (R << 8);
        R += rand() % 256;
        R = (R << 8);
        R += rand() % 256;
        ip[i] = R;
        // std::cout << "R=" << std::hex << R << std::endl;
    }
    // read database to in-memory buffer
    std::vector<std::string> geoip;
    readGeoIP(filename, geoip);

    uint64_t* netsID = aligned_alloc<uint64_t>(N16);
    unsigned int* netsBegin = aligned_alloc<unsigned int>(geoip.size());
    unsigned int* netsEnd = aligned_alloc<unsigned int>(geoip.size());
    int netsLow21_size = (geoip.size() + 15) / 16;
    std::cout << "netsLow21 create buffer size is " << netsLow21_size << std::endl;

    uint512* netsLow21_512 = aligned_alloc<uint512>(netsLow21_size);
    geoipConvert(geoip, netsID, (uint512*)netsLow21_512, netsBegin, netsEnd);

#ifdef HLS_TEST
    std::cout << "start GeoIP kernel\n";
    GeoIP_kernel(ipNum, (uint32*)ip, (uint64*)netsID, (uint512*)netsLow21_512, (uint512*)netsLow21_512, (uint32*)id);
#else
    // do pre-process on CPU
    struct timeval start_time, end_time;
    cl_int cl_err;
    // platform related operations
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Creating Context and Command Queue for selected Device
    cl::Context context(device, NULL, NULL, NULL, &cl_err);
    logger.logCreateContext(cl_err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &cl_err);
    logger.logCreateCommandQueue(cl_err);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("Found Device=%s\n", devName.c_str());

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &cl_err);
    logger.logCreateProgram(cl_err);
    cl::Kernel GeoIPKernel(program, "GeoIP_kernel", &cl_err);
    logger.logCreateKernel(cl_err);
    std::cout << "kernel has been created" << std::endl;

    cl_mem_ext_ptr_t mext_o[5];
    mext_o[0] = {1, ip, GeoIPKernel()};
    mext_o[1] = {2, netsID, GeoIPKernel()};
    mext_o[2] = {3, netsLow21_512, GeoIPKernel()};
    // mext_o[3] = {3, netsLow21_512, GeoIPKernel()};
    mext_o[4] = {5, id, GeoIPKernel()};

    // create device buffer and map dev buf to host buf
    cl::Buffer ip_buf, netsAddr_buf, netsLow_buf, netsLow2_buf, id_buf;
    ip_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                        sizeof(uint32) * ipNum, &mext_o[0]);
    netsAddr_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                              sizeof(uint64) * N16, &mext_o[1]);
    netsLow_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             sizeof(uint512) * netsLow21_size, &mext_o[2]);
    // netsLow2_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
    //                          sizeof(uint512) * netsLow21_size, &mext_o[3]);
    id_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                        sizeof(uint32) * ipNum, &mext_o[4]);

    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_buf;
    std::vector<cl::Memory> ob_out;
    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(1);
    std::vector<cl::Event> events_read(1);

    ob_in.push_back(ip_buf);
    ob_in.push_back(netsAddr_buf);
    ob_in.push_back(netsLow_buf);
    // ob_in.push_back(netsLow2_buf);
    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &events_write[0]);
    ob_out.push_back(id_buf);

    // launch kernel and calculate kernel execution time
    std::cout << "kernel start------" << std::endl;
    gettimeofday(&start_time, 0);
    int j = 0;
    GeoIPKernel.setArg(j++, ipNum);
    GeoIPKernel.setArg(j++, ip_buf);
    GeoIPKernel.setArg(j++, netsAddr_buf);
    GeoIPKernel.setArg(j++, netsLow_buf);
    GeoIPKernel.setArg(j++, netsLow_buf);
    // GeoIPKernel.setArg(j++, netsLow2_buf);
    GeoIPKernel.setArg(j++, id_buf);

    q.enqueueTask(GeoIPKernel, &events_write, &events_kernel[0]);

    q.enqueueMigrateMemObjects(ob_out, 1, &events_kernel, &events_read[0]);
    q.finish();

    gettimeofday(&end_time, 0);
    std::cout << "kernel end------" << std::endl;
    std::cout << "Execution time " << tvdiff(&start_time, &end_time) / 1000.0 << "ms" << std::endl;

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
    std::cout << "Total Execution time " << total_time / 1000000.0 << " ms" << std::endl;
#endif
    std::vector<std::string> geoip_out;
    nerr = geoip_check(ipNum, ip, id, netsBegin, netsEnd, geoip, geoip_out);
    delete[] netsID;
    delete[] netsBegin;
    delete[] netsEnd;
    // delete[] netsLow21_512;
    delete[] ip;
    delete[] id;

    nerr ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
         : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return nerr;
}
