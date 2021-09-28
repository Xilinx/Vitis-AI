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
#include "xf_utils_sw/logger.hpp"
#endif
#include "utils.hpp"
#include <iostream>
#include "crc32_kernel.hpp"
#include <fstream>
#include <vector>

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
    std::cout << "\n--------------------- CRC32 -----------------\n";
    int nerr = 0;
    // cmd parser
    ArgParser parser(argc, argv);
    std::string xclbin_path;

    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }

    std::string filename;
    if (!parser.getCmdOption("-data", filename)) { // offset
        std::cout << "ERROR: file path is not set!\n";
        return -1;
    }
    ap_uint<32> golden = 0xff7e73d8;
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        std::cout << "ERROR: read file failure!\n";
        return 1;
    }

    uint32_t size;
    ifs.seekg(0, std::ios::end);
    size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    int size_w1 = (size + W - 1) / W;

    std::vector<ap_uint<W * 8> > in((size + W - 1) / W);
    ifs.read(reinterpret_cast<char*>(in.data()), size);

    int num = 1;
    std::string input_num;
    if (!parser.getCmdOption("-num", input_num)) {
        num = 1;
    } else {
        num = std::stoi(input_num);
    }
    // size *= num;
    // num = 1;
    int size_w = (size + W - 1) / W;

    ap_uint<32>* len = aligned_alloc<ap_uint<32> >(num);
    ap_uint<32>* crcInit = aligned_alloc<ap_uint<32> >(num);
    ap_uint<32>* crc32_out = aligned_alloc<ap_uint<32> >(num);
    ap_uint<8 * W>* data = aligned_alloc<ap_uint<8 * W> >(size_w * num);

    int offset = 0;
    for (int i = 0; i < num; i++) {
        len[i] = size;
        crcInit[i] = ~0;
        for (int j = 0; j < size_w; j++) {
            data[j + offset] = in[j % size_w1];
        }
        offset += size_w;
    }

    // do pre-process on CPU
    struct timeval start_time, end_time, test_time;
    // platform related operations
    xf::common::utils_sw::Logger logger;
    cl_int err = CL_SUCCESS;

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

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

    cl::Kernel kernel(program, "CRC32Kernel", &err);
    logger.logCreateKernel(err);

    std::cout << "kernel has been created" << std::endl;

    cl_mem_ext_ptr_t mext_o[5];
    int j = 0;
    mext_o[j++] = {2, len, kernel()};
    mext_o[j++] = {3, crcInit, kernel()};
    mext_o[j++] = {4, data, kernel()};
    mext_o[j++] = {5, crc32_out, kernel()};

    j = 0;
    // create device buffer and map dev buf to host buf
    cl::Buffer len_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                    sizeof(ap_uint<32>) * num, &mext_o[j++]);
    cl::Buffer crcInit_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                        sizeof(ap_uint<32>) * num, &mext_o[j++]);
    cl::Buffer data_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                     sizeof(ap_uint<8 * W>) * size_w * num, &mext_o[j++]);
    cl::Buffer crc32_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                      sizeof(ap_uint<32>) * num, &mext_o[j++]);

    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_buf;
    std::vector<cl::Memory> ob_out;
    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(1);
    std::vector<cl::Event> events_read(1);

    ob_in.push_back(len_buf);
    ob_in.push_back(crcInit_buf);
    ob_in.push_back(data_buf);
    ob_out.push_back(crc32_buf);

    // launch kernel and calculate kernel execution time
    std::cout << "kernel start------" << std::endl;
    gettimeofday(&start_time, 0);
    j = 0;
    kernel.setArg(j++, num);
    kernel.setArg(j++, offset);
    kernel.setArg(j++, len_buf);
    kernel.setArg(j++, crcInit_buf);
    kernel.setArg(j++, data_buf);
    kernel.setArg(j++, crc32_buf);

    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &events_write[0]);
    q.enqueueTask(kernel, &events_write, &events_kernel[0]);
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
    // CRC32Kernel(num, len, crcInit, data, crc32_out);

    for (int i = 0; i < num; i++) {
        ap_uint<32> crc_out = crc32_out[i];
        if (golden != crc_out) {
            std::cout << std::hex << "crc_out=" << crc_out << ",golden=" << golden << std::endl;
            nerr = 1;
        }
    }
    if (nerr == 0) {
        logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    } else {
        logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL);
    }
    return nerr;
}
