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

#include "ap_int.h"
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

template <typename MType>
union f_cast {
    MType f;
    MType i;
};

template <>
union f_cast<unsigned int> {
    unsigned int f;
    unsigned int i;
};

template <>
union f_cast<unsigned long long> {
    unsigned long long f;
    unsigned long long i;
};

template <>
union f_cast<double> {
    double f;
    unsigned long long i;
};

template <>
union f_cast<float> {
    float f;
    unsigned int i;
};

inline void splitStr(const std::string& s, std::vector<std::string>& v, const std::string& c) {
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (std::string::npos != pos2) {
        v.push_back(s.substr(pos1, pos2 - pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length()) v.push_back(s.substr(pos1));
}

int load_dat(std::vector<ap_uint<64> >& dataset, const std::string& file_name) {
    std::string fn = file_name;
    std::ifstream ifs(fn, std::ifstream::in);

    for (int i = 0; i < 8; i++) dataset.push_back(0);

    if (ifs) {
        while (ifs.good()) {
            std::string str;
            std::vector<std::string> std_vec;

            std::getline(ifs, str);
            if (ifs.good()) {
                splitStr(str, std_vec, " ");

                ap_uint<12> type = std::stoi(std_vec[0]);
                for (unsigned int i = 1; i < std_vec.size(); i++) {
                    std::vector<std::string> vec_t;
                    splitStr(std_vec[i], vec_t, ":");
                    if (vec_t.size() != 2)
                        return -1;
                    else {
                        ap_uint<20> term = std::stoi(vec_t[0]);
                        ap_uint<32> tf = std::stoi(vec_t[1]);
                        dataset.push_back((type, term, tf));
                    }
                }

                ap_uint<64> end;
                end(31, 0) = 0;
                end(51, 32) = -1;
                end(63, 52) = type;
                dataset.push_back(end); // end of this sample
            }
        }
    } else {
        std::cerr << "ERROR: "
                  << "Failed to open dat file!\n";
        return -1;
    }

    ifs.close();

    return 0;
}

int main(int argc, const char* argv[]) {
    std::cout << "\n---------------------Multinomial Training Test of Naive Bayes-----------------\n";
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

    std::string in_file;
    if (!parser.getCmdOption("-in", in_file)) { // input CSR path
        std::cout << "ERROR: input dat file is not set!\n";
#ifdef HLS_TEST
        in_file = "../train.dat";
#else
        return 1;
#endif
    }

    std::string g_file;
    if (!parser.getCmdOption("-g", g_file)) { // input CSR path
        std::cout << "ERROR: golden file is not set!\n";
#ifdef HLS_TEST
        in_file = "../train_g.dat";
#else
        return 1;
#endif
    }

    std::string str_class;
    int num_of_class = 2;
    if (parser.getCmdOption("-c", str_class)) {
        try {
            num_of_class = std::stoi(str_class);
        } catch (...) {
            std::cout << "Warning: invaild number of class, pls re-run.\n";
        }
    } else {
        std::cout << "ERROR: number of class is not set!\n";
    }

    std::string str_term;
    int num_of_term = 100;
    if (parser.getCmdOption("-t", str_term)) {
        try {
            num_of_term = std::stoi(str_term);
        } catch (...) {
            std::cout << "Warning: invaild number of feature, pls re-run.\n";
        }
    } else {
        std::cout << "ERROR: number of feature is not set!\n";
    }

    int nerror = 0;

    std::vector<ap_uint<64> > dataset;
    nerror = load_dat(dataset, in_file);
    if (nerror) return 1;
    int fnum = dataset.size();
    if (fnum % 8 != 0) {
        for (int i = 0; i < (8 - fnum % 8); i++) {
            dataset.push_back(0);
        }
    }
    dataset[0] = (dataset.size() - 8) / 8;

    ap_uint<512>* buf_in = (ap_uint<512>*)dataset.data();
    size_t buf_in_bytes = 8 * dataset.size();
    size_t depth_buf_out0 = 10 + buf_in_bytes / 64;
    size_t depth_buf_out1 = num_of_class;

    ap_uint<512>* buf_out0 = aligned_alloc<ap_uint<512> >(depth_buf_out0);
    ap_uint<512>* buf_out1 = aligned_alloc<ap_uint<512> >(depth_buf_out1);
    memset(buf_out0, 0, 4);
    memset(buf_out1, 0, 4);

#ifndef HLS_TEST
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
    cl::Kernel kernel(program, "naiveBayesTrain_kernel", &cl_err);
    logger.logCreateKernel(cl_err);
    std::cout << "kernel has been created" << std::endl;

    cl_mem_ext_ptr_t mext_o[3];
    mext_o[0] = {2, buf_in, kernel()};
    mext_o[1] = {3, buf_out0, kernel()};
    mext_o[2] = {4, buf_out1, kernel()};

    // create device buffer and map dev buf to host buf
    cl::Buffer buffer_in =
        cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, buf_in_bytes, &mext_o[0]);
    cl::Buffer buffer_out0 = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                        64 * depth_buf_out0, &mext_o[1]);
    cl::Buffer buffer_out1 = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                        sizeof(ap_uint<512>) * depth_buf_out1, &mext_o[2]);

    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;
    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(1);
    std::vector<cl::Event> events_read(1);

    int j = 0;
    kernel.setArg(j++, num_of_class);
    kernel.setArg(j++, num_of_term);
    kernel.setArg(j++, buffer_in);
    kernel.setArg(j++, buffer_out0);
    kernel.setArg(j++, buffer_out1);

    ob_in.push_back(buffer_in);
    ob_out.push_back(buffer_out0);
    ob_out.push_back(buffer_out1);

    // launch kernel and calculate kernel execution time
    std::cout << "kernel start------" << std::endl;
    gettimeofday(&start_time, 0);
    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &events_write[0]);

    q.enqueueTask(kernel, &events_write, &events_kernel[0]);

    q.enqueueMigrateMemObjects(ob_out, 1, &events_kernel, &events_read[0]);
    q.finish();

    gettimeofday(&end_time, 0);
    std::cout << "kernel end------" << std::endl;
    std::cout << "Total Execution time " << tvdiff(&start_time, &end_time) / 1000.0 << "ms" << std::endl << std::endl;

    std::cout << "Start Profiling..." << std::endl;
    unsigned long time1, time2, total_time;
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    std::cout << "Write DDR Execution time " << (time2 - time1) / 1000000.0 << "ms" << std::endl;
    total_time = time2 - time1;
    events_kernel[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    events_kernel[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    std::cout << "Kernel Execution time " << (time2 - time1) / 1000000.0 << "ms" << std::endl;
    total_time += time2 - time1;
    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    std::cout << "Read DDR Execution time " << (time2 - time1) / 1000000.0 << "ms" << std::endl;
    total_time += time2 - time1;
    std::cout << "Total Execution time " << total_time / 1000000.0 << "ms" << std::endl;
#endif

    std::cout << "============================================================" << std::endl;

    std::vector<double> golden_result, hw_result;
    std::fstream goldenfstream(g_file.c_str(), std::ios::in);
    if (!goldenfstream) {
        std::cout << "Error : " << g_file << " file doesn't exist !" << std::endl;
        exit(1);
    }

    char line[1024] = {0};
    while (goldenfstream.getline(line, sizeof(line))) {
        std::stringstream data(line);
        for (unsigned int i = 0; i < buf_out0[0](31, 0); i++) {
            std::string tmp;
            data >> tmp;

            double tmpd = std::stod(tmp);
            golden_result.push_back(tmpd);
        }
    }

    // check the result
    int index = 0;
    ap_uint<32> nr0 = buf_out0[0](63, 32);
    ap_uint<32> nm0 = buf_out0[0](31, 0);
    for (int i = 0; i < nr0; i++) {
        ap_uint<512> t = buf_out0[i + 1];
        for (int j = 0; j < 8; j++) {
            f_cast<double> cc0;
            cc0.i = t(64 * j + 63, 64 * j);

            if (index < nm0) {
                hw_result.push_back(cc0.f);
                index++;
            }
        }

        if (index >= nm0) index = 0;
    }

    for (int i = 0; i < (num_of_class * nm0); i++) {
        if (std::abs(hw_result[i] - golden_result[i]) > 1e-8) {
            std::cout << "Mismatch found!" << std::endl;
            nerror++;
            break;
        }
    }

    std::cout << "\nPrior probability:" << std::endl;
    ap_uint<32> nr1 = buf_out1[0](63, 32);
    ap_uint<32> nm1 = buf_out1[0](31, 0);
    for (int i = 0; i < nr1; i++) {
        ap_uint<512> t = buf_out1[i + 1];
        for (int j = 0; j < 8; j++) {
            f_cast<double> cc0;
            cc0.i = t(64 * j + 63, 64 * j);
            std::cout << cc0.f << " ";
        }
        std::cout << std::endl;
    }

    if (nerror == 0) std::cout << "Check pass.\n";

    free(buf_out0);
    free(buf_out1);

    nerror ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
           : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return nerror;
}
