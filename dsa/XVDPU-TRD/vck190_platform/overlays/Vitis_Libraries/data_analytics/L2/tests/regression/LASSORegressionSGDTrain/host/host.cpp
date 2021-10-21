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
#include <fstream>
#include "utils.hpp"
#include <ap_int.h>
#include <CL/cl_ext_xilinx.h>
#include <xcl2.hpp>
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

int main(int argc, const char* argv[]) {
    //
    std::cout << "\n--------- LASSO Regression SGD Train Test ---------\n";
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);

    // cmd arg parser.
    ArgParser parser(argc, argv);
    std::string xclbin_path; // eg. q5kernel_VCU1525_hw.xclbin
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR: xclbin path is not set!\n";
        return 1;
    }

    // Allocate Memory in Host Memory
    const int rows = 100;
    const int cols = 23;
    const float fraction = 1.0;
    const bool ifJump = false;
    const int bucketSize = 1;
    const ap_uint<32> seed = 42;
    double stepSize = 1.0;
    double tolerance = 0.001;
    bool withIntercept = false;
    ap_uint<32> maxIter = 99;
    const ap_uint<32> offset = 2;
    double regVal = 0.1;

    ap_uint<512>* table;
    int table_size = (rows * cols + 16 + 7) / 8;
    table = aligned_alloc<ap_uint<512> >(table_size);
    double* ttable = (double*)table;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols - 1; j++) {
            ttable[16 + i * cols + j] = (i + j) * 0.01;
        }
        ttable[16 + i * cols + cols - 1] = (i + i + cols - 2) * (cols - 1) / 2.0 * 0.01 * 0.3;
    }

    *(ap_uint<32>*)(ttable + 0) = seed;
    *(double*)(ttable + 1) = stepSize;
    *(double*)(ttable + 2) = tolerance;
    *(ap_uint<32>*)(ttable + 3) = withIntercept ? 1 : 0;
    *(ap_uint<32>*)(ttable + 4) = maxIter;
    *(ap_uint<32>*)(ttable + 5) = offset;
    *(ap_uint<32>*)(ttable + 6) = rows;
    *(ap_uint<32>*)(ttable + 7) = cols;
    *(ap_uint<32>*)(ttable + 8) = bucketSize;
    *(float*)(ttable + 9) = fraction;
    *(ap_uint<32>*)(ttable + 10) = ifJump ? 1 : 0;
    *(double*)(ttable + 11) = regVal;

    double weightGolden[cols - 1] = {0.17378717673636357, 0.18429218206666115, 0.19479719980974586, 0.2053022068754798,
                                     0.21580721874474093, 0.2263122340588962,  0.23681723067533744, 0.24732224644881173,
                                     0.25782726865771066, 0.2683322646505349,  0.2788372865485966,  0.2893422869269156,
                                     0.29984730573249074, 0.3103522922935321,  0.32085731417615854, 0.3313623342311564,
                                     0.34186734339762165, 0.35237236166588226, 0.3628773537166093,  0.3733823629723242,
                                     0.38388738819319396, 0.39439239190673836};

    int res_size = (200 + 7) / 8;
    ap_uint<512>* res = aligned_alloc<ap_uint<512> >(res_size);
    double* tres = (double*)res;

    cl_int cl_err;
    // Get CL devices.
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Create context and command queue for selected device
    cl::Context context(device, NULL, NULL, NULL, &cl_err);
    logger.logCreateContext(cl_err);
    cl::CommandQueue q(context, device,
                       // CL_QUEUE_PROFILING_ENABLE);
                       CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &cl_err);
    logger.logCreateCommandQueue(cl_err);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Selected Device " << devName << "\n";

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    std::vector<cl::Device> devices_h;
    devices_h.push_back(device);
    cl::Program program(context, devices, xclBins, NULL, &cl_err);
    logger.logCreateProgram(cl_err);

    cl::Kernel kernel;
    kernel = cl::Kernel(program, "LASSORegressionTrain", &cl_err);
    logger.logCreateKernel(cl_err);

#ifdef USE_DDR
    cl_mem_ext_ptr_t mext_table = {XCL_BANK0, table, 0};
    cl_mem_ext_ptr_t mext_res = {XCL_BANK0, res, 0};
#else
    cl_mem_ext_ptr_t mext_table = {(unsigned int)(0), table, 0};
    cl_mem_ext_ptr_t mext_res = {(unsigned int)(0), res, 0};
#endif

    // Map buffers
    int err;
    cl::Buffer buf_table(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,

                         (size_t)(sizeof(ap_uint<512>) * table_size), &mext_table, &err);
    printf("creating buf_table\n");

    cl::Buffer buf_res(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                       (size_t)(sizeof(ap_uint<512>) * res_size), &mext_res, &err);
    printf("creating buf_res\n");

    q.finish();
    std::cout << "DDR buffers have been mapped/copy-and-mapped\n";

    int num_rep = 1;
    std::vector<std::vector<cl::Event> > write_events(num_rep);
    std::vector<std::vector<cl::Event> > kernel_events(num_rep);
    std::vector<std::vector<cl::Event> > read_events(num_rep);
    for (int i = 0; i < num_rep; ++i) {
        write_events[i].resize(1);
        kernel_events[i].resize(1);
        read_events[i].resize(1);
    }
    std::vector<cl::Memory> buffwrite;
    buffwrite.push_back(buf_table);
    buffwrite.push_back(buf_res);

    struct timeval start_time, end_time;
    std::cout << "INFO: kernel start------" << std::endl;
    gettimeofday(&start_time, 0);

    int j = 0;
    kernel.setArg(j++, buf_table);
    kernel.setArg(j++, buf_res);

    q.enqueueMigrateMemObjects(buffwrite, 0, nullptr, &write_events[0][0]);
    q.enqueueTask(kernel, &write_events[0], &kernel_events[0][0]);
    std::vector<cl::Memory> buffread;
    buffread.push_back(buf_res);
    q.enqueueMigrateMemObjects(buffread, CL_MIGRATE_MEM_OBJECT_HOST, &kernel_events[0], &read_events[0][0]);
    q.finish();

    gettimeofday(&end_time, 0);
    std::cout << "INFO: kernel end------" << std::endl;
    std::cout << "INFO: Execution time " << tvdiff(&start_time, &end_time) / 1000.0 << "ms" << std::endl;
    unsigned long time1, time2, total_time;
    write_events[0][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    write_events[0][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    std::cout << "INFO: Write DDR Execution time " << (time2 - time1) / 1000000.0 << "ms" << std::endl;
    total_time = time2 - time1;
    kernel_events[0][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    kernel_events[0][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    std::cout << "INFO: Kernel Execution time " << (time2 - time1) / 1000000.0 << "ms" << std::endl;
    total_time += time2 - time1;
    read_events[0][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    read_events[0][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    std::cout << "INFO: Read DDR Execution time " << (time2 - time1) / 1000000.0 << "ms" << std::endl;
    total_time += time2 - time1;
    std::cout << "INFO: Total Execution time " << total_time / 1000000.0 << "ms" << std::endl;

    // check tree by computing precision and recall
    bool tested = 0;

    for (int i = 0; i < cols - 1; i++) {
        double diff = tres[i] - weightGolden[i];
        double rel_err = diff / weightGolden[i];
        if (rel_err < -0.00001 || rel_err > 0.00001) {
            std::cout << i << " th weight: " << tres[i] << std::endl;
            tested = 1;
        }
    }
    tested ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
           : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return tested;
}
