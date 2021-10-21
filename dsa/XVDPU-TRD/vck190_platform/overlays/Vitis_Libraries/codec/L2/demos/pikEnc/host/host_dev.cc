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

#include <iostream>
#include <sys/time.h>
#include "host_dev.hpp"

#ifndef HLS_TEST

#include "xcl2.hpp"
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
#define XCL_BANK16 XCL_BANK(16)
#define XCL_BANK17 XCL_BANK(17)
#define XCL_BANK18 XCL_BANK(18)
#define XCL_BANK19 XCL_BANK(19)
#define XCL_BANK20 XCL_BANK(20)
#define XCL_BANK21 XCL_BANK(21)
#define XCL_BANK22 XCL_BANK(22)
#define XCL_BANK23 XCL_BANK(23)
#define XCL_BANK24 XCL_BANK(24)
#define XCL_BANK25 XCL_BANK(25)
#define XCL_BANK26 XCL_BANK(26)
#define XCL_BANK27 XCL_BANK(27)
#define XCL_BANK28 XCL_BANK(28)
#define XCL_BANK29 XCL_BANK(29)
#define XCL_BANK30 XCL_BANK(30)
#define XCL_BANK31 XCL_BANK(31)
#define XCL_BANK32 XCL_BANK(32)
#define XCL_BANK33 XCL_BANK(33)

unsigned long diff(const struct timeval* newTime, const struct timeval* oldTime) {
    return (newTime->tv_sec - oldTime->tv_sec) * 1000000 + (newTime->tv_usec - oldTime->tv_usec);
}

template <typename T>
T* aligned_alloc(std::size_t num) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, 4096, num * sizeof(T))) throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
}

void host_func(std::string xclbinPath,
               float* dataDDR,
               ap_uint<AXI_SZ> k1_config[MAX_NUM_CONFIG],
               ap_uint<AXI_SZ> k2_config[MAX_NUM_CONFIG],
               ap_uint<AXI_SZ> k3_config[MAX_NUM_CONFIG],

               ap_uint<AXI_SZ> cmap[AXI_CMAP],
               ap_uint<AXI_SZ> order[MAX_NUM_ORDER],
               ap_uint<AXI_SZ> quant_field[AXI_QF],

               int len_dc_histo[2 * MAX_DC_GROUP],
               int len_dc[2 * MAX_DC_GROUP],
               ap_uint<AXI_SZ> dc_histo_code_out[2 * MAX_DC_GROUP * MAX_DC_HISTO_SIZE],
               ap_uint<AXI_SZ> dc_code_out[2 * MAX_DC_GROUP * MAX_DC_SIZE],

               int len_ac_histo[MAX_AC_GROUP],
               int len_ac[MAX_AC_GROUP],
               ap_uint<AXI_SZ> ac_histo_code_out[MAX_AC_GROUP * MAX_AC_HISTO_SIZE],
               ap_uint<AXI_SZ> ac_code_out[MAX_AC_GROUP * MAX_AC_SIZE]) {
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    cl_int fail;

    struct timeval start_time; // End to end time clock start
    gettimeofday(&start_time, 0);

    // platform related operations
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Creating Context and Command Queue for selected Device
    cl::Context context(device, NULL, NULL, NULL, &fail);
    logger.logCreateContext(fail);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &fail);
    logger.logCreateCommandQueue(fail);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("INFO: Found Device=%s\n", devName.c_str());
    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbinPath);

    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &fail);
    logger.logCreateProgram(fail);

    int repInt = 1;
    // create kernels
    std::vector<cl::Kernel> pik_kernel1(repInt);
    std::vector<cl::Kernel> pik_kernel2(repInt);
    std::vector<cl::Kernel> pik_kernel3(repInt);
    for (int i = 0; i < repInt; i++) {
        pik_kernel1[i] = cl::Kernel(program, "kernel1Top", &fail);
        logger.logCreateKernel(fail);
        pik_kernel2[i] = cl::Kernel(program, "kernel2Top", &fail);
        logger.logCreateKernel(fail);
        pik_kernel3[i] = cl::Kernel(program, "kernel3Top", &fail);
        logger.logCreateKernel(fail);
    }
    std::cout << "INFO: kernel has been created" << std::endl;

    // declare map of host buffers
    std::cout << "kernel config size:" << MAX_NUM_CONFIG << std::endl;
    std::cout << "buf size:" << k2_config[8] << std::endl;
    std::cout << "ac size:" << k2_config[8] << std::endl;
    std::cout << "dc size:" << k2_config[24] << std::endl;
    std::cout << "acs size:" << k2_config[13] << std::endl;
    std::cout << "cmap size:" << k2_config[10] << std::endl;
    std::cout << "order size:" << k2_config[23] << std::endl;
    std::cout << "quant size:" << k2_config[15] << std::endl;
    std::cout << "ac_histo size:" << k3_config[12] * MAX_AC_HISTO_SIZE << std::endl;
    std::cout << "dc_histo size:" << 2 * k3_config[13] * MAX_DC_HISTO_SIZE << std::endl;
    std::cout << "ac_code size:" << k3_config[12] * MAX_AC_SIZE << std::endl;
    std::cout << "dc_code size:" << 2 * k3_config[13] * MAX_DC_SIZE << std::endl;

    ap_uint<32>* hb_config1 = aligned_alloc<ap_uint<32> >(MAX_NUM_CONFIG);
    ap_uint<32>* hb_config2 = aligned_alloc<ap_uint<32> >(MAX_NUM_CONFIG);
    ap_uint<32>* hb_config3 = aligned_alloc<ap_uint<32> >(MAX_NUM_CONFIG);
    float* hb_data_in = aligned_alloc<float>(BUF_DEPTH);

    ap_uint<32>* hb_buf_out = aligned_alloc<ap_uint<32> >(k2_config[8]);
    ap_uint<32>* hb_qf = aligned_alloc<ap_uint<32> >(k2_config[9]);
    ap_uint<32>* hb_cmap = aligned_alloc<ap_uint<32> >(k2_config[10]);

    ap_uint<32>* hb_ac = aligned_alloc<ap_uint<32> >(k2_config[8]);
    ap_uint<32>* hb_dc = aligned_alloc<ap_uint<32> >(k2_config[24]);
    ap_uint<32>* hb_order = aligned_alloc<ap_uint<32> >(k2_config[23]);
    ap_uint<32>* hb_strategy = aligned_alloc<ap_uint<32> >(k2_config[13]);
    ap_uint<32>* hb_block = aligned_alloc<ap_uint<32> >(k2_config[14]);
    ap_uint<32>* hb_quant = aligned_alloc<ap_uint<32> >(k2_config[15]);

    ap_uint<32>* hb_histo_cfg = aligned_alloc<ap_uint<32> >(4 * k3_config[13] + 2 * k3_config[12]);
    ap_uint<32>* hb_dc_histo = aligned_alloc<ap_uint<32> >(2 * k3_config[13] * MAX_DC_HISTO_SIZE);
    ap_uint<32>* hb_dc_code = aligned_alloc<ap_uint<32> >(2 * k3_config[13] * MAX_DC_SIZE);
    ap_uint<32>* hb_ac_histo = aligned_alloc<ap_uint<32> >(k3_config[12] * MAX_AC_HISTO_SIZE);
    ap_uint<32>* hb_ac_code = aligned_alloc<ap_uint<32> >(k3_config[12] * MAX_AC_SIZE);

    for (int j = 0; j < MAX_NUM_CONFIG; j++) {
        hb_config1[j] = k1_config[j];
        hb_config2[j] = k2_config[j];
        hb_config3[j] = k3_config[j];
    }

    for (int j = 0; j < BUF_DEPTH; j++) hb_data_in[j] = dataDDR[j];

    for (int j = 0; j < k2_config[8]; j++) hb_buf_out[j] = 0;

    for (int j = 0; j < k2_config[8]; j++) {
        hb_ac[j] = 0;
    }

    for (int j = 0; j < k2_config[24]; j++) {
        hb_dc[j] = 0;
    }

    for (int j = 0; j < k2_config[13]; j++) {
        hb_strategy[j] = 0;
        hb_block[j] = 0;
        hb_quant[j] = 0;
    }

    for (int j = 0; j < k2_config[10]; j++) {
        hb_cmap[j] = 0;
    }

    for (int j = 0; j < k2_config[23]; j++) hb_order[j] = 0;

    for (int j = 0; j < 4 * k3_config[13] + 2 * k3_config[12]; j++) {
        hb_histo_cfg[j] = 0;
    }

    for (int j = 0; j < 2 * k3_config[13] * MAX_DC_HISTO_SIZE; j++) {
        hb_dc_histo[j] = 0;
    }

    for (int j = 0; j < 2 * k3_config[13] * MAX_DC_SIZE; j++) {
        hb_dc_code[j] = 0;
    }

    for (int j = 0; j < k3_config[12] * MAX_AC_HISTO_SIZE; j++) {
        hb_ac_histo[j] = 0;
    }

    for (int j = 0; j < k3_config[12] * MAX_AC_SIZE; j++) {
        hb_ac_code[j] = 0;
    }

    std::vector<cl_mem_ext_ptr_t> mext_o(18);
    mext_o[0] = {XCL_BANK(0), hb_config1, 0};
    mext_o[1] = {XCL_BANK(0), hb_data_in, 0};
    mext_o[2] = {XCL_BANK(1), hb_buf_out, 0};
    mext_o[3] = {XCL_BANK(1), hb_qf, 0};
    mext_o[4] = {XCL_BANK(1), hb_cmap, 0};

    mext_o[5] = {XCL_BANK(1), hb_config2, 0};
    mext_o[6] = {XCL_BANK(2), hb_ac, 0};
    mext_o[7] = {XCL_BANK(2), hb_dc, 0};
    mext_o[8] = {XCL_BANK(2), hb_order, 0};
    mext_o[9] = {XCL_BANK(2), hb_strategy, 0};
    mext_o[10] = {XCL_BANK(2), hb_block, 0};
    mext_o[11] = {XCL_BANK(2), hb_quant, 0};

    mext_o[12] = {XCL_BANK(2), hb_config3, 0};
    mext_o[13] = {XCL_BANK(3), hb_histo_cfg, 0};
    mext_o[14] = {XCL_BANK(3), hb_ac_histo, 0};
    mext_o[15] = {XCL_BANK(3), hb_ac_code, 0};
    mext_o[16] = {XCL_BANK(3), hb_dc_histo, 0};
    mext_o[17] = {XCL_BANK(3), hb_dc_code, 0};

    // create device buffer and map dev buf to host buf
    cl::Buffer db_conf1, db_buf_in, db_buf_out, db_qf, db_cmap;
    cl::Buffer db_conf2, db_ac, db_dc, db_order, db_strategy, db_quant, db_block;
    cl::Buffer db_conf3, db_histo_cfg, db_dc_histo, db_dc_code, db_ac_histo, db_ac_code;

    db_conf1 = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                          sizeof(ap_int<32>) * MAX_NUM_CONFIG, &mext_o[0]);
    db_buf_in = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                           sizeof(ap_int<32>) * k2_config[8], &mext_o[1]);
    db_buf_out = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            sizeof(ap_int<32>) * k2_config[8], &mext_o[2]);
    db_qf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                       sizeof(ap_int<32>) * k2_config[9], &mext_o[3]);
    db_cmap = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                         sizeof(ap_int<32>) * k2_config[10], &mext_o[4]);

    db_conf2 = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                          sizeof(ap_int<32>) * MAX_NUM_CONFIG, &mext_o[5]);
    db_ac = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                       sizeof(ap_int<32>) * k2_config[8], &mext_o[6]);
    db_dc = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                       sizeof(ap_int<32>) * k2_config[24], &mext_o[7]);
    db_order = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                          sizeof(ap_int<32>) * k2_config[23], &mext_o[8]);
    db_strategy = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             sizeof(ap_int<32>) * k2_config[13], &mext_o[9]);
    db_block = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                          sizeof(ap_int<32>) * k2_config[14], &mext_o[10]);
    db_quant = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                          sizeof(ap_int<32>) * k2_config[15], &mext_o[11]);

    db_conf3 = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                          sizeof(ap_int<32>) * MAX_NUM_CONFIG, &mext_o[12]);
    db_histo_cfg = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                              sizeof(ap_int<32>) * (4 * k3_config[13] + 2 * k3_config[12]), &mext_o[13]);
    db_ac_histo = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             sizeof(ap_int<32>) * k3_config[12] * MAX_AC_HISTO_SIZE, &mext_o[14]);
    db_ac_code = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            sizeof(ap_int<32>) * k3_config[12] * MAX_AC_SIZE, &mext_o[15]);
    db_dc_histo = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             sizeof(ap_int<32>) * 2 * k3_config[13] * MAX_DC_HISTO_SIZE, &mext_o[16]);
    db_dc_code = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            sizeof(ap_int<32>) * 2 * k3_config[13] * MAX_DC_SIZE, &mext_o[17]);

    // add buffers to migrate
    std::vector<cl::Memory> init;

    init.push_back(db_conf1);
    init.push_back(db_buf_in);
    init.push_back(db_buf_out);
    init.push_back(db_qf);
    init.push_back(db_cmap);

    init.push_back(db_conf2);
    init.push_back(db_ac);
    init.push_back(db_dc);
    init.push_back(db_order);
    init.push_back(db_strategy);
    init.push_back(db_block);
    init.push_back(db_quant);

    init.push_back(db_conf3);
    init.push_back(db_histo_cfg);
    init.push_back(db_ac_histo);
    init.push_back(db_ac_code);
    init.push_back(db_dc_histo);
    init.push_back(db_dc_code);

    // migrate data from host to device
    q.enqueueMigrateMemObjects(init, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, nullptr);
    q.finish();

    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out1, ob_out2, ob_out3;

    ob_in.push_back(db_conf1);
    ob_in.push_back(db_buf_in);
    ob_out1.push_back(db_buf_out);
    ob_out1.push_back(db_qf);
    ob_out1.push_back(db_cmap);

    ob_in.push_back(db_conf2);
    ob_out2.push_back(db_ac);
    ob_out2.push_back(db_dc);
    ob_out2.push_back(db_order);
    ob_out2.push_back(db_strategy);
    ob_out2.push_back(db_block);
    ob_out2.push_back(db_quant);

    ob_in.push_back(db_conf3);
    ob_out3.push_back(db_histo_cfg);
    ob_out3.push_back(db_ac_histo);
    ob_out3.push_back(db_ac_code);
    ob_out3.push_back(db_dc_histo);
    ob_out3.push_back(db_dc_code);

    // declare events
    std::vector<cl::Event> events_write(1);
    std::vector<std::vector<cl::Event> > events_kernel(3);
    std::vector<std::vector<cl::Event> > events_read(3);
    for (int i = 0; i < 3; ++i) {
        events_kernel[i].resize(1);
        events_read[i].resize(1);
    }

    // set kernel args
    for (int i = 0; i < repInt; i++) {
        pik_kernel1[i].setArg(0, db_conf1);
        pik_kernel1[i].setArg(1, db_buf_in);
        pik_kernel1[i].setArg(2, db_buf_out);
        pik_kernel1[i].setArg(3, db_cmap);
        pik_kernel1[i].setArg(4, db_qf);

        pik_kernel2[i].setArg(0, db_conf2);
        pik_kernel2[i].setArg(1, db_buf_out);
        pik_kernel2[i].setArg(2, db_qf);
        pik_kernel2[i].setArg(3, db_cmap);
        pik_kernel2[i].setArg(4, db_ac);
        pik_kernel2[i].setArg(5, db_dc);
        pik_kernel2[i].setArg(6, db_quant);
        pik_kernel2[i].setArg(7, db_strategy);
        pik_kernel2[i].setArg(8, db_block);
        pik_kernel2[i].setArg(9, db_order);

        pik_kernel3[i].setArg(0, db_conf3);
        pik_kernel3[i].setArg(1, db_ac);
        pik_kernel3[i].setArg(2, db_dc);
        pik_kernel3[i].setArg(3, db_quant);
        pik_kernel3[i].setArg(4, db_strategy);
        pik_kernel3[i].setArg(5, db_block);
        pik_kernel3[i].setArg(6, db_order);
        pik_kernel3[i].setArg(7, db_histo_cfg);
        pik_kernel3[i].setArg(8, db_dc_histo);
        pik_kernel3[i].setArg(9, db_dc_code);
        pik_kernel3[i].setArg(10, db_ac_histo);
        pik_kernel3[i].setArg(11, db_ac_code);
    }

    // launch kernel and calculate kernel execution time
    std::cout << "INFO: Kernel Start" << std::endl;

    // migrate
    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &events_write[0]);
    q.enqueueTask(pik_kernel1[0], &events_write, &events_kernel[0][0]);
    q.enqueueMigrateMemObjects(ob_out1, 1, &events_kernel[0], &events_read[0][0]);
    q.enqueueTask(pik_kernel2[0], &events_read[0], &events_kernel[1][0]);
    q.enqueueMigrateMemObjects(ob_out2, 1, &events_kernel[1], &events_read[1][0]);
    q.enqueueTask(pik_kernel3[0], &events_read[1], &events_kernel[2][0]);
    q.enqueueMigrateMemObjects(ob_out3, 1, &events_kernel[2], &events_read[2][0]);
    q.finish();

    struct timeval end_time;
    gettimeofday(&end_time, 0);
    std::cout << "INFO: Finish kernel execution" << std::endl;
    std::cout << "INFO: Finish E2E execution" << std::endl;

    // print related times
    unsigned long timeStart, timeEnd, exec_time0;
    std::cout << "-------------------------------------------------------" << std::endl;
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &timeStart);
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &timeEnd);
    exec_time0 = (timeEnd - timeStart) / 1000.0;
    std::cout << "INFO: Data transfer from host to device: " << exec_time0 << " us\n";
    std::cout << "-------------------------------------------------------" << std::endl;
    events_read[0][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &timeStart);
    events_read[0][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &timeEnd);
    exec_time0 = (timeEnd - timeStart) / 1000.0;
    std::cout << "INFO: Kernel1 Data transfer from device to host: " << exec_time0 << " us\n";
    std::cout << "-------------------------------------------------------" << std::endl;
    events_read[1][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &timeStart);
    events_read[1][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &timeEnd);
    exec_time0 = (timeEnd - timeStart) / 1000.0;
    std::cout << "INFO: Kernel2 Data transfer from device to host: " << exec_time0 << " us\n";
    std::cout << "-------------------------------------------------------" << std::endl;
    events_read[2][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &timeStart);
    events_read[2][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &timeEnd);
    exec_time0 = (timeEnd - timeStart) / 1000.0;
    std::cout << "INFO: Kernel3 Data transfer from device to host: " << exec_time0 << " us\n";
    std::cout << "-------------------------------------------------------" << std::endl;
    exec_time0 = 0;
    for (int i = 0; i < 3; ++i) {
        events_kernel[i][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &timeStart);
        events_kernel[i][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &timeEnd);
        exec_time0 += (timeEnd - timeStart) / 1000.0;

        std::cout << "INFO: Kernel" << i + 1 << " execution: " << (timeEnd - timeStart) / 1000.0 << " us\n";
        std::cout << "-------------------------------------------------------" << std::endl;
    }
    std::cout << "INFO: kernel total execution: " << exec_time0 << " us\n";
    std::cout << "-------------------------------------------------------" << std::endl;
    unsigned long exec_timeE2E = diff(&end_time, &start_time);
    std::cout << "INFO: FPGA execution time:" << exec_timeE2E << " us\n";
    std::cout << "-------------------------------------------------------" << std::endl;

    // output
    for (int i = 0; i < k2_config[10]; i++) {
        cmap[i] = hb_cmap[i];
    }
    std::cout << "cmap finish" << std::endl;

    for (int i = 0; i < k2_config[23]; i++) {
        order[i] = hb_order[i];
    }
    std::cout << "order finish" << std::endl;

    for (int i = 0; i < k2_config[15]; i++) {
        quant_field[i] = hb_quant[i];
    }
    std::cout << "quant_field finish" << std::endl;

    int dc_histo_sum = 0;
    for (int j = 0; j < 2 * k3_config[13]; j++) {
        len_dc_histo[j] = hb_histo_cfg[j];
        dc_histo_sum += len_dc_histo[j];
        std::cout << "len_dc_h:" << (int)hb_histo_cfg[j] << std::endl;
    }
    std::cout << "dc_histo_sum:" << dc_histo_sum << std::endl;

    int ac_histo_sum = 0;
    for (int j = 0; j < k3_config[12]; j++) {
        len_ac_histo[j] = hb_histo_cfg[2 * k3_config[13] + j];
        ac_histo_sum += len_ac_histo[j];
        std::cout << "len_ac_h:" << (int)hb_histo_cfg[2 * k3_config[13] + j] << std::endl;
    }
    std::cout << "ac_histo_sum:" << ac_histo_sum << std::endl;

    int len_dc_sum = 0;
    for (int j = 0; j < 2 * k3_config[13]; j++) {
        len_dc[j] = hb_histo_cfg[2 * k3_config[13] + k3_config[12] + j];
        len_dc_sum += (len_dc[j] + 1) / 2;
        std::cout << "len_dc_c:" << (int)hb_histo_cfg[2 * k3_config[13] + k3_config[12] + j] << std::endl;
    }
    std::cout << "len_dc_sum:" << len_dc_sum << std::endl;

    int len_ac_sum = 0;
    for (int j = 0; j < k3_config[12]; j++) {
        len_ac[j] = hb_histo_cfg[4 * k3_config[13] + k3_config[12] + j];
        len_ac_sum += (len_ac[j] + 1) / 2;
        std::cout << "len_ac_c:" << (int)hb_histo_cfg[4 * k3_config[13] + k3_config[12] + j] << std::endl;
    }
    std::cout << "len_ac_sum:" << len_ac_sum << std::endl;

    const uint64_t num_dc_histo = 2 * k3_config[13] * MAX_DC_HISTO_SIZE;
    const uint64_t num_dc = 2 * k3_config[13] * MAX_DC_SIZE;
    const uint64_t num_ac_histo = k3_config[12] * MAX_AC_HISTO_SIZE;
    const uint64_t num_ac = k3_config[12] * MAX_AC_SIZE;
    memcpy(dc_histo_code_out, hb_dc_histo, sizeof(ap_uint<AXI_SZ>) * num_dc_histo);
    memcpy(dc_code_out, hb_dc_code, sizeof(ap_uint<AXI_SZ>) * num_dc);
    memcpy(ac_histo_code_out, hb_ac_histo, sizeof(ap_uint<AXI_SZ>) * num_ac_histo);
    memcpy(ac_code_out, hb_ac_code, sizeof(ap_uint<AXI_SZ>) * num_ac);

    std::cout << "k2 order:" << std::endl;
    for (int i = 0; i < k2_config[6] * k2_config[7]; i++) {
        for (int j = 0; j < 64 * 3; j++) {
            std::cout << (int)order[i * 3 * 64 + j] << ",";
        }
        std::cout << std::endl;
    }

    std::cout << "k2 quant:" << std::endl;
    for (int i = 0; i < k2_config[3]; i++) {
        for (int j = 0; j < k2_config[2]; j++) {
            std::cout << (int)quant_field[i * k2_config[2] + j] << ",";
        }
        std::cout << std::endl;
    }

    std::cout << "k2 global_scale:" << (int)quant_field[k2_config[15] - 2] << std::endl;

    std::cout << "k2 dequant:" << (int)quant_field[k2_config[15] - 1] << std::endl;

    std::cout << "k2 acs:" << std::endl;
    for (int i = 0; i < k2_config[3]; i++) {
        for (int j = 0; j < k2_config[2]; j++) {
            std::cout << (int)hb_strategy[i * k2_config[2] + j] << ",";
        }
        std::cout << std::endl;
    }

    std::cout << "k2 block:" << std::endl;
    for (int i = 0; i < k2_config[3]; i++) {
        for (int j = 0; j < k2_config[2]; j++) {
            std::cout << (int)hb_block[i * k2_config[2] + j] << ",";
        }
        std::cout << std::endl;
    }

    std::cout << "k2 dc x:" << std::endl;
    for (int i = 0; i < k2_config[3]; i++) {
        for (int j = 0; j < k2_config[2]; j++) {
            std::cout << (int)hb_dc[i * k2_config[2] + j] << ",";
        }
        std::cout << std::endl;
    }

    std::cout << "k2 dc y:" << std::endl;
    for (int i = 0; i < k2_config[3]; i++) {
        for (int j = 0; j < k2_config[2]; j++) {
            std::cout << (int)hb_dc[i * k2_config[2] + k2_config[13]] << ",";
        }
        std::cout << std::endl;
    }

    std::cout << "k2 dc b:" << std::endl;
    for (int i = 0; i < k2_config[3]; i++) {
        for (int j = 0; j < k2_config[2]; j++) {
            std::cout << (int)hb_dc[i * k2_config[2] + 2 * k2_config[13]] << ",";
        }
        std::cout << std::endl;
    }

    std::cout << "k2 ac:" << std::endl;
    for (int i = 0; i < k2_config[3]; i++) {
        for (int j = 0; j < k2_config[2]; j++) {
            std::cout << "id=" << (i * k2_config[2] + j) << " ";
            for (int k = 0; k < 64; k++) {
                std::cout << (int)hb_ac[(i * k2_config[2] + j) * 64 + k] << ",";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "dc_histo_code_out:" << std::endl;
    for (int j = 0; j < dc_histo_sum; j++) {
        std::cout << ", " << (int)dc_histo_code_out[j];
        if (j != 0 && j % 32 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "dc_code_out:" << std::endl;
    for (int j = 0; j < len_dc_sum; j++) {
        std::cout << ", " << (int)dc_code_out[j];
        if (j != 0 && j % 32 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "ac_histo_code_out:" << std::endl;
    for (int j = 0; j < ac_histo_sum; j++) {
        std::cout << ", " << (int)ac_histo_code_out[j];
        if (j != 0 && j % 32 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "ac_code_out:" << std::endl;
    for (int j = 0; j < len_ac_sum; j++) {
        std::cout << ", " << (int)ac_code_out[j];
        if (j != 0 && j % 32 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    free(hb_buf_out);
    free(hb_ac);
    free(hb_dc);
    free(hb_strategy);
    free(hb_block);
    free(hb_cmap);
    free(hb_order);
    free(hb_quant);
    free(hb_histo_cfg);
    free(hb_dc_histo);
    free(hb_dc_code);
    free(hb_ac_histo);
    free(hb_ac_code);
}

#endif
