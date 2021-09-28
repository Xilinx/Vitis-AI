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

// L3
#include "xf_database/gqe_init.hpp"
#include "xf_database/gqe_utils.hpp"
#include "ap_int.h"

namespace xf {
namespace database {
namespace gqe {

// initialize the fpga with xclbin
FpgaInit::FpgaInit(std::string xclbin) {
    xclbin_path = xclbin;
    cl_int err = xf::database::gqe::init_hardware(&ctx, &dev_id, &cq,
                                                  CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to init hardware\n");
        exit(1);
    }

    err = xf::database::gqe::load_binary(&prg, ctx, dev_id, xclbin_path.c_str());
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to program PL\n");
        exit(1);
    }
    // createBuffers();
}

FpgaInit::~FpgaInit() {
    cl_int err;
    err = clReleaseMemObject(dbuf_ddr0);
    err = clReleaseMemObject(dbuf_ddr1);
    for (int i = 0; i < PU_NM * 2; i++) {
        err = clReleaseMemObject(dbuf_hbm[i]);
    }
    err = clReleaseProgram(prg);
    if (err != CL_SUCCESS) {
        std::cout << "deconstructor" << std::endl;
        exit(1);
    }

    clReleaseCommandQueue(cq);
    clReleaseContext(ctx);
};

// create host buffer
void FpgaInit::createHostBufs() {
    hbuf_ddr0 = mm.aligned_alloc<char>(DDR_SIZE_IN_BYTE);
    hbuf_ddr1 = mm.aligned_alloc<char>(DDR_SIZE_IN_BYTE);
    memset(hbuf_ddr0, 0, DDR_SIZE_IN_BYTE * sizeof(char));
    memset(hbuf_ddr1, 0, DDR_SIZE_IN_BYTE * sizeof(char));
    // for gqeFilter passing on hash-table into HBMs
    for (int i = 0; i < PU_NM; i++) {
        hbuf_hbm[i] = mm.aligned_alloc<char>(HBM_SIZE_IN_BYTE);
        memset(hbuf_hbm[i], 0, HBM_SIZE_IN_BYTE * sizeof(char));
    }
}

// create consecutive big device buffer while init FPGA
void FpgaInit::createDevBufs() {
    // create device buffer DDR0 and DDR1
    // ddr0
    mext_ddr0 = {XCL_MEM_TOPOLOGY | unsigned(32), hbuf_ddr0, 0};
    dbuf_ddr0 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX,
                               (int64_t)(DDR_SIZE_IN_BYTE * sizeof(char)), &mext_ddr0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: failed to create DDR0 buffer.\n");
        exit(1);
    }
    // ddr1
    mext_ddr1 = {XCL_MEM_TOPOLOGY | unsigned(33), hbuf_ddr1, 0};
    dbuf_ddr1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX,
                               (int64_t)(DDR_SIZE_IN_BYTE * sizeof(char)), &mext_ddr1, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: failed to create DDR1 buffer.\n");
        exit(1);
    }
    // hbm
    cl_mem_ext_ptr_t mext_hbm[16];
    mext_hbm[0] = {((unsigned int)(0) | XCL_MEM_TOPOLOGY), hbuf_hbm[0], 0};
    mext_hbm[1] = {((unsigned int)(2) | XCL_MEM_TOPOLOGY), hbuf_hbm[1], 0};
    mext_hbm[2] = {((unsigned int)(6) | XCL_MEM_TOPOLOGY), hbuf_hbm[2], 0};
    mext_hbm[3] = {((unsigned int)(8) | XCL_MEM_TOPOLOGY), hbuf_hbm[3], 0};
    mext_hbm[4] = {((unsigned int)(12) | XCL_MEM_TOPOLOGY), hbuf_hbm[4], 0};
    mext_hbm[5] = {((unsigned int)(22) | XCL_MEM_TOPOLOGY), hbuf_hbm[5], 0};
    mext_hbm[6] = {((unsigned int)(24) | XCL_MEM_TOPOLOGY), hbuf_hbm[6], 0};
    mext_hbm[7] = {((unsigned int)(28) | XCL_MEM_TOPOLOGY), hbuf_hbm[7], 0};

    mext_hbm[8] = {((unsigned int)(1) | XCL_MEM_TOPOLOGY), NULL, 0};
    mext_hbm[9] = {((unsigned int)(3) | XCL_MEM_TOPOLOGY), NULL, 0};
    mext_hbm[10] = {((unsigned int)(7) | XCL_MEM_TOPOLOGY), NULL, 0};
    mext_hbm[11] = {((unsigned int)(9) | XCL_MEM_TOPOLOGY), NULL, 0};
    mext_hbm[12] = {((unsigned int)(13) | XCL_MEM_TOPOLOGY), NULL, 0};
    mext_hbm[13] = {((unsigned int)(23) | XCL_MEM_TOPOLOGY), NULL, 0};
    mext_hbm[14] = {((unsigned int)(25) | XCL_MEM_TOPOLOGY), NULL, 0};
    mext_hbm[15] = {((unsigned int)(29) | XCL_MEM_TOPOLOGY), NULL, 0};

    // htb
    for (int j = 0; j < PU_NM; j++) {
        dbuf_hbm[j] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX,
                                     (int64_t)(sizeof(ap_uint<256>) * HT_BUFF_DEPTH), &mext_hbm[j], &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create HBM buffer.\n");
            exit(1);
        }
    }
    // make sure buffers resident on dev
    std::vector<cl_mem> in_hbms;
    for (int c = 0; c < PU_NM; c++) {
        in_hbms.push_back(dbuf_hbm[c]);
    }
    clEnqueueMigrateMemObjects(cq, in_hbms.size(), in_hbms.data(), CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, nullptr,
                               nullptr);
    // stb
    for (int j = PU_NM; j < PU_NM * 2; j++) {
        dbuf_hbm[j] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_EXT_PTR_XILINX,
                                     (int64_t)(sizeof(ap_uint<256>) * S_BUFF_DEPTH), &mext_hbm[j], &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create HBM buffer.\n");
            exit(1);
        }
    }

    // make sure all buffers in DDR0 DDR1 and HBM0 - HBM7 resident on dev
    std::vector<cl_mem> mem_vec;
    mem_vec.push_back(dbuf_ddr0);
    mem_vec.push_back(dbuf_ddr1);
    for (int c = 0; c < PU_NM; c++) {
        mem_vec.push_back(dbuf_hbm[c]);
    }
    clEnqueueMigrateMemObjects(cq, mem_vec.size(), mem_vec.data(), CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, nullptr,
                               nullptr);
}

} // database
} // gqe
} // xf
