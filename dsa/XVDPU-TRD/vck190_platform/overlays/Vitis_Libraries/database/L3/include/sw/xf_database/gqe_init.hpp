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

#ifndef _GQE_INIT_L3_
#define _GQE_INIT_L3_

#include <iostream>
#include <thread>
#include <atomic>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <queue>

// commmon
// for opencl
// helpers for OpenCL C API
#ifndef HLS_TEST
#include "xf_database/gqe_ocl.hpp"
#include "xf_database/gqe_utils.hpp"
#endif

namespace xf {
namespace database {
namespace gqe {

// DDR buffer size
const int64_t DDR_SIZE_IN_BYTE = 4000000000;
// HBM size
const int64_t HBM_SIZE_IN_BYTE = (1 << 28);

enum {
    PU_NM = 8,
    // the size of HTB and STB, each using single 256MB HBM, width:256bit
    // so the depth = 256MB /256bit = 8M
    HT_BUFF_DEPTH = (1 << 23), // 8M
    S_BUFF_DEPTH = (1 << 23),
    HBM_BUFF_DEPTH = (1 << 23),
    VEC_LEN = 8,
    KEY_SZ = sizeof(int64_t)
};

class FpgaInit {
   private:
    gqe::utils::MM mm;

   public:
    cl_device_id dev_id;
    cl_mem_ext_ptr_t mext_ddr0;
    cl_mem_ext_ptr_t mext_ddr1;
    cl_mem_ext_ptr_t mext_hbm[2 * PU_NM];

    // for platform init
    cl_context ctx;
    cl_command_queue cq;
    cl_program prg;
    cl_int err;
    std::string xclbin_path;

    // host buffer
    char* hbuf_ddr0;
    char* hbuf_ddr1;
    char* hbuf_hbm[PU_NM]; // for gqeFilter only

    // the device buffers created, on ddr
    cl_mem dbuf_ddr0;
    cl_mem dbuf_ddr1;

    // the device buffer created, on hbm
    cl_mem dbuf_hbm[2 * PU_NM];

    // init the fpag with xclbin
    FpgaInit(std::string xclbin);

    ~FpgaInit();

    // create host buffer
    void createHostBufs();

    // create consecutive big device buffer while init FPGA
    void createDevBufs();
};
//-----------------------------------------------------------------------------------------------

} // gqe
} // database
} // xf
#endif
