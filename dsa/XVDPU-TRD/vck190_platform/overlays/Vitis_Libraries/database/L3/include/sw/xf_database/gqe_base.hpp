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

#ifndef _GQE_BASE_L3_
#define _GQE_BASE_L3_

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
#include "xf_database/gqe_init.hpp"
#endif

namespace xf {
namespace database {
namespace gqe {

class Base {
   protected:
    // for platform init
    cl_context ctx;
    // cl_device_id dev_id;
    cl_command_queue cq;
    cl_program prg;

    // dev buffer
    cl_mem dbuf_ddr0;
    cl_mem dbuf_ddr1;
    cl_mem dbuf_hbm[2 * PU_NM];

    // host buffer
    char* hbuf_ddr0;
    char* hbuf_ddr1;
    char* hbuf_hbm[PU_NM];

    // the first element data position
    std::vector<std::vector<int64_t> > buf_head;
    // the size of each buffer
    std::vector<std::vector<int64_t> > buf_size;
    // the latest accumulated ddr0 used data size
    int64_t buf_accu_ddr0_size;
    // the latest accumulated ddr1 used data size
    int64_t buf_accu_ddr1_size;
    // the actual number of buffers allocated in DDR0
    int64_t buf_idx0;
    // the actual number of buffers allocated in DDR1
    int64_t buf_idx1;

    /**
     * @brief constructor of Base.
     *
     * @param obj: the initialized fpga init
     *
     * context, program, command queue are created and ready after fpga init.
     *
     */
    Base(FpgaInit& obj);

    // deconstructor
    ~Base();

   public:
    /**
     * @brief Setting the maximum number of buffers that are allowed to allocate for each ddr
     *
     * @param _num: the maximum ddr buffer num
     *
     */
    void SetBufAllocMaxNum(int _num);

    /**
    * @brief Request one host buffer in hbuf ddr0/1, the size and starting ptr of this buf is recorded.
    *
    * @param _ddr_idx: the device buffer DDR idx that corresponding to requested host buf
    * @param _size: the requested buf size
    *
    */
    char* AllocHostBuf(bool _ddr_idx, int64_t _size);

    /**
    * @brief Reset host Buf related record, but won't destroy host buffer
    */
    void ResetHostBuf() {
        buf_accu_ddr0_size = 0;
        buf_accu_ddr1_size = 0;
        buf_idx0 = 0;
        buf_idx1 = 0;
    }
};
//-----------------------------------------------------------------------------------------------

} // gqe
} // database
} // xf
#endif
