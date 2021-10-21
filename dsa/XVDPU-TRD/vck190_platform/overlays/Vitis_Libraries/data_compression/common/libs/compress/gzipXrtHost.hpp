/*
 * (c) Copyright 2019-2021 Xilinx, Inc. All rights reserved.
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
 *
 */
#ifndef _XFCOMPRESSION_GZIPXRTHOST_HPP_
#define _XFCOMPRESSION_GZIPXRTHOST_HPP_

#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <thread>
//#include "xcl2.hpp"
#include <sys/stat.h>
#include <random>
#include <new>
#include <map>
#include <future>
#include <queue>
#include <assert.h>
#include <syslog.h>
#include "compressBase.hpp"
#include "gzipBase.hpp"

// XRT Host API Headers
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

// Maximum host buffer used to operate
// per kernel invocation
#define MAX_HOST_BUFFER_SIZE (2 * 1024 * 1024)
#define HOST_BUFFER_SIZE (1024 * 1024)

// Default block size
#define BLOCK_SIZE_IN_KB 32

class gzipXrtHost : public gzipBase {
   public:
    // Constructor
    gzipXrtHost(enum State flow,
                const std::string& xclbin,
                const std::string& uncompFileName,
                const bool isSeq = false,
                uint8_t device_id = 0,
                bool enable_profile = false,
                uint8_t decKernelType = FULL,
                uint8_t dflow = XILINX_GZIP,
                bool freeRunKernel = false);

    /**
     * @brief returns check sum value (CRC32/Adler32)
     */
    uint32_t get_checksum(void);

    void set_checksum(uint32_t cval);

    // Compress
    uint64_t compressEngineSeq(uint8_t* in,
                               uint8_t* out,
                               uint64_t actual_size,
                               int level = 1,
                               int strategy = 0,
                               int window_bits = 15,
                               uint32_t* checksum = nullptr) override;
    uint64_t compressEngineOverlap(uint8_t* in,
                                   uint8_t* out,
                                   uint64_t actual_size,
                                   int cu = 0,
                                   int level = 1,
                                   int strategy = 0,
                                   int window_bits = 15,
                                   uint32_t* checksum = nullptr) {
        return 0;
    };
    uint64_t decompressEngine(uint8_t* in, uint8_t* out, uint64_t input_size, uint64_t max_output_size, int cu = 0) {
        return 0;
    };
    uint64_t decompressEngineSeq(uint8_t* in, uint8_t* out, uint64_t input_size, uint64_t max_output_size, int cu = 0) {
        return 0;
    };

    size_t compress_buffer(
        uint8_t* in, uint8_t* out, size_t input_size, uint32_t host_buffer_size = HOST_BUFFER_SIZE, int cu = 0);

    // Destructor
    ~gzipXrtHost(){};

   private:
    xrt::device m_device;
    xrt::uuid m_uuid;
    std::vector<std::string> m_cuNames;
    uint32_t m_checksum = 0;
    uint8_t m_cdflow;
    uint8_t m_deviceid = 0;
    enum design_flow m_zlibFlow;
    uint32_t m_kidx;
    uint8_t m_pending = 0;
    xrt::kernel datamoverKernel;
    xrt::kernel compressKernel;
};
#endif
