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
/**
 * @file zstdOCLHost.hpp
 * @brief Header for ZSTD Base functionality
 *
 * This file is part of Vitis Data Compression Library host code for zstd compression.
 */

#ifndef _XFCOMPRESSION_ZSTD_OCL_HOST_HPP_
#define _XFCOMPRESSION_ZSTD_OCL_HOST_HPP_

#include <iomanip>
#include "xcl2.hpp"
#include "compressBase.hpp"
#include "zstdBase.hpp"

#define CMP_HOST_BUF_SIZE (1024 * 1024)

/**
 *  zstdOCLHost class. Class containing methods for ZSTD
 * compression and decompression to be executed on host side.
 */
class zstdOCLHost : public zstdBase {
   private:
    enum State m_flow;
    std::string m_xclbin;
    uint8_t m_deviceId;
    bool m_enableProfile;
    uint8_t m_maxCr;
    uint32_t m_testItrCount;
    cl::Program* m_program;
    cl::Context* m_context;
#ifndef FREE_RUNNING_KERNEL
    cl::CommandQueue* m_q_dec;
    cl::CommandQueue* m_q_cmp;
#endif
    cl::CommandQueue* m_q_rd;
    cl::CommandQueue* m_q_wr;
    cl::CommandQueue* m_q_cdm;

    // Decompress Kernel Declaration
    cl::Kernel* compress_kernel;
    cl::Kernel* decompress_kernel;
    cl::Kernel* cmp_dm_kernel;
    cl::Kernel* data_writer_kernel;
    cl::Kernel* data_reader_kernel;

    std::chrono::system_clock::time_point kernel_start;
    std::chrono::system_clock::time_point kernel_end;
    std::chrono::system_clock::time_point total_start;
    std::chrono::system_clock::time_point total_end;

    // Compression related
    std::vector<uint8_t, aligned_allocator<uint8_t> > cbuf_in;
    std::vector<uint8_t, aligned_allocator<uint8_t> > cbuf_out;
    std::vector<uint64_t, aligned_allocator<uint64_t> > cbuf_outSize;

    // Decompression related
    std::vector<uint8_t, aligned_allocator<uint8_t> > dbuf_in;
    std::vector<uint8_t, aligned_allocator<uint8_t> > dbuf_out;
    std::vector<uint64_t, aligned_allocator<uint64_t> > dbuf_outSize;
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_dcompressStatus;

    // Device buffers
    cl::Buffer* buffer_cmp_input;
    cl::Buffer* buffer_cmp_output;
    cl::Buffer* buffer_cmp_size;

    cl::Buffer* buffer_dec_input;
    cl::Buffer* buffer_dec_output;
    cl::Buffer* buffer_size;
    cl::Buffer* buffer_status;

    // Kernel names
    std::string compress_kernel_name = "xilZstdCompress";
    std::string decompress_kernel_name = "xilZstdDecompressStream";
    std::string cmp_dm_kernel_name = "xilZstdDataMover";
    std::string data_writer_kernel_name = "xilGzipMM2S";
    std::string data_reader_kernel_name = "xilGzipS2MM";

   public:
    /**
     * @brief Initialize host/device and OpenCL Setup
     *
     */
    zstdOCLHost(enum State flow,
                const std::string& binaryFileName,
                uint8_t device_id,
                uint8_t max_cr,
                bool enable_profile = false,
                uint32_t itrCnt = 1);

    /**
     * @brief Xilinx Compress
     *
     * @param in input byte sequence
     * @param out output byte sequence
     * @param actual_size input size
     */
    uint64_t xilCompress(uint8_t* in, uint8_t* out, size_t input_size) override;

    /**
     * @brief Xilinx Decompress
     *
     * @param in input byte sequence
     * @param out output byte sequence
     * @param compressed size
     */

    uint64_t xilDecompress(uint8_t* in, uint8_t* out, size_t input_size) override;

    /**
     * @brief Decompress sequential.
     *
     * @param in input byte sequence
     * @param out output byte sequence
     * @param actual_size original size
     */
    uint64_t compressEngine(uint8_t* in, uint8_t* out, size_t actual_size) override;

    /**
     * @brief Decompress sequential.
     *
     * @param in input byte sequence
     * @param out output byte sequence
     * @param original_size original size
     */
    uint64_t decompressEngine(uint8_t* in, uint8_t* out, size_t actual_size) override;

    /**
     * @brief Setter method for testItrCount.
     */
    void setTestItrCount(uint16_t itrcnt);

    /**
     * @brief Release host/device and OpenCL setup
     */
    ~zstdOCLHost();
};

#endif // _XFCOMPRESSION_ZSTD_OCL_HPP_
