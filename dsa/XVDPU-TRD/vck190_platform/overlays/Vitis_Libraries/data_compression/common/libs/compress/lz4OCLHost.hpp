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
 * @file lz4OCLHost.hpp
 * @brief Header for LZ4 Base functionality
 *
 * This file is part of Vitis Data Compression Library host code for lz4 compression.
 */

#ifndef _XFCOMPRESSION_LZ4_OCL_HOST_HPP_
#define _XFCOMPRESSION_LZ4_OCL_HOST_HPP_

#include <cassert>
#include <iomanip>
#include "xcl2.hpp"
#include "compressBase.hpp"
#include "lz4Base.hpp"

/**
 *  lz4OCLHost class. Class containing methods for LZ4
 * compression and decompression to be executed on host side.
 */
class lz4OCLHost : public lz4Base {
   public:
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
     * @brief Initialize host/device and OpenCL Setup
     *
     */
    lz4OCLHost(enum State flow,
               const std::string& binaryFileName,
               uint8_t device_id,
               uint32_t block_size_kb,
               bool lz4_stream,
               bool enable_profile = false);

    /**
     * @brief Compress sequential
     *
     * @param in input byte sequence
     * @param out output byte sequence
     * @param actual_size input size
     */
    uint64_t compressEngineSeq(uint8_t* in, uint8_t* out, size_t input_size);
    uint64_t compressEngineStreamSeq(uint8_t* in, uint8_t* out, size_t input_size);

    /**
     * @brief Decompress sequential.
     *
     * @param in input byte sequence
     * @param out output byte sequence
     * @param actual_size input size
     */
    uint64_t decompressEngineSeq(uint8_t* in, uint8_t* out, size_t input_size);
    uint64_t decompressEngineStreamSeq(uint8_t* in, uint8_t* out, size_t input_size);
    /**
     * @brief Release host/device and OpenCL setup
     */
    ~lz4OCLHost();

   private:
    std::string m_xclbin;
    bool m_enableProfile;
    uint32_t m_deviceId;
    size_t m_inputSize;
    bool m_lz4Stream;
    enum State m_flow;
    cl::Program* m_program;
    cl::Context* m_context;
    cl::CommandQueue* m_q;
    cl::Kernel* compress_kernel_lz4;
    cl::Kernel* compress_data_mover_kernel;
    cl::Kernel* decompress_kernel_lz4;
    cl::Kernel* decompress_data_mover_kernel;

    std::chrono::system_clock::time_point kernel_start;
    std::chrono::system_clock::time_point kernel_end;
    std::chrono::system_clock::time_point total_start;
    std::chrono::system_clock::time_point total_end;

    // Compression related
    std::vector<uint8_t, aligned_allocator<uint8_t> > h_buf_in;
    std::vector<uint8_t, aligned_allocator<uint8_t> > h_buf_out;
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_buf_decompressSize;
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_blksize;
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_compressSize;
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_decSize;

    // Device buffers
    cl::Buffer* buffer_input;
    cl::Buffer* buffer_output;
    cl::Buffer* bufferOutputSize;
    cl::Buffer* buffer_compressed_size;
    cl::Buffer* buffer_block_size;
    cl::Buffer* buffer_dec_size;

    // Decompression related
    std::vector<uint32_t> m_blkSize;
    std::vector<uint32_t> m_compressSize;
};

#endif // _XFCOMPRESSION_LZ4_OCL_HPP_
