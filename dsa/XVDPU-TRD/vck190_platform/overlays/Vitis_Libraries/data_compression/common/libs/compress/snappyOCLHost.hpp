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
 * @file snappyOCLHost.hpp
 * @brief Header for SNAPPY Base functionality
 *
 * This file is part of Vitis Data Compression Library host code for snappy compression.
 */

#ifndef _XFCOMPRESSION_SNAPPY_OCL_HOST_HPP_
#define _XFCOMPRESSION_SNAPPY_OCL_HOST_HPP_

#include <cassert>
#include <iomanip>
#include "xcl2.hpp"
#include <fcntl.h>  /* For O_RDWR */
#include <unistd.h> /* For open(), creat() */
#include "compressBase.hpp"
#include "snappyBase.hpp"
#include <string>
#include <sys/stat.h>

/**
 *  snappyOCLHost class. Class containing methods for SNAPPY
 * compression and decompression to be executed on host side.
 */
class snappyOCLHost : public snappyBase {
   public:
    /**
     * @brief Initialize host/device and OpenCL Setup
     *
     */
    snappyOCLHost(enum State flow,
                  const std::string& binaryFileName,
                  uint8_t device_id,
                  uint32_t block_size_kb,
                  bool enable_profile = false,
                  bool enable_p2p = 0);

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
     * @brief Compress sequential
     *
     * @param in input byte sequence
     * @param out output byte sequence
     * @param actual_size input size
     */
    uint64_t compressEngineSeq(uint8_t* in, uint8_t* out, uint64_t actual_size);
    uint64_t compressEngineStreamSeq(uint8_t* in, uint8_t* out, size_t actual_size);

    /**
     * @brief Decompress sequential.
     *
     * @param in input byte sequence
     * @param out output byte sequence
     * @param original_size original size
     */
    uint64_t decompressEngineSeq(uint8_t* in, uint8_t* out, uint64_t original_size);
    uint64_t decompressEngineStreamSeq(uint8_t* in, uint8_t* out, size_t original_size);

    /**
     * @brief Release host/device and OpenCL setup
     */
    ~snappyOCLHost();

   private:
    std::string m_xclbin;
    bool m_enableProfile;
    bool m_enableP2P;
    uint8_t m_deviceId;
    size_t m_inputSize;
    enum State m_flow;

    cl::Program* m_program;
    cl::Context* m_context;
    cl::CommandQueue* m_q;
    cl::Kernel* compress_kernel_snappy;
    cl::Kernel* compress_data_mover_kernel;
    cl::Kernel* decompress_kernel_snappy;
    cl::Kernel* decompress_data_mover_kernel;

    std::chrono::system_clock::time_point kernel_start;
    std::chrono::system_clock::time_point kernel_end;
    std::chrono::system_clock::time_point total_start;
    std::chrono::system_clock::time_point total_end;

    // Compression related
    std::vector<uint8_t, aligned_allocator<uint8_t> > h_buf_in;
    std::vector<uint8_t, aligned_allocator<uint8_t> > h_buf_out;
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_compressSize;
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_blksize;
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_buf_decompressSize;
    // Device buffers
    cl::Buffer* buffer_input;
    cl::Buffer* buffer_output;
    cl::Buffer* buffer_compressed_size;
    cl::Buffer* buffer_block_size;

    // Decompression related
    std::vector<uint32_t> m_blkSize;
    std::vector<uint32_t> m_compressSize;
    // Kernel names
    std::vector<std::string> compress_kernel_names = {"xilSnappyCompress", "xilSnappyCompressStream"};
    std::string compress_dm_kernel_names = "xilCompDatamover";
    std::vector<std::string> decompress_kernel_names = {"xilSnappyDecompress", "xilSnappyDecompressStream"};
    std::string decompress_dm_kernel_names = "xilDecompDatamover";
};

#endif // _XFCOMPRESSION_SNAPPY_OCL_HPP_
