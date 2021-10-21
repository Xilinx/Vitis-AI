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
 * @file zlib_stream.hpp
 * @brief Header for ZLIB host functionality
 *
 * This file is part of Vitis Data Compression Library host code for lz4
 * compression.
 */

#ifndef _XFCOMPRESSION_ZLIB_COMPRESS_HPP_
#define _XFCOMPRESSION_ZLIB_COMPRESS_HPP_

#include "xcl2.hpp"
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <string>
#include <thread>
#include <time.h>
#include <vector>

/**
 * Maximum host buffer used to operate per kernel invocation
 */
#ifndef HOST_BUFFER_SIZE
#define HOST_BUFFER_SIZE (100 * 1024 * 1024)
#endif

/**
 * Default block size
 */
#ifndef BLOCK_SIZE_IN_KB
#define BLOCK_SIZE_IN_KB 32
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE (BLOCK_SIZE_IN_KB * 1024)
#endif

/**
 * @brief Validate the compressed file.
 *
 * @param inFile_name input file name
 * @param outFile_name output file name
 */
int validate(std::string& inFile_name, std::string& outFile_name);

/**
 *  xfLz4 class. Class containing methods for ZLIB
 * compression and decompression to be executed on host side.
 */
class xfZlib {
   public:
    /**
     * @brief Initialize the class object.
     *
     * @param binaryFile file to be read
     */
    int init(const std::string& binaryFile);

    /**
     * @brief release
     *
     */
    int release();

    /**
     * @brief Compress sequential
     *
     * @param in input byte sequence
     * @param out output byte sequence
     * @param actual_size input size
     * @param host_buffer_size host buffer size
     */
    uint64_t compressSequential(uint8_t* in, uint8_t* out, uint64_t actual_size, uint32_t host_buffer_size);

    /**
     * @brief Compress Full
     *
     * @param in input byte sequence
     * @param out output byte sequence
     * @param actual_size input size
     */
    uint64_t compressFull(uint8_t* in, uint8_t* out, uint64_t actual_size);

    /**
     * @brief Compress the input file.
     *
     * @param inFile_name input file name
     * @param outFile_name output file name
     * @param actual_size input size
     */
    uint64_t compressFile(std::string& inFile_name, std::string& outFile_name, uint64_t actual_size);

    /**
     * @brief Compress the input file.
     *
     * @param inFile_name input file name
     * @param outFile_name output file name
     * @param actual_size input size
     */
    uint64_t compressFileFull(std::string& inFile_name, std::string& outFile_name, uint64_t actual_size);

    /**
     * @brief Initialize host/device and OpenCL Setup
     *
     */
    xfZlib(const std::string& binaryFileName, uint32_t block_size_kb);

    /**
     * @brief Release host/device and OpenCL setup
     */
    ~xfZlib();

   private:
    /**
     * Block Size
     */
    uint32_t m_BlockSizeInKb;

    cl::Program* m_program;
    cl::Context* m_context;
    cl::CommandQueue* m_q;
    cl::Kernel* compress_kernel_zlib;
    cl::Kernel* datamover_kernel;

    // Compression related
    std::vector<uint8_t, aligned_allocator<uint8_t> > h_buf_in;
    std::vector<uint8_t, aligned_allocator<uint8_t> > h_buf_out;
    std::vector<uint32_t, aligned_allocator<uint8_t> > h_compressSize;

    // Device buffers
    cl::Buffer* buffer_input;
    cl::Buffer* buffer_output;
    cl::Buffer* buffer_compressed_size;
    cl::Buffer* buffer_checksum_data;

    // Kernel names
    std::vector<std::string> compress_kernel_names = {"xilZlibCompressFull", "xilGzipComp", "xilGzipCompBlock"};
    std::string datamover_kernel_name = "xilDataMover";
};

#endif // _XFCOMPRESSION_ZLIB_COMPRESS_HPP_
