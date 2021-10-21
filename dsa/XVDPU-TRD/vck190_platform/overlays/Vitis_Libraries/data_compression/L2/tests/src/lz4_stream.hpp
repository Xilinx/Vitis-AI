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
 * @file xil_lz4.hpp
 * @brief Header for LZ4 host functionality
 *
 * This file is part of Vitis Data Compression Library host code for lz4
 * compression.
 */

#ifndef _XFCOMPRESSION_LZ4_STREAM_HPP_
#define _XFCOMPRESSION_LZ4_STREAM_HPP_

#include "xcl2.hpp"
#include <assert.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <string>
#include <time.h>
#include <vector>
// This extension file is required for stream APIs
#include <CL/cl_ext_xilinx.h>

/**
 * Maximum compute units supported
 */
#if (C_COMPUTE_UNIT > D_COMPUTE_UNIT)
#define MAX_COMPUTE_UNITS C_COMPUTE_UNIT
#else
#define MAX_COMPUTE_UNITS D_COMPUTE_UNIT
#endif

/**
 * Enable/diasble p2p flow
 * by default disable
 */
#ifndef ENABLE_P2P
#define ENABLE_P2P 0
#endif

/**
 * Maximum host buffer used to operate per kernel invocation
 */
#define HOST_BUFFER_SIZE (64 * 1024 * 1024)

/**
 * Default block size
 */
#define BLOCK_SIZE_IN_KB 64

/**
 * Value below is used to associate with
 * Overlapped buffers, ideally overlapped
 * execution requires 2 resources per invocation
 */
#define OVERLAP_BUF_COUNT 2

/**
 * Maximum number of blocks based on host buffer size
 */
#define MAX_NUMBER_BLOCKS (HOST_BUFFER_SIZE / (BLOCK_SIZE_IN_KB * 1024))

/**
 * Below are the codes as per LZ4 standard for
 * various maximum block sizes supported.
 */
#define BSIZE_STD_64KB 0x40
#define BSIZE_STD_256KB 0x50
#define BSIZE_STD_1024KB 0x60
#define BSIZE_STD_4096KB 0x70

/**
 * Maximum block sizes supported by LZ4
 */
#define MAX_BSIZE_64KB (64 * 1024)
#define MAX_BSIZE_256KB (256 * 1024)
#define MAX_BSIZE_1024KB (1024 * 1024)
#define MAX_BSIZE_4096KB (4096 * 1024)

/**
 * This value is used to set
 * uncompressed block size value.
 * 4th byte is always set to below
 * and placed as uncompressed byte
 */
#define NO_COMPRESS_BIT 128

/**
 * In case of uncompressed block
 * Values below are used to set
 * 3rd byte to following values
 * w.r.t various maximum block sizes
 * supported by standard
 */
#define BSIZE_NCOMP_64 1
#define BSIZE_NCOMP_256 4
#define BSIZE_NCOMP_1024 16
#define BSIZE_NCOMP_4096 64

/**
 * @brief Validate the compressed file.
 *
 * @param inFile_name input file name
 * @param outFile_name output file name
 */
int validate(std::string& inFile_name, std::string& outFile_name);

/**
 *  xfLz4Streaming class. Class containing methods for LZ4
 * compression and decompression to be executed on host side.
 */
class xfLz4Streaming {
   public:
    /**
     * @brief Compress streaming.
     *
     * @param in input byte sequence
     * @param out output byte sequence
     * @param actual_size input size
     * @param host_buffer_size host buffer size
     */
    uint64_t compressStream(uint8_t* in, uint8_t* out, uint64_t actual_size);

    /**
     * @brief Compress the input file.
     *
     * @param inFile_name input file name
     * @param outFile_name output file name
     * @param actual_size input size
     */
    uint64_t compressFile(std::string& inFile_name, std::string& outFile_name, uint64_t actual_size, bool m_flow);

    /**
     * @brief Decompress the input file.
     *
     * @param inFile_name input file name
     * @param outFile_name output file name
     * @param actual_size input size
     */
    uint64_t decompressFile(std::string& inFile_name, std::string& outFile_name, uint64_t actual_size, bool m_flow);

    /**
    * @brief Decompress the input file full.
    *
    * @param inFile_name input file name
    * @param outFile_name output file name
    * @param input_size input size
    * @param m_flow decompress flow
    * @param enable_p2p runs p2p mode
    */
    uint32_t decompressFileFull(
        std::string& inFile_name, std::string& outFile_name, uint32_t inputSize, bool m_flow, bool enable_p2p = 0);

    /**
     * @brief Decompress streaming.
     *
     * @param in input byte sequence
     * @param out output byte sequence
     * @param actual_size input size
     * @param original_size original size
     * @param host_buffer_size host buffer size
     */
    uint64_t decompressStream(uint8_t* in, uint8_t* out, uint64_t actual_size, uint64_t original_size);

    /**
    * @brief Decompress sequential migrate full inputSize.
    *
    * @param in input byte sequence
    * @param out output byte sequence
    * @param input_size input size
    * @param m_flow decompress flow
    * @param enable_p2p runs p2p mode
    */
    uint32_t decompressFull(uint8_t* in, uint8_t* out, uint32_t inputSize, bool enable_p2p = 0);

    /**
     * @brief Get the duration of input event
     *
     * @param event event to get duration for
     */
    uint64_t getEventDurationNs(const cl::Event& event);

    /**
     * @brief Class constructor
     *
     * @param binaryFile file to be read
     * @param flow 0 for compression, 1 for decompression
     */
    xfLz4Streaming(const std::string& binaryFile, uint8_t flow, uint32_t m_block_kb);

    /**
     * @brief Class destructor.
     */
    ~xfLz4Streaming();

   private:
    /**
     * Binary flow compress/decompress
     */
    bool m_BinFlow;
    /**
     * Block Size
     */
    uint32_t m_BlockSizeInKb;
    /**
      * Switch between FPGA/Standard flows
      */
    bool m_SwitchFlow;

    cl::Device m_device;
    cl::Program* m_program;
    cl::Context* m_context;
    cl::CommandQueue* m_q;
    cl::Kernel* compress_kernel_lz4;
    cl::Kernel* compress_data_mover_kernel;
    cl::Kernel* decompress_kernel_lz4;
    cl::Kernel* decompress_data_mover_kernel;

    // Compression related
    std::vector<uint8_t, aligned_allocator<uint8_t> > h_buf_in;
    std::vector<uint8_t, aligned_allocator<uint8_t> > h_buf_out;
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_buf_decompressSize;
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_blksize;
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_compressSize;

    // Device buffers
    cl::Buffer* buffer_input;
    cl::Buffer* buffer_output;
    cl::Buffer* bufferOutputSize;
    cl::Buffer* buffer_compressed_size;
    cl::Buffer* buffer_block_size;

    // Decompression related
    std::vector<uint32_t> m_blkSize;
    std::vector<uint32_t> m_compressSize;
    std::vector<bool> m_is_compressed;

    // Kernel names
    std::string compress_kernel_name = "xilLz4CompressStream";
    std::string compress_dm_kernel_name = "xilCompDatamover";
    std::string decompress_kernel_name = "xilLz4DecompressStream";
    std::string decompress_dm_kernel_name = "xilDecompDatamover";
};

#endif // _XFCOMPRESSION_LZ4_STREAM_HPP_
