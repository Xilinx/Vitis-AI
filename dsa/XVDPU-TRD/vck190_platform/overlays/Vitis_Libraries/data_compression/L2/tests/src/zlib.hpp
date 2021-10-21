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
#ifndef _XFCOMPRESSION_ZLIB_HPP_
#define _XFCOMPRESSION_ZLIB_HPP_

#pragma once

#include "xcl2.hpp"
#include "zlib_specs.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <string>
#include <thread>
#include <time.h>
#include <vector>

#pragma once

#define PARALLEL_ENGINES 8
#define C_COMPUTE_UNIT 1
#define D_COMPUTE_UNIT 1
#define H_COMPUTE_UNIT 1
#define MAX_CCOMP_UNITS C_COMPUTE_UNIT
#define MAX_DDCOMP_UNITS D_COMPUTE_UNIT

#define MIN_BLOCK_SIZE 1024
// Default block size
#define BLOCK_SIZE_IN_KB 1024

// zlib maximum cr
#define MAX_CR 20

// Input and output buffer size
#define INPUT_BUFFER_SIZE (16 * 1024 * 1024)

#define OUTPUT_BUFFER_SIZE (32 * 1024 * 1024)

// buffer count for data in
#define DIN_BUFFERCOUNT 2
#define DOUT_BUFFERCOUNT 4

// Maximum host buffer used to operate
// per kernel invocation
#define HOST_BUFFER_SIZE (PARALLEL_ENGINES * BLOCK_SIZE_IN_KB * 1024)

// Value below is used to associate with
// Overlapped buffers, ideally overlapped
// execution requires 2 resources per invocation
#define OVERLAP_BUF_COUNT 2

// Maximum number of blocks based on host buffer size
#define MAX_NUMBER_BLOCKS (HOST_BUFFER_SIZE / (BLOCK_SIZE_IN_KB * 1024))

enum d_type { DYNAMIC = 0, FIXED = 1, FULL = 2 };

void error_message(const std::string& val);

int validate(std::string& inFile_name, std::string& outFile_name);

uint64_t get_file_size(std::ifstream& file);

class xil_zlib {
   public:
    int init(const std::string& binaryFile, uint8_t flow, uint8_t d_type);
    int release();
    uint32_t compress(uint8_t* in, uint8_t* out, uint64_t actual_size, uint32_t host_buffer_size);
    uint32_t compressFull(uint8_t* in, uint8_t* out, uint64_t actual_size);
    uint32_t decompress(uint8_t* in, uint8_t* out, uint32_t actual_size, int cu_run = 0, bool enable_p2p = 0);
    uint32_t decompressSeq(uint8_t* in, uint8_t* out, uint32_t actual_size, int cu_run);
    uint32_t compress_file(std::string& inFile_name, std::string& outFile_name, uint64_t input_size);
    uint32_t decompress_file(
        std::string& inFile_name, std::string& outFile_name, uint64_t input_size, int cu_run = 0, bool enable_p2p = 0);
    uint64_t get_event_duration_ns(const cl::Event& event);
    // Binary flow compress/decompress
    xil_zlib(const std::string& binaryFile,
             uint8_t flow,
             uint8_t max_cr = MAX_CR,
             uint8_t device_id = 0,
             uint8_t d_type = DYNAMIC,
             bool is_dec_overlap = false);
    ~xil_zlib();

   private:
    void _enqueue_writes(uint32_t bufSize, uint8_t* in, uint32_t inputSize, bool enable_p2p);
    void _enqueue_reads(uint32_t bufSize, uint8_t* out, uint32_t* decompSize, uint32_t max_outbuf);

    uint8_t m_BinFlow;
    uint8_t m_deviceid;
    const uint32_t m_minfilesize = 200;
    bool m_useOverlapDec = false;

    // Max cr
    uint8_t m_max_cr;

    cl::Program* m_program;
    cl::Context* m_context;
    cl::CommandQueue* m_q[C_COMPUTE_UNIT * OVERLAP_BUF_COUNT];
#ifndef FREE_RUNNING_KERNEL
    cl::CommandQueue* m_q_dec;
#endif
    cl::CommandQueue* m_q_rd;
    cl::CommandQueue* m_q_rdd;
    cl::CommandQueue* m_q_wr;
    cl::CommandQueue* m_q_wrd;

    // Compress Kernel Declaration
    cl::Kernel* compress_kernel;
    cl::Kernel* huffman_kernel;
    cl::Kernel* treegen_kernel;

    // Decompress Kernel Declaration
    cl::Kernel* decompress_kernel;
    cl::Kernel* data_writer_kernel;
    cl::Kernel* data_reader_kernel;

    // Compression related
    std::vector<uint8_t, aligned_allocator<uint8_t> > h_buf_in[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    std::vector<uint8_t, aligned_allocator<uint8_t> > h_buf_out[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    std::vector<uint8_t, aligned_allocator<uint8_t> > h_buf_zlibout[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_blksize[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_compressSize[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];

    // Decompression Related
    std::vector<uint8_t, aligned_allocator<uint8_t> > h_dbuf_in[DIN_BUFFERCOUNT];
    std::vector<uint8_t, aligned_allocator<uint8_t> > h_dbuf_zlibout[DOUT_BUFFERCOUNT];
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_dcompressSize[DOUT_BUFFERCOUNT];
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_dcompressStatus;

    // Buffers related to Dynamic Huffman

    // Literal & length frequency tree
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_dyn_ltree_freq[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    // Distance frequency tree
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_dyn_dtree_freq[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    // Bit Length frequency
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_dyn_bltree_freq[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];

    // Literal Codes
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_dyn_ltree_codes[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    // Distance Codes
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_dyn_dtree_codes[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    // Bit Length Codes
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_dyn_bltree_codes[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];

    // Literal Bitlength
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_dyn_ltree_blen[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    // Distance Bitlength
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_dyn_dtree_blen[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    // Bit Length Bitlength
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_dyn_bltree_blen[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];

    std::vector<uint32_t, aligned_allocator<uint32_t> > h_buff_max_codes[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];

    // Device buffers
    cl::Buffer* buffer_input[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    cl::Buffer* buffer_output[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    cl::Buffer* buffer_lz77_output[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    cl::Buffer* buffer_zlib_output[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    cl::Buffer* buffer_compress_size[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    cl::Buffer* buffer_inblk_size[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];

    cl::Buffer* buffer_dyn_ltree_freq[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    cl::Buffer* buffer_dyn_dtree_freq[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    cl::Buffer* buffer_dyn_bltree_freq[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];

    cl::Buffer* buffer_dyn_ltree_codes[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    cl::Buffer* buffer_dyn_dtree_codes[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    cl::Buffer* buffer_dyn_bltree_codes[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];

    cl::Buffer* buffer_dyn_ltree_blen[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    cl::Buffer* buffer_dyn_dtree_blen[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
    cl::Buffer* buffer_dyn_bltree_blen[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];

    cl::Buffer* buffer_max_codes[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];

    cl::Buffer* buffer_checksum_data;

    // Decompress Device Buffers
    cl::Buffer* buffer_dec_input[MAX_DDCOMP_UNITS];
    cl::Buffer* buffer_dec_zlib_output[MAX_DDCOMP_UNITS];
    cl::Buffer* buffer_dec_compress_size[MAX_DDCOMP_UNITS];

    // Kernel names
    std::vector<std::string> compress_kernel_names = {"xilLz77Compress", "xilGzipCompBlock"};
    std::vector<std::string> huffman_kernel_names = {"xilHuffmanKernel"};
    std::vector<std::string> treegen_kernel_names = {"xilTreegenKernel"};
    std::vector<std::string> decompress_kernel_names = {"xilDecompressDynamic", "xilDecompressFixed", "xilDecompress"};
    std::vector<std::string> data_writer_kernel_names = {"xilGzipMM2S"};
    std::vector<std::string> data_reader_kernel_names = {"xilGzipS2MM"};
};
#endif // _XFCOMPRESSION_ZLIB_HPP_
