/*
 * Copyright 2019-2021 Xilinx, Inc.
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

#ifndef _XFCOMPRESSION_LZ4_P2P_DEC_HPP_
#define _XFCOMPRESSION_LZ4_P2P_DEC_HPP_

#include <iomanip>
#include "xcl2.hpp"
#include <fcntl.h>
#include <unistd.h>

// Maximum host buffer used to operate
// per kernel invocation
#define HOST_BUFFER_SIZE (2 * 1024 * 1024)

// Default block size
#define BLOCK_SIZE_IN_KB 64

#define KB 1024

// Max Input buffer Size
#define MAX_IN_BUFFER_SIZE (1024 * 1024 * 1024)

// Max Input Buffer Partitions
#define MAX_IN_BUFFER_PARTITION MAX_IN_BUFFER_SIZE / HOST_BUFFER_SIZE

// Maximum number of blocks based on host buffer size
#define MAX_NUMBER_BLOCKS (HOST_BUFFER_SIZE / (BLOCK_SIZE_IN_KB * 1024))

class xfLz4 {
   public:
    xfLz4(const std::string& binaryFile);
    void decompress_in_line_multiple_files(const std::vector<std::string>& inFileList,
                                           std::vector<int>& fd_p2p_vec,
                                           std::vector<char*>& outVec,
                                           std::vector<uint64_t>& orgSizeVec,
                                           std::vector<uint32_t>& inSizeVec4k,
                                           bool enable_p2p,
                                           uint8_t maxCR);
    ~xfLz4();
    static uint64_t get_file_size(std::ifstream& file) {
        file.seekg(0, file.end);
        uint64_t file_size = file.tellg();
        file.seekg(0, file.beg);
        return file_size;
    }

   private:
    cl_program m_program;
    cl_context m_context;
    cl_device_id m_device;
    cl_command_queue ooo_q;

    std::vector<std::string> datamover_kernel_names = {"xilDecompDatamover"};

    std::vector<std::string> decompress_kernel_names = {"xilLz4DecompressStream"};
};

#endif // _XFCOMPRESSION_LZ4_P2P_DEC_HPP_
