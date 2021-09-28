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
 * @file lz4Base.hpp
 * @brief Header for LZ4 Base functionality
 *
 * This file is part of Vitis Data Compression Library host code for lz4 compression.
 */

#ifndef _XFCOMPRESSION_LZ4_BASE_HPP_
#define _XFCOMPRESSION_LZ4_BASE_HPP_

#include <cassert>
#include <iomanip>
#include <cstring>
#include <vector>
#include "lz4_specs.hpp"
#include "compressBase.hpp"

/**
 *
 *  Maximum host buffer used to operate per kernel invocation
 */
const auto HOST_BUFFER_SIZE = (32 * 1024 * 1024);

/*
 * Default block size
 *
 */
const auto BLOCK_SIZE_IN_KB = 64;

/**
 * Maximum number of blocks based on host buffer size
 *
 */
const auto MAX_NUMBER_BLOCKS = (HOST_BUFFER_SIZE / (BLOCK_SIZE_IN_KB * 1024));

// Kernel names
const std::vector<std::string> compress_kernel_names = {"xilLz4Compress", "xilLz4CompressStream"};
const std::vector<std::string> decompress_kernel_names = {"xilLz4Decompress", "xilLz4DecompressStream"};

const std::string compress_dm_kernel_name = "xilCompDatamover";
const std::string decompress_dm_kernel_name = "xilDecompDatamover";

/**
 *  lz4Base class. Class containing methods for LZ4
 * compression and decompression to be executed on host side.
 */

class lz4Base : public compressBase {
    uint64_t xilCompress(uint8_t* in, uint8_t* out, size_t input_size) override { return input_size; }

    uint64_t xilDecompress(uint8_t* in, uint8_t* out, size_t input_size) override { return input_size; }

   public:
    lz4Base() {}
    lz4Base(bool content_size) : m_addContentSize(content_size) {}

    /**
             * @brief Header Writer
             *
             * @param compress out stream
             */

    uint8_t writeHeader(uint8_t* out);

    /**
         * @brief Footer Writer
         *
         * @param compress out stream
         */

    void writeFooter(uint8_t* in, uint8_t* out);

    /**
         * @brief Header Reader
         *
         * @param Compress stream input header read
         */

    uint8_t readHeader(uint8_t* in);

    /**
             * @brief Header Reader
             *
             * @param Compress stream input header read
             */
    uint8_t get_bsize(uint32_t input_size);

    template <typename T>
    void writeCompressedBlock(size_t input_size,
                              uint32_t block_length,
                              uint32_t* compressSize,
                              uint8_t* out,
                              T* in,
                              T* buf_out,
                              uint64_t& outIdx,
                              uint64_t& inIdx) {
        uint32_t no_blocks = (input_size - 1) / block_length + 1;
        uint32_t idx = 0;
        for (uint32_t bIdx = 0; bIdx < no_blocks; bIdx++, idx += block_length) {
            // Default block size in bytes i.e., 64 * 1024
            uint32_t block_size = block_length;
            if (idx + block_size > input_size) {
                block_size = input_size - idx;
            }
            uint32_t compressed_size = compressSize[bIdx];
            assert(compressed_size != 0);

            int orig_block_size = input_size;
            int perc_cal = orig_block_size * 10;
            perc_cal = perc_cal / block_size;

            if (compressed_size < block_size && perc_cal >= 10) {
                memcpy(&out[outIdx], &compressed_size, 4);
                outIdx += 4;
                std::memcpy(&out[outIdx], &(buf_out[bIdx * block_length]), compressed_size);
                outIdx += compressed_size;
            } else {
                // No Compression, so copy raw data
                if (block_size == 65536) {
                    out[outIdx++] = 0;
                    out[outIdx++] = 0;
                    out[outIdx++] = 1;
                    out[outIdx++] = 128;
                } else {
                    uint8_t temp = 0;
                    temp = block_size;
                    out[outIdx++] = temp;
                    temp = block_size >> 8;
                    out[outIdx++] = temp;
                    out[outIdx++] = 0;
                    out[outIdx++] = 128;
                }
                std::memcpy(&out[outIdx], &in[inIdx + idx], block_size);
                outIdx += block_size;
            }
        }
    }

    ~lz4Base(){};

    size_t m_InputSize;

   protected:
    bool m_addContentSize;
    uint32_t m_BlockSizeInKb = 64;
    uint32_t m_HostBufferSize;
    uint8_t m_frameByteCount;
    size_t m_ActualSize;
};
#endif // _XFCOMPRESSION_LZ4_BASE_HPP_
