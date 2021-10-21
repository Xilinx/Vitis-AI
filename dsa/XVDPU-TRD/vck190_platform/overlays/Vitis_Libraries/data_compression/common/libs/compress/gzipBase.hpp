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
 * @file gzipBase.hpp
 * @brief Header for GZIP Base functionality
 *
 * This file is part of Vitis Data Compression Library host code for gzip compression.
 */

#ifndef _XFCOMPRESSION_GZIP_BASE_HPP_
#define _XFCOMPRESSION_GZIP_BASE_HPP_

#include <cassert>
#include <iomanip>
#include <sys/stat.h>
#include "xcl2.hpp"
#include "compressBase.hpp"
#include <algorithm>
#include <iterator>

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

// Enums for
enum comp_decom_flows { BOTH, COMP_ONLY, DECOMP_ONLY };
enum d_type { DYNAMIC = 0, FIXED = 1, FULL = 2 };
enum design_flow { XILINX_GZIP, XILINX_ZLIB };

// Kernel names
const std::vector<std::string> compress_kernel_names = {"xilLz77Compress", "xilGzipComp", "xilGzipCompBlock"};
const std::vector<std::string> stream_decompress_kernel_name = {"xilDecompressDynamic", "xilDecompressFixed",
                                                                "xilDecompress"};
const std::string data_writer_kernel_name = "xilGzipMM2S";
const std::string data_reader_kernel_name = "xilGzipS2MM";
const std::string datamover_kernel_name = "xilDataMover";

/**
 *  gzipBase class. Class containing methods for GZIP
 * compression and decompression to be executed on host side.
 */
class gzipBase : public compressBase {
   private:
    uint32_t m_BlockSizeInKb = 64;
    uint32_t m_HostBufferSize;
    uint64_t m_InputSize;

   protected:
    bool m_isZlib = false;
    std::string m_inFileName;
    bool m_isSeq = false;
    bool m_freeRunKernel = false;
    int m_level = 1;
    int m_strategy = 0;
    int m_windowbits = 15;
    std::chrono::system_clock::time_point total_start;
    std::chrono::system_clock::time_point total_end;

   public:
    enum d_type { DYNAMIC = 0, FIXED = 1, FULL = 2 };
    enum design_flow { XILINX_GZIP, XILINX_ZLIB };

    bool is_freeRunKernel(void) { return m_freeRunKernel; };

    /**
      * @brief Enable Profile
      * True: Prints the end to end throughput.
      * False: Prints Kernel throughput.
      */
    bool m_enableProfile;

    gzipBase(bool enable_profile) : m_enableProfile(enable_profile) {}

    /**
         * @brief Xilinx Compress
         *
         * @param in input byte sequence
         * @param out output byte sequence
         * @param actual_size input size
         */

    uint64_t xilCompress(uint8_t* in, uint8_t* out, uint64_t input_size) override;

    /**
         * @brief Xilinx Decompress
         *
         * @param in input byte sequence
         * @param out output byte sequence
         * @param compressed size
         */

    uint64_t xilDecompress(uint8_t* in, uint8_t* out, uint64_t input_size) override;

    /**
         * @brief Header Writer
         *
         * @param compress out stream
         */

    size_t writeHeader(uint8_t* out);
    size_t writeFooter(uint8_t* out, size_t compressSize, uint32_t checksum = 0);

    /**
         * @brief Header Reader
         *
         * @param Compress stream input header read
         */

    bool readHeader(uint8_t* in);

    /**
     * @brief Compress Engine
     *
     * @param in input byte sequence
     * @param out output byte sequence
     * @param actual_size input size
     * @param host_buffer_size host buffer size
     */
    virtual uint64_t compressEngineOverlap(uint8_t* in,
                                           uint8_t* out,
                                           uint64_t actual_size,
                                           int cu,
                                           int level,
                                           int strategy,
                                           int window_bits,
                                           uint32_t* checksum) = 0;
    virtual uint64_t compressEngineSeq(uint8_t* in,
                                       uint8_t* out,
                                       uint64_t actual_size,
                                       int level,
                                       int strategy,
                                       int window_bits,
                                       uint32_t* checksum) = 0;

    /**
     * @brief Decompress Engine.
     *
     * @param in input byte sequence
     * @param out output byte sequence
     * @param actual_size input size
     * @param original_size original size
     * @param host_buffer_size host buffer size
     */
    virtual uint64_t decompressEngine(
        uint8_t* in, uint8_t* out, uint64_t actual_size, uint64_t max_outbuf_size, int cu = 0) = 0;
    virtual uint64_t decompressEngineSeq(
        uint8_t* in, uint8_t* out, uint64_t actual_size, uint64_t max_outbuf_size, int cu = 0) = 0;
    gzipBase(){};
};
#endif // _XFCOMPRESSION_GZIP_BASE_HPP_
