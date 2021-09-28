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
#ifndef _XFCOMPRESSION_BLOCK_PACKER_HPP_
#define _XFCOMPRESSION_BLOCK_PACKER_HPP_

/**
 * @file block_packer.hpp
 * @brief Header for module used in packer kernels.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "hls_stream.h"

#include <ap_int.h>
#include <assert.h>
#include <stdint.h>

#include "block_packer.hpp"

namespace xf {
namespace compression {

/**
 * @brief Compression Packer module packs the compressed data.
 *
 * @tparam DATAWIDTH input data width
 *
 * @param inStream input data
 * @param outStream output data
 * @param inStreamSize size of the data in input stream
 * @param outStreamEos output end of stream
 * @param outCompressedSize total compressed packed data size
 */
template <int DATAWIDTH = 512>
void blockPacker(hls::stream<ap_uint<DATAWIDTH> >& inStream,
                 hls::stream<uint32_t>& inStreamSize,
                 hls::stream<ap_uint<DATAWIDTH> >& outStream,
                 hls::stream<bool>& outStreamEos,
                 hls::stream<uint32_t>& outCompressedSize) {
    // Main Module Starts
    // Local Variable Declaration
    const int c_byteSize = 8;
    uint32_t lbuf_idx = 0;
    uint32_t index = 0;
    uint32_t prodLbuf = 0;
    uint32_t endSizeCnt = 0;

    // Parallel Byte Variable
    uint32_t c_parallelByte = DATAWIDTH / c_byteSize;

    // Local Buffer for shift operation
    ap_uint<DATAWIDTH* 2> lcl_buffer = 0;

// Loop for processing each compressed size block
size_stream_loop:
    for (uint32_t size = inStreamSize.read(); size != 0; size = inStreamSize.read()) {
        // Increment the size variable.
        endSizeCnt += size;

        // Calculation for byte processing
        uint32_t leftBytes = size % c_parallelByte;
        uint32_t alignedBytes = size - leftBytes;

        // One-time multiplication calculation for index
        prodLbuf = lbuf_idx * c_byteSize;

    loop_aligned:
        for (uint32_t i = 0; i < alignedBytes; i += c_parallelByte) {
#pragma HLS PIPELINE II = 1
            // Reading Input Data
            lcl_buffer.range(prodLbuf + DATAWIDTH - 1, prodLbuf) = inStream.read();

            // Writing output into memory
            outStream << lcl_buffer.range(DATAWIDTH - 1, 0);
            outStreamEos << 0;
            // Shifting by Global datawidth
            lcl_buffer >>= DATAWIDTH;
        }

        // Left bytes from each block
        if (leftBytes) {
            lcl_buffer.range(prodLbuf + DATAWIDTH - 1, prodLbuf) = inStream.read();
            lbuf_idx += leftBytes;
        }

        // Check for PARALLEL BYTE data and write to output stream
        if (lbuf_idx >= c_parallelByte) {
            outStream << lcl_buffer.range(DATAWIDTH - 1, 0);
            outStreamEos << 0;
            lcl_buffer >>= DATAWIDTH;
            lbuf_idx -= c_parallelByte;
        }
    }

    // Left Over Data Handling
    if (lbuf_idx) {
        lcl_buffer.range(DATAWIDTH - 1, (lbuf_idx * c_byteSize)) = 0;
        outStream << lcl_buffer.range(DATAWIDTH - 1, 0);
        outStreamEos << 0;
        lbuf_idx = 0;
    }

    // Dummy data
    outStreamEos << 1;
    outStream << 0;

    // Total Size
    outCompressedSize << endSizeCnt;
}

} // namespace compression
} // namespace xf
#endif // _XFCOMPRESSION_BLOCK_PACKER_HPP_
