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
#include "lz4Base.hpp"
#include "xxhash.h"
#include <iostream>
#include <cstring>

using namespace xf::compression;

uint8_t lz4Base::writeHeader(uint8_t* out) {
    size_t input_size = m_InputSize;
    uint8_t fileIdx = 0;
    (out[fileIdx++]) = MAGIC_BYTE_1;
    (out[fileIdx++]) = MAGIC_BYTE_2;
    (out[fileIdx++]) = MAGIC_BYTE_3;
    (out[fileIdx++]) = MAGIC_BYTE_4;
    m_frameByteCount += 4;

    // FLG & BD bytes
    // --no-frame-crc flow
    // --content-size
    (out[fileIdx++]) = FLG_BYTE;
    m_frameByteCount += 1;

    size_t block_size_header = 0;
    // Default value 64K
    switch (m_BlockSizeInKb) {
        case 64:
            (out[fileIdx++]) = BSIZE_STD_64KB;
            block_size_header = BSIZE_STD_64KB;
            break;
        case 256:
            (out[fileIdx++]) = BSIZE_STD_256KB;
            block_size_header = BSIZE_STD_256KB;
            break;
        case 1024:
            (out[fileIdx++]) = BSIZE_STD_1024KB;
            block_size_header = BSIZE_STD_1024KB;
            break;
        case 4096:
            (out[fileIdx++]) = BSIZE_STD_4096KB;
            block_size_header = BSIZE_STD_4096KB;
            break;
        default:
            std::cout << "Invalid Block Size" << std::endl;
            break;
    }
    m_frameByteCount += 1;

    m_HostBufferSize = HOST_BUFFER_SIZE;

    if ((m_BlockSizeInKb * 1024) > input_size) m_HostBufferSize = m_BlockSizeInKb * 1024;

    uint8_t temp_buff[10] = {FLG_BYTE,
                             (uint8_t)block_size_header,
                             (uint8_t)input_size,
                             (uint8_t)(input_size >> 8),
                             (uint8_t)(input_size >> 16),
                             (uint8_t)(input_size >> 24),
                             (uint8_t)(input_size >> 32),
                             (uint8_t)(input_size >> 40),
                             (uint8_t)(input_size >> 48),
                             (uint8_t)(input_size >> 56)};

    if (m_addContentSize) {
        memcpy(&out[fileIdx], &temp_buff[2], 8);
        fileIdx += 8;
        m_frameByteCount += 8;
    }

    // xxhash is used to calculate hash value
    uint32_t xxh = (m_addContentSize) ? XXH32(temp_buff, 10, 0) : XXH32(temp_buff, 2, 0);

    //  Header CRC
    out[fileIdx++] = (uint8_t)(xxh >> 8);
    m_frameByteCount += 1;

    return fileIdx;
}

void lz4Base::writeFooter(uint8_t* in, uint8_t* out) {
    uint32_t* zero_ptr = 0;
    memcpy(out, &zero_ptr, 4);
    out += 4;
    m_frameByteCount += 4;

    size_t input_size = m_InputSize;
    // xxhash is used to calculate content checksum value
    uint32_t xxh = XXH32(in, input_size, 0);
    memcpy(out, &xxh, 4);
    out += 4;
    m_frameByteCount += 4;
}

uint8_t lz4Base::readHeader(uint8_t* in) {
    uint8_t fileIdx = 0;
    // Read magic header 4 bytes
    char c = 0;
    int magic_hdr[] = {MAGIC_BYTE_1, MAGIC_BYTE_2, MAGIC_BYTE_3, MAGIC_BYTE_4};
    for (uint32_t i = 0; i < MAGIC_HEADER_SIZE; i++) {
        // inFile.get(v);
        c = in[fileIdx++];
        if (int(c) == magic_hdr[i])
            continue;
        else {
            std::cerr << "Problem with magic header " << c << " " << i << std::endl;
            exit(1);
        }
    }

    // FLG Byte
    c = in[fileIdx++];

    // Check if block size is 64 KB
    c = in[fileIdx++];

    switch (c) {
        case BSIZE_STD_64KB:
            m_BlockSizeInKb = 64;
            break;
        case BSIZE_STD_256KB:
            m_BlockSizeInKb = 256;
            break;
        case BSIZE_STD_1024KB:
            m_BlockSizeInKb = 1024;
            break;
        case BSIZE_STD_4096KB:
            m_BlockSizeInKb = 4096;
            break;
        default:
            std::cout << "Invalid Block Size" << std::endl;
            break;
    }

    if (FLG_BYTE == 104) {
        // Original size
        size_t original_size = 0;

        memcpy(&original_size, &in[fileIdx], 8);

        // file size(8)
        fileIdx += 8;
    }

    // Header Checksum
    c = in[fileIdx++];

    m_HostBufferSize = (m_BlockSizeInKb * 1024) * MAX_NUMBER_BLOCKS;

    return fileIdx;
}

uint8_t lz4Base::get_bsize(uint32_t c_input_size) {
    switch (c_input_size) {
        case MAX_BSIZE_64KB:
            return BSIZE_NCOMP_64;
            break;
        case MAX_BSIZE_256KB:
            return BSIZE_NCOMP_256;
            break;
        case MAX_BSIZE_1024KB:
            return BSIZE_NCOMP_1024;
            break;
        case MAX_BSIZE_4096KB:
            return BSIZE_NCOMP_4096;
            break;
        default:
            return BSIZE_NCOMP_64;
            break;
    }
}
