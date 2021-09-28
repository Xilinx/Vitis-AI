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
#include "gzipBase.hpp"
constexpr auto DEFLATE_METHOD = 8;

size_t gzipBase::writeHeader(uint8_t* out) {
    size_t outIdx = 0;
    if (m_isZlib) {
        // Compression method
        uint8_t CM = DEFLATE_METHOD;

        // Compression Window information
        uint8_t CINFO = m_windowbits - 8;

        // Create CMF header
        uint16_t header = (CINFO << 4);
        header |= CM;
        header <<= 8;

        if (m_level < 2 || m_strategy > 2)
            m_level = 0;
        else if (m_level < 6)
            m_level = 1;
        else if (m_level == 6)
            m_level = 2;
        else
            m_level = 3;

        // CreatE FLG header based on level
        // Strategy information
        header |= (m_level << 6);

        // Ensure that Header (CMF + FLG) is
        // divisible by 31
        header += 31 - (header % 31);

        out[outIdx] = (uint8_t)(header >> 8);
        out[outIdx + 1] = (uint8_t)(header);
        outIdx += 2;
    } else {
        long int magic_headers = 0x0000000008088B1F;
        std::memcpy(out + outIdx, &magic_headers, 8);
        outIdx += 8;

        long int osheader = 0x00780300;
        std::memcpy(out + outIdx, &osheader, 4);
        outIdx += 4;
    }

    return outIdx;
}

size_t gzipBase::writeFooter(uint8_t* out, size_t compressSize, uint32_t checksum) {
    size_t outIdx = compressSize;

    outIdx = compressSize;

    if (m_isZlib) {
        out[outIdx++] = checksum >> 24;
        out[outIdx++] = checksum >> 16;
        out[outIdx++] = checksum >> 8;
        out[outIdx++] = checksum;
    } else {
        struct stat istat;
        stat(m_inFileName.c_str(), &istat);
        unsigned long ifile_size = istat.st_size;
        out[outIdx++] = checksum;
        out[outIdx++] = checksum >> 8;
        out[outIdx++] = checksum >> 16;
        out[outIdx++] = checksum >> 24;

        out[outIdx++] = ifile_size;
        out[outIdx++] = ifile_size >> 8;
        out[outIdx++] = ifile_size >> 16;
        out[outIdx++] = ifile_size >> 24;
    }
    return outIdx;
}

uint64_t gzipBase::xilCompress(uint8_t* in, uint8_t* out, uint64_t input_size) {
    m_InputSize = input_size;
    size_t outIdx = 0;

    // GZip Header
    outIdx = (this->is_freeRunKernel()) ? outIdx : writeHeader(out);

    uint64_t enbytes = 0;

    uint32_t checksum = 0;

    if (m_isSeq) {
        // GZip single cu sequential version
        enbytes = compressEngineSeq(in, out + outIdx, m_InputSize, 1, 0, 15, &checksum);
    } else {
        // GZIP multiple/single cu overlap version
        enbytes = compressEngineOverlap(in, out + outIdx, m_InputSize, 0, 1, 0, 15, &checksum);
    }

    size_t final_bytes = enbytes;

    // GZip Footer
    final_bytes = (this->is_freeRunKernel()) ? final_bytes : writeFooter(out, outIdx + enbytes, checksum);

    return final_bytes;
}

bool gzipBase::readHeader(uint8_t* in) {
    uint8_t hidx = 0;
    if (in[hidx++] == 0x1F && in[hidx++] == 0x8B) {
        // Check for magic header
        // Check if method is deflate or not
        if (in[hidx++] != 0x08) {
            std::cerr << "\n";
            std::cerr << "Deflate Header Check Fails" << std::endl;
            return 0;
        }

        // Check if the FLAG has correct value
        // Supported file name or no file name
        // 0x00: No File Name
        // 0x08: File Name
        if (in[hidx] != 0 && in[hidx] != 0x08) {
            std::cerr << "\n";
            std::cerr << "Deflate -n option check failed" << std::endl;
            return 0;
        }
        hidx++;

        // Skip time stamp bytes
        // time stamp contains 4 bytes
        hidx += 4;

        // One extra 0  ending byte
        hidx += 1;

        // Check the operating system code
        // for Unix its 3
        uint8_t oscode_in = in[hidx];
        std::vector<uint8_t> oscodes{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
        bool ochck = std::find(oscodes.cbegin(), oscodes.cend(), oscode_in) == oscodes.cend();
        if (ochck) {
            std::cerr << "\n";
            std::cerr << "GZip header mismatch: OS code is unknown" << std::endl;
            return 0;
        }
    } else {
        hidx = 0;
        // ZLIB Header Checks
        // CMF
        // FLG
        uint8_t cmf = 0x78;
        // 0x01: Fast Mode
        // 0x5E: 1 to 5 levels
        // 0x9C: Default compression: level 6
        // 0xDA: High compression
        std::vector<uint8_t> zlib_flags{0x01, 0x5E, 0x9C, 0xDA};
        if (in[hidx++] == cmf) {
            uint8_t flg = in[hidx];
            bool hchck = std::find(zlib_flags.cbegin(), zlib_flags.cend(), flg) == zlib_flags.cend();
            if (hchck) {
                std::cerr << "\n";
                std::cerr << "Header check fails" << std::endl;
                return 0;
            }
        } else {
            std::cerr << "\n";
            std::cerr << "Zlib Header mismatch" << std::endl;
            return 0;
        }
    }
    return true;
}

uint64_t gzipBase::xilDecompress(uint8_t* in, uint8_t* out, uint64_t input_size) {
    bool hcheck = readHeader(in);
    if (!hcheck) {
        std::cerr << "Header Check Failed" << std::endl;
        return 0;
    }
    uint64_t debytes;

    if (m_isSeq) {
        // Decompression Engine multiple cus.
        debytes = decompressEngineSeq(in, out, input_size, input_size * m_maxCR);
    } else {
        // Decompression Engine multiple cus.
        debytes = decompressEngine(in, out, input_size, input_size * m_maxCR);
    }
    return debytes;
}
