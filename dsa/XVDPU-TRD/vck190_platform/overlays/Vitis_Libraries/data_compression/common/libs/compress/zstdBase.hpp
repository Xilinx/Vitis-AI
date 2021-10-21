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
 * @file zstdBase.hpp
 * @brief Header for ZSTD Base functionality
 *
 * This file is part of Vitis Data Compression Library host code for zstd compression.
 */

#ifndef _XFCOMPRESSION_ZSTD_BASE_HPP_
#define _XFCOMPRESSION_ZSTD_BASE_HPP_

#include <iomanip>
#include "xcl2.hpp"
#include "compressBase.hpp"

/**
 *  zstdBase class. Class containing methods for ZSTD
 *  decompression to be executed on host side.
 */
class zstdBase : public compressBase {
   public:
    /**
     * @brief Decompress sequential.
     *
     * @param in input byte sequence
     * @param out output byte sequence
     * @param actual_size original size
     */
    virtual uint64_t compressEngine(uint8_t* in, uint8_t* out, size_t actual_size) = 0;
    /**
     * @brief Decompress Engine.
     *
     * @param in input byte sequence
     * @param out output byte sequence
     * @param compressed_size input size
     *
     * */
    virtual uint64_t decompressEngine(uint8_t* in, uint8_t* out, size_t compressed_size) = 0;

    ~zstdBase() = default;
};
#endif // _XFCOMPRESSION_ZSTD_BASE_HPP_
