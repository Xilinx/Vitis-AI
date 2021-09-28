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
 * @file snappyBase.hpp
 * @brief Header for SNAPPY Base functionality
 *
 * This file is part of Vitis Data Compression Library host code for snappy compression.
 */

#ifndef _XFCOMPRESSION_SNAPPY_BASE_HPP_
#define _XFCOMPRESSION_SNAPPY_BASE_HPP_

#include <cassert>
#include <iomanip>
#include "xcl2.hpp"
#include "compressBase.hpp"

#ifndef ENABLE_P2P
#define ENABLE_P2P 0
#endif
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

const int RESIDUE_4K = 4096;

/**
 *  snappyBase class. Class containing methods for SNAPPY
 * compression and decompression to be executed on host side.
 */
class snappyBase : public compressBase {
   public:
    /**
      * @brief Header Writer
      *
      * @param compress out stream
      */

    uint8_t writeHeader(uint8_t* out);

    /**
      * @brief Header Reader
      *
      * @param Compress stream input header read
      */

    uint8_t readHeader(uint8_t* in);

   protected:
    uint32_t m_BlockSizeInKb = 64;
    uint32_t m_HostBufferSize;
    size_t m_InputSize;
};
#endif // _XFCOMPRESSION_SNAPPY_BASE_HPP_
