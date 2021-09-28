/*
 *Copyright 2019-2021 Xilinx, Inc. All rights reserved.
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
 * @file lz4App.hpp
 * @brief lz4 realted  host functionality
 *
 * This file is part of Vitis Data Compression Library host code.
 */

#ifndef _XFCOMPRESSION_LZ4_APP_HPP_
#define _XFCOMPRESSION_LZ4_APP_HPP_
#include "compressApp.hpp"

class lz4App : public compressApp {
   public:
    /**
     * @brief Initialize lz4App content
     *
     */
    lz4App(const int argc, char** argv, bool is_seq, bool enable_profile = false);
    /**
     * @brief lz4 block size
     *
     */
    uint32_t getBlockSize(void);

   private:
    std::string m_blocksize;
};

#endif // _XFCOMPRESSION_LZ4_APP_HPP_
