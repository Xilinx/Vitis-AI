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
#ifndef _XFCOMPRESSION_LZ4_SPECS_HPP
#define _XFCOMPRESSION_LZ4_SPECS_HPP

/**
 * @file lz4_specs.hpp
 * @brief Header containing some lz4 Specification related constant parameters.
 *
 * This file is part of Vitis Data Compression Library host code for lz4 compression.
 */

namespace xf {
namespace compression {

/**
 * Below are the codes as per LZ4 standard for
 * various maximum block sizes supported.
 */
const auto BSIZE_STD_64KB = 0x40;
const auto BSIZE_STD_256KB = 0x50;
const auto BSIZE_STD_1024KB = 0x60;
const auto BSIZE_STD_4096KB = 0x70;

/**
 * Maximum block sizes supported by LZ4
 */
const auto MAX_BSIZE_64KB = (64 * 1024);
const auto MAX_BSIZE_256KB = (256 * 1024);
const auto MAX_BSIZE_1024KB = (1024 * 1024);
const auto MAX_BSIZE_4096KB = (4096 * 1024);

const auto MAGIC_HEADER_SIZE = 4;
const auto MAGIC_BYTE_1 = 4;
const auto MAGIC_BYTE_2 = 34;
const auto MAGIC_BYTE_3 = 77;
const auto MAGIC_BYTE_4 = 24;
const auto FLG_BYTE = 100;

/**
 * This value is used to set
 * uncompressed block size value.
 * 4th byte is always set to below
 * and placed as uncompressed byte
 */
const auto NO_COMPRESS_BIT = 128;

/**
 * In case of uncompressed block
 * Values below are used to set
 * 3rd byte to following values
 * w.r.t various maximum block sizes
 * supported by standard
 */
const auto BSIZE_NCOMP_64 = 1;
const auto BSIZE_NCOMP_256 = 4;
const auto BSIZE_NCOMP_1024 = 16;
const auto BSIZE_NCOMP_4096 = 64;

} // end namespace compression
} // end namespace xf

#endif // _XFCOMPRESSION_LZ4_SPECS_HPP
