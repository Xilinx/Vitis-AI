/*
 * Copyright 2019 Xilinx, Inc.
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

/**
 * @file hash_hive.hpp
 * @brief Implementation of Hash Murmur3 which is used in Hive.
 *
 * This file is part of Vitis Database Library.
 */

#ifndef XF_DATABASE_HASH_MURMUR3_HIVE_HPP
#define XF_DATABASE_HASH_MURMUR3_HIVE_HPP

#ifndef __cplusplus
#error "Vitis Database Library only works with C++."
#endif

#include "hls_stream.h"
#include <ap_int.h>
#include "xf_database/types.hpp"

// For debug
#if !defined(__SYNTHSIS__) && __XF_DATABASE_DEBUG_HASH_MURMUR3_HIVE__ == 1
#include <iostream>
#endif

// XXX HLS currently does not support cross-compilation, and assumes the target
// to have same endianness as build host. HLS is only available on X86 machines,
// and thus the code is always little-endian.
#if !defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "HLS only supports little endian systems!"
#endif

namespace xf {
namespace database {
namespace details {

/// mix function
static inline ap_int<64> fmix64(ap_int<64> h) {
#pragma HLS inline
    h ^= (ap_uint<64>)(h) >> 33;
#if !defined(__SYNTHSIS__) && __XF_DATABASE_DEBUG_HASH_MURMUR3_HIVE__ == 1
    std::cout << "hash = " << std::hex << (ap_uint<64>)h << std::endl;
#endif
    h *= 0xff51afd7ed558ccdL;
#if !defined(__SYNTHSIS__) && __XF_DATABASE_DEBUG_HASH_MURMUR3_HIVE__ == 1
    std::cout << "hash = " << std::hex << (ap_uint<64>)h << std::endl;
#endif
    h ^= (ap_uint<64>)(h) >> 33;
#if !defined(__SYNTHSIS__) && __XF_DATABASE_DEBUG_HASH_MURMUR3_HIVE__ == 1
    std::cout << "hash = " << std::hex << (ap_uint<64>)h << std::endl;
#endif
    h *= 0xc4ceb9fe1a85ec53L;
#if !defined(__SYNTHSIS__) && __XF_DATABASE_DEBUG_HASH_MURMUR3_HIVE__ == 1
    std::cout << "hash = " << std::hex << (ap_uint<64>)h << std::endl;
#endif
    h ^= (ap_uint<64>)(h) >> 33;
#if !defined(__SYNTHSIS__) && __XF_DATABASE_DEBUG_HASH_MURMUR3_HIVE__ == 1
    std::cout << "hash = " << std::hex << (ap_uint<64>)h << std::endl;
#endif
    return h;
}

/// brief Generate 64-bit output hash value for the 64-bit input key.
static inline ap_int<64> hash_hive(ap_int<64> key) {
#pragma HLS inline

    // constants in hash64
    const ap_int<64> DEFAULT_SEED = 104729;
    const ap_int<64> C1 = 0x87c37b91114253d5L;
    const ap_int<64> C2 = 0x4cf5ad432745937fL;
    const int R1 = 31;
    const int R2 = 27;
    const int M = 5;
    const int N1 = 0x52dce729;
    const ap_int<64> LONG_BYTES = 8;

    // initialize hash with default seed
    ap_int<64> hash = DEFAULT_SEED;
    // reverse byte of key
    ap_int<64> k;
    k.range(7, 0) = key.range(63, 56);
    k.range(15, 8) = key.range(55, 48);
    k.range(23, 16) = key.range(47, 40);
    k.range(31, 24) = key.range(39, 32);
    k.range(39, 32) = key.range(31, 24);
    k.range(47, 40) = key.range(23, 16);
    k.range(55, 48) = key.range(15, 8);
    k.range(63, 56) = key.range(7, 0);
#if !defined(__SYNTHSIS__) && __XF_DATABASE_DEBUG_HASH_MURMUR3_HIVE__ == 1
    std::cout << "k = " << std::hex << (ap_uint<64>)k << std::endl;
#endif

    // mix functions
    k *= C1;
#if !defined(__SYNTHSIS__) && __XF_DATABASE_DEBUG_HASH_MURMUR3_HIVE__ == 1
    std::cout << "k = " << std::hex << (ap_uint<64>)k << std::endl;
#endif
    // Long.rotateLeft(k, R1)
    k = ((ap_uint<64>)(k) << R1) | (ap_uint<64>(k) >> (64 - R1));
#if !defined(__SYNTHSIS__) && __XF_DATABASE_DEBUG_HASH_MURMUR3_HIVE__ == 1
    std::cout << "k = " << std::hex << (ap_uint<64>)k << std::endl;
#endif
    k *= C2;
#if !defined(__SYNTHSIS__) && __XF_DATABASE_DEBUG_HASH_MURMUR3_HIVE__ == 1
    std::cout << "k = " << std::hex << (ap_uint<64>)k << std::endl;
#endif
    hash ^= k;
#if !defined(__SYNTHSIS__) && __XF_DATABASE_DEBUG_HASH_MURMUR3_HIVE__ == 1
    std::cout << "hash = " << std::hex << (ap_uint<64>)hash << std::endl;
#endif
    hash = (ap_int<64>)(((ap_uint<64>)(hash) << R2) | ((ap_uint<64>)(hash) >> (64 - R2))) * M + N1;
#if !defined(__SYNTHSIS__) && __XF_DATABASE_DEBUG_HASH_MURMUR3_HIVE__ == 1
    std::cout << "hash = " << std::hex << (ap_uint<64>)hash << std::endl;
#endif

    // finalization
    hash ^= LONG_BYTES;
#if !defined(__SYNTHSIS__) && __XF_DATABASE_DEBUG_HASH_MURMUR3_HIVE__ == 1
    std::cout << "hash = " << std::hex << (ap_uint<64>)hash << std::endl;
#endif
    hash = fmix64(hash);

    // output 64-bit hash value
    return hash;

} // hash_hive

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/// @brief Murmur3 algorithm in 64-bit version.
/// @param keyStrm Message being hashed.
/// @param hashStrm The hash value.
static void hashMurmur3Hive(hls::stream<ap_int<64> >& keyStrm, hls::stream<ap_int<64> >& hashStrm) {
    ap_int<64> key = keyStrm.read();
    ap_int<64> hash = details::hash_hive(key);
    hashStrm.write(hash);
}

} // namespace database
} // namespace xf

//-----------------------------------------------------------------------------
#undef _XF_DATABASE_VOID_CAST
#undef _XF_DATABASE_PRINT
#endif // XF_DATABASE_HASH_MURMUR3_HIVE_HPP
