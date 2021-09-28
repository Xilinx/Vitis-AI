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
 * @file hash_murmur3.hpp
 * @brief Murmur3 hash function implementation.
 *
 * This file is part of Vitis Database Library.
 */

#ifndef XF_DATABASE_HASH_MURMUR3_H
#define XF_DATABASE_HASH_MURMUR3_H

#ifndef __cplusplus
#error "Vitis Database Library only works with C++."
#endif

#include "hls_stream.h"
// for wide input&output
#include <ap_int.h>
#include "xf_database/types.hpp"

// For debug
#include <cstdio>
#define _XF_DATABASE_VOID_CAST static_cast<void>
// XXX toggle here to debug this file
#if 0
#define _XF_DATABASE_PRINT(msg...) \
    do {                           \
        printf(msg);               \
    } while (0)
#else
#define _XF_DATABASE_PRINT(msg...) (_XF_DATABASE_VOID_CAST(0))
#endif

//-----------------------------------------------------------------------------
#include <stdint.h>

// XXX HLS currently does not support cross-compilation, and assumes the target
// to have same endianness as build host. HLS is only available on X86 machines,
// and thus the code is always little-endian.
#if !defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "HLS only supports little endian systems!"
#endif

namespace xf {
namespace database {
namespace details {
//-------------------------------------------------/#pragma HLS
// INLINE----------------------------

/// brief Generate 32bit output hash value for the input key.
template <int W, int H>
inline void hashmurmur3_strm(hls::stream<ap_uint<W> >& key_strm, hls::stream<ap_uint<H> >& hash_strm) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II = 1
#endif
    const int nblocks = W / H;

    // keyBlen is the BYTE len of the key.
    const ap_uint<H> keyBlen = W / 8;
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;
    const uint32_t c3 = 0xe6546b64;
    const uint32_t c4 = 0x85ebca6b;
    const uint32_t c5 = 0xc2b2ae35;
    //----------
    // body

    // hash is the output/return value. use magic word(seed) to initial the output
    ap_uint<H> hash = 13; // seed;

    ap_uint<32> kt;   // temp32
    ap_uint<W> key_t; // temp512
    key_t = key_strm.read();

LOOP_MURMUR3_MAIN:
    for (int j = 0; j < nblocks; ++j) {
        kt = key_t(32 * j + 31, 32 * j);
        kt *= c1;
        kt = (kt << 15) | (kt >> (32 - 15)); // ROTL32(kt,15);//
        kt *= c2;

        hash ^= kt;
        hash = (hash << 13) | (hash >> (32 - 13)); // ROTL32(hash,13);//
        hash = hash * 5 + c3;
    }

    // finalization
    hash ^= keyBlen;
    hash ^= hash >> 16;
    hash *= c4;
    hash ^= hash >> 13;
    hash *= c5;
    hash ^= hash >> 16;

    /// output hash stream width in 32bit.
    hash_strm.write(hash);
} // murmur3

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/// @brief murmur3 algorithm.
/// @tparam W the bit width of ap_uint type for input message stream.
/// @tparam h the bit width of ap_uint type for output hash stream.
/// @param key_strm the message being hashed.
/// @param hash_strm the result.
template <int W, int H>
void hashMurmur3(hls::stream<ap_uint<W> >& key_strm, hls::stream<ap_uint<H> >& hash_strm) {
    details::hashmurmur3_strm(key_strm, hash_strm);
}

} // namespace database
} // namespace xf

//-----------------------------------------------------------------------------
#undef _XF_DATABASE_VOID_CAST
#undef _XF_DATABASE_PRINT
#endif // XF_DATABASE_HASH_MURMUR3_H
