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
 * @file hash_lookup3.hpp
 * @brief hash-little lookup3 template function implementation.
 *
 * This hashlookup3 return a 32-bit value or a 64-bit value.
 * If you need less than 32 bits, use a bitmask.
 */

#ifndef XF_DATABASE_HASH_LOOKUP3_H
#define XF_DATABASE_HASH_LOOKUP3_H

#ifndef __cplusplus
#error "Vitis Database Library only works with C++."
#endif

#include "hls_stream.h"
// for wide input&output
#include <ap_int.h>
// for standard integer
#include "xf_database/types.hpp"
// for static assert
#include "xf_database/utils.hpp"

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

#include <stdint.h>

// XXX HLS currently does not support cross-compilation, and assumes the target
// to have same endianness as build host. HLS is only available on X86 machines,
// and thus the code is always little-endian.
#if !defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "HLS only supports little endian systems!"
#endif

#define rot(x, k) (((x) << (k)) | ((x) >> (32 - (k))));

//-----------------------------------------------------------------------------

namespace xf {
namespace database {
namespace details {

/// @brief Generate 32bit output hash value for the input key.
/// @tparam W width of input in bit.
/// @param key_val input key value
/// @param hash_val output hash value.
template <int W>
inline void hashlookup3_core(ap_uint<W> key_val, ap_uint<32>& hash_val) {
    const int key96blen = W / 96;

    // key8blen is the BYTE len of the key.
    const int key8blen = W / 8;
    const ap_uint<32> c1 = 0xdeadbeef;
    //----------
    // body

    // use magic word(seed) to initial the output
    ap_uint<32> seed = 1032032634; // 0x3D83917A

    // loop value 32 bit
    uint32_t a, b, c;
    a = b = c = c1 + ((ap_uint<32>)key8blen) + ((ap_uint<32>)seed);

    _XF_DATABASE_PRINT("DEBUG: a =\t%08x (pad 0)\n b =\t%08x\n c =\t%08x\n", a, b, c);

LOOP_lookup3_MAIN:
    for (int j = 0; j < key96blen; ++j) {
        a += key_val(96 * j + 31, 96 * j);
        b += key_val(96 * j + 63, 96 * j + 32);
        c += key_val(96 * j + 95, 96 * j + 64);
        _XF_DATABASE_PRINT("DEBUG: a =\t%08x (pad 1)\n b =\t%08x\n c =\t%08x\n", a, b, c);

        a -= c;
        a ^= rot(c, 4);
        c += b;

        b -= a;
        b ^= rot(a, 6);
        a += c;

        c -= b;
        c ^= rot(b, 8);
        b += a;

        a -= c;
        a ^= rot(c, 16);
        c += b;

        b -= a;
        b ^= rot(a, 19);
        a += c;

        c -= b;
        c ^= rot(b, 4);
        b += a;
        _XF_DATABASE_PRINT("DEBUG: a =\t%08x (pad 2)\n b =\t%08x\n c =\t%08x\n", a, b, c);
    }

    // tail	k8 is a temp
    // key8blen-12*key96blen will not large than 11
    switch (key8blen - 12 * key96blen) {
        case 12:
            c += key_val(W - 1, key96blen * 3 * 32 + 64);
            b += key_val(key96blen * 3 * 32 + 63, key96blen * 3 * 32 + 32);
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 11:
            c += key_val(W - 1, key96blen * 3 * 32 + 64) & 0xffffff;
            b += key_val(key96blen * 3 * 32 + 63, key96blen * 3 * 32 + 32);
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 10:
            c += key_val(W - 1, key96blen * 3 * 32 + 64) & 0xffff;
            b += key_val(key96blen * 3 * 32 + 63, key96blen * 3 * 32 + 32);
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 9:
            c += key_val(W - 1, key96blen * 3 * 32 + 64) & 0xff;
            b += key_val(key96blen * 3 * 32 + 63, key96blen * 3 * 32 + 32);
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;

        case 8:
            b += key_val(W - 1, key96blen * 3 * 32 + 32);
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 7:
            b += key_val(W - 1, key96blen * 3 * 32 + 32) & 0xffffff;
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 6:
            b += key_val(W - 1, key96blen * 3 * 32 + 32) & 0xffff;
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 5:
            b += key_val(W - 1, key96blen * 3 * 32 + 32) & 0xff;
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;

        case 4:
            a += key_val(W - 1, key96blen * 3 * 32);
            break;
        case 3:
            a += key_val(W - 1, key96blen * 3 * 32) & 0xffffff;
            break;
        case 2:
            a += key_val(W - 1, key96blen * 3 * 32) & 0xffff;
            break;
        case 1:
            a += key_val(W - 1, key96blen * 3 * 32) & 0xff;
            break;

        default:
            break; // in the original algorithm case:0 will not appear
    }
    _XF_DATABASE_PRINT("DEBUG: a =\t%08x ((ap_uint<32>)pad 3)\n b =\t%08x\n c =\t%08x\n", a, b, c);
    // finalization
    c ^= b;
    c -= rot(b, 14);

    a ^= c;
    a -= rot(c, 11);

    b ^= a;
    b -= rot(a, 25);

    c ^= b;
    c -= rot(b, 16);

    a ^= c;
    a -= rot(c, 4);

    b ^= a;
    b -= rot(a, 14);

    c ^= b;
    c -= rot(b, 24);
    _XF_DATABASE_PRINT("DEBUG: a =\t%08x (pad 4)\n b =\t%08x\n c =\t%08x\n", a, b, c);

    hash_val = (ap_uint<32>)c;
} // lookup3_32

/// brief Generate 64bit output hash value for the input key.
/// @param key_val input, key value
/// @param hash_val output, hash value.
template <int W>
inline void hashlookup3_core(ap_uint<W> key_val, ap_uint<64>& hash_val) {
    const int key96blen = W / 96;

    // key8blen is the BYTE len of the key.
    const int key8blen = W / 8;
    const ap_uint<32> c1 = 0xdeadbeef;
    //----------
    // body

    // use magic word(seed) to initial the output
    ap_uint<64> hash1 = 1032032634; // 0x3D83917A
    ap_uint<64> hash2 = 2818135537; // 0xA7F955F1

    // loop value 32 bit
    uint32_t a, b, c;
    a = b = c = c1 + ((ap_uint<32>)key8blen) + ((ap_uint<32>)hash1);
    c += (ap_uint<32>)hash2;

    _XF_DATABASE_PRINT("DEBUG: a =\t%08x (pad 0)\n b =\t%08x\n c =\t%08x\n", a, b, c);

LOOP_lookup3_MAIN:
    for (int j = 0; j < key96blen; ++j) {
        a += key_val(96 * j + 31, 96 * j);
        b += key_val(96 * j + 63, 96 * j + 32);
        c += key_val(96 * j + 95, 96 * j + 64);
        _XF_DATABASE_PRINT("DEBUG: a =\t%08x (pad 1)\n b =\t%08x\n c =\t%08x\n", a, b, c);

        a -= c;
        a ^= rot(c, 4);
        c += b;

        b -= a;
        b ^= rot(a, 6);
        a += c;

        c -= b;
        c ^= rot(b, 8);
        b += a;

        a -= c;
        a ^= rot(c, 16);
        c += b;

        b -= a;
        b ^= rot(a, 19);
        a += c;

        c -= b;
        c ^= rot(b, 4);
        b += a;
        _XF_DATABASE_PRINT("DEBUG: a =\t%08x (pad 2)\n b =\t%08x\n c =\t%08x\n", a, b, c);
    }

    // tail	k8 is a temp
    // key8blen-12*key96blen will not large than 11
    switch (key8blen - 12 * key96blen) {
        case 12:
            c += key_val(W - 1, key96blen * 3 * 32 + 64);
            b += key_val(key96blen * 3 * 32 + 63, key96blen * 3 * 32 + 32);
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 11:
            c += key_val(W - 1, key96blen * 3 * 32 + 64) & 0xffffff;
            b += key_val(key96blen * 3 * 32 + 63, key96blen * 3 * 32 + 32);
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 10:
            c += key_val(W - 1, key96blen * 3 * 32 + 64) & 0xffff;
            b += key_val(key96blen * 3 * 32 + 63, key96blen * 3 * 32 + 32);
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 9:
            c += key_val(W - 1, key96blen * 3 * 32 + 64) & 0xff;
            b += key_val(key96blen * 3 * 32 + 63, key96blen * 3 * 32 + 32);
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;

        case 8:
            b += key_val(W - 1, key96blen * 3 * 32 + 32);
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 7:
            b += key_val(W - 1, key96blen * 3 * 32 + 32) & 0xffffff;
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 6:
            b += key_val(W - 1, key96blen * 3 * 32 + 32) & 0xffff;
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 5:
            b += key_val(W - 1, key96blen * 3 * 32 + 32) & 0xff;
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;

        case 4:
            a += key_val(W - 1, key96blen * 3 * 32);
            break;
        case 3:
            a += key_val(W - 1, key96blen * 3 * 32) & 0xffffff;
            break;
        case 2:
            a += key_val(W - 1, key96blen * 3 * 32) & 0xffff;
            break;
        case 1:
            a += key_val(W - 1, key96blen * 3 * 32) & 0xff;
            break;

        default:
            break; // in the original algorithm case:0 will not appear
    }
    _XF_DATABASE_PRINT("DEBUG: a =\t%08x (pad 3)\n b =\t%08x\n c =\t%08x\n", a, b, c);
    // finalization
    c ^= b;
    c -= rot(b, 14);

    a ^= c;
    a -= rot(c, 11);

    b ^= a;
    b -= rot(a, 25);

    c ^= b;
    c -= rot(b, 16);

    a ^= c;
    a -= rot(c, 4);

    b ^= a;
    b -= rot(a, 14);

    c ^= b;
    c -= rot(b, 24);
    _XF_DATABASE_PRINT("DEBUG: a =\t%08x (pad 4)\n b =\t%08x\n c =\t%08x\n", a, b, c);

    hash1 = (ap_uint<64>)c;
    hash2 = (ap_uint<64>)b;

    hash_val = hash1 << 32 | hash2;
    /// output hash stream width in 64bit.
    _XF_DATABASE_PRINT("DEBUG: hash1 =%s (pad 5)\n hash2 =%s\n hash_val =%s\n", hash1.to_string().c_str(),
                       hash2.to_string().c_str(), hash_val.to_string().c_str());
} // lookup3_64

/// brief Generate 64bit output hash value for the input key.
/// @tparam W input key width
/// @param key_val input, key value
/// @param seed input, seed value
/// @param hash_val output, hash value
template <int W>
inline void hashlookup3_seed_core(ap_uint<W> key_val, ap_uint<32> seed, ap_uint<64>& hash_val) {
    const int key96blen = W / 96;

    // key8blen is the BYTE len of the key.
    const int key8blen = W / 8;

    //----------
    // body

    // use magic word(seed) to initial the output
    ap_uint<64> hash1 = 1032032634; // 0x3D83917A
    ap_uint<64> hash2 = 2818135537; // 0xA7F955F1

    // loop value 32 bit
    uint32_t a, b, c;
    a = b = c = seed + ((ap_uint<32>)key8blen) + ((ap_uint<32>)hash1);
    c += (ap_uint<32>)hash2;

    _XF_DATABASE_PRINT("DEBUG: a =\t%08x (pad 0)\n b =\t%08x\n c =\t%08x\n", a, b, c);

LOOP_lookup3_MAIN:
    for (int j = 0; j < key96blen; ++j) {
        a += key_val(96 * j + 31, 96 * j);
        b += key_val(96 * j + 63, 96 * j + 32);
        c += key_val(96 * j + 95, 96 * j + 64);
        _XF_DATABASE_PRINT("DEBUG: a =\t%08x (pad 1)\n b =\t%08x\n c =\t%08x\n", a, b, c);

        a -= c;
        a ^= rot(c, 4);
        c += b;

        b -= a;
        b ^= rot(a, 6);
        a += c;

        c -= b;
        c ^= rot(b, 8);
        b += a;

        a -= c;
        a ^= rot(c, 16);
        c += b;

        b -= a;
        b ^= rot(a, 19);
        a += c;

        c -= b;
        c ^= rot(b, 4);
        b += a;
        _XF_DATABASE_PRINT("DEBUG: a =\t%08x (pad 2)\n b =\t%08x\n c =\t%08x\n", a, b, c);
    }

    // tail	k8 is a temp
    // key8blen-12*key96blen will not large than 11
    switch (key8blen - 12 * key96blen) {
        case 12:
            c += key_val(W - 1, key96blen * 3 * 32 + 64);
            b += key_val(key96blen * 3 * 32 + 63, key96blen * 3 * 32 + 32);
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 11:
            c += key_val(W - 1, key96blen * 3 * 32 + 64) & 0xffffff;
            b += key_val(key96blen * 3 * 32 + 63, key96blen * 3 * 32 + 32);
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 10:
            c += key_val(W - 1, key96blen * 3 * 32 + 64) & 0xffff;
            b += key_val(key96blen * 3 * 32 + 63, key96blen * 3 * 32 + 32);
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 9:
            c += key_val(W - 1, key96blen * 3 * 32 + 64) & 0xff;
            b += key_val(key96blen * 3 * 32 + 63, key96blen * 3 * 32 + 32);
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;

        case 8:
            b += key_val(W - 1, key96blen * 3 * 32 + 32);
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 7:
            b += key_val(W - 1, key96blen * 3 * 32 + 32) & 0xffffff;
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 6:
            b += key_val(W - 1, key96blen * 3 * 32 + 32) & 0xffff;
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 5:
            b += key_val(W - 1, key96blen * 3 * 32 + 32) & 0xff;
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;

        case 4:
            a += key_val(W - 1, key96blen * 3 * 32);
            break;
        case 3:
            a += key_val(W - 1, key96blen * 3 * 32) & 0xffffff;
            break;
        case 2:
            a += key_val(W - 1, key96blen * 3 * 32) & 0xffff;
            break;
        case 1:
            a += key_val(W - 1, key96blen * 3 * 32) & 0xff;
            break;

        default:
            break; // in the original algorithm case:0 will not appear
    }
    _XF_DATABASE_PRINT("DEBUG: a =\t%08x (pad 3)\n b =\t%08x\n c =\t%08x\n", a, b, c);
    // finalization
    c ^= b;
    c -= rot(b, 14);

    a ^= c;
    a -= rot(c, 11);

    b ^= a;
    b -= rot(a, 25);

    c ^= b;
    c -= rot(b, 16);

    a ^= c;
    a -= rot(c, 4);

    b ^= a;
    b -= rot(a, 14);

    c ^= b;
    c -= rot(b, 24);
    _XF_DATABASE_PRINT("DEBUG: a =\t%08x (pad 4)\n b =\t%08x\n c =\t%08x\n", a, b, c);

    hash1 = (ap_uint<64>)c;
    hash2 = (ap_uint<64>)b;

    hash_val = hash1 << 32 | hash2;
    /// output hash stream width in 64bit.
    _XF_DATABASE_PRINT("DEBUG: hash1 =%s (pad 5)\n hash2 =%s\n hash_val =%s\n", hash1.to_string().c_str(),
                       hash2.to_string().c_str(), hash_val.to_string().c_str());
} // lookup3_64

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/// @brief lookup3 algorithm, 64-bit hash.
/// II=1 when W<=96, otherwise II=(W/96).
/// @tparam W the bit width of ap_uint type for input message stream.
/// @param key_strm the message being hashed.
/// @param hash_strm the result.
template <int W>
void hashLookup3(hls::stream<ap_uint<W> >& key_strm, hls::stream<ap_uint<64> >& hash_strm) {
#pragma HLS inline
    ap_uint<W> k = key_strm.read();
    ap_uint<64> h;
    details::hashlookup3_core(k, h);
    hash_strm.write(h);
}

template <int W>
void hashLookup3(ap_uint<32> seed, hls::stream<ap_uint<W> >& key_strm, hls::stream<ap_uint<64> >& hash_strm) {
#pragma HLS inline
    ap_uint<W> k = key_strm.read();
    ap_uint<64> h;
    details::hashlookup3_seed_core(k, seed, h);
    hash_strm.write(h);
}

/// @brief lookup3 algorithm, 32-bit hash.
/// II=1 when W<=96, otherwise II=(W/96).
/// @tparam W the bit width of ap_uint type for input message stream.
/// @param key_strm the message being hashed.
/// @param hash_strm the result.
template <int W>
void hashLookup3(hls::stream<ap_uint<W> >& key_strm, hls::stream<ap_uint<32> >& hash_strm) {
#pragma HLS inline
    ap_uint<W> k = key_strm.read();
    ap_uint<32> h;
    details::hashlookup3_core(k, h);
    hash_strm.write(h);
}

/**
 * @brief lookup3 algorithm, 64-bit or 32-bit hash.
 *
 * @tparam WK the bit width of input message stream.
 * @tparam WH the bit width of output hash stream, must be 64 or 32.
 *
 * @param key_strm the message being hashed.
 * @param e_key_strm end of key flag stream.
 * @param hash_strm the result.
 * @param e_hash_strm end of hash flag stream.
 */
template <int WK, int WH>
void hashLookup3(hls::stream<ap_uint<WK> >& key_strm,
                 hls::stream<bool>& e_key_strm,
                 hls::stream<ap_uint<WH> >& hash_strm,
                 hls::stream<bool>& e_hash_strm) {
    XF_DATABASE_STATIC_ASSERT((WH == 32 || WH == 64),
                              "lookup3 only support 32bit"
                              " or 64bit hash now.");
    bool e = e_key_strm.read();
    while (!e) {
        ap_uint<WK> k = key_strm.read();
        ap_uint<WH> h;
        details::hashlookup3_core(k, h);
        hash_strm.write(h);
        e_hash_strm.write(false);
        //
        e = e_key_strm.read();
    }
    e_hash_strm.write(true);
}

} // namespace database
} // namespace xf
//-----------------------------------------------------------------------------

#undef rot
#undef _XF_DATABASE_VOID_CAST
#undef _XF_DATABASE_PRINT

#endif // XF_DATABASE_HASH_LOOKUP3_H
