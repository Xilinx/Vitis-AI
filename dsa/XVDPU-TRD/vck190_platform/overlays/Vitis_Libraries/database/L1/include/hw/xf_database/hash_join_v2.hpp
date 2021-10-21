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
 * @file hash_join_v2.hpp
 * @brief hash join implementation, targeting HBM/DDR devices.
 *
 * The limitations are:
 * (1) less than 8M entries from small table.
 * (2) max number of key with same hash is less than 512.
 *
 * This file is part of Vitis Database Library.
 */

#ifndef XF_DATABASE_HASH_JOIN_v2_H
#define XF_DATABASE_HASH_JOIN_v2_H

#ifndef __cplusplus
#error "Vitis Database Library only works with C++."
#endif

#include "hls_stream.h"
#include "ap_int.h"

#include "xf_database/bloom_filter.hpp"
#include "xf_database/hash_lookup3.hpp"
#include "xf_database/utils.hpp"

// FIXME For debug
#ifndef __SYNTHESIS__
#include <iostream>
#endif

#ifdef URAM_SPLITTING
#undef URAM_SPLITTING
#endif
#define URAM_SPLITTING 1

#ifdef URAM_SPLITTING

#define write_bit_vector(i, v)             \
    do {                                   \
        ap_uint<2> s = (i);                \
        switch (s) {                       \
            case 0:                        \
                bit_vector0[(i) >> 2] = v; \
                break;                     \
            case 1:                        \
                bit_vector1[(i) >> 2] = v; \
                break;                     \
            case 2:                        \
                bit_vector2[(i) >> 2] = v; \
                break;                     \
            default:                       \
                bit_vector3[(i) >> 2] = v; \
        }                                  \
    } while (0)

#define read_bit_vector(i, v)              \
    do {                                   \
        ap_uint<2> s = (i);                \
        switch (s) {                       \
            case 0:                        \
                v = bit_vector0[(i) >> 2]; \
                break;                     \
            case 1:                        \
                v = bit_vector1[(i) >> 2]; \
                break;                     \
            case 2:                        \
                v = bit_vector2[(i) >> 2]; \
                break;                     \
            default:                       \
                v = bit_vector3[(i) >> 2]; \
        }                                  \
    } while (0)

#else // !defined(URAM_SPLITTING)

#define write_bit_vector(i, v) \
    do {                       \
        bit_vector[(i)] = v;   \
    } while (0)

#define read_bit_vector(i, v) \
    do {                      \
        v = bit_vector[(i)];  \
    } while (0)

#endif // !defined(URAM_SPLITTING)

namespace xf {
namespace database {
namespace details {
namespace join_v2 {

/// @brief Multiplier
/// Only support mux8_1, mux4_1, mux2_1
template <int IN_NM>
ap_uint<3> mux(ap_uint<IN_NM> rd) {
#pragma HLS inline
    ap_uint<3> o = 0;
    if (IN_NM == 8) {
        o[0] = rd[1] | rd[3] | rd[5] | rd[7];
        o[1] = rd[2] | rd[3] | rd[6] | rd[7];
        o[2] = rd[4] | rd[5] | rd[6] | rd[7];
    } else if (IN_NM == 4) {
        o[0] = rd[1] | rd[3];
        o[1] = rd[2] | rd[3];
    } else if (IN_NM == 2) {
        o[0] = rd[1];
    } else {
        o = 0;
    }
    return o;
}

template <int CH_NM>
ap_uint<CH_NM> mul_ch_read(ap_uint<CH_NM> empty) {
    ap_uint<CH_NM> rd = 0;
#pragma HLS inline
    for (int i = 0; i < CH_NM; i++) {
#pragma HLS unroll
        ap_uint<CH_NM> t_e = 0;
        if (i > 0) t_e = empty(i - 1, 0);
        rd[i] = t_e > 0 ? (bool)0 : (bool)empty[i];
    }
    return rd;
}

// ------------------------------------------------------------

/// @brief calculate hash value based on key
template <int HASH_MODE, int KEYW, int HASHW, int EN_BF>
void hash_wrapper(hls::stream<ap_uint<KEYW> >& i_key_strm,
                  hls::stream<bool>& i_e_strm,
                  hls::stream<ap_uint<64> >& l_hash_strm,
                  hls::stream<ap_uint<HASHW> >& s_hash_strm,
                  hls::stream<ap_uint<KEYW> >& o_key_strm,
                  hls::stream<bool>& o_e0_strm,
                  hls::stream<bool>& o_e1_strm) {
    hls::stream<ap_uint<KEYW> > key_strm_in;
#pragma HLS STREAM variable = key_strm_in depth = 24
#pragma HLS bind_storage variable = key_strm_in type = fifo impl = srl
    hls::stream<ap_uint<64> > hash_strm_out;
#pragma HLS STREAM variable = hash_strm_out depth = 8
// radix hash function
#ifndef __SYNTHESIS__
#ifdef DEBUG
    unsigned int i = 0;
#endif
#endif
    bool last = i_e_strm.read();
BUILD_HASH_LOOP:
    while (!last) {
#pragma HLS loop_tripcount min = 1 max = 1000
#pragma HLS PIPELINE II = 1
        bool blk = i_e_strm.empty() || i_key_strm.empty() || o_key_strm.full();
        if (!blk) {
            last = i_e_strm.read();
            o_e0_strm.write(0);
            ap_uint<KEYW> key = i_key_strm.read();
            o_key_strm.write(key);
            if (HASH_MODE != 0) key_strm_in.write(key);
            if (HASH_MODE == 0) {
                ap_uint<HASHW> s_hash_val = key(HASHW - 1, 0);
                s_hash_strm.write(s_hash_val);
                if (EN_BF) {
                    ap_uint<64> l_hash_val = key;
                    l_hash_strm.write(l_hash_val);
                    o_e1_strm.write(0);
                }
            } else {
                database::hashLookup3<KEYW>(key_strm_in, hash_strm_out);
                ap_uint<64> l_hash_val = hash_strm_out.read();
                if (EN_BF) {
                    l_hash_strm.write(l_hash_val);
                    o_e1_strm.write(0);
                }
                ap_uint<HASHW> s_hash_val = l_hash_val(HASHW - 1, 0);
                s_hash_strm.write(s_hash_val);
#ifndef __SYNTHESIS__
#ifdef DEBUG
                i++;
                if (i < 10) std::cout << "hash_val = " << hash_val << std::endl;
#endif
#endif
            }
        }
    }
    o_e0_strm.write(1);
    if (EN_BF) {
        o_e1_strm.write(1);
    }
}

template <int BF_W>
void bloomfilter_wrapper(hls::stream<ap_uint<64> >& i_hash_strm,
                         hls::stream<bool>& i_e_strm,
                         hls::stream<bool>& o_vld_strm,
                         const int is_build,
                         ap_uint<16>* bit_vec_0,
                         ap_uint<16>* bit_vec_1,
                         ap_uint<16>* bit_vec_2) {
    hls::stream<ap_uint<64> > hash_strm;
#pragma HLS STREAM variable = hash_strm depth = 8
    hls::stream<bool> vld_strm;
#pragma HLS STREAM variable = vld_strm depth = 8
    ap_uint<BF_W - 4> cnt = 0;
    if (is_build == 0) {
        bool last = i_e_strm.read();
        while (!last) {
#pragma HLS pipeline II = 1
            ap_uint<64> tmp = i_hash_strm.read();
            last = i_e_strm.read();
            cnt++;
            // clear up
            bit_vec_0[cnt] = 0;
            bit_vec_1[cnt] = 0;
            bit_vec_2[cnt] = 0;
        }
    } else if (is_build == 1) {
        details::bv_update_bram<64, BF_W>(i_hash_strm, i_e_strm, bit_vec_0, bit_vec_1, bit_vec_2);
    } else {
        bool last = i_e_strm.read();
        while (!last) {
#pragma HLS pipeline II = 1
            if (!i_e_strm.empty()) {
                last = i_e_strm.read();
                ap_uint<64> hash_in = i_hash_strm.read();
                hash_strm.write(hash_in);
                details::bv_check_one<64, BF_W>(hash_strm, bit_vec_0, bit_vec_1, bit_vec_2, vld_strm);
                bool vld = vld_strm.read();
                o_vld_strm.write(vld);
            }
        }
    }
}

/// @brief dispatch data to multiple PU based one the hash value
/// every PU with different hash_value.
template <int KEYW, int PW, int HASHWH, int HASHWL, int PU, int EN_BF>
void dispatch(hls::stream<ap_uint<KEYW> >& i_key_strm,
              hls::stream<ap_uint<PW> >& i_pld_strm,
              hls::stream<ap_uint<HASHWH + HASHWL> >& i_hash_strm,
              hls::stream<bool>& i_vld_strm,
              hls::stream<bool>& i_e_strm,
              hls::stream<ap_uint<KEYW> > o_key_strm[PU],
              hls::stream<ap_uint<PW> > o_pld_strm[PU],
              hls::stream<ap_uint<HASHWL> > o_hash_strm[PU],
              hls::stream<bool> o_e_strm[PU],
              const int is_build) {
    bool last = i_e_strm.read();
    while (!last) {
#pragma HLS pipeline II = 1
        ap_uint<HASHWH + HASHWL> hash_val = i_hash_strm.read();
        ap_uint<HASHWH> idx = hash_val(HASHWH + HASHWL - 1, HASHWL);
        ap_uint<HASHWL> hash_out = hash_val(HASHWL - 1, 0);
        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<PW> pld = i_pld_strm.read();
        bool vld = 1;
        if (EN_BF && (is_build == 2)) {
            vld = i_vld_strm.read();
        }
        if (vld) {
            o_key_strm[idx].write(key);
            o_pld_strm[idx].write(pld);
            o_hash_strm[idx].write(hash_out);
            o_e_strm[idx].write(false);
        }
        last = i_e_strm.read();
    }
    // for do_while
    for (int i = 0; i < PU; i++) {
#pragma HLS unroll
        // if add merge module, need uncomment
        o_key_strm[i].write(0);
        o_pld_strm[i].write(0);
        o_hash_strm[i].write(0);
        o_e_strm[i].write(true);
    }
}

/// @brief read data from multiple channel,
/// dispatch data based on hash value to multiple PU.
template <int HASH_MODE, int KEYW, int PW, int HASHWH, int HASHWL, int PU, int BF_W, bool EN_BF>
void dispatch_unit(hls::stream<ap_uint<KEYW> >& i_key_strm,
                   hls::stream<ap_uint<PW> >& i_pld_strm,
                   hls::stream<bool>& i_e_strm,
                   hls::stream<ap_uint<KEYW> > o_key_strm[PU],
                   hls::stream<ap_uint<PW> > o_pld_strm[PU],
                   hls::stream<ap_uint<HASHWL> > o_hash_strm[PU],
                   hls::stream<bool> o_e_strm[PU],
                   ap_uint<16>* bit_vec_0,
                   ap_uint<16>* bit_vec_1,
                   ap_uint<16>* bit_vec_2,
                   const int is_build) {
    hls::stream<ap_uint<64> > l_hash_strm;
#pragma HLS STREAM variable = l_hash_strm depth = 4
    hls::stream<ap_uint<HASHWH + HASHWL> > s_hash_strm;
#pragma HLS STREAM variable = s_hash_strm depth = 4

    hls::stream<bool> io_e0_strm;
#pragma HLS STREAM variable = io_e0_strm depth = 16
    hls::stream<bool> io_e1_strm;
#pragma HLS STREAM variable = io_e1_strm depth = 4

    hls::stream<ap_uint<KEYW> > io_k0_strm;
#pragma HLS STREAM variable = io_k0_strm depth = 16

    hls::stream<bool> io_vld_strm;
#pragma HLS STREAM variable = io_vld_strm depth = 4

#pragma HLS DATAFLOW

    hash_wrapper<HASH_MODE, KEYW, HASHWH + HASHWL, EN_BF>(i_key_strm, i_e_strm, l_hash_strm, s_hash_strm, io_k0_strm,
                                                          io_e0_strm, io_e1_strm);

    if (EN_BF) {
        bloomfilter_wrapper<BF_W>(l_hash_strm, io_e1_strm, io_vld_strm, is_build, bit_vec_0, bit_vec_1, bit_vec_2);
    }

    dispatch<KEYW, PW, HASHWH, HASHWL, PU, EN_BF>(io_k0_strm, i_pld_strm, s_hash_strm, io_vld_strm, io_e0_strm,
                                                  o_key_strm, o_pld_strm, o_hash_strm, o_e_strm, is_build);
}

template <int HASH_MODE, int KEYW, int PW, int HASHWH, int HASHWL, int PU, int BF_W, bool EN_BF>
void dispatch_wrapper(hls::stream<ap_uint<KEYW> >& i_key_strm,
                      hls::stream<ap_uint<PW> >& i_pld_strm,
                      hls::stream<bool>& i_e_strm,
                      hls::stream<ap_uint<KEYW> > o_key_strm[PU],
                      hls::stream<ap_uint<PW> > o_pld_strm[PU],
                      hls::stream<ap_uint<HASHWL> > o_hash_strm[PU],
                      hls::stream<bool> o_e_strm[PU]) {
    const int DH = EN_BF ? (1 << (BF_W - 4)) : 1;
    ap_uint<16> bit_vec_0[DH];
    ap_uint<16> bit_vec_1[DH];
    ap_uint<16> bit_vec_2[DH];
    // first: build bitmap
    // second: build hash table
    // third: probe_hash_table
    for (int r = 0; r < 3; r++) {
        dispatch_unit<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU, BF_W, EN_BF>(
            i_key_strm, i_pld_strm, i_e_strm, o_key_strm, o_pld_strm, o_hash_strm, o_e_strm, bit_vec_0, bit_vec_1,
            bit_vec_2, r);
    }
}

// ------------------------------------------------------------

template <int KEYW, int PW, int HASHW>
void merge1_1(hls::stream<ap_uint<KEYW> >& i_key_strm,
              hls::stream<ap_uint<PW> >& i_pld_strm,
              hls::stream<ap_uint<HASHW> >& i_hash_strm,
              hls::stream<bool>& i_e_strm,
              hls::stream<ap_uint<KEYW> >& o_key_strm,
              hls::stream<ap_uint<PW> >& o_pld_strm,
              hls::stream<ap_uint<HASHW> >& o_hash_strm,
              hls::stream<bool>& o_e_strm) {
    bool last = 0; // i_e_strm.read();
LOOP_MERGE1_1:
    do {
#pragma HLS pipeline II = 1
        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<PW> pld = i_pld_strm.read();
        ap_uint<HASHW> hash_val = i_hash_strm.read();
        last = i_e_strm.read();
        if (!last) {
            o_key_strm.write(key);
            o_pld_strm.write(pld);
            o_hash_strm.write(hash_val);
            o_e_strm.write(false);
        }
    } while (!last);
    o_e_strm.write(true);
}

template <int KEYW, int PW, int HASHW>
void merge2_1(hls::stream<ap_uint<KEYW> >& i0_key_strm,
              hls::stream<ap_uint<KEYW> >& i1_key_strm,
              hls::stream<ap_uint<PW> >& i0_pld_strm,
              hls::stream<ap_uint<PW> >& i1_pld_strm,
              hls::stream<ap_uint<HASHW> >& i0_hash_strm,
              hls::stream<ap_uint<HASHW> >& i1_hash_strm,
              hls::stream<bool>& i0_e_strm,
              hls::stream<bool>& i1_e_strm,
              hls::stream<ap_uint<KEYW> >& o_key_strm,
              hls::stream<ap_uint<PW> >& o_pld_strm,
              hls::stream<ap_uint<HASHW> >& o_hash_strm,
              hls::stream<bool>& o_e_strm) {
    ap_uint<KEYW> key_arry[2];
#pragma HLS array_partition variable = key_arry dim = 1
    ap_uint<PW> pld_arry[2];
#pragma HLS array_partition variable = pld_arry dim = 1
    ap_uint<HASHW> hash_val_arry[2];
#pragma HLS array_partition variable = hash_val_arry dim = 1
    ap_uint<2> empty_e = 0;
    ;
    ap_uint<2> rd_e = 0;
    ;
    ap_uint<2> last = 0;
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
LOOP_MERGE2_1:
    do {
#pragma HLS loop_tripcount min = 1 max = 5000
#pragma HLS PIPELINE II = 1
        empty_e[0] = !i0_e_strm.empty() && !last[0];
        empty_e[1] = !i1_e_strm.empty() && !last[1];
        rd_e = mul_ch_read(empty_e);
        if (rd_e[0]) {
            key_arry[0] = i0_key_strm.read();
            pld_arry[0] = i0_pld_strm.read();
            hash_val_arry[0] = i0_hash_strm.read();
            last[0] = i0_e_strm.read();
        }
        if (rd_e[1]) {
            key_arry[1] = i1_key_strm.read();
            pld_arry[1] = i1_pld_strm.read();
            hash_val_arry[1] = i1_hash_strm.read();
            last[1] = i1_e_strm.read();
        }
        // only support 8 channels, 4 channels and 2 channels
        ap_uint<3> id = mux<2>(rd_e);
        ap_uint<KEYW> key = key_arry[id];
        ap_uint<PW> pld = pld_arry[id];
        ap_uint<HASHW> hash_val = hash_val_arry[id];
        bool valid_n = last[id];
        if (!valid_n && rd_e != 0) {
#ifndef __SYNTHESIS__
            cnt++;
#endif
            o_key_strm.write(key);
            o_pld_strm.write(pld);
            o_hash_strm.write(hash_val);
            o_e_strm.write(false);
        }
    } while (last != 3);
    o_e_strm.write(true);
}

template <int KEYW, int PW, int HASHW>
void merge4_1(hls::stream<ap_uint<KEYW> >& i0_key_strm,
              hls::stream<ap_uint<KEYW> >& i1_key_strm,
              hls::stream<ap_uint<KEYW> >& i2_key_strm,
              hls::stream<ap_uint<KEYW> >& i3_key_strm,
              hls::stream<ap_uint<PW> >& i0_pld_strm,
              hls::stream<ap_uint<PW> >& i1_pld_strm,
              hls::stream<ap_uint<PW> >& i2_pld_strm,
              hls::stream<ap_uint<PW> >& i3_pld_strm,
              hls::stream<ap_uint<HASHW> >& i0_hash_strm,
              hls::stream<ap_uint<HASHW> >& i1_hash_strm,
              hls::stream<ap_uint<HASHW> >& i2_hash_strm,
              hls::stream<ap_uint<HASHW> >& i3_hash_strm,
              hls::stream<bool>& i0_e_strm,
              hls::stream<bool>& i1_e_strm,
              hls::stream<bool>& i2_e_strm,
              hls::stream<bool>& i3_e_strm,
              hls::stream<ap_uint<KEYW> >& o_key_strm,
              hls::stream<ap_uint<PW> >& o_pld_strm,
              hls::stream<ap_uint<HASHW> >& o_hash_strm,
              hls::stream<bool>& o_e_strm) {
    ap_uint<KEYW> key_arry[4];
#pragma HLS array_partition variable = key_arry dim = 1
    ap_uint<PW> pld_arry[4];
#pragma HLS array_partition variable = pld_arry dim = 1
    ap_uint<HASHW> hash_val_arry[4];
#pragma HLS array_partition variable = hash_val_arry dim = 1
    ap_uint<4> empty_e = 0;
    ;
    ap_uint<4> rd_e = 0;
    ;
    ap_uint<4> last = 0;
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
LOOP_MERGE4_1:
    do {
#pragma HLS loop_tripcount min = 1 max = 5000
#pragma HLS PIPELINE II = 1
        empty_e[0] = !i0_e_strm.empty() && !last[0];
        empty_e[1] = !i1_e_strm.empty() && !last[1];
        empty_e[2] = !i2_e_strm.empty() && !last[2];
        empty_e[3] = !i3_e_strm.empty() && !last[3];
        rd_e = mul_ch_read(empty_e);
        if (rd_e[0]) {
            key_arry[0] = i0_key_strm.read();
            pld_arry[0] = i0_pld_strm.read();
            hash_val_arry[0] = i0_hash_strm.read();
            last[0] = i0_e_strm.read();
        }
        if (rd_e[1]) {
            key_arry[1] = i1_key_strm.read();
            pld_arry[1] = i1_pld_strm.read();
            hash_val_arry[1] = i1_hash_strm.read();
            last[1] = i1_e_strm.read();
        }
        if (rd_e[2]) {
            key_arry[2] = i2_key_strm.read();
            pld_arry[2] = i2_pld_strm.read();
            hash_val_arry[2] = i2_hash_strm.read();
            last[2] = i2_e_strm.read();
        }
        if (rd_e[3]) {
            key_arry[3] = i3_key_strm.read();
            pld_arry[3] = i3_pld_strm.read();
            hash_val_arry[3] = i3_hash_strm.read();
            last[3] = i3_e_strm.read();
        }
        // only support 8 channels, 4 channels and 2 channels
        ap_uint<3> id = mux<4>(rd_e);
        ap_uint<KEYW> key = key_arry[id];
        ap_uint<PW> pld = pld_arry[id];
        ap_uint<HASHW> hash_val = hash_val_arry[id];
        bool valid_n = last[id];
        if (!valid_n && rd_e != 0) {
#ifndef __SYNTHESIS__
            cnt++;
#endif
            o_key_strm.write(key);
            o_pld_strm.write(pld);
            o_hash_strm.write(hash_val);
            o_e_strm.write(false);
        }
    } while (last != 15);
    o_e_strm.write(true);
}

template <int KEYW, int PW, int HASHW>
void merge8_1(hls::stream<ap_uint<KEYW> >& i0_key_strm,
              hls::stream<ap_uint<KEYW> >& i1_key_strm,
              hls::stream<ap_uint<KEYW> >& i2_key_strm,
              hls::stream<ap_uint<KEYW> >& i3_key_strm,
              hls::stream<ap_uint<KEYW> >& i4_key_strm,
              hls::stream<ap_uint<KEYW> >& i5_key_strm,
              hls::stream<ap_uint<KEYW> >& i6_key_strm,
              hls::stream<ap_uint<KEYW> >& i7_key_strm,
              hls::stream<ap_uint<PW> >& i0_pld_strm,
              hls::stream<ap_uint<PW> >& i1_pld_strm,
              hls::stream<ap_uint<PW> >& i2_pld_strm,
              hls::stream<ap_uint<PW> >& i3_pld_strm,
              hls::stream<ap_uint<PW> >& i4_pld_strm,
              hls::stream<ap_uint<PW> >& i5_pld_strm,
              hls::stream<ap_uint<PW> >& i6_pld_strm,
              hls::stream<ap_uint<PW> >& i7_pld_strm,
              hls::stream<ap_uint<HASHW> >& i0_hash_strm,
              hls::stream<ap_uint<HASHW> >& i1_hash_strm,
              hls::stream<ap_uint<HASHW> >& i2_hash_strm,
              hls::stream<ap_uint<HASHW> >& i3_hash_strm,
              hls::stream<ap_uint<HASHW> >& i4_hash_strm,
              hls::stream<ap_uint<HASHW> >& i5_hash_strm,
              hls::stream<ap_uint<HASHW> >& i6_hash_strm,
              hls::stream<ap_uint<HASHW> >& i7_hash_strm,
              hls::stream<bool>& i0_e_strm,
              hls::stream<bool>& i1_e_strm,
              hls::stream<bool>& i2_e_strm,
              hls::stream<bool>& i3_e_strm,
              hls::stream<bool>& i4_e_strm,
              hls::stream<bool>& i5_e_strm,
              hls::stream<bool>& i6_e_strm,
              hls::stream<bool>& i7_e_strm,
              hls::stream<ap_uint<KEYW> >& o_key_strm,
              hls::stream<ap_uint<PW> >& o_pld_strm,
              hls::stream<ap_uint<HASHW> >& o_hash_strm,
              hls::stream<bool>& o_e_strm) {
    ap_uint<KEYW> key_arry[8];
#pragma HLS array_partition variable = key_arry dim = 1
    ap_uint<PW> pld_arry[8];
#pragma HLS array_partition variable = pld_arry dim = 1
    ap_uint<HASHW> hash_val_arry[8];
#pragma HLS array_partition variable = hash_val_arry dim = 1
    ap_uint<8> empty_e = 0;
    ;
    ap_uint<8> rd_e = 0;
    ;
    ap_uint<8> last = 0;
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
LOOP_MERGE8_1:
    do {
#pragma HLS loop_tripcount min = 1 max = 5000
#pragma HLS PIPELINE II = 1
        empty_e[0] = !i0_e_strm.empty() && !last[0];
        empty_e[1] = !i1_e_strm.empty() && !last[1];
        empty_e[2] = !i2_e_strm.empty() && !last[2];
        empty_e[3] = !i3_e_strm.empty() && !last[3];
        empty_e[4] = !i4_e_strm.empty() && !last[4];
        empty_e[5] = !i5_e_strm.empty() && !last[5];
        empty_e[6] = !i6_e_strm.empty() && !last[6];
        empty_e[7] = !i7_e_strm.empty() && !last[7];
        rd_e = mul_ch_read(empty_e);
        if (rd_e[0]) {
            key_arry[0] = i0_key_strm.read();
            pld_arry[0] = i0_pld_strm.read();
            hash_val_arry[0] = i0_hash_strm.read();
            last[0] = i0_e_strm.read();
        }
        if (rd_e[1]) {
            key_arry[1] = i1_key_strm.read();
            pld_arry[1] = i1_pld_strm.read();
            hash_val_arry[1] = i1_hash_strm.read();
            last[1] = i1_e_strm.read();
        }
        if (rd_e[2]) {
            key_arry[2] = i2_key_strm.read();
            pld_arry[2] = i2_pld_strm.read();
            hash_val_arry[2] = i2_hash_strm.read();
            last[2] = i2_e_strm.read();
        }
        if (rd_e[3]) {
            key_arry[3] = i3_key_strm.read();
            pld_arry[3] = i3_pld_strm.read();
            hash_val_arry[3] = i3_hash_strm.read();
            last[3] = i3_e_strm.read();
        }
        if (rd_e[4]) {
            key_arry[4] = i4_key_strm.read();
            pld_arry[4] = i4_pld_strm.read();
            hash_val_arry[4] = i4_hash_strm.read();
            last[4] = i4_e_strm.read();
        }
        if (rd_e[5]) {
            key_arry[5] = i5_key_strm.read();
            pld_arry[5] = i5_pld_strm.read();
            hash_val_arry[5] = i5_hash_strm.read();
            last[5] = i5_e_strm.read();
        }
        if (rd_e[6]) {
            key_arry[6] = i6_key_strm.read();
            pld_arry[6] = i6_pld_strm.read();
            hash_val_arry[6] = i6_hash_strm.read();
            last[6] = i6_e_strm.read();
        }
        if (rd_e[7]) {
            key_arry[7] = i7_key_strm.read();
            pld_arry[7] = i7_pld_strm.read();
            hash_val_arry[7] = i7_hash_strm.read();
            last[7] = i7_e_strm.read();
        }
        // only support 8 channels, 4 channels and 2 channels
        ap_uint<3> id = mux<8>(rd_e);
        ap_uint<KEYW> key = key_arry[id];
        ap_uint<PW> pld = pld_arry[id];
        ap_uint<HASHW> hash_val = hash_val_arry[id];
        bool valid_n = last[id];
        if (!valid_n && rd_e != 0) {
#ifndef __SYNTHESIS__
            cnt++;
#endif
            o_key_strm.write(key);
            o_pld_strm.write(pld);
            o_hash_strm.write(hash_val);
            o_e_strm.write(false);
        }
    } while (last != 255);
    o_e_strm.write(true);
}

template <int KEYW, int PW, int HASHW>
void merge1_1_wrapper(hls::stream<ap_uint<KEYW> >& i_key_strm,
                      hls::stream<ap_uint<PW> >& i_pld_strm,
                      hls::stream<ap_uint<HASHW> >& i_hash_strm,
                      hls::stream<bool>& i_e_strm,
                      hls::stream<ap_uint<KEYW> >& o_key_strm,
                      hls::stream<ap_uint<PW> >& o_pld_strm,
                      hls::stream<ap_uint<HASHW> >& o_hash_strm,
                      hls::stream<bool>& o_e_strm) {
    for (int r = 0; r < 3; ++r) {
        merge1_1<KEYW, PW, HASHW>(i_key_strm, i_pld_strm, i_hash_strm, i_e_strm, o_key_strm, o_pld_strm, o_hash_strm,
                                  o_e_strm);
    }
}

template <int KEYW, int PW, int HASHW>
void merge2_1_wrapper(hls::stream<ap_uint<KEYW> >& i0_key_strm,
                      hls::stream<ap_uint<KEYW> >& i1_key_strm,
                      hls::stream<ap_uint<PW> >& i0_pld_strm,
                      hls::stream<ap_uint<PW> >& i1_pld_strm,
                      hls::stream<ap_uint<HASHW> >& i0_hash_strm,
                      hls::stream<ap_uint<HASHW> >& i1_hash_strm,
                      hls::stream<bool>& i0_e_strm,
                      hls::stream<bool>& i1_e_strm,
                      hls::stream<ap_uint<KEYW> >& o_key_strm,
                      hls::stream<ap_uint<PW> >& o_pld_strm,
                      hls::stream<ap_uint<HASHW> >& o_hash_strm,
                      hls::stream<bool>& o_e_strm) {
    for (int r = 0; r < 3; r++) {
        merge2_1<KEYW, PW, HASHW>(i0_key_strm, i1_key_strm, i0_pld_strm, i1_pld_strm, i0_hash_strm, i1_hash_strm,
                                  i0_e_strm, i1_e_strm, o_key_strm, o_pld_strm, o_hash_strm, o_e_strm);
    }
}

template <int KEYW, int PW, int HASHW>
void merge4_1_wrapper(hls::stream<ap_uint<KEYW> >& i0_key_strm,
                      hls::stream<ap_uint<KEYW> >& i1_key_strm,
                      hls::stream<ap_uint<KEYW> >& i2_key_strm,
                      hls::stream<ap_uint<KEYW> >& i3_key_strm,
                      hls::stream<ap_uint<PW> >& i0_pld_strm,
                      hls::stream<ap_uint<PW> >& i1_pld_strm,
                      hls::stream<ap_uint<PW> >& i2_pld_strm,
                      hls::stream<ap_uint<PW> >& i3_pld_strm,
                      hls::stream<ap_uint<HASHW> >& i0_hash_strm,
                      hls::stream<ap_uint<HASHW> >& i1_hash_strm,
                      hls::stream<ap_uint<HASHW> >& i2_hash_strm,
                      hls::stream<ap_uint<HASHW> >& i3_hash_strm,
                      hls::stream<bool>& i0_e_strm,
                      hls::stream<bool>& i1_e_strm,
                      hls::stream<bool>& i2_e_strm,
                      hls::stream<bool>& i3_e_strm,
                      hls::stream<ap_uint<KEYW> >& o_key_strm,
                      hls::stream<ap_uint<PW> >& o_pld_strm,
                      hls::stream<ap_uint<HASHW> >& o_hash_strm,
                      hls::stream<bool>& o_e_strm) {
    for (int r = 0; r < 3; r++) {
        merge4_1<KEYW, PW, HASHW>(i0_key_strm, i1_key_strm, i2_key_strm, i3_key_strm, i0_pld_strm, i1_pld_strm,
                                  i2_pld_strm, i3_pld_strm, i0_hash_strm, i1_hash_strm, i2_hash_strm, i3_hash_strm,
                                  i0_e_strm, i1_e_strm, i2_e_strm, i3_e_strm, o_key_strm, o_pld_strm, o_hash_strm,
                                  o_e_strm);
    }
}

// ------------------------------------------------------------

/// @brief scan small table to build bitmap
template <int HASHW, int KEYW, int PW, int ARW>
void build_bitmap(hls::stream<ap_uint<HASHW> >& i_hash_strm,
                  hls::stream<ap_uint<KEYW> >& i_key_strm,
                  hls::stream<ap_uint<PW> >& i_pld_strm,
                  hls::stream<bool>& i_e_strm,
#ifdef URAM_SPLITTING
                  ap_uint<72>* bit_vector0,
                  ap_uint<72>* bit_vector1,
                  ap_uint<72>* bit_vector2,
                  ap_uint<72>* bit_vector3
#else  // !defined(URAM_SPLITTING)
                  ap_uint<72>* bit_vector
#endif // !defined(URAM_SPLITTING)
                  ) {
    const int HASH_DEPTH = 1 << (HASHW - 2);
    ap_uint<HASHW - 1> array_idx_r0 = 0xffffffff;
    ap_uint<HASHW - 1> array_idx_r1 = 0xffffffff;
    ap_uint<HASHW - 1> array_idx_r2 = 0xffffffff;
    ap_uint<HASHW - 1> array_idx_r3 = 0xffffffff;
    ap_uint<HASHW - 1> array_idx_r4 = 0xffffffff;
    ap_uint<HASHW - 1> array_idx_r5 = 0xffffffff;
    ap_uint<HASHW - 1> array_idx_r6 = 0xffffffff;
    ap_uint<HASHW - 1> array_idx_r7 = 0xffffffff;
    ap_uint<72> array_val_r0 = 0;
    ap_uint<72> array_val_r1 = 0;
    ap_uint<72> array_val_r2 = 0;
    ap_uint<72> array_val_r3 = 0;
    ap_uint<72> array_val_r4 = 0;
    ap_uint<72> array_val_r5 = 0;
    ap_uint<72> array_val_r6 = 0;
    ap_uint<72> array_val_r7 = 0;

    ap_uint<ARW> base_addr = 0;
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
    unsigned int max_col = 0;
#endif
INIT_LOOP:
    for (int i = 0; i < HASH_DEPTH; i++) {
#pragma HLS PIPELINE II = 1
#ifdef URAM_SPLITTING
#pragma HLS dependence variable = bit_vector0 inter false
#pragma HLS dependence variable = bit_vector1 inter false
#pragma HLS dependence variable = bit_vector2 inter false
#pragma HLS dependence variable = bit_vector3 inter false
#else // !defined(URAM_SPLITTING)
#pragma HLS dependence variable = bit_vector inter false
#endif // !defined(URAM_SPLITTING)
        write_bit_vector(i, 0);
    }
    bool last = i_e_strm.read();
    const int idx_arry[4] = {0, 18, 36, 54};
BITMAP_OFFSET_LOOP:
    while (!last) {
#pragma HLS PIPELINE II = 1
#ifdef URAM_SPLITTING
#pragma HLS dependence variable = bit_vector0 inter false
#pragma HLS dependence variable = bit_vector1 inter false
#pragma HLS dependence variable = bit_vector2 inter false
#pragma HLS dependence variable = bit_vector3 inter false
#else // !defined(URAM_SPLITTING)
#pragma HLS dependence variable = bit_vector inter false
#endif // !defined(URAM_SPLITTING)
        if (!i_e_strm.empty()) {
            ap_uint<HASHW> hash_val = i_hash_strm.read();
            // discard
            ap_uint<KEYW> key = i_key_strm.read();
            ap_uint<PW> pld = i_pld_strm.read();

            last = i_e_strm.read();

            ap_uint<2> bit_idx = hash_val(1, 0);
            ap_uint<7> idx = idx_arry[bit_idx];
            ap_uint<HASHW - 2> array_idx = hash_val(HASHW - 1, 2);

            ap_uint<72> elem;
            if (array_idx == array_idx_r0) {
                elem = array_val_r0;
            } else if (array_idx == array_idx_r1) {
                elem = array_val_r1;
            } else if (array_idx == array_idx_r2) {
                elem = array_val_r2;
            } else if (array_idx == array_idx_r3) {
                elem = array_val_r3;
            } else if (array_idx == array_idx_r4) {
                elem = array_val_r4;
            } else if (array_idx == array_idx_r5) {
                elem = array_val_r5;
            } else if (array_idx == array_idx_r6) {
                elem = array_val_r6;
            } else if (array_idx == array_idx_r7) {
                elem = array_val_r7;
            } else {
                read_bit_vector(array_idx, elem);
            }
            /* XXX the following code causes ii = 2 with array_idx_r1
            ap_uint<18> old_val = elem(idx+17, idx);
            ap_uint<72> new_elem = elem;
            new_elem(idx+17,idx) = old_val + 1;
            */

            ap_uint<18> v0 = elem(17, 0);
            ap_uint<18> v1 = elem(35, 18);
            ap_uint<18> v2 = elem(53, 36);
            ap_uint<18> v3 = elem(71, 54);

            ap_uint<18> v0a = (bit_idx == 0) ? ap_uint<18>(v0 + 1) : v0;
            ap_uint<18> v1a = (bit_idx == 1) ? ap_uint<18>(v1 + 1) : v1;
            ap_uint<18> v2a = (bit_idx == 2) ? ap_uint<18>(v2 + 1) : v2;
            ap_uint<18> v3a = (bit_idx == 3) ? ap_uint<18>(v3 + 1) : v3;

            ap_uint<72> new_elem;
            new_elem(17, 0) = v0a;
            new_elem(35, 18) = v1a;
            new_elem(53, 36) = v2a;
            new_elem(71, 54) = v3a;
            ;

            ap_uint<ARW> old_val = (bit_idx == 0) ? v0 : ((bit_idx == 1) ? v1 : ((bit_idx == 2) ? v2 : v3));

            array_val_r7 = array_val_r6;
            array_val_r6 = array_val_r5;
            array_val_r5 = array_val_r4;
            array_val_r4 = array_val_r3;
            array_val_r3 = array_val_r2;
            array_val_r2 = array_val_r1;
            array_val_r1 = array_val_r0;
            array_val_r0 = new_elem;

            array_idx_r7 = array_idx_r6;
            array_idx_r6 = array_idx_r5;
            array_idx_r5 = array_idx_r4;
            array_idx_r4 = array_idx_r3;
            array_idx_r3 = array_idx_r2;
            array_idx_r2 = array_idx_r1;
            array_idx_r1 = array_idx_r0;
            array_idx_r0 = array_idx;

            write_bit_vector(array_idx, new_elem);
#ifndef __SYNTHESIS__
            cnt++;
            if (old_val > max_col) max_col = old_val;
#endif
        }
    }
#ifndef __SYNTHESIS__
    std::cout << "Get " << cnt << " to build bitmap" << std::endl;
    std::cout << "collision probility " << max_col << std::endl;
#endif
BITMAP_ADDR_LOOP:
    for (int i = 0; i < HASH_DEPTH; i++) {
#pragma HLS PIPELINE II = 1
#ifdef URAM_SPLITTING
#pragma HLS dependence variable = bit_vector0 inter false
#pragma HLS dependence variable = bit_vector1 inter false
#pragma HLS dependence variable = bit_vector2 inter false
#pragma HLS dependence variable = bit_vector3 inter false
#else // !defined(URAM_SPLITTING)
#pragma HLS dependence variable = bit_vector inter false
#endif // !defined(URAM_SPLITTING)
        ap_uint<72> elem;
        read_bit_vector(i, elem);
        ap_uint<ARW> val_0 = elem(17, 0);
        ap_uint<ARW> val_1 = elem(35, 18);
        ap_uint<ARW> val_2 = elem(53, 36);
        ap_uint<ARW> val_3 = elem(71, 54);
        ap_uint<ARW> sum_0 = val_0;
        ap_uint<ARW> sum_1 = val_0 + val_1;
        ap_uint<ARW> sum_2 = val_0 + val_1 + val_2;
        ap_uint<ARW> sum_3 = val_0 + val_1 + val_2 + val_3;
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "i=" << i << " ,coll_nm_0=" << val_0 << std::endl;
        std::cout << "i=" << i << " ,coll_nm_1=" << val_1 << std::endl;
        std::cout << "i=" << i << " ,coll_nm_2=" << val_2 << std::endl;
        std::cout << "i=" << i << " ,coll_nm_3=" << val_3 << std::endl;
#endif
#endif
        ap_uint<72> head = 0;
        head(17, 0) = base_addr;
        head(35, 18) = base_addr + sum_0;
        head(53, 36) = base_addr + sum_1;
        head(71, 54) = base_addr + sum_2;
        base_addr = base_addr + sum_3;
        write_bit_vector(i, head);
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "i=" << i << std::endl;
        ap_int<18> a = head(17, 0);
        std::cout << " ,head[0]=" << std::hex << a << std::endl;
        ap_int<18> a1 = head(35, 18);
        std::cout << " ,head[1]=" << std::hex << a1 << std::endl;
        ap_uint<18> a2 = head(53, 36);
        std::cout << " ,head[2]=" << std::hex << a2 << std::endl;
        ap_uint<18> a3 = head(71, 54);
        std::cout << " ,head[3]=" << std::hex << a3 << std::endl;
        ;
#endif
#endif
    }
}

/// @brief Read data from multiple channel
/// Build hash table using BRAM/URAM, cache small table to HBM.
template <int HASHW, int KEYW, int PW, int S_PW, int ARW>
void build_unit(hls::stream<ap_uint<HASHW> >& i_hash_strm,
                hls::stream<ap_uint<KEYW> >& i_key_strm,
                hls::stream<ap_uint<PW> >& i_pld_strm,
                hls::stream<bool>& i_e_strm,
                hls::stream<ap_uint<KEYW + S_PW> >& o_row_strm,
                hls::stream<ap_uint<ARW> >& o_addr_strm,
                hls::stream<bool>& o_e_strm,
#ifdef URAM_SPLITTING
                ap_uint<72>* bit_vector0,
                ap_uint<72>* bit_vector1,
                ap_uint<72>* bit_vector2,
                ap_uint<72>* bit_vector3
#else  // !defined(URAM_SPLITTING)
                ap_uint<72>* bit_vector
#endif // !defined(URAM_SPLITTING)
                ) {
#pragma HLS inline off
    ap_uint<ARW> head = 0;
    ap_uint<HASHW - 1> array_idx_r0 = 0xffffffff;
    ap_uint<HASHW - 1> array_idx_r1 = 0xffffffff;
    ap_uint<HASHW - 1> array_idx_r2 = 0xffffffff;
    ap_uint<HASHW - 1> array_idx_r3 = 0xffffffff;
    ap_uint<HASHW - 1> array_idx_r4 = 0xffffffff;
    ap_uint<HASHW - 1> array_idx_r5 = 0xffffffff;
    ap_uint<HASHW - 1> array_idx_r6 = 0xffffffff;
    ap_uint<HASHW - 1> array_idx_r7 = 0xffffffff;
    ap_uint<72> array_val_r0 = 0;
    ap_uint<72> array_val_r1 = 0;
    ap_uint<72> array_val_r2 = 0;
    ap_uint<72> array_val_r3 = 0;
    ap_uint<72> array_val_r4 = 0;
    ap_uint<72> array_val_r5 = 0;
    ap_uint<72> array_val_r6 = 0;
    ap_uint<72> array_val_r7 = 0;
    const int idx_arry[4] = {0, 18, 36, 54};
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
    bool last = i_e_strm.read();
LOOP_BUILD:
    while (!last) {
#pragma HLS loop_tripcount min = 1 max = 5000
#pragma HLS PIPELINE II = 1
#ifdef URAM_SPLITTING
#pragma HLS dependence variable = bit_vector0 inter false
#pragma HLS dependence variable = bit_vector1 inter false
#pragma HLS dependence variable = bit_vector2 inter false
#pragma HLS dependence variable = bit_vector3 inter false
#else // !defined(URAM_SPLITTING)
#pragma HLS dependence variable = bit_vector inter false
#endif // !defined(URAM_SPLITTING)
        if (!i_e_strm.empty()) {
            ap_uint<KEYW> key = i_key_strm.read();
            ap_uint<PW> pld = i_pld_strm.read();
            ap_uint<HASHW> hash_val = i_hash_strm.read();
            last = i_e_strm.read();
            // only support 8 channels, 4 channels and 2 channels
            ap_uint<KEYW + S_PW> stb_row = 0;
            if (S_PW > 0) stb_row(S_PW - 1, 0) = pld(S_PW - 1, 0);
            stb_row(S_PW + KEYW - 1, S_PW) = key;
            // avoid read after write
            ap_uint<2> bit_idx = hash_val(1, 0);
            ap_uint<7> idx = idx_arry[bit_idx];
            ap_uint<HASHW - 1> array_idx = hash_val(HASHW - 1, 2);

            ap_uint<72> elem;
            if (array_idx == array_idx_r0) {
                elem = array_val_r0;
            } else if (array_idx == array_idx_r1) {
                elem = array_val_r1;
            } else if (array_idx == array_idx_r2) {
                elem = array_val_r2;
            } else if (array_idx == array_idx_r3) {
                elem = array_val_r3;
            } else if (array_idx == array_idx_r4) {
                elem = array_val_r4;
            } else if (array_idx == array_idx_r5) {
                elem = array_val_r5;
            } else if (array_idx == array_idx_r6) {
                elem = array_val_r6;
            } else if (array_idx == array_idx_r7) {
                elem = array_val_r7;
            } else {
                read_bit_vector(array_idx, elem);
            }
            /* XXX the following code causes ii = 2 with array_idx_r1
            ap_uint<18> old_val = elem(idx+17, idx);
            ap_uint<72> new_elem = elem;
            new_elem(idx+17,idx) = old_val + 1;
            */

            ap_uint<18> v0 = elem(17, 0);
            ap_uint<18> v1 = elem(35, 18);
            ap_uint<18> v2 = elem(53, 36);
            ap_uint<18> v3 = elem(71, 54);

            ap_uint<18> v0a = (bit_idx == 0) ? ap_uint<18>(v0 + 1) : v0;
            ap_uint<18> v1a = (bit_idx == 1) ? ap_uint<18>(v1 + 1) : v1;
            ap_uint<18> v2a = (bit_idx == 2) ? ap_uint<18>(v2 + 1) : v2;
            ap_uint<18> v3a = (bit_idx == 3) ? ap_uint<18>(v3 + 1) : v3;

            ap_uint<72> new_elem;
            new_elem(17, 0) = v0a;
            new_elem(35, 18) = v1a;
            new_elem(53, 36) = v2a;
            new_elem(71, 54) = v3a;
            ;

            ap_uint<ARW> old_val = (bit_idx == 0) ? v0 : ((bit_idx == 1) ? v1 : ((bit_idx == 2) ? v2 : v3));

            array_val_r7 = array_val_r6;
            array_val_r6 = array_val_r5;
            array_val_r5 = array_val_r4;
            array_val_r4 = array_val_r3;
            array_val_r3 = array_val_r2;
            array_val_r2 = array_val_r1;
            array_val_r1 = array_val_r0;
            array_val_r0 = new_elem;

            array_idx_r7 = array_idx_r6;
            array_idx_r6 = array_idx_r5;
            array_idx_r5 = array_idx_r4;
            array_idx_r4 = array_idx_r3;
            array_idx_r3 = array_idx_r2;
            array_idx_r2 = array_idx_r1;
            array_idx_r1 = array_idx_r0;
            array_idx_r0 = array_idx;

            write_bit_vector(array_idx, new_elem);

            ap_uint<ARW> o_addr = old_val;
#ifndef __SYNTHESIS__
            cnt++;
#endif
            o_addr_strm.write(o_addr);
            o_row_strm.write(stb_row);
            o_e_strm.write(false);
        }
    }
#ifndef __SYNTHESIS__
    std::cout << "build unit write " << cnt << " row to ssbm" << std::endl;
#ifdef DEBUG
    int H = 1 << (HASHW - 2);
    for (int i = 0; i < H; i++) {
        ap_uint<72> t;
        read_bit_vector(i, t);
        std::cout << "i = " << i << " ,addr = " << t << std::endl;
    }
#endif
#endif
    o_e_strm.write(true);
}

/// @brief probe the hash table and output address which hash same hash_value
template <int HASHW, int KEYW, int PW, int ARW>
void probe_head(hls::stream<ap_uint<HASHW> >& i_hash_strm,
                hls::stream<ap_uint<KEYW> >& i_key_strm,
                hls::stream<ap_uint<PW> >& i_pld_strm,
                hls::stream<bool>& i_e_strm,
                hls::stream<ap_uint<ARW> >& o_addr_strm,
                hls::stream<ap_uint<18> >& o_nm0_strm,
                hls::stream<ap_uint<18> >& o_nm1_strm,
                hls::stream<ap_uint<KEYW> >& o_key_strm,
                hls::stream<ap_uint<PW> >& o_pld_strm,
                hls::stream<bool>& o_e0_strm,
                hls::stream<bool>& o_e1_strm,
#ifdef URAM_SPLITTING
                ap_uint<72>* bit_vector0,
                ap_uint<72>* bit_vector1,
                ap_uint<72>* bit_vector2,
                ap_uint<72>* bit_vector3
#else  // !defined(URAM_SPLITTING)
                ap_uint<72>* bit_vector
#endif // !defined(URAM_SPLITTING)
                ) {
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
    bool last = i_e_strm.read();
    const int idx_arry[4] = {0, 18, 36, 54};
LOOP_PROBE:
    while (!last) {
#pragma HLS loop_tripcount min = 1 max = 5000
#pragma HLS PIPELINE II = 1
#ifdef URAM_SPLITTING
#pragma HLS dependence variable = bit_vector0 inter false
#pragma HLS dependence variable = bit_vector1 inter false
#pragma HLS dependence variable = bit_vector2 inter false
#pragma HLS dependence variable = bit_vector3 inter false
#else // !defined(URAM_SPLITTING)
#pragma HLS dependence variable = bit_vector inter false
#endif // !defined(URAM_SPLITTING)
        // read select field from stream and store them on local ram.
        ap_uint<HASHW> hash_val = i_hash_strm.read();
        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<PW> pld = i_pld_strm.read();
        last = i_e_strm.read();
        ap_uint<HASHW - 2> array_idx = hash_val(HASHW - 1, 2);
        ap_uint<2> bit_idx = hash_val(1, 0);
        ap_uint<72> pre = 0;
        if (array_idx > 0) {
            read_bit_vector(array_idx - 1, pre);
        }
        ap_uint<72> cur;
        read_bit_vector(array_idx, cur);
        ap_uint<ARW> pre_addr = 0;
        ap_uint<ARW> cur_addr = 0;
        ap_uint<7> idx = idx_arry[bit_idx];
        if (bit_idx == 0) {
            pre_addr = pre(71, 54);
        } else {
            pre_addr = cur(idx - 1, idx - 18);
        }
        cur_addr = cur(idx + 17, idx);
        ap_uint<18> nm = cur_addr - pre_addr;
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "key = " << key << ", hash_val = " << hash_val << ", nm = " << nm << ", pre = " << pre_addr
                  << std::endl;
#endif
#endif
        // optimization: add bloom filter to filter out more row
        if (nm > 0) {
#ifndef __SYNTHESIS__
            cnt++;
#endif
            o_addr_strm.write(pre_addr);
            o_nm0_strm.write(nm);
            o_nm1_strm.write(nm);
            o_key_strm.write(key);
            o_pld_strm.write(pld);
            o_e0_strm.write(false);
            o_e1_strm.write(false);
        }
    }
#ifndef __SYNTHESIS__
    std::cout << "probe unit read " << cnt << " block from ssbm" << std::endl;
    ;
#endif
    o_e0_strm.write(true);
    o_e1_strm.write(true);
}

template <int ARW, int WCOLLISION>
void probe_addr_gen(hls::stream<ap_uint<ARW> >& i_addr_strm,
                    hls::stream<ap_uint<WCOLLISION> >& i_nm_strm,
                    hls::stream<bool>& i_e_strm,
                    hls::stream<ap_uint<ARW> >& o_addr_strm,
                    hls::stream<bool>& o_e_strm) {
    ap_uint<WCOLLISION> nm = 0;
    ap_uint<ARW> addr = 0;
    bool last = i_e_strm.read();
    while (!last) {
#pragma HLS pipeline II = 1
        if (nm == 0) {
            addr = i_addr_strm.read();
            last = i_e_strm.read();
            nm = i_nm_strm.read() - 1;
            o_addr_strm.write(addr++);
            o_e_strm.write(0);
        } else {
            nm--;
            o_addr_strm.write(addr++);
            o_e_strm.write(0);
        }
    }
    while (nm != 0) {
#pragma HLS pipeline II = 1
        nm--;
        o_addr_strm.write(addr++);
        o_e_strm.write(0);
    }
    o_e_strm.write(1);
}

template <int HASHW, int KEYW, int PW, int ARW>
void probe_unit(hls::stream<ap_uint<HASHW> >& i_hash_strm,
                hls::stream<ap_uint<KEYW> >& i_key_strm,
                hls::stream<ap_uint<PW> >& i_pld_strm,
                hls::stream<bool>& i_e_strm,
                hls::stream<ap_uint<ARW> >& o_addr_strm,
                hls::stream<ap_uint<18> >& o_nm_strm,
                hls::stream<ap_uint<KEYW> >& o_key_strm,
                hls::stream<ap_uint<PW> >& o_pld_strm,
                hls::stream<bool>& o_e0_strm,
                hls::stream<bool>& o_e1_strm,
#ifdef URAM_SPLITTING
                ap_uint<72>* bit_vector0,
                ap_uint<72>* bit_vector1,
                ap_uint<72>* bit_vector2,
                ap_uint<72>* bit_vector3
#else  // !defined(URAM_SPLITTING)
                ap_uint<72>* bit_vector
#endif // !defined(URAM_SPLITTING)
                ) {
    hls::stream<ap_uint<18> > nm_strm;
#pragma HLS stream variable = nm_strm depth = 8
    hls::stream<ap_uint<ARW> > addr_strm;
#pragma HLS stream variable = addr_strm depth = 8
    hls::stream<bool> e_strm;
#pragma HLS stream variable = e_strm depth = 8
#pragma HLS dataflow
    probe_head<HASHW, KEYW, PW, ARW>(i_hash_strm, i_key_strm, i_pld_strm, i_e_strm, addr_strm, o_nm_strm, nm_strm,
                                     o_key_strm, o_pld_strm, e_strm, o_e1_strm,
#ifdef URAM_SPLITTING
                                     bit_vector0, bit_vector1, bit_vector2, bit_vector3
#else  // !defined(URAM_SPLITTING)
                                     bit_vector
#endif // !defined(URAM_SPLITTING)
                                     );
    probe_addr_gen<ARW, 18>(addr_strm, nm_strm, e_strm, o_addr_strm, o_e0_strm);
}

template <int HASHW, int KEYW, int PW, int S_PW, int ARW>
void build_probe_wrapper(hls::stream<ap_uint<HASHW> >& i_hash_strm,
                         hls::stream<ap_uint<KEYW> >& i_key_strm,
                         hls::stream<ap_uint<PW> >& i_pld_strm,
                         hls::stream<bool>& i_e_strm,
                         hls::stream<ap_uint<KEYW + S_PW> >& o_row_strm,
                         hls::stream<ap_uint<KEYW> >& o_key_strm,
                         hls::stream<ap_uint<PW> >& o_pld_strm,
                         hls::stream<ap_uint<ARW> >& o_addr_strm,
                         hls::stream<ap_uint<18> >& o_nm0_strm,
                         hls::stream<bool>& o_e0_strm,
                         hls::stream<bool>& o_e1_strm,
#ifdef URAM_SPLITTING
                         ap_uint<72>* bit_vector0,
                         ap_uint<72>* bit_vector1,
                         ap_uint<72>* bit_vector2,
                         ap_uint<72>* bit_vector3
#else
                         ap_uint<72>* bit_vector
#endif
                         ) {

    build_bitmap<HASHW, KEYW, PW, ARW>(                //
        i_hash_strm, i_key_strm, i_pld_strm, i_e_strm, //
#ifdef URAM_SPLITTING
        bit_vector0, bit_vector1, bit_vector2, bit_vector3
#else
        bit_vector
#endif
        );

    build_unit<HASHW, KEYW, PW, S_PW, ARW>(            //
        i_hash_strm, i_key_strm, i_pld_strm, i_e_strm, //
        o_row_strm, o_addr_strm, o_e0_strm,            //
#ifdef URAM_SPLITTING
        bit_vector0, bit_vector1, bit_vector2, bit_vector3
#else
        bit_vector
#endif
        );

    probe_unit<HASHW, KEYW, PW, ARW>(                                          //
        i_hash_strm, i_key_strm, i_pld_strm, i_e_strm,                         //
        o_addr_strm, o_nm0_strm, o_key_strm, o_pld_strm, o_e0_strm, o_e1_strm, //
#ifdef URAM_SPLITTING
        bit_vector0, bit_vector1, bit_vector2, bit_vector3
#else
        bit_vector
#endif
        );
}

// ------------------------------------------------------------

template <int width, int ARW, int RW>
void access_srow_w(ap_uint<width>* stb_buf,
                   hls::stream<ap_uint<ARW> >& i_addr_strm,
                   hls::stream<ap_uint<RW> >& i_row_strm,
                   hls::stream<bool>& i_e_strm,
                   hls::stream<ap_uint<RW> >& o_row_strm) {
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
    const int sz = RW >> 3;
    bool last = i_e_strm.read();
    while (!last) {
#pragma HLS pipeline II = 1
        last = i_e_strm.read();
        ap_uint<ARW> addr = i_addr_strm.read();
        ap_uint<RW> srow = i_row_strm.read();
        stb_buf[addr] = srow;
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "write addr =" << addr << ", row=" << srow << std::endl;
#endif
        cnt++;
#endif
    }
#ifndef __SYNTHESIS__
    std::cout << "SSBM write " << cnt << " rows" << std::endl;
#endif
}

template <int width, int ARW, int RW>
void access_srow_r(ap_uint<width>* stb_buf,
                   hls::stream<ap_uint<ARW> >& i_addr_strm,
                   hls::stream<ap_uint<RW> >& i_row_strm,
                   hls::stream<bool>& i_e_strm,
                   hls::stream<ap_uint<RW> >& o_row_strm) {
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
    const int sz = RW >> 3;
    bool last = i_e_strm.read();
READ_BUFF_LOOP:
    while (!last) {
#pragma HLS pipeline II = 1
        if (!i_e_strm.empty()) {
            last = i_e_strm.read();
            ap_uint<ARW> addr = i_addr_strm.read();
            ap_int<RW> row = stb_buf[addr];
            o_row_strm.write(row);
#ifndef __SYNTHESIS__
            cnt++;
#ifdef DEBUG
            std::cout << "read addr =" << addr << ", row=" << row << std::endl;
#endif
#endif
        }
    }
#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "SSBM read " << cnt << " block" << std::endl;
#endif
    cnt = 0;
#endif
}

template <int width, int ARW, int RW>
void access_srow(ap_uint<width>* stb_buf,
                 hls::stream<ap_uint<ARW> >& i_addr_strm,
                 hls::stream<ap_uint<RW> >& i_row_strm,
                 hls::stream<bool>& i_e_strm,
                 hls::stream<ap_uint<RW> >& o_row_strm) {
    // XXX for simpler pipeline FSM.
    access_srow_w(stb_buf,                           //
                  i_addr_strm, i_row_strm, i_e_strm, //
                  o_row_strm);
    access_srow_r(stb_buf,                           //
                  i_addr_strm, i_row_strm, i_e_strm, //
                  o_row_strm);
}

// ------------------------------------------------------------

/// @brief compare key, if match output joined row
template <int PW, int S_PW, int B_PW, int KEYW, int WCOLLISION>
void join_unit(hls::stream<ap_uint<KEYW + S_PW> >& i_srow_strm,
               hls::stream<ap_uint<KEYW> >& i_bkey_strm,
               hls::stream<ap_uint<PW> >& i_bpld_strm,
               hls::stream<ap_uint<WCOLLISION> >& i_nm_strm,
               hls::stream<bool>& i_e_strm,
               hls::stream<ap_uint<S_PW + B_PW> >& o_j_strm,
               hls::stream<bool>& o_e_strm) {
    ap_uint<KEYW> s_key = 0;
    ap_uint<KEYW> b_key = 0;
    ap_uint<B_PW> b_pld = 0;
    bool last = i_e_strm.read();
    ap_uint<WCOLLISION> nm = 0;
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
    const int S_PW_R = (S_PW == 0) ? 1 : S_PW;
JOIN_LOOP:
    while (!last) {
#pragma HLS PIPELINE II = 1
        if (nm == 0) {
            last = i_e_strm.read();
            b_pld = i_bpld_strm.read();
            b_key = i_bkey_strm.read();
            nm = i_nm_strm.read();
        } else {
            nm--;
            ap_uint<KEYW + S_PW> stb_row = i_srow_strm.read();
            s_key = stb_row(KEYW + S_PW - 1, S_PW);
            if (s_key == b_key) {
                ap_uint<S_PW + B_PW> j = 0;
                ap_uint<S_PW_R> s_pld = stb_row(S_PW_R - 1, 0);
                if (S_PW > 0) {
                    j(S_PW + B_PW - 1, B_PW) = s_pld;
                }
                j(B_PW - 1, 0) = b_pld;
                o_j_strm.write(j);
                o_e_strm.write(false);
#ifndef __SYNTHESIS__
                cnt++;
#endif
            }
        }
    }
    if (nm != 0) {
    JOIN_CLEAR_LOOP:
        for (int i = 0; i < nm; i++) {
#pragma HLS PIPELINE II = 1
            ap_uint<KEYW + S_PW> stb_row = i_srow_strm.read();
            s_key = stb_row(KEYW + S_PW - 1, S_PW);
            if (s_key == b_key) {
                ap_uint<S_PW + B_PW> j = 0;
                ap_uint<S_PW_R> s_pld = stb_row(S_PW_R - 1, 0);
                if (S_PW > 0) {
                    j(S_PW + B_PW - 1, B_PW) = s_pld;
                }
                j(B_PW - 1, 0) = b_pld;
                o_j_strm.write(j);
                o_e_strm.write(false);
#ifndef __SYNTHESIS__
                cnt++;
#endif
            }
        }
    }
    o_j_strm.write(0);
    o_e_strm.write(true);
#ifndef __SYNTHESIS__
    std::cout << "Join Unit output " << cnt << " rows" << std::endl;
#endif
} // join_unit

// ------------------------------------------------------------

template <int PU, int JW>
void collect_unit(hls::stream<ap_uint<JW> > i_jrow_strm[PU],
                  hls::stream<bool> i_e_strm[PU],
                  hls::stream<ap_uint<JW> >& o_jrow_strm,
                  hls::stream<bool>& o_e_strm) {
    const int MAX = (1 << PU) - 1;
    ap_uint<JW> jrow_arr[PU];
#pragma HLS array_partition variable = jrow_arr dim = 1
    ap_uint<PU> empty_e = 0;
    ;
    ap_uint<PU> last = 0;
    ap_uint<PU> rd_e = 0;
    ;
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
    std::cout << "PU=" << PU << std::endl;
#endif
    do {
#pragma HLS pipeline II = 1
        for (int i = 0; i < PU; i++) {
#pragma HLS unroll
            empty_e[i] = !i_e_strm[i].empty() && !last[i];
        }
        rd_e = mul_ch_read(empty_e);
        for (int i = 0; i < PU; i++) {
#pragma HLS unroll
            if (rd_e[i]) {
                jrow_arr[i] = i_jrow_strm[i].read();
                last[i] = i_e_strm[i].read();
            }
        }
        ap_uint<3> id = mux<PU>(rd_e);
        ap_uint<JW> j = jrow_arr[id];
        bool valid_n = last[id];
        if (!valid_n && rd_e != 0) {
            o_jrow_strm.write(j);
            o_e_strm.write(false);
#ifndef __SYNTHESIS__
            cnt++;
#endif
        }
    } while (last != MAX);
#ifndef __SYNTHESIS__
    std::cout << "Collect " << cnt << " rows" << std::endl;
#endif
    o_e_strm.write(true);
} // collect_unit

} // namespace join_v2
} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/**
 * @brief Multi-PU Hash-Join primitive, using multiple DDR/HBM buffers.
 *
 * The max number of lines of small table is 2M in this design.
 * It is assumed that the hash-conflict is within 512 per bin.
 *
 * This module can accept more than 1 input row per cycle, via multiple
 * input channels.
 * The small table and the big table shares the same input ports,
 * so the width of the payload should be the max of both, while the data
 * should be aligned to the little-end.
 * The small table should be fed TWICE, followed by the big table once.
 *
 * @tparam HASH_MODE 0 for radix and 1 for Jenkin's Lookup3 hash.
 * @tparam KEYW width of key, in bit.
 * @tparam PW width of max payload, in bit.
 * @tparam S_PW width of payload of small table.
 * @tparam B_PW width of payload of big table.
 * @tparam HASHWH number of hash bits used for PU/buffer selection, 1~3.
 * @tparam HASHWL number of hash bits used for hash-table in PU.
 * @tparam ARW width of address, log2(small table max num of rows).
 * @tparam BFW width of buffer.
 * @tparam CH_NM number of input channels, 1,2,4.
 * @tparam BF_W bloom-filter hash width.
 * @tparam EN_BF bloom-filter switch, 0 for off, 1 for on.
 *
 * @param k0_strm_arry input of key columns of both tables.
 * @param p0_strm_arry input of payload columns of both tables.
 * @param e0_strm_arry input of end signal of both tables.
 * @param stb0_buf HBM/DDR buffer of PU0
 * @param stb1_buf HBM/DDR buffer of PU1
 * @param stb2_buf HBM/DDR buffer of PU2
 * @param stb3_buf HBM/DDR buffer of PU3
 * @param stb4_buf HBM/DDR buffer of PU4
 * @param stb5_buf HBM/DDR buffer of PU5
 * @param stb6_buf HBM/DDR buffer of PU6
 * @param stb7_buf HBM/DDR buffer of PU7
 * @param j1_strm output of joined rows.
 * @param e5_strm end signal of joined rows.
 */
template <int HASH_MODE,
          int KEYW,
          int PW,
          int S_PW,
          int B_PW,
          int HASHWH,
          int HASHWL,
          int ARW,
          int BFW,
          int CH_NM,
          int BF_W,
          int EN_BF>
static void hashJoinMPU(hls::stream<ap_uint<KEYW> > k0_strm_arry[CH_NM],
                        hls::stream<ap_uint<PW> > p0_strm_arry[CH_NM],
                        hls::stream<bool> e0_strm_arry[CH_NM],
                        ap_uint<BFW>* stb0_buf,
                        ap_uint<BFW>* stb1_buf,
                        ap_uint<BFW>* stb2_buf,
                        ap_uint<BFW>* stb3_buf,
                        ap_uint<BFW>* stb4_buf,
                        ap_uint<BFW>* stb5_buf,
                        ap_uint<BFW>* stb6_buf,
                        ap_uint<BFW>* stb7_buf,
                        hls::stream<ap_uint<S_PW + B_PW> >& j1_strm,
                        hls::stream<bool>& e5_strm) {
    enum { HDP_J = (1 << (HASHWL - 2)) }; // 4 entries per slot, so -2.
    enum { PU = (1 << HASHWH) };          // high hash for distribution.

#pragma HLS dataflow

    hls::stream<ap_uint<KEYW> > k1_strm_arry[PU];
#pragma HLS stream variable = k1_strm_arry depth = 8
#pragma HLS array_partition variable = k1_strm_arry dim = 0
    hls::stream<ap_uint<PW> > p1_strm_arry[PU];
#pragma HLS stream variable = p1_strm_arry depth = 8
#pragma HLS array_partition variable = p1_strm_arry dim = 0
#pragma HLS bind_storage variable = p1_strm_arry type = fifo impl = srl
    hls::stream<ap_uint<HASHWL> > hash_strm_arry[PU];
#pragma HLS stream variable = hash_strm_arry depth = 8
#pragma HLS array_partition variable = hash_strm_arry dim = 0
    hls::stream<bool> e1_strm_arry[PU];
#pragma HLS stream variable = e1_strm_arry depth = 8
#pragma HLS array_partition variable = e1_strm_arry dim = 0

    // CH_NM >= 1
    hls::stream<ap_uint<KEYW> > k1_strm_arry_c0[PU];
#pragma HLS stream variable = k1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c0 dim = 0
    hls::stream<ap_uint<PW> > p1_strm_arry_c0[PU];
#pragma HLS stream variable = p1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c0 dim = 0
#pragma HLS bind_storage variable = p1_strm_arry_c0 type = fifo impl = srl
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_c0[PU];
#pragma HLS stream variable = hash_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c0 dim = 0
    hls::stream<bool> e1_strm_arry_c0[PU];
#pragma HLS stream variable = e1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c0 dim = 0
    // CH_NM >= 2
    hls::stream<ap_uint<KEYW> > k1_strm_arry_c1[PU];
#pragma HLS stream variable = k1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c1 dim = 0
    hls::stream<ap_uint<PW> > p1_strm_arry_c1[PU];
#pragma HLS stream variable = p1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c1 dim = 0
#pragma HLS bind_storage variable = p1_strm_arry_c1 type = fifo impl = srl
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_c1[PU];
#pragma HLS stream variable = hash_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c1 dim = 0
    hls::stream<bool> e1_strm_arry_c1[PU];
#pragma HLS stream variable = e1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c1 dim = 0
    // CH_NM >= 4
    hls::stream<ap_uint<KEYW> > k1_strm_arry_c2[PU];
#pragma HLS stream variable = k1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c2 dim = 0
    hls::stream<ap_uint<PW> > p1_strm_arry_c2[PU];
#pragma HLS stream variable = p1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c2 dim = 0
#pragma HLS bind_storage variable = p1_strm_arry_c2 type = fifo impl = srl
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_c2[PU];
#pragma HLS stream variable = hash_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c2 dim = 0
    hls::stream<bool> e1_strm_arry_c2[PU];
#pragma HLS stream variable = e1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c2 dim = 0
    hls::stream<ap_uint<KEYW> > k1_strm_arry_c3[PU];
#pragma HLS stream variable = k1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c3 dim = 0
    hls::stream<ap_uint<PW> > p1_strm_arry_c3[PU];
#pragma HLS stream variable = p1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c3 dim = 0
#pragma HLS bind_storage variable = p1_strm_arry_c3 type = fifo impl = srl
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_c3[PU];
#pragma HLS stream variable = hash_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c3 dim = 0
    hls::stream<bool> e1_strm_arry_c3[PU];
#pragma HLS stream variable = e1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c3 dim = 0

    hls::stream<ap_uint<KEYW + S_PW> > w_row_strm[PU];
#pragma HLS stream variable = w_row_strm depth = 8
#pragma HLS array_partition variable = w_row_strm dim = 0
    hls::stream<ap_uint<KEYW + S_PW> > r_row_strm[PU];
#pragma HLS stream variable = r_row_strm depth = 8
    hls::stream<ap_uint<KEYW> > k2_strm_arry[PU];
#pragma HLS stream variable = k2_strm_arry depth = 32
#pragma HLS array_partition variable = k2_strm_arry dim = 0
#pragma HLS bind_storage variable = k2_strm_arry type = fifo impl = srl
    hls::stream<ap_uint<PW> > p2_strm_arry[PU];
#pragma HLS stream variable = p2_strm_arry depth = 32
#pragma HLS array_partition variable = p2_strm_arry dim = 0
#pragma HLS bind_storage variable = p2_strm_arry type = fifo impl = srl
    hls::stream<bool> e2_strm_arry[PU];
#pragma HLS stream variable = e2_strm_arry depth = 8
#pragma HLS array_partition variable = e2_strm_arry dim = 0
    hls::stream<bool> e3_strm_arry[PU];
#pragma HLS stream variable = e3_strm_arry depth = 32
#pragma HLS array_partition variable = e3_strm_arry dim = 0
    hls::stream<bool> e4_strm_arry[PU];
#pragma HLS array_partition variable = e4_strm_arry dim = 0
#pragma HLS stream variable = e4_strm_arry depth = 8
    hls::stream<ap_uint<ARW> > addr_strm[PU];
#pragma HLS stream variable = addr_strm depth = 8
#pragma HLS array_partition variable = addr_strm dim = 0
    hls::stream<ap_uint<18> > nm0_strm_arry[PU];
#pragma HLS stream variable = nm0_strm_arry depth = 32
#pragma HLS array_partition variable = nm0_strm_arry dim = 0
    hls::stream<ap_uint<S_PW + B_PW> > j0_strm_arry[PU];
#pragma HLS stream variable = j0_strm_arry depth = 8
#pragma HLS array_partition variable = j0_strm_arry dim = 0
#pragma HLS bind_storage variable = j0_strm_arry type = fifo impl = srl

#ifndef __SYNTHESIS__
    ap_uint<72>* bit_vector0[PU];
    ap_uint<72>* bit_vector1[PU];
    ap_uint<72>* bit_vector2[PU];
    ap_uint<72>* bit_vector3[PU];
    for (int i = 0; i < PU; i++) {
        bit_vector0[i] = (ap_uint<72>*)malloc((HDP_J >> 2) * sizeof(ap_uint<72>));
        bit_vector1[i] = (ap_uint<72>*)malloc((HDP_J >> 2) * sizeof(ap_uint<72>));
        bit_vector2[i] = (ap_uint<72>*)malloc((HDP_J >> 2) * sizeof(ap_uint<72>));
        bit_vector3[i] = (ap_uint<72>*)malloc((HDP_J >> 2) * sizeof(ap_uint<72>));
    }
#else
    ap_uint<72> bit_vector0[PU][(HDP_J >> 2)];
    ap_uint<72> bit_vector1[PU][(HDP_J >> 2)];
    ap_uint<72> bit_vector2[PU][(HDP_J >> 2)];
    ap_uint<72> bit_vector3[PU][(HDP_J >> 2)];
#pragma HLS array_partition variable = bit_vector0 dim = 1
#pragma HLS bind_storage variable = bit_vector0 type = ram_2p impl = uram
#pragma HLS array_partition variable = bit_vector1 dim = 1
#pragma HLS bind_storage variable = bit_vector1 type = ram_2p impl = uram
#pragma HLS array_partition variable = bit_vector2 dim = 1
#pragma HLS bind_storage variable = bit_vector2 type = ram_2p impl = uram
#pragma HLS array_partition variable = bit_vector3 dim = 1
#pragma HLS bind_storage variable = bit_vector3 type = ram_2p impl = uram
#endif

    // clang-format off
      // -------------|----------|------------|------|------|------|--------
      //   Dispatch   |          |Bloom Filter|Bitmap|Build |Probe |        
      // -------------|          |------------|------|------|------|        
      //   Dispatch   | switcher |Bloom filter|Bitmap|Build |Probe |Collect 
      // -------------|          |------------|------|------|------|        
      //   Dispatch   |          |Bloom filter|Bitmap|Build |Porbe |        
      // -------------|----------|------------|------|------|------|--------
    // clang-format on
    ;

    if (CH_NM >= 1) {
        details::join_v2::dispatch_wrapper<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU, BF_W, EN_BF>(
            k0_strm_arry[0], p0_strm_arry[0], e0_strm_arry[0], k1_strm_arry_c0, p1_strm_arry_c0, hash_strm_arry_c0,
            e1_strm_arry_c0);
    }
    if (CH_NM >= 2) {
        details::join_v2::dispatch_wrapper<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU, BF_W, EN_BF>(
            k0_strm_arry[1], p0_strm_arry[1], e0_strm_arry[1], k1_strm_arry_c1, p1_strm_arry_c1, hash_strm_arry_c1,
            e1_strm_arry_c1);
    }
    if (CH_NM >= 4) {
        details::join_v2::dispatch_wrapper<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU, BF_W, EN_BF>(
            k0_strm_arry[2], p0_strm_arry[2], e0_strm_arry[2], k1_strm_arry_c2, p1_strm_arry_c2, hash_strm_arry_c2,
            e1_strm_arry_c2);
        details::join_v2::dispatch_wrapper<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU, BF_W, EN_BF>(
            k0_strm_arry[3], p0_strm_arry[3], e0_strm_arry[3], k1_strm_arry_c3, p1_strm_arry_c3, hash_strm_arry_c3,
            e1_strm_arry_c3);
    }

    if (CH_NM == 1) {
        for (int p = 0; p < PU; ++p) {
#pragma HLS unroll
            details::join_v2::merge1_1_wrapper(k1_strm_arry_c0[p], p1_strm_arry_c0[p], hash_strm_arry_c0[p],
                                               e1_strm_arry_c0[p], k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p],
                                               e1_strm_arry[p]);
        }
    }
    if (CH_NM == 2) {
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll

            details::join_v2::merge2_1_wrapper(k1_strm_arry_c0[p], k1_strm_arry_c1[p], p1_strm_arry_c0[p],
                                               p1_strm_arry_c1[p], hash_strm_arry_c0[p], hash_strm_arry_c1[p],
                                               e1_strm_arry_c0[p], e1_strm_arry_c1[p], k1_strm_arry[p], p1_strm_arry[p],
                                               hash_strm_arry[p], e1_strm_arry[p]);
        }
    }
    if (CH_NM == 4) {
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            details::join_v2::merge4_1_wrapper(
                k1_strm_arry_c0[p], k1_strm_arry_c1[p], k1_strm_arry_c2[p], k1_strm_arry_c3[p], p1_strm_arry_c0[p],
                p1_strm_arry_c1[p], p1_strm_arry_c2[p], p1_strm_arry_c3[p], hash_strm_arry_c0[p], hash_strm_arry_c1[p],
                hash_strm_arry_c2[p], hash_strm_arry_c3[p], e1_strm_arry_c0[p], e1_strm_arry_c1[p], e1_strm_arry_c2[p],
                e1_strm_arry_c3[p], k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p], e1_strm_arry[p]);
        }
    }

    for (int i = 0; i < PU; i++) {
#pragma HLS unroll
        details::join_v2::build_probe_wrapper<HASHWL, KEYW, PW, S_PW, ARW>(
            hash_strm_arry[i], k1_strm_arry[i], p1_strm_arry[i], e1_strm_arry[i], w_row_strm[i], k2_strm_arry[i],
            p2_strm_arry[i], addr_strm[i], nm0_strm_arry[i], e2_strm_arry[i], e3_strm_arry[i], bit_vector0[i],
            bit_vector1[i], bit_vector2[i], bit_vector3[i]);
    }

    if (PU >= 4) {
        details::join_v2::access_srow<BFW, ARW, KEYW + S_PW>(stb0_buf, addr_strm[0], w_row_strm[0], e2_strm_arry[0],
                                                             r_row_strm[0]);
        details::join_v2::access_srow<BFW, ARW, KEYW + S_PW>(stb1_buf, addr_strm[1], w_row_strm[1], e2_strm_arry[1],
                                                             r_row_strm[1]);
        details::join_v2::access_srow<BFW, ARW, KEYW + S_PW>(stb2_buf, addr_strm[2], w_row_strm[2], e2_strm_arry[2],
                                                             r_row_strm[2]);
        details::join_v2::access_srow<BFW, ARW, KEYW + S_PW>(stb3_buf, addr_strm[3], w_row_strm[3], e2_strm_arry[3],
                                                             r_row_strm[3]);
    }
    if (PU >= 8) {
        details::join_v2::access_srow<BFW, ARW, KEYW + S_PW>(stb4_buf, addr_strm[4], w_row_strm[4], e2_strm_arry[4],
                                                             r_row_strm[4]);
        details::join_v2::access_srow<BFW, ARW, KEYW + S_PW>(stb5_buf, addr_strm[5], w_row_strm[5], e2_strm_arry[5],
                                                             r_row_strm[5]);
        details::join_v2::access_srow<BFW, ARW, KEYW + S_PW>(stb6_buf, addr_strm[6], w_row_strm[6], e2_strm_arry[6],
                                                             r_row_strm[6]);
        details::join_v2::access_srow<BFW, ARW, KEYW + S_PW>(stb7_buf, addr_strm[7], w_row_strm[7], e2_strm_arry[7],
                                                             r_row_strm[7]);
    }

    for (int i = 0; i < PU; i++) {
#pragma HLS unroll
        details::join_v2::join_unit<PW, S_PW, B_PW, KEYW, 18>(r_row_strm[i], k2_strm_arry[i], p2_strm_arry[i],
                                                              nm0_strm_arry[i], e3_strm_arry[i], j0_strm_arry[i],
                                                              e4_strm_arry[i]);
    }

    // Collect
    details::join_v2::collect_unit<PU, S_PW + B_PW>(j0_strm_arry, e4_strm_arry, j1_strm, e5_strm);
} // hash_join_mpu

} // namespace database
} // namespace xf

namespace xf {
namespace database {
namespace details {
namespace v2_8m {

template <int HASH_MODE, int KEYW, int HASHW>
void hash_wrapper(hls::stream<ap_uint<KEYW> >& i_key_strm,
                  hls::stream<bool>& i_e_strm,
                  hls::stream<ap_uint<HASHW> >& o_hash_strm,
                  hls::stream<ap_uint<KEYW> >& o_key_strm,
                  hls::stream<bool>& o_e_strm) {
    hls::stream<ap_uint<KEYW> > key_strm_in;
#pragma HLS STREAM variable = key_strm_in depth = 24
#pragma HLS bind_storage variable = key_strm_in type = fifo impl = srl
    hls::stream<ap_uint<64> > hash_strm_out;
#pragma HLS STREAM variable = hash_strm_out depth = 8
// radix hash function
#ifndef __SYNTHESIS__
#ifdef DEBUG
    unsigned int i = 0;
#endif
#endif
    bool last = i_e_strm.read();
BUILD_HASH_LOOP:
    while (!last) {
#pragma HLS loop_tripcount min = 1 max = 1000
#pragma HLS PIPELINE II = 1
        bool blk = i_e_strm.empty() || i_key_strm.empty() || o_key_strm.full();
        if (!blk) {
            last = i_e_strm.read();
            o_e_strm.write(0);
            ap_uint<KEYW> key = i_key_strm.read();
            o_key_strm.write(key);
            if (HASH_MODE != 0) key_strm_in.write(key);
            if (HASH_MODE == 0) {
                ap_uint<HASHW> s_hash_val = key(HASHW - 1, 0);
                o_hash_strm.write(s_hash_val);
            } else {
                database::hashLookup3<KEYW>(key_strm_in, hash_strm_out);
                ap_uint<64> l_hash_val = hash_strm_out.read();
                ap_uint<HASHW> s_hash_val = l_hash_val(HASHW - 1, 0);
                o_hash_strm.write(s_hash_val);
#ifndef __SYNTHESIS__
#ifdef DEBUG
                i++;
                if (i < 10) std::cout << "hash_val = " << s_hash_val << std::endl;
#endif
#endif
            }
        }
    }
    o_e_strm.write(1);
} // hash_wrapper

/// @brief dispatch data to multiple PU based on the hash value
/// Each PU is assigned with a different hash_value
template <int KEYW, int PW, int HASHWH, int HASHWL, int PU>
void dispatch(hls::stream<ap_uint<KEYW> >& i_key_strm,
              hls::stream<ap_uint<PW> >& i_pld_strm,
              hls::stream<ap_uint<HASHWH + HASHWL> >& i_hash_strm,
              hls::stream<bool>& i_e_strm,
              hls::stream<ap_uint<KEYW> > o_key_strm[PU],
              hls::stream<ap_uint<PW> > o_pld_strm[PU],
              hls::stream<ap_uint<HASHWL> > o_hash_strm[PU],
              hls::stream<bool> o_e_strm[PU]) {
    bool last = i_e_strm.read();
    while (!last) {
#pragma HLS pipeline II = 1
        ap_uint<HASHWH + HASHWL> hash_val = i_hash_strm.read();
        ap_uint<HASHWH> idx = hash_val(HASHWH + HASHWL - 1, HASHWL);
        ap_uint<HASHWL> hash_out = hash_val(HASHWL - 1, 0);
        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<PW> pld = i_pld_strm.read();
        o_key_strm[idx].write(key);
        o_pld_strm[idx].write(pld);
        o_hash_strm[idx].write(hash_out);
        o_e_strm[idx].write(false);
        last = i_e_strm.read();
    }
    // for do_while
    for (int i = 0; i < PU; i++) {
#pragma HLS unroll
        // if add merge module, need uncomment
        o_key_strm[i].write(0);
        o_pld_strm[i].write(0);
        o_hash_strm[i].write(0);
        o_e_strm[i].write(true);
    }
} // dispatch

/// @brief dispatch data based on hash value to multiple PU
template <int HASH_MODE, int KEYW, int PW, int HASHWH, int HASHWL, int PU>
void dispatch_unit(hls::stream<ap_uint<KEYW> >& i_key_strm,
                   hls::stream<ap_uint<PW> >& i_pld_strm,
                   hls::stream<bool>& i_e_strm,
                   hls::stream<ap_uint<KEYW> > o_key_strm[PU],
                   hls::stream<ap_uint<PW> > o_pld_strm[PU],
                   hls::stream<ap_uint<HASHWL> > o_hash_strm[PU],
                   hls::stream<bool> o_e_strm[PU]) {
    hls::stream<ap_uint<HASHWH + HASHWL> > s_hash_strm;
#pragma HLS STREAM variable = s_hash_strm depth = 4

    hls::stream<bool> io_e0_strm;
#pragma HLS STREAM variable = io_e0_strm depth = 16

    hls::stream<ap_uint<KEYW> > io_k0_strm;
#pragma HLS STREAM variable = io_k0_strm depth = 16

#pragma HLS DATAFLOW
    hash_wrapper<HASH_MODE, KEYW, HASHWH + HASHWL>(i_key_strm, i_e_strm, s_hash_strm, io_k0_strm, io_e0_strm);
    dispatch<KEYW, PW, HASHWH, HASHWL, PU>(io_k0_strm, i_pld_strm, s_hash_strm, io_e0_strm, o_key_strm, o_pld_strm,
                                           o_hash_strm, o_e_strm);
} // dispatch_unit

template <int HASH_MODE, int KEYW, int PW, int HASHWH, int HASHWL, int PU>
void dispatch_wrapper(hls::stream<ap_uint<KEYW> >& i_key_strm,
                      hls::stream<ap_uint<PW> >& i_pld_strm,
                      hls::stream<bool>& i_e_strm,
                      hls::stream<ap_uint<KEYW> > o_key_strm[PU],
                      hls::stream<ap_uint<PW> > o_pld_strm[PU],
                      hls::stream<ap_uint<HASHWL> > o_hash_strm[PU],
                      hls::stream<bool> o_e_strm[PU]) {
    // first: build bitmap
    // second: build hash table
    // third: probe_hash_table
    for (int r = 0; r < 3; r++) {
        dispatch_unit<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(i_key_strm, i_pld_strm, i_e_strm, o_key_strm, o_pld_strm,
                                                               o_hash_strm, o_e_strm);
    }
} // dispatch_wrapper

// ------------------------------------------------------------

/// @brief scan small table to build bitmap
template <int HASHW, int KEYW, int PW, int ARW>
void build_bitmap_lp(hls::stream<ap_uint<HASHW> >& i_hash_strm,
                     hls::stream<ap_uint<KEYW> >& i_key_strm,
                     hls::stream<ap_uint<PW> >& i_pld_strm,
                     hls::stream<bool>& i_e_strm,
                     hls::stream<ap_uint<HASHW> > o_hv_strm[8],   // used to build hp_table
                     hls::stream<ap_uint<ARW - 9> > o_hp_strm[8], // used to build hp_addr_table
                     hls::stream<bool> o_e_strm[8],               // used to build hp_addr_table
#ifdef URAM_SPLITTING
                     ap_uint<72>* bit_vector0,
                     ap_uint<72>* bit_vector1,
                     ap_uint<72>* bit_vector2,
                     ap_uint<72>* bit_vector3
#else
                     ap_uint<72>* bit_vector
#endif
                     ) {
    const int HASH_DEPTH = 1 << (HASHW - 3); // every 8 item in one row
    ap_uint<HASHW - 2> array_idx_r0 = 0xffffffff;
    ap_uint<HASHW - 2> array_idx_r1 = 0xffffffff;
    ap_uint<HASHW - 2> array_idx_r2 = 0xffffffff;
    ap_uint<HASHW - 2> array_idx_r3 = 0xffffffff;
    ap_uint<HASHW - 2> array_idx_r4 = 0xffffffff;
    ap_uint<HASHW - 2> array_idx_r5 = 0xffffffff;
    ap_uint<HASHW - 2> array_idx_r6 = 0xffffffff;
    ap_uint<HASHW - 2> array_idx_r7 = 0xffffffff;
    ap_uint<72> array_val_r0 = 0;
    ap_uint<72> array_val_r1 = 0;
    ap_uint<72> array_val_r2 = 0;
    ap_uint<72> array_val_r3 = 0;
    ap_uint<72> array_val_r4 = 0;
    ap_uint<72> array_val_r5 = 0;
    ap_uint<72> array_val_r6 = 0;
    ap_uint<72> array_val_r7 = 0;

    ap_uint<ARW> base_addr = 0;
    ap_uint<ARW - 9> hp_pre = 0;
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
    unsigned int max_col = 0;
#endif
INIT_LOOP:
    for (int i = 0; i < HASH_DEPTH; i++) {
#pragma HLS PIPELINE II = 1
#ifdef URAM_SPLITTING
#pragma HLS dependence variable = bit_vector0 inter false
#pragma HLS dependence variable = bit_vector1 inter false
#pragma HLS dependence variable = bit_vector2 inter false
#pragma HLS dependence variable = bit_vector3 inter false
#else
#pragma HLS dependence variable = bit_vector inter false
#endif
        write_bit_vector(i, 0);
    }
    bool last = i_e_strm.read();
    // low-part-0: 8:0
    // low-part-1: 17:9
    // low-part-2: 26:18
    // low-part-3: 35:27
    // low-part-4: 44:36
    // low-part-5: 53:45
    // low-part-6: 62:54
    // low-part-7: 71:63
    const int idx_arry[8] = {0, 9, 18, 27, 36, 45, 54, 63};
BITMAP_OFFSET_LOOP:
    while (!last) {
#pragma HLS PIPELINE II = 1
#ifdef URAM_SPLITTING
#pragma HLS dependence variable = bit_vector0 inter false
#pragma HLS dependence variable = bit_vector1 inter false
#pragma HLS dependence variable = bit_vector2 inter false
#pragma HLS dependence variable = bit_vector3 inter false
#else
#pragma HLS dependence variable = bit_vector inter false
#endif
        if (!i_e_strm.empty()) {
            ap_uint<HASHW> hash_val = i_hash_strm.read();
            // discard
            ap_uint<KEYW> key = i_key_strm.read();
            ap_uint<PW> pld = i_pld_strm.read();

            last = i_e_strm.read();

            ap_uint<3> bit_idx = hash_val(2, 0);
            ap_uint<7> idx = idx_arry[bit_idx];
            ap_uint<HASHW - 2> array_idx = hash_val(HASHW - 1, 3);

            ap_uint<72> elem = 0;
            if (array_idx == array_idx_r0) {
                elem = array_val_r0;
            } else if (array_idx == array_idx_r1) {
                elem = array_val_r1;
            } else if (array_idx == array_idx_r2) {
                elem = array_val_r2;
            } else if (array_idx == array_idx_r3) {
                elem = array_val_r3;
            } else if (array_idx == array_idx_r4) {
                elem = array_val_r4;
            } else if (array_idx == array_idx_r5) {
                elem = array_val_r5;
            } else if (array_idx == array_idx_r6) {
                elem = array_val_r6;
            } else if (array_idx == array_idx_r7) {
                elem = array_val_r7;
            } else {
                read_bit_vector(array_idx, elem);
            }
            ap_uint<9> old_val = elem(idx + 8, idx);
            ap_uint<72> new_elem = elem;
            new_elem(idx + 8, idx) = old_val + 1;
            write_bit_vector(array_idx, new_elem);
            array_val_r7 = array_val_r6;
            array_val_r6 = array_val_r5;
            array_val_r5 = array_val_r4;
            array_val_r4 = array_val_r3;
            array_val_r3 = array_val_r2;
            array_val_r2 = array_val_r1;
            array_val_r1 = array_val_r0;
            array_val_r0 = new_elem;
            array_idx_r7 = array_idx_r6;
            array_idx_r6 = array_idx_r5;
            array_idx_r5 = array_idx_r4;
            array_idx_r4 = array_idx_r3;
            array_idx_r3 = array_idx_r2;
            array_idx_r2 = array_idx_r1;
            array_idx_r1 = array_idx_r0;
            array_idx_r0 = array_idx;
#ifndef __SYNTHESIS__
            cnt++;
            if (old_val > max_col) max_col = old_val;
#endif
        }
    }
#ifndef __SYNTHESIS__
    std::cout << "Get " << cnt << " to build bitmap" << std::endl;
    std::cout << "collision probility " << max_col << std::endl;
#endif
BITMAP_ADDR_LOOP:
    for (int h = 0; h < HASH_DEPTH; h++) {
#pragma HLS PIPELINE II = 1
#ifdef URAM_SPLITTING
#pragma HLS dependence variable = bit_vector0 inter false
#pragma HLS dependence variable = bit_vector1 inter false
#pragma HLS dependence variable = bit_vector2 inter false
#pragma HLS dependence variable = bit_vector3 inter false
#else
#pragma HLS dependence variable = bit_vector inter false
#endif
        ap_uint<72> elem;
        read_bit_vector(h, elem);
        ap_uint<ARW> val[8];
        // get the 8 part
        for (int i = 0; i < 8; i++) {
#pragma HLS unroll
            val[i] = elem(i * 9 + 8, i * 9);
        }
        ap_uint<ARW> addr[9];
        addr[0] = base_addr;
        // calculate the address
        for (int i = 1; i < 9; i++) {
#pragma HLS unroll
            addr[i] = addr[i - 1] + val[i - 1];
        }

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "h=" << i << " ,coll_nm_0=" << val[0] << std::endl;
        std::cout << "h=" << i << " ,coll_nm_1=" << val[1] << std::endl;
        std::cout << "h=" << i << " ,coll_nm_2=" << val[2] << std::endl;
        std::cout << "h=" << i << " ,coll_nm_3=" << val[3] << std::endl;
        std::cout << "h=" << i << " ,coll_nm_4=" << val[4] << std::endl;
        std::cout << "h=" << i << " ,coll_nm_5=" << val[5] << std::endl;
        std::cout << "h=" << i << " ,coll_nm_6=" << val[6] << std::endl;
        std::cout << "h=" << i << " ,coll_nm_7=" << val[7] << std::endl;
#endif
#endif
        ap_uint<HASHW> hv = 0;
        hv(HASHW - 1, 3) = h; // left shift 2 bit
        ap_uint<ARW - 9> hp[8];
        for (int i = 0; i < 8; i++) {
#pragma HLS unroll
            hp[i] = addr[i](ARW - 1, 9); // ARW must be > 9
        }
        if (hp[0][0] != hp_pre[0]) {
            o_hv_strm[0].write(hv - 1);
            o_hp_strm[0].write(hp[0]);
            o_e_strm[0].write(0);
        }
        for (int i = 1; i < 8; i++) {
#pragma HLS unroll
            if (hp[i][0] != hp[i - 1][0]) {
                o_hv_strm[i].write(hv + i - 1);
                o_hp_strm[i].write(hp[i]);
                o_e_strm[i].write(false);
            }
        }
        ap_uint<72> head = 0;
        for (int i = 0; i < 8; i++) {
#pragma HLS unroll
            head(i * 9 + 8, i * 9) = addr[i](8, 0);
        }
        base_addr = addr[8];
        hp_pre = addr[7](ARW - 1, 9);
        write_bit_vector(h, head);
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "h=" << h << std::endl;
        ap_int<18> a = head(17, 0);
        std::cout << " ,head[0]=" << std::hex << a << std::endl;
        ap_int<18> a1 = head(35, 18);
        std::cout << " ,head[1]=" << std::hex << a1 << std::endl;
        ap_uint<18> a2 = head(53, 36);
        std::cout << " ,head[2]=" << std::hex << a2 << std::endl;
        ap_uint<18> a3 = head(71, 54);
        std::cout << " ,head[3]=" << std::hex << a3 << std::endl;
        ;
#endif
#endif
    }
    for (int i = 0; i < 8; i++) {
#pragma HLS unroll
        o_hv_strm[i].write(0);
        o_hp_strm[i].write(0);
        o_e_strm[i].write(true);
    }
} // build_bitmap_lp

template <int HASHW, int ARW_H, int HP_DH>
void build_hp_tb(hls::stream<ap_uint<HASHW> > i_hv_strm[8],
                 hls::stream<ap_uint<ARW_H> > i_hp_strm[8],
                 hls::stream<bool> i_e_strm[8],
                 ap_uint<HASHW> hp_addr_tb[ARW_H + 1][HP_DH],
                 ap_uint<1>* tg_tb) {
    const int MAX_HASH = (1 << HASHW) - 1;
    ap_uint<HASHW> hv_arry[8];
#pragma HLS array_partition variable = hv_arry dim = 1
    ap_uint<ARW_H> hp_arry[8];
#pragma HLS array_partition variable = hp_arry dim = 1
    ap_uint<8> empty_e = 0;
    ;
    ap_uint<8> rd_e = 0;
    ;
    ap_uint<8> last = 0;
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
INIT_TB:
    for (int i = 0; i < HP_DH; i++) {
#pragma HLS pipeline II = 1
        if (i == 0)
            tg_tb[i] = 1;
        else
            tg_tb[i] = 0;
        for (int j = 0; j < ARW_H + 1; ++j) {
#pragma HLS unroll
            if (i != 0)
                hp_addr_tb[j][i] = MAX_HASH;
            else
                hp_addr_tb[j][i] = 0;
        }
    }
LOOP_MERGE8_1:
    do {
#pragma HLS loop_tripcount min = 1 max = 5000
#pragma HLS pipeline II = 1
        for (int i = 0; i < 8; i++) {
#pragma HLS unroll
            empty_e[i] = !i_e_strm[i].empty() && !last[i];
        }
        rd_e = details::join_v2::mul_ch_read(empty_e);
        for (int i = 0; i < 8; i++) {
#pragma HLS unroll
            if (rd_e[i]) {
                hv_arry[i] = i_hv_strm[i].read();
                hp_arry[i] = i_hp_strm[i].read();
                last[i] = i_e_strm[i].read();
            }
        }
        // only support 8 channels, 4 channels and 2 channels
        ap_uint<3> id = details::join_v2::mux<8>(rd_e);
        ap_uint<HASHW> hv = hv_arry[id];
        ap_uint<ARW_H> hp = hp_arry[id];
        bool valid_n = last[id];
        if (!valid_n && rd_e != 0) {
#ifndef __SYNTHESIS__
            cnt++;
#endif
            for (int i = 0; i < ARW_H + 1; ++i) {
#pragma HLS unroll
                hp_addr_tb[i][hp] = hv; // store the hash value in duplicated table.
            }
        }
    } while (last != 255);
} // build_hp_tb

/// @brief scan small table to build bitmap
template <int HASHW, int KEYW, int PW, int ARW, int HP_DH>
void build_bitmap(hls::stream<ap_uint<HASHW> >& i_hash_strm,
                  hls::stream<ap_uint<KEYW> >& i_key_strm,
                  hls::stream<ap_uint<PW> >& i_pld_strm,
                  hls::stream<bool>& i_e_strm,
#ifdef URAM_SPLITTING
                  ap_uint<72>* bit_vector0,
                  ap_uint<72>* bit_vector1,
                  ap_uint<72>* bit_vector2,
                  ap_uint<72>* bit_vector3,
#else
                  ap_uint<72>* bit_vector,
#endif
                  ap_uint<HASHW> hp_addr_tb[ARW - 8][HP_DH],
                  ap_uint<1>* tg_tb // more 1 bit for toggle flag
                  ) {
    hls::stream<ap_uint<HASHW> > hv_strm[8];
#pragma HLS stream variable = hv_strm depth = 8
    hls::stream<ap_uint<ARW - 9> > hp_strm[8];
#pragma HLS stream variable = hp_strm depth = 8
    hls::stream<bool> e_strm[8];
#pragma HLS stream variable = e_strm depth = 8

#pragma HLS dataflow

    build_bitmap_lp<HASHW, KEYW, PW, ARW>(i_hash_strm, i_key_strm, i_pld_strm, i_e_strm, hv_strm, hp_strm, e_strm,
#ifdef URAM_SPLITTING
                                          bit_vector0, bit_vector1, bit_vector2, bit_vector3
#else
                                          bit_vector
#endif
                                          );

    build_hp_tb<HASHW, ARW - 9, HP_DH>(hv_strm, hp_strm, e_strm, hp_addr_tb, tg_tb);

#ifndef __SYNTHESIS__
    for (int i = 0; i < HP_DH; ++i) {
        std::cout << "hp = " << i << ", hv = " << hp_addr_tb[0][i] << ", tg =" << tg_tb[i] << std::endl;
    }
#endif
} // build_bitmap

/// @brief read data from multiple channel.
/// Build hash table using BRAM/URAM.
/// Cache small table to HBM.
template <int HASHW, int KEYW, int PW, int S_PW, int ARW>
void build_unit_lp(hls::stream<ap_uint<HASHW> >& i_hash_strm,
                   hls::stream<ap_uint<KEYW> >& i_key_strm,
                   hls::stream<ap_uint<PW> >& i_pld_strm,
                   hls::stream<bool>& i_e_strm,
                   hls::stream<ap_uint<KEYW + S_PW> >& o_row_strm,
                   hls::stream<ap_uint<9> >& o_lp_strm,     // low part of address
                   hls::stream<ap_uint<HASHW> >& o_hv_strm, // hash value used to lookup the high part of address
                   hls::stream<bool>& o_tg_strm,            // toggle flag used to lookup the high part of address
                   hls::stream<bool>& o_e_strm,
#ifdef URAM_SPLITTING
                   ap_uint<72>* bit_vector0,
                   ap_uint<72>* bit_vector1,
                   ap_uint<72>* bit_vector2,
                   ap_uint<72>* bit_vector3
#else
                   ap_uint<72>* bit_vector
#endif
                   ) {
#pragma HLS inline off
    ap_uint<HASHW - 2> array_idx_r0 = 0xffffffff;
    ap_uint<HASHW - 2> array_idx_r1 = 0xffffffff;
    ap_uint<HASHW - 2> array_idx_r2 = 0xffffffff;
    ap_uint<HASHW - 2> array_idx_r3 = 0xffffffff;
    ap_uint<HASHW - 2> array_idx_r4 = 0xffffffff;
    ap_uint<HASHW - 2> array_idx_r5 = 0xffffffff;
    ap_uint<HASHW - 2> array_idx_r6 = 0xffffffff;
    ap_uint<HASHW - 2> array_idx_r7 = 0xffffffff;
    ap_uint<72> array_val_r0 = 0;
    ap_uint<72> array_val_r1 = 0;
    ap_uint<72> array_val_r2 = 0;
    ap_uint<72> array_val_r3 = 0;
    ap_uint<72> array_val_r4 = 0;
    ap_uint<72> array_val_r5 = 0;
    ap_uint<72> array_val_r6 = 0;
    ap_uint<72> array_val_r7 = 0;
    const int idx_arry[8] = {0, 9, 18, 27, 36, 45, 54, 63};
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
    bool last = i_e_strm.read();
LOOP_BUILD:
    while (!last) {
#pragma HLS loop_tripcount min = 1 max = 5000
#pragma HLS PIPELINE II = 1
#ifdef URAM_SPLITTING
#pragma HLS dependence variable = bit_vector0 inter false
#pragma HLS dependence variable = bit_vector1 inter false
#pragma HLS dependence variable = bit_vector2 inter false
#pragma HLS dependence variable = bit_vector3 inter false
#else
#pragma HLS dependence variable = bit_vector inter false
#endif
        if (!i_e_strm.empty()) {
            ap_uint<KEYW> key = i_key_strm.read();
            ap_uint<PW> pld = i_pld_strm.read();
            ap_uint<HASHW> hash_val = i_hash_strm.read();
            last = i_e_strm.read();
            // only support 8 channels, 4 channels and 2 channels
            ap_uint<KEYW + S_PW> stb_row = 0;
            if (S_PW > 0) stb_row(S_PW - 1, 0) = pld(S_PW - 1, 0);
            stb_row(S_PW + KEYW - 1, S_PW) = key;
            // avoid read after write
            ap_uint<3> bit_idx = hash_val(2, 0);
            ap_uint<7> idx = idx_arry[bit_idx];
            ap_uint<HASHW - 2> array_idx = hash_val(HASHW - 1, 3);

            ap_uint<72> elem = 0;
            if (array_idx == array_idx_r0) {
                elem = array_val_r0;
            } else if (array_idx == array_idx_r1) {
                elem = array_val_r1;
            } else if (array_idx == array_idx_r2) {
                elem = array_val_r2;
            } else if (array_idx == array_idx_r3) {
                elem = array_val_r3;
            } else if (array_idx == array_idx_r4) {
                elem = array_val_r4;
            } else if (array_idx == array_idx_r5) {
                elem = array_val_r5;
            } else if (array_idx == array_idx_r6) {
                elem = array_val_r6;
            } else if (array_idx == array_idx_r7) {
                elem = array_val_r7;
            } else {
                read_bit_vector(array_idx, elem);
            }
            ap_uint<9> old_val = elem(idx + 8, idx);
            if (old_val == ap_uint<9>(0x1ff)) {
                o_tg_strm.write(1); // for 1-1111-1111 to 0-0000-0000
            } else {
                o_tg_strm.write(0);
            }
            ap_uint<72> new_elem = elem;
            new_elem(idx + 8, idx) = old_val + 1;
            write_bit_vector(array_idx, new_elem);

            array_val_r7 = array_val_r6;
            array_val_r6 = array_val_r5;
            array_val_r5 = array_val_r4;
            array_val_r4 = array_val_r3;
            array_val_r3 = array_val_r2;
            array_val_r2 = array_val_r1;
            array_val_r1 = array_val_r0;
            array_val_r0 = new_elem;

            array_idx_r7 = array_idx_r6;
            array_idx_r6 = array_idx_r5;
            array_idx_r5 = array_idx_r4;
            array_idx_r4 = array_idx_r3;
            array_idx_r3 = array_idx_r2;
            array_idx_r2 = array_idx_r1;
            array_idx_r1 = array_idx_r0;
            array_idx_r0 = array_idx;

            ap_uint<9> o_addr = old_val;
#ifndef __SYNTHESIS__
            cnt++;
#endif
            o_lp_strm.write(o_addr); // low part of address
            o_hv_strm.write(hash_val);
            o_row_strm.write(stb_row);
            o_e_strm.write(false);
        }
    }
#ifndef __SYNTHESIS__
    std::cout << "build unit write " << cnt << " row to ssbm" << std::endl;
#ifdef DEBUG
    int H = 1 << (HASHW - 2);
    for (int i = 0; i < H; i++) {
        ap_uint<72> t;
        read_bit_vector(i, t);
        std::cout << "i = " << i << " ,addr = " << t << std::endl;
    }
#endif
#endif
    o_e_strm.write(true);
} // build_unit_lp

// reuse this moudule of build phase and probe phase
template <int HASHW, int ARW, int HP_DH>
void probe_hp_tb(hls::stream<ap_uint<HASHW> >& i_hv_strm,
                 hls::stream<bool>& i_tg_strm,
                 hls::stream<ap_uint<9> >& i_lp_strm,
                 hls::stream<bool>& i_e_strm,
                 hls::stream<ap_uint<ARW> >& o_addr_strm,
                 hls::stream<bool>& o_e_strm,
                 ap_uint<HASHW> hp_addr_tb[ARW - 8][HP_DH],
                 ap_uint<1>* tg_tb,
                 bool build_phase) {
    bool e = i_e_strm.read();

    ap_uint<ARW - 9> hp_reg = 1;
    ap_uint<1> tg_reg = 0;

    while (!e) {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = tg_tb inter false
        if (!i_e_strm.empty()) {
            e = i_e_strm.read();
            ap_uint<HASHW> hv_ref = i_hv_strm.read();
            // binary search
            ap_uint<ARW - 9> begin = 0;
            ap_uint<ARW - 9> end = (1 << (ARW - 9)) - 1;
            for (int i = 0; i < (ARW - 9); ++i) {
#pragma HLS unroll
                ap_uint<ARW - 9> mid = (begin + end) >> 1;
                ap_uint<HASHW> hv = hp_addr_tb[i][mid];
                if (hv > hv_ref) {
                    end = mid - 1;
                } else {
                    begin = mid + 1;
                }
#ifndef __SYNTHESIS__
#ifdef DEBUG
                std::cout << "hv = " << hv_ref << ", begin = " << begin << ", end=" << end << ", mid=" << mid
                          << std::endl;
#endif
#endif
            }
            // need one more
            ap_uint<HASHW> hv_end = hp_addr_tb[ARW - 9][end];
            ap_uint<HASHW> hv_end_m1 = hp_addr_tb[ARW - 9][end - 1];
            bool is_equal = 0;
            if (hv_end > hv_ref) {
                end--;
            }
            if (hv_end == hv_ref || hv_end_m1 == hv_ref) is_equal = 1;
#ifndef __SYNTHESIS__
#ifdef DEBUG
            std::cout << "hv_end = " << hv_end << ", hv = " << hv_ref << ", begin = " << begin << ", end=" << end
                      << std::endl;
#endif
#endif
            bool tg = 0;
            if (end == hp_reg) {
                tg = tg_reg;
            } else {
                tg = tg_tb[end];
            }
            bool i_tg = 0;
            i_tg = i_tg_strm.read();
#ifndef __SYNTHESIS__
#ifdef DEBUG
            std::cout << "i_tg = " << i_tg << std::endl;
            std::cout << "tg = " << tg << std::endl;
            std::cout << "is_equal = " << is_equal << std::endl;
#endif
#endif
            if (build_phase && i_tg) {
                tg_tb[end] = 1;
                hp_reg = end;
                tg_reg = 1;
            }
            ap_uint<ARW> o_addr = 0;
            o_addr(8, 0) = i_lp_strm.read();
            if (((build_phase && !tg) || (!build_phase && end != 0)) && is_equal) {
                o_addr(ARW - 1, 9) = end - 1;
            } else {
                o_addr(ARW - 1, 9) = end;
            }
            o_addr_strm.write(o_addr);
            o_e_strm.write(false);
#ifndef __SYNTHESIS__
#ifdef DEBUG
            std::cout << "lp_addr = " << o_addr(8, 0) << std::endl;
            std::cout << "addr = " << o_addr << std::endl;
#endif
#endif
        }
    }
    o_e_strm.write(true);
} // probe_hp_tb

template <int HASHW, int KEYW, int PW, int S_PW, int ARW, int HP_DH>
void build_unit(hls::stream<ap_uint<HASHW> >& i_hash_strm,
                hls::stream<ap_uint<KEYW> >& i_key_strm,
                hls::stream<ap_uint<PW> >& i_pld_strm,
                hls::stream<bool>& i_e_strm,
                hls::stream<ap_uint<KEYW + S_PW> >& o_row_strm,
                hls::stream<ap_uint<ARW> >& o_addr_strm, // high part of address
                hls::stream<bool>& o_e_strm,
#ifdef URAM_SPLITTING
                ap_uint<72>* bit_vector0,
                ap_uint<72>* bit_vector1,
                ap_uint<72>* bit_vector2,
                ap_uint<72>* bit_vector3,
#else
                ap_uint<72>* bit_vector,
#endif
                ap_uint<HASHW> hp_addr_tb[ARW - 8][HP_DH],
                ap_uint<1>* tg_tb) {
    hls::stream<ap_uint<HASHW> > hv_strm;
#pragma HLS stream variable = hv_strm depth = 8

    hls::stream<bool> tg_strm;
#pragma HLS stream variable = tg_strm depth = 8
    hls::stream<bool> e_strm;
#pragma HLS stream variable = e_strm depth = 8

    hls::stream<ap_uint<9> > lp_strm;
#pragma HLS stream variable = lp_strm depth = 8

#pragma HLS dataflow
    build_unit_lp<HASHW, KEYW, PW, S_PW, ARW>           //
        (i_hash_strm, i_key_strm, i_pld_strm, i_e_strm, //
         o_row_strm, lp_strm, hv_strm, tg_strm, e_strm, //
#ifdef URAM_SPLITTING
         bit_vector0, bit_vector1, bit_vector2, bit_vector3
#else
         bit_vector
#endif
         );

    probe_hp_tb<HASHW, ARW, HP_DH>          //
        (hv_strm, tg_strm, lp_strm, e_strm, //
         o_addr_strm, o_e_strm, hp_addr_tb, tg_tb, 1);
}

/// @brief read data from multiple channel
/// @brief probe the hash table and output address which hash same hash_value
/// @brief maybe add bloom filter
template <int HASHW, int KEYW, int PW, int ARW>
void probe_head(hls::stream<ap_uint<HASHW> >& i_hash_strm,
                hls::stream<ap_uint<KEYW> >& i_key_strm,
                hls::stream<ap_uint<PW> >& i_pld_strm,
                hls::stream<bool>& i_e_strm,
                hls::stream<ap_uint<9> >& o_lp_strm,
                // output low part of address; more bit to avoid overflow
                hls::stream<ap_uint<8> >& o_nm0_strm,
                hls::stream<ap_uint<8> >& o_nm1_strm,
                hls::stream<ap_uint<KEYW> >& o_key_strm,
                hls::stream<ap_uint<PW> >& o_pld_strm,
                hls::stream<ap_uint<HASHW> >& o_hv_strm, // hash value
                hls::stream<bool>& o_tg_strm,            // h
                hls::stream<bool>& o_e0_strm,
                hls::stream<bool>& o_e1_strm,
#ifdef URAM_SPLITTING
                ap_uint<72>* bit_vector0,
                ap_uint<72>* bit_vector1,
                ap_uint<72>* bit_vector2,
                ap_uint<72>* bit_vector3
#else
                ap_uint<72>* bit_vector
#endif
                ) {
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
    bool last = i_e_strm.read();
    const int idx_arry[8] = {0, 9, 18, 27, 36, 45, 54, 63};
LOOP_PROBE:
    while (!last) {
#pragma HLS loop_tripcount min = 1 max = 5000
#pragma HLS PIPELINE II = 1
#ifdef URAM_SPLITTING
#pragma HLS dependence variable = bit_vector0 inter false
#pragma HLS dependence variable = bit_vector1 inter false
#pragma HLS dependence variable = bit_vector2 inter false
#pragma HLS dependence variable = bit_vector3 inter false
#else
#pragma HLS dependence variable = bit_vector inter false
#endif
        // read select field from stream and store them on local ram.
        ap_uint<HASHW> hash_val = i_hash_strm.read();
        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<PW> pld = i_pld_strm.read();
        last = i_e_strm.read();
        ap_uint<HASHW - 3> array_idx = hash_val(HASHW - 1, 3);
        ap_uint<3> bit_idx = hash_val(2, 0);
        ap_uint<72> pre = 0;
        if (array_idx > 0) {
            read_bit_vector(array_idx - 1, pre);
        }
        ap_uint<72> cur;
        read_bit_vector(array_idx, cur);
        ap_uint<10> pre_addr = 0;
        ap_uint<10> cur_addr = 0;
        ap_uint<7> idx = idx_arry[bit_idx];
        if (bit_idx == 0) {
            pre_addr = pre(71, 63);
        } else {
            pre_addr = cur(idx - 1, idx - 9);
        }
        cur_addr = cur(idx + 8, idx);
        if (cur_addr < pre_addr) cur_addr[9] = 1;
        ap_uint<8> nm = cur_addr - pre_addr;
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "key = " << key << ", hash_val = " << hash_val << ", nm = " << nm << ", pre = " << pre_addr
                  << std::endl;
#endif
#endif
        // optimization: add bloom filter to filter out more row
        if (nm > 0) {
#ifndef __SYNTHESIS__
            cnt++;
#endif
            o_lp_strm.write(pre_addr);
            o_nm0_strm.write(nm);
            o_nm1_strm.write(nm);
            o_key_strm.write(key);
            o_pld_strm.write(pld);
            o_hv_strm.write(hash_val);
            o_tg_strm.write(false);
            o_e0_strm.write(false);
            o_e1_strm.write(false);
        }
    }
#ifndef __SYNTHESIS__
    std::cout << "probe unit read " << cnt << " block from ssbm" << std::endl;
    ;
#endif
    o_e0_strm.write(true);
    o_e1_strm.write(true);
} // probe_head

template <int HASHW, int KEYW, int PW, int ARW, int HP_DH>
void probe_unit(hls::stream<ap_uint<HASHW> >& i_hash_strm,
                hls::stream<ap_uint<KEYW> >& i_key_strm,
                hls::stream<ap_uint<PW> >& i_pld_strm,
                hls::stream<bool>& i_e_strm,
                hls::stream<ap_uint<ARW> >& o_addr_strm,
                hls::stream<ap_uint<8> >& o_nm_strm,
                hls::stream<ap_uint<KEYW> >& o_key_strm,
                hls::stream<ap_uint<PW> >& o_pld_strm,
                hls::stream<bool>& o_e0_strm,
                hls::stream<bool>& o_e1_strm,
#ifdef URAM_SPLITTING
                ap_uint<72>* bit_vector0,
                ap_uint<72>* bit_vector1,
                ap_uint<72>* bit_vector2,
                ap_uint<72>* bit_vector3,
#else
                ap_uint<72>* bit_vector,
#endif
                ap_uint<HASHW> hp_addr_tb[ARW - 8][HP_DH],
                ap_uint<1>* tg_tb) {
    hls::stream<ap_uint<8> > nm_strm;
#pragma HLS stream variable = nm_strm depth = 32
    hls::stream<ap_uint<ARW> > addr_strm;
#pragma HLS stream variable = addr_strm depth = 8
    hls::stream<bool> e_strm;
#pragma HLS stream variable = e_strm depth = 8
    hls::stream<ap_uint<9> > lp_strm;
#pragma HLS stream variable = lp_strm depth = 8
    hls::stream<ap_uint<HASHW> > hv_strm;
#pragma HLS stream variable = hv_strm depth = 8
    hls::stream<bool> tg_strm;
#pragma HLS stream variable = tg_strm depth = 8
    hls::stream<bool> e1_strm;
#pragma HLS stream variable = e1_strm depth = 8

#pragma HLS dataflow

    probe_head<HASHW, KEYW, PW, ARW>                          //
        (i_hash_strm, i_key_strm, i_pld_strm, i_e_strm,       //
         lp_strm, o_nm_strm, nm_strm, o_key_strm, o_pld_strm, //
         hv_strm, tg_strm, e_strm, o_e1_strm,                 //
#ifdef URAM_SPLITTING
         bit_vector0, bit_vector1, bit_vector2, bit_vector3
#else
         bit_vector
#endif
         );

    probe_hp_tb<HASHW, ARW, HP_DH>          //
        (hv_strm, tg_strm, lp_strm, e_strm, //
         addr_strm, e1_strm, hp_addr_tb, tg_tb, 0);

    details::join_v2::probe_addr_gen<ARW, 8> //
        (addr_strm, nm_strm, e1_strm, o_addr_strm, o_e0_strm);
} // probe_unit

template <int HASHW, int KEYW, int PW, int S_PW, int ARW>
void build_probe_wrapper(hls::stream<ap_uint<HASHW> >& i_hash_strm,
                         hls::stream<ap_uint<KEYW> >& i_key_strm,
                         hls::stream<ap_uint<PW> >& i_pld_strm,
                         hls::stream<bool>& i_e_strm,
                         hls::stream<ap_uint<KEYW + S_PW> >& o_row_strm,
                         hls::stream<ap_uint<KEYW> >& o_key_strm,
                         hls::stream<ap_uint<PW> >& o_pld_strm,
                         hls::stream<ap_uint<ARW> >& o_addr_strm,
                         hls::stream<ap_uint<8> >& o_nm0_strm,
                         hls::stream<bool>& o_e0_strm,
                         hls::stream<bool>& o_e1_strm,
                         ap_uint<72>* bit_vector0,
                         ap_uint<72>* bit_vector1,
                         ap_uint<72>* bit_vector2,
                         ap_uint<72>* bit_vector3) {
    const int DH = 1 << (ARW - 9);

    ap_uint<HASHW> hp_addr_tb[ARW - 8][DH];
#pragma HLS array_partition variable = hp_addr_tb dim = 1
#pragma HLS bind_storage variable = hp_addr_tb type = ram_2p impl = bram

    ap_uint<1> tg_tb[DH];

    build_bitmap<HASHW, KEYW, PW, ARW, DH>(i_hash_strm, i_key_strm, i_pld_strm, i_e_strm, bit_vector0, bit_vector1,
                                           bit_vector2, bit_vector3, hp_addr_tb, tg_tb);

    build_unit<HASHW, KEYW, PW, S_PW, ARW, DH>(i_hash_strm, i_key_strm, i_pld_strm, i_e_strm, o_row_strm, o_addr_strm,
                                               o_e0_strm, bit_vector0, bit_vector1, bit_vector2, bit_vector3,
                                               hp_addr_tb, tg_tb);

    probe_unit<HASHW, KEYW, PW, ARW, DH>(i_hash_strm, i_key_strm, i_pld_strm, i_e_strm, o_addr_strm, o_nm0_strm,
                                         o_key_strm, o_pld_strm, o_e0_strm, o_e1_strm, bit_vector0, bit_vector1,
                                         bit_vector2, bit_vector3, hp_addr_tb, tg_tb);
}

} // namespace v2_8m
} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/**
 * @brief Multi-PU Hash-Join primitive, using multiple DDR/HBM buffers.
 *
 * The max number of lines of small table is 8M in this design.
 * It is assumed that the hash-conflict is within 512 per bin.
 *
 * This module can accept more than 1 input row per cycle, via multiple
 * input channels.
 * The small table and the big table shares the same input ports,
 * so the width of the payload should be the max of both, while the data
 * should be aligned to the little-end.
 * The small table should be fed TWICE, followed by the big table once.
 *
 * @tparam HASH_MODE 0 for radix and 1 for Jenkin's Lookup3 hash.
 * @tparam KEYW width of key, in bit.
 * @tparam PW width of max payload, in bit.
 * @tparam S_PW width of payload of small table.
 * @tparam B_PW width of payload of big table.
 * @tparam HASHWH number of hash bits used for PU/buffer selection, 1~3.
 * @tparam HASHWL number of hash bits used for hash-table in PU.
 * @tparam ARW width of address, log2(small table max num of rows).
 * @tparam BFW width of buffer.
 * @tparam CH_NM number of input channels, 1,2,4.
 *
 * @param k0_strm_arry input of key columns of both tables.
 * @param p0_strm_arry input of payload columns of both tables.
 * @param e0_strm_arry input of end signal of both tables.
 * @param stb0_buf HBM/DDR buffer of PU0
 * @param stb1_buf HBM/DDR buffer of PU1
 * @param stb2_buf HBM/DDR buffer of PU2
 * @param stb3_buf HBM/DDR buffer of PU3
 * @param stb4_buf HBM/DDR buffer of PU4
 * @param stb5_buf HBM/DDR buffer of PU5
 * @param stb6_buf HBM/DDR buffer of PU6
 * @param stb7_buf HBM/DDR buffer of PU7
 * @param j1_strm output of joined rows.
 * @param e5_strm end signal of joined rows.
 */
template <int HASH_MODE, int KEYW, int PW, int S_PW, int B_PW, int HASHWH, int HASHWL, int ARW, int BFW, int CH_NM>
void hashJoinMPU(hls::stream<ap_uint<KEYW> > k0_strm_arry[CH_NM],
                 hls::stream<ap_uint<PW> > p0_strm_arry[CH_NM],
                 hls::stream<bool> e0_strm_arry[CH_NM],
                 ap_uint<BFW>* stb0_buf,
                 ap_uint<BFW>* stb1_buf,
                 ap_uint<BFW>* stb2_buf,
                 ap_uint<BFW>* stb3_buf,
                 ap_uint<BFW>* stb4_buf,
                 ap_uint<BFW>* stb5_buf,
                 ap_uint<BFW>* stb6_buf,
                 ap_uint<BFW>* stb7_buf,
                 hls::stream<ap_uint<S_PW + B_PW> >& j1_strm,
                 hls::stream<bool>& e5_strm) {
    // constants
    enum { HDP_J = (1 << (HASHWL - 3)), PU = (1 << HASHWH) };

    XF_DATABASE_STATIC_ASSERT(CH_NM <= 4, "No more than 4 input channel for HASH-JOIN");
    XF_DATABASE_STATIC_ASSERT(CH_NM > 0, "At least 1 input channel for HASH-JOIN");

#pragma HLS dataflow

    hls::stream<ap_uint<KEYW> > k1_strm_arry[PU];
#pragma HLS stream variable = k1_strm_arry depth = 8
#pragma HLS array_partition variable = k1_strm_arry dim = 0
    hls::stream<ap_uint<PW> > p1_strm_arry[PU];
#pragma HLS stream variable = p1_strm_arry depth = 8
#pragma HLS array_partition variable = p1_strm_arry dim = 0
#pragma HLS bind_storage variable = p1_strm_arry type = fifo impl = srl
    hls::stream<ap_uint<HASHWL> > hash_strm_arry[PU];
#pragma HLS stream variable = hash_strm_arry depth = 8
#pragma HLS array_partition variable = hash_strm_arry dim = 0
    hls::stream<bool> e1_strm_arry[PU];
#pragma HLS stream variable = e1_strm_arry depth = 8
#pragma HLS array_partition variable = e1_strm_arry dim = 0

    // CH_NM >= 1
    hls::stream<ap_uint<KEYW> > k1_strm_arry_c0[PU];
#pragma HLS stream variable = k1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c0 dim = 0
    hls::stream<ap_uint<PW> > p1_strm_arry_c0[PU];
#pragma HLS stream variable = p1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c0 dim = 0
#pragma HLS bind_storage variable = p1_strm_arry_c0 type = fifo impl = srl
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_c0[PU];
#pragma HLS stream variable = hash_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c0 dim = 0
    hls::stream<bool> e1_strm_arry_c0[PU];
#pragma HLS stream variable = e1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c0 dim = 0

    // CH_NM >= 2
    hls::stream<ap_uint<KEYW> > k1_strm_arry_c1[PU];
#pragma HLS stream variable = k1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c1 dim = 0
    hls::stream<ap_uint<PW> > p1_strm_arry_c1[PU];
#pragma HLS stream variable = p1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c1 dim = 0
#pragma HLS bind_storage variable = p1_strm_arry_c1 type = fifo impl = srl
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_c1[PU];
#pragma HLS stream variable = hash_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c1 dim = 0
    hls::stream<bool> e1_strm_arry_c1[PU];
#pragma HLS stream variable = e1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c1 dim = 0

    // CH_NM >= 4
    hls::stream<ap_uint<KEYW> > k1_strm_arry_c2[PU];
#pragma HLS stream variable = k1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c2 dim = 0
    hls::stream<ap_uint<PW> > p1_strm_arry_c2[PU];
#pragma HLS stream variable = p1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c2 dim = 0
#pragma HLS bind_storage variable = p1_strm_arry_c2 type = fifo impl = srl
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_c2[PU];
#pragma HLS stream variable = hash_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c2 dim = 0
    hls::stream<bool> e1_strm_arry_c2[PU];
#pragma HLS stream variable = e1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c2 dim = 0
    hls::stream<ap_uint<KEYW> > k1_strm_arry_c3[PU];
#pragma HLS stream variable = k1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c3 dim = 0
    hls::stream<ap_uint<PW> > p1_strm_arry_c3[PU];
#pragma HLS stream variable = p1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c3 dim = 0
#pragma HLS bind_storage variable = p1_strm_arry_c3 type = fifo impl = srl
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_c3[PU];
#pragma HLS stream variable = hash_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c3 dim = 0
    hls::stream<bool> e1_strm_arry_c3[PU];
#pragma HLS stream variable = e1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c3 dim = 0

    hls::stream<ap_uint<KEYW + S_PW> > w_row_strm[PU];
#pragma HLS stream variable = w_row_strm depth = 64
#pragma HLS array_partition variable = w_row_strm dim = 0
    hls::stream<ap_uint<KEYW + S_PW> > r_row_strm[PU];
#pragma HLS stream variable = r_row_strm depth = 8
    hls::stream<ap_uint<KEYW> > k2_strm_arry[PU];
#pragma HLS stream variable = k2_strm_arry depth = 64
#pragma HLS array_partition variable = k2_strm_arry dim = 0
#pragma HLS bind_storage variable = k2_strm_arry type = fifo impl = srl
    hls::stream<ap_uint<PW> > p2_strm_arry[PU];
#pragma HLS stream variable = p2_strm_arry depth = 64
#pragma HLS array_partition variable = p2_strm_arry dim = 0
#pragma HLS bind_storage variable = p2_strm_arry type = fifo impl = srl
    hls::stream<bool> e2_strm_arry[PU];
#pragma HLS stream variable = e2_strm_arry depth = 8
#pragma HLS array_partition variable = e2_strm_arry dim = 0
    hls::stream<bool> e3_strm_arry[PU];
#pragma HLS stream variable = e3_strm_arry depth = 64
#pragma HLS array_partition variable = e3_strm_arry dim = 0
    hls::stream<bool> e4_strm_arry[PU];
#pragma HLS array_partition variable = e4_strm_arry dim = 0
#pragma HLS stream variable = e4_strm_arry depth = 8
    hls::stream<ap_uint<ARW> > addr_strm[PU];
#pragma HLS stream variable = addr_strm depth = 8
#pragma HLS array_partition variable = addr_strm dim = 0
    hls::stream<ap_uint<8> > nm0_strm_arry[PU];
#pragma HLS stream variable = nm0_strm_arry depth = 8
#pragma HLS array_partition variable = nm0_strm_arry dim = 0
    hls::stream<ap_uint<S_PW + B_PW> > j0_strm_arry[PU];
#pragma HLS stream variable = j0_strm_arry depth = 8
#pragma HLS array_partition variable = j0_strm_arry dim = 0
#pragma HLS bind_storage variable = j0_strm_arry type = fifo impl = srl

#ifndef __SYNTHESIS__
    ap_uint<72>* bit_vector0[PU];
    ap_uint<72>* bit_vector1[PU];
    ap_uint<72>* bit_vector2[PU];
    ap_uint<72>* bit_vector3[PU];
    for (int i = 0; i < PU; i++) {
        bit_vector0[i] = (ap_uint<72>*)malloc((HDP_J >> 2) * sizeof(ap_uint<72>));
        bit_vector1[i] = (ap_uint<72>*)malloc((HDP_J >> 2) * sizeof(ap_uint<72>));
        bit_vector2[i] = (ap_uint<72>*)malloc((HDP_J >> 2) * sizeof(ap_uint<72>));
        bit_vector3[i] = (ap_uint<72>*)malloc((HDP_J >> 2) * sizeof(ap_uint<72>));
    }
#else
    ap_uint<72> bit_vector0[PU][(HDP_J >> 2)];
    ap_uint<72> bit_vector1[PU][(HDP_J >> 2)];
    ap_uint<72> bit_vector2[PU][(HDP_J >> 2)];
    ap_uint<72> bit_vector3[PU][(HDP_J >> 2)];
#pragma HLS array_partition variable = bit_vector0 dim = 1
#pragma HLS bind_storage variable = bit_vector0 type = ram_2p impl = uram
#pragma HLS array_partition variable = bit_vector1 dim = 1
#pragma HLS bind_storage variable = bit_vector1 type = ram_2p impl = uram
#pragma HLS array_partition variable = bit_vector2 dim = 1
#pragma HLS bind_storage variable = bit_vector2 type = ram_2p impl = uram
#pragma HLS array_partition variable = bit_vector3 dim = 1
#pragma HLS bind_storage variable = bit_vector3 type = ram_2p impl = uram
#endif

    // clang-format off
      ;
      // -----------|----------|------------|------|-------|-------|----------
      //   Dispatch |          |Bloom Filter|Bitmap| Build | Probe | 
      // -----------|          |------------|------|-------|-------|
      //   Dispatch | Switcher |Bloom filter|Bitmap| Build | Probe | Collect
      // -----------|          |------------|------|-------|-------|
      //   Dispatch |          |Bloom filter|Bitmap| Build | Porbe |
      // -----------|----------|------------|------|-------|-------|----------

    // clang-format on
    ;

    // These dispatcher will distribute small table twice and big table once.

    details::v2_8m::dispatch_wrapper<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(
        k0_strm_arry[0], p0_strm_arry[0], e0_strm_arry[0], k1_strm_arry_c0, p1_strm_arry_c0, hash_strm_arry_c0,
        e1_strm_arry_c0);
    if (CH_NM >= 2) {
        details::v2_8m::dispatch_wrapper<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(
            k0_strm_arry[1], p0_strm_arry[1], e0_strm_arry[1], k1_strm_arry_c1, p1_strm_arry_c1, hash_strm_arry_c1,
            e1_strm_arry_c1);
    }
    if (CH_NM >= 4) {
        details::v2_8m::dispatch_wrapper<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(
            k0_strm_arry[2], p0_strm_arry[2], e0_strm_arry[2], k1_strm_arry_c2, p1_strm_arry_c2, hash_strm_arry_c2,
            e1_strm_arry_c2);
        details::v2_8m::dispatch_wrapper<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(
            k0_strm_arry[3], p0_strm_arry[3], e0_strm_arry[3], k1_strm_arry_c3, p1_strm_arry_c3, hash_strm_arry_c3,
            e1_strm_arry_c3);
    }

    // These units merges entries for each PU into one stream.

    if (CH_NM == 1) {
        for (int p = 0; p < PU; ++p) {
#pragma HLS unroll
            details::join_v2::merge1_1_wrapper(k1_strm_arry_c0[p], p1_strm_arry_c0[p], hash_strm_arry_c0[p],
                                               e1_strm_arry_c0[p], k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p],
                                               e1_strm_arry[p]);
        }
    } else if (CH_NM == 2) {
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            details::join_v2::merge2_1_wrapper(k1_strm_arry_c0[p], k1_strm_arry_c1[p], p1_strm_arry_c0[p],
                                               p1_strm_arry_c1[p], hash_strm_arry_c0[p], hash_strm_arry_c1[p],
                                               e1_strm_arry_c0[p], e1_strm_arry_c1[p], k1_strm_arry[p], p1_strm_arry[p],
                                               hash_strm_arry[p], e1_strm_arry[p]);
        }
    }
    // These units merges entries for each PU into one stream.

    if (CH_NM == 1) {
        for (int p = 0; p < PU; ++p) {
#pragma HLS unroll
            details::join_v2::merge1_1_wrapper(k1_strm_arry_c0[p], p1_strm_arry_c0[p], hash_strm_arry_c0[p],
                                               e1_strm_arry_c0[p], k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p],
                                               e1_strm_arry[p]);
        }
    } else if (CH_NM == 2) {
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            details::join_v2::merge4_1_wrapper(
                k1_strm_arry_c0[p], k1_strm_arry_c1[p], k1_strm_arry_c2[p], k1_strm_arry_c3[p], p1_strm_arry_c0[p],
                p1_strm_arry_c1[p], p1_strm_arry_c2[p], p1_strm_arry_c3[p], hash_strm_arry_c0[p], hash_strm_arry_c1[p],
                hash_strm_arry_c2[p], hash_strm_arry_c3[p], e1_strm_arry_c0[p], e1_strm_arry_c1[p], e1_strm_arry_c2[p],
                e1_strm_arry_c3[p], k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p], e1_strm_arry[p]);
        }
    }

    // PUs
    for (int i = 0; i < PU; i++) {
#pragma HLS unroll
        details::v2_8m::build_probe_wrapper<HASHWL, KEYW, PW, S_PW, ARW>(
            hash_strm_arry[i], k1_strm_arry[i], p1_strm_arry[i], e1_strm_arry[i], w_row_strm[i], k2_strm_arry[i],
            p2_strm_arry[i], addr_strm[i], nm0_strm_arry[i], e2_strm_arry[i], e3_strm_arry[i], bit_vector0[i],
            bit_vector1[i], bit_vector2[i], bit_vector3[i]);
    }

    if (PU >= 4) {
        details::join_v2::access_srow<BFW, ARW, KEYW + S_PW>(stb0_buf, addr_strm[0], w_row_strm[0], e2_strm_arry[0],
                                                             r_row_strm[0]);
        details::join_v2::access_srow<BFW, ARW, KEYW + S_PW>(stb1_buf, addr_strm[1], w_row_strm[1], e2_strm_arry[1],
                                                             r_row_strm[1]);
        details::join_v2::access_srow<BFW, ARW, KEYW + S_PW>(stb2_buf, addr_strm[2], w_row_strm[2], e2_strm_arry[2],
                                                             r_row_strm[2]);
        details::join_v2::access_srow<BFW, ARW, KEYW + S_PW>(stb3_buf, addr_strm[3], w_row_strm[3], e2_strm_arry[3],
                                                             r_row_strm[3]);
    }
    if (PU >= 8) {
        details::join_v2::access_srow<BFW, ARW, KEYW + S_PW>(stb4_buf, addr_strm[4], w_row_strm[4], e2_strm_arry[4],
                                                             r_row_strm[4]);
        details::join_v2::access_srow<BFW, ARW, KEYW + S_PW>(stb5_buf, addr_strm[5], w_row_strm[5], e2_strm_arry[5],
                                                             r_row_strm[5]);
        details::join_v2::access_srow<BFW, ARW, KEYW + S_PW>(stb6_buf, addr_strm[6], w_row_strm[6], e2_strm_arry[6],
                                                             r_row_strm[6]);
        details::join_v2::access_srow<BFW, ARW, KEYW + S_PW>(stb7_buf, addr_strm[7], w_row_strm[7], e2_strm_arry[7],
                                                             r_row_strm[7]);
    }

    for (int i = 0; i < PU; i++) {
#pragma HLS unroll
        details::join_v2::join_unit<PW, S_PW, B_PW, KEYW, 8>(r_row_strm[i], k2_strm_arry[i], p2_strm_arry[i],
                                                             nm0_strm_arry[i], e3_strm_arry[i], j0_strm_arry[i],
                                                             e4_strm_arry[i]);
    }

    // Collect
    details::join_v2::collect_unit<PU, S_PW + B_PW>(j0_strm_arry, e4_strm_arry, j1_strm, e5_strm);
} // hash_join_mpu

} // namespace database
} // namespace xf
#undef write_bit_vector
#undef read_bit_vector

#endif // !defined(XF_DATABASE_HASH_JOIN_V2_H)
