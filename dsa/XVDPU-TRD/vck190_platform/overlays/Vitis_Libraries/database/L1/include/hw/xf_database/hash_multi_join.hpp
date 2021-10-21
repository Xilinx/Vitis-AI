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
 * @file hash_multi_join.hpp
 * @brief hash join template function implementation, targeting HBM devices.
 *
 *
 * This file is part of Vitis Database Library.
 */

#ifndef XF_DATABASE_HASH_MULTI_JOIN_H
#define XF_DATABASE_HASH_MULTI_JOIN_H

#ifndef __cplusplus
#error "Vitis Database Library only works with C++."
#endif

#include "ap_int.h"
#include "hls_stream.h"

#include "xf_database/combine_split_col.hpp"
#include "xf_database/enums.hpp"
#include "xf_database/hash_join_v2.hpp"
#include "xf_database/hash_join_v3.hpp"
#include "xf_database/hash_lookup3.hpp"
#include "xf_database/utils.hpp"
//#define DEBUG true

#ifdef DEBUG
//#define DEBUG_BUILD true
//#define DEBUG_PROBE true
//#define DEBUG_JOIN true
//#define DEBUG_MISS true
#endif

// FIXME For debug
#ifndef __SYNTHESIS__
#include <iostream>
#endif

#define write_bit_vector0(i, v) \
    do {                        \
        bit_vector0[(i)] = v;   \
    } while (0)

#define read_bit_vector0(i, v) \
    do {                       \
        v = bit_vector0[(i)];  \
    } while (0)

#define write_bit_vector1(i, v) \
    do {                        \
        bit_vector1[(i)] = v;   \
    } while (0)

#define read_bit_vector1(i, v) \
    do {                       \
        v = bit_vector1[(i)];  \
    } while (0)

namespace xf {
namespace database {
namespace details {
namespace hash_multi_join {

//---------------------------------------------------probe------------------------------------------------

/// @brief Probe the hash table
template <int HASHW, int KEYW, int PW, int B_PW, int ARW>
void multi_probe_htb(ap_uint<32>& depth,

                     // input large table
                     hls::stream<ap_uint<HASHW> >& i_hash_strm,
                     hls::stream<ap_uint<KEYW> >& i_key_strm,
                     hls::stream<ap_uint<PW> >& i_pld_strm,
                     hls::stream<bool>& i_e_strm,

                     // output to generate probe addr
                     hls::stream<ap_uint<ARW> >& o_base_addr_strm,
                     hls::stream<ap_uint<ARW> >& o_nm0_strm,
                     hls::stream<bool>& o_e0_strm,
                     hls::stream<ap_uint<ARW> >& o_overflow_addr_strm,
                     hls::stream<ap_uint<ARW> >& o_nm1_strm,
                     hls::stream<bool>& o_e1_strm,

                     // output to join
                     hls::stream<ap_uint<KEYW> >& o_key_strm,
                     hls::stream<ap_uint<B_PW> >& o_pld_strm,
                     hls::stream<ap_uint<ARW> >& o_nm2_strm,
                     hls::stream<bool>& o_e2_strm,

                     ap_uint<72>* bit_vector0,
                     ap_uint<72>* bit_vector1) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
    unsigned int base_cnt = 0;
    unsigned int overflow_cnt = 0;
#endif

    const int HASH_DEPTH = (1 << HASHW) / 3 + 1;
    const int HASH_NUMBER = 1 << HASHW;

    ap_uint<72> base_bitmap;
    ap_uint<72> overflow_bitmap;
    ap_uint<ARW> base_ht_addr;
    ap_uint<ARW> overflow_ht_addr;

    bool last = i_e_strm.read();
LOOP_PROBE:
    while (!last) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = bit_vector0 inter false
#pragma HLS dependence variable = bit_vector1 inter false

        // read select field from stream and store them on local ram.
        ap_uint<HASHW> hash_val = i_hash_strm.read();
        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<B_PW> pld = i_pld_strm.read(); // XXX trunc
        last = i_e_strm.read();

        // mod 3 to calculate index for 24bit address
        ap_uint<HASHW> array_idx = hash_val / 3;
        ap_uint<HASHW> temp = array_idx * 3;
        ap_uint<2> bit_idx = hash_val - temp;

        // total hash count
        ap_uint<ARW> nm;
        read_bit_vector0(array_idx, base_bitmap);

        if (bit_idx == 0 || bit_idx == 3) {
            nm = base_bitmap(23, 0);
        } else if (bit_idx == 1) {
            nm = base_bitmap(47, 24);
        } else {
            nm = base_bitmap(71, 48);
        }

        // calculate addr
        base_ht_addr = hash_val * depth;

        if ((bit_idx == 0 || bit_idx == 3) && array_idx > 0)
            read_bit_vector1(array_idx - 1, overflow_bitmap);
        else
            read_bit_vector1(array_idx, overflow_bitmap);

        if (bit_idx == 0 || bit_idx == 3) {
            if (array_idx > 0) {
                overflow_ht_addr = overflow_bitmap(71, 48);
            } else {
                overflow_ht_addr = 0;
            }
        } else if (bit_idx == 1) {
            overflow_ht_addr = overflow_bitmap(23, 0);
        } else {
            overflow_ht_addr = overflow_bitmap(47, 24);
        }

#ifndef __SYNTHESIS__
#ifdef DEBUG_MISS
        if (key == 68444 || key == 129169)
            std::cout << std::hex << "probe_ahead: key=" << key << " hash_val=" << hash_val
                      << " array_idx=" << array_idx << " bit_idx=" << bit_idx << " nm=" << nm
                      << " base_ht_addr=" << base_ht_addr << " base_bitmap=" << base_bitmap
                      << " overflow_addr=" << overflow_ht_addr << " overflow_bitmap=" << overflow_bitmap << std::endl;
#endif

#ifdef DEBUG_PROBE
        std::cout << std::hex << "probe_ahead: key=" << key << " hash_val=" << hash_val << " array_idx=" << array_idx
                  << " bit_idx=" << bit_idx << " nm=" << nm << " base_ht_addr=" << base_ht_addr
                  << " base_bitmap=" << base_bitmap << " overflow_addr=" << overflow_ht_addr
                  << " overflow_bitmap=" << overflow_bitmap << std::endl;
#endif
#endif
        // optimization: add bloom filter to filter out more row
        if (nm >= 0) {
#ifndef __SYNTHESIS__
            cnt++;
#endif

            ap_uint<ARW> nm0; // base number
            ap_uint<ARW> nm1; // overflow number

            if (nm > depth) {
                nm0 = depth;
                nm1 = nm - depth;
            } else {
                nm0 = nm;
                nm1 = 0;
            }

            if (nm0 > 0) {
                o_base_addr_strm.write(base_ht_addr);
                o_nm0_strm.write(nm0);
                o_e0_strm.write(false);
#ifndef __SYNTHESIS__
                base_cnt++;
#endif
            }
            if (nm1 > 0) {
#ifndef __SYNTHESIS__
                overflow_cnt++;
#endif
                o_overflow_addr_strm.write(overflow_ht_addr);
                o_nm1_strm.write(nm1);
                o_e1_strm.write(false);
            }
            o_key_strm.write(key);
            o_pld_strm.write(pld);
            o_nm2_strm.write(nm);
            o_e2_strm.write(false);
        }
    }

#ifndef __SYNTHESIS__
    std::cout << std::dec << "probe will read " << cnt << " block from stb, including " << base_cnt << " base block, "
              << overflow_cnt << " overflow block" << std::endl;
#endif

    o_e0_strm.write(true);
    o_e2_strm.write(true);

    // for do-while in probe overflow stb
    o_overflow_addr_strm.write(0);
    o_nm1_strm.write(0);
    o_e1_strm.write(true);
}

/// @brief Top function of hash multi join probe
template <int HASHW, int KEYW, int PW, int S_PW, int T_PW, int ARW>
void multi_probe_wrapper(ap_uint<32>& depth,

                         // input large table
                         hls::stream<ap_uint<HASHW> >& i_hash_strm,
                         hls::stream<ap_uint<KEYW> >& i_key_strm,
                         hls::stream<ap_uint<PW> >& i_pld_strm,
                         hls::stream<bool>& i_e_strm,

                         // output for join
                         hls::stream<ap_uint<KEYW> >& o_t_key_strm,
                         hls::stream<ap_uint<T_PW> >& o_t_pld_strm,
                         hls::stream<ap_uint<ARW> >& o_nm_strm,
                         hls::stream<bool>& o_e0_strm,

                         hls::stream<ap_uint<KEYW> >& o_base_s_key_strm,
                         hls::stream<ap_uint<S_PW> >& o_base_s_pld_strm,
                         hls::stream<ap_uint<KEYW> >& o_overflow_s_key_strm,
                         hls::stream<ap_uint<S_PW> >& o_overflow_s_pld_strm,

                         ap_uint<256>* htb_buf,
                         ap_uint<256>* stb_buf,
                         ap_uint<72>* bit_vector0,
                         ap_uint<72>* bit_vector1) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    // for probe_addr_gen
    hls::stream<ap_uint<ARW> > base_addr_strm;
#pragma HLS stream variable = base_addr_strm depth = 512
#pragma HLS bind_storage variable = base_addr_strm type = fifo impl = bram
    hls::stream<ap_uint<ARW> > nm0_strm;
#pragma HLS stream variable = nm0_strm depth = 512
#pragma HLS bind_storage variable = nm0_strm type = fifo impl = bram
    hls::stream<bool> e0_strm;
#pragma HLS stream variable = e0_strm depth = 512
#pragma HLS bind_storage variable = e0_strm type = fifo impl = srl

    hls::stream<ap_uint<ARW> > overflow_addr_strm;
#pragma HLS stream variable = overflow_addr_strm depth = 8
#pragma HLS bind_storage variable = overflow_addr_strm type = fifo impl = srl
    hls::stream<ap_uint<ARW> > nm1_strm;
#pragma HLS stream variable = nm1_strm depth = 8
#pragma HLS bind_storage variable = nm1_strm type = fifo impl = srl
    hls::stream<bool> e1_strm;
#pragma HLS stream variable = e1_strm depth = 8
#pragma HLS bind_storage variable = e1_strm type = fifo impl = srl

    // calculate number of srow need to probe in HBM/DDR
    multi_probe_htb<HASHW, KEYW, PW, T_PW, ARW>(depth, i_hash_strm, i_key_strm, i_pld_strm, i_e_strm,

                                                base_addr_strm, nm0_strm, e0_strm, overflow_addr_strm, nm1_strm,
                                                e1_strm, o_t_key_strm, o_t_pld_strm, o_nm_strm, o_e0_strm,

                                                bit_vector0, bit_vector1);

    join_v3::sc::probe_base_stb<KEYW, S_PW, ARW>(depth, base_addr_strm, nm0_strm, e0_strm,

                                                 o_base_s_key_strm, o_base_s_pld_strm,

                                                 stb_buf);

    join_v3::sc::probe_overflow_stb<KEYW, S_PW, ARW>(overflow_addr_strm, nm1_strm, e1_strm,

                                                     o_overflow_s_key_strm, o_overflow_s_pld_strm,

                                                     htb_buf);
}

//----------------------------------------------build+merge+probe------------------------------------------------

/// @brief Top function of hash multi join PU
template <int HASH_MODE, int HASHWH, int HASHWL, int KEYW, int S_PW, int T_PW, int ARW>
void build_merge_multi_probe_wrapper(
    // input status
    ap_uint<32>& depth,

    // input table
    hls::stream<ap_uint<HASHWL> >& i_hash_strm,
    hls::stream<ap_uint<KEYW> >& i_key_strm,
    hls::stream<ap_uint<T_PW> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    // output for join
    hls::stream<ap_uint<KEYW> >& o_t_key_strm,
    hls::stream<ap_uint<T_PW> >& o_t_pld_strm,
    hls::stream<ap_uint<ARW> >& o_nm_strm,
    hls::stream<bool>& o_e_strm,

    hls::stream<ap_uint<KEYW> >& o_base_s_key_strm,
    hls::stream<ap_uint<S_PW> >& o_base_s_pld_strm,
    hls::stream<ap_uint<KEYW> >& o_overflow_s_key_strm,
    hls::stream<ap_uint<S_PW> >& o_overflow_s_pld_strm,

    ap_uint<256>* htb_buf,
    ap_uint<256>* stb_buf) {
#pragma HLS INLINE off

    const int PW = (S_PW > T_PW) ? S_PW : T_PW;
    // alllocate uram storage
    const int HASH_DEPTH = (1 << HASHWL) / 3 + 1;

#ifndef __SYNTHESIS__

    ap_uint<72>* bit_vector0;
    ap_uint<72>* bit_vector1;
    bit_vector0 = (ap_uint<72>*)malloc(HASH_DEPTH * sizeof(ap_uint<72>));
    bit_vector1 = (ap_uint<72>*)malloc(HASH_DEPTH * sizeof(ap_uint<72>));

#else

    ap_uint<72> bit_vector0[HASH_DEPTH];
#pragma HLS ARRAY_PARTITION variable = bit_vector0 block factor = 4 dim = 1
#pragma HLS bind_storage variable = bit_vector0 type = ram_2p impl = uram
    ap_uint<72> bit_vector1[HASH_DEPTH];
#pragma HLS ARRAY_PARTITION variable = bit_vector1 block factor = 4 dim = 1
#pragma HLS bind_storage variable = bit_vector1 type = ram_2p impl = uram

#endif

    ap_uint<32> overflow_length = 0;

    // initilize uram by previous hash build or probe
    join_v3::sc::initiate_uram<HASHWL, ARW>(bit_vector0, bit_vector1);

#ifndef __SYNTHESIS__
    std::cout << "----------------------build------------------------" << std::endl;
#endif

    // build
    join_v3::sc::build_wrapper<HASHWL, KEYW, PW, S_PW, ARW>(
        // input status
        depth, overflow_length,

        // input s-table
        i_hash_strm, i_key_strm, i_pld_strm, i_e_strm,

        // HBM/DDR
        stb_buf, bit_vector0, bit_vector1);

    // merge
    if (overflow_length > 0) {
// the first time to probe, need to fully build bitmap
#ifndef __SYNTHESIS__
        std::cout << "----------------------merge------------------------" << std::endl;
#endif

        join_v3::sc::merge_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, ARW>(
            // input status
            depth, overflow_length,

            htb_buf, stb_buf, bit_vector1);
    }

// probe
#ifndef __SYNTHESIS__
    std::cout << "-----------------------Probe------------------------" << std::endl;
#endif

    multi_probe_wrapper<HASHWL, KEYW, PW, S_PW, T_PW, ARW>(depth,
                                                           // join_flag_strm,

                                                           // input large table
                                                           i_hash_strm, i_key_strm, i_pld_strm, i_e_strm,

                                                           // output for join
                                                           o_t_key_strm, o_t_pld_strm, o_nm_strm, o_e_strm,
                                                           o_base_s_key_strm, o_base_s_pld_strm, o_overflow_s_key_strm,
                                                           o_overflow_s_pld_strm,
                                                           // join_flag_strm_o,

                                                           htb_buf, stb_buf, bit_vector0, bit_vector1);

#ifndef __SYNTHESIS__

    free(bit_vector0);
    free(bit_vector1);

#endif
}

//-----------------------------------------------join-----------------------------------------------

/// @brief hash hit branch of t_strm
template <int KEYW, int S_PW, int T_PW, int ARW>
void join_unit_1(

    ap_uint<32>& join_depth,
    hls::stream<ap_uint<3> >& join_flag_strm_o,
    // input large table
    hls::stream<ap_uint<KEYW> >& i1_t_key_strm,
    hls::stream<ap_uint<T_PW> >& i1_t_pld_strm,
    hls::stream<ap_uint<ARW> >& i1_nm_strm,
    hls::stream<bool>& i1_e0_strm,

    // input small table
    hls::stream<ap_uint<KEYW> >& i_base_s_key_strm,
    hls::stream<ap_uint<S_PW> >& i_base_s_pld_strm,
    hls::stream<ap_uint<KEYW> >& i_overflow_s_key_strm,
    hls::stream<ap_uint<S_PW> >& i_overflow_s_pld_strm,

    // output join result
    hls::stream<ap_uint<KEYW + S_PW + T_PW> >& o_j_strm,
    hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    ap_uint<KEYW> s1_key;
    ap_uint<S_PW> s1_pld;
    ap_uint<KEYW> t1_key;
    ap_uint<T_PW> t1_pld;
    ap_uint<KEYW + S_PW + T_PW> j;
    ap_uint<ARW> depth = join_depth;

    ap_uint<3> join_flag_t = join_flag_strm_o.read();
    int join_flag_i = join_flag_t;
    xf::database::enums::JoinType join_flag = static_cast<xf::database::enums::JoinType>(join_flag_i);

    bool t1_last = i1_e0_strm.read();
JOIN_LOOP_1:
    while (!t1_last) {
        t1_key = i1_t_key_strm.read();
        t1_pld = i1_t_pld_strm.read();
        ap_uint<ARW> nm_1 = i1_nm_strm.read();
        t1_last = i1_e0_strm.read();
        bool flag = 0;
        ap_uint<ARW> base1_nm, overflow1_nm;
        if (nm_1 > depth) {
            base1_nm = depth;
            overflow1_nm = nm_1 - depth;
        } else {
            base1_nm = nm_1;
            overflow1_nm = 0;
        }
        j(KEYW + S_PW + T_PW - 1, S_PW + T_PW) = t1_key;
        if (T_PW > 0) j(T_PW - 1, 0) = t1_pld;
    JOIN_COMPARE_LOOP:
        while (base1_nm > 0 || overflow1_nm > 0) {
#pragma HLS PIPELINE II = 1

            if (base1_nm > 0) {
                s1_key = i_base_s_key_strm.read();
                s1_pld = i_base_s_pld_strm.read();
                base1_nm--;
            } else if (overflow1_nm > 0) {
                s1_key = i_overflow_s_key_strm.read();
                s1_pld = i_overflow_s_pld_strm.read();
                overflow1_nm--;
            }

            if (S_PW > 0) j(S_PW + T_PW - 1, T_PW) = s1_pld;

            if (join_flag == xf::database::enums::JT_INNER && s1_key == t1_key) {
                o_j_strm.write(j);
                o_e_strm.write(false);
            }

            flag = flag || (join_flag == 3 && s1_key == t1_key && s1_pld.range(31, 0) != t1_pld.range(31, 0)) ||
                   (join_flag != 3 && s1_key == t1_key);
        }

        if (join_flag == xf::database::enums::JT_ANTI && !flag) {
            o_j_strm.write(j);
            o_e_strm.write(false);
        } else if ((join_flag == xf::database::enums::JT_SEMI || join_flag == 3) && flag) {
            o_j_strm.write(j);
            o_e_strm.write(false);
        }
    }
    o_j_strm.write(0);
    o_e_strm.write(true);
}

/// @brief hash unhit branch of t_strm
template <int KEYW, int S_PW, int T_PW, int ARW>
void join_unit_2(
    // input large table
    hls::stream<ap_uint<3> >& join_flag_strm_o,
    hls::stream<ap_uint<KEYW> >& i2_t_key_strm,
    hls::stream<ap_uint<T_PW> >& i2_t_pld_strm,
    hls::stream<ap_uint<ARW> >& i2_nm_strm,
    hls::stream<bool>& i2_e0_strm,

    // output join result
    hls::stream<ap_uint<KEYW + S_PW + T_PW> >& o2_j_strm,
    hls::stream<bool>& o2_e_strm) {
#pragma HLS INLINE off

    ap_uint<KEYW> s2_key;
    ap_uint<S_PW> s2_pld;
    ap_uint<KEYW> t2_key;
    ap_uint<T_PW> t2_pld;
    ap_uint<KEYW + S_PW + T_PW> j2;

    ap_uint<3> join_flag_t = join_flag_strm_o.read();
    int join_flag_i = join_flag_t;
    xf::database::enums::JoinType join_flag = static_cast<xf::database::enums::JoinType>(join_flag_i);

    bool t2_last = i2_e0_strm.read();
JOIN_LOOP_2:
    while (!t2_last) {
        t2_key = i2_t_key_strm.read();
        t2_pld = i2_t_pld_strm.read();
        ap_uint<ARW> nm_2 = i2_nm_strm.read();
        t2_last = i2_e0_strm.read();

        if (join_flag == xf::database::enums::JT_ANTI) {
            if (nm_2 == 0) {
                j2(KEYW + S_PW + T_PW - 1, S_PW + T_PW) = t2_key;
                if (S_PW > 0) {
                    j2(S_PW + T_PW - 1, T_PW) = 0;
                }
                if (T_PW > 0) {
                    j2(T_PW - 1, 0) = t2_pld;
                }

                o2_j_strm.write(j2);
                o2_e_strm.write(false);
            }
        }
    }

    // for do while in collect
    o2_j_strm.write(0);
    o2_e_strm.write(true);
}

/// @brief split t_strm to hash hit branch and hash unhit branch
template <int KEYW, int S_PW, int T_PW, int ARW>
void split_stream(

    // input
    hls::stream<ap_uint<3> >& join_flag_strm_o,
    hls::stream<ap_uint<KEYW> >& i_t_key_strm,
    hls::stream<ap_uint<T_PW> >& i_t_pld_strm,
    hls::stream<ap_uint<ARW> >& i_nm_strm,
    hls::stream<bool>& i_e0_strm,

    // output
    hls::stream<ap_uint<3> >& join1_flag_strm,
    hls::stream<ap_uint<KEYW> >& i1_t_key_strm,
    hls::stream<ap_uint<T_PW> >& i1_t_pld_strm,
    hls::stream<ap_uint<ARW> >& i1_nm_strm,
    hls::stream<bool>& i1_e0_strm,

    hls::stream<ap_uint<3> >& join2_flag_strm,
    hls::stream<ap_uint<KEYW> >& i2_t_key_strm,
    hls::stream<ap_uint<T_PW> >& i2_t_pld_strm,
    hls::stream<ap_uint<ARW> >& i2_nm_strm,
    hls::stream<bool>& i2_e0_strm) {
#pragma HLS INLINE off

    ap_uint<KEYW> t_key;
    ap_uint<T_PW> t_pld;
    ap_uint<3> join_flag = join_flag_strm_o.read();

    join1_flag_strm.write(join_flag);
    join2_flag_strm.write(join_flag);

    bool t_last = i_e0_strm.read();
DIVIDE_LOOP:

    while (!t_last) {
        t_key = i_t_key_strm.read();
        t_pld = i_t_pld_strm.read();
        ap_uint<ARW> nm = i_nm_strm.read();
        t_last = i_e0_strm.read();
        if (nm > 0) {
            i1_t_key_strm.write(t_key);
            i1_t_pld_strm.write(t_pld);
            i1_nm_strm.write(nm);
            i1_e0_strm.write(false);

        } else if (nm == 0) {
            i2_t_key_strm.write(t_key);
            i2_t_pld_strm.write(t_pld);
            i2_nm_strm.write(nm);
            i2_e0_strm.write(false);
        }
    }

    i1_e0_strm.write(true);
    i2_e0_strm.write(true);
}

/// @brief combine hash hit and unhit branches
template <int KEYW, int S_PW, int T_PW, int ARW>
void combine_stream(

    // input
    hls::stream<ap_uint<KEYW + S_PW + T_PW> > i_j_strm[2],
    hls::stream<bool> i_e_strm[2],

    // output join result
    hls::stream<ap_uint<KEYW + S_PW + T_PW> >& o_j_strm,
    hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    ap_uint<2> last = 0;
    ap_uint<2> empty_e = 0;
    ap_uint<2> rd_e = 0;
    ap_uint<KEYW + S_PW + T_PW> j_arr[2];
#pragma HLS array_partition variable = j_arr dim = 1

    do {
#pragma HLS pipeline II = 1
        for (int i = 0; i < 2; i++) {
#pragma HLS unroll
            empty_e[i] = !i_e_strm[i].empty() && !last[i];
        }

        for (int i = 0; i < 2; i++) {
#pragma HLS unroll
            ap_uint<2> t_e = 0;
            if (i > 0) t_e = empty_e(i - 1, 0);
            rd_e[i] = t_e > 0 ? (bool)0 : (bool)empty_e[i];
        }

        for (int i = 0; i < 2; i++) {
#pragma HLS unroll
            if (rd_e[i]) {
                j_arr[i] = i_j_strm[i].read();
                last[i] = i_e_strm[i].read();
            }
        }

        ap_uint<3> id = join_v2::mux<2>(rd_e);
        ap_uint<KEYW + S_PW + T_PW> j = j_arr[id];
        bool valid_n = last[id];

        if (!valid_n && rd_e != 0) {
            o_j_strm.write(j);
            o_e_strm.write(false);
        }
    } while (last != 3);

    o_j_strm.write(0);
    o_e_strm.write(true);
}

/// @brief top function of multi join
template <int KEYW, int S_PW, int T_PW, int ARW>
void multi_join_unit(
#ifndef __SYNTHESIS__
    int pu_id,
#endif

    ap_uint<32>& depth,
    hls::stream<ap_uint<3> >& join_flag_strm,
    // input large table
    hls::stream<ap_uint<KEYW> >& i_t_key_strm,
    hls::stream<ap_uint<T_PW> >& i_t_pld_strm,
    hls::stream<ap_uint<ARW> >& i_nm_strm,
    hls::stream<bool>& i_e0_strm,

    // input small table
    hls::stream<ap_uint<KEYW> >& i_base_s_key_strm,
    hls::stream<ap_uint<S_PW> >& i_base_s_pld_strm,
    hls::stream<ap_uint<KEYW> >& i_overflow_s_key_strm,
    hls::stream<ap_uint<S_PW> >& i_overflow_s_pld_strm,

    // output join result
    hls::stream<ap_uint<KEYW + S_PW + T_PW> >& o_j_strm,
    hls::stream<bool>& o_e_strm) {

#pragma HLS INLINE off
#pragma HLS DATAFLOW

#ifndef __SYNTHESIS__
    unsigned int cnt0 = 0;
    unsigned int cnt1 = 0;
    unsigned int cnt2 = 0;
    unsigned int cnt3 = 0;

    bool hit_failed;
#endif

    hls::stream<ap_uint<KEYW> > i1_t_key_strm;
#pragma HLS STREAM variable = i1_t_key_strm depth = 1024
#pragma HLS bind_storage variable = i1_t_key_strm type = fifo impl = bram
    hls::stream<ap_uint<T_PW> > i1_t_pld_strm;
#pragma HLS STREAM variable = i1_t_pld_strm depth = 1024
#pragma HLS bind_storage variable = i1_t_pld_strm type = fifo impl = bram
    hls::stream<ap_uint<ARW> > i1_nm_strm;
#pragma HLS STREAM variable = i1_nm_strm depth = 1024
#pragma HLS bind_storage variable = i1_nm_strm type = fifo impl = bram
    hls::stream<ap_uint<3> > join1_flag_strm;
#pragma HLS STREAM variable = join1_flag_strm depth = 16
#pragma HLS bind_storage variable = join1_flag_strm type = fifo impl = srl
    hls::stream<bool> i1_e0_strm;
#pragma HLS STREAM variable = i1_e0_strm depth = 1024
#pragma HLS bind_storage variable = i1_e0_strm type = fifo impl = srl
    hls::stream<ap_uint<KEYW> > i2_t_key_strm;
#pragma HLS STREAM variable = i2_t_key_strm depth = 16
#pragma HLS bind_storage variable = i2_t_key_strm type = fifo impl = srl
    hls::stream<ap_uint<T_PW> > i2_t_pld_strm;
#pragma HLS STREAM variable = i2_t_pld_strm depth = 16
#pragma HLS bind_storage variable = i2_t_pld_strm type = fifo impl = srl
    hls::stream<ap_uint<ARW> > i2_nm_strm;
#pragma HLS STREAM variable = i2_nm_strm depth = 16
#pragma HLS bind_storage variable = i2_nm_strm type = fifo impl = srl
    hls::stream<ap_uint<3> > join2_flag_strm;
#pragma HLS STREAM variable = join2_flag_strm depth = 16
#pragma HLS bind_storage variable = join2_flag_strm type = fifo impl = srl
    hls::stream<bool> i2_e0_strm;
#pragma HLS STREAM variable = i2_e0_strm depth = 16
#pragma HLS bind_storage variable = i2_e0_strm type = fifo impl = srl

    hls::stream<ap_uint<KEYW + S_PW + T_PW> > i_j_strm[2];
#pragma HLS STREAM variable = i_j_strm depth = 16
#pragma HLS array_partition variable = i_j_strm dim = 0
#pragma HLS bind_storage variable = i_j_strm type = fifo impl = srl
    hls::stream<bool> i_e_strm[2];
#pragma HLS STREAM variable = i_e_strm depth = 16
#pragma HLS array_partition variable = i_e_strm dim = 0
#pragma HLS bind_storage variable = i_e_strm type = fifo impl = srl

    split_stream<KEYW, S_PW, T_PW, ARW>(join_flag_strm, i_t_key_strm, i_t_pld_strm, i_nm_strm, i_e0_strm,
                                        join1_flag_strm, i1_t_key_strm, i1_t_pld_strm, i1_nm_strm, i1_e0_strm,
                                        join2_flag_strm, i2_t_key_strm, i2_t_pld_strm, i2_nm_strm, i2_e0_strm);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << std::dec << "i1_t_key_strm output " << i1_t_key_strm.size() << std::endl;
    std::cout << std::dec << "i2_t_key_strm output " << i2_t_key_strm.size() << std::endl;
#endif
#endif

    join_unit_1<KEYW, S_PW, T_PW, ARW>(depth, join1_flag_strm, i1_t_key_strm, i1_t_pld_strm, i1_nm_strm, i1_e0_strm,
                                       i_base_s_key_strm, i_base_s_pld_strm, i_overflow_s_key_strm,
                                       i_overflow_s_pld_strm, i_j_strm[0], i_e_strm[0]);

    join_unit_2<KEYW, S_PW, T_PW, ARW>(join2_flag_strm, i2_t_key_strm, i2_t_pld_strm, i2_nm_strm, i2_e0_strm,
                                       i_j_strm[1], i_e_strm[1]);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << std::dec << "Join Unit 1 output " << i_j_strm[0].size() << std::endl;
    std::cout << std::dec << "Join Unit 2 output " << i_j_strm[1].size() << std::endl;
#endif
#endif

    combine_stream<KEYW, S_PW, T_PW, ARW>(i_j_strm, i_e_strm, o_j_strm, o_e_strm);
#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << std::dec << "Join Unit output " << o_j_strm.size() << std::endl;
#endif
#endif
}

template <int _NOut>
void dup_join_flag(hls::stream<ap_uint<3> >& join_flag_strm, hls::stream<ap_uint<3> > join_flags[_NOut]) {
    ap_uint<3> flag = join_flag_strm.read();
    for (int i = 0; i < _NOut; ++i) {
#pragma HLS unroll
        join_flags[i].write(flag);
    }
}

} // namespace hash_multi_join
} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/**
 * @brief Multi-PU Hash-Multi-Join primitive, using multiple DDR/HBM buffers.
 *
 * This primitive shares most of the structure of ``hashJoinV3``.
 * The inner table should be fed once, followed by the outer table once.
 *
 * @tparam HASH_MODE 0 for radix and 1 for Jenkin's Lookup3 hash.
 * @tparam KEYW width of key, in bit.
 * @tparam PW width of max payload, in bit.
 * @tparam S_PW width of payload of small table.
 * @tparam B_PW width of payload of big table.
 * @tparam HASHWH number of hash bits used for PU/buffer selection, 1~3.
 * @tparam HASHWL number of hash bits used for hash-table in PU.
 * @tparam ARW width of address, larger than 24 is suggested.
 * @tparam CH_NM number of input channels, 1,2,4.
 *
 * @param join_flag_strm specifies the join type, this flag is only read once.
 *
 * @param k0_strm_arry input of key columns of both tables.
 * @param p0_strm_arry input of payload columns of both tables.
 * @param e0_strm_arry input of end signal of both tables.
 *
 * @param htb0_buf HBM/DDR buffer of hash_table0
 * @param htb1_buf HBM/DDR buffer of hash_table1
 * @param htb2_buf HBM/DDR buffer of hash_table2
 * @param htb3_buf HBM/DDR buffer of hash_table3
 * @param htb4_buf HBM/DDR buffer of hash_table4
 * @param htb5_buf HBM/DDR buffer of hash_table5
 * @param htb6_buf HBM/DDR buffer of hash_table6
 * @param htb7_buf HBM/DDR buffer of hash_table7
 *
 * @param stb0_buf HBM/DDR buffer of PU0
 * @param stb1_buf HBM/DDR buffer of PU1
 * @param stb2_buf HBM/DDR buffer of PU2
 * @param stb3_buf HBM/DDR buffer of PU3
 * @param stb4_buf HBM/DDR buffer of PU4
 * @param stb5_buf HBM/DDR buffer of PU5
 * @param stb6_buf HBM/DDR buffer of PU6
 * @param stb7_buf HBM/DDR buffer of PU7
 *
 * @param pu_begin_status_strms constains depth of hash, row number of join result
 * @param pu_end_status_strms constains depth of hash, row number of join result
 *
 * @param j_strm output of joined result
 * @param j_e_strm end flag of joined result
 */
template <int HASH_MODE, int KEYW, int PW, int S_PW, int B_PW, int HASHWH, int HASHWL, int ARW, int CH_NM>
void hashMultiJoin(
    // type
    hls::stream<ap_uint<3> >& join_flag_strm,
    // input
    hls::stream<ap_uint<KEYW> > k0_strm_arry[CH_NM],
    hls::stream<ap_uint<PW> > p0_strm_arry[CH_NM],
    hls::stream<bool> e0_strm_arry[CH_NM],

    // output hash table
    ap_uint<256>* htb0_buf,
    ap_uint<256>* htb1_buf,
    ap_uint<256>* htb2_buf,
    ap_uint<256>* htb3_buf,
    ap_uint<256>* htb4_buf,
    ap_uint<256>* htb5_buf,
    ap_uint<256>* htb6_buf,
    ap_uint<256>* htb7_buf,

    // output
    ap_uint<256>* stb0_buf,
    ap_uint<256>* stb1_buf,
    ap_uint<256>* stb2_buf,
    ap_uint<256>* stb3_buf,
    ap_uint<256>* stb4_buf,
    ap_uint<256>* stb5_buf,
    ap_uint<256>* stb6_buf,
    ap_uint<256>* stb7_buf,

    hls::stream<ap_uint<32> >& pu_begin_status_strms,
    hls::stream<ap_uint<32> >& pu_end_status_strms,

    hls::stream<ap_uint<KEYW + S_PW + B_PW> >& j_strm,
    hls::stream<bool>& j_e_strm) {
    enum { PU = (1 << HASHWH) }; // high hash for distribution.

#pragma HLS DATAFLOW

    hls::stream<ap_uint<3> > join_flag_strms[8];
    details::hash_multi_join::dup_join_flag<8>(join_flag_strm, join_flag_strms);

    ap_uint<32> depth;
    ap_uint<32> join_num;

    // dispatch k0_strm_arry, p0_strm_arry, e0strm_arry to channel1-4
    // Channel1
    hls::stream<ap_uint<KEYW> > k1_strm_arry_c0[PU];
#pragma HLS stream variable = k1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c0 dim = 1
#pragma HLS bind_storage variable = k1_strm_arry_c0 type = fifo impl = srl
    hls::stream<ap_uint<PW> > p1_strm_arry_c0[PU];
#pragma HLS stream variable = p1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c0 dim = 1
#pragma HLS bind_storage variable = p1_strm_arry_c0 type = fifo impl = srl
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_c0[PU];
#pragma HLS stream variable = hash_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c0 dim = 1
#pragma HLS bind_storage variable = hash_strm_arry_c0 type = fifo impl = srl
    hls::stream<bool> e1_strm_arry_c0[PU];
#pragma HLS stream variable = e1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c0 dim = 1

    // Channel2
    hls::stream<ap_uint<KEYW> > k1_strm_arry_c1[PU];
#pragma HLS stream variable = k1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c1 dim = 1
#pragma HLS bind_storage variable = k1_strm_arry_c1 type = fifo impl = srl
    hls::stream<ap_uint<PW> > p1_strm_arry_c1[PU];
#pragma HLS stream variable = p1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c1 dim = 1
#pragma HLS bind_storage variable = p1_strm_arry_c1 type = fifo impl = srl
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_c1[PU];
#pragma HLS stream variable = hash_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c1 dim = 1
#pragma HLS bind_storage variable = hash_strm_arry_c1 type = fifo impl = srl
    hls::stream<bool> e1_strm_arry_c1[PU];
#pragma HLS stream variable = e1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c1 dim = 1

    // Channel3
    hls::stream<ap_uint<KEYW> > k1_strm_arry_c2[PU];
#pragma HLS stream variable = k1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c2 dim = 1
#pragma HLS bind_storage variable = k1_strm_arry_c2 type = fifo impl = srl
    hls::stream<ap_uint<PW> > p1_strm_arry_c2[PU];
#pragma HLS stream variable = p1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c2 dim = 1
#pragma HLS bind_storage variable = p1_strm_arry_c2 type = fifo impl = srl
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_c2[PU];
#pragma HLS stream variable = hash_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c2 dim = 1
#pragma HLS bind_storage variable = hash_strm_arry_c2 type = fifo impl = srl
    hls::stream<bool> e1_strm_arry_c2[PU];
#pragma HLS stream variable = e1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c2 dim = 1

    // Channel4
    hls::stream<ap_uint<KEYW> > k1_strm_arry_c3[PU];
#pragma HLS stream variable = k1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c3 dim = 1
#pragma HLS bind_storage variable = k1_strm_arry_c3 type = fifo impl = srl
    hls::stream<ap_uint<PW> > p1_strm_arry_c3[PU];
#pragma HLS stream variable = p1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c3 dim = 1
#pragma HLS bind_storage variable = p1_strm_arry_c3 type = fifo impl = srl
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_c3[PU];
#pragma HLS stream variable = hash_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c3 dim = 1
#pragma HLS bind_storage variable = hash_strm_arry_c3 type = fifo impl = srl
    hls::stream<bool> e1_strm_arry_c3[PU];
#pragma HLS stream variable = e1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c3 dim = 1

    // merge channel1-channel4 to here, then perform build or probe
    hls::stream<ap_uint<KEYW> > k1_strm_arry[PU];
#pragma HLS stream variable = k1_strm_arry depth = 8
#pragma HLS array_partition variable = k1_strm_arry dim = 1
#pragma HLS bind_storage variable = k1_strm_arry type = fifo impl = srl
    hls::stream<ap_uint<PW> > p1_strm_arry[PU];
#pragma HLS stream variable = p1_strm_arry depth = 8
#pragma HLS array_partition variable = p1_strm_arry dim = 1
#pragma HLS bind_storage variable = p1_strm_arry type = fifo impl = srl
    hls::stream<ap_uint<HASHWL> > hash_strm_arry[PU];
#pragma HLS stream variable = hash_strm_arry depth = 8
#pragma HLS array_partition variable = hash_strm_arry dim = 1
    hls::stream<bool> e1_strm_arry[PU];
#pragma HLS stream variable = e1_strm_arry depth = 8
#pragma HLS array_partition variable = e1_strm_arry dim = 1

    // output of probe for join
    hls::stream<ap_uint<KEYW> > t_key_strm_arry[PU];
#pragma HLS stream variable = t_key_strm_arry depth = 16
#pragma HLS array_partition variable = t_key_strm_arry dim = 1
#pragma HLS bind_storage variable = t_key_strm_arry type = fifo impl = srl
    hls::stream<ap_uint<B_PW> > t_pld_strm_arry[PU];
#pragma HLS stream variable = t_pld_strm_arry depth = 16
#pragma HLS array_partition variable = t_pld_strm_arry dim = 1
#pragma HLS bind_storage variable = t_pld_strm_arry type = fifo impl = srl
    hls::stream<ap_uint<ARW> > nm_strm_arry[PU];
#pragma HLS stream variable = nm_strm_arry depth = 16
#pragma HLS array_partition variable = nm_strm_arry dim = 1
#pragma HLS bind_storage variable = nm_strm_arry type = fifo impl = srl
    hls::stream<bool> e2_strm_arry[PU];
#pragma HLS stream variable = e2_strm_arry depth = 16
#pragma HLS array_partition variable = e2_strm_arry dim = 1
#pragma HLS bind_storage variable = e2_strm_arry type = fifo impl = srl

    hls::stream<ap_uint<KEYW> > s_base_key_strm_arry[PU];
#pragma HLS stream variable = s_base_key_strm_arry depth = 512
#pragma HLS array_partition variable = s_base_key_strm_arry dim = 1
#pragma HLS bind_storage variable = s_base_key_strm_arry type = fifo impl = bram
    hls::stream<ap_uint<S_PW> > s_base_pld_strm_arry[PU];
#pragma HLS stream variable = s_base_pld_strm_arry depth = 512
#pragma HLS array_partition variable = s_base_pld_strm_arry dim = 1
#pragma HLS bind_storage variable = s_base_pld_strm_arry type = fifo impl = bram
    hls::stream<ap_uint<KEYW> > s_overflow_key_strm_arry[PU];
#pragma HLS stream variable = s_overflow_key_strm_arry depth = 512
#pragma HLS array_partition variable = s_overflow_key_strm_arry dim = 1
#pragma HLS bind_storage variable = s_overflow_key_strm_arry type = fifo impl = bram
    hls::stream<ap_uint<S_PW> > s_overflow_pld_strm_arry[PU];
#pragma HLS stream variable = s_overflow_pld_strm_arry depth = 512
#pragma HLS array_partition variable = s_overflow_pld_strm_arry dim = 1
#pragma HLS bind_storage variable = s_overflow_pld_strm_arry type = fifo impl = bram

    // output of join for collect
    hls::stream<ap_uint<KEYW + S_PW + B_PW> > j0_strm_arry[PU];
#pragma HLS stream variable = j0_strm_arry depth = 512
#pragma HLS array_partition variable = j0_strm_arry dim = 1
#pragma HLS bind_storage variable = j0_strm_arry type = fifo impl = bram
    hls::stream<bool> e3_strm_arry[PU];
#pragma HLS array_partition variable = e3_strm_arry dim = 1
#pragma HLS stream variable = e3_strm_arry depth = 512
#pragma HLS bind_storage variable = e3_strm_arry type = fifo impl = srl

//------------------------------read status-----------------------------------
#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "------------------------read status------------------------" << std::endl;
#endif
#endif
    details::join_v3::sc::read_status<PU>(pu_begin_status_strms, depth);

//---------------------------------dispatch PU-------------------------------
#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "------------------------dispatch PU------------------------" << std::endl;
#endif
#endif

    if (CH_NM >= 1) {
        details::join_v3::sc::dispatch_wrapper<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(
            k0_strm_arry[0], p0_strm_arry[0], e0_strm_arry[0], k1_strm_arry_c0, p1_strm_arry_c0, hash_strm_arry_c0,
            e1_strm_arry_c0);
    }

    if (CH_NM >= 2) {
        details::join_v3::sc::dispatch_wrapper<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(
            k0_strm_arry[1], p0_strm_arry[1], e0_strm_arry[1], k1_strm_arry_c1, p1_strm_arry_c1, hash_strm_arry_c1,
            e1_strm_arry_c1);
    }

    if (CH_NM >= 4) {
        details::join_v3::sc::dispatch_wrapper<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(
            k0_strm_arry[2], p0_strm_arry[2], e0_strm_arry[2], k1_strm_arry_c2, p1_strm_arry_c2, hash_strm_arry_c2,
            e1_strm_arry_c2);

        details::join_v3::sc::dispatch_wrapper<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(
            k0_strm_arry[3], p0_strm_arry[3], e0_strm_arry[3], k1_strm_arry_c3, p1_strm_arry_c3, hash_strm_arry_c3,
            e1_strm_arry_c3);
    }

//---------------------------------merge PU---------------------------------
#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "------------------------merge PU------------------------" << std::endl;
#endif
#endif

    if (CH_NM == 1) {
        for (int p = 0; p < PU; ++p) {
#pragma HLS unroll
            details::join_v3::sc::merge1_1_wrapper(
                k1_strm_arry_c0[p], p1_strm_arry_c0[p], hash_strm_arry_c0[p], e1_strm_arry_c0[p],

                k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p], e1_strm_arry[p]);
        }
    } else if (CH_NM == 2) {
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            details::join_v3::sc::merge2_1_wrapper(
                k1_strm_arry_c0[p], k1_strm_arry_c1[p], p1_strm_arry_c0[p], p1_strm_arry_c1[p], hash_strm_arry_c0[p],
                hash_strm_arry_c1[p], e1_strm_arry_c0[p], e1_strm_arry_c1[p],

                k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p], e1_strm_arry[p]);
        }
    } else {
        // CH_NM == 4
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            details::join_v3::sc::merge4_1_wrapper(
                k1_strm_arry_c0[p], k1_strm_arry_c1[p], k1_strm_arry_c2[p], k1_strm_arry_c3[p], p1_strm_arry_c0[p],
                p1_strm_arry_c1[p], p1_strm_arry_c2[p], p1_strm_arry_c3[p], hash_strm_arry_c0[p], hash_strm_arry_c1[p],
                hash_strm_arry_c2[p], hash_strm_arry_c3[p], e1_strm_arry_c0[p], e1_strm_arry_c1[p], e1_strm_arry_c2[p],
                e1_strm_arry_c3[p],

                k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p], e1_strm_arry[p]);
        }
    }

    //-------------------------------PU----------------------------------------
    if (PU >= 1) {
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU0------------------------" << std::endl;
#endif
#endif
        details::hash_multi_join::build_merge_multi_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
            // input status
            depth,

            // input table
            hash_strm_arry[0], k1_strm_arry[0], p1_strm_arry[0], e1_strm_arry[0],

            // output for join
            t_key_strm_arry[0], t_pld_strm_arry[0], nm_strm_arry[0], e2_strm_arry[0], s_base_key_strm_arry[0],
            s_base_pld_strm_arry[0], s_overflow_key_strm_arry[0], s_overflow_pld_strm_arry[0],

            // HBM/DDR
            htb0_buf, stb0_buf);
    }

    if (PU >= 2) {
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU1------------------------" << std::endl;
#endif
#endif
        details::hash_multi_join::build_merge_multi_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
            // input status
            depth,

            // input t-table
            hash_strm_arry[1], k1_strm_arry[1], p1_strm_arry[1], e1_strm_arry[1],

            // output for join
            t_key_strm_arry[1], t_pld_strm_arry[1], nm_strm_arry[1], e2_strm_arry[1], s_base_key_strm_arry[1],
            s_base_pld_strm_arry[1], s_overflow_key_strm_arry[1], s_overflow_pld_strm_arry[1],

            // HBM/DDR
            htb1_buf, stb1_buf);
    }

    if (PU >= 4) {
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU2------------------------" << std::endl;
#endif
#endif
        details::hash_multi_join::build_merge_multi_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
            // input status
            depth,

            // input t-table
            hash_strm_arry[2], k1_strm_arry[2], p1_strm_arry[2], e1_strm_arry[2],

            // output for join
            t_key_strm_arry[2], t_pld_strm_arry[2], nm_strm_arry[2], e2_strm_arry[2], s_base_key_strm_arry[2],
            s_base_pld_strm_arry[2], s_overflow_key_strm_arry[2], s_overflow_pld_strm_arry[2],

            // HBM/DDR
            htb2_buf, stb2_buf);

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU3------------------------" << std::endl;
#endif
#endif
        details::hash_multi_join::build_merge_multi_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
            // input status
            depth,

            // input t-table
            hash_strm_arry[3], k1_strm_arry[3], p1_strm_arry[3], e1_strm_arry[3],

            // output for join
            t_key_strm_arry[3], t_pld_strm_arry[3], nm_strm_arry[3], e2_strm_arry[3], s_base_key_strm_arry[3],
            s_base_pld_strm_arry[3], s_overflow_key_strm_arry[3], s_overflow_pld_strm_arry[3],

            // HBM/DDR
            htb3_buf, stb3_buf);
    }

    if (PU >= 8) {
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU4------------------------" << std::endl;
#endif
#endif
        details::hash_multi_join::build_merge_multi_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
            // input status
            depth,

            // input t-table
            hash_strm_arry[4], k1_strm_arry[4], p1_strm_arry[4], e1_strm_arry[4],

            // output for join
            t_key_strm_arry[4], t_pld_strm_arry[4], nm_strm_arry[4], e2_strm_arry[4], s_base_key_strm_arry[4],
            s_base_pld_strm_arry[4], s_overflow_key_strm_arry[4], s_overflow_pld_strm_arry[4],

            // HBM/DDR
            htb4_buf, stb4_buf);

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU5------------------------" << std::endl;
#endif
#endif
        details::hash_multi_join::build_merge_multi_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
            // input status
            depth,

            // input t-table
            hash_strm_arry[5], k1_strm_arry[5], p1_strm_arry[5], e1_strm_arry[5],

            // output for join
            t_key_strm_arry[5], t_pld_strm_arry[5], nm_strm_arry[5], e2_strm_arry[5], s_base_key_strm_arry[5],
            s_base_pld_strm_arry[5], s_overflow_key_strm_arry[5], s_overflow_pld_strm_arry[5],

            // HBM/DDR
            htb5_buf, stb5_buf);

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU6------------------------" << std::endl;
#endif
#endif
        details::hash_multi_join::build_merge_multi_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
            // input status
            depth,

            // input t-table
            hash_strm_arry[6], k1_strm_arry[6], p1_strm_arry[6], e1_strm_arry[6],

            // output for join
            t_key_strm_arry[6], t_pld_strm_arry[6], nm_strm_arry[6], e2_strm_arry[6], s_base_key_strm_arry[6],
            s_base_pld_strm_arry[6], s_overflow_key_strm_arry[6], s_overflow_pld_strm_arry[6],

            // HBM/DDR
            htb6_buf, stb6_buf);

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU7------------------------" << std::endl;
#endif
#endif
        details::hash_multi_join::build_merge_multi_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
            // input status
            depth,

            // input t-table
            hash_strm_arry[7], k1_strm_arry[7], p1_strm_arry[7], e1_strm_arry[7],

            // output for join
            t_key_strm_arry[7], t_pld_strm_arry[7], nm_strm_arry[7], e2_strm_arry[7], s_base_key_strm_arry[7],
            s_base_pld_strm_arry[7], s_overflow_key_strm_arry[7], s_overflow_pld_strm_arry[7],

            // HBM/DDR
            htb7_buf, stb7_buf);
    }

    //-----------------------------------join--------------------------------------
    for (int i = 0; i < PU; i++) {
#pragma HLS unroll
        details::hash_multi_join::multi_join_unit<KEYW, S_PW, B_PW, ARW>(
#ifndef __SYNTHESIS__
            i,
#endif

            depth, join_flag_strms[i], t_key_strm_arry[i], t_pld_strm_arry[i], nm_strm_arry[i], e2_strm_arry[i],
            s_base_key_strm_arry[i], s_base_pld_strm_arry[i], s_overflow_key_strm_arry[i], s_overflow_pld_strm_arry[i],
            j0_strm_arry[i], e3_strm_arry[i]);
    }

    //-----------------------------------Collect-----------------------------------
    details::join_v3::sc::collect_unit<PU, KEYW + S_PW + B_PW>(j0_strm_arry, e3_strm_arry, join_num, j_strm, j_e_strm);

    //------------------------------Write Status-----------------------------------
    details::join_v3::sc::write_status<PU>(pu_end_status_strms, depth, join_num);

} // hash_multi_join

} // namespace database
} // namespace xf

#undef write_bit_vector0
#undef read_bit_vector0

#undef write_bit_vector1
#undef read_bit_vector1

#endif // !defined(XF_DATABASE_HASH_MULTI_JOIN_H)
