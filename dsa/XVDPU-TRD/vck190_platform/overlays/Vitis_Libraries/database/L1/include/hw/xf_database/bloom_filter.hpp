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
 * @file bloom_filter.hpp
 * @brief BLOOMFILTER template function implementation.
 *
 * This file is part of Vitis Database Library.
 */

#ifndef XF_DATABASE_BLOOMFILTER_H
#define XF_DATABASE_BLOOMFILTER_H

#ifndef __cplusplus
#error "Vitis Database Library only works with C++."
#endif

#include "ap_int.h"
#include "hls_stream.h"
#include "xf_database/hash_lookup3.hpp"

namespace xf {
namespace database {
namespace details {

template <int STR_IN_W, int HASH_OUT_W>
inline void simple_hash_strm(hls::stream<ap_uint<STR_IN_W> >& msg_strm, hls::stream<ap_uint<HASH_OUT_W> >& hash_strm) {
// for internal test only
#pragma HLS INLINE
    ap_uint<STR_IN_W> key_t;
    key_t = msg_strm.read();
    ap_uint<HASH_OUT_W> hash = key_t(HASH_OUT_W - 1, 0);
    hash_strm.write(hash);
    //	std::cout << "hash is " << std::hex <<hash<< std::endl;
}

template <int STR_IN_W, int HASH_OUT_W>
void hashloop(hls::stream<ap_uint<STR_IN_W> >& msg_strm,
              hls::stream<bool>& in_e_strm,
              hls::stream<ap_uint<HASH_OUT_W> >& hash_strm,
              hls::stream<bool>& out_e_strm) {
    bool e = in_e_strm.read();
    while (!e) {
#pragma HLS LOOP_TRIPCOUNT max = 65536
#pragma HLS PIPELINE II = 1
        //        simple_hash_strm<STR_IN_W,HASH_OUT_W>(msg_strm,hash_strm);
        database::hashLookup3<STR_IN_W>(msg_strm, hash_strm);
        out_e_strm.write(0);
        e = in_e_strm.read();
    }
    out_e_strm.write(1);
}

template <int HASH_OUT_W, int BV_W>
inline void bv_update_one(hls::stream<ap_uint<HASH_OUT_W> >& hash_strm,
                          ap_uint<16>* bit_vector_ptr0,
                          ap_uint<16>* bit_vector_ptr1,
                          ap_uint<16>* bit_vector_ptr2) {
#pragma HLS PIPELINE II = 1
#pragma HLS inline
    ap_uint<BV_W - 3> array_idx_l_r0 = 0xffffffff;
    ap_uint<BV_W - 3> array_idx_l_r1 = 0xffffffff; // should not be accessed in first three cycles
    ap_uint<BV_W - 3> array_idx_l_r2 = 0xffffffff;
    ap_uint<BV_W - 3> array_idx_l_r3 = 0xffffffff;
    ap_uint<BV_W - 3> array_idx_h_r0 = 0xffffffff;
    ap_uint<BV_W - 3> array_idx_h_r1 = 0xffffffff;
    ap_uint<BV_W - 3> array_idx_h_r2 = 0xffffffff;
    ap_uint<BV_W - 3> array_idx_h_r3 = 0xffffffff;
    ap_uint<BV_W - 3> array_idx_a_r0 = 0xffffffff;
    ap_uint<BV_W - 3> array_idx_a_r1 = 0xffffffff;
    ap_uint<BV_W - 3> array_idx_a_r2 = 0xffffffff;
    ap_uint<BV_W - 3> array_idx_a_r3 = 0xffffffff;
    ap_uint<16> bv_val_l_c, bv_val_l_r0, bv_val_l_r1, bv_val_l_r2, bv_val_l_r3; // 16 is the width of bram used.
    ap_uint<16> bv_val_h_c, bv_val_h_r0, bv_val_h_r1, bv_val_h_r2, bv_val_h_r3;
    ap_uint<16> bv_val_a_c, bv_val_a_r0, bv_val_a_r1, bv_val_a_r2, bv_val_a_r3;

    // 0) divide the hash strm into two parts: high16bits and low16bits
    ap_uint<HASH_OUT_W> hash_res = hash_strm.read();
    ap_uint<HASH_OUT_W / 2> lowpart = hash_res(HASH_OUT_W / 2 - 1, 0);
    ap_uint<HASH_OUT_W / 2> highpart = hash_res(HASH_OUT_W - 1, HASH_OUT_W / 2);
    // obtain the final positions of three bits in bit-vector
    lowpart %= (1 << BV_W);
    highpart %= (1 << BV_W);
    ap_uint<HASH_OUT_W / 2> avgpart = (highpart + lowpart) % (1 << BV_W);
    //		std::cout <<"kernel lowpart, highpart, and avgpart is " << lowpart
    //<<", " << highpart <<", and " <<avgpart<<std::endl;
    ap_uint<HASH_OUT_W / 2 - 3> lha_max = HASH_OUT_W / 2 > BV_W ? BV_W : HASH_OUT_W / 2;
    // store bv in 16 bits bram
    // 1) calculate the array-idx and bit-idx
    ap_uint<4> bit_idx_l_c = lowpart(3, 0);
    ap_uint<BV_W - 3> array_idx_l_c = lowpart(lha_max - 1, 4);

    ap_uint<4> bit_idx_h_c = highpart(3, 0);
    ap_uint<BV_W - 3> array_idx_h_c = highpart(lha_max - 1, 4);

    ap_uint<4> bit_idx_a_c = avgpart(3, 0);
    ap_uint<BV_W - 3> array_idx_a_c = avgpart(lha_max - 1, 4);

    // 2) read Bram and select the state_c. low16
    if (array_idx_l_c == array_idx_l_r0) // no read from bram
        bv_val_l_c = bv_val_l_r0;
    else if (array_idx_l_c == array_idx_l_r1)
        bv_val_l_c = bv_val_l_r1;
    else if (array_idx_l_c == array_idx_l_r2)
        bv_val_l_c = bv_val_l_r2;
    else if (array_idx_l_c == array_idx_l_r3)
        bv_val_l_c = bv_val_l_r3;
    else
        bv_val_l_c = bit_vector_ptr0[array_idx_l_c];
    // high16
    if (array_idx_h_c == array_idx_h_r0) // no read from bram
        bv_val_h_c = bv_val_h_r0;
    else if (array_idx_h_c == array_idx_h_r1)
        bv_val_h_c = bv_val_h_r1;
    else if (array_idx_h_c == array_idx_h_r2)
        bv_val_h_c = bv_val_h_r2;
    else if (array_idx_h_c == array_idx_h_r3)
        bv_val_h_c = bv_val_h_r3;
    else
        bv_val_h_c = bit_vector_ptr1[array_idx_h_c];
    // avg16
    if (array_idx_a_c == array_idx_a_r0) // no read from bram
        bv_val_a_c = bv_val_a_r0;
    else if (array_idx_a_c == array_idx_a_r1)
        bv_val_a_c = bv_val_a_r1;
    else if (array_idx_a_c == array_idx_a_r2)
        bv_val_a_c = bv_val_a_r2;
    else if (array_idx_a_c == array_idx_a_r3)
        bv_val_a_c = bv_val_a_r3;
    else
        bv_val_a_c = bit_vector_ptr2[array_idx_a_c];

    // 3) update bv vector value
    bv_val_l_c[bit_idx_l_c] = 1;
    bv_val_h_c[bit_idx_h_c] = 1;
    bv_val_a_c[bit_idx_a_c] = 1;

// 4) write back to bram
#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "w_idx_l=" << array_idx_l_c << " ,w_idx_h=" << array_idx_h_c << " ,w_idx_a=" << array_idx_a_c
              << std::endl;
    std::cout << "val_l=" << bv_val_l_c << " ,val_h=" << bv_val_h_c << " ,val_a" << bv_val_a_c << std::endl;
#endif
#endif

    bit_vector_ptr0[array_idx_l_c] = bv_val_l_c;
    bit_vector_ptr1[array_idx_h_c] = bv_val_h_c;
    bit_vector_ptr2[array_idx_a_c] = bv_val_a_c;

    // 5) shift the whole data line 1 cycle for bv_val and addr
    // low16
    bv_val_l_r3 = bv_val_l_r2;
    bv_val_l_r2 = bv_val_l_r1;
    bv_val_l_r1 = bv_val_l_r0;
    bv_val_l_r0 = bv_val_l_c;
    array_idx_l_r3 = array_idx_l_r2;
    array_idx_l_r2 = array_idx_l_r1;
    array_idx_l_r1 = array_idx_l_r0;
    array_idx_l_r0 = array_idx_l_c;
    // high16
    bv_val_h_r3 = bv_val_h_r2;
    bv_val_h_r2 = bv_val_h_r1;
    bv_val_h_r1 = bv_val_h_r0;
    bv_val_h_r0 = bv_val_h_c;
    array_idx_h_r3 = array_idx_h_r2;
    array_idx_h_r2 = array_idx_h_r1;
    array_idx_h_r1 = array_idx_h_r0;
    array_idx_h_r0 = array_idx_h_c;
    // avg16
    bv_val_a_r3 = bv_val_a_r2;
    bv_val_a_r2 = bv_val_a_r1;
    bv_val_a_r1 = bv_val_a_r0;
    bv_val_a_r0 = bv_val_a_c;
    array_idx_a_r3 = array_idx_a_r2;
    array_idx_a_r2 = array_idx_a_r1;
    array_idx_a_r1 = array_idx_a_r0;
    array_idx_a_r0 = array_idx_a_c;
}

/// @brief read in hash_strm and update the bit_vector
template <int HASH_OUT_W, int BV_W>
void bv_update_bram(hls::stream<ap_uint<HASH_OUT_W> >& hash_strm,
                    hls::stream<bool>& in_e_strm,
                    ap_uint<16>* bit_vector_ptr0,
                    ap_uint<16>* bit_vector_ptr1,
                    ap_uint<16>* bit_vector_ptr2) {
    //#pragma HLS inline
    //#pragma HLS PIPELINE II=1
    ap_uint<BV_W - 3> array_idx_l_r0 = 0xffffffff;
    ap_uint<BV_W - 3> array_idx_l_r1 = 0xffffffff; // should not be accessed in first three cycles
    ap_uint<BV_W - 3> array_idx_l_r2 = 0xffffffff;
    ap_uint<BV_W - 3> array_idx_l_r3 = 0xffffffff;
    ap_uint<BV_W - 3> array_idx_h_r0 = 0xffffffff;
    ap_uint<BV_W - 3> array_idx_h_r1 = 0xffffffff;
    ap_uint<BV_W - 3> array_idx_h_r2 = 0xffffffff;
    ap_uint<BV_W - 3> array_idx_h_r3 = 0xffffffff;
    ap_uint<BV_W - 3> array_idx_a_r0 = 0xffffffff;
    ap_uint<BV_W - 3> array_idx_a_r1 = 0xffffffff;
    ap_uint<BV_W - 3> array_idx_a_r2 = 0xffffffff;
    ap_uint<BV_W - 3> array_idx_a_r3 = 0xffffffff;
    ap_uint<16> bv_val_l_c, bv_val_l_r0, bv_val_l_r1, bv_val_l_r2, bv_val_l_r3; // 16 is the width of bram used.
    ap_uint<16> bv_val_h_c, bv_val_h_r0, bv_val_h_r1, bv_val_h_r2, bv_val_h_r3;
    ap_uint<16> bv_val_a_c, bv_val_a_r0, bv_val_a_r1, bv_val_a_r2, bv_val_a_r3;

    bool e = in_e_strm.read();

BV_UPDATE:
    //	for(int i=0; i< n_of_input; i++){
    while (!e) {
#pragma HLS DEPENDENCE array inter false
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 64 max = 65536
        // 0) divide the hash strm into two parts: high16bits and low16bits
        ap_uint<HASH_OUT_W> hash_res = hash_strm.read();
        ap_uint<HASH_OUT_W / 2> lowpart = hash_res(HASH_OUT_W / 2 - 1, 0);
        ap_uint<HASH_OUT_W / 2> highpart = hash_res(HASH_OUT_W - 1, HASH_OUT_W / 2);
        // obtain the final positions of three bits in bit-vector
        lowpart %= (1 << BV_W);
        highpart %= (1 << BV_W);
        ap_uint<HASH_OUT_W / 2> avgpart = (highpart + lowpart) % (1 << BV_W);
        //		std::cout <<"kernel lowpart, highpart, and avgpart is " << lowpart
        //<<", " << highpart <<", and " <<avgpart<<std::endl;
        ap_uint<HASH_OUT_W / 2 - 3> lha_max = HASH_OUT_W / 2 > BV_W ? BV_W : HASH_OUT_W / 2;
        // store bv in 16 bits bram
        // 1) calculate the array-idx and bit-idx
        ap_uint<4> bit_idx_l_c = lowpart(3, 0);
        ap_uint<BV_W - 3> array_idx_l_c = lowpart(lha_max - 1, 4);

        ap_uint<4> bit_idx_h_c = highpart(3, 0);
        ap_uint<BV_W - 3> array_idx_h_c = highpart(lha_max - 1, 4);

        ap_uint<4> bit_idx_a_c = avgpart(3, 0);
        ap_uint<BV_W - 3> array_idx_a_c = avgpart(lha_max - 1, 4);

        // 2) read Bram and select the state_c. low16
        if (array_idx_l_c == array_idx_l_r0) // no read from bram
            bv_val_l_c = bv_val_l_r0;
        else if (array_idx_l_c == array_idx_l_r1)
            bv_val_l_c = bv_val_l_r1;
        else if (array_idx_l_c == array_idx_l_r2)
            bv_val_l_c = bv_val_l_r2;
        else if (array_idx_l_c == array_idx_l_r3)
            bv_val_l_c = bv_val_l_r3;
        else
            bv_val_l_c = bit_vector_ptr0[array_idx_l_c];
        // high16
        if (array_idx_h_c == array_idx_h_r0) // no read from bram
            bv_val_h_c = bv_val_h_r0;
        else if (array_idx_h_c == array_idx_h_r1)
            bv_val_h_c = bv_val_h_r1;
        else if (array_idx_h_c == array_idx_h_r2)
            bv_val_h_c = bv_val_h_r2;
        else if (array_idx_h_c == array_idx_h_r3)
            bv_val_h_c = bv_val_h_r3;
        else
            bv_val_h_c = bit_vector_ptr1[array_idx_h_c];
        // avg16
        if (array_idx_a_c == array_idx_a_r0) // no read from bram
            bv_val_a_c = bv_val_a_r0;
        else if (array_idx_a_c == array_idx_a_r1)
            bv_val_a_c = bv_val_a_r1;
        else if (array_idx_a_c == array_idx_a_r2)
            bv_val_a_c = bv_val_a_r2;
        else if (array_idx_a_c == array_idx_a_r3)
            bv_val_a_c = bv_val_a_r3;
        else
            bv_val_a_c = bit_vector_ptr2[array_idx_a_c];

        // 3) update bv vector value
        bv_val_l_c[bit_idx_l_c] = 1;
        bv_val_h_c[bit_idx_h_c] = 1;
        bv_val_a_c[bit_idx_a_c] = 1;

        // 4) write back to bram
        bit_vector_ptr0[array_idx_l_c] = bv_val_l_c;
        bit_vector_ptr1[array_idx_h_c] = bv_val_h_c;
        bit_vector_ptr2[array_idx_a_c] = bv_val_a_c;

        // 5) shift the whole data line 1 cycle for bv_val and addr
        // low16
        bv_val_l_r3 = bv_val_l_r2;
        bv_val_l_r2 = bv_val_l_r1;
        bv_val_l_r1 = bv_val_l_r0;
        bv_val_l_r0 = bv_val_l_c;
        array_idx_l_r3 = array_idx_l_r2;
        array_idx_l_r2 = array_idx_l_r1;
        array_idx_l_r1 = array_idx_l_r0;
        array_idx_l_r0 = array_idx_l_c;
        // high16
        bv_val_h_r3 = bv_val_h_r2;
        bv_val_h_r2 = bv_val_h_r1;
        bv_val_h_r1 = bv_val_h_r0;
        bv_val_h_r0 = bv_val_h_c;
        array_idx_h_r3 = array_idx_h_r2;
        array_idx_h_r2 = array_idx_h_r1;
        array_idx_h_r1 = array_idx_h_r0;
        array_idx_h_r0 = array_idx_h_c;
        // avg16
        bv_val_a_r3 = bv_val_a_r2;
        bv_val_a_r2 = bv_val_a_r1;
        bv_val_a_r1 = bv_val_a_r0;
        bv_val_a_r0 = bv_val_a_c;
        array_idx_a_r3 = array_idx_a_r2;
        array_idx_a_r2 = array_idx_a_r1;
        array_idx_a_r1 = array_idx_a_r0;
        array_idx_a_r0 = array_idx_a_c;

        e = in_e_strm.read();
    }
}

template <int HASH_OUT_W, int BV_W>
inline void bv_check_one(hls::stream<ap_uint<HASH_OUT_W> >& chk_hash_strm,
                         ap_uint<16>* bit_vector_ptr0,
                         ap_uint<16>* bit_vector_ptr1,
                         ap_uint<16>* bit_vector_ptr2,
                         hls::stream<bool>& out_v_strm) {
#pragma HLS pipeline II = 1
#pragma HLS inline
    ap_uint<HASH_OUT_W> chk_hash_data;
    ap_uint<HASH_OUT_W / 2> lowpart, highpart, avgpart;
    //		std::cout <<"e is " << e <<std::endl;
    chk_hash_data = chk_hash_strm.read();
    lowpart = chk_hash_data(HASH_OUT_W / 2 - 1, 0);
    highpart = chk_hash_data(HASH_OUT_W - 1, HASH_OUT_W / 2);
    lowpart %= (1 << BV_W);
    highpart %= (1 << BV_W);
    avgpart = (highpart + lowpart) % (1 << BV_W);
    ap_uint<HASH_OUT_W / 2 - 4> lha_max = HASH_OUT_W / 2 > BV_W ? BV_W : HASH_OUT_W / 2;
    //		std::cout <<"check kernel lowpart, highpart, and avgpart is " <<
    // lowpart <<", " << highpart <<", and " <<avgpart<<std::endl;
    ap_uint<4> bit_idx_l = lowpart(3, 0);
    ap_uint<BV_W - 4> array_idx_l = lowpart(lha_max - 1, 4);

    ap_uint<4> bit_idx_h = highpart(3, 0);
    ap_uint<BV_W - 4> array_idx_h = highpart(lha_max - 1, 4);

    ap_uint<4> bit_idx_a = avgpart(3, 0);
    ap_uint<BV_W - 4> array_idx_a = avgpart(lha_max - 1, 4);
    ap_uint<16> bit_res0 = bit_vector_ptr0[array_idx_l];
    ap_uint<16> bit_res1 = bit_vector_ptr1[array_idx_h];
    ap_uint<16> bit_res2 = bit_vector_ptr2[array_idx_a];

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "r_idx_l=" << array_idx_l << " ,r_idx_h=" << array_idx_h << " ,r_idx_a=" << array_idx_a << std::endl;
    std::cout << "res0=" << bit_res0 << " ,res1=" << bit_res1 << " ,res2=" << bit_res2 << std::endl;
#endif
#endif
    if (bit_res0[bit_idx_l] == 1 && bit_res1[bit_idx_h] == 1 && bit_res2[bit_idx_a] == 1) {
        out_v_strm.write(1);
    } else {
        out_v_strm.write(0);
    }
}

template <int HASH_OUT_W, int BV_W>
void bv_check_bram(hls::stream<ap_uint<HASH_OUT_W> >& chk_hash_strm,
                   hls::stream<bool>& chk_hash_e_strm,
                   ap_uint<16>* bit_vector_ptr0,
                   ap_uint<16>* bit_vector_ptr1,
                   ap_uint<16>* bit_vector_ptr2,
                   hls::stream<bool>& out_v_strm,
                   hls::stream<bool>& out_e_strm) {
    bool e = chk_hash_e_strm.read();
    ap_uint<HASH_OUT_W> chk_hash_data;
    ap_uint<HASH_OUT_W / 2> lowpart, highpart, avgpart;
BV_LOOP:
    while (!e) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 65536 max = 65536
        //		std::cout <<"e is " << e <<std::endl;
        chk_hash_data = chk_hash_strm.read();
        lowpart = chk_hash_data(HASH_OUT_W / 2 - 1, 0);
        highpart = chk_hash_data(HASH_OUT_W - 1, HASH_OUT_W / 2);
        lowpart %= (1 << BV_W);
        highpart %= (1 << BV_W);
        avgpart = (highpart + lowpart) % (1 << BV_W);
        ap_uint<HASH_OUT_W / 2 - 4> lha_max = HASH_OUT_W / 2 > BV_W ? BV_W : HASH_OUT_W / 2;
        //		std::cout <<"check kernel lowpart, highpart, and avgpart is " <<
        // lowpart <<", " << highpart <<", and " <<avgpart<<std::endl;
        ap_uint<4> bit_idx_l = lowpart(3, 0);
        ap_uint<BV_W - 4> array_idx_l = lowpart(lha_max - 1, 4);

        ap_uint<4> bit_idx_h = highpart(3, 0);
        ap_uint<BV_W - 4> array_idx_h = highpart(lha_max - 1, 4);

        ap_uint<4> bit_idx_a = avgpart(3, 0);
        ap_uint<BV_W - 4> array_idx_a = avgpart(lha_max - 1, 4);

        ap_uint<16> bit_res0 = bit_vector_ptr0[array_idx_l];
        ap_uint<16> bit_res1 = bit_vector_ptr1[array_idx_h];
        ap_uint<16> bit_res2 = bit_vector_ptr2[array_idx_a];
        if (bit_res0[bit_idx_l] == 1 && bit_res1[bit_idx_h] == 1 && bit_res2[bit_idx_a] == 1) {
            //		std::cout <<"array_idx_l[]bit_res0] is " <<
            // array_idx_l<<"["<<bit_res0<<"]"<<std::endl;
            //		std::cout <<"bit_vector_ptr0[array_idx_l] is "
            //<<bit_vector_ptr0[array_idx_l]<<std::endl;
            //		ap_uint<4> bit_res1 = bit_vector_ptr1[array_idx_h];
            //		ap_uint<4> bit_res2 = bit_vector_ptr2[array_idx_a];
            //		if(bit_res0 == bit_idx_l && bit_res1 == bit_idx_h &&
            // bit_res2 == bit_idx_a)
            out_v_strm.write(1);
        } else {
            out_v_strm.write(0);
        }
        out_e_strm.write(0);
        e = chk_hash_e_strm.read();
    }
    out_e_strm.write(1);
}

/// @brief read hash_strm and update the bit_vector
template <int BV_W>
void bv_update_uram(hls::stream<ap_uint<BV_W> >& hash_strm,
                    hls::stream<bool>& in_e_strm,

                    ap_uint<72>* bit_vector_ptr) {
    //#pragma HLS inline

    ap_uint<BV_W - 6> array_idx_l_r0 = 0xffffffff; // should not be accessed in first three cycles
    ap_uint<BV_W - 6> array_idx_l_r1 = 0xffffffff;
    ap_uint<BV_W - 6> array_idx_l_r2 = 0xffffffff;
    ap_uint<BV_W - 6> array_idx_l_r3 = 0xffffffff;

    ap_uint<72> bv_val_l_c, bv_val_l_r0, bv_val_l_r1, bv_val_l_r2, bv_val_l_r3; // 72 is the width of Uram used.

    bool e = in_e_strm.read();

BV_UPDATE:
    while (!e) {
#pragma HLS DEPENDENCE array inter false
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 64 max = 65536
        // 0) read the hash strm
        ap_uint<BV_W> hash = hash_strm.read();

        // store bv in 72 bits Uram
        // 1) calculate the array-idx and bit-idx
        ap_uint<6> bit_idx_l_c = hash(5, 0);
        ap_uint<BV_W - 6> array_idx_l_c = hash(BV_W - 1, 6);

        // 2) read Uram and select the state_c.
        if (array_idx_l_c == array_idx_l_r0) // no read from Uram
            bv_val_l_c = bv_val_l_r0;
        else if (array_idx_l_c == array_idx_l_r1)
            bv_val_l_c = bv_val_l_r1;
        else if (array_idx_l_c == array_idx_l_r2)
            bv_val_l_c = bv_val_l_r2;
        else if (array_idx_l_c == array_idx_l_r3)
            bv_val_l_c = bv_val_l_r3;
        else
            bv_val_l_c = bit_vector_ptr[array_idx_l_c];

        // 3) update bv vector value
        bv_val_l_c[bit_idx_l_c] = 1;

        // 4) write back to Uram
        bit_vector_ptr[array_idx_l_c] = bv_val_l_c;

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << std::hex << "bloom filter: hash0=" << hash << " bf_addr0=" << array_idx_l_c
                  << " bf_in0=" << bit_idx_l_c << " bf_new0=" << bv_val_l_c << std::endl;
#endif
#endif

        // 5) shift the whole data line 1 cycle for bv_val and addr
        bv_val_l_r3 = bv_val_l_r2;
        bv_val_l_r2 = bv_val_l_r1;
        bv_val_l_r1 = bv_val_l_r0;
        bv_val_l_r0 = bv_val_l_c;
        array_idx_l_r3 = array_idx_l_r2;
        array_idx_l_r2 = array_idx_l_r1;
        array_idx_l_r1 = array_idx_l_r0;
        array_idx_l_r0 = array_idx_l_c;

        e = in_e_strm.read();
    }
}

///@brief check one hash strm
template <int BV_W>
void bv_check_uram(hls::stream<ap_uint<BV_W> >& chk_hash_strm,
                   hls::stream<bool>& chk_hash_e_strm,

                   ap_uint<72>* bit_vector_ptr,

                   hls::stream<bool>& out_v_strm,
                   hls::stream<bool>& out_e_strm) {
    bool e = chk_hash_e_strm.read();
    ap_uint<BV_W> chk_hash;

BV_LOOP:
    while (!e) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 65536 max = 65536

        chk_hash = chk_hash_strm.read();
        e = chk_hash_e_strm.read();

        ap_uint<6> bit_idx_l = chk_hash(5, 0);
        ap_uint<BV_W - 6> array_idx_l = chk_hash(BV_W - 1, 6);
        ap_uint<72> bit_res = bit_vector_ptr[array_idx_l];

        if (bit_res[bit_idx_l] == 1) {
            out_v_strm.write(true);
        } else {
            out_v_strm.write(false);
        }

        out_e_strm.write(false);
    }
    out_e_strm.write(true);
}

/// @brief read two hash_strm and update the bit_vector
template <int BV_W>
void bv_update_uram(hls::stream<ap_uint<BV_W> >& hash_strm0,
                    hls::stream<ap_uint<BV_W> >& hash_strm1,
                    hls::stream<bool>& in_e_strm,

                    ap_uint<72>* bit_vector_ptr0,
                    ap_uint<72>* bit_vector_ptr1) {
    //#pragma HLS inline

    ap_uint<BV_W - 6> array_idx_l_r0 = 0xffffffff; // should not be accessed in first three cycles
    ap_uint<BV_W - 6> array_idx_l_r1 = 0xffffffff;
    ap_uint<BV_W - 6> array_idx_l_r2 = 0xffffffff;
    ap_uint<BV_W - 6> array_idx_l_r3 = 0xffffffff;
    ap_uint<BV_W - 6> array_idx_h_r0 = 0xffffffff;
    ap_uint<BV_W - 6> array_idx_h_r1 = 0xffffffff;
    ap_uint<BV_W - 6> array_idx_h_r2 = 0xffffffff;
    ap_uint<BV_W - 6> array_idx_h_r3 = 0xffffffff;

    ap_uint<72> bv_val_l_c, bv_val_l_r0, bv_val_l_r1, bv_val_l_r2, bv_val_l_r3; // 72 is the width of Uram used.
    ap_uint<72> bv_val_h_c, bv_val_h_r0, bv_val_h_r1, bv_val_h_r2, bv_val_h_r3;

    bool e = in_e_strm.read();

BV_UPDATE:
    //	for(int i=0; i< n_of_input; i++){
    while (!e) {
#pragma HLS DEPENDENCE array inter false
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 64 max = 65536
        // 0) read the hash strm
        ap_uint<BV_W> hash0 = hash_strm0.read();
        ap_uint<BV_W> hash1 = hash_strm1.read();

        // store bv in 72 bits Uram
        // 1) calculate the array-idx and bit-idx
        ap_uint<6> bit_idx_l_c = hash0(5, 0);
        ap_uint<BV_W - 6> array_idx_l_c = hash0(BV_W - 1, 6);

        ap_uint<6> bit_idx_h_c = hash1(5, 0);
        ap_uint<BV_W - 6> array_idx_h_c = hash1(BV_W - 1, 6);

        // 2) read Uram and select the state_c.
        // hash0
        if (array_idx_l_c == array_idx_l_r0) // no read from Uram
            bv_val_l_c = bv_val_l_r0;
        else if (array_idx_l_c == array_idx_l_r1)
            bv_val_l_c = bv_val_l_r1;
        else if (array_idx_l_c == array_idx_l_r2)
            bv_val_l_c = bv_val_l_r2;
        else if (array_idx_l_c == array_idx_l_r3)
            bv_val_l_c = bv_val_l_r3;
        else
            bv_val_l_c = bit_vector_ptr0[array_idx_l_c];
        // hash1
        if (array_idx_h_c == array_idx_h_r0) // no read from Uram
            bv_val_h_c = bv_val_h_r0;
        else if (array_idx_h_c == array_idx_h_r1)
            bv_val_h_c = bv_val_h_r1;
        else if (array_idx_h_c == array_idx_h_r2)
            bv_val_h_c = bv_val_h_r2;
        else if (array_idx_h_c == array_idx_h_r3)
            bv_val_h_c = bv_val_h_r3;
        else
            bv_val_h_c = bit_vector_ptr1[array_idx_h_c];

        // 3) update bv vector value
        bv_val_l_c[bit_idx_l_c] = 1;
        bv_val_h_c[bit_idx_h_c] = 1;

        // 4) write back to Uram
        bit_vector_ptr0[array_idx_l_c] = bv_val_l_c;
        bit_vector_ptr1[array_idx_h_c] = bv_val_h_c;

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << std::hex << "bloom filter: hash0=" << hash0 << " bf_addr0=" << array_idx_l_c
                  << " bf_in0=" << bit_idx_l_c << " bf_new0=" << bv_val_l_c << " bf_addr1=" << array_idx_h_c
                  << " hash1=" << hash1 << " bf_in1=" << bit_idx_h_c << " bf_new1=" << bv_val_h_c << std::endl;
#endif
#endif

        // 5) shift the whole data line 1 cycle for bv_val and addr
        // hash0
        bv_val_l_r3 = bv_val_l_r2;
        bv_val_l_r2 = bv_val_l_r1;
        bv_val_l_r1 = bv_val_l_r0;
        bv_val_l_r0 = bv_val_l_c;
        array_idx_l_r3 = array_idx_l_r2;
        array_idx_l_r2 = array_idx_l_r1;
        array_idx_l_r1 = array_idx_l_r0;
        array_idx_l_r0 = array_idx_l_c;
        // hash1
        bv_val_h_r3 = bv_val_h_r2;
        bv_val_h_r2 = bv_val_h_r1;
        bv_val_h_r1 = bv_val_h_r0;
        bv_val_h_r0 = bv_val_h_c;
        array_idx_h_r3 = array_idx_h_r2;
        array_idx_h_r2 = array_idx_h_r1;
        array_idx_h_r1 = array_idx_h_r0;
        array_idx_h_r0 = array_idx_h_c;

        e = in_e_strm.read();
    }
}

///@brief check two hash stream
template <int BV_W>
void bv_check_uram(hls::stream<ap_uint<BV_W> >& chk_hash_strm0,
                   hls::stream<ap_uint<BV_W> >& chk_hash_strm1,
                   hls::stream<bool>& chk_hash_e_strm,

                   ap_uint<72>* bit_vector_ptr0,
                   ap_uint<72>* bit_vector_ptr1,

                   hls::stream<bool>& out_v_strm,
                   hls::stream<bool>& out_e_strm) {
    bool e = chk_hash_e_strm.read();
    ap_uint<BV_W> chk_hash0;
    ap_uint<BV_W> chk_hash1;

BV_LOOP:
    while (!e) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 65536 max = 65536

        chk_hash0 = chk_hash_strm0.read();
        chk_hash1 = chk_hash_strm1.read();
        e = chk_hash_e_strm.read();

        ap_uint<6> bit_idx_l = chk_hash0(5, 0);
        ap_uint<BV_W - 6> array_idx_l = chk_hash0(BV_W - 1, 6);

        ap_uint<6> bit_idx_h = chk_hash1(5, 0);
        ap_uint<BV_W - 6> array_idx_h = chk_hash1(BV_W - 1, 6);

        ap_uint<72> bit_res0 = bit_vector_ptr0[array_idx_l];
        ap_uint<72> bit_res1 = bit_vector_ptr1[array_idx_h];

        if (bit_res0[bit_idx_l] == 1 && bit_res1[bit_idx_h] == 1) {
            out_v_strm.write(true);
        } else {
            out_v_strm.write(false);
        }

        out_e_strm.write(false);
    }
    out_e_strm.write(true);
}

///@brief generate 3 hash strm
template <int STR_IN_W, int HASH_OUT_W>
void hashloop(hls::stream<ap_uint<STR_IN_W> >& msg_strm,
              hls::stream<bool>& in_e_strm,

              hls::stream<ap_uint<HASH_OUT_W> >& hash_strm0,
              hls::stream<ap_uint<HASH_OUT_W> >& hash_strm1,
              hls::stream<ap_uint<HASH_OUT_W> >& hash_strm2,
              hls::stream<bool>& out_e_strm) {
    hls::stream<ap_uint<STR_IN_W> > msg;
#pragma HLS STREAM variable = msg depth = 8
#pragma HLS bind_storage variable = msg type = fifo impl = srl

    hls::stream<ap_uint<64> > hash;
#pragma HLS STREAM variable = hash depth = 8
#pragma HLS bind_storage variable = hash type = fifo impl = srl

    const int max_width = STR_IN_W > HASH_OUT_W ? HASH_OUT_W : STR_IN_W;

    bool e = in_e_strm.read();
    while (!e) {
#pragma HLS LOOP_TRIPCOUNT max = 65536
#pragma HLS PIPELINE II = 1

        ap_uint<STR_IN_W> temp = msg_strm.read();
        e = in_e_strm.read();

        // generate hash
        msg.write(temp);
        database::hashLookup3<STR_IN_W>(msg, hash);

        ap_uint<64> hash_temp;
        hash_temp = hash.read();
        hash_strm0.write(hash_temp(HASH_OUT_W - 1, 0));
        hash_strm1.write(hash_temp(2 * HASH_OUT_W - 1, HASH_OUT_W));
        hash_strm2.write(temp(max_width - 1, 0));

        out_e_strm.write(false);
    }
    out_e_strm.write(true);
}

/// @brief read three hash_strm and update the bit_vector
template <int BV_W>
void bv_update_uram(hls::stream<ap_uint<BV_W> >& hash_strm0,
                    hls::stream<ap_uint<BV_W> >& hash_strm1,
                    hls::stream<ap_uint<BV_W> >& hash_strm2,
                    hls::stream<bool>& in_e_strm,

                    ap_uint<72>* bit_vector_ptr0,
                    ap_uint<72>* bit_vector_ptr1,
                    ap_uint<72>* bit_vector_ptr2) {
    //#pragma HLS inline

    ap_uint<BV_W - 6> array_idx_l_r0 = 0xffffffff;
    ap_uint<BV_W - 6> array_idx_l_r1 = 0xffffffff; // should not be accessed in first three cycles
    ap_uint<BV_W - 6> array_idx_l_r2 = 0xffffffff;
    ap_uint<BV_W - 6> array_idx_l_r3 = 0xffffffff;
    ap_uint<BV_W - 6> array_idx_h_r0 = 0xffffffff;
    ap_uint<BV_W - 6> array_idx_h_r1 = 0xffffffff;
    ap_uint<BV_W - 6> array_idx_h_r2 = 0xffffffff;
    ap_uint<BV_W - 6> array_idx_h_r3 = 0xffffffff;
    ap_uint<BV_W - 6> array_idx_a_r0 = 0xffffffff;
    ap_uint<BV_W - 6> array_idx_a_r1 = 0xffffffff;
    ap_uint<BV_W - 6> array_idx_a_r2 = 0xffffffff;
    ap_uint<BV_W - 6> array_idx_a_r3 = 0xffffffff;
    ap_uint<72> bv_val_l_c, bv_val_l_r0, bv_val_l_r1, bv_val_l_r2, bv_val_l_r3; // 72 is the width of Uram used.
    ap_uint<72> bv_val_h_c, bv_val_h_r0, bv_val_h_r1, bv_val_h_r2, bv_val_h_r3;
    ap_uint<72> bv_val_a_c, bv_val_a_r0, bv_val_a_r1, bv_val_a_r2, bv_val_a_r3;

    bool e = in_e_strm.read();

BV_UPDATE:
    //	for(int i=0; i< n_of_input; i++){
    while (!e) {
#pragma HLS DEPENDENCE array inter false
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 64 max = 65536
        // 0) read the hash strm
        ap_uint<BV_W> hash0 = hash_strm0.read();
        ap_uint<BV_W> hash1 = hash_strm1.read();
        ap_uint<BV_W> hash2 = hash_strm2.read();

        // store bv in 72 bits Uram
        // 1) calculate the array-idx and bit-idx
        ap_uint<6> bit_idx_l_c = hash0(5, 0);
        ap_uint<BV_W - 6> array_idx_l_c = hash0(BV_W - 1, 6);

        ap_uint<6> bit_idx_h_c = hash1(5, 0);
        ap_uint<BV_W - 6> array_idx_h_c = hash1(BV_W - 1, 6);

        ap_uint<6> bit_idx_a_c = hash2(5, 0);
        ap_uint<BV_W - 6> array_idx_a_c = hash2(BV_W - 1, 6);

        // 2) read Uram and select the state_c.
        // hash0
        if (array_idx_l_c == array_idx_l_r0) // no read from Uram
            bv_val_l_c = bv_val_l_r0;
        else if (array_idx_l_c == array_idx_l_r1)
            bv_val_l_c = bv_val_l_r1;
        else if (array_idx_l_c == array_idx_l_r2)
            bv_val_l_c = bv_val_l_r2;
        else if (array_idx_l_c == array_idx_l_r3)
            bv_val_l_c = bv_val_l_r3;
        else
            bv_val_l_c = bit_vector_ptr0[array_idx_l_c];
        // hash1
        if (array_idx_h_c == array_idx_h_r0) // no read from Uram
            bv_val_h_c = bv_val_h_r0;
        else if (array_idx_h_c == array_idx_h_r1)
            bv_val_h_c = bv_val_h_r1;
        else if (array_idx_h_c == array_idx_h_r2)
            bv_val_h_c = bv_val_h_r2;
        else if (array_idx_h_c == array_idx_h_r3)
            bv_val_h_c = bv_val_h_r3;
        else
            bv_val_h_c = bit_vector_ptr1[array_idx_h_c];
        // hash2
        if (array_idx_a_c == array_idx_a_r0) // no read from Uram
            bv_val_a_c = bv_val_a_r0;
        else if (array_idx_a_c == array_idx_a_r1)
            bv_val_a_c = bv_val_a_r1;
        else if (array_idx_a_c == array_idx_a_r2)
            bv_val_a_c = bv_val_a_r2;
        else if (array_idx_a_c == array_idx_a_r3)
            bv_val_a_c = bv_val_a_r3;
        else
            bv_val_a_c = bit_vector_ptr2[array_idx_a_c];

        // 3) update bv vector value
        bv_val_l_c[bit_idx_l_c] = 1;
        bv_val_h_c[bit_idx_h_c] = 1;
        bv_val_a_c[bit_idx_a_c] = 1;

        // 4) write back to Uram
        bit_vector_ptr0[array_idx_l_c] = bv_val_l_c;
        bit_vector_ptr1[array_idx_h_c] = bv_val_h_c;
        bit_vector_ptr2[array_idx_a_c] = bv_val_a_c;

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << std::hex << "bloom filter update: hash0=" << hash0 << " bf_addr0=" << array_idx_l_c
                  << " bit_idx0=" << bit_idx_l_c << " bf_in0=" << (1 << bit_idx_l_c) << " bf_new0=" << bv_val_l_c
                  << " hash1=" << hash1 << " bf_addr1=" << array_idx_h_c << " bit_idx1=" << bit_idx_h_c
                  << " bf_in1=" << (1 << bit_idx_h_c) << " bf_new1=" << bv_val_h_c << " hash2=" << hash2
                  << " bf_addr2=" << array_idx_a_c << " bit_idx2=" << bit_idx_a_c << " bf_in2=" << (1 << bit_idx_a_c)
                  << " bf_new2=" << bv_val_a_c << std::endl;
#endif
#endif

        // 5) shift the whole data line 1 cycle for bv_val and addr
        // hash0
        bv_val_l_r3 = bv_val_l_r2;
        bv_val_l_r2 = bv_val_l_r1;
        bv_val_l_r1 = bv_val_l_r0;
        bv_val_l_r0 = bv_val_l_c;
        array_idx_l_r3 = array_idx_l_r2;
        array_idx_l_r2 = array_idx_l_r1;
        array_idx_l_r1 = array_idx_l_r0;
        array_idx_l_r0 = array_idx_l_c;
        // hash1
        bv_val_h_r3 = bv_val_h_r2;
        bv_val_h_r2 = bv_val_h_r1;
        bv_val_h_r1 = bv_val_h_r0;
        bv_val_h_r0 = bv_val_h_c;
        array_idx_h_r3 = array_idx_h_r2;
        array_idx_h_r2 = array_idx_h_r1;
        array_idx_h_r1 = array_idx_h_r0;
        array_idx_h_r0 = array_idx_h_c;
        // hash2
        bv_val_a_r3 = bv_val_a_r2;
        bv_val_a_r2 = bv_val_a_r1;
        bv_val_a_r1 = bv_val_a_r0;
        bv_val_a_r0 = bv_val_a_c;
        array_idx_a_r3 = array_idx_a_r2;
        array_idx_a_r2 = array_idx_a_r1;
        array_idx_a_r1 = array_idx_a_r0;
        array_idx_a_r0 = array_idx_a_c;

        e = in_e_strm.read();
    }
}

///@brief check three hash strm
template <int BV_W>
void bv_check_uram(hls::stream<ap_uint<BV_W> >& chk_hash_strm0,
                   hls::stream<ap_uint<BV_W> >& chk_hash_strm1,
                   hls::stream<ap_uint<BV_W> >& chk_hash_strm2,
                   hls::stream<bool>& chk_hash_e_strm,

                   ap_uint<72>* bit_vector_ptr0,
                   ap_uint<72>* bit_vector_ptr1,
                   ap_uint<72>* bit_vector_ptr2,

                   hls::stream<bool>& out_v_strm,
                   hls::stream<bool>& out_e_strm) {
    bool e = chk_hash_e_strm.read();
    ap_uint<BV_W> chk_hash0;
    ap_uint<BV_W> chk_hash1;
    ap_uint<BV_W> chk_hash2;

BV_LOOP:
    while (!e) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 65536 max = 65536

        chk_hash0 = chk_hash_strm0.read();
        chk_hash1 = chk_hash_strm1.read();
        chk_hash2 = chk_hash_strm2.read();
        e = chk_hash_e_strm.read();

        ap_uint<6> bit_idx_l = chk_hash0(5, 0);
        ap_uint<BV_W - 6> array_idx_l = chk_hash0(BV_W - 1, 6);

        ap_uint<6> bit_idx_h = chk_hash1(5, 0);
        ap_uint<BV_W - 6> array_idx_h = chk_hash1(BV_W - 1, 6);

        ap_uint<6> bit_idx_a = chk_hash2(5, 0);
        ap_uint<BV_W - 6> array_idx_a = chk_hash2(BV_W - 1, 6);

        ap_uint<72> bit_res0 = bit_vector_ptr0[array_idx_l];
        ap_uint<72> bit_res1 = bit_vector_ptr1[array_idx_h];
        ap_uint<72> bit_res2 = bit_vector_ptr2[array_idx_a];

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << std::hex << "bloom filter check: hash0=" << chk_hash0 << " bf_addr0=" << array_idx_l
                  << " bf_val0=" << bit_res0 << " bf_check0=" << bit_idx_l << " bf_bool0=" << bit_res0[bit_idx_l]
                  << " hash1=" << chk_hash1 << " bf_addr1=" << array_idx_h << " bf_val1=" << bit_res1
                  << " bf_check1=" << bit_idx_h << " bf_bool1=" << bit_res1[bit_idx_h] << " hash2=" << chk_hash2
                  << " bf_addr2=" << array_idx_a << " bf_val2=" << bit_res2 << " bf_check2=" << bit_idx_a
                  << " bf_bool2=" << bit_res2[bit_idx_a] << std::endl;
#endif
#endif

        if (bit_res0[bit_idx_l] == 1 && bit_res1[bit_idx_h] == 1 && bit_res2[bit_idx_a] == 1) {
            out_v_strm.write(true);
        } else {
            out_v_strm.write(false);
        }

        out_e_strm.write(false);
    }
    out_e_strm.write(true);
}

/// @brief generate the bloomfilter with list of strings, the output is 3
/// pointers that point to Bram tables.
/// @tparam STR_IN_W W width of the streamed input message, e.g., W=512.
/// @tparam HASH_OUT_W H width of the hash function output stream, lookup3 32bit
/// 64bit hash can be used.
/// @tparam BV_W depth of the input list (items/rows of input data).
/// @param msg_strm input message stream.
/// @param in_e_strm the flag that indicate the end of input message stream.
/// @param bit_vector_ptr0 the pointer of bit_vector0.
/// @param bit_vector_ptr1 the pointer of bit_vector1.
/// @param bit_vector_ptr2 the pointer of bit_vector2.
template <int STR_IN_W, int HASH_OUT_W, int BV_W>
void bf_gen_bram(hls::stream<ap_uint<STR_IN_W> >& msg_strm,
                 hls::stream<bool>& in_e_strm,
                 ap_uint<16>* bit_vector_ptr0,
                 ap_uint<16>* bit_vector_ptr1,
                 ap_uint<16>* bit_vector_ptr2) {
#pragma HLS DATAFLOW
    hls::stream<ap_uint<HASH_OUT_W> > hash_strm("hash_strm");
#pragma HLS STREAM variable = hash_strm depth = 64
    hls::stream<bool> hash_e_strm("hash_e_strm");
#pragma HLS STREAM variable = hash_e_strm depth = 64
    //	#pragma HLS STREAM variable=hash_strm depth=128
    hashloop<STR_IN_W, HASH_OUT_W>(msg_strm, in_e_strm, hash_strm, hash_e_strm);
    bv_update_bram<HASH_OUT_W, BV_W>(hash_strm, hash_e_strm, bit_vector_ptr0, bit_vector_ptr1, bit_vector_ptr2);
}

/// @brief generate the bloomfilter with list of strings, the output is a
/// stream, which can be transferred to ddr.
/// @tparam STR_IN_W W width of the streamed input message, e.g., W=512.
/// @tparam HASH_OUT_W H width of the hash function output stream.
/// @tparam BV_W depth of the input list (items/rows of input data).
/// @param msg_strm input message stream.
/// @param in_e_strm the flag that indicate the end of input message stream.
/// @param bit_vet_strm the output stream of bit_vector.
/// @param out_e_strm the flag that indicate the end of output stream.
template <bool IS_BRAM, int STR_IN_W, int HASH_OUT_W, int BV_W>
void bf_gen_bram_and_stream(hls::stream<ap_uint<STR_IN_W> >& msg_strm,
                            hls::stream<bool>& in_e_strm,
                            hls::stream<ap_uint<IS_BRAM ? 16 : 64> >& bit_vet_strm,
                            hls::stream<bool>& out_e_strm) {
    // define the bit-vector used for bf, each hash function corresponds to one
    // bit_vector
    // store the bit-vector in BRAM and width of each data is 16-bit
    ap_uint<16> bit_vector0[(1 << (BV_W - 4))];
#pragma HLS bind_storage variable = bit_vector0 type = ram_s2p impl = bram
    ap_uint<16> bit_vector1[(1 << (BV_W - 4))];
#pragma HLS bind_storage variable = bit_vector1 type = ram_s2p impl = bram
    ap_uint<16> bit_vector2[(1 << (BV_W - 4))];
#pragma HLS bind_storage variable = bit_vector2 type = ram_s2p impl = bram

INIT_LOOP: // initialize bit_vector to zero
    for (int i = 0; i < (1 << (BV_W - 4)); i++) {
        //		#pragma HLS UNROLL factor=64
        bit_vector0[i] = 0;
        bit_vector1[i] = 0;
        bit_vector2[i] = 0;
    }

    ap_uint<16>* bit_vector_ptr0;
    bit_vector_ptr0 = bit_vector0;
    ap_uint<16>* bit_vector_ptr1;
    bit_vector_ptr1 = bit_vector1;
    ap_uint<16>* bit_vector_ptr2;
    bit_vector_ptr2 = bit_vector2;

    // update all values in bit_vector
    bf_gen_bram<STR_IN_W, HASH_OUT_W, BV_W>(msg_strm, in_e_strm, bit_vector_ptr0, bit_vector_ptr1, bit_vector_ptr2);

OUT_LOOP:
    for (int i = 0; i < (1 << (BV_W - 4)); i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<16> bv_result = (bit_vector0[i] | bit_vector1[i] | bit_vector2[i]);
        bit_vet_strm.write(bv_result);
        out_e_strm.write(0);
    }
    out_e_strm.write(1);
}

/// @brief generate the bloomfilter with list of strings, the output is 3
/// pointers that point to Bram tables.
/// @tparam STR_IN_W W width of the streamed input message, e.g., W=512.
/// @tparam HASH_OUT_W H width of the hash function output stream, lookup3 32bit
/// 64bit hash can be used.
/// @tparam BV_W depth of the input list (items/rows of input data).
/// @param msg_strm input message stream.
/// @param in_e_strm the flag that indicate the end of input message stream.
/// @param bit_vector_ptr0 the pointer of bit_vector0.
/// @param bit_vector_ptr1 the pointer of bit_vector1.
/// @param bit_vector_ptr2 the pointer of bit_vector2.
/// @param out_v_strm the output stream that indicate whether the msg_strm is
/// inside,[in 1,not-in 0].
/// @param out_e_strm the output end flag stream.
template <int STR_IN_W, int HASH_OUT_W, int BV_W>
void bf_check_bram(hls::stream<ap_uint<STR_IN_W> >& msg_strm,
                   hls::stream<bool>& in_e_strm,
                   ap_uint<16>* bit_vector_ptr0,
                   ap_uint<16>* bit_vector_ptr1,
                   ap_uint<16>* bit_vector_ptr2,
                   hls::stream<bool>& out_v_strm,
                   hls::stream<bool>& out_e_strm) {
#pragma HLS DATAFLOW
    hls::stream<ap_uint<HASH_OUT_W> > chk_hash_strm("chk_hash_strm");
#pragma HLS STREAM variable = chk_hash_strm depth = 64
    hls::stream<bool> chk_hash_e_strm("chk_hash_e_strm");
#pragma HLS STREAM variable = chk_hash_e_strm depth = 64
    hashloop<STR_IN_W, HASH_OUT_W>(msg_strm, in_e_strm, chk_hash_strm, chk_hash_e_strm);
    bv_check_bram<HASH_OUT_W, BV_W>(chk_hash_strm, chk_hash_e_strm, bit_vector_ptr0, bit_vector_ptr1, bit_vector_ptr2,
                                    out_v_strm, out_e_strm);
}

/// @brief generate the bloomfilter with list of strings, the output is 3
/// pointers that point to Uram tables.
/// @tparam STR_IN_W W width of the streamed input message, e.g., W=512.
/// @tparam BV_W depth of the input list (items/rows of input data).
/// @param msg_strm input message stream.
/// @param in_e_strm the flag that indicate the end of input message stream.
/// @param bit_vector_ptr0 the pointer of bit_vector0.
/// @param bit_vector_ptr1 the pointer of bit_vector1.
/// @param bit_vector_ptr2 the pointer of bit_vector2.
template <int STR_IN_W, int BV_W>
void bf_gen_uram(hls::stream<ap_uint<STR_IN_W> >& msg_strm,
                 hls::stream<bool>& in_e_strm,
                 ap_uint<72>* bit_vector_ptr0,
                 ap_uint<72>* bit_vector_ptr1,
                 ap_uint<72>* bit_vector_ptr2) {
#pragma HLS DATAFLOW

    hls::stream<ap_uint<BV_W> > hash_strm0("hash_strm0");
#pragma HLS STREAM variable = hash_strm0 depth = 64
    hls::stream<ap_uint<BV_W> > hash_strm1("hash_strm1");
#pragma HLS STREAM variable = hash_strm1 depth = 64
    hls::stream<ap_uint<BV_W> > hash_strm2("hash_strm2");
#pragma HLS STREAM variable = hash_strm2 depth = 64
    hls::stream<bool> hash_e_strm("hash_e_strm");
#pragma HLS STREAM variable = hash_e_strm depth = 64

    hashloop<STR_IN_W, BV_W>(msg_strm, in_e_strm, hash_strm0, hash_strm1, hash_strm2, hash_e_strm);

    bv_update_uram<BV_W>(hash_strm0, hash_strm1, hash_strm2, hash_e_strm, bit_vector_ptr0, bit_vector_ptr1,
                         bit_vector_ptr2);
}

/// @brief generate the bloomfilter with list of strings, the output is a
/// stream, which can be transferred to ddr.
/// @tparam STR_IN_W W width of the streamed input message, e.g., W=512.
/// @tparam BV_W depth of the input list (items/rows of input data).
/// @param msg_strm input message stream.
/// @param in_e_strm the flag that indicate the end of input message stream.
/// @param bit_vet_strm the output stream of bit_vector.
/// @param out_e_strm the flag that indicate the end of output stream.
template <bool IS_BRAM, int STR_IN_W, int BV_W>
void bf_gen_uram_and_stream(hls::stream<ap_uint<STR_IN_W> >& msg_strm,
                            hls::stream<bool>& in_e_strm,
                            hls::stream<ap_uint<IS_BRAM ? 16 : 64> >& bit_vet_strm,
                            hls::stream<bool>& out_e_strm) {
    // define the bit-vector used for bf, each hash function corresponds to one
    // bit_vector
    // store the bit-vector in URAM and width of each data is 72-bit
    ap_uint<72> bit_vector0[(1 << (BV_W - 6))];
#pragma HLS bind_storage variable = bit_vector0 type = ram_2p impl = uram
    ap_uint<72> bit_vector1[(1 << (BV_W - 6))];
#pragma HLS bind_storage variable = bit_vector1 type = ram_2p impl = uram
    ap_uint<72> bit_vector2[(1 << (BV_W - 6))];
#pragma HLS bind_storage variable = bit_vector2 type = ram_2p impl = uram

INIT_LOOP: // initialize bit_vector to zero
    for (int i = 0; i < (1 << (BV_W - 6)); i++) {
        bit_vector0[i] = 0;
        bit_vector1[i] = 0;
        bit_vector2[i] = 0;
    }

    ap_uint<72>* bit_vector_ptr0;
    bit_vector_ptr0 = bit_vector0;
    ap_uint<72>* bit_vector_ptr1;
    bit_vector_ptr1 = bit_vector1;
    ap_uint<72>* bit_vector_ptr2;
    bit_vector_ptr2 = bit_vector2;

    // update all values in bit_vector
    bf_gen_uram<STR_IN_W, BV_W>(msg_strm, in_e_strm, bit_vector_ptr0, bit_vector_ptr1, bit_vector_ptr2);

OUT_LOOP:
    for (int i = 0; i < (1 << (BV_W - 6)); i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<72> bv_result = (bit_vector0[i] | bit_vector1[i] | bit_vector2[i]);
        bit_vet_strm.write(bv_result(IS_BRAM ? 16 : 64, 0));
        out_e_strm.write(false);
    }
    out_e_strm.write(true);
}

/// @brief generate the bloomfilter with list of strings, the output is 3
/// pointers that point to Uram tables.
/// @tparam STR_IN_W W width of the streamed input message, e.g., W=512.
/// @tparam BV_W depth of the input list (items/rows of input data).
/// @param msg_strm input message stream.
/// @param in_e_strm the flag that indicate the end of input message stream.
/// @param bit_vector_ptr0 the pointer of bit_vector0.
/// @param bit_vector_ptr1 the pointer of bit_vector1.
/// @param bit_vector_ptr2 the pointer of bit_vector2.
/// @param out_v_strm the output stream that indicate whether the msg_strm is
/// inside,[in 1,not-in 0].
/// @param out_e_strm the output end flag stream.
template <int STR_IN_W, int BV_W>
void bf_check_uram(hls::stream<ap_uint<STR_IN_W> >& msg_strm,
                   hls::stream<bool>& in_e_strm,
                   ap_uint<72>* bit_vector_ptr0,
                   ap_uint<72>* bit_vector_ptr1,
                   ap_uint<72>* bit_vector_ptr2,
                   hls::stream<bool>& out_v_strm,
                   hls::stream<bool>& out_e_strm) {
#pragma HLS DATAFLOW

    hls::stream<ap_uint<BV_W> > hash_strm0("hash_strm0");
#pragma HLS STREAM variable = hash_strm0 depth = 64
    hls::stream<ap_uint<BV_W> > hash_strm1("hash_strm1");
#pragma HLS STREAM variable = hash_strm1 depth = 64
    hls::stream<ap_uint<BV_W> > hash_strm2("hash_strm2");
#pragma HLS STREAM variable = hash_strm2 depth = 64
    hls::stream<bool> hash_e_strm("hash_e_strm");
#pragma HLS STREAM variable = hash_e_strm depth = 64

    hashloop<STR_IN_W, BV_W>(msg_strm, in_e_strm, hash_strm0, hash_strm1, hash_strm2, hash_e_strm);

    bv_check_uram<BV_W>(hash_strm0, hash_strm1, hash_strm2, hash_e_strm, bit_vector_ptr0, bit_vector_ptr1,
                        bit_vector_ptr2, out_v_strm, out_e_strm);
}
} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {

/**
 * @brief Generate the bloomfilter in on-chip RAM blocks.
 *
 * This primitive calculates hash of input values, and marks corresponding bits in the on-chip RAM blocks.
 * RAM blocks can be configured to be 18-bit BRAM or 72-bit URAM.
 *
 * The bloom-filter bit vectors are passed as three pointers, and behind the scene, one hash value is calculated and
 * manipulated into three distint marker locatins in these vectors.
 *
 * To check for existance of a value with generated vector, use the ``bfCheck`` primitive.
 *
 * @tparam STR_IN_W W width of the streamed input message, e.g., W=512.
 * @tparam BV_W width of the hash value. ptr0, ptr1 and ptr2 should point at MEM_SPACE=2^BV_W (bit).
 *
 * @param msg_strm input message stream.
 * @param in_e_strm the flag that indicate the end of input message stream.
 * @param bit_vector_ptr0 the pointer of bit_vector0.
 * @param bit_vector_ptr1 the pointer of bit_vector1.
 * @param bit_vector_ptr2 the pointer of bit_vector2.
 */
template <bool IS_BRAM, int STR_IN_W, int BV_W>
void bfGen(hls::stream<ap_uint<STR_IN_W> >& msg_strm,
           hls::stream<bool>& in_e_strm,
           ap_uint<IS_BRAM ? 16 : 72>* bit_vector_ptr0,
           ap_uint<IS_BRAM ? 16 : 72>* bit_vector_ptr1,
           ap_uint<IS_BRAM ? 16 : 72>* bit_vector_ptr2) {
    if (IS_BRAM) {
        details::bf_gen_bram<STR_IN_W, 64, BV_W>(msg_strm, in_e_strm, (ap_uint<16>*)bit_vector_ptr0,
                                                 (ap_uint<16>*)bit_vector_ptr1, (ap_uint<16>*)bit_vector_ptr2);
    } else {
        details::bf_gen_uram<STR_IN_W, BV_W>(msg_strm, in_e_strm, (ap_uint<72>*)bit_vector_ptr0,
                                             (ap_uint<72>*)bit_vector_ptr1, (ap_uint<72>*)bit_vector_ptr2);
    }
}

/**
 * @brief Generate the bloomfilter in on-chip RAM blocks, and emit the vectors upon finish.
 *
 * This primitive calculates hash values of input, and marks corresponding bits in the on-chip RAM blocks.
 * RAM blocks can be configured to be 18-bit BRAM or 72-bit URAM.
 *
 * The bloom-filter bit vectors are built into internally allocated buffers, and streamed out after the filter has been
 * fully built.
 *
 * @tparam STR_IN_W W width of the streamed input message, e.g., W=512.
 * @tparam BV_W width of the hash value. bit_vet_strm should send out MEM_SPACE=2^BV_W (bit) data in total.
 *
 * @param msg_strm input message stream.
 * @param in_e_strm the flag that indicate the end of input message stream.
 * @param bit_vet_strm the output stream of bit_vector.
 * @param out_e_strm the flag that indicate the end of output stream.
 */
template <bool IS_BRAM, int STR_IN_W, int BV_W>
void bfGenStream(hls::stream<ap_uint<STR_IN_W> >& msg_strm,
                 hls::stream<bool>& in_e_strm,
                 hls::stream<ap_uint<IS_BRAM ? 16 : 64> >& bit_vet_strm,
                 hls::stream<bool>& out_e_strm) {
    if (IS_BRAM) {
        details::bf_gen_bram_and_stream<IS_BRAM, STR_IN_W, 64, BV_W>(msg_strm, in_e_strm, bit_vet_strm, out_e_strm);
    } else {
        details::bf_gen_uram_and_stream<IS_BRAM, STR_IN_W, BV_W>(msg_strm, in_e_strm, bit_vet_strm, out_e_strm);
    }
}

/**
 * @brief Check existance of value using bloom-filter vectors.
 *
 * This primitive is designed to work with the bloom-filter vectors generated by the ``bfGen`` primitive.
 * Basically, it detects the existance of value by hashing it and check for the corresponding vector bits.
 * When hit, it is likely to be in the set of generating values, otherwise, it cannot be element of the set.
 * RAM blocks can be configured to be 18-bit BRAM or 72-bit URAM, the setting must match ``bfGen``.
 *
 * @tparam IS_BRAM choose which types of memory to use. True for BRAM. False for URAM
 * @tparam STR_IN_W W width of the streamed input message, e.g., W=512.
 * @tparam BV_W width of the hash value. ptr0, ptr1 and ptr2 should point at MEM_SPACE=2^BV_W (bit).
 *
 * @param msg_strm input message stream.
 * @param in_e_strm the flag that indicate the end of input message stream.
 * @param bit_vector_ptr0 the pointer of bit_vector0.
 * @param bit_vector_ptr1 the pointer of bit_vector1.
 * @param bit_vector_ptr2 the pointer of bit_vector2.
 * @param out_v_strm the output stream that indicate whether the value may exist <1 for true, 0 for false>.
 * @param out_e_strm the output end flag stream.
 */
template <bool IS_BRAM, int STR_IN_W, int BV_W>
void bfCheck(hls::stream<ap_uint<STR_IN_W> >& msg_strm,
             hls::stream<bool>& in_e_strm,
             ap_uint<IS_BRAM ? 16 : 72>* bit_vector_ptr0,
             ap_uint<IS_BRAM ? 16 : 72>* bit_vector_ptr1,
             ap_uint<IS_BRAM ? 16 : 72>* bit_vector_ptr2,
             hls::stream<bool>& out_v_strm,
             hls::stream<bool>& out_e_strm) {
    if (IS_BRAM) {
        details::bf_check_bram<STR_IN_W, 64, BV_W>(msg_strm, in_e_strm, (ap_uint<16>*)bit_vector_ptr0,
                                                   (ap_uint<16>*)bit_vector_ptr1, (ap_uint<16>*)bit_vector_ptr2,
                                                   out_v_strm, out_e_strm);
    } else {
        details::bf_check_uram<STR_IN_W, BV_W>(msg_strm, in_e_strm, (ap_uint<72>*)bit_vector_ptr0,
                                               (ap_uint<72>*)bit_vector_ptr1, (ap_uint<72>*)bit_vector_ptr2, out_v_strm,
                                               out_e_strm);
    }
}

} // namespace database
} // namespace xf
#endif
