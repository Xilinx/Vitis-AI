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

#ifndef _XF_GRAPH_HASH_MAX_FREQ_HPP_
#define _XF_GRAPH_HASH_MAX_FREQ_HPP_

#include "hls_stream.h"
#include <ap_int.h>

#include "xf_utils_hw/axi_to_stream.hpp"
#include "xf_utils_hw/stream_to_axi.hpp"
#include "xf_utils_hw/stream_n_to_one/load_balance.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

#define rot(x, k) (((x) << (k)) | ((x) >> (32 - (k))));

namespace xf {
namespace graph {
namespace internal {
namespace hash_group_aggregate {

template <int W>
inline void hashlookup3_seed_core(ap_uint<W> key_val, ap_uint<32> seed, ap_uint<64>& hash_val) {
    const int key32blen = W / 32;
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

LOOP_lookup3_MAIN:
    for (int j = 0; j < key96blen; ++j) {
        a += key_val(96 * j + 31, 96 * j);
        b += key_val(96 * j + 63, 96 * j + 32);
        c += key_val(96 * j + 95, 96 * j + 64);

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

    hash1 = (ap_uint<64>)c;
    hash2 = (ap_uint<64>)b;

    hash_val = hash1 << 32 | hash2;
} // lookup3_64

template <int _Width, int _ColumnNM>
struct COLUMN_DATA {
    ap_uint<_Width> data[_ColumnNM];
};

// ---------------------------------write HBM/DDR---------------------------

template <int _WKey, int _KeyNM, int _WPay, int _PayNM, int _WBuffer>
void combine_col(
    // input stream
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> >& kin_strm,
    hls::stream<COLUMN_DATA<_WPay, _PayNM> >& pin_strm,
    hls::stream<bool>& ein_strm,

    // output streams
    hls::stream<ap_uint<_WBuffer> >& out_strm,
    hls::stream<bool>& eout_strm) {
#pragma HLS inline off

    ap_uint<_WBuffer> out = 0;
    bool end = ein_strm.read();
    while (!end) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 10 min = 10

        COLUMN_DATA<_WKey, _KeyNM> key = kin_strm.read();
        for (int i = 0; i < _KeyNM; i++) {
#pragma HLS unroll
            out((i + 1) * _WKey - 1, i * _WKey) = key.data[i];
        }

        COLUMN_DATA<_WPay, _PayNM> pld = pin_strm.read();
        for (int i = 0; i < _PayNM; i++) {
#pragma HLS unroll
            out((i + 1) * _WPay + _WKey * _KeyNM - 1, i * _WPay + _WKey * _KeyNM) = pld.data[i];
        }
        end = ein_strm.read();
        out_strm.write(out);
        eout_strm.write(false);
    }
    eout_strm.write(true);
}

/// @brief write data from streams to DDR/HBM
template <int _WKey, int _KeyNM, int _WPay, int _PayNM, int _WBuffer, int _BurstLenW>
void stream_to_buf(
    // unhandled streams from aggr_spill
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> >& strm_undo_key,
    hls::stream<COLUMN_DATA<_WPay, _PayNM> >& strm_undo_pld,
    hls::stream<bool>& strm_undo_e,

    // buffer for storing the unhandled data
    ap_uint<_WBuffer>* pong_buf) {
#pragma HLS DATAFLOW

    // streams between combine_col and stream_to_axi
    hls::stream<ap_uint<_WBuffer> > strm_unhandle_data;
    hls::stream<bool> strm_unhandle_e;
#pragma HLS RESOURCE variable = strm_unhandle_data core = FIFO_SRL
#pragma HLS STREAM variable = strm_unhandle_data depth = 32
#pragma HLS RESOURCE variable = strm_unhandle_e core = FIFO_SRL
#pragma HLS STREAM variable = strm_unhandle_e depth = 32

    // combine unhandled streams from aggr_spill
    combine_col<_WKey, _KeyNM, _WPay, _PayNM, _WBuffer>(strm_undo_key, strm_undo_pld, strm_undo_e, // stream in
                                                        strm_unhandle_data, strm_unhandle_e);      // stream out

    // write unhandled data back to DDR
    xf::common::utils_hw::streamToAxi<_BurstLenW, _WBuffer, _WBuffer>(pong_buf, strm_unhandle_data, strm_unhandle_e);
}

// ---------------------------------read HBM/DDR---------------------------

template <int _WKey, int _KeyNM, int _WPay, int _PayNM, int _WBuffer>
void split_col(
    // input stream
    hls::stream<ap_uint<_WBuffer> >& in_strm,
    hls::stream<bool>& ein_strm,

    // output streams
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> >& kout_strm,
    hls::stream<COLUMN_DATA<_WPay, _PayNM> >& pout_strm,
    hls::stream<bool>& eout_strm) {
#pragma HLS inline off

    bool end = ein_strm.read();
    while (!end) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 10 min = 10
        ap_uint<_WBuffer> in = in_strm.read();
        end = ein_strm.read();

        COLUMN_DATA<_WKey, _KeyNM> key;
        for (int i = 0; i < _KeyNM; i++) {
            key.data[i] = in((i + 1) * _WKey - 1, i * _WKey);
        }

        COLUMN_DATA<_WPay, _PayNM> pld;
        for (int i = 0; i < _PayNM; i++) {
            pld.data[i] = in((i + 1) * _WPay + _WKey * _KeyNM - 1, i * _WPay + _WKey * _KeyNM);
        }

        kout_strm.write(key);
        pout_strm.write(pld);
        eout_strm.write(false);
    }
    eout_strm.write(true);
}

/// @brief read data from overflow-buffer to streams
template <int _WKey, int _KeyNM, int _WPay, int _PayNM, int _WBuffer, int _BurstLenR>
void buf_to_stream(
    // input buffer
    ap_uint<_WBuffer>* in_buf,
    int unhandle_cnt,

    // output streams
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> >& kout_strm,
    hls::stream<COLUMN_DATA<_WPay, _PayNM> >& pout_strm,
    hls::stream<bool>& eout_strm) {
#pragma HLS DATAFLOW

    // streams between axi_to_stream and split_col
    hls::stream<ap_uint<_WBuffer> > ostrm;
#pragma HLS RESOURCE variable = ostrm core = FIFO_SRL
#pragma HLS STREAM variable = ostrm depth = 32
    hls::stream<bool> e0_strm;
#pragma HLS RESOURCE variable = e0_strm core = FIFO_SRL
#pragma HLS STREAM variable = e0_strm depth = 32

    xf::common::utils_hw::axiToStream<_BurstLenR, _WBuffer, ap_uint<_WBuffer> >(in_buf, unhandle_cnt, ostrm, e0_strm);

    split_col<_WKey, _KeyNM, _WPay, _PayNM, _WBuffer>(ostrm, e0_strm, kout_strm, pout_strm, eout_strm);

} // end buf_to_stream

// ---------------------------------hash---------------------------------------

/// @brief Calculate hash value based on key
template <int HASH_MODE, int KEYW, int KeyNM, int HASHW>
void hash_wrapper(hls::stream<COLUMN_DATA<KEYW, KeyNM> >& i_key_strm,
                  hls::stream<bool>& i_e_strm,

                  ap_uint<32> round,

                  hls::stream<ap_uint<HASHW> >& o_hash_strm,
                  hls::stream<COLUMN_DATA<KEYW, KeyNM> >& o_key_strm,
                  hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off
    bool last = i_e_strm.read();
BUILD_HASH_LOOP:
    while (!last) {
#pragma HLS loop_tripcount min = 1 max = 10
#pragma HLS PIPELINE II = 1

        COLUMN_DATA<KEYW, KeyNM> key = i_key_strm.read();
        o_key_strm.write(key);
        last = i_e_strm.read();
        o_e_strm.write(false);

        if (HASH_MODE == 0) {
            // radix hash
            ap_uint<HASHW> s_hash_val = key.data[0];
            o_hash_strm.write(s_hash_val);

        } else {
            // Jekins lookup3 hash
            ap_uint<64> l_hash_val;
            ap_uint<HASHW> s_hash_val;
            ap_uint<32> seed = round + 0xdeadbeef;

            hashlookup3_seed_core<32>((ap_uint<32>)(key.data[0]), seed, l_hash_val);

            s_hash_val = l_hash_val(HASHW - 1, 0);
            o_hash_strm.write(s_hash_val);
        }
    }
    o_e_strm.write(true);
}

// -----------------------------------dispatch------------------------------

// prepare the input data for hash_aggr
template <int _WKey, int _KeyNM, int _WPay, int _PayNM, int _WBuffer, int _BurstLenR>
void input_mux(
    // extern input
    hls::stream<ap_uint<_WKey> >& kin_strm,
    hls::stream<bool>& ein_strm,

    // input buffer
    ap_uint<_WBuffer>* in_buf,
    ap_uint<32> unhandle_cnt,
    ap_uint<32> round,

    // stream out
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> >& kout_strm,
    hls::stream<COLUMN_DATA<_WPay, _PayNM> >& pout_strm,
    hls::stream<bool>& eout_strm) {
#pragma HLS inline off

    if (round == 0) { // use data from extern input
        COLUMN_DATA<_WKey, _KeyNM> key;
#pragma HLS array_partition variable = key complete
        COLUMN_DATA<_WPay, _PayNM> pld;
#pragma HLS array_partition variable = pld complete

        bool e = ein_strm.read();
        while (!e) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 10 min = 10

            ap_uint<_WKey> tmp = kin_strm.read();
            for (int i = 0; i < _KeyNM; i++) {
                key.data[i] = tmp;
            }
            kout_strm.write(key);
            pout_strm.write(key);

            e = ein_strm.read();
            eout_strm.write(false);
        } // end while
        eout_strm.write(true);
    } else { // use data from HBM
        if (unhandle_cnt != 0) {
            buf_to_stream<_WKey, _KeyNM, _WPay, _PayNM, _WBuffer, _BurstLenR>(in_buf, unhandle_cnt, kout_strm,
                                                                              pout_strm, eout_strm);
        } else {
            eout_strm.write(true);
        }
    } // end if-else
} // end input_mux

// dispatch multi column key && pld
template <int KEYW, int KeyNM, int PW, int PayNM, int HASHWH, int HASHWL, int PU>
void dispatch(hls::stream<COLUMN_DATA<KEYW, KeyNM> >& i_key_strm,
              hls::stream<COLUMN_DATA<PW, PayNM> >& i_pld_strm,
              hls::stream<ap_uint<HASHWH + HASHWL> >& i_hash_strm,
              hls::stream<bool>& i_e_strm,

              hls::stream<COLUMN_DATA<KEYW, KeyNM> > o_key_strm[PU],
              hls::stream<COLUMN_DATA<PW, PayNM> > o_pld_strm[PU],
              hls::stream<ap_uint<HASHWL> > o_hash_strm[PU],
              hls::stream<bool> o_e_strm[PU]) {
#pragma HLS INLINE off

    bool last = i_e_strm.read();
LOOP_DISPATCH:
    while (!last) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 10 min = 10

        ap_uint<HASHWH + HASHWL> hash_val = i_hash_strm.read();
        ap_uint<HASHWL> hash_out = hash_val(HASHWL - 1, 0);
        ap_uint<4> idx;
        if (HASHWH > 0)
            idx = hash_val(HASHWH + HASHWL - 1, HASHWL);
        else
            idx = 0;

        COLUMN_DATA<KEYW, KeyNM> key = i_key_strm.read();
        COLUMN_DATA<PW, PayNM> pld = i_pld_strm.read();

        o_key_strm[idx].write(key);
        o_pld_strm[idx].write(pld);
        last = i_e_strm.read();
        o_hash_strm[idx].write(hash_out);
        o_e_strm[idx].write(false);
    }

    // for do_while in merge function
    COLUMN_DATA<KEYW, KeyNM> pad_key;
    COLUMN_DATA<PW, PayNM> pad_pld;

    for (int i = 0; i < PU; i++) {
#pragma HLS unroll

        o_key_strm[i].write(pad_key);
        o_pld_strm[i].write(pad_pld);
        o_hash_strm[i].write(0);
        o_e_strm[i].write(true);
    }
}

// dispatch data based on hash value to multiple PU.
template <int _HashMode,
          int _WKey,
          int _KeyNM,
          int _WPay,
          int _PayNM,
          int _HASHWH,
          int _HASHWL,
          int PU,
          int _WBuffer,
          int _BurstLenR>
void dispatch_wrapper(hls::stream<ap_uint<_WKey> >& i_key_strm,
                      hls::stream<bool>& i_e_strm,

                      ap_uint<_WBuffer>* in_buf,
                      ap_uint<32> unhandle_cnt,
                      ap_uint<32> round,

                      hls::stream<COLUMN_DATA<_WKey, _KeyNM> > o_key_strm[PU],
                      hls::stream<COLUMN_DATA<_WPay, _PayNM> > o_pld_strm[PU],
                      hls::stream<ap_uint<_HASHWL> > o_hash_strm[PU],
                      hls::stream<bool> o_e_strm[PU]) {
#pragma HLS DATAFLOW

    hls::stream<COLUMN_DATA<_WKey, _KeyNM> > key0_strm;
#pragma HLS STREAM variable = key0_strm depth = 8
#pragma HLS resource variable = key0_strm core = FIFO_SRL
    hls::stream<COLUMN_DATA<_WPay, _PayNM> > pld0_strm;
#pragma HLS STREAM variable = pld0_strm depth = 32
#pragma HLS resource variable = pld0_strm core = FIFO_SRL
    hls::stream<bool> e0_strm;
#pragma HLS STREAM variable = e0_strm depth = 8

    hls::stream<ap_uint<_HASHWH + _HASHWL> > hash_strm;
#pragma HLS STREAM variable = hash_strm depth = 8
#pragma HLS resource variable = hash_strm core = FIFO_SRL
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> > key1_strm;
#pragma HLS STREAM variable = key1_strm depth = 8
#pragma HLS resource variable = key1_strm core = FIFO_SRL
    hls::stream<bool> e1_strm;
#pragma HLS STREAM variable = e1_strm depth = 8

    input_mux<_WKey, _KeyNM, _WPay, _PayNM, _WBuffer, _BurstLenR>(i_key_strm, i_e_strm, in_buf, unhandle_cnt, round,
                                                                  key0_strm, pld0_strm, e0_strm);

    hash_wrapper<_HashMode, _WKey, _KeyNM, _HASHWH + _HASHWL>(key0_strm, e0_strm, round, hash_strm, key1_strm, e1_strm);

    dispatch<_WKey, _KeyNM, _WPay, _PayNM, _HASHWH, _HASHWL, PU>(key1_strm, pld0_strm, hash_strm, e1_strm, o_key_strm,
                                                                 o_pld_strm, o_hash_strm, o_e_strm);
}

// -------------------------------------merge-------------------------------

/// @brief Merge stream of multiple channels into one PU, merge 1 to 1
template <int KEYW, int KeyNM, int PW, int PayNM, int HASHW>
void merge1_1(hls::stream<COLUMN_DATA<KEYW, KeyNM> >& i_key_strm,
              hls::stream<COLUMN_DATA<PW, PayNM> >& i_pld_strm,
              hls::stream<ap_uint<HASHW> >& i_hash_strm,
              hls::stream<bool>& i_e_strm,
              hls::stream<COLUMN_DATA<KEYW, KeyNM> >& o_key_strm,
              hls::stream<COLUMN_DATA<PW, PayNM> >& o_pld_strm,
              hls::stream<ap_uint<HASHW> >& o_hash_strm,
              hls::stream<bool>& o_e_strm) {
    bool last = 0;
LOOP_MERGE1_1:
    do {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 10 min = 10
        COLUMN_DATA<KEYW, KeyNM> key;
        COLUMN_DATA<PW, PayNM> pld;
        ap_uint<HASHW> hash_val;

        key = i_key_strm.read();
        pld = i_pld_strm.read();
        hash_val = i_hash_strm.read();
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

template <int _WPay>
ap_uint<_WPay> aggr_cnt_nz(ap_uint<_WPay> pld, ap_uint<_WPay> uram, bool sign, bool enable) {
#pragma HLS inline

    if (enable) {
        return (pld == 0) && sign ? uram : (ap_uint<_WPay>)(uram + 1);
    } else {
        return uram;
    }
}

// -----------------------------initialize uram-------------------------------
/// @brief initiaalize uram
template <int _WKey, int _WPay, int _WHash, int _Wcnt, int PL>
void initial_uram(
    // controling param
    ap_uint<4> op,

    // uram buffer
    ap_uint<_WKey + _Wcnt>* key_uram,
    ap_uint<_WPay>* pld_uram0) {
#pragma HLS inline

    enum { depth = (1 << (_WHash - 3 - PL)) }; // depth of URAM

loop_initial_uram:
    for (int i = 0; i < depth; i++) {
#pragma HLS pipeline II = 1
        for (int j = 0; j < (1 << PL); j++) {
            key_uram[i + j * depth] = 0;
            pld_uram0[i + j * depth] = 0;
        }
    }
}

// -----------------------------update key uram-------------------------------

// calculate new key element
template <int _WKey, int _KeyNM, int _PayNM, int _Wcnt>
void update_key_elem(
    // in
    ap_uint<_WKey * _KeyNM> i_key,
    ap_uint<_WKey * _KeyNM + _Wcnt> i_uram,
    ap_uint<3> i_bit_idx,

    // controling param
    ap_uint<_Wcnt> max_col,

    // out
    bool& o_write_success,
    ap_uint<_WKey * _KeyNM + _Wcnt>& o_uram) {
#pragma HLS inline

    ap_uint<_Wcnt> col_status = i_uram(_WKey * _KeyNM + _Wcnt - 1, _WKey * _KeyNM);
    ap_uint<_Wcnt> new_col_status;

    // cope with different column number
    ap_uint<_WKey * _KeyNM + _Wcnt> uram;
    ap_uint<_KeyNM> not_find;
    ap_uint<_KeyNM> write_success;
    ap_uint<_KeyNM> enable;

    ap_uint<_WKey> insert_key[_KeyNM];
#pragma HLS array_partition variable = insert_key complete
    ap_uint<_WKey> old_key[_KeyNM];
#pragma HLS array_partition variable = old_key complete
    ap_uint<_WKey> new_key[_KeyNM];
#pragma HLS array_partition variable = new_key complete

    // generate input key for aggregate
    for (int i = 0; i < _KeyNM; i++) {
#pragma HLS unroll
        if (max_col == 1) {
            insert_key[i] = i_key(_WKey - 1, 0);
        } else if (max_col == 2) {
            insert_key[i] = i_key(((i % 2) + 1) * _WKey - 1, (i % 2) * _WKey);
        } else if (max_col > 2 && max_col <= 4) {
            insert_key[i] = i_key((i % 4 + 1) * _WKey - 1, i % 4 * _WKey);
        } else {
            insert_key[i] = i_key((i + 1) * _WKey - 1, i * _WKey);
        }
    }

    // initialize LUT for enable
    if (max_col == 1) {
        if (i_bit_idx == 0) {
            enable = 1; // 0000_0001
        } else if (i_bit_idx == 1) {
            enable = 2;
        } else if (i_bit_idx == 2) {
            enable = 4;
        } else if (i_bit_idx == 3) {
            enable = 8;
        } else if (i_bit_idx == 4) {
            enable = 16;
        } else if (i_bit_idx == 5) {
            enable = 32;
        } else if (i_bit_idx == 6) {
            enable = 64;
        } else {
            enable = 128;
        }
    } else if (max_col == 2) {
        if (i_bit_idx(1, 0) == 0) {
            enable = 3; // 0000_0011
        } else if (i_bit_idx(1, 0) == 1) {
            enable = 12; // 0000_1100
        } else if (i_bit_idx(1, 0) == 2) {
            enable = 48; // 0011_0000
        } else {
            enable = 192; // 1100_0000
        }
    } else if (max_col > 2 && max_col <= 4) {
        if (i_bit_idx[0] == 0) {
            enable = 15; // 0000_1111
        } else {
            enable = 240; // 1111_0000
        }
    } else {
        enable = 255; // 1111_1111
    }

    // generate control signal
    for (int i = 0; i < _KeyNM; i++) {
#pragma HLS unroll
        // compare
        old_key[i] = i_uram((i + 1) * _WKey - 1, i * _WKey);
        not_find[i] = old_key[i] == insert_key[i] ? 1 : 0;

        // control insert
        if (enable[i] == 1) {
            if (col_status[i] == 0 && not_find[i] == 0) {
                // insert new key
                write_success[i] = 1;
            } else if (col_status[i] == 0 && not_find[i] == 1) {
                // insert same key
                write_success[i] = 1;
            } else if (col_status[i] == 1 && not_find[i] == 0) {
                // insert failed because of overflow
                write_success[i] = 0;
            } else {
                // insert old key
                write_success[i] = 1;
            }
        } else {
            // not insert
            write_success[i] = 0;
        }
    }

    // update key uram
    for (int i = 0; i < _KeyNM; i++) {
#pragma HLS unroll
        if (write_success[i] == 1) {
            new_key[i] = insert_key[i];
            new_col_status[i] = 1;
        } else {
            new_key[i] = old_key[i];
            new_col_status[i] = col_status[i];
        }
        uram((i + 1) * _WKey - 1, i * _WKey) = new_key[i];
    }
    uram(_WKey * _KeyNM + _Wcnt - 1, _WKey * _KeyNM) = new_col_status;

    // assign output
    if (max_col == 1) {
        o_write_success = write_success != 0;
    } else if (max_col == 2) {
        o_write_success = write_success(1, 0) == 3 || write_success(3, 2) == 3 || write_success(5, 4) == 3 ||
                          write_success(7, 6) == 3;
    } else if (max_col > 2 && max_col <= 4) {
        o_write_success = write_success(3, 0) == 15 || write_success(7, 4) == 15;
    } else {
        o_write_success = write_success == 255;
    }

    o_uram = uram;
}

/// @brief update key uram for insert new key
template <int _WKey, int _KeyNM, int _WPay, int _PayNM, int _WHash, int _Wcnt>
void update_key_uram(
    // stream in
    hls::stream<ap_uint<_WHash> >& i_hash_strm,
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> >& i_key_strm,
    hls::stream<COLUMN_DATA<_WPay, _PayNM> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    // controling param
    ap_uint<_Wcnt> key_column,
    ap_uint<32>& unhandle_cnt,

    // uram
    ap_uint<_WKey + _Wcnt>* key_uram,

    // to HBM and wait for process again
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> >& undokey_strm,
    hls::stream<COLUMN_DATA<_WPay, _PayNM> >& undopld_strm,
    hls::stream<bool>& o_e0_strm,

    // to update pld uram
    hls::stream<ap_uint<_WHash> >& o_hash_strm,
    hls::stream<COLUMN_DATA<_WPay, _KeyNM> >& o_key_strm,
    hls::stream<COLUMN_DATA<_WPay, _PayNM> >& o_pld_strm,
    hls::stream<bool>& o_e1_strm) {
#pragma HLS inline off

    ap_uint<_WKey * _KeyNM + _Wcnt> elem;
    ap_uint<_WKey * _KeyNM + _Wcnt> elem_temp[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ap_uint<32> arry_idx_temp[12] = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                                     0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
#pragma HLS array_partition variable = elem_temp complete
#pragma HLS array_partition variable = arry_idx_temp complete

    COLUMN_DATA<_WKey, _KeyNM> key_in;
    COLUMN_DATA<_WPay, _PayNM> pld_in;
#pragma HLS array_partition variable = key_in complete
#pragma HLS array_partition variable = pld_in complete

    ap_uint<_WKey * _KeyNM> key;
    ap_uint<32> cnt = 0;
    bool last = i_e_strm.read();
    while (!last) {
#pragma HLS loop_tripcount max = 10 min = 10
#pragma HLS DEPENDENCE variable = key_uram pointer inter false
#pragma HLS pipeline II = 1
#pragma HLS LATENCY min = 8

        ap_uint<_WHash> hash = i_hash_strm.read();
        ap_uint<_WHash - 3> arry_idx = hash(_WHash - 4, 0);
        ap_uint<3> bit_idx = hash(_WHash - 1, _WHash - 3);
        key_in = i_key_strm.read();
        pld_in = i_pld_strm.read();
        for (int i = 0; i < _KeyNM; i++) {
#pragma HLS unroll
            key((i + 1) * _WKey - 1, i * _WKey) = key_in.data[i];
        }
        last = i_e_strm.read();

        // read temp, prevent duplicate key
        if (arry_idx == arry_idx_temp[0]) {
            elem = elem_temp[0];
        } else if (arry_idx == arry_idx_temp[1]) {
            elem = elem_temp[1];
        } else if (arry_idx == arry_idx_temp[2]) {
            elem = elem_temp[2];
        } else if (arry_idx == arry_idx_temp[3]) {
            elem = elem_temp[3];
        } else if (arry_idx == arry_idx_temp[4]) {
            elem = elem_temp[4];
        } else if (arry_idx == arry_idx_temp[5]) {
            elem = elem_temp[5];
        } else if (arry_idx == arry_idx_temp[6]) {
            elem = elem_temp[6];
        } else if (arry_idx == arry_idx_temp[7]) {
            elem = elem_temp[7];
        } else if (arry_idx == arry_idx_temp[8]) {
            elem = elem_temp[8];
        } else if (arry_idx == arry_idx_temp[9]) {
            elem = elem_temp[9];
        } else if (arry_idx == arry_idx_temp[10]) {
            elem = elem_temp[10];
        } else if (arry_idx == arry_idx_temp[11]) {
            elem = elem_temp[11];
        } else {
            elem = key_uram[arry_idx];
        }
        // calculate new element to update uram
        bool write_success;
        ap_uint<_WKey * _KeyNM + _Wcnt> new_elem;

        update_key_elem<_WKey, _KeyNM, _PayNM, _Wcnt>(key, elem, bit_idx, key_column, write_success, new_elem);

        // write key to uram
        key_uram[arry_idx] = new_elem;

        // right shift temp
        for (int i = 11; i > 0; i--) {
#pragma HLS unroll
            arry_idx_temp[i] = arry_idx_temp[i - 1];
            elem_temp[i] = elem_temp[i - 1];
        }
        arry_idx_temp[0] = arry_idx;
        elem_temp[0] = new_elem;

        // output
        if (write_success) {
            o_key_strm.write(key_in);
            o_pld_strm.write(pld_in);
            o_hash_strm.write(hash);
            o_e1_strm.write(false);
        } else {
            cnt++;
            undokey_strm.write(key_in);
            undopld_strm.write(pld_in);
            o_e0_strm.write(false);
        }
    }
    unhandle_cnt = cnt;
    o_e0_strm.write(true);
    o_e1_strm.write(true);
}

// generate pld for different col number
template <int _WPay, int _PayNM, int _WHash, int _Wcnt>
void update_pld_elem(
    // in
    ap_uint<_WPay * _PayNM> i_key,
    ap_uint<_WPay> i_rng,
    ap_uint<_WPay * _PayNM> i_pld,
    ap_uint<_WPay * _PayNM> i_pld_uram[3],

    // controlling param
    ap_uint<32> op_type,
    ap_uint<_Wcnt> max_col,
    ap_uint<3> bit_idx,

    // elem out
    unsigned int& max_count,
    unsigned int& max_key,
    unsigned int& max_rng,
    hls::stream<ap_uint<_WPay> >& o_max_key_strm,
    hls::stream<bool>& o_max_end_strm,
    ap_uint<_WPay * _PayNM> o_pld_uram[1]) {
#pragma HLS inline

    ap_uint<_WPay> pld[_PayNM];
#pragma HLS array_partition variable = pld
    ap_uint<_WPay> col_agg[_PayNM][3];
#pragma HLS array_partition variable = col_agg
    ap_uint<_WPay> new_col_agg[_PayNM][3];
#pragma HLS array_partition variable = new_col_agg
    ap_uint<_WPay * _PayNM> o_pld[3];
#pragma HLS array_partition variable = o_pld

    ap_uint<_PayNM> pld_status;
    ap_uint<4> op[_PayNM];

    // generate enable signal from bit_idx
    if (max_col == 1) {
        if (bit_idx == 0) {
            pld_status = 1;
        } else if (bit_idx == 1) {
            pld_status = 2;
        } else if (bit_idx == 2) {
            pld_status = 4;
        } else if (bit_idx == 3) {
            pld_status = 8;
        } else if (bit_idx == 4) {
            pld_status = 16;
        } else if (bit_idx == 5) {
            pld_status = 32;
        } else if (bit_idx == 6) {
            pld_status = 64;
        } else {
            pld_status = 128;
        }
    } else if (max_col == 2) {
        if (bit_idx(1, 0) == 0) {
            pld_status = 3;
        } else if (bit_idx(1, 0) == 1) {
            pld_status = 12;
        } else if (bit_idx(1, 0) == 2) {
            pld_status = 48;
        } else {
            pld_status = 192;
        }
    } else if (max_col > 2 && max_col <= 4) {
        if (bit_idx[0] == 0) {
            pld_status = 15;
        } else {
            pld_status = 240;
        }
    } else if (max_col > 4 && max_col <= 8) {
        pld_status = 255;
    } else {
        // not supported yet
    }

    int i = 0;
    // generate input pld for aggregate
    if (max_col == 1) {
        pld[i] = i_pld(_WPay - 1, 0);
    } else if (max_col == 2) {
        pld[i] = i_pld(((i % 2) + 1) * _WPay - 1, (i % 2) * _WPay);
    } else if (max_col > 2 && max_col <= 4) {
        pld[i] = i_pld((i % 4 + 1) * _WPay - 1, i % 4 * _WPay);
    } else {
        pld[i] = i_pld((i + 1) * _WPay - 1, i * _WPay);
    }

    // calculate aggr

    bool enable = pld_status[i] == 1;
    op[i] = op_type((i + 1) * 4 - 1, i * 4);
    col_agg[i][0] = i_pld_uram[0]((i + 1) * _WPay - 1, i * _WPay);
    new_col_agg[i][0] = aggr_cnt_nz(pld[i], col_agg[i][0], 0, enable);

    if (i == 0) {
        if (new_col_agg[i][0] > max_count) {
            max_count = new_col_agg[i][0];
            max_key = i_key(_WPay - 1, 0);
            max_rng = i_rng;

            o_max_key_strm.write(max_key);
            o_max_end_strm.write(false);
        } else if (new_col_agg[i][0] == max_count) {
            if (i_rng > max_rng) {
                max_count = new_col_agg[i][0];
                max_key = i_key(_WPay - 1, 0);
                max_rng = i_rng;

                o_max_key_strm.write(max_key);
                o_max_end_strm.write(false);
            }
        }
    }

    o_pld_uram[0]((i + 1) * _WPay - 1, i * _WPay) = new_col_agg[i][0];
}

/// @brief update pld uram to calculate aggregate
template <int _WPay, int _PayNM, int _WHash, int _Wcnt>
void update_pld_uram(
    // stream in
    hls::stream<ap_uint<_WHash> >& i_hash_strm,
    hls::stream<COLUMN_DATA<_WPay, _PayNM> >& i_key_strm,
    hls::stream<ap_uint<_WPay> >& strm_rng_in,
    hls::stream<COLUMN_DATA<_WPay, _PayNM> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    unsigned int& max_count,
    unsigned int& max_key,
    unsigned int& max_rng,

    // controlling param
    ap_uint<32> op,
    ap_uint<_Wcnt> pld_column,

    hls::stream<ap_uint<_WPay> >& o_max_key_strm,
    hls::stream<bool>& o_max_end_strm,

    // uram
    ap_uint<_WPay>* pld_uram0 // min, cnt
    ) {
#pragma HLS inline off

    ap_uint<_WPay * _PayNM> elem[1];
    ap_uint<_WPay * _PayNM> new_elem[1];
    ap_uint<_WPay * _PayNM> elem_temp0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    ap_uint<_WPay * _PayNM> elem_temp1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    ap_uint<_WPay * _PayNM> elem_temp2[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    ap_uint<32> arry_idx_temp[8] = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                                    0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
    COLUMN_DATA<_WPay, _PayNM> pld_in;
    COLUMN_DATA<_WPay, _PayNM> key_in;
    ap_uint<_WPay * _PayNM> pld;
    ap_uint<_WPay * _PayNM> key;

#pragma HLS array_partition variable = elem_temp0 complete
#pragma HLS array_partition variable = elem_temp1 complete
#pragma HLS array_partition variable = elem_temp2 complete
#pragma HLS array_partition variable = arry_idx_temp complete
#pragma HLS array_partition variable = pld_in complete
    bool last = i_e_strm.read();

update_pld_loop:
    while (!last) {
#pragma HLS loop_tripcount max = 10 min = 10
#pragma HLS DEPENDENCE variable = pld_uram0 pointer inter false

#pragma HLS pipeline II = 1
#pragma HLS LATENCY min = 8

        ap_uint<_WHash> hash = i_hash_strm.read();
        ap_uint<_WHash - 3> arry_idx = hash(_WHash - 4, 0);
        ap_uint<3> bit_idx = hash(_WHash - 1, _WHash - 3);
        pld_in = i_pld_strm.read();
        key_in = i_key_strm.read();
        ap_uint<_WPay> rng = strm_rng_in.read();
        for (int i = 0; i < _PayNM; i++) {
#pragma HLS unroll
            pld((i + 1) * _WPay - 1, i * _WPay) = pld_in.data[i];
            key((i + 1) * _WPay - 1, i * _WPay) = key_in.data[i];
        }
        last = i_e_strm.read();

        // read temp, prevent duplicate idx
        if (arry_idx == arry_idx_temp[0]) {
            elem[0] = elem_temp0[0];
        } else if (arry_idx == arry_idx_temp[1]) {
            elem[0] = elem_temp0[1];
        } else if (arry_idx == arry_idx_temp[2]) {
            elem[0] = elem_temp0[2];
        } else if (arry_idx == arry_idx_temp[3]) {
            elem[0] = elem_temp0[3];
        } else if (arry_idx == arry_idx_temp[4]) {
            elem[0] = elem_temp0[4];
        } else if (arry_idx == arry_idx_temp[5]) {
            elem[0] = elem_temp0[5];
        } else if (arry_idx == arry_idx_temp[6]) {
            elem[0] = elem_temp0[6];
        } else if (arry_idx == arry_idx_temp[7]) {
            elem[0] = elem_temp0[7];
        } else {
            elem[0] = pld_uram0[arry_idx];
        }

        // calculate aggregate result && cope with different column size
        update_pld_elem<_WPay, _PayNM, _WHash>(key, rng, pld, elem, op, pld_column, bit_idx, max_count, max_key,
                                               max_rng, o_max_key_strm, o_max_end_strm, new_elem);

        // write uram
        pld_uram0[arry_idx] = new_elem[0];

        // right shift temp
        for (int i = 7; i > 0; i--) {
            elem_temp0[i] = elem_temp0[i - 1];
            arry_idx_temp[i] = arry_idx_temp[i - 1];
        }
        elem_temp0[0] = new_elem[0];
        arry_idx_temp[0] = arry_idx;
    }
}

/// @brief compute aggregate result and update URAM
template <int _HashMode,
          int _WHash,
          int _WKey,
          int _KeyNM,
          int _WPay,
          int _PayNM,
          int _Wcnt,
          int _WBuffer,
          int _BurstLenW>
void update_uram(
    // stream in
    hls::stream<ap_uint<_WHash> >& i_hash_strm,
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> >& i_key_strm,
    hls::stream<ap_uint<_WKey> >& strm_rng_in,
    hls::stream<COLUMN_DATA<_WPay, _PayNM> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    unsigned int& max_count,
    unsigned int& max_key,
    unsigned int& max_rng,
    // controling param
    ap_uint<32> op,
    ap_uint<32> key_column,
    ap_uint<32> pld_column,
    ap_uint<32> round,
    ap_uint<32>& unhandle_cnt,

    hls::stream<ap_uint<_WPay> >& o_max_key_strm,
    hls::stream<bool>& o_max_end_strm,

    // uram
    ap_uint<_WKey + _Wcnt>* key_uram,
    ap_uint<_WPay>* pld_uram0,

    // buffer
    ap_uint<_WBuffer>* out_buf) {
#pragma HLS dataflow

    // streams between update_key_uram and stream_to_buf
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> > key0_strm;
#pragma HLS RESOURCE variable = key0_strm core = FIFO_SRL
#pragma HLS STREAM variable = key0_strm depth = 8
    hls::stream<COLUMN_DATA<_WPay, _PayNM> > pld0_strm;
#pragma HLS RESOURCE variable = pld0_strm core = FIFO_SRL
#pragma HLS STREAM variable = pld0_strm depth = 8
    hls::stream<bool> e0_strm;
#pragma HLS RESOURCE variable = e0_strm core = FIFO_SRL
#pragma HLS STREAM variable = e0_strm depth = 8

    // streams between update_key_uram and update_pld_uram
    hls::stream<ap_uint<_WHash> > hash_strm;
#pragma HLS RESOURCE variable = hash_strm core = FIFO_SRL
#pragma HLS STREAM variable = hash_strm depth = 8
    hls::stream<COLUMN_DATA<_WPay, _KeyNM> > key1_strm;
#pragma HLS RESOURCE variable = key1_strm core = FIFO_SRL
#pragma HLS STREAM variable = key1_strm depth = 8
    hls::stream<COLUMN_DATA<_WPay, _PayNM> > pld1_strm;
#pragma HLS RESOURCE variable = pld1_strm core = FIFO_SRL
#pragma HLS STREAM variable = pld1_strm depth = 8
    hls::stream<bool> e1_strm;
#pragma HLS RESOURCE variable = e1_strm core = FIFO_SRL
#pragma HLS STREAM variable = e1_strm depth = 8

    ap_uint<_Wcnt> max_col = key_column > pld_column ? key_column : pld_column;

    update_key_uram<_WKey, _KeyNM, _WPay, _PayNM, _WHash, _Wcnt>(i_hash_strm, i_key_strm, i_pld_strm, i_e_strm,

                                                                 max_col, unhandle_cnt, key_uram,

                                                                 key0_strm, pld0_strm, e0_strm,

                                                                 hash_strm, key1_strm, pld1_strm, e1_strm);

    stream_to_buf<_WKey, _KeyNM, _WPay, _PayNM, _WBuffer, _BurstLenW>(key0_strm, pld0_strm, e0_strm,

                                                                      out_buf);

    update_pld_uram<_WPay, _PayNM, _WHash, _Wcnt>(hash_strm, key1_strm, strm_rng_in, pld1_strm, e1_strm,

                                                  max_count, max_key, max_rng, op, max_col, o_max_key_strm,
                                                  o_max_end_strm, pld_uram0);
}

//--------------------------------pu top---------------------------------

/// @brief hash aggregate processing unit
template <int _HashMode,
          int _WHash,
          int _WKey,
          int _KeyNM,
          int _WPay,
          int _PayNM,
          int _Wcnt,
          int _WBuffer,
          int _BurstLenW>
void hash_aggr_pu_wrapper(
    // stream in
    hls::stream<ap_uint<_WHash> >& i_hash_strm,
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> >& i_key_strm,
    hls::stream<ap_uint<_WKey> >& strm_rng_in,
    hls::stream<COLUMN_DATA<_WPay, _PayNM> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    // out buffer for undo strm
    ap_uint<_WBuffer>* out_buf,

    unsigned int& max_count,
    unsigned int& max_key,
    unsigned int& max_rng,
    // controling param
    ap_uint<32> op,
    ap_uint<32> key_column,
    ap_uint<32> pld_column,
    ap_uint<32> round,
    ap_uint<32>& unhandle_cnt,

    hls::stream<ap_uint<_WPay> >& o_max_key_strm,
    hls::stream<bool>& o_max_end_strm) {
#pragma HLS inline off

    enum { depth = (1 << (_WHash - 3)) }; // depth of URAM
    const int PL = 7;
    const int PL2 = (1 << PL);

// define URAM structure
#ifndef __SYNTHESIS__

    ap_uint<_WKey + _Wcnt>* key_uram;
    ap_uint<_WPay>* pld_uram0;

    key_uram = (ap_uint<_WKey + _Wcnt>*)malloc(depth * sizeof(ap_uint<_WKey + _Wcnt>));
    pld_uram0 = (ap_uint<_WPay>*)malloc(depth * sizeof(ap_uint<_WPay>));

#else

    ap_uint<_WKey + _Wcnt> key_uram[depth];
#pragma HLS resource variable = key_uram core = RAM_2P_LUTRAM
#pragma HLS ARRAY_PARTITION variable = key_uram block factor = PL2
    ap_uint<_WPay> pld_uram0[depth];
#pragma HLS resource variable = pld_uram0 core = RAM_2P_LUTRAM
#pragma HLS ARRAY_PARTITION variable = pld_uram0 block factor = PL2

#endif

    initial_uram<_WKey, _WPay, _WHash, _Wcnt, PL>(op, key_uram, pld_uram0);

    update_uram<_HashMode, _WHash, _WKey, _KeyNM, _WPay, _PayNM, _Wcnt, _WBuffer, _BurstLenW>(
        i_hash_strm, i_key_strm, strm_rng_in, i_pld_strm, i_e_strm, max_count, max_key, max_rng, op, key_column,
        pld_column, round, unhandle_cnt, o_max_key_strm, o_max_end_strm, key_uram, pld_uram0, out_buf);

#ifndef __SYNTHESIS__

    free(key_uram);
    free(pld_uram0);

#endif
}

// ------------------------top function of hash aggregate--------------------

/// @brief do hash_group_aggregate for once
template <int _WKey,
          int _KeyNM,
          int _WPay,
          int _PayNM,
          int _HashMode,
          int _WHashHigh,
          int _WHashLow,
          int _CHNM,
          int _Wcnt,
          int _WBuffer,
          int _BurstLenW,
          int _BurstLenR>
void hash_aggr_top(
    // stream in
    hls::stream<ap_uint<_WKey> >& strm_key_in,
    hls::stream<ap_uint<_WKey> >& strm_rng_in,
    hls::stream<bool>& strm_e_in,

    unsigned int& max_count,
    unsigned int& max_key,
    unsigned int& max_rng,
    // operation type
    ap_uint<32> op_type[1 << _WHashHigh + 1],
    ap_uint<32> key_column,
    ap_uint<32> pld_column,
    ap_uint<32> round,
    ap_uint<32> unhandle_cnt_r[1 << _WHashHigh],
    ap_uint<32> unhandle_cnt_w[1 << _WHashHigh],
    ap_uint<32>& aggregate_num,

    // input & output buffer
    ap_uint<_WBuffer>* in_buf0,
    ap_uint<_WBuffer>* out_buf0,
    // stream out
    hls::stream<ap_uint<_WPay> >& o_max_key_strm,
    hls::stream<bool>& o_max_end_strm) {
#pragma HLS inline off
#pragma HLS DATAFLOW

    enum { PU = (1 << _WHashHigh) }; // high hash for distribution.

    // dispatch
    // Channel1
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> > k1_strm_arry_c0[PU];
#pragma HLS stream variable = k1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c0 complete
#pragma HLS resource variable = k1_strm_arry_c0 core = FIFO_SRL
    hls::stream<COLUMN_DATA<_WPay, _PayNM> > p1_strm_arry_c0[PU];
#pragma HLS stream variable = p1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c0 complete
#pragma HLS resource variable = p1_strm_arry_c0 core = FIFO_SRL
    hls::stream<ap_uint<_WHashLow> > hash_strm_arry_c0[PU];
#pragma HLS stream variable = hash_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c0 complete
#pragma HLS resource variable = hash_strm_arry_c0 core = FIFO_SRL
    hls::stream<bool> e1_strm_arry_c0[PU];
#pragma HLS stream variable = e1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c0 complete

    // merge channel1-channel4 to here, then mux HBM data as input of hash
    // aggregate PU
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> > k1_strm_arry[PU];
#pragma HLS stream variable = k1_strm_arry depth = 8
#pragma HLS array_partition variable = k1_strm_arry complete
#pragma HLS resource variable = k1_strm_arry core = FIFO_SRL
    hls::stream<COLUMN_DATA<_WPay, _PayNM> > p1_strm_arry[PU];
#pragma HLS stream variable = p1_strm_arry depth = 8
#pragma HLS array_partition variable = p1_strm_arry complete
#pragma HLS resource variable = p1_strm_arry core = FIFO_SRL
    hls::stream<ap_uint<_WHashLow> > hash1_strm_arry[PU];
#pragma HLS stream variable = hash1_strm_arry depth = 8
#pragma HLS array_partition variable = hash1_strm_arry complete
#pragma HLS resource variable = hash1_strm_arry core = FIFO_SRL
    hls::stream<bool> e1_strm_arry[PU];
#pragma HLS stream variable = e1_strm_arry depth = 8
#pragma HLS array_partition variable = e1_strm_arry complete

    //---------------------------------dispatch PU-------------------------------
    internal::hash_group_aggregate::dispatch_wrapper<_HashMode, _WKey, _KeyNM, _WPay, _PayNM, _WHashHigh, _WHashLow, PU,
                                                     _WBuffer, _BurstLenR>(
        strm_key_in, strm_e_in, in_buf0, unhandle_cnt_r[0], round, k1_strm_arry_c0, p1_strm_arry_c0, hash_strm_arry_c0,
        e1_strm_arry_c0);

    //---------------------------------merge PU---------------------------------
    for (int p = 0; p < PU; ++p) {
#pragma HLS unroll
        internal::hash_group_aggregate::merge1_1<_WKey, _KeyNM, _WPay, _PayNM, _WHashLow>(
            k1_strm_arry_c0[p], p1_strm_arry_c0[p], hash_strm_arry_c0[p], e1_strm_arry_c0[p], k1_strm_arry[p],
            p1_strm_arry[p], hash1_strm_arry[p], e1_strm_arry[p]);
    }

    //---------------------------------aggregate PU---------------------------------
    // hash aggregate processing unit
    internal::hash_group_aggregate::hash_aggr_pu_wrapper<_HashMode, _WHashLow, _WKey, _KeyNM, _WPay, _PayNM, _Wcnt,
                                                         _WBuffer, _BurstLenW>(
        hash1_strm_arry[0], k1_strm_arry[0], strm_rng_in, p1_strm_arry[0],
        e1_strm_arry[0], // input stream
        out_buf0,        // buffer
        max_count, max_key, max_rng,
        op_type[0],                      // operation
        key_column, pld_column,          // column number
        round,                           // loop cnt
        unhandle_cnt_w[0],               // overflow cnt
        o_max_key_strm, o_max_end_strm); // output stream

} // end hash_aggr_top

} // namespace hash_group_aggregate

template <int _WDT, int _WHashLow, int _WBuffer, int _BurstLenW = 32, int _BurstLenR = 32>
void hashGroupAggregate(
    // stream in
    hls::stream<ap_uint<_WDT> >& strm_key_in,
    hls::stream<ap_uint<_WDT> >& strm_rng_in,
    hls::stream<bool>& strm_e_in,

    // ping-pong buffer
    ap_uint<_WBuffer>* ping_buf0,
    ap_uint<_WBuffer>* pong_buf0,
    // stream out
    hls::stream<ap_uint<_WDT> >& o_max_key_strm,
    hls::stream<bool>& o_max_end_strm) {
#pragma HLS inline off
    enum { PU = 1 }; // high hash for distribution.

    bool loop_continue;
    ap_uint<4> op1 = 3;
    ap_uint<32> op = (op1, op1, op1, op1, op1, op1, op1, op1);
    ap_uint<32> key_column = 8;
    ap_uint<32> pld_column = 8;
    ap_uint<32> aggr_num = 0;
    ap_uint<32> round = 0;
    ap_uint<32> unhandle_cnt_r[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma HLS ARRAY_PARTITION variable = unhandle_cnt_r complete
    ap_uint<32> unhandle_cnt_w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma HLS ARRAY_PARTITION variable = unhandle_cnt_w complete

    ap_uint<32> op_type[9] = {op, op, op, op, op, op, op, op, op};

    unsigned int max_count = 0;
    unsigned int max_key = 0;
    unsigned int max_rng = 0;

    do {
#pragma HLS ALLOCATION instances = hash_aggr_top limit = 1 function
#pragma HLS loop_tripcount min = 1 max = 1
        if (round[0] == 0) {
            hash_group_aggregate::hash_aggr_top<_WDT, 8, _WDT, 8, 1, 0, _WHashLow, 1, 8, _WBuffer, _BurstLenW,
                                                _BurstLenR>(strm_key_in, strm_rng_in, strm_e_in, max_count, max_key,
                                                            max_rng, op_type, key_column, pld_column, round,
                                                            unhandle_cnt_r, unhandle_cnt_w, aggr_num,

                                                            ping_buf0, pong_buf0, o_max_key_strm, o_max_end_strm);
        } else {
            hash_group_aggregate::hash_aggr_top<_WDT, 8, _WDT, 8, 1, 0, _WHashLow, 1, 8, _WBuffer, _BurstLenW,
                                                _BurstLenR>(strm_key_in, strm_rng_in, strm_e_in, max_count, max_key,
                                                            max_rng, op_type, key_column, pld_column, round,
                                                            unhandle_cnt_r, unhandle_cnt_w, aggr_num,

                                                            pong_buf0, ping_buf0, o_max_key_strm, o_max_end_strm);
        }

        // generate control signal
        round++;
        loop_continue = false;

        loop_continue |= (unhandle_cnt_w[0] != 0);
        unhandle_cnt_r[0] = unhandle_cnt_w[0];

    } while (loop_continue);
    o_max_end_strm.write(true);

} // end hashGroupAggregate

} // namespace internal
} // namespace graph
} // namespace xf

#undef rot
#endif
