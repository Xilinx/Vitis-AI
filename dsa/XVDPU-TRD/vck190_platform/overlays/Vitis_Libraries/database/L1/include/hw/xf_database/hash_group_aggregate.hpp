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
 * @file hash_group_aggregate.hpp
 * @brief HASH GROUP AGGREGATE template function implementation.
 *
 * This file is part of Vitis Database Library.
 */

#ifndef _XF_DATABASE_HASH_AGGR_GENERAL_H_
#define _XF_DATABASE_HASH_AGGR_GENERAL_H_

#ifndef __cplusplus
#error "Vitis Database Library only works with C++."
#endif

#include "xf_database/utils.hpp"
#include "hls_stream.h"

#include "xf_database/combine_split_col.hpp"
#include "xf_database/hash_lookup3.hpp"
#include "xf_database/hash_join_v2.hpp"
#include "xf_database/types.hpp"
#include "xf_database/enums.hpp"

#include "xf_utils_hw/axi_to_stream.hpp"
#include "xf_utils_hw/stream_to_axi.hpp"
#include "xf_utils_hw/stream_n_to_one/load_balance.hpp"

// for wide output
#include <ap_int.h>

#ifdef DEBUG_AGGR
#undef DEBUG_AGGR
#endif

#ifdef DEBUG_UPDATE_KEY
#undef DEBUG_UPDATE_KEY
#endif

#ifdef DEBUG_UPDATE_PLD
#undef DEBUG_UPDATE_PLD
#endif

#ifdef DEBUG_READ_KEY
#undef DEBUG_READ_KEY
#endif

#ifdef DEBUG_READ_PLD
#undef DEBUG_READ_PLD
#endif

#ifdef DEBUG_MISS
#undef DEBUG_MISS
#endif

#ifdef DEBUG_HBM
#undef DEBUG_HBM
#endif

#ifdef DEBUG_RESULT
#undef DEBUG_RESULT
#endif

//#define DEBUG_AGGR true

#ifdef DEBUG_AGGR
#include <iostream>
//#define DEBUG_UPDATE_KEY true
//#define DEBUG_UPDATE_PLD true
//#define DEBUG_READ_KEY true
//#define DEBUG_READ_PLD true
//#define DEBUG_MISS true
//#define DEBUG_HBM true
//#define DEBUG_RESULT true
#endif

namespace xf {
namespace database {
namespace details {
namespace hash_group_aggregate {

template <int _Width, int _ColumnNM>
struct COLUMN_DATA {
    ap_uint<_Width> data[_ColumnNM];
};

// -------------------------------config && info----------------------------

template <int _W>
void read_config(hls::stream<ap_uint<_W> >& config_strm,
                 ap_uint<_W>& op,
                 ap_uint<_W>& key_column,
                 ap_uint<_W>& pld_column,
                 ap_uint<_W>& aggr_num) {
    op = config_strm.read();
    key_column = config_strm.read();
    pld_column = config_strm.read();
    aggr_num = config_strm.read();
}

template <int _W>
void write_info(hls::stream<ap_uint<_W> >& info_strm,
                ap_uint<_W> op,
                ap_uint<_W> key_column,
                ap_uint<_W> pld_column,
                ap_uint<_W> aggr_num) {
    info_strm.write(op);
    info_strm.write(key_column);
    info_strm.write(pld_column);
    info_strm.write(aggr_num);
}

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

#ifndef __SYNTHESIS__
#ifdef DEBUG_HBM
        std::cout << std::hex << "HBM out: " << out << std::endl;
#endif
#endif
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
#pragma HLS bind_storage variable = strm_unhandle_data type = fifo impl = srl
#pragma HLS STREAM variable = strm_unhandle_data depth = 32
#pragma HLS bind_storage variable = strm_unhandle_e type = fifo impl = srl
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

#ifndef __SYNTHESIS__
#ifdef DEBUG_HBM
        std::cout << std::hex << "HBM in: " << in << std::endl;
#endif
#endif
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
#pragma HLS bind_storage variable = ostrm type = fifo impl = srl
#pragma HLS STREAM variable = ostrm depth = 32
    hls::stream<bool> e0_strm;
#pragma HLS bind_storage variable = e0_strm type = fifo impl = srl
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

#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif

    bool last = i_e_strm.read();
BUILD_HASH_LOOP:
    while (!last) {
#pragma HLS loop_tripcount min = 1 max = 1000
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

            if (KeyNM == 1)
                database::details::hashlookup3_seed_core<32>((ap_uint<32>)(key.data[0]), seed, l_hash_val);
            else if (KeyNM == 2)
                database::details::hashlookup3_seed_core<96>((ap_uint<96>)(0, key.data[1], key.data[0]), seed,
                                                             l_hash_val);
            else
                database::details::hashlookup3_seed_core<96>((ap_uint<96>)(key.data[2], key.data[1], key.data[0]), seed,
                                                             l_hash_val);

            s_hash_val = l_hash_val(HASHW - 1, 0);
            o_hash_strm.write(s_hash_val);

#ifndef __SYNTHESIS__

#ifdef DEBUG_MISS
            if (key.data[0] == 1881824) {
                std::cout << std::hex << "hash wrapper: cnt=" << cnt << " key = " << key.data[0]
                          << " hash_val = " << s_hash_val << std::endl;
            }
#endif
            cnt++;
#endif
        }
    }
    o_e_strm.write(true);
}

// -----------------------------------dispatch------------------------------

// prepare the input data for hash_aggr
template <int _WKey, int _KeyNM, int _WPay, int _PayNM, int _WBuffer, int _BurstLenR>
void input_mux(
    // extern input
    hls::stream<ap_uint<_WKey> > kin_strm[_KeyNM],
    hls::stream<ap_uint<_WPay> > pin_strm[_PayNM],
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

            for (int i = 0; i < _KeyNM; i++) {
                key.data[i] = kin_strm[i].read();
            }
            kout_strm.write(key);

            for (int i = 0; i < _PayNM; i++) {
                pld.data[i] = pin_strm[i].read();
            }
            pout_strm.write(pld);

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

#ifndef __SYNTHESIS__
    int cnt = 0;
#endif

    bool last = i_e_strm.read();
LOOP_DISPATCH:
    while (!last) {
#pragma HLS pipeline II = 1

        ap_uint<HASHWH + HASHWL> hash_val = i_hash_strm.read();
        ap_uint<HASHWL> hash_out = hash_val(HASHWL - 1, 0);
        ap_uint<4> idx;
        if (HASHWH > 0)
            idx = hash_val(HASHWH + HASHWL - 1, HASHWL);
        else
            idx = 0;

        COLUMN_DATA<KEYW, KeyNM> key = i_key_strm.read();
        COLUMN_DATA<PW, PayNM> pld = i_pld_strm.read();

#ifndef __SYNTHESIS__

#ifdef DEBUG_AGGR
        if (cnt < 10) {
            std::cout << "dispatch:" << std::endl;
            for (int i = 0; i < KeyNM; i++) {
                std::cout << std::dec << "key_col" << i << "= " << key.data[i] << " ";
            }
            std::cout << std::endl;

            for (int i = 0; i < PayNM; i++) {
                std::cout << std::dec << "pld_col" << i << "= " << pld.data[i] << " ";
            }
            std::cout << std::endl;
        } else if (key.data[1] == 35) {
            std::cout << "dispatch find error data:" << std::endl;
            for (int i = 0; i < KeyNM; i++) {
                std::cout << std::dec << "key_col" << i << "= " << key.data[i] << " ";
            }
            std::cout << std::endl;

            for (int i = 0; i < PayNM; i++) {
                std::cout << std::dec << "pld_col" << i << "= " << pld.data[i] << " ";
            }
            std::cout << std::endl;
        }
#endif
        cnt++;
#endif

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
void dispatch_wrapper(hls::stream<ap_uint<_WKey> > i_key_strm[_KeyNM],
                      hls::stream<ap_uint<_WPay> > i_pld_strm[_PayNM],
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
#pragma HLS bind_storage variable = key0_strm type = fifo impl = srl
    hls::stream<COLUMN_DATA<_WPay, _PayNM> > pld0_strm;
#pragma HLS STREAM variable = pld0_strm depth = 32
#pragma HLS bind_storage variable = pld0_strm type = fifo impl = srl
    hls::stream<bool> e0_strm;
#pragma HLS STREAM variable = e0_strm depth = 8

    hls::stream<ap_uint<_HASHWH + _HASHWL> > hash_strm;
#pragma HLS STREAM variable = hash_strm depth = 8
#pragma HLS bind_storage variable = hash_strm type = fifo impl = srl
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> > key1_strm;
#pragma HLS STREAM variable = key1_strm depth = 8
#pragma HLS bind_storage variable = key1_strm type = fifo impl = srl
    hls::stream<bool> e1_strm;
#pragma HLS STREAM variable = e1_strm depth = 8

#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
    std::cout << "dispatch: round=" << round << " unhandle_cnt=" << unhandle_cnt << " in_buf=" << in_buf << std::endl;
#endif
#endif

    input_mux<_WKey, _KeyNM, _WPay, _PayNM, _WBuffer, _BurstLenR>(i_key_strm, i_pld_strm, i_e_strm, in_buf,
                                                                  unhandle_cnt, round, key0_strm, pld0_strm, e0_strm);

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

/// @brief Merge stream of multiple channels into one PU, merge 2 to 1
template <int KEYW, int KeyNM, int PW, int PayNM, int HASHW>
void merge2_1(hls::stream<COLUMN_DATA<KEYW, KeyNM> >& i0_key_strm,
              hls::stream<COLUMN_DATA<KEYW, KeyNM> >& i1_key_strm,
              hls::stream<COLUMN_DATA<PW, PayNM> >& i0_pld_strm,
              hls::stream<COLUMN_DATA<PW, PayNM> >& i1_pld_strm,
              hls::stream<ap_uint<HASHW> >& i0_hash_strm,
              hls::stream<ap_uint<HASHW> >& i1_hash_strm,
              hls::stream<bool>& i0_e_strm,
              hls::stream<bool>& i1_e_strm,
              hls::stream<COLUMN_DATA<KEYW, KeyNM> >& o_key_strm,
              hls::stream<COLUMN_DATA<PW, PayNM> >& o_pld_strm,
              hls::stream<ap_uint<HASHW> >& o_hash_strm,
              hls::stream<bool>& o_e_strm) {
    COLUMN_DATA<KEYW, KeyNM> key_arry[2];
#pragma HLS array_partition variable = key_arry complete dim = 0
    COLUMN_DATA<PW, PayNM> pld_arry[2];
#pragma HLS array_partition variable = pld_arry complete dim = 0
    ap_uint<HASHW> hash_val_arry[2];
#pragma HLS array_partition variable = hash_val_arry complete dim = 0

    ap_uint<2> empty_e = 0;
    ap_uint<2> rd_e = 0;
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

        rd_e = details::join_v2::mul_ch_read(empty_e);
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
        ap_uint<1> id = details::join_v2::mux<2>(rd_e);
        COLUMN_DATA<KEYW, KeyNM> key = key_arry[id];
        COLUMN_DATA<PW, PayNM> pld = pld_arry[id];
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

/// @brief Merge stream of multiple channels into one PU, merge 4 to 1
template <int KEYW, int KeyNM, int PW, int PayNM, int HASHW>
void merge4_1(hls::stream<COLUMN_DATA<KEYW, KeyNM> >& i0_key_strm,
              hls::stream<COLUMN_DATA<KEYW, KeyNM> >& i1_key_strm,
              hls::stream<COLUMN_DATA<KEYW, KeyNM> >& i2_key_strm,
              hls::stream<COLUMN_DATA<KEYW, KeyNM> >& i3_key_strm,
              hls::stream<COLUMN_DATA<PW, PayNM> >& i0_pld_strm,
              hls::stream<COLUMN_DATA<PW, PayNM> >& i1_pld_strm,
              hls::stream<COLUMN_DATA<PW, PayNM> >& i2_pld_strm,
              hls::stream<COLUMN_DATA<PW, PayNM> >& i3_pld_strm,
              hls::stream<ap_uint<HASHW> >& i0_hash_strm,
              hls::stream<ap_uint<HASHW> >& i1_hash_strm,
              hls::stream<ap_uint<HASHW> >& i2_hash_strm,
              hls::stream<ap_uint<HASHW> >& i3_hash_strm,
              hls::stream<bool>& i0_e_strm,
              hls::stream<bool>& i1_e_strm,
              hls::stream<bool>& i2_e_strm,
              hls::stream<bool>& i3_e_strm,
              hls::stream<COLUMN_DATA<KEYW, KeyNM> >& o_key_strm,
              hls::stream<COLUMN_DATA<PW, PayNM> >& o_pld_strm,
              hls::stream<ap_uint<HASHW> >& o_hash_strm,
              hls::stream<bool>& o_e_strm) {
    COLUMN_DATA<KEYW, KeyNM> key_arry[4];
#pragma HLS array_partition variable = key_arry complete dim = 0
    COLUMN_DATA<PW, PayNM> pld_arry[4];
#pragma HLS array_partition variable = pld_arry complete dim = 0
    ap_uint<HASHW> hash_val_arry[4];
#pragma HLS array_partition variable = hash_val_arry complete dim = 0

    ap_uint<4> empty_e = 0;
    ap_uint<4> rd_e = 0;
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

        rd_e = details::join_v2::mul_ch_read(empty_e);
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
        ap_uint<2> id = details::join_v2::mux<4>(rd_e);
        COLUMN_DATA<KEYW, KeyNM> key = key_arry[id];
        COLUMN_DATA<PW, PayNM> pld = pld_arry[id];
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

// ---------------------------------aggr function---------------------------

template <int _WPay>
ap_uint<_WPay> aggr_min_max(ap_uint<_WPay> pld, ap_uint<_WPay> uram, bool sign, bool enable) {
#pragma HLS inline

    if (enable) {
        return (pld > uram) ^ sign ? pld : uram;
    } else {
        return uram;
    }
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

template <int _WPay>
void aggr_sum(ap_uint<_WPay> pld, ap_uint<2 * _WPay> uram, ap_uint<2 * _WPay>& result, bool enable) {
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS pipeline II = 1
#pragma HLS inline off

    if (enable) {
        result = pld + uram;
    } else {
        result = uram;
    }
}

template <int _WPay>
void aggr_mean(ap_uint<2 * _WPay> sum, ap_uint<_WPay> cnt, ap_uint<2 * _WPay>& result) {
#pragma HLS inline

    if (cnt != 0)
        result = sum / cnt;
    else
        result = -1;
}

// -----------------------------initialize uram-------------------------------

/// @brief initiaalize uram
template <int _WKey, int _KeyNM, int _WPay, int _PayNM, int _WHash, int _Wcnt>
void initial_uram(
    // controling param
    ap_uint<4> op,

    // uram buffer
    ap_uint<_WKey * _KeyNM + _Wcnt>* key_uram,
    ap_uint<_WPay * _PayNM>* pld_uram0,
    ap_uint<_WPay * _PayNM>* pld_uram1,
    ap_uint<_WPay * _PayNM>* pld_uram2) {
#pragma HLS inline off

    enum { depth = (1 << (_WHash - 3)) }; // depth of URAM

    ap_uint<_WPay * _PayNM> initial_value;
    if (op == enums::AOP_MIN) {
        initial_value = ap_uint<_WPay * _PayNM>(-1);
    } else if (op == enums::AOP_MAX) {
        initial_value = ap_uint<_WPay * _PayNM>(0);
    } else if (op == enums::AOP_SUM || op == enums::AOP_COUNT || op == enums::AOP_COUNTNONZEROS ||
               op == enums::AOP_MEAN) {
        initial_value = 0;
    } else {
        // not supported yet
        initial_value = 0;
    }

    for (int i = 0; i < depth; i++) {
#pragma HLS pipeline II = 1
        key_uram[i] = 0;
        pld_uram0[i] = initial_value;
        pld_uram1[i] = 0;
        pld_uram2[i] = 0;
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

#ifndef __SYNTHESIS__
#ifdef DEBUG_MISS
    if (i_key(_WKey - 1, 0) == 1881824) {
        std::cout << std::hex << "Update_key_elem: key=" << i_key(_WKey - 1, 0) << " not_find=" << not_find
                  << " col_cnt=" << new_col_status << " max_col=" << max_col << " write_success=" << write_success
                  << " overflow=" << o_write_success << std::endl;
    }
#endif
#endif
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
    ap_uint<_WKey * _KeyNM + _Wcnt>* key_uram,

    // to HBM and wait for process again
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> >& undokey_strm,
    hls::stream<COLUMN_DATA<_WPay, _PayNM> >& undopld_strm,
    hls::stream<bool>& o_e0_strm,

    // to update pld uram
    hls::stream<ap_uint<_WHash> >& o_hash_strm,
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
            o_pld_strm.write(pld_in);
            o_hash_strm.write(hash);
            o_e1_strm.write(false);
        } else {
            cnt++;
            undokey_strm.write(key_in);
            undopld_strm.write(pld_in);
            o_e0_strm.write(false);
        }

#ifndef __SYNTHESIS__
#ifdef DEBUG_UPDATE_KEY
        if (write_success) {
            std::cout << std::hex << "Update_key_uram: array_idx=" << arry_idx << " bit_idx=" << bit_idx
                      << " col_cnt=" << new_elem(_WKey * _KeyNM + _Wcnt - 1, _WKey * _KeyNM)
                      << " key=" << key_in.data[0] << " pld=" << pld_in.data[0] << " old_uram=" << elem
                      << " new_uram=" << new_elem << std::endl;
        } else {
            std::cout << std::hex << "Update_key_uram: array_idx=" << arry_idx << " key=" << key_in.data[0]
                      << " pld=" << pld_in.data[0] << " old_uram=" << elem << " overflow_col_cnt=" << cnt << std::endl;
        }
#endif

#ifdef DEBUG_MISS
        if (key_in.data[0] == 1881824) {
            std::cout << std::hex << "Update_key_uram: array_idx=" << arry_idx << " bit_idx=" << bit_idx
                      << " col_cnt=" << new_elem(_WKey * _KeyNM + _Wcnt - 1, _WKey * _KeyNM) << " unhandle_cnt=" << cnt
                      << " key=" << key << " old_uram=" << elem << " new_uram=" << new_elem << std::endl;
        }
#endif

#endif
    }
    unhandle_cnt = cnt;
    o_e0_strm.write(true);
    o_e1_strm.write(true);
}

// generate pld for different col number
template <int _WPay, int _PayNM, int _WHash, int _Wcnt>
void update_pld_elem(
    // in
    ap_uint<_WPay * _PayNM> i_pld,
    ap_uint<_WPay * _PayNM> i_pld_uram[3],

    // controlling param
    ap_uint<32> op_type,
    ap_uint<_Wcnt> max_col,
    ap_uint<3> bit_idx,

    // elem out
    ap_uint<_WPay * _PayNM> o_pld_uram[3]) {
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
    ap_uint<4> op_r[_PayNM];

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

    // generate input pld for aggregate
    for (int i = 0; i < _PayNM; i++) {
#pragma HLS unroll
        if (max_col == 1) {
            pld[i] = i_pld(_WPay - 1, 0);
        } else if (max_col == 2) {
            pld[i] = i_pld(((i % 2) + 1) * _WPay - 1, (i % 2) * _WPay);
        } else if (max_col > 2 && max_col <= 4) {
            pld[i] = i_pld((i % 4 + 1) * _WPay - 1, i % 4 * _WPay);
        } else {
            pld[i] = i_pld((i + 1) * _WPay - 1, i * _WPay);
        }

        op[i] = op_type((i + 1) * 4 - 1, i * 4);
    }

    // calculate aggr
    for (int i = 0; i < _PayNM; i++) {
#pragma HLS unroll

        if (max_col == 1) {
            op_r[i] = op[0];
        } else if (max_col == 2) {
            op_r[i] = op[i % 2];
        } else if (max_col > 2 && max_col <= 4) {
            op_r[i] = op[i % 4];
        } else {
            op_r[i] = op[i];
        }

        bool enable = pld_status[i] == 1;
        col_agg[i][0] = i_pld_uram[0]((i + 1) * _WPay - 1, i * _WPay);
        col_agg[i][1] = i_pld_uram[1]((i + 1) * _WPay - 1, i * _WPay);
        col_agg[i][2] = i_pld_uram[2]((i + 1) * _WPay - 1, i * _WPay);

        // col0: min-max cnt-cnt_nz default:max
        // col1: sum_l avg_l default:sum
        // col2: sum_h avg_h default:sum

        if (op_r[i] == enums::AOP_COUNT || op_r[i] == enums::AOP_COUNTNONZEROS || op_r[i] == enums::AOP_SUM ||
            op_r[i] == enums::AOP_MEAN) {
            // cnt-cnt_nz
            new_col_agg[i][0] = aggr_cnt_nz(pld[i], col_agg[i][0], op_r[i] == enums::AOP_COUNTNONZEROS, enable);
        } else {
            // min-max
            new_col_agg[i][0] = aggr_min_max(pld[i], col_agg[i][0], op_r[i] == enums::AOP_MIN, enable);
        }

        // sum
        ap_uint<2 * _WPay> accum_old = (col_agg[i][2], col_agg[i][1]);
        ap_uint<2 * _WPay> accum_new;
        aggr_sum(pld[i], accum_old, accum_new, enable);
        new_col_agg[i][1] = accum_new(_WPay - 1, 0);
        new_col_agg[i][2] = accum_new(2 * _WPay - 1, _WPay);

#ifndef __SYNTHESIS__
#ifdef DEBUG_UPDATE_PLD
        if (i == 0) {
            std::cout << std::hex << "Update_pld_elem: i=" << i << " enable=" << enable << " op_r=" << op_r[i]
                      << " pld_status=" << pld_status << " pld_col=" << max_col << std::endl;
            std::cout << std::hex << "Update_pld_elem: accum_old=" << accum_old << " accum_new=" << accum_new
                      << std::endl;
            std::cout << std::hex << "Update_pld_elem: col0_old=" << col_agg[i][0] << " col0_new=" << new_col_agg[i][0]
                      << std::endl;
        }
#endif
#endif

        o_pld_uram[0]((i + 1) * _WPay - 1, i * _WPay) = new_col_agg[i][0];
        o_pld_uram[1]((i + 1) * _WPay - 1, i * _WPay) = new_col_agg[i][1];
        o_pld_uram[2]((i + 1) * _WPay - 1, i * _WPay) = new_col_agg[i][2];
    }

#ifndef __SYNTHESIS__
#ifdef DEBUG_UPDATE_PLD

    std::cout << std::hex << "Update_pld_elem: i_pld=" << i_pld << std::endl;
    std::cout << std::hex << "Update_pld_elem: old_pld[0]=" << i_pld_uram[0] << " new_pld[0]=" << o_pld_uram[0]
              << std::endl;
    std::cout << std::hex << "Update_pld_elem: old_pld[1]=" << i_pld_uram[1] << " new_pld[1]=" << o_pld_uram[1]
              << std::endl;
    std::cout << std::hex << "Update_pld_elem: old_pld[2]=" << i_pld_uram[2] << " new_pld[2]=" << o_pld_uram[2] << "\n"
              << std::endl;

#endif
#endif
}

/// @brief update pld uram to calculate aggregate
template <int _WPay, int _PayNM, int _WHash, int _Wcnt>
void update_pld_uram(
    // stream in
    hls::stream<ap_uint<_WHash> >& i_hash_strm,
    hls::stream<COLUMN_DATA<_WPay, _PayNM> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    // controlling param
    ap_uint<32> op,
    ap_uint<_Wcnt> pld_column,

    // uram
    ap_uint<_WPay * _PayNM>* pld_uram0, // min, cnt
    ap_uint<_WPay * _PayNM>* pld_uram1, // max, sum_l
    ap_uint<_WPay * _PayNM>* pld_uram2  // cnt_nz, sum_h
    ) {
#pragma HLS inline off

    ap_uint<_WPay * _PayNM> elem[3];
    ap_uint<_WPay * _PayNM> new_elem[3];
    ap_uint<_WPay * _PayNM> elem_temp0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    ap_uint<_WPay * _PayNM> elem_temp1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    ap_uint<_WPay * _PayNM> elem_temp2[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    ap_uint<32> arry_idx_temp[8] = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                                    0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
    COLUMN_DATA<_WPay, _PayNM> pld_in;
    ap_uint<_WPay * _PayNM> pld;

#pragma HLS array_partition variable = elem complete
#pragma HLS array_partition variable = new_elem complete
#pragma HLS array_partition variable = elem_temp0 complete
#pragma HLS array_partition variable = elem_temp1 complete
#pragma HLS array_partition variable = elem_temp2 complete
#pragma HLS array_partition variable = arry_idx_temp complete
#pragma HLS array_partition variable = pld_in complete
    bool last = i_e_strm.read();
update_pld_loop:
    while (!last) {
#pragma HLS DEPENDENCE variable = pld_uram0 pointer inter false
#pragma HLS DEPENDENCE variable = pld_uram1 pointer inter false
#pragma HLS DEPENDENCE variable = pld_uram2 pointer inter false

#pragma HLS pipeline II = 1
#pragma HLS LATENCY min = 8

        ap_uint<_WHash> hash = i_hash_strm.read();
        ap_uint<_WHash - 3> arry_idx = hash(_WHash - 4, 0);
        ap_uint<3> bit_idx = hash(_WHash - 1, _WHash - 3);
        pld_in = i_pld_strm.read();
        for (int i = 0; i < _PayNM; i++) {
#pragma HLS unroll
            pld((i + 1) * _WPay - 1, i * _WPay) = pld_in.data[i];
        }
        last = i_e_strm.read();

        // read temp, prevent duplicate idx
        if (arry_idx == arry_idx_temp[0]) {
            elem[0] = elem_temp0[0];
            elem[1] = elem_temp1[0];
            elem[2] = elem_temp2[0];
        } else if (arry_idx == arry_idx_temp[1]) {
            elem[0] = elem_temp0[1];
            elem[1] = elem_temp1[1];
            elem[2] = elem_temp2[1];
        } else if (arry_idx == arry_idx_temp[2]) {
            elem[0] = elem_temp0[2];
            elem[1] = elem_temp1[2];
            elem[2] = elem_temp2[2];
        } else if (arry_idx == arry_idx_temp[3]) {
            elem[0] = elem_temp0[3];
            elem[1] = elem_temp1[3];
            elem[2] = elem_temp2[3];
        } else if (arry_idx == arry_idx_temp[4]) {
            elem[0] = elem_temp0[4];
            elem[1] = elem_temp1[4];
            elem[2] = elem_temp2[4];
        } else if (arry_idx == arry_idx_temp[5]) {
            elem[0] = elem_temp0[5];
            elem[1] = elem_temp1[5];
            elem[2] = elem_temp2[5];
        } else if (arry_idx == arry_idx_temp[6]) {
            elem[0] = elem_temp0[6];
            elem[1] = elem_temp1[6];
            elem[2] = elem_temp2[6];
        } else if (arry_idx == arry_idx_temp[7]) {
            elem[0] = elem_temp0[7];
            elem[1] = elem_temp1[7];
            elem[2] = elem_temp2[7];
        } else {
            elem[0] = pld_uram0[arry_idx];
            elem[1] = pld_uram1[arry_idx];
            elem[2] = pld_uram2[arry_idx];
        }

#ifndef __SYNTHESIS__
#ifdef DEBUG_UPDATE_PLD

        std::cout << std::hex << "Update_pld_uram: array_idx=" << arry_idx << " bit_idx=" << bit_idx << std::endl;

#endif
#endif

        // calculate aggregate result && cope with different column size
        update_pld_elem<_WPay, _PayNM, _WHash>(pld, elem, op, pld_column, bit_idx, new_elem);

        // write uram
        pld_uram0[arry_idx] = new_elem[0];
        pld_uram1[arry_idx] = new_elem[1];
        pld_uram2[arry_idx] = new_elem[2];

        // right shift temp
        for (int i = 7; i > 0; i--) {
            elem_temp0[i] = elem_temp0[i - 1];
            elem_temp1[i] = elem_temp1[i - 1];
            elem_temp2[i] = elem_temp2[i - 1];
            arry_idx_temp[i] = arry_idx_temp[i - 1];
        }
        elem_temp0[0] = new_elem[0];
        elem_temp1[0] = new_elem[1];
        elem_temp2[0] = new_elem[2];
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
    hls::stream<COLUMN_DATA<_WPay, _PayNM> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    // controling param
    ap_uint<32> op,
    ap_uint<32> key_column,
    ap_uint<32> pld_column,
    ap_uint<32> round,
    ap_uint<32>& unhandle_cnt,

    // uram
    ap_uint<_WKey * _KeyNM + _Wcnt>* key_uram,
    ap_uint<_WPay * _PayNM>* pld_uram0,
    ap_uint<_WPay * _PayNM>* pld_uram1,
    ap_uint<_WPay * _PayNM>* pld_uram2,

    // buffer
    ap_uint<_WBuffer>* out_buf) {
#pragma HLS dataflow

    // streams between update_key_uram and stream_to_buf
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> > key0_strm;
#pragma HLS bind_storage variable = key0_strm type = fifo impl = srl
#pragma HLS STREAM variable = key0_strm depth = 8
    hls::stream<COLUMN_DATA<_WPay, _PayNM> > pld0_strm;
#pragma HLS bind_storage variable = pld0_strm type = fifo impl = srl
#pragma HLS STREAM variable = pld0_strm depth = 8
    hls::stream<bool> e0_strm;
#pragma HLS bind_storage variable = e0_strm type = fifo impl = srl
#pragma HLS STREAM variable = e0_strm depth = 8

    // streams between update_key_uram and update_pld_uram
    hls::stream<ap_uint<_WHash> > hash_strm;
#pragma HLS bind_storage variable = hash_strm type = fifo impl = srl
#pragma HLS STREAM variable = hash_strm depth = 8
    hls::stream<COLUMN_DATA<_WPay, _PayNM> > pld1_strm;
#pragma HLS bind_storage variable = pld1_strm type = fifo impl = srl
#pragma HLS STREAM variable = pld1_strm depth = 8
    hls::stream<bool> e1_strm;
#pragma HLS bind_storage variable = e1_strm type = fifo impl = srl
#pragma HLS STREAM variable = e1_strm depth = 8

    ap_uint<_Wcnt> max_col = key_column > pld_column ? key_column : pld_column;

#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
    std::cout << "-------------------update key uram----------------------" << std::endl;
#endif
#endif

    update_key_uram<_WKey, _KeyNM, _WPay, _PayNM, _WHash, _Wcnt>(i_hash_strm, i_key_strm, i_pld_strm, i_e_strm,

                                                                 max_col, unhandle_cnt, key_uram,

                                                                 key0_strm, pld0_strm, e0_strm,

                                                                 hash_strm, pld1_strm, e1_strm);

#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
    std::cout << "-------------------write out overflow-------------------" << std::endl;
#endif
#endif

    stream_to_buf<_WKey, _KeyNM, _WPay, _PayNM, _WBuffer, _BurstLenW>(key0_strm, pld0_strm, e0_strm,

                                                                      out_buf);

#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
    std::cout << "--------------------update pld uram---------------------" << std::endl;
#endif
#endif

    update_pld_uram<_WPay, _PayNM, _WHash, _Wcnt>(hash_strm, pld1_strm, e1_strm,

                                                  op, max_col, pld_uram0, pld_uram1, pld_uram2);
}

// read key from uram with different column number
template <int _WKey, int _KeyNM, int _PayNM, int _WHash, int _Wcnt>
void read_key_uram(
    // uram buffer
    ap_uint<_WKey * _KeyNM + _Wcnt>* key_uram,

    // control param
    ap_uint<_Wcnt> max_col,

    // to output
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> >& o_key_strm,
    hls::stream<ap_uint<_WHash> >& o_array_idx_strm,
    hls::stream<ap_uint<_Wcnt> >& o_cnt_strm,
    hls::stream<bool>& o_e_strm) {
#pragma HLS inline off

    enum { depth = (1 << (_WHash - 3)) }; // depth of URAM

    COLUMN_DATA<_WKey, _KeyNM> o_key_split;
#pragma HLS array_partition variable = o_key_split complete

loop_read_key_uram:
    for (int i = 0; i < depth; i++) {
#pragma HLS pipeline off

        ap_uint<_WKey* _KeyNM + _Wcnt> data = key_uram[i];
        ap_uint<_Wcnt> col_cnt = data(_WKey * _KeyNM + _Wcnt - 1, _WKey * _KeyNM);
        ap_uint<_WKey* _KeyNM> key = data(_WKey * _KeyNM - 1, 0);

        if (col_cnt > 0) {
            if (max_col == 1) {
            loop_column1:
                for (int j = 0; j < 8; j++) {
#pragma HLS pipeline II = 1
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 8

                    ap_uint<_WKey* _KeyNM> o_key = key(_WKey - 1, 0);
                    for (int k = 0; k < _KeyNM; k++) {
#pragma HLS unroll
                        o_key_split.data[k] = o_key((k + 1) * _WKey - 1, k * _WKey);
                    }

                    // write key
                    if (col_cnt[j] == 1) {
                        o_key_strm.write(o_key_split);

#ifndef __SYNTHESIS__
#ifdef DEBUG_READ_KEY
                        std::cout << "Read_key_uram: array_idx=" << i << " col_cnt=" << col_cnt
                                  << " o_key=" << o_key_split.data[0] << std::endl;
#endif
#endif
                    }

                    // left shift
                    for (int k = 0; k < _KeyNM - 1; k++) {
#pragma HLS unroll
                        key(_WKey * (k + 1) - 1, _WKey * k) = key(_WKey * (k + 2) - 1, _WKey * (k + 1));
                    }
                }
            } else if (max_col == 2) {
            loop_column2:
                for (int j = 0; j < 4; j++) {
#pragma HLS pipeline II = 1
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 4

                    ap_uint<_WKey* _KeyNM> o_key = key(_WKey * 2 - 1, 0);
                    for (int k = 0; k < _KeyNM; k++) {
#pragma HLS unroll
                        o_key_split.data[k] = o_key((k + 1) * _WKey - 1, k * _WKey);
                    }

                    // write key
                    if (col_cnt(2 * j + 1, 2 * j) != 0) {
                        o_key_strm.write(o_key_split);

#ifndef __SYNTHESIS__
#ifdef DEBUG_READ_KEY
                        std::cout << "Read_key_uram: array_idx=" << i << " col_cnt=" << col_cnt
                                  << " o_key=" << o_key_split.data[0] << std::endl;
#endif
#endif
                    }

                    // left shift
                    for (int k = 0; k < _KeyNM - 2; k++) {
#pragma HLS unroll
                        key(_WKey * (k + 1) - 1, _WKey * k) = key(_WKey * (k + 3) - 1, _WKey * (k + 2));
                    }
                }
            } else if (max_col > 2 && max_col <= 4) {
            loop_column4:
                for (int j = 0; j < 2; j++) {
#pragma HLS pipeline II = 1
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 2

                    ap_uint<_WKey* _KeyNM> o_key = key(_WKey * 4 - 1, 0);
                    for (int k = 0; k < _KeyNM; k++) {
#pragma HLS unroll
                        o_key_split.data[k] = o_key((k + 1) * _WKey - 1, k * _WKey);
                    }

                    // write key
                    if (col_cnt(4 * j + 3, 4 * j) != 0) {
                        o_key_strm.write(o_key_split);

#ifndef __SYNTHESIS__
#ifdef DEBUG_READ_KEY
                        std::cout << "Read_key_uram: array_idx=" << i << " col_cnt=" << col_cnt
                                  << " o_key=" << o_key_split.data[0] << std::endl;
#endif
#endif
                    }

                    // left shift
                    for (int k = 0; k < _KeyNM - 4; k++) {
#pragma HLS unroll
                        key(_WKey * (k + 1) - 1, _WKey * k) = key(_WKey * (k + 5) - 1, _WKey * (k + 4));
                    }
                }
            } else {
                // max_col > 4
                ap_uint<_WKey* _KeyNM> o_key = key;
                for (int k = 0; k < _KeyNM; k++) {
#pragma HLS unroll
                    o_key_split.data[k] = o_key((k + 1) * _WKey - 1, k * _WKey);
                }

                // write key
                if (col_cnt != 0) {
                    o_key_strm.write(o_key_split);

#ifndef __SYNTHESIS__
#ifdef DEBUG_READ_KEY
                    std::cout << "Read_key_uram: array_idx=" << i << " col_cnt=" << col_cnt
                              << " o_key=" << o_key_split.data[0] << std::endl;
#endif
#endif
                }
            }

            // write control param for read pld uram
            o_array_idx_strm.write(i);
            o_cnt_strm.write(col_cnt);
            o_e_strm.write(false);
        }
    }

    o_e_strm.write(true);
}

// padding pld data according to different column number
template <int _WPay, int _PayNM, int _WHash, int _Wcnt>
void read_pld_uram(
    // uram buffer
    ap_uint<_WPay * _PayNM>* pld_uram0,
    ap_uint<_WPay * _PayNM>* pld_uram1,
    ap_uint<_WPay * _PayNM>* pld_uram2,

    // control param
    ap_uint<_Wcnt> max_col,

    // stream in
    hls::stream<ap_uint<_WHash> >& i_array_idx_strm,
    hls::stream<ap_uint<_Wcnt> >& i_cnt_strm,
    hls::stream<bool>& i_e_strm,

    // to output
    hls::stream<COLUMN_DATA<_WPay, _PayNM> > o_pld_strm[3],
    hls::stream<bool>& o_e_strm) {
#pragma HLS inline off

    bool last = i_e_strm.read();

    ap_uint<_Wcnt> col_cnt;
    ap_uint<_WPay * _PayNM> i_pld0, i_pld1, i_pld2;

    ap_uint<_WPay> pld0[_PayNM];
#pragma HLS array_partition variable = pld0 complete
    ap_uint<_WPay> pld1[_PayNM];
#pragma HLS array_partition variable = pld1 complete
    ap_uint<_WPay> pld2[_PayNM];
#pragma HLS array_partition variable = pld2 complete

    COLUMN_DATA<_WPay, _PayNM> o_pld0;
#pragma HLS array_partition variable = o_pld0 complete
    COLUMN_DATA<_WPay, _PayNM> o_pld1;
#pragma HLS array_partition variable = o_pld1 complete
    COLUMN_DATA<_WPay, _PayNM> o_pld2;
#pragma HLS array_partition variable = o_pld2 complete

LOOP_PLD_READ:
    while (!last) {
#pragma HLS pipeline off

        ap_uint<_WHash> array_idx = i_array_idx_strm.read();
        ap_uint<_Wcnt> col_cnt = i_cnt_strm.read();
        last = i_e_strm.read();

        i_pld0 = pld_uram0[array_idx];
        i_pld1 = pld_uram1[array_idx];
        i_pld2 = pld_uram2[array_idx];

        for (int i = 0; i < _PayNM; i++) {
#pragma HLS unroll
            pld0[i] = i_pld0((i + 1) * _WPay - 1, i * _WPay);
            pld1[i] = i_pld1((i + 1) * _WPay - 1, i * _WPay);
            pld2[i] = i_pld2((i + 1) * _WPay - 1, i * _WPay);
        }

        if (max_col == 1) {
        column1_cnt:
            for (int i = 0; i < 8; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 8
#pragma HLS pipeline II = 1

                ap_uint<_WPay* _PayNM> pld_column0 = (0, pld0[0]);
                ap_uint<_WPay* _PayNM> pld_column1 = (0, pld1[0]);
                ap_uint<_WPay* _PayNM> pld_column2 = (0, pld2[0]);

                // left shift
                for (int j = 0; j < _PayNM - 1; j++) {
#pragma HLS unroll
                    pld0[j] = pld0[j + 1];
                    pld1[j] = pld1[j + 1];
                    pld2[j] = pld2[j + 1];
                }

                // assign output
                for (int j = 0; j < _PayNM; j++) {
#pragma HLS unroll
                    o_pld0.data[j] = pld_column0((j + 1) * _WPay - 1, j * _WPay);
                    o_pld1.data[j] = pld_column1((j + 1) * _WPay - 1, j * _WPay);
                    o_pld2.data[j] = pld_column2((j + 1) * _WPay - 1, j * _WPay);
                }

                // control output
                if (col_cnt[i] == 1) {
                    o_pld_strm[0].write(o_pld0);
                    o_pld_strm[1].write(o_pld1);
                    o_pld_strm[2].write(o_pld2);
                    o_e_strm.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG_READ_PLD
                    std::cout << "Read_pld_uram: array_idx=" << array_idx << " col_cnt=" << col_cnt << std::endl;
                    std::cout << std::hex << "Read_pld_uram: o_pld0=" << pld_column0 << std::endl;
                    std::cout << std::hex << "Read_pld_uram: o_pld1=" << pld_column1 << std::endl;
                    std::cout << std::hex << "Read_pld_uram: o_pld2=" << pld_column2 << std::endl;
#endif
#endif
                }
            }
        } else if (max_col == 2) {
        column2_cnt:
            for (int i = 0; i < 4; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 4
#pragma HLS pipeline II = 1

                ap_uint<_WPay* _PayNM> pld_column0 = (0, pld0[1], pld0[0]);
                ap_uint<_WPay* _PayNM> pld_column1 = (0, pld1[1], pld1[0]);
                ap_uint<_WPay* _PayNM> pld_column2 = (0, pld2[1], pld2[0]);

                // left shift
                for (int j = 0; j < _PayNM - 2; j++) {
#pragma HLS unroll
                    pld0[j] = pld0[j + 2];
                    pld1[j] = pld1[j + 2];
                    pld2[j] = pld2[j + 2];
                }

                // assign output
                for (int j = 0; j < _PayNM; j++) {
#pragma HLS unroll
                    o_pld0.data[j] = pld_column0((j + 1) * _WPay - 1, j * _WPay);
                    o_pld1.data[j] = pld_column1((j + 1) * _WPay - 1, j * _WPay);
                    o_pld2.data[j] = pld_column2((j + 1) * _WPay - 1, j * _WPay);
                }

                // control output
                if (col_cnt(2 * i + 1, 2 * i) != 0) {
                    o_pld_strm[0].write(o_pld0);
                    o_pld_strm[1].write(o_pld1);
                    o_pld_strm[2].write(o_pld2);
                    o_e_strm.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG_READ_PLD
                    std::cout << "Read_pld_uram: array_idx=" << array_idx << " col_cnt=" << col_cnt << std::endl;
                    std::cout << std::hex << "Read_pld_uram: o_pld0=" << pld_column0 << std::endl;
                    std::cout << std::hex << "Read_pld_uram: o_pld1=" << pld_column1 << std::endl;
                    std::cout << std::hex << "Read_pld_uram: o_pld2=" << pld_column2 << std::endl;
#endif
#endif
                }
            }
        } else if (max_col > 2 && max_col <= 4) {
        column4_cnt:
            for (int i = 0; i < 2; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 2
#pragma HLS pipeline II = 1

                ap_uint<_WPay* _PayNM> pld_column0 = (0, pld0[3], pld0[2], pld0[1], pld0[0]);
                ap_uint<_WPay* _PayNM> pld_column1 = (0, pld1[3], pld1[2], pld1[1], pld1[0]);
                ap_uint<_WPay* _PayNM> pld_column2 = (0, pld2[3], pld2[2], pld2[1], pld2[0]);

                // left shift
                for (int j = 0; j < _PayNM - 4; j++) {
#pragma HLS unroll
                    pld0[j] = pld0[j + 4];
                    pld1[j] = pld1[j + 4];
                    pld2[j] = pld2[j + 4];
                }

                // assign output
                for (int j = 0; j < _PayNM; j++) {
#pragma HLS unroll
                    o_pld0.data[j] = pld_column0((j + 1) * _WPay - 1, j * _WPay);
                    o_pld1.data[j] = pld_column1((j + 1) * _WPay - 1, j * _WPay);
                    o_pld2.data[j] = pld_column2((j + 1) * _WPay - 1, j * _WPay);
                }

                // control output
                if (col_cnt(4 * i + 3, 4 * i) != 0) {
                    o_pld_strm[0].write(o_pld0);
                    o_pld_strm[1].write(o_pld1);
                    o_pld_strm[2].write(o_pld2);
                    o_e_strm.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG_READ_PLD
                    std::cout << "Read_pld_uram: array_idx=" << array_idx << " col_cnt=" << col_cnt << std::endl;
                    std::cout << std::hex << "Read_pld_uram: o_pld0=" << pld_column0 << std::endl;
                    std::cout << std::hex << "Read_pld_uram: o_pld1=" << pld_column1 << std::endl;
                    std::cout << std::hex << "Read_pld_uram: o_pld2=" << pld_column2 << std::endl;
#endif
#endif
                }
            }
        } else {
            // max_col >4

            // assign output
            for (int i = 0; i < _PayNM; i++) {
#pragma HLS unroll
                o_pld0.data[i] = i_pld0((i + 1) * _WPay - 1, i * _WPay);
                o_pld1.data[i] = i_pld1((i + 1) * _WPay - 1, i * _WPay);
                o_pld2.data[i] = i_pld2((i + 1) * _WPay - 1, i * _WPay);
            }

            // control output
            if (col_cnt != 0) {
                o_pld_strm[0].write(o_pld0);
                o_pld_strm[1].write(o_pld1);
                o_pld_strm[2].write(o_pld2);
                o_e_strm.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG_READ_PLD
                std::cout << "Read_pld_uram: array_idx=" << array_idx << " col_cnt=" << col_cnt << std::endl;
                std::cout << std::hex << "Read_pld_uram: o_pld0=" << i_pld0 << std::endl;
                std::cout << std::hex << "Read_pld_uram: o_pld1=" << i_pld1 << std::endl;
                std::cout << std::hex << "Read_pld_uram: o_pld2=" << i_pld2 << std::endl;
#endif
#endif
            }
        }

#ifndef __SYNTHESIS__
#ifdef DEBUG_READ_PLD
        std::cout << std::hex << "Read_pld_uram: i_pld0=" << i_pld0 << std::endl;
        std::cout << std::hex << "Read_pld_uram: i_pld1=" << i_pld1 << std::endl;
        std::cout << std::hex << "Read_pld_uram: i_pld2=" << i_pld2 << std::endl;
#endif
#endif
    }

    o_e_strm.write(true);
}

/// @brief output aggregate result stored in URAM
template <int _WKey, int _KeyNM, int _WPay, int _PayNM, int _WHash, int _Wcnt>
void uram_to_stream(
    // uram buffer
    ap_uint<_WKey * _KeyNM + _Wcnt>* key_uram,
    ap_uint<_WPay * _PayNM>* pld_uram0,
    ap_uint<_WPay * _PayNM>* pld_uram1,
    ap_uint<_WPay * _PayNM>* pld_uram2,

    // controling param
    ap_uint<32> key_column,
    ap_uint<32> pld_column,

    // to output
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> >& aggr_key_out,
    hls::stream<COLUMN_DATA<_WPay, _PayNM> > aggr_pld_out[3],
    hls::stream<bool>& o_e_strm) {
#pragma HLS dataflow

    hls::stream<ap_uint<_WHash> > array_idx_strm("uram_to_stream1");
#pragma HLS bind_storage variable = array_idx_strm type = fifo impl = srl
#pragma HLS STREAM variable = array_idx_strm depth = 8
    hls::stream<ap_uint<_Wcnt> > i_cnt_strm("uram_to_stream2");
#pragma HLS bind_storage variable = i_cnt_strm type = fifo impl = srl
#pragma HLS STREAM variable = i_cnt_strm depth = 8
    hls::stream<bool> e0_strm("uram_to_stream3");
#pragma HLS bind_storage variable = e0_strm type = fifo impl = srl
#pragma HLS STREAM variable = e0_strm depth = 8

    ap_uint<_Wcnt> max_col = key_column > pld_column ? key_column : pld_column;

#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
    std::cout << "------------------write out key uram--------------------" << std::endl;
#endif
#endif

    read_key_uram<_WKey, _KeyNM, _PayNM, _WHash, _Wcnt>(key_uram, max_col, aggr_key_out, array_idx_strm, i_cnt_strm,
                                                        e0_strm);

#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
    std::cout << "------------------write out pld uram--------------------" << std::endl;
#endif
#endif

    read_pld_uram<_WPay, _PayNM, _WHash, _Wcnt>(pld_uram0, pld_uram1, pld_uram2, max_col, array_idx_strm, i_cnt_strm,
                                                e0_strm, aggr_pld_out, o_e_strm);
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
    hls::stream<COLUMN_DATA<_WPay, _PayNM> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    // out buffer for undo strm
    ap_uint<_WBuffer>* out_buf,

    // controling param
    ap_uint<32> op,
    ap_uint<32> key_column,
    ap_uint<32> pld_column,
    ap_uint<32> round,
    ap_uint<32>& unhandle_cnt,

    // output result
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> >& aggr_key_out,
    hls::stream<COLUMN_DATA<_WPay, _PayNM> > aggr_pld_out[3],
    hls::stream<bool>& o_e_strm) {
#pragma HLS inline off

#ifndef __SYNTHESIS__
    std::cout << std::hex << "PU_wrapper: op=" << op << " key_column=" << key_column << " pld_column=" << pld_column
              << " round=" << round << " unhandle_cnt=" << unhandle_cnt << std::endl;
#endif

    enum { depth = (1 << (_WHash - 3)) }; // depth of URAM

// define URAM structure
#ifndef __SYNTHESIS__

    ap_uint<_WKey * _KeyNM + _Wcnt>* key_uram;
    ap_uint<_WPay * _PayNM>* pld_uram0;
    ap_uint<_WPay * _PayNM>* pld_uram1;
    ap_uint<_WPay * _PayNM>* pld_uram2;

    key_uram = (ap_uint<_WKey * _KeyNM + _Wcnt>*)malloc(depth * sizeof(ap_uint<_WKey * _KeyNM + _Wcnt>));
    pld_uram0 = (ap_uint<_WPay * _PayNM>*)malloc(depth * sizeof(ap_uint<_WPay * _PayNM>));
    pld_uram1 = (ap_uint<_WPay * _PayNM>*)malloc(depth * sizeof(ap_uint<_WPay * _PayNM>));
    pld_uram2 = (ap_uint<_WPay * _PayNM>*)malloc(depth * sizeof(ap_uint<_WPay * _PayNM>));

#else

    ap_uint<_WKey * _KeyNM + _Wcnt> key_uram[depth];
#pragma HLS bind_storage variable = key_uram type = ram_2p impl = uram
#pragma HLS ARRAY_PARTITION variable = key_uram block factor = 4
    ap_uint<_WPay * _PayNM> pld_uram0[depth];
#pragma HLS bind_storage variable = pld_uram0 type = ram_2p impl = uram
#pragma HLS ARRAY_PARTITION variable = pld_uram0 block factor = 4
    ap_uint<_WPay * _PayNM> pld_uram1[depth];
#pragma HLS bind_storage variable = pld_uram1 type = ram_2p impl = uram
#pragma HLS ARRAY_PARTITION variable = pld_uram1 block factor = 4
    ap_uint<_WPay * _PayNM> pld_uram2[depth];
#pragma HLS bind_storage variable = pld_uram2 type = ram_2p impl = uram
#pragma HLS ARRAY_PARTITION variable = pld_uram2 block factor = 4

#endif

#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
    std::cout << "-------------------initialize uram----------------------" << std::endl;
#endif
#endif

    initial_uram<_WKey, _KeyNM, _WPay, _PayNM, _WHash, _Wcnt>(op, key_uram, pld_uram0, pld_uram1, pld_uram2);

#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
    std::cout << "----------------------update uram-----------------------" << std::endl;
#endif
#endif

    update_uram<_HashMode, _WHash, _WKey, _KeyNM, _WPay, _PayNM, _Wcnt, _WBuffer, _BurstLenW>(
        i_hash_strm, i_key_strm, i_pld_strm, i_e_strm, op, key_column, pld_column, round, unhandle_cnt, key_uram,
        pld_uram0, pld_uram1, pld_uram2, out_buf);

#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
    std::cout << "-------------------write out overflow--------------------" << std::endl;
#endif
#endif

    uram_to_stream<_WKey, _KeyNM, _WPay, _PayNM, _WHash, _Wcnt>(key_uram, pld_uram0, pld_uram1, pld_uram2, key_column,
                                                                pld_column, aggr_key_out, aggr_pld_out, o_e_strm);

    // for do-while in collect
    COLUMN_DATA<_WKey, _KeyNM> pad_key;
    COLUMN_DATA<_WPay, _PayNM> pad_pld;

    aggr_key_out.write(pad_key);
    aggr_pld_out[0].write(pad_pld);
    aggr_pld_out[1].write(pad_pld);
    aggr_pld_out[2].write(pad_pld);

#ifndef __SYNTHESIS__

    free(key_uram);
    free(pld_uram0);
    free(pld_uram1);
    free(pld_uram2);

#endif
}

//--------------------------------collect----------------------------------

// collect join result of multiple PU
template <int PU, int WKey, int KeyNM, int WPay, int PayNM>
void collect_unit(hls::stream<COLUMN_DATA<WKey, KeyNM> > i_key_strm[PU],
                  hls::stream<COLUMN_DATA<WPay, PayNM> > i_pay_strm[PU][3],
                  hls::stream<bool> i_e_strm[PU],

                  ap_uint<32>& aggregate_num,

                  hls::stream<COLUMN_DATA<WKey, KeyNM> >& o_key_strm,
                  hls::stream<COLUMN_DATA<WPay, PayNM> > o_pay_strm[3],
                  hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    const int MAX = (1 << PU) - 1;

    COLUMN_DATA<WKey, KeyNM> key_arry[PU];
#pragma HLS array_partition variable = key_arry complete dim = 0
    COLUMN_DATA<WPay, PayNM> pay_arry[PU][3];
#pragma HLS array_partition variable = pay_arry complete dim = 0
#pragma HLS array_partition variable = pay_arry complete dim = 1
    COLUMN_DATA<WKey, KeyNM> o_key;
#pragma HLS array_partition variable = o_key complete
    COLUMN_DATA<WPay, PayNM> o_pay[3];
#pragma HLS array_partition variable = o_pay complete dim = 0

    ap_uint<32> cnt = 0;
    ap_uint<PU> empty_e = 0;
    ap_uint<PU> last = 0;
    ap_uint<PU> rd_e = 0;

#ifndef __SYNTHESIS__
    std::cout << std::dec << "PU=" << PU << std::endl;
#endif

    do {
#pragma HLS pipeline II = 1
        for (int i = 0; i < PU; i++) {
#pragma HLS unroll
            empty_e[i] = !i_e_strm[i].empty() && !last[i];
        }

        rd_e = details::join_v2::mul_ch_read(empty_e);

        for (int i = 0; i < PU; i++) {
#pragma HLS unroll

            if (rd_e[i]) {
                key_arry[i] = i_key_strm[i].read();

                pay_arry[i][0] = i_pay_strm[i][0].read();
                pay_arry[i][1] = i_pay_strm[i][1].read();
                pay_arry[i][2] = i_pay_strm[i][2].read();

                last[i] = i_e_strm[i].read();
            }
        }

        ap_uint<3> id = details::join_v2::mux<PU>(rd_e);
        o_key = key_arry[id];
        o_pay[0] = pay_arry[id][0];
        o_pay[1] = pay_arry[id][1];
        o_pay[2] = pay_arry[id][2];

        bool valid_n = last[id];
        if (!valid_n && rd_e != 0) {
            o_key_strm.write(o_key);
            o_pay_strm[0].write(o_pay[0]);
            o_pay_strm[1].write(o_pay[1]);
            o_pay_strm[2].write(o_pay[2]);
            o_e_strm.write(false);

            cnt++;
        }
    } while (last != MAX);

#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
    std::cout << std::dec << "Collect " << cnt << " rows" << std::endl;
#endif
#endif

    aggregate_num += cnt;
    o_e_strm.write(true);
}

//-----------------------------aggr mean------------------------------
template <int _WKey, int _KeyNM, int _WPay, int _PayNM, int MeanColNM>
void calculate_aggr_mean(hls::stream<COLUMN_DATA<_WKey, _KeyNM> >& i_key_strm,
                         hls::stream<COLUMN_DATA<_WPay, _PayNM> > i_pld_strm[3],
                         hls::stream<bool>& i_e_strm,

                         ap_uint<32> op_type,

                         hls::stream<ap_uint<_WKey> > o_key_strm[_KeyNM],
                         hls::stream<ap_uint<_WPay> > o_pld_strm[3][_PayNM],
                         hls::stream<bool>& o_e_strm) {
#ifndef __SYNTHESIS__
    ap_uint<64> cnt = 0;
#endif

    ap_uint<4> op[_PayNM];
#pragma HLS array_partition variable = op complete
    COLUMN_DATA<_WKey, _KeyNM> i_key;
#pragma HLS array_partition variable = i_key complete
    COLUMN_DATA<_WPay, _PayNM> i_pld[3];
#pragma HLS array_partition variable = i_pld complete
    ap_uint<_WPay> i_pld_col[3][_PayNM];
#pragma HLS array_partition variable = i_pld_col complete dim = 0
#pragma HLS array_partition variable = i_pld_col complete dim = 1
    ap_uint<_WPay> o_pld_col[3][_PayNM];
#pragma HLS array_partition variable = o_pld_col complete dim = 0
#pragma HLS array_partition variable = o_pld_col complete dim = 1

    bool last = i_e_strm.read();
    while (!last) {
#pragma HLS pipeline II = 1

        i_key = i_key_strm.read();
        i_pld[0] = i_pld_strm[0].read();
        i_pld[1] = i_pld_strm[1].read();
        i_pld[2] = i_pld_strm[2].read();

        for (int i = 0; i < _KeyNM; i++) {
#pragma HLS unroll
            o_key_strm[i].write(i_key.data[i]);
        }

        for (int i = 0; i < _PayNM; i++) {
#pragma HLS unroll
            i_pld_col[0][i] = i_pld[0].data[i];
            i_pld_col[1][i] = i_pld[1].data[i];
            i_pld_col[2][i] = i_pld[2].data[i];
            op[i] = op_type((i + 1) * 4 - 1, 4 * i);

            if (op[i] == enums::AOP_MEAN && i < MeanColNM) {
                ap_uint<_WPay> cnt = i_pld_col[0][i];
                ap_uint<2 * _WPay> sum = (i_pld_col[2][i], i_pld_col[1][i]);
                ap_uint<2 * _WPay> ave;

                aggr_mean<_WPay>(sum, cnt, ave);

                o_pld_col[0][i] = cnt;
                o_pld_col[1][i] = ave(_WPay - 1, 0);
                o_pld_col[2][i] = ave(2 * _WPay - 1, _WPay);
            } else {
                o_pld_col[0][i] = i_pld_col[0][i];
                o_pld_col[1][i] = i_pld_col[1][i];
                o_pld_col[2][i] = i_pld_col[2][i];
            }

            o_pld_strm[0][i].write(o_pld_col[0][i]);
            o_pld_strm[1][i].write(o_pld_col[1][i]);
            o_pld_strm[2][i].write(o_pld_col[2][i]);
        }
        last = i_e_strm.read();
        o_e_strm.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG_RESULT
        if (cnt < 10) {
            for (int i = 0; i < _KeyNM; i++) {
                std::cout << std::dec << "aggr_result: col" << i << " key=" << i_key.data[i] << std::endl;
            }
            for (int i = 0; i < _PayNM; i++) {
                std::cout << std::dec << "aggr_result: col" << i << " pld0=" << o_pld_col[0][i]
                          << " pld1=" << o_pld_col[1][i] << " pld2=" << o_pld_col[2][i] << std::endl;
            }
            std::cout << std::dec << "cnt=" << cnt++ << std::endl;
        }
#endif
#endif
    }
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
    hls::stream<ap_uint<_WKey> > strm_key_in[_CHNM][_KeyNM],
    hls::stream<ap_uint<_WPay> > strm_pld_in[_CHNM][_PayNM],
    hls::stream<bool> strm_e_in[_CHNM],

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
    ap_uint<_WBuffer>* in_buf1,
    ap_uint<_WBuffer>* in_buf2,
    ap_uint<_WBuffer>* in_buf3,

    ap_uint<_WBuffer>* out_buf0,
    ap_uint<_WBuffer>* out_buf1,
    ap_uint<_WBuffer>* out_buf2,
    ap_uint<_WBuffer>* out_buf3,

    // stream out
    hls::stream<ap_uint<_WKey> > aggr_key_out[_KeyNM],
    hls::stream<ap_uint<_WPay> > aggr_pld_out[3][_PayNM],
    hls::stream<bool>& strm_e_out) {
#pragma HLS inline off
#pragma HLS DATAFLOW

    enum { PU = (1 << _WHashHigh) }; // high hash for distribution.

    // dispatch
    // Channel1
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> > k1_strm_arry_c0[PU];
#pragma HLS stream variable = k1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c0 complete
#pragma HLS bind_storage variable = k1_strm_arry_c0 type = fifo impl = srl
    hls::stream<COLUMN_DATA<_WPay, _PayNM> > p1_strm_arry_c0[PU];
#pragma HLS stream variable = p1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c0 complete
#pragma HLS bind_storage variable = p1_strm_arry_c0 type = fifo impl = srl
    hls::stream<ap_uint<_WHashLow> > hash_strm_arry_c0[PU];
#pragma HLS stream variable = hash_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c0 complete
#pragma HLS bind_storage variable = hash_strm_arry_c0 type = fifo impl = srl
    hls::stream<bool> e1_strm_arry_c0[PU];
#pragma HLS stream variable = e1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c0 complete

    // Channel2
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> > k1_strm_arry_c1[PU];
#pragma HLS stream variable = k1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c1 complete
#pragma HLS bind_storage variable = k1_strm_arry_c1 type = fifo impl = srl
    hls::stream<COLUMN_DATA<_WPay, _PayNM> > p1_strm_arry_c1[PU];
#pragma HLS stream variable = p1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c1 complete
#pragma HLS bind_storage variable = p1_strm_arry_c1 type = fifo impl = srl
    hls::stream<ap_uint<_WHashLow> > hash_strm_arry_c1[PU];
#pragma HLS stream variable = hash_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c1 complete
#pragma HLS bind_storage variable = hash_strm_arry_c1 type = fifo impl = srl
    hls::stream<bool> e1_strm_arry_c1[PU];
#pragma HLS stream variable = e1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c1 complete

    // Channel3
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> > k1_strm_arry_c2[PU];
#pragma HLS stream variable = k1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c2 complete
#pragma HLS bind_storage variable = k1_strm_arry_c2 type = fifo impl = srl
    hls::stream<COLUMN_DATA<_WPay, _PayNM> > p1_strm_arry_c2[PU];
#pragma HLS stream variable = p1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c2 complete
#pragma HLS bind_storage variable = p1_strm_arry_c2 type = fifo impl = srl
    hls::stream<ap_uint<_WHashLow> > hash_strm_arry_c2[PU];
#pragma HLS stream variable = hash_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c2 complete
#pragma HLS bind_storage variable = hash_strm_arry_c2 type = fifo impl = srl
    hls::stream<bool> e1_strm_arry_c2[PU];
#pragma HLS stream variable = e1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c2 complete

    // Channel4
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> > k1_strm_arry_c3[PU];
#pragma HLS stream variable = k1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c3 complete
#pragma HLS bind_storage variable = k1_strm_arry_c3 type = fifo impl = srl
    hls::stream<COLUMN_DATA<_WPay, _PayNM> > p1_strm_arry_c3[PU];
#pragma HLS stream variable = p1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c3 complete
#pragma HLS bind_storage variable = p1_strm_arry_c3 type = fifo impl = srl
    hls::stream<ap_uint<_WHashLow> > hash_strm_arry_c3[PU];
#pragma HLS stream variable = hash_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c3 complete
#pragma HLS bind_storage variable = hash_strm_arry_c3 type = fifo impl = srl
    hls::stream<bool> e1_strm_arry_c3[PU];
#pragma HLS stream variable = e1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c3 complete

    // merge channel1-channel4 to here, then mux HBM data as input of hash
    // aggregate PU
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> > k1_strm_arry[PU];
#pragma HLS stream variable = k1_strm_arry depth = 8
#pragma HLS array_partition variable = k1_strm_arry complete
#pragma HLS bind_storage variable = k1_strm_arry type = fifo impl = srl
    hls::stream<COLUMN_DATA<_WPay, _PayNM> > p1_strm_arry[PU];
#pragma HLS stream variable = p1_strm_arry depth = 8
#pragma HLS array_partition variable = p1_strm_arry complete
#pragma HLS bind_storage variable = p1_strm_arry type = fifo impl = srl
    hls::stream<ap_uint<_WHashLow> > hash1_strm_arry[PU];
#pragma HLS stream variable = hash1_strm_arry depth = 8
#pragma HLS array_partition variable = hash1_strm_arry complete
#pragma HLS bind_storage variable = hash1_strm_arry type = fifo impl = srl
    hls::stream<bool> e1_strm_arry[PU];
#pragma HLS stream variable = e1_strm_arry depth = 8
#pragma HLS array_partition variable = e1_strm_arry complete

    // output aggregate result of PU
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> > aggr_key_arry[PU];
#pragma HLS stream variable = aggr_key_arry depth = 64
#pragma HLS array_partition variable = aggr_key_arry complete
#pragma HLS bind_storage variable = aggr_key_arry type = fifo impl = srl
    hls::stream<COLUMN_DATA<_WPay, _PayNM> > aggr_pld_array[PU][3];
#pragma HLS stream variable = aggr_pld_array depth = 8
#pragma HLS array_partition variable = aggr_pld_array complete
#pragma HLS bind_storage variable = aggr_pld_array type = fifo impl = srl
    hls::stream<bool> e2_strm_arry[PU];
#pragma HLS stream variable = e2_strm_arry depth = 8
#pragma HLS array_partition variable = e2_strm_arry complete

    // output of collect pu
    hls::stream<COLUMN_DATA<_WKey, _KeyNM> > collect_key;
#pragma HLS stream variable = collect_key depth = 512
#pragma HLS bind_storage variable = collect_key type = fifo impl = bram
    hls::stream<COLUMN_DATA<_WPay, _PayNM> > collect_pld[3];
#pragma HLS stream variable = collect_pld depth = 512
#pragma HLS array_partition variable = collect_pld complete
#pragma HLS bind_storage variable = collect_pld type = fifo impl = bram
    hls::stream<bool> e3_strm;
#pragma HLS stream variable = e3_strm depth = 512

//---------------------------------dispatch PU-------------------------------
#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
    std::cout << "---------------------dispatch PU------------------------" << std::endl;
#endif
#endif

    if (_CHNM >= 1) {
        details::hash_group_aggregate::dispatch_wrapper<_HashMode, _WKey, _KeyNM, _WPay, _PayNM, _WHashHigh, _WHashLow,
                                                        PU, _WBuffer, _BurstLenR>(
            strm_key_in[0], strm_pld_in[0], strm_e_in[0], in_buf0, unhandle_cnt_r[0], round, k1_strm_arry_c0,
            p1_strm_arry_c0, hash_strm_arry_c0, e1_strm_arry_c0);
    }

    if (_CHNM >= 2) {
        details::hash_group_aggregate::dispatch_wrapper<_HashMode, _WKey, _KeyNM, _WPay, _PayNM, _WHashHigh, _WHashLow,
                                                        PU, _WBuffer, _BurstLenR>(
            strm_key_in[1], strm_pld_in[1], strm_e_in[1], in_buf1, unhandle_cnt_r[1], round, k1_strm_arry_c1,
            p1_strm_arry_c1, hash_strm_arry_c1, e1_strm_arry_c1);
    }

    if (_CHNM >= 4) {
        details::hash_group_aggregate::dispatch_wrapper<_HashMode, _WKey, _KeyNM, _WPay, _PayNM, _WHashHigh, _WHashLow,
                                                        PU, _WBuffer, _BurstLenR>(
            strm_key_in[2], strm_pld_in[2], strm_e_in[2], in_buf2, unhandle_cnt_r[2], round, k1_strm_arry_c2,
            p1_strm_arry_c2, hash_strm_arry_c2, e1_strm_arry_c2);

        details::hash_group_aggregate::dispatch_wrapper<_HashMode, _WKey, _KeyNM, _WPay, _PayNM, _WHashHigh, _WHashLow,
                                                        PU, _WBuffer, _BurstLenR>(
            strm_key_in[3], strm_pld_in[3], strm_e_in[3], in_buf3, unhandle_cnt_r[3], round, k1_strm_arry_c3,
            p1_strm_arry_c3, hash_strm_arry_c3, e1_strm_arry_c3);
    }

//---------------------------------merge PU---------------------------------

#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
    std::cout << "---------------------merge PU---------------------------" << std::endl;
#endif
#endif

    if (_CHNM == 1) {
        for (int p = 0; p < PU; ++p) {
#pragma HLS unroll
            details::hash_group_aggregate::merge1_1<_WKey, _KeyNM, _WPay, _PayNM, _WHashLow>(
                k1_strm_arry_c0[p], p1_strm_arry_c0[p], hash_strm_arry_c0[p], e1_strm_arry_c0[p], k1_strm_arry[p],
                p1_strm_arry[p], hash1_strm_arry[p], e1_strm_arry[p]);
        }
    } else if (_CHNM == 2) {
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            details::hash_group_aggregate::merge2_1<_WKey, _KeyNM, _WPay, _PayNM, _WHashLow>(
                k1_strm_arry_c0[p], k1_strm_arry_c1[p], p1_strm_arry_c0[p], p1_strm_arry_c1[p], hash_strm_arry_c0[p],
                hash_strm_arry_c1[p], e1_strm_arry_c0[p], e1_strm_arry_c1[p], k1_strm_arry[p], p1_strm_arry[p],
                hash1_strm_arry[p], e1_strm_arry[p]);
        }
    } else {
        // _CHNM == 4
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
        details:
            hash_group_aggregate::merge4_1<_WKey, _KeyNM, _WPay, _PayNM, _WHashLow>(
                k1_strm_arry_c0[p], k1_strm_arry_c1[p], k1_strm_arry_c2[p], k1_strm_arry_c3[p], p1_strm_arry_c0[p],
                p1_strm_arry_c1[p], p1_strm_arry_c2[p], p1_strm_arry_c3[p], hash_strm_arry_c0[p], hash_strm_arry_c1[p],
                hash_strm_arry_c2[p], hash_strm_arry_c3[p], e1_strm_arry_c0[p], e1_strm_arry_c1[p], e1_strm_arry_c2[p],
                e1_strm_arry_c3[p], k1_strm_arry[p], p1_strm_arry[p], hash1_strm_arry[p], e1_strm_arry[p]);
        }
    }

//---------------------------------aggregate PU---------------------------------

#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
    std::cout << "---------------------aggregate PU-----------------------" << std::endl;
#endif
#endif

    // hash aggregate processing unit
    if (PU >= 1) {
#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
        std::cout << "-------------------------PU0----------------------------" << std::endl;
#endif
#endif

        details::hash_group_aggregate::hash_aggr_pu_wrapper<_HashMode, _WHashLow, _WKey, _KeyNM, _WPay, _PayNM, _Wcnt,
                                                            _WBuffer, _BurstLenW>(
            hash1_strm_arry[0], k1_strm_arry[0], p1_strm_arry[0],
            e1_strm_arry[0],                                       // input stream
            out_buf0,                                              // buffer
            op_type[0],                                            // operation
            key_column, pld_column,                                // column number
            round,                                                 // loop cnt
            unhandle_cnt_w[0],                                     // overflow cnt
            aggr_key_arry[0], aggr_pld_array[0], e2_strm_arry[0]); // output stream
    }

    if (PU >= 2) {
#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
        std::cout << "-------------------------PU1----------------------------" << std::endl;
#endif
#endif

        details::hash_group_aggregate::hash_aggr_pu_wrapper<_HashMode, _WHashLow, _WKey, _KeyNM, _WPay, _PayNM, _Wcnt,
                                                            _WBuffer, _BurstLenW>(
            hash1_strm_arry[1], k1_strm_arry[1], p1_strm_arry[1],
            e1_strm_arry[1],                                       // input stream
            out_buf1,                                              // buffer
            op_type[1],                                            // operation
            key_column, pld_column,                                // column number
            round,                                                 // loop cnt
            unhandle_cnt_w[1],                                     // overflow cnt
            aggr_key_arry[1], aggr_pld_array[1], e2_strm_arry[1]); // output stream
    }

    if (PU >= 4) {
#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
        std::cout << "-------------------------PU2----------------------------" << std::endl;
#endif
#endif

        details::hash_group_aggregate::hash_aggr_pu_wrapper<_HashMode, _WHashLow, _WKey, _KeyNM, _WPay, _PayNM, _Wcnt,
                                                            _WBuffer, _BurstLenW>(
            hash1_strm_arry[2], k1_strm_arry[2], p1_strm_arry[2],
            e1_strm_arry[2],                                       // input stream
            out_buf2,                                              // buffer
            op_type[2],                                            // operation
            key_column, pld_column,                                // column number
            round,                                                 // loop cnt
            unhandle_cnt_w[2],                                     // overflow cnt
            aggr_key_arry[2], aggr_pld_array[2], e2_strm_arry[2]); // output stream

#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
        std::cout << "-------------------------PU3----------------------------" << std::endl;
#endif
#endif

        details::hash_group_aggregate::hash_aggr_pu_wrapper<_HashMode, _WHashLow, _WKey, _KeyNM, _WPay, _PayNM, _Wcnt,
                                                            _WBuffer, _BurstLenW>(
            hash1_strm_arry[3], k1_strm_arry[3], p1_strm_arry[3],
            e1_strm_arry[3],                                       // input stream
            out_buf3,                                              // buffer
            op_type[3],                                            // operation
            key_column, pld_column,                                // column number
            round,                                                 // loop cnt
            unhandle_cnt_w[3],                                     // overflow cnt
            aggr_key_arry[3], aggr_pld_array[3], e2_strm_arry[3]); // output stream
    }

//---------------------------------collect-----------------------------------

#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
    std::cout << "-----------------collect aggr result--------------------" << std::endl;
#endif
#endif

    details::hash_group_aggregate::collect_unit<PU, _WKey, _KeyNM, _WPay, _PayNM>(
        aggr_key_arry, aggr_pld_array, e2_strm_arry, aggregate_num, collect_key, collect_pld, e3_strm);

//---------------------------------aggr mean-----------------------------------

#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
    std::cout << "-----------------calculate aggr mean--------------------" << std::endl;
#endif
#endif

    details::hash_group_aggregate::calculate_aggr_mean<_WKey, _KeyNM, _WPay, _PayNM, 4>(
        collect_key, collect_pld, e3_strm, op_type[PU], aggr_key_out, aggr_pld_out, strm_e_out);

} // end hash_aggr_top

} // namespace hash_group_aggregate
} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {

/**
 * @brief Generic hash group aggregate primitive.
 *
 * With this primitive, the max number of lines of aggregate table is bound by the AXI buffer size.
 *
 * The group aggregation values are updated inside the chip, and when a hash-bucket overflows, the overflowed rows are
 * spilled into
 * external buffers. The overflow buffer will be automatically re-scanned, and within each round, a number of distinct
 * groups will be aggregated and emitted. This algorithm ends when the overflow buffer is empty and all groups are
 * aggregated.
 *
 * \rst
 *
 * .. ATTENTION::
 *     1. This module can accept multiple input row of key and payload pair per cycle.
 *     2. The max distinct groups aggregated in one pass is ``2 ^ (1 + _WHash)``.
 *     3. When the width of the input stream is not fully used, data should be aligned to the little-end.
 *     4. It is highly recommended to assign the ping buffer and pong buffer in different HBM banks, input and output in
 *        different DDR banks for a better performance.
 *     5. The max number of lines of aggregate table cannot bigger than the max DDR/HBM SIZE used in this design.
 *     6. When the bit-width of group key is known to be small, say 10-bit, please consider the ``directAggregate``
 *        primitive, which offers smaller utilization, and requires no external buffer access.
 *
 * \endrst
 *
 * @tparam _WKey width of key, in bit.
 * @tparam _KeyNM maximum number of key column, maximum is 8.
 * @tparam _WPay width of max payload, in bit.
 * @tparam _PayNM maximum number of payload column, maximum is 8.
 * @tparam _HashMode control hash algotithm, 0: radix 1: lookup3.
 * @tparam _WHashHigh number of hash bits used for dispatch pu.
 * @tparam _WHashLow number of hash bits used for hash-table.
 * @tparam _CHNM number of input channels.
 * @tparam _WBuffer width of HBM/DDR buffer(ping_buf and pong_buf).
 * @tparam _BurstLenW burst len of writting unhandled data.
 * @tparam _BurstLenR burst len of reloading unhandled data.
 *
 * @param strm_key_in input of key streams.
 * @param strm_pld_in input of payload streams.
 * @param strm_e_in input of end signal.
 * @param config information for initializing primitive, contains op for maximum
 * of 8 columns, key column number(less than 8),
 * pld column number(less than 8) and initial aggregate cnt.
 * @param result_info result information at kernel end, contains op, key_column,
 * pld_column and aggregate result cnt
 * @param ping_buf0 DDR/HBM ping buffer for unhandled data.
 * @param ping_buf1 DDR/HBM ping buffer for unhandled data.
 * @param ping_buf2 DDR/HBM ping buffer for unhandled data.
 * @param ping_buf3 DDR/HBM ping buffer for unhandled data.
 * @param pong_buf0 DDR/HBM pong buffer for unhandled data.
 * @param pong_buf1 DDR/HBM pong buffer for unhandled data.
 * @param pong_buf2 DDR/HBM pong buffer for unhandled data.
 * @param pong_buf3 DDR/HBM pong buffer for unhandled data.
 * @param aggr_key_out output of key columns.
 * @param aggr_pld_out output of pld columns. [0][*] is the result of min/max/cnt for
 * pld columns, [1][*] is the low-bit value of sum/average, [2][*] is the hight-bit
 * value of sum/average.
 * @param strm_e_out is the end signal of output.
 */
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
          int _BurstLenW = 32,
          int _BurstLenR = 32>
void hashGroupAggregate(
    // stream in
    hls::stream<ap_uint<_WKey> > strm_key_in[_CHNM][_KeyNM],
    hls::stream<ap_uint<_WPay> > strm_pld_in[_CHNM][_PayNM],
    hls::stream<bool> strm_e_in[_CHNM],

    // control param
    hls::stream<ap_uint<32> >& config,
    hls::stream<ap_uint<32> >& result_info,

    // ping-pong buffer
    ap_uint<_WBuffer>* ping_buf0,
    ap_uint<_WBuffer>* ping_buf1,
    ap_uint<_WBuffer>* ping_buf2,
    ap_uint<_WBuffer>* ping_buf3,

    ap_uint<_WBuffer>* pong_buf0,
    ap_uint<_WBuffer>* pong_buf1,
    ap_uint<_WBuffer>* pong_buf2,
    ap_uint<_WBuffer>* pong_buf3,

    // stream out
    hls::stream<ap_uint<_WKey> > aggr_key_out[_KeyNM],
    hls::stream<ap_uint<_WPay> > aggr_pld_out[3][_PayNM],
    hls::stream<bool>& strm_e_out) {
#pragma HLS inline off

    enum { PU = (1 << _WHashHigh) }; // high hash for distribution.

    bool loop_continue;
    ap_uint<32> op;
    ap_uint<32> key_column;
    ap_uint<32> pld_column;
    ap_uint<32> aggr_num;
    ap_uint<32> round = 0;
    ap_uint<32> unhandle_cnt_r[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma HLS ARRAY_PARTITION variable = unhandle_cnt_r complete
    ap_uint<32> unhandle_cnt_w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma HLS ARRAY_PARTITION variable = unhandle_cnt_w complete

    details::hash_group_aggregate::read_config<32>(config, op, key_column, pld_column, aggr_num);

    ap_uint<32> op_type[9] = {op, op, op, op, op, op, op, op, op};

#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
    std::cout << "-----------------hash group aggr mpu---------------------" << std::endl;
#endif
#endif

    do {
// clang-format off
#pragma HLS ALLOCATION function instances = details::hash_group_aggregate::hash_aggr_top<_WKey, _KeyNM, _WPay, _PayNM, _HashMode, _WHashHigh,  _WHashLow, _CHNM, _Wcnt, _WBuffer, _BurstLenW, _BurstLenR> limit = 1
// clang-format on

#ifndef __SYNTHESIS__
        std::cout << std::hex << "hash_aggr_top: op=" << op << " key_column=" << key_column
                  << " pld_column=" << pld_column << " round=" << round << std::endl;
#endif

        if (round[0] == 0) {
            details::hash_group_aggregate::hash_aggr_top<_WKey, _KeyNM, _WPay, _PayNM, _HashMode, _WHashHigh, _WHashLow,
                                                         _CHNM, _Wcnt, _WBuffer, _BurstLenW, _BurstLenR>(
                strm_key_in, strm_pld_in, strm_e_in, op_type, key_column, pld_column, round, unhandle_cnt_r,
                unhandle_cnt_w, aggr_num,

                ping_buf0, ping_buf1, ping_buf2, ping_buf3, pong_buf0, pong_buf1, pong_buf2, pong_buf3, aggr_key_out,
                aggr_pld_out, strm_e_out);
        } else {
            details::hash_group_aggregate::hash_aggr_top<_WKey, _KeyNM, _WPay, _PayNM, _HashMode, _WHashHigh, _WHashLow,
                                                         _CHNM, _Wcnt, _WBuffer, _BurstLenW, _BurstLenR>(
                strm_key_in, strm_pld_in, strm_e_in, op_type, key_column, pld_column, round, unhandle_cnt_r,
                unhandle_cnt_w, aggr_num,

                pong_buf0, pong_buf1, pong_buf2, pong_buf3, ping_buf0, ping_buf1, ping_buf2, ping_buf3, aggr_key_out,
                aggr_pld_out, strm_e_out);
        }

        // generate control signal
        round++;
        loop_continue = false;

        for (int i = 0; i < PU; i++) {
#pragma HLS unroll
            loop_continue |= (unhandle_cnt_w[i] != 0);
            unhandle_cnt_r[i] = unhandle_cnt_w[i];

#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
            std::cout << "hash_group_aggregate unhandle_cnt[" << i << "]=" << unhandle_cnt_r[i] << std::dec
                      << std::endl;

#endif
#endif
        }

#ifndef __SYNTHESIS__
#ifdef DEBUG_AGGR
        std::cout << "hash_group_aggregate round=" << round << " loop_end=" << loop_continue
                  << " ping_buf0=" << ping_buf0 << " ping_buf1=" << ping_buf1 << " ping_buf2=" << ping_buf2
                  << " ping_buf3=" << ping_buf3 << " pong_buf0=" << pong_buf0 << " pong_buf1=" << pong_buf1
                  << " pong_buf2=" << pong_buf2 << " pong_buf3=" << pong_buf3 << std::endl;

#endif
#endif

    } while (loop_continue);

    details::hash_group_aggregate::write_info<32>(result_info, op, key_column, pld_column, aggr_num);

    // generate end signal for the last element
    strm_e_out.write(true);

} // end hashGroupAggregate

} // namespace database
} // namespace xf
#endif
