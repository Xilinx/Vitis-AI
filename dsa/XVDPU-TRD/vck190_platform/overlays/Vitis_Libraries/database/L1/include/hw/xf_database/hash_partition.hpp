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
 * @file hash_partition.hpp
 * @brief hash partition implementation
 *
 * This file is part of Vitis Database Library.
 */

#ifndef XF_DATABASE_HASH_PARTITION_H
#define XF_DATABASE_HASH_PARTITION_H

#ifndef __cplusplus
#error "Vitis Database Library only works with C++."
#endif

#include <ap_int.h>
#include <hls_stream.h>
#include "xf_database/hash_lookup3.hpp"
#include "xf_utils_hw/uram_array.hpp"

#define DEBUG

#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {
namespace database {
namespace details {
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

template <int HASH_MODE, int KEYW, int HASHW>
void hash_wrapper(hls::stream<ap_uint<KEYW> >& i_key_strm,
                  hls::stream<bool>& i_e_strm,

                  hls::stream<ap_uint<HASHW> >& o_hash_strm,
                  hls::stream<ap_uint<KEYW> >& o_key_strm,
                  hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    hls::stream<ap_uint<KEYW> > key_strm_in;
#pragma HLS STREAM variable = key_strm_in depth = 8
#pragma HLS bind_storage variable = key_strm_in type = fifo impl = srl
    hls::stream<ap_uint<64> > hash_strm_out;
#pragma HLS STREAM variable = hash_strm_out depth = 8
#pragma HLS bind_storage variable = hash_strm_out type = fifo impl = srl

#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif

    bool last = i_e_strm.read();
BUILD_HASH_LOOP:
    while (!last) {
#pragma HLS loop_tripcount min = 1 max = 1000
#pragma HLS PIPELINE II = 1

        // bool blk = i_e_strm.empty() || i_key_strm.empty() || o_key_strm.full();
        // if (!blk) {
        last = i_e_strm.read();
        ap_uint<KEYW> key = i_key_strm.read();

        o_key_strm.write(key);
        o_e_strm.write(false);

        if (HASH_MODE == 0) {
            // radix hash
            ap_uint<HASHW> s_hash_val = key(HASHW - 1, 0);
            o_hash_strm.write(s_hash_val);

        } else {
            // Jekins lookup3 hash
            key_strm_in.write(key);
            xf::database::hashLookup3<KEYW>(13, key_strm_in, hash_strm_out);
            ap_uint<64> l_hash_val = hash_strm_out.read();

            ap_uint<HASHW> s_hash_val = l_hash_val(HASHW - 1, 0);
            o_hash_strm.write(s_hash_val);

#ifndef __SYNTHESIS__

#ifdef DEBUG_MISS
            if (key == 3680482 || key == 3691265 || key == 4605699 || key == 4987782)
                std::cout << std::hex << "hashwrapper: key=" << key << " hash=" << s_hash_val << std::endl;
#endif

#ifdef DEBUG
// if (cnt < 10) {
// std::cout << std::hex << "hash wrapper: cnt=" << cnt << " key = " << key
//           << " hash_val = " << s_hash_val << std::endl;
// }
#endif
            cnt++;
#endif
        }
        //}
    }
    o_e_strm.write(true);
}

template <int KEYW, int PW, int HASHWH, int HASHWL, int PU>
void dispatch_to_pu(hls::stream<ap_uint<KEYW> >& i_key_strm,
                    hls::stream<ap_uint<PW> >& i_pld_strm,
                    hls::stream<ap_uint<HASHWH + HASHWL> >& i_hash_strm,
                    hls::stream<bool>& i_e_strm,

                    hls::stream<ap_uint<KEYW> > o_key_strm[PU],
                    hls::stream<ap_uint<PW> > o_pld_strm[PU],
                    hls::stream<ap_uint<HASHWL> > o_hash_strm[PU],
                    hls::stream<bool> o_e_strm[PU]) {
#pragma HLS INLINE off

    bool last = i_e_strm.read();
LOOP_DISPATCH:
    while (!last) {
#pragma HLS pipeline II = 1

        ap_uint<HASHWH + HASHWL> hash_val = i_hash_strm.read();
        ap_uint<HASHWL> hash_out = hash_val(HASHWL - 1, 0);
        ap_uint<HASHWH + 1> pu_idx;
        if (HASHWH > 0)
            pu_idx = hash_val(HASHWH + HASHWL - 1, HASHWL);
        else
            pu_idx = 0;

        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<PW> pld = i_pld_strm.read();
        last = i_e_strm.read();

        o_key_strm[pu_idx].write(key);
        o_pld_strm[pu_idx].write(pld);
        o_hash_strm[pu_idx].write(hash_out);
        o_e_strm[pu_idx].write(false);
    }

    for (int i = 0; i < PU; i++) {
#pragma HLS unroll
        // if add merge module, need uncomment
        o_key_strm[i].write(0);
        o_pld_strm[i].write(0);
        o_hash_strm[i].write(0);
        o_e_strm[i].write(true);
    }
}

// dispatch data based on hash value to multiple PU.
template <int HASH_MODE, int KEYW, int PW, int HASHWH, int HASHWL, int PU>
void dispatch_unit(hls::stream<ap_uint<KEYW> >& i_key_strm,
                   hls::stream<ap_uint<PW> >& i_pld_strm,
                   hls::stream<bool>& i_e_strm,

                   hls::stream<ap_uint<KEYW> > o_key_strm[PU],
                   hls::stream<ap_uint<PW> > o_pld_strm[PU],
                   hls::stream<ap_uint<HASHWL> > o_hash_strm[PU],
                   hls::stream<bool> o_e_strm[PU]) {
#pragma HLS DATAFLOW

    hls::stream<ap_uint<HASHWH + HASHWL> > hash_strm;
#pragma HLS STREAM variable = hash_strm depth = 8
#pragma HLS bind_storage variable = hash_strm type = fifo impl = srl
    hls::stream<ap_uint<KEYW> > key_strm;
#pragma HLS STREAM variable = key_strm depth = 8
#pragma HLS bind_storage variable = key_strm type = fifo impl = srl
    hls::stream<bool> e_strm;
#pragma HLS STREAM variable = e_strm depth = 8

    hash_wrapper<HASH_MODE, KEYW, HASHWH + HASHWL>(i_key_strm, i_e_strm, hash_strm, key_strm, e_strm);

    dispatch_to_pu<KEYW, PW, HASHWH, HASHWL, PU>(key_strm, i_pld_strm, hash_strm, e_strm, o_key_strm, o_pld_strm,
                                                 o_hash_strm, o_e_strm);
}

/// @brief Merge stream of mutiple channels into one PU, merge 1 to 1
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
#ifndef __SYNTHESIS__
    static int pu = 0;
    unsigned int cnt = 0;
#endif
LOOP_MERGE1_1:
    do {
#pragma HLS pipeline II = 1
        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<PW> pld = i_pld_strm.read();
        ap_uint<HASHW> hash_val = i_hash_strm.read();
        last = i_e_strm.read();
        if (!last) {
#ifndef __SYNTHESIS__
            cnt++;
#endif
            o_key_strm.write(key);
            o_pld_strm.write(pld);
            o_hash_strm.write(hash_val);
            o_e_strm.write(false);
        }
    } while (!last);
    o_e_strm.write(true);
#ifndef __SYNTHESIS__
    std::cout << "PU" << pu++ << " row number:" << cnt << std::endl;
#endif
}

/// @brief Merge stream of mutiple channels into one PU, merge 2 to 1
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
    ap_uint<2> rd_e = 0;
    ap_uint<2> last = 0;
#ifndef __SYNTHESIS__
    static int pu = 0;
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
#ifndef __SYNTHESIS__
    std::cout << "PU" << pu++ << " row number:" << cnt << std::endl;
#endif
}

/// @brief Merge stream of mutiple channels into one PU, merge 4 to 1
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
    ap_uint<4> rd_e = 0;
    ap_uint<4> last = 0;
#ifndef __SYNTHESIS__
    static int pu = 0;
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
// std::cout << std::hex << "rd_e:" << rd_e << std::dec << ", cnt:" << cnt << std::endl;
#endif
            o_key_strm.write(key);
            o_pld_strm.write(pld);
            o_hash_strm.write(hash_val);
            o_e_strm.write(false);
        }
    } while (last != 15);
    o_e_strm.write(true);
#ifndef __SYNTHESIS__
    std::cout << "PU" << pu++ << " row number:" << cnt << std::endl;
#endif
}

/*
 * @tparam HASHW, valid data width of hash value, log2(BK)
 */
template <int HASHW, int KEYW, int PW, int ARW>
void build_unit(int k_depth,
                hls::stream<int>& bit_num_strm,

                hls::stream<ap_uint<HASHW> >& i_hash_strm,
                hls::stream<ap_uint<KEYW> >& i_key_strm,
                hls::stream<ap_uint<PW> >& i_pld_strm,
                hls::stream<bool>& i_e_strm,

                hls::stream<ap_uint<KEYW + PW> >& o_kpld_strm,
                hls::stream<ap_uint<10> >& o_nm_strm,
                hls::stream<ap_uint<10> >& o_bk_nm_strm) {
    const ap_uint<10> depth = k_depth % 1024;
    const int bit_num = bit_num_strm.read();
    const int BK = 1 << bit_num;
    const int uram_depth = 1 << ARW; // base + overflow
    bool ren = false;
    ap_uint<HASHW + 1> bk_num;
    ap_uint<10> rcnt = 0; // counter for uram read
    xf::common::utils_hw::UramArray<KEYW + PW, uram_depth, 8> uram_inst;
    hls::stream<ap_uint<HASHW + 1> > wlist;
#pragma HLS stream variable = wlist depth = 256
    ap_uint<10> bucket_cnt[256];
#pragma HLS array_partition variable = bucket_cnt complete dim = 0
    ap_uint<10> bucket_reg;
    for (int i = 0; i < 256; i++) {
#pragma HLS unroll
        bucket_cnt[i] = 0;
    }
#ifndef __SYNTHESIS__
    static int pu = 0;
    unsigned int cnt = 0;
#endif

    bool last = i_e_strm.read();
BUILD_UNIT_CORE_LOOP:
    while (!last) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = uram_inst.blocks inter false
        ap_uint<HASHW> hash_val = i_hash_strm.read();
        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<PW> pld = i_pld_strm.read();
        last = i_e_strm.read();

        bucket_reg = bucket_cnt[hash_val(bit_num - 1, 0)];

        ap_uint<ARW> waddr = hash_val(bit_num - 1, 0) * depth * 2 + bucket_reg;
        ap_uint<KEYW + PW> wdata = (pld, key);
        uram_inst.write(waddr, wdata);

#ifndef __SYNTHESIS__
        cnt++;
// if ((pu == 0))
// std::cout << "wdata:" << std::dec << pld(31, 0) << ", addr:" << waddr << std::dec
//           << ",bucket_reg:" << bucket_reg << " " << std::endl;
#endif

        if ((bucket_reg == (depth - 1)) || (bucket_reg == (2 * depth - 1))) {
            ap_uint<1> sector = (bucket_reg >= depth);
            wlist.write((hash_val(bit_num - 1, 0), sector));
        }

        if (bucket_reg == (2 * depth - 1))
            bucket_reg = 0;
        else
            bucket_reg++;

        // update increased value
        bucket_cnt[hash_val(bit_num - 1, 0)] = bucket_reg;

        if (!wlist.empty() && (rcnt == 0)) { // only update when one full bucket is read out
            bk_num = wlist.read();
            ren = true;
        } else
            ren = false;

        if (ren || (rcnt != 0)) { // write and read simulatously
            ap_uint<ARW> raddr = bk_num(bit_num, 0) * depth + rcnt;
            ap_uint<KEYW + PW> rdata = uram_inst.read(raddr);
#ifndef __SYNTHESIS__
// if ((pu == 0) && (bk_num[1] == 0))
// std::cout << "rdata:" << std::dec << rdata(95, 64) << ", addr:" << raddr << std::dec << ",rcnt:" << rcnt
//           << std::endl;
#endif
            o_kpld_strm.write(rdata);
            if (rcnt == depth - 1) {
                o_nm_strm.write(depth);
                o_bk_nm_strm.write(bk_num(bit_num, 1));
                rcnt = 0;
            } else
                rcnt++;
        }

    } // end while loop

#ifndef __SYNTHESIS__
    std::cout << "wlist.size() = " << wlist.size() << ", rcnt = " << rcnt << std::endl;
#endif

    if (rcnt != 0) {
    BUILD_REMAIN_ONE_BUCKET_LOOP:
        for (int i = rcnt; i < depth; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = uram_inst.blocks inter false
            ap_uint<ARW> raddr = bk_num(bit_num, 0) * depth + i;
            ap_uint<KEYW + PW> rdata = uram_inst.read(raddr);
            o_kpld_strm.write(rdata);
        }
        o_nm_strm.write(depth);
        o_bk_nm_strm.write(bk_num(bit_num, 1));
    }

    while (!wlist.empty()) {
        bk_num = wlist.read();
    BUILD_REMAIN_COMPLETE_BUCKET_OUT_LOOP:
        for (int i = 0; i < depth; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = uram_inst.blocks inter false
            ap_uint<ARW> raddr = bk_num(bit_num, 0) * depth + i;
            ap_uint<KEYW + PW> rdata = uram_inst.read(raddr);
            o_kpld_strm.write(rdata);
        }
        o_nm_strm.write(depth);
        o_bk_nm_strm.write(bk_num(bit_num, 1));
    }

    ap_uint<10> len;
    ap_uint<ARW> offset;
    for (int i = 0; i < BK; i++) {
        bucket_reg = bucket_cnt[i];
        if (bucket_reg == 0) {
            len = 0;
            offset = 0;
        } else if ((bucket_reg > 0) && (bucket_reg < depth)) {
            len = bucket_reg;
            offset = 0;
        } else {
            len = bucket_reg - depth;
            offset = depth;
        }
        // if (pu == 0)
        // std::cout << "bucket: " << i << ",bucket_reg:" << bucket_reg << ",remain:" << len << ",offset:" << offset
        //           << std::endl;
        if (len != 0) {
            o_nm_strm.write(len);
            o_bk_nm_strm.write(i);
        BUILD_INCOMPLETE_BUCKET_OUT_LOOP:
            for (int j = 0; j < len; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = uram_inst.blocks inter false
                ap_uint<ARW> raddr = 2 * i * depth + offset + j; // need optimize
                ap_uint<KEYW + PW> rdata = uram_inst.read(raddr);
                // if (pu == 0)
                // std::cout << "data:" << std::hex << rdata << ", address:" << std::dec << raddr << std::endl;
                o_kpld_strm.write(rdata);
            }
        }
    }

    o_nm_strm.write(0);
    o_bk_nm_strm.write(0);
#ifndef __SYNTHESIS__
    std::cout << "PU-" << pu << " build output row number:" << o_kpld_strm.size() << std::endl;
    pu++;
#endif
}

// load balance read multi-PU and split into column
template <int KEYW, int PW, int EW, int PU, int COL_NM>
void split_read_load_balance(bool mk_on,
                             hls::stream<ap_uint<KEYW + PW> > i_kpld_strm_array[PU],
                             hls::stream<ap_uint<10> > i_nm_strm_array[PU],
                             hls::stream<ap_uint<10> > i_bk_nm_array[PU],

                             hls::stream<ap_uint<16> >& o_bkpu_num_strm,
                             hls::stream<ap_uint<10> >& o_nm_strm,
                             hls::stream<ap_uint<EW> > out_strm[COL_NM]) {
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
    ap_uint<6> pu_idx = 0;
    ap_uint<PU> cond = 0;
    ap_uint<PU> last = -1;
    do {
        if (!i_nm_strm_array[pu_idx].empty()) {
            const ap_uint<10> nm = i_nm_strm_array[pu_idx].read();
            const ap_uint<10> bk = i_bk_nm_array[pu_idx].read();
            if (nm > 0) {
                cond[pu_idx] = 0;
                o_nm_strm.write(nm);
                o_bkpu_num_strm.write((pu_idx, bk));
            } else {
                cond[pu_idx] = 1;
            }
        WRITE_ONE_BUCKET_PU:
            for (int i = 0; i < nm; i++) {
#pragma HLS PIPELINE II = 1
                ap_uint<KEYW + PW> dt = i_kpld_strm_array[pu_idx].read();
#ifndef __SYNTHESIS__
                if (pu_idx == 0) cnt++;
#endif
                ap_uint<EW> st[COL_NM];
                for (int c = 0; c < COL_NM; c++) {
#pragma HLS unroll
                    if (c == 0) {
                        st[c] = dt(EW - 1, 0);
                    } else if (c < COL_NM - 1) {
                        st[c] = mk_on ? dt((c + 1) * EW - 1, c * EW) : dt((c + 2) * EW - 1, (c + 1) * EW);
#ifndef __SYNTHESIS__
// if (c == 1) std::cout << "split:" << st[c] << std::endl;
#endif
                    } else {
                        st[c] = mk_on ? 0 : dt((c + 1) * EW - 1, c * EW);
                    }
                    out_strm[c].write(st[c]);
                }
            } // end one bucket read
        }

        if (pu_idx == PU - 1)
            pu_idx = 0;
        else
            pu_idx++;
    } while (cond != last);

#ifndef __SYNTHESIS__
    std::cout << "PU0 read counter: " << cnt << std::endl;
#endif
    o_nm_strm.write(0);
}

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {

/**
 * @brief Hash-Partition primitive
 *
 * @tparam HASH_MODE 0 for radix and 1 for Jenkin's Lookup3 hash.
 * @tparam KEYW width of key, in bit.
 * @tparam PW width of max payload, in bit.
 * @tparam EW element data width of input table, in bit.
 * @tparam HASHWH number of hash bits used for PU selection.
 * @tparam HASHWL number of hash bits used for partition selection.
 * @tparam ARW width of address for URAM
 * @tparam CH_NM number of input channels, 1,2,4.
 * @tparam COL_NM number of input columns, 1~8.
 *
 * @param mk_on input of double key flag, 0 for off, 1 for on.
 * @param depth input of depth of each hash bucket in URAM.
 * @param bit_num_strm input of partition number, log2(number of partition).
 *
 * @param k0_strm_arry input of key columns of both tables.
 * @param p0_strm_arry input of payload columns of both tables.
 * @param e0_strm_arry input of end signal of both tables.
 *
 * @param o_bkpu_num_strm output of index for bucket and PU
 * @param o_nm_strm output of row number each time
 * @param o_kpld_strm output of key+payload
 *
 */
template <int HASH_MODE, int KEYW, int PW, int EW, int HASHWH, int HASHWL, int ARW, int CH_NM, int COL_NM>
void hashPartition(
    // input
    bool mk_on,
    int depth,
    hls::stream<int>& bit_num_strm,

    hls::stream<ap_uint<KEYW> > k0_strm_arry[CH_NM],
    hls::stream<ap_uint<PW> > p0_strm_arry[CH_NM],
    hls::stream<bool> e0_strm_arry[CH_NM],

    // output
    hls::stream<ap_uint<16> >& o_bkpu_num_strm,
    hls::stream<ap_uint<10> >& o_nm_strm,
    hls::stream<ap_uint<EW> > o_kpld_strm[COL_NM]) {
    enum { PU = (1 << HASHWH), BDEPTH = 512 * 4 };

#pragma HLS DATAFLOW

    // dispatch k0_strm_arry, p0_strm_arry, e0strm_arry to channel1-4
    // Channel 1~4
    hls::stream<ap_uint<KEYW> > k1_strm_arry_mc[CH_NM][PU];
#pragma HLS stream variable = k1_strm_arry_mc depth = 8
#pragma HLS array_partition variable = k1_strm_arry_mc complete dim = 0
#pragma HLS bind_storage variable = k1_strm_arry_mc type = fifo impl = srl
    hls::stream<ap_uint<PW> > p1_strm_arry_mc[CH_NM][PU];
#pragma HLS stream variable = p1_strm_arry_mc depth = 8
#pragma HLS array_partition variable = p1_strm_arry_mc complete dim = 0
#pragma HLS bind_storage variable = p1_strm_arry_mc type = fifo impl = srl
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_mc[CH_NM][PU];
#pragma HLS stream variable = hash_strm_arry_mc depth = 8
#pragma HLS array_partition variable = hash_strm_arry_mc complete dim = 0
#pragma HLS bind_storage variable = hash_strm_arry_mc type = fifo impl = srl
    hls::stream<bool> e1_strm_arry_mc[CH_NM][PU];
#pragma HLS stream variable = e1_strm_arry_mc depth = 8
#pragma HLS array_partition variable = e1_strm_arry_mc complete dim = 0

    // merge channel 1~4 here, then perform build process
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

    // output of PU
    hls::stream<ap_uint<KEYW + PW> > t_kpld_strm_arry[PU];
#pragma HLS stream variable = t_kpld_strm_arry depth = BDEPTH
#pragma HLS array_partition variable = t_kpld_strm_arry dim = 1
#pragma HLS bind_storage variable = t_kpld_strm_arry type = fifo impl = bram
    hls::stream<ap_uint<10> > t_nm_strm_arry[PU];
#pragma HLS stream variable = t_nm_strm_arry depth = BDEPTH
#pragma HLS array_partition variable = t_nm_strm_arry dim = 1
#pragma HLS bind_storage variable = t_nm_strm_arry type = fifo impl = bram
    hls::stream<ap_uint<10> > t_bk_nm_strm_arry[PU];
#pragma HLS stream variable = t_bk_nm_strm_arry depth = BDEPTH
#pragma HLS array_partition variable = t_bk_nm_strm_arry dim = 1
#pragma HLS bind_storage variable = t_bk_nm_strm_arry type = fifo impl = bram

//---------------------------------dispatch PU-------------------------------
#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "------------------------dispatch PU------------------------" << std::endl;
#endif
#endif

    if (CH_NM >= 1) {
        details::dispatch_unit<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(
            k0_strm_arry[0], p0_strm_arry[0], e0_strm_arry[0], k1_strm_arry_mc[0], p1_strm_arry_mc[0],
            hash_strm_arry_mc[0], e1_strm_arry_mc[0]);
    }

    if (CH_NM >= 2) {
        details::dispatch_unit<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(
            k0_strm_arry[1], p0_strm_arry[1], e0_strm_arry[1], k1_strm_arry_mc[1], p1_strm_arry_mc[1],
            hash_strm_arry_mc[1], e1_strm_arry_mc[1]);
    }

    if (CH_NM >= 4) {
        details::dispatch_unit<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(
            k0_strm_arry[2], p0_strm_arry[2], e0_strm_arry[2], k1_strm_arry_mc[2], p1_strm_arry_mc[2],
            hash_strm_arry_mc[2], e1_strm_arry_mc[2]);

        details::dispatch_unit<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(
            k0_strm_arry[3], p0_strm_arry[3], e0_strm_arry[3], k1_strm_arry_mc[3], p1_strm_arry_mc[3],
            hash_strm_arry_mc[3], e1_strm_arry_mc[3]);
    }

//---------------------------------merge PU---------------------------------
#ifndef __SYNTHESIS__
#ifdef DEBUG
    long long rcnt = 0;
    std::cout << "\t\t|";
    for (int i = 0; i < PU; i++) {
        std::cout << "PU" << i << "|\t\t";
    }
    if (CH_NM >= 1) {
        std::cout << "\nch0:\t|";
        for (int i = 0; i < PU; i++) {
            std::cout << k1_strm_arry_mc[0][i].size() << "|\t\t";
            rcnt += k1_strm_arry_mc[0][i].size();
        }
    }
    if (CH_NM >= 2) {
        std::cout << "\nch1:\t|";
        for (int i = 0; i < PU; i++) {
            std::cout << k1_strm_arry_mc[1][i].size() << "|\t\t";
            rcnt += k1_strm_arry_mc[1][i].size();
        }
    }
    if (CH_NM >= 4) {
        std::cout << "\nch2:\t|";
        for (int i = 0; i < PU; i++) {
            std::cout << k1_strm_arry_mc[2][i].size() << "|\t\t";
            rcnt += k1_strm_arry_mc[2][i].size();
        }
        std::cout << "\nch3:\t|";
        for (int i = 0; i < PU; i++) {
            std::cout << k1_strm_arry_mc[3][i].size() << "|\t\t";
            rcnt += k1_strm_arry_mc[3][i].size();
        }
    }
    std::cout << "\nTotal, dispatch row number is " << rcnt << std::endl;

    std::cout << "------------------------merge PU------------------------" << std::endl;
#endif
#endif

    if (CH_NM == 1) {
        for (int p = 0; p < PU; ++p) {
#pragma HLS unroll
            details::merge1_1(k1_strm_arry_mc[0][p], p1_strm_arry_mc[0][p], hash_strm_arry_mc[0][p],
                              e1_strm_arry_mc[0][p],

                              k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p], e1_strm_arry[p]);
        }
    } else if (CH_NM == 2) {
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            details::merge2_1(k1_strm_arry_mc[0][p], k1_strm_arry_mc[1][p], p1_strm_arry_mc[0][p],
                              p1_strm_arry_mc[1][p], hash_strm_arry_mc[0][p], hash_strm_arry_mc[1][p],
                              e1_strm_arry_mc[0][p], e1_strm_arry_mc[1][p],

                              k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p], e1_strm_arry[p]);
        }
    } else {
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            details::merge4_1(
                k1_strm_arry_mc[0][p], k1_strm_arry_mc[1][p], k1_strm_arry_mc[2][p], k1_strm_arry_mc[3][p],
                p1_strm_arry_mc[0][p], p1_strm_arry_mc[1][p], p1_strm_arry_mc[2][p], p1_strm_arry_mc[3][p],
                hash_strm_arry_mc[0][p], hash_strm_arry_mc[1][p], hash_strm_arry_mc[2][p], hash_strm_arry_mc[3][p],
                e1_strm_arry_mc[0][p], e1_strm_arry_mc[1][p], e1_strm_arry_mc[2][p], e1_strm_arry_mc[3][p],

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
        details::build_unit<HASHWL, KEYW, PW, ARW>(depth, bit_num_strm,

                                                   hash_strm_arry[0], k1_strm_arry[0], p1_strm_arry[0], e1_strm_arry[0],

                                                   t_kpld_strm_arry[0], t_nm_strm_arry[0], t_bk_nm_strm_arry[0]);
    }

    if (PU >= 2) {
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "PU0 is done\n\n";
        std::cout << "------------------------PU1------------------------" << std::endl;
#endif
#endif
        details::build_unit<HASHWL, KEYW, PW, ARW>(depth, bit_num_strm,

                                                   hash_strm_arry[1], k1_strm_arry[1], p1_strm_arry[1], e1_strm_arry[1],

                                                   t_kpld_strm_arry[1], t_nm_strm_arry[1], t_bk_nm_strm_arry[1]);
    }

    if (PU >= 4) {
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "PU1 is done\n\n";
        std::cout << "------------------------PU2------------------------" << std::endl;
#endif
#endif
        details::build_unit<HASHWL, KEYW, PW, ARW>(depth, bit_num_strm,

                                                   hash_strm_arry[2], k1_strm_arry[2], p1_strm_arry[2], e1_strm_arry[2],

                                                   t_kpld_strm_arry[2], t_nm_strm_arry[2], t_bk_nm_strm_arry[2]);

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "PU2 is done\n\n";
        std::cout << "------------------------PU3------------------------" << std::endl;
#endif
#endif
        details::build_unit<HASHWL, KEYW, PW, ARW>(depth, bit_num_strm,

                                                   hash_strm_arry[3], k1_strm_arry[3], p1_strm_arry[3], e1_strm_arry[3],

                                                   t_kpld_strm_arry[3], t_nm_strm_arry[3], t_bk_nm_strm_arry[3]);
    }

    if (PU >= 8) {
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "PU3 is done\n\n";
        std::cout << "------------------------PU4------------------------" << std::endl;
#endif
#endif
        details::build_unit<HASHWL, KEYW, PW, ARW>(depth, bit_num_strm,

                                                   hash_strm_arry[4], k1_strm_arry[4], p1_strm_arry[4], e1_strm_arry[4],

                                                   t_kpld_strm_arry[4], t_nm_strm_arry[4], t_bk_nm_strm_arry[4]);

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "PU4 is done\n\n";
        std::cout << "------------------------PU5------------------------" << std::endl;
#endif
#endif
        details::build_unit<HASHWL, KEYW, PW, ARW>(depth, bit_num_strm,

                                                   hash_strm_arry[5], k1_strm_arry[5], p1_strm_arry[5], e1_strm_arry[5],

                                                   t_kpld_strm_arry[5], t_nm_strm_arry[5], t_bk_nm_strm_arry[5]);

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "PU5 is done\n\n";
        std::cout << "------------------------PU6------------------------" << std::endl;
#endif
#endif
        details::build_unit<HASHWL, KEYW, PW, ARW>(depth, bit_num_strm,

                                                   hash_strm_arry[6], k1_strm_arry[6], p1_strm_arry[6], e1_strm_arry[6],

                                                   t_kpld_strm_arry[6], t_nm_strm_arry[6], t_bk_nm_strm_arry[6]);

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "PU6 is done\n\n";
        std::cout << "------------------------PU7------------------------" << std::endl;
#endif
#endif
        details::build_unit<HASHWL, KEYW, PW, ARW>(depth, bit_num_strm,

                                                   hash_strm_arry[7], k1_strm_arry[7], p1_strm_arry[7], e1_strm_arry[7],

                                                   t_kpld_strm_arry[7], t_nm_strm_arry[7], t_bk_nm_strm_arry[7]);
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "PU7 is done\n";
#endif
#endif
    }

    //-----------------------------------round-robin-----------------------------
    details::split_read_load_balance<KEYW, PW, EW, PU, COL_NM>(
        mk_on, t_kpld_strm_arry, t_nm_strm_arry, t_bk_nm_strm_arry, o_bkpu_num_strm, o_nm_strm, o_kpld_strm);
}

} // namespace database
} // namespace xf

#endif // XF_DATABASE_HASH_PARTITION_H
