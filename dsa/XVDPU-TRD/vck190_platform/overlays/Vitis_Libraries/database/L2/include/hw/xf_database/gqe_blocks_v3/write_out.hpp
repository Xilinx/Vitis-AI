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
#ifndef GQE_ISV_WRITE_OUT_HPP
#define GQE_ISV_WRITE_OUT_HPP

#include <ap_int.h>
#include <hls_stream.h>

#include "xf_database/utils.hpp"
#include "xf_utils_hw/stream_shuffle.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
//#define USER_DEBUG true
//#define XDEBUG true
#endif

namespace xf {
namespace database {
namespace gqe {

template <int elem_size, int vec_len, int col_num>
void write_prepare(bool bp_flag,
                   hls::stream<bool>& bp_flag_strm,
                   hls::stream<ap_uint<8> >& write_out_cfg_strm,
                   hls::stream<ap_uint<8> >& write_out_en_strm) {
#ifdef USER_DEBUG
    std::cout << "useful when append mode is on" << std::endl;
#endif
    bp_flag_strm.write(bp_flag);
    ap_uint<8> write_out_cfg = write_out_cfg_strm.read();
#ifdef USER_DEBUG
    std::cout << "write_out_cfg: " << (int)write_out_cfg << std::endl;
#endif
    write_out_en_strm.write(write_out_cfg);
    bool append_mode = write_out_cfg[7];
#ifdef USER_DEBUG
    std::cout << "append mode is " << append_mode << std::endl;
    if (append_mode) {
        std::cout << "error!!!! Append mode is not supported yet" << std::endl;
    }
#endif
}

template <int burst_len, int elem_size, int vec_len, int col_num>
void count_for_burst(hls::stream<bool>& i_bp_flag_strm,
                     hls::stream<bool>& o_bp_flag_strm,
                     hls::stream<ap_uint<elem_size> > in_strm[col_num], // 64-bit in
                     hls::stream<bool>& e_in_strm,
                     hls::stream<ap_uint<elem_size * vec_len> > out_strm[col_num], // 512-bit out
                     hls::stream<ap_uint<8> >& o_nm_strm,
                     hls::stream<ap_uint<32> >& rnm_strm,
                     hls::stream<ap_uint<8> >& i_wr_en_strm,
                     hls::stream<ap_uint<8> >& o_wr_en_strm) {
#ifdef USER_DEBUG
    std::cout << "e_in_strm.size(): " << e_in_strm.size() << std::endl;
    for (int c = 0; c < col_num; c++) {
        std::cout << "in_strm[c].size(): " << in_strm[c].size() << std::endl;
    }

#endif
    ap_uint<8> wr_en = i_wr_en_strm.read();
    o_wr_en_strm.write(wr_en);
#ifndef __SYNTHESIS__
    std::cout << "wr_en: " << wr_en << std::endl;
#endif

    bool build_probe_flag = i_bp_flag_strm.read();
    o_bp_flag_strm.write(build_probe_flag);
    bool e = e_in_strm.read();
    if (build_probe_flag) {
        ap_uint<elem_size * vec_len> vecs[col_num];
        int n = 0; // nrow count
        int r = 0; // element count in one 512b
        int b = 0; // burst length count
    LOOP_COUNT:
        while (!e) {
#pragma HLS pipeline II = 1
            e = e_in_strm.read();
            ++n;
            for (int c = 0; c < col_num; ++c) {
#pragma HLS unroll
                ap_uint<elem_size> t = in_strm[c].read();
                //#ifndef __SYNTHESIS__
                //                std::cout << "c: " << c << ", t:" << (int64_t)t << std::endl;
                //#endif
                vecs[c].range(elem_size * (vec_len - 1) - 1, 0) = vecs[c].range(elem_size * vec_len - 1, elem_size);
                vecs[c].range(elem_size * vec_len - 1, elem_size * (vec_len - 1)) = t;
            }
            if (r == vec_len - 1) { // 512bit, write 1 data
                for (int c = 0; c < col_num; ++c) {
#pragma HLS unroll
                    if (wr_en[c]) {
                        out_strm[c].write(vecs[c]);
                    }
                }
                if (b == burst_len - 1) {
                    o_nm_strm.write(burst_len);
                    b = 0;
                } else {
                    ++b;
                }
                r = 0;
            } else {
                ++r;
            }
        }

        // algin to 512bit with 0
        if (r != 0) {
        LOOP_FOR:
            for (; r < vec_len; ++r) {
#pragma HLS pipeline II = 1
                for (int c = 0; c < col_num; ++c) {
#pragma HLS unroll
                    vecs[c].range(elem_size * (vec_len - 1) - 1, 0) = vecs[c].range(elem_size * vec_len - 1, elem_size);
                    vecs[c].range(elem_size * vec_len - 1, elem_size * (vec_len - 1)) = 0;
                }
            }
            for (int c = 0; c < col_num; ++c) {
#pragma HLS unroll
                if (wr_en[c]) {
                    out_strm[c].write(vecs[c]);
                }
            }
            ++b;
        }

        // last burst
        if (b != 0) {
            XF_DATABASE_ASSERT(b <= burst_len);
            o_nm_strm.write(b);
        }
        o_nm_strm.write(0);
#ifndef __SYNTHESIS__
        std::cout << "o_nm_strm.size(): " << o_nm_strm.size() << std::endl;
#endif

        // write number of resulting rows to metaout
        // dout_meta[0].range(71, 8) = n;
        rnm_strm.write(n);
    }
}

// one time burst write for 1 col
template <int elem_size, int vec_len>
void bwrite(const int nm,
            const int bnm,
            const int burst_len,
            hls::stream<ap_uint<elem_size * vec_len> >& i_strm,
            ap_uint<elem_size * vec_len>* ptr) {
#pragma HLS inline off
    for (int n = 0; n < nm; ++n) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 32 max = 32
        ap_uint<elem_size* vec_len> out = i_strm.read();
        ptr[bnm * burst_len + n] = out;
    }
}

template <int burst_len, int elem_size, int vec_len, int col_num>
void burst_write_hj(hls::stream<bool>& bp_flag_strm,
                    hls::stream<ap_uint<elem_size * vec_len> > in_strm[col_num], // 512-bit out
                    hls::stream<ap_uint<8> >& i_nm_strm,
                    hls::stream<ap_uint<32> >& i_rnm_strm,
                    hls::stream<ap_uint<8> >& i_wr_en_strm,
                    ap_uint<elem_size * vec_len>* ptr0,
                    ap_uint<elem_size * vec_len>* ptr1,
                    ap_uint<elem_size * vec_len>* ptr2,
                    ap_uint<elem_size * vec_len>* ptr3,
                    ap_uint<512>* dout_meta) {
    bool bp_flag = bp_flag_strm.read();
    ap_uint<8> wr_en = i_wr_en_strm.read();

    if (bp_flag) {
        ap_uint<8> nm = i_nm_strm.read();
        int bnm = 0;
        while (nm > 0) {
            // write each column one burst
            if (wr_en[0]) bwrite<elem_size, vec_len>(nm, bnm, burst_len, in_strm[0], ptr0);
            if (wr_en[1]) bwrite<elem_size, vec_len>(nm, bnm, burst_len, in_strm[1], ptr1);
            if (wr_en[2]) bwrite<elem_size, vec_len>(nm, bnm, burst_len, in_strm[2], ptr2);
            if (wr_en[3]) bwrite<elem_size, vec_len>(nm, bnm, burst_len, in_strm[3], ptr3);
            bnm++;
            nm = i_nm_strm.read();
        }

        // write number of resulting rows to metaout
        ap_uint<32> rnm = i_rnm_strm.read();
        dout_meta[0].range(71, 8) = rnm;
#ifdef USER_DEBUG
        std::cout << "total hj result row number: " << rnm << std::endl;
#endif
    }
}

template <int burst_len, int elem_size, int vec_len, int col_num>
void write_table_hj_core(hls::stream<bool>& bp_flag_strm,
                         hls::stream<ap_uint<elem_size> > in_strm[col_num],
                         hls::stream<bool>& e_in_strm,
                         hls::stream<ap_uint<8> >& write_out_en_strm,
                         ap_uint<elem_size * vec_len>* ptr0,
                         ap_uint<elem_size * vec_len>* ptr1,
                         ap_uint<elem_size * vec_len>* ptr2,
                         ap_uint<elem_size * vec_len>* ptr3,
                         ap_uint<512>* dout_meta) {
#pragma HLS dataflow

    hls::stream<ap_uint<8> > write_out_en_mid_strm;
#pragma HLS stream variable = write_out_en_mid_strm depth = 2
    hls::stream<ap_uint<elem_size * vec_len> > mid_vecs_strm[col_num];
#pragma HLS stream variable = mid_vecs_strm depth = 64
    hls::stream<ap_uint<8> > mid_nm_strm;
#pragma HLS stream variable = mid_nm_strm depth = 4
    hls::stream<ap_uint<32> > mid_rnm_strm;
#pragma HLS stream variable = mid_rnm_strm depth = 2
    hls::stream<bool> mid_bp_flag_strm;
#pragma HLS stream variable = mid_bp_flag_strm depth = 2

    count_for_burst<burst_len, elem_size, vec_len, col_num>(bp_flag_strm, mid_bp_flag_strm, in_strm, e_in_strm,
                                                            mid_vecs_strm, mid_nm_strm, mid_rnm_strm, write_out_en_strm,
                                                            write_out_en_mid_strm);
#ifdef USER_DEBUG
    std::cout << "mid_nm_strm.size() = " << mid_nm_strm.size() << std::endl;
    for (int c = 0; c < col_num; c++) {
        std::cout << "mid_vecs_strm[" << c << "].size() = " << mid_vecs_strm[c].size() << std::endl;
    }
#endif
    burst_write_hj<burst_len, elem_size, vec_len, col_num>(mid_bp_flag_strm, mid_vecs_strm, mid_nm_strm, mid_rnm_strm,
                                                           write_out_en_mid_strm, ptr0, ptr1, ptr2, ptr3, dout_meta);
}

// gqeJoin used write data out
template <int burst_len, int elem_size, int vec_len, int col_num>
void write_table_hj(hls::stream<ap_uint<6> >& join_cfg_strm,
                    hls::stream<ap_uint<elem_size> > in_strm[col_num],
                    hls::stream<bool>& e_in_strm,
                    hls::stream<ap_uint<8> >& write_out_cfg_strm,
                    ap_uint<elem_size * vec_len>* ptr0,
                    ap_uint<elem_size * vec_len>* ptr1,
                    ap_uint<elem_size * vec_len>* ptr2,
                    ap_uint<elem_size * vec_len>* ptr3,
                    ap_uint<512>* dout_meta) {
    ap_uint<6> join_cfg = join_cfg_strm.read();
    bool bp_flag = join_cfg[5];

    hls::stream<ap_uint<8> > write_out_en_strm;
#pragma HLS stream variable = write_out_en_strm depth = 2

    hls::stream<bool> bp_flag_strm;
#pragma HLS stream variable = bp_flag_strm depth = 2
    // if append mode is on
    write_prepare<elem_size, vec_len, col_num>(bp_flag, bp_flag_strm, write_out_cfg_strm, write_out_en_strm);

    write_table_hj_core<burst_len, elem_size, vec_len, col_num>(bp_flag_strm, in_strm, e_in_strm, write_out_en_strm,
                                                                ptr0, ptr1, ptr2, ptr3, dout_meta);
}

//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------

// for partition kernel below
// count and prepare data for burst write
template <int elem_size, int vec_len, int col_num, int hash_wh>
void countForBurstPart(hls::stream<ap_uint<elem_size*(1 << hash_wh)> > i_post_Agg[col_num],
                       hls::stream<int>& i_partition_size_strm,
                       hls::stream<ap_uint<8> >& i_wr_cfg_strm,
                       hls::stream<int>& i_bit_num_strm,
                       hls::stream<ap_uint<10> >& i_nm_strm,
                       hls::stream<ap_uint<12> >& i_bkpu_num_strm,

                       hls::stream<ap_uint<elem_size * vec_len> > o_post_Agg[col_num],
                       hls::stream<ap_uint<8> > o_nm_512_strm[3],
                       hls::stream<ap_uint<8> > o_wr_cfg_strm[3],
                       hls::stream<int> o_p_base_addr_strm[3],
                       hls::stream<int> o_burst_strm_cnt_reg_strm[3],
                       hls::stream<int>& o_per_part_nm_strm) {
    enum { PU = 1 << hash_wh };
    // read out the partition size
    int PARTITION_SIZE = i_partition_size_strm.read();

#if !defined(__SYNTHESIS__) && XDEBUG == 1
    std::cout << "Partition size:" << PARTITION_SIZE << std::endl;
    long long cnt = 0;
#endif

    const int sz = elem_size;
    ap_uint<2> pu_idx;
    ap_uint<10> bk_idx;
    int n = 0; // nrow count
    int r = 0; // element count in one 512b
    int b = 0; // burst lentg count
    ap_uint<sz * vec_len> vecs[col_num];
#pragma HLS array_partition variable = vecs complete
    int nrow_cnt[512];
#pragma HLS resource variable = nrow_cnt core = RAM_S2P_BRAM
    for (int i = 0; i < 512; i++) {
#pragma HLS PIPELINE II = 1
        nrow_cnt[i] = 0;
    }

    ap_uint<8> write_out_cfg = i_wr_cfg_strm.read();
#ifndef __SYNTHESIS__
    std::cout << "write_out_cfg: " << (int)write_out_cfg << std::endl;
#endif
    o_wr_cfg_strm[0].write(write_out_cfg);
    o_wr_cfg_strm[1].write(write_out_cfg);
    o_wr_cfg_strm[2].write(write_out_cfg);
    const int bit_num_org = i_bit_num_strm.read();
    // get the partition num
    const int partition_num = 1 << bit_num_org;
    // get the bit num after minus PU idx 2 bits
    const int bit_num = bit_num_org - hash_wh;
    // get the part num in each PU
    const int part_num_per_PU = 1 << bit_num;

    ap_uint<10> nm = i_nm_strm.read();
    while (nm != 0) {
        (pu_idx, bk_idx) = i_bkpu_num_strm.read();

        // the partition idx among all PUs
        ap_uint<16> location = pu_idx * part_num_per_PU + bk_idx;

        // get write ddr base addr
        int p_base_addr = PARTITION_SIZE * location;
        o_p_base_addr_strm[0].write(p_base_addr);
        o_p_base_addr_strm[1].write(p_base_addr);
        o_p_base_addr_strm[2].write(p_base_addr);

        // get the offset after nm times of write
        int burst_step_cnt_reg = (nrow_cnt[location] + vec_len - 1) / vec_len;
        o_burst_strm_cnt_reg_strm[0].write(burst_step_cnt_reg);
        o_burst_strm_cnt_reg_strm[1].write(burst_step_cnt_reg);
        o_burst_strm_cnt_reg_strm[2].write(burst_step_cnt_reg);

        // convert dat number from nm * 32bit to nm/16 * 512bit
        const ap_uint<8> burst_len = (nm + vec_len - 1) / vec_len;
        for (int i = 0; i < (nm + PU - 1) / PU; i++) {
#pragma HLS pipeline II = 1
            for (int c = 0; c < col_num; ++c) {
#pragma HLS unroll
                ap_uint<sz* PU> t = i_post_Agg[c].read();
#ifdef USER_DEBUG
                std::cout << "t.range(63,0): " << t.range(63, 0) << std::endl;
                std::cout << "t.range(63,0): " << t.range(127, 64) << std::endl;
                std::cout << "t.range(63,0): " << t.range(191, 128) << std::endl;
                std::cout << "sz: " << sz << std::endl;

                std::cout << "t: " << t << std::endl;
#endif
                vecs[c].range(sz * PU * (vec_len / PU - 1) - 1, 0) = vecs[c].range(sz * vec_len - 1, sz * PU);
                vecs[c].range(sz * vec_len - 1, sz * PU * (vec_len / PU - 1)) = t;
            }

            if (r == vec_len - PU) {
                for (int c = 0; c < col_num; ++c) {
#pragma HLS unroll
                    if (write_out_cfg[c]) o_post_Agg[c].write(vecs[c]);
                }
                if (b == burst_len - 1) {
                    o_nm_512_strm[0].write(burst_len);
                    o_nm_512_strm[1].write(burst_len);
                    o_nm_512_strm[2].write(burst_len);
                    b = 0;
                } else {
                    ++b;
                }
                r = 0;
            } else {
                r = r + PU;
            }
        }

        if (r != 0) {
            // handle incomplete vecs
            for (; r < vec_len; r = r + PU) {
                for (int c = 0; c < col_num; ++c) {
#pragma HLS unroll
                    vecs[c].range(sz * PU * (vec_len / PU - 1) - 1, 0) = vecs[c].range(sz * vec_len - 1, sz * PU);
                    vecs[c].range(sz * vec_len - 1, sz * PU * (vec_len / PU - 1)) = 0;
                }
            }

            for (int c = 0; c < col_num; ++c) {
#pragma HLS unroll
                if (write_out_cfg[c]) o_post_Agg[c].write(vecs[c]);
            }
            ++b;
        }

        if (b != 0) {
            o_nm_512_strm[0].write(b);
            o_nm_512_strm[1].write(b);
            o_nm_512_strm[2].write(b);
        }

        b = 0;
        r = 0;

        nrow_cnt[location] += nm;

        nm = i_nm_strm.read();
    }

    o_nm_512_strm[0].write(0);
    o_nm_512_strm[1].write(0);
    o_nm_512_strm[2].write(0);
    // output nrow of each partition
    for (int i = 0; i < partition_num; i++) {
#pragma HLS PIPELINE II = 1
        o_per_part_nm_strm.write(nrow_cnt[i]);
    }
}

// one time burst write for 1 col
template <int elem_size, int vec_len>
void bwrite_part(ap_uint<8> nm,
                 const int part_base_addr,
                 const int burst_step_cnt_reg,
                 hls::stream<ap_uint<elem_size * vec_len> >& i_strm,
                 ap_uint<elem_size * vec_len>* ptr) {
    for (int n = 0; n < nm; ++n) {
#pragma HLS PIPELINE II = 1
        ap_uint<elem_size* vec_len> out = i_strm.read();
        //#ifndef __SYNTHESIS__
        //        for (int i = 0; i < vec_len; i++) {
        //            std::cout << "i: " << i << ", " << out.range(elem_size * (i + 1) - 1, elem_size * i) << std::endl;
        //        }
        //#endif
        ptr[part_base_addr + burst_step_cnt_reg + n] = out;
    }
}
// for part, burst write to col 0,3,6, using axi0
template <int elem_size, int vec_len, int col_num>
void burst_write_col0(hls::stream<ap_uint<elem_size * vec_len> >& i_strm0,
                      hls::stream<ap_uint<8> >& i_wr_cfg_strm,
                      hls::stream<ap_uint<8> >& i_nm_512_strm,
                      hls::stream<int>& i_p_base_addr_strm,
                      hls::stream<int>& i_burst_step_cnt_reg_strm,

                      ap_uint<elem_size * vec_len>* ptr0) {
    ap_uint<32> write_out_cfg = i_wr_cfg_strm.read();
    ap_uint<8> nm = i_nm_512_strm.read();
    while (nm != 0) {
        int p_base_addr = i_p_base_addr_strm.read();
        int burst_step_cnt_reg = i_burst_step_cnt_reg_strm.read();

        if (write_out_cfg[0]) bwrite_part<elem_size, vec_len>(nm, p_base_addr, burst_step_cnt_reg, i_strm0, ptr0);

        nm = i_nm_512_strm.read();
    }
}
template <int elem_size, int vec_len, int col_num>
void burst_write_col1(hls::stream<ap_uint<elem_size * vec_len> >& i_strm1,
                      hls::stream<ap_uint<8> >& i_wr_cfg_strm,
                      hls::stream<ap_uint<8> >& i_nm_512_strm,
                      hls::stream<int>& i_p_base_addr_strm,
                      hls::stream<int>& i_burst_step_cnt_reg_strm,

                      ap_uint<elem_size * vec_len>* ptr1) {
    ap_uint<32> write_out_cfg = i_wr_cfg_strm.read();
    ap_uint<8> nm = i_nm_512_strm.read();
    while (nm != 0) {
        int p_base_addr = i_p_base_addr_strm.read();
        int burst_step_cnt_reg = i_burst_step_cnt_reg_strm.read();

        if (write_out_cfg[1]) bwrite_part<elem_size, vec_len>(nm, p_base_addr, burst_step_cnt_reg, i_strm1, ptr1);

        nm = i_nm_512_strm.read();
    }
}

template <int elem_size, int vec_len, int col_num>
void burst_write_col2(hls::stream<ap_uint<elem_size * vec_len> >& i_strm2,
                      hls::stream<ap_uint<8> >& i_wr_cfg_strm,
                      hls::stream<ap_uint<8> >& i_nm_512_strm,
                      hls::stream<int>& i_p_base_addr_strm,
                      hls::stream<int>& i_burst_step_cnt_reg_strm,

                      ap_uint<elem_size * vec_len>* ptr2) {
    ap_uint<32> write_out_cfg = i_wr_cfg_strm.read();
    ap_uint<8> nm = i_nm_512_strm.read();
    while (nm != 0) {
        int p_base_addr = i_p_base_addr_strm.read();
        int burst_step_cnt_reg = i_burst_step_cnt_reg_strm.read();

        if (write_out_cfg[2]) bwrite_part<elem_size, vec_len>(nm, p_base_addr, burst_step_cnt_reg, i_strm2, ptr2);

        nm = i_nm_512_strm.read();
    }
}

template <int elem_size, int vec_len, int col_num, int hash_wh>
void write_table_part_core(hls::stream<ap_uint<elem_size*(1 << hash_wh)> > post_Agg[col_num],
                           hls::stream<int>& i_partition_size_strm,
                           hls::stream<ap_uint<8> >& i_wr_cfg_strm,
                           hls::stream<int>& i_bit_num_strm,
                           hls::stream<ap_uint<10> >& i_nm_strm,
                           hls::stream<ap_uint<12> >& i_bkpu_num_strm,

                           hls::stream<int>& o_per_part_nm_strm,
                           ap_uint<elem_size * vec_len>* ptr0,
                           ap_uint<elem_size * vec_len>* ptr1,
                           ap_uint<elem_size * vec_len>* ptr2) {
    hls::stream<ap_uint<elem_size * vec_len> > mid_post_Agg[col_num];
#pragma HLS stream variable = mid_post_Agg depth = 64
#pragma HLS resource variable = mid_post_Agg core = FIFO_LUTRAM

    hls::stream<ap_uint<8> > mid_nm_512_strm[3];
#pragma HLS array_partition variable = mid_nm_512_strm complete
#pragma HLS stream variable = mid_nm_512_strm depth = 8

    hls::stream<ap_uint<8> > mid_write_cfg_strm[3];
#pragma HLS array_partition variable = mid_write_cfg_strm complete
#pragma HLS stream variable = mid_write_cfg_strm depth = 4

    hls::stream<int> mid_p_base_addr_strm[3];
#pragma HLS array_partition variable = mid_p_base_addr_strm complete
#pragma HLS stream variable = mid_p_base_addr_strm depth = 4

    hls::stream<int> mid_burst_step_cnt_reg_strm[3];
#pragma HLS array_partition variable = mid_burst_step_cnt_reg_strm complete
#pragma HLS stream variable = mid_burst_step_cnt_reg_strm depth = 4

#pragma HLS dataflow

    countForBurstPart<elem_size, vec_len, col_num, hash_wh>(
        post_Agg, i_partition_size_strm, i_wr_cfg_strm, i_bit_num_strm, i_nm_strm, i_bkpu_num_strm, mid_post_Agg,
        mid_nm_512_strm, mid_write_cfg_strm, mid_p_base_addr_strm, mid_burst_step_cnt_reg_strm, o_per_part_nm_strm);

    burst_write_col0<elem_size, vec_len, col_num>(mid_post_Agg[0], mid_write_cfg_strm[0], mid_nm_512_strm[0],
                                                  mid_p_base_addr_strm[0], mid_burst_step_cnt_reg_strm[0], ptr0);
    burst_write_col1<elem_size, vec_len, col_num>(mid_post_Agg[1], mid_write_cfg_strm[1], mid_nm_512_strm[1],
                                                  mid_p_base_addr_strm[1], mid_burst_step_cnt_reg_strm[1], ptr1);
    burst_write_col2<elem_size, vec_len, col_num>(mid_post_Agg[2], mid_write_cfg_strm[2], mid_nm_512_strm[2],
                                                  mid_p_base_addr_strm[2], mid_burst_step_cnt_reg_strm[2], ptr2);
}

// write out for partition
template <int elem_size, int vec_len, int col_num, int hash_wh>
void write_table_part(hls::stream<ap_uint<elem_size*(1 << hash_wh)> > post_Agg[col_num],
                      hls::stream<ap_uint<8> >& i_wr_cfg_strm,
                      hls::stream<int>& i_bit_num_strm,
                      hls::stream<ap_uint<10> >& i_nm_strm,
                      hls::stream<ap_uint<12> >& i_bkpu_num_strm,

                      ap_uint<elem_size * vec_len>* ptr0,
                      ap_uint<elem_size * vec_len>* ptr1,
                      ap_uint<elem_size * vec_len>* ptr2,
                      ap_uint<512>* dout_meta) {
    hls::stream<int> mid_partition_size_strm;
#pragma HLS stream variable = mid_partition_size_strm depth = 2
    hls::stream<int> mid_bit_num_strm;
#pragma HLS stream variable = mid_bit_num_strm depth = 2
    hls::stream<int> mid_per_part_nm_strm;
#pragma HLS stream variable = mid_per_part_nm_strm depth = 256
#pragma HLS resource variable = mid_per_part_nm_strm core = FIFO_BRAM
    // read out the partition size
    int PARTITION_SIZE = dout_meta[0].range(135, 104);
    mid_partition_size_strm.write(PARTITION_SIZE);

    // read in the bit num
    const int bit_num = i_bit_num_strm.read();
    mid_bit_num_strm.write(bit_num);
    // get the partition num
    const int partition_num = 1 << bit_num;

    write_table_part_core<elem_size, vec_len, col_num, hash_wh>(post_Agg, mid_partition_size_strm, i_wr_cfg_strm,
                                                                mid_bit_num_strm, i_nm_strm, i_bkpu_num_strm,
                                                                mid_per_part_nm_strm, ptr0, ptr1, ptr2);
//----------update meta------------
#if !defined(__SYNTHESIS__) && XDEBUG == 1
    int cnt = 0;
#endif
    // define the buffer used to save nrow of each partition
    ap_uint<512> metaout[16];
#pragma HLS resource variable = metaout core = RAM_S2P_BRAM
// update tout meta
FINAL_WRITE_HEAD_LOOP:
    for (int i = 0; i < partition_num; i++) {
#pragma HLS PIPELINE II = 1
        int rnm = mid_per_part_nm_strm.read();
        ap_uint<8> idx = i / 16;
        ap_uint<8> idx_ex = i % 16;
        metaout[idx].range(32 * (idx_ex + 1) - 1, 32 * idx_ex) = rnm;

#if !defined(__SYNTHESIS__) && XDEBUG == 1
        std::cout << "P" << i << "\twrite out row number = " << rnm << std::endl;
        cnt += rnm;
#endif
    }
    // write nrow of each partition to dout_meta, ddr
    for (int i = 0; i < (partition_num + 15) / 16; i++) {
#pragma HLS PIPELINE II = 1
        dout_meta[8 + i] = metaout[i];
    }
#if !defined(__SYNTHESIS__) && XDEBUG == 1
    std::cout << "Total number of write-out row: " << cnt << std::endl;
#endif
}

} // namespace gqe
} // namespace database
} // namespace xf

#endif
