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

#ifndef __SYNTHESIS__
#include <iostream>
#endif

#include <ap_int.h>
#include <hls_stream.h>

#include "xf_database/utils.hpp"
#include "xf_utils_hw/stream_shuffle.hpp"

namespace xf {
namespace database {
namespace gqe {

template <int burst_len, int elem_size, int vec_len, int col_num>
void countForBurst(hls::stream<ap_uint<elem_size> > i_post_Agg[col_num],
                   hls::stream<bool>& i_e_strm,
                   hls::stream<ap_uint<elem_size * vec_len> > o_post_Agg[col_num],
                   hls::stream<ap_uint<8> >& o_nm_strm,
                   hls::stream<ap_uint<32> >& rnm_strm,
                   hls::stream<ap_uint<32> >& wr_cfg_istrm,
                   hls::stream<ap_uint<32> >& wr_cfg_ostrm) {
    const int sz = elem_size;
    bool e = i_e_strm.read();
    ap_uint<32> write_out_cfg = wr_cfg_istrm.read();
    wr_cfg_ostrm.write(write_out_cfg);
#if !defined(__SYNTHESIS__) && XDEBUG == 1
    std::cout << std::hex << "write out config" << write_out_cfg << std::endl;
#endif

    ap_uint<sz * vec_len> vecs[col_num];
    int n = 0; // nrow count
    int r = 0; // element count in one 512b
    int b = 0; // burst lentg count
#pragma HLS array_partition variable = vecs complete
    while (!e) {
#pragma HLS pipeline II = 1
        e = i_e_strm.read();
        ++n;
        for (int c = 0; c < col_num; ++c) {
#pragma HLS unroll
            ap_uint<sz> t = i_post_Agg[c].read();
            vecs[c].range(sz * (vec_len - 1) - 1, 0) = vecs[c].range(sz * vec_len - 1, sz);
            vecs[c].range(sz * vec_len - 1, sz * (vec_len - 1)) = t;
            // vecs[c].range(sz * (r + 1) - 1, sz * r) = t;
        }
        if (r == vec_len - 1) {
            for (int c = 0; c < col_num; ++c) {
#pragma HLS unroll
                if (write_out_cfg[c]) {
                    o_post_Agg[c].write(vecs[c]);
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
    if (r != 0) {
        // handle incomplete vecs
        for (; r < vec_len; ++r) {
#pragma HLS unroll
            for (int c = 0; c < col_num; ++c) {
#pragma HLS unroll
                vecs[c].range(sz * (vec_len - 1) - 1, 0) = vecs[c].range(sz * vec_len - 1, sz);
                vecs[c].range(sz * vec_len - 1, sz * (vec_len - 1)) = 0;
                // vecs[c].range(sz * (r + 1) - 1, sz * r) = 0;
            }
        }
        for (int c = 0; c < col_num; ++c) {
#pragma HLS unroll
            if (write_out_cfg[c]) {
                o_post_Agg[c].write(vecs[c]);
            }
        }
        ++b;
    }
    if (b != 0) {
        XF_DATABASE_ASSERT(b <= burst_len);
        o_nm_strm.write(b);
        o_nm_strm.write(0);
    } else {
        o_nm_strm.write(0);
    }
    rnm_strm.write(n);
#if !defined(__SYNTHESIS__) && XDEBUG == 1
    for (int c = 0; c < col_num; ++c) {
        printf("write out col %d size is %d\n", c, o_post_Agg[c].size());
    }
    printf("write out nm stream size is %d\n", o_nm_strm.size());
#endif
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
        ap_uint<elem_size* vec_len> out = i_strm.read();
        ptr[bnm * burst_len + n] = out;
    }
}

template <int burst_len, int elem_size, int vec_len, int col_num>
void burstWriteAggr(hls::stream<ap_uint<elem_size * vec_len> > i_strm[col_num],
                    ap_uint<elem_size * vec_len>* ptr0,
                    ap_uint<elem_size * vec_len>* ptr1,
                    ap_uint<elem_size * vec_len>* ptr2,
                    ap_uint<elem_size * vec_len>* ptr3,
                    ap_uint<elem_size * vec_len>* ptr4,
                    ap_uint<elem_size * vec_len>* ptr5,
                    ap_uint<elem_size * vec_len>* ptr6,
                    ap_uint<elem_size * vec_len>* ptr7,
                    ap_uint<elem_size * vec_len>* ptr8,
                    ap_uint<elem_size * vec_len>* ptr9,
                    ap_uint<elem_size * vec_len>* ptr10,
                    ap_uint<elem_size * vec_len>* ptr11,
                    ap_uint<elem_size * vec_len>* ptr12,
                    ap_uint<elem_size * vec_len>* ptr13,
                    ap_uint<elem_size * vec_len>* ptr14,
                    ap_uint<elem_size * vec_len>* ptr15,
                    ap_uint<512>* tout_meta,
                    hls::stream<ap_uint<8> >& nm_strm,
                    hls::stream<ap_uint<32> >& rnm_strm,
                    hls::stream<ap_uint<32> >& write_out_cfg_strm) {
    // record the burst nubmer
    unsigned bnm = 0;
    // write_out_cfg, write out of bypass
    ap_uint<32> write_out_cfg = write_out_cfg_strm.read();

    int nm = nm_strm.read();
    while (nm) {
        // write each col one burst
        if (write_out_cfg[0]) bwrite<elem_size, vec_len>(nm, bnm, burst_len, i_strm[0], ptr0);
        if (write_out_cfg[1]) bwrite<elem_size, vec_len>(nm, bnm, burst_len, i_strm[1], ptr1);
        if (write_out_cfg[2]) bwrite<elem_size, vec_len>(nm, bnm, burst_len, i_strm[2], ptr2);
        if (write_out_cfg[3]) bwrite<elem_size, vec_len>(nm, bnm, burst_len, i_strm[3], ptr3);
        if (write_out_cfg[4]) bwrite<elem_size, vec_len>(nm, bnm, burst_len, i_strm[4], ptr4);
        if (write_out_cfg[5]) bwrite<elem_size, vec_len>(nm, bnm, burst_len, i_strm[5], ptr5);
        if (write_out_cfg[6]) bwrite<elem_size, vec_len>(nm, bnm, burst_len, i_strm[6], ptr6);
        if (write_out_cfg[7]) bwrite<elem_size, vec_len>(nm, bnm, burst_len, i_strm[7], ptr7);
        if (write_out_cfg[8]) bwrite<elem_size, vec_len>(nm, bnm, burst_len, i_strm[8], ptr8);
        if (write_out_cfg[9]) bwrite<elem_size, vec_len>(nm, bnm, burst_len, i_strm[9], ptr9);
        if (write_out_cfg[10]) bwrite<elem_size, vec_len>(nm, bnm, burst_len, i_strm[10], ptr10);
        if (write_out_cfg[11]) bwrite<elem_size, vec_len>(nm, bnm, burst_len, i_strm[11], ptr11);
        if (write_out_cfg[12]) bwrite<elem_size, vec_len>(nm, bnm, burst_len, i_strm[12], ptr12);
        if (write_out_cfg[13]) bwrite<elem_size, vec_len>(nm, bnm, burst_len, i_strm[13], ptr13);
        if (write_out_cfg[14]) bwrite<elem_size, vec_len>(nm, bnm, burst_len, i_strm[14], ptr14);
        if (write_out_cfg[15]) bwrite<elem_size, vec_len>(nm, bnm, burst_len, i_strm[15], ptr15);
        bnm++;
        nm = nm_strm.read();
    }

    ap_uint<32> rnm = rnm_strm.read();
#ifndef __SYNTHESIS__
    std::cout << std::dec << "write out row=" << rnm.to_int() << std::endl;
#endif
#if !defined(__SYNTHESIS__) && XDEBUG == 1
    std::cout << std::hex << "write out config" << write_out_cfg << std::endl;
    std::cout << std::dec << "write out row=" << rnm.to_int() << " col_nm=" << wcol << std::endl;
#endif
    // write number of resulting rows to metaout
    tout_meta[0].range(71, 8) = rnm;
}

template <int burst_len, int elem_size, int vec_len, int col_num>
void writeTableAggr(hls::stream<ap_uint<elem_size> > post_Agg[col_num],
                    hls::stream<bool>& e_post_Agg,
                    ap_uint<elem_size * vec_len>* ptr0,
                    ap_uint<elem_size * vec_len>* ptr1,
                    ap_uint<elem_size * vec_len>* ptr2,
                    ap_uint<elem_size * vec_len>* ptr3,
                    ap_uint<elem_size * vec_len>* ptr4,
                    ap_uint<elem_size * vec_len>* ptr5,
                    ap_uint<elem_size * vec_len>* ptr6,
                    ap_uint<elem_size * vec_len>* ptr7,
                    ap_uint<elem_size * vec_len>* ptr8,
                    ap_uint<elem_size * vec_len>* ptr9,
                    ap_uint<elem_size * vec_len>* ptr10,
                    ap_uint<elem_size * vec_len>* ptr11,
                    ap_uint<elem_size * vec_len>* ptr12,
                    ap_uint<elem_size * vec_len>* ptr13,
                    ap_uint<elem_size * vec_len>* ptr14,
                    ap_uint<elem_size * vec_len>* ptr15,
                    ap_uint<512>* tout_meta,
                    hls::stream<ap_uint<32> >& write_out_cfg_strm) {
    const int k_fifo_buf = burst_len * 2;

    hls::stream<ap_uint<elem_size * vec_len> > m_post_Agg[col_num];
#pragma HLS array_partition variable = m_post_Agg dim = 0
#pragma HLS stream variable = m_post_Agg depth = k_fifo_buf
#pragma HLS bind_storage variable = m_post_Agg type = fifo impl = lutram
    hls::stream<ap_uint<8> > nm_strm;
#pragma HLS stream variable = nm_strm depth = 2
    hls::stream<ap_uint<32> > rnm_strm;
#pragma HLS stream variable = rnm_strm depth = 4
    hls::stream<ap_uint<32> > mid_wr_cfg_strm;
#pragma HLS stream variable = mid_wr_cfg_strm depth = 1

#pragma HLS dataflow

    countForBurst<burst_len, elem_size, vec_len, col_num>(post_Agg, e_post_Agg, m_post_Agg, nm_strm, rnm_strm,
                                                          write_out_cfg_strm, mid_wr_cfg_strm);

#if !defined(__SYNTHESIS__) && XDEBUG == 1
    printf("\npost_Agg: %ld, e_post_Agg:%ld, m_post_Agg:%ld, nm_strm:%ld, rnm_strm:%ld\n\n", post_Agg[0].size(),
           e_post_Agg.size(), m_post_Agg[0].size(), nm_strm.size(), rnm_strm.size());
#endif

    burstWriteAggr<burst_len, elem_size, vec_len, col_num>(m_post_Agg, ptr0, ptr1, ptr2, ptr3, ptr4, ptr5, ptr6, ptr7,
                                                           ptr8, ptr9, ptr10, ptr11, ptr12, ptr13, ptr14, ptr15,
                                                           tout_meta, nm_strm, rnm_strm, mid_wr_cfg_strm);
}
// for part
// count and prepare data for burst write
template <int elem_size, int vec_len, int col_num, int hash_wh>
void countForBurstPart(hls::stream<ap_uint<elem_size*(1 << hash_wh)> > i_post_Agg[col_num],
                       hls::stream<int>& i_partition_size_strm,
                       hls::stream<ap_uint<32> >& i_wr_cfg_strm,
                       hls::stream<int>& i_bit_num_strm,
                       hls::stream<ap_uint<10> >& i_nm_strm,
                       hls::stream<ap_uint<12> >& i_bkpu_num_strm,

                       hls::stream<ap_uint<elem_size * vec_len> > o_post_Agg[col_num],
                       hls::stream<ap_uint<8> > o_nm_512_strm[3],
                       hls::stream<ap_uint<32> > o_wr_cfg_strm[3],
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
#pragma HLS bind_storage variable = nrow_cnt type = ram_s2p impl = bram
    for (int i = 0; i < 512; i++) {
#pragma HLS PIPELINE II = 1
        nrow_cnt[i] = 0;
    }

    ap_uint<32> write_out_cfg = i_wr_cfg_strm.read();
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
        // int burst_step_cnt_reg = nrow_cnt[location] / vec_len;
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
        ptr[part_base_addr + burst_step_cnt_reg + n] = out;
    }
}
// for part, burst write to col 0,3,6, using axi0
template <int elem_size, int vec_len, int col_num>
void burstwrite_col036(hls::stream<ap_uint<elem_size * vec_len> >& i_strm0,
                       hls::stream<ap_uint<elem_size * vec_len> >& i_strm1,
                       hls::stream<ap_uint<elem_size * vec_len> >& i_strm2,
                       hls::stream<ap_uint<32> >& i_wr_cfg_strm,
                       hls::stream<ap_uint<8> >& i_nm_512_strm,
                       hls::stream<int>& i_p_base_addr_strm,
                       hls::stream<int>& i_burst_step_cnt_reg_strm,

                       ap_uint<elem_size * vec_len>* ptr0,
                       ap_uint<elem_size * vec_len>* ptr1,
                       ap_uint<elem_size * vec_len>* ptr2) {
    ap_uint<32> write_out_cfg = i_wr_cfg_strm.read();
    ap_uint<8> nm = i_nm_512_strm.read();
    while (nm != 0) {
        int p_base_addr = i_p_base_addr_strm.read();
        int burst_step_cnt_reg = i_burst_step_cnt_reg_strm.read();

        if (write_out_cfg[0]) bwrite_part<elem_size, vec_len>(nm, p_base_addr, burst_step_cnt_reg, i_strm0, ptr0);
        if (write_out_cfg[3]) bwrite_part<elem_size, vec_len>(nm, p_base_addr, burst_step_cnt_reg, i_strm1, ptr1);
        if (write_out_cfg[6]) bwrite_part<elem_size, vec_len>(nm, p_base_addr, burst_step_cnt_reg, i_strm2, ptr2);

        nm = i_nm_512_strm.read();
    }
}
// for part, burst write to col 1,4,7, using axi1
template <int elem_size, int vec_len, int col_num>
void burstwrite_col147(hls::stream<ap_uint<elem_size * vec_len> >& i_strm0,
                       hls::stream<ap_uint<elem_size * vec_len> >& i_strm1,
                       hls::stream<ap_uint<elem_size * vec_len> >& i_strm2,
                       hls::stream<ap_uint<32> >& i_wr_cfg_strm,
                       hls::stream<ap_uint<8> >& i_nm_512_strm,
                       hls::stream<int>& i_p_base_addr_strm,
                       hls::stream<int>& i_burst_step_cnt_reg_strm,

                       ap_uint<elem_size * vec_len>* ptr0,
                       ap_uint<elem_size * vec_len>* ptr1,
                       ap_uint<elem_size * vec_len>* ptr2) {
    ap_uint<32> write_out_cfg = i_wr_cfg_strm.read();
    ap_uint<8> nm = i_nm_512_strm.read();
    while (nm != 0) {
        int p_base_addr = i_p_base_addr_strm.read();
        int burst_step_cnt_reg = i_burst_step_cnt_reg_strm.read();

        if (write_out_cfg[1]) bwrite_part<elem_size, vec_len>(nm, p_base_addr, burst_step_cnt_reg, i_strm0, ptr0);
        if (write_out_cfg[4]) bwrite_part<elem_size, vec_len>(nm, p_base_addr, burst_step_cnt_reg, i_strm1, ptr1);
        if (write_out_cfg[7]) bwrite_part<elem_size, vec_len>(nm, p_base_addr, burst_step_cnt_reg, i_strm2, ptr2);

        nm = i_nm_512_strm.read();
    }
}
// for part, burst write to col 2,5, using axi2
template <int elem_size, int vec_len, int col_num>
void burstwrite_col25(hls::stream<ap_uint<elem_size * vec_len> >& i_strm0,
                      hls::stream<ap_uint<elem_size * vec_len> >& i_strm1,
                      hls::stream<ap_uint<32> >& i_wr_cfg_strm,
                      hls::stream<ap_uint<8> >& i_nm_512_strm,
                      hls::stream<int>& i_p_base_addr_strm,
                      hls::stream<int>& i_burst_step_cnt_reg_strm,

                      ap_uint<elem_size * vec_len>* ptr0,
                      ap_uint<elem_size * vec_len>* ptr1) {
    ap_uint<32> write_out_cfg = i_wr_cfg_strm.read();
    ap_uint<8> nm = i_nm_512_strm.read();
    while (nm != 0) {
        int p_base_addr = i_p_base_addr_strm.read();
        int burst_step_cnt_reg = i_burst_step_cnt_reg_strm.read();

        if (write_out_cfg[2]) bwrite_part<elem_size, vec_len>(nm, p_base_addr, burst_step_cnt_reg, i_strm0, ptr0);
        if (write_out_cfg[5]) bwrite_part<elem_size, vec_len>(nm, p_base_addr, burst_step_cnt_reg, i_strm1, ptr1);

        nm = i_nm_512_strm.read();
    }
}

// write out for partition
template <int elem_size, int vec_len, int col_num, int hash_wh>
void writeTablePart(hls::stream<ap_uint<elem_size*(1 << hash_wh)> > post_Agg[col_num],
                    hls::stream<int>& i_partition_size_strm,
                    hls::stream<ap_uint<32> >& i_wr_cfg_strm,
                    hls::stream<int>& i_bit_num_strm,
                    hls::stream<ap_uint<10> >& i_nm_strm,
                    hls::stream<ap_uint<12> >& i_bkpu_num_strm,

                    hls::stream<int>& o_per_part_nm_strm,

                    ap_uint<elem_size * vec_len>* ptr0,
                    ap_uint<elem_size * vec_len>* ptr1,
                    ap_uint<elem_size * vec_len>* ptr2,
                    ap_uint<elem_size * vec_len>* ptr3,
                    ap_uint<elem_size * vec_len>* ptr4,
                    ap_uint<elem_size * vec_len>* ptr5,
                    ap_uint<elem_size * vec_len>* ptr6,
                    ap_uint<elem_size * vec_len>* ptr7) {
    const int burst_len = BURST_LEN;
    const int k_fifo_buf = burst_len * 2;

    hls::stream<ap_uint<elem_size * vec_len> > mid_post_Agg[col_num];
#pragma HLS stream variable = mid_post_Agg depth = k_fifo_buf
#pragma HLS bind_storage variable = mid_post_Agg type = fifo impl = lutram

    hls::stream<ap_uint<8> > mid_nm_512_strm[3];
#pragma HLS array_partition variable = mid_nm_512_strm complete
#pragma HLS stream variable = mid_nm_512_strm depth = 8

    hls::stream<ap_uint<32> > mid_write_cfg_strm[3];
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

    burstwrite_col036<elem_size, vec_len, col_num>(mid_post_Agg[0], mid_post_Agg[3], mid_post_Agg[6],
                                                   mid_write_cfg_strm[0], mid_nm_512_strm[0], mid_p_base_addr_strm[0],
                                                   mid_burst_step_cnt_reg_strm[0], ptr0, ptr3, ptr6);
    burstwrite_col147<elem_size, vec_len, col_num>(mid_post_Agg[1], mid_post_Agg[4], mid_post_Agg[7],
                                                   mid_write_cfg_strm[1], mid_nm_512_strm[1], mid_p_base_addr_strm[1],
                                                   mid_burst_step_cnt_reg_strm[1], ptr1, ptr4, ptr7);

    burstwrite_col25<elem_size, vec_len, col_num>(mid_post_Agg[2], mid_post_Agg[5], mid_write_cfg_strm[2],
                                                  mid_nm_512_strm[2], mid_p_base_addr_strm[2],
                                                  mid_burst_step_cnt_reg_strm[2], ptr2, ptr5);
}

// write out for partition
template <int elem_size, int vec_len, int col_num, int hash_wh>
void writeTablePartWrapper(hls::stream<ap_uint<elem_size*(1 << hash_wh)> > post_Agg[col_num],
                           hls::stream<ap_uint<32> >& i_wr_cfg_strm,
                           hls::stream<int>& i_bit_num_strm,
                           hls::stream<ap_uint<10> >& i_nm_strm,
                           hls::stream<ap_uint<12> >& i_bkpu_num_strm,

                           ap_uint<elem_size * vec_len>* ptr0,
                           ap_uint<elem_size * vec_len>* ptr1,
                           ap_uint<elem_size * vec_len>* ptr2,
                           ap_uint<elem_size * vec_len>* ptr3,
                           ap_uint<elem_size * vec_len>* ptr4,
                           ap_uint<elem_size * vec_len>* ptr5,
                           ap_uint<elem_size * vec_len>* ptr6,
                           ap_uint<elem_size * vec_len>* ptr7,
                           ap_uint<512>* tout_meta) {
    hls::stream<int> mid_partition_size_strm;
#pragma HLS stream variable = mid_partition_size_strm depth = 2
    hls::stream<int> mid_bit_num_strm;
#pragma HLS stream variable = mid_bit_num_strm depth = 2
    hls::stream<int> mid_per_part_nm_strm;
#pragma HLS stream variable = mid_per_part_nm_strm depth = 256
#pragma HLS bind_storage variable = mid_per_part_nm_strm type = fifo impl = bram
    // read out the partition size
    int PARTITION_SIZE = tout_meta[0].range(135, 104);
    mid_partition_size_strm.write(PARTITION_SIZE);

    // read in the bit num
    const int bit_num = i_bit_num_strm.read();
    mid_bit_num_strm.write(bit_num);
    // get the partition num
    const int partition_num = 1 << bit_num;

    writeTablePart<elem_size, vec_len, col_num, hash_wh>(
        post_Agg, mid_partition_size_strm, i_wr_cfg_strm, mid_bit_num_strm, i_nm_strm, i_bkpu_num_strm,
        mid_per_part_nm_strm, ptr0, ptr1, ptr2, ptr3, ptr4, ptr5, ptr6, ptr7);
//----------update meta------------
#if !defined(__SYNTHESIS__) && XDEBUG == 1
    int cnt = 0;
#endif
    // define the buffer used to save nrow of each partition
    ap_uint<512> metaout[16];
#pragma HLS bind_storage variable = metaout type = ram_s2p impl = bram
// update tout meta
FINAL_WRITE_HEAD_LOOP:
    for (int i = 0; i < partition_num; i++) {
#pragma HLS PIPELINE II = 1
        int rnm = mid_per_part_nm_strm.read();
        ap_uint<8> idx = i / 16;
        ap_uint<8> idx_ex = i % 16;
        metaout[idx].range(32 * (idx_ex + 1) - 1, 32 * idx_ex) = rnm;

// tout_meta[8 + idx].range(32 * (idx_ex + 1) - 1, 32 * idx_ex) = rnm;

#if !defined(__SYNTHESIS__) && XDEBUG == 1
        std::cout << "P" << i << "\twrite out row number = " << rnm << std::endl;
        cnt += rnm;
#endif
    }
    // write nrow of each partition to tout_meta, ddr
    for (int i = 0; i < (partition_num + 15) / 16; i++) {
#pragma HLS PIPELINE II = 1
        tout_meta[8 + i] = metaout[i];
    }
#if !defined(__SYNTHESIS__) && XDEBUG == 1
    std::cout << "Total number of write-out row: " << cnt << std::endl;
#endif
}

} // namespace gqe
} // namespace database
} // namespace xf

#endif
