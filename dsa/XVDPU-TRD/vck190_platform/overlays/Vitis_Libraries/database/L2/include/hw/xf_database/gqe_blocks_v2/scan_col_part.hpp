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
#ifndef GQE_ISV_SCAN_COL_FOR_HP_HPP
#define GQE_ISV_SCAN_COL_FOR_HP_HPP

#ifndef __SYNTHESIS__
#include <stdio.h>
#endif

#include <ap_int.h>
#include <hls_stream.h>

#include "xf_database/types.hpp"
#include "xf_database/utils.hpp"
#include <iostream>

namespace xf {
namespace database {
namespace gqe {

template <int NCOL>
void dup_config(int n_burst, hls::stream<int8_t>& col_id_strm, hls::stream<int> config_copies[NCOL + 2]) {
    int en[8];
#pragma HLS array_partition variable = en complete
    for (ap_uint<4> i = 0; i < 8; i++) {
        en[i] = col_id_strm.read();
    }

    // for vec read
    config_copies[0].write(n_burst);
    config_copies[1].write(n_burst);
    config_copies[2].write(n_burst);
    config_copies[3].write(n_burst);
    config_copies[4].write(n_burst);
    config_copies[5].write(n_burst);
    config_copies[6].write(n_burst);
    config_copies[7].write(n_burst);
    config_copies[0].write(en[0]);
    config_copies[1].write(en[1]);
    config_copies[2].write(en[2]);
    config_copies[3].write(en[3]);
    config_copies[4].write(en[4]);
    config_copies[5].write(en[5]);
    config_copies[6].write(en[6]);
    config_copies[7].write(en[7]);
    // for b_sync
    config_copies[8].write(n_burst);
    // for vecs_to_ch_col
    config_copies[9].write(n_burst);
    config_copies[9].write(en[0]);
    config_copies[9].write(en[1]);
    config_copies[9].write(en[2]);
    config_copies[9].write(en[3]);
    config_copies[9].write(en[4]);
    config_copies[9].write(en[5]);
    config_copies[9].write(en[6]);
    config_copies[9].write(en[7]);
}

template <int WD, int VEC, int BLEN>
void read_3_vec(const ap_uint<WD * VEC>* ptr_1,
                hls::stream<int>& config_copy_1,
                hls::stream<ap_uint<WD * VEC> >& vs_1,
                hls::stream<bool>& b_1,
                const ap_uint<WD * VEC>* ptr_2,
                hls::stream<int>& config_copy_2,
                hls::stream<ap_uint<WD * VEC> >& vs_2,
                hls::stream<bool>& b_2,
                const ap_uint<WD * VEC>* ptr_3,
                hls::stream<int>& config_copy_3,
                hls::stream<ap_uint<WD * VEC> >& vs_3,
                hls::stream<bool>& b_3) {
    int num_1 = config_copy_1.read();
    int num_2 = config_copy_2.read();
    int num_3 = config_copy_3.read();
    int n_burst = (num_1 + VEC * BLEN - 1) / (VEC * BLEN);
    int en[3];
    en[0] = config_copy_1.read();
    en[1] = config_copy_2.read();
    en[2] = config_copy_3.read();
    int o = 0;
    const int hdr = 0;
    for (int i = 0; i < n_burst - 1; ++i) {
#pragma HLS loop_flatten off
        if (en[0] >= 0) {
            for (int j = 0; j < BLEN; ++j) {
#pragma HLS pipeline II = 1
                ap_uint<WD* VEC> v_1 = ptr_1[hdr + i * BLEN + j + o];
                vs_1.write(v_1);
            }
        }
        if (en[1] >= 0) {
            for (int j = 0; j < BLEN; ++j) {
#pragma HLS pipeline II = 1
                ap_uint<WD* VEC> v_2 = ptr_2[hdr + i * BLEN + j + o];
                vs_2.write(v_2);
            }
        }
        if (en[2] >= 0) {
            for (int j = 0; j < BLEN; ++j) {
#pragma HLS pipeline II = 1
                ap_uint<WD* VEC> v_3 = ptr_3[hdr + i * BLEN + j + o];
                vs_3.write(v_3);
            }
        }
        o = 1;
        // start sync
        b_1.write(true);
        b_2.write(true);
        b_3.write(true);
        while (b_1.full()) {
#pragma HLS pipeline II = 1
            ;
        }
        o = 0;
        // now all column has been read once.
    }
    const int len = (n_burst * BLEN * VEC > num_1) ? (num_1 - (n_burst - 1) * BLEN * VEC + VEC - 1) / VEC : BLEN;

    if (en[0] >= 0) {
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            ap_uint<WD* VEC> v_1 = ptr_1[hdr + (n_burst - 1) * BLEN + j + o];
            vs_1.write(v_1);
        }
    }
    if (en[1] >= 0) {
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            ap_uint<WD* VEC> v_2 = ptr_2[hdr + (n_burst - 1) * BLEN + j + o];
            vs_2.write(v_2);
        }
    }
    if (en[2] >= 0) {
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            ap_uint<WD* VEC> v_3 = ptr_3[hdr + (n_burst - 1) * BLEN + j + o];
            vs_3.write(v_3);
        }
    }
    o = 1;
    // start sync
    b_1.write(true);
    b_2.write(true);
    b_3.write(true);
    while (b_1.full()) {
#pragma HLS pipeline II = 1
        ;
    }
    o = 0;
}

template <int WD, int VEC, int BLEN>
void read_2_vec(const ap_uint<WD * VEC>* ptr_1,
                hls::stream<int>& config_copy_1,
                hls::stream<ap_uint<WD * VEC> >& vs_1,
                hls::stream<bool>& b_1,
                const ap_uint<WD * VEC>* ptr_2,
                hls::stream<int>& config_copy_2,
                hls::stream<ap_uint<WD * VEC> >& vs_2,
                hls::stream<bool>& b_2) {
    int num_1 = config_copy_1.read();
    int num_2 = config_copy_2.read();
    int n_burst = (num_1 + VEC * BLEN - 1) / (VEC * BLEN);
    int en[2];
    en[0] = config_copy_1.read();
    en[1] = config_copy_2.read();
    int o = 0;
    const int hdr = 0;
    for (int i = 0; i < n_burst - 1; ++i) {
#pragma HLS loop_flatten off
        if (en[0] >= 0) {
            for (int j = 0; j < BLEN; ++j) {
#pragma HLS pipeline II = 1
                ap_uint<WD* VEC> v_1 = ptr_1[hdr + i * BLEN + j + o];
                vs_1.write(v_1);
            }
        }
        if (en[1] >= 0) {
            for (int j = 0; j < BLEN; ++j) {
#pragma HLS pipeline II = 1
                ap_uint<WD* VEC> v_2 = ptr_2[hdr + i * BLEN + j + o];
                vs_2.write(v_2);
            }
        }
        o = 1;
        // start sync
        b_1.write(true);
        b_2.write(true);
        while (b_1.full()) {
#pragma HLS pipeline II = 1
            ;
        }
        o = 0;
        // now all column has been read once.
    }
    const int len = (n_burst * BLEN * VEC > num_1) ? (num_1 - (n_burst - 1) * BLEN * VEC + VEC - 1) / VEC : BLEN;

    if (en[0] >= 0) {
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            ap_uint<WD* VEC> v_1 = ptr_1[hdr + (n_burst - 1) * BLEN + j + o];
            vs_1.write(v_1);
        }
    }
    if (en[1] >= 0) {
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            ap_uint<WD* VEC> v_2 = ptr_2[hdr + (n_burst - 1) * BLEN + j + o];
            vs_2.write(v_2);
        }
    }
    o = 1;
    // start sync
    b_1.write(true);
    b_2.write(true);
    while (b_1.full()) {
#pragma HLS pipeline II = 1
        ;
    }
    o = 0;
}

template <int NCOL, int WD, int VEC, int BLEN, int NCH>
void vecs_to_ch_col(hls::stream<ap_uint<WD * VEC> > colvec_strm[NCOL],
                    hls::stream<int>& config_copy,
                    hls::stream<ap_uint<WD> > col_strm[NCH][NCOL],
                    hls::stream<bool> e_strm[NCH]) {
    enum { col_num = NCOL, ch_num = NCH, per_ch = VEC / NCH };
    int nrow = config_copy.read();
    int en[col_num];
    for (int c = 0; c < col_num; ++c) {
        en[c] = config_copy.read();
    }
SPLIT_COL_VEC:
    for (int i = 0; i < nrow; i += VEC) {
#pragma HLS pipeline II = per_ch
        ap_uint<WD * VEC> colvec[col_num];
        for (int c = 0; c < col_num; ++c) {
#pragma HLS unroll
            if (en[c] >= 0) colvec[c] = colvec_strm[c].read();
        }
        int n = (i + VEC) > nrow ? (nrow - i) : VEC;
        XF_DATABASE_ASSERT((VEC % ch_num == 0) && (VEC >= ch_num));
        // j for word in vec
        for (int j = 0; j < per_ch; ++j) {
            // ch for channel
            for (int ch = 0; ch < ch_num; ++ch) {
#pragma HLS unroll
                ap_uint<WD> ct[col_num];
                for (int c = 0; c < col_num; ++c) {
#pragma HLS unroll
                    if (en[c] >= 0)
                        ct[c] = colvec[c].range(WD * (j * ch_num + ch + 1) - 1, WD * (j * ch_num + ch));
                    else if (en[c] == -1)
                        ct[c] = 0;
                    else if (en[c] == -2)
                        ct[c] = i + j * ch_num + ch;
                }
                if ((j * ch_num + ch) < n) {
                    for (int c = 0; c < col_num; ++c) {
#pragma HLS unroll
                        col_strm[ch][c].write(ct[c]);
                    }
                    e_strm[ch].write(false);
                }
            }
        }
    }
    for (int ch = 0; ch < ch_num; ++ch) {
#pragma HLS unroll
        e_strm[ch].write(true);
    }
}

template <int _n>
inline bool andTree(bool flag[], int _o = 0) {
#pragma HLS inline
    return andTree<_n / 2>(flag, _o) && andTree<_n / 2>(flag, _o + (_n / 2));
}

template <>
inline bool andTree<2>(bool flag[], int _o) {
#pragma HLS inline
    return flag[_o] && flag[_o + 1];
}

template <int _n>
inline bool orTree(bool flag[], int _o = 0) {
#pragma HLS inline
    return orTree<_n / 2>(flag, _o) || orTree<_n / 2>(flag, _o + (_n / 2));
}

template <>
inline bool orTree<2>(bool flag[], int _o) {
#pragma HLS inline
    return flag[_o] || flag[_o + 1];
}

template <int VEC, int BLEN>
void b_sync(hls::stream<bool> b[8], hls::stream<int>& config_copy) {
    int num = config_copy.read();
    int n_burst = (num + VEC * BLEN - 1) / (VEC * BLEN);
    for (int i = 0; i < n_burst; ++i) {
        bool has_empty = true;
        while (has_empty) {
#pragma HLS pipeline II = 1
            bool p[8];
#pragma HLS array_partition variable = p
            for (int i = 0; i < 8; ++i) {
                p[i] = b[i].empty();
            }
            has_empty = orTree<8>(p);
        }
        // no empty, can proceed
        for (int j = 0; j < 8; ++j) {
#pragma HLS unroll
            b[j].read();
        }
    }
}

template <int WD, int VEC, int NCH, int NCOL>
void scan_cols(const ap_uint<WD * VEC>* ptr0,
               const ap_uint<WD * VEC>* ptr1,
               const ap_uint<WD * VEC>* ptr2,
               const ap_uint<WD * VEC>* ptr3,
               const ap_uint<WD * VEC>* ptr4,
               const ap_uint<WD * VEC>* ptr5,
               const ap_uint<WD * VEC>* ptr6,
               const ap_uint<WD * VEC>* ptr7,
               hls::stream<int>& nrow_strm,
               hls::stream<int8_t>& col_id_strm,
               hls::stream<ap_uint<WD> > s[NCH][NCOL],
               hls::stream<bool> e_s[NCH]) {
    // clang-format off
    enum {  burstLen = BURST_LEN, burstLen2 = (BURST_LEN * 2), nCol = NCOL, nCh = NCH };
#pragma HLS dataflow

    // need wide buffering for fast input, slow output, wide to narrow.
    hls::stream<ap_uint<WD * VEC> > vs[nCol];
#pragma HLS stream variable = vs depth = burstLen2
#pragma HLS bind_storage variable = vs type = fifo impl = lutram

    hls::stream<int> config_copies[nCol + 2];
#pragma HLS stream variable = config_copies depth = 2
#pragma HLS bind_storage variable = config_copies type = fifo impl = lutram

    int nrowA = nrow_strm.read(); 

    dup_config<NCOL>( nrowA, col_id_strm, config_copies);

    hls::stream<bool> b[nCol];
#pragma HLS stream variable = b depth = 1
#pragma HLS array_partition variable = b dim = 0

    // TODO change with nCol
    read_3_vec<8*TPCH_INT_SZ, VEC_SCAN, BURST_LEN>(ptr0, config_copies[0], vs[0], b[0], ptr3, config_copies[3], vs[3], b[3],
                                                     ptr6, config_copies[6], vs[6], b[6]);
    read_3_vec<8*TPCH_INT_SZ, VEC_SCAN, BURST_LEN>(ptr1, config_copies[1], vs[1], b[1], ptr4, config_copies[4], vs[4], b[4],
                                                     ptr7, config_copies[7], vs[7], b[7]);
    read_2_vec<8*TPCH_INT_SZ, VEC_SCAN, BURST_LEN>(ptr2, config_copies[2], vs[2], b[2], ptr5, config_copies[5], vs[5], b[5]);
    b_sync<VEC_SCAN, BURST_LEN>(b, config_copies[nCol]);

    vecs_to_ch_col<NCOL, WD, VEC_SCAN, BURST_LEN, NCH>(vs, config_copies[nCol + 1], s, e_s);
}

} // namespace gqe
} // namespace database
} // namespace xf

#endif // GQE_SCAN_TO_CHANNEL_HPP
