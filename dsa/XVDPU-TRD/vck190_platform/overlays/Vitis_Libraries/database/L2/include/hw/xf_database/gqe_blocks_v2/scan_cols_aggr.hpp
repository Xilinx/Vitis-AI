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
#ifndef GQE_ISV_SCAN_COL_AGGR_HPP
#define GQE_ISV_SCAN_COL_AGGR_HPP

#ifndef __SYNTHESIS__
#include <stdio.h>
#endif

#include <ap_int.h>
#include <hls_stream.h>

#include "xf_database/utils.hpp"
#include "xf_database/types.hpp"
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
void read_two_vec(const ap_uint<WD * VEC>* ptr_1,
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
    int en_1 = config_copy_1.read();
    int en_2 = config_copy_2.read();
    const int hdr = 0;
    if (n_burst > 0) {
        if (en_1 >= 0 && en_2 >= 0) {
            for (int i = 0; i < n_burst - 1; ++i) {
#pragma HLS loop_flatten off
                for (int j = 0; j < BLEN; ++j) {
#pragma HLS pipeline II = 1
                    ap_uint<WD* VEC> v_1 = ptr_1[hdr + i * BLEN + j];
                    vs_1.write(v_1);
                }
                for (int j = 0; j < BLEN; ++j) {
#pragma HLS pipeline II = 1
                    ap_uint<WD* VEC> v_2 = ptr_2[hdr + i * BLEN + j];
                    vs_2.write(v_2);
                }
                // start sync
                b_1.write(true);
                b_2.write(true);
                while (b_1.full()) {
#pragma HLS pipeline II = 1
                    ;
                }
                // now all column has been read once.
            }
            const int len =
                (n_burst * BLEN * VEC > num_1) ? (num_1 - (n_burst - 1) * BLEN * VEC + VEC - 1) / VEC : BLEN;
            for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
                ap_uint<WD* VEC> v_1 = ptr_1[hdr + (n_burst - 1) * BLEN + j];
                vs_1.write(v_1);
            }
            for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
                ap_uint<WD* VEC> v_2 = ptr_2[hdr + (n_burst - 1) * BLEN + j];
                vs_2.write(v_2);
            }
            // start sync
            b_1.write(true);
            b_2.write(true);
            while (b_1.full()) {
#pragma HLS pipeline II = 1
                ;
            }
        } else if (en_1 >= 0 && en_2 < 0) {
            for (int i = 0; i < n_burst - 1; ++i) {
#pragma HLS loop_flatten off
                for (int j = 0; j < BLEN; ++j) {
#pragma HLS pipeline II = 1
                    ap_uint<WD* VEC> v_1 = ptr_1[hdr + i * BLEN + j];
                    vs_1.write(v_1);
                }
                // start sync
                b_1.write(true);
                b_2.write(true);
                while (b_1.full()) {
#pragma HLS pipeline II = 1
                    ;
                }
                // now all column has been read once.
            }
            const int len =
                (n_burst * BLEN * VEC > num_1) ? (num_1 - (n_burst - 1) * BLEN * VEC + VEC - 1) / VEC : BLEN;
            for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
                ap_uint<WD* VEC> v_1 = ptr_1[hdr + (n_burst - 1) * BLEN + j];
                vs_1.write(v_1);
            }
            // start sync
            b_1.write(true);
            b_2.write(true);
            while (b_1.full()) {
#pragma HLS pipeline II = 1
                ;
            }
        } else if (en_1 < 0 && en_2 >= 0) {
            for (int i = 0; i < n_burst - 1; ++i) {
#pragma HLS loop_flatten off
                for (int j = 0; j < BLEN; ++j) {
#pragma HLS pipeline II = 1
                    ap_uint<WD* VEC> v_2 = ptr_2[hdr + i * BLEN + j];
                    vs_2.write(v_2);
                }
                // start sync
                b_1.write(true);
                b_2.write(true);
                while (b_1.full()) {
#pragma HLS pipeline II = 1
                    ;
                }
                // now all column has been read once.
            }
            const int len =
                (n_burst * BLEN * VEC > num_1) ? (num_1 - (n_burst - 1) * BLEN * VEC + VEC - 1) / VEC : BLEN;
            for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
                ap_uint<WD* VEC> v_2 = ptr_2[hdr + (n_burst - 1) * BLEN + j];
                vs_2.write(v_2);
            }
            // start sync
            b_1.write(true);
            b_2.write(true);
            while (b_1.full()) {
#pragma HLS pipeline II = 1
                ;
            }
        } else {
            for (int i = 0; i < n_burst; ++i) {
                b_1.write(true);
                b_2.write(true);
            }
            while (b_1.full()) {
#pragma HLS pipeline II = 1
                ;
            }
        }
    }
}

template <int NCOL, int WD, int VEC, int BLEN, int NCH>
void vecs_to_ch_col(hls::stream<ap_uint<WD * VEC> > vs[NCOL],
                    hls::stream<int>& config_copy,
                    hls::stream<ap_uint<WD> > s[NCH][NCOL],
                    hls::stream<bool> e_s[NCH]) {
    enum { nCol = NCOL, nCh = NCH };
    int num = config_copy.read();
    int n_burst = (num + VEC * BLEN - 1) / (VEC * BLEN);
    int en[nCol];
    for (int c = 0; c < nCol; ++c) {
        en[c] = config_copy.read();
    }
#if __cplusplus >= 201103L
//    static_assert(VEC == NCH, "VEC != NCH is not supported!");
// TODO check and rewrite
#endif
    if (n_burst > 0) {
        for (int i = 0; i < n_burst - 1; ++i) {
#pragma HLS loop_flatten off
        L_VEC_SPLIT:
            for (int j = 0; j < BLEN; ++j) {
#pragma HLS pipeline II = 1
                for (int c = 0; c < nCol; ++c) {
#pragma HLS unroll
                    if (en[c] >= 0) {
                        ap_uint<WD* VEC> v = vs[c].read();
                        for (int k = 0; k < NCH; ++k) {
                            ap_uint<WD> d = v.range(WD * k + WD - 1, WD * k);
                            s[k][c].write(d);
                            if (c == 0) e_s[k].write(false);
                        }
                        for (int k = NCH; k < 2 * NCH; ++k) {
                            ap_uint<WD> d = v.range(WD * k + WD - 1, WD * k);
                            s[k - NCH][c].write(d);
                            if (c == 0) e_s[k - NCH].write(false);
                        }
                    } else if (en[c] == -1) {
                        for (int k = 0; k < NCH; ++k) {
                            ap_uint<WD> d = 0;
                            s[k][c].write(d);
                            if (c == 0) e_s[k].write(false);
                        }
                        for (int k = NCH; k < 2 * NCH; ++k) {
                            ap_uint<WD> d = 0;
                            s[k - NCH][c].write(d);
                            if (c == 0) e_s[k - NCH].write(false);
                        }
                    } else if (en[c] == -2) {
                        for (int k = 0; k < NCH; ++k) {
                            ap_uint<WD> d = i * BLEN * VEC + j * VEC + k;
                            s[k][c].write(d);
                            if (c == 0) e_s[k].write(false);
                        }
                        for (int k = NCH; k < 2 * NCH; ++k) {
                            ap_uint<WD> d = i * BLEN * VEC + j * VEC + k;
                            s[k - NCH][c].write(d);
                            if (c == 0) e_s[k - NCH].write(false);
                        }
                    }
                } // for column
            }     // for BLEN x VEC
        }         // for n_burst

        const int len = (n_burst * BLEN * VEC >= num) ? (num - (n_burst - 1) * BLEN * VEC + VEC - 1) / VEC : BLEN;
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            for (int c = 0; c < nCol; ++c) {
#pragma HLS unroll
                if (en[c] >= 0) {
                    ap_uint<WD* VEC> v = vs[c].read();
                    for (int k = 0; k < NCH; ++k) {
                        ap_uint<WD> d = v.range(WD * k + WD - 1, WD * k);
                        if (((n_burst - 1) * BLEN * VEC + j * VEC + k) < num) {
                            s[k][c].write(d);
                            if (c == 0) e_s[k].write(false);
                        }
                    }
                    for (int k = NCH; k < 2 * NCH; ++k) {
                        ap_uint<WD> d = v.range(WD * k + WD - 1, WD * k);
                        if (((n_burst - 1) * BLEN * VEC + j * VEC + k) < num) {
                            s[k - NCH][c].write(d);
                            if (c == 0) e_s[k - NCH].write(false);
                        }
                    }
                } else if (en[c] == -1) {
                    for (int k = 0; k < NCH; ++k) {
                        ap_uint<WD> d = 0;
                        if (((n_burst - 1) * BLEN * VEC + j * VEC + k) < num) {
                            s[k][c].write(d);
                            if (c == 0) e_s[k].write(false);
                        }
                    }
                    for (int k = NCH; k < 2 * NCH; ++k) {
                        ap_uint<WD> d = 0;
                        if (((n_burst - 1) * BLEN * VEC + j * VEC + k) < num) {
                            s[k - NCH][c].write(d);
                            if (c == 0) e_s[k - NCH].write(false);
                        }
                    }
                } else if (en[c] == -2) {
                    for (int k = 0; k < NCH; ++k) {
                        ap_uint<WD> d = (n_burst - 1) * BLEN * VEC + j * VEC + k;
                        if (((n_burst - 1) * BLEN * VEC + j * VEC + k) < num) {
                            s[k][c].write(d);
                            if (c == 0) e_s[k].write(false);
                        }
                    }
                    for (int k = NCH; k < 2 * NCH; ++k) {
                        ap_uint<WD> d = (n_burst - 1) * BLEN * VEC + j * VEC + k;
                        if (((n_burst - 1) * BLEN * VEC + j * VEC + k) < num) {
                            s[k - NCH][c].write(d);
                            if (c == 0) e_s[k - NCH].write(false);
                        }
                    }
                }
            } // for column
        }     // for BLEN x VEC
    }

    for (int c = 0; c < NCH; c++) {
#pragma HLS unroll
        e_s[c].write(true);
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
               const int num,
               hls::stream<int8_t>& col_id_strm,
               hls::stream<ap_uint<WD> > s[NCH][NCOL],
               hls::stream<bool> e_s[NCH]) {
    // clang-format off
    enum {  burstLen = BURST_LEN, burstLen2 = (BURST_LEN * 2), nCol = NCOL, nCh = NCH };
// clang-format on
#pragma HLS dataflow

    // need wide buffering for fast input, slow output, wide to narrow.
    hls::stream<ap_uint<WD * VEC> > vs[nCol];
#pragma HLS stream variable = vs depth = burstLen2
#pragma HLS bind_storage variable = vs type = fifo impl = lutram

    hls::stream<int> config_copies[nCol + 2];
#pragma HLS stream variable = config_copies depth = 2
#pragma HLS bind_storage variable = config_copies type = fifo impl = lutram

    dup_config<NCOL>(num, col_id_strm, config_copies);

    hls::stream<bool> b[nCol];
#pragma HLS stream variable = b depth = 1
#pragma HLS array_partition variable = b dim = 0

    read_two_vec<8 * TPCH_INT_SZ, VEC_SCAN, BURST_LEN>(ptr0, config_copies[0], vs[0], b[0], ptr4, config_copies[4],
                                                       vs[4], b[4]);
    read_two_vec<8 * TPCH_INT_SZ, VEC_SCAN, BURST_LEN>(ptr1, config_copies[1], vs[1], b[1], ptr5, config_copies[5],
                                                       vs[5], b[5]);
    read_two_vec<8 * TPCH_INT_SZ, VEC_SCAN, BURST_LEN>(ptr2, config_copies[2], vs[2], b[2], ptr6, config_copies[6],
                                                       vs[6], b[6]);
    read_two_vec<8 * TPCH_INT_SZ, VEC_SCAN, BURST_LEN>(ptr3, config_copies[3], vs[3], b[3], ptr7, config_copies[7],
                                                       vs[7], b[7]);

    b_sync<VEC_SCAN, BURST_LEN>(b, config_copies[nCol]);

    vecs_to_ch_col<NCOL, WD, VEC_SCAN, BURST_LEN, NCH>(vs, config_copies[nCol + 1], s, e_s);
}

template <int WD, int VEC, int NCH, int NCOL>
void scan_cols_wrapper(const ap_uint<WD * VEC>* ptr0,
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
    int num = nrow_strm.read();
    scan_cols<WD, VEC, NCH, NCOL>(ptr0, ptr1, ptr2, ptr3, ptr4, ptr5, ptr6, ptr7, num, col_id_strm, s, e_s);
}

} // namespace gqe
} // namespace database
} // namespace xf

#endif // GQE_SCAN_TO_CHANNEL_HPP
