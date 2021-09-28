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
#ifndef GQE_ISV_SCAN_TO_CHANNEL_HPP
#define GQE_ISV_SCAN_TO_CHANNEL_HPP

#ifndef __SYNTHESIS__
#include <stdio.h>
#include <iostream>
#endif

#include <ap_int.h>
#include <hls_stream.h>
#include "hls_burst_maxi.h"

#ifndef __SYNTHESIS__
//#define USER_DEBUG true
#endif

namespace xf {
namespace database {
namespace gqe {

template <int CH_NM, int BLEN>
void read_col(hls::burst_maxi<ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> > din_col,
              int64_t nrow,
              bool din_col_en,
              hls::stream<ap_uint<8 * TPCH_INT_SZ> >& ch_strm0,
              hls::stream<ap_uint<8 * TPCH_INT_SZ> >& ch_strm1,
              hls::stream<ap_uint<8 * TPCH_INT_SZ> >& ch_strm2,
              hls::stream<ap_uint<8 * TPCH_INT_SZ> >& ch_strm3,
              bool gen_e,
              hls::stream<bool> e_ch_strms[CH_NM]) {
    // each burst data: 512bit * burst_len, burst num:
    int n_burst = (nrow + BLEN * VEC_SCAN - 1) / (BLEN * VEC_SCAN);
    if (n_burst > 0) {
        if (din_col_en > 0) {
            // first n-1 burst

            if (n_burst > 3) {
                din_col.read_request(0, BLEN);
                din_col.read_request(BLEN, BLEN);
                din_col.read_request(2 * BLEN, BLEN);
            } else if (n_burst == 2) {
                din_col.read_request(0, BLEN);
            } else if (n_burst == 3) {
                din_col.read_request(0, BLEN);
                din_col.read_request(BLEN, BLEN);
            }

            for (int i = 0; i < n_burst - 1; i++) {
                for (int j = 0; j < BLEN; ++j) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 64 max = 64
                    if (i > 2 && j == 0) {
                        din_col.read_request(i * BLEN, BLEN);
                    }
                    ap_uint<8 * TPCH_INT_SZ* VEC_SCAN> tmp = din_col.read();
                    ch_strm0.write(tmp.range(63, 0));
                    ch_strm1.write(tmp.range(127, 64));
                    ch_strm2.write(tmp.range(191, 128));
                    ch_strm3.write(tmp.range(255, 192));
                    if (gen_e) {
                        e_ch_strms[0].write(false);
                        e_ch_strms[1].write(false);
                        e_ch_strms[2].write(false);
                        e_ch_strms[3].write(false);
                    }
                }
            }
            // last burst round
            const int left_nrow = nrow - (n_burst - 1) * BLEN * VEC_SCAN;
            const int left_nrow_vec = (left_nrow + VEC_SCAN - 1) / VEC_SCAN;
            din_col.read_request((n_burst - 1) * BLEN, left_nrow_vec);
            for (int j = 0; j < left_nrow_vec; ++j) {
#pragma HLS pipeline
                ap_uint<8 * TPCH_INT_SZ* VEC_SCAN> tmp = din_col.read();
                if ((j * VEC_SCAN) < left_nrow) {
                    ch_strm0.write(tmp.range(63, 0));
                    //#ifndef __SYNTHESIS__
                    //                    std::cout << "read_col: " << tmp.range(63, 0) << std::endl;
                    //#endif

                    if (gen_e) e_ch_strms[0].write(false);
                }
                if ((j * VEC_SCAN + 1) < left_nrow) {
                    ch_strm1.write(tmp.range(127, 64));
                    //#ifndef __SYNTHESIS__
                    //                    std::cout << "read_col: " << tmp.range(127, 64) << std::endl;
                    //#endif
                    if (gen_e) e_ch_strms[1].write(false);
                }
                if ((j * VEC_SCAN + 2) < left_nrow) {
                    ch_strm2.write(tmp.range(191, 128));
                    //#ifndef __SYNTHESIS__
                    //                    std::cout << "read_col: " << tmp.range(191, 128) << std::endl;
                    //#endif
                    if (gen_e) e_ch_strms[2].write(false);
                }
                if ((j * VEC_SCAN + 3) < left_nrow) {
                    ch_strm3.write(tmp.range(255, 192));
                    //#ifndef __SYNTHESIS__
                    //                    std::cout << "read_col: " << tmp.range(255, 192) << std::endl;
                    //#endif
                    if (gen_e) e_ch_strms[3].write(false);
                }
            }

        } else { // fill inputs to 0
            // first n-1 burst
            for (int i = 0; i < n_burst - 1; i++) {
                for (int j = 0; j < BLEN; ++j) {
#pragma HLS pipeline
                    ch_strm0.write(0);
                    ch_strm1.write(0);
                    ch_strm2.write(0);
                    ch_strm3.write(0);
                    if (gen_e) {
                        e_ch_strms[0].write(false);
                        e_ch_strms[1].write(false);
                        e_ch_strms[2].write(false);
                        e_ch_strms[3].write(false);
                    }
                }
            }
            // last round burst
            const int left_nrow = nrow - (n_burst - 1) * BLEN * VEC_SCAN;
            const int left_nrow_vec = (left_nrow + VEC_SCAN - 1) / VEC_SCAN;
            for (int j = 0; j < left_nrow_vec; ++j) {
#pragma HLS pipeline
                if ((j * VEC_SCAN) < left_nrow) {
                    ch_strm0.write(0);
                    if (gen_e) e_ch_strms[0].write(false);
                }
                if ((j * VEC_SCAN + 1) < left_nrow) {
                    ch_strm1.write(0);
                    if (gen_e) e_ch_strms[1].write(false);
                }
                if ((j * VEC_SCAN + 2) < left_nrow) {
                    ch_strm2.write(0);
                    if (gen_e) e_ch_strms[2].write(false);
                }
                if ((j * VEC_SCAN + 3) < left_nrow) {
                    ch_strm3.write(0);
                    if (gen_e) e_ch_strms[3].write(false);
                }
            }
        }
    }

    for (int ch = 0; ch < CH_NM; ++ch) {
#pragma HLS unroll
        if (gen_e) e_ch_strms[ch].write(true);
    }
}

// read the validation buffer
template <int BLEN>
void read_validation(bool read_valid,
                     int64_t nrow,
                     hls::burst_maxi<ap_uint<64> > din_val,
                     hls::stream<ap_uint<64> >& valid_strm) {
    int nrow_64 = (nrow + 63) / 64;
    int n_burst = (nrow_64 + BLEN - 1) / BLEN;
    if (n_burst > 0) {
        // read the first n_burst-1 data
        for (int n = 0; n < n_burst - 1; ++n) {
            if (read_valid) din_val.read_request(n * BLEN, BLEN);
            for (int j = 0; j < BLEN; ++j) {
#pragma HLS pipeline
                //#ifndef __SYNTHESIS__
                //                std::cout << "read_valid = " << read_valid << std::endl;
                //#endif
                ap_uint<64> valid = read_valid ? din_val.read() : ap_uint<64>(0xffffffffffffffff);
                //#ifndef __SYNTHESIS__
                //                std::cout << "valid: " << std::hex << (int64_t)valid << std::dec << std::endl;
                //#endif
                valid_strm.write(valid);
            }
        }
        // the last round data
        int left_nrow = nrow - (n_burst - 1) * BLEN * 64;
        int left_nrow_vec = (left_nrow + 63) / 64;
        // std::cout << "n_burst: " << n_burst << std::endl;
        // std::cout << "left_nrow: " << left_nrow << std::endl;
        // std::cout << "left_nrow_vec: " << left_nrow_vec << std::endl;
        //#ifndef __SYNTHESIS__
        //        std::cout << "read_valid = " << read_valid << std::endl;
        //#endif
        if (read_valid) din_val.read_request((n_burst - 1) * BLEN, left_nrow_vec);
        for (int j = 0; j < left_nrow_vec; ++j) {
            ap_uint<64> valid = read_valid ? din_val.read() : ap_uint<64>(0xffffffffffffffff);
            //#ifndef __SYNTHESIS__
            //            std::cout << "valid: " << std::hex << (int64_t)valid << std::dec << std::endl;
            //#endif
            valid_strm.write(valid);
        }
    }
}

template <int BLEN>
void write_row_id(int sec_id,
                  int64_t nrow,
                  hls::stream<ap_uint<64> >& valid_strm,
                  hls::stream<ap_uint<8 * TPCH_INT_SZ> >& ch_strm0,
                  hls::stream<ap_uint<8 * TPCH_INT_SZ> >& ch_strm1,
                  hls::stream<ap_uint<8 * TPCH_INT_SZ> >& ch_strm2,
                  hls::stream<ap_uint<8 * TPCH_INT_SZ> >& ch_strm3) {
    int nrow_64 = (nrow + 63) / 64;
    ap_uint<64> row_id_dat[4];
    for (int b = 0; b < 4; b++) {
        row_id_dat[b].range(63, 32) = sec_id;
    }
    ap_uint<64> valid;
    ap_uint<64> rowID[4];
    int it = 1;
    // read the first n-1 strm
    for (int i = 0; i < nrow_64 - 1; ++i) {
        valid = valid_strm.read();
        bool val[4];
        for (int j = 0; j < 64; j = j + 4) {
#pragma HLS pipeline
            for (int b = 0; b < 4; b++) {
#pragma HLS unroll
                // row_id_dat[b].range(31, 0) = i * 64 + j + b;
                row_id_dat[b].range(31, 0) = it++;
                val[b] = valid.range(b + j, b + j);
                rowID[b] = val[b] ? row_id_dat[b] : ap_uint<64>(0);
            }

            ch_strm0.write(rowID[0]);
            ch_strm1.write(rowID[1]);
            ch_strm2.write(rowID[2]);
            ch_strm3.write(rowID[3]);
        }
    }
    // the last n_row, which is <=64
    valid = valid_strm.read();
    int left_nrow = nrow - (nrow_64 - 1) * 64;
    int row_id_offset = (nrow_64 - 1) * 64;
    int left_nrow_4 = left_nrow / 4 * 4;
    bool val[4];
    for (int i = 0; i < left_nrow_4; i = i + 4) {
#pragma HLS pipeline
        for (int b = 0; b < 4; b++) {
#pragma HLS unroll
            // row_id_dat.range(31, 0) = row_id_offset + i + b;
            row_id_dat[b].range(31, 0) = it++;
            val[b] = valid.range(i + b, i + b);
            rowID[b] = val[b] ? row_id_dat[b] : ap_uint<64>(0);
        }

        ch_strm0.write(rowID[0]);
        ch_strm1.write(rowID[1]);
        ch_strm2.write(rowID[2]);
        ch_strm3.write(rowID[3]);
    }
    // the last 0-3 dat
    if (nrow % 4 == 1) {
        bool val = valid.range(left_nrow - 1, left_nrow - 1);

        row_id_dat[0].range(31, 0) = it++;
        ap_uint<64> rowID0 = val ? row_id_dat[0] : ap_uint<64>(0);
        ch_strm0.write(rowID0);
    }
    if (nrow % 4 == 2) {
        bool val0 = valid.range(left_nrow - 2, left_nrow - 2);
        bool val1 = valid.range(left_nrow - 1, left_nrow - 1);
        row_id_dat[0].range(31, 0) = it++;
        row_id_dat[1].range(31, 0) = it++;
        ap_uint<64> rowID0 = val0 ? row_id_dat[0] : ap_uint<64>(0);
        ap_uint<64> rowID1 = val1 ? row_id_dat[1] : ap_uint<64>(0);
        ch_strm0.write(rowID0);
        ch_strm1.write(rowID1);
    }
    if (nrow % 4 == 3) {
        bool val0 = valid.range(left_nrow - 3, left_nrow - 3);
        bool val1 = valid.range(left_nrow - 2, left_nrow - 2);
        bool val2 = valid.range(left_nrow - 1, left_nrow - 1);
        row_id_dat[0].range(31, 0) = it++;
        row_id_dat[1].range(31, 0) = it++;
        row_id_dat[2].range(31, 0) = it++;
        ap_uint<64> rowID0 = val0 ? row_id_dat[0] : ap_uint<64>(0);
        ap_uint<64> rowID1 = val1 ? row_id_dat[1] : ap_uint<64>(0);
        ap_uint<64> rowID2 = val2 ? row_id_dat[2] : ap_uint<64>(0);

        ch_strm0.write(rowID0);
        ch_strm1.write(rowID1);
        ch_strm2.write(rowID2);
    }
}

template <int BLEN>
void gen_row_id_col(int sec_id,
                    int64_t nrow,
                    bool din_val_en,
                    hls::burst_maxi<ap_uint<64> > din_val,
                    hls::stream<ap_uint<8 * TPCH_INT_SZ> >& ch_strm0,
                    hls::stream<ap_uint<8 * TPCH_INT_SZ> >& ch_strm1,
                    hls::stream<ap_uint<8 * TPCH_INT_SZ> >& ch_strm2,
                    hls::stream<ap_uint<8 * TPCH_INT_SZ> >& ch_strm3) {
#pragma HLS dataflow
    hls::stream<ap_uint<64> > valid_strm;
#pragma HLS stream variable = valid_strm depth = 32
    read_validation<BLEN>(din_val_en, nrow, din_val, valid_strm);
    write_row_id<BLEN>(sec_id, nrow, valid_strm, ch_strm0, ch_strm1, ch_strm2, ch_strm3);
}

// row-id column
template <int CH_NM, int BLEN>
void read_gen_row_id(ap_uint<2> rowID_flags,
                     int sec_id,
                     hls::burst_maxi<ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> > din_col,
                     hls::burst_maxi<ap_uint<64> > din_val,
                     int64_t nrow,
                     bool din_col_en,
                     hls::stream<ap_uint<8 * TPCH_INT_SZ> >& ch_strm0,
                     hls::stream<ap_uint<8 * TPCH_INT_SZ> >& ch_strm1,
                     hls::stream<ap_uint<8 * TPCH_INT_SZ> >& ch_strm2,
                     hls::stream<ap_uint<8 * TPCH_INT_SZ> >& ch_strm3,
                     bool gen_e,
                     hls::stream<bool> e_ch_strms[CH_NM]) {
    bool gen_row_id = rowID_flags.range(0, 0);
    bool din_val_en = rowID_flags.range(1, 1);
#ifndef __SYNTHESIS__
    std::cout << "gen_row_id: " << gen_row_id << std::endl;
    std::cout << "din_val_en: " << din_val_en << std::endl;
#endif

    if (gen_row_id) { // gen rowid inside kernel
        gen_row_id_col<BLEN>(sec_id, nrow, din_val_en, din_val, ch_strm0, ch_strm1, ch_strm2, ch_strm3);
    } else { // read row-id from col dat
        read_col<CH_NM, BLEN>(din_col, nrow, din_col_en, ch_strm0, ch_strm1, ch_strm2, ch_strm3, gen_e, e_ch_strms);
    }
}

// scan column data in
template <int CH_NM, int COL_NM, int BLEN>
void scan_cols(ap_uint<2> rowID_flags,
               int64_t nrow,
               int secID,
               ap_uint<3> din_col_en,
               hls::burst_maxi<ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> > din_col0,
               hls::burst_maxi<ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> > din_col1,
               hls::burst_maxi<ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> > din_col2,
               hls::burst_maxi<ap_uint<64> > din_val,
               hls::stream<ap_uint<8 * TPCH_INT_SZ> > ch_strms[CH_NM][COL_NM],
               hls::stream<bool> e_ch_strms[CH_NM]) {
    hls::stream<bool> e_ch_strms_nouse[2][CH_NM];
#pragma HLS stream variable = e_ch_strms_nouse depth = 2
#ifndef __SYNTHESIS__
    std::cout << "din_col_en: " << (int)din_col_en << std::endl;
#endif

#pragma HLS dataflow
    read_col<CH_NM, BLEN>(din_col0, nrow, din_col_en[0], ch_strms[0][0], ch_strms[1][0], ch_strms[2][0], ch_strms[3][0],
                          1, e_ch_strms);
    read_col<CH_NM, BLEN>(din_col1, nrow, din_col_en[1], ch_strms[0][1], ch_strms[1][1], ch_strms[2][1], ch_strms[3][1],
                          0, e_ch_strms_nouse[0]);
    read_gen_row_id<CH_NM, BLEN>(rowID_flags, secID, din_col2, din_val, nrow, din_col_en[2], ch_strms[0][2],
                                 ch_strms[1][2], ch_strms[2][2], ch_strms[3][2], 0, e_ch_strms_nouse[1]);
}

} // namespace gqe
} // namespace database
} // namespace xf

#endif // GQE_SCAN_TO_CHANNEL_HPP
