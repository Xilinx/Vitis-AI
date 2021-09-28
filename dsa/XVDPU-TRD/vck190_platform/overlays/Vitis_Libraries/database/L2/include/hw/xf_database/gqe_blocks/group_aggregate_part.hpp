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
#ifndef GQE_GROUP_AGGREGATE_PART_HPP
#define GQE_GROUP_AGGREGATE_PART_HPP

#include <hls_stream.h>
#include <ap_int.h>

#include "xf_utils_hw/stream_shuffle.hpp"
#include "xf_database/hash_group_aggregate.hpp"

#include "xf_database/gqe_blocks/gqe_types.hpp"

namespace xf {
namespace database {
namespace gqe {

// duplicate stream with double strm end
template <int WStrm, int COL_NM>
void dup_strm(hls::stream<ap_uint<WStrm> > in_strm[COL_NM],
              hls::stream<bool>& e_in_strm,
              hls::stream<ap_uint<WStrm> > out_strm1[COL_NM],
              hls::stream<bool>& e_out1_strm,
              hls::stream<ap_uint<WStrm> > out_strm2[COL_NM],
              hls::stream<bool>& e_out2_strm) {
    bool e = e_in_strm.read();
    while (!e) {
#pragma HLS pipeline II = 1
        e = e_in_strm.read();
        for (int i = 0; i < COL_NM; i++) {
#pragma HLS unroll
            ap_uint<WStrm> tmp = in_strm[i].read();
            out_strm1[i].write(tmp);
            out_strm2[i].write(tmp);
        }
        e_out1_strm.write(false);
        e_out2_strm.write(false);
    }
    e_out1_strm.write(true);
    e_out2_strm.write(true);
}

// merge column and discard some column
/*
 * merge_level1:key---pld0  key---pld1  key---pld2
 *                  |           |           |
 *                merge0      merge1      merge2
 *
 * merge_level2:merge0---merge1  merge1---merge2
 *                     |                |
 *                    out0             out1
 */
template <int WStrm, int ColNM>
void merge_column(hls::stream<ap_uint<32> >& merge_column_cfg_strm,
                  hls::stream<ap_uint<WStrm> > in_key[ColNM],
                  hls::stream<ap_uint<WStrm> > in_pld0[ColNM],
                  hls::stream<ap_uint<WStrm> > in_pld1[ColNM],
                  hls::stream<ap_uint<WStrm> > in_pld2[ColNM],
                  hls::stream<bool>& e_in_strm,
                  hls::stream<ap_uint<WStrm> > out_strm[2 * ColNM],
                  hls::stream<bool>& e_out_strm) {
    ap_uint<32> merge_column_cfg1 = merge_column_cfg_strm.read();
    ap_uint<32> merge_column_cfg2 = merge_column_cfg_strm.read();

    ap_uint<ColNM> merge_level1_cfg0 = merge_column_cfg1(ColNM - 1, 0);
    ap_uint<ColNM> merge_level1_cfg1 = merge_column_cfg1(2 * ColNM - 1, ColNM);
    ap_uint<ColNM> merge_level1_cfg2 = merge_column_cfg1(3 * ColNM - 1, 2 * ColNM);
    ap_uint<1> merge_level1_reverse = merge_column_cfg1[3 * ColNM];

    ap_uint<ColNM> merge_level2_cfg0 = merge_column_cfg2(ColNM - 1, 0);
    ap_uint<ColNM> merge_level2_cfg1 = merge_column_cfg2(2 * ColNM - 1, ColNM);
    ap_uint<1> merge_level2_reverse = merge_column_cfg2[2 * ColNM];

    ap_uint<WStrm> key[ColNM];
    ap_uint<WStrm> pld0[ColNM];
    ap_uint<WStrm> pld1[ColNM];
    ap_uint<WStrm> pld2[ColNM];
    ap_uint<WStrm> merge0[ColNM];
    ap_uint<WStrm> merge1[ColNM];
    ap_uint<WStrm> merge2[ColNM];
    ap_uint<WStrm> out[2 * ColNM];

#if !defined __SYNTHESIS__ && XDEBUG == 1
    int cnt = 0;
    std::cout << std::hex << "merge_cfg0:" << merge_column_cfg1 << "   merge_cfg1:" << merge_column_cfg2 << std::endl;
#endif // !defined __SYNTHESIS__ && XDEBUG == 1

    bool e = e_in_strm.read();
    while (!e) {
#pragma HLS PIPELINE II = 1

        for (int i = 0; i < ColNM; i++) {
#pragma HLS UNROLL
            key[i] = in_key[i].read();
            pld0[i] = in_pld0[i].read();
            pld1[i] = in_pld1[i].read();
            pld2[i] = in_pld2[i].read();
        }
        e = e_in_strm.read();

#if !defined __SYNTHESIS__ && XDEBUG == 1
        if (cnt < 10) {
            std::cout << std::dec << "cnt=" << cnt << std::endl;
            std::cout << "key: ";
            for (int i = 0; i < ColNM; i++) {
                std::cout << "col" << i << ": " << key[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "pld0: ";
            for (int i = 0; i < ColNM; i++) {
                std::cout << "col" << i << ": " << pld0[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "pld1: ";
            for (int i = 0; i < ColNM; i++) {
                std::cout << "col" << i << ": " << pld1[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "pld2: ";
            for (int i = 0; i < ColNM; i++) {
                std::cout << "col" << i << ": " << pld2[i] << " ";
            }
            std::cout << std::endl;
        }
#endif // !defined __SYNTHESIS__ && XDEBUG == 1

        // merge_level1
        for (int i = 0; i < ColNM; i++) {
#pragma HLS UNROLL
            if (merge_level1_reverse == 0) {
                merge0[i] = merge_level1_cfg0[i] == 1 ? key[i] : pld0[i];
                merge1[i] = merge_level1_cfg1[i] == 1 ? key[i] : pld1[i];
                merge2[i] = merge_level1_cfg2[i] == 1 ? key[i] : pld2[i];
            } else {
                merge0[i] = merge_level1_cfg0[i] == 1 ? key[ColNM - 1 - i] : pld0[i];
                merge1[i] = merge_level1_cfg1[i] == 1 ? key[ColNM - 1 - i] : pld1[i];
                merge2[i] = merge_level1_cfg2[i] == 1 ? key[ColNM - 1 - i] : pld2[i];
            }
        }

#if !defined __SYNTHESIS__ && defined XDEBUG
        if (cnt < 10) {
            std::cout << std::dec << "cnt=" << cnt << std::endl;
            std::cout << "merge0: ";
            for (int i = 0; i < ColNM; i++) {
                std::cout << "col" << i << ": " << merge0[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "merge1: ";
            for (int i = 0; i < ColNM; i++) {
                std::cout << "col" << i << ": " << merge1[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "merge2: ";
            for (int i = 0; i < ColNM; i++) {
                std::cout << "col" << i << ": " << merge2[i] << " ";
            }
            std::cout << std::endl;
        }
#endif // !defined __SYNTHESIS__ && XDEBUG == 1

        // merge_level2 inverse direction
        for (int i = 0; i < ColNM; i++) {
#pragma HLS UNROLL
            if (merge_level2_reverse == 0) {
                out[i] = merge_level2_cfg0[i] == 1 ? merge0[i] : merge1[i];
                out[i + ColNM] = merge_level2_cfg1[i] == 1 ? merge0[i] : merge2[i];
            } else {
                out[i] = merge_level2_cfg0[i] == 1 ? merge0[ColNM - 1 - i] : merge1[i];
                out[i + ColNM] = merge_level2_cfg1[i] == 1 ? merge0[ColNM - 1 - i] : merge2[i];
            }
        }

        for (int i = 0; i < 2 * ColNM; i++) {
#pragma HLS UNROLL
            out_strm[i].write(out[i]);
        }
        e_out_strm.write(false);

#if !defined __SYNTHESIS__ && XDEBUG == 1
        cnt++;
#endif // !defined __SYNTHESIS__ && XDEBUG == 1
    }
    e_out_strm.write(true);
}

// eliminate strm end
template <int CHNM>
void eliminate_strm_end(hls::stream<bool>& strm_end) {
#pragma HLS inline off

    bool end = strm_end.read();
    while (!end) {
        end = strm_end.read();
    }
}

// group aggr wrapper
template <int _WStrm,
          int _ColNM,
          int _HashMode,
          int _WHashHigh,
          int _WHashLow,
          int _CHNM,
          int _Wcnt,
          int _WBuffer,
          int _BurstLenW = 32,
          int _BurstLenR = 32>
void group_aggregate_wrapper(
    // stream in
    hls::stream<ap_uint<_WStrm> > strm_in[_CHNM][_ColNM],
    hls::stream<bool> strm_e_in[_CHNM],

    // control param
    hls::stream<ap_uint<_ColNM * _ColNM> > shuffle_key_cfg[_CHNM],
    hls::stream<ap_uint<_ColNM * _ColNM> > shuffle_pld_cfg[_CHNM],
    hls::stream<ap_uint<32> >& merge_column_cfg,
    hls::stream<ap_uint<32> >& aggr_cfg,
    hls::stream<ap_uint<32> >& aggr_result_info,

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
    hls::stream<ap_uint<_WStrm> > strm_out[2 * _ColNM],
    hls::stream<bool>& strm_e_out) {
#pragma HLS dataflow

    hls::stream<ap_uint<_WStrm> > dup0_strm[_CHNM][_ColNM];
#pragma HLS stream variable = dup0_strm depth = 8
#pragma HLS array_partition variable = dup0_strm complete
#pragma HLS bind_storage variable = dup0_strm type = fifo impl = lutram
    hls::stream<ap_uint<_WStrm> > dup1_strm[_CHNM][_ColNM];
#pragma HLS stream variable = dup1_strm depth = 8
#pragma HLS array_partition variable = dup1_strm complete
#pragma HLS bind_storage variable = dup1_strm type = fifo impl = lutram
    hls::stream<bool> e0_strm[_CHNM];
#pragma HLS stream variable = e0_strm depth = 8
#pragma HLS array_partition variable = e0_strm complete
#pragma HLS bind_storage variable = e0_strm type = fifo impl = lutram
    hls::stream<bool> e1_strm[_CHNM];
#pragma HLS stream variable = e1_strm depth = 8
#pragma HLS array_partition variable = e1_strm complete
#pragma HLS bind_storage variable = e1_strm type = fifo impl = lutram

    hls::stream<ap_uint<_WStrm> > strm_key_in[_CHNM][_ColNM];
#pragma HLS stream variable = strm_key_in depth = 8
#pragma HLS array_partition variable = strm_key_in complete
#pragma HLS bind_storage variable = strm_key_in type = fifo impl = lutram
    hls::stream<ap_uint<_WStrm> > strm_pld_in[_CHNM][_ColNM];
#pragma HLS stream variable = strm_pld_in depth = 8
#pragma HLS array_partition variable = strm_pld_in complete
#pragma HLS bind_storage variable = strm_pld_in type = fifo impl = lutram
    hls::stream<bool> e2_strm[_CHNM];
#pragma HLS stream variable = e2_strm depth = 8
#pragma HLS array_partition variable = e2_strm complete
#pragma HLS bind_storage variable = e1_strm type = fifo impl = lutram
    hls::stream<bool> e3_strm[_CHNM];
#pragma HLS stream variable = e3_strm depth = 8
#pragma HLS bind_storage variable = e3_strm type = fifo impl = lutram

    hls::stream<ap_uint<_WStrm> > strm_key_out[_ColNM];
#pragma HLS stream variable = strm_key_out depth = 512
#pragma HLS array_partition variable = strm_key_out complete
#pragma HLS bind_storage variable = strm_key_out type = fifo impl = bram
    hls::stream<ap_uint<_WStrm> > strm_pld_out[3][_ColNM];
#pragma HLS stream variable = strm_pld_out depth = 512
#pragma HLS array_partition variable = strm_pld_out complete
#pragma HLS bind_storage variable = strm_pld_out type = fifo impl = bram
    hls::stream<bool> e4_strm;
#pragma HLS stream variable = e4_strm depth = 512
#pragma HLS bind_storage variable = e4_strm type = fifo impl = lutram

    for (int i = 0; i < _CHNM; i++) {
#pragma HLS unroll

        // duplicate col streams
        dup_strm<_WStrm, _ColNM>(strm_in[i], strm_e_in[i], dup0_strm[i], e0_strm[i], dup1_strm[i], e1_strm[i]);

        // shuffle key
        xf::common::utils_hw::streamShuffle<_ColNM, _ColNM>(shuffle_key_cfg[i], dup0_strm[i], e0_strm[i],
                                                            strm_key_in[i], e2_strm[i]);

        // shuffle pld
        xf::common::utils_hw::streamShuffle<_ColNM, _ColNM>(shuffle_pld_cfg[i], dup1_strm[i], e1_strm[i],
                                                            strm_pld_in[i], e3_strm[i]);

        // consume end streams
        eliminate_strm_end<_CHNM>(e2_strm[i]);
    }

    // group aggr primitive
    xf::database::hashGroupAggregate<_WStrm, _ColNM, _WStrm, _ColNM, _HashMode, _WHashHigh, _WHashLow, _CHNM, _Wcnt,
                                     _WBuffer, _BurstLenW, _BurstLenR>(
        strm_key_in, strm_pld_in, e3_strm, aggr_cfg, aggr_result_info, ping_buf0, ping_buf1, ping_buf2, ping_buf3,
        pong_buf0, pong_buf1, pong_buf2, pong_buf3, strm_key_out, strm_pld_out, e4_strm);

    // merge column from 32 to 16
    merge_column<_WStrm, _ColNM>(merge_column_cfg, strm_key_out, strm_pld_out[0], strm_pld_out[1], strm_pld_out[2],
                                 e4_strm, strm_out, strm_e_out);
}

} // namespace gqe
} // namespace database
} // namespace xf

#endif
