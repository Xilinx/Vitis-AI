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
#ifndef GQE_EVAL_PART_HPP
#define GQE_EVAL_PART_HPP

#include <hls_stream.h>
#include <ap_int.h>

#include "xf_utils_hw/stream_shuffle.hpp"
#include "xf_utils_hw/stream_dup.hpp"
#include "xf_database/dynamic_eval.hpp"
#include "xf_database/gqe_blocks/gqe_types.hpp"

namespace xf {
namespace database {
namespace gqe {

// scaling
template <int WStrm = 32>
void scaling(hls::stream<ap_uint<2 * WStrm> >& in_strm,
             hls::stream<bool>& i_e_strm,
             ap_uint<3> scaling_cfg,
             hls::stream<ap_uint<WStrm> >& out_strm,
             hls::stream<bool>& o_e_strm) {
#pragma HLS inline off

#if !defined __SYNTHESIS__ && XDEBUG == 1
    int cnt = 0;
#endif

    ap_uint<2> scale = scaling_cfg(1, 0);
    ap_uint<1> enable = scaling_cfg[2];

    ap_uint<64> scale_factor;

#if !defined __SYNTHESIS__ && XDEBUG == 1
    std::cout << std::hex << "scaling: cfg=" << scaling_cfg << std::endl;
#endif // !defined __SYNTHESIS__ && XDEBUG == 1

    if (scale == 0) {
        scale_factor = 10;
    } else if (scale == 1) {
        scale_factor = 100;
    } else if (scale == 2) {
        scale_factor = 1000;
    } else if (scale == 3) {
        scale_factor = 10000;
    }

    bool last = i_e_strm.read();
    while (!last) {
#pragma HLS pipeline II = 1

        ap_uint<2 * WStrm> in = in_strm.read();
        last = i_e_strm.read();

        ap_uint<WStrm> out;
        if (enable == 1) {
            if (scale == 0)
                out = (in / 10)(WStrm - 1, 0);
            else if (scale == 1)
                out = (in / 100)(WStrm - 1, 0);
            else if (scale == 2)
                out = (in / 1000)(WStrm - 1, 0);
            else if (scale == 3)
                out = (in / 10000)(WStrm - 1, 0);
        } else {
            out = in(WStrm - 1, 0);
        }

#if !defined __SYNTHESIS__ && defined XDEBUG
        if (cnt < 10) {
            std::cout << std::dec << "scaling: in=" << in << " scale_factor=" << scale_factor << std::endl;
            std::cout << std::dec << "scaling: out=" << out << " temp=" << temp << std::endl;
            cnt++;
        }
#endif

        out_strm.write(out);
        o_e_strm.write(false);
    }
    o_e_strm.write(true);
}

// combine new column to high column
template <int WStrm, int ColNM0, int ColNM1, int ColNM2>
void combine(hls::stream<ap_uint<WStrm> > in_strm0[ColNM0],
             hls::stream<ap_uint<WStrm> > in_strm1[ColNM1],
             hls::stream<ap_uint<WStrm> > in_strm2[ColNM2],
             hls::stream<bool>& e_in_strm,
             hls::stream<ap_uint<WStrm> > out_strm[ColNM0 + ColNM1 + ColNM2],
             hls::stream<bool>& e_out_strm) {
#if !defined __SYNTHESIS__ && XDEBUG == 1
    int cnt = 0;
#endif // !defined __SYNTHESIS__ && XDEBUG == 1

    bool e = e_in_strm.read();
    while (!e) {
#pragma HLS PIPELINE II = 1

        ap_uint<WStrm> temp[ColNM0 + ColNM1 + ColNM2];
#pragma HLS array_partition variable = temp complete

        e = e_in_strm.read();
        for (int i = 0; i < ColNM0; i++) {
#pragma HLS UNROLL
            temp[i] = in_strm0[i].read();
            out_strm[i].write(temp[i]);
        }

        for (int i = 0; i < ColNM1; i++) {
#pragma HLS UNROLL
            temp[i + ColNM0] = in_strm1[i].read();
            out_strm[i + ColNM0].write(temp[i + ColNM0]);
        }

        for (int i = 0; i < ColNM2; i++) {
#pragma HLS UNROLL
            temp[i + ColNM0 + ColNM1] = in_strm2[i].read();
            out_strm[i + ColNM0 + ColNM1].write(temp[i + ColNM0 + ColNM1]);
        }

#if !defined __SYNTHESIS__ && defined XDEBUG
        if (cnt < 10) {
            std::cout << "eval: ";
            for (int i = 0; i < ColNM0 + ColNM1 + ColNM2; i++) {
                std::cout << "col" << i << ": " << temp[i] << " ";
            }
            std::cout << std::endl;
            cnt++;
        }
#endif

        e_out_strm.write(false);
    }
    e_out_strm.write(true);
}

// split low bit column
template <int WStrm, int ColNM0, int ColNM1>
void split(hls::stream<ap_uint<WStrm> > in_strm[ColNM0 + ColNM1],
           hls::stream<bool>& e_in_strm,
           hls::stream<ap_uint<WStrm> > out_strm0[ColNM0],
           hls::stream<ap_uint<WStrm> > out_strm1[ColNM1],
           hls::stream<bool>& e_out_strm) {
    bool e = e_in_strm.read();
    while (!e) {
#pragma HLS PIPELINE II = 1

        e = e_in_strm.read();
        for (int i = 0; i < ColNM0; i++) {
#pragma HLS UNROLL
            out_strm0[i].write(in_strm[i].read());
        }

        for (int i = 0; i < ColNM1; i++) {
#pragma HLS UNROLL
            out_strm1[i].write(in_strm[i + ColNM0].read());
        }
        e_out_strm.write(false);
    }
    e_out_strm.write(true);
}

// dynamic eval wrapper: plit + dup + eval + cpmbine + shuffle
template <int ColNM, int WStrm, int WConst>
void dynamic_eval_wrapper(ap_uint<289> alu_cfg,
                          ap_uint<3> scaling_cfg,
                          hls::stream<ap_uint<ColNM * ColNM> >& shuffle_cfg_strm,
                          hls::stream<ap_uint<WStrm> > in_strm[ColNM],
                          hls::stream<bool>& e_in_strm,
                          hls::stream<ap_uint<WStrm> > out_strm[ColNM],
                          hls::stream<bool>& e_out_strm) {
#pragma HLS dataflow

    hls::stream<ap_uint<WStrm> > split0_strm[ColNM - 4];
#pragma HLS stream variable = split0_strm depth = 512
#pragma HLS array_partition variable = split0_strm complete
#pragma HLS bind_storage variable = split0_strm type = fifo impl = bram
    hls::stream<ap_uint<WStrm> > split1_strm[4];
#pragma HLS stream variable = split1_strm depth = 8
#pragma HLS array_partition variable = split1_strm complete
    hls::stream<bool> e0_strm;
#pragma HLS stream variable = e0_strm depth = 8

    const unsigned int choose[4] = {0, 1, 2, 3};
    hls::stream<ap_uint<WStrm> > dup0_strm[4];
#pragma HLS stream variable = dup0_strm depth = 512
#pragma HLS array_partition variable = dup0_strm complete
#pragma HLS bind_storage variable = dup0_strm type = fifo impl = bram
    hls::stream<ap_uint<WStrm> > dup1_strm[1][4];
#pragma HLS stream variable = dup1_strm depth = 8
#pragma HLS array_partition variable = dup1_strm complete
    hls::stream<bool> e1_strm;
#pragma HLS stream variable = e1_strm depth = 8

    hls::stream<ap_uint<2 * WStrm> > eval_strm;
#pragma HLS stream variable = eval_strm depth = 16
    hls::stream<bool> e2_strm;
#pragma HLS stream variable = e2_strm depth = 16

    hls::stream<ap_uint<WStrm> > scaling_strm[1];
#pragma HLS stream variable = scaling_strm depth = 8
    hls::stream<bool> e3_strm;
#pragma HLS stream variable = e3_strm depth = 8

    hls::stream<ap_uint<WStrm> > combine_strm[ColNM + 1];
#pragma HLS stream variable = combine_strm depth = 8
#pragma HLS array_partition variable = combine_strm complete
    hls::stream<bool> e4_strm;
#pragma HLS stream variable = e4_strm depth = 8

    // split col for duplicate
    split<WStrm, 4, ColNM - 4>(in_strm, e_in_strm, split1_strm, split0_strm, e0_strm);

    // duplicate 4 col streams
    xf::common::utils_hw::streamDup<ap_uint<WStrm>, 4, 4, 1>(choose, split1_strm, e0_strm, dup0_strm, dup1_strm,
                                                             e1_strm);

    // evaluate into 5th col stream
    xf::database::dynamicEval<ap_uint<WStrm>, ap_uint<WStrm>, ap_uint<WStrm>, ap_uint<WStrm>, ap_uint<WConst>,
                              ap_uint<WConst>, ap_uint<WConst>, ap_uint<WConst>,
                              ap_uint<2 * WStrm> >(alu_cfg, // cfg
                                                   dup1_strm[0][0], dup1_strm[0][1], dup1_strm[0][2], dup1_strm[0][3],
                                                   e1_strm,             // in
                                                   eval_strm, e2_strm); // out

    // scaling
    scaling<WStrm>(eval_strm, e2_strm, scaling_cfg, scaling_strm[0], e3_strm);

    // combine eval result with original stream
    combine<WStrm, 4, ColNM - 4, 1>(dup0_strm, split0_strm, scaling_strm, e3_strm, combine_strm, e4_strm);

    // shuffle stream order
    xf::common::utils_hw::streamShuffle<ColNM + 1, ColNM, ap_uint<WStrm> >(shuffle_cfg_strm, combine_strm, e4_strm,
                                                                           out_strm, e_out_strm);
}

// muti channel dynamic eval
template <int CHNM, int ColNM, int WStrm, int WConst>
void multi_dynamic_eval_wrapper(hls::stream<ap_uint<32> >& alu_cfg_strm,
                                hls::stream<ap_uint<ColNM * ColNM> > shuffle_cfg_strm[CHNM],
                                hls::stream<ap_uint<WStrm> > in_strm[CHNM][ColNM],
                                hls::stream<bool> e_in_strm[CHNM],
                                hls::stream<ap_uint<WStrm> > out_strm[CHNM][ColNM],
                                hls::stream<bool> e_out_strm[CHNM]) {
#pragma HLS inline off

    ap_uint<320> alu_cfg;
    ap_uint<3> scaling_cfg;

    alu_cfg(31, 0) = alu_cfg_strm.read();
    for (int i = 0; i < 9; i++) {
#pragma HLS pipeline II = 1
        alu_cfg(319, 32) = alu_cfg(287, 0);
        alu_cfg(31, 0) = alu_cfg_strm.read();
    }
    scaling_cfg = alu_cfg(291, 289);

#if !defined __SYNTHESIS__ && XDEBUG == 1
    std::cout << std::hex << "alu_cfg:" << alu_cfg << std::endl;
#endif // !defined __SYNTHESIS__ && XDEBUG == 1

    for (int i = 0; i < CHNM; i++) {
#pragma HLS unroll

        dynamic_eval_wrapper<ColNM, WStrm, WConst>(alu_cfg(288, 0), scaling_cfg, shuffle_cfg_strm[i], in_strm[i],
                                                   e_in_strm[i], out_strm[i], e_out_strm[i]);
    }
}

} // namespace gqe
} // namespace database
} // namespace xf

#endif
