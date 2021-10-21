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
#ifndef GQE_STREAM_HELPER_HPP
#define GQE_STREAM_HELPER_HPP

#include "ap_int.h"
#include "hls_stream.h"

#ifndef __SYNTHESIS__
#include <stdio.h>
#endif

namespace xf {
namespace database {
namespace gqe {

/* @brief stream_demux switch one common input stream to one of several seperate output stream
 *
 * @tparam -WCfg input configuration width.
 * @tparam _WIn input stream width.
 * @tparam _NOutStrm number of output stream.
 *
 * @param select_cfg binary encoded selection, used to select which stream to output.
 * @param istrm input data streams.
 * @param e_istrm end flag for input stream.
 * @param ostrms output data stream array.
 * @param e_ostrm end flag stream for output data.
 */
template <int _WCfg, int _WIn, int _NOutStrm>
void stream_demux(hls::stream<ap_uint<_WCfg> >& select_cfg,

                  hls::stream<ap_uint<_WIn> >& istrm,
                  hls::stream<bool>& e_istrm,

                  hls::stream<ap_uint<_WIn> > ostrms[_NOutStrm],
                  hls::stream<bool> e_ostrm[_NOutStrm]) {
    ap_uint<_WCfg> cfg = select_cfg.read();
    bool e = e_istrm.read();

    while (!e) {
#pragma HLS pipeline II = 1
        ap_uint<_WIn> in = istrm.read();
        ostrms[cfg].write(in);
        e_ostrm[cfg].write(false);
        e = e_istrm.read();
    }
    e_ostrm[cfg].write(true);
#if !defined __SYNTHESIS__ && XDEBUG == 1
    if (istrm.size() != 0) printf("input stream left data %d after demux", istrm.size());
#endif
}

/* @brief stream_mux switch several input input stream to one output stream
 *
 * @tparam _WCfg input configuration width.
 * @tparam _WIn input stream width.
 * @tparam _NInStrm number of input stream.
 *
 * @param select_cfg binary encoded selection, used to select which stream to input.
 * @param istrms input data streams array.
 * @param e_istrm end flag for input stream.
 * @param ostrm output data stream
 * @param e_ostrm end flag stream for output data.
 */
template <int _WCfg, int _WIn, int _NInStrm>
void stream_mux(hls::stream<ap_uint<_WCfg> >& select_cfg,

                hls::stream<ap_uint<_WIn> > istrms[_NInStrm],
                hls::stream<bool> e_istrm[_NInStrm],

                hls::stream<ap_uint<_WIn> >& ostrm,
                hls::stream<bool>& e_ostrm) {
    ap_uint<_WCfg> cfg = select_cfg.read();
    bool e = e_istrm[cfg].read();

    while (!e) {
#pragma HLS pipeline II = 1
        ap_uint<_WIn> in = istrms[cfg].read();
        ostrm.write(in);
        e_ostrm.write(false);
        e = e_istrm[cfg].read();
    }
    e_ostrm.write(true);
#if !defined __SYNTHESIS__ && XDEBUG == 1
    for (int i = 0; i < _NInStrm; ++i) {
        if (istrms[i].size() != 0) printf("Input stream %d left data %d after mux\n", i, istrms[i].size());
    }
#endif
}

/* @brief 1Dstream_demux1To2 swtich 1 common input flow to 2 ouput flow. where the input is a 1 dimensional array.
 *
 * @tparam _WIn input stream width.
 * @tparam _NStrm size of 1th-dim for input stream array.
 *
 * @param select_cfg binary encoded selection, used to select which stream to input.
 * @param istrms inpput data stream
 * @param e_istrm end flag stream for input data.
 *
 * @param ostrms_0 first output data streams array.
 * @param ostrms_1 second output data streams array.
 * @param e_ostrm_0 end flag for first output stream.
 * @param e_ostrm_1 end flag for second output stream.
 *
 */
template <int _WIn, int _NStrm>
void stream1D_demux1To2(hls::stream<bool>& select_cfg,

                        hls::stream<ap_uint<_WIn> > istrms[_NStrm],
                        hls::stream<bool>& e_istrm,

                        hls::stream<ap_uint<_WIn> > ostrms_0[_NStrm],
                        hls::stream<ap_uint<_WIn> > ostrms_1[_NStrm],
                        hls::stream<bool>& e_ostrm_0,
                        hls::stream<bool>& e_ostrm_1) {
    bool cfg = select_cfg.read();
#if !defined __SYNTHESIS__ && XDEBUG == 1
    for (int i = 0; i < _NStrm; ++i) {
        printf("Input stream %d left data %d before mux \n", i, istrms[i].size());
    }
    printf("Input stream left end %d before mux \n", e_istrm.size());
#endif
    bool e = e_istrm.read();
    while (!e) {
#pragma HLS pipeline II = 1
        for (int i = 0; i < _NStrm; ++i) {
#pragma HLS unroll
            ap_uint<_WIn> in = istrms[i].read();
            if (cfg == 0) {
                ostrms_0[i].write(in);
            } else {
                ostrms_1[i].write(in);
            }
        }
        if (cfg == 0) {
            e_ostrm_0.write(false);
        } else {
            e_ostrm_1.write(false);
        }
        e = e_istrm.read();
    }
    if (cfg == 0) {
        e_ostrm_0.write(true);
    } else {
        e_ostrm_1.write(true);
    }
#if !defined __SYNTHESIS__ && XDEBUG == 1
    for (int i = 0; i < _NStrm; ++i) {
        if (istrms[i].size() != 0) printf("Input stream %d left data %d after demux\n", i, istrms[i].size());
    }
#endif
}
/* @brief 1Dtream_mux2To1 select the data from 2 input and output 1, where the input is a 1 dimensional array.
 *
 * @tparam _WIn input stream width.
 * @tparam _NStrm size of 1th-dim for input stream array.
 *
 * @param select_cfg binary encoded selection, used to select which stream to input.
 * @param istrms_0 first input data streams array.
 * @param istrms_1 second input data streams array.
 * @param e_istrm_0 end flag for first input stream.
 * @param e_istrm_1 end flag for second input stream.
 *
 * @param ostrms output data stream
 * @param e_ostrm end flag stream for output data.
 */

template <int _WIn, int _NStrm, int _NStrm2>
void stream1D_mux2To1(hls::stream<ap_uint<6> >& i_join_cfg_strm,
                      hls::stream<ap_uint<6> >& o_join_cfg_strm,

                      hls::stream<ap_uint<_WIn> > istrms_0[_NStrm],
                      hls::stream<ap_uint<_WIn> > istrms_1[_NStrm2],
                      hls::stream<bool>& e_istrm_0,
                      hls::stream<bool>& e_istrm_1,

                      hls::stream<ap_uint<_WIn> > ostrms[_NStrm],
                      hls::stream<bool>& e_ostrm) {
    ap_uint<6> join_cfg = i_join_cfg_strm.read();
    o_join_cfg_strm.write(join_cfg);
    bool cfg = join_cfg[0];
    bool e;
#if !defined __SYNTHESIS__ && XDEBUG == 1
    for (int i = 0; i < _NStrm2; ++i) {
        if (i < _NStrm) {
            printf("Input-0 stream %d left data %d before mux \n", i, istrms_0[i].size());
            printf("Input-1 stream %d left data %d before mux \n", i, istrms_1[i].size());
        }
    }
    printf("Input-0 stream left end %d before mux \n", e_istrm_0.size());
    printf("Input-1 stream left end %d before mux \n", e_istrm_1.size());
#endif
    if (cfg == 0) {
        e = e_istrm_0.read();
    } else {
        e = e_istrm_1.read();
    }
    int cnt = 0;
    while (!e) {
#pragma HLS pipeline II = 1
        if (cfg == 0) {
            e = e_istrm_0.read();
        } else {
            e = e_istrm_1.read();
        }

        for (int i = 0; i < _NStrm2; ++i) {
#pragma HLS unroll
            ap_uint<_WIn> in;
            if (cfg == 0) {
                if (i < _NStrm) {
                    in = istrms_0[i].read();
                } else {
                    in = 0;
                }
            } else {
                in = istrms_1[i].read();
            }
            ostrms[i].write(in);
        }
        cnt++;
        e_ostrm.write(false);
    }
    e_ostrm.write(true);
#if !defined __SYNTHESIS__ && XDEBUG == 1
    for (int i = 0; i < _NStrm2; ++i) {
        if (i < _NStrm) {
            if (istrms_0[i].size() != 0) printf("Input-0 stream %d left data %d after mux \n", i, istrms_0[i].size());
            if (istrms_1[i].size() != 0) printf("Input-1 stream %d left data %d after mux \n", i, istrms_1[i].size());
        }
    }
#endif
}
/* @brief 1Dtream_mux2To1 select the data from 2 input and output 1, where the input is a 1 dimensional array.
 *
 * @tparam _WIn input stream width.
 * @tparam _NStrm size of 1th-dim for input stream array.
 *
 * @param select_cfg binary encoded selection, used to select which stream to input.
 * @param istrms_0 first input data streams array.
 * @param istrms_1 second input data streams array.
 * @param e_istrm_0 end flag for first input stream.
 * @param e_istrm_1 end flag for second input stream.
 *
 * @param ostrms output data stream
 * @param e_ostrm end flag stream for output data.
 */

template <int _WIn, int _NStrm>
void stream1D_mux2To1(hls::stream<bool>& select_cfg,

                      hls::stream<ap_uint<_WIn> > istrms_0[_NStrm],
                      hls::stream<ap_uint<_WIn> > istrms_1[_NStrm],
                      hls::stream<bool>& e_istrm_0,
                      hls::stream<bool>& e_istrm_1,

                      hls::stream<ap_uint<_WIn> > ostrms[_NStrm],
                      hls::stream<bool>& e_ostrm) {
    bool cfg = select_cfg.read();
    bool e;
#if !defined __SYNTHESIS__ && XDEBUG == 1
    for (int i = 0; i < _NStrm; ++i) {
        printf("Input-0 stream %d left data %d before mux \n", i, istrms_0[i].size());
        printf("Input-1 stream %d left data %d before mux \n", i, istrms_1[i].size());
    }
    printf("Input-0 stream left end %d before mux \n", e_istrm_0.size());
    printf("Input-1 stream left end %d before mux \n", e_istrm_1.size());
#endif
    if (cfg == 0) {
        e = e_istrm_0.read();
    } else {
        e = e_istrm_1.read();
    }
    int cnt = 0;
    while (!e) {
#pragma HLS pipeline II = 1
        if (cfg == 0) {
            e = e_istrm_0.read();
        } else {
            e = e_istrm_1.read();
        }

        for (int i = 0; i < _NStrm; ++i) {
#pragma HLS unroll
            ap_uint<_WIn> in;
            if (cfg == 0) {
                in = istrms_0[i].read();
            } else {
                in = istrms_1[i].read();
            }
            ostrms[i].write(in);
        }
        cnt++;
        e_ostrm.write(false);
    }
    e_ostrm.write(true);
#if !defined __SYNTHESIS__ && XDEBUG == 1
    for (int i = 0; i < _NStrm; ++i) {
        if (istrms_0[i].size() != 0) printf("Input-0 stream %d left data %d after mux \n", i, istrms_0[i].size());
        if (istrms_1[i].size() != 0) printf("Input-1 stream %d left data %d after mux \n", i, istrms_1[i].size());
    }
#endif
}

/* @brief 2Dstream_demux1To2 swtich 1 common input flow to 2 ouput flow. where the input is a 2 dimensional array.
 *
 * @tparam _WIn input stream width.
 * @tparam _MStrm size of 1th-dim for input stream array.
 * @tparam _NStrm size of 2nd-dim for input stream array.
 *
 * @param select_cfg binary encoded selection, used to select which stream to input.
 * @param istrms inpput data stream
 * @param e_istrm end flag stream for input data.
 *
 * @param ostrms_0 first output data streams array.
 * @param ostrms_1 second output data streams array.
 * @param e_ostrm_0 end flag for first output stream.
 * @param e_ostrm_1 end flag for second output stream.
 *
 */
template <int _WIn, int _MStrm, int _NStrm>
void stream2D_demux1To2(hls::stream<bool>& select_cfg,

                        hls::stream<ap_uint<_WIn> > istrms[_MStrm][_NStrm],
                        hls::stream<bool> e_istrm[_MStrm],

                        hls::stream<ap_uint<_WIn> > ostrms_0[_MStrm][_NStrm],
                        hls::stream<ap_uint<_WIn> > ostrms_1[_MStrm][_NStrm],
                        hls::stream<bool> e_ostrm_0[_MStrm],
                        hls::stream<bool> e_ostrm_1[_MStrm]) {
    hls::stream<bool> dup_sel_cfg[_MStrm];
#pragma HLS stream variable = dup_sel_cfg depth = 1
#pragma HLS array_partition variable = dup_sel_cfg dim = 0
    bool cfg = select_cfg.read();
    for (int i = 0; i < _MStrm; ++i) {
#pragma HLS unroll
        dup_sel_cfg[i].write(cfg);
    }
    for (int i = 0; i < _MStrm; ++i) {
#pragma HLS unroll
        stream1D_demux1To2<_WIn, _NStrm>(dup_sel_cfg[i], istrms[i], e_istrm[i], ostrms_0[i], ostrms_1[i], e_ostrm_0[i],
                                         e_ostrm_1[i]);
    }
}
/* @brief 2Dtream_mux2To1 select the data from 2 input and output 1, where the input is a 2 dimensional array.
 *
 * @tparam _WIn input stream width.
 * @tparam _MStrm number of 1th-dim for input stream.
 * @tparam _NStrm number of 2nd-dim for input stream.
 *
 * @param select_cfg binary encoded selection, used to select which stream to input.
 * @param istrms_0 first input data streams array.
 * @param istrms_1 second input data streams array.
 * @param e_istrm_0 end flag for first input stream.
 * @param e_istrm_1 end flag for second input stream.
 *
 * @param ostrms output data stream
 * @param e_ostrm end flag stream for output data.
 */
template <int _WIn, int _MStrm, int _NStrm>
void stream2D_mux2To1(hls::stream<bool>& select_cfg,

                      hls::stream<ap_uint<_WIn> > istrms_0[_MStrm][_NStrm],
                      hls::stream<ap_uint<_WIn> > istrms_1[_MStrm][_NStrm],
                      hls::stream<bool> e_istrm_0[_MStrm],
                      hls::stream<bool> e_istrm_1[_MStrm],

                      hls::stream<ap_uint<_WIn> > ostrms[_MStrm][_NStrm],
                      hls::stream<bool> e_ostrm[_MStrm]) {
    hls::stream<bool> dup_sel_cfg[_MStrm];
#pragma HLS stream variable = dup_sel_cfg depth = 1
#pragma HLS array_partition variable = dup_sel_cfg dim = 0
    bool cfg = select_cfg.read();
    for (int i = 0; i < _MStrm; ++i) {
#pragma HLS unroll
        dup_sel_cfg[i].write(cfg);
    }
    for (int i = 0; i < _MStrm; ++i) {
#pragma HLS unroll
        stream1D_mux2To1<_WIn, _NStrm>(dup_sel_cfg[i], istrms_0[i], istrms_1[i], e_istrm_0[i], e_istrm_1[i], ostrms[i],
                                       e_ostrm[i]);
    }
}

template <int COL_NM, int CH_NM, int TB_NM>
void demux_wrapper(hls::stream<bool>& join_on_strm,
                   hls::stream<ap_uint<8 * TPCH_INT_SZ> > istrm[CH_NM][COL_NM],
                   hls::stream<bool> e_istrm[CH_NM],
                   hls::stream<ap_uint<8 * TPCH_INT_SZ> > ostrm_0[CH_NM][COL_NM],
                   hls::stream<ap_uint<8 * TPCH_INT_SZ> > ostrm_1[CH_NM][COL_NM],
                   hls::stream<bool> e_ostrm_0[CH_NM],
                   hls::stream<bool> e_ostrm_1[CH_NM]) {
    bool join_on = join_on_strm.read();
    hls::stream<bool> jn_on_strm;
#pragma HLS stream variable = jn_on_strm depth = 3
    int lp_nm = join_on ? TB_NM : 1;
    for (int i = 0; i < lp_nm; ++i) {
        jn_on_strm.write(join_on);
        stream2D_demux1To2<8 * TPCH_INT_SZ, CH_NM, COL_NM>(jn_on_strm, istrm, e_istrm, ostrm_0, ostrm_1, e_ostrm_0,
                                                           e_ostrm_1);
    }
    // add more data to prepare for do-while
    if (!join_on) {
        for (int col = 0; col < COL_NM; ++col) {
#pragma HLS pipeline II = 1
            for (int ch = 0; ch < CH_NM; ++ch) {
#pragma HLS unroll
                ostrm_0[ch][col].write(0);
            }
        }
    }
}
template <int COL_NM, int CH_NM, int TB_NM>
void demux_wrapper(hls::stream<ap_uint<6> >& i_join_cfg_strm,
                   hls::stream<ap_uint<8 * TPCH_INT_SZ> > istrm[CH_NM][COL_NM],
                   hls::stream<bool> e_istrm[CH_NM],
                   hls::stream<ap_uint<6> > o_join_cfg_strm[2],
                   hls::stream<ap_uint<8 * TPCH_INT_SZ> > ostrm_0[CH_NM][COL_NM],
                   hls::stream<ap_uint<8 * TPCH_INT_SZ> > ostrm_1[CH_NM][COL_NM],
                   hls::stream<bool> e_ostrm_0[CH_NM],
                   hls::stream<bool> e_ostrm_1[CH_NM]) {
    ap_uint<6> join_cfg = i_join_cfg_strm.read();
    o_join_cfg_strm[0].write(join_cfg);
    o_join_cfg_strm[1].write(join_cfg);
    bool join_on = join_cfg[0];
    hls::stream<bool> jn_on_strm;
#pragma HLS stream variable = jn_on_strm depth = 3
    int lp_nm = join_on ? TB_NM : 1;
    for (int i = 0; i < lp_nm; ++i) {
        jn_on_strm.write(join_on);
        stream2D_demux1To2<8 * TPCH_INT_SZ, CH_NM, COL_NM>(jn_on_strm, istrm, e_istrm, ostrm_0, ostrm_1, e_ostrm_0,
                                                           e_ostrm_1);
    }
    // add more data to prepare for do-while
    if (!join_on) {
        for (int col = 0; col < COL_NM; ++col) {
#pragma HLS pipeline II = 1
            for (int ch = 0; ch < CH_NM; ++ch) {
#pragma HLS unroll
                ostrm_0[ch][col].write(0);
            }
        }
    }
}

} // namespace gqe
} // namespace database
} // namespace xf

#endif
