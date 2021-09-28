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
#ifndef XF_UTILS_HW_STREAM_1N_LB_H
#define XF_UTILS_HW_STREAM_1N_LB_H

/**
 * @file stream_one_to_n/load_balance.hpp
 * @brief header for distribute in load-balancing way.
 */

#include "xf_utils_hw/types.hpp"
#include "xf_utils_hw/enums.hpp"
#include "xf_utils_hw/common.hpp"

// Forward decl ===============================================================

namespace xf {
namespace common {
namespace utils_hw {

/**
 * @brief stream distribute, using load-balancing algorithm.
 *
 * The input stream is assumed to be compact data to be splitted into
 * _WOutStrm wide data into output streams.
 *
 * @tparam _WInStrm input stream width.
 * @tparam _WOutStrm output stream width.
 * @tparam _NStrm number of output stream.
 *
 * @param istrm input data stream.
 * @param e_istrm end flag stream for input data.
 * @param ostrms output data streams.
 * @param e_ostrms end flag streams, one for each output data stream.
 * @param alg algorithm selector.
 */
template <int _WInStrm, int _WOutStrm, int _NStrm>
void streamOneToN(hls::stream<ap_uint<_WInStrm> >& istrm,
                  hls::stream<bool>& e_istrm,
                  hls::stream<ap_uint<_WOutStrm> > ostrms[_NStrm],
                  hls::stream<bool> e_ostrms[_NStrm],
                  LoadBalanceT alg);

/**
 * @brief stream distribute, using load-balancing algorithm.
 *
 * @tparam _TIn the type of input stream.
 * @tparam _NStrm number of output stream.
 *
 * @param istrm input data stream.
 * @param e_istrm end flag stream for input data.
 * @param ostrms output data streams.
 * @param e_ostrms end flag streams, one for each output data stream.
 * @param alg algorithm selector.
 */
template <typename _TIn, int _NStrm>
void streamOneToN(hls::stream<_TIn>& istrm,
                  hls::stream<bool>& e_istrm,
                  hls::stream<_TIn> ostrms[_NStrm],
                  hls::stream<bool> e_ostrms[_NStrm],
                  LoadBalanceT alg);

} // utils_hw
} // common
} // xf

// Implementation =============================================================

namespace xf {
namespace common {
namespace utils_hw {

// support ap_uint<>
namespace details {

/* @brief  buffer data to solve different input and output width.
 *
 * @tparam _WInStrm input stream width.
 * @tparam _WOutStrm output stream width.
 * @tparam _NStrm number of output stream.
 *
 * @param istrm input data stream.
 * @param e_istrm end flag stream for input data.
 * @param buf_lcm_strm output data streams.
 * @param left_lcm the number of useful data(_WInStrm) in last output data.
 * @param e_buf_lcm_ostrms end flag streams
 */
template <int _WInStrm, int _WOutStrm, int _NStrm>
void stream_one_to_n_read(hls::stream<ap_uint<_WInStrm> >& istrm,
                          hls::stream<bool>& e_istrm,
                          hls::stream<ap_uint<LCM<_WInStrm, _NStrm * _WOutStrm>::value> >& buf_lcm_strm,
                          hls::stream<int>& left_lcm,
                          hls::stream<bool>& e_buf_lcm_strm) {
    const int buf_width = LCM<_WInStrm, _NStrm * _WOutStrm>::value;
    const int num_in = buf_width / _WInStrm;
    ap_uint<buf_width> buff = 0;
#if !defined(__SYNTHESIS__) && XF_UTILS_HW_STRM_1NRR_DEBUG == 1
    std::cout << "lcm(" << _WInStrm << ", " << _WOutStrm << ")=" << buf_width << std::endl;
    std::cout << "_WInStrm =" << _WInStrm << ","
              << "num_in =" << num_in << std::endl;
    std::cout << "_WOutStrm =" << _WOutStrm << ","
              << "num_out =" << num_out << std::endl;
#endif

    bool last = e_istrm.read();
    int p = 0;
    while (!last) {
#pragma HLS pipeline II = 1
        ap_uint<_WInStrm> d = istrm.read();
        buff.range((p + 1) * _WInStrm - 1, p * _WInStrm) = d;
        last = e_istrm.read();
        if (p + 1 >= num_in) {
            // collect num_in input data and output them together
            buf_lcm_strm.write(buff);
            e_buf_lcm_strm.write(false);
        }
        p = (p + 1) >= num_in ? 0 : p + 1;
    }
    // output buff which has p useful data, that is to say, others in buff are
    // unuseful but output.
    buf_lcm_strm.write(buff);
    e_buf_lcm_strm.write(true);
    left_lcm.write(p);
}

/* @brief convert the input data with multiple _WOutStrm width to  output with
 * _WOutStrm width.
 *
 * @tparam _WInStrm input stream width.
 * @tparam _WOutStrm output stream width.
 * @tparam _NStrm number of output stream.
 *
 * @param buf_lcm_strm input data stream.
 * @param e_buf_lcm_strm end flag stream for input data.
 * @param left_lcm the number of useful data(_WInStrm) in last input data.
 * @param left_n the number of useful data(_WOutStrm) in last output.
 * @param buf_n_strm  output stream.
 * @param e_buf_n_strm  end flag stream.
 */
template <int _WInStrm, int _WOutStrm, int _NStrm>
void stream_one_to_n_reduce(hls::stream<ap_uint<LCM<_WInStrm, _WOutStrm * _NStrm>::value> >& buf_lcm_strm,
                            hls::stream<bool>& e_buf_lcm_strm,
                            hls::stream<int>& left_lcm,
                            hls::stream<int>& left_n,
                            hls::stream<ap_uint<_NStrm * _WOutStrm> >& buf_n_strm,
                            hls::stream<bool>& e_buf_n_strm) {
    const int buf_width = LCM<_WInStrm, _NStrm * _WOutStrm>::value;
    const int num_out = buf_width / _WOutStrm;
    const int count_out = num_out / _NStrm;
    ap_uint<buf_width> inner_buf = 0;
    while (!e_buf_lcm_strm.read()) {
// FIX
#pragma HLS pipeline II = 1
        inner_buf = buf_lcm_strm.read();
        for (int i = 0; i < count_out; ++i) {
            buf_n_strm.write(inner_buf.range((i + 1) * _NStrm * _WOutStrm - 1, i * _NStrm * _WOutStrm));
            e_buf_n_strm.write(false);
        }
    }
    // when end of input stream, pick up the useful data from the last input data
    int ln_in = left_lcm.read();
    inner_buf = buf_lcm_strm.read();
    int flg = 0;
    for (int i = 0; i < count_out; ++i) {
// FIX
#pragma HLS pipeline II = 1
        if ((i + 1) * _NStrm * _WOutStrm <= ln_in * _WInStrm) {
            buf_n_strm.write(inner_buf.range((i + 1) * _NStrm * _WOutStrm - 1, i * _NStrm * _WOutStrm));
            e_buf_n_strm.write(false);
            flg = i + 1;
        }
    }
    // output  the remaining data  when the above for-loop quits.
    buf_n_strm.write(inner_buf.range((flg + 1) * _NStrm * _WOutStrm - 1, flg * _NStrm * _WOutStrm));
    int tmp_ln_out = ln_in * _WInStrm / _WOutStrm - flg * _NStrm;
    int ln_out = tmp_ln_out < 0 ? 0 : tmp_ln_out;
    e_buf_n_strm.write(true);
    left_n.write(ln_out);
#if !defined(__SYNTHESIS__) && XF_UTILS_HW_STRM_1NRR_DEBUG == 1
    std::cout << "ln_in =  " << std::dec << ln_in << std::endl;
    std::cout << "ln_out =  " << std::dec << ln_out << std::endl;
#endif
}

/* @brief ditribution of the input data.
 *
 * @tparam _WInStrm input stream width.
 * @tparam _WOutStrm output stream width.
 * @tparam _NStrm number of output stream.
 *
 * @param buf_n_strm  output stream.
 * @param e_buf_n_strm  end flag stream.
 * @param left_n the number of useful data(_WOutStrm) in last input.
 * @param ostrms the output streams.
 * @param e_ostrms the end flags.
 */
template <int _WInStrm, int _WOutStrm, int _NStrm>
void stream_one_to_n_distribute(hls::stream<ap_uint<_NStrm * _WOutStrm> >& buf_n_strm,
                                hls::stream<bool>& e_buf_n_strm,
                                hls::stream<int>& left_n,
                                hls::stream<ap_uint<_WOutStrm> > ostrms[_NStrm],
                                hls::stream<bool> e_ostrms[_NStrm]) {
    const int buf_width = _NStrm * _WOutStrm;
    const int num_in = buf_width / _WInStrm;
    const int num_out = buf_width / _WOutStrm;
    const int count_out = num_out / _NStrm;
    // const int up_nstrm  = _NStrm;
    const int up_nstrm = UpBound<_NStrm>::value;
    const int deq_depth = 8;
    const int deq_width = 4;
    ap_uint<buf_width> buff_r = 0;
    ap_uint<buf_width> buff_p = 0;
    ap_uint<buf_width> buff_q = 0;
    ap_uint<_WOutStrm> buf_arr[2 * _NStrm];
#pragma HLS ARRAY_PARTITION variable = buf_arr complete
    ap_uint<_WOutStrm> deq[_NStrm][deq_depth];
#pragma HLS ARRAY_PARTITION variable = deq dim = 0
    ap_uint<deq_width> frt[_NStrm];
#pragma HLS ARRAY_PARTITION variable = frt complete
    ap_uint<deq_width> rr[_NStrm];
#pragma HLS ARRAY_PARTITION variable = rr complete
    int pos[_NStrm];
#pragma HLS ARRAY_PARTITION variable = pos complete

    ap_uint<_NStrm> full = 0;
    ap_uint<_NStrm> all_full = ~full;
    ap_uint<_NStrm> bak_full = 0;
    ap_uint<_NStrm> bak_full1 = 0;
    ap_uint<_NStrm> inv_bak_full = 0;
    ap_uint<_NStrm> last_full = 0;
    const int mult_nstrm = _NStrm * 2;
    int base = 0;
    int next_base = 0;
    int bs_n = 0;
    int p = 0;
    int rn = 0;
    int rn2 = 0;
    int ld = 0;
    int ld2 = 0;
    bool ld_flg = true;
    bool wb = true;
    bool high = true;
    int tc = 0;
    bool be = e_buf_n_strm.read();
    /**********************************************************************************
   *  iterator  1      2       3
   *  input    4321   8765    no input
   *  output  ( f=full, no output)
   *    A       1      3      5
   *    B       f      f      6
   *    C       f      f      7
   *    D       2      4      8
   *
   * buff_p(L) 4321  4321   4321
   * buff_q(H) 0000  8765   8765
   *   base    0       2      4
   *
   *  pos ( index=0, 1, ..., _NStrm-1)
   *    A      0       2      4
   *    B      0       2      5
   *    C      0       2      6
   *    D      1       3      7
   *  stream C is full, so don't care its pos at the first time
   *  So does D at the second time
   *
   **************************************************************************************/

    for (int i = 0; i < _NStrm; ++i) {
#pragma HLS unroll
        bak_full1[i] = ostrms[i].full();
        frt[i] = 0;
        rr[i] = 0;
    }

    bool new_buff_r = false;
LOOP_core:
    while (!be) {
#pragma HLS pipeline II = 1
        // output data from deq
        for (int i = 0; i < _NStrm; ++i) {
#pragma HLS unroll
            int f = frt[i];
            int r = rr[i];
            ap_uint<_WOutStrm> d = deq[i][f];
            bool fl = ostrms[i].full();
            full[i] = fl;
            if (f != r && fl == false) {
                ostrms[i].write(d);
                e_ostrms[i].write(false);
                int nf = (f + 1) == deq_depth ? 0 : (f + 1);
                frt[i] = nf;
            }
#if !defined(__SYNTHESIS__) && XF_UTILS_HW_STRM_1NRR_DEBUG == 1
            std::cout << "i =  " << std::dec << i << std::endl;
            std::cout << "d =  " << std::hex << d << std::endl;
#endif
        } // for

        // read state
        // bak_full= full;
        bak_full = bak_full1;
        inv_bak_full = ~bak_full1;
        bak_full1 = full;

        base = next_base;

        if (bak_full != all_full && ld_flg) {
            //  read new data when left data is not enough
            be = e_buf_n_strm.read();
            buff_r = buf_n_strm.read();
            high = !high;
            tc = _NStrm;
            new_buff_r = true;
        } else {
            tc = 0;
        }
        // ld is the number of input data, including data in buf_arr and buff_r
        // Updating `ld` before `if (ld < _NStrm)` leads to big latency because the
        // critical path includes read the input stream.
        ld = ld - rn2 + tc;
        // ld2 is the number of input data stored in buf_arr
        ld2 = ld2 - rn2;
        int tmp_ld = ld2 < _NStrm ? ld2 + mult_nstrm : ld2 + _NStrm;
        ld_flg = ld < mult_nstrm && ld <= tmp_ld;
        // get data when the data in buf_arr is not enough
        if (new_buff_r && ld2 < _NStrm) {
            new_buff_r = false;
            wb = true;
            ld2 += _NStrm;
            buff_q = high ? buff_r : buff_q;
            buff_p = high ? buff_p : buff_r;
        } else {
            wb = false;
        }
        // compute the index that deq[i] read the data in buf_arr
        pos[0] = base;
        for (int i = 1; i < _NStrm; ++i) {
#pragma HLS unroll
            ap_uint<up_nstrm> bf = inv_bak_full.range(i - 1, 0);
            // the number of not full streams befor the i-th streams ( i.e, among the
            // first i-1 streams).
            int nb = countOnes(bf);
            // no-blockings
            int mv = nb + base;
            pos[i] = (mv >= mult_nstrm) ? (mv - mult_nstrm) : mv;
#if !defined(__SYNTHESIS__) && XF_UTILS_HW_STRM_1NRR_DEBUG == 1
            std::cout << "bf =  " << std::hex << bf << std::endl;
            std::cout << "c =  " << std::dec << c << std::endl;
#endif
        }
        // double buffers service buf_arr in order of deq accessing data
        for (int i = 0; i < _NStrm; ++i) {
#pragma HLS unroll
            buf_arr[i] = buff_p.range((i + 1) * _WOutStrm - 1, i * _WOutStrm);
            buf_arr[i + _NStrm] = buff_q.range((i + 1) * _WOutStrm - 1, i * _WOutStrm);
        }
        last_full = bak_full;
        // the numbers of data which are written to deq from buf_arr in this
        // iteration
        rn2 = countOnes(inv_bak_full);
#if !defined(__SYNTHESIS__) && XF_UTILS_HW_STRM_1NRR_DEBUG == 1
        std::cout << "last_full =  " << std::hex << last_full << std::endl;
        std::cout << "full =  " << std::hex << full << std::endl;
        std::cout << "rn2 =  " << std::dec << rn2 << std::endl;
#endif
        // deq should read data from the position in buf_arr in next time
        int temp = bs_n + rn2;
        int temp2 = base + rn2;
        next_base = temp2 >= mult_nstrm ? temp : temp2;
        bs_n = next_base - mult_nstrm;
        // write to deq
        for (int i = 0; i < _NStrm; ++i) {
#pragma HLS unroll
            int ps = pos[i];
            if (!last_full[i]) {
                ap_uint<_WOutStrm> d = buf_arr[ps];
                int r = rr[i];
                deq[i][r] = d;
                int nr = (r + 1) == deq_depth ? 0 : (r + 1);
                rr[i] = nr;
            }
        } // for
    }     // while

    ////////////////////////////////////////////////////
    /*
     * the data are moved from input to output:
     *  input stream --> buff_r--> buf_arr --> deq --> output streams
     * so it has to output the remaining data in buffers when the above while-loop
     * quits.
     */

    /* sequentially  output the remaining data in deq
     * here, the next for-lopp will be unrolled to _NStrm while-loops, which will
     * be run in sequence instead of in parallel.
     */
    int sid = 0;
    for (int i = 0; i < _NStrm; ++i) {
#pragma HLS unroll
        int r = rr[i];
        int f = frt[i];
        while (f != r) {
#pragma HLS pipeline II = 1
            ap_uint<_WOutStrm> d = deq[i][f];
            if (false == ostrms[sid].full()) {
                ostrms[sid].write(d);
                e_ostrms[sid].write(false);
                f = (f + 1) == deq_depth ? 0 : (f + 1);
                sid = (sid + 1) == _NStrm ? 0 : (sid + 1);
            }
        } // while (f!=r)
    }     // for

    // sequentially  output the remaining data in buf_arr
    ld2 -= rn2;
    base = next_base;
    int pb = base > 0 ? base : 0;
    while (ld2 > 0) {
#pragma HLS pipeline II = 1
        if (false == ostrms[sid].full()) {
            ap_uint<_WOutStrm> d = buf_arr[pb];
            pb = (pb + 1) == mult_nstrm ? 0 : pb + 1;
            ostrms[sid].write(d);
            e_ostrms[sid].write(false);
            ld2--;
        }
        sid = (sid + 1 == _NStrm) ? 0 : sid + 1;
    }

    // sequentially  output the remaining data in buff_r
    if (wb == false) {
        int ib = 0;
        while (ib < _NStrm) {
#pragma HLS pipeline II = 1
            if (false == ostrms[sid].full()) {
                ostrms[sid].write(buff_r.range((ib + 1) * _WOutStrm - 1, ib * _WOutStrm));
                e_ostrms[sid].write(false);
                ib++;
            }
            sid = (sid + 1 == _NStrm) ? 0 : sid + 1;
        }
    }

    // sequentially  output the last data from input stream
    int left = left_n.read();
    buff_r = buf_n_strm.read();
    int ib2 = 0;
    while (ib2 < left) {
#pragma HLS pipeline II = 1
        if (false == ostrms[sid].full()) {
            ostrms[sid].write(buff_r.range((ib2 + 1) * _WOutStrm - 1, ib2 * _WOutStrm));
            e_ostrms[sid].write(false);
            ib2++;
        }
        sid = (sid + 1 == _NStrm) ? 0 : sid + 1;
    }

    // end flags for output streams
    for (int i = 0; i < _NStrm; ++i) {
#pragma HLS unroll
        e_ostrms[i].write(true);
    }
}

/* @brief stream distribute, using load-balancing algorithm.
 *
 * The input stream is assumed to be compact data to be splitted into
 * _WOutStrm wide data into output streams.
 *
 * @tparam _WInStrm input stream width.
 * @tparam _WOutStrm output stream width.
 * @tparam _NStrm number of output stream.
 *
 * @param istrm input data stream.
 * @param e_istrm end flag stream for input data.
 * @param ostrms output data streams.
 * @param e_ostrms end flag streams, one for each output data stream.
 */
template <int _WInStrm, int _WOutStrm, int _NStrm>
void stream_one_to_n_load_balance(hls::stream<ap_uint<_WInStrm> >& istrm,
                                  hls::stream<bool>& e_istrm,
                                  hls::stream<ap_uint<_WOutStrm> > ostrms[_NStrm],
                                  hls::stream<bool> e_ostrms[_NStrm]) {
#pragma HLS dataflow
    // least common multiple of _WInStrm and _WOutStrm as the width of inner
    // buffer
    hls::stream<ap_uint<LCM<_WInStrm, _NStrm * _WOutStrm>::value> > buf_lcm_strm;
#pragma HLS stream variable = buf_lcm_strm depth = 32
    hls::stream<int> left_lcm;
    hls::stream<int> left_n;
    hls::stream<bool> e_buf_lcm_strm;
#pragma HLS stream variable = e_buf_lcm_strm depth = 32
    hls::stream<ap_uint<_NStrm * _WOutStrm> > buf_n_strm;
#pragma HLS stream variable = buf_n_strm depth = 32
    hls::stream<bool> e_buf_n_strm;
#pragma HLS stream variable = e_buf_n_strm depth = 32
    /* there are 3 steps:
     *    collect input data     |         solve different widths  |  dipatch to
     *_NStrm streams
     *connected streams and their widths:
     *   istrms  --->       buf_lcm_strm    ---->             buf_n_strm
     *---->  ostrms
     * _WInStrm       lcm(_WInStrm, _WOutStrm * _NStrm)   _WOutStrm * _NStrm
     *_WOutStrm
    *
    */
    stream_one_to_n_read<_WInStrm, _WOutStrm, _NStrm>(istrm, e_istrm, buf_lcm_strm, left_lcm, e_buf_lcm_strm);

    stream_one_to_n_reduce<_WInStrm, _WOutStrm, _NStrm>(buf_lcm_strm, e_buf_lcm_strm, left_lcm, left_n, buf_n_strm,
                                                        e_buf_n_strm);

    stream_one_to_n_distribute<_WInStrm, _WOutStrm, _NStrm>(buf_n_strm, e_buf_n_strm, left_n, ostrms, e_ostrms);
}

/////////////////////////////////////////////////////////////////////////
} // details

/* @brief stream distribute, using load-balancing algorithm.
 *
 * The input stream is assumed to be compact data to be splitted into
 * _WOutStrm wide data into output streams.
 *
 * @tparam _WInStrm input stream width.
 * @tparam _WOutStrm output stream width.
 * @tparam _NStrm number of output stream.
 *
 * @param istrm input data stream.
 * @param e_istrm end flag stream for input data.
 * @param ostrms output data streams.
 * @param e_ostrms end flag streams, one for each output data stream.
 * @param alg algorithm selector.
 */
template <int _WInStrm, int _WOutStrm, int _NStrm>
void streamOneToN(hls::stream<ap_uint<_WInStrm> >& istrm,
                  hls::stream<bool>& e_istrm,
                  hls::stream<ap_uint<_WOutStrm> > ostrms[_NStrm],
                  hls::stream<bool> e_ostrms[_NStrm],
                  LoadBalanceT alg) {
    details::stream_one_to_n_load_balance<_WInStrm, _WOutStrm, _NStrm>(istrm, e_istrm, ostrms, e_ostrms);
}

// support  _TIn
namespace details {

/* @brief stream distribute, using load-balancing algorithm.
 *
 * @tparam _TIn the type of input stream.
 * @tparam _NStrm number of output stream.
 *
 * @param istrm input data stream.
 * @param e_istrm end flag stream for input data.
 * @param ostrms output data streams.
 * @param e_ostrms end flag streams, one for each output data stream.
 */
template <typename _TIn, int _NStrm>
void stream_one_to_n_load_balance_type(hls::stream<_TIn>& istrm,
                                       hls::stream<bool>& e_istrm,
                                       hls::stream<_TIn> ostrms[_NStrm],
                                       hls::stream<bool> e_ostrms[_NStrm]) {
    int id = 0;
    bool last = e_istrm.read();
    while (!last) {
#pragma HLS pipeline II = 1
        if (false == ostrms[id].full()) {
            // read data and ouput them  when ouput is available
            _TIn data = istrm.read();
            ostrms[id].write(data);
            e_ostrms[id].write(false);
            last = e_istrm.read();
        }
        id = (id + 1 == _NStrm) ? 0 : (id + 1);
    }
    for (int i = 0; i < _NStrm; ++i) {
#pragma HLS unroll
        e_ostrms[i].write(true);
    }
}

} // details

/* @brief stream distribute, using load-balancing algorithm.
 *
 * @tparam _TIn the type of input stream.
 * @tparam _NStrm number of output stream.
 *
 * @param istrm input data stream.
 * @param e_istrm end flag stream for input data.
 * @param ostrms output data streams.
 * @param e_ostrms end flag streams, one for each output data stream.
 * @param alg algorithm selector.
 */
template <typename _TIn, int _NStrm>
void streamOneToN(hls::stream<_TIn>& istrm,
                  hls::stream<bool>& e_istrm,
                  hls::stream<_TIn> ostrms[_NStrm],
                  hls::stream<bool> e_ostrms[_NStrm],
                  LoadBalanceT alg) {
    details::stream_one_to_n_load_balance_type<_TIn, _NStrm>(istrm, e_istrm, ostrms, e_ostrms);
}

} // utils_hw
} // common
} // xf
///////////////////////////////////////////////////

#endif // XF_UTILS_HW_STREAM_1N_LB_H
