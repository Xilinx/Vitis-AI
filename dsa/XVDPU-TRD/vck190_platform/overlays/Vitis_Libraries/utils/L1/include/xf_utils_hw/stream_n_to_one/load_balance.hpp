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
#ifndef XF_UTILS_HW_STREAM_N1_LB
#define XF_UTILS_HW_STREAM_N1_LB

/**
 * @file stream_n_to_one/load_balance.hpp
 * @brief header for collect in round-roubin order.
 */

#include "xf_utils_hw/enums.hpp"
#include "xf_utils_hw/types.hpp"
#include "xf_utils_hw/common.hpp"

// Forward decl

namespace xf {
namespace common {
namespace utils_hw {

/**
 * @brief stream distribute, skip to read the empty input streams.
 *
 * @tparam _WInStrm input stream width.
 * @tparam _WOutStrm output stream width.
 * @tparam _NStrm number of input streams.
 *
 * @param istrms input data streams.
 * @param e_istrms end flag streams for input data.
 * @param ostrm output data stream.
 * @param e_ostrm end flag stream.
 * @param alg algorithm selector.
 */
template <int _WInStrm, int _WOutStrm, int _NStrm>
void streamNToOne(hls::stream<ap_uint<_WInStrm> > istrms[_NStrm],
                  hls::stream<bool> e_istrms[_NStrm],
                  hls::stream<ap_uint<_WOutStrm> >& ostrm,
                  hls::stream<bool>& e_ostrm,
                  LoadBalanceT alg);

/**
 * @brief stream distribute, skip to read the empty input streams.
 *
 * @tparam _TIn the type of stream.
 * @tparam _NStrm number of input streams.
 *
 * @param istrms input data streams.
 * @param e_istrms end flag streams for input data.
 * @param ostrm output data stream.
 * @param e_ostrm end flag stream.
 * @param alg algorithm selector.
 */
template <typename _TIn, int _NStrm>
void streamNToOne(hls::stream<_TIn> istrms[_NStrm],
                  hls::stream<bool> e_istrms[_NStrm],
                  hls::stream<_TIn>& ostrm,
                  hls::stream<bool>& e_ostrm,
                  LoadBalanceT alg);

} // utils_hw
} // common
} // xf

namespace xf {
namespace common {
namespace utils_hw {

namespace details {

/**
 * @brief read the data from _NStrm streams, skip the empty streams
 * collect  _NStrm data(_WInStrm bits) from input streams and output
 * one(_NStrm*_NStrm bits)
 * @tparam _WInStrm input stream width.
 * @tparam _NStrm number of input streams.
 *
 * @param istrms input data streams.
 * @param e_istrms end flag streams for input data.
 * @param left_n the number of available data in last buf_lcm_strm.
 * @param buf_lcm_strm output stream.
 * @param e_buf_lcm_strm end flag stream.
 */
template <int _WInStrm, int _NStrm>
void stream_n_to_one_read_lb(hls::stream<ap_uint<_WInStrm> > istrms[_NStrm],
                             hls::stream<bool> e_istrms[_NStrm],
                             hls::stream<ap_uint<32> >& left_n,
                             hls::stream<ap_uint<_WInStrm * _NStrm> >& buf_lcm_strm,
                             hls::stream<bool>& e_buf_lcm_strm) {
    const int buf_width = _WInStrm * _NStrm;
    const int num_in = _NStrm;
    const int up_nstrm = UpBound<_NStrm>::value;
    ap_uint<2 * buf_width> buff_a = 0;
    ap_uint<buf_width> buff_b = 0;
    ap_uint<_NStrm> last = 0;
    ap_uint<_NStrm> bak_last = 0;
    const ap_uint<_NStrm> ends = ~last;
    ap_uint<_NStrm> val = 0;
    ap_uint<_NStrm> bak_val = 0;
    int base = 0;
    ap_uint<_NStrm * _WInStrm> tmpb[_NStrm];
#pragma HLS ARRAY_PARTITION variable = tmpb complete
    ap_uint<_WInStrm> ttm[_NStrm];
#pragma HLS ARRAY_PARTITION variable = ttm complete

#if !defined(__SYNTHESIS__) && XF_UTIL_STRM_1NRR_DEBUG == 1
    std::cout << std::dec << std::endl;
    std::cout << "_NStrm =" << _NStrm << std::endl;
    std::cout << "_WInStrm =" << _WInStrm << std::endl;
    std::cout << "Win*_NStrm =" << buf_width << std::endl;
    std::cout << "num_in =" << num_in << std::endl;
    std::cout << "UpBound<_NStrm> =" << up_nstrm << std::endl;
#endif
    /*
   * assume _NStrm=4, WInStm=4bits
   * streams  ( - empty )  (hex)
   * strms[0]    -    3    end
   * strms[1]    -    4    end
   * strms[2]    1    -    6      8     end
   * strms[3]    2    5    7      9     end
   *
   * val
   *  val[0]     0    1    0      0     0
   *  val[1]     0    1    0      0     0
   *  val[2]     1    0    1      1     0
   *  val[3]     1    1    1      1     0
   *
   * tmpb(hex)
   * tmpb[0]   0000  0003  0000  0000
   * tmpb[1]   0000  0040  0000  0000
   * tmpb[2]   0001  0000  0006  0008
   * tmpb[3]   0020  0500  0070  0090
   *
   * buff_b    0021  0543  0076  0098
   * un          2     3     2      2
   *
   * buff_a      21 54321   765  98765
   * output       -  4321   -     8765
   * buff_a      21     5   765      9
   */

    for (int i = 0; i < _NStrm; ++i) {
#pragma HLS unroll
        last[i] = e_istrms[i].empty() ? false : e_istrms[i].read();
    } // for
    while (last != ends) {
#pragma HLS pipeline II = 1
        bak_last = last;
        for (int i = 0; i < _NStrm; ++i) {
#pragma HLS unroll
            bool et = istrms[i].empty();
            bool vl = !et && !bak_last[i];
            val[i] = vl; // flag of available data
            // neither  empty nor finished stream, read it, or default zero
            ttm[i] = vl ? istrms[i].read() : (ap_uint<_WInStrm>)0;
            // empty or end stream, keep the last flag; or read it
            last[i] = (bak_last[i] || e_istrms[i].empty()) ? bak_last[i] : e_istrms[i].read();

#if !defined(__SYNTHESIS__) && XF_UTIL_STRM_1NRR_DEBUG == 1
            std::cout << std::hex << "et= " << et << std::endl;
            std::cout << std::dec << "i= " << i << " ttm[i]= " << ttm[i] << std::endl;
            std::cout << std::dec << "i= " << i << " val[i]= " << val[i] << std::endl;
            std::cout << std::dec << "i= " << i << " last[i]= " << last[i] << std::endl;
#endif
        } // for
        // move the available data to "the right position"
        // if a stream is empty or finished, there is no point in moving its data
        // since its data is 0
        tmpb[0] = ttm[0];
        for (int i = 1; i < _NStrm; ++i) {
#pragma HLS unroll
            ap_uint<up_nstrm> v = val.range(i - 1, 0);
            int ones = countOnes(v); // it's similar to round robin  if ones always is i.
            int p = ones;            // index of tmpb[i].range(), where  istrm[i] is stored if it
                                     // is not empty
            ap_uint<_NStrm* _WInStrm> d = ttm[i];
            tmpb[i] = d << (p * _WInStrm);
#if !defined(__SYNTHESIS__) && XF_UTIL_STRM_1NRR_DEBUG == 1
            std::cout << std::hex << "v= " << v << " d= " << d << " tmpb[i]=" << tmpb[i] << std::endl;
            std::cout << std::dec << "i= " << i << " p= " << p << " p*_WInStrm=" << p * _WInStrm << std::endl;
#endif
        } // for
        buff_b = 0;
        for (int i = 0; i < _NStrm; ++i) {
#pragma HLS pipeline II = 1
            // merge data,
            buff_b |= tmpb[i];
        }
        int un = countOnes(val); // how many new data are collected to buffer at this time

        // accumulate data
        if (un > 0) {
            buff_a.range((base + un) * _WInStrm - 1, base * _WInStrm) = buff_b.range(un * _WInStrm - 1, 0);
        }
        // output one data
        if (base + un >= num_in) {
            //  the size of buff_a is big enough
            base = base + un - num_in;
            buf_lcm_strm.write(buff_a.range(buf_width - 1, 0));
            e_buf_lcm_strm.write(false);
            // move the remaining data to the start position
            buff_a = buff_a >> buf_width;
        } else {
            // accumulated data
            base += un;
        } // if-else

    } // while
    // output the last data(_NStrm*_WInStrm), which have base available
    // data(_WInStrm bits)
    buf_lcm_strm.write(buff_a.range(buf_width - 1, 0));
    e_buf_lcm_strm.write(true);
    left_n.write(base);
}

/**
 * @brief buffer data to solve different input and output width.
 *
 * input  _WInStrm * _NStrm bit --> output lcm(_WInStrm*_NStrm, WOutStrm) bits
 * @tparam _WInStrm input stream width.
 * @tparam _WOutStrm output stream width.
 * @tparam _NStrm number of input streams.
 *
 * @param buf_n_strm input data stream.
 * @param e_buf_n_strm end flag stream for input data.
 * @param left_n the number of available data in last buf_n_strm, input port
 * @param left_lcm the number of available data in last buf_lcm_strm, output
 * port.
 * @param buf_lcm_strm output stream.
 * @param e_buf_lcm_strm end flag stream.
 */
template <int _WInStrm, int _WOutStrm, int _NStrm>
void stream_n_to_one_collect_lb(hls::stream<ap_uint<_WInStrm * _NStrm> >& buf_n_strm,
                                hls::stream<bool>& e_buf_n_strm,
                                hls::stream<ap_uint<32> >& left_n,
                                hls::stream<ap_uint<32> >& left_lcm,
                                hls::stream<ap_uint<LCM<_WInStrm * _NStrm, _WOutStrm>::value> >& buf_lcm_strm,
                                hls::stream<bool>& e_buf_lcm_strm) {
    const int buf_width = LCM<_WInStrm * _NStrm, _WOutStrm>::value;
    const int num_in = buf_width / _WInStrm;
    const int count_in = num_in / _NStrm;
    ap_uint<buf_width> inner_buf;
    int p = 0;
    int pos = 0;
    bool last = e_buf_n_strm.read();
    while (!last) {
#pragma HLS pipeline II = 1
        int low = pos;
        pos += _NStrm * _WInStrm;
        inner_buf.range(pos - 1, low) = buf_n_strm.read();
        last = e_buf_n_strm.read();
        if (p + 1 == count_in) {
            e_buf_lcm_strm.write(false);
            buf_lcm_strm.write(inner_buf);
            p = 0;
            pos = 0;
        } else
            p++;
    } // for

    int end_pos = pos + _NStrm * _WInStrm;
    inner_buf.range(end_pos - 1, pos) = buf_n_strm.read();

    buf_lcm_strm.write(inner_buf);
    e_buf_lcm_strm.write(true); // even if true, the data in buf_strm is available.

    int left = left_n.read();
    left_lcm.write(p * _NStrm + left);
#if !defined(__SYNTHESIS__) && XF_UTIL_STRM_1NRR_DEBUG == 1
    std::cout << "p=" << p << "  "
              << "left=" << left << std::endl;
    std::cout << "pos=" << pos << "  "
              << "end_pos=" << end_pos << std::endl;
#endif
}

/**
 * @brief output data sequentially from input stream with big width.
 *
 * output buf_width=lcm(_WInStrm*_NStrm, WOutStrm) bits in buf_width/_WOutStrm
 * cycles
 *
 * @tparam _WInStrm input stream width.
 * @tparam _WOutStrm output stream width.
 * @tparam _NStrm number of input streams.
 *
 * @param buf_lcm_strm input data stream.
 * @param e_buf_lcm_strm end flag stream for input data.
 * @param left_lcm the number of available data in last buf_lcm_strm, input
 * port.
 * @param ostrm output stream.
 * @param e_ostrm end flag stream.
 */
template <int _WInStrm, int _WOutStrm, int _NStrm>
void stream_n_to_one_distribute_lb(hls::stream<ap_uint<LCM<_WInStrm * _NStrm, _WOutStrm>::value> >& buf_lcm_strm,
                                   hls::stream<bool>& e_buf_lcm_strm,
                                   hls::stream<ap_uint<32> >& left_lcm,
                                   hls::stream<ap_uint<_WOutStrm> >& ostrm,
                                   hls::stream<bool>& e_ostrm) {
    const int buf_width = LCM<_WInStrm * _NStrm, _WOutStrm>::value;
    const int num_out = buf_width / _WOutStrm;

    ap_uint<buf_width> inner_buf;
    int low = 0;
    int up = 0;
    int c = num_out;
    unsigned int up_pos = -1;
    bool last = false;
    while (!last) {
#pragma HLS pipeline II = 1
        if (c == num_out) {
            inner_buf = buf_lcm_strm.read();
            c = 0;
            last = e_buf_lcm_strm.read();
            if (last) up_pos = left_lcm.read() * _WInStrm;
        } // if
        if ((c + 1) * _WOutStrm <= up_pos) {
            ostrm.write(inner_buf.range((c + 1) * _WOutStrm - 1, c * _WOutStrm));
            e_ostrm.write(false);
        } // if
        c++;
    } // for
    e_ostrm.write(true);
}

/**
 * @brief stream distribute, skip to read the empty input streams.
 *
 * @tparam _WInStrm input stream width.
 * @tparam _WOutStrm output stream width.
 * @tparam _NStrm number of input streams.
 *
 * @param istrms input data streams.
 * @param e_istrms end flag streams for input data.
 * @param ostrm output data stream.
 * @param e_ostrm end flag stream.
 */
template <int _WInStrm, int _WOutStrm, int _NStrm>
void stream_n_to_one_load_balance(hls::stream<ap_uint<_WInStrm> > istrms[_NStrm],
                                  hls::stream<bool> e_istrms[_NStrm],
                                  hls::stream<ap_uint<_WOutStrm> >& ostrm,
                                  hls::stream<bool>& e_ostrm) {
    const int buf_width = LCM<_WInStrm * _NStrm, _WOutStrm>::value;
    hls::stream<ap_uint<_WInStrm * _NStrm> > buf_n_strm;
#pragma HLS stream variable = buf_n_strm depth = 8
    hls::stream<bool> e_buf_n_strm;
#pragma HLS stream variable = e_buf_n_strm depth = 8
    hls::stream<ap_uint<32> > left_n; // how many input(_WInStrm bits) are stored
                                      // in  last data(_NStrm*WInstrm bits) in
                                      // buf_n_strm
    hls::stream<ap_uint<32> > left_lcm;
    hls::stream<ap_uint<buf_width> > buf_lcm_strm;
#pragma HLS stream variable = buf_lcm_strm depth = 8

    hls::stream<bool> e_buf_lcm_strm;
#pragma HLS stream variable = e_buf_lcm_strm depth = 8

#if !defined(__SYNTHESIS__) && XF_UTIL_STRM_1NRR_DEBUG == 1
    std::cout << "start n ---->1  " << std::endl;
#endif
#pragma HLS dataflow

    /*  read data   : read data from input streams, output _NStrm * _WInStrm bits
     *  collect data: buffer  buf_width=lcm(_WInStrm*_NStrm, _WOutStrm) bits and
     * output them.
     *  distribute  : output buf_width/_WOutStrm  data when read buf_width bits
     * once
     *
     * least common mutiple(lcm) is used for solving the difference between  the
     * input width and output width
     *
     * */
    stream_n_to_one_read_lb<_WInStrm, _NStrm>(istrms, e_istrms, left_n, buf_n_strm, e_buf_n_strm);

    stream_n_to_one_collect_lb<_WInStrm, _WOutStrm, _NStrm>(buf_n_strm, e_buf_n_strm, left_n, left_lcm, buf_lcm_strm,
                                                            e_buf_lcm_strm);

    stream_n_to_one_distribute_lb<_WInStrm, _WOutStrm, _NStrm>(buf_lcm_strm, e_buf_lcm_strm, left_lcm, ostrm, e_ostrm);
}

} // details

/**
 * @brief stream distribute, skip to read the empty input streams.
 *
 * @tparam _WInStrm input stream width.
 * @tparam _WOutStrm output stream width.
 * @tparam _NStrm number of input streams.
 *
 * @param istrms input data streams.
 * @param e_istrms end flag streams for input data.
 * @param ostrm output data stream.
 * @param e_ostrm end flag stream.
 * @param alg algorithm selector.
 */
template <int _WInStrm, int _WOutStrm, int _NStrm>
void streamNToOne(hls::stream<ap_uint<_WInStrm> > istrms[_NStrm],
                  hls::stream<bool> e_istrms[_NStrm],
                  hls::stream<ap_uint<_WOutStrm> >& ostrm,
                  hls::stream<bool>& e_ostrm,
                  LoadBalanceT alg) {
    details::stream_n_to_one_load_balance<_WInStrm, _WOutStrm, _NStrm>(istrms, e_istrms, ostrm, e_ostrm);
}

// support _TIn
namespace details {

/**
 * @brief stream distribute, skip to read the empty input streams.
 *
 * @tparam _TIn the type of streams.
 * @tparam _NStrm number of input streams.
 *
 * @param istrms input data streams.
 * @param e_istrms end flag streams for input data.
 * @param ostrm output data stream.
 * @param e_ostrm end flag stream.
 */
template <typename _TIn, int _NStrm>
void stream_n_to_one_load_balance_type(hls::stream<_TIn> istrms[_NStrm],
                                       hls::stream<bool> e_istrms[_NStrm],
                                       hls::stream<_TIn>& ostrm,
                                       hls::stream<bool>& e_ostrm) {
    ap_uint<_NStrm> last = 0;
    ap_uint<_NStrm> end = ~last;

    for (int i = 0; i < _NStrm; ++i) {
#pragma HLS unroll
        //    last[i] = e_istrms[i].empty() ? false: e_istrms[i].read();
        last[i] = e_istrms[i].read();
    }

    int id = 0;
    while (last != end) {
#pragma HLS pipeline II = 1
        bool lst = last[id];
        bool em = istrms[id].empty();
        // read data if not end and not empty
        if (!lst && !em) {
            _TIn data = istrms[id].read();
            ostrm.write(data);
            e_ostrm.write(false);
        }
        bool ee = e_istrms[id].empty();
        // keep its flag if a stream is finished or empty
        last[id] = (lst || ee) ? lst : e_istrms[id].read();
        id = (id + 1) == _NStrm ? 0 : (id + 1);
    } // while
    e_ostrm.write(true);
}

} // details

/**
 * @brief stream distribute, skip to read the empty input streams.
 *
 * @tparam _TIn the type of stream.
 * @tparam _NStrm number of input streams.
 *
 * @param istrms input data streams.
 * @param e_istrms end flag streams for input data.
 * @param ostrm output data stream.
 * @param e_ostrm end flag stream.
 * @param alg algorithm selector.
 */
template <typename _TIn, int _NStrm>
void streamNToOne(hls::stream<_TIn> istrms[_NStrm],
                  hls::stream<bool> e_istrms[_NStrm],
                  hls::stream<_TIn>& ostrm,
                  hls::stream<bool>& e_ostrm,
                  LoadBalanceT alg) {
    details::stream_n_to_one_load_balance_type<_TIn, _NStrm>(istrms, e_istrms, ostrm, e_ostrm);
}
} // utils_hw
} // common
} // xf
#endif // XF_UTILS_HW_STREAM_N1_LB
