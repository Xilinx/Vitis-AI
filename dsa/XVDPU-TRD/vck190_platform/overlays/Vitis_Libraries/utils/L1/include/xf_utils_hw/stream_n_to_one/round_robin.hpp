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
#ifndef XF_UTILS_HW_STREAM_N1_RR
#define XF_UTILS_HW_STREAM_N1_RR

/**
 * @file stream_n_to_one/round_robin.hpp
 * @brief header for collect in round-roubin order.
 */

#include "xf_utils_hw/enums.hpp"
#include "xf_utils_hw/types.hpp"
#include "xf_utils_hw/common.hpp"

// Forward decl ===============================================================

namespace xf {
namespace common {
namespace utils_hw {
/**
 * @brief stream distribute, in round-robin order from NStrm input streams.
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
                  RoundRobinT alg);

/**
 * @brief stream distribute, in round-robin order from NStrm input streams.
 *
 * @tparam _TIn the type of input and output stream.
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
                  RoundRobinT alg);

} // utils_hw
} // common
} // xf

// Implementation =============================================================

namespace xf {
namespace common {
namespace utils_hw {

namespace details {

/* @brief read data in round-robin order from NStrm input streams.
 *
 * This function collects multiple narrow streams into one same-wider or wider
 * stream.
 *
 * @tparam _WInStrm input stream width.
 * @tparam _NStrm number of input streams.
 *
 * @param istrms input data streams.
 * @param e_istrms end flag streams for input data.
 * @param lef_n the number of available data in last buf_n_strm.
 * @param buf_n_strm output stream.
 * @param e_buf_n_strm end flag stream.
 */
template <int _WInStrm, int _NStrm>
void stream_n_to_one_read(hls::stream<ap_uint<_WInStrm> > istrms[_NStrm],
                          hls::stream<bool> e_istrms[_NStrm],
                          hls::stream<ap_uint<32> >& left_n,
                          hls::stream<ap_uint<_WInStrm * _NStrm> >& buf_n_strm,
                          hls::stream<bool>& e_buf_n_strm) {
    ap_uint<_NStrm> ends = 0;
    ap_uint<_WInStrm* _NStrm> cmb = 0;
    for (int id = 0; id < _NStrm; ++id) {
#pragma HLS unroll
        ends[id] = e_istrms[id].read();
    }
    // read _NStrm data from input streams at the  same time, then output them
    while (ends == 0) {
#pragma HLS pipeline II = 1
        for (int id = 0; id < _NStrm; ++id) {
#pragma HLS unroll
            ap_uint<_WInStrm> d = istrms[id].read();
            cmb.range((id + 1) * _WInStrm - 1, id * _WInStrm) = d;
            ends[id] = e_istrms[id].read();
        }
#if !defined(__SYNTHESIS__) && XF_UTILS_HW_STRM_1NRR_DEBUG == 1
        std::cout << "comb=" << cmb << std::endl;
#endif
        buf_n_strm.write(cmb);
        e_buf_n_strm.write(false);
    }
    // read data from unfinished streams
    int left = 0;
    for (int id = 0; id < _NStrm; ++id) {
#pragma HLS pipeline II = 1
        if (!ends[id]) {
            ap_uint<_WInStrm> d = istrms[id].read();
            cmb.range((left + 1) * _WInStrm - 1, left * _WInStrm) = d;
            ends[id] = e_istrms[id].read();
            left++;
        }
    }
    // output the data from last loop
    buf_n_strm.write(cmb);
    e_buf_n_strm.write(true);
    // how many data are available in the last loop
    left_n.write(left);
#if !defined(__SYNTHESIS__) && XF_UTILS_HW_STRM_1NRR_DEBUG == 1
    std::cout << "comb=" << cmb << std::endl;
#endif
}
/* @brief buffer data to solve different input and output width.
 *
 * input  _WInStrm * _NStrm bit --> output lcm(_WInStrm*_NStrm, WOutStrm) bits
 *
 * @tparam _WInStrm input stream width.
 * @tparam _WOutStrm output stream width.
 * @tparam _NStrm number of input streams.
 *
 * @param buf_n_strm input data stream.
 * @param e_buf_n_strm end flag stream for input data.
 * @param lef_n the number of available data in last buf_n_strm, input port
 * @param lef_b the number of available data in last buf_lcm_strm, output port.
 * @param buf_lcm_strm output stream.
 * @param e_buf_lcm_strm end flag stream.
 */
template <int _WInStrm, int _WOutStrm, int _NStrm>
void stream_n_to_one_collect(hls::stream<ap_uint<_WInStrm * _NStrm> >& buf_n_strm,
                             hls::stream<bool>& e_buf_n_strm,
                             hls::stream<ap_uint<32> >& left_n,
                             hls::stream<ap_uint<32> >& left_lcm,
                             hls::stream<ap_uint<LCM<_WInStrm * _NStrm, _WOutStrm>::value> >& buf_lcm_strm,
                             hls::stream<bool>& e_buf_lcm_strm) {
    const int buf_size = LCM<_WInStrm * _NStrm, _WOutStrm>::value;
    const int num_in = buf_size / _WInStrm;
    const int count_in = num_in / _NStrm;
    ap_uint<buf_size> buf_a;
    int p = 0;
    int pos = 0;
    bool last = false;
    while (!last) {
#pragma HLS pipeline II = 1
        int low = pos;
        // pos stands for total available bits in buf_a except last =true
        pos += _NStrm * _WInStrm;
        buf_a.range(pos - 1, low) = buf_n_strm.read();
        last = e_buf_n_strm.read();
        // output when the buf_a is full
        if (!last && p + 1 == count_in) {
            e_buf_lcm_strm.write(false);
            buf_lcm_strm.write(buf_a);
            p = 0;
            pos = 0;
#if !defined(__SYNTHESIS__) && XF_UTILS_HW_STRM_1NRR_DEBUG == 1
            std::cout << "buf_a=" << buf_a << std::endl;
#endif
        } else
            p++;
    }
    // here, there are some useless data in buf_a, and left_lcm is the number of
    // useful data
    buf_lcm_strm.write(buf_a);
    e_buf_lcm_strm.write(true); // even if true, the data in buf_lcm_strm is available.
    int left = left_n.read();
    // 0<= left < _Nstrm
    left_lcm.write(pos - (_NStrm - left) * _WInStrm);
}

/* @brief output data sequentially from input stream with big width.
 *
 * output buf_size=lcm(_WInStrm*_NStrm, WOutStrm) bits in buf_size/_WOutStrm
 * cycles
 * @tparam _WInStrm input stream width.
 * @tparam _WOutStrm output stream width.
 * @tparam _NStrm number of input streams.
 *
 * @param buf_lcm_strm input data stream.
 * @param e_buf_lcm_strm end flag stream for input data.
 * @param left_lcm the number of available data in last buf_lcm_strm, input
 * port.
 * @param ostrm output stream.
 * @param estrm end flag stream.
 */
template <int _WInStrm, int _WOutStrm, int _NStrm>
void stream_n_to_one_distribute(hls::stream<ap_uint<LCM<_WInStrm * _NStrm, _WOutStrm>::value> >& buf_lcm_strm,
                                hls::stream<bool>& e_buf_lcm_strm,
                                hls::stream<ap_uint<32> >& left_lcm,
                                hls::stream<ap_uint<_WOutStrm> >& ostrm,
                                hls::stream<bool>& e_ostrm) {
    const int buf_size = LCM<_WInStrm * _NStrm, _WOutStrm>::value;
    const int num_out = buf_size / _WOutStrm;

    ap_uint<buf_size> buf_b;
    int low = 0;
    int up = 0;
    int c = num_out;
    bool last = false;
    unsigned int up_pos = -1;
    // assume the lengths of buf_lcm_strm and e_buf_lcm_strm are same and >=1
    while (!last) {
#pragma HLS pipeline II = 1
        // read once, output num_out data
        // the lengths of buf_b and e_buf_b are same and >=1
        if (c == num_out) {
            c = 0;
            buf_b = buf_lcm_strm.read();
            last = e_buf_lcm_strm.read();
            // when  e_buf_lcm_strm is true, read left_lcm
            if (last) up_pos = left_lcm.read(); // up_pos is changed only if the input will end
        }
        // ouput data at every cycle
        if (!last) {
            ostrm.write(buf_b.range((c + 1) * _WOutStrm - 1, c * _WOutStrm));
            e_ostrm.write(false);
        }
        c++;
    } // while
    // output the last data from input
    for (int i = 0; i < num_out; ++i) {
#pragma HLS pipeline II = 1
        if ((i + 1) * _WOutStrm <= up_pos) {
            ostrm.write(buf_b.range((i + 1) * _WOutStrm - 1, i * _WOutStrm));
            e_ostrm.write(false);
        }
    }
    e_ostrm.write(true);
}

/* read data from _NStrm input streams and output them to one stream in order
 * of round robin
 * assume the input and output width( _WInStrm and _WOutStrm) are different.
 */
template <int _WInStrm, int _WOutStrm, int _NStrm>
void stream_n_to_one_round_robin(hls::stream<ap_uint<_WInStrm> > istrms[_NStrm],
                                 hls::stream<bool> e_istrms[_NStrm],
                                 hls::stream<ap_uint<_WOutStrm> >& ostrm,
                                 hls::stream<bool>& e_ostrm) {
    const int buf_size = LCM<_WInStrm * _NStrm, _WOutStrm>::value;
    hls::stream<ap_uint<_WInStrm * _NStrm> > buf_n_strm;
#pragma HLS stream variable = buf_n_strm depth = 8
    hls::stream<bool> e_buf_n_strm;
#pragma HLS stream variable = e_buf_n_strm depth = 8
    hls::stream<ap_uint<buf_size> > buf_lcm_strm;
#pragma HLS stream variable = buf_lcm_strm depth = 8
    hls::stream<bool> e_buf_lcm_strm;
#pragma HLS stream variable = e_buf_lcm_strm depth = 8
    hls::stream<ap_uint<32> > left_n;   // how many input(_WInStrm bits) are stored
                                        // in  last data(_NStrm*WInstrm bits) in
                                        // buf_n_strm
    hls::stream<ap_uint<32> > left_lcm; // how many input(_WInStrm bits) are
                                        // stored in last data(buf_size bits) in
                                        // buf_lcm_strm

#pragma HLS dataflow

    /* read data    : read data from input streams, output _NStrm * _WInStrm bits
     * collect data : buffer  buf_size=lcm(_WInStrm*_NStrm, _WOutStrm) bits and
     *                output them.
     * distribute   : output buf_size/_WOutStrm  data when read buf_size bits once
     *
     * least common mutiple(lcm) is used for solving the difference between  the
     * input width and output width
     */

    stream_n_to_one_read<_WInStrm, _NStrm>(istrms, e_istrms, left_n, buf_n_strm, e_buf_n_strm);

    stream_n_to_one_collect<_WInStrm, _WOutStrm, _NStrm>(buf_n_strm, e_buf_n_strm, left_n, left_lcm, buf_lcm_strm,
                                                         e_buf_lcm_strm);

    stream_n_to_one_distribute<_WInStrm, _WOutStrm, _NStrm>(buf_lcm_strm, e_buf_lcm_strm, left_lcm, ostrm, e_ostrm);
}

} // details

template <int _WInStrm, int _WOutStrm, int _NStrm>
void streamNToOne(hls::stream<ap_uint<_WInStrm> > istrms[_NStrm],
                  hls::stream<bool> e_istrms[_NStrm],
                  hls::stream<ap_uint<_WOutStrm> >& ostrm,
                  hls::stream<bool>& e_ostrm,
                  RoundRobinT alg) {
    details::stream_n_to_one_round_robin<_WInStrm, _WOutStrm, _NStrm>(istrms, e_istrms, ostrm, e_ostrm);
}

//------------------------------------------------------------------------------//

namespace details {
template <typename _TIn, int _NStrm>
void stream_n_to_one_round_robin_type(hls::stream<_TIn> istrms[_NStrm],
                                      hls::stream<bool> e_istrms[_NStrm],
                                      hls::stream<_TIn>& ostrm,
                                      hls::stream<bool>& e_ostrm) {
    int id = 0;
    ap_uint<_NStrm> end = 0;
    ap_uint<_NStrm> end_flag = ~end;
    for (int i = 0; i < _NStrm; ++i) {
#pragma HLS unroll
        end[i] = e_istrms[i].read();
    }
    // output the data from input in order of round robin
    while (end != end_flag) {
#pragma HLS pipeline II = 1
        if (!end[id]) {
            _TIn d = istrms[id].read();
            end[id] = e_istrms[id].read();
            id = (id + 1) == _NStrm ? 0 : (id + 1);
            ostrm.write(d);
            e_ostrm.write(false);
        }
    } // while
    e_ostrm.write(true);
}

} // details

template <typename _TIn, int _NStrm>
void streamNToOne(hls::stream<_TIn> istrms[_NStrm],
                  hls::stream<bool> e_istrms[_NStrm],
                  hls::stream<_TIn>& ostrm,
                  hls::stream<bool>& e_ostrm,
                  RoundRobinT alg) {
    details::stream_n_to_one_round_robin_type<_TIn, _NStrm>(istrms, e_istrms, ostrm, e_ostrm);
}

} // utils_hw
} // common
} // xf

#endif // XF_UTILS_HW_STREAM_N1_RR
