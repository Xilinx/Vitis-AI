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
#ifndef XF_UTILS_HW_STREAM_1N_RR_H
#define XF_UTILS_HW_STREAM_1N_RR_H

/**
 * @file stream_one_to_n/round_robin.hpp
 * @brief header for distribute in round-roubin order.
 */

#include "xf_utils_hw/types.hpp"
#include "xf_utils_hw/enums.hpp"
#include "xf_utils_hw/common.hpp"

// Forward decl ===============================================================

namespace xf {
namespace common {
namespace utils_hw {

/**
 * @brief stream distribute, in round-robin order from first output.
 *
 * The input stream is assumed to be conpact data to be splitted into
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
                  RoundRobinT alg);

/**
 * @brief stream distribute, in round-robin order from first output.
 *
 * @tparam _TIn  the type of input stream.
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
                  RoundRobinT alg);
} // utils_hw
} // common
} // xf

// Implementation =============================================================

namespace xf {
namespace common {
namespace utils_hw {

namespace details {

template <int _WInStrm, int _WOutStrm, int _NStrm>
void stream_one_to_n_rr(hls::stream<ap_uint<_WInStrm> >& istrm,
                        hls::stream<bool>& e_istrm,
                        hls::stream<ap_uint<_WOutStrm> > ostrms[_NStrm],
                        hls::stream<bool> e_ostrms[_NStrm]) {
    /*
     * while a buffer is being filled by input data, another one is being read to
     * ouput streams except the first time buff_a is not wrote yet.
     * p is an iterator, range from 0 to  max(num_in, count_out), it executes at
     * least one operation(buffer or output data) at each cycle.
     * when p=max, it means a buffer is consumed and another is full already,so
     * switch them.
     * e.g  _WInStrm=3, _WOutStrm=2, _NStrm=1  the width of buffer = 6 ,
     * ie 1 buffer = 2 input data = 3 ouput data,switch buffers every  max(3,2)=3
     * cycles.
     *  6 bits:  2 input data |---|---| = 1 buffer |------| = 3 output data
     * |--|--|--|
     *  pipeline:
     *    write buff   wb   wa  wb  wa  wb  wa
     *    input       |-- |-- |-- |-- |-- |-- |
     *    output          |---|---|---|---|---|---|
     *    read buff        rb   ra  rb  ra  rb  ra
     *    in which, wa: write buff_a,  wb: write buff_b
     *              ra: read  buff_a,  rb: write buff_b
     *
     * when _NStrm=3
     * pipeline:
     *    write buff   wb wa wb wa wb wa
     *    input       |--|--|--|--|--|--|
     *    output         |- |- |- |- |- |- |
     *    read buff       rb ra rb ra rb ra
     *
     */
    // least common multiple of _WInStrm and _WOutStrm*_NStrm as the width of
    // ping-pong buffer
    const int buf_size = LCM<_WInStrm, _NStrm * _WOutStrm>::value;
    // the number of input data to fill the buffer to full
    const int num_in = buf_size / _WInStrm;
    const int num_out = buf_size / _WOutStrm;
    const int count_out = num_out / _NStrm;
    const int max = num_in > count_out ? num_in : count_out;
    ap_uint<buf_size> buff_a = 0;
    ap_uint<buf_size> buff_b = 0;
#if !defined(__SYNTHESIS__) && XF_UTILS_HW_STRM_1NRR_DEBUG == 1
    std::cout << "LCM(" << _WInStrm << ", " << _WOutStrm << ")=" << buf_size << std::endl;
    std::cout << "_WInStrm =" << _WInStrm << ","
              << "num_in =" << num_in << std::endl;
    std::cout << "_WOutStrm =" << _WOutStrm << ","
              << "num_out =" << num_out << std::endl;
#endif
    int p = 0;
    bool sw = false;
    bool start = false;
    bool last = e_istrm.read();
    while (!last) {
#pragma HLS pipeline II = 1
        // buffer data
        if (p < num_in) {
            ap_uint<_WInStrm> d = istrm.read();
            if (sw)
                buff_a.range((p + 1) * _WInStrm - 1, p * _WInStrm) = d;
            else
                buff_b.range((p + 1) * _WInStrm - 1, p * _WInStrm) = d;
            last = e_istrm.read();
        }

        // output data
        if (p < count_out && start) {
            for (int id = 0; id < _NStrm; ++id) {
#pragma HLS unroll
                ap_uint<_WOutStrm> d =
                    sw ? buff_b.range((p * _NStrm + id + 1) * _WOutStrm - 1, (p * _NStrm + id) * _WOutStrm)
                       : buff_a.range((p * _NStrm + id + 1) * _WOutStrm - 1, (p * _NStrm + id) * _WOutStrm);
                ostrms[id].write(d);
                e_ostrms[id].write(false);
            }
        }
        if (++p == max) {
            // switch them to  access
            sw = !sw;
            p = 0;
            start = true;
#if !defined(__SYNTHESIS__) && XF_UTILS_HW_STRM_1NRR_DEBUG == 1
            std::cout << "buff_a =  " << std::hex << buff_a << std::endl;
            std::cout << "buff_b =  " << std::hex << buff_b << std::endl;
#endif
        }
    } // end while

#if !defined(__SYNTHESIS__) && XF_UTILS_HW_STRM_1NRR_DEBUG == 1
    std::cout << "sw     =  " << sw << std::endl;
    std::cout << "p      =  " << std::dec << p << std::endl;
    std::cout << "buff_a =  " << std::hex << buff_a << std::endl;
    std::cout << "buff_b =  " << std::hex << buff_b << std::endl;
#endif

    int id = 0;
    //  the output operation is suspended in the  while-loop when last=true, so
    //  continully ouput data from the same buffer.
    for (int c = p * _NStrm; c < num_out; ++c) {
#pragma HLS pipeline II = 1
        if (start) {
            int low = c * _WOutStrm;
            int up = low + _WOutStrm - 1;
            ap_uint<_WOutStrm> d = sw ? buff_b.range(up, low) : buff_a.range(up, low);
            ostrms[id].write(d);
            e_ostrms[id].write(false);
            id = (id + 1) == _NStrm ? 0 : (id + 1);
        }
    }
    // output the data in another buffer.
    for (int c = 0; c < num_out; ++c) {
#pragma HLS pipeline II = 1
        if ((c + 1) * _WOutStrm <= p * _WInStrm) {
            int low = c * _WOutStrm;
            int up = low + _WOutStrm - 1;
            ap_uint<_WOutStrm> d = sw ? buff_a.range(up, low) : buff_b.range(up, low);
            ostrms[id].write(d);
            e_ostrms[id].write(false);
            id = (id + 1) == _NStrm ? 0 : (id + 1);
        }
    }
    for (int i = 0; i < _NStrm; ++i) {
#pragma HLS unroll
        e_ostrms[i].write(true);
    }
}

} // details

template <int _WInStrm, int _WOutStrm, int _NStrm>
void streamOneToN(hls::stream<ap_uint<_WInStrm> >& istrm,
                  hls::stream<bool>& e_istrm,
                  hls::stream<ap_uint<_WOutStrm> > ostrms[_NStrm],
                  hls::stream<bool> e_ostrms[_NStrm],
                  RoundRobinT alg) {
    details::stream_one_to_n_rr<_WInStrm, _WOutStrm, _NStrm>(istrm, e_istrm, ostrms, e_ostrms);
}

// -------------------------------------------------------------------

namespace details {
template <typename _TIn, int _NStrm>
void stream_one_to_n_rr_type(hls::stream<_TIn>& istrm,
                             hls::stream<bool>& e_istrm,
                             hls::stream<_TIn> ostrms[_NStrm],
                             hls::stream<bool> e_ostrms[_NStrm]) {
    int id = 0;
    bool last = e_istrm.read();
    while (!last) {
#pragma HLS pipeline II = 1
        _TIn data = istrm.read();
        ostrms[id].write(data);
        e_ostrms[id].write(false);
        last = e_istrm.read();
        id = (id + 1 == _NStrm) ? 0 : (id + 1);
    }

    for (int i = 0; i < _NStrm; ++i) {
#pragma HLS unroll
        e_ostrms[i].write(true);
    }
}
} // details

template <typename _TIn, int _NStrm>
void streamOneToN(hls::stream<_TIn>& istrm,
                  hls::stream<bool>& e_istrm,
                  hls::stream<_TIn> ostrms[_NStrm],
                  hls::stream<bool> e_ostrms[_NStrm],
                  RoundRobinT alg) {
    details::stream_one_to_n_rr_type<_TIn, _NStrm>(istrm, e_istrm, ostrms, e_ostrms);
}

} // utils_hw
} // common
} // xf

#endif // XF_UTILS_HW_STREAM_1N_RR_H
