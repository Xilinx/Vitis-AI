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

/**
 * @file multiplexer.hpp
 * @brief This file provides implementation of FIFO multiplexer wrapper class.
 *
 * This file is part of Vitis Utility Library.
 */

#ifndef XF_UTILS_HW_MUX_HPP
#define XF_UTILS_HW_MUX_HPP

#include "ap_int.h"
#include "hls_stream.h"

#include "xf_utils_hw/common.hpp"

namespace xf {
namespace common {
namespace utils_hw {

namespace details {

template <int N>
struct uchar_array {
    unsigned char c[N];
};

// XXX user can create their own specialization to remove bubbles, etc.
template <typename T>
struct as_ap_uint {
    typedef ap_uint<sizeof(T) * 8> type;
    static type cast(const T& v) {
#pragma HLS inline
        type d;
        union {
            T t;
            uchar_array<sizeof(T)> ca;
        } un;
        un.t = v;
        uchar_array<sizeof(T)> ca = un.ca;
        for (int i = 0; i < sizeof(T); ++i) {
#pragma HLS unroll
            d.range(7 + 8 * i, 8 * i) = ca.c[i];
        }
        return d;
    }
};

template <typename T>
struct ap_uint_as {
    static T cast(const ap_uint<sizeof(T) * 8>& v) {
#pragma HLS inline
        uchar_array<sizeof(T)> ca;
        for (int i = 0; i < sizeof(T); ++i) {
#pragma HLS unroll
            ca.c[i] = v.range(7 + 8 * i, 8 * i);
        }
        union {
            T t;
            uchar_array<sizeof(T)> ca;
        } un;
        un.ca = ca;
        T d = un.t;
        return d;
    }
};
} /* details */

enum MuxSide { MUX_SENDER, MUX_RECEIVER };

/**
 * @class Multiplexer multiplexer.hpp "xf_utils_hw/multiplexer.hpp"
 *
 * @brief Multiplexer is a wrapper around FIFO for sending/receiving data of different types.
 *
 * @tparam S specifies this wrapper is on which side of the FIFO.
 * @tparam W specifies the width of the FIFO.
 */
template <MuxSide S, int W>
class Multiplexer {
   private:
    hls::stream<ap_uint<W> >& _strm;

   public:
    Multiplexer(hls::stream<ap_uint<W> >& s) : _strm(s) {
#pragma HLS inline
    }
    Multiplexer() = delete;
    Multiplexer(const Multiplexer&) = delete;
    Multiplexer& operator=(const Multiplexer&) = delete;
    Multiplexer(Multiplexer&& o) : _strm(o._strm) {
#pragma HLS inline
    }
    Multiplexer& operator=(Multiplexer&&) = delete;

    /// get
    /// @tparam T type to receive
    /// @return one value
    template <typename T>
    T get() {
#pragma HLS inline
        static_assert(S == MUX_RECEIVER, "Multiplexer sender cannot get()");
        typename details::as_ap_uint<T>::type v; // v is ap_uint
        // use potentially multiple cycles to collect one instance of T
        for (int i = 0; i < ((v.width - 1) / W + 1); ++i) {
#pragma HLS unroll
            ap_uint<W> t = _strm.read();
            v.range(AP_MIN(W, v.width - W * i) - 1 + W * i, W * i) = t;
            XF_UTILS_HW_PRINT("read bus word %d (%s)", i, t.to_string(16).c_str());
        }
        return details::ap_uint_as<T>::cast(v);
    }

    /// get
    /// @tparam T type to receive
    /// @return one value
    template <typename T>
    void get(T& d) {
#pragma HLS inline
        d = get<T>();
    }

    /// put
    /// @tparam T type of data to send
    /// @param d data to send
    template <typename T>
    void put(const T& d) {
#pragma HLS inline
        static_assert(S == MUX_SENDER, "Multiplexer receiver cannot put()");
        auto v = details::as_ap_uint<T>::cast(d); // v is ap_uint
        // use potentially multiple cycles to transmit one instance of T
        for (int i = 0; i < ((v.width - 1) / W + 1); ++i) {
#pragma HLS unroll
            ap_uint<W> t = v.range(AP_MIN(W, v.width - W * i) - 1 + W * i, W * i);
            _strm.write(t);
            XF_UTILS_HW_PRINT("write bus word %d (%s)", i, t.to_string(16).c_str());
        }
    }
};

/**
 * @brief Create Multiplexer object with FIFO width inferred.
 *
 * @tparam S side of FIFO, must be provide by user.
 * @tparam W width of FIFO, can be inferred.
 *
 * @param p FIFO to be wrapped.
 *
 * @return Multiplexer object
 */
template <MuxSide S, int W>
Multiplexer<S, W> makeMux(hls::stream<ap_uint<W> >& p) {
    Multiplexer<S, W> mux(p);
    return mux;
}

} /* utils_hw */
} /* common */
} /* xf */

#endif
