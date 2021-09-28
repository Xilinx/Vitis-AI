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

#ifndef DATAMOVER_READ_CONST_HPP
#define DATAMOVER_READ_CONST_HPP

#include "xf_datamover/types.hpp"
#include <hls_stream.h>

namespace xf {
namespace datamover {

namespace details {
void readOneConst(ConstData::type* mm, hls::stream<ConstData::type>& cs, uint64_t sz) {
#pragma HLS inline
    static const int bytePerData = ConstData::Port_Width / 8;
    int nBlks = sz / bytePerData + ((sz % bytePerData) > 0);
LOOP_FULL_BLK:
    for (int i = 0; i < nBlks; i++) {
#pragma HLS pipeline II = 1
        ConstData::type tmp = mm[i];
        cs.write(tmp);
    }
}
} /* details */

template <typename Tm, typename Ts, typename Tz>
void readConst(Tm& mm, Ts& cs, Tz& sz) {
    details::readOneConst(mm, cs, sz);
}

/**
 * Read constant data from AXI master port to initilization stream.
 *
 * Stream and pointer pairs are handled **sequentially**.
 * It is allowed that all these pointers are bundled together to one AXI master port.
 *
 * @tparam Tm the type of pointer.
 * @tparam Ts the type of stream.
 * @tparam Tz the type of size.
 *
 * @param mm pointer.
 * @param cs constant stream.
 * @param sz the size of initializing constant, in bytes.
 */
template <typename Tm, typename Ts, typename Tz, typename... Args>
void readConst(Tm& mm, Ts& cs, Tz& sz, Args&... args) {
    details::readOneConst(mm, cs, sz);
    // C++ will expand the recursive call in variadic template.
    readConst(args...);
}

} /* datamover */
} /* xf */
#endif
