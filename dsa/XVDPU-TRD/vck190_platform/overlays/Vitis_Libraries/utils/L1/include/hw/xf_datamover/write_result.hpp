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

#ifndef DATAMOVER_WRITE_RESULT_HPP
#define DATAMOVER_WRITE_RESULT_HPP

#include "xf_datamover/types.hpp"
#include <hls_stream.h>

namespace xf {
namespace datamover {

namespace details {
void writeOneResult(hls::stream<CheckResult::type>& rs, CheckResult::type* rm) {
#pragma HLS inline
    CheckResult::type tmp = rs.read();
    *rm = tmp;
}
} /* details */

template <typename Ts, typename Tm>
void writeResult(Ts& rs, Tm& rm) {
    details::writeOneResult(rs, rm);
}

/**
 * Write CheckResult to AXI master port.
 *
 * Stream and pointer pairs are handled **sequentially**.
 * It is allowed that all these pointers are bundled together to one AXI master port.
 *
 * This function is expected to write just small amount of data to output pointers.
 * It is not designed to write result at line-speed.
 *
 * @tparam Ts the type of CheckResult stream.
 * @tparam Tm the type of CheckResult pointer.
 *
 * @param rs CheckResult stream.
 * @param rm CheckResult master port.
 */
template <typename Ts, typename Tm, typename... Args>
void writeResult(Ts& rs, Tm& rm, Args&... args) {
    details::writeOneResult(rs, rm);
    // C++ will expand the recursive call in variadic template.
    writeResult(args...);
}

} /* datamover */
} /* xf */
#endif
