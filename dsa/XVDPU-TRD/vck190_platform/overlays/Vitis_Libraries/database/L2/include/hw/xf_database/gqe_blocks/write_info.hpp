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
#ifndef GQE_WRITE_INFO_HPP
#define GQE_WRITE_INFO_HPP

#include "gqe_types.hpp"
#include <ap_int.h>
#include <hls_stream.h>

namespace xf {
namespace database {
namespace gqe {

// for gqe_aggr
template <int _WStrm, int _Lens>
void write_info(hls::stream<ap_uint<_WStrm> >& info_strm, ap_uint<_WStrm>* buff) {
    ap_uint<32> addr = 0;
    for (int i = 0; i < _Lens; i++) {
        buff[addr++] = info_strm.read();
    }
}

} // namespace gqe
} // namespace database
} // namespace xf

#endif
