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
 * @file duplicate_col.hpp
 * @brief This file contains column data manipulate helpers.
 */

#ifndef XF_DATABASE_DUPLICATE_COL_H
#define XF_DATABASE_DUPLICATE_COL_H

#ifndef __cplusplus
#error "Databse Library only works with C++."
#endif

#include <ap_int.h>
#include <hls_stream.h>

namespace xf {
namespace database {
/**
 * @brief Duplicate one column into two columns.
 *
 * @tparam W column data width in bits.
 *
 * @param d_in_strm input data stream.
 * @param e_in_strm end flag for input data.
 * @param d0_out_strm output data stream 0.
 * @param d1_out_strm output data stream 1.
 * @param e_out_strm end flag for output data.
 */
template <int W>
void duplicateCol(hls::stream<ap_uint<W> >& d_in_strm,
                  hls::stream<bool>& e_in_strm,
                  hls::stream<ap_uint<W> >& d0_out_strm,
                  hls::stream<ap_uint<W> >& d1_out_strm,
                  hls::stream<bool>& e_out_strm) {
    bool e = e_in_strm.read();
    while (!e) {
        ap_uint<W> d = d_in_strm.read();
        e = e_in_strm.read();
        //
        d0_out_strm.write(d);
        d1_out_strm.write(d);
        e_out_strm.write(false);
    }
    e_out_strm.write(true);
}
} // namespace database
} // namespace xf

#endif // XF_DATABASE_DUPLICATE_COL_H
