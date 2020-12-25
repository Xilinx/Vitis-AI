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

/*
 * This file contains the definition of the data types for AXI streaming.
 * ap_axi_s is a signed interpretation of the AXI stream
 * ap_axi_u is an unsigned interpretation of the AXI stream
 */

#ifndef __XF__AXI_SDATA__
#define __XF__AXI_SDATA__

#include "ap_int.h"

namespace xf {
namespace cv {

template <int D, int U, int TI, int TD>
struct ap_axis {
    ap_int<D> data;
    ap_uint<(D + 7) / 8> keep;
    ap_uint<(D + 7) / 8> strb;
    ap_uint<U> user;
    ap_uint<1> last;
    ap_uint<TI> id;
    ap_uint<TD> dest;
};

template <int D, int U, int TI, int TD>
struct ap_axiu {
    ap_uint<D> data;
    ap_uint<(D + 7) / 8> keep;
    ap_uint<(D + 7) / 8> strb;
    ap_uint<U> user;
    ap_uint<1> last;
    ap_uint<TI> id;
    ap_uint<TD> dest;
};

// typedef ap_axis<int D, int U, int TI, int TD> ap_axis_unsigned<int D, int U, int TI, int TD>;
} // namespace xf
} // namespace cv
#endif
