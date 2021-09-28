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

#include "kernel_pseudosqrt.hpp"
#include "xf_solver_L1.hpp"

#ifdef KERNEL0
#ifdef _USE_STRM_
extern "C" void kernel_pseudosqrt_0(int nrows,
                                    hls::stream<ap_uint<DTLen * TO> >& matIn,
                                    hls::stream<ap_uint<DTLen * TO> >& matOut) {
    xf::solver::pseudosqrtStrm<DT, matSize, unrollNm1, DTLen, TO>(nrows, matIn, matOut);
}
#else
extern "C" void kernel_pseudosqrt_0(int nrows, DT* matIn, DT* matOut) {
    xf::solver::pseudosqrt<DT, matSize, unrollNm1>(nrows, matIn, matOut);
}
#endif
#endif
