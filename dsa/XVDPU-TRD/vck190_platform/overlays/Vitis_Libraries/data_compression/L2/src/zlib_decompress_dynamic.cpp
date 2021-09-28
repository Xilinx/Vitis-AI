/*
 * (c) Copyright 2019-2021 Xilinx, Inc. All rights reserved.
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
 *
 */

#include "zlib_decompress_dynamic.hpp"

const int c_historySize = LZ_MAX_OFFSET_LIMIT;

extern "C" {
void xilDecompressDynamic(hls::stream<ap_axiu<16, 0, 0, 0> >& inaxistreamd,
                          hls::stream<ap_axiu<MULTIPLE_BYTES * 8, 0, 0, 0> >& outaxistreamd) {
#ifdef FREE_RUNNING_KERNEL
#pragma HLS interface ap_ctrl_none port = return
#endif
#pragma HLS interface axis port = inaxistreamd
#pragma HLS interface axis port = outaxistreamd

    // Call for decompression
    xf::compression::inflateMultiByte<DECODER_TYPE, MULTIPLE_BYTES, LL_MODEL, c_historySize>(inaxistreamd,
                                                                                             outaxistreamd);
}
}
