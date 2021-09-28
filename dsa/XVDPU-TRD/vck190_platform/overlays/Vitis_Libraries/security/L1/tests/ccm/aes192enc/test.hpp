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
#ifndef _TEST_H_
#define _TEST_H_

#include <ap_int.h>
#include "hls_stream.h"

// size of the tag in byte
#define TAG_SIZE 4
// size of the nonce in byte
#define N_SIZE 7

void test(hls::stream<ap_uint<128> >& payloadStrm,
          hls::stream<ap_uint<192> >& cipherkey,
          hls::stream<ap_uint<8 * N_SIZE> >& nonceStrm,
          hls::stream<ap_uint<128> >& ADStrm,
          hls::stream<ap_uint<64> >& lenADStrm,
          hls::stream<ap_uint<64> >& lenPldStrm,
          hls::stream<bool>& endLenStrm,
          hls::stream<ap_uint<128> >& cipherStrm,
          hls::stream<ap_uint<64> >& lenCphStrm,
          hls::stream<ap_uint<8 * TAG_SIZE> >& tagStrm,
          hls::stream<bool>& endTagStrm);
#endif
