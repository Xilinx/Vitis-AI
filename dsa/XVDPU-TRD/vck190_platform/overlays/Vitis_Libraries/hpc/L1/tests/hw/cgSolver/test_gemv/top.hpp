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
#pragma once
#include <cstdint>
#include "ap_int.h"
#include "hls_stream.h"
#include "params.hpp"

void top(uint32_t p_size,
         hls::stream<ap_uint<sizeof(CG_dataType) * 8> >& p_dot,
         hls::stream<ap_uint<CG_memBits> > p_A[CG_numChannels],
         hls::stream<ap_uint<CG_memBits> >& p_pk,
         hls::stream<ap_uint<sizeof(CG_dataType) * 8> >& p_pkc,
         hls::stream<ap_uint<sizeof(CG_dataType) * 8> >& p_Apk);
