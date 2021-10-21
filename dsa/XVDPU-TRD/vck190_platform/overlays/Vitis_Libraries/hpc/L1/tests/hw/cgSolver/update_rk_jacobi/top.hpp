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
#include "ap_int.h"
#include "hls_stream.h"
#include "params.hpp"

void top(ap_uint<CG_memBits>* p_rk_in,
         ap_uint<CG_memBits>* p_rk_out,
         ap_uint<CG_memBits>* p_zk,
         ap_uint<CG_memBits>* p_jacobi,
         ap_uint<CG_memBits>* p_Apk,
         hls::stream<ap_uint<CG_tkWidth> >& p_tokenIn,
         hls::stream<ap_uint<CG_tkWidth> >& p_tokenOut);
