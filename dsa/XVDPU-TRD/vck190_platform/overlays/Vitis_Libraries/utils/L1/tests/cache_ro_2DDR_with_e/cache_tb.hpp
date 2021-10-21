/*
 * Copyright 2020 Xilinx, Inc.
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
#include <iostream>
#include <stdlib.h>
#include <hls_stream.h>
#include <ap_int.h>

#define DDRSIZEIN512 (5000)
#define BUSADDRWIDTH (32)
#define BUSDATAWIDTH (64)
#define EACHLINE (512 / BUSDATAWIDTH)
#define RAMROW (4096)
#define GRPRAM (4)

#define TOTALADDRWIDTH (15) // should be smaller for CoSim 15 for example

void syn_top(hls::stream<ap_uint<BUSADDRWIDTH> >& raddrStrm,
             hls::stream<bool>& e_raddrStrm,
             hls::stream<ap_uint<BUSDATAWIDTH> >& rdataStrm0,
             hls::stream<bool>& e_rdataStrm0,
             hls::stream<ap_uint<BUSDATAWIDTH> >& rdataStrm1,
             hls::stream<bool>& e_rdataStrm1,
             ap_uint<512>* ddrMem0,
             ap_uint<512>* ddrMem1);
