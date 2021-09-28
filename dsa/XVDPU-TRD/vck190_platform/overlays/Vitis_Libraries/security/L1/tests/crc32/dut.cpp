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
 * @file dut.cpp
 *
 * @brief This file contains top function of test case.
 */

#include "dut.hpp"

void dut(hls::stream<ap_uint<32> >& crcInitStrm,
         hls::stream<ap_uint<8 * W> >& inStrm,
         hls::stream<ap_uint<32> >& inLenStrm,
         hls::stream<bool>& endInStrm,
         hls::stream<ap_uint<32> >& outStrm,
         hls::stream<bool>& endOutStrm) {
    xf::security::crc32<W>(crcInitStrm, inStrm, inLenStrm, endInStrm, outStrm, endOutStrm);
}
