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

#include "geoip_kernel.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

extern "C" void GeoIP_kernel(
    int ipNum, uint32* ip, uint64* netHigh16, uint512* netLow21, uint512* net2Low21, uint32* netID) {
    const int Depth = NIP;
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem0_0 port = ip depth = Depth
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem0_1 port = netHigh16 depth = 65536
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem0_2 port = netLow21 depth = 4194304
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem0_3 port = net2Low21 depth = 4194304
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 2 bundle = gmem0_4 port = netID depth = Depth

#pragma HLS INTERFACE s_axilite port = ipNum bundle = control
#pragma HLS INTERFACE s_axilite port = ip bundle = control
#pragma HLS INTERFACE s_axilite port = netHigh16 bundle = control
#pragma HLS INTERFACE s_axilite port = netLow21 bundle = control
#pragma HLS INTERFACE s_axilite port = net2Low21 bundle = control
#pragma HLS INTERFACE s_axilite port = netID bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#ifndef __SYNTHESIS__
    std::cout << "start GeoIP kernel" << std::endl;
#endif
    xf::data_analytics::text::GeoIP<uint512, uint64, uint32, uint16, 32, 24, TH1, TH2> geoip;
#ifndef __SYNTHESIS__
    std::cout << "start geoip.init" << std::endl;
#endif
    geoip.init(65536, netHigh16);
#ifndef __SYNTHESIS__
    std::cout << "start geoip.run, ipNum=" << ipNum << std::endl;
#endif
    geoip.run(ipNum, ip, netLow21, net2Low21, netID);
#ifndef __SYNTHESIS__
    std::cout << "end GeoIP kernel" << std::endl;
#endif
}
