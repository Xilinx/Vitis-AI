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

#include "WJ_kernel.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

extern "C" void WJ_kernel(ap_uint<W_IN>* cfgBuff,
                          ap_uint<W_IN>* msgBuff,
                          ap_uint<16>* msgLenBuff,
                          ap_uint<32>* msgPosBuff,
                          ap_uint<256>* geoBuff,
                          ap_uint<64>* geoLenBuff,
                          ap_uint<32>* geoPosBuff,
                          ap_uint<W_OUT>* outJson) {
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    8 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem0_0 port = cfgBuff
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    8 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem0_0 port = msgBuff
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem0_1 port = msgLenBuff
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem0_2 port = msgPosBuff
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    256 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem0_3 port = geoBuff
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem0_4 port = geoLenBuff
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem0_5 port = geoPosBuff
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 2 bundle = gmem0_6 port = outJson

#pragma HLS INTERFACE s_axilite port = cfgBuff bundle = control
#pragma HLS INTERFACE s_axilite port = msgBuff bundle = control
#pragma HLS INTERFACE s_axilite port = msgLenBuff bundle = control
#pragma HLS INTERFACE s_axilite port = msgPosBuff bundle = control
#pragma HLS INTERFACE s_axilite port = geoBuff bundle = control
#pragma HLS INTERFACE s_axilite port = geoLenBuff bundle = control
#pragma HLS INTERFACE s_axilite port = geoPosBuff bundle = control
#pragma HLS INTERFACE s_axilite port = outJson bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#ifndef __SYNTHESIS__
    std::cout << "start write json kernel" << std::endl;
#endif

    ap_uint<256> msgFldArr[32];
    ap_uint<16> msgFldLenArr[32];

    // get the message number
    ap_uint<32> num = 0;
    num(31, 16) = msgLenBuff[0];
    num(15, 0) = msgLenBuff[1];
    num = num - 2;

    ap_uint<16> fielddNum = 0;
    ap_uint<32> head_0 = cfgBuff[0];
    ap_uint<64> head_1 = cfgBuff[1];
    ap_uint<16> fieldNum = head_1.range(63, 48);
    fieldNum = fieldNum + 1;
#ifndef __SYNTHESIS__
    printf("total message number: %d, field number: %d\n", (unsigned int)num, (unsigned int)fieldNum - 2);
#endif

    // store field name and field length
    for (unsigned int i = 0; i < fieldNum - 2; ++i) {
#pragma HLS pipeline II = 1
        msgFldLenArr[i] = cfgBuff[head_0 + i];
        //#ifndef __SYNTHESIS__
        //    printf("lentgh of field[%d]: %d\n", i, (unsigned int)msgFldLenArr[i]);
        //#endif
    }
    // ToDO: need to fix if the length of capture name is longer than 256 bits.
    const int vec = 256 / W_IN;
    for (unsigned int i = 0; i < fieldNum - 2; ++i) {
        ap_uint<256> fld_name = 0;
        for (int j = 0; j < vec; ++j) {
#pragma HLS pipeline II = 1
            fld_name.range((j + 1) * W_IN - 1, j * W_IN) = cfgBuff[head_0 + fieldNum - 2 + i * vec + j];
        }
        msgFldArr[i] = fld_name;
    }
    //#ifndef __SYNTHESIS__
    // for(int i = 0; i < fieldNum - 2; ++i) {
    //   for(int j = 0;j < msgFldLenArr[i]; ++j) {
    //       printf("%c", (unsigned int)msgFldArr[i].range((j+1)*8-1, j*8));
    //   }
    //   printf("\n");
    //}
    //#endif

    xf::data_analytics::text::writeJson<W_IN, W_OUT>(num, fieldNum, msgBuff, &msgLenBuff[2], &msgPosBuff[1], msgFldArr,
                                                     msgFldLenArr, geoBuff, geoLenBuff, geoPosBuff, outJson);
#ifndef __SYNTHESIS__
    std::cout << "outJson = " << outJson[0] << std::endl;
    std::cout << "end write json kernel" << std::endl;
#endif
}
