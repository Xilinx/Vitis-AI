
/**********
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
 * **********/

#ifndef XF_BLAS_BLASKERNELS_HPP
#define XF_BLAS_BLASKERNELS_HPP

#ifndef __cplusplus
#error "BLAS Library only works with C++."
#endif

#include "blasInstr.hpp"
#include "gemmMatMoverL2.hpp"

namespace xf {
namespace blas {

template <unsigned int t_MaxNumInstrs,
          unsigned int t_MemWordsPerInstr,
          unsigned int t_ParEntries,
          unsigned int t_MparWords,
          unsigned int t_KparWords,
          unsigned int t_NparWords,
          unsigned int t_MemBits>
void gemmLoad(ap_uint<t_MemBits>* p_ptr,
              hls::stream<ap_uint<t_MemBits> >& p_aStr,
              hls::stream<ap_uint<t_MemBits> >& p_bStr,
              hls::stream<ap_uint<t_MemBits> >& p_xStr,
              hls::stream<ap_uint<16> >& p_opCodeStr) {
    static const unsigned int t_MemWordBytes = t_MemBits / 8;

    ap_uint<t_MemBits> l_progMem[t_MaxNumInstrs][t_MemWordsPerInstr];
#pragma HLS ARRAY_PARTITION variable = l_progMem complete dim = 2

    // load Instructions
    loadInstr<t_MaxNumInstrs, t_MemWordsPerInstr, t_MemBits>(p_ptr, l_progMem);

    unsigned int l_pc = 0;
    uint16_t l_opCode = OpCodeType::OpControl;
    ap_uint<t_MemBits> l_instr[t_MemWordsPerInstr];
    do {
#pragma HLS ARRAY_PARTITION variable = l_instr complete dim = 1
        getInstr<t_MaxNumInstrs, t_MemWordsPerInstr, t_MemBits>(l_progMem, l_pc, l_instr);
        l_opCode = getOpCode<t_MemWordsPerInstr, t_MemBits>(l_instr);
        switch (l_opCode) {
            case OpCodeType::OpGemmLdSt: {
                p_opCodeStr.write(l_opCode);
                for (unsigned int i = 0; i < t_MemWordsPerInstr; ++i) {
#pragma HLS PIPELINE
                    p_aStr.write(l_instr[i]);
                    p_bStr.write(l_instr[i]);
                    p_xStr.write(l_instr[i]);
                }
                uint32_t l_aOffset, l_bOffset, l_xOffset, l_aWrOffset, l_bWrOffset, l_xWrOffset, l_m, l_k, l_n;
                decodeGemmLdStInstr<t_MemWordsPerInstr, t_MemBits>(
                    l_instr, l_aOffset, l_bOffset, l_xOffset, l_aWrOffset, l_bWrOffset, l_xWrOffset, l_m, l_k, l_n);
                ap_uint<t_MemBits>* l_aPtr = p_ptr + l_aOffset;
                ap_uint<t_MemBits>* l_bPtr = p_ptr + l_bOffset;
                ap_uint<t_MemBits>* l_xPtr = p_ptr + l_xOffset;
                gemmLoadDat<t_ParEntries, t_MparWords, t_KparWords, t_NparWords, t_MemBits>(
                    l_aPtr, l_bPtr, l_xPtr, l_m, l_k, l_n, p_aStr, p_bStr, p_xStr);
                break;
            }
            case OpCodeType::OpControl: {
                break;
            }
            default: {
#ifndef __SYNTHESIS__
                assert(false);
#endif
            }
        }
        l_pc++;
    } while ((l_pc < t_MaxNumInstrs) && (l_opCode != OpCodeType::OpControl));

    encodeControlInstr<t_MemWordsPerInstr, t_MemBits>(l_instr);
    for (unsigned int i = 0; i < t_MemWordsPerInstr; ++i) {
#pragma HLS PIPELINE
        p_aStr.write(l_instr[i]);
        p_bStr.write(l_instr[i]);
        p_xStr.write(l_instr[i]);
    }
    p_opCodeStr.write(OpCodeType::OpControl);
}

template <unsigned int t_ResOffsetBytes,
          unsigned int t_MemWordsPerInstr,
          unsigned int t_ParEntries,
          unsigned int t_MparWords,
          unsigned int t_KparWords,
          unsigned int t_NparWords,
          unsigned int t_MemBits>
void gemmStoreABX(hls::stream<ap_uint<t_MemBits> >& p_aStr,
                  hls::stream<ap_uint<t_MemBits> >& p_bStr,
                  hls::stream<ap_uint<t_MemBits> >& p_xStr,
                  hls::stream<ap_uint<32> >& p_resStr,
                  ap_uint<t_MemBits>* p_ptr) {
    static unsigned int t_MemWordBytes = t_MemBits / 8;
    static unsigned int t_ResOffset = t_ResOffsetBytes / t_MemWordBytes;
    static unsigned int t_InstrBytes = t_MemWordsPerInstr * t_MemWordBytes;

    ap_uint<t_MemBits> l_instrA[t_MemWordsPerInstr];
    ap_uint<t_MemBits> l_instrB[t_MemWordsPerInstr];
    ap_uint<t_MemBits> l_instrX[t_MemWordsPerInstr];
#pragma HLS ARRAY_PARTITION variable = l_instrA complete dim = 1
#pragma HLS ARRAY_PARTITION variable = l_instrB complete dim = 1
#pragma HLS ARRAY_PARTITION variable = l_instrX complete dim = 1
    for (unsigned int i = 0; i < t_MemWordsPerInstr; ++i) {
#pragma HLS PIPELINE
        l_instrA[i] = p_aStr.read();
        l_instrB[i] = p_bStr.read();
        l_instrX[i] = p_xStr.read();
    }
    uint16_t l_opCode = getOpCode<t_MemWordsPerInstr, t_MemBits>(l_instrA);
    while (l_opCode != OpCodeType::OpControl) {
        uint32_t l_aOffset, l_bOffset, l_xOffset, l_aWrOffset, l_bWrOffset, l_xWrOffset, l_m, l_k, l_n;
        decodeGemmLdStInstr<t_MemWordsPerInstr, t_MemBits>(l_instrX, l_aOffset, l_bOffset, l_xOffset, l_aWrOffset,
                                                           l_bWrOffset, l_xWrOffset, l_m, l_k, l_n);
        ap_uint<t_MemBits>* l_a = p_ptr + l_aWrOffset;
        ap_uint<t_MemBits>* l_b = p_ptr + l_bWrOffset;
        ap_uint<t_MemBits>* l_x = p_ptr + l_xWrOffset;
        gemmStoreDatABX<t_ParEntries, t_MparWords, t_KparWords, t_NparWords, t_MemBits>(p_aStr, p_bStr, p_xStr, l_m,
                                                                                        l_k, l_n, l_a, l_b, l_x);
        for (unsigned int i = 0; i < t_MemWordsPerInstr; ++i) {
#pragma HLS PIPELINE
            l_instrA[i] = p_aStr.read();
            l_instrB[i] = p_bStr.read();
            l_instrX[i] = p_xStr.read();
        }
        l_opCode = getOpCode<t_MemWordsPerInstr, t_MemBits>(l_instrA);
    }
    uint32_t l_cycles = (uint32_t)(p_resStr.read());
    encodeResInstr<t_MemWordsPerInstr, t_MemBits>(l_cycles, l_instrX);
    for (unsigned int i = 0; i < t_MemWordsPerInstr; ++i) {
#pragma HLS PIPELINE
        p_ptr[t_ResOffset + i] = l_instrX[i];
    }
}

void gemmLdStTimer(hls::stream<ap_uint<16> >& p_inStr, hls::stream<ap_uint<32> >& p_outStr) {
    ap_uint<32> l_cycles = 0;
    ap_uint<16> l_opCode = p_inStr.read();
    while (l_opCode != OpCodeType::OpControl) {
#pragma HLS PIPELINE
        (void)p_inStr.read_nb(l_opCode);
        l_cycles++;
    }
    p_outStr.write(l_cycles);
}

template <typename t_OpCodePktType, typename t_StatsPktType>
void gemmLdStTimerPkt(hls::stream<t_OpCodePktType>& p_inStr, hls::stream<t_StatsPktType>& p_outStr) {
    ap_uint<32> l_cycles = 0;
    t_OpCodePktType l_opPkt = p_inStr.read();
    ap_uint<16> l_opCode = l_opPkt.data;
    while (l_opCode != OpCodeType::OpControl) {
#pragma HLS PIPELINE
        (void)p_inStr.read_nb(l_opPkt);
        l_opCode = l_opPkt.data;
        l_cycles++;
    }
    t_StatsPktType l_statsPkt;
    l_statsPkt.data = l_cycles;
    p_outStr.write(l_statsPkt);
}

template <typename t_PktType, unsigned int t_DataBits>
void dataStr2PktStr(hls::stream<ap_uint<t_DataBits> >& p_inDatStr, hls::stream<t_PktType>& p_outPktStr) {
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS PIPELINE
    ap_uint<t_DataBits> l_dat = p_inDatStr.read();
    t_PktType l_pkt;
    l_pkt.data = l_dat;
    p_outPktStr.write(l_pkt);
}

template <typename t_PktType, unsigned int t_DataBits>
void pktStr2DatStr(hls::stream<t_PktType>& p_inPktStr, hls::stream<ap_uint<t_DataBits> >& p_outDatStr) {
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS PIPELINE
    t_PktType l_pkt = p_inPktStr.read();
    ap_uint<t_DataBits> l_dat = l_pkt.data;
    p_outDatStr.write(l_dat);
}

template <typename t_DataPktType,
          typename t_OpCodePktType,
          unsigned int t_MaxNumInstrs,
          unsigned int t_MemWordsPerInstr,
          unsigned int t_ParEntries,
          unsigned int t_MparWords,
          unsigned int t_KparWords,
          unsigned int t_NparWords,
          unsigned int t_MemBits>
void gemmLoadPkt(ap_uint<t_MemBits>* p_ptr,
                 hls::stream<t_DataPktType>& p_aStr,
                 hls::stream<t_DataPktType>& p_bStr,
                 hls::stream<t_DataPktType>& p_xStr,
                 hls::stream<t_OpCodePktType>& p_opCodeStr) {
#pragma HLS DATAFLOW
    hls::stream<ap_uint<t_MemBits> > l_aStr;
    hls::stream<ap_uint<t_MemBits> > l_bStr;
    hls::stream<ap_uint<t_MemBits> > l_xStr;
    hls::stream<ap_uint<16> > l_opCodeStr;

    gemmLoad<t_MaxNumInstrs, t_MemWordsPerInstr, t_ParEntries, t_MparWords, t_KparWords, t_NparWords, t_MemBits>(
        p_ptr, l_aStr, l_bStr, l_xStr, l_opCodeStr);
    dataStr2PktStr<t_DataPktType, t_MemBits>(l_aStr, p_aStr);
    dataStr2PktStr<t_DataPktType, t_MemBits>(l_bStr, p_bStr);
    dataStr2PktStr<t_DataPktType, t_MemBits>(l_xStr, p_xStr);
    dataStr2PktStr<t_OpCodePktType, 16>(l_opCodeStr, p_opCodeStr);
}

template <typename t_DataPktType,
          typename t_StatsPktType,
          unsigned int t_ResOffsetBytes,
          unsigned int t_MemWordsPerInstr,
          unsigned int t_ParEntries,
          unsigned int t_MparWords,
          unsigned int t_KparWords,
          unsigned int t_NparWords,
          unsigned int t_MemBits>
void gemmStoreABXpkt(hls::stream<t_DataPktType>& p_aStr,
                     hls::stream<t_DataPktType>& p_bStr,
                     hls::stream<t_DataPktType>& p_xStr,
                     hls::stream<t_StatsPktType>& p_resStr,
                     ap_uint<t_MemBits>* p_ptr) {
#pragma HLS DATAFLOW
    hls::stream<ap_uint<t_MemBits> > l_aStr;
    hls::stream<ap_uint<t_MemBits> > l_bStr;
    hls::stream<ap_uint<t_MemBits> > l_xStr;
    hls::stream<ap_uint<32> > l_resStr;

    pktStr2DatStr<t_DataPktType, t_MemBits>(p_aStr, l_aStr);
    pktStr2DatStr<t_DataPktType, t_MemBits>(p_bStr, l_bStr);
    pktStr2DatStr<t_DataPktType, t_MemBits>(p_xStr, l_xStr);
    pktStr2DatStr<t_StatsPktType, 32>(p_resStr, l_resStr);

    gemmStoreABX<t_ResOffsetBytes, t_MemWordsPerInstr, t_ParEntries, t_MparWords, t_KparWords, t_NparWords, t_MemBits>(
        l_aStr, l_bStr, l_xStr, l_resStr, p_ptr);
}
}
}

#endif
