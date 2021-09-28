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
 * @file geoip.hpp
 * @brief This file include the class GeoIP
 *
 */
#ifndef XF_TEXT_GEOIP_HPP
#define XF_TEXT_GEOIP_HPP

#ifndef __SYNTHESIS__
#include <iostream>
#endif
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>

namespace xf {
namespace data_analytics {
namespace text {

/**
 * @brief GeoIP Query geographic location ID based on IP
 *
 * @tparam uint512 512-bit wide unsigned integer data type
 * @tparam uint64 64-bit wide unsigned integer data type
 * @tparam uint32 32-bit wide unsigned integer data type
 * @tparam uint16 16-bit wide unsigned integer data type
 * @tparam Len1 The number of IPs placed in a 512-bit wide data
 * @tparam Len2 The number of networks placed in a 512-bit wide data
 * @tparam TH1 Query the query depth of the IP range
 * @tparam TH2 Query the depth of the query where the IP is located
 */
template <typename uint512,
          typename uint64,
          typename uint32,
          typename uint16,
          int Len1 = 32,
          int Len2 = 24,
          int TH1 = 1,
          int TH2 = 1>
class GeoIP {
   private:
    // buffer to store the address for the low 16-bit a3a4 in IP
    // index is the high 16-bit a2a2 in IP
    // for each item, the high 32-bit is the accurate number of a3a4.
    // the low 32-bit is the number of a3a4 stored in 512-bit wide data.
    ap_uint<64> ipHigh16Arr[65536];

    // if not find ID, return 0XFFFFFFFF.
    const uint32 Fail = 0xFFFFFFFF;
    const int T = 16;
    const int W = 32;
    const int MSG_BYTE = 8;

    // burst length of index
    // a3a4 is divieded into serval blocks by Delta
    const int Delta = TH2 * Len2;

    // number of fields
    uint16 fieldNum;
    // the postion of sub-string to be extracted from the message.
    uint16 ipPos;

    // get the address from ipHigh16Arr buffer for specified ip
    // return the address in DDR buffer and row number
    void findAddrID(uint32 len,
                    hls::stream<ap_uint<32> >& ipStrm,
                    hls::stream<ap_uint<16> >& ipLow16Strm,
                    hls::stream<ap_uint<64> >& addrStrm,
                    hls::stream<ap_uint<64> >& baseIDStrm) {
    loop_find:
        for (uint32 i = 0; i < len; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 10000 min = 10000
            ap_uint<32> ip = ipStrm.read();
            ap_uint<64> addr, idRange;
            // get 2 values, the difference is the length
            ap_uint<64> tmp1 = ipHigh16Arr[ip(31, 16)];
            ap_uint<64> tmp2 = ipHigh16Arr[ip(31, 16) + 1];
            // a3a4 address in DDR buffer
            addr.range(31, 0) = tmp1(63, 32);
            addr.range(63, 32) = tmp2(63, 32);
            // row number
            idRange.range(31, 0) = tmp1(31, 0);
            idRange.range(63, 32) = tmp2(31, 0);
            // the low 16-bit a1a2 in IP
            ipLow16Strm.write(ip(15, 0));
            // the address for 512-bit wide data
            addrStrm.write(addr);
            // the accurate row number
            baseIDStrm.write(idRange);
        }
    }

    // get the burst read address
    void readHead(uint32 len,
                  uint512* netLow21,
                  hls::stream<ap_uint<64> >& addrInStrm,
                  hls::stream<ap_uint<64> >& baseIDInStrm,
                  hls::stream<ap_uint<64> >& addrOutStrm,
                  hls::stream<ap_uint<64> >& baseIDOutStrm,
                  hls::stream<ap_uint<512> >& headOutStrm) {
    loop_readHead:
        for (uint32 i = 0; i < len; i++) {
#pragma HLS loop_tripcount max = 10000 min = 10000
            ap_uint<64> baseID = baseIDInStrm.read();
            ap_uint<64> addr = addrInStrm.read();
            uint32 delta = addr.range(63, 32) - addr.range(31, 0);
            // if the delta is greater than the burst length TH1
            // then the data is divided into multiple burst reads
            // For this scenario, the a3a4 range is divided into several blocks.
            // store the start of a3a4 of each blocks as index
            // find the right block by comparing with index.
            if (delta > TH1) {
                uint32 begin = addr.range(31, 0);
                // the number of burst number after divided
                uint32 headLen = (baseID(63, 33) - baseID(31, 1) + Delta - 1) / Delta;
                // the 512-bt wide index for reach burst read
                uint32 head512 = (headLen + Len1 - 2) / Len1;
            loop_readHead2:
                for (uint32 j = 0; j < head512; j++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 1 min = 1
                    // burst read the index first
                    headOutStrm.write(netLow21[begin + j]);
                }
                // calculate the real start address of a3a4.
                addr(31, 0) = addr(31, 0) + head512;
            }
            addrOutStrm.write(addr);
            baseIDOutStrm.write(baseID);
        }
    }

    // get the start address for each burst read
    void findChunk(uint32 len,
                   hls::stream<ap_uint<512> >& headInStrm,
                   hls::stream<ap_uint<64> >& addrInStrm,
                   hls::stream<ap_uint<16> >& ipLow16Strm,
                   hls::stream<ap_uint<64> >& baseIDInStrm,
                   hls::stream<ap_uint<64> >& addrOutStrm,
                   hls::stream<ap_uint<16> >& ipLow16OutStrm,
                   hls::stream<ap_uint<32> >& baseIDOutStrm,
                   hls::stream<ap_uint<2> >& idFlagStrm) {
    loop_findChunk:
        for (uint32 i = 0; i < len; i++) {
#pragma HLS loop_tripcount max = 10000 min = 10000
            ap_uint<64> baseID = baseIDInStrm.read();
            ap_uint<64> addr = addrInStrm.read();
            ap_uint<16> ipLow16 = ipLow16Strm.read();
            ipLow16OutStrm.write(ipLow16);
            uint32 delta = addr.range(63, 32) - addr.range(31, 0);
            // for mask without a3a4
            if (baseID[0]) {
                idFlagStrm.write(2);
                addrOutStrm.write(0);
                baseIDOutStrm.write(baseID(31, 1));
                // no a3a4
            } else if (delta == 0) {
                idFlagStrm.write(3);
                addrOutStrm.write(0);
                baseIDOutStrm.write(baseID(31, 1));
                // only one burst
            } else if (delta <= TH1) {
                idFlagStrm.write(0);
                addrOutStrm.write(addr);
                baseIDOutStrm.write(baseID(31, 1));
                // needs multiple burst reads
                // find the burst-read start address
            } else {
                idFlagStrm.write(0);
                uint512 temp;
                uint32 diff = 0;
                uint32 headLen = (baseID(63, 33) - baseID(31, 1) + Delta - 1) / Delta;
            loop_findChunk2:
                for (uint32 j = 0; j < headLen - 1; j++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 1 min = 1
                    uint32 index = j / Len1;
                    uint32 offset = j % Len1;
                    if (offset == 0) temp = headInStrm.read();
                    ap_uint<16> netBegin = temp.range(16 * offset + 15, 16 * offset);
                    if (ipLow16 > netBegin && netBegin > 0) {
                        diff++;
                    }
                }
                addr(31, 0) = addr(31, 0) + diff * TH2;
                if (addr(63, 32) > addr(31, 0) + TH2) addr(63, 32) = addr(31, 0) + TH2;
                addrOutStrm.write(addr);
                baseIDOutStrm.write(baseID(31, 1) + diff * Delta);
            }
        }
    }

    void addrConvert(uint32 len,
                     hls::stream<ap_uint<64> >& addrInStrm,
                     hls::stream<ap_uint<64> >& addrOutStrm,
                     hls::stream<ap_uint<32> >& addrNumStrm,
                     hls::stream<bool>& addrNumEndStrm) {
    loop_addrConvert1:
        for (uint32 i = 0; i < len; i++) {
#pragma HLS loop_tripcount max = 10000 min = 10000
            ap_uint<64> addr = addrInStrm.read();
            addrOutStrm.write(addr);
        loop_addrConvert2:
            for (uint32 j = addr(31, 0); j < addr(63, 32); j++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 10 min = 10
                addrNumStrm.write(j);
                addrNumEndStrm.write(0);
            }
        }
        addrNumEndStrm.write(1);
    }
    // why not use burst read????
    // merge this module with addrConvert
    void readNetLow21(uint512* netLow21,
                      hls::stream<ap_uint<32> >& addrNumStrm,
                      hls::stream<bool>& addrNumEndStrm,
                      hls::stream<ap_uint<512> >& netOutStrm) {
        bool flag = addrNumEndStrm.read();
        bool rd_flag_success = 1;
    loop_readNet:
        while (!flag) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 100000 min = 100000
            if (rd_flag_success) {
                ap_uint<32> addr = addrNumStrm.read();
                netOutStrm.write(netLow21[addr]);
            }
            rd_flag_success = addrNumEndStrm.read_nb(flag);
        }
    }

    // accuratly compare the a3a4 with input and get the ID
    void searchID(uint32 len,
                  hls::stream<ap_uint<512> >& netInStrm,
                  hls::stream<ap_uint<64> >& addrInStrm,
                  hls::stream<ap_uint<16> >& ipLow16Strm,
                  hls::stream<ap_uint<32> >& baseIDInStrm,
                  hls::stream<ap_uint<2> >& idFlagStrm,
                  hls::stream<ap_uint<32> >& idOutStrm) {
    loop_searchID:
        for (uint32 i = 0; i < len; i++) {
#pragma HLS loop_tripcount max = 10000 min = 10000
            ap_uint<64> addr = addrInStrm.read();
            ap_uint<16> ipLow16 = ipLow16Strm.read();
            ap_uint<32> baseID = baseIDInStrm.read();
            ap_uint<2> idFlag = idFlagStrm.read();
            uint32 id = Fail;
            bool isBreakLoop = 1;
            // only 1 a3a4 ??
            if (idFlag == 2) {
                id = baseID;
                // not find
            } else if (idFlag == 3)
                id = Fail;
            else {
                uint512 temp;
            loop_searchID2:
                // compare with each item
                // maybe consider to optimize substract of addr, it appeared in serveral places!!!!
                for (uint32 j = 0; j < addr(63, 32) - addr(31, 0); j++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 10 min = 10
                    temp = netInStrm.read();
                    if (isBreakLoop) {
                    loop_searchID3:
                        for (uint32 k = 0; k < Len2; k++) {
#pragma HLS loop_tripcount max = 10 min = 10
                            ap_uint<32> netBegin = temp.range(21 * k + 15 + 8, 21 * k + 8);
                            ap_uint<32> netB = temp.range(21 * k + 20 + 8, 21 * k + 16 + 8);
                            ap_uint<32> netEnd = netBegin + (1 << (31 - netB));
                            if (ipLow16 >= netBegin && ipLow16 < netEnd && netB > 0) {
                                id = k + baseID + j * Len2; // no consider multi-bank512
                                isBreakLoop = 0;
                            }
                        }
                    }
                }
            }
            idOutStrm.write(id);
        }
    }

    // read data from external DDR/HBM memory
    void arr2Strm(uint32 len, uint32* arrIn, hls::stream<uint32>& outStrm) {
        uint512 tmp;
    loop_arr2Strm:
        for (int i = 0; i < len; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 10000 min = 10000
            outStrm.write(arrIn[i]);
        }
    }

    // write the result to external DDR/HMB memory
    void strm2Arr(uint32 len, hls::stream<uint32>& inStrm, uint32* arrOut) {
    loop_strm2Arr:
        for (uint32 i = 0; i < len; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 10000 min = 10000
            arrOut[i] = inStrm.read();
        }
    }

    // read message from external DDR/HBM memory
    void readMsg2Strm(uint64* msg, hls::stream<uint64>& msgStrm, hls::stream<bool>& msgEndStrm) {
        // the first one store the row number of this buffer
        uint64 len = msg[0] - 1;
        for (uint64 i = 0; i < len; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 10000 min = 10000
            msgStrm.write(msg[i + 1]);
            msgEndStrm.write(0);
        }
        msgEndStrm.write(1);
    }

    // read length of message from external DDR/HBM memory
    void readMsgLen2Strm(uint16* msgLen, hls::stream<uint16>& msgLenStrm) {
        // the first two data store the row number of this buffer.
        uint32 num;
        num(31, 16) = msgLen[0];
        num(15, 0) = msgLen[1];
        num -= 2;
        for (uint32 i = 0; i < num; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 10000 min = 10000
            msgLenStrm.write(msgLen[i + 2]);
            //#ifndef __SYNTHESIS__
            //            std::cout << "msgLen[" << i << "] = " << msgLen[i + 2] << std::endl;
            //#endif
        }
    }

    // read the offset from external DDR/HBM memory and extract specified group
    void readOffset2Strm(uint32* fieldOffset, hls::stream<uint64>& ipOffsetStrm) {
        uint32 fieldSize = fieldOffset[0] - 1;
#ifndef __SYNTHESIS__
        std::cout << "fieldSize  = " << fieldSize << std::endl;
#endif
        uint64 tmp = 0;
        for (uint32 i = 0; i < fieldSize; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 10000 min = 10000
            uint32 fo = fieldOffset[i + 1];
            // the high-32bit store the match/mismatch result.
            if (i % fieldNum == 0) tmp(63, 32) = fo;
            // the low-32bit store offset
            if (i % fieldNum == 2 + ipPos) {
                tmp(31, 0) = fo;
                ipOffsetStrm.write(tmp);
            }
        }
    }

    // get addr of ip in msg
    void readIPAddrInMsg(int len,
                         hls::stream<ap_uint<16> >& msgLenStrm,
                         hls::stream<ap_uint<64> >& ipOffsetStrm,
                         hls::stream<ap_uint<16> >& ipOffsetOutStrm,
                         hls::stream<ap_uint<64> >& ipAddrStrm,
                         hls::stream<bool>& ipAddrEndStrm) {
        ap_uint<64> addr = 0;
        for (int i = 0; i < len; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 10000 min = 10000
            ap_uint<64> tmp = ipOffsetStrm.read();
            ap_uint<16> tmp2 = 0;
            if (tmp(63, 32) == 1) {
                ap_uint<64> begin = tmp(15, 3) + addr;
                ipAddrStrm.write(begin);
                ipAddrEndStrm.write(0);
                ap_uint<8> x0 = tmp(2, 0);
                ap_uint<8> x1 = tmp(31, 16) - tmp(15, 0) + x0;
                tmp2(7, 0) = x0;
                tmp2(15, 8) = x1;
            }
            ipOffsetOutStrm.write(tmp2);
            addr += (msgLenStrm.read() + 7) / 8;
        }
        ipAddrEndStrm.write(1);
    }

    // the IP length is less than 15-byte, so it cross three 64-bit data at most.
    // get 3 continue address
    void msgAddrProcess(hls::stream<ap_uint<64> >& ipAddrStrm,
                        hls::stream<bool>& ipAddrEndStrm,
                        hls::stream<ap_uint<64> >& ipAddr3Strm,
                        hls::stream<bool>& ipAddr3EndStrm) {
        bool flag = ipAddrEndStrm.read();
        bool rd_flag_success = 1;
        while (!flag) {
#pragma HLS pipeline ii = 3
#pragma HLS loop_tripcount max = 10000 min = 10000
            if (rd_flag_success) {
                ap_uint<64> ipAddr = ipAddrStrm.read();
                ipAddr3Strm.write(ipAddr);
                ipAddr3Strm.write(ipAddr + 1);
                ipAddr3Strm.write(ipAddr + 2);
                ipAddr3EndStrm.write(0);
                ipAddr3EndStrm.write(0);
                ipAddr3EndStrm.write(0);
            }
            rd_flag_success = ipAddrEndStrm.read_nb(flag);
        }
        ipAddr3EndStrm.write(1);
    }

    // get ip from msg
    void readIPInMsg(uint64* msgArr,
                     hls::stream<ap_uint<64> >& ipAddr3Strm,
                     hls::stream<bool>& ipAddr3EndStrm,
                     hls::stream<ap_uint<192> >& ipStrStrm) {
        int i = 0;
        ap_uint<192> ipStr;
        bool flag = ipAddr3EndStrm.read();
        bool rd_flag_success = 1;
        while (!flag) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 10000 min = 10000
            if (rd_flag_success) {
                ap_uint<64> addr = ipAddr3Strm.read();
                ipStr((i + 1) * 64 - 1, i * 64) = msgArr[addr + 1];
                if (i == 2) {
                    ipStrStrm.write(ipStr);
                    i = 0;
                } else {
                    i++;
                }
            }
            rd_flag_success = ipAddr3EndStrm.read_nb(flag);
        }
    }

    // get the sub-string based on the start and end offset.
    // the IP length is less than 15-byte, so it cross three 64-bit data at most.
    // this function output the three 64-bit data.
    void getIPStr(hls::stream<uint64>& msgStrm,
                  hls::stream<bool>& msgEndStrm,
                  hls::stream<uint16>& msgLenStrm,
                  hls::stream<uint64>& ipOffsetInStrm,
                  hls::stream<ap_uint<192> >& ipStrStrm,
                  hls::stream<uint32>& ipOffsetOutStrm) {
        uint16 oneLen = 0;
        uint16 begin, end;
        uint64 msg;
        uint32 msgIndex = 0;
        ap_uint<192> ipStr;
        uint32 ipOffset;
        uint16 ipStrOffset = 0;
        while (!msgEndStrm.read()) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 10000 min = 10000
            if (oneLen <= msgIndex) {
                if (ipStrOffset > 0) {
                    ipStrStrm.write(ipStr);
                    ipOffsetOutStrm.write(ipOffset);
                }
                // read length of message
                oneLen = msgLenStrm.read();
                // read offset of sub-string
                uint64 tmp = ipOffsetInStrm.read();
                // mismatch, still read the message from input stream.
                if (tmp(63, 32) == 0) {
                    ipOffset = 0;
                    begin = 0;
                    end = 0;
                    ipStrOffset = 1;
                    // match
                } else {
                    ipOffset = tmp(31, 0);
                    begin = ipOffset(15, 0);
                    end = ipOffset(31, 16);
                    ipStrOffset = 0;
                }
                msgIndex = 0;
            }
            msg = msgStrm.read();
            if ((begin < (msgIndex + 8)) && (end > msgIndex)) {
                ipStr(63 + 64 * ipStrOffset, 64 * ipStrOffset) = msg;
                ipStrOffset++;
            } else if (begin >= (msgIndex + 8)) {
                uint16 low16 = ipOffset(15, 0);
                uint16 high16 = ipOffset(31, 16);
                low16 -= 8;
                high16 -= 8;
                ipOffset(15, 0) = low16;
                ipOffset(31, 16) = high16;
            }
            msgIndex += 8;
        }
        if (ipStrOffset > 0) {
            ipStrStrm.write(ipStr);
            ipOffsetOutStrm.write(ipOffset);
        }
    }

    // convert the IP from string to integer
    // this maybe is the bottleneck of performance.
    // if needs about 15 clock cycles to generate one IP
    void processIP(uint32 len,
                   hls::stream<ap_uint<192> >& ipStrStrm,
                   hls::stream<ap_uint<16> >& ipOffsetStrm,
                   hls::stream<uint32>& ipStrm) {
        ap_uint<192> ipStr;
        ap_uint<16> ipOffset;
        ap_uint<8> begin, end;
        for (uint32 i = 0; i < len; i++) {
#pragma HLS loop_tripcount max = 10000 min = 10000
            ipOffset = ipOffsetStrm.read();
            if (ipOffset > 0) ipStr = ipStrStrm.read();
            begin = ipOffset(7, 0);
            end = ipOffset(15, 8);
            ap_uint<8> ipIndex = 24;
            uint32 ip = 0;
            ap_uint<8> ipTmp = 0;
            for (ap_uint<8> j = begin; j < end; j++) {
#pragma HLS pipeline ii = 1
                ap_uint<8> oneChar = ipStr(7 + 8 * j, 8 * j);
                if (oneChar == '.') {
                    if ((ipIndex <= 24) && (ipTmp < 0x100))
                        ip(7 + ipIndex, ipIndex) = ipTmp;
                    else
                        ip = 0;
                    ipIndex -= 8;
                    ipTmp = 0;
                } else
                    ipTmp = ipTmp * 10 + (oneChar - '0');
            }
            if ((ipIndex == 0) && (ipTmp < 0x100))
                ip(7, 0) = ipTmp;
            else
                ip = 0;
            ipStrm.write(ip);
            //#ifndef __SYNTHESIS__
            //            std::cout << std::hex << "ipStrStrm[" << i << "]=" << ipStr << ", ipOffsetStrm=" << ipOffset
            //            << ",ip=" << ip
            //                      << std::endl;
            //#endif
        }
    }

   public:
    /**
     * @brief GeoIP constructor
     */
    GeoIP() {
#pragma HLS inline
#pragma HLS bind_storage variable = ipHigh16Arr type = ram_t2p impl = uram
#ifndef __SYNTHESIS__
        std::cout << "GeoIP constructor\n";
#endif
    }
    /**
     * brief init Initialize parameters
     * @param len The length of netHigh16, default value is 65536
     * @param netHigh16 The netHigh16 store the position of the lower 21 bits of the network in the DDR
     */
    void init(uint32 len, uint64* netHigh16) {
        uint512 ip, id;
#ifndef __SYNTHESIS__
        std::cout << "GeoIP init start\n";
#endif
    loop_init:
        for (uint32 i = 0; i < len; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 65536 min = 65536
            ipHigh16Arr[i] = netHigh16[i];
        }
#ifndef __SYNTHESIS__
        std::cout << "GeoIP init end\n";
#endif
    }
    /**
     * brief init Initialize parameters
     * @param len The length of netHigh16, default value is 65536
     * @param netHigh16 The netHigh16 store the position of the lower 21 bits of the network in the DDR
     */
    void init(uint32 len, uint64* netHigh16, uint32 ipPosIn) {
        uint512 ip, id;
        // the low 16-bit store total field number
        fieldNum = ipPosIn & 0xFFFF;
        // the high 16-bit store postion of sub-string to be extracted.
        ipPos = (ipPosIn >> 8);
#ifndef __SYNTHESIS__
        std::cout << "GeoIP init start\n";
#endif
    loop_init:
        for (uint32 i = 0; i < len; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 65536 min = 65536
            ipHigh16Arr[i] = netHigh16[i];
        }
#ifndef __SYNTHESIS__
        std::cout << "GeoIP init end\n";
#endif
    }

    /**
     * @brief run GeoIP's core execution module
     * @param len The length of IP
     * @param ipIn The input of 32bit IP
     * @param netLow21 The low 21bit of the network
     * @param net2Low21 In order to achieve faster throughput, the content is the same as netLow21
     * @param idOut Return the id result of the query, 0xFFFFFFFF means the query failed
     */
    void run(uint32 len, uint32* ipIn, uint512* netLow21, uint512* net2Low21, uint32* idOut) {
#ifndef __SYNTHESIS__
        std::cout << "GeoIP run\n";
#endif
        // all the length of stream is set to 32, maybe needs to optimize!!!!
        const int N = 32;
#pragma HLS dataflow
        hls::stream<ap_uint<32> > ipStrm("ipStrm");
#pragma HLS stream variable = ipStrm depth = N
#pragma HLS bind_storage variable = ipStrm type = fifo impl = lutram
        hls::stream<ap_uint<16> > ipLow16Strm("ipLow16Strm");
#pragma HLS stream variable = ipLow16Strm depth = N
#pragma HLS bind_storage variable = ipLow16Strm type = fifo impl = lutram
        hls::stream<ap_uint<64> > addrStrm("addrStrm");
#pragma HLS stream variable = addrStrm depth = N
#pragma HLS bind_storage variable = addrStrm type = fifo impl = lutram
        hls::stream<ap_uint<64> > baseIDStrm("baseIDStrm");
#pragma HLS stream variable = baseIDStrm depth = N
#pragma HLS bind_storage variable = baseIDStrm type = fifo impl = lutram
        hls::stream<ap_uint<64> > addrStrm1("addrStrm1");
#pragma HLS stream variable = addrStrm1 depth = N
#pragma HLS bind_storage variable = addrStrm1 type = fifo impl = lutram
        hls::stream<ap_uint<64> > baseIDStrm1("baseIDStrm1");
#pragma HLS stream variable = baseIDStrm1 depth = N
#pragma HLS bind_storage variable = baseIDStrm1 type = fifo impl = lutram
        hls::stream<ap_uint<16> > ipLow16Strm2("ipLow16Strm2");
#pragma HLS stream variable = ipLow16Strm2 depth = N
        hls::stream<ap_uint<64> > addrStrm2("addrStrm2");
#pragma HLS stream variable = addrStrm2 depth = N
        hls::stream<ap_uint<32> > baseIDStrm2("baseIDStrm2");
#pragma HLS stream variable = baseIDStrm2 depth = N
        hls::stream<ap_uint<2> > idFlagStrm("idFlagStrm");
#pragma HLS stream variable = idFlagStrm depth = N

        hls::stream<ap_uint<64> > addrStrm3("addrStrm3");
#pragma HLS stream variable = addrStrm3 depth = N
#pragma HLS bind_storage variable = addrStrm3 type = fifo impl = lutram
        hls::stream<ap_uint<32> > addrNumStrm("addrNumStrm");
#pragma HLS stream variable = addrNumStrm depth = N
#pragma HLS bind_storage variable = addrNumStrm type = fifo impl = lutram
        hls::stream<bool> addrNumEndStrm("addrNumEndStrm");
#pragma HLS stream variable = addrNumEndStrm depth = N
#pragma HLS bind_storage variable = addrNumEndStrm type = fifo impl = lutram
        hls::stream<ap_uint<512> > netLow21Strm("netLow21Strm");
#pragma HLS stream variable = netLow21Strm depth = N
#pragma HLS bind_storage variable = netLow21Strm type = fifo impl = lutram
        hls::stream<ap_uint<512> > headStrm("headStrm");
#pragma HLS stream variable = headStrm depth = N
#pragma HLS bind_storage variable = headStrm type = fifo impl = lutram
        hls::stream<ap_uint<32> > idStrm("idStrm");
#pragma HLS stream variable = idStrm depth = N
#pragma HLS bind_storage variable = idStrm type = fifo impl = lutram
        // read data from DDR/HBM
        arr2Strm(len, ipIn, ipStrm);

        // get the information from internal buffer
        findAddrID(len, ipStrm, ipLow16Strm, addrStrm, baseIDStrm);

        // find the start address of a2a3 in 512-bit buffer
        readHead(len, netLow21, addrStrm, baseIDStrm, addrStrm1, baseIDStrm1, headStrm);

        // find the start address of burst read
        findChunk(len, headStrm, addrStrm1, ipLow16Strm, baseIDStrm1, addrStrm2, ipLow16Strm2, baseIDStrm2, idFlagStrm);
        // useless???, merge with readNetLow21.
        addrConvert(len, addrStrm2, addrStrm3, addrNumStrm, addrNumEndStrm);
        readNetLow21(net2Low21, addrNumStrm, addrNumEndStrm, netLow21Strm);

        // accuratly compare to find the ID
        searchID(len, netLow21Strm, addrStrm3, ipLow16Strm2, baseIDStrm2, idFlagStrm, idStrm);

        // write result to DDR/HBM
        strm2Arr(len, idStrm, idOut);
    }

    /**
     * @brief extract IP from full message then run GeoIP's core execution module to find the ID.
     * @param len The length of IP
     * @param msg the input message
     * @param msgLen the input length of each message.
     * @param fieldOffset the input offset for each sub-stirng
     * @param netLow21 The low 21bit of the network
     * @param net2Low21 In order to achieve faster throughput, the content is the same as netLow21
     * @param idOut Return the id result of the query, 0xFFFFFFFF means the query failed
     */
    void run(uint32 len,
             uint64* msg,
             uint16* msgLen,
             uint32* fieldOffset,
             uint512* netLow21,
             uint512* net2Low21,
             uint32* idOut) {
#ifndef __SYNTHESIS__
        std::cout << "GeoIP run\n";
#endif
        // setting all length of stream to 32, is it reasonable???? maybe need optimize.
        const int N = 32;
#pragma HLS dataflow
        hls::stream<ap_uint<32> > ipStrm("ipStrm");
#pragma HLS stream variable = ipStrm depth = N
#pragma HLS bind_storage variable = ipStrm type = fifo impl = lutram
        hls::stream<ap_uint<16> > ipLow16Strm("ipLow16Strm");
#pragma HLS stream variable = ipLow16Strm depth = N
#pragma HLS bind_storage variable = ipLow16Strm type = fifo impl = lutram
        hls::stream<ap_uint<64> > addrStrm("addrStrm");
#pragma HLS stream variable = addrStrm depth = N
#pragma HLS bind_storage variable = addrStrm type = fifo impl = lutram
        hls::stream<ap_uint<64> > baseIDStrm("baseIDStrm");
#pragma HLS stream variable = baseIDStrm depth = N
#pragma HLS bind_storage variable = baseIDStrm type = fifo impl = lutram
        hls::stream<ap_uint<64> > addrStrm1("addrStrm1");
#pragma HLS stream variable = addrStrm1 depth = N
#pragma HLS bind_storage variable = addrStrm1 type = fifo impl = lutram
        hls::stream<ap_uint<64> > baseIDStrm1("baseIDStrm1");
#pragma HLS stream variable = baseIDStrm1 depth = N
#pragma HLS bind_storage variable = baseIDStrm1 type = fifo impl = lutram
        hls::stream<ap_uint<16> > ipLow16Strm2("ipLow16Strm2");
#pragma HLS stream variable = ipLow16Strm2 depth = N
        hls::stream<ap_uint<64> > addrStrm2("addrStrm2");
#pragma HLS stream variable = addrStrm2 depth = N
        hls::stream<ap_uint<32> > baseIDStrm2("baseIDStrm2");
#pragma HLS stream variable = baseIDStrm2 depth = N
        hls::stream<ap_uint<2> > idFlagStrm("idFlagStrm");
#pragma HLS stream variable = idFlagStrm depth = N

        hls::stream<ap_uint<64> > addrStrm3("addrStrm3");
#pragma HLS stream variable = addrStrm3 depth = N
#pragma HLS bind_storage variable = addrStrm3 type = fifo impl = lutram
        hls::stream<ap_uint<32> > addrNumStrm("addrNumStrm");
#pragma HLS stream variable = addrNumStrm depth = 1024
#pragma HLS bind_storage variable = addrNumStrm type = fifo impl = lutram
        hls::stream<bool> addrNumEndStrm("addrNumEndStrm");
#pragma HLS stream variable = addrNumEndStrm depth = 1024
#pragma HLS bind_storage variable = addrNumEndStrm type = fifo impl = lutram
        hls::stream<ap_uint<512> > netLow21Strm("netLow21Strm");
#pragma HLS stream variable = netLow21Strm depth = 1024
#pragma HLS bind_storage variable = netLow21Strm type = fifo impl = lutram
        hls::stream<ap_uint<512> > headStrm("headStrm");

#pragma HLS stream variable = headStrm depth = N
#pragma HLS bind_storage variable = headStrm type = fifo impl = lutram
        hls::stream<ap_uint<32> > idStrm("idStrm");
#pragma HLS stream variable = idStrm depth = N
#pragma HLS bind_storage variable = idStrm type = fifo impl = lutram
        // arr2Strm(len, ipIn, ipStrm);
        hls::stream<uint64> msgStrm("msgStrm");
#pragma HLS stream variable = msgStrm depth = 1024
        hls::stream<bool> msgEndStrm("msgEndStrm");
#pragma HLS stream variable = msgEndStrm depth = 1024
        hls::stream<uint16> msgLenStrm("msgLenStrm");
#pragma HLS stream variable = msgLenStrm depth = N
        hls::stream<uint64> ipOffsetStrm("ipOffsetStrm");
#pragma HLS stream variable = ipOffsetStrm depth = N

        hls::stream<ap_uint<192> > ipStrStrm("ipStrStrm");
#pragma HLS stream variable = ipStrStrm depth = N
        //        hls::stream<ap_uint<16> > ipOffset2Strm("ipOffset2Strm");
        //#pragma HLS stream variable = ipOffset2Strm depth = N
        // read message from DDR/HBM
        // readMsg2Strm(msg, msgStrm, msgEndStrm);

        // read length of message from DDR/HBM
        readMsgLen2Strm(msgLen, msgLenStrm);

        // read offset for each sub-string
        readOffset2Strm(fieldOffset, ipOffsetStrm);

        hls::stream<ap_uint<16> > ipOffsetStrm2("ipOffsetStrm2");
#pragma HLS stream variable = ipOffsetStrm2 depth = N
        hls::stream<ap_uint<64> > ipAddrStrm("ipAddrStrm");
#pragma HLS stream variable = ipAddrStrm depth = N
        hls::stream<bool> ipAddrEndStrm("ipAddrEndStrm");
#pragma HLS stream variable = ipAddrEndStrm depth = N
        readIPAddrInMsg(len, msgLenStrm, ipOffsetStrm, ipOffsetStrm2, ipAddrStrm, ipAddrEndStrm);
        hls::stream<ap_uint<64> > ipAddr3Strm("ipAddr3Strm");
#pragma HLS stream variable = ipAddr3Strm depth = N
        hls::stream<bool> ipAddr3EndStrm("ipAddr3EndStrm");
#pragma HLS stream variable = ipAddr3EndStrm depth = N
        msgAddrProcess(ipAddrStrm, ipAddrEndStrm, ipAddr3Strm, ipAddr3EndStrm);
        readIPInMsg(msg, ipAddr3Strm, ipAddr3EndStrm, ipStrStrm);
        // get the sub-string with ap_uint<192> width
        // getIPStr(msgStrm, msgEndStrm, msgLenStrm, ipOffsetStrm, ipStrStrm, ipOffset2Strm);

        // extract the actual IP from ap_uint<192> width sub-string
        processIP(len, ipStrStrm, ipOffsetStrm2, ipStrm);

        // get address information from internal buffer with IP
        findAddrID(len, ipStrm, ipLow16Strm, addrStrm, baseIDStrm);

        // get the address of a3a3 in external DDR buffer with IP
        readHead(len, netLow21, addrStrm, baseIDStrm, addrStrm1, baseIDStrm1, headStrm);

        // get the actual burst address
        findChunk(len, headStrm, addrStrm1, ipLow16Strm, baseIDStrm1, addrStrm2, ipLow16Strm2, baseIDStrm2, idFlagStrm);
        // meaningless, merge with readNetLow21
        addrConvert(len, addrStrm2, addrStrm3, addrNumStrm, addrNumEndStrm);
        readNetLow21(net2Low21, addrNumStrm, addrNumEndStrm, netLow21Strm);

        // accuratly comparison of a3a4 to get ID.
        searchID(len, netLow21Strm, addrStrm3, ipLow16Strm2, baseIDStrm2, idFlagStrm, idStrm);

        // store the result into external DDR/HBM
        strm2Arr(len, idStrm, idOut);
    }
}; // class

} // namespace text
} // namespace data_analytics
} // namespace xf
#endif
