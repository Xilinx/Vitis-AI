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

#ifndef _XF_TEXT_WRITE_JSON_HPP_
#define _XF_TEXT_WRITE_JSON_HPP_

#include "hls_stream.h"
#include "ap_int.h"

namespace xf {
namespace data_analytics {
namespace text {
namespace internal {

template <int W1, int W2, int Flag = 0>
void printStr(ap_uint<W1> len, ap_uint<W2> data) {
#ifndef __SYNTHESIS__
    char e = 0;
    for (ap_uint<W1> k = 0; k < len; k++) {
        e = data(k * 8 + 7, k * 8);
        std::cout << e;
    }
    if (Flag) std::cout << std::endl;
#endif
}

template <int W>
void readArr2Strm(ap_uint<W>* arrIn, hls::stream<ap_uint<W> >& outStrm, hls::stream<bool>& msgEndStrm) {
    ap_uint<64> num = arrIn[0](63, 0);
loop_readArr2Strm:
    for (ap_uint<W> i = 1; i < num; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 10000 min = 10000
        outStrm.write(arrIn[i]);
        msgEndStrm.write(0);
    }
    msgEndStrm.write(1);
}

template <int W>
void readArr2Strm(ap_uint<W> num, ap_uint<W>* arrIn, hls::stream<ap_uint<W> >& outStrm) {
loop_readArr2Strm:
    for (ap_uint<W> i = 0; i < num; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 10000 min = 10000
        outStrm.write(arrIn[i]);
    }
}
// byte number
template <int W, int WM, int BN = 128>
void msgLineProcess(hls::stream<ap_uint<W> >& msgStrm,
                    hls::stream<bool>& msgEndStrm,
                    hls::stream<ap_uint<16> >& msgLenStrm,
                    hls::stream<ap_uint<WM> >& msgLineStrm,
                    hls::stream<ap_uint<16> >& msgLenOutStrm) {
    ap_uint<W> msg;
    ap_uint<16> byteCnt = 0, byteCnt2 = 0;
    ap_uint<WM> msgLine = 0;
    ap_uint<16> oneLen = 0;
loop_msgLineProcess:
    while (!msgEndStrm.read()) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 10000 min = 10000
        if (oneLen <= byteCnt2) {
            if (oneLen > 0) {
                msgLineStrm.write(msgLine);
                // printStr<16, WM, 1>(BN, msgLine);
            }
            oneLen = msgLenStrm.read();
            msgLenOutStrm.write(oneLen);
            byteCnt = 0;
            byteCnt2 = 0;
            msgLine = 0;
        } else if (byteCnt == BN) {
            msgLineStrm.write(msgLine);
            // printStr<16, WM>(BN, msgLine);
            byteCnt = 0;
        }

        // std::cout << "oneLen=" << oneLen << ", byteCnt2=" << byteCnt2 << ", byteCnt=" << byteCnt << std::endl;
        if (oneLen > 0) {
            msgLine((8 + byteCnt) * 8 - 1, byteCnt * 8) = msgStrm.read();
            byteCnt += 8;
            byteCnt2 += 8;
        }
    }
    // std::cout << "oneLen=" << oneLen << ", byteCnt2=" << byteCnt2 << ", byteCnt=" << byteCnt << std::endl;
    if (oneLen <= byteCnt2 && oneLen > 0) {
        msgLineStrm.write(msgLine);
        // printStr<16, WM, 1>(oneLen % BN, msgLine);
    }
}

template <int BN>
void msgPosProcess(int num,
                   int fieldNum,
                   hls::stream<ap_uint<32> >& msgPosInStrm,
                   hls::stream<ap_uint<32> >& msgPosOutStrm,
                   hls::stream<ap_uint<8> >& fldNumStrm,
                   hls::stream<ap_uint<2> >& msgPosFlagStrm) {
    bool grokFlag;
    ap_uint<32> pos, pos2;
    ap_uint<16> begin = 0, end = 0, last_end = 0, delta = 0;
loop_msgPosProcess1:
    for (int i = 0; i < num; i++) {
#pragma HLS loop_tripcount max = 10000 min = 10000
    loop_msgPosProcess2:
        for (int j = 0; j < fieldNum;) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 13 min = 13
            if (j == 0) {
                // std::cout << "msgPosProcess: i=" << i << ", j=" << j << std::endl;
                pos = msgPosInStrm.read();
                grokFlag = pos[0];
                j++;
            } else if (j == 1) {
                // std::cout << "msgPosProcess: i=" << i << ", j=" << j << std::endl;
                pos = msgPosInStrm.read();
                j++;
            } else if (grokFlag) {
                // std::cout << "msgPosProcess: i=" << i << ", j=" << j << ", begin=" << begin << ", end=" << end
                //          << std::endl;
                if (begin + delta >= end) {
                    pos = msgPosInStrm.read();
                    begin = pos(15, 0);
                    end = pos(31, 16);
                } else {
                    begin += delta;
                }
                if (end > last_end + BN && begin < last_end + BN) {
                    fldNumStrm.write(j - 2);
                    delta = last_end + BN - begin;
                    last_end += BN;
                } else if (end > begin + BN) {
                    pos2(15, 0) = begin;
                    pos2(31, 16) = begin + BN;
                    fldNumStrm.write(j - 2);
                    delta = BN;
                    last_end = begin + BN;
                } else {
                    pos2(15, 0) = begin;
                    pos2(31, 16) = end;
                    fldNumStrm.write(j - 2);
                    delta = end - begin;
                    last_end = end;
                    j++;
                }
                pos2(15, 0) = begin;
                pos2(31, 16) = last_end;
                msgPosOutStrm.write(pos2);
                msgPosFlagStrm.write(1);
            } else {
                // std::cout << "msgPosProcess: i=" << i << ", j=" << j << std::endl;
                pos = msgPosInStrm.read();
                msgPosFlagStrm.write(2);
                j++;
            }
        }
        msgPosFlagStrm.write(0);
    }
}

template <int W, int BN = 32>
void msgSplitProcess(int num,
                     hls::stream<ap_uint<W> >& msgLineStrm,
                     hls::stream<ap_uint<16> >& msgLenStrm,
                     hls::stream<ap_uint<32> >& msgPosStrm,
                     hls::stream<ap_uint<8> >& fldNumStrm,
                     hls::stream<ap_uint<2> >& msgPosFlagStrm,
                     hls::stream<ap_uint<W> >& msgSplitStrm,
                     hls::stream<ap_uint<16> >& splitLenStrm,
                     hls::stream<ap_uint<8> >& fldNumOutStrm,
                     hls::stream<bool>& msgSplitEndStrm) {
loop_msgSplitProcess1:
    for (int i = 0; i < num; i++) {
#pragma HLS loop_tripcount max = 10000 min = 10000
        ap_int<16> msgLen = msgLenStrm.read();
        ap_uint<16> begin = 0, end = 0;
        ap_uint<W> msgLine[2];
        msgLine[1] = msgLineStrm.read();
        ap_int<16> msgNowPos = BN;
        ap_uint<2> msgPosFlag = msgPosFlagStrm.read();
    loop_msgSplitProcess2:
        while (msgPosFlag) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 13 min = 13
            if (msgPosFlag == 1) {
                fldNumOutStrm.write(fldNumStrm.read());
                ap_uint<32> msgPos = msgPosStrm.read();
                begin = msgPos(15, 0);
                end = msgPos(31, 16);
#ifndef __SYNTHESIS__
                // printf("msgPosFlag = %d, begin = %d, end = %d\n", (unsigned int)msgPosFlag, (unsigned int) begin,
                // (unsigned int)end);
                assert(end >= begin);
#endif
                ap_uint<16> diff = end - begin;
                if (end > msgNowPos) {
                    msgLine[0] = msgLine[1];
                    msgLine[1] = msgLineStrm.read();
                    msgNowPos += BN;
                }
                ap_uint<W> oneField;
                ap_uint<W> oneField2 = 0;
                if (begin + BN < msgNowPos) {
                    oneField((msgNowPos - BN - begin) * 8 - 1, 0) =
                        msgLine[0](BN * 8 - 1, (begin + 2 * BN - msgNowPos) * 8);
                    oneField(BN * 8 - 1, (msgNowPos - BN - begin) * 8) =
                        msgLine[1]((begin + 2 * BN - msgNowPos) * 8 - 1, 0);
                } else {
                    if (msgNowPos > begin) {
                        oneField((msgNowPos - begin) * 8 - 1, 0) = msgLine[1](BN * 8 - 1, (begin + BN - msgNowPos) * 8);
                    }
                }

                msgSplitStrm.write(oneField);
                splitLenStrm.write(diff);
                msgSplitEndStrm.write(0);
                // std::cout << "begin=" << begin << ",end=" << end << ",msgNowPos=" << msgNowPos << std::endl;
                // printStr<16, W, 1>(end - begin, oneField);
            } else {
                if (msgNowPos < msgLen) {
                    msgLineStrm.read();
                    msgNowPos += BN;
                }
            }
            msgPosFlag = msgPosFlagStrm.read();
        }
        msgSplitEndStrm.write(1);
        while (msgNowPos < msgLen) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 1 min = 1
            msgLineStrm.read();
            msgNowPos += BN;
        }
    }
#ifndef __SYNTHESIS__
    std::cout << "msgSplitProcess Done\n";
#endif
}

// special process "-" -> "\"-\"", After, need optimize based on actual input messages
template <int W = 256, int BN = 32>
void doubleQuoteEscape(int num,
                       hls::stream<ap_uint<W> >& msgSplitStrm,
                       hls::stream<ap_uint<16> >& splitLenStrm,
                       hls::stream<ap_uint<8> >& fldNumStrm,
                       hls::stream<bool>& msgSplitEndStrm,
                       hls::stream<ap_uint<W> >& msgSplitOutStrm,
                       hls::stream<ap_uint<16> >& splitLenOutStrm,
                       hls::stream<ap_uint<8> >& fldNumOutStrm,
                       hls::stream<bool>& msgSplitEndOutStrm) {
loop_doubleQuoteEscape1:
    for (int i = 0; i < num; i++) {
#pragma HLS loop_tripcount max = 10000 min = 10000
        bool msgSplitEnd = msgSplitEndStrm.read();
    loop_doubleQuoteEscape2:
        while (!msgSplitEnd) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 13 min = 13
            ap_uint<W> oneField = msgSplitStrm.read();
            ap_uint<16> diff = splitLenStrm.read();
            ap_uint<8> fldNum = fldNumStrm.read();

            // if (diff >= 2 && diff <= BN - 2) {
            //    if (oneField(7, 0) == 0x22 && (oneField(diff * 8 - 1, diff * 8 - 8) == 0x22)) {
            //        ap_uint<W> oneField2;
            //        oneField2(7, 0) = 0x5C;
            //        oneField2(diff * 8 - 1, 8) = oneField((diff - 1) * 8 - 1, 0);
            //        oneField2((diff + 2) * 8 - 1, diff * 8) = 0x225C;
            //        oneField = oneField2;
            //        diff += 2;
            //    }
            //}

            if (diff == 3 && oneField(7, 0) == 0x22 && (oneField(23, 16) == 0x22)) {
                oneField = 0x225C2D225C;
                diff = 5;
            }

            msgSplitOutStrm.write(oneField);
            splitLenOutStrm.write(diff);
            fldNumOutStrm.write(fldNum);
            msgSplitEndOutStrm.write(0);

            msgSplitEnd = msgSplitEndStrm.read();
        }
        msgSplitEndOutStrm.write(1);
    }
    //#ifndef __SYNTHESIS__
    //  std::cout<<"doubleQuoteEscape Done\n";
    //#endif
}

template <int W, int BN = 32>
void msgKeyValueProcess(int num,
                        hls::stream<ap_uint<W> >& msgSplitStrm,
                        hls::stream<ap_uint<16> >& splitLenStrm,
                        hls::stream<ap_uint<8> >& fldNumStrm,
                        hls::stream<bool>& msgSplitEndStrm,
                        ap_uint<W>* msgFldBuff,
                        ap_uint<16>* msgFldLenBuff,
                        hls::stream<ap_uint<W> >& oneFieldStrm,
                        hls::stream<ap_uint<16> >& oneLenStrm,
                        hls::stream<bool>& oneFieldEndStrm) {
loop_msgKeyValueProcess1:
    for (int i = 0; i < num; i++) {
#pragma HLS loop_tripcount max = 10000 min = 10000
        bool msgSplitEnd = msgSplitEndStrm.read();
        bool readFlag = 1;
        ap_uint<W> msgSplit;
        ap_uint<16> splitLen;
        ap_uint<8> fldNum;
        ap_uint<8> fldNumLast = 255;
        ap_uint<W> msgFld;
        ap_uint<16> msgFldLen;
        ap_uint<2> mergeFlag = 0;
    // std::cout << "msgKeyValueProcess: msgSplitEnd=" << msgSplitEnd << std::endl;
    loop_msgKeyValueProcess2:
        while (!msgSplitEnd) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 13 min = 13
            msgFldLen = 0;
            ap_uint<W> oneLineField;
            ap_uint<W> oneLineLen;
            if (readFlag) {
                msgSplit = msgSplitStrm.read();
                splitLen = splitLenStrm.read();
                // printStr<16, W,1>(splitLen, msgSplit);
                fldNum = fldNumStrm.read();
                if (fldNum != fldNumLast) {
                    fldNumLast = fldNum;
                    msgFldLen = msgFldLenBuff[fldNum];
                    oneLineField = msgFldBuff[fldNum];
                    oneLineField(BN * 8 - 1, msgFldLen * 8) = msgSplit((BN - msgFldLen) * 8 - 1, 0);
                    if (msgFldLen + splitLen > BN)
                        oneLineLen = BN;
                    else
                        oneLineLen = msgFldLen + splitLen;
                } else {
                    oneLineField = msgSplit;
                    oneLineLen = splitLen;
                }
            } else {
                ap_uint<16> fldLenTmp = msgFldLenBuff[fldNum];
                oneLineField(fldLenTmp * 8 - 1, 0) = msgSplit(BN * 8 - 1, (BN - fldLenTmp) * 8);
                oneLineLen = splitLen + fldLenTmp - BN;
            }
            oneFieldStrm.write(oneLineField);
            oneLenStrm.write(oneLineLen);
            oneFieldEndStrm.write(0);
            // std::cout << "oneLineLen=" << oneLineLen << std::endl;
            // printStr<16, W, 1>(oneLineLen, oneLineField);

            // std::cout << "msgFldLen=" << msgFldLen << ", splitLen=" << splitLen << std::endl;
            if (msgFldLen + splitLen <= BN) {
                // std::cout << "msgSplitEnd=" << msgSplitEnd << std::endl;
                msgSplitEnd = msgSplitEndStrm.read();
                readFlag = 1;
            } else {
                readFlag = 0;
            }
        }
        oneFieldEndStrm.write(1);
    }
#ifndef __SYNTHESIS__
    std::cout << "msgKeyValueProcess Done\n";
#endif
}
template <int W = 256, int BN = 32, int BN2 = 5>
void mergeMsgProcess(int num,
                     hls::stream<ap_uint<W> >& oneFieldStrm,
                     hls::stream<ap_uint<16> >& oneLenStrm,
                     hls::stream<bool>& oneFieldEndStrm,
                     hls::stream<ap_uint<W> >& oneLineMsgStrm,
                     hls::stream<ap_uint<16> >& oneLineMsgLenStrm,
                     hls::stream<bool>& oneLineMsgEndStrm) {
loop_mergeMsgProcess1:
    for (int i = 0; i < num; i++) {
#pragma HLS loop_tripcount max = 10000 min = 10000
        ap_uint<BN2> cnt = 0;
        ap_uint<W> oneChunk;
    loop_mergeMsgProcess2:
        while (!oneFieldEndStrm.read()) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 13 min = 13
            ap_uint<W> oneField = oneFieldStrm.read();
            ap_uint<16> oneLen = oneLenStrm.read();
            oneChunk(BN * 8 - 1, cnt * 8) = oneField((BN - cnt) * 8 - 1, 0);
            ap_uint<16> cnt2 = cnt + oneLen;
            if (cnt2 >= BN) {
                oneLineMsgStrm.write(oneChunk);
                oneLineMsgLenStrm.write(BN);
                oneLineMsgEndStrm.write(0);
                if (cnt2 > BN) {
                    oneChunk(cnt * 8 - 1, 0) = oneField(BN * 8 - 1, (BN - cnt) * 8);
                }
            }
            cnt = cnt2;
        }
        if (cnt > 0) {
            oneLineMsgStrm.write(oneChunk);
            oneLineMsgLenStrm.write(cnt);
            oneLineMsgEndStrm.write(0);
        }
        oneLineMsgEndStrm.write(1);
    }
    //#ifndef __SYNTHESIS__
    //  std::cout<<"mergeMsgProcess Done\n";
    //#endif
}

inline void geoLenProcess(int num,
                          hls::stream<ap_uint<32> >& geoPosStrm,
                          ap_uint<64>* geoLenBuff,
                          hls::stream<ap_uint<64> >& geoLenStrm) {
#ifndef __SYNTHESIS__
    std::cout << "geoLenProcess" << std::endl;
#endif
loop_geoLenProcess:
    for (int i = 0; i < num; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 10000 min = 10000
        ap_uint<32> geoPos = geoPosStrm.read();
        //#ifndef __SYNTHESIS__
        //       if(i < 50) printf("geoPos = %d\n", (unsigned int)geoPos);
        //#endif
        if (geoPos == 0xFFFFFFFF)
            geoLenStrm.write(0xFFFFFFFFFFFFFFFF);
        else
            geoLenStrm.write(geoLenBuff[geoPos]);
    }
}

// get geoip addr
template <int BN = 32>
void geoAddrProcess(int num,
                    hls::stream<ap_uint<64> >& geoLenStrm,
                    hls::stream<ap_uint<64> >& geoLenOutStrm,
                    hls::stream<ap_uint<32> >& geoAddrStrm,
                    hls::stream<bool>& geoAddrEndStrm) {
    for (int i = 0; i < num; i++) {
#pragma HLS loop_tripcount max = 10000 min = 10000
        ap_uint<64> geoLen = geoLenStrm.read();
        geoLenOutStrm.write(geoLen);
        ap_uint<32> begin = geoLen(31, 0);
        ap_uint<32> end = geoLen(63, 32);
        //#ifndef __SYNTHESIS__
        //       if(i < 50) printf("begin = %d, end = %d\n", (unsigned int)begin, (unsigned int)end);
        //#endif
        if (begin < end) {
            ap_uint<32> b1 = begin / BN;
            ap_uint<32> e1 = (end + BN - 1) / BN;
            for (int j = 0; j < e1 - b1; j++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 10 min = 10
                geoAddrStrm.write(j + b1);
                geoAddrEndStrm.write(0);
            }
        }
    }
    geoAddrEndStrm.write(1);
}

// according to addr read geoip data
template <int W = 256>
void geoDataProcess(ap_uint<W>* geoBuff,
                    hls::stream<ap_uint<32> >& geoAddrStrm,
                    hls::stream<bool>& geoAddrEndStrm,
                    hls::stream<ap_uint<W> >& geoDataStrm) {
    bool flag = geoAddrEndStrm.read();
    bool rd_flag_success = 1;
    while (!flag) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 10000 min = 10000
        if (rd_flag_success) {
            ap_uint<32> addr = geoAddrStrm.read();
            geoDataStrm.write(geoBuff[addr]);
        }
        rd_flag_success = geoAddrEndStrm.read_nb(flag);
    }
}

template <int W, int K>
void geoLineProcess(int num,
                    hls::stream<ap_uint<64> >& geoLenStrm,
                    hls::stream<ap_uint<W> >& geoDataStrm,
                    hls::stream<ap_uint<W> >& geoChunkStrm,
                    hls::stream<ap_uint<16> >& oneGeoLenStrm,
                    hls::stream<bool>& geoChunkEndStrm) {
    for (int i = 0; i < num; i++) {
#pragma HLS loop_tripcount max = 10000 min = 10000
        ap_uint<64> geoLen = geoLenStrm.read();
        ap_uint<32> begin = geoLen(31, 0);
        ap_uint<32> end = geoLen(63, 32);
        ap_uint<16> oneLen = end - begin;
        ap_uint<W> oneChunk;
        ap_uint<16> oneChunkLen;
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "geoLineProcess: i=" << i << ",begin=" << begin << ",end=" << end << std::endl;
#endif
#endif
        if (begin < end) {
            ap_uint<32> b1 = begin / K;
            ap_uint<32> e1 = (end + K - 1) / K;
            ap_uint<16> b2 = begin - b1 * K;
            ap_uint<16> e2 = end - end / K * K;
            for (int j = 0; j < e1 - b1; j++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 10 min = 10
                // ap_uint<W> geoTmp = geoBuff[j + b1];
                ap_uint<W> geoTmp = geoDataStrm.read();
                if (j == 0) {
                    oneChunk((K - b2) * 8 - 1, 0) = geoTmp(K * 8 - 1, b2 * 8);
                } else {
                    if (oneLen > K) {
                        oneChunkLen = K;
                        oneLen -= K;
                    } else {
                        oneChunkLen = oneLen;
                        oneLen = 0;
                    }
                    if (b2 == 0) {
                        geoChunkStrm.write(oneChunk);
                        // printStr<16, W>(oneChunkLen, oneChunk);
                        oneChunk = geoTmp;
                    } else {
                        oneChunk(K * 8 - 1, (K - b2) * 8) = geoTmp(b2 * 8 - 1, 0);
                        geoChunkStrm.write(oneChunk);
                        // printStr<16, W>(oneChunkLen, oneChunk);
                        oneChunk((K - b2) * 8 - 1, 0) = geoTmp(K * 8 - 1, b2 * 8);
                    }
                    oneGeoLenStrm.write(oneChunkLen);
                    geoChunkEndStrm.write(0);
                }
            }
            if (0 < oneLen) {
                geoChunkStrm.write(oneChunk);
                oneGeoLenStrm.write(oneLen);
                geoChunkEndStrm.write(0);
                // printStr<16, W, 1>(oneLen, oneChunk);
            }
#ifndef __SYNTHESIS__
#ifdef DEBUG
            std::cout << std::endl << "oneLen=" << oneLen << ",e1=" << e1 << ",b1=" << b1 << std::endl;
#endif
#endif
        } else {
            oneChunk = ap_uint<W>("0x0A7D226572756c6961665f70696f6567223a22676174222C22"); //"tag":"geoip_failure"
            geoChunkStrm.write(oneChunk);
            oneGeoLenStrm.write(25);
            geoChunkEndStrm.write(0);
        }
        geoChunkEndStrm.write(1);
    }
    //#ifndef __SYNTHESIS__
    //    std::cout << "geoLineProcess end" << std::endl;
    //#endif
}

template <int W>
void mergeMsgGeoControl(int num,
                        hls::stream<bool>& msgChunkEndStrm,
                        hls::stream<bool>& geoChunkEndStrm,
                        hls::stream<ap_uint<2> >& stateStrm) {
loop_mergeMsgGeoControl1:
    for (int i = 0; i < num; i++) {
#pragma HLS loop_tripcount max = 10000 min = 10000
        bool msgEnd = msgChunkEndStrm.read();
        bool geoEnd = geoChunkEndStrm.read();
        if (msgEnd) {
            stateStrm.write(1);
            geoChunkEndStrm.read();
        } else {
        loop_mergeMsgGeoControl2:
            while (!(msgEnd && geoEnd)) {
#pragma HLS pipeline ii = 1 rewind
#pragma HLS loop_tripcount max = 20 min = 20
                // std::cout << "msgEnd=" << msgEnd << ", geoEnd=" << geoEnd << std::endl;
                if (!msgEnd) {
                    msgEnd = msgChunkEndStrm.read();
                    stateStrm.write(2);
                } else {
                    geoEnd = geoChunkEndStrm.read();
                    stateStrm.write(3);
                }
            }
        }
    }
    stateStrm.write(0);
    //#ifndef __SYNTHESIS__
    //    std::cout << "mergeMsgGeoControl end" << std::endl;
    //#endif
}

template <int W, int BN = 32, int BN2 = 5>
void mergeMsgGeoProcess(hls::stream<ap_uint<W> >& msgChunkStrm,
                        hls::stream<ap_uint<16> >& oneMsgLenStrm,
                        hls::stream<ap_uint<W> >& geoChunkStrm,
                        hls::stream<ap_uint<16> >& oneGeoLenStrm,
                        hls::stream<ap_uint<2> >& stateStrm,
                        hls::stream<ap_uint<W> >& jsonChunkStrm,
                        hls::stream<ap_uint<64> >& jsonLenStrm,
                        hls::stream<bool>& jsonChunkEndStrm) {
    ap_uint<64> totalCnt = 0;
    ap_uint<BN2> cnt = 0;
    ap_uint<W> oneChunk;
    ap_uint<2> state = stateStrm.read();
loop_mergeMsgGeoProcess:
    while (state) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 200000 min = 200000
        ap_uint<W> inChunk;
        ap_uint<16> inChunkLen;
        // std::cout << "msgEnd=" << msgEnd << ", geoEnd=" << geoEnd << std::endl;
        if (state == 1) {
            // oneChunk = 0x0A7D226572756C69;
            inChunk = ap_uint<W>("0x0A7D226572756C6961665f6b6f7267223a22676174227B"); //"tag":"grok_failure"
            inChunkLen = 23;
            totalCnt += inChunkLen;
            oneChunk(BN * 8 - 1, cnt * 8) = inChunk((BN - cnt) * 8 - 1, 0);
            ap_uint<16> cnt2 = cnt + inChunkLen;
            // std::cout << "cnt=" << cnt << ",inChunkLen=" << inChunkLen << ",cnt2=" << cnt2 << std::endl;
            if (cnt2 >= BN) {
                jsonChunkStrm.write(oneChunk);
                jsonChunkEndStrm.write(0);
                // printStr<16, W>(BN, oneChunk);
                if (cnt2 > BN) {
                    oneChunk(cnt * 8 - 1, 0) = inChunk(BN * 8 - 1, (BN - cnt) * 8);
                }
            }
            cnt = cnt2;
            geoChunkStrm.read();
            oneGeoLenStrm.read();
        } else {
            // std::cout << "msgEnd=" << msgEnd << ", geoEnd=" << geoEnd << std::endl;
            if (state == 2) {
                inChunk = msgChunkStrm.read();
                inChunkLen = oneMsgLenStrm.read();
            } else {
                inChunk = geoChunkStrm.read();
                inChunkLen = oneGeoLenStrm.read();
            }
            totalCnt += inChunkLen;
            // std::cout << "BN=" << BN << ",cnt=" << cnt << ",inChunkLen=" << inChunkLen << std::endl;
            oneChunk(BN * 8 - 1, cnt * 8) = inChunk((BN - cnt) * 8 - 1, 0);
            ap_uint<16> cnt2 = cnt + inChunkLen;
            // std::cout << "cnt=" << cnt << ",inChunkLen=" << inChunkLen << ",cnt2=" << cnt2 << std::endl;
            if (cnt2 >= BN) {
                jsonChunkStrm.write(oneChunk);
                jsonChunkEndStrm.write(0);
                // printStr<16, W>(BN, oneChunk);
                if (cnt2 > BN) {
                    oneChunk(cnt * 8 - 1, 0) = inChunk(BN * 8 - 1, (BN - cnt) * 8);
                }
            }
            cnt = cnt2;
        }
        state = stateStrm.read();
    }
    if (cnt > 0) {
        jsonChunkStrm.write(oneChunk);
        jsonChunkEndStrm.write(0);
        // printStr<16, W, 1>(cnt, oneChunk);
    }
    jsonChunkEndStrm.write(1);
    jsonLenStrm.write(totalCnt);
    //#ifndef __SYNTHESIS__
    //    std::cout << "mergeMsgGeoProcess end" << std::endl;
    //#endif
}

template <int W>
void strm2Arr(hls::stream<ap_uint<W> >& outJsonStrm,
              hls::stream<bool>& outJsonEndStrm,
              hls::stream<ap_uint<64> >& jsonLenStrm,
              ap_uint<W>* outJson) {
#ifndef __SYNTHESIS__
    std::cout << "strm2Arr" << std::endl;
#endif
    const int M = 2048;
    int i = 0;
    ap_uint<W> tmp = 0;
    bool flag = outJsonEndStrm.read();
loop_strm2Arr:
    while (!flag) {
#pragma HLS loop_tripcount max = 10000 / M min = 10000 / M
        for (int j = 0; j < M; j++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = M min = M
            if (flag == 0) {
                flag = outJsonEndStrm.read();
                tmp = outJsonStrm.read();
            }
            outJson[i * M + j + 1] = tmp;
        }
        i++;
    }
    outJson[0] = jsonLenStrm.read() + W / 8;
}

} // internal

/**
 * @brief writeJson json format writer
 * @tparam W_IN input data bit-width
 * @tparam W_OUT output data bit-width
 * @tparam BW bit width of W_OUT / 8
 * @param num input massage number
 * @param fieldNum field number
 * @param msgBuff input of messages
 * @param msgLenBuff input of length for each message.
 * @param msgPosBuff each field position in message
 * @param msgFldBuff field name
 * @param msgFldLenBuff field name length
 * @param geoBuff input of geoip data
 * @param geoLenBuff length for each geo
 * @param geoPosBuff position for each geo
 * @param outJson output json format data
 */
template <int W_IN = 64, int W_OUT = 256, int BW = 5>
void writeJson(int num,
               int fieldNum,
               ap_uint<W_IN>* msgBuff,
               ap_uint<16>* msgLenBuff,
               ap_uint<32>* msgPosBuff,
               ap_uint<256>* msgFldBuff,
               ap_uint<16>* msgFldLenBuff,
               ap_uint<256>* geoBuff,
               ap_uint<64>* geoLenBuff,
               ap_uint<32>* geoPosBuff,
               ap_uint<W_OUT>* outJson) {
#pragma HLS dataflow
    const int N = 64;
    const int WM = 256;
    hls::stream<ap_uint<W_IN> > msgStrm("msgStrm");
#pragma HLS stream variable = msgStrm depth = N
    hls::stream<bool> msgEndStrm("msgEndStrm");
#pragma HLS stream variable = msgEndStrm depth = N
    hls::stream<ap_uint<16> > msgLenStrm("msgLenStrm");
#pragma HLS stream variable = msgLenStrm depth = N
    hls::stream<ap_uint<32> > msgPosStrm("msgPosStrm");
#pragma HLS stream variable = msgPosStrm depth = N

    hls::stream<ap_uint<64> > geoLenStrm("geoLenStrm");
#pragma HLS stream variable = geoLenStrm depth = N
    hls::stream<ap_uint<32> > geoPosStrm("geoPosStrm");
#pragma HLS stream variable = geoPosStrm depth = N

    internal::readArr2Strm<W_IN>(msgBuff, msgStrm, msgEndStrm);
    internal::readArr2Strm<16>(num, msgLenBuff, msgLenStrm);
    internal::readArr2Strm<32>(num * fieldNum, msgPosBuff, msgPosStrm);

    hls::stream<ap_uint<8> > fldNumStrm("fldNumStrm");
#pragma HLS stream variable = fldNumStrm depth = N
    hls::stream<ap_uint<32> > msgPosStrm2("msgPosStrm2");
#pragma HLS stream variable = msgPosStrm2 depth = N
    hls::stream<ap_uint<2> > msgPosFlagStrm("msgPosFlagStrm");
#pragma HLS stream variable = msgPosFlagStrm depth = N
    internal::msgPosProcess<32>(num, fieldNum, msgPosStrm, msgPosStrm2, fldNumStrm, msgPosFlagStrm);

    internal::readArr2Strm<32>(num, geoPosBuff, geoPosStrm);
    internal::geoLenProcess(num, geoPosStrm, geoLenBuff, geoLenStrm);
    hls::stream<ap_uint<64> > geoLenStrm2("geoLenStrm2");
#pragma HLS stream variable = geoLenStrm2 depth = N
    hls::stream<ap_uint<32> > geoAddrStrm("geoAddrStrm");
#pragma HLS stream variable = geoAddrStrm depth = N
    hls::stream<bool> geoAddrEndStrm("geoAddrEndStrm");
#pragma HLS stream variable = geoAddrEndStrm depth = N
    internal::geoAddrProcess<WM / 8>(num, geoLenStrm, geoLenStrm2, geoAddrStrm, geoAddrEndStrm);
    hls::stream<ap_uint<WM> > geoDataStrm("geoDataStrm");
#pragma HLS stream variable = geoDataStrm depth = N
    internal::geoDataProcess<WM>(geoBuff, geoAddrStrm, geoAddrEndStrm, geoDataStrm);
    hls::stream<ap_uint<WM> > geoChunkStrm("geoChunkStrm");
#pragma HLS stream variable = geoChunkStrm depth = N
    hls::stream<ap_uint<16> > oneGeoLenStrm("oneGeoLenStrm");
#pragma HLS stream variable = oneGeoLenStrm depth = N
    hls::stream<bool> geoChunkEndStrm("geoChunkEndStrm");
#pragma HLS stream variable = geoChunkEndStrm depth = N
    internal::geoLineProcess<WM, WM / 8>(num, geoLenStrm2, geoDataStrm, geoChunkStrm, oneGeoLenStrm, geoChunkEndStrm);

    hls::stream<ap_uint<WM> > msgLineStrm("msgLineStrm");
#pragma HLS stream variable = msgLineStrm depth = N
    hls::stream<ap_uint<16> > msgLenStrm2("msgLenStrm2");
#pragma HLS stream variable = msgLenStrm2 depth = N
    internal::msgLineProcess<W_IN, WM, WM / 8>(msgStrm, msgEndStrm, msgLenStrm, msgLineStrm, msgLenStrm2);

    hls::stream<ap_uint<WM> > msgSplitStrm("msgSplitStrm");
#pragma HLS stream variable = msgSplitStrm depth = N
    hls::stream<ap_uint<16> > splitLenStrm("splitLenStrm");
#pragma HLS stream variable = splitLenStrm depth = N
    hls::stream<ap_uint<8> > fldNumStrm2("fldNumStrm2");
#pragma HLS stream variable = fldNumStrm2 depth = N
    hls::stream<bool> msgSplitEndStrm("msgSplitEndStrm");
#pragma HLS stream variable = msgSplitEndStrm depth = N
    // split message by field
    internal::msgSplitProcess<WM, WM / 8>(num, msgLineStrm, msgLenStrm2, msgPosStrm2, fldNumStrm, msgPosFlagStrm,
                                          msgSplitStrm, splitLenStrm, fldNumStrm2, msgSplitEndStrm);

    hls::stream<ap_uint<WM> > msgSplitStrm2("msgSplitStrm2");
#pragma HLS stream variable = msgSplitStrm2 depth = N
    hls::stream<ap_uint<16> > splitLenStrm2("splitLenStrm2");
#pragma HLS stream variable = splitLenStrm2 depth = N
    hls::stream<ap_uint<8> > fldNumStrm3("fldNumStrm3");
#pragma HLS stream variable = fldNumStrm3 depth = N
    hls::stream<bool> msgSplitEndStrm2("msgSplitEndStrm2");
#pragma HLS stream variable = msgSplitEndStrm2 depth = N
    internal::doubleQuoteEscape<WM, WM / 8>(num, msgSplitStrm, splitLenStrm, fldNumStrm2, msgSplitEndStrm,
                                            msgSplitStrm2, splitLenStrm2, fldNumStrm3, msgSplitEndStrm2);

    hls::stream<ap_uint<WM> > oneFieldStrm("oneFieldStrm");
#pragma HLS stream variable = oneFieldStrm depth = N
    hls::stream<ap_uint<16> > oneLenStrm("oneLenStrm");
#pragma HLS stream variable = oneLenStrm depth = N
    hls::stream<bool> oneFieldEndStrm("oneFieldEndStrm");
#pragma HLS stream variable = oneFieldEndStrm depth = N
    internal::msgKeyValueProcess<WM, WM / 8>(num, msgSplitStrm2, splitLenStrm2, fldNumStrm3, msgSplitEndStrm2,
                                             msgFldBuff, msgFldLenBuff, oneFieldStrm, oneLenStrm, oneFieldEndStrm);

    hls::stream<ap_uint<WM> > oneLineMsgStrm("oneLineMsgStrm");
#pragma HLS stream variable = oneLineMsgStrm depth = N
    hls::stream<ap_uint<16> > oneLineMsgLenStrm("oneLineMsgLenStrm");
#pragma HLS stream variable = oneLineMsgLenStrm depth = N
    hls::stream<bool> oneLineMsgEndStrm("oneLineMsgEndStrm");
#pragma HLS stream variable = oneLineMsgEndStrm depth = N
    internal::mergeMsgProcess<256, 32, 5>(num, oneFieldStrm, oneLenStrm, oneFieldEndStrm, oneLineMsgStrm,
                                          oneLineMsgLenStrm, oneLineMsgEndStrm);
    hls::stream<ap_uint<WM> > jsonChunkStrm("jsonChunkStrm");
#pragma HLS stream variable = jsonChunkStrm depth = N
    hls::stream<bool> jsonChunkEndStrm("jsonChunkEndStrm");
#pragma HLS stream variable = jsonChunkEndStrm depth = N
    hls::stream<ap_uint<64> > jsonLenStrm("jsonLenStrm");
    hls::stream<ap_uint<2> > stateStrm("stateStrm");
#pragma HLS stream variable = stateStrm depth = N
    internal::mergeMsgGeoControl<WM>(num, oneLineMsgEndStrm, geoChunkEndStrm, stateStrm);
    internal::mergeMsgGeoProcess<WM, WM / 8, BW>(oneLineMsgStrm, oneLineMsgLenStrm, geoChunkStrm, oneGeoLenStrm,
                                                 stateStrm, jsonChunkStrm, jsonLenStrm, jsonChunkEndStrm);

    internal::strm2Arr<W_OUT>(jsonChunkStrm, jsonChunkEndStrm, jsonLenStrm, outJson);
}

} // namespace text
} // data_analytics
} // xf
#endif
