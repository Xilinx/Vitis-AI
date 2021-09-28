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
 * @file shortest_path.hpp
 *
 */

#ifndef __XF_GRAPH_SHORTESTPATH_HPP_
#define __XF_GRAPH_SHORTESTPATH_HPP_

#include "ap_int.h"
#include "hls_stream.h"
#include <stdint.h>

namespace xf {
namespace graph {
namespace internal {

template <typename MType>
union f_cast;

template <>
union f_cast<ap_uint<32> > {
    uint32_t f;
    uint32_t i;
};

template <>
union f_cast<ap_uint<64> > {
    uint64_t f;
    uint64_t i;
};

template <>
union f_cast<int> {
    int f;
    int i;
};

template <>
union f_cast<unsigned int> {
    unsigned int f;
    unsigned int i;
};

template <>
union f_cast<double> {
    double f;
    uint64_t i;
};

template <>
union f_cast<float> {
    float f;
    uint32_t i;
};

namespace sssp {

static bool que_valid;
static ap_uint<512> que_reg;
static ap_uint<32> que_addr;

static bool offset_valid;
static ap_uint<512> offset_reg;
static ap_uint<32> offset_addr;

static bool column_valid;
static ap_uint<512> column_reg;
static ap_uint<32> column_addr;

static bool weight_valid;
static ap_uint<512> weight_reg;
static ap_uint<32> weight_addr;

static bool result_valid;
static ap_uint<512> result_reg;
static ap_uint<32> result_addr;

inline void queCtrl(ap_uint<32> qcfg,
                    ap_uint<32> queStatus[7],
                    ap_uint<512>* ddrQue,

                    hls::stream<ap_uint<32> >& fromIDStrm) {
    ap_uint<32> fromID;
    ap_uint<64> que_rd_cnt;
    ap_uint<64> que_wr_cnt;
    ap_uint<32> que_rd_ptr;

    que_rd_cnt = queStatus[1].concat(queStatus[0]);
    que_wr_cnt = queStatus[3].concat(queStatus[2]);
    que_rd_ptr = queStatus[4];

    if (que_rd_cnt != que_wr_cnt) {
        fromIDStrm.write(1);
        int que_rd_ptrH = que_rd_ptr.range(31, 4);
        int que_rd_ptrL = que_rd_ptr.range(3, 0);
        if (que_valid == 0 || que_rd_ptrH != que_addr) {
            que_reg = ddrQue[que_rd_ptrH];
            que_valid = 1;
            que_addr = que_rd_ptrH;
        }
        fromID = que_reg.range(32 * (que_rd_ptrL + 1) - 1, 32 * que_rd_ptrL);
        que_rd_cnt++;
        if (que_rd_ptr != qcfg - 1)
            que_rd_ptr++;
        else
            que_rd_ptr = 0;
        fromIDStrm.write(fromID);
    } else {
        fromIDStrm.write(0);
    }

    queStatus[0] = que_rd_cnt.range(31, 0);
    queStatus[1] = que_rd_cnt.range(63, 32);
    queStatus[4] = que_rd_ptr;
}

inline void loadoffsetSendAddr(hls::stream<ap_uint<32> >& fromIDInStrm,

                               hls::stream<ap_uint<32> >& fromIDOutStrm,
                               hls::stream<ap_uint<32> >& offsetAddrStrm) {
    ap_uint<32> fromID;
    int ctrl;

    ctrl = fromIDInStrm.read();

    if (ctrl == 1) {
        fromID = fromIDInStrm.read();
        fromIDOutStrm.write(fromID);
        offsetAddrStrm.write(2);
        offsetAddrStrm.write(fromID);
        offsetAddrStrm.write(fromID + 1);
    } else {
        offsetAddrStrm.write(0);
        fromIDOutStrm.write(0);
    }
}

inline void readOffset(hls::stream<ap_uint<32> >& raddrStrm,
                       hls::stream<ap_uint<32> >& rdataStrm,
                       ap_uint<512>* ddrMem) {
    int cnt = raddrStrm.read();
    rdataStrm.write(cnt);

    for (int i = 0; i < cnt; i++) {
#pragma HLS pipeline II = 1
        ap_uint<32> raddr = raddrStrm.read();
        int raddrH = raddr.range(31, 4);
        int raddrL = raddr.range(3, 0);
        if (offset_valid == 0 || raddrH != offset_addr) {
            offset_valid = 1;
            offset_addr = raddrH;
            offset_reg = ddrMem[raddrH];
        }
        rdataStrm.write(offset_reg.range(32 * (raddrL + 1) - 1, 32 * raddrL));
    }
}

inline void loadoffsetCollect(hls::stream<ap_uint<32> >& fromIDInStrm,
                              hls::stream<ap_uint<32> >& offsetDataStrm,

                              hls::stream<ap_uint<32> >& fromIDOutStrm,
                              hls::stream<ap_uint<32> >& offsetLowStrm,
                              hls::stream<ap_uint<32> >& offsetHighStrm) {
    ap_uint<32> fromID;
    fromIDOutStrm.write(fromIDInStrm.read());
    int cnt = offsetDataStrm.read();

    if (cnt == 2) {
        offsetLowStrm.write(offsetDataStrm.read());
        offsetHighStrm.write(offsetDataStrm.read());
    } else {
        offsetLowStrm.write(0);
        offsetHighStrm.write(0);
    }
}

inline void loadToIDWeiSendAddr(hls::stream<ap_uint<32> >& fromIDInStrm,
                                hls::stream<ap_uint<32> >& offsetLowStrm,
                                hls::stream<ap_uint<32> >& offsetHighStrm,

                                hls::stream<ap_uint<32> >& fromIDOutStrm,
                                hls::stream<ap_uint<32> >& columnAddrStrm,
                                hls::stream<ap_uint<32> >& weightAddrStrm) {
    ap_uint<32> fromID;
    bool ctrl;
    ap_uint<32> offsetLow;
    ap_uint<32> offsetHigh;

    fromID = fromIDInStrm.read();
    offsetLow = offsetLowStrm.read();
    offsetHigh = offsetHighStrm.read();
    fromIDOutStrm.write(fromID);

    columnAddrStrm.write(offsetHigh - offsetLow);
    weightAddrStrm.write(offsetHigh - offsetLow);
    for (ap_uint<32> i = offsetLow; i < offsetHigh; i++) {
#pragma HLS PIPELINE II = 1
        columnAddrStrm.write(i);
        weightAddrStrm.write(i);
    }
}

inline void loadToIDWeiSendAddr(hls::stream<ap_uint<32> >& fromIDInStrm,
                                hls::stream<ap_uint<32> >& offsetLowStrm,
                                hls::stream<ap_uint<32> >& offsetHighStrm,

                                hls::stream<ap_uint<32> >& fromIDOutStrm,
                                hls::stream<ap_uint<32> >& columnAddrStrm) {
    ap_uint<32> fromID;
    bool ctrl;
    ap_uint<32> offsetLow;
    ap_uint<32> offsetHigh;

    fromID = fromIDInStrm.read();
    offsetLow = offsetLowStrm.read();
    offsetHigh = offsetHighStrm.read();
    fromIDOutStrm.write(fromID);

    columnAddrStrm.write(offsetHigh - offsetLow);
    for (ap_uint<32> i = offsetLow; i < offsetHigh; i++) {
#pragma HLS PIPELINE II = 1
        columnAddrStrm.write(i);
    }
}

inline void readColumn(hls::stream<ap_uint<32> >& raddrStrm,
                       hls::stream<ap_uint<32> >& rdataStrm,
                       ap_uint<512>* ddrMem) {
    int cnt = raddrStrm.read();
    rdataStrm.write(cnt);
    for (int i = 0; i < cnt; i++) {
#pragma HLS pipeline II = 1
        ap_uint<32> raddr = raddrStrm.read();
        int raddrH = raddr.range(31, 4);
        int raddrL = raddr.range(3, 0);
        if (column_valid == 0 || raddrH != column_addr) {
            column_valid = 1;
            column_addr = raddrH;
            column_reg = ddrMem[raddrH];
        }
        rdataStrm.write(column_reg.range(32 * (raddrL + 1) - 1, 32 * raddrL));
    }
}

template <int WIDTH>
inline void readWeight(bool enable,
                       int type, // 0 for float, 1 for fixed
                       hls::stream<ap_uint<32> >& raddrStrm,
                       hls::stream<ap_uint<WIDTH> >& rdataStrm,
                       ap_uint<512>* ddrMem) {
    int cnt = raddrStrm.read();
    rdataStrm.write(cnt);
    for (int i = 0; i < cnt; i++) {
#pragma HLS pipeline II = 1
        ap_uint<32> raddr = raddrStrm.read();
        if (WIDTH == 32) {
            if (enable) {
                int raddrH = raddr.range(31, 4);
                int raddrL = raddr.range(3, 0);
                if (weight_valid == 0 || raddrH != weight_addr) {
                    weight_valid = 1;
                    weight_addr = raddrH;
                    weight_reg = ddrMem[raddrH];
                }
                rdataStrm.write(weight_reg.range(32 * (raddrL + 1) - 1, 32 * raddrL));
            } else {
                if (type == 0) {
                    f_cast<float> tmp;
                    tmp.f = 1.0;
                    rdataStrm.write(tmp.i);
                } else {
                    rdataStrm.write(1);
                }
            }
        } else if (WIDTH == 64) {
            if (enable) {
                int raddrH = raddr.range(31, 3);
                int raddrL = raddr.range(2, 0);
                if (weight_valid == 0 || raddrH != weight_addr) {
                    weight_valid = 1;
                    weight_addr = raddrH;
                    weight_reg = ddrMem[raddrH];
                }
                rdataStrm.write(weight_reg.range(64 * (raddrL + 1) - 1, 64 * raddrL));
            } else {
                if (type == 0) {
                    f_cast<double> tmp;
                    tmp.f = 1.0;
                    rdataStrm.write(tmp.i);
                } else {
                    rdataStrm.write(1);
                }
            }
        }
    }
}

template <int WIDTH>
void loadResSendAddr(hls::stream<ap_uint<32> >& fromIDInStrm,
                     hls::stream<ap_uint<32> >& columnDataStrm,
                     hls::stream<ap_uint<WIDTH> >& weightDataStrm,

                     hls::stream<ap_uint<32> >& resAddrStrm,
                     hls::stream<ap_uint<32> >& toIDOutStrm,
                     hls::stream<ap_uint<WIDTH> >& weiOutStrm) {
    int cnt = columnDataStrm.read();
    weightDataStrm.read();
    ap_uint<32> fromID = fromIDInStrm.read();
    resAddrStrm.write(cnt + 1);
    resAddrStrm.write(fromID);
    toIDOutStrm.write(fromID);
    for (int i = 0; i < cnt; i++) {
#pragma HLS pipeline II = 1
        ap_uint<32> toID = columnDataStrm.read();
        resAddrStrm.write(toID);
        toIDOutStrm.write(toID);
        weiOutStrm.write(weightDataStrm.read());
    }
}

template <int WIDTH>
void readResult(hls::stream<ap_uint<32> >& raddrStrm, hls::stream<ap_uint<WIDTH> >& rdataStrm, ap_uint<512>* ddrMem) {
    int cnt = raddrStrm.read();
    rdataStrm.write(cnt);
    for (int i = 0; i < cnt; i++) {
#pragma HLS pipeline II = 1
        ap_uint<32> raddr = raddrStrm.read();
        if (WIDTH == 32) {
            int raddrH = raddr.range(31, 4);
            int raddrL = raddr.range(3, 0);
            if (result_valid == 0 || raddrH != result_addr) {
                result_valid = 1;
                result_addr = raddrH;
                result_reg = ddrMem[raddrH];
            }
            rdataStrm.write(result_reg.range(32 * (raddrL + 1) - 1, 32 * raddrL));
        } else if (WIDTH == 64) {
            int raddrH = raddr.range(31, 3);
            int raddrL = raddr.range(2, 0);
            if (result_valid == 0 || raddrH != result_addr) {
                result_valid = 1;
                result_addr = raddrH;
                result_reg = ddrMem[raddrH];
            }
            rdataStrm.write(result_reg.range(64 * (raddrL + 1) - 1, 64 * raddrL));
        }
    }
}

template <int WIDTH>
void loadResCollectData(int type, // 0 for float, 1 for fixed
                        hls::stream<ap_uint<WIDTH> >& dataStrm,
                        hls::stream<ap_uint<32> >& toIDStrm,
                        hls::stream<ap_uint<WIDTH> >& weiStrm,

                        ap_uint<32> tableStatus[3],
                        ap_uint<32>* toIDTable,
                        ap_uint<WIDTH>* distTable) {
    int cnt = dataStrm.read();
    tableStatus[2] = toIDStrm.read();

    if (WIDTH == 32) {
        ap_uint<32> reg = dataStrm.read();
        int addr = 0;
        for (int i = 0; i < cnt - 1; i++) {
#pragma HLS pipeline II = 1
            if (type == 0) {
                f_cast<float> fromDist;
                fromDist.i = reg;
                f_cast<float> weight;
                weight.i = weiStrm.read();
                f_cast<float> toDist;
                toDist.i = dataStrm.read();
                f_cast<float> curDist;
                curDist.f = fromDist.f + weight.f;
                if (curDist.f < toDist.f) {
                    toIDTable[addr] = toIDStrm.read();
                    distTable[addr] = curDist.i;
                    addr++;
                } else {
                    toIDStrm.read();
                }
            } else {
                ap_uint<32> fromDist;
                fromDist = reg;
                ap_uint<32> weight;
                weight = weiStrm.read();
                ap_uint<32> toDist;
                toDist = dataStrm.read();
                ap_uint<32> curDist = fromDist + weight;
                if (curDist < toDist) {
                    toIDTable[addr] = toIDStrm.read();
                    distTable[addr] = curDist;
                    addr++;
                } else {
                    toIDStrm.read();
                }
            }
        }
        tableStatus[0] = addr;
    } else if (WIDTH == 64) {
        ap_uint<64> reg = dataStrm.read();
        int addr = 0;
        for (int i = 0; i < cnt - 1; i++) {
#pragma HLS pipeline II = 1
            if (type == 0) {
                f_cast<double> fromDist;
                fromDist.i = reg;
                f_cast<double> weight;
                weight.i = weiStrm.read();
                f_cast<double> toDist;
                toDist.i = dataStrm.read();
                f_cast<double> curDist;
                curDist.f = fromDist.f + weight.f;
                if (curDist.f < toDist.f) {
                    toIDTable[addr] = toIDStrm.read();
                    distTable[addr] = curDist.i;
                    addr++;
                } else {
                    toIDStrm.read();
                }
            } else {
                ap_uint<64> fromDist;
                fromDist = reg;
                ap_uint<64> weight;
                weight = weiStrm.read();
                ap_uint<64> toDist;
                toDist = dataStrm.read();
                ap_uint<64> curDist = fromDist + weight;
                if (curDist < toDist) {
                    toIDTable[addr] = toIDStrm.read();
                    distTable[addr] = curDist;
                    addr++;
                } else {
                    toIDStrm.read();
                }
            }
        }
        tableStatus[0] = addr;
    }
}

template <int WIDTH>
void shortestPathS1Dataflow(bool enaWei,
                            int type, // 0 for float, 1 for fixed
                            ap_uint<32> qcfg,
                            ap_uint<32> queStatus[7],
                            ap_uint<512>* ddrQue,
                            ap_uint<512>* offset,
                            ap_uint<512>* column,
                            ap_uint<512>* weight,
                            ap_uint<512>* result512,
                            ap_uint<32> tableStatus[3],
                            ap_uint<32>* toIDTable,
                            ap_uint<WIDTH>* distTable) {
#pragma HLS dataflow
    hls::stream<ap_uint<32> > fromIDQueStrm;
#pragma HLS stream variable = fromIDQueStrm depth = 32
#pragma HLS resource variable = fromIDQueStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > fromIDStrm2;
#pragma HLS stream variable = fromIDStrm2 depth = 32
#pragma HLS resource variable = fromIDStrm2 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > offsetLowStrm;
#pragma HLS stream variable = offsetLowStrm depth = 32
#pragma HLS resource variable = offsetLowStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > offsetHighStrm;
#pragma HLS stream variable = offsetHighStrm depth = 32
#pragma HLS resource variable = offsetHighStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > resAddrStrm;
#pragma HLS stream variable = resAddrStrm depth = 32
#pragma HLS resource variable = resAddrStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<WIDTH> > resDataStrm;
#pragma HLS stream variable = resDataStrm depth = 32
#pragma HLS resource variable = resDataStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > toIDStrm3;
#pragma HLS stream variable = toIDStrm3 depth = 1024
#pragma HLS resource variable = toIDStrm3 core = FIFO_BRAM

    hls::stream<ap_uint<WIDTH> > weiStrm2;
#pragma HLS stream variable = weiStrm2 depth = 1024
#pragma HLS resource variable = weiStrm2 core = FIFO_BRAM

    hls::stream<ap_uint<32> > fromIDoffsetStrm;
#pragma HLS stream variable = fromIDoffsetStrm depth = 32
#pragma HLS resource variable = fromIDoffsetStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > offsetAddrStrm;
#pragma HLS stream variable = offsetAddrStrm depth = 32
#pragma HLS resource variable = offsetAddrStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > offsetDataStrm;
#pragma HLS stream variable = offsetDataStrm depth = 32
#pragma HLS resource variable = offsetDataStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > fromIDweiStrm;
#pragma HLS stream variable = fromIDweiStrm depth = 32
#pragma HLS resource variable = fromIDweiStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > columnAddrStrm;
#pragma HLS stream variable = columnAddrStrm depth = 32
#pragma HLS resource variable = columnAddrStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > columnDataStrm;
#pragma HLS stream variable = columnDataStrm depth = 1024
#pragma HLS resource variable = columnDataStrm core = FIFO_BRAM

    hls::stream<ap_uint<32> > weightAddrStrm;
#pragma HLS stream variable = weightAddrStrm depth = 32
#pragma HLS resource variable = weightAddrStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<WIDTH> > weightDataStrm;
#pragma HLS stream variable = weightDataStrm depth = 1024
#pragma HLS resource variable = weightDataStrm core = FIFO_BRAM

    queCtrl(qcfg, queStatus, ddrQue, fromIDQueStrm);

    loadoffsetSendAddr(fromIDQueStrm,

                       fromIDoffsetStrm, offsetAddrStrm);

    readOffset(offsetAddrStrm, offsetDataStrm, offset);

    loadoffsetCollect(fromIDoffsetStrm, offsetDataStrm,

                      fromIDStrm2, offsetLowStrm, offsetHighStrm);

    loadToIDWeiSendAddr(fromIDStrm2, offsetLowStrm, offsetHighStrm,

                        fromIDweiStrm, columnAddrStrm, weightAddrStrm);

    readColumn(columnAddrStrm, columnDataStrm, column);

    readWeight<WIDTH>(enaWei, type, weightAddrStrm, weightDataStrm, weight);

    loadResSendAddr(fromIDweiStrm, columnDataStrm, weightDataStrm,

                    resAddrStrm, toIDStrm3, weiStrm2);

    readResult<WIDTH>(resAddrStrm, resDataStrm, result512);

    loadResCollectData<WIDTH>(type, resDataStrm, toIDStrm3, weiStrm2, tableStatus, toIDTable, distTable);
}

template <int WIDTH, int MAXOUTDEGREE>
void updateDispatch(ap_uint<32> tableStatus[3],
                    ap_uint<32>* toIDTable,
                    ap_uint<WIDTH>* distTable,

                    hls::stream<ap_uint<32> >& resToIDStrm,
                    hls::stream<ap_uint<WIDTH> >& resDistStrm,
                    hls::stream<ap_uint<32> >& queToIDStrm) {
    ap_uint<32> tableCnt = tableStatus[0];
    if (tableCnt > MAXOUTDEGREE) tableStatus[1] = 1;
    resToIDStrm.write(tableCnt);
    resDistStrm.write(tableStatus[2]);
    queToIDStrm.write(tableCnt);
    for (int i = 0; i < tableCnt; i++) {
#pragma HLS PIPELINE II = 1
        if (i < MAXOUTDEGREE) {
            ap_uint<32> reg_toID = toIDTable[i];
            resToIDStrm.write(reg_toID);
            resDistStrm.write(distTable[i]);
            queToIDStrm.write(reg_toID);
        } else {
            resToIDStrm.write(0);
            resDistStrm.write(0);
            queToIDStrm.write(0);
        }
    }
}

inline void upadateQue(ap_uint<32> qcfg,
                       hls::stream<ap_uint<32> >& queToIDStrm,

                       ap_uint<32> queStatus[7],
                       ap_uint<32>* ddrQue) {
    ap_uint<64> que_rd_cnt;
    ap_uint<64> que_wr_cnt;
    ap_uint<32> que_wr_ptr;

    que_rd_cnt = queStatus[1].concat(queStatus[0]);
    que_wr_cnt = queStatus[3].concat(queStatus[2]);
    que_wr_ptr = queStatus[5];

    int cnt = queToIDStrm.read();
    for (int i = 0; i < cnt; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<32> toID = queToIDStrm.read();
        ddrQue[que_wr_ptr] = toID;
        int que_wr_ptrH = que_wr_ptr.range(31, 4);
        int que_wr_ptrL = que_wr_ptr.range(3, 0);
        if (que_wr_ptrH == que_addr) {
            que_valid = 0;
        }

        if (que_wr_ptr != qcfg - 1)
            que_wr_ptr++;
        else
            que_wr_ptr = 0;
        que_wr_cnt++;
        if (que_wr_cnt - que_rd_cnt > qcfg - 1) queStatus[6] = 1;
    }

    queStatus[2] = que_wr_cnt.range(31, 0);
    queStatus[3] = que_wr_cnt.range(63, 32);
    queStatus[5] = que_wr_ptr;
}

template <int WIDTH>
void initRes(ap_uint<32> numVertex, ap_uint<WIDTH> maxValue, ap_uint<32> sourceID, ap_uint<512>* result512) {
    ap_uint<512> max512;
    ap_uint<32> each;
    if (WIDTH == 32) {
        each = (numVertex * 4 + 4095) / 4096;
        for (int i = 0; i < 16; i++) {
#pragma HLS unroll
            max512.range((i + 1) * 32 - 1, i * 32) = maxValue;
        }

        for (int j = 0; j < each; j++) {
            for (int i = 0; i < 64; i++) {
#pragma HLS pipeline II = 1
                result512[j * 64 + i] = max512;
            }
        }

        ap_uint<32> sourceIDH = sourceID(31, 4);
        ap_uint<32> sourceIDL = sourceID(3, 0);
        max512.range((sourceIDL + 1) * 32 - 1, sourceIDL * 32) = 0;
        result512[sourceIDH] = max512;

    } else if (WIDTH == 64) {
        each = (numVertex * 8 + 4095) / 4096;
        for (int i = 0; i < 8; i++) {
#pragma HLS unroll
            max512.range((i + 1) * 64 - 1, i * 64) = maxValue;
        }

        for (int j = 0; j < each; j++) {
            for (int i = 0; i < 64; i++) {
#pragma HLS pipeline II = 1
                result512[j * 64 + i] = max512;
            }
        }

        ap_uint<32> sourceIDH = sourceID(31, 3);
        ap_uint<32> sourceIDL = sourceID(2, 0);
        max512.range((sourceIDL + 1) * 64 - 1, sourceIDL * 64) = 0;
        result512[sourceIDH] = max512;
    }
}

namespace pred {

template <int WIDTH>
void updateRes(bool enaPred,
               hls::stream<ap_uint<32> >& resToIDStrm,
               hls::stream<ap_uint<WIDTH> >& resDistStrm,
               ap_uint<WIDTH>* result,
               ap_uint<32>* pred) {
    int cnt = resToIDStrm.read();
    ap_uint<32> fromID = resDistStrm.read();
    for (int i = 0; i < cnt; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<32> toID = resToIDStrm.read();
        ap_uint<32> toID512 = toID.range(31, 4);

        if (toID512 == result_addr) result_valid = 0;

        result[toID] = resDistStrm.read();
        if (enaPred) pred[toID] = fromID;
    }
}

template <int WIDTH, int MAXOUTDEGREE>
void shortestPathS2Dataflow(bool enaPred,
                            ap_uint<32> qcfg,
                            ap_uint<32> tableStatus[3],
                            ap_uint<32>* toIDTable,
                            ap_uint<WIDTH>* distTable,

                            ap_uint<WIDTH>* result,
                            ap_uint<32>* pred,

                            ap_uint<32> queStatus[7],
                            ap_uint<32>* ddrQue) {
#pragma HLS dataflow
    hls::stream<ap_uint<32> > resToIDStrm;
#pragma HLS stream variable = resToIDStrm depth = 32
#pragma HLS resource variable = resToIDStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<WIDTH> > resDistStrm;
#pragma HLS stream variable = resDistStrm depth = 32
#pragma HLS resource variable = resDistStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > queToIDStrm;
#pragma HLS stream variable = queToIDStrm depth = 32
#pragma HLS resource variable = queToIDStrm core = FIFO_LUTRAM

    updateDispatch<WIDTH, MAXOUTDEGREE>(tableStatus, toIDTable, distTable, resToIDStrm, resDistStrm, queToIDStrm);
    updateRes(enaPred, resToIDStrm, resDistStrm, result, pred);
    upadateQue(qcfg, queToIDStrm, queStatus, ddrQue);
}

inline void initPred(bool enaPred, ap_uint<32> numVertex, ap_uint<32> sourceID, ap_uint<512>* pred512) {
    ap_uint<512> max512 = -1;
    ap_uint<32> each;

    each = (numVertex * 4 + 4095) / 4096;
    for (int j = 0; j < each; j++) {
        for (int i = 0; i < 64; i++) {
#pragma HLS pipeline II = 1
            if (enaPred) pred512[j * 64 + i] = -1;
        }
    }

    ap_uint<32> sourceIDH = sourceID(31, 4);
    ap_uint<32> sourceIDL = sourceID(3, 0);
    max512.range((sourceIDL + 1) * 32 - 1, sourceIDL * 32) = sourceID;
    if (enaPred) pred512[sourceIDH] = max512;
}

template <int WIDTH>
void init(bool enaPred,
          ap_uint<32> numVertex,
          ap_uint<WIDTH> maxValue,
          ap_uint<32> sourceID,
          ap_uint<512>* result512,
          ap_uint<512>* pred512) {
    initRes<WIDTH>(numVertex, maxValue, sourceID, result512);
    initPred(enaPred, numVertex, sourceID, pred512);
}

template <int WIDTH, int MAXOUTDEGREE>
void shortestPathInner(ap_uint<32>* config,
                       ap_uint<512>* offset,
                       ap_uint<512>* column,
                       ap_uint<512>* weight,

                       ap_uint<512>* ddrQue512,
                       ap_uint<32>* ddrQue,

                       ap_uint<512>* result512,
                       ap_uint<WIDTH>* result,
                       ap_uint<512>* pred512,
                       ap_uint<32>* pred,
                       ap_uint<8>* info) {
    ap_uint<32> tableStatus[3];
#pragma HLS ARRAY_PARTITION variable = tableStatus complete dim = 1

#ifndef __SYNTHESIS__
    ap_uint<32>* toIDTable = (ap_uint<32>*)malloc(MAXOUTDEGREE * sizeof(ap_uint<32>));
    ap_uint<WIDTH>* distTable = (ap_uint<WIDTH>*)malloc(MAXOUTDEGREE * sizeof(ap_uint<WIDTH>));
#else
    ap_uint<32> toIDTable[MAXOUTDEGREE];
#pragma HLS RESOURCE variable = toIDTable core = RAM_S2P_URAM

    ap_uint<WIDTH> distTable[MAXOUTDEGREE];
#pragma HLS RESOURCE variable = distTable core = RAM_S2P_URAM
#endif

    offset_valid = 0;
    column_valid = 0;
    weight_valid = 0;
    result_valid = 0;
    que_valid = 0;

    tableStatus[0] = 0;
    tableStatus[1] = 0;
    tableStatus[2] = 0;

    ap_uint<32> queStatus[7];
#pragma HLS ARRAY_PARTITION variable = queStatus complete dim = 1

    queStatus[0] = 0;
    queStatus[1] = 0;
    queStatus[2] = 1;
    queStatus[3] = 0;
    queStatus[4] = 0;
    queStatus[5] = 1;
    queStatus[6] = 0;

    ap_uint<64> que_rd_cnt;
    ap_uint<64> que_wr_cnt;
    ap_uint<32> que_rd_ptr;

    que_rd_cnt = queStatus[1].concat(queStatus[0]);
    que_wr_cnt = queStatus[3].concat(queStatus[2]);

    ap_uint<32> sourceID = config[5];
    ddrQue[0] = sourceID;
    ap_uint<32> numVertex = config[0];
    ap_uint<WIDTH> maxValue;
    if (WIDTH == 32) {
        maxValue = config[1];
    } else if (WIDTH == 64) {
        maxValue = config[2].concat(config[1]);
    }
    ap_uint<32> qcfg = config[3];
    ap_uint<32> cmd = config[4];
    bool enaWei = cmd.get_bit(0);
    bool enaPred = cmd.get_bit(1);
    int type = cmd.get_bit(2);

    init(enaPred, numVertex, maxValue, sourceID, result512, pred512);
    while (que_rd_cnt != que_wr_cnt) {
        shortestPathS1Dataflow<WIDTH>(enaWei, type, qcfg, queStatus, ddrQue512, offset, column, weight, result512,
                                      tableStatus, toIDTable, distTable);
        shortestPathS2Dataflow<WIDTH, MAXOUTDEGREE>(enaPred, qcfg, tableStatus, toIDTable, distTable, result, pred,
                                                    queStatus, ddrQue);

        que_rd_cnt = queStatus[1].concat(queStatus[0]);
        que_wr_cnt = queStatus[3].concat(queStatus[2]);
    }

    info[0] = queStatus[6];
    info[1] = tableStatus[1];
}

} // namespace pred

namespace nopred {

template <int WIDTH>
void updateRes(hls::stream<ap_uint<32> >& resToIDStrm,
               hls::stream<ap_uint<WIDTH> >& resDistStrm,
               ap_uint<WIDTH>* result) {
    int cnt = resToIDStrm.read();
    ap_uint<32> fromID = resDistStrm.read();
    for (int i = 0; i < cnt; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<32> toID = resToIDStrm.read();
        ap_uint<32> toID512 = toID.range(31, 4);

        if (toID512 == result_addr) result_valid = 0;

        result[toID] = resDistStrm.read();
    }
}

template <int WIDTH, int MAXOUTDEGREE>
void shortestPathS2Dataflow(ap_uint<32> qcfg,
                            ap_uint<32> tableStatus[3],
                            ap_uint<32>* toIDTable,
                            ap_uint<WIDTH>* distTable,

                            ap_uint<WIDTH>* result,

                            ap_uint<32> queStatus[7],
                            ap_uint<32>* ddrQue) {
#pragma HLS dataflow
    hls::stream<ap_uint<32> > resToIDStrm;
#pragma HLS stream variable = resToIDStrm depth = 32
#pragma HLS resource variable = resToIDStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<WIDTH> > resDistStrm;
#pragma HLS stream variable = resDistStrm depth = 32
#pragma HLS resource variable = resDistStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > queToIDStrm;
#pragma HLS stream variable = queToIDStrm depth = 32
#pragma HLS resource variable = queToIDStrm core = FIFO_LUTRAM

    updateDispatch<WIDTH, MAXOUTDEGREE>(tableStatus, toIDTable, distTable, resToIDStrm, resDistStrm, queToIDStrm);
    updateRes(resToIDStrm, resDistStrm, result);
    upadateQue(qcfg, queToIDStrm, queStatus, ddrQue);
}

template <int WIDTH, int MAXOUTDEGREE>
void shortestPathInner(ap_uint<32>* config,
                       ap_uint<512>* offset,
                       ap_uint<512>* column,
                       ap_uint<512>* weight,

                       ap_uint<512>* ddrQue512,
                       ap_uint<32>* ddrQue,

                       ap_uint<512>* result512,
                       ap_uint<WIDTH>* result,
                       ap_uint<8>* info) {
    ap_uint<32> tableStatus[3];
#pragma HLS ARRAY_PARTITION variable = tableStatus complete dim = 1

#ifndef __SYNTHESIS__
    ap_uint<32>* toIDTable = (ap_uint<32>*)malloc(MAXOUTDEGREE * sizeof(ap_uint<32>));
    ap_uint<WIDTH>* distTable = (ap_uint<WIDTH>*)malloc(MAXOUTDEGREE * sizeof(ap_uint<WIDTH>));
#else
    ap_uint<32> toIDTable[MAXOUTDEGREE];
#pragma HLS RESOURCE variable = toIDTable core = RAM_S2P_URAM

    ap_uint<WIDTH> distTable[MAXOUTDEGREE];
#pragma HLS RESOURCE variable = distTable core = RAM_S2P_URAM
#endif

    offset_valid = 0;
    column_valid = 0;
    weight_valid = 0;
    result_valid = 0;
    que_valid = 0;

    tableStatus[0] = 0;
    tableStatus[1] = 0;
    tableStatus[2] = 0;

    ap_uint<32> queStatus[7];
#pragma HLS ARRAY_PARTITION variable = queStatus complete dim = 1

    queStatus[0] = 0;
    queStatus[1] = 0;
    queStatus[2] = 1;
    queStatus[3] = 0;
    queStatus[4] = 0;
    queStatus[5] = 1;
    queStatus[6] = 0;

    ap_uint<64> que_rd_cnt;
    ap_uint<64> que_wr_cnt;
    ap_uint<32> que_rd_ptr;

    que_rd_cnt = queStatus[1].concat(queStatus[0]);
    que_wr_cnt = queStatus[3].concat(queStatus[2]);

    ap_uint<32> sourceID = config[5];
    ddrQue[0] = sourceID;
    ap_uint<32> numVertex = config[0];
    ap_uint<WIDTH> maxValue;
    if (WIDTH == 32) {
        maxValue = config[1];
    } else if (WIDTH == 64) {
        maxValue = config[2].concat(config[1]);
    }
    ap_uint<32> qcfg = config[3];
    ap_uint<32> cmd = config[4];
    bool enaWei = cmd.get_bit(0);
    int type = cmd.get_bit(2);

    initRes<WIDTH>(numVertex, maxValue, sourceID, result512);
    while (que_rd_cnt != que_wr_cnt) {
        shortestPathS1Dataflow<WIDTH>(enaWei, type, qcfg, queStatus, ddrQue512, offset, column, weight, result512,
                                      tableStatus, toIDTable, distTable);
        shortestPathS2Dataflow<WIDTH, MAXOUTDEGREE>(qcfg, tableStatus, toIDTable, distTable, result, queStatus, ddrQue);

        que_rd_cnt = queStatus[1].concat(queStatus[0]);
        que_wr_cnt = queStatus[3].concat(queStatus[2]);
    }

    info[0] = queStatus[6];
    info[1] = tableStatus[1];
}

} // namespace nopred

} // namespace sssp

namespace mssp {} // namespace mssp

} // namespace internal

/**
 * @brief singleSourceShortestPath the single source shortest path algorithm is implemented, the input is the matrix in
 * CSR format.
 *
 * @tparam WIDTH date width of the weight and the result distance
 * @tparam MAXOUTDEGREE The max out put degree of the input graph supported. Large max out degree value
 *
 * @param config    The config data. config[0] is the number of vertices in the graph. config[1]
 * is the lower 32bit of the max distance value. config[2] is the higher 32bit of the max distance value. config[3] is
 * the depth of the queue. config[4] is the option for the sssp: config[4].get_bit(0) is the weight/unweight switch,
 * config[4].get_bit(1) is the path switch, config[4].get_bit(2) is the fixed/float switch.
 * config[5] is sourceID.
 * @param offsetCSR    The offsetCSR buffer that stores the offsetCSR data in CSR format
 * @param indexCSR    The indexCSR buffer that stores the indexCSR dada in CSR format
 * @param weightCSR    The weight buffer that stores the weight data in CSR format
 * @param queue512 The shortest path requires a queue. The queue will be stored here in 512bit. Please allocate the
 * same buffer with queue32 and budle to the same gmem port with queue32.
 * @param queue32    The shortest path requires a queue. The queue will be stored here in 32bit. Please allocate the
 * same buffer with queue512 and budle to the same gmem port with queue512.
 * @param distance512 The distance32 data. The width is 512. When allocating buffers, distance512 and distance32 should
 * point to the same buffer. And please bundle distance512 and distance32 to the same gmem port.
 * @param distance32    The distance32 data is stored here. When allocating buffers, distance512 and distance32 should
 * point to the same buffer. And please bundle distance512 and distance32 to the same gmem port.
 * @param pred512 The predecessor data. The width is 512. When allocating buffers, pred512 and pred32 should
 * point to the same buffer. And please bundle pred512 and pred32 to the same gmem port.
 * @param pred32  The predecessor data is stored here. When allocating buffers, pred512 and pred32 should
 * point to the same buffer. And please bundle pred512 and pred32 to the same gmem port.
 * @param info The debug information. info[0] shows if the queue overflow. info[1] shows if the precessed graph exceeds
 * the supported maxoutdegree.
 *
 */
template <int WIDTH, int MAXOUTDEGREE>
void singleSourceShortestPath(ap_uint<32>* config,
                              ap_uint<512>* offsetCSR,
                              ap_uint<512>* indexCSR,
                              ap_uint<512>* weightCSR,

                              ap_uint<512>* queue512,
                              ap_uint<32>* queue32,

                              ap_uint<512>* distance512,
                              ap_uint<WIDTH>* distance32,
                              ap_uint<512>* pred512,
                              ap_uint<32>* pred32,
                              ap_uint<8>* info) {
    xf::graph::internal::sssp::pred::shortestPathInner<WIDTH, MAXOUTDEGREE>(
        config, offsetCSR, indexCSR, weightCSR, queue512, queue32, distance512, distance32, pred512, pred32, info);
}

/**
 * @brief singleSourceShortestPath the single source shortest path algorithm is implemented, the input is the matrix in
 * CSR format.
 *
 * @tparam WIDTH date width of the weight and the result distance
 * @tparam MAXOUTDEGREE The max out put degree of the input graph supported. Large max out degree value
 *
 * @param config    The config data. config[0] is the number of vertices in the graph. config[1]
 * is the lower 32bit of the max distance value. config[2] is the higher 32bit of the max distance value. config[3] is
 * the depth of the queue. config[4] is the option for the sssp: config[4].get_bit(0) is the weight/unweight switch,
 * config[4].get_bit(2) is the fixed/float switch.
 * config[5] is the sourceID.
 * @param offsetCSR    The offsetCSR buffer that stores the offsetCSR data in CSR format
 * @param indexCSR    The indexCSR buffer that stores the indexCSR dada in CSR format
 * @param weightCSR    The weight buffer that stores the weight data in CSR format
 * @param queue512 The shortest path requires a queue. The queue will be stored here in 512bit. Please allocate the
 * same buffer with queue32 and budle to the same gmem port with queue32.
 * @param queue32    The shortest path requires a queue. The queue will be stored here in 32bit. Please allocate the
 * same buffer with queue512 and budle to the same gmem port with queue512.
 * @param distance512 The distance32 data. The width is 512. When allocating buffers, distance512 and distance32 should
 * point to the same buffer. And please bundle distance512 and distance32 to the same gmem port.
 * @param distance32    The distance32 data is stored here. When allocating buffers, distance512 and distance32 should
 * point to the same buffer. And please bundle distance512 and distance32 to the same gmem port.
 * @param info The debug information. info[0] shows if the queue overflow. info[1] shows if the precessed graph exceeds
 * the supported maxoutdegree.
 *
 */
template <int WIDTH, int MAXOUTDEGREE>
void singleSourceShortestPath(ap_uint<32>* config,
                              ap_uint<512>* offsetCSR,
                              ap_uint<512>* indexCSR,
                              ap_uint<512>* weightCSR,

                              ap_uint<512>* queue512,
                              ap_uint<32>* queue32,

                              ap_uint<512>* distance512,
                              ap_uint<WIDTH>* distance32,
                              ap_uint<8>* info) {
    xf::graph::internal::sssp::nopred::shortestPathInner<WIDTH, MAXOUTDEGREE>(
        config, offsetCSR, indexCSR, weightCSR, queue512, queue32, distance512, distance32, info);
}

} // namespace graph
} // namespace xf
#endif
