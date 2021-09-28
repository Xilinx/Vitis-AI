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

#ifndef _XF_GRAPH_CONNECTED_COMPONENTS_HPP_
#define _XF_GRAPH_CONNECTED_COMPONENTS_HPP_

#include <ap_int.h>
#include <hls_stream.h>

#include "convert_csr_csc.hpp"
#include "xf_database/merge_sort.hpp"

namespace xf {
namespace graph {
namespace internal {
namespace connected_components {

static bool que_valid;
static ap_uint<512> que_reg;
static ap_uint<32> que_addr;

static bool column_valid0;
static ap_uint<512> column_reg0;
static ap_uint<32> column_addr0;
static bool column_valid1;
static ap_uint<512> column_reg1;
static ap_uint<32> column_addr1;

static bool result_valid;
static ap_uint<512> result_reg;
static ap_uint<32> result_addr;

inline void loadQueue(const int vertexNum,
                      ap_uint<32> queStatus[8],
                      ap_uint<512>* ddrQue,

                      hls::stream<ap_uint<32> >& fromIDStrm0,
                      hls::stream<ap_uint<32> >& fromIDStrm1,
                      hls::stream<bool>& ctrlStrm0,
                      hls::stream<bool>& ctrlStrm1) {
#pragma HLS inline off
    ap_uint<32> fromID;
    ap_uint<64> que_rd_cnt;
    ap_uint<64> que_wr_cnt;
    ap_uint<32> que_rd_ptr;

    que_rd_cnt = queStatus[1].concat(queStatus[0]);
    que_wr_cnt = queStatus[3].concat(queStatus[2]);
    que_rd_ptr = queStatus[4];

    if (que_rd_cnt != que_wr_cnt) {
        int que_rd_ptrH = que_rd_ptr.range(31, 4);
        int que_rd_ptrL = que_rd_ptr.range(3, 0);
        if (que_valid == 0 || que_rd_ptrH != que_addr) {
            que_reg = ddrQue[que_rd_ptrH];
            que_valid = 1;
            que_addr = que_rd_ptrH;
        }

        fromID = que_reg.range(32 * (que_rd_ptrL + 1) - 1, 32 * que_rd_ptrL);

        que_rd_cnt++;
        if (que_rd_ptr != vertexNum - 1)
            que_rd_ptr++;
        else
            que_rd_ptr = 0;

        fromIDStrm0.write(fromID);
        fromIDStrm1.write(fromID);
        ctrlStrm0.write(0);
        ctrlStrm1.write(0);
    } else {
        fromIDStrm0.write(0);
        fromIDStrm1.write(0);
        ctrlStrm0.write(1);
        ctrlStrm1.write(1);
    }

    queStatus[0] = que_rd_cnt.range(31, 0);
    queStatus[1] = que_rd_cnt.range(63, 32);
    queStatus[4] = que_rd_ptr;
}

inline void loadOffset(int idxoffset_s,
                       ap_uint<512> offset_reg,
                       hls::stream<ap_uint<32> >& fromIDStrm,
                       hls::stream<bool>& ctrlInStrm,

                       ap_uint<512>* offset,

                       hls::stream<ap_uint<32> >& offsetLowStrm,
                       hls::stream<ap_uint<32> >& offsetHighStrm,
                       hls::stream<bool>& ctrlOutStrm) {
#pragma HLS inline off
    ap_uint<32> fromID;
    bool ctrl;

    fromID = fromIDStrm.read();
    ctrl = ctrlInStrm.read();

    if (ctrl == 0) {
        int idxHcur = fromID.range(31, 4);
        int idxLcur = fromID.range(3, 0);
        int idxHnxt = (fromID + 1).range(31, 4);

        ap_uint<32> offsetLow;
        ap_uint<32> offsetHigh;
        if (idxHcur == idxoffset_s && idxHcur == idxHnxt) { // hit, and offsets in one line
            ap_uint<64> t0 = offset_reg.range(32 * (idxLcur + 2) - 1, 32 * idxLcur);
            offsetLow = t0(31, 0);
            offsetHigh = t0(63, 32);
        } else if (idxHcur == idxHnxt) { // no hit, but offsets in one line
            offset_reg = offset[idxHcur];
            idxoffset_s = idxHcur;
            ap_uint<64> t0 = offset_reg.range(32 * (idxLcur + 2) - 1, 32 * idxLcur);
            offsetLow = t0(31, 0);
            offsetHigh = t0(63, 32);
        } else { // no hit, and offsets in different line
            ap_uint<512> offset_t[2];
#pragma HLS array_partition variable = offset_t complete dim = 0
            for (int i = 0; i < 2; i++) {
#pragma HLS PIPELINE II = 1
                offset_t[i] = offset[idxHcur + i];
            }

            offset_reg = offset_t[1];
            idxoffset_s = idxHnxt;
            offsetLow = offset_t[0](511, 480);
            offsetHigh = offset_t[1](31, 0);
        }

        offsetLowStrm.write(offsetLow);
        offsetHighStrm.write(offsetHigh);
        ctrlOutStrm.write(0);
    } else {
        offsetLowStrm.write(0);
        offsetHighStrm.write(0);
        ctrlOutStrm.write(1);
    }
}

inline void loadColumnSendAddr(hls::stream<ap_uint<32> >& offsetLowStrm,
                               hls::stream<ap_uint<32> >& offsetHighStrm,
                               hls::stream<bool>& ctrlInStrm,

                               hls::stream<ap_uint<32> >& columnAddrStrm,
                               hls::stream<bool>& ctrlOutStrm) {
#pragma HLS inline off
    bool ctrl;
    ap_uint<32> offsetLow;
    ap_uint<32> offsetHigh;

    offsetLow = offsetLowStrm.read();
    offsetHigh = offsetHighStrm.read();
    ctrl = ctrlInStrm.read();

    if (ctrl == 0) {
        for (ap_uint<32> i = offsetLow; i < offsetHigh; i++) {
#pragma HLS PIPELINE II = 1
            columnAddrStrm.write(i);
            ctrlOutStrm.write(0);
        }
    }

    ctrlOutStrm.write(1);
}

inline void loadColumnCSR(hls::stream<ap_uint<32> >& raddrStrm,
                          hls::stream<bool>& ctrlInStrm,

                          ap_uint<512>* column,

                          hls::stream<ap_uint<32> >& toIDStrm,
                          hls::stream<bool>& ctrlOutStrm) {
#pragma HLS inline off

    bool ctrl = ctrlInStrm.read();

    while (!ctrl) {
#pragma HLS PIPELINE II = 1
        ap_uint<32> raddr = raddrStrm.read();
        int idxHc = raddr.range(31, 4);
        int idxLc = raddr.range(3, 0);

        if (column_valid0 == 0 || idxHc != column_addr0) {
            column_valid0 = 1;
            column_addr0 = idxHc;
            column_reg0 = column[idxHc];
        }

        toIDStrm.write(column_reg0.range(32 * (idxLc + 1) - 1, 32 * idxLc));
        ctrlOutStrm.write(0);
        ctrl = ctrlInStrm.read();
    }

    ctrlOutStrm.write(1);
}

inline void loadColumnCSC(hls::stream<ap_uint<32> >& raddrStrm,
                          hls::stream<bool>& ctrlInStrm,

                          ap_uint<512>* column,

                          hls::stream<ap_uint<32> >& toIDStrm,
                          hls::stream<bool>& ctrlOutStrm) {
#pragma HLS inline off

    bool ctrl = ctrlInStrm.read();

    while (!ctrl) {
#pragma HLS PIPELINE II = 1
        ap_uint<32> raddr = raddrStrm.read();
        int idxHc = raddr.range(31, 4);
        int idxLc = raddr.range(3, 0);

        if (column_valid1 == 0 || idxHc != column_addr1) {
            column_valid1 = 1;
            column_addr1 = idxHc;
            column_reg1 = column[idxHc];
        }

        toIDStrm.write(column_reg1.range(32 * (idxLc + 1) - 1, 32 * idxLc));
        ctrlOutStrm.write(0);
        ctrl = ctrlInStrm.read();
    }

    ctrlOutStrm.write(1);
}

inline void removeDup(hls::stream<ap_uint<32> >& toIDStrm,
                      hls::stream<bool>& ctrlInStrm,

                      hls::stream<ap_uint<32> >& toIDOutStrm,
                      hls::stream<bool>& ctrlOutStrm) {
#pragma HLS inline off
    bool ctrl;
    ap_uint<32> toID0 = 0;
    ap_uint<32> toID1 = -1;

    ctrl = ctrlInStrm.read();

    while (!ctrl) {
#pragma HLS PIPELINE II = 1
        ctrl = ctrlInStrm.read();
        toID0 = toIDStrm.read();

        if (toID0 != toID1) {
            toIDOutStrm.write(toID0);
            ctrlOutStrm.write(false);
        }

        toID1 = toID0;
    }

    ctrlOutStrm.write(true);
}

inline void loadRes(hls::stream<ap_uint<32> >& toIDStrm,
                    hls::stream<bool>& ctrlInStrm,

                    ap_uint<512>* color512,

                    hls::stream<ap_uint<32> >& toIDOutStrm,
                    hls::stream<ap_uint<32> >& colorOutStrm,
                    hls::stream<bool>& ctrlOutStrm) {
#pragma HLS inline off
    bool ctrl;

    ctrl = ctrlInStrm.read();
    while (!ctrl) {
#pragma HLS PIPELINE II = 1
        ctrl = ctrlInStrm.read();
        ap_uint<32> toID = toIDStrm.read();
        int raddrH = toID.range(31, 4);
        int raddrL = toID.range(3, 0);

        if (result_valid == 0 || raddrH != result_addr) {
            result_valid = 1;
            result_addr = raddrH;
            result_reg = color512[raddrH];
        }

        ap_uint<32> color = result_reg.range(32 * (raddrL + 1) - 1, 32 * raddrL);
        colorOutStrm.write(color);
        toIDOutStrm.write(toID);
        ctrlOutStrm.write(0);
    }

    ctrlOutStrm.write(1);
}

inline void storeUpdate(hls::stream<ap_uint<32> >& toIDStrm,
                        hls::stream<ap_uint<32> >& colorInStrm,
                        hls::stream<bool>& ctrlInStrm,

                        ap_uint<32> tableStatus[3],
                        ap_uint<32>* toIDTable) {
#pragma HLS inline off
    ap_uint<32> cnt = 0;

    bool ctrl = ctrlInStrm.read();
    while (!ctrl) {
#pragma HLS PIPELINE II = 1
        ctrl = ctrlInStrm.read();
        ap_uint<32> toID = toIDStrm.read();
        ap_uint<32> color = colorInStrm.read();

        if (color == (ap_uint<32>)-1) {
            toIDTable[cnt] = toID;
            cnt++;
        }
    }

    tableStatus[1] = cnt;
}

inline void forwardBFSS1Dataflow(const int vertexNum,
                                 int idxoffset_s[2],
                                 ap_uint<512> offset_reg[2],
                                 ap_uint<32> queStatus[8],
                                 ap_uint<512>* ddrQue,

                                 ap_uint<512>* offsetG1,
                                 ap_uint<512>* columnG1,
                                 ap_uint<512>* offsetG2,
                                 ap_uint<512>* columnG2,

                                 ap_uint<512>* color512,

                                 ap_uint<32> tableStatus[3],
                                 ap_uint<32>* toIDTable) {
#pragma HLS inline off
#pragma HLS dataflow
    hls::stream<ap_uint<32> > fromIDStrm0;
#pragma HLS stream variable = fromIDStrm0 depth = 32
#pragma HLS resource variable = fromIDStrm0 core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm0;
#pragma HLS stream variable = ctrlStrm0 depth = 32
#pragma HLS resource variable = ctrlStrm0 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > fromIDStrm1;
#pragma HLS stream variable = fromIDStrm1 depth = 32
#pragma HLS resource variable = fromIDStrm1 core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm1;
#pragma HLS stream variable = ctrlStrm1 depth = 32
#pragma HLS resource variable = ctrlStrm1 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > offsetLowStrm0;
#pragma HLS stream variable = offsetLowStrm0 depth = 32
#pragma HLS resource variable = offsetLowStrm0 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > offsetHighStrm0;
#pragma HLS stream variable = offsetHighStrm0 depth = 32
#pragma HLS resource variable = offsetHighStrm0 core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm2;
#pragma HLS stream variable = ctrlStrm2 depth = 32
#pragma HLS resource variable = ctrlStrm2 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > columnAddrStrm0;
#pragma HLS stream variable = columnAddrStrm0 depth = 32
#pragma HLS resource variable = columnAddrStrm0 core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm3;
#pragma HLS stream variable = ctrlStrm3 depth = 32
#pragma HLS resource variable = ctrlStrm3 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > offsetLowStrm1;
#pragma HLS stream variable = offsetLowStrm1 depth = 32
#pragma HLS resource variable = offsetLowStrm1 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > offsetHighStrm1;
#pragma HLS stream variable = offsetHighStrm1 depth = 32
#pragma HLS resource variable = offsetHighStrm1 core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm5;
#pragma HLS stream variable = ctrlStrm5 depth = 32
#pragma HLS resource variable = ctrlStrm5 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > columnAddrStrm1;
#pragma HLS stream variable = columnAddrStrm1 depth = 32
#pragma HLS resource variable = columnAddrStrm1 core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm6;
#pragma HLS stream variable = ctrlStrm6 depth = 32
#pragma HLS resource variable = ctrlStrm6 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > toIDStrm1;
#pragma HLS stream variable = toIDStrm1 depth = 32
#pragma HLS resource variable = toIDStrm1 core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm4;
#pragma HLS stream variable = ctrlStrm4 depth = 32
#pragma HLS resource variable = ctrlStrm4 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > toIDStrm2;
#pragma HLS stream variable = toIDStrm2 depth = 32
#pragma HLS resource variable = toIDStrm2 core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm7;
#pragma HLS stream variable = ctrlStrm7 depth = 32
#pragma HLS resource variable = ctrlStrm7 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > merge_data_strm;
#pragma HLS stream variable = merge_data_strm depth = 32
#pragma HLS resource variable = merge_data_strm core = FIFO_LUTRAM

    hls::stream<bool> merge_e_strm;
#pragma HLS stream variable = merge_e_strm depth = 32
#pragma HLS resource variable = merge_e_strm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > toIDStrm3;
#pragma HLS stream variable = toIDStrm3 depth = 32
#pragma HLS resource variable = toIDStrm3 core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm8;
#pragma HLS stream variable = ctrlStrm8 depth = 32
#pragma HLS resource variable = ctrlStrm8 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > toIDStrm4;
#pragma HLS stream variable = toIDStrm4 depth = 32
#pragma HLS resource variable = toIDStrm4 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > colorStrm;
#pragma HLS stream variable = colorStrm depth = 32
#pragma HLS resource variable = colorStrm core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm9;
#pragma HLS stream variable = ctrlStrm9 depth = 32
#pragma HLS resource variable = ctrlStrm9 core = FIFO_LUTRAM

    loadQueue(vertexNum, queStatus, ddrQue, fromIDStrm0, fromIDStrm1, ctrlStrm0, ctrlStrm1);

    // read CSR Graph
    loadOffset(idxoffset_s[0], offset_reg[0], fromIDStrm0, ctrlStrm0, offsetG1, offsetLowStrm0, offsetHighStrm0,
               ctrlStrm2);
    loadColumnSendAddr(offsetLowStrm0, offsetHighStrm0, ctrlStrm2, columnAddrStrm0, ctrlStrm3);
    loadColumnCSR(columnAddrStrm0, ctrlStrm3, columnG1, toIDStrm1, ctrlStrm4);

    // read CSC Graph
    loadOffset(idxoffset_s[1], offset_reg[1], fromIDStrm1, ctrlStrm1, offsetG2, offsetLowStrm1, offsetHighStrm1,
               ctrlStrm5);
    loadColumnSendAddr(offsetLowStrm1, offsetHighStrm1, ctrlStrm5, columnAddrStrm1, ctrlStrm6);
    loadColumnCSC(columnAddrStrm1, ctrlStrm6, columnG2, toIDStrm2, ctrlStrm7);

    // sort and remomve the duplication
    xf::database::mergeSort<ap_uint<32> >(toIDStrm1, ctrlStrm4, toIDStrm2, ctrlStrm7, merge_data_strm, merge_e_strm,
                                          true);
    removeDup(merge_data_strm, merge_e_strm, toIDStrm3, ctrlStrm8);

    // load unvisited node
    loadRes(toIDStrm3, ctrlStrm8, color512, toIDStrm4, colorStrm, ctrlStrm9);

    storeUpdate(toIDStrm4, colorStrm, ctrlStrm9, tableStatus, toIDTable);
}

template <int MAXDEGREE>
void updateDispatch(ap_uint<32> tableStatus[3],
                    ap_uint<32>* toIDTable,

                    hls::stream<ap_uint<32> >& rootIDStrm,
                    hls::stream<ap_uint<32> >& colorToIDStrm,
                    hls::stream<bool>& colorCtrlStrm,
                    hls::stream<ap_uint<32> >& queToIDStrm,
                    hls::stream<bool>& queCtrlStrm) {
#pragma HLS inline off
    ap_uint<32> rootID = tableStatus[0] + 1;
    ap_uint<32> tableCnt = tableStatus[1];
    if (tableCnt > MAXDEGREE) tableStatus[2] = 1;

    rootIDStrm.write(rootID);
    for (int i = 0; i < tableCnt; i++) {
#pragma HLS PIPELINE II = 1
        if (i < MAXDEGREE) {
            ap_uint<32> reg_toID = toIDTable[i];
            colorToIDStrm.write(reg_toID);
            colorCtrlStrm.write(0);

            queToIDStrm.write(reg_toID);
            queCtrlStrm.write(0);
        } else {
            colorToIDStrm.write(0);
            colorCtrlStrm.write(0);

            queToIDStrm.write(0);
            queCtrlStrm.write(0);
        }
    }

    colorCtrlStrm.write(1);
    queCtrlStrm.write(1);
}

inline void updateRes(hls::stream<ap_uint<32> >& rootIDStrm,
                      hls::stream<ap_uint<32> >& colorToIDStrm,
                      hls::stream<bool>& colorCtrlStrm,

                      ap_uint<32>* color32) {
#pragma HLS inline off
    ap_uint<32> rootID = rootIDStrm.read();
    bool ctrl = colorCtrlStrm.read();
    while (!ctrl) {
#pragma HLS PIPELINE II = 1
        ap_uint<32> toID = colorToIDStrm.read();
        color32[toID] = rootID;

        if (toID(31, 4) == result_addr) result_valid = 0;

        ctrl = colorCtrlStrm.read();
    }
}

inline void upadateQue(const int vertexNum,
                       hls::stream<ap_uint<32> >& queToIDStrm,
                       hls::stream<bool>& queCtrlStrm,

                       ap_uint<32> queStatus[8],
                       ap_uint<32>* ddrQue) {
#pragma HLS inline off
    ap_uint<64> que_rd_cnt;
    ap_uint<64> que_wr_cnt;
    ap_uint<32> que_wr_ptr;

    que_rd_cnt = queStatus[1].concat(queStatus[0]);
    que_wr_cnt = queStatus[3].concat(queStatus[2]);
    que_wr_ptr = queStatus[5];

    bool ctrl = queCtrlStrm.read();
    while (!ctrl) {
#pragma HLS PIPELINE II = 1
        ap_uint<32> toID = queToIDStrm.read();
        ddrQue[que_wr_ptr] = toID;
        int que_wr_ptrH = que_wr_ptr.range(31, 4);
        if (que_wr_ptrH == que_addr) {
            que_valid = 0;
        }

        que_wr_cnt++;
        if (que_wr_ptr != vertexNum - 1)
            que_wr_ptr++;
        else
            que_wr_ptr = 0;
        if (que_wr_cnt - que_rd_cnt > vertexNum) queStatus[6] = 1;
        ctrl = queCtrlStrm.read();
    }

    queStatus[2] = que_wr_cnt.range(31, 0);
    queStatus[3] = que_wr_cnt.range(63, 32);
    queStatus[5] = que_wr_ptr;
}

template <int MAXDEGREE>
void forwardBFSS2Dataflow(const int vertexNum,
                          ap_uint<32> tableStatus[3],
                          ap_uint<32> queStatus[8],
                          ap_uint<32>* toIDTable,

                          ap_uint<32>* color32,
                          ap_uint<32>* ddrQue) {
#pragma HLS inline off
#pragma HLS dataflow

    hls::stream<ap_uint<32> > rootIDStrm;
#pragma HLS stream variable = rootIDStrm depth = 2
#pragma HLS resource variable = rootIDStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > colorToIDStrm;
#pragma HLS stream variable = colorToIDStrm depth = 32
#pragma HLS resource variable = colorToIDStrm core = FIFO_LUTRAM

    hls::stream<bool> colorCtrlStrm;
#pragma HLS stream variable = colorCtrlStrm depth = 32
#pragma HLS resource variable = colorCtrlStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > queToIDStrm;
#pragma HLS stream variable = queToIDStrm depth = 32
#pragma HLS resource variable = queToIDStrm core = FIFO_LUTRAM

    hls::stream<bool> queCtrlStrm;
#pragma HLS stream variable = queCtrlStrm depth = 32
#pragma HLS resource variable = queCtrlStrm core = FIFO_LUTRAM

    updateDispatch<MAXDEGREE>(tableStatus, toIDTable, rootIDStrm, colorToIDStrm, colorCtrlStrm, queToIDStrm,
                              queCtrlStrm);

    updateRes(rootIDStrm, colorToIDStrm, colorCtrlStrm, color32);

    upadateQue(vertexNum, queToIDStrm, queCtrlStrm, queStatus, ddrQue);
}

inline void writeQueStatus(const int vertexNum, ap_uint<32> queStatus[8], ap_uint<32>* queue, ap_uint<32> value) {
    ap_uint<64> que_wr_cnt = queStatus[3].concat(queStatus[2]);
    ap_uint<32> que_wr_ptr = queStatus[5];
    queStatus[7] = value;
    queue[que_wr_ptr] = value;

    que_wr_cnt++;
    if (que_wr_ptr != vertexNum - 1)
        que_wr_ptr++;
    else
        que_wr_ptr = 0;

    queStatus[2] = que_wr_cnt.range(31, 0);
    queStatus[3] = que_wr_cnt.range(63, 32);
    queStatus[5] = que_wr_ptr;
}

inline void searchNewRoot(const int vertexNum,
                          ap_uint<32> queStatus[8],
                          ap_uint<512>* colorMap512,
                          ap_uint<32>* colorMap32,

                          ap_uint<32>* queue,
                          ap_uint<32>& newNode) {
#pragma HLS inline off

    int ptr = newNode;

    bool f = false;
    ap_uint<32> pos = 0;
    ap_uint<32> array[64] = {0};

    while (!f) {
        for (int i = 0; i < 64; i++) {
#pragma HLS pipeline II = 1
            array[i] = colorMap32[ptr + i];

            if (!f && (array[i] == (ap_uint<32>)-1 || (ptr + i == vertexNum))) {
                pos = ptr + i;
                f = true;
            }
        }

        ptr += 64;
    }

    writeQueStatus(vertexNum, queStatus, queue, pos);
    colorMap32[pos] = pos + 1;

    newNode = pos;
}

inline void initBuffer(const int depth, ap_uint<512>* result) {
    for (int i = 0; i < depth; i++) {
#pragma HLS PIPELINE II = 1
        result[i] = (ap_uint<512>)-1;
    }
}

template <int MAXDEGREE>
void forwardBFS(const int vertexNum,
                ap_uint<32> queStatus[8],
                ap_uint<512>* columnG1,
                ap_uint<512>* offsetG1,
                ap_uint<512>* columnG2,
                ap_uint<512>* offsetG2,

                ap_uint<512>* queue512,
                ap_uint<32>* queue,
                ap_uint<512>* result512,
                ap_uint<32>* result32) {
#pragma HLS inline off
    int idxoffset_s[2];
    idxoffset_s[0] = -1;
    idxoffset_s[1] = -1;
#pragma HLS ARRAY_PARTITION variable = idxoffset_s complete dim = 1
    ap_uint<512> offset_reg[2];
#pragma HLS ARRAY_PARTITION variable = offset_reg complete dim = 1

    ap_uint<32> tableStatus[3];
#pragma HLS ARRAY_PARTITION variable = tableStatus complete dim = 1

#ifndef __SYNTHESIS__
    ap_uint<32>* toIDTable = (ap_uint<32>*)malloc(MAXDEGREE * sizeof(ap_uint<32>));
#else
    ap_uint<32> toIDTable[MAXDEGREE];
#pragma HLS RESOURCE variable = toIDTable core = RAM_2P_URAM
#endif

    que_valid = 0;
    column_valid0 = 0;
    column_valid1 = 0;
    result_valid = 0;

    tableStatus[0] = queStatus[7]; // start from certain root node
    tableStatus[1] = 0;
    tableStatus[2] = 0;

    ap_uint<64> que_rd_cnt;
    ap_uint<64> que_wr_cnt;
    que_rd_cnt = queStatus[1].concat(queStatus[0]);
    que_wr_cnt = queStatus[3].concat(queStatus[2]);

    while (que_rd_cnt != que_wr_cnt) {
        forwardBFSS1Dataflow(vertexNum, idxoffset_s, offset_reg, queStatus, queue512, offsetG1, columnG1, offsetG2,
                             columnG2, result512, tableStatus, toIDTable);

        forwardBFSS2Dataflow<MAXDEGREE>(vertexNum, tableStatus, queStatus, toIDTable, result32, queue);

        que_rd_cnt = queStatus[1].concat(queStatus[0]);
        que_wr_cnt = queStatus[3].concat(queStatus[2]);
    }
}

template <int MAXDEGREE>
void connectedComponentImpl(const int vertexNum,

                            ap_uint<512>* columnG1,
                            ap_uint<512>* offsetG1,
                            ap_uint<512>* columnG2,
                            ap_uint<512>* offsetG2,

                            ap_uint<512>* queue512,
                            ap_uint<32>* queue,
                            ap_uint<512>* result512,

                            ap_uint<32>* result32) {
#pragma HLS inline off

    ap_uint<32> queStatus[8];
#pragma HLS ARRAY_PARTITION variable = queStatus complete dim = 1

    queStatus[0] = 0;
    queStatus[1] = 0;
    queStatus[2] = 1;
    queStatus[3] = 0;
    queStatus[4] = 0;
    queStatus[5] = 1;
    queStatus[6] = 0;
    queStatus[7] = 0; // start from node 0

    queue[0] = 0;
    result32[0] = 1;

    ap_uint<32> node = 0;

    while (node != vertexNum) {
        forwardBFS<MAXDEGREE>(vertexNum, queStatus, columnG1, offsetG1, columnG2, offsetG2, queue512, queue, result512,
                              result32);

        searchNewRoot(vertexNum, queStatus, result512, result32, queue, node);
    }
}
} // namespace connected_components
} // namespace internal

/**
 * @brief connectedComponents Compute the connected component membership of each vertex only for undirected graph
 *
 * @tparam MAXVERTEX CSC/CSR data vertex(offset) array maxsize
 * @tparam MAXEDGE CSC/CSR data edge(indice) array maxsize
 * @tparam LOG2CACHEDEPTH cache depth in Binary, the cache onchip memory is 512 bit x uramRow
 * @tparam LOG2DATAPERCACHELINE The log2 of number of data in each cache line (512bit), for double, it's 3, for float,
 * it's 4
 * @tparam RAMTYPE flag to tell use URAM LUTRAM or BRAM, 0 : LUTRAM, 1 : URAM, 2 : BRAM
 * @tparam MAXOUTDEGREE the max sum of indegree and outdegree of the input graph supported. Large value will result in
 * more
 * URAM usage
 *
 * @param numEdge edge number of the input graph
 * @param numVertex vertex number of the input graph
 * @param indexCSR column index of CSR format
 * @param offsetCSR row offset of CSR format
 * @param indexCSC512 column index of CSC format in 512bit, which should be shared the same buffer with indexCSC32 in
 * host
 * @param indexCSC32 column index of CSC format in 32bit, which should be shared the same buffer with indexCSC512 in
 * host
 * @param offsetCSC row offset of CSC format
 * @param offsetCSCTmp1 temp row offset for CSR2CSC convert
 * @param offsetCSCTmp2 temp row offset of CSR2CSC convert
 * @param queue32 intermediate queue32 used during the internal BFS
 * @param component512 Same as component32 but in 512bit, which shoude be shared the same buffer with component32 in
 * host
 * @param component32 return result buffer with the vertex label containing the lowest vertex id in the connnected
 * component containing that vertex
 *
 */
template <int MAXVERTEX, int MAXEDGE, int LOG2CACHEDEPTH, int LOG2DATAPERCACHELINE, int RAMTYPE, int MAXOUTDEGREE>
void connectedComponents(const int numEdge,
                         const int numVertex,

                         ap_uint<512>* offsetCSR,
                         ap_uint<512>* indexCSR,

                         ap_uint<512>* offsetCSC,
                         ap_uint<512>* indexCSC512,
                         ap_uint<32>* indexCSC32,

                         ap_uint<512>* offsetCSCTmp1,
                         ap_uint<512>* offsetCSCTmp2,

                         ap_uint<512>* queue512,
                         ap_uint<32>* queue32,

                         ap_uint<512>* component512,
                         ap_uint<32>* component32) {
#pragma HLS inline off

    const int depth = (numVertex + 15) / 16;

    convertCsrCsc<ap_uint<32>, MAXVERTEX, MAXEDGE, LOG2CACHEDEPTH, LOG2DATAPERCACHELINE, RAMTYPE>(
        numEdge, numVertex, offsetCSR, indexCSR, offsetCSC, indexCSC32, offsetCSCTmp1, offsetCSCTmp2);

    internal::connected_components::initBuffer(depth, component512);

    internal::connected_components::connectedComponentImpl<MAXOUTDEGREE>(
        numVertex, indexCSR, offsetCSR, indexCSC512, offsetCSC, queue512, queue32, component512, component32);
}

} // namespace graph
} // namespace xf
#endif
