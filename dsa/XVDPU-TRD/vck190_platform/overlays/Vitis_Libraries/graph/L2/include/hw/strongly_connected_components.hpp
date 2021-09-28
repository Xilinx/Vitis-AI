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

#ifndef _XF_GRAPH_STRONGLY_CONNECTED_COMPONENTS_HPP_
#define _XF_GRAPH_STRONGLY_CONNECTED_COMPONENTS_HPP_

#include <ap_int.h>
#include <hls_stream.h>

#include "convert_csr_csc.hpp"

namespace xf {
namespace graph {
namespace internal {
namespace scc {
inline void loadQueue(const int vertexNum,
                      ap_uint<32> queStatus[8],
                      ap_uint<32>* ddrQue,

                      hls::stream<ap_uint<32> >& rootIDStrm,
                      hls::stream<ap_uint<32> >& fromIDStrm,
                      hls::stream<bool>& ctrlStrm) {
#pragma HLS inline off
    ap_uint<32> fromID;
    ap_uint<64> que_rd_cnt;
    ap_uint<64> que_wr_cnt;
    ap_uint<32> que_rd_ptr;
    ap_uint<32> rootID;

    que_rd_cnt = queStatus[1].concat(queStatus[0]);
    que_wr_cnt = queStatus[3].concat(queStatus[2]);
    que_rd_ptr = queStatus[4];
    rootID = queStatus[7];

    rootIDStrm.write(rootID);
    if (que_rd_cnt != que_wr_cnt) {
        fromID = ddrQue[que_rd_ptr];
        que_rd_cnt++;
        if (que_rd_ptr != vertexNum - 1)
            que_rd_ptr++;
        else
            que_rd_ptr = 0;

        fromIDStrm.write(fromID);
        ctrlStrm.write(0);
    } else {
        fromIDStrm.write(0);
        ctrlStrm.write(1);
    }

    queStatus[0] = que_rd_cnt.range(31, 0);
    queStatus[1] = que_rd_cnt.range(63, 32);
    queStatus[4] = que_rd_ptr;
}

inline void loadOffset(int idxoffset_s,
                       ap_uint<512> offset_reg,
                       hls::stream<ap_uint<32> >& rootIDStrm,
                       hls::stream<ap_uint<32> >& fromIDStrm,
                       hls::stream<bool>& ctrlInStrm,

                       ap_uint<512>* offset,

                       hls::stream<ap_uint<32> >& rootIDOutStrm,
                       hls::stream<ap_uint<32> >& offsetLowStrm,
                       hls::stream<ap_uint<32> >& offsetHighStrm,
                       hls::stream<bool>& ctrlOutStrm) {
#pragma HLS inline off
    ap_uint<32> fromID;
    ap_uint<32> rootID;
    bool ctrl;

    rootID = rootIDStrm.read();
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

        rootIDOutStrm.write(rootID);
        offsetLowStrm.write(offsetLow);
        offsetHighStrm.write(offsetHigh);
        ctrlOutStrm.write(0);
    } else {
        rootIDOutStrm.write(0);
        offsetLowStrm.write(0);
        offsetHighStrm.write(0);
        ctrlOutStrm.write(1);
    }
}

inline void loadColumn(hls::stream<ap_uint<32> >& rootIDInStrm,
                       hls::stream<ap_uint<32> >& offsetLowStrm,
                       hls::stream<ap_uint<32> >& offsetHighStrm,
                       hls::stream<bool>& ctrlInStrm,

                       ap_uint<512>* column,

                       hls::stream<ap_uint<32> >& rootIDOutStrm,
                       hls::stream<ap_uint<32> >& toIDStrm,
                       hls::stream<bool>& ctrlOutStrm) {
#pragma HLS inline off
    ap_uint<32> rootID;
    bool ctrl;
    ap_uint<32> offsetLow;
    ap_uint<32> offsetHigh;
    ap_uint<512> column_reg;

    rootID = rootIDInStrm.read();
    offsetLow = offsetLowStrm.read();
    offsetHigh = offsetHighStrm.read();
    ctrl = ctrlInStrm.read();

    rootIDOutStrm.write(rootID);
    if (ctrl == 0) {
        if (offsetLow.range(3, 0) != 0) column_reg = column[offsetLow.range(31, 4)];
    READ_COLUMN_LOOP:
        for (ap_uint<32> i = offsetLow; i < offsetHigh; i++) {
#pragma HLS PIPELINE II = 1
            int idxHc = i.range(31, 4);
            int idxLc = i.range(3, 0);

            if (idxLc == 0) column_reg = column[idxHc];
            toIDStrm.write(column_reg.range(32 * (idxLc + 1) - 1, 32 * idxLc));
            ctrlOutStrm.write(0);
        }
    }

    ctrlOutStrm.write(1);
}

inline void loadRes(int idxoffset_s,
                    ap_uint<512> offset_reg,
                    hls::stream<ap_uint<32> >& rootIDStrm,
                    hls::stream<ap_uint<32> >& toIDStrm,
                    hls::stream<bool>& ctrlInStrm,

                    ap_uint<512>* color512,

                    hls::stream<ap_uint<32> >& rootIDOutStrm,
                    hls::stream<ap_uint<32> >& toIDOutStrm,
                    hls::stream<ap_uint<32> >& colorOutStrm,
                    hls::stream<bool>& ctrlOutStrm) {
#pragma HLS inline off
    bool ctrl;

    ap_uint<32> rootID = rootIDStrm.read();
    rootIDOutStrm.write(rootID);

    ctrl = ctrlInStrm.read();
    while (!ctrl) {
#pragma HLS PIPELINE II = 1
        ctrl = ctrlInStrm.read();
        ap_uint<32> toID = toIDStrm.read();
        int raddrH = toID.range(31, 4);
        int raddrL = toID.range(3, 0);

        ap_uint<512> line;
        if (raddrH == idxoffset_s) { // hit
            line = offset_reg;
        } else { // no hit
            offset_reg = color512[raddrH];
            idxoffset_s = raddrH;
            line = offset_reg;
        }

        ap_uint<32> color = line.range(32 * (raddrL + 1) - 1, 32 * raddrL);
        colorOutStrm.write(color);
        toIDOutStrm.write(toID);
        ctrlOutStrm.write(0);
    }

    ctrlOutStrm.write(1);
}

inline void storeUpdate(hls::stream<ap_uint<32> >& rootIDStrm,
                        hls::stream<ap_uint<32> >& toIDStrm,
                        hls::stream<ap_uint<32> >& colorInStrm,
                        hls::stream<bool>& ctrlInStrm,

                        ap_uint<32> tableStatus[3],
                        ap_uint<32>* toIDTable) {
#pragma HLS inline off
    ap_uint<32> rootID;
    ap_uint<32> cnt = 0;

    rootID = rootIDStrm.read();

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

    tableStatus[0] = rootID;
    tableStatus[1] = cnt;
}

inline void forwardSCCS1Dataflow(const int vertexNum,
                                 int idxoffset_s[2],
                                 ap_uint<512> offset_reg[2],
                                 ap_uint<32> queStatus[8],
                                 ap_uint<32>* ddrQue,

                                 ap_uint<512>* offset,
                                 ap_uint<512>* column,
                                 ap_uint<512>* color512,

                                 ap_uint<32> tableStatus[3],
                                 ap_uint<32>* toIDTable) {
#pragma HLS inline off
#pragma HLS dataflow
    hls::stream<ap_uint<32> > fromIDStrm1;
#pragma HLS stream variable = fromIDStrm1 depth = 32
#pragma HLS resource variable = fromIDStrm1 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > rootIDStrm1;
#pragma HLS stream variable = rootIDStrm1 depth = 2
#pragma HLS resource variable = rootIDStrm1 core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm1;
#pragma HLS stream variable = ctrlStrm1 depth = 32
#pragma HLS resource variable = ctrlStrm1 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > rootIDStrm2;
#pragma HLS stream variable = rootIDStrm2 depth = 2
#pragma HLS resource variable = rootIDStrm2 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > offsetLowStrm;
#pragma HLS stream variable = offsetLowStrm depth = 32
#pragma HLS resource variable = offsetLowStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > offsetHighStrm;
#pragma HLS stream variable = offsetHighStrm depth = 32
#pragma HLS resource variable = offsetHighStrm core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm2;
#pragma HLS stream variable = ctrlStrm2 depth = 32
#pragma HLS resource variable = ctrlStrm2 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > rootIDStrm3;
#pragma HLS stream variable = rootIDStrm3 depth = 2
#pragma HLS resource variable = rootIDStrm3 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > toIDStrm1;
#pragma HLS stream variable = toIDStrm1 depth = 32
#pragma HLS resource variable = toIDStrm1 core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm3;
#pragma HLS stream variable = ctrlStrm3 depth = 32
#pragma HLS resource variable = ctrlStrm3 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > rootIDStrm4;
#pragma HLS stream variable = rootIDStrm4 depth = 2
#pragma HLS resource variable = rootIDStrm4 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > toIDStrm2;
#pragma HLS stream variable = toIDStrm2 depth = 32
#pragma HLS resource variable = toIDStrm2 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > colorStrm;
#pragma HLS stream variable = colorStrm depth = 32
#pragma HLS resource variable = colorStrm core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm4;
#pragma HLS stream variable = ctrlStrm4 depth = 32
#pragma HLS resource variable = ctrlStrm4 core = FIFO_LUTRAM

    loadQueue(vertexNum, queStatus, ddrQue, rootIDStrm1, fromIDStrm1, ctrlStrm1);

    loadOffset(idxoffset_s[0], offset_reg[0], rootIDStrm1, fromIDStrm1, ctrlStrm1, offset, rootIDStrm2, offsetLowStrm,
               offsetHighStrm, ctrlStrm2);

    loadColumn(rootIDStrm2, offsetLowStrm, offsetHighStrm, ctrlStrm2, column, rootIDStrm3, toIDStrm1, ctrlStrm3);

    loadRes(idxoffset_s[1], offset_reg[1], rootIDStrm3, toIDStrm1, ctrlStrm3, color512, rootIDStrm4, toIDStrm2,
            colorStrm, ctrlStrm4);

    storeUpdate(rootIDStrm4, toIDStrm2, colorStrm, ctrlStrm4, tableStatus, toIDTable);
}

inline void backwardLoadRes(int idxoffset_s,
                            ap_uint<512> offset_reg,
                            bool mode,
                            ap_uint<32> rootIDp,
                            hls::stream<ap_uint<32> >& rootIDStrm,
                            hls::stream<ap_uint<32> >& toIDStrm,
                            hls::stream<bool>& ctrlInStrm,

                            ap_uint<512>* color512,
                            ap_uint<32>* result,

                            hls::stream<ap_uint<32> >& rootIDOutStrm,
                            hls::stream<ap_uint<32> >& toIDOutStrm,
                            hls::stream<ap_uint<32> >& colorOutStrm,
                            hls::stream<bool>& ctrlOutStrm) {
#pragma HLS inline off
    bool ctrl;

    ap_uint<32> rootID = rootIDStrm.read();
    ap_uint<32> rootID_t = mode ? (ap_uint<32>)(rootID + 1) : rootID;
    rootIDOutStrm.write(rootID_t);

    ctrl = ctrlInStrm.read();
    while (!ctrl) {
#pragma HLS PIPELINE II = 1
        ctrl = ctrlInStrm.read();
        ap_uint<32> toID = toIDStrm.read();
        int raddrH = toID.range(31, 4);
        int raddrL = toID.range(3, 0);

        ap_uint<32> visited = result[toID];

        ap_uint<512> line;
        if (raddrH == idxoffset_s) { // hit
            line = offset_reg;
        } else { // no hit
            offset_reg = color512[raddrH];
            idxoffset_s = raddrH;
            line = offset_reg;
        }

        ap_uint<32> color = line.range(32 * (raddrL + 1) - 1, 32 * raddrL);
        if (color == rootIDp) {
            colorOutStrm.write(visited);
            toIDOutStrm.write(toID);
            ctrlOutStrm.write(0);
        }
    }

    ctrlOutStrm.write(1);
}

inline void backwardSCCS1Dataflow(const int vertexNum,
                                  int idxoffset_s[2],
                                  ap_uint<512> offset_reg[2],
                                  bool mode,
                                  ap_uint<32> rootID,
                                  ap_uint<32> queStatus[8],
                                  ap_uint<32>* ddrQue,

                                  ap_uint<512>* offset,
                                  ap_uint<512>* column,
                                  ap_uint<512>* color512,
                                  ap_uint<32>* result,

                                  ap_uint<32> tableStatus[3],
                                  ap_uint<32>* toIDTable) {
#pragma HLS inline off
#pragma HLS dataflow
    hls::stream<ap_uint<32> > fromIDStrm1;
#pragma HLS stream variable = fromIDStrm1 depth = 32
#pragma HLS resource variable = fromIDStrm1 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > rootIDStrm1;
#pragma HLS stream variable = rootIDStrm1 depth = 2
#pragma HLS resource variable = rootIDStrm1 core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm1;
#pragma HLS stream variable = ctrlStrm1 depth = 32
#pragma HLS resource variable = ctrlStrm1 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > rootIDStrm2;
#pragma HLS stream variable = rootIDStrm2 depth = 2
#pragma HLS resource variable = rootIDStrm2 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > offsetLowStrm;
#pragma HLS stream variable = offsetLowStrm depth = 32
#pragma HLS resource variable = offsetLowStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > offsetHighStrm;
#pragma HLS stream variable = offsetHighStrm depth = 32
#pragma HLS resource variable = offsetHighStrm core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm2;
#pragma HLS stream variable = ctrlStrm2 depth = 32
#pragma HLS resource variable = ctrlStrm2 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > rootIDStrm3;
#pragma HLS stream variable = rootIDStrm3 depth = 2
#pragma HLS resource variable = rootIDStrm3 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > toIDStrm1;
#pragma HLS stream variable = toIDStrm1 depth = 32
#pragma HLS resource variable = toIDStrm1 core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm3;
#pragma HLS stream variable = ctrlStrm3 depth = 32
#pragma HLS resource variable = ctrlStrm3 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > rootIDStrm4;
#pragma HLS stream variable = rootIDStrm4 depth = 2
#pragma HLS resource variable = rootIDStrm4 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > toIDStrm2;
#pragma HLS stream variable = toIDStrm2 depth = 32
#pragma HLS resource variable = toIDStrm2 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > colorStrm;
#pragma HLS stream variable = colorStrm depth = 32
#pragma HLS resource variable = colorStrm core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm4;
#pragma HLS stream variable = ctrlStrm4 depth = 32
#pragma HLS resource variable = ctrlStrm4 core = FIFO_LUTRAM

    loadQueue(vertexNum, queStatus, ddrQue, rootIDStrm1, fromIDStrm1, ctrlStrm1);

    loadOffset(idxoffset_s[0], offset_reg[0], rootIDStrm1, fromIDStrm1, ctrlStrm1, offset, rootIDStrm2, offsetLowStrm,
               offsetHighStrm, ctrlStrm2);

    loadColumn(rootIDStrm2, offsetLowStrm, offsetHighStrm, ctrlStrm2, column, rootIDStrm3, toIDStrm1, ctrlStrm3);

    backwardLoadRes(idxoffset_s[1], offset_reg[1], mode, rootID, rootIDStrm3, toIDStrm1, ctrlStrm3, color512, result,
                    rootIDStrm4, toIDStrm2, colorStrm, ctrlStrm4);

    storeUpdate(rootIDStrm4, toIDStrm2, colorStrm, ctrlStrm4, tableStatus, toIDTable);
}

inline void backwardStoreUpdate(ap_uint<32> rootIDp,
                                hls::stream<ap_uint<32> >& rootIDStrm,
                                hls::stream<ap_uint<32> >& toIDStrm,
                                hls::stream<ap_uint<32> >& colorInStrm,
                                hls::stream<bool>& ctrlInStrm,

                                ap_uint<32> tableStatus[3],
                                ap_uint<32>* toIDTable) {
#pragma HLS inline off
    ap_uint<32> rootID;
    ap_uint<32> cnt = 0;

    rootID = rootIDStrm.read();

    bool ctrl = ctrlInStrm.read();
    while (!ctrl) {
#pragma HLS PIPELINE II = 1
        ctrl = ctrlInStrm.read();
        ap_uint<32> toID = toIDStrm.read();
        ap_uint<32> color = colorInStrm.read();

        if (color == rootIDp) {
            toIDTable[cnt] = toID;
            cnt++;
        }
    }

    tableStatus[0] = rootID;
    tableStatus[1] = cnt;
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
    ap_uint<32> rootID = tableStatus[0];
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
void forwardSCCS2Dataflow(const int vertexNum,
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

inline void updateBWRes(hls::stream<ap_uint<32> >& rootIDStrm,
                        hls::stream<ap_uint<32> >& colorToIDStrm,
                        hls::stream<bool>& colorCtrlStrm,

                        ap_uint<32>* color32,
                        ap_uint<32>& node) {
#pragma HLS inline off
    ap_uint<32> rootID = rootIDStrm.read();

    bool ctrl = colorCtrlStrm.read();
    while (!ctrl) {
#pragma HLS PIPELINE II = 1
        ap_uint<32> toID = colorToIDStrm.read();
        ctrl = colorCtrlStrm.read();
        color32[toID] = rootID;

        node = (node > toID) ? toID : node;
    }
}

template <int MAXDEGREE>
void forwardInBWS2Dataflow(const int vertexNum,
                           ap_uint<32> tableStatus[3],
                           ap_uint<32> queStatus[8],
                           ap_uint<32>* toIDTable,

                           ap_uint<32>* color32,
                           ap_uint<32>* ddrQue,
                           ap_uint<32>& node) {
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

    updateBWRes(rootIDStrm, colorToIDStrm, colorCtrlStrm, color32, node);

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

inline void searchNewRoot(
    const int vertexNum, ap_uint<32> queStatus[8], ap_uint<32>& node, ap_uint<32>* colorMap, ap_uint<32>* queue) {
#pragma HLS inline off
    int ptr = node;

    bool f = false;
    ap_uint<32> pos = 0;
    ap_uint<32> array[64] = {0};

    while (!f) {
        for (int i = 0; i < 64; i++) {
#pragma HLS pipeline II = 1
            array[i] = colorMap[ptr + i];

            if (!f && (array[i] == (ap_uint<32>)-1 || (ptr + i == vertexNum))) {
                pos = ptr + i;
                f = true;
            }
        }

        ptr += 64;
    }

    if (pos < vertexNum) {
        writeQueStatus(vertexNum, queStatus, queue, pos);
        colorMap[pos] = pos;
    }

    node = pos;
}

inline void searchOnQueue(const int vertexNum,
                          ap_uint<32> base,
                          ap_uint<32> queStatus[8],

                          ap_uint<32>* colorMap32,
                          ap_uint<32>* result,
                          ap_uint<32>* queueG1,
                          ap_uint<32>* queueG2,

                          ap_uint<32>& offset,
                          ap_uint<32>& node) {
#pragma HLS inline off
    int ptr = base + offset;

    bool f = false;
    ap_uint<32> pos = 0;
    ap_uint<32> burstLine[64] = {0};

    while (!f) {
        for (int i = 0; i < 64; i++) {
#pragma HLS pipeline II = 1
            burstLine[i] = queueG1[ptr + i];
        }

        for (int i = 0; i < 64; i++) {
#pragma HLS pipeline II = 1
            ap_uint<32> addr = burstLine[i];

            ap_uint<32> r = 0;
            if (addr < vertexNum) r = result[addr];
            if (!f && r == (ap_uint<32>)-1) {
                pos = addr;
                node = addr;
                offset = ptr + i - base;
                f = true;
            }
        }

        ptr += 64;
    }

    writeQueStatus(vertexNum, queStatus, queueG2, pos);
    colorMap32[pos] = pos;
    result[pos] = pos + 1; // XXX
}

inline void loadQueAddr(const int vertexNum,
                        ap_uint<32> queueStatus,
                        ap_uint<32> cnt[2],
                        ap_uint<32>* queue,
                        hls::stream<int>& nodeCntStrm,
                        hls::stream<ap_uint<32> >& toIDStrm) {
#pragma HLS inline off
    const int num = (cnt[1].to_int() - cnt[0].to_int() > 0) ? (cnt[1].to_int() - 1) : 0;
    nodeCntStrm.write(num);

    int end1 = (queueStatus < cnt[0]) ? (queueStatus.to_int() + vertexNum - cnt[0].to_int())
                                      : (queueStatus.to_int() - cnt[0].to_int());
    int offset = (end1 - num < 0) ? (vertexNum - num + end1) : (end1 - num);
    int len1 = (end1 - num < 0) ? (num - end1) : num;
    int len2 = (end1 - num < 0) ? end1 : 0;

    for (int i = 0; i < len1; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<32> t = queue[offset + i];
        toIDStrm.write(t);
    }

    if (len2 > 0) {
        for (int i = 0; i < len2; i++) {
#pragma HLS PIPELINE II = 1
            ap_uint<32> t = queue[i];
            toIDStrm.write(t);
        }
    }
}

inline void loadResFlag(ap_uint<32> rootID,
                        hls::stream<int>& nodeCntStrm,
                        hls::stream<ap_uint<32> >& toIDInStrm,

                        ap_uint<32>* result,

                        hls::stream<ap_uint<32> >& resetValueStrm,
                        hls::stream<ap_uint<32> >& toIDOutStrm,
                        hls::stream<bool>& ctrOutStrm) {
#pragma HLS inline off
    int vertexNum = nodeCntStrm.read();
    resetValueStrm.write(rootID);

    for (int i = 0; i < vertexNum; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<32> toID = toIDInStrm.read();
        ap_uint<32> t = result[toID];
        if (t == (ap_uint<32>)-1) {
            toIDOutStrm.write(toID);
            ctrOutStrm.write(false);
        }
    }

    ctrOutStrm.write(true);
}

inline void resetColor(const int vertexNum,
                       ap_uint<32> rootID,
                       ap_uint<32> queStatus,
                       ap_uint<32> cnt[2],
                       ap_uint<32>* queue,
                       ap_uint<32>* result,
                       ap_uint<32>* colorMap32) {
#pragma HLS inline off
#pragma HLS dataflow
    hls::stream<int> nodeCntStrm;
#pragma HLS stream variable = nodeCntStrm depth = 2
#pragma HLS resource variable = nodeCntStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > toIDStrm0;
#pragma HLS stream variable = toIDStrm0 depth = 32
#pragma HLS resource variable = toIDStrm0 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > dataStrm;
#pragma HLS stream variable = dataStrm depth = 2
#pragma HLS resource variable = dataStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > toIDStrm1;
#pragma HLS stream variable = toIDStrm1 depth = 32
#pragma HLS resource variable = toIDStrm1 core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm;
#pragma HLS stream variable = ctrlStrm depth = 32
#pragma HLS resource variable = ctrlStrm core = FIFO_LUTRAM

    loadQueAddr(vertexNum, queStatus, cnt, queue, nodeCntStrm, toIDStrm0);

    loadResFlag(rootID, nodeCntStrm, toIDStrm0, result, dataStrm, toIDStrm1, ctrlStrm);

    updateRes(dataStrm, toIDStrm1, ctrlStrm, colorMap32);
}

inline void searchAddrGen(ap_uint<32> offset,
                          ap_uint<32> inSize,
                          ap_uint<32>* queueG1,
                          hls::stream<ap_uint<32> > toIDStrm[3],
                          hls::stream<bool> ctrlStrm[3]) {
    for (int i = 0; i < inSize; i++) {
#pragma HLS pipeline II = 1
        ap_uint<32> t = queueG1[offset + i];
        for (int j = 0; j < 3; j++) {
#pragma HLS unroll
            toIDStrm[j].write(t);
            ctrlStrm[j].write(false);
        }
    }

    ctrlStrm[0].write(true);
    ctrlStrm[1].write(true);
    ctrlStrm[2].write(true);
}

inline void searchDegree(int idxoffset_s,
                         ap_uint<512> offset_reg,
                         hls::stream<ap_uint<32> >& fromIDStrm,
                         hls::stream<bool>& ctrlInStrm,

                         ap_uint<512>* offset,

                         hls::stream<bool>& isTrimStrm) {
#pragma HLS inline off
    ap_uint<32> fromID;
    bool ctrl;

    ctrl = ctrlInStrm.read();

    while (!ctrl) {
#pragma HLS pipeline II = 1
        fromID = fromIDStrm.read();
        ctrl = ctrlInStrm.read();

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

        bool isTrim = (offsetHigh - offsetLow == 0);
        isTrimStrm.write(isTrim);
    }
}

inline void searchCompare(ap_uint<32> rootID,
                          hls::stream<ap_uint<32> >& toIDStrm,
                          hls::stream<bool> isTrimStrm[2],
                          hls::stream<bool>& ctrlInStrm,

                          ap_uint<32>* colorMap32,
                          ap_uint<32>* result,
                          ap_uint<32>& trimNum) {
    ap_uint<32> cnt = 0;

    bool ctrl = ctrlInStrm.read();

    while (!ctrl) {
#pragma HLS pipeline II = 1
        ap_uint<32> toID = toIDStrm.read();
        bool isTrim0 = isTrimStrm[0].read();
        bool isTrim1 = isTrimStrm[1].read();
        bool isTrim = isTrim0 || isTrim1;
        ctrl = ctrlInStrm.read();

        if (isTrim && toID != rootID) {
            colorMap32[toID] = toID;
            result[toID] = toID + 1; // XXX
            cnt++;
        }
    }

    trimNum = cnt;
}

inline void simpleTrim(int idxoffset_s[2],
                       ap_uint<512> offset_reg[2],
                       ap_uint<32> offset,
                       ap_uint<32> inSize,
                       ap_uint<32> rootID,

                       ap_uint<512>* offsetG1,
                       ap_uint<512>* offsetG2,
                       ap_uint<32>* colorMap32,
                       ap_uint<32>* result,
                       ap_uint<32>* queueG1,

                       ap_uint<32>& trimNum) {
#pragma HLS inline off
#pragma HLS dataflow

    hls::stream<ap_uint<32> > toIDStrm0[3];
#pragma HLS stream variable = toIDStrm0 depth = 128
#pragma HLS resource variable = toIDStrm0 core = FIFO_BRAM

    hls::stream<bool> ctrlStrm0[3];
#pragma HLS stream variable = ctrlStrm0 depth = 128
#pragma HLS resource variable = ctrlStrm0 core = FIFO_LUTRAM

    hls::stream<bool> isTrimStrm[2];
#pragma HLS stream variable = isTrimStrm depth = 32
#pragma HLS resource variable = isTrimStrm core = FIFO_LUTRAM

    searchAddrGen(offset, inSize, queueG1, toIDStrm0, ctrlStrm0);

    searchDegree(idxoffset_s[0], offset_reg[0], toIDStrm0[0], ctrlStrm0[0], offsetG1, isTrimStrm[0]);

    searchDegree(idxoffset_s[1], offset_reg[1], toIDStrm0[1], ctrlStrm0[1], offsetG2, isTrimStrm[1]);

    searchCompare(rootID, toIDStrm0[2], isTrimStrm, ctrlStrm0[2], colorMap32, result, trimNum);
}

inline void activateBackward(ap_uint<32> queStatus,
                             ap_uint<32> cnt,
                             hls::stream<ap_uint<32> >& data_strm,
                             hls::stream<bool>& e_strm) {
#pragma HLS inline off
    data_strm.write(queStatus);
    data_strm.write(cnt);
    e_strm.write(false);
    e_strm.write(false);
    e_strm.write(false);
}

inline void initBuffer(const int vertexNum, ap_uint<32>* color32, ap_uint<32>* result) {
    for (int i = 0; i < vertexNum; i++) {
#pragma HLS PIPELINE II = 1
        color32[i] = (ap_uint<32>)-1;
        result[i] = (ap_uint<32>)-1;
    }
}

template <int MAXDEGREE>
void forwardSCC(const int vertexNum,
                ap_uint<512>* columnG1,
                ap_uint<512>* offsetG1,
                ap_uint<32>* queue,
                ap_uint<512>* colorMap512,
                ap_uint<32>* colorMap32,
                hls::stream<ap_uint<32> >& data_strm,
                hls::stream<bool>& e_strm) {
#pragma HLS inline off
    int idxoffset_s[2] = {-1, -1};
#pragma HLS ARRAY_PARTITION variable = idxoffset_s complete dim = 1
    ap_uint<512> offset_reg[2];
#pragma HLS ARRAY_PARTITION variable = offset_reg complete dim = 1

    ap_uint<32> tableStatus[3];
#pragma HLS ARRAY_PARTITION variable = tableStatus complete dim = 1

    tableStatus[0] = 0;
    tableStatus[1] = 0;
    tableStatus[2] = 0;

    ap_uint<32> queStatus[8];
#pragma HLS ARRAY_PARTITION variable = queStatus complete dim = 1

    queStatus[0] = 0;
    queStatus[1] = 0;
    queStatus[2] = 1;
    queStatus[3] = 0;
    queStatus[4] = 0;
    queStatus[5] = 1;
    queStatus[6] = 0;
    queStatus[7] = 0;

#ifndef __SYNTHESIS__
    ap_uint<32>* toIDTable = (ap_uint<32>*)malloc(MAXDEGREE * sizeof(ap_uint<32>));
#else
    ap_uint<32> toIDTable[MAXDEGREE];
#pragma HLS RESOURCE variable = toIDTable core = RAM_2P_URAM
#endif

    ap_uint<32> node = 0;

    queue[0] = 0;
    colorMap32[0] = 0;

    while (node != vertexNum) {
        ap_uint<64> que_rd_cnt;
        ap_uint<64> que_wr_cnt;
        que_rd_cnt = queStatus[1].concat(queStatus[0]);
        que_wr_cnt = queStatus[3].concat(queStatus[2]);
        ap_uint<32> cnt = 0;

        while (que_rd_cnt != que_wr_cnt) {
            forwardSCCS1Dataflow(vertexNum, idxoffset_s, offset_reg, queStatus, queue, offsetG1, columnG1, colorMap512,
                                 tableStatus, toIDTable);

            forwardSCCS2Dataflow<MAXDEGREE>(vertexNum, tableStatus, queStatus, toIDTable, colorMap32, queue);

            que_rd_cnt = queStatus[1].concat(queStatus[0]);
            que_wr_cnt = queStatus[3].concat(queStatus[2]);
            cnt += tableStatus[1];
        }

        cnt++; // contain the root node
        activateBackward(queStatus[7], cnt, data_strm, e_strm);
        searchNewRoot(vertexNum, queStatus, node, colorMap32, queue);
    }
}

template <int MAXDEGREE>
void forwardSCCInBackward(const int vertexNum,
                          ap_uint<32> rootID,
                          ap_uint<32> queStatus[8],
                          ap_uint<512>* columnG1,
                          ap_uint<512>* offsetG1,
                          ap_uint<32>* queue,
                          ap_uint<32>* result,
                          ap_uint<512>* colorMap512,
                          ap_uint<32>* colorMap32,
                          ap_uint<32>& nodeCnt,
                          ap_uint<32>& node) {
#pragma HLS inline off
    int idxoffset_s1[2] = {-1, -1};
#pragma HLS ARRAY_PARTITION variable = idxoffset_s1 complete dim = 1
    ap_uint<512> offset_reg_s1[2];
#pragma HLS ARRAY_PARTITION variable = offset_reg_s1 complete dim = 1

    ap_uint<32> tableStatus[3];
#pragma HLS ARRAY_PARTITION variable = tableStatus complete dim = 1

    tableStatus[0] = 0;
    tableStatus[1] = 0;
    tableStatus[2] = 0;

#ifndef __SYNTHESIS__
    ap_uint<32>* toIDTable = (ap_uint<32>*)malloc(MAXDEGREE * sizeof(ap_uint<32>));
#else
    ap_uint<32> toIDTable[MAXDEGREE];
#pragma HLS RESOURCE variable = toIDTable core = RAM_2P_URAM
#endif

    ap_uint<32> nodeCnt_t = 0;

    ap_uint<64> que_rd_cnt;
    ap_uint<64> que_wr_cnt;
    que_rd_cnt = queStatus[1].concat(queStatus[0]);
    que_wr_cnt = queStatus[3].concat(queStatus[2]);

    while (que_rd_cnt != que_wr_cnt) {
        backwardSCCS1Dataflow(vertexNum, idxoffset_s1, offset_reg_s1, false, rootID, queStatus, queue, offsetG1,
                              columnG1, colorMap512, result, tableStatus, toIDTable);

        forwardInBWS2Dataflow<MAXDEGREE>(vertexNum, tableStatus, queStatus, toIDTable, colorMap32, queue, node);

        que_rd_cnt = queStatus[1].concat(queStatus[0]);
        que_wr_cnt = queStatus[3].concat(queStatus[2]);
        nodeCnt_t += tableStatus[1];
    }

    nodeCnt = nodeCnt_t + 1;
}

template <int MAXDEGREE>
void backwardSCC(const int vertexNum,
                 hls::stream<ap_uint<32> >& data_strm,
                 hls::stream<bool>& e_strm,

                 ap_uint<512>* columnG1, // CSR
                 ap_uint<512>* offsetG1,
                 ap_uint<512>* columnG2, // CSC
                 ap_uint<512>* offsetG2,

                 ap_uint<32>* queueG1, // FW
                 ap_uint<32>* queueG2, // BW
                 ap_uint<512>* colorMap512,
                 ap_uint<32>* colorMap32,

                 ap_uint<32>* result) {
#pragma HLS inline off
    int idxoffset_s1[2] = {-1, -1};
#pragma HLS ARRAY_PARTITION variable = idxoffset_s1 complete dim = 1
    ap_uint<512> offset_reg_s1[2];
#pragma HLS ARRAY_PARTITION variable = offset_reg_s1 complete dim = 1
    int idxoffset_s2[2] = {-1, -1};
#pragma HLS ARRAY_PARTITION variable = idxoffset_s2 complete dim = 1
    ap_uint<512> offset_reg_s2[2];
#pragma HLS ARRAY_PARTITION variable = offset_reg_s2 complete dim = 1

#ifndef __SYNTHESIS__
    ap_uint<32>* toIDTable = (ap_uint<32>*)malloc(MAXDEGREE * sizeof(ap_uint<32>));
#else
    ap_uint<32> toIDTable[MAXDEGREE];
#pragma HLS RESOURCE variable = toIDTable core = RAM_2P_URAM
#endif

    ap_uint<32> tableStatus[3];
#pragma HLS ARRAY_PARTITION variable = tableStatus complete dim = 1

    ap_uint<32> queStatus[8];
#pragma HLS ARRAY_PARTITION variable = queStatus complete dim = 1
    ap_uint<32> queStatus_tmp[8];
#pragma HLS ARRAY_PARTITION variable = queStatus_tmp complete dim = 1

    queStatus[0] = 0;
    queStatus[1] = 0;
    queStatus[2] = 0;
    queStatus[3] = 0;
    queStatus[4] = 0;
    queStatus[5] = 0;
    queStatus[6] = 0;

    ap_uint<32> cnt2 = 0;

    while (cnt2 != vertexNum) {
        ap_uint<32> rootID = data_strm.read();
        ap_uint<32> inSize = data_strm.read();
        bool e = e_strm.read();
        e = e_strm.read();
        e = e_strm.read();

        writeQueStatus(vertexNum, queStatus, queueG2, rootID);

        result[rootID] = rootID + 1; // XXX
        ap_uint<32> simpleTrimNum = 0;
        if (inSize > 1) // do simple Triming, only trim nodes with zero indegree/outdegree
            simpleTrim(idxoffset_s1, offset_reg_s1, cnt2, inSize, rootID, offsetG1, offsetG2, colorMap32, result,
                       queueG1, simpleTrimNum);

        ap_uint<32> cnt[2] = {0, 0};
        ap_uint<32> cnt0 = 0;
        ap_uint<32> cnt1 = simpleTrimNum;
        ap_uint<32> offset = 0;

        while (cnt1 < inSize) { // process all node in current color region
            ap_uint<32> cnt_tmp = 0;
            ap_uint<32> node[2] = {-1, -1};
            ap_uint<32> rootIDp = rootID;

            // Enter after once BW-BFS
            if (cnt1 > simpleTrimNum) {
                // Reset color in the sub-graph
                resetColor(vertexNum, rootID, queStatus[4], cnt, queueG2, result, colorMap32);

                searchOnQueue(vertexNum, cnt2, queStatus, colorMap32, result, queueG1, queueG2, offset, node[0]);

                // Find the new root node and do FW-BFS
                int exitCnt = 0;
                do {
                    forwardSCCInBackward<MAXDEGREE>(vertexNum, rootIDp, queStatus, columnG1, offsetG1, queueG2, result,
                                                    colorMap512, colorMap32, cnt0,
                                                    node[1]); // BFS in the same color and result=-1

                    if (cnt0 > 1 && node[1] < node[0] && exitCnt < 1) {
                        writeQueStatus(vertexNum, queStatus, queueG2, node[1]);
                        result[node[0]] = -1;

                        result[node[1]] = node[1] + 1;
                        colorMap32[node[1]] = node[1];
                        rootIDp = node[0]; // go into new sub-graph

                        // copy the QueStatus
                        for (int i = 0; i < 8; i++) {
#pragma HLS unroll
                            queStatus_tmp[i] = queStatus[i];
                        }
                    }

                    if (exitCnt == 0) cnt[1] = cnt0;
                    exitCnt++;
                } while (cnt0 > 1 && node[1] < node[0] && exitCnt < 2);

                if (exitCnt == 1 && node[1] >= node[0]) writeQueStatus(vertexNum, queStatus, queueG2, node[0]);

                // Need recovery only after once copy, and reset the rd/wr pointer in the first FW-BFS
                if (exitCnt == 2) {
                    for (int i = 0; i < 8; i++) {
#pragma HLS unroll
                        queStatus[i] = queStatus_tmp[i];
                    }
                }
            }

            ap_uint<64> que_rd_cnt;
            ap_uint<64> que_wr_cnt;
            que_rd_cnt = queStatus[1].concat(queStatus[0]);
            que_wr_cnt = queStatus[3].concat(queStatus[2]);
            while (que_rd_cnt != que_wr_cnt) { // find one SCC
                backwardSCCS1Dataflow(vertexNum, idxoffset_s2, offset_reg_s2, true, queStatus[7], queStatus, queueG2,
                                      offsetG2, columnG2, colorMap512, result, tableStatus, toIDTable);

                forwardSCCS2Dataflow<MAXDEGREE>(vertexNum, tableStatus, queStatus, toIDTable, result, queueG2);

                que_rd_cnt = queStatus[1].concat(queStatus[0]);
                que_wr_cnt = queStatus[3].concat(queStatus[2]);
                cnt_tmp += tableStatus[1];
            }

            cnt_tmp = cnt_tmp + 1; // contain the root node

            cnt1 += cnt_tmp;
            cnt[0] = cnt_tmp;
        }

        cnt2 += inSize;
    }
}

template <int MAXDEGREE>
void stronglyConnectedComponentImpl(const int vertexNum,

                                    ap_uint<512>* columnG1,
                                    ap_uint<512>* offsetG1,
                                    ap_uint<512>* columnG2,
                                    ap_uint<512>* offsetG2,
                                    ap_uint<512>* columnG3,
                                    ap_uint<512>* offsetG3,

                                    ap_uint<512>* colorMap512G1,
                                    ap_uint<32>* colorMap32G1,
                                    ap_uint<32>* queueG1,
                                    ap_uint<512>* colorMap512G2,
                                    ap_uint<32>* colorMap32G2,
                                    ap_uint<32>* queueG2,
                                    ap_uint<32>* queueG3,

                                    ap_uint<32>* result) {
#pragma HLS inline off
#pragma HLS dataflow

    hls::stream<ap_uint<32> > data_strm;
#pragma HLS stream variable = data_strm depth = 2
    hls::stream<bool> e_strm;
#pragma HLS stream variable = e_strm depth = 2

    forwardSCC<MAXDEGREE>(vertexNum, columnG1, offsetG1, queueG1, colorMap512G1, colorMap32G1, data_strm, e_strm);

    backwardSCC<MAXDEGREE>(vertexNum, data_strm, e_strm, columnG3, offsetG3, columnG2, offsetG2, queueG3, queueG2,
                           colorMap512G2, colorMap32G2, result);
}
} // namespace scc
} // namespace internal

/**
 * @brief stronglyConnectedComponents Compute the strongly connected component membership of each vertex only for
 * directed graph, and label each vertex with one value containing the lowest vertex id in the SCC containing
 * that vertex.
 *
 * @tparam MAXVERTEX CSC/CSR data vertex(offset) array maxsize
 * @tparam MAXEDGE CSC/CSR data edge(indice) array maxsize
 * @tparam LOG2CACHEDEPTH cache depth in Binary, the cache onchip memory is 512 bit x uramRow
 * @tparam LOG2DATAPERCACHELINE number of data in one 512bit in Binary, for double, it's 3, for float, it's 4
 * @tparam RAMTYPE flag to tell use URAM LUTRAM or BRAM, 0 : LUTRAM, 1 : URAM, 2 : BRAM
 * @tparam MAXOUTDEGREE the max indegree or outdegree of the input graph supported. Large value will result in more
 * URAM usage
 *
 * @param edgeNum edge number of the input graph
 * @param vertexNum vertex number of the input graph
 * @param indexCSR0 column index of CSR format
 * @param offsetCSR0 row offset of CSR format
 * @param indxeCSC512 column index of CSC format in 512bit, which should be shared the same buffer with indexCSC32 in
 * host
 * @param indexCSC32 column index of CSC format in 32bit, which should be shared the same buffer with indxeCSC512 in
 * host
 * @param offsetCSC row offset of CSC format
 * @param indexCSR1 column index of CSR format, which should be shared the same buffer with indexCSR0 in host
 * @param offsetCSR1 row offset of CSR format, which should be shared the same buffer with offsetCSR0 in host
 * @param offsetCSCTmp1 temp row offset for CSR2CSC convert
 * @param offsetCSCTmp2 temp row offset of CSR2CSC convert
 *
 * @param colorCSR0512 intermediate color map in 512bit for forward-BFS
 * @param colorCSR032 intermediate color map in 32bit for forward-BFS, which should be shared the same buffer with
 * colorCSR0512 in host
 * @param queueCSR0 intermediate queue used during the internal forward-BFS
 *
 * @param colorCSC512 intermediate color map in 512bit for backward-BFS, which should be shared the same buffer with
 * colorCSR0512 in host
 * @param colorCSC32 intermediate color map in 32bit for backward-BFS, which should be shared the same buffer with
 * colorCSR0512 in host
 * @param queueCSC intermediate queue used during the internal backward-BFS
 * @param queueCSR1 intermediate queue which should be shared the same buffer with queueCSR0
 *
 * @param component return component buffer with the vertex label containing the lowest vertex id in the strongly
 * connnected
 * component containing that vertex
 *
 */
template <int MAXVERTEX, int MAXEDGE, int LOG2CACHEDEPTH, int LOG2DATAPERCACHELINE, int RAMTYPE, int MAXOUTDEGREE>
void stronglyConnectedComponents(const int edgeNum,
                                 const int vertexNum,
                                 ap_uint<512>* offsetCSR0,
                                 ap_uint<512>* indexCSR0,
                                 ap_uint<512>* offsetCSC,
                                 ap_uint<512>* indxeCSC512,
                                 ap_uint<32>* indexCSC32,
                                 ap_uint<512>* offsetCSR1,
                                 ap_uint<512>* indexCSR1,

                                 ap_uint<512>* offsetCSCTmp1,
                                 ap_uint<512>* offsetCSCTmp2,

                                 ap_uint<512>* colorCSR0512,
                                 ap_uint<32>* colorCSR032,
                                 ap_uint<32>* queueCSR0,
                                 ap_uint<512>* colorCSC512,
                                 ap_uint<32>* colorCSC32,
                                 ap_uint<32>* queueCSC,
                                 ap_uint<32>* queueCSR1,

                                 ap_uint<32>* component) {
#pragma HLS inline off

    convertCsrCsc<ap_uint<32>, MAXVERTEX, MAXEDGE, LOG2CACHEDEPTH, LOG2DATAPERCACHELINE, RAMTYPE>(
        edgeNum, vertexNum, offsetCSR0, indexCSR0, offsetCSC, indexCSC32, offsetCSCTmp1, offsetCSCTmp2);

    internal::scc::initBuffer(vertexNum, colorCSR032, component);

    internal::scc::stronglyConnectedComponentImpl<MAXOUTDEGREE>(
        vertexNum, indexCSR0, offsetCSR0, indxeCSC512, offsetCSC, indexCSR1, offsetCSR1, colorCSR0512, colorCSR032,
        queueCSR0, colorCSC512, colorCSC32, queueCSC, queueCSR1, component);
}

} // namespace graph
} // namespace xf
#endif
