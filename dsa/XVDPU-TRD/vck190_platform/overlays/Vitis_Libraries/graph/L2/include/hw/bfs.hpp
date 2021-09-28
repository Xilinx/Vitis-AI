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

#ifndef _XF_GRAPH_BFS_HPP_
#define _XF_GRAPH_BFS_HPP_

#include "ap_int.h"
#include "hls_stream.h"

namespace xf {
namespace graph {
namespace internal {
namespace bfs {

static bool que_valid;
static ap_uint<512> que_reg;
static ap_uint<32> que_addr;

static bool column_valid;
static ap_uint<512> column_reg;
static ap_uint<32> column_addr;

static bool result_valid;
static ap_uint<512> result_reg;
static ap_uint<32> result_addr;

inline void loadQueue(const int vertexNum,
                      ap_uint<32> queStatus[8],
                      ap_uint<32>& tableStatus,
                      ap_uint<512>* ddrQue,

                      hls::stream<ap_uint<32> >& fromIDStrm,
                      hls::stream<bool>& ctrlStrm) {
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
        tableStatus = fromID;

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

inline void loadColumn(hls::stream<ap_uint<32> >& raddrStrm,
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

        if (column_valid == 0 || idxHc != column_addr) {
            column_valid = 1;
            column_addr = idxHc;
            column_reg = column[idxHc];
        }

        toIDStrm.write(column_reg.range(32 * (idxLc + 1) - 1, 32 * idxLc));
        ctrlOutStrm.write(0);
        ctrl = ctrlInStrm.read();
    }

    ctrlOutStrm.write(1);
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

                        ap_uint<32>& tableStatus,
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

    tableStatus = cnt;
}

inline void BFSS1Dataflow(const int vertexNum,
                          int idxoffset_s,
                          ap_uint<512> offset_reg,
                          ap_uint<32> queStatus[8],
                          ap_uint<512>* ddrQue,

                          ap_uint<512>* offsetG1,
                          ap_uint<512>* columnG1,

                          ap_uint<512>* color512,

                          ap_uint<32> tableStatus[6],
                          ap_uint<32>* toIDTable) {
#pragma HLS inline off
#pragma HLS dataflow
    hls::stream<ap_uint<32> > fromIDStrm;
#pragma HLS stream variable = fromIDStrm depth = 32
#pragma HLS resource variable = fromIDStrm core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm0;
#pragma HLS stream variable = ctrlStrm0 depth = 32
#pragma HLS resource variable = ctrlStrm0 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > offsetLowStrm;
#pragma HLS stream variable = offsetLowStrm depth = 32
#pragma HLS resource variable = offsetLowStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > offsetHighStrm;
#pragma HLS stream variable = offsetHighStrm depth = 32
#pragma HLS resource variable = offsetHighStrm core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm2;
#pragma HLS stream variable = ctrlStrm2 depth = 32
#pragma HLS resource variable = ctrlStrm2 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > toIDStrm1;
#pragma HLS stream variable = toIDStrm1 depth = 32
#pragma HLS resource variable = toIDStrm1 core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm3;
#pragma HLS stream variable = ctrlStrm3 depth = 32
#pragma HLS resource variable = ctrlStrm3 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > toIDStrm2;
#pragma HLS stream variable = toIDStrm2 depth = 32
#pragma HLS resource variable = toIDStrm2 core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm4;
#pragma HLS stream variable = ctrlStrm4 depth = 32
#pragma HLS resource variable = ctrlStrm4 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > toIDStrm3;
#pragma HLS stream variable = toIDStrm3 depth = 32
#pragma HLS resource variable = toIDStrm3 core = FIFO_LUTRAM

    hls::stream<bool> ctrlStrm5;
#pragma HLS stream variable = ctrlStrm5 depth = 32
#pragma HLS resource variable = ctrlStrm5 core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > colorStrm;
#pragma HLS stream variable = colorStrm depth = 32
#pragma HLS resource variable = colorStrm core = FIFO_LUTRAM

    // load one node from queue
    loadQueue(vertexNum, queStatus, tableStatus[0], ddrQue, fromIDStrm, ctrlStrm0);

    // read CSR Graph offset
    loadOffset(idxoffset_s, offset_reg, fromIDStrm, ctrlStrm0, offsetG1, offsetLowStrm, offsetHighStrm, ctrlStrm2);

    // get address for read column
    loadColumnSendAddr(offsetLowStrm, offsetHighStrm, ctrlStrm2, toIDStrm1, ctrlStrm3);

    // read CSR Graph column
    loadColumn(toIDStrm1, ctrlStrm3, columnG1, toIDStrm2, ctrlStrm4);

    // load unvisited node
    loadRes(toIDStrm2, ctrlStrm4, color512, toIDStrm3, colorStrm, ctrlStrm5);

    storeUpdate(toIDStrm3, colorStrm, ctrlStrm5, tableStatus[1], toIDTable);
}

template <int MAXDEGREE>
void updateDispatch(ap_uint<32>& visitor,
                    ap_uint<32> tableStatus[6],
                    ap_uint<32>* toIDTable,

                    hls::stream<ap_uint<32> >& rootIDStrm,
                    hls::stream<ap_uint<32> >& colorToIDStrm,
                    hls::stream<ap_uint<32> >& levelToIDStrm,
                    hls::stream<bool>& colorCtrlStrm,
                    hls::stream<ap_uint<32> >& queToIDStrm,
                    hls::stream<bool>& queCtrlStrm) {
#pragma HLS inline off
    ap_uint<32> level = visitor;

    ap_uint<32> rootID = tableStatus[0];
    ap_uint<32> tableCnt = tableStatus[1];
    ap_uint<32> cnt_cur = tableStatus[2];
    ap_uint<32> level_len = tableStatus[3];
    ap_uint<32> cnt_nxt = tableStatus[4];

    rootIDStrm.write(rootID);
    levelToIDStrm.write(level);

    if (tableCnt > MAXDEGREE) tableStatus[5] = 1;

    if (cnt_cur < level_len - 1) {
        tableStatus[2] = cnt_cur + 1;
        tableStatus[4] = cnt_nxt + tableCnt;
    } else {
        tableStatus[2] = 0;                  // reset the current counter
        tableStatus[3] = cnt_nxt + tableCnt; // for next level
        tableStatus[4] = 0;                  // add the first outdegree
        level++;
    }

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
    visitor = level;
}

inline void updateRes(ap_uint<32>& visitor,
                      hls::stream<ap_uint<32> >& rootIDStrm,
                      hls::stream<ap_uint<32> >& colorToIDStrm,
                      hls::stream<ap_uint<32> >& levelToIDStrm,
                      hls::stream<bool>& colorCtrlStrm,

                      ap_uint<32>* result_dt,
                      ap_uint<32>* result_ft,
                      ap_uint<32>* result_pt,
                      ap_uint<32>* result_lv) {
#pragma HLS inline off

    ap_uint<32> dtime = visitor;

    ap_uint<32> rootID = rootIDStrm.read();
    ap_uint<32> level = levelToIDStrm.read();
    bool ctrl = colorCtrlStrm.read();
    while (!ctrl) {
#pragma HLS PIPELINE II = 1
        ap_uint<32> toID = colorToIDStrm.read();
        result_dt[toID] = dtime++;
        result_lv[toID] = level;
        result_pt[toID] = rootID;

        if (toID(31, 4) == result_addr) result_valid = 0;

        ctrl = colorCtrlStrm.read();
    }

    result_ft[rootID] = dtime++;
    visitor = dtime;
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
void BFSS2Dataflow(const int vertexNum,
                   ap_uint<32> visitor[2],
                   ap_uint<32> tableStatus[6],
                   ap_uint<32> queStatus[8],
                   ap_uint<32>* toIDTable,

                   ap_uint<32>* ddrQue,

                   ap_uint<32>* result_dt,
                   ap_uint<32>* result_ft,
                   ap_uint<32>* result_pt,
                   ap_uint<32>* result_lv) {
#pragma HLS inline off
#pragma HLS dataflow

    hls::stream<ap_uint<32> > rootIDStrm;
#pragma HLS stream variable = rootIDStrm depth = 2
#pragma HLS resource variable = rootIDStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > levelToIDStrm;
#pragma HLS stream variable = levelToIDStrm depth = 2
#pragma HLS resource variable = levelToIDStrm core = FIFO_LUTRAM

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

    updateDispatch<MAXDEGREE>(visitor[0], tableStatus, toIDTable, rootIDStrm, colorToIDStrm, levelToIDStrm,
                              colorCtrlStrm, queToIDStrm, queCtrlStrm);

    updateRes(visitor[1], rootIDStrm, colorToIDStrm, levelToIDStrm, colorCtrlStrm, result_dt, result_ft, result_pt,
              result_lv);

    upadateQue(vertexNum, queToIDStrm, queCtrlStrm, queStatus, ddrQue);
}

inline void initBuffer(const int depth, ap_uint<512>* result) {
    for (int i = 0; i < depth; i++) {
#pragma HLS PIPELINE II = 1
        result[i] = (ap_uint<512>)-1;
    }
}

template <int MAXDEGREE>
void bfsImpl(const int srcID,
             const int vertexNum,
             ap_uint<512>* columnG1,
             ap_uint<512>* offsetG1,

             ap_uint<512>* queue512,
             ap_uint<32>* queue,
             ap_uint<512>* color512,

             ap_uint<32>* result_dt,
             ap_uint<32>* result_ft,
             ap_uint<32>* result_pt,
             ap_uint<32>* result_lv) {
#pragma HLS inline off

    int idxoffset_s = -1;
    ap_uint<512> offset_reg;

    ap_uint<32> visitor[2] = {1, 1};
#pragma HLS ARRAY_PARTITION variable = visitor complete dim = 1

    ap_uint<32> queStatus[8];
#pragma HLS ARRAY_PARTITION variable = queStatus complete dim = 1

    ap_uint<32> tableStatus[6];
#pragma HLS ARRAY_PARTITION variable = tableStatus complete dim = 1

#ifndef __SYNTHESIS__
    ap_uint<32>* toIDTable = (ap_uint<32>*)malloc(MAXDEGREE * sizeof(ap_uint<32>));
#else
    ap_uint<32> toIDTable[MAXDEGREE];
#pragma HLS RESOURCE variable = toIDTable core = RAM_2P_URAM
#endif

    que_valid = 0;
    column_valid = 0;
    result_valid = 0;

    tableStatus[0] = 0;
    tableStatus[1] = 0;
    tableStatus[2] = 0;
    tableStatus[3] = 1;
    tableStatus[4] = 0;
    tableStatus[5] = 0; // overflow info

    queue[0] = srcID;         // push into queue
    result_dt[srcID] = 0;     // color into visited
    result_pt[srcID] = srcID; // parent node
    result_lv[srcID] = 0;     // distance value

    queStatus[0] = 0;
    queStatus[1] = 0;
    queStatus[2] = 1;
    queStatus[3] = 0;
    queStatus[4] = 0;
    queStatus[5] = 1;
    queStatus[6] = 0;

    ap_uint<64> que_rd_cnt;
    ap_uint<64> que_wr_cnt;
    que_rd_cnt = queStatus[1].concat(queStatus[0]);
    que_wr_cnt = queStatus[3].concat(queStatus[2]);

    while (que_rd_cnt != que_wr_cnt) {
        BFSS1Dataflow(vertexNum, idxoffset_s, offset_reg, queStatus, queue512, offsetG1, columnG1, color512,
                      tableStatus, toIDTable);

        BFSS2Dataflow<MAXDEGREE>(vertexNum, visitor, tableStatus, queStatus, toIDTable, queue, result_dt, result_ft,
                                 result_pt, result_lv);

        que_rd_cnt = queStatus[1].concat(queStatus[0]);
        que_wr_cnt = queStatus[3].concat(queStatus[2]);
    }
}
} // namespace bfs
} // namespace internal

/**
 * @brief bfs Implement the directed graph traversal by breath-first search algorithm
 *
 * @tparam MAXOUTDEGREE the maximum outdegree of input graphs. Large value will result in more URAM usage.
 *
 * @param srcID the source vertex ID for this search, starting from 0
 * @param numVertex vertex number of the input graph
 * @param indexCSR column index of CSR format
 * @param offsetCSR row offset of CSR format
 * @param color512 intermediate color map which should be shared with dtime, pred or distance in host.
 * @param queue32 intermediate queue32 used during the BFS
 * @param dtime the result of discovery time stamp for each vertex
 * @param ftime the result of finish time stamp for each vertex
 * @param pred the result of parent index of each vertex
 * @param distance the distance result from given source vertex for each vertex
 *
 */
template <int MAXOUTDEGREE>
void bfs(const int srcID,
         const int numVertex,

         ap_uint<512>* offsetCSR,
         ap_uint<512>* indexCSR,

         ap_uint<512>* queue512,
         ap_uint<32>* queue32,
         ap_uint<512>* color512,

         ap_uint<32>* dtime,
         ap_uint<32>* ftime,
         ap_uint<32>* pred,
         ap_uint<32>* distance) {
#pragma HLS inline off

    const int depth = (numVertex + 15) / 16;

    internal::bfs::initBuffer(depth, color512);

    internal::bfs::bfsImpl<MAXOUTDEGREE>(srcID, numVertex, indexCSR, offsetCSR, queue512, queue32, color512, dtime,
                                         ftime, pred, distance);
}

} // namespace graph
} // namespace xf
#endif
