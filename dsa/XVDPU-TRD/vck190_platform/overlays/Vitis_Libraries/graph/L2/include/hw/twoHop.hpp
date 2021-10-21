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

#ifndef __XF_GRAPH_TWOHOP_HPP_
#define __XF_GRAPH_TWOHOP_HPP_

#include <ap_int.h>
#include <hls_stream.h>

namespace xf {
namespace graph {
namespace internal {

inline void load_pair(ap_uint<32> numPairs,
                      ap_uint<64>* pair,
                      hls::stream<ap_uint<32> >& srcStream,
                      hls::stream<ap_uint<32> >& desStream) {
    for (unsigned i = 0; i < numPairs.range(31, 9); i++) {
        for (unsigned j = 0; j < 512; j++) {
#pragma HLS PIPELINE II = 1
            ap_uint<64> tmp = pair[i * 512 + j];
            srcStream.write(tmp.range(63, 32));
            desStream.write(tmp.range(31, 0));
        }
    }

    for (unsigned i = 0; i < numPairs.range(8, 0); i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<64> tmp = pair[numPairs.range(31, 9) * 512 + i];
        srcStream.write(tmp.range(63, 32));
        desStream.write(tmp.range(31, 0));
    }
}

inline void loadOneHopOffset(ap_uint<32> numPairs,
                             hls::stream<ap_uint<32> >& srcStream,
                             hls::stream<ap_uint<32> >& desStream,
                             unsigned* offset,
                             hls::stream<ap_uint<64> >& offsetStream,
                             hls::stream<ap_uint<32> >& desOutStream) {
    for (unsigned i = 0; i < numPairs; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<64> tmp;
        ap_uint<32> idx = srcStream.read();
        tmp.range(63, 32) = offset[idx];
        tmp.range(31, 0) = offset[idx + 1];
        offsetStream.write(tmp);
        desOutStream.write(desStream.read());
    }
}

inline void loadOneHopIndex(ap_uint<32> numPairs,
                            hls::stream<ap_uint<64> >& offsetStream,
                            hls::stream<ap_uint<32> >& desStream,
                            unsigned* index,
                            hls::stream<ap_uint<32> >& indexStream) {
    ap_uint<32> idx;

    for (unsigned i = 0; i < numPairs; i++) {
        ap_uint<64> tmp = offsetStream.read();
        ap_uint<32> begin = tmp.range(63, 32);
        ap_uint<32> end = tmp.range(31, 0);
        ap_uint<32> des = desStream.read();
        idx[31] = 1;
        idx.range(30, 0) = des.range(30, 0);
        indexStream.write(idx);
        for (unsigned j = begin; j < end; j++) {
#pragma HLS PIPELINE II = 1
            indexStream.write(index[j]);
        }
    }

    idx[31] = 1;
    indexStream.write(idx);
}

inline void loadTwoHopOffset(ap_uint<32> numPairs,
                             hls::stream<ap_uint<32> >& indexStream,
                             unsigned* offset,
                             hls::stream<ap_uint<64> >& offsetStream) {
    unsigned cnt = 0;

    while (cnt < numPairs + 1) {
#pragma HLS PIPELINE II = 1
        ap_uint<32> idx = indexStream.read();
        if (idx[31] != 1) {
            ap_uint<64> tmp;
            tmp.range(63, 32) = offset[idx];
            tmp.range(31, 0) = offset[idx + 1];
            offsetStream.write(tmp);
        } else {
            ap_uint<64> reg;
            reg.range(63, 32) = idx;
            offsetStream.write(reg);
            cnt++;
        }
    }
}

inline void loadTwoHopIndex(ap_uint<32> numPairs,
                            hls::stream<ap_uint<64> >& offsetStream,
                            unsigned* index,
                            hls::stream<ap_uint<32> >& indexStream) {
    unsigned cnt = 0;

    while (cnt < numPairs + 1) {
        ap_uint<64> tmp = offsetStream.read();
        if (tmp[63] != 1) {
            ap_uint<32> begin = tmp.range(63, 32);
            ap_uint<32> end = tmp.range(31, 0);
            for (unsigned j = begin; j < end; j++) {
#pragma HLS PIPELINE II = 1
                indexStream.write(index[j]);
            }
        } else {
            ap_uint<32> reg = tmp.range(63, 32);
            indexStream.write(reg);
            cnt++;
        }
    }
}

inline void counter(ap_uint<32> numPairs,
                    hls::stream<ap_uint<32> >& indexStream,
                    hls::stream<ap_uint<32> >& resStream) {
    unsigned cnt = 0;
    unsigned res = 0;

    ap_uint<32> idx = indexStream.read();
    idx[31] = 0;
    ap_uint<32> des = idx;

    while (cnt < numPairs) {
        while (idx[31] != 1) {
#pragma HLS PIPELINE II = 1
            idx = indexStream.read();
            if (idx[31] != 1 && idx == des) {
                res++;
            }
        }
        idx[31] = 0;
        des = idx;
        resStream.write(res);
        res = 0;
        cnt++;
    }
}

inline void writeOut(ap_uint<32> numPairs, hls::stream<ap_uint<32> >& resStream, unsigned* cnt_res) {
    for (unsigned i = 0; i < numPairs; i++) {
#pragma HLS PIPELINE II = 1
        cnt_res[i] = resStream.read();
    }
}

inline void twoHopCore(ap_uint<32> numPairs,
                       ap_uint<64>* pair,

                       unsigned* offsetOneHop,
                       unsigned* indexOneHop,
                       unsigned* offsetTwoHop,
                       unsigned* indexTwoHop,

                       unsigned* cnt_res) {
#pragma HLS DATAFLOW

    hls::stream<ap_uint<32> > srcStream;
#pragma HLS stream variable = srcStream depth = 32
#pragma HLS resource variable = srcStream core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > desStream;
#pragma HLS stream variable = desStream depth = 32
#pragma HLS resource variable = desStream core = FIFO_LUTRAM

    load_pair(numPairs, pair, srcStream, desStream);

    hls::stream<ap_uint<64> > offsetOneHopStream;
#pragma HLS stream variable = offsetOneHopStream depth = 32
#pragma HLS resource variable = offsetOneHopStream core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > desOneHopStream;
#pragma HLS stream variable = desOneHopStream depth = 32
#pragma HLS resource variable = desOneHopStream core = FIFO_LUTRAM

    loadOneHopOffset(numPairs, srcStream, desStream, offsetOneHop, offsetOneHopStream, desOneHopStream);

    hls::stream<ap_uint<32> > indexOneHopStream;
#pragma HLS stream variable = indexOneHopStream depth = 32
#pragma HLS resource variable = indexOneHopStream core = FIFO_LUTRAM

    loadOneHopIndex(numPairs, offsetOneHopStream, desOneHopStream, indexOneHop, indexOneHopStream);

    hls::stream<ap_uint<64> > offsetTwoHopStream;
#pragma HLS stream variable = offsetTwoHopStream depth = 32
#pragma HLS resource variable = offsetTwoHopStream core = FIFO_LUTRAM

    loadTwoHopOffset(numPairs, indexOneHopStream, offsetTwoHop, offsetTwoHopStream);

    hls::stream<ap_uint<32> > indexTwoHopStream;
#pragma HLS stream variable = indexTwoHopStream depth = 32
#pragma HLS resource variable = indexTwoHopStream core = FIFO_LUTRAM

    loadTwoHopIndex(numPairs, offsetTwoHopStream, indexTwoHop, indexTwoHopStream);

    hls::stream<ap_uint<32> > resStream;
#pragma HLS stream variable = resStream depth = 32
#pragma HLS resource variable = resStream core = FIFO_LUTRAM

    counter(numPairs, indexTwoHopStream, resStream);

    writeOut(numPairs, resStream, cnt_res);
}

} // namespace internal

/**
 * @brief twoHop this API can find the how many 2-hop pathes between two vertices. The input graph is the matrix in
 * CSR format. And a list of src and destination pairs whose 2-hop pathes will be counted.
 *
 * @param numPairs  How many pairs of source and destination vertices to be counted.
 * @param pair  The source and destination of pairs are stored in this pointer.
 * @param offsetOneHop The CSR offset is stored in this pointer.
 * @param indexOneHop The CSR index is stored in this pointer.
 * @param offsetTwoHop The CSR offset is stored in this pointer. The graph should be replicated and stored here. This
 * pointer is for an independent AXI port to increase performance.
 * @param indexTwoop The CSR index is stored in this pointer. The graph should be replicated and stored here. This
 * pointer is for an independent AXI port to increase performance.
 * @param cnt_res The result of the twoHop API. The order of the result matches the order of the input source and
 * destination pairs.
 *
 */

inline void twoHop(ap_uint<32> numPairs,
                   ap_uint<64>* pair,

                   unsigned* offsetOneHop,
                   unsigned* indexOneHop,
                   unsigned* offsetTwoHop,
                   unsigned* indexTwoHop,

                   unsigned* cnt_res) {
    xf::graph::internal::twoHopCore(numPairs, pair, offsetOneHop, indexOneHop, offsetTwoHop, indexTwoHop, cnt_res);
}

} // namespace graph
} // namespace xf

#endif
