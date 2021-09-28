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

#ifndef _LOUVAIN_COLORING_H_
#define _LOUVAIN_COLORING_H_

#include "louvain_modularity.hpp"
#include "louvain_samecolor.hpp"
#include "kernel_renumber.hpp"
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>

//#define DEBUG_CALCMODULARITY 1

namespace xf {
namespace graph {

template <int N_E>
ap_uint<32> loopNumGen(ap_uint<32> num) {
#pragma HLS INLINE

    ap_uint<32> base = num / N_E;
    ap_uint<4> fraction = num % N_E;
    return fraction == 0 ? base : (ap_uint<32>)(base + 1);
}

template <int DWIDTH>
void burstLoad(ap_uint<32> num, ap_uint<DWIDTH>* data, hls::stream<ap_uint<DWIDTH> >& strm) {
#pragma HLS INLINE off

    ap_uint<DWIDTH> in;
    const int N_E = DWIDTH / 32;
    ap_uint<32> numV = loopNumGen<N_E>(num);
    ap_uint<32> base = numV(31, 6);
    ap_uint<32> fraction = numV(5, 0);

#ifndef __SYNTHESIS__
#ifdef DEBUG_INIT_COMM
    std::cout << "Load num=" << num << " base=" << base << " fraction=" << fraction << std::endl;
#endif
#endif

LOAD_BASE:
    for (ap_uint<32> i = 0; i < base; i++) {
        for (ap_uint<32> j = 0; j < 64; j++) {
#pragma HLS PIPELINE II = 1

            ap_uint<32> addr;
            addr(31, 6) = i;
            addr(5, 0) = j;

            in = data[addr];
            strm.write(in);
        }
    }

LOAD_FRACTION:
    for (ap_uint<32> i = 0; i < fraction; i++) {
#pragma HLS PIPELINE II = 1

        ap_uint<32> addr;
        addr(31, 6) = base;
        addr(5, 0) = i;

        in = data[addr];
        strm.write(in);

#ifndef __SYNTHESIS__
#ifdef DEBUG_INIT_COMM
        std::cout << std::hex << "Loading addr=" << addr << " data=" << in << std::endl;
#endif
#endif
    }
}

template <int DWIDTH>
void burstWrite(ap_uint<32> num, hls::stream<ap_uint<DWIDTH> >& strm, ap_uint<DWIDTH>* data) {
#pragma HLS INLINE off

    ap_uint<DWIDTH> in;
    const int N_E = DWIDTH / 32;
    ap_uint<32> numV = loopNumGen<N_E>(num);
    ap_uint<32> base = numV(31, 6);
    ap_uint<32> fraction = numV(5, 0);

#ifndef __SYNTHESIS__
#ifdef DEBUG_INIT_COMM
    std::cout << std::hex << "Write num=" << num << " base=" << base << " fraction=" << fraction << std::endl;
#endif
#endif

WRITE_BASE:
    for (ap_uint<32> i = 0; i < base; i++) {
        for (ap_uint<32> j = 0; j < 64; j++) {
#pragma HLS PIPELINE II = 1

            ap_uint<32> addr;
            addr(31, 6) = i;
            addr(5, 0) = j;

            in = strm.read();
            data[addr] = in;
        }
    }

WRITE_FRACTION:
    for (ap_uint<32> i = 0; i < fraction; i++) {
#pragma HLS PIPELINE II = 1

        ap_uint<32> addr;
        addr(31, 6) = base;
        addr(5, 0) = i;

        in = strm.read();
        data[addr] = in;

#ifndef __SYNTHESIS__
#ifdef DEBUG_INIT_COMM
        std::cout << std::hex << "Writing addr=" << addr << " data=" << in << std::endl;
#endif
#endif
    }
}

template <int DWIDTH>
void transformOffset(int numVertex, hls::stream<ap_uint<DWIDTH> >& offset_in, hls::stream<ap_uint<32> >& offset_out) {
#pragma HLS INLINE off

    const int N_E = DWIDTH / 32;

    ap_uint<DWIDTH> offset;
    ap_uint<32> offset_tmp[N_E];
#pragma HLS array_partition variable = offset_tmp complete
    ap_uint<32> out;

LOAD_OFFSET:
    for (ap_uint<32> k = 0; k < numVertex + 1; k++) {
#pragma HLS PIPELINE II = 1

        bool read_sign;
        if (N_E == 16) {
            if (k(3, 0) == 0)
                read_sign = true;
            else
                read_sign = false;
        } else if (N_E == 8) {
            if (k(2, 0) == 0)
                read_sign = true;
            else
                read_sign = false;
        } else if (N_E == 4) {
            if (k(1, 0) == 0)
                read_sign = true;
            else
                read_sign = false;
        } else if (N_E == 2) {
            if (k[0] == 0)
                read_sign = true;
            else
                read_sign = false;
        } else {
            read_sign = true;
        }

        if (read_sign) offset = offset_in.read();

        for (ap_uint<32> j = 0; j < N_E; j++) {
#pragma HLS unroll
            offset_tmp[j] = offset((j + 1) * 32 - 1, j * 32);
        }

        if (N_E == 16) {
            out = offset_tmp[k(3, 0)];
        } else if (N_E == 8) {
            out = offset_tmp[k(2, 0)];
        } else if (N_E == 4) {
            out = offset_tmp[k(1, 0)];
        } else if (N_E == 2) {
            out = offset_tmp[k[0]];
        } else {
            out = offset_tmp[0];
        }
        out[31] = 0;
        offset_out.write(out);
    }
}

template <int DWIDTH>
void transformWeight(int numEdge, hls::stream<ap_uint<DWIDTH> >& weight_in, hls::stream<float>& weight_out) {
#pragma HLS INLINE off

    const int N_E = DWIDTH / 32;

    ap_uint<DWIDTH> weight;
    ap_uint<32> weight_tmp[N_E];
#pragma HLS array_partition variable = weight_tmp complete
    ap_uint<32> out;

LOAD_OFFSET:
    for (ap_uint<32> k = 0; k < numEdge; k++) {
#pragma HLS PIPELINE II = 1

        bool read_sign;
        if (N_E == 16) {
            if (k(3, 0) == 0)
                read_sign = true;
            else
                read_sign = false;
        } else if (N_E == 8) {
            if (k(2, 0) == 0)
                read_sign = true;
            else
                read_sign = false;
        } else if (N_E == 4) {
            if (k(1, 0) == 0)
                read_sign = true;
            else
                read_sign = false;
        } else if (N_E == 2) {
            if (k[0] == 0)
                read_sign = true;
            else
                read_sign = false;
        } else {
            read_sign = true;
        }

        if (read_sign) weight = weight_in.read();

        for (ap_uint<32> j = 0; j < N_E; j++) {
#pragma HLS unroll
            f_cast<ap_uint<32> > tmp;
            tmp.i = weight((j + 1) * 32 - 1, j * 32);
            weight_tmp[j] = tmp.f;
        }

        if (N_E == 16) {
            out = weight_tmp[k(3, 0)];
        } else if (N_E == 8) {
            out = weight_tmp[k(2, 0)];
        } else if (N_E == 4) {
            out = weight_tmp[k(1, 0)];
        } else if (N_E == 2) {
            out = weight_tmp[k[0]];
        } else {
            out = weight_tmp[0];
        }

        weight_out.write(out);
    }
}

template <typename DWEIGHT, int DWIDTH, int MAXVERTEX, int MAXEDGE>
void calculateTot(int numVertex,
                  int numEdge,
                  hls::stream<ap_uint<32> >& offset,
                  hls::stream<float>& weight,
                  hls::stream<float>& totPrev,
                  DWEIGHT& constant_recip) {
#pragma HLS INLINE off
    DWEIGHT total_weight = 0;
    DWEIGHT kiTmp = 0;
    long long total_weight_i = 0;
    long long kiTmp_i = 0;
    const int scl = 31;

    ap_uint<32> adj1 = offset.read();
    ap_uint<32> adj2 = offset.read();
    bool vertex_end = adj1 == adj2;
    ap_uint<32> vertex_cnt = 1;

CALC_TOT:
    while ((vertex_cnt < numVertex) || (adj1 < numEdge)) {
#pragma HLS PIPELINE
        if (vertex_end) {
            adj2 = offset.read(); // offset[k + 1];
            totPrev.write((float)ToDouble(kiTmp_i, scl));

#ifndef __SYNTHESIS__
#ifdef DEBUG_INIT_COMM
            std::cout << "offset=" << adj2 << " current=" << adj1 << " totPrev=" << (float)ToDouble(kiTmp_i, scl)
                      << std::endl;
#endif
#endif

            kiTmp_i = 0;
            vertex_cnt++;
        } else {
            DWEIGHT weight_tmp = weight.read();
            kiTmp_i += ToInt(weight_tmp, scl);
            total_weight_i += ToInt(weight_tmp, scl);
            adj1++;

#ifndef __SYNTHESIS__
#ifdef DEBUG_INIT_COMM
            std::cout << "weight=" << weight_tmp << " total=" << total_weight << std::endl;
#endif
#endif
        }

        if (adj1 < adj2) {
            vertex_end = false;
        } else {
            vertex_end = true;
        }
    }
    constant_recip = 1.0 / ToDouble(total_weight_i, scl);
    totPrev.write((float)ToDouble(kiTmp_i, scl));

#ifndef __SYNTHESIS__
    std::cout << "const_recip=" << constant_recip << std::endl;
#endif
}

template <typename DWEIGHT, int DWIDTH, int MAXVERTEX, int MAXEDGE>
void initTot(int numVertex, hls::stream<float>& totPrev_strm, ap_uint<DWIDTH> totPrev[MAXVERTEX]) {
#pragma HLS INLINE off

    const int N_E = DWIDTH / 32;

    ap_uint<DWIDTH> totPrev_tmp;
    ap_uint<4> cnt = 0;
    ap_uint<32> addr;

INIT_TOT:
    for (int k = 0; k < numVertex + N_E - 1; k++) {
#pragma HLS LOOP_TRIPCOUNT MIN = 1696415 MAX = 1696415
#pragma HLS PIPELINE
        f_cast<float> tmp;
        if (k < numVertex)
            tmp.f = totPrev_strm.read();
        else
            tmp.f = 0;
        totPrev_tmp(32 * (cnt + 1) - 1, 32 * cnt) = tmp.i;

        if (cnt == (N_E - 1)) {
            if (N_E == 16)
                addr = k >> 4;
            else if (N_E == 8)
                addr = k >> 3;
            else if (N_E == 4)
                addr = k >> 2;
            else
                addr = k;

            totPrev[addr] = totPrev_tmp;
            cnt = 0;

#ifndef __SYNTHESIS__
#ifdef DEBUG_INIT_COMM
            std::cout << "totPrev[" << addr << "]=" << totPrev_tmp << std::endl;
#endif
#endif
        } else
            cnt++;
    }
}

template <typename DWEIGHT, int DWIDTH, int MAXVERTEX, int MAXEDGE>
void initCid(int numVertex,
             ap_uint<DWIDTH> cidPrev[MAXVERTEX],
             ap_uint<DWIDTH> cidCurr[MAXVERTEX],
             ap_uint<DWIDTH> cidSize[MAXVERTEX]) {
#pragma HLS INLINE off

    AxiMap<int, DWIDTH> axi_cidPrev(cidPrev);
    AxiMap<int, DWIDTH> axi_cidCurr(cidCurr);
    AxiMap<int, DWIDTH> axi_cidSize(cidSize);

    const int N_E = DWIDTH / 32;
    const int loop_size = (numVertex % N_E) == 0 ? numVertex / N_E : numVertex / N_E + 1;

INIT_CID:
    for (int k = 0; k < loop_size; k++) {
#pragma HLS LOOP_TRIPCOUNT MIN = 1696415 MAX = 1696415
#pragma HLS PIPELINE
        for (int i = 0; i < N_E; i++) {
#pragma HLS UNROLL

            axi_cidPrev.wr(k * N_E + i, k * N_E + i);
            axi_cidCurr.wr(k * N_E + i, k * N_E + i);
            axi_cidSize.wr(k * N_E + i, 1);

#ifndef __SYNTHESIS__
#ifdef DEBUG_INIT_COMM
            std::cout << "cid_addr=" << k * N_E + i << std::endl;
#endif
#endif
        }
    }
}

template <typename DWEIGHT, int DWIDTH, int MAXVERTEX, int MAXEDGE>
void initComm(int64_t numVertex,
              int64_t numEdge,
              ap_uint<DWIDTH> offset[MAXVERTEX],
              ap_uint<DWIDTH> weight[MAXEDGE],
              ap_uint<DWIDTH> cidPrev[MAXVERTEX],
              ap_uint<DWIDTH> cidCurr[MAXVERTEX],
              ap_uint<DWIDTH> cidSize[MAXVERTEX],
              ap_uint<DWIDTH> totPrev[MAXVERTEX],
              DWEIGHT& constant_recip) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    const int N_E = DWIDTH / 32;

    hls::stream<ap_uint<DWIDTH> > offset_strm0("load_offset");
#pragma HLS RESOURCE variable = offset_strm0 core = FIFO_SRL
#pragma HLS STREAM variable = offset_strm0 depth = 64

    hls::stream<ap_uint<DWIDTH> > weight_strm0("load_weight");
#pragma HLS RESOURCE variable = weight_strm0 core = FIFO_SRL
#pragma HLS STREAM variable = weight_strm0 depth = 64

    hls::stream<ap_uint<32> > offset_strm1("transform_offset");
#pragma HLS RESOURCE variable = offset_strm1 core = FIFO_SRL
#pragma HLS STREAM variable = offset_strm1 depth = 32

    hls::stream<float> weight_strm1("transform_weight");
#pragma HLS RESOURCE variable = weight_strm1 core = FIFO_SRL
#pragma HLS STREAM variable = weight_strm1 depth = 32

    hls::stream<float> totPrev_strm("init_tot");
#pragma HLS RESOURCE variable = totPrev_strm core = FIFO_SRL
#pragma HLS STREAM variable = totPrev_strm depth = 32

    initCid<DWEIGHT, DWIDTH, MAXVERTEX, MAXEDGE>(numVertex, cidPrev, cidCurr, cidSize);

    burstLoad<DWIDTH>(numVertex + 1, offset, offset_strm0);

    burstLoad<DWIDTH>(numEdge, weight, weight_strm0);

    transformOffset<DWIDTH>(numVertex, offset_strm0, offset_strm1);

    transformWeight<DWIDTH>(numEdge, weight_strm0, weight_strm1);

    calculateTot<DWEIGHT, DWIDTH, MAXVERTEX, MAXEDGE>(numVertex, numEdge, offset_strm1, weight_strm1, totPrev_strm,
                                                      constant_recip);

    initTot<DWEIGHT, DWIDTH, MAXVERTEX, MAXEDGE>(numVertex, totPrev_strm, totPrev);
#ifndef __SYNTHESIS__
    std::cout << "constant_recip=" << constant_recip << std::endl;
#endif
}

template <int COLORWIDTH, int MAXVERTEX, int MAXCOLOR>
void update_color_ptr0(int numVertex, hls::stream<ap_uint<COLORWIDTH> >& color_strm, int colorPtr[MAXCOLOR]) {
#pragma HLS INLINE off

    int idx_tmp[8] = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
#pragma HLS ARRAY_PARTITION variable = idx_tmp complete
    int elem_tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma HLS ARRAY_PARTITION variable = elem_tmp complete

INIT_COLOR_SIZE:
    for (int i = 0; i < numVertex; i++) {
#pragma HLS DEPENDENCE variable = colorPtr inter false
#pragma HLS PIPELINE II = 1
#pragma HLS LATENCY min = 8

        int idx = color_strm.read();
        int addr = idx + 1;
        int elem, new_elem;
        if (idx == idx_tmp[0]) {
            elem = elem_tmp[0];
        } else if (idx == idx_tmp[1]) {
            elem = elem_tmp[1];
        } else if (idx == idx_tmp[2]) {
            elem = elem_tmp[2];
        } else if (idx == idx_tmp[3]) {
            elem = elem_tmp[3];
        } else if (idx == idx_tmp[4]) {
            elem = elem_tmp[4];
        } else if (idx == idx_tmp[5]) {
            elem = elem_tmp[5];
        } else if (idx == idx_tmp[6]) {
            elem = elem_tmp[6];
        } else if (idx == idx_tmp[7]) {
            elem = elem_tmp[7];
        } else {
            elem = colorPtr[addr];
        }
        new_elem = elem + 1;

        colorPtr[addr] = new_elem;

        for (int j = 7; j > 0; j--) {
#pragma HLS UNROLL
            idx_tmp[j] = idx_tmp[j - 1];
            elem_tmp[j] = elem_tmp[j - 1];
        }
        idx_tmp[0] = idx;
        elem_tmp[0] = new_elem;
    }
}

template <int COLORWIDTH, int MAXVERTEX, int MAXCOLOR>
void init_color_size(int numVertex, ap_uint<COLORWIDTH> color[MAXVERTEX], int colorPtr[MAXCOLOR]) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<COLORWIDTH> > color_strm;
#pragma HLS RESOURCE variable = color_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = color_strm depth = 64

    burstLoad<COLORWIDTH>(numVertex, color, color_strm);

    update_color_ptr0<COLORWIDTH, MAXVERTEX, MAXCOLOR>(numVertex, color_strm, colorPtr);
}

template <int COLORWIDTH, int MAXVERTEX, int MAXCOLOR>
void update_color_ptr1(int numVertex,
                       hls::stream<ap_uint<COLORWIDTH> >& color_strm,
                       int colorPtr[MAXCOLOR],
                       hls::stream<int>& addr_strm) {
#pragma HLS INLINE off

    int idx_tmp[8] = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
#pragma HLS ARRAY_PARTITION variable = idx_tmp complete
    int elem_tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma HLS ARRAY_PARTITION variable = elem_tmp complete

UPDATE_COLOR_PTR:
    for (int i = 0; i < numVertex; i++) {
#pragma HLS DEPENDENCE variable = colorPtr inter false
#pragma HLS PIPELINE II = 1
#pragma HLS LATENCY min = 8

        int idx = color_strm.read();
        int elem, new_elem;
        if (idx == idx_tmp[0]) {
            elem = elem_tmp[0];
        } else if (idx == idx_tmp[1]) {
            elem = elem_tmp[1];
        } else if (idx == idx_tmp[2]) {
            elem = elem_tmp[2];
        } else if (idx == idx_tmp[3]) {
            elem = elem_tmp[3];
        } else if (idx == idx_tmp[4]) {
            elem = elem_tmp[4];
        } else if (idx == idx_tmp[5]) {
            elem = elem_tmp[5];
        } else if (idx == idx_tmp[6]) {
            elem = elem_tmp[6];
        } else if (idx == idx_tmp[7]) {
            elem = elem_tmp[7];
        } else {
            elem = colorPtr[idx];
        }
        new_elem = elem + 1;

        colorPtr[idx] = new_elem;
        addr_strm.write(elem);

        for (int j = 7; j > 0; j--) {
#pragma HLS UNROLL
            idx_tmp[j] = idx_tmp[j - 1];
            elem_tmp[j] = elem_tmp[j - 1];
        }
        idx_tmp[0] = idx;
        elem_tmp[0] = new_elem;
    }
}

template <int COLORWIDTH, int MAXVERTEX>
void update_color_inx(int numVertex, hls::stream<int>& addr_strm, ap_uint<COLORWIDTH> colorInx[MAXVERTEX]) {
#pragma HLS INLINE off

REMAP_VERTEX:
    for (int i = 0; i < numVertex; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = colorInx inter false

        int addr = addr_strm.read();
        colorInx[addr] = i;
    }
}

template <int COLORWIDTH, int MAXVERTEX, int MAXCOLOR>
void remap_vertex(int numVertex,
                  ap_uint<COLORWIDTH> color[MAXVERTEX],
                  ap_uint<COLORWIDTH> colorInx[MAXVERTEX],
                  int colorPtr[MAXCOLOR]) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<COLORWIDTH> > color_strm;
#pragma HLS RESOURCE variable = color_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = color_strm depth = 16
    hls::stream<int> addr_strm;
#pragma HLS RESOURCE variable = addr_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = addr_strm depth = 16

    burstLoad<COLORWIDTH>(numVertex, color, color_strm);

    update_color_ptr1<COLORWIDTH, MAXVERTEX, MAXCOLOR>(numVertex, color_strm, colorPtr, addr_strm);

    update_color_inx<COLORWIDTH, MAXVERTEX>(numVertex, addr_strm, colorInx);
}

template <int DWIDTH>
void calculate_tot_cidSize(int numVertex,
                           hls::stream<ap_uint<DWIDTH> >& cidSizePrev_strm,
                           hls::stream<ap_uint<DWIDTH> >& totPrev_strm,
                           hls::stream<ap_uint<DWIDTH> >& cidSizeUpdate_strm,
                           hls::stream<ap_uint<DWIDTH> >& totUpdate_strm,
                           hls::stream<ap_uint<DWIDTH> >& cidSizeCurr_strm,
                           hls::stream<ap_uint<DWIDTH> >& totCurr_strm) {
#pragma HLS INLINE off

    const int NvBt = 32;                           // 32 bits
    const int NV = DWIDTH / NvBt;                  // number of vertex each DWIDTH
    const int numNV = (numVertex + (NV - 1)) / NV; // aligned alloc as NV

    for (int i = 0; i < numNV; i++) {
#pragma HLS PIPELINE II = 1

        ap_uint<DWIDTH> cidSizePrev = cidSizePrev_strm.read();
        ap_uint<DWIDTH> totPrev = totPrev_strm.read();
        ap_uint<DWIDTH> cidSizeUpdate = cidSizeUpdate_strm.read();
        ap_uint<DWIDTH> totUpdate = totUpdate_strm.read();

        ap_uint<DWIDTH> cidSizeCurr, totCurr;
        for (int j = 0; j < NV; j++) {
#pragma HLS UNROLL
            cidSizeCurr((j + 1) * NvBt - 1, j * NvBt) =
                cidSizePrev((j + 1) * NvBt - 1, j * NvBt) + cidSizeUpdate((j + 1) * NvBt - 1, j * NvBt);
        }

        for (int j = 0; j < NV; j++) {
#pragma HLS UNROLL
            f_cast<float> pre, update, curr;
            pre.i = totPrev((j + 1) * NvBt - 1, j * NvBt);
            update.i = totUpdate((j + 1) * NvBt - 1, j * NvBt);
            curr.f = pre.f + update.f;
            totCurr((j + 1) * NvBt - 1, j * NvBt) = curr.i;
        }
        cidSizeCurr_strm.write(cidSizeCurr);
        totCurr_strm.write(totCurr);
    }
}

template <int DWIDTH, int MAXVERTEX>
void updateSameColor(int numVertex,
                     ap_uint<DWIDTH> cidPrev[MAXVERTEX],
                     ap_uint<DWIDTH> cidCurr[MAXVERTEX],
                     ap_uint<DWIDTH> cidSizePrev[MAXVERTEX],
                     ap_uint<DWIDTH> cidSizeUpdate[MAXVERTEX],
                     ap_uint<DWIDTH> cidSizeCurr[MAXVERTEX],
                     ap_uint<DWIDTH> totPrev[MAXVERTEX],
                     ap_uint<DWIDTH> totUpdate[MAXVERTEX],
                     ap_uint<DWIDTH> totCurr[MAXVERTEX]) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<DWIDTH> > cidCurr_strm;
#pragma HLS RESOURCE variable = cidCurr_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = cidCurr_strm depth = 64

    hls::stream<ap_uint<DWIDTH> > cidSizePrev_strm;
#pragma HLS RESOURCE variable = cidSizePrev_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = cidSizePrev_strm depth = 64
    hls::stream<ap_uint<DWIDTH> > cidSizeUpdate_strm;
#pragma HLS RESOURCE variable = cidSizeUpdate_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = cidSizeUpdate_strm depth = 64
    hls::stream<ap_uint<DWIDTH> > cidSizeCurr_strm;
#pragma HLS RESOURCE variable = cidSizeCurr_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = cidSizeCurr_strm depth = 64

    hls::stream<ap_uint<DWIDTH> > totPrev_strm;
#pragma HLS RESOURCE variable = totPrev_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = totPrev_strm depth = 64
    hls::stream<ap_uint<DWIDTH> > totUpdate_strm;
#pragma HLS RESOURCE variable = totUpdate_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = totUpdate_strm depth = 64
    hls::stream<ap_uint<DWIDTH> > totCurr_strm;
#pragma HLS RESOURCE variable = totCurr_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = totCurr_strm depth = 64

    // load cid prev
    burstLoad<DWIDTH>(numVertex, cidCurr, cidCurr_strm);

    // load cid size prev
    burstLoad<DWIDTH>(numVertex, cidSizePrev, cidSizePrev_strm);

    burstLoad<DWIDTH>(numVertex, totPrev, totPrev_strm);

    // load tot prev
    burstLoad<DWIDTH>(numVertex, cidSizeUpdate, cidSizeUpdate_strm);

    burstLoad<DWIDTH>(numVertex, totUpdate, totUpdate_strm);

    // update
    calculate_tot_cidSize<DWIDTH>(numVertex, cidSizePrev_strm, totPrev_strm, cidSizeUpdate_strm, totUpdate_strm,
                                  cidSizeCurr_strm, totCurr_strm);

    // write back
    burstWrite<DWIDTH>(numVertex, cidCurr_strm, cidPrev);

    burstWrite<DWIDTH>(numVertex, cidSizeCurr_strm, cidSizeCurr);

    burstWrite<DWIDTH>(numVertex, totCurr_strm, totCurr);
}

template <typename DWEIGHT, int DWIDTH, int MAXVERTEX>
void calcModularity(ap_uint<32>& loop_cnt,
                    int numVertex,
                    DWEIGHT& Q_prev,
                    DWEIGHT& Q_curr,
                    DWEIGHT& deltaQ,
                    DWEIGHT constant_recip,
                    ap_uint<DWIDTH> totPrev[MAXVERTEX],
                    ap_uint<DWIDTH> totCurr[MAXVERTEX],
                    ap_uint<DWIDTH> commWeight[MAXVERTEX]) {
#pragma HLS INLINE off

    const int NvBt = 32;                           // 32 bits
    const int NV = DWIDTH / NvBt;                  // number of vertex each DWIDTH
    const int numNV = (numVertex + (NV - 1)) / NV; // aligned alloc as NV

    const int scl = 23;
    const int scl_sqr = 10;

    long long inTmp_i[NV] = {0};
#pragma HLS array_partition variable = inTmp_i complete
    long long totTmp_i[NV] = {0};
#pragma HLS array_partition variable = totTmp_i complete

    long long totalIn_i = 0;
    long long totalTot_i = 0;

MODULARITY:
    for (int i = 0; i < numNV; i++) {
#pragma HLS PIPELINE
        ap_uint<DWIDTH> tmpTot;
        if (loop_cnt[0] == 0) {
            tmpTot = totPrev[i];
        } else {
            tmpTot = totCurr[i];
        }

        ap_uint<DWIDTH> tmpIn = commWeight[i];

        for (int j = 0; j < NV; j++) {
#pragma HLS UNROLL

            f_cast<ap_uint<32> > eTot;
            eTot.i = tmpTot.range(32 * (j + 1) - 1, 32 * j);
            DWEIGHT tmp0 = (DWEIGHT)eTot.f;
            long long tmp1 = ToInt((double)(tmp0 * tmp0), scl_sqr);

            f_cast<ap_uint<32> > eIn;
            eIn.i = tmpIn.range(32 * (j + 1) - 1, 32 * j);
            long long tmp2 = ToInt((double)eIn.f, scl);

            if ((i * NV + j) < numVertex) {
                totTmp_i[j] += tmp1;
                inTmp_i[j] += tmp2;
            }
#ifndef __SYNTHESIS__
#if DEBUG_CALCMODULARITY1
            std::cout << "j=" << j << " tmp1=" << (int)tmp1 << " eTot=" << eTot.f << " tmp2=" << (int)tmp2
                      << " eIn=" << eIn.f << " totTmp=" << totTmp_i[j] << " inTmp=" << inTmp_i[j] << std::endl;
#endif
#if DEBUG_CALCMODULARITY
            std::cout << "i=" << (i * NV + j) << " tmp1=" << ToDouble(tmp1, scl_sqr) << " tmp2=" << ToDouble(tmp2, scl)
                      << std::endl;
#endif
#endif
        }
    }

Fraction:
    for (int j = 0; j < NV; j++) {
        totalTot_i += totTmp_i[j];
        totalIn_i += inTmp_i[j];
        //#if DEBUG_CALCMODULARITY
        //            std::cout << "i=" << j << " tmp1=" << ToDouble(totTmp_i[j], scl_sqr) << " tmp2=" <<
        //            ToDouble(inTmp_i[j], scl)
        //                      << std::endl;
        //#endif
    }

    Q_curr =
        ToDouble(totalIn_i, scl) * constant_recip - ToDouble(totalTot_i, scl_sqr) * constant_recip * constant_recip;
    deltaQ = Q_curr - Q_prev;

#ifndef __SYNTHESIS__
    std::cout << "totalIn=" << ToDouble(totalIn_i, scl) << " totalTot=" << ToDouble(totalTot_i, scl_sqr)
              << " Q_prev=" << Q_prev << " Q_curr=" << Q_curr << " deltaQ=" << deltaQ << std::endl;
#endif
}

template <int DWIDTH, int MAXVERTEX>
void initSameColor(int c,
                   int numVertex,
                   ap_uint<DWIDTH> cidSizeUpdate[MAXVERTEX],
                   ap_uint<DWIDTH> totUpdate[MAXVERTEX],
                   ap_uint<DWIDTH> commWeight[MAXVERTEX],
                   ap_uint<FLAGW> flagUpdate[MAXVERTEX]) {
#pragma HLS INLINE off

    const int NvBt = 32;                           // 32 bits
    const int NV = DWIDTH / NvBt;                  // number of vertex each DWIDTH
    const int numNV = (numVertex + (NV - 1)) / NV; // aligned alloc as NV

INIT_SAME_COLOR:
    for (int i = 0; i < numNV; i++) {
#pragma HLS LOOP_TRIPCOUNT MIN = 1000000 MAX = 1000000
#pragma HLS PIPELINE II = 1
        cidSizeUpdate[i] = 0;
        totUpdate[i] = 0;
        if (c == 0) {
            commWeight[i] = 0;
            flagUpdate[i] = 0;
        }
    }
}

template <typename DWEIGHT, int DWIDTH, int CSRWIDTH, int COLORWIDTH, int MAXVERTEX>
void sameColorWrapper(int numVertex,
                      int numColor,
                      int& moves,
                      long long total_w_i,
                      short scalor,
                      DWEIGHT& constant_recip,
                      int* colorPtr,
                      ap_uint<32>& loop_cnt,
                      bool use_push_flag,
                      bool write_push_flag,
                      ap_uint<CSRWIDTH>* offset,
                      ap_uint<CSRWIDTH>* index,
                      ap_uint<CSRWIDTH>* weight,
                      ap_uint<COLORWIDTH>* colorInx,
                      ap_uint<DWIDTH>* cidPrev,
                      ap_uint<DWIDTH>* cidSizePrev,
                      ap_uint<DWIDTH>* totPrev,
                      ap_uint<DWIDTH>* cidCurr,
                      ap_uint<DWIDTH>* cidSizeCurr,
                      ap_uint<DWIDTH>* totCurr,
                      ap_uint<DWIDTH>* cidSizeUpdate,
                      ap_uint<DWIDTH>* totUpdate,
                      ap_uint<DWIDTH>* commWeight,
                      ap_uint<CSRWIDTH>* offsetDup,
                      ap_uint<CSRWIDTH>* indexDup,
                      ap_uint<8>* flag,
                      ap_uint<8>* flagUpdate) {
#pragma HLS INLINE off

COLOR:
    for (int c = 0; c < numColor; c++) {
        initSameColor<DWIDTH, MAXVERTEX>(c, numVertex, cidSizeUpdate, totUpdate, commWeight, flagUpdate); //, flagUpdate

        int coloradj1 = c == 0 ? 0 : colorPtr[c - 1];
        int coloradj2 = colorPtr[c];

        if (loop_cnt[0] == 0) {
            // cidSize and tot are ping_pong buffers
            SameColor_dataflow<DWIDTH, CSRWIDTH, COLORWIDTH>(numVertex, coloradj1, coloradj2, moves, scalor, total_w_i,
                                                             use_push_flag, write_push_flag, constant_recip,
                                                             offset,        //[MAXVERTEX],
                                                             index,         //[MAXEDGE],
                                                             weight,        //[MAXEDGE],
                                                             colorInx,      //[MAXVERTEX],
                                                             cidPrev,       //[MAXVERTEX],
                                                             cidSizePrev,   //[MAXVERTEX],
                                                             totPrev,       //[MAXVERTEX],
                                                             cidCurr,       //[MAXVERTEX],
                                                             cidSizeUpdate, //[MAXVERTEX],
                                                             totUpdate,     //[MAXVERTEX],
                                                             commWeight, offsetDup, indexDup, flag,
                                                             flagUpdate); //[MAXVERTEX]);

            updateSameColor<DWIDTH, MAXVERTEX>(numVertex, cidPrev, cidCurr, cidSizePrev, cidSizeUpdate, cidSizeCurr,
                                               totPrev, totUpdate, totCurr);
        } else {
            SameColor_dataflow<DWIDTH, CSRWIDTH, COLORWIDTH>(numVertex, coloradj1, coloradj2, moves, scalor, total_w_i,
                                                             use_push_flag, write_push_flag, constant_recip,
                                                             offset,        //[MAXVERTEX],
                                                             index,         //[MAXEDGE],
                                                             weight,        //[MAXEDGE],
                                                             colorInx,      //[MAXVERTEX],
                                                             cidPrev,       //[MAXVERTEX],
                                                             cidSizeCurr,   //[MAXVERTEX],
                                                             totCurr,       //[MAXVERTEX],
                                                             cidCurr,       //[MAXVERTEX],
                                                             cidSizeUpdate, //[MAXVERTEX],
                                                             totUpdate,     //[MAXVERTEX],
                                                             commWeight, offsetDup, indexDup, flag,
                                                             flagUpdate); //[MAXVERTEX]);

            updateSameColor<DWIDTH, MAXVERTEX>(numVertex, cidPrev, cidCurr, cidSizeCurr, cidSizeUpdate, cidSizePrev,
                                               totCurr, totUpdate, totPrev);
        }
        loop_cnt++;
    } // c
}

template <typename DWEIGHT,
          int DWIDTH,
          int CSRWIDTH,
          int COLORWIDTH,
          int MAXVERTEX,
          int MAXEDGE,
          int MAXDEGREE,
          int MAXCOLOR>
void louvainWithColoring(int64_t numVertex,
                         int64_t numColor,
                         int64_t totItr,
                         DWEIGHT epsilon,
                         DWEIGHT& constant_recip,
                         ap_uint<CSRWIDTH> offset[MAXVERTEX],
                         ap_uint<CSRWIDTH> index[MAXEDGE],
                         ap_uint<CSRWIDTH> weight[MAXEDGE],
                         ap_uint<COLORWIDTH> color[MAXVERTEX],
                         ap_uint<COLORWIDTH> colorInx[MAXVERTEX],
                         ap_uint<DWIDTH> cidPrev[MAXVERTEX],
                         ap_uint<DWIDTH> cidSizePrev[MAXVERTEX],
                         ap_uint<DWIDTH> totPrev[MAXVERTEX],
                         ap_uint<DWIDTH> cidCurr[MAXVERTEX],
                         ap_uint<DWIDTH> cidSizeCurr[MAXVERTEX],
                         ap_uint<DWIDTH> totCurr[MAXVERTEX],
                         ap_uint<DWIDTH> cidSizeUpdate[MAXVERTEX],
                         ap_uint<DWIDTH> totUpdate[MAXVERTEX],
                         ap_uint<DWIDTH> commWeight[MAXVERTEX],
                         ap_uint<CSRWIDTH> offsetDup[MAXVERTEX],
                         ap_uint<CSRWIDTH> indexDup[MAXEDGE],
                         ap_uint<8> flag[MAXVERTEX],
                         ap_uint<8> flagUpdate[MAXVERTEX],
                         int64_t& iteration,
                         DWEIGHT& modularity) {
#pragma HLS INLINE off

    long long total_w_i = (long long)(1.0 / constant_recip);
    short scalor = 0;
    for (scalor = 0; scalor < 63; scalor++)
#pragma HLS PIPELINE II = 1
        if ((total_w_i >> scalor) == 0) break;
    scalor = 63 - scalor;
#ifndef __SYNTHESIS__
    printf("INFO total_w_i: %ld,  scalor = %d, totItr=%ld\n", total_w_i, (int)scalor, totItr);
#endif

    DWEIGHT Q_prev = modularity;
    DWEIGHT Q_curr = 0;
    DWEIGHT deltaQ;

#ifndef __SYNTHESIS__
    int* colorPtr = (int*)malloc(MAXCOLOR * sizeof(int));
#else
    int colorPtr[MAXCOLOR];
#pragma HLS RESOURCE variable = colorPtr core = RAM_2P_URAM
#endif

INIT_COLOR_INDEX:
    for (int i = 0; i < numColor + 1; i++) {
#pragma HLS PIPELINE II = 1
        colorPtr[i] = 0;
    }

    init_color_size<COLORWIDTH, MAXVERTEX, MAXCOLOR>(numVertex, color, colorPtr);

INIT_COLOR_DEGREE:
    for (int i = 0; i < numColor; i++) {
#pragma HLS LOOP_TRIPCOUNT MIN = 1000000 MAX = 1000000
#pragma HLS PIPELINE II = 3
        colorPtr[i + 1] += colorPtr[i];
    }

    remap_vertex<COLORWIDTH, MAXVERTEX, MAXCOLOR>(numVertex, color, colorInx, colorPtr);

    int moves;
    ap_uint<32> loop_cnt = 0;
    ap_uint<32> tmpItr = 0;
    bool config_push_flag = 0;
    bool wirte_push_flag = 0;

MOVE:
    do {
        moves = 0;

        ap_uint<32> tmpItrop = tmpItr + (ap_uint<32>)totItr;
        if (tmpItrop == 1) {
            wirte_push_flag = 1;
        }
        if (tmpItrop < 2) {
            config_push_flag = 0;
        } else {
            config_push_flag = 1;
        }

        pushcnt = 0;
        if (tmpItrop[0] == 0) {
            sameColorWrapper<DWEIGHT, DWIDTH, CSRWIDTH, COLORWIDTH, MAXVERTEX>(
                numVertex, numColor, moves, total_w_i, scalor, constant_recip, colorPtr, loop_cnt, config_push_flag,
                wirte_push_flag, offset, index, weight, colorInx, cidPrev, cidSizePrev, totPrev, cidCurr, cidSizeCurr,
                totCurr, cidSizeUpdate, totUpdate, commWeight, offsetDup, indexDup, flag, flagUpdate);
        } else {
            sameColorWrapper<DWEIGHT, DWIDTH, CSRWIDTH, COLORWIDTH, MAXVERTEX>(
                numVertex, numColor, moves, total_w_i, scalor, constant_recip, colorPtr, loop_cnt, config_push_flag,
                wirte_push_flag, offset, index, weight, colorInx, cidPrev, cidSizePrev, totPrev, cidCurr, cidSizeCurr,
                totCurr, cidSizeUpdate, totUpdate, commWeight, offsetDup, indexDup, flagUpdate, flag);
        }

#ifndef __SYNTHESIS__
        printf("INFO begin calcModularity\n");
#endif
        calcModularity<DWEIGHT, DWIDTH, MAXVERTEX>(loop_cnt, numVertex, Q_prev, Q_curr, deltaQ, constant_recip, totPrev,
                                                   totCurr, commWeight);

#ifndef __SYNTHESIS__
        printf("INFO PRINT MODULARITY\n");
        printf("INFO the modularity is: %f , Total push is: %d\n", Q_curr, (int)pushcnt);
        printf("INFO the deltaQ is: %f , epsilon is: %f, move=%d\n", deltaQ, epsilon, moves);
#endif
        modularity = Q_prev;
        Q_prev = Q_curr;
        tmpItr++;
    } while (moves > 0 && deltaQ > epsilon);
    iteration += tmpItr;

#ifndef __SYNTHESIS__
    printf("===============================\n");
    printf("INFO Total iterations is: %d \n", (int)tmpItr);
    printf("===============================\n");
    // clean up
    free(colorPtr);
#endif
}

// Reference public parallel algorithm: https://arxiv.org/pdf/1410.1237.pdf

/**
 * @brief Level2: louvain kernel implement
 *
 * @param config0 table for the int64 parameter of louvain, numVertex, numColor, total iteration times and so on.
 * @param config1 table for the double parameter of louvain, epsilon, constant_recip, modularity and so on.
 * @param offsets CSR data offset array.
 * @param indices CSR data index array.
 * @param weights CSR data weight array, support type float.
 * @param colorAxi color result before ordered.
 * @param colorInx color result of vertexes before louvain phase.
 * @param cidPrev pingpang buffer of community id (cid) for all vertex
 * @param cidSizePrev pingpang buffer of size of community for all cid
 * @param totPrev pingpang buffer of tot of community for all cid
 * @param cidCurr pingpang buffer of community id (cid) for all vertex
 * @param cidSizeCurr pingpang buffer of size of community for all cid
 * @param totCurr pingpang buffer of tot of community for all cid
 * @param cidSizeUpdate buffer for update
 * @param totUpdate buffer for update
 * @param cWeight community weight of all cid
 * @param offsetsDup duplicate of offset for fast pruning
 * @param indicesDup duplicate of index for fast pruning
 * @param flag pingpang buffer of pruning flag
 * @param flagUpdate pingpang buffer of pruning flag
 */
inline void kernelLouvainTop(int64_t* config0,
                             DWEIGHT* config1,
                             ap_uint<CSRWIDTHS>* offsets,
                             ap_uint<CSRWIDTHS>* indices,
                             ap_uint<CSRWIDTHS>* weights,
                             ap_uint<COLORWIDTHS>* colorAxi,
                             ap_uint<COLORWIDTHS>* colorInx,
                             ap_uint<DWIDTHS>* cidPrev,
                             ap_uint<DWIDTHS>* cidSizePrev,
                             ap_uint<DWIDTHS>* totPrev,
                             ap_uint<DWIDTHS>* cidCurr,
                             ap_uint<DWIDTHS>* cidSizeCurr,
                             ap_uint<DWIDTHS>* totCurr,
                             ap_uint<DWIDTHS>* cidSizeUpdate,
                             ap_uint<DWIDTHS>* totUpdate,
                             ap_uint<DWIDTHS>* cWeight,
                             ap_uint<CSRWIDTHS>* offsetsDup,
                             ap_uint<CSRWIDTHS>* indicesDup,
                             ap_uint<8>* flag,
                             ap_uint<8>* flagUpdate) {
    // clang-format off
    DWEIGHT constant_recip = 0;

    // clang-format on
    xf::graph::initComm<DWEIGHT, DWIDTHS, VERTEXS, EDGES>(config0[0], config0[3], offsets, weights, cidPrev, cidCurr,
                                                          cidSizePrev, totPrev, constant_recip);

    xf::graph::louvainWithColoring<DWEIGHT, DWIDTHS, CSRWIDTHS, COLORWIDTHS, VERTEXS, EDGES, DEGREES, COLORS>(
        config0[0], config0[1], 0, config1[0], constant_recip, offsets, indices, weights, colorAxi, colorInx, cidPrev,
        cidSizePrev, totPrev, cidCurr, cidSizeCurr, totCurr, cidSizeUpdate, totUpdate, cWeight, offsetsDup, indicesDup,
        flag, flagUpdate, config0[2], config1[1]);

    xf::graph::renumberClusters_ghost<DWIDTHS, VERTEXS>(config0[0], config0[4], config0[5], cidPrev, cidSizePrev);
}

} // graph
} // xf
#endif
