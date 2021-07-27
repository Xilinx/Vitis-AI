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

#ifndef __XF_DENSE_NONPYR_OPTICAL_FLOW__
#define __XF_DENSE_NONPYR_OPTICAL_FLOW__
#include <stdio.h>
#include <string.h>
#include <ap_int.h>
#include <hls_stream.h>
#include "assert.h"
#include "video/xf_dense_npyr_optical_flow_types.h"
#include "common/xf_common.hpp"

namespace xf {
namespace cv {

// enable to run c-sim
//#define HLS_SIM

// Custom low cost colorizer. Uses RGB to show 4 directions. For
// simple demo purposes only. Real applications would take flow values and do
// other useful analysis on them.
static void getPseudoColorInt(pix_t pix, float fx, float fy, rgba_t& rgba) {
    // normalization factor is key for good visualization. Make this auto-ranging
    // or controllable from the host TODO
    // const int normFac = 127/2;
    const int normFac = 10;

    int y = 127 + (int)(fy * normFac);
    int x = 127 + (int)(fx * normFac);
    if (y > 255) y = 255;
    if (y < 0) y = 0;
    if (x > 255) x = 255;
    if (x < 0) x = 0;

    rgb_t rgb;
    if (x > 127) {
        if (y < 128) {
            // 1 quad
            rgb.r = x - 127 + (127 - y) / 2;
            rgb.g = (127 - y) / 2;
            rgb.b = 0;
        } else {
            // 4 quad
            rgb.r = x - 127;
            rgb.g = 0;
            rgb.b = y - 127;
        }
    } else {
        if (y < 128) {
            // 2 quad
            rgb.r = (127 - y) / 2;
            rgb.g = 127 - x + (127 - y) / 2;
            rgb.b = 0;
        } else {
            // 3 quad
            rgb.r = 0;
            rgb.g = 128 - x;
            rgb.b = y - 127;
        }
    }

    rgba.r = pix / 4 + 3 * rgb.r / 4;
    rgba.g = pix / 4 + 3 * rgb.g / 4;
    rgba.b = pix / 4 + 3 * rgb.b / 4;
    rgba.a = 255;
    // rgba.r = rgb.r;
    // rgba.g = rgb.g;
    // rgba.b = rgb.b ;
}
// read external array matB and stream.
// Can be simplified to a single loop with II=1 TODO
// void readMatRows ( mywide_t< XF_NPIXPERCYCLE(NPC) >  *matB, hls::stream < mywide_t< XF_NPIXPERCYCLE(NPC) > >&
// pixStream)
template <int TYPE, int ROWS, int COLS, int NPC, int WINDOW_SIZE>
static void readMatRows16(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& src,
                          hls::stream<mywide_t<XF_NPIXPERCYCLE(NPC)> >& pixStream,
                          int rows,
                          int cols,
                          int size) {
    unsigned int count = 0;
    for (int i = 0; i < size; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS*ROWS/NPC
        #pragma HLS PIPELINE
        // clang-format on
        unsigned short t;
        t = src.read(i);
        mywide_t<XF_NPIXPERCYCLE(NPC)> tmpData;
        tmpData.data[0] = t & 0x00FF;
        tmpData.data[1] = t >> 8;
        pixStream.write(tmpData);
    }
}

template <int ROWS, int COLS, int NPC, int WINDOW_SIZE>
static void writeMatRowsRGBA16(
    hls::stream<rgba_t>& pixStream0, hls::stream<rgba_t>& pixStream1, ap_uint<64>* dst, int rows, int cols, int size) {
    for (int i = 0; i < size; ++i) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS*COLS/NPC
        #pragma HLS PIPELINE
        // clang-format on
        rgba_t d0 = pixStream0.read();
        rgba_t d1 = pixStream1.read();

        // unsigned int t1 = (unsigned int)d1.a << 24 | (unsigned int)d1.b << 16 | (unsigned int)d1.g << 8 | (unsigned
        // int)d1.r; unsigned int t0 = (unsigned int)d0.a << 24 | (unsigned int)d0.b << 16 | (unsigned int)d0.g << 8 |
        // (unsigned int)d0.r;
        // who is at MSB? t0 or t1 TODO
        // unsigned long long l = (unsigned long long) t1 << 32 | (unsigned long long) t0;

        unsigned long long l = (unsigned long long)d1.a << 56 | (unsigned long long)d1.b << 48 |
                               (unsigned long long)d1.g << 40 | (unsigned long long)d1.r << 32 |
                               (unsigned long long)d0.a << 24 | (unsigned long long)d0.b << 16 |
                               (unsigned long long)d0.g << 8 | (unsigned long long)d0.r;
        *(dst + i) = l;
    }
}

// write rgba stream to external array dst. The "a" is just padding and is
// unused
template <int TYPE, int ROWS, int COLS, int NPC, int WINDOW_SIZE>
static void pack2Vectors(hls::stream<float>& flow0,
                         hls::stream<float>& flow1,
                         xf::cv::Mat<XF_32FC1, ROWS, COLS, NPC>& out_flow,
                         int rows,
                         int cols,
                         int size) {
    for (int i = 0; i < size; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS*COLS/NPC
        #pragma HLS PIPELINE
        // clang-format on
        float d0 = flow0.read();
        float d1 = flow1.read();
        ap_uint<32>* d0_int;
        d0_int = (ap_uint<32>*)&d0;
        ap_uint<32>* d1_int;
        d1_int = (ap_uint<32>*)&d1;
        // as 0th word will have d0_int and 1st word will have d1_int
        ap_uint<64> l = ((unsigned long long)(*d1_int) << 32) | (unsigned long long)(*d0_int);
        out_flow.write(i, l);
    }
}

// Compute sums for bottom-right and top-right pixel and update the column sums.
// Use column sums to update the integrals. Implements O(1) sliding window.
//
// TODO:
// 1. Dont need the entire column for img1Win and img2Win. Need only the kernel
// 2. Full line buffer is not needed
template <int ROWS, int COLS, int NPC, int WINDOW_SIZE, bool USE_URAM>
static void computeSums16(hls::stream<mywide_t<XF_NPIXPERCYCLE(NPC)> > img1Col[(WINDOW_SIZE + 1)],
                          hls::stream<mywide_t<XF_NPIXPERCYCLE(NPC)> > img2Col[(WINDOW_SIZE + 1)],
                          hls::stream<int>& ixix_out0,
                          hls::stream<int>& ixiy_out0,
                          hls::stream<int>& iyiy_out0,
                          hls::stream<int>& dix_out0,
                          hls::stream<int>& diy_out0,

                          hls::stream<int>& ixix_out1,
                          hls::stream<int>& ixiy_out1,
                          hls::stream<int>& iyiy_out1,
                          hls::stream<int>& dix_out1,
                          hls::stream<int>& diy_out1,
                          int rows,
                          int cols,
                          int size)

{
    pix_t img1Col0[(WINDOW_SIZE + 1)], img2Col0[(WINDOW_SIZE + 1)];
    pix_t img1Col1[(WINDOW_SIZE + 1)], img2Col1[(WINDOW_SIZE + 1)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=img1Col0 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=img2Col0 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=img1Col1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=img2Col1 complete dim=0
    // clang-format on

    static pix_t img1Win[2 * (WINDOW_SIZE + 1)], img2Win[1 * (WINDOW_SIZE + 1)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=img1Win complete dim=0
    #pragma HLS ARRAY_PARTITION variable=img2Win complete dim=0
    // clang-format on
    // static pix_t img1Win1 [2 * (WINDOW_SIZE+1)], img2Win1 [1 * (WINDOW_SIZE+1)];
    //#pragma HLS ARRAY_PARTITION variable=img1Win1 complete dim=0
    //#pragma HLS ARRAY_PARTITION variable=img2Win1 complete dim=0

    static int ixix = 0, ixiy = 0, iyiy = 0, dix = 0, diy = 0;

    // column sums:
    // need left-shift. Array-Part leads to FF with big Muxes. Try to do with
    // classic array and pointer. Need current and current-WINDOW_SIZE ptrs
    // For II=1 pipelining, need two read and 1 write ports. Simulating it with
    // two arrays that have their write ports tied together.
    // TODO need only MAX_WODTH/2. Have to adjust zIdx and nIdx as well
    static int csIxixO[COLS / 2], csIxiyO[COLS / 2], csIyiyO[COLS / 2], csDixO[COLS / 2], csDiyO[COLS / 2];
    static int csIxixE[COLS / 2], csIxiyE[COLS / 2], csIyiyE[COLS / 2], csDixE[COLS / 2], csDiyE[COLS / 2];

    static int cbIxixO[COLS / 2], cbIxiyO[COLS / 2], cbIyiyO[COLS / 2], cbDixO[COLS / 2], cbDiyO[COLS / 2];
    static int cbIxixE[COLS / 2], cbIxiyE[COLS / 2], cbIyiyE[COLS / 2], cbDixE[COLS / 2], cbDiyE[COLS / 2];

    int zIdx = -(WINDOW_SIZE / 2 - 1);
    int nIdx = zIdx + WINDOW_SIZE / 2 - 1;

// clang-format off
    #pragma HLS ARRAY_MAP variable=csIxixO instance=csO vertical
    #pragma HLS ARRAY_MAP variable=csIxiyO instance=csO vertical
    #pragma HLS ARRAY_MAP variable=csIyiyO instance=csO vertical
    #pragma HLS ARRAY_MAP variable=csDixO  instance=csO vertical
    #pragma HLS ARRAY_MAP variable=csDiyO  instance=csO vertical
// clang-format on

// clang-format off
    #pragma HLS ARRAY_MAP variable=csIxixE instance=csE vertical
    #pragma HLS ARRAY_MAP variable=csIxiyE instance=csE vertical
    #pragma HLS ARRAY_MAP variable=csIyiyE instance=csE vertical
    #pragma HLS ARRAY_MAP variable=csDixE  instance=csE vertical
    #pragma HLS ARRAY_MAP variable=csDiyE  instance=csE vertical
// clang-format on

// clang-format off
    #pragma HLS ARRAY_MAP variable=cbIxixO instance=cb vertical
    #pragma HLS ARRAY_MAP variable=cbIxiyO instance=cb vertical
    #pragma HLS ARRAY_MAP variable=cbIyiyO instance=cb vertical
    #pragma HLS ARRAY_MAP variable=cbDixO  instance=cb vertical
    #pragma HLS ARRAY_MAP variable=cbDiyO  instance=cb vertical
    #pragma HLS ARRAY_MAP variable=cbIxixE instance=cb vertical
    #pragma HLS ARRAY_MAP variable=cbIxiyE instance=cb vertical
    #pragma HLS ARRAY_MAP variable=cbIyiyE instance=cb vertical
    #pragma HLS ARRAY_MAP variable=cbDixE  instance=cb vertical
    #pragma HLS ARRAY_MAP variable=cbDiyE  instance=cb vertical
    // clang-format on

    if (USE_URAM) {
// clang-format off
        #pragma HLS RESOURCE variable=csIxixO core=RAM_2P_URAM
        #pragma HLS RESOURCE variable=csIxixE core=RAM_2P_URAM
        #pragma HLS RESOURCE variable=cbIxixO core=RAM_2P_URAM
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=csIxixO core=RAM_2P_BRAM
        #pragma HLS RESOURCE variable=csIxixE core=RAM_2P_BRAM
        #pragma HLS RESOURCE variable=cbIxixO core=RAM_2P_BRAM
        // clang-format on
    }

// clang-format off
    #pragma HLS DEPENDENCE variable=csIxixO inter RAW false
    #pragma HLS DEPENDENCE variable=csIxiyO inter RAW false
    #pragma HLS DEPENDENCE variable=csIyiyO inter RAW false
    #pragma HLS DEPENDENCE variable=csDixO  inter RAW false
    #pragma HLS DEPENDENCE variable=csDiyO  inter RAW false
// clang-format on

// clang-format off
    #pragma HLS DEPENDENCE variable=csIxixE inter WAR false
    #pragma HLS DEPENDENCE variable=csIxiyE inter WAR false
    #pragma HLS DEPENDENCE variable=csIyiyE inter WAR false
    #pragma HLS DEPENDENCE variable=csDixE  inter WAR false
    #pragma HLS DEPENDENCE variable=csDiyE  inter WAR false
// clang-format on

// clang-format off
    #pragma HLS DEPENDENCE variable=cbIxixO inter RAW false
    #pragma HLS DEPENDENCE variable=cbIxiyO inter RAW false
    #pragma HLS DEPENDENCE variable=cbIyiyO inter RAW false
    #pragma HLS DEPENDENCE variable=cbDixO  inter RAW false
    #pragma HLS DEPENDENCE variable=cbDiyO  inter RAW false
// clang-format on

// clang-format off
    #pragma HLS DEPENDENCE variable=cbIxixE inter RAW false
    #pragma HLS DEPENDENCE variable=cbIxiyE inter RAW false
    #pragma HLS DEPENDENCE variable=cbIyiyE inter RAW false
    #pragma HLS DEPENDENCE variable=cbDixE  inter RAW false
    #pragma HLS DEPENDENCE variable=cbDiyE  inter RAW false
    // clang-format on

    int csIxixR0, csIxiyR0, csIyiyR0, csDixR0, csDiyR0;
    int csIxixR1, csIxiyR1, csIyiyR1, csDixR1, csDiyR1;

    for (int r = 0; r < rows; r++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        for (int c = 0; c < cols / 2; c++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS/2
            #pragma HLS PIPELINE
            // clang-format on

            int csIxixL0 = 0, csIxiyL0 = 0, csIyiyL0 = 0, csDixL0 = 0, csDiyL0 = 0;
            int csIxixL1 = 0, csIxiyL1 = 0, csIyiyL1 = 0, csDixL1 = 0, csDiyL1 = 0;

            for (int wr = 0; wr < (WINDOW_SIZE + 1); ++wr) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=1 max=WINDOW_SIZE+1
                // clang-format on
                mywide_t<XF_NPIXPERCYCLE(NPC)> tmp1 = img1Col[wr].read();
                img1Col0[wr] = tmp1.data[0];
                img1Col1[wr] = tmp1.data[1];

                mywide_t<XF_NPIXPERCYCLE(NPC)> tmp2 = img2Col[wr].read();
                img2Col0[wr] = tmp2.data[0];
                img2Col1[wr] = tmp2.data[1];
            }

            // p(x+1,y) and p(x-1,y)
            int wrt = 1;
            int cIxTopR0 = (img1Col0[wrt] - img1Win[wrt * 2 + 2 - 2]) / 2;
            // p(x,y+1) and p(x,y-1)
            int cIyTopR0 = (img1Win[(wrt + 1) * 2 + 2 - 1] - img1Win[(wrt - 1) * 2 + 2 - 1]) / 2;
            // p1(x,y) and p2(x,y)
            int delTopR0 = img1Win[wrt * 2 + 2 - 1] - img2Win[wrt * 1 + 1 - 1];

            int wrb = WINDOW_SIZE - 1;
            int cIxBotR0 = (img1Col0[wrb] - img1Win[wrb * 2 + 2 - 2]) / 2;
            int cIyBotR0 = (img1Win[(wrb + 1) * 2 + 2 - 1] - img1Win[(wrb - 1) * 2 + 2 - 1]) / 2;
            int delBotR0 = img1Win[wrb * 2 + 2 - 1] - img2Win[wrb * 1 + 1 - 1];
            if (0 && r < WINDOW_SIZE) {
                cIxTopR0 = 0;
                cIyTopR0 = 0;
                delTopR0 = 0;
            }

            // p(x+1,y) and p(x-1,y)
            wrt = 1;
            int cIxTopR1 = (img1Col1[wrt] - img1Win[wrt * 2 + 2 - 1]) / 2;
            // p(x,y+1) and p(x,y-1)
            int cIyTopR1 = (img1Col0[wrt + 1] - img1Col0[wrt - 1]) / 2;
            // p1(x,y) and p2(x,y)
            int delTopR1 = (img1Col0[wrt] - img2Col0[wrt]);

            wrb = WINDOW_SIZE - 1;
            int cIxBotR1 = (img1Col1[wrb] - img1Win[wrb * 2 + 2 - 1]) / 2;
            int cIyBotR1 = (img1Col0[wrb + 1] - img1Col0[wrb - 1]) / 2;
            int delBotR1 = (img1Col0[wrb] - img2Col0[wrb]);

            csIxixR0 = cbIxixE[nIdx] + cIxBotR0 * cIxBotR0 - cIxTopR0 * cIxTopR0;
            csIxiyR0 = cbIxiyE[nIdx] + cIxBotR0 * cIyBotR0 - cIxTopR0 * cIyTopR0;
            csIyiyR0 = cbIyiyE[nIdx] + cIyBotR0 * cIyBotR0 - cIyTopR0 * cIyTopR0;
            csDixR0 = cbDixE[nIdx] + delBotR0 * cIxBotR0 - delTopR0 * cIxTopR0;
            csDiyR0 = cbDiyE[nIdx] + delBotR0 * cIyBotR0 - delTopR0 * cIyTopR0;

            csIxixR1 = cbIxixO[nIdx] + cIxBotR1 * cIxBotR1 - cIxTopR1 * cIxTopR1;
            csIxiyR1 = cbIxiyO[nIdx] + cIxBotR1 * cIyBotR1 - cIxTopR1 * cIyTopR1;
            csIyiyR1 = cbIyiyO[nIdx] + cIyBotR1 * cIyBotR1 - cIyTopR1 * cIyTopR1;
            csDixR1 = cbDixO[nIdx] + delBotR1 * cIxBotR1 - delTopR1 * cIxTopR1;
            csDiyR1 = cbDiyO[nIdx] + delBotR1 * cIyBotR1 - delTopR1 * cIyTopR1;

            csIxixE[nIdx] = csIxixR0;
            csIxiyE[nIdx] = csIxiyR0;
            csIyiyE[nIdx] = csIyiyR0;
            csDixE[nIdx] = csDixR0;
            csDiyE[nIdx] = csDiyR0;

            if (zIdx >= 0) {
                int const zIdxPrev = zIdx == 0 ? cols / 2 - 1 : zIdx - 1;
                csIxixL0 = csIxixO[zIdxPrev];
                csIxiyL0 = csIxiyO[zIdxPrev];
                csIyiyL0 = csIyiyO[zIdxPrev];
                csDixL0 = csDixO[zIdxPrev];
                csDiyL0 = csDiyO[zIdxPrev];

                csIxixL1 = csIxixE[zIdx];
                csIxiyL1 = csIxiyE[zIdx];
                csIyiyL1 = csIyiyE[zIdx];
                csDixL1 = csDixE[zIdx];
                csDiyL1 = csDiyE[zIdx];
            }

            int tmpixix0 = (csIxixR0 - csIxixL0);
            int tmpixix1 = (csIxixR0 - csIxixL0) + (csIxixR1 - csIxixL1);
            int tmpixiy0 = (csIxiyR0 - csIxiyL0);
            int tmpixiy1 = (csIxiyR0 - csIxiyL0) + (csIxiyR1 - csIxiyL1);
            int tmpiyiy0 = (csIyiyR0 - csIyiyL0);
            int tmpiyiy1 = (csIyiyR0 - csIyiyL0) + (csIyiyR1 - csIyiyL1);
            int tmpdix0 = (csDixR0 - csDixL0);
            int tmpdix1 = (csDixR0 - csDixL0) + (csDixR1 - csDixL1);
            int tmpdiy0 = (csDiyR0 - csDiyL0);
            int tmpdiy1 = (csDiyR0 - csDiyL0) + (csDiyR1 - csDiyL1);

            // ixix += (csIxixR0 - csIxixL0);
            // ixiy += (csIxiyR0 - csIxiyL0);
            // iyiy += (csIyiyR0 - csIyiyL0);
            // dix += (csDixR0 - csDixL0);
            // diy += (csDiyR0 - csDiyL0);

            // ixix_out0. write (ixix);
            // ixiy_out0. write (ixiy);
            // iyiy_out0. write (iyiy);
            // dix_out0. write (dix);
            // diy_out0. write (diy);
            ixix_out0.write(ixix + tmpixix0);
            ixiy_out0.write(ixiy + tmpixiy0);
            iyiy_out0.write(iyiy + tmpiyiy0);
            dix_out0.write(dix + tmpdix0);
            diy_out0.write(diy + tmpdiy0);

            // now compute the second pixel
            // ixix += (csIxixR1 - csIxixL1);
            // ixiy += (csIxiyR1 - csIxiyL1);
            // iyiy += (csIyiyR1 - csIyiyL1);
            // dix += (csDixR1 - csDixL1);
            // diy += (csDiyR1 - csDiyL1);
            ixix += tmpixix1;
            ixiy += tmpixiy1;
            iyiy += tmpiyiy1;
            dix += tmpdix1;
            diy += tmpdiy1;

            ixix_out1.write(ixix);
            ixiy_out1.write(ixiy);
            iyiy_out1.write(iyiy);
            dix_out1.write(dix);
            diy_out1.write(diy);

            for (int i = 0; i < (WINDOW_SIZE + 1); i++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=1 max=WINDOW_SIZE+1
                // clang-format on
                img1Win[i * 2] = img1Col0[i];
                img1Win[i * 2 + 1] = img1Col1[i];
                img2Win[i] = img2Col1[i];
            }

            cbIxixE[nIdx] = csIxixR0;
            cbIxiyE[nIdx] = csIxiyR0;
            cbIyiyE[nIdx] = csIyiyR0;
            cbDixE[nIdx] = csDixR0;
            cbDiyE[nIdx] = csDiyR0;

            /* csIxixE [nIdx] = csIxixR0;
             csIxiyE [nIdx] = csIxiyR0;
             csIyiyE [nIdx] = csIyiyR0;
             csDixE  [nIdx] = csDixR0;
             csDiyE  [nIdx] = csDiyR0;*/

            cbIxixO[nIdx] = csIxixR1;
            cbIxiyO[nIdx] = csIxiyR1;
            cbIyiyO[nIdx] = csIyiyR1;
            cbDixO[nIdx] = csDixR1;
            cbDiyO[nIdx] = csDiyR1;

            csIxixO[nIdx] = csIxixR1;
            csIxiyO[nIdx] = csIxiyR1;
            csIyiyO[nIdx] = csIyiyR1;
            csDixO[nIdx] = csDixR1;
            csDiyO[nIdx] = csDiyR1;

            zIdx++;
            if (zIdx == cols / 2) zIdx = 0;
            nIdx++;
            if (nIdx == cols / 2) nIdx = 0;
        }
    }

    // Cleanup. If kernel is called multiple times with different inputs, not
    // cleaning these vars would pollute the subsequent frames.
    // TODO zero in the line buffer instead, for r < WINDOW_SIZE
    for (int r = 0; r < (WINDOW_SIZE + 1); r++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=WINDOW_SIZE+1
        #pragma HLS UNROLL
        // clang-format on
        img1Win[r] = 0;
        img1Win[r + (WINDOW_SIZE + 1)] = 0;
        img2Win[r] = 0;
        img1Col0[r] = 0;
        img2Col0[r] = 0;
        img1Col1[r] = 0;
        img2Col1[r] = 0;
    }
    for (int r = 0; r < cols / 2; ++r) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
        #pragma HLS PIPELINE
        // clang-format on
        csIxixO[r] = 0;
        csIxiyO[r] = 0;
        csIyiyO[r] = 0;
        csDixO[r] = 0;
        csDiyO[r] = 0;
        cbIxixO[r] = 0;
        cbIxiyO[r] = 0;
        cbIyiyO[r] = 0;
        cbDixO[r] = 0;
        cbDiyO[r] = 0;

        csIxixE[r] = 0;
        csIxiyE[r] = 0;
        csIyiyE[r] = 0;
        csDixE[r] = 0;
        csDiyE[r] = 0;
        cbIxixE[r] = 0;
        cbIxiyE[r] = 0;
        cbIyiyE[r] = 0;
        cbDixE[r] = 0;
        cbDiyE[r] = 0;
    }
    ixix = 0;
    ixiy = 0;
    iyiy = 0;
    dix = 0;
    diy = 0;
}

// consume the integrals and compute flow vectors
template <int ROWS, int COLS, int NPC, int WINDOW_SIZE>
static void computeFlow16(hls::stream<int>& ixix,
                          hls::stream<int>& ixiy,
                          hls::stream<int>& iyiy,
                          hls::stream<int>& dix,
                          hls::stream<int>& diy,
                          hls::stream<float>& fx_out,
                          hls::stream<float>& fy_out,
                          int rows,
                          int cols,
                          int size) {
    for (int r = 0; r < rows; r++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        for (int c = 0; c < cols / 2; c++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS/2
            #pragma HLS PIPELINE
            // clang-format on
            int ixix_ = ixix.read();
            int ixiy_ = ixiy.read();
            int iyiy_ = iyiy.read();
            int dix_ = dix.read();
            int diy_ = diy.read();
            float fx_ = 0, fy_ = 0;

            // matrix inv
            float det = (float)ixix_ * iyiy_ - (float)ixiy_ * ixiy_;
            if (det <= 1.0f || r < (WINDOW_SIZE) || c < ((WINDOW_SIZE + 1) / 2)) {
                fx_ = 0.0;
                fy_ = 0.0;
            } else {
                // res est: (dsp,ff,lut)
                // fdiv (0,748,800), fmul (3,143,139), fadd (2,306,246), fsub (2,306,246)
                // sitofp (0,229,365), fcmp (0,66,72), imul(1,0,0) (in cs)
                // float detInv = 1.0/det;
                float i00 = (float)iyiy_ / det;
                float i01 = (float)(-ixiy_) / det;
                float i10 = (float)(-ixiy_) / det;
                float i11 = (float)ixix_ / det;

                fx_ = i00 * dix_ + i01 * diy_;
                fy_ = i10 * dix_ + i11 * diy_;
            }
            fx_out.write(fx_);
            fy_out.write(fy_);
        }
    }
}

// convert flow values to visualizable pixel. For simple demo purposes only
template <int ROWS, int COLS, int NPC, int WINDOW_SIZE>
static void getOutPix16(
    hls::stream<float>& fx, hls::stream<float>& fy, hls::stream<rgba_t>& out_pix, int rows, int cols, int size) {
    for (int r = 0; r < rows; r++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        for (int c = 0; c < cols / 2; c++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS/2
            #pragma HLS PIPELINE
            // clang-format on
            float fx_ = fx.read();
            float fy_ = fy.read();

            pix_t p_ = 0;
            rgba_t out_pix_;
            getPseudoColorInt(p_, fx_, fy_, out_pix_);

            out_pix.write(out_pix_);
        }
    }
}

// line buffer for both input images. Can be split to a fn that models a single
// linebuffer
template <int ROWS, int COLS, int NPC, int WINDOW_SIZE, bool USE_URAM>
static void lbWrapper16(hls::stream<mywide_t<XF_NPIXPERCYCLE(NPC)> >& f0Stream,
                        hls::stream<mywide_t<XF_NPIXPERCYCLE(NPC)> >& f1Stream,
                        hls::stream<mywide_t<XF_NPIXPERCYCLE(NPC)> > img1Col[(WINDOW_SIZE + 1)],
                        hls::stream<mywide_t<XF_NPIXPERCYCLE(NPC)> > img2Col[(WINDOW_SIZE + 1)],
                        int rows,
                        int cols,
                        int size) {
    static pix_t lb1[(WINDOW_SIZE + 1)][COLS / XF_NPIXPERCYCLE(NPC)][XF_NPIXPERCYCLE(NPC)],
        lb2[(WINDOW_SIZE + 1)][COLS / XF_NPIXPERCYCLE(NPC)][XF_NPIXPERCYCLE(NPC)];

// clang-format off
    #pragma HLS ARRAY_MAP variable=lb1 instance=lbMap vertical
    #pragma HLS ARRAY_MAP variable=lb2 instance=lbMap vertical
// clang-format on

// clang-format off
    #pragma HLS ARRAY_RESHAPE variable=lb1 complete dim=1
    #pragma HLS ARRAY_RESHAPE variable=lb2 complete dim=1
    #pragma HLS ARRAY_RESHAPE variable=lb1 complete dim=3
    #pragma HLS ARRAY_RESHAPE variable=lb2 complete dim=3
    // clang-format on

    if (USE_URAM) {
// clang-format off
        #pragma HLS RESOURCE variable=lb1 core=RAM_T2P_URAM
        #pragma HLS RESOURCE variable=lb2 core=RAM_T2P_URAM
        // clang-format on
    }

    for (int r = 0; r < rows; r++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        #pragma HLS LOOP_FLATTEN OFF
        // clang-format on
        for (int c = 0; c < cols / 2; c++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS/2
            #pragma HLS pipeline
            // clang-format on
            // shift up both linebuffers at col=c
            for (int i = 0; i < ((WINDOW_SIZE + 1) - 1); i++) {
                mywide_t<XF_NPIXPERCYCLE(NPC)> lb;

                for (int k = 0; k < XF_NPIXPERCYCLE(NPC); k++) {
                    lb.data[k] = lb1[i + 1][c][k];
                    lb1[i][c][k] = lb.data[k];
                }
                img1Col[i].write(lb);

                for (int k = 0; k < XF_NPIXPERCYCLE(NPC); k++) {
                    lb.data[k] = lb2[i + 1][c][k];
                    lb2[i][c][k] = lb.data[k];
                }
                img2Col[i].write(lb);
            }

            // read in the new pixels at col=c and row=bottom_of_lb
            mywide_t<XF_NPIXPERCYCLE(NPC)> pix0 = f0Stream.read();
            img1Col[(WINDOW_SIZE + 1) - 1].write(pix0);

            mywide_t<XF_NPIXPERCYCLE(NPC)> pix1 = f1Stream.read();
            img2Col[(WINDOW_SIZE + 1) - 1].write(pix1);

            for (int k = 0; k < XF_NPIXPERCYCLE(NPC); k++) {
                lb1[(WINDOW_SIZE + 1) - 1][c][k] = pix0.data[k];
                lb2[(WINDOW_SIZE + 1) - 1][c][k] = pix1.data[k];
            }
        }
    }

    // cleanup
    for (int c = 0; c < cols / 2; c++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS/2
        #pragma HLS PIPELINE
        // clang-format on
        for (int r = 0; r < (WINDOW_SIZE + 1); r++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=WINDOW_SIZE+1
            // clang-format on
            for (int k = 0; k < XF_NPIXPERCYCLE(NPC); k++) {
                lb1[r][c][k] = 0;
                lb2[r][c][k] = 0;
            }
        }
    }
}

// top level wrapper to avoid dataflow problems
// void flowWrap (mywide_t frame0[NUM_WORDS], mywide_t frame1[NUM_WORDS], rgba2_t framef[NUM_WORDS])
template <int TYPE, int ROWS, int COLS, int NPC, int WINDOW_SIZE, bool USE_URAM>
static void flowWrap16(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& frame0,
                       xf::cv::Mat<TYPE, ROWS, COLS, NPC>& frame1,
                       xf::cv::Mat<XF_32FC1, ROWS, COLS, NPC>& flowx,
                       xf::cv::Mat<XF_32FC1, ROWS, COLS, NPC>& flowy,
                       int rows,
                       int cols,
                       int size) {
//#pragma HLS data_pack variable=frame0
//#pragma HLS data_pack variable=frame1
//#pragma HLS data_pack variable=framef

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    // ddr <-> kernel streams. Stream depths are probably too large and can be
    // trimmed
    hls::stream<mywide_t<XF_NPIXPERCYCLE(NPC)> > f0Stream, f1Stream;
// clang-format off
    #pragma HLS data_pack variable=f0Stream
    #pragma HLS data_pack variable=f1Stream
    #pragma HLS STREAM variable=f0Stream depth=16
    #pragma HLS STREAM variable=f1Stream depth=16
    // clang-format on

    // hls::stream <rgba_t> ff0Stream, ff1Stream;
    // #pragma HLS data_pack variable=ff0Stream
    // #pragma HLS data_pack variable=ff1Stream
    // #pragma HLS STREAM variable=ff0Stream depth=16
    // #pragma HLS STREAM variable=ff1Stream depth=16

    hls::stream<mywide_t<XF_NPIXPERCYCLE(NPC)> > img1Col[(WINDOW_SIZE + 1)], img2Col[(WINDOW_SIZE + 1)];
// clang-format off
    #pragma HLS data_pack variable=img1Col
    #pragma HLS data_pack variable=img2Col
    #pragma HLS STREAM variable=img1Col  depth=16
    #pragma HLS STREAM variable=img2Col  depth=16
    #pragma HLS ARRAY_PARTITION variable=img1Col complete dim=0
    #pragma HLS ARRAY_PARTITION variable=img2Col complete dim=0
    // clang-format on

    hls::stream<int> ixix0, ixiy0, iyiy0, dix0, diy0;
    hls::stream<float> fx0("fx0"), fy0("fy0");
// clang-format off
    #pragma HLS STREAM variable=ixix0 depth=16
    #pragma HLS STREAM variable=ixiy0 depth=16
    #pragma HLS STREAM variable=iyiy0 depth=16
    #pragma HLS STREAM variable=dix0 depth=16
    #pragma HLS STREAM variable=diy0 depth=16
    #pragma HLS STREAM variable=fx0  depth=16
    #pragma HLS STREAM variable=fy0  depth=16
    // clang-format on

    hls::stream<int> ixix1, ixiy1, iyiy1, dix1, diy1;
    hls::stream<float> fx1("fx1"), fy1("fy1");
// clang-format off
    #pragma HLS STREAM variable=ixix1 depth=16
    #pragma HLS STREAM variable=ixiy1 depth=16
    #pragma HLS STREAM variable=iyiy1 depth=16
    #pragma HLS STREAM variable=dix1 depth=16
    #pragma HLS STREAM variable=diy1 depth=16
    #pragma HLS STREAM variable=fx1  depth=16
    #pragma HLS STREAM variable=fy1  depth=16
    // clang-format on

    readMatRows16<TYPE, ROWS, COLS, NPC, WINDOW_SIZE>(frame0, f0Stream, rows, cols, size);
    readMatRows16<TYPE, ROWS, COLS, NPC, WINDOW_SIZE>(frame1, f1Stream, rows, cols, size);

    lbWrapper16<ROWS, COLS, NPC, WINDOW_SIZE, USE_URAM>(f0Stream, f1Stream, img1Col, img2Col, rows, cols, size);
    computeSums16<ROWS, COLS, NPC, WINDOW_SIZE, USE_URAM>(img1Col, img2Col, ixix0, ixiy0, iyiy0, dix0, diy0, ixix1,
                                                          ixiy1, iyiy1, dix1, diy1, rows, cols, size);

    computeFlow16<ROWS, COLS, NPC, WINDOW_SIZE>(ixix0, ixiy0, iyiy0, dix0, diy0, fx0, fy0, rows, cols, size);
    computeFlow16<ROWS, COLS, NPC, WINDOW_SIZE>(ixix1, ixiy1, iyiy1, dix1, diy1, fx1, fy1, rows, cols, size);

    pack2Vectors<TYPE, ROWS, COLS, NPC, WINDOW_SIZE>(fx0, fx1, flowx, rows, cols, size);
    pack2Vectors<TYPE, ROWS, COLS, NPC, WINDOW_SIZE>(fy0, fy1, flowy, rows, cols, size);
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

// external interface to the kernel.
//  frame0 - First input frame (grayscale 1 byte per pixel)
//  frame1 - Second input frame (grayscale 1 byte per pixel)
//  framef - Output frame with flows visualized. 3 bytes per pixel + 1 byte padding
// void fpga_optflow ( mywide_t< XF_NPIXPERCYCLE(NPC) >  *frame0,  mywide_t< XF_NPIXPERCYCLE(NPC) >  *frame1, rgba2_t
// *framef);
// ushort = 16bits, 8 bits per grayscale pix, so two pix
// ulonglong = 64 bits, 32 bits per color pixel (rgba), so two color pix
// void fpga_optflow (unsigned short *frame0, unsigned short *frame1, unsigned long long *framef)
// void fpga_optflow (unsigned short frame0[NUM_WORDS], unsigned short frame1[NUM_WORDS], unsigned long long
// framef[NUM_WORDS])
template <int TYPE, int ROWS, int COLS, int NPC, int WINDOW_SIZE, bool USE_URAM>
static void fpga_optflow16(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& frame0,
                           xf::cv::Mat<TYPE, ROWS, COLS, NPC>& frame1,
                           xf::cv::Mat<XF_32FC1, ROWS, COLS, NPC>& flowx,
                           xf::cv::Mat<XF_32FC1, ROWS, COLS, NPC>& flowy,
                           int rows,
                           int cols,
                           int size) {
// clang-format off
    #pragma HLS inline off
    // clang-format on

    flowWrap16<TYPE, ROWS, COLS, NPC, WINDOW_SIZE, USE_URAM>(frame0, frame1, flowx, flowy, rows, cols, size);

    return;
}

// read external array matB and stream.
// Can be simplified to a single loop with II=1 TODO, hls::stream< mywide_t< XF_NPIXPERCYCLE(NPC) > > &frame1,
// hls::stream<rgba_t> &framef
template <int TYPE, int ROWS, int COLS, int NPC, int WINDOW_SIZE>
static void readMatRows(
    xf::cv::Mat<TYPE, ROWS, COLS, NPC>& matB, hls::stream<pix_t>& pixStream, int rows, int cols, int size) {
    const int WORD_SIZE = (NPC == XF_NPPC1) ? 1 : 2;
    for (int i = 0; i < size; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS*ROWS/NPC
        #pragma HLS PIPELINE
        // clang-format on
        mywide_t<XF_NPIXPERCYCLE(NPC)> tmpData;
        tmpData.data[0] = matB.read(i);

        // for (int k = 0; k < WORD_SIZE; ++k) {
        //    pixStream.write(tmpData.data[k]);
        // }
        pixStream.write(tmpData.data[0]);
    }
}

// write rgba stream to external array dst. The "a" is just padding and is
// unused
template <int ROWS, int COLS, int NPC, int WINDOW_SIZE>
static void writeMatRowsRGBA(hls::stream<rgba_t>& pixStream, unsigned int* dst, int rows, int cols, int size) {
    for (int i = 0; i < size; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS*COLS/NPC
        #pragma HLS PIPELINE
        // clang-format on
        rgba_t tmpData = pixStream.read();
        *(dst + i) = (unsigned int)tmpData.a << 24 | (unsigned int)tmpData.b << 16 | (unsigned int)tmpData.g << 8 |
                     (unsigned int)tmpData.r;
    }
}

// Compute sums for bottom-right and top-right pixel and update the column sums.
// Use column sums to update the integrals. Implements O(1) sliding window.
//
// TODO:
// 1. Dont need the entire column for img1Win and img2Win. Need only the kernel
// 2. Full line buffer is not needed
template <int ROWS, int COLS, int NPC, int WINDOW_SIZE, bool USE_URAM>
static void computeSums(hls::stream<pix_t> img1Col[(WINDOW_SIZE + 1)],
                        hls::stream<pix_t> img2Col[(WINDOW_SIZE + 1)],
                        hls::stream<int>& ixix_out,
                        hls::stream<int>& ixiy_out,
                        hls::stream<int>& iyiy_out,
                        hls::stream<int>& dix_out,
                        hls::stream<int>& diy_out,
                        int rows,
                        int cols,
                        int size) {
    pix_t img1Col_[(WINDOW_SIZE + 1)], img2Col_[(WINDOW_SIZE + 1)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=img1Col_ complete dim=0
    #pragma HLS ARRAY_PARTITION variable=img2Col_ complete dim=0
    // clang-format on

    static pix_t img1Win[2 * (WINDOW_SIZE + 1)], img2Win[1 * (WINDOW_SIZE + 1)];
    static int ixix = 0, ixiy = 0, iyiy = 0, dix = 0, diy = 0;
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=img1Win complete dim=0
    #pragma HLS ARRAY_PARTITION variable=img2Win complete dim=0
    // clang-format on

    // column sums:
    // need left-shift. Array-Part leads to FF with big Muxes. Try to do with
    // classic array and pointer. Need current and current-WINDOW_SIZE ptrs
    // For II=1 pipelining, need two read and 1 write ports. Simulating it with
    // two arrays that have their write ports tied together.
    static int csIxix[COLS], csIxiy[COLS], csIyiy[COLS], csDix[COLS], csDiy[COLS];
    static int cbIxix[COLS], cbIxiy[COLS], cbIyiy[COLS], cbDix[COLS], cbDiy[COLS];

    int zIdx = -(WINDOW_SIZE - 2);
    int nIdx = zIdx + WINDOW_SIZE - 2;

// clang-format off
    #pragma HLS ARRAY_MAP variable=csIxix instance=cs vertical
    #pragma HLS ARRAY_MAP variable=csIxiy instance=cs vertical
    #pragma HLS ARRAY_MAP variable=csIyiy instance=cs vertical
    #pragma HLS ARRAY_MAP variable=csDix  instance=cs vertical
    #pragma HLS ARRAY_MAP variable=csDiy  instance=cs vertical
    // clang-format on

    if (USE_URAM) {
// clang-format off
        #pragma HLS RESOURCE variable=csIxix core=RAM_2P_URAM
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=csIxix core=RAM_2P_BRAM
        // clang-format on
    }

// clang-format off
    #pragma HLS DEPENDENCE variable=csIxix inter RAW false
    #pragma HLS DEPENDENCE variable=csIxiy inter RAW false
    #pragma HLS DEPENDENCE variable=csIyiy inter RAW false
    #pragma HLS DEPENDENCE variable=csDix  inter RAW false
    #pragma HLS DEPENDENCE variable=csDiy  inter RAW false
// clang-format on

// clang-format off
    #pragma HLS ARRAY_MAP variable=cbIxix instance=cb vertical
    #pragma HLS ARRAY_MAP variable=cbIxiy instance=cb vertical
    #pragma HLS ARRAY_MAP variable=cbIyiy instance=cb vertical
    #pragma HLS ARRAY_MAP variable=cbDix  instance=cb vertical
    #pragma HLS ARRAY_MAP variable=cbDiy  instance=cb vertical
    // clang-format on

    if (USE_URAM) {
// clang-format off
        #pragma HLS RESOURCE variable=cbIxix core=RAM_2P_URAM
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=cbIxix core=RAM_2P_BRAM
        // clang-format on
    }

// clang-format off
    #pragma HLS DEPENDENCE variable=cbIxix inter RAW false
    #pragma HLS DEPENDENCE variable=cbIxiy inter RAW false
    #pragma HLS DEPENDENCE variable=cbIyiy inter RAW false
    #pragma HLS DEPENDENCE variable=cbDix  inter RAW false
    #pragma HLS DEPENDENCE variable=cbDiy  inter RAW false
    // clang-format on

    int csIxixR, csIxiyR, csIyiyR, csDixR, csDiyR;

    for (int r = 0; r < rows; r++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        for (int c = 0; c < cols; c++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
            #pragma HLS PIPELINE II=1
            // clang-format on

            int csIxixL = 0;
            int csIxiyL = 0;
            int csIyiyL = 0;
            int csDixL = 0;
            int csDiyL = 0;
            if (zIdx >= 0) {
                csIxixL = csIxix[zIdx];
                csIxiyL = csIxiy[zIdx];
                csIyiyL = csIyiy[zIdx];
                csDixL = csDix[zIdx];
                csDiyL = csDiy[zIdx];
            }

            for (int wr = 0; wr < (WINDOW_SIZE + 1); ++wr) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=1 max=WINDOW_SIZE+1
                // clang-format on
                img1Col_[wr] = img1Col[wr].read();
                img2Col_[wr] = img2Col[wr].read();
            }

            // p(x+1,y) and p(x-1,y)
            int wrt = 1;
            int cIxTopR = (img1Col_[wrt] - img1Win[wrt * 2 + 2 - 2]) / 2;
            // p(x,y+1) and p(x,y-1)
            int cIyTopR = (img1Win[(wrt + 1) * 2 + 2 - 1] - img1Win[(wrt - 1) * 2 + 2 - 1]) / 2;
            // p1(x,y) and p2(x,y)
            int delTopR = img1Win[wrt * 2 + 2 - 1] - img2Win[wrt * 1 + 1 - 1];

            int wrb = WINDOW_SIZE - 1;
            int cIxBotR = (img1Col_[wrb] - img1Win[wrb * 2 + 2 - 2]) / 2;
            int cIyBotR = (img1Win[(wrb + 1) * 2 + 2 - 1] - img1Win[(wrb - 1) * 2 + 2 - 1]) / 2;
            int delBotR = img1Win[wrb * 2 + 2 - 1] - img2Win[wrb * 1 + 1 - 1];
            if (0 && r < WINDOW_SIZE) {
                cIxTopR = 0;
                cIyTopR = 0;
                delTopR = 0;
            }

            csIxixR = cbIxix[nIdx] + cIxBotR * cIxBotR - cIxTopR * cIxTopR;
            csIxiyR = cbIxiy[nIdx] + cIxBotR * cIyBotR - cIxTopR * cIyTopR;
            csIyiyR = cbIyiy[nIdx] + cIyBotR * cIyBotR - cIyTopR * cIyTopR;
            csDixR = cbDix[nIdx] + delBotR * cIxBotR - delTopR * cIxTopR;
            csDiyR = cbDiy[nIdx] + delBotR * cIyBotR - delTopR * cIyTopR;

            ixix += (csIxixR - csIxixL);
            ixiy += (csIxiyR - csIxiyL);
            iyiy += (csIyiyR - csIyiyL);
            dix += (csDixR - csDixL);
            diy += (csDiyR - csDiyL);

            ixix_out.write(ixix);
            ixiy_out.write(ixiy);
            iyiy_out.write(iyiy);
            dix_out.write(dix);
            diy_out.write(diy);

            // we dont have the shifted pixel anymore to do overlay TODO
            // img1Delayed. write (0);

            for (int i = 0; i < (WINDOW_SIZE + 1); i++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=1 max=WINDOW_SIZE+1
                // clang-format on
                img1Win[i * 2] = img1Win[i * 2 + 1];
            }
            for (int i = 0; i < (WINDOW_SIZE + 1); ++i) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=1 max=WINDOW_SIZE+1
                // clang-format on
                img1Win[i * 2 + 1] = img1Col_[i];
                img2Win[i] = img2Col_[i];
            }

            cbIxix[nIdx] = csIxixR;
            cbIxiy[nIdx] = csIxiyR;
            cbIyiy[nIdx] = csIyiyR;
            cbDix[nIdx] = csDixR;
            cbDiy[nIdx] = csDiyR;

            csIxix[nIdx] = csIxixR;
            csIxiy[nIdx] = csIxiyR;
            csIyiy[nIdx] = csIyiyR;
            csDix[nIdx] = csDixR;
            csDiy[nIdx] = csDiyR;
            zIdx++;
            if (zIdx == cols) zIdx = 0;
            nIdx++;
            if (nIdx == cols) nIdx = 0;
        }
    }

    // Cleanup. If kernel is called multiple times with different inputs, not
    // cleaning these vars would pollute the subsequent frames.
    // TODO zero in the line buffer instead, for r < WINDOW_SIZE
    for (int r = 0; r < (WINDOW_SIZE + 1); r++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=WINDOW_SIZE+1
        #pragma HLS UNROLL
        // clang-format on
        img1Win[r] = 0;
        img1Win[r + (WINDOW_SIZE + 1)] = 0;
        img2Win[r] = 0;
        img1Col_[r] = 0;
        img2Col_[r] = 0;
    }
    for (int r = 0; r < cols; ++r) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
        #pragma HLS PIPELINE II=1
        // clang-format on
        csIxix[r] = 0;
        csIxiy[r] = 0;
        csIyiy[r] = 0;
        csDix[r] = 0;
        csDiy[r] = 0;
        cbIxix[r] = 0;
        cbIxiy[r] = 0;
        cbIyiy[r] = 0;
        cbDix[r] = 0;
        cbDiy[r] = 0;
    }
    ixix = 0;
    ixiy = 0;
    iyiy = 0;
    dix = 0;
    diy = 0;
}

// consume the integrals and compute flow vectors
template <int ROWS, int COLS, int NPC, int WINDOW_SIZE>
static void computeFlow(hls::stream<int>& ixix,
                        hls::stream<int>& ixiy,
                        hls::stream<int>& iyiy,
                        hls::stream<int>& dix,
                        hls::stream<int>& diy,
                        hls::stream<float>& fx_out,
                        hls::stream<float>& fy_out,
                        int rows,
                        int cols,
                        int size) {
    for (int r = 0; r < rows; r++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        for (int c = 0; c < cols; c++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
            #pragma HLS PIPELINE
            // clang-format on
            int ixix_ = ixix.read();
            int ixiy_ = ixiy.read();
            int iyiy_ = iyiy.read();
            int dix_ = dix.read();
            int diy_ = diy.read();
            float fx_ = 0, fy_ = 0;

            // matrix inv
            float det = (float)ixix_ * iyiy_ - (float)ixiy_ * ixiy_;
            if (det <= 1.0f || r < (WINDOW_SIZE) || c < (WINDOW_SIZE + 1)) {
                fx_ = 0.0;
                fy_ = 0.0;
            } else {
                // res est: (dsp,ff,lut)
                // fdiv (0,748,800), fmul (3,143,139), fadd (2,306,246), fsub (2,306,246)
                // sitofp (0,229,365), fcmp (0,66,72), imul(1,0,0) (in cs)
                // float detInv = 1.0/det;
                float i00 = (float)iyiy_ / det;
                float i01 = (float)(-ixiy_) / det;
                float i10 = (float)(-ixiy_) / det;
                float i11 = (float)ixix_ / det;

                fx_ = i00 * dix_ + i01 * diy_;
                fy_ = i10 * dix_ + i11 * diy_;
            }
            fx_out.write(fx_);
            fy_out.write(fy_);
        }
    }
}

template <int TYPE, int ROWS, int COLS, int NPC, int WINDOW_SIZE>
static void writeOutput8(hls::stream<float>& fx_in,
                         hls::stream<float>& fy_in,
                         xf::cv::Mat<XF_32FC1, ROWS, COLS, NPC>& flowx,
                         xf::cv::Mat<XF_32FC1, ROWS, COLS, NPC>& flowy,
                         int size) {
    for (int r = 0; r < size; r++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS*COLS
        #pragma HLS PIPELINE
        // clang-format on

        float fx_out = fx_in.read();
        float fy_out = fy_in.read();
        ap_uint<32>* fx_out_int;
        fx_out_int = (ap_uint<32>*)&fx_out;
        ap_uint<32>* fy_out_int;
        fy_out_int = (ap_uint<32>*)&fy_out;
        flowx.write(r, *fx_out_int);
        flowy.write(r, *fy_out_int);
        // ap_uint<32> a32 = flowx.read(r);
        // ap_uint<32> b32 = flowy.read(r);
    }
}

// convert flow values to visualizable pixel. For simple demo purposes only
template <int ROWS, int COLS, int NPC>
static void getOutPix(float* fx, float* fy, pix_t* p, hls::stream<rgba_t>& out_pix, int rows, int cols, int size) {
    for (int r = 0; r < rows; r++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        for (int c = 0; c < cols; c++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
            #pragma HLS PIPELINE
            // clang-format on
            float fx_ = *(fx + r * cols + c);
            float fy_ = *(fy + r * cols + c);

            pix_t p_ = *(p + r * cols + c);
            rgba_t out_pix_;
            getPseudoColorInt(p_, fx_, fy_, out_pix_);

            out_pix.write(out_pix_);
        }
    }
}

// line buffer for both input images. Can be split to a fn that models a single
// linebuffer
template <int ROWS, int COLS, int NPC, int WINDOW_SIZE, bool USE_URAM>
static void lbWrapper(hls::stream<pix_t>& f0Stream,
                      hls::stream<pix_t>& f1Stream,
                      hls::stream<pix_t> img1Col[(WINDOW_SIZE + 1)],
                      hls::stream<pix_t> img2Col[(WINDOW_SIZE + 1)],
                      int rows,
                      int cols,
                      int size) {
    static pix_t lb1[(WINDOW_SIZE + 1)][COLS], lb2[(WINDOW_SIZE + 1)][COLS];
// clang-format off
    #pragma HLS ARRAY_MAP variable=lb1 instance=lbMap vertical
    #pragma HLS ARRAY_MAP variable=lb2 instance=lbMap vertical
    #pragma HLS ARRAY_RESHAPE variable=lb1 complete dim=1
    #pragma HLS ARRAY_RESHAPE variable=lb2 complete dim=1
    // clang-format on
    if (USE_URAM) {
// clang-format off
        #pragma HLS RESOURCE variable=lb1 core=RAM_T2P_URAM
        #pragma HLS RESOURCE variable=lb2 core=RAM_T2P_URAM
        // clang-format on
    }
loop_rows:
    for (int r = 0; r < rows; r++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        #pragma HLS LOOP_FLATTEN OFF
    // clang-format on
    loop_cols:
        for (int c = 0; c < cols; c++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
        #pragma HLS pipeline
        // clang-format on

        // shift up both linebuffers at col=c
        loop_ws:
            for (int i = 0; i < (WINDOW_SIZE + 1) - 1; i++) {
                lb1[i][c] = lb1[i + 1][c];
                img1Col[i].write(lb1[i][c]);

                lb2[i][c] = lb2[i + 1][c];
                img2Col[i].write(lb2[i][c]);
            }

            // read in the new pixels at col=c and row=bottom_of_lb
            pix_t pix0 = f0Stream.read();

            lb1[(WINDOW_SIZE + 1) - 1][c] = pix0;
            img1Col[(WINDOW_SIZE + 1) - 1].write(pix0);

            pix_t pix1 = f1Stream.read();

            lb2[(WINDOW_SIZE + 1) - 1][c] = pix1;
            img2Col[(WINDOW_SIZE + 1) - 1].write(pix1);
        }
    }

    // cleanup
    for (int c = 0; c < cols; c++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
        #pragma HLS PIPELINE
        // clang-format on
        for (int r = 0; r < (WINDOW_SIZE + 1); r++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=WINDOW_SIZE+1
            // clang-format on
            lb1[r][c] = 0;
            lb2[r][c] = 0;
        }
    }
}

// top level wrapper to avoid dataflow problems
template <int TYPE, int ROWS, int COLS, int NPC, int WINDOW_SIZE, bool USE_URAM>
static void flowWrap(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& frame0,
                     xf::cv::Mat<TYPE, ROWS, COLS, NPC>& frame1,
                     xf::cv::Mat<XF_32FC1, ROWS, COLS, NPC>& flowx,
                     xf::cv::Mat<XF_32FC1, ROWS, COLS, NPC>& flowy,
                     int rows,
                     int cols,
                     int size) {
// clang-format off
    #pragma HLS inline off
    #pragma HLS DATAFLOW
    // clang-format on

    // ddr <-> kernel streams. Stream depths are probably too large and can be
    // trimmed
    hls::stream<pix_t> f0Stream("f0Stream"), f1Stream("f1Stream");
    hls::stream<pix_t> f0Delayed("f0Delayed");
// clang-format off
    #pragma HLS STREAM variable=f0Stream depth=16
    #pragma HLS STREAM variable=f1Stream depth=16
    // clang-format on
    // #pragma HLS STREAM variable=f0Delayed depth=128

    // hls::stream <rgba_t> ffStream ("ffStream");
    // #pragma HLS data_pack variable=ffStream
    // #pragma HLS STREAM variable=ffStream depth=16

    hls::stream<pix_t> img1Col[(WINDOW_SIZE + 1)], img2Col[(WINDOW_SIZE + 1)];
    hls::stream<int> ixix, ixiy, iyiy, dix, diy;
    hls::stream<float> fx, fy;

// clang-format off
    #pragma HLS STREAM variable=ixix depth=16
    #pragma HLS STREAM variable=ixiy depth=16
    #pragma HLS STREAM variable=iyiy depth=16
    #pragma HLS STREAM variable=dix depth=16
    #pragma HLS STREAM variable=diy depth=16
// clang-format on
// #pragma HLS STREAM variable=fx  depth=16
// #pragma HLS STREAM variable=fy  depth=16
// clang-format off
    #pragma HLS STREAM variable=img1Col  depth=16
    #pragma HLS STREAM variable=img2Col  depth=16
// clang-format on

// clang-format off
    #pragma HLS ARRAY_PARTITION variable=img1Col complete dim=0
    #pragma HLS ARRAY_PARTITION variable=img2Col complete dim=0
    // clang-format on

    readMatRows<TYPE, ROWS, COLS, NPC, WINDOW_SIZE>(frame0, f0Stream, rows, cols, size);
    readMatRows<TYPE, ROWS, COLS, NPC, WINDOW_SIZE>(frame1, f1Stream, rows, cols, size);

    lbWrapper<ROWS, COLS, NPC, WINDOW_SIZE, USE_URAM>(f0Stream, f1Stream, img1Col, img2Col, rows, cols, size);

    computeSums<ROWS, COLS, NPC, WINDOW_SIZE, USE_URAM>(img1Col, img2Col, ixix, ixiy, iyiy, dix, diy, rows, cols, size);

    computeFlow<ROWS, COLS, NPC, WINDOW_SIZE>(ixix, ixiy, iyiy, dix, diy, fx, fy, rows, cols, size);

    writeOutput8<TYPE, ROWS, COLS, NPC, WINDOW_SIZE>(fx, fy, flowx, flowy, size);
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

// external interface to the kernel.
//  frame0 - First input frame (grayscale 1 byte per pixel)
//  frame1 - Second input frame (grayscale 1 byte per pixel)
//  framef - Output frame with flows visualized. 3 bytes per pixel + 1 byte padding
template <int TYPE, int ROWS, int COLS, int NPC, int WINDOW_SIZE, bool USE_URAM>
static void fpga_optflow8(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& frame0,
                          xf::cv::Mat<TYPE, ROWS, COLS, NPC>& frame1,
                          xf::cv::Mat<XF_32FC1, ROWS, COLS, NPC>& flowx,
                          xf::cv::Mat<XF_32FC1, ROWS, COLS, NPC>& flowy,
                          int rows,
                          int cols,
                          int size) {
// clang-format off
    #pragma HLS inline off
    // clang-format on

    flowWrap<TYPE, ROWS, COLS, NPC, WINDOW_SIZE, USE_URAM>(frame0, frame1, flowx, flowy, rows, cols, size);

    return;
}
template <int WINDOW_SIZE, int TYPE, int ROWS, int COLS, int NPC, bool USE_URAM = false>
void DenseNonPyrLKOpticalFlow(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& frame0,
                              xf::cv::Mat<TYPE, ROWS, COLS, NPC>& frame1,
                              xf::cv::Mat<XF_32FC1, ROWS, COLS, NPC>& flowx,
                              xf::cv::Mat<XF_32FC1, ROWS, COLS, NPC>& flowy) {
    if (NPC == XF_NPPC1) {
        fpga_optflow8<TYPE, ROWS, COLS, NPC, WINDOW_SIZE, USE_URAM>(frame0, frame1, flowx, flowy, frame0.rows,
                                                                    frame0.cols, frame0.size);
    } else {
        fpga_optflow16<TYPE, ROWS, COLS, NPC, WINDOW_SIZE, USE_URAM>(frame0, frame1, flowx, flowy, frame0.rows,
                                                                     frame0.cols, frame0.size);
    }
}
} // namespace cv
} // namespace xf
#endif
