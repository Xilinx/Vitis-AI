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

#ifndef __XF_PYR_DENSE_OPTICAL_FLOW_FIND_GRADIENTS__
#define __XF_PYR_DENSE_OPTICAL_FLOW_FIND_GRADIENTS__
#include <algorithm>

template <unsigned short MAXHEIGHT,
          unsigned short MAXWIDTH,
          int NUM_LINES,
          int IT_WIDTH,
          int IT_INT,
          int ITCMP_WIDTH,
          int ITCMP_INT,
          int FLOW_WIDTH,
          int FLOW_INT,
          int RMAPPX_WIDTH,
          int RMAPPX_INT>
ap_fixed<IT_WIDTH, IT_INT> findIntensity(unsigned char lineBuffer[NUM_LINES + 1][MAXWIDTH],
                                         ap_uint<11> i,
                                         ap_uint<11> j,
                                         ap_fixed<FLOW_WIDTH, FLOW_INT> u,
                                         ap_fixed<FLOW_WIDTH, FLOW_INT> v,
                                         int totalLinesInBuffer,
                                         unsigned char effBufferedLines,
                                         int lineStore,
                                         unsigned int rows,
                                         unsigned int cols,
                                         unsigned char* fi0,
                                         unsigned char* fi1,
                                         unsigned char* fi2,
                                         unsigned char* fi3) {
    ap_fixed<ITCMP_WIDTH, ITCMP_INT> tmp_locj = u + j;
    ap_fixed<ITCMP_WIDTH, ITCMP_INT> tmp_loci = v + i;

    ap_fixed<ITCMP_WIDTH, ITCMP_INT> locj = tmp_locj;
    ap_fixed<ITCMP_WIDTH, ITCMP_INT> loci = tmp_loci;
    if (tmp_loci < 0 || (tmp_loci) > rows - 1 || (tmp_locj) < 0 || (tmp_locj) > cols - 1 ||
        (v) < (-effBufferedLines / 2) || (v) > (effBufferedLines / 2)) {
        int temp = lineStore - totalLinesInBuffer / 2;
        if (temp > effBufferedLines)
            temp -= (totalLinesInBuffer);
        else if (temp < 0)
            temp = (effBufferedLines + temp);

        return lineBuffer[temp][j];
    } else {
        // Finding which linebuffer to access
        ap_fixed<ITCMP_WIDTH, ITCMP_INT> b = ((lineStore) - (totalLinesInBuffer) / 2) + v;
        int b0 = b;

        ap_fixed<ITCMP_WIDTH, ITCMP_INT> fracy = ap_fixed<ITCMP_WIDTH, ITCMP_INT>(b - b0);
        if (b < 0) {
            fracy = 1 + fracy;
        }

        if (b0 > effBufferedLines)
            b0 -= (totalLinesInBuffer);
        else if (b < 0)
            b0 = (effBufferedLines + b0);

        int b1 = b0 + 1;
        if (b1 > effBufferedLines)
            b1 -= (totalLinesInBuffer);
        else if (b1 < 0)
            b1 = (effBufferedLines + b1);

        // Find which location in linebuffers to access
        int lx0 = tmp_locj;
        // AK,ZoTech: here out of bound of current level picture access may happen, thus workaround for bound padding
        // suggested: int lx1 = lx0 + ((lx0<(cols-1)) ? 1:0);
        int lx1 = lx0 + 1;

        ap_fixed<ITCMP_WIDTH, ITCMP_INT> fracx = ap_fixed<ITCMP_WIDTH, ITCMP_INT>(tmp_locj - lx0);
        ap_uint<8> i0 = 0;
        ap_uint<8> i1 = 0;
        ap_uint<8> i2 = 0;
        ap_uint<8> i3 = 0;
        i0 = lineBuffer[b0][lx0];
        i1 = lineBuffer[b0][lx1];
        i2 = lineBuffer[b1][lx0];
        i3 = lineBuffer[b1][lx1];
        *fi0 = i0;
        *fi1 = i1;
        *fi2 = i2;
        *fi3 = i3;

        ap_fixed<ITCMP_WIDTH, ITCMP_INT> fracx_fx = fracx;
        ap_fixed<ITCMP_WIDTH, ITCMP_INT> fracy_fx = fracy;
        // ap_fixed<RMAPPX_WIDTH,RMAPPX_INT> frac_mul = fracx * fracy;
        ap_fixed<ITCMP_WIDTH, ITCMP_INT> tempmul = 1;
        ap_fixed<RMAPPX_WIDTH, RMAPPX_INT> resf_fx = (tempmul - fracx_fx - fracy_fx + fracx_fx * fracy_fx) * i0 +
                                                     (fracx_fx - fracx_fx * fracy_fx) * i1 +
                                                     (fracy_fx - fracx_fx * fracy_fx) * i2 + fracx_fx * fracy_fx * i3;
        ap_fixed<IT_WIDTH, IT_INT> resit = resf_fx;
        return resit;
    }

} // end findIntensity()

template <unsigned short MAXHEIGHT,
          unsigned short MAXWIDTH,
          int NUM_PYR_LEVELS,
          int NUM_LINES,
          int WINSIZE,
          int IT_WIDTH,
          int IT_INT,
          int ITCMP_WIDTH,
          int ITCMP_INT,
          int FLOW_WIDTH,
          int FLOW_INT,
          int RMAPPX_WIDTH,
          int RMAPPX_INT,
          bool USE_URAM>
void findGradients(xf::cv::Mat<XF_8UC1, MAXHEIGHT, MAXWIDTH, XF_NPPC1>& currImg3,
                   xf::cv::Mat<XF_8UC1, MAXHEIGHT, MAXWIDTH, XF_NPPC1>& nextImg,
                   hls::stream<ap_fixed<IT_WIDTH, IT_INT> >& strmIt,
                   hls::stream<ap_int<9> >& strmIx,
                   hls::stream<ap_int<9> >& strmIy,
                   unsigned int rows,
                   unsigned int cols,
                   hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& strmFlowUin,
                   hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& strmFlowVin,
                   hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& strmFlowU_in1,
                   hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& strmFlowV_in1,
                   int level) {
// clang-format off
#pragma HLS inline off
// clang-format on
#if DEBUG
    char name[200];
    sprintf(name, "It_hw%d.txt", level);
    FILE* fpit = fopen(name, "w");
    sprintf(name, "idxx_hw%d.txt", level);
    FILE* fpidxx = fopen(name, "w");
    sprintf(name, "idxy_hw%d.txt", level);
    FILE* fpidxy = fopen(name, "w");
    sprintf(name, "It_i0_hw%d.txt", level);
    FILE* fpiti0 = fopen(name, "w");
    sprintf(name, "It_i1_hw%d.txt", level);
    FILE* fpiti1 = fopen(name, "w");
    sprintf(name, "It_i2_hw%d.txt", level);
    FILE* fpiti2 = fopen(name, "w");
    sprintf(name, "It_i3_hw%d.txt", level);
    FILE* fpiti3 = fopen(name, "w");
    sprintf(name, "gx_hw%d.txt", level);
    FILE* fpgx = fopen(name, "w");
    sprintf(name, "gy_hw%d.txt", level);
    FILE* fpgy = fopen(name, "w");
#endif

    unsigned char prev = 0;
    unsigned char curr = 0;
    ap_int<9> gradx = 0;
    ap_int<9> grad = 0;
    unsigned int read_curimg = 0;
    unsigned int read_nxtimg = 0;

    // AK,ZoTech: this buffer needs initialization as workaround to exclude "X" values in co-sim.
    unsigned char lineBuffer[NUM_LINES + 1][MAXWIDTH];
    if (USE_URAM) {
// clang-format off
        #pragma HLS array_reshape variable=lineBuffer complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS array_partition variable=lineBuffer complete dim=1
        // clang-format on
    }

    unsigned char curr_img_buf[2][MAXWIDTH];
    if (USE_URAM) {
// clang-format off
        #pragma HLS array_reshape variable=curr_img_buf complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS array_partition variable=curr_img_buf complete dim=1
        // clang-format on
    }

    if (USE_URAM) {
// clang-format off
        #pragma HLS RESOURCE variable=lineBuffer   core=RAM_2P_URAM
        #pragma HLS RESOURCE variable=curr_img_buf core=RAM_2P_URAM
        // clang-format on
    }

    unsigned char effBufferedLines =
        std::min(NUM_LINES, (1 << (NUM_PYR_LEVELS - 1 - level)) * (WINSIZE - 1) +
                                1); /**** Change this appropriately in original function***/
    ap_uint<8> totalLinesInBuffer = effBufferedLines + 1;

    int lineStore = totalLinesInBuffer / 2;
L3:
    for (ap_uint<16> I = 0; I < (rows + totalLinesInBuffer / 2); I++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXHEIGHT // 1080 + 11-81
    // clang-format on
    L4:
        for (ap_uint<16> j = 0; j < cols + 1; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXWIDTH
            #pragma HLS pipeline ii=1
            #pragma HLS LOOP_FLATTEN OFF
            #pragma HLS dependence array intra false
            // clang-format on
            // #pragma HLS loop_flatten off

            if (I < totalLinesInBuffer / 2) {
                if (j < cols) {
                    if (I == 0) {
                        curr_img_buf[1][j] = currImg3.read(read_curimg);
                        read_curimg++;
                        // curr_img_buf[1][j] =  *(currImg3 + I*cols + j);
                    }
                    lineBuffer[I][j] = nextImg.read(read_nxtimg);
                    read_nxtimg++;
                    // lineBuffer[I][j] = *(nextImg + I*cols + j);
                }
            } else {
                ap_uint<11> i = I - totalLinesInBuffer / 2;
                if (j < cols) {
                    if (i < rows - totalLinesInBuffer / 2) {
                        lineBuffer[lineStore][j] = nextImg.read(read_nxtimg);

                        read_nxtimg++;
                        // lineBuffer[lineStore][j] = *(nextImg + (i+(totalLinesInBuffer/2))*cols + j);
                    }

                    ap_fixed<FLOW_WIDTH, FLOW_INT> u = strmFlowUin.read();
                    ap_fixed<FLOW_WIDTH, FLOW_INT> v = strmFlowVin.read();

                    strmFlowU_in1.write(u);
                    strmFlowV_in1.write(v);
                    // Figure out the intensity to be subtracted using (u,v) on lineBuffer
                    unsigned char i0 = 0;
                    unsigned char i1 = 0;
                    unsigned char i2 = 0;
                    unsigned char i3 = 0;
                    ap_fixed<IT_WIDTH, IT_INT> nextI =
                        findIntensity<MAXHEIGHT, MAXWIDTH, NUM_LINES, IT_WIDTH, IT_INT, ITCMP_WIDTH, ITCMP_INT,
                                      FLOW_WIDTH, FLOW_INT, RMAPPX_WIDTH, RMAPPX_INT>(
                            lineBuffer, i, j, u, v, totalLinesInBuffer, effBufferedLines, lineStore, rows, cols, &i0,
                            &i1, &i2, &i3);

                    unsigned char reg = 0;
                    unsigned char curr_px = 0;
                    ap_uint<8> regSub = 0;
                    ap_uint<8> prevreg = 0;
                    if (i == 0) {
                        reg = currImg3.read(read_curimg);
                        read_curimg++;
                        // reg = *(currImg3 + (i+1)*cols + j);
                        regSub = curr_img_buf[1][j];

                        curr_img_buf[1][j] = reg;
                        curr_img_buf[0][j] = regSub;
                        curr_px = regSub;

                        grad = reg - curr_px;
                        strmIy.write(grad);
#if DEBUG
                        fprintf(fpgy, "%d ", short(grad));
#endif

                        if (j == 0) {
                            prev = curr_px;
                        } // end if
                        else if (j == 1) {
                            gradx = curr_px - prev;
                            curr = curr_px;
                            strmIx.write(gradx);
#if DEBUG
                            fprintf(fpgx, "%d ", short(gradx));
#endif
                        } // end else-if
                        else {
                            gradx = curr_px - prev;
                            prev = curr;
                            curr = curr_px;
                            strmIx.write(gradx);
#if DEBUG
                            fprintf(fpgx, "%d ", short(gradx));
#endif
                        } // end else

                    } // end else-if
                    else if (i == rows - 1) {
                        regSub = curr_img_buf[1][j];
                        grad = regSub - curr_img_buf[0][j];
                        strmIy.write(grad);
#if DEBUG
                        fprintf(fpgy, "%d ", short(grad));
#endif
                        curr_px = regSub;

                        if (j == 0) {
                            prev = curr_px;
                        } // end if
                        else if (j == 1) {
                            gradx = curr_px - prev;
                            curr = curr_px;
                            strmIx.write(gradx);
#if DEBUG
                            fprintf(fpgx, "%d ", short(gradx));
#endif
                        } // end else-if
                        else {
                            gradx = curr_px - prev;
                            prev = curr;
                            curr = curr_px;
                            strmIx.write(gradx);
#if DEBUG
                            fprintf(fpgx, "%d ", short(gradx));
#endif
                        } // end else
                    }     // end else-if
                    else {
                        reg = currImg3.read(read_curimg);
                        read_curimg++;
                        // reg = *(currImg3 + (i+1)*cols + j);
                        unsigned char regSub0 = curr_img_buf[0][j];
                        regSub = curr_img_buf[1][j];
                        curr_img_buf[0][j] = regSub;
                        curr_img_buf[1][j] = reg;

                        grad = reg - regSub0;
                        strmIy.write(grad);
#if DEBUG
                        fprintf(fpgy, "%d ", short(grad));
#endif
                        curr_px = regSub;

                        if (j == 0) {
                            prev = curr_px;
                        } // end if
                        else if (j == 1) {
                            gradx = curr_px - prev;
                            curr = curr_px;
                            strmIx.write(gradx);
#if DEBUG
                            fprintf(fpgx, "%d ", short(gradx));
#endif
                        } // end else-if
                        else {
                            gradx = curr_px - prev;
                            prev = curr;
                            curr = curr_px;
                            strmIx.write(gradx);
#if DEBUG
                            fprintf(fpgx, "%d ", short(gradx));
#endif
                        } // end else

                    } // end else
                    ap_fixed<IT_WIDTH, IT_INT> regIT = curr_px - nextI;
                    strmIt.write(regIT);

#if DEBUG
                    fprintf(fpit, "%12.2f ", float(regIT));
                    fprintf(fpidxx, "%12.2f ", float(u));
                    fprintf(fpidxy, "%12.2f ", float(v));
                    fprintf(fpiti0, "%12.2f ", float(i0));
                    fprintf(fpiti1, "%12.2f ", float(i1));
                    fprintf(fpiti2, "%12.2f ", float(i2));
                    fprintf(fpiti3, "%12.2f ", float(i3));
#endif

                    if (j == cols - 1) {
                        lineStore++;
                        if (lineStore > effBufferedLines) lineStore = 0;
                    } // end if j==cols-1
                } else {
                    gradx = curr - prev;
                    strmIx.write(gradx);
#if DEBUG
                    fprintf(fpgx, "%d ", short(gradx));
#endif
                }
            }
        } // end j loop
#if DEBUG
        if (I >= totalLinesInBuffer / 2) {
            fprintf(fpit, "\n");
            fprintf(fpidxx, "\n");
            fprintf(fpidxy, "\n");
            fprintf(fpiti0, "\n");
            fprintf(fpiti1, "\n");
            fprintf(fpiti2, "\n");
            fprintf(fpiti3, "\n");
        }
        fprintf(fpgx, "\n");
        fprintf(fpgy, "\n");
#endif
    } // end i loop

#if DEBUG
    fclose(fpit);
    fclose(fpidxx);
    fclose(fpidxy);
    fclose(fpiti0);
    fclose(fpiti1);
    fclose(fpiti2);
    fclose(fpiti3);
    fclose(fpgx);
    fclose(fpgy);
#endif
} // end find_It()

#endif
