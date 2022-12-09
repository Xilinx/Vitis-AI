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

#ifndef __XF_PYR_DENSE_OPTICAL_FLOW_OFLOW_PROCESS__
#define __XF_PYR_DENSE_OPTICAL_FLOW_OFLOW_PROCESS__
template <unsigned short MAXHEIGHT,
          unsigned short MAXWIDTH,
          int WINSIZE,
          int IT_WIDTH,
          int IT_INT,
          int SIXIY_WIDTH,
          int SIXIY_INT,
          int SIXYIT_WIDTH,
          int SIXYIT_INT,
          bool USE_URAM>
void find_G_and_b_matrix(hls::stream<ap_int<9> >& strmIx,
                         hls::stream<ap_int<9> >& strmIy,
                         hls::stream<ap_fixed<IT_WIDTH, IT_INT> >& strmIt,
                         hls::stream<ap_fixed<SIXIY_WIDTH, SIXIY_INT> >& sigmaIx2,
                         hls::stream<ap_fixed<SIXIY_WIDTH, SIXIY_INT> >& sigmaIy2,
                         hls::stream<ap_fixed<SIXIY_WIDTH, SIXIY_INT> >& sigmaIxIy,
                         hls::stream<ap_fixed<SIXYIT_WIDTH, SIXYIT_INT> >& sigmaIxIt,
                         hls::stream<ap_fixed<SIXYIT_WIDTH, SIXYIT_INT> >& sigmaIyIt,
                         unsigned int rows,
                         unsigned int cols,
                         int level) {
// clang-format off
    #pragma HLS inline off
    // clang-format on
    // bufLines is used to buffer Ix, Iy, It in that order
    ap_int<9> bufLines_ix[WINSIZE][MAXWIDTH + (WINSIZE >> 1)];

    ap_int<9> bufLines_iy[WINSIZE][MAXWIDTH + (WINSIZE >> 1)];

    ap_fixed<IT_WIDTH, IT_INT> bufLines_it[WINSIZE][MAXWIDTH + (WINSIZE >> 1)];

    if (USE_URAM) {
// clang-format off
        #pragma HLS array_reshape variable=bufLines_ix complete dim=1
        #pragma HLS array_reshape variable=bufLines_iy complete dim=1
        #pragma HLS array_reshape variable=bufLines_it complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS array_partition variable=bufLines_ix complete dim=1
        #pragma HLS array_partition variable=bufLines_iy complete dim=1
        #pragma HLS array_partition variable=bufLines_it complete dim=1
        // clang-format on
    }

    ap_fixed<SIXIY_WIDTH, SIXIY_INT> colsum_IxIx[MAXWIDTH + (WINSIZE >> 1)];
    ap_fixed<SIXIY_WIDTH, SIXIY_INT> colsum_IxIy[MAXWIDTH + (WINSIZE >> 1)];
    ap_fixed<SIXIY_WIDTH, SIXIY_INT> colsum_IyIy[MAXWIDTH + (WINSIZE >> 1)];
    ap_fixed<SIXYIT_WIDTH, SIXYIT_INT> colsum_IxIt[MAXWIDTH + (WINSIZE >> 1)];
    ap_fixed<SIXYIT_WIDTH, SIXYIT_INT> colsum_IyIt[MAXWIDTH + (WINSIZE >> 1)];
    if (USE_URAM) {
// clang-format off
        #pragma HLS ARRAY_MAP variable=bufLines_ix instance=buffers vertical
        #pragma HLS ARRAY_MAP variable=bufLines_iy instance=buffers vertical
        #pragma HLS ARRAY_MAP variable=bufLines_it instance=buffers vertical
// clang-format on

// clang-format off
        #pragma HLS ARRAY_MAP variable=colsum_IxIx instance=buffers vertical
        #pragma HLS ARRAY_MAP variable=colsum_IxIy instance=buffers vertical
        #pragma HLS ARRAY_MAP variable=colsum_IyIy instance=buffers vertical
        #pragma HLS ARRAY_MAP variable=colsum_IxIt instance=buffers vertical
        #pragma HLS ARRAY_MAP variable=colsum_IyIt instance=buffers vertical
// clang-format on

// clang-format off
        #pragma HLS RESOURCE variable=bufLines_ix core=RAM_S2P_URAM
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=colsum_IxIx core=RAM_T2P_BRAM
        #pragma HLS RESOURCE variable=colsum_IxIy core=RAM_T2P_BRAM
        #pragma HLS RESOURCE variable=colsum_IyIy core=RAM_T2P_BRAM
        #pragma HLS RESOURCE variable=colsum_IxIt core=RAM_T2P_BRAM
        #pragma HLS RESOURCE variable=colsum_IyIt core=RAM_T2P_BRAM
        // clang-format on
    }

    ap_fixed<SIXIY_WIDTH, SIXIY_INT> colsum_prevWIN_IxIx[WINSIZE];
    ap_fixed<SIXIY_WIDTH, SIXIY_INT> colsum_prevWIN_IxIy[WINSIZE];
    ap_fixed<SIXIY_WIDTH, SIXIY_INT> colsum_prevWIN_IyIy[WINSIZE];
    ap_fixed<SIXYIT_WIDTH, SIXYIT_INT> colsum_prevWIN_IxIt[WINSIZE];
    ap_fixed<SIXYIT_WIDTH, SIXYIT_INT> colsum_prevWIN_IyIt[WINSIZE];
// clang-format off
    #pragma HLS array_partition variable=colsum_prevWIN_IxIx complete dim=1
    #pragma HLS array_partition variable=colsum_prevWIN_IxIy complete dim=1
    #pragma HLS array_partition variable=colsum_prevWIN_IyIy complete dim=1
    #pragma HLS array_partition variable=colsum_prevWIN_IxIt complete dim=1
    #pragma HLS array_partition variable=colsum_prevWIN_IyIt complete dim=1
    // clang-format on

    for (int i = 0; i < WINSIZE; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=11
        // clang-format on
        for (int j = 0; j < cols + (WINSIZE >> 1); j++) {
// clang-format off
            #pragma HLS pipeline ii=1
            #pragma HLS LOOP_FLATTEN OFF
            #pragma HLS LOOP_TRIPCOUNT min=1 max=1920
            // clang-format on
            bufLines_ix[i][j] = 0;
            bufLines_iy[i][j] = 0;
            bufLines_it[i][j] = 0;
            if (i == 0) {
                colsum_IxIx[j] = 0;
                colsum_IxIy[j] = 0;
                colsum_IyIy[j] = 0;
                colsum_IxIt[j] = 0;
                colsum_IyIt[j] = 0;
            }
        }
    }
    ap_uint<7> lineStore = 0;

#if DEBUG
    char name[200];
    sprintf(name, "sumIxt_hw%d.txt", level);
    FILE* fpixt = fopen(name, "w");
    sprintf(name, "sumIyt_hw%d.txt", level);
    FILE* fpiyt = fopen(name, "w");
    sprintf(name, "sumIx2_hw%d.txt", level);
    FILE* fpix2 = fopen(name, "w");
    sprintf(name, "sumIy2_hw%d.txt", level);
    FILE* fpiy2 = fopen(name, "w");
    sprintf(name, "sumIxy_hw%d.txt", level);
    FILE* fpixy = fopen(name, "w");
#endif

    ap_fixed<SIXIY_WIDTH, SIXIY_INT> sumIx2, sumIy2, sumIxIy;
    ap_fixed<SIXYIT_WIDTH, SIXYIT_INT> sumIxIt, sumIyIt;
    for (ap_uint<16> i = 0; i < rows + (WINSIZE >> 1); i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXHEIGHT
        // clang-format on
        for (ap_uint<16> j = 0; j < cols + (WINSIZE >> 1); j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXWIDTH
            #pragma HLS pipeline ii=1
            #pragma HLS LOOP_FLATTEN OFF
            // clang-format on

            if (j == 0) {
                sumIx2 = 0;
                sumIy2 = 0;
                sumIxIy = 0;
                sumIxIt = 0;
                sumIyIt = 0;
            }
            ap_int<9> regIx = 0, regIy = 0;
            ap_fixed<IT_WIDTH, IT_INT> regIt = 0;
            ap_int<9> top_Ix = 0, top_Iy = 0;
            ap_fixed<IT_WIDTH, IT_INT> top_It = 0;

            ap_fixed<SIXIY_WIDTH, SIXIY_INT> current_ixix = 0, current_iyiy = 0, current_ixiy = 0;
            ap_fixed<SIXYIT_WIDTH, SIXYIT_INT> current_ixit = 0, current_iyit = 0;
            ap_fixed<SIXIY_WIDTH, SIXIY_INT> leftwin_ixix = 0, leftwin_iyiy = 0, leftwin_ixiy = 0;
            ap_fixed<SIXYIT_WIDTH, SIXYIT_INT> leftwin_ixit = 0, leftwin_iyit = 0;

            if (j < cols && i < rows) {
                regIx = strmIx.read();
                regIy = strmIy.read();
                regIt = strmIt.read();
            } else {
                regIx = 0;
                regIy = 0;
                regIt = 0;
            }

            if (j < cols) {
                top_Ix = bufLines_ix[0][j];
                top_Iy = bufLines_iy[0][j];
                top_It = bufLines_it[0][j];
            } else {
                top_Ix = 0;
                top_Iy = 0;
                top_It = 0;
            }
            for (int shiftuprow = 0; shiftuprow < WINSIZE - 1; shiftuprow++) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on
                bufLines_ix[shiftuprow][j] = bufLines_ix[shiftuprow + 1][j];
                bufLines_iy[shiftuprow][j] = bufLines_iy[shiftuprow + 1][j];
                bufLines_it[shiftuprow][j] = bufLines_it[shiftuprow + 1][j];
            }
            bufLines_ix[WINSIZE - 1][j] = regIx;
            bufLines_iy[WINSIZE - 1][j] = regIy;
            bufLines_it[WINSIZE - 1][j] = regIt;

            current_ixix = colsum_IxIx[j] + (regIx * regIx) - (top_Ix * top_Ix);
            current_ixiy = colsum_IxIy[j] + (regIx * regIy) - (top_Ix * top_Iy);
            current_iyiy = colsum_IyIy[j] + (regIy * regIy) - (top_Iy * top_Iy);
            current_ixit = colsum_IxIt[j] + (regIx * regIt) - (top_Ix * top_It);
            current_iyit = colsum_IyIt[j] + (regIy * regIt) - (top_Iy * top_It);

            colsum_IxIx[j] = current_ixix;
            colsum_IxIy[j] = current_ixiy;
            colsum_IyIy[j] = current_iyiy;
            colsum_IxIt[j] = current_ixit;
            colsum_IyIt[j] = current_iyit;

            ap_fixed<SIXIY_WIDTH, SIXIY_INT> prev_win_ixix = colsum_prevWIN_IxIx[0];
            ap_fixed<SIXIY_WIDTH, SIXIY_INT> prev_win_iyiy = colsum_prevWIN_IxIy[0];
            ap_fixed<SIXIY_WIDTH, SIXIY_INT> prev_win_ixiy = colsum_prevWIN_IyIy[0];
            ap_fixed<SIXYIT_WIDTH, SIXYIT_INT> prev_win_ixit = colsum_prevWIN_IxIt[0];
            ap_fixed<SIXYIT_WIDTH, SIXYIT_INT> prev_win_iyit = colsum_prevWIN_IyIt[0];

            for (int shiftregwin = 0; shiftregwin < WINSIZE - 1; shiftregwin++) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on
                colsum_prevWIN_IxIx[shiftregwin] = colsum_prevWIN_IxIx[shiftregwin + 1];
                colsum_prevWIN_IxIy[shiftregwin] = colsum_prevWIN_IxIy[shiftregwin + 1];
                colsum_prevWIN_IyIy[shiftregwin] = colsum_prevWIN_IyIy[shiftregwin + 1];
                colsum_prevWIN_IxIt[shiftregwin] = colsum_prevWIN_IxIt[shiftregwin + 1];
                colsum_prevWIN_IyIt[shiftregwin] = colsum_prevWIN_IyIt[shiftregwin + 1];
            }

            colsum_prevWIN_IxIx[WINSIZE - 1] = current_ixix;
            colsum_prevWIN_IxIy[WINSIZE - 1] = current_ixiy;
            colsum_prevWIN_IyIy[WINSIZE - 1] = current_iyiy;
            colsum_prevWIN_IxIt[WINSIZE - 1] = current_ixit;
            colsum_prevWIN_IyIt[WINSIZE - 1] = current_iyit;
            if (j >= WINSIZE)
            // if(0)
            {
                leftwin_ixix = current_ixix - prev_win_ixix;
                leftwin_ixiy = current_ixiy - prev_win_iyiy;
                leftwin_iyiy = current_iyiy - prev_win_ixiy;
                leftwin_ixit = current_ixit - prev_win_ixit;
                leftwin_iyit = current_iyit - prev_win_iyit;
            } else {
                leftwin_ixix = current_ixix;
                leftwin_ixiy = current_ixiy;
                leftwin_iyiy = current_iyiy;
                leftwin_ixit = current_ixit;
                leftwin_iyit = current_iyit;
            }

            sumIx2 += leftwin_ixix;
            sumIy2 += leftwin_iyiy;
            sumIxIy += leftwin_ixiy;
            sumIxIt += leftwin_ixit;
            sumIyIt += leftwin_iyit;

            ap_fixed<SIXIY_WIDTH, SIXIY_INT> Ix2out = ap_fixed<SIXIY_WIDTH, SIXIY_INT>(sumIx2 >> 2);
            ap_fixed<SIXIY_WIDTH, SIXIY_INT> Iy2out = ap_fixed<SIXIY_WIDTH, SIXIY_INT>(sumIy2 >> 2);
            ap_fixed<SIXIY_WIDTH, SIXIY_INT> IxIyout = ap_fixed<SIXIY_WIDTH, SIXIY_INT>(sumIxIy >> 2);
            ap_fixed<SIXYIT_WIDTH, SIXYIT_INT> IxItout = ap_fixed<SIXYIT_WIDTH, SIXIY_INT>(sumIxIt >> 1);
            ap_fixed<SIXYIT_WIDTH, SIXYIT_INT> IyItout = ap_fixed<SIXYIT_WIDTH, SIXIY_INT>(sumIyIt >> 1);

            if (j >= WINSIZE >> 1 && i >= WINSIZE >> 1) {
                sigmaIx2.write(Ix2out);
                sigmaIy2.write(Iy2out);
                sigmaIxIy.write(IxIyout);
                sigmaIxIt.write(IxItout);
                sigmaIyIt.write(IyItout);
#if DEBUG
                fprintf(fpixt, "%12.4f ", float(IxItout));
                fprintf(fpiyt, "%12.4f ", float(IyItout));
                fprintf(fpix2, "%12.2f ", float(Ix2out));
                fprintf(fpiy2, "%12.2f ", float(Iy2out));
                fprintf(fpixy, "%12.2f ", float(IxIyout));
#endif
            }
        } // end j loop
#if DEBUG
        if (i >= WINSIZE >> 1) {
            fprintf(fpixt, "\n");
            fprintf(fpiyt, "\n");
            fprintf(fpix2, "\n");
            fprintf(fpiy2, "\n");
            fprintf(fpixy, "\n");
        }
#endif
    }
#if DEBUG
    fclose(fpixt);
    fclose(fpiyt);
    fclose(fpix2);
    fclose(fpiy2);
    fclose(fpixy);
#endif
} // end find_G()
#endif
