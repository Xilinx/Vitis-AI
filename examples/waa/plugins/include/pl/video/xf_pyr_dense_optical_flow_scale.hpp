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

#ifndef __XF_PYR_DENSE_OPTICAL_FLOW_SCALE__
#define __XF_PYR_DENSE_OPTICAL_FLOW_SCALE__

template <int MAXWIDTH, int FLOW_WIDTH, int FLOW_INT, int SCCMP_WIDTH, int SCCMP_INT, int SCALE_WIDTH, int SCALE_INT>
void load_data(hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& inStrm,
               ap_fixed<FLOW_WIDTH, FLOW_INT> buf[MAXWIDTH],
               int rows,
               int cols,
               bool& flagLoaded,
               int inCurrRow,
               ap_ufixed<SCALE_WIDTH, SCALE_INT> scaleI,
               ap_fixed<SCCMP_WIDTH, SCCMP_INT>& fracI,
               int& prevIceil) {
// clang-format off
    #pragma HLS inline off
    // clang-format on
    // Calculate the input row needed to compute the current output
    ap_fixed<SCCMP_WIDTH, SCCMP_INT> iSmall = inCurrRow * scaleI;
    // integer index of the input row needed to compute the output row
    int iSmallFloor = (int)iSmall;
    // fractional value of the input row, i.e., weight needed for bilateral interpolation
    fracI = iSmall - (ap_fixed<SCCMP_WIDTH, SCCMP_INT>)iSmallFloor;
    // two rows are needed for bilinear interpolation. So, if the second row is not already in the buffer, read another
    // row. this is also enabled when the row count is less than 2
    if ((iSmallFloor + 1 > prevIceil || inCurrRow < 2) && (iSmallFloor < rows - 1)) {
        // setting a flag that the input is read
        flagLoaded = 1;
        for (int i = 0; i < cols; i++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXWIDTH
            #pragma HLS pipeline ii=1
            #pragma HLS LOOP_FLATTEN OFF
            // clang-format on
            buf[i] = inStrm.read();
        }
        // after the read, increment the input row by 1
        prevIceil = iSmallFloor + 1;
    } else {
        // setting a flag that the input is not read
        flagLoaded = 0;
    }
} // end load_data()

template <int FLOW_WIDTH, int FLOW_INT, int SCCMP_WIDTH, int SCCMP_INT, int RMAPPX_WIDTH, int RMAPPX_INT>
ap_fixed<FLOW_WIDTH, FLOW_INT> compute_result(ap_fixed<SCCMP_WIDTH, SCCMP_INT> fracI,
                                              ap_fixed<SCCMP_WIDTH, SCCMP_INT> fracJ,
                                              ap_fixed<FLOW_WIDTH, FLOW_INT> i0,
                                              ap_fixed<FLOW_WIDTH, FLOW_INT> i1,
                                              ap_fixed<FLOW_WIDTH, FLOW_INT> i2,
                                              ap_fixed<FLOW_WIDTH, FLOW_INT> i3) {
// clang-format off
    #pragma HLS inline off
    // clang-format on
    ap_fixed<18, 1> fi = (fracI);
    ap_fixed<18, 1> fj = (fracJ);
    ap_fixed<36, 1> fij = (ap_fixed<36, 1>)fi * (ap_fixed<36, 1>)fj;

    ap_fixed<18, 1> p3 = (ap_fixed<18, 1>)fij;
    ap_fixed<18, 1> p2 = (ap_fixed<18, 1>)((ap_fixed<36, 1>)fi - fij);
    ap_fixed<18, 1> p1 = (ap_fixed<18, 1>)((ap_fixed<36, 1>)fj - fij);
    ap_fixed<21, 4> p0 = ap_fixed<21, 4>(1.0) - ap_fixed<21, 4>(p1) - ap_fixed<21, 4>(p2) - ap_fixed<21, 4>(p3);
    ap_fixed<FLOW_WIDTH + 2, FLOW_INT + 2> resIf =
        (ap_fixed<FLOW_WIDTH + 2, FLOW_INT + 2>)i0 * p0 + (ap_fixed<FLOW_WIDTH + 2, FLOW_INT + 2>)i1 * p1 +
        (ap_fixed<FLOW_WIDTH + 2, FLOW_INT + 2>)i2 * p2 + (ap_fixed<FLOW_WIDTH + 2, FLOW_INT + 2>)i3 * p3;
    return (ap_fixed<FLOW_WIDTH, FLOW_INT>)resIf;
} // end compute_result()

template <unsigned short MAXHEIGHT,
          unsigned short MAXWIDTH,
          int FLOW_WIDTH,
          int FLOW_INT,
          int SCCMP_WIDTH,
          int SCCMP_INT,
          int RMAPPX_WIDTH,
          int RMAPPX_INT,
          int SCALE_WIDTH,
          int SCALE_INT>
void process(ap_fixed<FLOW_WIDTH, FLOW_INT> buf[MAXWIDTH],
             ap_fixed<FLOW_WIDTH, FLOW_INT> buffer[2][MAXWIDTH],
             unsigned short int outRows,
             unsigned short int outCols,
             hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& outStrm,
             bool flagLoaded,
             int row,
             ap_ufixed<SCALE_WIDTH, SCALE_INT> scaleI,
             ap_ufixed<SCALE_WIDTH, SCALE_INT> scaleJ,
             ap_fixed<SCCMP_WIDTH, SCCMP_INT> fracI,
             int mul) {
// clang-format off
    #pragma HLS array_partition variable=buffer dim=1 complete
    #pragma HLS inline off
    // clang-format on
    int bufCount = 0;
    ap_fixed<FLOW_WIDTH, FLOW_INT> regLoad;
    int prevJceil = -1;
    ap_fixed<FLOW_WIDTH, FLOW_INT> i0 = 0, i1 = 0, i2 = 0, i3 = 0;
L3:
    for (ap_uint<16> j = 0; j < outCols; j++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXWIDTH
        #pragma HLS pipeline
        #pragma HLS LOOP_FLATTEN OFF
        #pragma HLS DEPENDENCE variable=buffer array inter false
        // clang-format on
        // calculate the current input column index needed for the current output
        ap_fixed<SCCMP_WIDTH, SCCMP_INT> jSmall = j * scaleJ;
        // integer part
        int jSmallFloor = (int)jSmall;
        // calculate the current input row index needed for the current output
        ap_fixed<SCCMP_WIDTH, SCCMP_INT> iSmall = row * scaleI;
        // integer part
        int iSmallFloor = (int)iSmall;
        // fractional index
        ap_fixed<SCCMP_WIDTH, SCCMP_INT> fracI = iSmall - (ap_fixed<SCCMP_WIDTH, SCCMP_INT>)iSmallFloor;
        ap_fixed<SCCMP_WIDTH, SCCMP_INT> fracJ = jSmall - (ap_fixed<SCCMP_WIDTH, SCCMP_INT>)jSmallFloor;

        // copy the input buffer buf into the internal buffer 'buffer' while shifting the row of the buffer up
        // i.e., buffer[0][column] = buffer[1][column]; buffer[1][column] = current read value
        // for the first row
        if (row == 0) {
            // only one row is available to process hence fractional index is 1
            fracI = 1;
            // when column count is 0, for the first pixel the left pixel i1 = 0 and all the other pixels are 0
            // only when the prevJceil is equal to the current column index, i.e., when another pixel is needed for
            // computing the next pixel, a pixel is read or no pixel is read from the input. each iteration, i2 = i3 and
            // i3 = current read value, top row is 0, hence i1 and i0 are always 0
            if (j == 0) {
                ap_fixed<FLOW_WIDTH, FLOW_INT> reg = buf[bufCount];
                buffer[1][bufCount] = reg;
                i3 = reg;
                fracI = 1;
                fracJ = 1;
                bufCount++;
                prevJceil = 0;
            } else if (j < outCols) {
                if (prevJceil == jSmallFloor) {
                    i2 = i3;
                    ap_fixed<FLOW_WIDTH, FLOW_INT> reg = buf[bufCount];
                    buffer[1][bufCount] = reg;
                    i3 = reg;
                    bufCount++;
                    prevJceil = jSmallFloor + 1;
                }
            } else {
                i3 = buffer[1][bufCount - 1];
                fracI = 1;
                fracJ = 1;
            }
        }
        // rows > 0 are processed, i0 and i2 are previous i1 and i3 and the current i1 and i3 are the current column
        // reads. again, the internal buffer is loaded with the input buf values. This happens only when a input row is
        // read during the previous iteration
        else if (row < outRows - 1) {
            if (j == 0) {
                i0 = 0;
                i2 = 0;
                fracJ = 1;
                if (flagLoaded) {
                    ap_fixed<FLOW_WIDTH, FLOW_INT> reg = buf[bufCount];
                    ap_fixed<FLOW_WIDTH, FLOW_INT> tmp = buffer[1][bufCount];
                    buffer[0][bufCount] = tmp;
                    i1 = tmp;
                    buffer[1][bufCount] = reg;
                    i3 = reg;
                    bufCount++;
                } else {
                    i1 = buffer[0][bufCount];
                    i3 = buffer[1][bufCount];
                    bufCount++;
                }
                prevJceil = 0;
            } else if (j < outCols) {
                if (prevJceil == jSmallFloor) {
                    i0 = i1;
                    i2 = i3;
                    if (flagLoaded) {
                        ap_fixed<FLOW_WIDTH, FLOW_INT> reg = buf[bufCount];
                        ap_fixed<FLOW_WIDTH, FLOW_INT> tmp = buffer[1][bufCount];
                        buffer[0][bufCount] = tmp;
                        i1 = tmp;
                        buffer[1][bufCount] = reg;
                        i3 = reg;
                        bufCount++;
                    } else {
                        i1 = buffer[0][bufCount];
                        i3 = buffer[1][bufCount];
                        bufCount++;
                    }
                    prevJceil = jSmallFloor + 1;
                }
            } else {
                fracJ = 1;
            }
        }
        // for the final row, only one row is processed, the fracI index is always 1. i2 = previous iteration's i3 and
        // i3 is the current buf read.
        else {
            if (j == 0) {
                i3 = buffer[1][bufCount];
                fracI = 1;
                fracJ = 1;
                bufCount++;
                prevJceil = 0;
            } else if (j < outCols) {
                if (prevJceil == jSmallFloor) {
                    i2 = i3;
                    ap_fixed<FLOW_WIDTH, FLOW_INT> reg = buffer[1][bufCount];
                    i3 = reg;
                    bufCount++;
                    prevJceil = jSmallFloor + 1;
                }
                fracI = 1;
            } else {
                i3 = buffer[1][bufCount - 1];
                fracI = 1;
                fracJ = 1;
            }

        } // end else
        // bilinear interpolation equation.
        ap_fixed<FLOW_WIDTH, FLOW_INT> resIf =
            compute_result<FLOW_WIDTH, FLOW_INT, SCCMP_WIDTH, SCCMP_INT, RMAPPX_WIDTH, RMAPPX_INT>(fracI, fracJ, i0, i1,
                                                                                                   i2, i3);

        // multiply the interpolation result by 2 as the image is scaled up by a factor of about 2 and the pixel
        // displacements are scaled up by a factor of 2 too.
        outStrm.write(resIf << 1);

    } // end L3
} // end process()
template <unsigned short MAXHEIGHT,
          unsigned short MAXWIDTH,
          int FLOW_WIDTH,
          int FLOW_INT,
          int SCCMP_WIDTH,
          int SCCMP_INT,
          int RMAPPX_WIDTH,
          int RMAPPX_INT,
          int SCALE_WIDTH,
          int SCALE_INT,
          bool USE_URAM>
void scale_up(hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& inStrm,
              hls::stream<ap_fixed<FLOW_WIDTH, FLOW_INT> >& outStrm,
              unsigned short int inRows,
              unsigned short int inCols,
              unsigned short int outRows,
              unsigned short int outCols,
              int mul,
              const bool scale_up_flag,
              float scale_comp) {
// clang-format off
    #pragma HLS inline off
    // clang-format on
    // Buffer to store two rows of the input image. These rows are updated in the process function
    ap_fixed<FLOW_WIDTH, FLOW_INT> buffer[2][MAXWIDTH];
    if (USE_URAM) {
// clang-format off
        #pragma HLS array_reshape variable=buffer dim=1 complete
        // clang-format on
    } else {
// clang-format off
        #pragma HLS array_partition variable=buffer dim=1 complete
        // clang-format on
    }
    // buf0 and buf1 are used as ping pong buffers to read and process. While one buffer is used to read the input
    // image, the other buffer is copied into the buffer variable declared above
    ap_fixed<FLOW_WIDTH, FLOW_INT> buf0[MAXWIDTH], buf1[MAXWIDTH];

    if (USE_URAM) {
// clang-format off
        #pragma HLS array_reshape variable=buf0 dim=1 complete
        #pragma HLS array_reshape variable=buf1 dim=1 complete
        #pragma HLS RESOURCE variable=buffer core=RAM_S2P_URAM
        #pragma HLS RESOURCE variable=buf0   core=RAM_S2P_URAM
        #pragma HLS RESOURCE variable=buf1   core=RAM_S2P_URAM
        // clang-format on
    }

    // Copy input scale into the following variable
    ap_ufixed<SCALE_WIDTH, SCALE_INT> scaleI = (ap_ufixed<SCALE_WIDTH, SCALE_INT>)scale_comp;
    ap_ufixed<SCALE_WIDTH, SCALE_INT> scaleJ = (ap_ufixed<SCALE_WIDTH, SCALE_INT>)scale_comp;
#if DEBUG
    cout << "Scale Flag: " << scale_up_flag << "\n";
    cout << "Scale Comp: " << scale_comp << "\n";
    cout << "Scale: " << float(scaleJ) << " " << float(scaleI) << "\n";
#endif
    // Variables to store the bilinear interpolation weights
    ap_fixed<SCCMP_WIDTH, SCCMP_INT> fracI0, fracI1;

    // flags to mark if the buffer is read
    bool flagLoaded0, flagLoaded1;
    // ping-pong operation flag
    bool flag = 0;

    // if the input scale-up flag is 0, i.e., if this module needs to be bypassed, the input stream is copied to the
    // output stream
    if (scale_up_flag == 0) {
        for (ap_uint<16> i = 0; i < outRows; i++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXHEIGHT
            // clang-format on
            for (ap_uint<16> j = 0; j < outCols; j++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXWIDTH
                #pragma HLS pipeline II=1
                #pragma HLS LOOP_FLATTEN OFF
                // clang-format on
                outStrm.write((ap_fixed<FLOW_WIDTH, FLOW_INT>)inStrm.read());
            }
        }
    }
    // Scale up enabled
    else {
        int prevIceil = -1;
        // load first row into the buf0 so that the output processing can have two rows at the same time.
        load_data<MAXWIDTH, FLOW_WIDTH, FLOW_INT, SCCMP_WIDTH, SCCMP_INT, SCALE_WIDTH, SCALE_INT>(
            inStrm, buf0, inRows, inCols, flagLoaded0, 0, scaleI, fracI0, prevIceil);
    // run the ping pong buffer for outRows -1 times
    L2:
        for (ap_uint<16> i = 0; i < outRows - 1; i++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXHEIGHT
            // clang-format on
            if (flag == 0) {
                load_data<MAXWIDTH, FLOW_WIDTH, FLOW_INT, SCCMP_WIDTH, SCCMP_INT, SCALE_WIDTH, SCALE_INT>(
                    inStrm, buf1, inRows, inCols, flagLoaded1, i + 1, scaleI, fracI1, prevIceil);
                process<MAXHEIGHT, MAXWIDTH, FLOW_WIDTH, FLOW_INT, SCCMP_WIDTH, SCCMP_INT, RMAPPX_WIDTH, RMAPPX_INT,
                        SCALE_WIDTH, SCALE_INT>(buf0, buffer, outRows, outCols, outStrm, flagLoaded0, i, scaleI, scaleJ,
                                                fracI0, mul);
                flag = 1;
            } else {
                load_data<MAXWIDTH, FLOW_WIDTH, FLOW_INT, SCCMP_WIDTH, SCCMP_INT, SCALE_WIDTH, SCALE_INT>(
                    inStrm, buf0, inRows, inCols, flagLoaded0, i + 1, scaleI, fracI0, prevIceil);
                process<MAXHEIGHT, MAXWIDTH, FLOW_WIDTH, FLOW_INT, SCCMP_WIDTH, SCCMP_INT, RMAPPX_WIDTH, RMAPPX_INT,
                        SCALE_WIDTH, SCALE_INT>(buf1, buffer, outRows, outCols, outStrm, flagLoaded1, i, scaleI, scaleJ,
                                                fracI1, mul);
                flag = 0;
            }
        } // end L2

        if (flag == 0) {
            process<MAXHEIGHT, MAXWIDTH, FLOW_WIDTH, FLOW_INT, SCCMP_WIDTH, SCCMP_INT, RMAPPX_WIDTH, RMAPPX_INT>(
                buf0, buffer, outRows, outCols, outStrm, flagLoaded0, outRows - 1, scaleI, scaleJ, fracI0, mul);
        } else {
            process<MAXHEIGHT, MAXWIDTH, FLOW_WIDTH, FLOW_INT, SCCMP_WIDTH, SCCMP_INT, RMAPPX_WIDTH, RMAPPX_INT>(
                buf1, buffer, outRows, outCols, outStrm, flagLoaded1, outRows - 1, scaleI, scaleJ, fracI1, mul);
        }
    }

} // end scale_up

#endif
