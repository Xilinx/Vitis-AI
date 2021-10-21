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

#ifndef _XF_MEAN_SHIFT_HPP_
#define _XF_MEAN_SHIFT_HPP_

#include "hls_stream.h"
#include "core/xf_math.h"

// flag to enable regressive object feature copying
#define QU_COPY 0
#define _MST_SETUP_FLAG_ 0

// change the datatypes before configuring TOTAL BINS
#define _MST_TOTAL_BINS_ 512

/* LUT to store squared values which is used in kernel computation which is
 * used in Histogram calculation and weight function
 */
static unsigned int xFTrackmulKernelLut[200] = {
    0,     1,     4,     9,     16,    25,    36,    49,    64,    81,    100,   121,   144,   169,   196,   225,
    256,   289,   324,   361,   400,   441,   484,   529,   576,   625,   676,   729,   784,   841,   900,   961,
    1024,  1089,  1156,  1225,  1296,  1369,  1444,  1521,  1600,  1681,  1764,  1849,  1936,  2025,  2116,  2209,
    2304,  2401,  2500,  2601,  2704,  2809,  2916,  3025,  3136,  3249,  3364,  3481,  3600,  3721,  3844,  3969,
    4096,  4225,  4356,  4489,  4624,  4761,  4900,  5041,  5184,  5329,  5476,  5625,  5776,  5929,  6084,  6241,
    6400,  6561,  6724,  6889,  7056,  7225,  7396,  7569,  7744,  7921,  8100,  8281,  8464,  8649,  8836,  9025,
    9216,  9409,  9604,  9801,  10000, 10201, 10404, 10609, 10816, 11025, 11236, 11449, 11664, 11881, 12100, 12321,
    12544, 12769, 12996, 13225, 13456, 13689, 13924, 14161, 14400, 14641, 14884, 15129, 15376, 15625, 15876, 16129,
    16384, 16641, 16900, 17161, 17424, 17689, 17956, 18225, 18496, 18769, 19044, 19321, 19600, 19881, 20164, 20449,
    20736, 21025, 21316, 21609, 21904, 22201, 22500, 22801, 23104, 23409, 23716, 24025, 24336, 24649, 24964, 25281,
    25600, 25921, 26244, 26569, 26896, 27225, 27556, 27889, 28224, 28561, 28900, 29241, 29584, 29929, 30276, 30625,
    30976, 31329, 31684, 32041, 32400, 32761, 33124, 33489, 33856, 34225, 34596, 34969, 35344, 35721, 36100, 36481,
    36864, 37249, 37636, 38025, 38416, 38809, 39204, 39601};

/* LUT to store square root values, used in xFTrackmulWeight function,
 * to find the weight of every pixel in the object window
 */
const unsigned char xFTrackmulSqrtLut[100] = {
    0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};

/* xFTrackmulBlkReadIn: Block reading inner loop, Reads the single pack and pushes into stream
 * ptr			--> To store the single pack of pixels
 * in			--> input image
 * rows 		--> Image Height
 * cols 		--> Image Width
 * i			--> To calculate the offset
 * input1		--> stream into which we have to push the elements
 * x1     		--> Top left corner x-coordinate
 * y1     		--> Top left corner y-coordinate
 * buf_size		--> number of elements to be read in one row
 */
template <int ROWS, int IN_TC, int COLS, int SRC_T, int ROWS_IMG, int COLS_IMG, int NPC>
void xFTrackmulBlkReadIn(XF_TNAME(SRC_T, NPC) ptr[1],
                         xf::cv::Mat<SRC_T, ROWS_IMG, COLS_IMG, NPC>& _in_mat,
                         int i,
                         hls::stream<XF_TNAME(SRC_T, NPC)>& input1,
                         int x1,
                         int y1,
                         unsigned short buf_size) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    int cols = _in_mat.cols;
    int src_off = (cols >> XF_BITSHIFT(NPC)) * (y1 + i) + (x1 >> XF_BITSHIFT(NPC));
    unsigned short size = 4 << XF_BITSHIFT(NPC);

loop_blockread_inner:
    for (int j = 0; j < buf_size; j++) {
// clang-format off
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=20 max=IN_TC
        // clang-format on
        ptr[0] = _in_mat.read(src_off + j);
        input1.write(ptr[0]);
    }
}

/* xFTrackmulBlkRead: Block reading outer loop
 * 					  calls the inner loop over height times
 * input1	--> stream into which we have to push the elements
 * in		--> input image
 * rows 	--> Image Height
 * cols 	--> Image Width
 * x1     	--> Top left corner x-coordinate
 * y1     	--> Top left corner y-coordinate
 * obj_hgt  --> height of the object
 * obj_wdt  --> width of the object
 * obj_num	--> object number in the video
 */
template <int ROWS, int IN_TC, int COLS, int SRC_T, int ROWS_IMG, int COLS_IMG, int NPC>
void xFTrackmulBlkRead(hls::stream<XF_TNAME(SRC_T, NPC)>& input1,
                       xf::cv::Mat<SRC_T, ROWS_IMG, COLS_IMG, NPC>& _in_mat,
                       uint16_t x1,
                       uint16_t y1,
                       uint16_t obj_hgt,
                       uint16_t obj_wdt) {
    XF_TNAME(SRC_T, NPC) dst[1];
    unsigned short h_y = obj_hgt >> 1;
    unsigned short h_x = obj_wdt >> 1;

    unsigned short buf_size = ((x1 + (h_x << 1)) >> XF_BITSHIFT(NPC)) - (x1 >> XF_BITSHIFT(NPC));

loop_blockread_outer:
    for (int i = 0; i < obj_hgt; i++) {
// clang-format off
        #pragma HLS PIPELINE
        #pragma HLS LOOP_TRIPCOUNT min=20 max=ROWS avg=ROWS
        // clang-format on
        xFTrackmulBlkReadIn<ROWS, IN_TC, COLS, SRC_T, ROWS_IMG, COLS_IMG, NPC>(dst, _in_mat, i, input1, x1, y1,
                                                                               buf_size);
    }
}

/* xFTrackmulFindbin : To find the bin corresponds to the pixel intensity value
 * R,G,B --> R,G,B values of a pixel
 * bin   --> bin value corresponding to the given pixel
 */
static uint16_t xFTrackmulFindbin(unsigned char R, unsigned char G, unsigned char B) {
    uint16_t bin;
    uint16_t r, g, b;
    r = (uint16_t)R >> 5;
    g = (uint16_t)G >> 5;
    b = (uint16_t)B >> 5;
    bin = r + (g << 3) + (b << 6);
    return bin;
}

/*
 * Bin increment and distance computation module
 */
template <int ROWS, int COLS, typename BINTYPE>
void xFTrackmulFindbinIncrement(ap_uint32_t val,
                                short& distance,
                                uint16_t& bin,
                                BINTYPE BIN[ROWS * COLS],
                                uint16_t i,
                                uint16_t j,
                                uint16_t obj_wdt,
                                uint16_t h_x,
                                uint16_t h_y,
                                unsigned int wh) {
    uint8_t R = val.range(7, 0);
    uint8_t G = val.range(15, 8);
    uint8_t B = val.range(23, 16);
    bin = xFTrackmulFindbin(R, G, B);
    uint16_t y = i;
    uint16_t x = j;
    int y_off = i * obj_wdt;
    int loc = y_off + x;
    BIN[loc] = bin;
    int a = y - h_y;
    int b = x - h_x;
    y = __ABS(a);
    x = __ABS(b);
    int xx = xFTrackmulKernelLut[x];
    int yy = xFTrackmulKernelLut[y];
    short K = ((xx + yy) * wh) >> 8; // K is in 0.8 format --------> original kernel K(x,y) = (x*x+y*y)/(w/2)*(h/2)
    if (K <= 256)                    // K is in 0.8 format, comparing with '1' in Q0.8 format
        distance = 256 - K;
}

/* xFTrackmulHist: Reads the values from stream and finds the histogram of the current frame
 * 				   and stores in Pu or Qu depending on frame status, Stores the bin values in _BIN array
 * 				   if frame_status is '0'; store in Qu else in Pu,
 * input	--> stream which contains the input data
 * x1     	--> Top left corner x-coordinate
 * obj_hgt  --> height of the object
 * y1     	--> Top left corner y-coordinate
 * obj_wdt  --> width of the object
 * Qu		--> object histogram
 * Pu		--> Array to store the histogram
 * BIN 		--> An array to store bin values
 * frame_status	--> frame number in video
 */
template <int ROWS, int IN_TC, int COLS, int NPC, int WORDWIDTH, typename QuPuTYPE, typename BINTYPE>
void xFTrackmulHist(hls::stream<XF_SNAME(WORDWIDTH)>& input,
                    uint16_t x1,
                    uint16_t obj_hgt,
                    uint16_t y1,
                    uint16_t obj_wdt,
                    QuPuTYPE Qu[_MST_TOTAL_BINS_],
                    QuPuTYPE Pu[_MST_TOTAL_BINS_],
                    BINTYPE BIN[ROWS * COLS],
                    uint8_t frame_status) {
// clang-format off
    #pragma HLS INLINE OFF
// clang-format on

// clang-format off
    #pragma HLS ARRAY_PARTITION variable=BIN cyclic factor=2 dim=1
    #pragma HLS DEPENDENCE variable=BIN array inter false
    // clang-format on

    QuPuTYPE tmp_hist1[_MST_TOTAL_BINS_];
    QuPuTYPE tmp_hist2[_MST_TOTAL_BINS_];

    uint16_t buf_size;
    uint16_t h_y = obj_hgt >> 1;
    uint16_t h_x = obj_wdt >> 1;

    //	buf_size = h_y;
    buf_size = ((x1 + (h_x << 1)) >> XF_BITSHIFT(NPC)) - (x1 >> XF_BITSHIFT(NPC));

    char shift1, shift2;
    unsigned int _height1 = xf::cv::Inverse(h_y, 16, &shift1); // Q(32-shift1).shift1 -----> format
    unsigned int _width1 = xf::cv::Inverse(h_x, 16, &shift2);  // Q(32-shift2).shift2 -----> format
    unsigned long int temp = _height1 * _width1;               // Q(64-shift1-shift2).(shift1+shift2) -----> format
    unsigned int wh = temp >> (shift1 + shift2 - 16);          // wh = 1/(w/2*h/2)  --------> 0.16 format

loop_hist_init:
    for (uint16_t i = 0; i < _MST_TOTAL_BINS_; i++) {
        tmp_hist1[i] = 0;
        tmp_hist2[i] = 0;
    }

loop_hist_height:
    for (uint16_t i = 0; i < obj_hgt; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=20 max=ROWS
    // clang-format on

    loop_hist_width:
        for (uint16_t j = 0; j < buf_size; j = j + 2) {
// clang-format off
            #pragma HLS LOOP_FLATTEN off
            #pragma HLS PIPELINE
            #pragma HLS LOOP_TRIPCOUNT min=20 max=IN_TC
            // clang-format on

            XF_SNAME(WORDWIDTH) val1 = input.read();
            XF_SNAME(WORDWIDTH) val2 = input.read();
            short dist1 = 0, dist2 = 0;
            uint16_t bin1, bin2;

            xFTrackmulFindbinIncrement<ROWS, COLS>((ap_uint32_t)val1, dist1, bin1, BIN, i, j, obj_wdt, h_x, h_y, wh);
            xFTrackmulFindbinIncrement<ROWS, COLS>((ap_uint32_t)val2, dist2, bin2, BIN, i, j + 1, obj_wdt, h_x, h_y,
                                                   wh);

            tmp_hist1[bin1] += dist1;
            tmp_hist2[bin2] += dist2;
        }
    }

// Accumulate the temporary histograms
loop_hist_accumulate:
    for (uint16_t i = 0; i < _MST_TOTAL_BINS_; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=512 max=512 avg=512
        #pragma HLS PIPELINE
        // clang-format on

        if (frame_status == _MST_SETUP_FLAG_)
            Qu[i] = tmp_hist1[i] + tmp_hist2[i];
        else
            Pu[i] = tmp_hist1[i] + tmp_hist2[i];
    }
}

/*xFTrackmulSqrt: Finds the square root of the given number using an xFTrackmulSqrtLut
 */
static int xFTrackmulSqrt(int temp) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    if (temp < 100)
        return xFTrackmulSqrtLut[temp];
    else
        return 10;
}

/* xFTrackmulWeight: Calculates the displacement in the center of the rectangle using
 * 					 Histograms Pu,Qu,BIN arrays
 * Qu		--> Model histogram
 * Pu		--> Current frame histogram
 * BIN 		--> An array which contains the bin values
 * dx   	--> displacement of center x-coordinate
 * dy   	--> displacement of center y-coordinate
 * x1     	--> Top left corner x-coordinate
 * y1     	--> Top left corner y-coordinate
 * obj_hgt  --> height of the object
 * obj_wdt  --> width of the object
 * C_x   	--> temporary histogram used in RO, PO cases for parallel processing
 * C_x  	--> object number in the video
 * track    --> track status of the current object
 * rows     --> height of the image
 * cols     --> width of the image
 */
template <int ROWS, int IN_TC, int COLS, int NPC, typename QuPuTYPE, typename BINTYPE>
void xFTrackmulWeight(QuPuTYPE Qu[_MST_TOTAL_BINS_],
                      QuPuTYPE Pu[_MST_TOTAL_BINS_],
                      BINTYPE BIN[ROWS * COLS],
                      uint16_t& x1,
                      uint16_t& y1,
                      uint16_t obj_hgt,
                      uint16_t obj_wdt,
                      uint16_t& C_x,
                      uint16_t& C_y,
                      bool& track,
                      uint16_t rows,
                      uint16_t cols) {
    BINTYPE loc, bin;
    uint16_t x, y;
    int total_x = 0, total_y = 0, total_w = 0, K, xx, yy;
    short weight;
    short A, B;
    uint16_t dispx = 0, dispy = 0;

    short buf_size;
    int y_off, a, b;

    unsigned short int h_x = obj_wdt >> 1;
    unsigned short int h_y = obj_hgt >> 1;

    buf_size = ((y1 + (h_y << 1)) >> XF_BITSHIFT(NPC)) - (y1 >> XF_BITSHIFT(NPC)) + 1;

    char shift1, shift2;

    unsigned int _width1 = xf::cv::Inverse(h_x, 16, &shift1);  // Q(32-shift1).shift1 -----> format
    unsigned int _height1 = xf::cv::Inverse(h_y, 16, &shift2); // Q(32-shift2).shift2 -----> format
    unsigned long int temp = _width1 * _height1;               // Q(64-shift1-shift2).(shift1+shift2) -----> format
    unsigned int wh = temp >> (shift1 + shift2 - 16);          // wh = 1/(w/2*h/2)  --------> 0.16 format

loop_weight_height:
    for (uint16_t i = 0; i < obj_hgt; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=20 max=ROWS avg=ROWS
        // clang-format on
        y_off = i * obj_wdt;

    loop_weight_width:
        for (uint16_t j = 0; j < buf_size; j++) {
// clang-format off
            #pragma HLS LOOP_FLATTEN off
            #pragma HLS PIPELINE
            #pragma HLS LOOP_TRIPCOUNT min=20 max=IN_TC avg=IN_TC
            // clang-format on

            y = i;
            x = j;
            loc = y_off + x;
            bin = BIN[loc];
            A = y - h_y;
            B = x - h_x;
            y = __ABS(A);
            x = __ABS(B);
            xx = xFTrackmulKernelLut[x];
            yy = xFTrackmulKernelLut[y];
            K = ((xx + yy) * wh) >> 8;

            // weight computation
            if (K > 256 || Qu[bin] == 0 || Pu[bin] >> 8 == 0)
                weight = 0;
            else {
                a = Qu[bin] >> 8;
                b = Pu[bin] >> 8;
                weight = xFTrackmulSqrt(a / b);
            }

            total_y += (weight * A);
            total_x += (weight * B);
            total_w += weight;
        }
    }

    // displacement computation
    if (total_w != 0) {
        dispx = (long int)total_x / (long int)total_w;
        dispy = (long int)total_y / (long int)total_w;
    } else {
        dispx = 0;
        dispy = 0;
    }

    // Add the displacement to the previous center
    C_x += dispx;
    C_y += dispy;

    // Check if the object goes out of the frame
    if (C_y + h_y >= rows) {
        C_y = rows - h_y - 1;
        track = 0;
    }
    if (C_x + h_x >= cols) {
        C_x = cols - h_x - 1;
        track = 0;
    }
    if (C_y - h_y <= 0) {
        C_y = h_y + 1;
        track = 0;
    }
    if (C_x - h_x <= 0) {
        C_x = h_x + 1;
        track = 0;
    }

    // Update the corners of the rectangle
    x1 = (uint16_t)(C_x - h_x);
    y1 = (uint16_t)(C_y - h_y);
}

/* xFTrackmulFindhist: Reads the window from DDR3 and computes the histogram and stores in Pu or Qu
 * 					   depending on frameno,and stores the bin values in the array BIN
 * 					   if frameno = 0; store in Qu else in Pu
 * in		--> Input image
 * rows 	--> Image Height
 * cols 	--> Image Width
 * x1     	--> Top left corner x-coordinate
 * y1     	--> Top left corner y-coordinate
 * obj_hgt  --> height of the object
 * obj_wdt  --> width of the object
 * Qu		--> Array to store the histograms of the objects in the first frame
 * Pu		--> array to store the histogram of current object
 * BIN 		--> An array to store the bin values
 * tmp_hist	--> temporary histogram used in RO, PO cases to compute histogram in parallel
 * obj_num 	--> object number in the video
 * frameno	--> frame number in video
 */
template <int ROWS,
          int IN_TC,
          int COLS,
          int SRC_T,
          int ROWS_IMG,
          int COLS_IMG,
          int NPC,
          typename QuPuTYPE,
          typename BINTYPE>
void xFTrackmulFindhist(xf::cv::Mat<SRC_T, ROWS_IMG, COLS_IMG, NPC>& _in_mat,
                        uint16_t x1,
                        uint16_t y1,
                        uint16_t obj_hgt,
                        uint16_t obj_wdt,
                        QuPuTYPE Qu[_MST_TOTAL_BINS_],
                        QuPuTYPE Pu[_MST_TOTAL_BINS_],
                        BINTYPE BIN[ROWS * COLS],
                        uint8_t frame_status) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    hls::stream<XF_TNAME(SRC_T, NPC)> input2;

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    // Read the block from DDR and push into stream
    xFTrackmulBlkRead<ROWS, IN_TC, COLS, SRC_T, ROWS_IMG, COLS_IMG, NPC>(input2, _in_mat, x1, y1, obj_hgt, obj_wdt);

    // Read the values from stream and find the histogram
    xFTrackmulHist<ROWS, (IN_TC >> 1), COLS, NPC, XF_WORDWIDTH(SRC_T, NPC)>(input2, x1, obj_hgt, y1, obj_wdt, Qu, Pu,
                                                                            BIN, frame_status);
}

/*
 * xFTrackmulKernelFunc: Kernel function which gives the next centroid given the earlier one
 * in		--> Input image
 * rows 	--> Image Height
 * cols 	--> Image Width
 * x1     	--> Top left corner x-coordinate
 * y1     	--> Top left corner y-coordinate
 * obj_hgt  --> height of the object
 * obj_wdt  --> width of the object
 * dx   	--> New center x-coordinate
 * dy   	--> New center y-coordinate
 * track 	--> object status, indicated if its valid for tracking in consecutive frames
 * frame_status --> current frame status
 * obj_num --> current object index
 * iters    --> Total number of iterations for the convergence
 */
template <int ROWS, int IN_TC, int COLS, int SRC_T, int ROWS_IMG, int COLS_IMG, int MAXOBJS, int MAXITERS, int NPC>
void xFTrackmulKernelFunc(xf::cv::Mat<SRC_T, ROWS_IMG, COLS_IMG, NPC>& _in_mat,
                          uint16_t x1,
                          uint16_t y1,
                          uint16_t obj_hgt,
                          uint16_t obj_wdt,
                          uint16_t& dx,
                          uint16_t& dy,
                          bool& track,
                          uint8_t frame_status,
                          uint8_t obj_num,
                          uint8_t iters) {
    uint16_t C_x = ((obj_wdt) >> 1) + x1;
    uint16_t C_y = ((obj_hgt) >> 1) + y1;

    uint8_t loop_count = iters << 1;
    // setup the object feature for the first frame
    if (frame_status == _MST_SETUP_FLAG_) loop_count = 1;

    // static array store the original object feature,
    // features stored when the first frame is processed and
    // for remaining frames data is read from it for the kernel computation
    static uint32_t Qu[MAXOBJS][_MST_TOTAL_BINS_];
    uint32_t Pu[_MST_TOTAL_BINS_];

    // storage to hold the bins values of each pixel
    uint16_t BIN[ROWS * COLS];
// clang-format off
    #pragma HLS DEPENDENCE variable=BIN array intra false
    // clang-format on

    uint16_t h_x = obj_wdt >> 1;
    uint16_t h_y = obj_hgt >> 1;

    bool flag = 0;
// For other frames, find histogram as well as centroid in iterative manner
loop_iterations:
    for (uint8_t i = 0; i < loop_count; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXITERS avg=MAXITERS
        // clang-format on

        if (flag == 0) {
            // Find histogram of the current frame and store in array Pu[512]
            xFTrackmulFindhist<ROWS, IN_TC, COLS, SRC_T, ROWS_IMG, COLS_IMG, NPC>(_in_mat, x1, y1, obj_hgt, obj_wdt,
                                                                                  Qu[obj_num], Pu, BIN, frame_status);
            flag = 1;
        }

        else {
            // Using Pu, Qu compute weights and displacement
            xFTrackmulWeight<ROWS, IN_TC, COLS, NPC>(Qu[obj_num], Pu, BIN, x1, y1, obj_hgt, obj_wdt, C_x, C_y, track,
                                                     _in_mat.rows, _in_mat.cols);
            flag = 0;
        }
    }

#if QU_COPY
loop_qu_copy:
    for (uint16_t i = 0; i < _MST_TOTAL_BINS_; i++) {
        Qu[obj_num][i] = Pu[i];
    }
#endif

    dx = C_x;
    dy = C_y;
}

/*
 * xFTrackmulFinalTop: calls the kernel function with the corresponding object coordinates and object number
 * in     	--> Input image
 * rows 	--> Image Height
 * cols 	--> Image Width
 * tlx     	--> Top left corner x-coordinate
 * tly     	--> Top left corner y-coordinate
 * obj_hgt  --> height of the object
 * obj_wdt  --> width of the object
 * dispx   	--> New center x-coordinate
 * dispy   	--> New center y-coordinate
 * status 	--> object status, indicated if its valid for tracking in consecutive frames
 * frame_status --> '0' if first frame, else '1'
 * no_objects --> total number of objects for tracking
 * iters    --> Total number of iterations for the centroid convergence, optimally '4'
 */

template <int ROWS, int COLS, int SRC_T, int ROWS_IMG, int COLS_IMG, int MAXOBJ, int MAXITERS, int NPC>
void xFMeanShiftKernel(xf::cv::Mat<SRC_T, ROWS_IMG, COLS_IMG, NPC>& _in_mat,
                       uint16_t tlx[MAXOBJ],
                       uint16_t tly[MAXOBJ],
                       uint16_t obj_hgt[MAXOBJ],
                       uint16_t obj_wdt[MAXOBJ],
                       uint16_t dispx[MAXOBJ],
                       uint16_t dispy[MAXOBJ],
                       uint16_t status[MAXOBJ],
                       uint8_t frame_status,
                       uint8_t no_objects,
                       uint8_t iters) {
    //#pragma HLS license key=IPAUVIZ_MST
    uint16_t dx, dy;
    uint8_t a;
    uint16_t x1, x2, y1, y2;
    bool track;

#ifndef __SYNTHESIS__
    assert((no_objects <= MAXOBJ) && "number of objects should be less than MAX_OBJECTS");
    assert((NPC == XF_NPPC1) && "NPC must be XF_NPPC1");
    //	assert((WORDWIDTH == XF_32UW) &&
    //			"WORDWIDTH must be XF_32UW");
    assert((COLS % 2 == 0) && "object width must be in multiples of two");
#endif

loop_objects:
    for (uint8_t i = 0; i < no_objects; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXOBJ
// clang-format on

#ifndef __SYNTHESIS__
        assert((obj_wdt[i] % 2 == 0) && "object width must be in multiples of two");
#endif

        a = i;
        x1 = tlx[i];
        y1 = tly[i];
        x2 = obj_wdt[i];
        y2 = obj_hgt[i];
        track = (bool)status[i];

#ifndef __SYNTHESIS__
        assert((x2 < 700) && (y2 < 700) && "object width and height should be less than 700");
        assert((x2 > 20) && (y2 > 20) && "object width and height should be greater than 20");
        assert((x2 <= COLS) && "The object width must be less than the MAX_WIDTH ");
        assert((y2 <= ROWS) && "The object height must be less than the MAX_HEIGHT ");
#endif

        if (track) {
            xFTrackmulKernelFunc<ROWS, (COLS >> XF_BITSHIFT(NPC)), COLS, SRC_T, ROWS_IMG, COLS_IMG, MAXOBJ,
                                 (MAXITERS << 1), NPC>(_in_mat, x1, y1, x2, y2, dx, dy, track, frame_status, a, iters);
        } else // If non-trackable displacement is Zero
        {
            dx = 0;
            dy = 0;
        }

        status[a] = (uint16_t)track;
        dispx[a] = dx;
        dispy[a] = dy;
    }
}
#endif // _XF_MEAN_SHIFT_HPP_
