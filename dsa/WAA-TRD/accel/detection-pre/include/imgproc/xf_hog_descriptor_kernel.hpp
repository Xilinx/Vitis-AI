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

#ifndef _XF_HOG_DESCRIPTOR_KERNEL_HPP_
#define _XF_HOG_DESCRIPTOR_KERNEL_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif

#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "core/xf_math.h"

#include "imgproc/xf_hog_descriptor_utility.hpp"
#include "imgproc/xf_hog_descriptor_gradients.hpp"
#include "imgproc/xf_hog_descriptor_pm.hpp"
#include "imgproc/xf_hog_descriptor_hist_norm.hpp"

/********************************************************************************************
 * 						xFDHOG function
 ********************************************************************************************
 *   This function calls the various pipelined functions for computing the HoG descriptors.
 *
 *   _in_stream: input image stream
 *   _block_stream: block stream (O) desc data written to this stream
 *
 ********************************************************************************************/
template <int WIN_HEIGHT,
          int WIN_WIDTH,
          int WIN_STRIDE,
          int CELL_HEIGHT,
          int CELL_WIDTH,
          int NOB,
          int NOHCPB,
          int NOVCPB,
          int MAT_WW,
          int ROWS,
          int COLS,
          int DEPTH,
          int DEPTH_BLOCK,
          int NPC,
          int WORDWIDTH,
          int WORDWIDTH_BLOCK,
          int NOC,
          bool USE_URAM>
void xFDHOGKernel(hls::stream<XF_SNAME(WORDWIDTH)> _in_stream[NOC],
                  hls::stream<XF_SNAME(WORDWIDTH_BLOCK)>& _block_stream,
                  uint16_t _height,
                  uint16_t _width) {
    // streams for dataflow between various processes
    hls::stream<XF_SNAME(MAT_WW)> grad_x_stream, grad_y_stream;
    hls::stream<XF_SNAME(XF_16UW)> phase_stream("phase_stream"), mag_stream("mag_stream");

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    //  gradient computation
    xFHOGgradients<ROWS, COLS, DEPTH, XF_9SP, XF_NPPC1, WORDWIDTH, MAT_WW, NOC, USE_URAM>(
        _in_stream, grad_x_stream, grad_y_stream, XF_BORDER_CONSTANT, _height, _width);

    // finding the magnitude and the phase for the gradient data
    xFHOGPhaseMagnitude<ROWS, COLS, XF_9SP, XF_16UP, XF_NPPC1, MAT_WW, XF_16UW>(
        grad_x_stream, grad_y_stream, phase_stream, mag_stream, _height, _width);

    // Descriptor function where the histogram is computed and the blocks are normalized
    xFDHOGDescriptor<WIN_HEIGHT, WIN_WIDTH, WIN_STRIDE, CELL_HEIGHT, CELL_WIDTH, NOB, NOHCPB, NOVCPB, ROWS, COLS,
                     XF_16UP, DEPTH_BLOCK, XF_NPPC1, XF_16UW, WORDWIDTH_BLOCK, USE_URAM>(
        phase_stream, mag_stream, _block_stream, _height, _width);
}

/***********************************************************************
 * 						xFDHOG function
 ***********************************************************************
 *   This function acts as wrapper function for xFDHOGKernel
 *
 *   _in_stream: This stream contains the input image data (I)
 *   _block_stream: This stream contaisn the output descriptor data (O)
 *
 ***********************************************************************/
template <int WIN_HEIGHT,
          int WIN_WIDTH,
          int WIN_STRIDE,
          int BLOCK_HEIGHT,
          int BLOCK_WIDTH,
          int CELL_HEIGHT,
          int CELL_WIDTH,
          int NOB,
          int ROWS,
          int COLS,
          int DEPTH,
          int DEPTH_BLOCK,
          int NPC,
          int WORDWIDTH,
          int WORDWIDTH_BLOCK,
          int NOC,
          bool USE_URAM>
void xFDHOG(hls::stream<XF_SNAME(WORDWIDTH)> _in_stream[NOC],
            hls::stream<XF_SNAME(WORDWIDTH_BLOCK)>& _block_stream,
            uint16_t _height,
            uint16_t _width) {
    //#pragma HLS license key=IPAUVIZ_HOG
    // Updating the _width based on NPC
    _width = _width >> XF_BITSHIFT(NPC);

#ifndef __SYNTHESIS__
    assert(((_height <= ROWS) && (_width <= COLS)) && "ROWS and COLS should be greater than input image");
    assert((NPC == XF_NPPC1) && "The NPC value must be XF_NPPC1");
#endif

    if (NPC == XF_NPPC1) {
        xFDHOGKernel<WIN_HEIGHT, WIN_WIDTH, WIN_STRIDE, CELL_HEIGHT, CELL_WIDTH, NOB, (BLOCK_WIDTH / CELL_WIDTH),
                     (BLOCK_HEIGHT / CELL_HEIGHT), XF_9UW, ROWS, COLS, DEPTH, DEPTH_BLOCK, NPC, WORDWIDTH,
                     WORDWIDTH_BLOCK, NOC, USE_URAM>(_in_stream, _block_stream, _height, _width);
    }
}
#endif // _XF_HOG_DESCRIPTOR_KERNEL_HPP_
