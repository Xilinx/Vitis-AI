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

#ifndef __XF_EXTRACT_EXPOSURE_FRAMES_HPP__
#define __XF_EXTRACT_EXPOSURE_FRAMES_HPP__

// =========================================================================
// Required files
// =========================================================================
#include "common/xf_common.hpp"
#include "common/xf_video_mem.hpp"

// =========================================================================
// Actual body
// =========================================================================
namespace xf {
namespace cv {

template <int SRC_T, int N_ROWS, int N_COLS, int MAX_ROWS, int MAX_COLS, int NPPC = XF_NPPC1, int USE_URAM = 0>
class ExposureFramesExtract {
   public:
    // Internal buffers, registers
    xf::cv::LineBuffer<N_ROWS,
                       ((MAX_COLS + N_COLS) >> (XF_BITSHIFT(NPPC))),
                       XF_TNAME(SRC_T, NPPC),
                       (USE_URAM ? RAM_S2P_URAM : RAM_S2P_BRAM),
                       (USE_URAM ? N_ROWS : 1)>
        buff;

    // Read and Write Pointer
    uint32_t fifo_rd_ptr;
    uint32_t fifo_wr_ptr;
    uint32_t src_rd_ptr;
    uint32_t lef_ptr;
    uint32_t sef_ptr;

    // ....................................................................................
    // Pointer initializer
    // ....................................................................................
    void initialize() {
// clang-format off
            #pragma HLS INLINE
        // clang-format on

        // Initialize read and write pointers
        fifo_rd_ptr = 0;
        fifo_wr_ptr = 0;

        src_rd_ptr = 0;

        lef_ptr = 0;
        sef_ptr = 0;

        return;
    }

    // ....................................................................................
    // Default Constructor
    // ....................................................................................
    ExposureFramesExtract() {
// clang-format off
            #pragma HLS INLINE
        // clang-format on

        initialize();
    }

    // ....................................................................................
    // Extraction function:
    //   This function synchronizes input frames of different exposures by storing initial
    //   blank lines into temporary buffer (BRAM or URAM)
    // ....................................................................................
    void extract(xf::cv::Mat<SRC_T, MAX_ROWS * 2, MAX_COLS + N_COLS, NPPC>& _hdrSrc,
                 xf::cv::Mat<SRC_T, MAX_ROWS, MAX_COLS, NPPC>& _lefSrc,
                 xf::cv::Mat<SRC_T, MAX_ROWS, MAX_COLS, NPPC>& _sefSrc) {
// clang-format off
            #pragma HLS INLINE OFF
        // clang-format on

        // Constants for loopcounts
        const uint32_t _TC1 = N_ROWS - 1;
        const uint32_t _TC2 = MAX_ROWS - N_ROWS;

        // Initialize read and write pointers
        initialize();

    // Part-1: Collect initial blank lines of Long Exposure Frames (LEF)
    // -----------------------------------------------------------------
    BUFFER_LINES:
        for (int row = 0; row < N_ROWS - 1; row++) {
            for (int col = 0; col<_hdrSrc.cols>> (XF_BITSHIFT(NPPC)); col++) {
// clang-format off
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=_TC1 max=_TC1
                // clang-format on

                buff.val[fifo_wr_ptr][col] = _hdrSrc.read(src_rd_ptr++);
            }

            /*if (fifo_wr_ptr == N_ROWS - 1)
                fifo_wr_ptr = 0;
            else*/
            fifo_wr_ptr++;
        }

    // Part-2: Writing out LEF and SEF (Short Exposure Frames)
    // -----------------------------------------------------------------
    SPLIT_LINES:
        for (int row = N_ROWS - 1; row < _lefSrc.rows; row++) {
            // LEF (Long Exposure Frame's Line, into internal buffer)
            for (int col = 0; col<_hdrSrc.cols>> (XF_BITSHIFT(NPPC)); col++) {
// clang-format off
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=_TC2
                // clang-format on

                buff.val[fifo_wr_ptr][col] = _hdrSrc.read(src_rd_ptr++);

                //_lefSrc.write(lef_ptr++, buff.val[fifo_rd_ptr][col]);
            }

            // SEF (Short Exposure Frame's Line, directly streamed out)
            for (int col = 0; col<_hdrSrc.cols>> (XF_BITSHIFT(NPPC)); col++) {
// clang-format off
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=_TC2
                // clang-format on

                if (col >= (N_COLS >> (XF_BITSHIFT(NPPC)))) {
                    _lefSrc.write(lef_ptr++, buff.val[fifo_rd_ptr][col]);
                    _sefSrc.write(sef_ptr++, _hdrSrc.read(src_rd_ptr++));
                } else {
                    // XF_TNAME(SRC_T, NPPC) dummy1 = buff.val[fifo_rd_ptr][col];
                    XF_TNAME(SRC_T, NPPC) dummy2 = _hdrSrc.read(src_rd_ptr++);
                }
            }

            // Handle read and write pointers:
            //    Reset them upon reaching end of the FIFO as it is a cyclic buffer
            if (fifo_wr_ptr == N_ROWS - 1)
                fifo_wr_ptr = 0;
            else
                fifo_wr_ptr++;

            if (fifo_rd_ptr == N_ROWS - 1)
                fifo_rd_ptr = 0;
            else
                fifo_rd_ptr++;
        }

    // Part-3: Get last N_ROWS of SEF
    // -----------------------------------------------------------------
    LAST_LINES:
        for (int row = 0; row < N_ROWS - 1; row++) {
            for (int col = 0; col<_hdrSrc.cols>> (XF_BITSHIFT(NPPC)); col++) {
// clang-format off
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=_TC1 max=_TC1
                // clang-format on

                //_lefSrc.write(lef_ptr++, buff.val[fifo_rd_ptr][col]);
                //_sefSrc.write(sef_ptr++, _hdrSrc.read(src_rd_ptr++));
                if (col >= (N_COLS >> (XF_BITSHIFT(NPPC)))) {
                    _lefSrc.write(lef_ptr++, buff.val[fifo_rd_ptr][col]);
                    _sefSrc.write(sef_ptr++, _hdrSrc.read(src_rd_ptr++));
                } else {
                    // XF_TNAME(SRC_T, NPPC) dummy1 = buff.val[fifo_rd_ptr][col];
                    XF_TNAME(SRC_T, NPPC) dummy2 = _hdrSrc.read(src_rd_ptr++);
                }
            }

            if (fifo_rd_ptr == N_ROWS - 1)
                fifo_rd_ptr = 0;
            else
                fifo_rd_ptr++;
        }

        return;
    } // End of extract
};

// Extract HDR exposure frames
template <int SRC_T, int N_ROWS, int N_COLS, int MAX_ROWS, int MAX_COLS, int NPPC = XF_NPPC1, int USE_URAM = 0>
void extractExposureFrames(xf::cv::Mat<SRC_T, MAX_ROWS * 2, MAX_COLS + N_COLS, NPPC>& _hdrSrc,
                           xf::cv::Mat<SRC_T, MAX_ROWS, MAX_COLS, NPPC>& _lefSrc,
                           xf::cv::Mat<SRC_T, MAX_ROWS, MAX_COLS, NPPC>& _sefSrc) {
// clang-format off
        #pragma HLS INLINE OFF
    // clang-format on

    xf::cv::ExposureFramesExtract<SRC_T, N_ROWS, N_COLS, MAX_ROWS, MAX_COLS, NPPC, USE_URAM> extractor;

    extractor.extract(_hdrSrc, _lefSrc, _sefSrc);

    return;
}
}
}

#endif // __XF_EXTRACT_EXPOSURE_FRAMES_HPP__
