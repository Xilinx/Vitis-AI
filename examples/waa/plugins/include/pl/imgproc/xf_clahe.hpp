/*
 * Copyright 2020 Xilinx, Inc.
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

#ifndef _XF_CLAHE_HPP_
#define _XF_CLAHE_HPP_

#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "algorithm"
#include "hls_math.h"
#include "hls_stream.h"
#include "imgproc/xf_duplicateimage.hpp"
#include <type_traits>

namespace xf {
namespace cv {
namespace clahe {

template <int IN_TYPE,
          int HEIGHT,
          int WIDTH,
          int NPC,
          int CLIPLIMIT,
          int TILES_Y_MAX,
          int TILES_X_MAX,
          int TILES_Y_MIN = 4,
          int TILES_X_MIN = 4>
class CLAHETile {
   public:
    static constexpr int TILE_HEIGHT_MAX =
        (HEIGHT % TILES_Y_MIN == 0) ? HEIGHT / TILES_Y_MIN : (HEIGHT + TILES_Y_MIN) / TILES_Y_MIN;
    static constexpr int TILE_WIDTH_MAX =
        (WIDTH % TILES_X_MIN == 0) ? WIDTH / TILES_X_MIN : (WIDTH + TILES_X_MIN) / TILES_X_MIN;
    static constexpr int CLIP_COUNTER_BITS = xf::cv::log2<TILE_HEIGHT_MAX * TILE_WIDTH_MAX>::cvalue;

    int mRows;
    int mCols;
    int mColsNPCAlligned;
    int mTilesY;
    int mTilesX;
    int mTileHeight;
    int mTileWidth;
    int mTileWidthNPCAlligned;
    ap_uint<CLIP_COUNTER_BITS> mClipValue;

    CLAHETile(int height, int width, int clip, int tilesY, int tilesX)
        : mRows(height), mCols(width), mTilesY(tilesY), mTilesX(tilesX) {
#ifndef __SYNTHESIS__
        assert((XF_CHANNELS(IN_TYPE, NPC) == 1));
        assert(mTilesY >= TILES_Y_MIN);
        assert(mTilesX >= TILES_X_MIN);
        assert(mTilesY <= TILES_Y_MAX);
        assert(mTilesX <= TILES_X_MAX);
        assert(height <= HEIGHT);
        assert(width <= WIDTH);
#endif

        mColsNPCAlligned = mCols >> XF_BITSHIFT(NPC);
        mTileHeight = height / mTilesY;
        mTileWidth = width / mTilesX;

        if ((mTileHeight * mTilesY) < height) mTileHeight++;

        if ((mTileWidth * mTilesX) < width) mTileWidth++;

        // Make tile width exact multiple of NPC
        mTileWidthNPCAlligned = (mTileWidth >> XF_BITSHIFT(NPC)) << XF_BITSHIFT(NPC);
        mClipValue = std::numeric_limits<unsigned int>::max();
        if (clip > 0) {
            mClipValue = std::max(((clip * mTileHeight * mTileWidth) >> XF_DTPIXELDEPTH(IN_TYPE, NPC)), 1);
        }
    }
};

#define _CLAHE_TILE \
    CLAHETile<IN_TYPE, HEIGHT, WIDTH, NPC, CLIPLIMIT, TILES_Y_MAX, TILES_X_MAX, TILES_Y_MIN, TILES_X_MIN>
template <int IN_TYPE,
          int HEIGHT,
          int WIDTH,
          int NPC,
          int CLIPLIMIT,
          int TILES_Y_MAX,
          int TILES_X_MAX,
          int TILES_Y_MIN = 4,
          int TILES_X_MIN = 4>
class CLAHEImpl {
   public:
    static constexpr int COLS_NPC_ALIGNED = (WIDTH + NPC - 1) >> XF_BITSHIFT(NPC);
    static constexpr int SATURATION_CAST = (1 << XF_DTPIXELDEPTH(IN_TYPE, NPC));

    static constexpr int TILE_HEIGHT_MAX = _CLAHE_TILE::TILE_HEIGHT_MAX;
    static constexpr int TILE_WIDTH_MAX = _CLAHE_TILE::TILE_WIDTH_MAX;
    static constexpr int CLIP_COUNTER_BITS = _CLAHE_TILE::CLIP_COUNTER_BITS;

    static constexpr int _MAXCLIPVALUE =
        ((CLIPLIMIT * TILE_HEIGHT_MAX * TILE_WIDTH_MAX) >> XF_DTPIXELDEPTH(IN_TYPE, NPC));
    static constexpr int MAXCLIPVALUE = (_MAXCLIPVALUE > 1) ? _MAXCLIPVALUE : 1;

    static constexpr int HIST_COUNTER_BITS = (XF_DTPIXELDEPTH(IN_TYPE, NPC) > xf::cv::log2<MAXCLIPVALUE>::cvalue)
                                                 ? XF_DTPIXELDEPTH(IN_TYPE, NPC)
                                                 : xf::cv::log2<MAXCLIPVALUE>::cvalue;

    void init(ap_uint<HIST_COUNTER_BITS> _lut[TILES_Y_MAX][TILES_X_MAX][(XF_NPIXPERCYCLE(NPC) << 1)]
                                             [1 << XF_DTPIXELDEPTH(IN_TYPE, NPC)],
              ap_uint<CLIP_COUNTER_BITS> _clipCounter[TILES_Y_MAX][TILES_X_MAX]) {
// clang-format off
    #pragma HLS inline off
    // clang-format on

    INITLY:
        for (int i = 0; i < TILES_Y_MAX; i++) {
        INITLX:
            for (int j = 0; j < TILES_X_MAX; j++) {
            INITLP:
                for (int k = 0; k < (1 << XF_DTPIXELDEPTH(IN_TYPE, NPC)); k++) {
// clang-format off
      #pragma HLS PIPELINE
                    // clang-format on
                    for (int l = 0; l < (XF_NPIXPERCYCLE(NPC) << 1); l++) {
// clang-format off
      #pragma HLS UNROLL
                        // clang-format on
                        _lut[i][j][l][k] = 0;
                    }
                    _clipCounter[i][j] = 0;
                }
            }
        }
    }

    void clipLut(_CLAHE_TILE& Tile,
                 ap_uint<HIST_COUNTER_BITS> _lut[TILES_Y_MAX][TILES_X_MAX][(XF_NPIXPERCYCLE(NPC) << 1)]
                                                [1 << XF_DTPIXELDEPTH(IN_TYPE, NPC)],
                 ap_uint<CLIP_COUNTER_BITS> _clipCounter[TILES_Y_MAX][TILES_X_MAX]) {
// clang-format off
      #pragma HLS inline off
    // clang-format on
    ACY:
        for (int i = 0; i < TILES_Y_MAX; i++) {
        ACX:
            for (int j = 0; j < TILES_X_MAX; j++) {
                ap_uint<CLIP_COUNTER_BITS> add = _clipCounter[i][j];
            ACN:
                for (int k = 0; k < (1 << XF_DTPIXELDEPTH(IN_TYPE, NPC)); k++) {
// clang-format off
      #pragma HLS PIPELINE
                    // clang-format on

                    ap_uint<CLIP_COUNTER_BITS> v = 0;
                    for (int l = 0; l < (XF_NPIXPERCYCLE(NPC) << 1); l++) {
// clang-format off
              #pragma HLS UNROLL
                        // clang-format on
                        v += _lut[i][j][l][k];
                    }
                    if (v > Tile.mClipValue) add += (v - Tile.mClipValue);
                }

                int sum = 0;
            ACK:
                for (int k = 0; k < (1 << XF_DTPIXELDEPTH(IN_TYPE, NPC)); k++) {
// clang-format off
      #pragma HLS PIPELINE
                    // clang-format on

                    ap_uint<CLIP_COUNTER_BITS> div = add >> XF_DTPIXELDEPTH(IN_TYPE, NPC);
                    ap_uint<XF_DTPIXELDEPTH(IN_TYPE, NPC)> rem = add.range(XF_DTPIXELDEPTH(IN_TYPE, NPC) - 1, 0);
                    ap_uint<CLIP_COUNTER_BITS> v = 0;

                    for (int l = 0; l < (XF_NPIXPERCYCLE(NPC) << 1); l++) {
// clang-format off
              #pragma HLS UNROLL
                        // clang-format on
                        v += _lut[i][j][l][k];
                    }

                    if (v > Tile.mClipValue) v = Tile.mClipValue;
                    v += div;

                    if (rem > 0) {
                        int step = (1 << XF_DTPIXELDEPTH(IN_TYPE, NPC)) / rem;
                        if (step < 1) step = 1;

                        if (((k % step) == 0) && (k < (rem * step))) v++;
                    }

                    sum += v;

                    int tileSize = Tile.mTileHeight * Tile.mTileWidth;
                    int fi = ((sum << XF_DTPIXELDEPTH(IN_TYPE, NPC)) - sum) / tileSize;
                    if (fi > (SATURATION_CAST - 1)) fi = (SATURATION_CAST - 1);

                    for (int l = 0; l < (XF_NPIXPERCYCLE(NPC) << 1); l++) {
// clang-format off
              #pragma HLS UNROLL
                        // clang-format on
                        _lut[i][j][l][k] = fi;
                    }
                }
            }
        }
    }

    void populateLutBlk(xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC>& in,
                        _CLAHE_TILE& Tile,
                        xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC>& in_copy,
                        ap_uint<HIST_COUNTER_BITS> _lut[TILES_Y_MAX][TILES_X_MAX][(XF_NPIXPERCYCLE(NPC) << 1)]
                                                       [1 << XF_DTPIXELDEPTH(IN_TYPE, NPC)],
                        ap_uint<CLIP_COUNTER_BITS> _clipCounter[TILES_Y_MAX][TILES_X_MAX]) {
// clang-format off
    #pragma HLS inline off
        // clang-format on
        static constexpr int COL_TRIPCOUNT = COLS_NPC_ALIGNED >> 1;

        int address = 0;
        int counterY = 0;
        int histY = 0;
        ap_uint<XF_DTPIXELDEPTH(IN_TYPE, NPC)> a_prev;
        int accum;
    R:
        for (int i = 0; i < Tile.mRows; i++) {
// clang-format off
      #pragma HLS LOOP_TRIPCOUNT min=1 max=HEIGHT
            // clang-format on
            int counterX = 0;
            int histX = 0;
        C:
            for (int j = 0; j < Tile.mColsNPCAlligned; j += 2) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=COL_TRIPCOUNT
        #pragma HLS LOOP_FLATTEN OFF
        #pragma HLS PIPELINE II=2
                // clang-format on
                XF_TNAME(IN_TYPE, NPC) pxl1;
                XF_TNAME(IN_TYPE, NPC) pxl2;
                pxl1 = in.read(address);
                in_copy.write(address, pxl1);
                if (j < (Tile.mColsNPCAlligned - 1)) {
                    pxl2 = in.read(address + 1);
                    in_copy.write(address + 1, pxl2);
                } else
                    pxl2 = 0;
                address += 2;

                ap_uint<XF_BITSHIFT(NPC) + 1> _clip1 = 0;
                ap_uint<XF_BITSHIFT(NPC) + 1> _clip2 = 0;
            N:
                for (int k = 0; k < XF_NPIXPERCYCLE(NPC); k++) {
// clang-format off
            #pragma HLS UNROLL
                    // clang-format on
                    ap_uint<XF_DTPIXELDEPTH(IN_TYPE, NPC)> a1;
                    ap_uint<XF_DTPIXELDEPTH(IN_TYPE, NPC)> a2;
                    a1 = pxl1.range(((k + 1) * XF_DTPIXELDEPTH(IN_TYPE, NPC)) - 1, (k * XF_DTPIXELDEPTH(IN_TYPE, NPC)));
                    a2 = pxl2.range(((k + 1) * XF_DTPIXELDEPTH(IN_TYPE, NPC)) - 1, (k * XF_DTPIXELDEPTH(IN_TYPE, NPC)));

                    ap_uint<HIST_COUNTER_BITS> val1 = _lut[histY][histX][2 * k][a1] + 1;
                    ap_uint<HIST_COUNTER_BITS> val2 = _lut[histY][histX][2 * k + 1][a2] + 1;

                    if (val1 <= Tile.mClipValue)
                        _lut[histY][histX][2 * k][a1] = val1;
                    else
                        _clip1++;

                    if (val2 <= Tile.mClipValue)
                        _lut[histY][histX][2 * k + 1][a2] = val2;
                    else
                        _clip2++;
                }
                _clipCounter[histY][histX] += (_clip1 + _clip2);

                counterX += (2 << XF_BITSHIFT(NPC));
                if (counterX >= Tile.mTileWidthNPCAlligned) {
                    histX++;
                    counterX = 0;
                }
            }
            counterY++;
            if (counterY == (Tile.mTileHeight)) {
                histY++;
                counterY = 0;
            }
        }
    }

    void populateLut(xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC>& in,
                     _CLAHE_TILE& Tile,
                     xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC>& in_copy,
                     ap_uint<HIST_COUNTER_BITS> _lut[TILES_Y_MAX][TILES_X_MAX][(XF_NPIXPERCYCLE(NPC) << 1)]
                                                    [1 << XF_DTPIXELDEPTH(IN_TYPE, NPC)],
                     ap_uint<CLIP_COUNTER_BITS> _clipCounter[TILES_Y_MAX][TILES_X_MAX]) {
// clang-format off
    #pragma HLS inline off
        // clang-format on
        int address = 0;

        // Initialize LUT
        init(_lut, _clipCounter);

        // Populate LUT
        populateLutBlk(in, Tile, in_copy, _lut, _clipCounter);

        // Accumulate values
        clipLut(Tile, _lut, _clipCounter);
    }

    void interpolate(xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC>& in,
                     _CLAHE_TILE& Tile,
                     ap_uint<HIST_COUNTER_BITS> _lut[TILES_Y_MAX][TILES_X_MAX][(XF_NPIXPERCYCLE(NPC) << 1)]
                                                    [1 << XF_DTPIXELDEPTH(IN_TYPE, NPC)],
                     xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC>& dst) {
        int counterY = Tile.mTileHeight >> 1;
        int histY = -1;
        int counterX;
        int address = 0;

    RI:
        for (int i = 0; i < Tile.mRows; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=HEIGHT
            // clang-format on
            int counterX = (Tile.mTileWidth >> 1);
            int histX = -1;
        CI:
            for (int j = 0; j < Tile.mColsNPCAlligned; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS_NPC_ALIGNED
            #pragma HLS PIPELINE
                // clang-format on
                XF_TNAME(IN_TYPE, NPC) pxl;
                pxl = in.read(address);

                XF_TNAME(IN_TYPE, NPC) pxlout;
            NI:
                for (int k = 0; k < XF_NPIXPERCYCLE(NPC); k++) {
// clang-format off
                #pragma HLS UNROLL
                    // clang-format on
                    ap_uint<XF_DTPIXELDEPTH(IN_TYPE, NPC)> a;
                    a = pxl.range(((k + 1) * XF_DTPIXELDEPTH(IN_TYPE, NPC)) - 1, (k * XF_DTPIXELDEPTH(IN_TYPE, NPC)));

                    int Y1 = std::max(histY, 0);
                    int X1 = std::max(histX, 0);

                    int Y2 = histY + 1;
                    if (Y2 >= Tile.mTilesY) Y2 = (Tile.mTilesY - 1);

                    int X2 = (histX + 1);
                    if (X2 >= Tile.mTilesX) X2 = (Tile.mTilesX - 1);

                    ap_uint<HIST_COUNTER_BITS> a1 = _lut[Y1][X1][2 * k][a];
                    ap_uint<HIST_COUNTER_BITS> a2 = _lut[Y1][X2][2 * k][a];
                    ap_uint<HIST_COUNTER_BITS> b1 = _lut[Y2][X1][2 * k + 1][a];
                    ap_uint<HIST_COUNTER_BITS> b2 = _lut[Y2][X2][2 * k + 1][a];

                    int tileSize = Tile.mTileHeight * Tile.mTileWidth;
                    int xa1 = Tile.mTileWidth - counterX;
                    int xa = counterX;
                    int ya1 = Tile.mTileHeight - counterY;
                    int ya = counterY;

                    ap_uint<HIST_COUNTER_BITS + xf::cv::log2<TILE_WIDTH_MAX>::cvalue> t1 = a1 * xa1 + a2 * xa;
                    ap_uint<HIST_COUNTER_BITS + xf::cv::log2<TILE_WIDTH_MAX>::cvalue> t2 = b1 * xa1 + b2 * xa;
                    ap_uint<HIST_COUNTER_BITS + xf::cv::log2<TILE_WIDTH_MAX>::cvalue +
                            xf::cv::log2<TILE_HEIGHT_MAX>::cvalue>
                        t3 = t1 * ya1 + t2 * ya;
                    ap_uint<HIST_COUNTER_BITS + xf::cv::log2<TILE_WIDTH_MAX>::cvalue +
                            xf::cv::log2<TILE_HEIGHT_MAX>::cvalue>
                        t4 = t3 / tileSize;

                    ap_uint<XF_DTPIXELDEPTH(IN_TYPE, NPC)> d;
                    if (t4 >= SATURATION_CAST)
                        d = (1 << XF_DTPIXELDEPTH(IN_TYPE, NPC)) - 1;
                    else
                        d = t4;

                    pxlout.range(((k + 1) * XF_DTPIXELDEPTH(IN_TYPE, NPC)) - 1, (k * XF_DTPIXELDEPTH(IN_TYPE, NPC))) =
                        d;
                }
                dst.write(address, pxlout);
                address++;

                counterX += (1 << XF_BITSHIFT(NPC));
                if (counterX >= Tile.mTileWidthNPCAlligned) {
                    counterX = 0;
                    histX++;
                }
            }
            counterY++;
            if (counterY == (Tile.mTileHeight)) {
                histY++;
                counterY = 0;
            }
        }
    }

    void process_i(xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC>& dst,
                   xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC>& in,
                   _CLAHE_TILE& Tile,
                   ap_uint<HIST_COUNTER_BITS> _lutw[TILES_Y_MAX][TILES_X_MAX][(XF_NPIXPERCYCLE(NPC) << 1)]
                                                   [1 << XF_DTPIXELDEPTH(IN_TYPE, NPC)],
                   ap_uint<HIST_COUNTER_BITS> _lutr[TILES_Y_MAX][TILES_X_MAX][(XF_NPIXPERCYCLE(NPC) << 1)]
                                                   [1 << XF_DTPIXELDEPTH(IN_TYPE, NPC)],
                   ap_uint<CLIP_COUNTER_BITS> _clipCounter[TILES_Y_MAX][TILES_X_MAX]) {
        xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC> in_copy(in.rows, in.cols);

// clang-format off
    #pragma HLS DATAFLOW
        // clang-format on
        populateLut(in, Tile, in_copy, _lutw, _clipCounter);
        interpolate(in_copy, Tile, _lutr, dst);
    }

    void process(xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC>& dst,
                 xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC>& in,
                 ap_uint<HIST_COUNTER_BITS> _lutw[TILES_Y_MAX][TILES_X_MAX][(XF_NPIXPERCYCLE(NPC) << 1)]
                                                 [1 << XF_DTPIXELDEPTH(IN_TYPE, NPC)],
                 ap_uint<HIST_COUNTER_BITS> _lutr[TILES_Y_MAX][TILES_X_MAX][(XF_NPIXPERCYCLE(NPC) << 1)]
                                                 [1 << XF_DTPIXELDEPTH(IN_TYPE, NPC)],
                 ap_uint<CLIP_COUNTER_BITS> _clipCounter[TILES_Y_MAX][TILES_X_MAX],
                 int height,
                 int width,
                 int clip,
                 int tilesY,
                 int tilesX) {
        _CLAHE_TILE Tile(height, width, clip, tilesY, tilesX);
        process_i(dst, in, Tile, _lutw, _lutr, _clipCounter);
    }
};

} // namespace clahe
} // namespace cv
} // namespace xf

#endif
