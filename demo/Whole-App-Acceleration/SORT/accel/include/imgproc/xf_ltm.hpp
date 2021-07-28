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

#ifndef _XF_LTM_HPP_
#define _XF_LTM_HPP_

#include "ap_int.h"
#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_duplicateimage.hpp"
#include "hls_math.h"
#include <type_traits>

namespace xf {
namespace cv {

template <int BLK_ROWS, int BLK_COLS, int ROWS, int COLS, int NPC>
class LTMTile {
   public:
    static constexpr int BLK_COLS_NPC_ALIGNED = BLK_COLS >> XF_BITSHIFT(NPC);
    static constexpr int COLS_NPC_ALIGNED = (COLS + NPC - 1) >> XF_BITSHIFT(NPC);

    static constexpr int MinMaxVArrSize =
        ((ROWS % BLK_ROWS) == 0) ? (ROWS / BLK_ROWS) + 1 : (((ROWS + BLK_ROWS - 1) / BLK_ROWS) + 1) + 1;
    static constexpr int MinMaxHArrSize =
        ((COLS_NPC_ALIGNED % BLK_COLS_NPC_ALIGNED) == 0)
            ? (COLS_NPC_ALIGNED / BLK_COLS_NPC_ALIGNED) + 1
            : (((COLS_NPC_ALIGNED + BLK_COLS_NPC_ALIGNED - 1) / BLK_COLS_NPC_ALIGNED) + 1) + 1;

    int mRows;
    int mCols;
    int mColsNPCAligned;

    int mBlkRows;
    int mBlkCols;
    int mBlkColsNPCAligned;

    int mVBlkCount;
    int mHBlkCount;

    int mVBlkSize[2]; // 0 - First / Last, 1 - Rest
    int mHBlkSize[2]; // 0 - First / Last, 1 - Rest

    // Use this contructor in case BLK_ROWS and BLK_COLS are fixed at compile
    LTMTile(int r, int c) : mRows(r), mCols(c), mBlkRows(BLK_ROWS), mBlkCols(BLK_COLS) {
        assert(BLK_ROWS >= 32);
        assert(BLK_COLS >= 32);
        assert(BLK_COLS > NPC);
        assert((BLK_ROWS & (BLK_ROWS - 1)) == 0);
        assert((BLK_COLS & (BLK_COLS - 1)) == 0);
        assert(mBlkRows < r);
        assert(mBlkCols < c);
        mColsNPCAligned = (mCols + NPC - 1) >> XF_BITSHIFT(NPC);
        mBlkColsNPCAligned = BLK_COLS_NPC_ALIGNED;

        mHBlkCount = ((mColsNPCAligned + (BLK_COLS_NPC_ALIGNED - 1)) >> xf::cv::log2<BLK_COLS_NPC_ALIGNED>::cvalue);
        int th = (mHBlkCount << xf::cv::log2<BLK_COLS_NPC_ALIGNED>::cvalue);
        mHBlkSize[0] = mHBlkSize[1] = BLK_COLS_NPC_ALIGNED;
        if (mColsNPCAligned < th) {
            // In case mColsNPCAligned is not perfectly divisible by HBlk count then add 2 boundary blocks
            int t = mColsNPCAligned - ((mColsNPCAligned >> xf::cv::log2<BLK_COLS_NPC_ALIGNED>::cvalue)
                                       << xf::cv::log2<BLK_COLS_NPC_ALIGNED>::cvalue);
            mHBlkSize[0] = t >> 1;
            mHBlkCount++;
        }

        mVBlkCount = ((mRows + (BLK_ROWS - 1)) >> xf::cv::log2<BLK_ROWS>::cvalue);
        int tv = (mVBlkCount << xf::cv::log2<BLK_ROWS>::cvalue);
        mVBlkSize[0] = mVBlkSize[1] = BLK_ROWS;
        if (mRows < tv) {
            // In case mRows is not perfectly divisible by VBlk count then add 2 boundary blocks
            int t = mRows - ((mRows >> xf::cv::log2<BLK_ROWS>::cvalue) << xf::cv::log2<BLK_ROWS>::cvalue);
            mVBlkSize[0] = t >> 1;
            mVBlkCount++;
        }
    }

    // Use this contructor in case BLK_ROWS and BLK_COLS can change at runtime
    // Note : BLK_ROWS / BLK_COLS should be smallest possible blk values
    LTMTile(int r, int c, int blk_r, int blk_c) : mRows(r), mCols(c), mBlkRows(blk_r), mBlkCols(blk_c) {
        assert(BLK_ROWS >= 32);
        assert(BLK_COLS >= 32);
        assert(BLK_COLS > NPC);
        assert((BLK_ROWS & (BLK_ROWS - 1)) == 0);
        assert((BLK_COLS & (BLK_COLS - 1)) == 0);

        assert(mBlkRows >= BLK_ROWS);
        assert(mBlkCols >= BLK_COLS);
        assert((mBlkRows & (mBlkRows - 1)) == 0);
        assert((mBlkCols & (mBlkCols - 1)) == 0);

        assert(mBlkRows < r);
        assert(mBlkCols < c);

        mColsNPCAligned = (mCols + NPC - 1) >> XF_BITSHIFT(NPC);
        int blkColsLog2;
        for (blkColsLog2 = xf::cv::log2<BLK_COLS>::cvalue; blkColsLog2 <= xf::cv::log2<COLS>::cvalue; blkColsLog2++) {
// clang-format off
#pragma HLS PIPELINE
            // clang-format on
            if ((1 << blkColsLog2) >= mBlkCols) break;
        }
        mBlkColsNPCAligned = 1 << (blkColsLog2 - XF_BITSHIFT(NPC));

        mHBlkCount = ((mColsNPCAligned + (mBlkColsNPCAligned - 1)) >> blkColsLog2);
        int th = (mHBlkCount << blkColsLog2);
        mHBlkSize[0] = mHBlkSize[1] = mBlkColsNPCAligned;
        if (mColsNPCAligned < th) {
            // In case mColsNPCAligned is not perfectly divisible by HBlk count then add 2 boundary blocks
            int t = mColsNPCAligned - ((mColsNPCAligned >> blkColsLog2) << blkColsLog2);
            mHBlkSize[0] = t >> 1;
            mHBlkCount++;
        }

        int blkRowsLog2;
        for (blkRowsLog2 = xf::cv::log2<BLK_ROWS>::cvalue; blkRowsLog2 <= xf::cv::log2<ROWS>::cvalue; blkRowsLog2++) {
// clang-format off
#pragma HLS PIPELINE
            // clang-format on
            if ((1 << blkRowsLog2) >= mBlkRows) break;
        }

        mVBlkCount = ((mRows + (mBlkRows - 1)) >> blkRowsLog2);
        int tv = (mVBlkCount << blkRowsLog2);
        mVBlkSize[0] = mVBlkSize[1] = mBlkRows;
        if (mRows < tv) {
            // In case mRows is not perfectly divisible by VBlk count then add 2 boundary blocks
            int t = mRows - ((mRows >> blkRowsLog2) << blkRowsLog2);
            mVBlkSize[0] = t >> 1;
            mVBlkCount++;
        }
    }

    int getVBlkSize(int index) {
        if (index == 0)
            return mVBlkSize[0];
        else if (index == (mVBlkCount - 1))
            return mVBlkSize[0];
        else
            return mVBlkSize[1];
    }

    // Special case function to calculate blk size for Min Max computation
    int getMinMaxVBlkSize(int index) {
        if (index == 0) return (mVBlkSize[0] >> 1);
        if (index == mVBlkCount) return (mVBlkSize[0] >> 1);
        if (index == 1) return (mVBlkSize[0] + mVBlkSize[1]) >> 1;
        if (index == (mVBlkCount - 1)) return (mVBlkSize[0] + mVBlkSize[1]) >> 1;
        if (index < (mVBlkCount - 1)) return mVBlkSize[1];
        return 0;
    }

    // Here addr is actual column number
    int getHBlkSize(int addr) {
        if (addr < mHBlkSize[0]) return mHBlkSize[0];
        if (addr >= (mCols - mHBlkSize[0])) return mHBlkSize[0];
        return mHBlkSize[1];
    }

    inline int getInputRows() { return mRows; }
    inline int getInputCols() { return mCols; }
    inline int getInputColsAlignedNPC() { return mColsNPCAligned; }

    inline int getVBlkCount() { return mVBlkCount; }
    inline int getHBlkCount() { return mHBlkCount; }
};

template <int T>
class is_floating_point {
   public:
    static constexpr bool value = false;
};

template <>
class is_floating_point<XF_32FC1> {
   public:
    static constexpr bool value = true;
};

template <>
class is_floating_point<XF_32FC3> {
   public:
    static constexpr bool value = true;
};

template <int IN_TYPE, int OUT_TYPE, int BLK_ROWS, int BLK_COLS, int ROWS, int COLS, int NPC>
class LTM {
   public:
    static constexpr int BILINEAR_INTERPOLATE_TYPE = XF_32FC1;
    static constexpr float OUT_DEPTH = (1 << XF_DTPIXELDEPTH(OUT_TYPE, NPC)) - 1;
    static constexpr int COLS_NPC_ALIGNED = LTMTile<BLK_ROWS, BLK_COLS, ROWS, COLS, NPC>::COLS_NPC_ALIGNED;
    static constexpr int MinMaxVArrSize = LTMTile<BLK_ROWS, BLK_COLS, ROWS, COLS, NPC>::MinMaxVArrSize;
    static constexpr int MinMaxHArrSize = LTMTile<BLK_ROWS, BLK_COLS, ROWS, COLS, NPC>::MinMaxHArrSize;

    LTM() { assert(!is_floating_point<OUT_TYPE>::value); }

    // Limit implementation SFINAE principal [[
    template <int T = IN_TYPE, typename std::enable_if<!is_floating_point<T>::value>::type* = nullptr>
    static constexpr XF_CTUNAME(IN_TYPE, NPC) LOW() {
        return 0;
    }

    template <int T = IN_TYPE, typename std::enable_if<is_floating_point<T>::value>::type* = nullptr>
    static constexpr XF_CTUNAME(IN_TYPE, NPC) LOW() {
        return floatToRawBits(0.0);
    }

    template <int T = IN_TYPE, typename std::enable_if<!is_floating_point<T>::value>::type* = nullptr>
    static constexpr XF_CTUNAME(IN_TYPE, NPC) HIGH() {
        return std::numeric_limits<unsigned int>::max();
    }

    template <int T = IN_TYPE, typename std::enable_if<is_floating_point<T>::value>::type* = nullptr>
    static constexpr XF_CTUNAME(IN_TYPE, NPC) HIGH() {
        return floatToRawBits(std::numeric_limits<float>::max());
    }
    //]]

    // Overload float conversion using SFINAE principal [[
    template <int T = IN_TYPE, typename std::enable_if<!is_floating_point<T>::value>::type* = nullptr>
    static float to_float(XF_CTUNAME(T, NPC) & a) {
        float f = a;
        return f;
    }

    template <int T = IN_TYPE, typename std::enable_if<is_floating_point<T>::value>::type* = nullptr>
    static float to_float(XF_CTUNAME(T, NPC) & a) {
        float f = rawBitsToFloat((unsigned long)a);
        return f;
    }
    //]]

    // Overload is_less using SFINAE principal [[
    template <int T = IN_TYPE, typename std::enable_if<!is_floating_point<T>::value>::type* = nullptr>
    static inline bool is_less(XF_CTUNAME(T, NPC) a, XF_CTUNAME(T, NPC) b) {
        return (a < b) ? true : false;
    }

    template <int T = IN_TYPE, typename std::enable_if<is_floating_point<T>::value>::type* = nullptr>
    static inline bool is_less(XF_CTUNAME(T, NPC) a, XF_CTUNAME(T, NPC) b) {
        return (rawBitsToFloat((unsigned long)a) < rawBitsToFloat((unsigned long)b)) ? true : false;
    }
    //]]

    static XF_CTUNAME(IN_TYPE, NPC) max(XF_DTUNAME(IN_TYPE, NPC) in) {
        XF_CTUNAME(IN_TYPE, NPC) ret = LOW();
    CH0:
        for (int i = 0; i < XF_CHANNELS(IN_TYPE, NPC); i++) {
            XF_CTUNAME(IN_TYPE, NPC)
            val = in.range((i + 1) * XF_DTPIXELDEPTH(IN_TYPE, NPC) - 1, i * XF_DTPIXELDEPTH(IN_TYPE, NPC));
            if (is_less(ret, val)) ret = val;
        }
        return ret;
    }

    static void getMaxImage(xf::cv::Mat<IN_TYPE, ROWS, COLS, NPC>& in,
                            LTMTile<BLK_ROWS, BLK_COLS, ROWS, COLS, NPC>& Tile,
                            xf::cv::Mat<IN_TYPE, ROWS, COLS, NPC>& in_copy,
                            hls::stream<ap_uint<XF_DTPIXELDEPTH(IN_TYPE, NPC) * XF_NPIXPERCYCLE(NPC)> >& in_max) {
        int addr = 0;
    R:
        for (int i = 0; i < Tile.getInputRows(); i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        // clang-format on
        C:
            for (int j = 0; j < Tile.getInputColsAlignedNPC(); j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=COLS_NPC_ALIGNED max=COLS_NPC_ALIGNED
#pragma HLS PIPELINE
                // clang-format on
                XF_TNAME(IN_TYPE, NPC) val = in.read(addr);
                in_copy.write(addr, val);
                addr++;
                ap_uint<XF_DTPIXELDEPTH(IN_TYPE, NPC) * XF_NPIXPERCYCLE(NPC)> valMax;
            N:
                for (int l = 0; l < XF_NPIXPERCYCLE(NPC); l++) {
                    XF_DTUNAME(IN_TYPE, NPC)
                    pinVal = val.range((l + 1) * XF_PIXELWIDTH(IN_TYPE, NPC) - 1, l * XF_PIXELWIDTH(IN_TYPE, NPC));
                    valMax.range((l + 1) * XF_DTPIXELDEPTH(IN_TYPE, NPC) - 1, l * XF_DTPIXELDEPTH(IN_TYPE, NPC)) =
                        max(pinVal);
                }
                in_max.write(valMax);
            }
        }
    }

    static void getMinMax(hls::stream<ap_uint<XF_DTPIXELDEPTH(IN_TYPE, NPC) * XF_NPIXPERCYCLE(NPC)> >& in_max,
                          int vBlkSize,
                          int vBlkIndex,
                          LTMTile<BLK_ROWS, BLK_COLS, ROWS, COLS, NPC>& Tile,
                          XF_CTUNAME(IN_TYPE, NPC) omin[MinMaxVArrSize][MinMaxHArrSize],
                          XF_CTUNAME(IN_TYPE, NPC) omax[MinMaxVArrSize][MinMaxHArrSize]) {
        XF_CTUNAME(IN_TYPE, NPC) omin_l[MinMaxHArrSize];
        XF_CTUNAME(IN_TYPE, NPC) omax_l[MinMaxHArrSize];
        bool bStart = true;
// clang-format off
#pragma HLS ARRAY_PARTITION variable=omin_l dim=1 complete
#pragma HLS ARRAY_PARTITION variable=omax_l dim=1 complete
    // clang-format on
    R0:
        for (int i = 0; i < vBlkSize; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=BLK_ROWS max=BLK_ROWS
        // clang-format on
        C0:
            for (int j = 0, blk = 0, offset = 0; j < Tile.getInputColsAlignedNPC(); j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=COLS_NPC_ALIGNED max=COLS_NPC_ALIGNED
#pragma HLS PIPELINE II=1
                // clang-format on
                ap_uint<XF_DTPIXELDEPTH(IN_TYPE, NPC) * XF_NPIXPERCYCLE(NPC)> val = in_max.read();
                int hBlkSize = Tile.getHBlkSize(j);

                XF_CTUNAME(IN_TYPE, NPC) pinValMin = HIGH();
                XF_CTUNAME(IN_TYPE, NPC) pinValMax = LOW();
            N0:
                for (int l = 0; l < XF_NPIXPERCYCLE(NPC); l++) {
                    XF_CTUNAME(IN_TYPE, NPC)
                    pinVal = val.range((l + 1) * XF_DTPIXELDEPTH(IN_TYPE, NPC) - 1, l * XF_DTPIXELDEPTH(IN_TYPE, NPC));

                    // Update min
                    if (is_less(pinVal, pinValMin)) pinValMin = pinVal;

                    // Update max
                    if (is_less(pinValMax, pinVal)) pinValMax = pinVal;
                }

                XF_CTUNAME(IN_TYPE, NPC) min = omin_l[blk];
                XF_CTUNAME(IN_TYPE, NPC) max = omax_l[blk];
                if (bStart) {
                    min = pinValMin;
                    max = pinValMin;
                } else {
                    min = (is_less(pinValMin, min)) ? pinValMin : min;
                    max = (is_less(max, pinValMax)) ? pinValMax : max;
                }

                offset++;
                bStart = false;
                if (offset == (hBlkSize >> 1)) {
                    blk++;
                    bStart = (i == 0);
                }
                if (offset == hBlkSize) offset = 0;

                omin_l[blk] = min;
                omax_l[blk] = max;

                omin[vBlkIndex][blk] = min;
                omax[vBlkIndex][blk] = max;
            }
        }
    }

    static void processMinMax(hls::stream<ap_uint<XF_DTPIXELDEPTH(IN_TYPE, NPC) * XF_NPIXPERCYCLE(NPC)> >& in_max,
                              LTMTile<BLK_ROWS, BLK_COLS, ROWS, COLS, NPC>& Tile,
                              XF_CTUNAME(IN_TYPE, NPC) omin[MinMaxVArrSize][MinMaxHArrSize],
                              XF_CTUNAME(IN_TYPE, NPC) omax[MinMaxVArrSize][MinMaxHArrSize]) {
    R1:
        for (int i = 0; i <= Tile.getVBlkCount(); i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=MinMaxVArrSize max=MinMaxVArrSize
            // clang-format on
            getMinMax(in_max, Tile.getMinMaxVBlkSize(i), i, Tile, omin, omax);
        }
    }

    template <int I_T,
              int O_T = BILINEAR_INTERPOLATE_TYPE,
              typename std::enable_if<!is_floating_point<O_T>::value>::type* = nullptr>
    static XF_CTUNAME(O_T, NPC)
        bilinear(XF_CTUNAME(I_T, NPC) & in1, XF_CTUNAME(I_T, NPC) & in2, float blkSize, float offset) {
        assert(is_floating_point<I_T>::value ==
               false); // If bilinear interpolation is not floating then input type cant be floating as well
        XF_CTUNAME(O_T, NPC) op1 = in1;
        XF_CTUNAME(O_T, NPC) op2 = (in2 - in1);
        XF_CTUNAME(O_T, NPC) op3 = offset / blkSize;
        XF_CTUNAME(O_T, NPC) op4 = op3 * op2;
        XF_CTUNAME(O_T, NPC) ret = op1 + op4;
        return ret;
    }

    template <int I_T,
              int O_T = BILINEAR_INTERPOLATE_TYPE,
              typename std::enable_if<is_floating_point<O_T>::value>::type* = nullptr>
    static XF_CTUNAME(O_T, NPC)
        bilinear(XF_CTUNAME(I_T, NPC) & in1, XF_CTUNAME(I_T, NPC) & in2, float blkSize, float offset) {
        float op1 = to_float<I_T>(in1);
        float op2 = (to_float<I_T>(in2) - to_float<I_T>(in1));
        float op3 = offset / blkSize;
        float op4 = op3 * op2;
        XF_CTUNAME(O_T, NPC) ret = floatToRawBits(op1 + op4);
        return ret;
    }

    static void interpolate(XF_CTUNAME(IN_TYPE, NPC) arr[MinMaxVArrSize][MinMaxHArrSize],
                            int vBlkSize,
                            int vBlkIndex,
                            LTMTile<BLK_ROWS, BLK_COLS, ROWS, COLS, NPC>& Tile,
                            hls::stream<XF_TNAME(BILINEAR_INTERPOLATE_TYPE, NPC)>& interpolateOut) {
    R2:
        for (int i = 0; i < vBlkSize; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=BLK_ROWS max=BLK_ROWS
        // clang-format on
        C2:
            for (int j = 0, blk = 0, offset = 0; j < Tile.getInputColsAlignedNPC(); j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=COLS_NPC_ALIGNED max=COLS_NPC_ALIGNED
#pragma HLS PIPELINE II=1
                // clang-format on
                XF_TNAME(BILINEAR_INTERPOLATE_TYPE, NPC) out;
                int hBlkSize = Tile.getHBlkSize(j);
                XF_CTUNAME(IN_TYPE, NPC) a[4];
                for (int m = 0; m < 2; m++) {
                    for (int n = 0; n < 2; n++) {
                        a[(m << 1) + n] = arr[vBlkIndex + m][blk + n];
                    }
                }

            N2:
                for (int l = 0; l < XF_NPIXPERCYCLE(NPC); l++) {
                    XF_CTUNAME(BILINEAR_INTERPOLATE_TYPE, NPC) x1 = bilinear<IN_TYPE>(a[0], a[1], hBlkSize, offset);
                    XF_CTUNAME(BILINEAR_INTERPOLATE_TYPE, NPC) x2 = bilinear<IN_TYPE>(a[2], a[3], hBlkSize, offset);
                    XF_CTUNAME(BILINEAR_INTERPOLATE_TYPE, NPC)
                    y = bilinear<BILINEAR_INTERPOLATE_TYPE>(x1, x2, vBlkSize, i);
                    out.range((l + 1) * XF_PIXELWIDTH(BILINEAR_INTERPOLATE_TYPE, NPC) - 1,
                              l * XF_PIXELWIDTH(BILINEAR_INTERPOLATE_TYPE, NPC)) = y;
                }

                offset++;
                if (offset == hBlkSize) {
                    offset = 0;
                    blk++;
                }
                interpolateOut.write(out);
            }
        }
    }

    static void processInterpolate(XF_CTUNAME(IN_TYPE, NPC) arr[MinMaxVArrSize][MinMaxHArrSize],
                                   LTMTile<BLK_ROWS, BLK_COLS, ROWS, COLS, NPC>& Tile,
                                   hls::stream<XF_TNAME(BILINEAR_INTERPOLATE_TYPE, NPC)>& interpolateOut) {
    R3:
        for (int i = 0; i < Tile.getVBlkCount(); i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=MinMaxVArrSize max=MinMaxVArrSize
            // clang-format on
            interpolate(arr, Tile.getVBlkSize(i), i, Tile, interpolateOut);
        }
    }

    static inline float compute(float val, float min, float max) {
        // Add epsillon to avoid floating point compute errors
        float op1 = std::log(val + std::numeric_limits<float>::epsilon());
        float op2 = std::log(min + std::numeric_limits<float>::epsilon());
        float op3 = std::log(max + std::numeric_limits<float>::epsilon());
        float ret = (op1 - op2) / ((op3 - op2) + std::numeric_limits<float>::epsilon());
        return ret;
    }

    static void compute(xf::cv::Mat<IN_TYPE, ROWS, COLS, NPC>& in,
                        LTMTile<BLK_ROWS, BLK_COLS, ROWS, COLS, NPC>& Tile,
                        hls::stream<XF_TNAME(BILINEAR_INTERPOLATE_TYPE, NPC)>& interpolateMin,
                        hls::stream<XF_TNAME(BILINEAR_INTERPOLATE_TYPE, NPC)>& interpolateMax,
                        xf::cv::Mat<OUT_TYPE, ROWS, COLS, NPC>& out) {
        int addr = 0;
    R4:
        for (int i = 0; i < Tile.getInputRows(); i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        // clang-format on
        C4:
            for (int j = 0; j < Tile.getInputColsAlignedNPC(); j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=COLS_NPC_ALIGNED max=COLS_NPC_ALIGNED
#pragma HLS PIPELINE II=1
                // clang-format on
                XF_TNAME(IN_TYPE, NPC) val = in.read(addr);
                XF_TNAME(BILINEAR_INTERPOLATE_TYPE, NPC) interpolateMinVal = interpolateMin.read();
                XF_TNAME(BILINEAR_INTERPOLATE_TYPE, NPC) interpolateMaxVal = interpolateMax.read();
                XF_TNAME(OUT_TYPE, NPC) valF;
            N4:
                for (int l = 0; l < XF_NPIXPERCYCLE(NPC); l++) {
                    XF_DTUNAME(IN_TYPE, NPC)
                    vali = val.range((l + 1) * XF_PIXELWIDTH(IN_TYPE, NPC) - 1, l * XF_PIXELWIDTH(IN_TYPE, NPC));
                    // Since interpolated value is single channel XF_DTUNAME is sane as XF_CTUNAME
                    XF_CTUNAME(IN_TYPE, NPC) maxVal = max(vali);
                    XF_DTUNAME(BILINEAR_INTERPOLATE_TYPE, NPC)
                    iMin = interpolateMinVal.range((l + 1) * XF_PIXELWIDTH(BILINEAR_INTERPOLATE_TYPE, NPC) - 1,
                                                   l * XF_PIXELWIDTH(BILINEAR_INTERPOLATE_TYPE, NPC));
                    XF_DTUNAME(BILINEAR_INTERPOLATE_TYPE, NPC)
                    iMax = interpolateMaxVal.range((l + 1) * XF_PIXELWIDTH(BILINEAR_INTERPOLATE_TYPE, NPC) - 1,
                                                   l * XF_PIXELWIDTH(BILINEAR_INTERPOLATE_TYPE, NPC));

                    // If difference between Max and Min is very small then it is a saturation point
                    float computeVal = compute(to_float<IN_TYPE>(maxVal), to_float<BILINEAR_INTERPOLATE_TYPE>(iMin),
                                               to_float<BILINEAR_INTERPOLATE_TYPE>(iMax));
                    float ratio = computeVal / to_float<IN_TYPE>(maxVal);

                    XF_DTUNAME(OUT_TYPE, NPC) valO;
                CH4:
                    for (int m = 0; m < XF_CHANNELS(OUT_TYPE, NPC); m++) {
                        XF_CTUNAME(IN_TYPE, NPC)
                        valiC =
                            vali.range((m + 1) * XF_DTPIXELDEPTH(IN_TYPE, NPC) - 1, m * XF_DTPIXELDEPTH(IN_TYPE, NPC));
                        float finalValue = (to_float<IN_TYPE>(valiC) * ratio * OUT_DEPTH);

                        XF_CTUNAME(OUT_TYPE, NPC) valiCOut;
                        if (finalValue < 0)
                            valiCOut = 0;
                        else if (finalValue > 255)
                            valiCOut = 255;
                        else
                            valiCOut = finalValue;

                        valO.range((m + 1) * XF_DTPIXELDEPTH(OUT_TYPE, NPC) - 1, m * XF_DTPIXELDEPTH(OUT_TYPE, NPC)) =
                            valiCOut;
                    }
                    valF.range((l + 1) * XF_PIXELWIDTH(OUT_TYPE, NPC) - 1, l * XF_PIXELWIDTH(OUT_TYPE, NPC)) = valO;
                }
                out.write(addr, valF);
                addr++;
            }
        }
    }

    static void process_i(xf::cv::Mat<IN_TYPE, ROWS, COLS, NPC>& in,
                          LTMTile<BLK_ROWS, BLK_COLS, ROWS, COLS, NPC>& Tile,
                          XF_CTUNAME(IN_TYPE, NPC) omin_r[MinMaxVArrSize][MinMaxHArrSize],
                          XF_CTUNAME(IN_TYPE, NPC) omax_r[MinMaxVArrSize][MinMaxHArrSize],
                          XF_CTUNAME(IN_TYPE, NPC) omin_w[MinMaxVArrSize][MinMaxHArrSize],
                          XF_CTUNAME(IN_TYPE, NPC) omax_w[MinMaxVArrSize][MinMaxHArrSize],
                          xf::cv::Mat<OUT_TYPE, ROWS, COLS, NPC>& out) {
        xf::cv::Mat<IN_TYPE, ROWS, COLS, NPC> in_copy(in.rows, in.cols);
        hls::stream<ap_uint<XF_DTPIXELDEPTH(IN_TYPE, NPC) * XF_NPIXPERCYCLE(NPC)> > in_max;
        hls::stream<XF_TNAME(BILINEAR_INTERPOLATE_TYPE, NPC)> interpolateMin;
        hls::stream<XF_TNAME(BILINEAR_INTERPOLATE_TYPE, NPC)> interpolateMax;

// clang-format off
#pragma HLS DATAFLOW
        // clang-format on
        getMaxImage(in, Tile, in_copy, in_max);
        processMinMax(in_max, Tile, omin_w, omax_w);
        processInterpolate(omin_r, Tile, interpolateMin);
        processInterpolate(omax_r, Tile, interpolateMax);
        compute(in_copy, Tile, interpolateMin, interpolateMax, out);
    }

    static void process(xf::cv::Mat<IN_TYPE, ROWS, COLS, NPC>& in,
                        XF_CTUNAME(IN_TYPE, NPC) omin_r[MinMaxVArrSize][MinMaxHArrSize],
                        XF_CTUNAME(IN_TYPE, NPC) omax_r[MinMaxVArrSize][MinMaxHArrSize],
                        XF_CTUNAME(IN_TYPE, NPC) omin_w[MinMaxVArrSize][MinMaxHArrSize],
                        XF_CTUNAME(IN_TYPE, NPC) omax_w[MinMaxVArrSize][MinMaxHArrSize],
                        xf::cv::Mat<OUT_TYPE, ROWS, COLS, NPC>& out) {
        LTMTile<BLK_ROWS, BLK_COLS, ROWS, COLS, NPC> Tile(in.rows, in.cols);
// clang-format off
#pragma HLS ARRAY_PARTITION variable=Tile.mVBlkSize dim=1 complete
#pragma HLS ARRAY_PARTITION variable=Tile.mHBlkSize dim=1 complete
        // clang-format on
        process_i(in, Tile, omin_r, omax_r, omin_w, omax_w, out);
    }

    static void process(xf::cv::Mat<IN_TYPE, ROWS, COLS, NPC>& in,
                        int block_rows,
                        int block_cols,
                        XF_CTUNAME(IN_TYPE, NPC) omin_r[MinMaxVArrSize][MinMaxHArrSize],
                        XF_CTUNAME(IN_TYPE, NPC) omax_r[MinMaxVArrSize][MinMaxHArrSize],
                        XF_CTUNAME(IN_TYPE, NPC) omin_w[MinMaxVArrSize][MinMaxHArrSize],
                        XF_CTUNAME(IN_TYPE, NPC) omax_w[MinMaxVArrSize][MinMaxHArrSize],
                        xf::cv::Mat<OUT_TYPE, ROWS, COLS, NPC>& out) {
        LTMTile<BLK_ROWS, BLK_COLS, ROWS, COLS, NPC> Tile(in.rows, in.cols, block_rows, block_cols);
// clang-format off
#pragma HLS ARRAY_PARTITION variable=Tile.mVBlkSize dim=1 complete
#pragma HLS ARRAY_PARTITION variable=Tile.mHBlkSize dim=1 complete
        // clang-format on
        process_i(in, Tile, omin_r, omax_r, omin_w, omax_w, out);
    }
};

} // namespace cv
} // namespace xf

#endif
