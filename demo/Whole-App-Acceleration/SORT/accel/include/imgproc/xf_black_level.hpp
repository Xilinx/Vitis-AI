#ifndef __XF_BLACK_LEVEL_HPP__
#define __XF_BLACK_LEVEL_HPP__

// =========================================================================
// Required files
// =========================================================================
#include "common/xf_common.hpp"

// =========================================================================
// Actual body
// =========================================================================

template <typename T>
T xf_satcast_bl(int in_val){};

template <>
inline ap_uint<8> xf_satcast_bl<ap_uint<8> >(int v) {
    v = (v > 255 ? 255 : v);
    v = (v < 0 ? 0 : v);
    return v;
};
template <>
inline ap_uint<10> xf_satcast_bl<ap_uint<10> >(int v) {
    v = (v > 1023 ? 1023 : v);
    v = (v < 0 ? 0 : v);
    return v;
};
template <>
inline ap_uint<12> xf_satcast_bl<ap_uint<12> >(int v) {
    v = (v > 4095 ? 4095 : v);
    v = (v < 0 ? 0 : v);
    return v;
};
template <>
inline ap_uint<16> xf_satcast_bl<ap_uint<16> >(int v) {
    v = (v > 65535 ? 65535 : v);
    v = (v < 0 ? 0 : v);
    return v;
};

namespace xf {
namespace cv {

template <int SRC_T,
          int MAX_ROWS,
          int MAX_COLS,
          int NPPC = XF_NPPC1,
          int MUL_VALUE_WIDTH = 16,
          int FL_POS = 15,
          int USE_DSP = 1>
void blackLevelCorrection(xf::cv::Mat<SRC_T, MAX_ROWS, MAX_COLS, NPPC>& _Src,
                          xf::cv::Mat<SRC_T, MAX_ROWS, MAX_COLS, NPPC>& _Dst,
                          XF_CTUNAME(SRC_T, NPPC) black_level,
                          float mul_value // ap_uint<MUL_VALUE_WIDTH> mul_value
                          ) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    // max/(max-black)

    const uint32_t _TC = MAX_ROWS * (MAX_COLS >> XF_BITSHIFT(NPPC));

    const int STEP = XF_DTPIXELDEPTH(SRC_T, NPPC);

    uint32_t LoopCount = _Src.rows * (_Src.cols >> XF_BITSHIFT(NPPC));
    uint32_t rw_ptr = 0, wrptr = 0;

    uint32_t max_value = (1 << (XF_DTPIXELDEPTH(SRC_T, NPPC))) - 1;

    ap_ufixed<16, 1> mulval = (ap_ufixed<16, 1>)mul_value;

    int value = 0;

    for (uint32_t i = 0; i < LoopCount; i++) {
// clang-format off
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=_TC max=_TC
        // clang-format on

        XF_TNAME(SRC_T, NPPC) wr_val = 0;
        XF_TNAME(SRC_T, NPPC) rd_val = _Src.read(rw_ptr++);

        for (uint8_t j = 0; j < NPPC; j++) {
// clang-format off
#pragma HLS UNROLL
            // clang-format on
            XF_CTUNAME(SRC_T, NPPC)
            in_val = rd_val.range(j * STEP + STEP - 1, j * STEP);

            int med_val = (in_val - black_level);

            if (in_val < black_level) {
                value = 0;
            } else {
                value = (int)(med_val * mulval);
            }

            wr_val.range(j * STEP + STEP - 1, j * STEP) = xf_satcast_bl<XF_CTUNAME(SRC_T, NPPC)>(value);
        }

        _Dst.write(wrptr++, wr_val);
    }
}
}
}

#endif // __XF_BLACK_LEVEL_HPP__
