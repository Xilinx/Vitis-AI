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

#include "kernel3/dc_shrink.hpp"

void hls_shink_fixed(const int xsize,
                     hls::stream<dct_t>& strm_dc_y,
                     dct_t line3_y[3][MAX_NUM_BLOCK88_W],
                     hls::stream<dct_t>& strm_dc_residuals) {
#pragma HLS INLINE OFF

    dct_t row_dc_left;

    for (int x = 0; x < xsize; ++x) {
#pragma HLS PIPELINE II = 1
        dct_t dc = strm_dc_y.read();
        line3_y[0][x] = dc;

        if (x == 0) {
            strm_dc_residuals.write(dc);
        } else {
            strm_dc_residuals.write(dc - row_dc_left);
        }

        row_dc_left = dc;
    }
}

inline dct_t hls_function(const int x,
                          const dct_t pred, // PixelV
                          const dct_t* row_b,
                          dct_t* residuals

                          ) {
    dct_t blow_pred = row_b[x];
    residuals[x] = blow_pred - pred;
    return blow_pred;
}

inline dct_t hls_saturated_add_16b(const dct_t v0, const dct_t v1) {
    dct_t add;
    ap_fixed<16, 16, AP_RND, AP_SAT_SYM> tmp =
        (ap_fixed<16, 16, AP_RND, AP_SAT_SYM>)v0 + (ap_fixed<16, 16, AP_RND, AP_SAT_SYM>)v1;
    add = tmp;

    return add;
}

inline dct_t hls_Average_16b(const dct_t v0, const dct_t v1) {
    dct_t add = hls_saturated_add_16b(v0, v1);
    dct_t avg = (add >> 1);
    return avg;
}

inline dct_t hls_saturated_subtract_16b(const dct_t v0, const dct_t v1) {
    dct_t subtract;
    ap_fixed<16, 16, AP_RND, AP_SAT_SYM> tmp =
        (ap_fixed<16, 16, AP_RND, AP_SAT_SYM>)v0 - (ap_fixed<16, 16, AP_RND, AP_SAT_SYM>)v1;
    subtract = tmp;
    return subtract;
}

inline dct_t hls_ClampedGradient(const dct_t n, const dct_t w, const dct_t l) {
    const dct_t grad = hls_saturated_subtract_16b(hls_saturated_add_16b(n, w), l); //(-32768, a.raw - b.raw), 32767)
    const dct_t vmin = hls::min(n, hls::min(w, l));
    const dct_t vmax = hls::max(n, hls::max(w, l));
    return hls::min(hls::max(vmin, grad), vmax);
}

inline void hls_Y_Predict(const dct_t n, const dct_t w, const dct_t l, const dct_t r, dct_t pred[hls_kNumPredictors]) {
    // Eight predictors for luminance (decreases coded size by ~0.5% vs four)
    pred[0] = hls_Average_16b(hls_Average_16b(n, w), r);
    pred[1] = hls_Average_16b(w, n);
    pred[2] = hls_Average_16b(n, r);
    pred[3] = hls_Average_16b(w, l);
    pred[4] = hls_Average_16b(n, l);
    pred[5] = w;
    pred[6] = hls_ClampedGradient(n, w, l);
    pred[7] = n;
}

inline dct_t hls_AbsResidual(const dct_t c, const dct_t pred) {
    return hls::abs(hls_saturated_subtract_16b(c, pred));
}

inline void hls_PredictorCosts(
    // input
    const int x,
    const dct_t* row_m,
    const dct_t* row_b,
    const dct_t* row_t,
    // store
    dct_t& tl_,
    dct_t& tn_,
    const dct_t l_,
    const dct_t n_,
    const dct_t w_,
    const dct_t pred_w_[hls_kNumPredictors],
    // output
    dct_t costs[hls_kNumPredictors]) {
    const dct_t tr = row_t[x + 1];
    dct_t pred_n[hls_kNumPredictors];
#pragma HLS ARRAY_PARTITION variable = pred_n complete
    hls_Y_Predict(tn_, l_, tl_, tr, pred_n);

    for (int i = 0; i < hls_kNumPredictors; ++i) {
#pragma HLS UNROLL
        costs[i] = hls_AbsResidual(n_, pred_n[i]) + hls_AbsResidual(w_, pred_w_[i]);
    }

    tl_ = tn_;
    tn_ = tr;
}

// check  timing TODO
inline uint8_t hls_IndexOfMinCost(const dct_t abs_costs[hls_kNumPredictors]) {
    // Algorithm must exactly match minpos_epu16.
    uint8_t idx_pred = 0;
    int16_t min_cost = abs_costs[0];
    for (uint8_t i = 0; i < hls_kNumPredictors; ++i) { // check the timing
#pragma HLS UNROLL
        const int16_t cost = abs_costs[i];
        if (cost < min_cost) {
            min_cost = cost;
            idx_pred = i;
        }
    }
    return idx_pred;
}

inline dct_t hls_Y_PredictC(const dct_t l_,
                            const dct_t n_,
                            const dct_t w_,
                            const dct_t r,
                            const dct_t costs[hls_kNumPredictors],
                            dct_t pred_w_[hls_kNumPredictors]) {
    hls_Y_Predict(n_, w_, l_, r, pred_w_);

    return pred_w_[hls_IndexOfMinCost(costs)];
}

inline void hls_Y_Advance(const dct_t r,
                          const dct_t c,

                          dct_t& l_,
                          dct_t& n_,
                          dct_t& w_) {
    l_ = n_;
    n_ = r;
    w_ = c;
}

void hls_ForeachPrediction(const int xsize,

                           const int t_idx,
                           const int m_idx,
                           const int b_idx,

                           dct_t tl_,
                           dct_t tn_, // row_t[2];
                           dct_t l_,  // row_m[1];
                           dct_t n_,  // row_m[2];
                           dct_t w_,
                           dct_t wl, // row_m[0];
                           dct_t ww, // row_b[0];

                           dct_t line3_y[3][MAX_NUM_BLOCK88_W],
                           hls::stream<dct_t>& strm_dc_y,
                           hls::stream<dct_t>& strm_dc_residuals) {
#pragma HLS INLINE OFF

    dct_t dc;
    dct_t pred_w_[hls_kNumPredictors];
#pragma HLS ARRAY_PARTITION variable = pred_w_ complete
    dct_t costs[hls_kNumPredictors]; // 60 55 ...
#pragma HLS ARRAY_PARTITION variable = costs complete

    hls_Y_Predict(l_, ww, wl, n_, pred_w_);

    if (xsize >= 2) { // Avoid out of bounds reads.

        // PixelNeighborsY uses w at x - 1 => two pixel margin.
        for (int x = 2; x < xsize - 1; ++x) {
#pragma HLS PIPELINE II = 1
            dct_t r = line3_y[m_idx][x + 1]; // row_m[x + 1];

            {
                const dct_t tr = line3_y[t_idx][x + 1]; // row_t[x + 1];
                dct_t pred_n[hls_kNumPredictors];
                hls_Y_Predict(tn_, l_, tl_, tr, pred_n);

                for (int i = 0; i < hls_kNumPredictors; ++i) {
#pragma HLS UNROLL
                    costs[i] = hls_AbsResidual(n_, pred_n[i]) + hls_AbsResidual(w_, pred_w_[i]);
                }

                tl_ = tn_;
                tn_ = tr;
            }

            const dct_t pred_c = hls_Y_PredictC(l_, n_, w_, r, costs, pred_w_);

            {
                dc = strm_dc_y.read();
                _XF_IMAGE_PRINT("%d \n", (int)dc);
                dct_t residuals = dc - pred_c;
                strm_dc_residuals.write(residuals);
                line3_y[b_idx][x] = dc;
            }
            hls_Y_Advance(r, dc, l_, n_, w_);
        }
    }
}

void hls_shink_Y_adaptive(const int xsize,
                          const bool is_by1,
                          const int m_idx,
                          dct_t line3_y[3][MAX_NUM_BLOCK88_W],
                          hls::stream<dct_t>& strm_dc_y,
                          hls::stream<dct_t>& strm_dc_residuals) {
#pragma HLS INLINE OFF

    const int t_idx = is_by1 ? 0 : ((m_idx == 0) ? 2 : (m_idx - 1)); // 0, 0, 1, 2, 0 , 1, 2
    const int b_idx = (m_idx == 2) ? 0 : (m_idx + 1);                // 1, 2, 0 , 1, 2
    dct_t residuals, dc;

    {
        dc = strm_dc_y.read();
        _XF_IMAGE_PRINT("%d \n", (int)dc);
        // output and store line buffer
        residuals = dc - line3_y[m_idx][0]; // row_b[0] - row_m[0];
        strm_dc_residuals.write(residuals);
        line3_y[b_idx][0] = dc;

        dct_t dc_left = dc;
        if (xsize > 2) { // diff with the origin
            dc = strm_dc_y.read();
            _XF_IMAGE_PRINT("%d \n", (int)dc);
            residuals = dc - dc_left; // row_b[0];
            strm_dc_residuals.write(residuals);
            line3_y[b_idx][1] = dc;
        }
    }

    dct_t tl_ = line3_y[t_idx][1];
    dct_t tn_ = line3_y[t_idx][2]; // row_t[2];
    dct_t l_ = line3_y[m_idx][1];  // row_m[1];
    dct_t n_ = line3_y[m_idx][2];  // row_m[2];
    dct_t wl = line3_y[m_idx][0];  // row_m[0];
    dct_t w_ = line3_y[b_idx][1];  // row_b[1];
    dct_t ww = line3_y[b_idx][0];  // row_b[0];

    hls_ForeachPrediction(xsize, t_idx, m_idx, b_idx, tl_, tn_, l_, n_, w_, wl, ww, line3_y, strm_dc_y,
                          strm_dc_residuals);

    {
        if (xsize >= 2) {
            dc = strm_dc_y.read();
            _XF_IMAGE_PRINT("%d \n", (int)dc);
            dct_t row_m_last = line3_y[b_idx][xsize - 2];

            residuals = dc - row_m_last; // row_m[xsize - 2];
            strm_dc_residuals.write(residuals);
            line3_y[b_idx][xsize - 1] = dc;
        }
    }
}

void hls_shink_xb_fixed(const int xsize,
                        hls::stream<dct_t>& strm_dc_y,
                        hls::stream<dct_t>& strm_dc_xb,
                        dct_t line2_y[2][MAX_NUM_BLOCK88_W],
                        dct_t line3_xb[3][MAX_NUM_BLOCK88_W],
                        hls::stream<dct_t>& strm_dc_residuals) {
#pragma HLS INLINE OFF

    dct_t row_dc_left;

    for (int x = 0; x < xsize; ++x) {
#pragma HLS PIPELINE II = 1
        dct_t dc = strm_dc_xb.read();
        dct_t y = strm_dc_y.read();
        line3_xb[0][x] = dc;
        line2_y[0][x] = y;

        if (x == 0) {
            strm_dc_residuals.write(dc);
        } else {
            strm_dc_residuals.write(dc - row_dc_left);
        }
        row_dc_left = dc;
    }
}

inline void hls_XB_Predict(const dct_t n, const dct_t w, const dct_t l, const dct_t r, dct_t pred[hls_kNumPredictors]) {
    ///#if SIMD_TARGET_VALUE == SIMD_NONE
    // Eight predictors for luminance (decreases coded size by ~0.5% vs four)
    pred[0] = hls_ClampedGradient(n, w, l);
    pred[1] = hls_Average_16b(n, w);
    pred[2] = n;
    pred[3] = hls_Average_16b(n, r);
    pred[4] = w;
    pred[5] = hls_Average_16b(w, l);
    pred[6] = r;
    pred[7] = hls_Average_16b(hls_Average_16b(w, r), n);
}

inline void hls_XB_PredictorCosts(
    // input
    const int x,

    // store
    dct_t& yn_,
    dct_t& yw_,
    dct_t& yl_,
    dct_t& n_,
    dct_t& w_,
    dct_t& l_,
    const dct_t yr,
    const dct_t yc,

    const dct_t pred_w_[hls_kNumPredictors],
    // output
    dct_t costs[hls_kNumPredictors]) {
    dct_t pred_y[hls_kNumPredictors];
#pragma HLS ARRAY_PARTITION variable = pred_y complete

    hls_XB_Predict(yn_, yw_, yl_, yr, pred_y);

    for (int i = 0; i < hls_kNumPredictors; ++i) {
#pragma HLS UNROLL
        costs[i] = hls_AbsResidual(yc, pred_y[i]);
    }

    yl_ = yn_;
    yn_ = yr;
    yw_ = yc;
}

inline dct_t hls_XB_PredictC(const dct_t l_,
                             const dct_t n_,
                             const dct_t w_,
                             const dct_t r,
                             const dct_t costs[hls_kNumPredictors],
                             dct_t pred_w_[hls_kNumPredictors]) {
    hls_XB_Predict(n_, w_, l_, r, pred_w_);

    return pred_w_[hls_IndexOfMinCost(costs)];
}

void hls_XB_ForeachPrediction(const int xsize,
                              const bool ym_idx,
                              const int yb_idx,
                              const int t_idx,
                              const int m_idx,
                              const int b_idx,
                              dct_t line2_y[2][MAX_NUM_BLOCK88_W],
                              dct_t line3_xb[3][MAX_NUM_BLOCK88_W],
                              hls::stream<dct_t>& strm_dc_y,
                              hls::stream<dct_t>& strm_dc_xb,
                              hls::stream<dct_t>& strm_dc_residuals) {
#pragma HLS INLINE OFF

    dct_t yn_ = line2_y[ym_idx][2]; // row_ym[2];
    dct_t yw_ = line2_y[yb_idx][1]; // row_yb[1];
    dct_t yl_ = line2_y[ym_idx][1]; // row_ym[1];
    dct_t n_ = line3_xb[m_idx][2];  // row_m[2];
    dct_t w_ = line3_xb[b_idx][1];  // row_b[1];
    dct_t l_ = line3_xb[m_idx][1];  // row_m[1];
    dct_t pred_w_[hls_kNumPredictors];
#pragma HLS ARRAY_PARTITION variable = pred_w_ complete
    dct_t costs[hls_kNumPredictors]; // 60 55 ...
#pragma HLS ARRAY_PARTITION variable = costs complete

    if (xsize >= 2) { // Avoid out of bounds reads.

        // PixelNeighborsY uses w at x - 1 => two pixel margin.
        for (int x = 2; x < xsize - 1; ++x) {
            const dct_t y = strm_dc_y.read();
            const dct_t dc = strm_dc_xb.read(); // row_b[x];

            const dct_t r = line3_xb[m_idx][x + 1];  // row_m[x + 1];//102
            const dct_t yr = line2_y[ym_idx][x + 1]; // row_ym[x + 1];
            const dct_t yc = y;                      // row_yb[x];
            hls_XB_PredictorCosts(x,                 // row_ym, row_yb, row_t,
                                  yn_, yw_, yl_, n_, w_, l_, yr, yc, pred_w_, costs);

            const dct_t pred_c = hls_XB_PredictC(l_, n_, w_, r, costs, pred_w_); // 188

            dct_t residuals = dc - pred_c;

            line2_y[yb_idx][x] = y;
            line3_xb[b_idx][x] = dc;
            strm_dc_residuals.write(residuals);
            hls_Y_Advance(r, dc, l_, n_, w_);
        }
    }
}

void hls_shink_XB_adaptive(const int xsize,
                           const bool is_by1,

                           const bool ym_idx, // 0,1,0,1
                           const int m_idx,   // 0,1,2,0,1,2

                           dct_t line2_y[2][MAX_NUM_BLOCK88_W],
                           dct_t line3_xb[3][MAX_NUM_BLOCK88_W],

                           hls::stream<dct_t>& strm_dc_y,
                           hls::stream<dct_t>& strm_dc_xb,

                           hls::stream<dct_t>& strm_dc_residuals) {
#pragma HLS INLINE OFF

    const int t_idx = is_by1 ? 0 : ((m_idx == 0) ? 2 : (m_idx - 1)); // 0, 0, 1, 2, 0 , 1, 2
    const int b_idx = (m_idx == 2) ? 0 : (m_idx + 1);                // 1, 2, 0 , 1, 2
    const bool yb_idx = !ym_idx;

    dct_t y = strm_dc_y.read();
    dct_t dc = strm_dc_xb.read();

    // output and store line buffer
    dct_t residuals = dc - line3_xb[m_idx][0]; // row_b[0] - row_m[0];

    line2_y[yb_idx][0] = y;
    line3_xb[b_idx][0] = dc;
    strm_dc_residuals.write(residuals);

    dct_t dc_left = dc;
    if (xsize > 2) {
        y = strm_dc_y.read();
        dc = strm_dc_xb.read();
        residuals = dc - dc_left; // row_b[0];

        line2_y[yb_idx][1] = y;
        line3_xb[b_idx][1] = dc;
        strm_dc_residuals.write(residuals);
    }

    hls_XB_ForeachPrediction(xsize, ym_idx, yb_idx, t_idx, m_idx, b_idx, line2_y, line3_xb, strm_dc_y, strm_dc_xb,
                             strm_dc_residuals);

    {
        if (xsize >= 2) {
            y = strm_dc_y.read();
            dc = strm_dc_xb.read();
            dct_t row_m_last = line3_xb[b_idx][xsize - 2];
            residuals = dc - row_m_last; // row_m[xsize - 2];
            line2_y[yb_idx][xsize - 1] = y;
            line3_xb[b_idx][xsize - 1] = dc;
            strm_dc_residuals.write(residuals);
        }
    }
}

void hls_ShrinkXB(const hls_Rect rect_in,
                  hls::stream<dct_t>& strm_dc_xb,
                  hls::stream<dct_t>& strm_dc_y,
                  hls::stream<dct_t>& strm_dc_residuals) {
#pragma HLS INLINE OFF

    const int xsize = rect_in.xsize;
    const int ysize = rect_in.ysize;

    // for init
    dct_t line3_xb[3][MAX_NUM_BLOCK88_W];
#pragma HLS RESOURCE variable = line3_xb core = RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable = line3_xb complete dim = 1

    dct_t line2_y[2][MAX_NUM_BLOCK88_W];
#pragma HLS RESOURCE variable = line2_y core = RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable = line2_y complete dim = 1

    hls_shink_xb_fixed((int)xsize, strm_dc_y, strm_dc_xb, line2_y, line3_xb, strm_dc_residuals);

    _XF_IMAGE_PRINT("\n start XB row 1 \n");

    bool is_by1 = true;
    bool ym_idx = 0;
    int m_idx = 0;
    if (ysize >= 2) {
        // Only one previous row, so row_t == row_m.
        is_by1 = true;
        hls_shink_XB_adaptive(xsize, is_by1, ym_idx, m_idx,

                              line2_y, line3_xb, strm_dc_y, strm_dc_xb, strm_dc_residuals);
    }

    for (int y = 2; y < ysize; ++y) {
        is_by1 = false;
        if (m_idx == 2) {
            m_idx = 0;
        } else {
            m_idx++;
        }
        ym_idx = !ym_idx;
        hls_shink_XB_adaptive(xsize, is_by1, ym_idx, m_idx,

                              line2_y, line3_xb, strm_dc_y, strm_dc_xb, strm_dc_residuals);
    }
}

void hls_ShrinkY(const hls_Rect rect_in, hls::stream<dct_t>& strm_dc_y, hls::stream<dct_t>& strm_dc_residuals) {
#pragma HLS INLINE OFF

    const int xsize = rect_in.xsize;
    const int ysize = rect_in.ysize;

    // for init
    dct_t line3_y[3][MAX_NUM_BLOCK88_W];
#pragma HLS RESOURCE variable = line3_y core = RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable = line3_y complete dim = 1

    hls_shink_fixed((int)xsize, strm_dc_y, line3_y, strm_dc_residuals);

    _XF_IMAGE_PRINT("\n start Y row 0 \n");
    _XF_IMAGE_PRINT("\n start Y row 1 \n");

    bool is_by1 = true;

    int m_idx = 0;
    if (ysize >= 2) {
        // Only one previous row, so row_t == row_m.
        is_by1 = true;
        hls_shink_Y_adaptive((int)xsize, is_by1, m_idx, line3_y, strm_dc_y, strm_dc_residuals);
        _XF_IMAGE_PRINT("\n start row 2 \n");
    }

    for (int y = 2; y < ysize; ++y) {
        is_by1 = false;
        if (m_idx == 2) {
            m_idx = 0;
        } else {
            m_idx++;
        }

        hls_shink_Y_adaptive((int)xsize, is_by1, m_idx, line3_y, strm_dc_y, strm_dc_residuals);

        _XF_IMAGE_PRINT("\n start row %d \n", (int)y + 1);
    }
}

void hls_ShrinkDC_top(const hls_Rect rect_dc,
                      hls::stream<dct_t>& strm_dc_y1,
                      hls::stream<dct_t>& strm_dc_y2,
                      hls::stream<dct_t>& strm_dc_y3,
                      hls::stream<dct_t>& strm_dc_x,
                      hls::stream<dct_t>& strm_dc_b,
                      hls::stream<dct_t>& strm_dc_residuals) {
#pragma HLS INLINE OFF

    hls_ShrinkXB(rect_dc, strm_dc_x, strm_dc_y1, strm_dc_residuals);

    hls_ShrinkY(rect_dc, strm_dc_y2, strm_dc_residuals);

    _XF_IMAGE_PRINT("row_residuals of Y\n");

    hls_ShrinkXB(rect_dc, strm_dc_b, strm_dc_y3, strm_dc_residuals);
}
