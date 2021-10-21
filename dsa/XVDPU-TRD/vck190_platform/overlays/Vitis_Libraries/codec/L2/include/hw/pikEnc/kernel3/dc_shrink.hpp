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

/**
 * @file dc_shrink.hpp
 */

#ifndef _XF_CODEC_DC_SHRINK_HPP_
#define _XF_CODEC_DC_SHRINK_HPP_

#include "kernel3/kernel3_common.hpp"

void hls_shink_Y_adaptive(const int xsize,
                          const bool is_by1,
                          const int m_idx, // 0,1,2,0,1,2
                          dct_t line3_y[3][MAX_NUM_BLOCK88_W],
                          hls::stream<dct_t>& strm_dc_y,
                          hls::stream<dct_t>& strm_dc_residuals);

void hls_shink_fixed(const int xsize,
                     hls::stream<dct_t>& strm_dc_y,
                     dct_t line3_y[3][MAX_NUM_BLOCK88_W],
                     hls::stream<dct_t>& strm_dc_residuals);

void hls_shink_XB_adaptive(const int xsize,
                           const bool is_by1,
                           const bool ym_idx, // 0,1,0,1
                           const int m_idx,   // 0,1,2,0,1,2
                           dct_t line2_y[2][MAX_NUM_BLOCK88_W],
                           dct_t line3_xb[3][MAX_NUM_BLOCK88_W],
                           hls::stream<dct_t>& strm_dc_y,
                           hls::stream<dct_t>& strm_dc_xb,
                           hls::stream<dct_t>& strm_dc_residuals);

void hls_shink_xb_fixed(const int xsize,
                        hls::stream<dct_t>& strm_dc_y,
                        hls::stream<dct_t>& strm_dc_xb,
                        dct_t line2_y[2][MAX_NUM_BLOCK88_W],
                        dct_t line3_xb[3][MAX_NUM_BLOCK88_W],
                        hls::stream<dct_t>& strm_dc_residuals);

void hls_ShrinkDC_top(const hls_Rect rect_dc,
                      hls::stream<dct_t>& strm_dc_y1,
                      hls::stream<dct_t>& strm_dc_y2,
                      hls::stream<dct_t>& strm_dc_y3,
                      hls::stream<dct_t>& strm_dc_x,
                      hls::stream<dct_t>& strm_dc_b,
                      hls::stream<dct_t>& strm_dc_residuals);

#endif
