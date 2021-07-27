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

#ifndef _XF_CVT_COLOR_1_HPP_
#define _XF_CVT_COLOR_1_HPP_

#ifndef __cplusplus
#error C++ is needed to compile this header !
#endif

#ifndef _XF_CVT_COLOR_HPP_
#error This file can not be included independently !
#endif

#include "xf_cvt_color_utils.hpp"

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC, int WORDWIDTH_SRC, int TC, int TCC>
void write_y(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src_y,
             xf::cv::Mat<DST_T, ROWS, COLS, NPC>& out_y,
             uint16_t height,
             uint16_t width) {
    XF_SNAME(WORDWIDTH_SRC) tmp;
    unsigned long long int idx = 0;
    for (int i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        #pragma HLS LOOP_FLATTEN off
        // clang-format on
        for (int j = 0; j < width; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            // clang-format on
            tmp = src_y.read(i * width + j);
            out_y.write(idx++, tmp);
        }
    }
}
template <int SRC_T, int UV_T, int ROWS, int COLS, int NPC, int NPC_UV, int WORDWIDTH_UV, int WORDWIDTH_DST, int TC>
void KernNv122Yuv4(xf::cv::Mat<UV_T, ROWS / 2, COLS / 2, NPC_UV>& _uv,
                   xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _u,
                   xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _v,
                   uint16_t height,
                   uint16_t width) {
    XF_PTNAME(XF_16UP) uv;
    XF_SNAME(WORDWIDTH_DST) u, v;
    XF_SNAME(WORDWIDTH_UV) uvPacked;
    XF_TNAME(SRC_T, NPC) arr_u[COLS];
    XF_TNAME(SRC_T, NPC) arr_v[COLS];

    unsigned long long int idx = 0, idx1 = 0;
    ap_uint<13> i, j;
    bool evenBlock = true;
RowLoop:
    for (i = 0; i < (height >> 1); i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    ColLoop:
        for (j = 0; j < width; j++) {
// clang-format off
            #pragma HLS pipeline
            #pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
            // clang-format on
            if (evenBlock) {
                uv = _uv.read(idx++);
                u.range(7, 0) = (uint8_t)uv.range(7, 0);
                v.range(7, 0) = (uint8_t)uv.range(15, 8);
            }
            arr_u[j] = u;
            arr_v[j] = v;

            _u.write(((i * 2) * (_u.cols >> XF_BITSHIFT(NPC))) + j, u);
            _v.write(((i * 2) * (_v.cols >> XF_BITSHIFT(NPC))) + j, v);
            evenBlock = evenBlock ? false : true;
        }
        for (int k = 0; k < width; k++) {
            _u.write((((i * 2) + 1) * (_u.cols >> XF_BITSHIFT(NPC))) + k, arr_u[k]);
            _v.write((((i * 2) + 1) * (_v.cols >> XF_BITSHIFT(NPC))) + k, arr_v[k]);
        }
    }
}

template <int SRC_T,
          int UV_T,
          int DST_T,
          int ROWS,
          int COLS,
          int NPC,
          int NPC_UV,
          int WORDWIDTH_Y,
          int WORDWIDTH_UV,
          int WORDWIDTH_DST>
void KernNv122Rgba(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _y,
                   xf::cv::Mat<UV_T, ROWS / 2, COLS / 2, NPC_UV>& _uv,
                   xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _rgba,
                   uint16_t height,
                   uint16_t width) {
    hls::stream<XF_SNAME(WORDWIDTH_UV)> uvStream;
// clang-format off
    #pragma HLS STREAM variable=&uvStream  depth=COLS
    // clang-format on
    XF_SNAME(WORDWIDTH_Y) yPacked;
    XF_SNAME(WORDWIDTH_UV) uvPacked;
    XF_SNAME(WORDWIDTH_DST) rgba;
    unsigned long long int idx = 0, idx1 = 0;
    uint8_t y1, y2;
    int32_t V2Rtemp, U2Gtemp, V2Gtemp, U2Btemp;
    int8_t u, v;
    bool evenRow = true, evenBlock = true;
RowLoop:
    for (int i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    ColLoop:
        for (int j = 0; j < width; j++) {
// clang-format off
            #pragma HLS pipeline
            #pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
            // clang-format on

            yPacked = _y.read(i * width + j);
            if (evenRow) {
                if (evenBlock) {
                    uvPacked = _uv.read(idx++);
                    uvStream.write(uvPacked);
                }
            } else { // Keep a copy of UV row data in stream to use for oddrow
                if (evenBlock) {
                    uvPacked = uvStream.read();
                }
            }
            //			auExtractPixels<NPC, WORDWIDTH_SRC,
            // XF_8UP>(UVbuf, UVPacked, 0);
            uint8_t t = yPacked.range(7, 0);
            y1 = t > 16 ? t - 16 : 0;
            v = (uint8_t)uvPacked.range(15, 8) - 128;
            u = (uint8_t)uvPacked.range(7, 0) - 128;

            V2Rtemp = v * (short int)V2R;
            U2Gtemp = (short int)U2G * u;
            V2Gtemp = (short int)V2G * v;
            U2Btemp = u * (short int)U2B;

            // R = 1.164*Y + 1.596*V = Y + 0.164*Y + V + 0.596*V
            // G = 1.164*Y - 0.813*V - 0.391*U = Y + 0.164*Y - 0.813*V - 0.391*U
            // B = 1.164*Y + 2.018*U = Y + 0.164 + 2*U + 0.018*U
            rgba.range(7, 0) = CalculateR(y1, V2Rtemp, v);        // R
            rgba.range(15, 8) = CalculateG(y1, U2Gtemp, V2Gtemp); // G
            rgba.range(23, 16) = CalculateB(y1, U2Btemp, u);      // B
            rgba.range(31, 24) = 255;                             // A

            //			PackedPixels =
            // PackRGBAPixels<WORDWIDTH_DST>(RGB);
            _rgba.write(idx1++, rgba);
            evenBlock = evenBlock ? false : true;
        }
        evenRow = evenRow ? false : true;
    }
    if (height & 1) {
        for (int i = 0; i < width; i++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
            // clang-format on
            uvStream.read();
        }
    }
}

template <int SRC_T, int UV_T, int ROWS, int COLS, int NPC, int NPC_UV, int WORDWIDTH_SRC, int WORDWIDTH_DST, int TC>
void KernNv122Iyuv(xf::cv::Mat<UV_T, ROWS / 2, COLS / 2, NPC_UV>& _uv,
                   xf::cv::Mat<SRC_T, ROWS / 4, COLS, NPC>& _u,
                   xf::cv::Mat<SRC_T, ROWS / 4, COLS, NPC>& _v,
                   uint16_t height,
                   uint16_t width) {
    XF_PTNAME(XF_8UP) u, v;
    XF_SNAME(WORDWIDTH_SRC) uv;
    unsigned long long int idx = 0;
    ap_uint<13> i, j;
RowLoop:
    for (i = 0; i<height>> 1; i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    ColLoop:
        for (j = 0; j < (width >> 1); j++) {
// clang-format off
            #pragma HLS pipeline
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            // clang-format on
            //			u = _uv.read();
            //			v = _uv.read();
            uv = _uv.read(i * (width >> 1) + j);

            _u.write(idx, uv.range(7, 0));
            _v.write(idx++, uv.range(15, 8));
        }
    }
}

template <int SRC_T, int UV_T, int ROWS, int COLS, int NPC, int NPC_UV, int WORDWIDTH_VU, int WORDWIDTH_DST, int TC>
void KernNv212Yuv4(xf::cv::Mat<UV_T, ROWS / 2, COLS / 2, NPC_UV>& _vu,
                   xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _u,
                   xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _v,
                   uint16_t height,
                   uint16_t width) {
    XF_PTNAME(XF_16UP) uv;
    XF_SNAME(WORDWIDTH_DST) u, v;
    XF_SNAME(WORDWIDTH_VU) uvPacked;
    XF_TNAME(SRC_T, NPC) arr_u[COLS];
    XF_TNAME(SRC_T, NPC) arr_v[COLS];

    unsigned long long int idx = 0, idx1 = 0;
    ap_uint<13> i, j;
    bool evenBlock = true;
RowLoop:
    for (i = 0; i < (height >> 1); i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    ColLoop:
        for (j = 0; j < width; j++) {
// clang-format off
            #pragma HLS pipeline
            #pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
            // clang-format on
            if (evenBlock) {
                uv = _vu.read(idx++);
                v.range(7, 0) = (uint8_t)uv.range(7, 0);
                u.range(7, 0) = (uint8_t)uv.range(15, 8);
            }
            arr_u[j] = u;
            arr_v[j] = v;

            _u.write(((i * 2) * (_u.cols >> XF_BITSHIFT(NPC))) + j, u);
            _v.write(((i * 2) * (_v.cols >> XF_BITSHIFT(NPC))) + j, v);
            evenBlock = evenBlock ? false : true;
        }
        for (int k = 0; k < width; k++) {
            _u.write((((i * 2) + 1) * (_u.cols >> XF_BITSHIFT(NPC))) + k, arr_u[k]);
            _v.write((((i * 2) + 1) * (_v.cols >> XF_BITSHIFT(NPC))) + k, arr_v[k]);
        }
    }
}

template <int SRC_T,
          int UV_T,
          int DST_T,
          int ROWS,
          int COLS,
          int NPC,
          int NPC_UV,
          int WORDWIDTH_Y,
          int WORDWIDTH_VU,
          int WORDWIDTH_DST>
void KernNv212Rgba(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _y,
                   xf::cv::Mat<UV_T, ROWS / 2, COLS / 2, NPC_UV>& _vu,
                   xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _rgba,
                   uint16_t height,
                   uint16_t width) {
    hls::stream<XF_SNAME(WORDWIDTH_VU)> vuStream;
// clang-format off
    #pragma HLS STREAM variable=&vuStream  depth=COLS
    // clang-format on
    XF_SNAME(WORDWIDTH_Y) yPacked;
    XF_SNAME(WORDWIDTH_VU) vuPacked;
    XF_SNAME(WORDWIDTH_DST) rgba;
    unsigned long long int idx = 0, idx1 = 0;
    ap_uint<13> i, j;
    uint8_t y1, y2;
    int32_t V2Rtemp, U2Gtemp, V2Gtemp, U2Btemp;
    int8_t u, v;
    bool evenRow = true, evenBlock = true;
RowLoop:
    for (i = 0; i < (height); i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    ColLoop:
        for (j = 0; j < width; j++) {
// clang-format off
            #pragma HLS pipeline
            #pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
            // clang-format on

            yPacked = _y.read(i * width + j);
            //			auExtractPixels<NPC, WORDWIDTH_SRC,
            // XF_8UP>(Ybuf, YPacked, 0);
            if (evenRow) {
                if (evenBlock) {
                    vuPacked = _vu.read(idx++);
                    vuStream.write(vuPacked);
                }
            } else { // Keep a copy of UV row data in stream to use for oddrow
                if (evenBlock) {
                    vuPacked = vuStream.read();
                }
            }
            //			auExtractPixels<NPC, WORDWIDTH_SRC,
            // XF_8UP>(UVbuf, UVPacked, 0);
            uint8_t t = yPacked.range(7, 0);
            y1 = t > 16 ? t - 16 : 0;
            u = (uint8_t)vuPacked.range(15, 8) - 128;
            v = (uint8_t)vuPacked.range(7, 0) - 128;

            V2Rtemp = v * (short int)V2R;
            U2Gtemp = (short int)U2G * u;
            V2Gtemp = (short int)V2G * v;
            U2Btemp = u * (short int)U2B;

            // R = 1.164*Y + 1.596*V = Y + 0.164*Y + V + 0.596*V
            // G = 1.164*Y - 0.813*V - 0.391*U = Y + 0.164*Y - 0.813*V - 0.391*U
            // B = 1.164*Y + 2.018*U = Y + 0.164 + 2*U + 0.018*U
            rgba.range(7, 0) = CalculateR(y1, V2Rtemp, v);        // R
            rgba.range(15, 8) = CalculateG(y1, U2Gtemp, V2Gtemp); // G
            rgba.range(23, 16) = CalculateB(y1, U2Btemp, u);      // B
            rgba.range(31, 24) = 255;                             // A

            //			PackedPixels =
            // PackRGBAPixels<WORDWIDTH_DST>(RGB);
            _rgba.write(idx1++, rgba);
            evenBlock = evenBlock ? false : true;
        }
        evenRow = evenRow ? false : true;
    }
    if (height & 1) {
        for (i = 0; i < width; i++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
            // clang-format on
            vuStream.read();
        }
    }
}

template <int SRC_T, int UV_T, int ROWS, int COLS, int NPC, int NPC_UV, int WORDWIDTH_SRC, int WORDWIDTH_DST, int TC>
void KernNv212Iyuv(xf::cv::Mat<UV_T, ROWS / 2, COLS / 2, NPC_UV>& _vu,
                   xf::cv::Mat<SRC_T, ROWS / 4, COLS, NPC>& _u,
                   xf::cv::Mat<SRC_T, ROWS / 4, COLS, NPC>& _v,
                   uint16_t height,
                   uint16_t width) {
    ap_uint<13> i, j;
    XF_PTNAME(XF_8UP) u, v;
    XF_SNAME(WORDWIDTH_SRC) VUPacked, UVPacked0, UVPacked1;
    unsigned long long int idx = 0, idx1 = 0;
RowLoop:
    for (i = 0; i < (height >> 1); i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    ColLoop:
        for (j = 0; j < (width >> 1); j++) {
// clang-format off
            #pragma HLS pipeline
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            // clang-format on
            VUPacked = _vu.read(idx++);
            u = (uint8_t)VUPacked.range(15, 8);
            v = (uint8_t)VUPacked.range(7, 0);
            _u.write(idx1, u);
            _v.write(idx1++, v);
        }
    }
}

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC, int WORDWIDTH_SRC, int WORDWIDTH_DST, int TC>
void KernIyuv2Rgba(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _y,
                   xf::cv::Mat<SRC_T, ROWS / 4, COLS, NPC>& _u,
                   xf::cv::Mat<SRC_T, ROWS / 4, COLS, NPC>& _v,
                   xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _rgba,
                   uint16_t height,
                   uint16_t width) {
    unsigned long long int idx = 0, idx1 = 0;
    ap_uint<13> i, j;
    hls::stream<XF_SNAME(WORDWIDTH_SRC)> uStream, vStream;
// clang-format off
    #pragma HLS STREAM variable=&uStream  depth=COLS
    #pragma HLS STREAM variable=&vStream  depth=COLS
    // clang-format on

    XF_SNAME(WORDWIDTH_SRC) yPacked, uPacked, vPacked;
    XF_SNAME(WORDWIDTH_DST) rgba;

    uint8_t y1, y2;
    int32_t V2Rtemp, U2Gtemp, V2Gtemp, U2Btemp;
    int8_t u, v;
    bool evenRow = true, evenBlock = true;
RowLoop:
    for (i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    ColLoop:
        for (j = 0; j < width; j++) {
// clang-format off
            #pragma HLS pipeline
            #pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
            // clang-format on
            yPacked = _y.read(i * width + j);

            if (evenBlock) {
                if (evenRow) {
                    uPacked = _u.read(idx);
                    uStream.write(uPacked);
                    vPacked = _v.read(idx++);
                    vStream.write(vPacked);
                } else {
                    /* Copy of the U and V values are pushed into stream to be used for
                     * next row */
                    uPacked = uStream.read();
                    vPacked = vStream.read();
                }
            }

            y1 = (uint8_t)yPacked.range(7, 0) > 16 ? (uint8_t)yPacked.range(7, 0) - 16 : 0;

            u = (uint8_t)uPacked.range(7, 0) - 128;
            v = (uint8_t)vPacked.range(7, 0) - 128;

            V2Rtemp = v * (short int)V2R;
            U2Gtemp = (short int)U2G * u;
            V2Gtemp = (short int)V2G * v;
            U2Btemp = u * (short int)U2B;

            // R = 1.164*Y + 1.596*V = Y + 0.164*Y + V + 0.596*V
            // G = 1.164*Y - 0.813*V - 0.391*U = Y + 0.164*Y - 0.813*V - 0.391*U
            // B = 1.164*Y + 2.018*U = Y + 0.164 + 2*U + 0.018*U
            rgba.range(7, 0) = CalculateR(y1, V2Rtemp, v);        // R
            rgba.range(15, 8) = CalculateG(y1, U2Gtemp, V2Gtemp); // G
            rgba.range(23, 16) = CalculateB(y1, U2Btemp, u);      // B
            rgba.range(31, 24) = 255;                             // A

            _rgba.write(idx1++, rgba);
            evenBlock = evenBlock ? false : true;
        }
        evenRow = evenRow ? false : true;
    }
}

template <int SRC_T, int ROWS, int COLS, int NPC, int WORDWIDTH, int rTC, int cTC>
void KernIyuv2Yuv4(xf::cv::Mat<SRC_T, ROWS / 4, COLS, NPC>& _in_u,
                   xf::cv::Mat<SRC_T, ROWS / 4, COLS, NPC>& _in_v,
                   xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _u_image,
                   xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _v_image,
                   uint16_t height,
                   uint16_t width) {
    hls::stream<XF_SNAME(WORDWIDTH)> inter_u;
// clang-format off
    #pragma HLS stream variable=inter_u depth=COLS
    // clang-format on

    hls::stream<XF_SNAME(WORDWIDTH)> inter_v;
// clang-format off
    #pragma HLS stream variable=inter_v depth=COLS
    // clang-format on

    XF_TNAME(SRC_T, NPC) arr_U[COLS];
    XF_TNAME(SRC_T, NPC) arr_V[COLS];

    XF_SNAME(WORDWIDTH) IUPacked, IVPacked;
    XF_PTNAME(XF_8UP) in_u, in_v;
    unsigned long long int idx = 0, idx1 = 0, in_idx1 = 0, in_idx2 = 0;
RowLoop:
    for (int i = 0; i < ((height >> 2) << 1); i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN
        #pragma HLS LOOP_TRIPCOUNT min=rTC max=rTC
    // clang-format on
    ColLoop:
        for (int j = 0, k = 0; j < (width >> 1); j++, k += 2) {
// clang-format off
            #pragma HLS pipeline
            #pragma HLS LOOP_TRIPCOUNT min=cTC max=cTC
            // clang-format on
            IUPacked = _in_u.read(in_idx1++);
            IVPacked = _in_v.read(in_idx2++);

            _u_image.write(((i * 2) * (width)) + k, IUPacked);
            _u_image.write(((i * 2) * (width)) + k + 1, IUPacked);
            _v_image.write(((i * 2) * (width)) + k, IVPacked);
            _v_image.write(((i * 2) * (width)) + k + 1, IVPacked);

            inter_u.write(IUPacked);
            inter_v.write(IVPacked);
            inter_u.write(IUPacked);
            inter_v.write(IVPacked);
        }
        for (int j = 0; j < width; j++) {
// clang-format off
            #pragma HLS pipeline
            // clang-format on
            _u_image.write((((i * 2) + 1) * (width) + j), inter_u.read());
            _v_image.write((((i * 2) + 1) * (width) + j), inter_v.read());
        }
    }
}

template <int SRC_T,
          int UV_T,
          int ROWS,
          int COLS,
          int NPC,
          int NPC_UV,
          int WORDWIDTH_SRC,
          int WORDWIDTH_UV,
          int rTC,
          int cTC>
void KernIyuv2Nv12(xf::cv::Mat<SRC_T, ROWS / 4, COLS, NPC>& _u,
                   xf::cv::Mat<SRC_T, ROWS / 4, COLS, NPC>& _v,
                   xf::cv::Mat<UV_T, ROWS / 2, COLS / 2, NPC_UV>& _uv,
                   uint16_t height,
                   uint16_t width) {
    ap_uint<13> i, j;
    XF_SNAME(WORDWIDTH_SRC) u, v;
    XF_SNAME(WORDWIDTH_UV) uv;
    unsigned long long int idx = 0;
RowLoop:
    for (i = 0; i<height>> 1; i++) {
// Reading the plane interleaved U and V data from streams,
// packing them in pixel interleaved and writing out to UV stream
// clang-format off
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT min=rTC max=rTC
    // clang-format on
    ColLoop:
        for (j = 0; j < (width >> 1); j++) {
// clang-format off
            #pragma HLS pipeline
            #pragma HLS LOOP_TRIPCOUNT min=cTC max=cTC
            // clang-format on
            u = _u.read(i * (width >> 1) + j);
            v = _v.read(i * (width >> 1) + j);
            uv.range(7, 0) = u;
            uv.range(15, 8) = v;
            _uv.write(idx++, uv);
        }
    }
}

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC, int WORDWIDTH_SRC, int WORDWIDTH_DST>
void KernRgba2Yuv4(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _rgba,
                   xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _y,
                   xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _u,
                   xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _v,
                   uint16_t height,
                   uint16_t width) {
    XF_SNAME(XF_32UW) rgba;
    uint8_t y, u, v;
    unsigned long long int idx = 0;
RowLoop:
    for (int i = 0; i < height; ++i) {
// clang-format off
        #pragma HLS LOOP_FLATTEN OFF
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    ColLoop:
        for (int j = 0; j < width; ++j) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
            #pragma HLS PIPELINE
            // clang-format on
            rgba = _rgba.read(i * width + j);

            y = CalculateY(rgba.range(7, 0), rgba.range(15, 8), rgba.range(23, 16));
            u = CalculateU(rgba.range(7, 0), rgba.range(15, 8), rgba.range(23, 16));
            v = CalculateV(rgba.range(7, 0), rgba.range(15, 8), rgba.range(23, 16));

            _y.write(idx, y);
            _u.write(idx, u);
            _v.write(idx++, v);
        }
    }
}

template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int ROWS_U,
          int ROWS_V>
void KernRgba2Iyuv(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _rgba,
                   xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _y,
                   xf::cv::Mat<DST_T, ROWS / 4, COLS, NPC>& _u,
                   xf::cv::Mat<DST_T, ROWS / 4, COLS, NPC>& _v,
                   uint16_t height,
                   uint16_t width) {
    XF_SNAME(XF_32UW) rgba;
    uint8_t y, u, v;
    bool evenRow = true, evenBlock = true;
    unsigned long long int idx = 0, idx1 = 0;
RowLoop:
    for (int i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    ColLoop:
        for (int j = 0; j < width; j++) {
// clang-format off
            #pragma HLS pipeline
            #pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
            // clang-format on
            rgba = _rgba.read(i * width + j);
            uint8_t r = rgba.range(7, 0);
            uint8_t g = rgba.range(15, 8);
            uint8_t b = rgba.range(23, 16);

            y = CalculateY(r, g, b);
            if (evenRow) {
                if (evenBlock) {
                    u = CalculateU(r, g, b);
                    v = CalculateV(r, g, b);
                }
            }
            _y.write(idx1++, y);
            if (evenRow & !evenBlock) {
                _u.write(idx, u);
                _v.write(idx++, v);
            }
            evenBlock = evenBlock ? false : true;
        }
        evenRow = evenRow ? false : true;
    }
}

template <int SRC_T,
          int Y_T,
          int UV_T,
          int ROWS,
          int COLS,
          int NPC,
          int NPC_UV,
          int WORDWIDTH_SRC,
          int WORDWIDTH_Y,
          int WORDWIDTH_UV>
void KernRgba2Nv12(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _rgba,
                   xf::cv::Mat<Y_T, ROWS, COLS, NPC>& _y,
                   xf::cv::Mat<UV_T, ROWS / 2, COLS / 2, NPC_UV>& _uv,
                   uint16_t height,
                   uint16_t width) {
    //	XF_SNAME(XF_32UW) rgba;
    XF_TNAME(SRC_T, NPC) rgba;
    ap_uint<16> val1;
    uint8_t y, u, v;
    unsigned long long int idx = 0, idx1 = 0;
    bool evenRow = true, evenBlock = true;

RowLoop:
    for (int i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    ColLoop:
        for (int j = 0; j < width; j++) {
// clang-format off
            #pragma HLS pipeline
            #pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
            // clang-format on
            rgba = _rgba.read(i * width + j);
            uint8_t r = rgba.range(7, 0);
            uint8_t g = rgba.range(15, 8);
            uint8_t b = rgba.range(23, 16);

            y = CalculateY(r, g, b);
            if (evenRow) {
                u = CalculateU(r, g, b);
                v = CalculateV(r, g, b);
            }
            _y.write(idx++, y);
            if (evenRow) {
                if ((j & 0x01) == 0)
                    //{
                    _uv.write(idx1++, u | (uint16_t)v << 8);
                //_uv.write(v);
                //}
                //	_uv.write(u | (uint16_t)v << 8);
            }
        }
        evenRow = evenRow ? false : true;
    }
}

template <int SRC_T,
          int Y_T,
          int UV_T,
          int ROWS,
          int COLS,
          int NPC,
          int NPC_UV,
          int WORDWIDTH_SRC,
          int WORDWIDTH_Y,
          int WORDWIDTH_VU>
void KernRgba2Nv21(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _rgba,
                   xf::cv::Mat<Y_T, ROWS, COLS, NPC>& _y,
                   xf::cv::Mat<UV_T, ROWS / 2, COLS / 2, NPC_UV>& _vu,
                   uint16_t height,
                   uint16_t width) {
    width = width >> XF_BITSHIFT(NPC);
    XF_TNAME(SRC_T, NPC) rgba;
    uint8_t y, u, v;
    unsigned long long int idx = 0, idx1 = 0;
    bool evenRow = true, evenBlock = true;

RowLoop:
    for (int i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    ColLoop:
        for (int j = 0; j < width; j++) {
// clang-format off
            #pragma HLS pipeline
            #pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
            // clang-format on
            rgba = _rgba.read(i * width + j);
            uint8_t r = rgba.range(7, 0);
            uint8_t g = rgba.range(15, 8);
            uint8_t b = rgba.range(23, 16);

            y = CalculateY(r, g, b);
            if (evenRow) {
                u = CalculateU(r, g, b);
                v = CalculateV(r, g, b);
            }
            _y.write(idx++, y);
            if (evenRow) {
                if ((j & 0x01) == 0) _vu.write(idx1++, v | ((uint16_t)u << 8));
            }
        }
        evenRow = evenRow ? false : true;
    }
}

// Yuyv2Rgba
template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC, int WORDWIDTH_SRC, int WORDWIDTH_DST, int TC>
void KernYuyv2Rgba(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _yuyv,
                   xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _rgba,
                   uint16_t height,
                   uint16_t width) {
    XF_SNAME(WORDWIDTH_DST) rgba;
    XF_SNAME(WORDWIDTH_SRC) yu, yv;
    XF_PTNAME(XF_8UP) r, g, b;
    int8_t y1, y2, u, v;
    int32_t V2Rtemp, U2Gtemp, V2Gtemp, U2Btemp;
    unsigned long long int idx = 0;
RowLoop:
    for (int i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        #pragma HLS LOOP_FLATTEN off
    // clang-format on
    ColLoop:
        for (int j = 0; j < width; j += 2) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            #pragma HLS pipeline
            // clang-format on

            yu = _yuyv.read(i * width + j);
            yv = _yuyv.read(i * width + j + 1);
            u = (uint8_t)yu.range(15, 8) - 128;
            y1 = (yu.range(7, 0) > 16) ? ((uint8_t)yu.range(7, 0) - 16) : 0;

            v = (uint8_t)yv.range(15, 8) - 128;
            y2 = (yv.range(7, 0) > 16) ? ((uint8_t)yv.range(7, 0) - 16) : 0;

            V2Rtemp = v * (short int)V2R;
            U2Gtemp = (short int)U2G * u;
            V2Gtemp = (short int)V2G * v;
            U2Btemp = u * (short int)U2B;

            r = CalculateR(y1, V2Rtemp, v);
            g = CalculateG(y1, U2Gtemp, V2Gtemp);
            b = CalculateB(y1, U2Btemp, u);

            rgba = ((ap_uint32_t)r) | ((ap_uint32_t)g << 8) | ((ap_uint32_t)b << 16) | (0xFF000000);
            _rgba.write(idx++, rgba);

            r = CalculateR(y2, V2Rtemp, v);
            g = CalculateG(y2, U2Gtemp, V2Gtemp);
            b = CalculateB(y2, U2Btemp, u);

            rgba = ((ap_uint32_t)r) | ((ap_uint32_t)g << 8) | ((ap_uint32_t)b << 16) | (0xFF000000);
            _rgba.write(idx++, rgba);
        }
    }
}

// Yuyv2Nv12
template <int SRC_T,
          int Y_T,
          int UV_T,
          int ROWS,
          int COLS,
          int NPC,
          int NPC_UV,
          int WORDWIDTH_SRC,
          int WORDWIDTH_Y,
          int WORDWIDTH_UV,
          int TC>
void KernYuyv2Nv12(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _yuyv,
                   xf::cv::Mat<Y_T, ROWS, COLS, NPC>& _y,
                   xf::cv::Mat<UV_T, ROWS / 2, COLS / 2, NPC_UV>& _uv,
                   uint16_t height,
                   uint16_t width) {
    XF_SNAME(WORDWIDTH_SRC) yu, yv;
    XF_PTNAME(XF_8UP) y1, y2;
    unsigned long long int idx = 0, idx1 = 0;
    XF_SNAME(WORDWIDTH_UV) uv;
    bool evenRow = true;
RowLoop:
    for (int i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    ColLoop:
        for (int j = 0; j < width; j += 2) {
// clang-format off
            #pragma HLS pipeline
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            // clang-format on

            yu = _yuyv.read(i * width + j);
            yv = _yuyv.read(i * width + j + 1);
            y1 = yu.range(7, 0);
            if (evenRow) uv.range(7, 0) = yu.range(15, 8);

            y2 = yv.range(7, 0);
            if (evenRow) uv.range(15, 8) = yv.range(15, 8);

            _y.write(idx++, y1);
            _y.write(idx++, y2);
            if (evenRow) {
                _uv.write(idx1++, uv);
            }
        }
        evenRow = evenRow ? false : true;
    }
}
// Yuyv2Nv12
template <int SRC_T,
          int Y_T,
          int UV_T,
          int ROWS,
          int COLS,
          int NPC,
          int NPC_UV,
          int WORDWIDTH_SRC,
          int WORDWIDTH_Y,
          int WORDWIDTH_UV,
          int TC>
void KernYuyv2Nv21(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _yuyv,
                   xf::cv::Mat<Y_T, ROWS, COLS, NPC>& _y,
                   xf::cv::Mat<UV_T, ROWS / 2, COLS / 2, NPC_UV>& _uv,
                   uint16_t height,
                   uint16_t width) {
    XF_SNAME(WORDWIDTH_SRC) yu, yv;
    XF_PTNAME(XF_8UP) y1, y2;
    unsigned long long int idx = 0, idx1 = 0;
    XF_SNAME(WORDWIDTH_UV) uv;
    bool evenRow = true;
RowLoop:
    for (int i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    ColLoop:
        for (int j = 0; j < width; j += 2) {
// clang-format off
            #pragma HLS pipeline
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            // clang-format on

            yu = _yuyv.read(i * width + j);
            yv = _yuyv.read(i * width + j + 1);
            y1 = yu.range(7, 0);
            if (evenRow) uv.range(7, 0) = yv.range(15, 8);

            y2 = yv.range(7, 0);
            if (evenRow) uv.range(15, 8) = yu.range(15, 8);

            _y.write(idx++, y1);
            _y.write(idx++, y2);
            if (evenRow) {
                _uv.write(idx1++, uv);
            }
        }
        evenRow = evenRow ? false : true;
    }
}

// Yuyv2Iyuv
template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC, int WORDWIDTH_SRC, int WORDWIDTH_DST, int TC>
void KernYuyv2Iyuv(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _yuyv,
                   xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _y,
                   xf::cv::Mat<DST_T, ROWS / 4, COLS, NPC>& _u,
                   xf::cv::Mat<DST_T, ROWS / 4, COLS, NPC>& _v,
                   uint16_t height,
                   uint16_t width) {
    XF_SNAME(WORDWIDTH_SRC) yu, yv;
    unsigned long long int idx = 0, idx1 = 0;
    bool evenRow = true, evenBlock = true;
    XF_PTNAME(XF_8UP) y1, y2, u, v;

RowLoop:
    for (int i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    ColLoop:
        for (int j = 0; j < width; j += 2) {
// clang-format off
            #pragma HLS pipeline
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            // clang-format on

            yu = _yuyv.read(i * width + j);
            yv = _yuyv.read(i * width + j + 1);
            y1 = yu.range(7, 0);
            y2 = yv.range(7, 0);
            _y.write(idx, y1);
            idx++;
            _y.write(idx, y2);
            idx++;
            if (evenRow) u = yu.range(15, 8);

            if (evenRow) v = yv.range(15, 8);

            if (evenRow) {
                _u.write(idx1, u);
                _v.write(idx1, v);
                idx1++;
            }
        }
        evenRow = evenRow ? false : true;
    }
}

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC, int WORDWIDTH_SRC, int WORDWIDTH_DST, int TC>
void KernUyvy2Iyuv(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _uyvy,
                   xf::cv::Mat<DST_T, ROWS, COLS, NPC>& y_plane,
                   xf::cv::Mat<DST_T, ROWS / 4, COLS, NPC>& u_plane,
                   xf::cv::Mat<DST_T, ROWS / 4, COLS, NPC>& v_plane,
                   uint16_t height,
                   uint16_t width) {
    XF_SNAME(WORDWIDTH_SRC) uy, vy;
    bool evenRow = true, evenBlock = true;
    XF_PTNAME(XF_8UP) y1, y2, u, v;
    unsigned long long int idx = 0, idx1 = 0;

RowLoop:
    for (int i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    ColLoop:
        for (int j = 0; j < width; j += 2) {
// clang-format off
            #pragma HLS pipeline
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            // clang-format on

            uy = _uyvy.read(i * width + j);
            vy = _uyvy.read(i * width + j + 1);

            y1 = uy.range(15, 8);

            y_plane.write(idx1, y1);
            idx1++;
            if (evenRow) u = uy.range(7, 0);

            y2 = vy.range(15, 8);

            y_plane.write(idx1, y2);
            idx1++;
            if (evenRow) v = vy.range(7, 0);

            if (evenRow) {
                u_plane.write(idx, u);
                v_plane.write(idx, v);
                idx++;
            }
        }

        evenRow = evenRow ? false : true;
    }
}

// Uyvy2Nv12
template <int SRC_T,
          int Y_T,
          int UV_T,
          int ROWS,
          int COLS,
          int NPC,
          int NPC_UV,
          int WORDWIDTH_SRC,
          int WORDWIDTH_Y,
          int WORDWIDTH_UV,
          int TC>
void KernUyvy2Nv12(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& uyvy,
                   xf::cv::Mat<Y_T, ROWS, COLS, NPC>& y_plane,
                   xf::cv::Mat<UV_T, ROWS / 2, COLS / 2, NPC_UV>& uv_plane,
                   uint16_t height,
                   uint16_t width) {
    XF_SNAME(WORDWIDTH_SRC) uy, vy;
    XF_PTNAME(XF_8UP) y1, y2;
    XF_SNAME(WORDWIDTH_UV) uv;
    bool evenRow = true;
    unsigned long long int idx = 0, idx1 = 0;
RowLoop:
    for (int i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    ColLoop:
        for (int j = 0; j < width; j += 2) {
// clang-format off
            #pragma HLS pipeline
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            // clang-format on

            uy = uyvy.read(i * width + j);
            vy = uyvy.read(i * width + j + 1);

            y1 = uy.range(15, 8);
            if (evenRow) uv.range(7, 0) = uy.range(7, 0);

            y2 = vy.range(15, 8);
            if (evenRow) uv.range(15, 8) = vy.range(7, 0);

            y_plane.write(idx1, y1);
            idx1++;
            y_plane.write(idx1, y2);
            idx1++;
            if (evenRow) {
                uv_plane.write(idx, uv);
                idx++;
            }
        }
        evenRow = evenRow ? false : true;
    }
}
// Uyvy2Nv12
template <int SRC_T,
          int Y_T,
          int UV_T,
          int ROWS,
          int COLS,
          int NPC,
          int NPC_UV,
          int WORDWIDTH_SRC,
          int WORDWIDTH_Y,
          int WORDWIDTH_UV,
          int TC>
void KernUyvy2Nv21(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& uyvy,
                   xf::cv::Mat<Y_T, ROWS, COLS, NPC>& y_plane,
                   xf::cv::Mat<UV_T, ROWS / 2, COLS / 2, NPC_UV>& uv_plane,
                   uint16_t height,
                   uint16_t width) {
    XF_SNAME(WORDWIDTH_SRC) uy, vy;
    XF_PTNAME(XF_8UP) y1, y2;
    XF_SNAME(WORDWIDTH_UV) uv;
    bool evenRow = true;
    unsigned long long int idx = 0, idx1 = 0;
RowLoop:
    for (int i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    ColLoop:
        for (int j = 0; j < width; j += 2) {
// clang-format off
            #pragma HLS pipeline
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            // clang-format on

            uy = uyvy.read(i * width + j);
            vy = uyvy.read(i * width + j + 1);

            y1 = uy.range(15, 8);
            if (evenRow) uv.range(7, 0) = vy.range(7, 0);

            y2 = vy.range(15, 8);
            if (evenRow) uv.range(15, 8) = uy.range(7, 0);

            y_plane.write(idx1, y1);
            idx1++;
            y_plane.write(idx1, y2);
            idx1++;
            if (evenRow) {
                uv_plane.write(idx, uv);
                idx++;
            }
        }
        evenRow = evenRow ? false : true;
    }
}
// Uyvy2Rgba
template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC, int WORDWIDTH_SRC, int WORDWIDTH_DST, int TC>
void KernUyvy2Rgba(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _uyvy,
                   xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _rgba,
                   uint16_t height,
                   uint16_t width) {
    XF_SNAME(WORDWIDTH_DST) rgba;

    XF_SNAME(WORDWIDTH_SRC) uyvy;

    XF_SNAME(WORDWIDTH_SRC) uy;
    XF_SNAME(WORDWIDTH_SRC) vy;
    unsigned long long int idx = 0;
    XF_PTNAME(XF_8UP) r, g, b;
    int8_t y1, y2, u, v;
    int32_t V2Rtemp, U2Gtemp, V2Gtemp, U2Btemp;

RowLoop:
    for (int i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        #pragma HLS LOOP_FLATTEN off
    // clang-format on
    ColLoop:
        for (int j = 0; j < width; j += 2) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            #pragma HLS pipeline
            // clang-format on

            uy = _uyvy.read(i * width + j);
            vy = _uyvy.read(i * width + j + 1);

            u = (uint8_t)uy.range(7, 0) - 128;

            /*			if(uyvy.range(15,8) > 16)
                                            y1 = (uint8_t)uyvy.range(15,8) - 16;
                                    else
                                            y1 = 0;*/

            y1 = (uy.range(15, 8) > 16) ? ((uint8_t)uy.range(15, 8) - 16) : 0;

            v = (uint8_t)vy.range(7, 0) - 128;

            /*			if(uyvy.range(31,24) > 16)
                                            y2 = ((uint8_t)uyvy.range(31,24) - 16);
                                    else
                                            y2 = 0;*/
            y2 = (vy.range(15, 8) > 16) ? ((uint8_t)vy.range(15, 8) - 16) : 0;

            V2Rtemp = v * (short int)V2R;
            U2Gtemp = (short int)U2G * u;
            V2Gtemp = (short int)V2G * v;
            U2Btemp = u * (short int)U2B;

            r = CalculateR(y1, V2Rtemp, v);
            g = CalculateG(y1, U2Gtemp, V2Gtemp);
            b = CalculateB(y1, U2Btemp, u);

            rgba = ((ap_uint32_t)r) | ((ap_uint32_t)g << 8) | ((ap_uint32_t)b << 16) | (0xFF000000);
            _rgba.write(idx, rgba);
            idx++;

            r = CalculateR(y2, V2Rtemp, v);
            g = CalculateG(y2, U2Gtemp, V2Gtemp);
            b = CalculateB(y2, U2Btemp, u);

            rgba = ((ap_uint32_t)r) | ((ap_uint32_t)g << 8) | ((ap_uint32_t)b << 16) | (0xFF000000);
            _rgba.write(idx, rgba);
            idx++;
        }
    }
}
#endif
