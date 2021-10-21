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

#include "kernel3/dc_tokenize.hpp"

void hls_EncodeHybridVarLenUint(uint32_t value, int* symbol, int* nbits, int* bits) {
#pragma HLS INLINE
    if (value < hls_kHybridEncodingSplitToken) {
        *symbol = value;
        *nbits = 0;
        *bits = 0;
    } else {
        uint32_t n = hls_Log2FloorNonZero_32b(value);
        uint32_t m = value - (1 << n);
        *symbol = hls_kHybridEncodingSplitToken + ((n - hls_kHybridEncodingDirectSplitExponent) << 1) + (m >> (n - 1));
        *nbits = n - 1;
        *bits = value & ((1 << (n - 1)) - 1);
    }
}

// Pack signed integer and encode value.
void hls_EncodeHybridVarLenInt(int32_t value, int* symbol, int* nbits, int* bits) {
#pragma HLS INLINE
    hls_EncodeHybridVarLenUint(hls_PackSigned_32b(value), symbol, nbits, bits);
}

void Tokenize_DC_top(const bool rle,
                     const hls_Rect rect,
                     hls::stream<dct_t>& strm_dc_residuals,

                     hls::stream<addr_t>& strm_token_addr,
                     hls::stream<hls_Token_symb>& strm_token_symb,
                     hls::stream<hls_Token_bits>& strm_token_bits,
                     hls::stream<bool>& strm_e_addr,
                     hls::stream<bool>& strm_e_dc) {
#pragma HLS INLINE OFF

    const int xsize = rect.xsize;
    const int ysize = rect.ysize;

    int cnt = 0;
    ap_uint<13> addr = 0;

    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < ysize; y++) {
            int x = 0;
            dct_t dc = strm_dc_residuals.read();
            while (x < xsize) {
#pragma HLS PIPELINE II = 1
                if (!rle || dc) { // nz must be encode and zero-flase is always better
                    if (cnt > 0) {
                        int symbol, nbits, bits;

                        hls_EncodeHybridVarLenUint(cnt - 1, &symbol, &nbits, &bits);
                        hls_Token_symb out_s;
                        hls_Token_bits out_t;
                        out_s.context = c;
                        out_s.symbol = hls_kRleSymStart + symbol;
                        out_t.nbits = nbits;
                        out_t.bits = bits;
                        addr = (c << 8) + hls_kRleSymStart + symbol;

                        strm_token_addr.write(addr);
                        strm_token_symb.write(out_s);
                        strm_token_bits.write(out_t);
                        strm_e_addr.write(false);
                        strm_e_dc.write(false);

                        _XF_IMAGE_PRINT("---write cnt token(%d,%d,%d,%d), run_bits=%.2x \n", c,
                                        hls_kRleSymStart + symbol, nbits, bits, cnt);

                        cnt = 0;
                    } else {
                        int symbol, nbits, bits;
                        hls_EncodeHybridVarLenInt(dc, &symbol, &nbits, &bits);
                        assert(symbol < hls_kRleSymStart);
                        hls_Token_symb out_s;
                        hls_Token_bits out_t;
                        out_s.context = c;
                        out_s.symbol = symbol;
                        out_t.nbits = nbits;
                        out_t.bits = bits;
                        addr = (c << 8) + symbol;

                        strm_token_addr.write(addr);
                        strm_token_symb.write(out_s);
                        strm_token_bits.write(out_t);
                        strm_e_addr.write(false);
                        strm_e_dc.write(false);

                        _XF_IMAGE_PRINT("---write token(%d,%d,%d,%d)\n", c, symbol, nbits, bits);

                        // update
                        if (x < xsize - 1) dc = strm_dc_residuals.read();
                        x++;
                    }

                } else {
                    if (x < xsize - 1) dc = strm_dc_residuals.read();
                    cnt++;
                    x++;
                }
            } // bx
        }     // by

        if (cnt > 0) {
            int symbol, nbits, bits;
            hls_EncodeHybridVarLenUint(cnt - 1, &symbol, &nbits, &bits);
            hls_Token_symb out_s;
            hls_Token_bits out_t;
            out_s.context = c;
            out_s.symbol = hls_kRleSymStart + symbol;
            out_t.nbits = nbits;
            out_t.bits = bits;
            addr = (c << 8) + hls_kRleSymStart + symbol;

            strm_token_addr.write(addr);
            strm_token_symb.write(out_s);
            strm_token_bits.write(out_t);
            strm_e_addr.write(false);
            strm_e_dc.write(false);

            _XF_IMAGE_PRINT("---write cnt token(%d,%d,%d,%d), run_bits=%.2x \n", c, hls_kRleSymStart + symbol, nbits,
                            bits, cnt);
            cnt = 0;
        } // per color
    }

    strm_e_addr.write(true);
    strm_e_dc.write(true);
}

void hls_encode_dc_top(const bool rle,
                       const hls_Rect rect_dc,
                       hls::stream<dct_t>& strm_dc_y1,
                       hls::stream<dct_t>& strm_dc_y2,
                       hls::stream<dct_t>& strm_dc_y3,

                       hls::stream<dct_t>& strm_dc_x,
                       hls::stream<dct_t>& strm_dc_b,

                       hls::stream<addr_t>& strm_token_addr,
                       hls::stream<hls_Token_symb>& strm_token_symb,
                       hls::stream<hls_Token_bits>& strm_token_bits,
                       hls::stream<bool>& strm_e_addr,
                       hls::stream<bool>& strm_e_dc) {
#pragma HLS INLINE OFF
#pragma HLS DATAFLOW
    // clang-format off
static hls::stream< dct_t >   strm_dc_residuals("dc_residuals");
#pragma HLS RESOURCE variable=strm_dc_residuals core=FIFO_LUTRAM
#pragma HLS STREAM   variable=strm_dc_residuals depth=32 dim=1
    // clang-format on

    hls_ShrinkDC_top(rect_dc, strm_dc_y1, strm_dc_y2, strm_dc_y3, strm_dc_x, strm_dc_b,

                     strm_dc_residuals);

    Tokenize_DC_top(false, rect_dc, strm_dc_residuals, strm_token_addr, strm_token_symb, strm_token_bits, strm_e_addr,
                    strm_e_dc);
}
