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

#include "kernel3/ac_tokenize.hpp"

// ------------------------------------------------------------
// XAcc_PredictFromTopAndLeft(cnt_nz, bx, is_top_row, cnt_nz_left, cnt_nz_abv,
// predicted_nz);
void XAcc_PredictFromTopAndLeft_nz(nzeros_t cnt_nz_here,
                                   uint32_t x,
                                   bool is_top_row,
                                   nzeros_t& cnt_nz_left,
                                   nzeros_t cnt_nz_abv[MAX_NUM_BLOCK88_W_TITLE],
                                   nzeros_t& predicted_nz) {
#pragma HLS INLINE

    nzeros_t default_val = 32;
    nzeros_t predicted_nzeros;
    // nzeros_t cnt_nz_here = cnt_nz.read();

    // read the regs and update
    if (x == 0 && is_top_row) {
        predicted_nzeros = default_val;
    } else if (x == 0) {
        predicted_nzeros = cnt_nz_abv[0];
    } else if (is_top_row) {
        predicted_nzeros = cnt_nz_left;
    } else {
        predicted_nzeros = (cnt_nz_abv[x] + cnt_nz_left + 1) / 2;
    }
    cnt_nz_left = cnt_nz_here;
    cnt_nz_abv[x] = cnt_nz_here;
    predicted_nz = predicted_nzeros;
}
// ------------------------------------------------------------

void XAcc_EncodeVarLenUint(uint16_t value, int& nbits, int& bits) {
#pragma HLS INLINE
    if (value == 0) {
        nbits = 0;
        bits = 0;
    } else {
        // int len = Log2FloorNonZero(value + 1);//because the __builtin_clz input
        // and return is a int
        uint8_t len = 31 ^ __builtin_clz(value + 1);
        nbits = len;
        bits = (value + 1) & ((1 << len) - 1);
    }
}

uint32_t NonZeroContext(uint32_t non_zeros, uint32_t block_ctx) {
    return hls_kOrderContexts * (non_zeros >> 1) + block_ctx;
}

// ------------------------------------------------------------

// ------------------------------------------------------------
// 1 coeffs per cycle in order
void hls_orderBlk(const hls_blksize rect,
                  const int32_t orders[3][64],
                  hls::stream<dct_t>& strm_coef_raster,
                  hls::stream<dct_t>& strm_coef_orderd // dct_t coef_blk[64]//
                  ) {
#pragma HLS INLINE OFF

#ifndef __SYNTHESIS__
    dct_t tmp_max = 0;
#endif

    bool ping = true;

    dct_t coef_blk[2][64];
#pragma HLS ARRAY_PARTITION variable = coef_blk dim = 1
#pragma HLS RESOURCE variable = coef_blk core = RAM_2P_BRAM

    assert(rect.ysize != 0);
    assert(rect.xsize != 0);
    const int nPingPong = rect.ysize * rect.xsize;
    for (int i = 0; i < BLKDIM; ++i) {     // 8
        for (int j = 0; j < BLKDIM; ++j) { // 8
#pragma HLS PIPELINE II = 1
            dct_t tmp = strm_coef_raster.read();
            coef_blk[ping][i * BLKDIM + j] = tmp;
        }
    }
    ping = !ping;

HLS_READ_BLK:
    for (int c = 0; c < 3; ++c) {
        for (int bx = 0; bx < (c == 2 ? (nPingPong - 1) : nPingPong); ++bx) {
            for (int i = 0; i < BLKDIM; ++i) {     // 8
                for (int j = 0; j < BLKDIM; ++j) { // 8
#pragma HLS PIPELINE II = 1
                    dct_t tmp = strm_coef_raster.read();
                    coef_blk[ping][i * BLKDIM + j] = tmp;

                    int zig = orders[c][i * BLKDIM + j]; // no dataflow
                    strm_coef_orderd.write((zig ? coef_blk[!ping][zig] : 0));
                }
            }
            ping = !ping;
        }
    } // end pingpang

    for (int i = 0; i < BLKDIM; ++i) {     // 8
        for (int j = 0; j < BLKDIM; ++j) { // 8
#pragma HLS PIPELINE II = 1
            int zig = orders[2][i * BLKDIM + j];
            strm_coef_orderd.write((zig ? coef_blk[!ping][zig] : 0));
        }
    }

    _XF_IMAGE_PRINT("--dct coeff max=%d - Tokenize\n", tmp_max);
}

// ------------------------------------------------------------
// goal: one NZs / 1 clock
/**
 * @brief count the NZs of DCT coeff in AC by block8x8 scanning.read the 64
 * coeff but write out 63 ac
 *
 * @param xsize_blocks num blocks per line of image.
 * @param coef coeffcients of one block line of DCT.
 * @param cnt_nz counter of non-zeros
 * XAcc_count_ac_nz(xsize_blocks, strm_coef, cnt_nz, lb_nz_write);
 */
void hls_count_ac_nz(hls_blksize rect,
                     hls::stream<dct_t>& strm_coef,

                     hls::stream<dct_t> strm_o_coef[64],
                     hls::stream<nzeros_t>& cnt_nz,
                     hls::stream<nzeros_t>& cnt_nz2) {
#pragma HLS INLINE OFF
    // counts
    nzeros_t reg_nz_cnt = 0;

    //    for (int c = 0; c < 3; ++c) {
    //    	for (int by = 0; by < rect.ysize; by++) {
    for (int bx = 0; bx < 3 * rect.xsize * rect.ysize; ++bx) { // while
        for (int i = 0; i < BLOCK_SIZE; i++) {
// for (int j = 0; j < BLKDIM; j++) {
#pragma HLS PIPELINE II = 1
            dct_t tmp = strm_coef.read();
            reg_nz_cnt += (tmp != 0);
            strm_o_coef[i].write(tmp);

            // write
            if (i == BLOCK_SIZE - 1) {
                cnt_nz.write(reg_nz_cnt);
                cnt_nz2.write(reg_nz_cnt);
                _XF_IMAGE_PRINT("%d,", (int)reg_nz_cnt);
                reg_nz_cnt = 0;
            }
        }
    } // bx
}

// ------------------------------------------------------------
// goal: one NZs / 1 clock
/**
 * @brief count the NZs of DCT coeff in AC by block8x8 scanning.read the 64
 * coeff but write out 63 ac
 *
 * @param xsize_blocks num blocks per line of image.
 * @param coef coeffcients of one block line of DCT.
 * @param cnt_nz counter of non-zeros
 * XAcc_count_ac_nz(xsize_blocks, strm_coef, cnt_nz, lb_nz_write);
 */
void hls_CountAcNz(hls_blksize rect,
                   hls::stream<dct_t>& strm_coef,

                   hls::stream<dct_t>& strm_o_coef,
                   hls::stream<nzeros_t>& cnt_nz,
                   hls::stream<nzeros_t>& cnt_nz2) {
#pragma HLS INLINE OFF
    // counts
    nzeros_t reg_nz_cnt = 0;

    //    for (int c = 0; c < 3; ++c) {
    //    	for (int by = 0; by < rect.ysize; by++) {
    for (int bx = 0; bx < 3 * rect.xsize * rect.ysize; ++bx) { // while
        for (int i = 0; i < BLOCK_SIZE; i++) {
// for (int j = 0; j < BLKDIM; j++) {
#pragma HLS PIPELINE II = 1
            dct_t tmp = strm_coef.read();
            reg_nz_cnt += (tmp != 0);
            strm_o_coef.write(tmp);

            // write
            if (i == BLOCK_SIZE - 1) {
                cnt_nz.write(reg_nz_cnt);
                cnt_nz2.write(reg_nz_cnt);
                _XF_IMAGE_PRINT("%d,", (int)reg_nz_cnt);
                reg_nz_cnt = 0;
            }
        }
    } // bx
}

// ------------------------------------------------------------
void hls_tokenize_nz(hls_blksize rect,
                     hls::stream<nzeros_t>& cnt_nz,

                     hls::stream<hls_Token>& strm_nz_token) {
#pragma HLS INLINE OFF
    nzeros_t cnt_nz_abv[MAX_NUM_BLOCK88_W_TITLE];
#pragma HLS ARRAY_PARTITION variable = cnt_nz_abv complete dim = 1
    nzeros_t cnt_nz_left;
    nzeros_t predicted_nz;

    hls_Token ac_token;
    hls_Token out_token;
    int total_token = 0; // init

    int bx = 0;

    for (int c = 0; c < 3; ++c) {
        for (int by = 0; by < rect.ysize;) { // while
#pragma HLS PIPELINE II = 1

            bool is_top_row = (by == 0);
            // for (int bx = 0; bx < rect.xsize; ++bx) {

            nzeros_t cnt_nz_here = cnt_nz.read();
            XAcc_PredictFromTopAndLeft_nz(cnt_nz_here, bx, is_top_row, cnt_nz_left, cnt_nz_abv, predicted_nz);

            int32_t predicted_nzeros = predicted_nz;

            ac_token.context = NonZeroContext(predicted_nzeros, c);
            ac_token.symbol = cnt_nz_here;

            /// write token (context symbol nbits bits)
            out_token.context = ac_token.context;
            out_token.symbol = ac_token.symbol;
            out_token.nbits = 0;
            out_token.bits = 0;

            strm_nz_token.write(out_token);

#ifndef __SYNTHESIS__
            total_token++;
#endif
            if (bx == rect.xsize - 1) {
                by++;
                bx = 0;
            } else {
                bx++;
            }

            //}
        } // end by
    }     // end tile
}

//// ------------------------------------------------------------
// void hls_orderblk_tokennz(const int32_t orders[3][64], // color dct_band
//                               const hls_blksize rect,
//							   //hls::stream<hls_blksize>
//&rect,
//                               hls::stream<dct_t> &strm_coef_raster,
//
//                               hls::stream<nzeros_t>& cnt_nz,
//                               hls::stream<hls_Token>& strm_nz_token,
//                               hls::stream<dct_t> strm_coef_orderd[64]
//                               // hls::stream< hls_Tokenbit >& strm_token_bit
//                               ) {
//#pragma HLS INLINE
//#pragma HLS DATAFLOW
//
//    static hls::stream<dct_t> strm_coef_ord("strm_coef_ord");
////#pragma HLS RESOURCE variable = strm_coef_ord core = FIFO_LUTRAM
//#pragma HLS STREAM variable = strm_coef_ord depth = 1024
//
//    // for predict
//    hls::stream<nzeros_t> cnt_nz2("strm_cnt_nz2");
//#pragma HLS RESOURCE variable = cnt_nz2 core = FIFO_LUTRAM
//#pragma HLS STREAM variable = cnt_nz2 depth = 32
//
//    _XF_IMAGE_PRINT("\n --2 tmp_num_nzeros begin - Tokenize \n");
//
//
//    hls_order_blk(rect, orders, strm_coef_raster, strm_coef_ord);
//
//    hls_count_ac_nz(rect, strm_coef_ord, strm_coef_orderd, cnt_nz, cnt_nz2);
//
//    hls_tokenize_nz(rect, cnt_nz2,  // ac_static_context_map,
//                         strm_nz_token); //, strm_coef_orderd
//}

// ------------------------------------------------------------
// goal: one token / clock
/**
 * @brief order_zig_zag the coeff in AC and encode to the
 * token(cxt,sym,nbits,bits).
 * @brief pick out the non-zero coeff, read the  63 ac and encode
 *
 * @param xsize_blocks num blocks per line of image.
 * @param coef coeffcients of one block line of DCT.
 * @param cnt_nz counter of non-zeros
 * XAcc_count_ac_nz(xsize_blocks, strm_coef, cnt_nz, lb_nz_write);
 */

void tokenize_blk_syn_test_org(

    const hls_blksize rect,
    // hls::stream<hls_blksize> &rect,
    hls::stream<dct_t> strm_coef_orderd[64],
    hls::stream<nzeros_t>& cnt_nz,

    hls::stream<nzeros_t>& strm_cnt_ac,
    hls::stream<hls_Token>& strm_ac_token

    ) {
#pragma HLS INLINE OFF

    nzeros_t reg_nz_cnt; // reg of last cnt
    nzeros_t cnt;        // cnt for nz--

    ap_uint<8> kSkipAndBits;
    hls_Token ac_token;

    nzeros_t len = 0;      // cnt for ac_tokens
    nzeros_t run = 0;      // cnt for run
    nzeros_t bpos = 1;     // cnt for pos in the block
    nzeros_t last_pos = 0; // reg of last pos
    // total_token = 4096;//init for nz tokens

    dct_t blk_orderd[64];
#pragma HLS ARRAY_PARTITION variable = blk_orderd complete dim = 1
#pragma HLS ARRAY_PARTITION variable = hls_kCoeffNumNonzeroContext complete dim = 1
#pragma HLS ARRAY_PARTITION variable = hls_kCoeffFreqContext complete dim = 1
    int bx = 0;

PICK_OUT_NZ_LOOP:
    for (int c = 0; c < 3; ++c) {
        // 1. init
        ap_uint<9> histo_offset = 96 + 105 * c; // ZeroDensityContextsOffset(c);

        if (rect.xsize || rect.ysize) {
            reg_nz_cnt = cnt_nz.read();
            cnt = reg_nz_cnt;

            for (int j = 0; j < 64; j++) {
#pragma HLS UNROLL
                blk_orderd[j] = strm_coef_orderd[j].read();
            }
        }

        for (int by = 0; by < rect.ysize;) { //++by// cnt for block// while(bx < rect.xsize){
#pragma HLS PIPELINE II = 1

            // 2. write out
            if (cnt > 0) {
                if (blk_orderd[bpos] != 0) { // find nz and encode nz token

                    int nbits, bits;
                    //  EncodeVarLenInt((int32_t)blk_orderd[bpos], &nbits,
                    //  &bits);//(-1,1,0)
                    XAcc_EncodeVarLenUint(hls_PackSigned((int16_t)blk_orderd[bpos]), nbits, bits);

                    kSkipAndBits(7, 4) = run;
                    kSkipAndBits(3, 0) = nbits;

                    //                    ac_token.context =
                    //                        histo_offset +
                    //                        hls_kCoeffNumNonzeroContext[cnt] +
                    //                        hls_kCoeffFreqContext[last_pos];
                    ac_token.symbol = kSkipAndBits;
                    ac_token.nbits = nbits;
                    ac_token.bits = bits;

                    len++;
                    strm_ac_token.write(ac_token);

                    run = 0;
                    last_pos = bpos;
                    cnt--;

                } else {             // find 0 and run++
                    if (run == 15) { // find ff and encode ff token
                        ap_uint<4> nbits = 0;
                        ap_uint<4> skip = 15;
                        kSkipAndBits(7, 4) = skip;
                        kSkipAndBits(3, 0) = nbits;

                        //                        ac_token.context =
                        //                            histo_offset +
                        //                            hls_kCoeffNumNonzeroContext[cnt] +
                        //                            hls_kCoeffFreqContext[last_pos];
                        ac_token.symbol = kSkipAndBits;
                        ac_token.nbits = nbits;
                        ac_token.bits = 0;

                        len++;
                        strm_ac_token.write(ac_token);

                        run = 0;
                        last_pos = bpos;
                    } else {
                        run++;
                    }
                }

                bpos++;
            }

            // 3. read new block
            if (!cnt) {
                strm_cnt_ac.write(len);
                len = 0;
                run = 0;
                last_pos = 0;
                bpos = 1;

                if (bx == rect.xsize - 1) {
                    if (by < rect.ysize - 1) {
                        reg_nz_cnt = cnt_nz.read();
                        cnt = reg_nz_cnt;

                        for (int j = 0; j < 64; j++) {
#pragma HLS UNROLL
                            blk_orderd[j] = strm_coef_orderd[j].read();
                        }
                    }

                    bx = 0;
                    by++; // break here
                } else {
                    reg_nz_cnt = cnt_nz.read();
                    cnt = reg_nz_cnt;

                    for (int j = 0; j < 64; j++) {
#pragma HLS UNROLL
                        blk_orderd[j] = strm_coef_orderd[j].read();
                    }

                    bx++;
                }
            }
        }
    }
} // end tile

void tokenize_blk_syn_test(

    const hls_blksize rect,
    // hls::stream<hls_blksize> &rect,
    hls::stream<dct_t> strm_coef_orderd[64],
    hls::stream<nzeros_t>& cnt_nz,

    hls::stream<nzeros_t>& strm_cnt_ac,
    hls::stream<ap_uint<9> >& strm_histo_offset,
    hls::stream<nzeros_t>& strm_cnt_lookup,
    hls::stream<nzeros_t>& strm_last_pos,
    // hls::stream<bool>& strm_e,
    hls::stream<hls_Token>& strm_ac_token

    ) {
#pragma HLS INLINE OFF

    nzeros_t reg_nz_cnt; // reg of last cnt
    nzeros_t cnt;        // cnt for nz--

    ap_uint<8> kSkipAndBits;
    hls_Token ac_token;

    nzeros_t len = 0;      // cnt for ac_tokens
    nzeros_t run = 0;      // cnt for run
    nzeros_t bpos = 1;     // cnt for pos in the block
    nzeros_t last_pos = 0; // reg of last pos
    // total_token = 4096;//init for nz tokens

    dct_t blk_orderd[64];
#pragma HLS ARRAY_PARTITION variable = blk_orderd complete dim = 1
#pragma HLS ARRAY_PARTITION variable = hls_kCoeffNumNonzeroContext complete dim = 1
#pragma HLS ARRAY_PARTITION variable = hls_kCoeffFreqContext complete dim = 1
    int bx = 0;

PICK_OUT_NZ_LOOP:
    for (int c = 0; c < 3; ++c) {
        // 1. init
        ap_uint<9> histo_offset = 96 + 105 * c; // ZeroDensityContextsOffset(c);

        if (rect.xsize || rect.ysize) {
            reg_nz_cnt = cnt_nz.read();
            cnt = reg_nz_cnt;

            for (int j = 0; j < 64; j++) {
#pragma HLS UNROLL
                blk_orderd[j] = strm_coef_orderd[j].read();
            }
        }

        for (int by = 0; by < rect.ysize;) { //++by// cnt for block// while(bx < rect.xsize){
#pragma HLS PIPELINE II = 1

            // 2. write out
            if (cnt > 0) {
                if (blk_orderd[bpos] != 0) { // find nz and encode nz token

                    int nbits, bits;
                    //  EncodeVarLenInt((int32_t)blk_orderd[bpos], &nbits,
                    //  &bits);//(-1,1,0)
                    XAcc_EncodeVarLenUint(hls_PackSigned((int16_t)blk_orderd[bpos]), nbits, bits);

                    kSkipAndBits(7, 4) = run;
                    kSkipAndBits(3, 0) = nbits;

                    //                    ac_token.context =
                    //                        histo_offset +
                    //                        hls_kCoeffNumNonzeroContext[cnt] +
                    //                        hls_kCoeffFreqContext[last_pos];
                    ac_token.symbol = kSkipAndBits;
                    ac_token.nbits = nbits;
                    ac_token.bits = bits;

                    len++;
                    strm_ac_token.write(ac_token);
                    strm_histo_offset.write(histo_offset);
                    strm_cnt_lookup.write(cnt);
                    strm_last_pos.write(last_pos);
                    // strm_e.write(false);

                    run = 0;
                    last_pos = bpos;
                    cnt--;

                } else {             // find 0 and run++
                    if (run == 15) { // find ff and encode ff token
                        ap_uint<4> nbits = 0;
                        ap_uint<4> skip = 15;
                        kSkipAndBits(7, 4) = skip;
                        kSkipAndBits(3, 0) = nbits;

                        //                        ac_token.context =
                        //                            histo_offset +
                        //                            hls_kCoeffNumNonzeroContext[cnt] +
                        //                            hls_kCoeffFreqContext[last_pos];
                        ac_token.symbol = kSkipAndBits;
                        ac_token.nbits = nbits;
                        ac_token.bits = 0;

                        len++;
                        strm_ac_token.write(ac_token);
                        strm_histo_offset.write(histo_offset);
                        strm_cnt_lookup.write(cnt);
                        strm_last_pos.write(last_pos);
                        // strm_e.write(false);

                        run = 0;
                        last_pos = bpos;
                    } else {
                        run++;
                    }
                }

                bpos++;
            }

            // 3. read new block
            if (!cnt) {
                strm_cnt_ac.write(len);
                len = 0;
                run = 0;
                last_pos = 0;
                bpos = 1;

                if (bx == rect.xsize - 1) {
                    if (by < rect.ysize - 1) {
                        reg_nz_cnt = cnt_nz.read();
                        cnt = reg_nz_cnt;

                        for (int j = 0; j < 64; j++) {
#pragma HLS UNROLL
                            blk_orderd[j] = strm_coef_orderd[j].read();
                        }
                    }

                    bx = 0;
                    by++; // break here
                } else {
                    reg_nz_cnt = cnt_nz.read();
                    cnt = reg_nz_cnt;

                    for (int j = 0; j < 64; j++) {
#pragma HLS UNROLL
                        blk_orderd[j] = strm_coef_orderd[j].read();
                    }

                    bx++;
                }
            }
        }
        // strm_e.write(true);
    }

} // end tile

// ------------------------------------------------------------
void hls_TokenizeBlk(

    const hls_blksize rect,
    hls::stream<dct_t>& strm_coef_orderd,
    hls::stream<nzeros_t>& cnt_nz,

    hls::stream<nzeros_t>& strm_cnt_ac,
    hls::stream<ap_uint<9> >& strm_histo_offset,
    hls::stream<nzeros_t>& strm_cnt_lookup,
    hls::stream<nzeros_t>& strm_last_pos,
    // hls::stream<bool>& strm_e,
    hls::stream<hls_Token>& strm_ac_token

    ) {
#pragma HLS INLINE OFF

    nzeros_t reg_nz_cnt; // reg of last cnt
    nzeros_t cnt;        // cnt for nz--

    ap_uint<8> kSkipAndBits;
    hls_Token ac_token;

    nzeros_t len = 0;      // cnt for ac_tokens
    nzeros_t run = 0;      // cnt for run
    nzeros_t bpos = 0;     // cnt for pos in the block
    nzeros_t last_pos = 0; // reg of last pos
                           // total_token = 4096;//init for nz tokens

#pragma HLS ARRAY_PARTITION variable = hls_kCoeffNumNonzeroContext complete dim = 1
#pragma HLS ARRAY_PARTITION variable = hls_kCoeffFreqContext complete dim = 1

    dct_t ac;

PICK_OUT_NZ_LOOP:
    for (int c = 0; c < 3; ++c) {
        for (int bx = 0; bx < rect.xsize * rect.ysize * BLOCK_SIZE; bx++) { // while
#pragma HLS PIPELINE II = 1

            if (bpos == 0) { // pos == 0 no token
                cnt = cnt_nz.read();
                dct_t tmp = strm_coef_orderd.read();
            } else {
                ac = strm_coef_orderd.read();
                ap_uint<9> histo_offset = 96 + 105 * c; // ZeroDensityContextsOffset(c);

                if (cnt > 0) {
                    if (ac != 0) { // find nz and encode nz token

                        int nbits, bits;
                        //  EncodeVarLenInt((int32_t)blk_orderd[bpos], &nbits,
                        //  &bits);//(-1,1,0)
                        XAcc_EncodeVarLenUint(hls_PackSigned((int16_t)ac), nbits, bits);

                        kSkipAndBits(7, 4) = run;
                        kSkipAndBits(3, 0) = nbits;

                        //                    ac_token.context =
                        //                        histo_offset +
                        //                        hls_kCoeffNumNonzeroContext[cnt] +
                        //                        hls_kCoeffFreqContext[last_pos];
                        ac_token.symbol = kSkipAndBits;
                        ac_token.nbits = nbits;
                        ac_token.bits = bits;

                        len++;
                        strm_ac_token.write(ac_token);
                        strm_histo_offset.write(histo_offset);
                        strm_cnt_lookup.write(cnt);
                        strm_last_pos.write(last_pos);
                        // strm_e.write(false);

                        run = 0;
                        last_pos = bpos;
                        cnt--;

                    } else {             // find 0 and run++
                        if (run == 15) { // find ff and encode ff token
                            ap_uint<4> nbits = 0;
                            ap_uint<4> skip = 15;
                            kSkipAndBits(7, 4) = skip;
                            kSkipAndBits(3, 0) = nbits;

                            //                        ac_token.context =
                            //                            histo_offset +
                            //                            hls_kCoeffNumNonzeroContext[cnt] +
                            //                            hls_kCoeffFreqContext[last_pos];
                            ac_token.symbol = kSkipAndBits;
                            ac_token.nbits = nbits;
                            ac_token.bits = 0;

                            len++;
                            strm_ac_token.write(ac_token);
                            strm_histo_offset.write(histo_offset);
                            strm_cnt_lookup.write(cnt);
                            strm_last_pos.write(last_pos);
                            // strm_e.write(false);

                            run = 0;
                            last_pos = bpos;
                        } else {
                            run++;
                        }
                    }
                }
            } // end ac

            if (bpos == BLOCK_SIZE - 1) {
                strm_cnt_ac.write(len);
                run = 0;
                last_pos = 0;
                len = 0;
                bpos = 0;
            } else {
                ++bpos;
            }

        } // end inter loop
    }

} // end tile

// ------------------------------------------------------------
void tokenize_lookup_table1(hls::stream<ap_uint<9> >& strm_histo_offset,
                            hls::stream<nzeros_t>& strm_cnt_lookup,
                            hls::stream<nzeros_t>& strm_last_pos,
                            hls::stream<hls_Token>& strm_ac_token,
                            hls::stream<bool>& strm_e,

                            hls::stream<hls_Token>& strm_ac_token_out

                            ) {
#pragma HLS INLINE OFF

    for (int c = 0; c < 3; ++c) {
        bool e = strm_e.read();
        while (!e) {
#pragma HLS PIPELINE II = 1
            ap_uint<9> histo_offset = strm_histo_offset.read();
            nzeros_t cnt = strm_cnt_lookup.read();
            nzeros_t last_pos = strm_last_pos.read();
            hls_Token ac_token = strm_ac_token.read();
            e = strm_e.read();

            ac_token.context = histo_offset + hls_kCoeffNumNonzeroContext[cnt] + hls_kCoeffFreqContext[last_pos];

            strm_ac_token_out.write(ac_token);
        }
    }
}

// ------------------------------------------------------------
void tokenize_lookup_table(const hls_blksize rect,
                           hls::stream<nzeros_t>& strm_cnt_ac,
                           hls::stream<ap_uint<9> >& strm_histo_offset,
                           hls::stream<nzeros_t>& strm_cnt_lookup,
                           hls::stream<nzeros_t>& strm_last_pos,
                           hls::stream<hls_Token>& strm_ac_token,

                           hls::stream<nzeros_t>& strm_cnt_ac_out,
                           hls::stream<hls_Token>& strm_ac_token_out

                           ) {
#pragma HLS INLINE OFF

    nzeros_t len = 0;
    for (int c = 0; c < 3; ++c) {
        for (int by = 0; by < rect.ysize; ++by) {
            // for (int bx = 0; bx < xsize_blocks; ++bx) {
            for (int bx = 0; bx < rect.xsize + 1;) {
#pragma HLS PIPELINE II = 1

                if (len == 0) { // no tockens
                    if (bx < rect.xsize) {
                        len = strm_cnt_ac.read();
                        strm_cnt_ac_out.write(len);
                    }

                    bx++; // break loop from here

                } else {
                    hls_Token ac_token = strm_ac_token.read();
                    ap_uint<9> histo_offset = strm_histo_offset.read();
                    nzeros_t cnt = strm_cnt_lookup.read();
                    nzeros_t last_pos = strm_last_pos.read();

                    ac_token.context =
                        histo_offset + hls_kCoeffNumNonzeroContext[cnt] + hls_kCoeffFreqContext[last_pos];

                    strm_ac_token_out.write(ac_token);

                    len--;
                }

            } // end bx
        }
    } // end tile
}

// ------------------------------------------------------------
void tokenize_blk(const hls_blksize rect,
                  hls::stream<dct_t> strm_coef_orderd[64],
                  hls::stream<nzeros_t>& cnt_nz,

                  hls::stream<nzeros_t>& strm_cnt_ac,
                  hls::stream<hls_Token>& strm_ac_token

                  ) {
    nzeros_t reg_nz_cnt; // reg of last cnt
    nzeros_t cnt;        // cnt for nz--

    ap_uint<8> kSkipAndBits;
    hls_Token ac_token;

    nzeros_t len = 0;      // cnt for ac_tokens
    nzeros_t run = 0;      // cnt for run
    nzeros_t bpos = 1;     // cnt for pos in the block
    nzeros_t last_pos = 0; // reg of last pos
    // total_token = 4096;//init for nz tokens

    dct_t blk_orderd[64];
#pragma HLS ARRAY_PARTITION variable = blk_orderd complete dim = 1
#pragma HLS ARRAY_PARTITION variable = hls_kCoeffNumNonzeroContext complete dim = 1
#pragma HLS ARRAY_PARTITION variable = hls_kCoeffFreqContext complete dim = 1
    // 1. init

    for (int c = 0; c < 3; ++c) {
        for (int by = 0; by < rect.ysize; ++by) {
            if (rect.xsize) {
                reg_nz_cnt = cnt_nz.read();
                // strm_cnt_nz.write(reg_nz_cnt);
                cnt = reg_nz_cnt;

                for (int j = 0; j < 64; j++) {
#pragma HLS UNROLL
                    blk_orderd[j] = strm_coef_orderd[j].read();
                }
            }

            ap_uint<9> histo_offset = 96 + 105 * c; // ZeroDensityContextsOffset(c);
            int bx = 0;                             // cnt for block
        PICK_OUT_NZ_LOOP:
            while (bx < rect.xsize) {
#pragma HLS PIPELINE II = 2

                // 2. write out
                if (cnt > 0) {
                    if (blk_orderd[bpos] != 0) { // find nz and encode nz token

                        int nbits, bits;
                        XAcc_EncodeVarLenUint(hls_PackSigned((int16_t)blk_orderd[bpos]), nbits, bits);

                        kSkipAndBits(7, 4) = run;
                        kSkipAndBits(3, 0) = nbits;

                        ac_token.context =
                            histo_offset + hls_kCoeffNumNonzeroContext[cnt] + hls_kCoeffFreqContext[last_pos];
                        ac_token.symbol = kSkipAndBits;
                        ac_token.nbits = nbits;
                        ac_token.bits = bits;

                        len++;
                        strm_ac_token.write(ac_token);

                        run = 0;
                        last_pos = bpos;
                        cnt--;

                    } else {             // find 0 and run++
                        if (run == 15) { // find ff and encode ff token
                            ap_uint<4> nbits = 0;
                            ap_uint<4> skip = 15;
                            kSkipAndBits(7, 4) = skip;
                            kSkipAndBits(3, 0) = nbits;

                            ac_token.context =
                                histo_offset + hls_kCoeffNumNonzeroContext[cnt] + hls_kCoeffFreqContext[last_pos];
                            ac_token.symbol = kSkipAndBits;
                            ac_token.nbits = nbits;
                            ac_token.bits = 0;

                            len++;
                            strm_ac_token.write(ac_token);

                            run = 0;
                            last_pos = bpos;
                        } else {
                            run++;
                        }
                    }

                    bpos++;
                }

                // 3. read new block
                if (!cnt) {
                    if (bx < rect.xsize - 1) {
                        reg_nz_cnt = cnt_nz.read();
                        cnt = reg_nz_cnt;

                        for (int j = 0; j < 64; j++) {
#pragma HLS UNROLL
                            blk_orderd[j] = strm_coef_orderd[j].read();
                        }
                    }

                    strm_cnt_ac.write(len);
                    len = 0;
                    run = 0;
                    last_pos = 0;
                    bpos = 1;
                    bx++; // break here
                }
            }
        }
    } // end tile
}
// ------------------------------------------------------------

void collect_token_syn(const hls_blksize rect,
                       // hls::stream<hls_blksize> &rect,
                       uint8_t ac_static_context_map[hls_kNumContexts],
                       bool e_tile,

                       hls::stream<hls_Token>& strm_nz_token,
                       hls::stream<nzeros_t>& strm_cnt_ac,
                       hls::stream<hls_Token>& strm_ac_token,

                       hls::stream<ap_uint<13> >& strm_token_addr,
                       hls::stream<bool>& strm_e_addr,
                       hls::stream<bool>& strm_e_token,
                       hls::stream<hls_Token_symb>& strm_token_symb,
                       hls::stream<hls_Token_bits>& strm_token_bits) {
#pragma HLS INLINE OFF

    hls_Token ac_token;
    nzeros_t len = 0;

#pragma HLS ARRAY_PARTITION variable = hls_kSkipAndBitsSymbol complete dim = 1

    for (int c = 0; c < 3; ++c) {
        for (int by = 0; by < rect.ysize; ++by) {
            // for (int bx = 0; bx < xsize_blocks; ++bx) {

            for (int bx = 0; bx < rect.xsize + 1;) {
// while(bx < rect.xsize+1 ){
#pragma HLS PIPELINE II = 1

                if (len == 0) { // no tockens

                    if (bx < rect.xsize) {
                        // nzeros_t reg_cnt_nz = strm_cnt_nz.read();// to be remove,
                        // non-used
                        len = strm_cnt_ac.read();
                        ac_token = strm_nz_token.read();
                        // Token out_token(0,0,0,0);

                        ap_uint<13> ac_token_addr =
                            ((uint16_t)ac_static_context_map[ac_token.context] << 8) + ac_token.symbol;
                        strm_token_addr.write(ac_token_addr);
                        strm_e_addr.write(false);
                        strm_e_token.write(false);

                        hls_Token_symb token_symb;
                        hls_Token_bits token_bits;
                        token_symb.context = ac_token.context;
                        token_symb.symbol = ac_token.symbol;
                        token_bits.bits = ac_token.bits;
                        token_bits.nbits = ac_token.nbits;

                        strm_token_symb.write(token_symb);
                        strm_token_bits.write(token_bits);
                        _XF_IMAGE_PRINT("---write token(%d,%d,0,0,%d) \n", (int)(ac_token.context),
                                        (int)(ac_token.symbol), (int)ac_token_addr.V.VAL);
                        //_XF_IMAGE_PRINT("---len=%d \n",len);
                    }

                    bx++; // break loop from here

                } else {
                    ac_token = strm_ac_token.read();
                    // addr_c = ac_token.addr;
                    ap_uint<13> ac_token_addr = ((uint16_t)ac_static_context_map[ac_token.context] << 8) +
                                                hls_kSkipAndBitsSymbol[ac_token.symbol];
                    strm_token_addr.write(ac_token_addr);
                    strm_e_addr.write(false);
                    strm_e_token.write(false);

                    hls_Token_symb token_symb;
                    hls_Token_bits token_bits;
                    token_symb.context = ac_token.context;
                    token_symb.symbol = hls_kSkipAndBitsSymbol[ac_token.symbol]; // ac_token.symbol;
                    token_bits.bits = ac_token.bits;
                    token_bits.nbits = ac_token.nbits;
                    strm_token_symb.write(token_symb);
                    strm_token_bits.write(token_bits);

                    len--;
                    _XF_IMAGE_PRINT("---write token(%d,%d,%d,%d,%d), skip_bits=\n", ac_token.context,
                                    hls_kSkipAndBitsSymbol[ac_token.symbol], ac_token.nbits, ac_token.bits,
                                    (int)ac_token_addr.V.VAL);
                }

            } // end bx
        }
    } // end tile

    if (e_tile) {
        strm_e_addr.write(true);
        strm_e_token.write(true);
    }
}
//
////----------------------------------------------------------
// void hls_read_config(hls::stream<hls_Rect>& strm_rect,
//                     hls::stream<bool>& strm_e_tile,
//					 hls::stream<hls_blksize>& rect_a,
//					 hls::stream<hls_blksize>& rect_b,
//					 hls::stream<hls_blksize>& rect_c,
//					 hls::stream<bool>& rect_e_tile) {
//#pragma HLS INLINE OFF
//    bool e_tile = strm_e_tile.read();
//
//    hls_Rect tmp = strm_rect.read();
//    hls_blksize tmp_out;
//    tmp_out.xsize = tmp.xsize;
//    tmp_out.ysize = tmp.ysize;
//    rect_a.write(tmp_out);
//    rect_b.write(tmp_out);
//    rect_c.write(tmp_out);
//    rect_e_tile.write(e_tile);
//
//}

//----------------------------------------------------------
void hls_read_config(hls::stream<hls_blksize>& strm_rect,
                     hls::stream<bool>& strm_e_tile,
                     hls_blksize& rect_a,
                     hls_blksize& rect_b,
                     hls_blksize& rect_c,
                     hls_blksize& rect_d,
                     hls_blksize& rect_e,
                     bool& rect_e_tile) {
#pragma HLS INLINE OFF
#pragma HLS PIPELINE II = 1
    bool e_tile = strm_e_tile.read();

    hls_blksize tmp = strm_rect.read();
    hls_blksize tmp_out;
    tmp_out.xsize = tmp.xsize;
    tmp_out.ysize = tmp.ysize;
    rect_a = (tmp_out);
    rect_b = (tmp_out);
    rect_c = (tmp_out);
    rect_d = (tmp_out);
    rect_e = (tmp_out);
    rect_e_tile = (e_tile);
}

//----------------------------------------------------------
void hls_tokenize_AC_tile_top(const int32_t orders[3][64], // color dct_band
                              hls::stream<hls_blksize>& strm_rect,
                              hls::stream<bool>& strm_e_tile,
                              // hls::stream<dct_t> strm_coef_raster[8],
                              hls::stream<dct_t>& strm_coef_raster,
                              uint8_t ac_static_context_map[hls_kNumContexts],
                              // hls::stream< hls_blksize > &strm_blk_size,

                              hls::stream<ap_uint<13> >& strm_token_addr,
                              hls::stream<hls_Token_symb>& strm_token_symb,
                              hls::stream<hls_Token_bits>& strm_token_bits,
                              hls::stream<bool>& strm_e_addr,
                              hls::stream<bool>& strm_e_token) {
#pragma HLS INLINE OFF
#pragma HLS DATAFLOW

    // clang-format off
	// strm_cnt_ac output every block with  1, 0, 0, 0, 1...
	// strm_ac_token2 output only there is a ac_tocken
	  static hls::stream< nzeros_t > cnt_nz("cnt_nz");
#pragma HLS RESOURCE  	  variable = cnt_nz core = FIFO_LUTRAM
#pragma HLS STREAM    	  variable = cnt_nz depth = 32
	  static hls::stream< hls_Token> strm_nz_token;
#pragma HLS DATA_PACK 	  variable = strm_nz_token
#pragma HLS RESOURCE  	  variable = strm_nz_token core = FIFO_LUTRAM
#pragma HLS STREAM    	  variable = strm_nz_token depth = 1024
//	  static hls::stream< dct_t >    strm_coef_orderd[64];
//#pragma HLS RESOURCE  	  variable = strm_coef_orderd core = FIFO_LUTRAM
//#pragma HLS ARRAY_PARTITION variable=strm_coef_orderd complete
//#pragma HLS STREAM    	  variable = strm_coef_orderd depth = 16

	  static hls::stream< dct_t >    strm_coef_orderd;
#pragma HLS RESOURCE  	  variable = strm_coef_orderd core = FIFO_LUTRAM
#pragma HLS STREAM    	  variable = strm_coef_orderd depth = 128

	  static hls::stream< nzeros_t > strm_cnt_ac("len_token");
#pragma HLS RESOURCE  	  variable = strm_cnt_ac core = FIFO_LUTRAM
#pragma HLS STREAM    	  variable = strm_cnt_ac depth = 64
	  static hls::stream< nzeros_t > strm_cnt_ac2("len_token2");
#pragma HLS RESOURCE  	  variable = strm_cnt_ac2 core = FIFO_LUTRAM
#pragma HLS STREAM    	  variable = strm_cnt_ac2 depth = 64
	  static hls::stream< hls_Token> strm_ac_token("ac_token");
#pragma HLS DATA_PACK 	  variable = strm_ac_token
#pragma HLS RESOURCE  	  variable = strm_ac_token core = FIFO_LUTRAM
#pragma HLS STREAM    	  variable = strm_ac_token depth = 64

  	hls::stream<ap_uint<9> > 		 strm_histo_offset;
#pragma HLS STREAM    	  variable = strm_histo_offset depth = 64
  	hls::stream<nzeros_t> 			 strm_cnt_lookup;
#pragma HLS STREAM    	  variable = strm_cnt_lookup depth = 64
  	hls::stream<nzeros_t> 			 strm_last_pos;
#pragma HLS STREAM    	  variable = strm_last_pos depth = 64
  	hls::stream<bool> 				 strm_e;
#pragma HLS STREAM    	  variable = strm_e depth = 64
  	hls::stream<hls_Token> 			 strm_ac_token2;
#pragma HLS STREAM    	  variable = strm_ac_token2 depth = 64
    // clang-format on
    hls_blksize rect_a;
    hls_blksize rect_b;
    hls_blksize rect_c;
    hls_blksize rect_d;
    hls_blksize rect_e;
    bool rect_e_tile;

    hls_read_config(strm_rect, strm_e_tile, rect_a, rect_b, rect_c, rect_d, rect_e, rect_e_tile);

    // hls_orderblk_tokennz(orders, rect_a, strm_coef_raster, //
    // ac_static_context_map,
    //                          cnt_nz, strm_nz_token, strm_coef_orderd);

    static hls::stream<dct_t> strm_coef_ord("strm_coef_ord");
//#pragma HLS RESOURCE variable = strm_coef_ord core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_coef_ord depth = 1024

    // for predict
    hls::stream<nzeros_t> cnt_nz2("strm_cnt_nz2");
#pragma HLS RESOURCE variable = cnt_nz2 core = FIFO_LUTRAM
#pragma HLS STREAM variable = cnt_nz2 depth = 32

    _XF_IMAGE_PRINT("\n --2 tmp_num_nzeros begin - Tokenize \n");

    hls_orderBlk(rect_a, orders, strm_coef_raster, strm_coef_ord); // II=1

    // hls_count_ac_nz(rect_d, strm_coef_ord, strm_coef_orderd, cnt_nz,
    // cnt_nz2);//II=1/8 in64
    hls_CountAcNz(rect_d, strm_coef_ord, strm_coef_orderd, cnt_nz, cnt_nz2);

    hls_tokenize_nz(rect_e, cnt_nz2, strm_nz_token); // II=1/8 in64

    // tokenize_blk(rect_b, strm_coef_orderd, cnt_nz, strm_cnt_ac, strm_ac_token);
    // //3.3ns II=2

    //    tokenize_blk_syn_test(rect_b, strm_coef_orderd, cnt_nz,
    //    		strm_cnt_ac, strm_histo_offset, strm_cnt_lookup,
    //    strm_last_pos,  strm_ac_token); //3.7ns II=1//strm_e,

    // tokenize_lookup_table(strm_histo_offset, strm_cnt_lookup, strm_last_pos,
    // strm_ac_token, strm_e, strm_ac_token2);

    hls_TokenizeBlk(rect_b, strm_coef_orderd, cnt_nz, strm_cnt_ac, strm_histo_offset, strm_cnt_lookup, strm_last_pos,
                    strm_ac_token);

    tokenize_lookup_table(rect_b, strm_cnt_ac, strm_histo_offset, strm_cnt_lookup, strm_last_pos, strm_ac_token,
                          strm_cnt_ac2, strm_ac_token2);

    // interleaving collect the nz token and ac token
    collect_token_syn(rect_c, ac_static_context_map, rect_e_tile, // strm_cnt_nz, num_tile,
                      strm_nz_token, strm_cnt_ac2, strm_ac_token2, strm_token_addr, strm_e_addr, strm_e_token,
                      strm_token_symb, strm_token_bits);
}

//----------------------------------------------------------
void hls_config_gen(group_rect rect, hls::stream<hls_blksize>& strm_rect, hls::stream<bool>& strm_e_tile) {
#pragma HLS INLINE OFF
    for (int tby = 0; tby < rect.ysize_tiles; ++tby) {
        for (int tbx = 0; tbx < rect.xsize_tiles; ++tbx) { // block
#pragma HLS PIPELINE II = 1
            hls_blksize tmp;
            int x0 = tbx * hls_kTileDimInBlocks;
            int y0 = tby * hls_kTileDimInBlocks;
            tmp.xsize =
                (x0 + hls_kTileDimInBlocks <= rect.xsize_blocks) ? hls_kTileDimInBlocks : (rect.xsize_blocks - x0);
            tmp.ysize =
                (y0 + hls_kTileDimInBlocks <= rect.ysize_blocks) ? hls_kTileDimInBlocks : (rect.ysize_blocks - y0);
            strm_rect.write(tmp);
            if ((tby == rect.ysize_tiles - 1) && (tbx == rect.xsize_tiles - 1)) {
                strm_e_tile.write(true);
            } else {
                strm_e_tile.write(false);
            }
        }
    }
}

//----------------------------------------------------------
void XAcc_TokenizeCoefficients6(const int32_t orders[3][64], // color dct_band
                                const group_rect rect,
                                // hls::stream<dct_t> strm_coef_raster[8],
                                hls::stream<dct_t>& strm_coef_raster,
                                uint8_t ac_static_context_map[hls_kNumContexts],
                                // hls::stream< hls_blksize > &strm_blk_size,

                                hls::stream<ap_uint<13> >& strm_token_addr,
                                hls::stream<hls_Token_symb>& strm_token_symb,
                                hls::stream<hls_Token_bits>& strm_token_bits,
                                hls::stream<bool>& strm_e_addr,
                                hls::stream<bool>& strm_e_token) {
#pragma HLS INLINE OFF
#pragma HLS DATAFLOW

    // clang-format off
	  static hls::stream< hls_blksize > strm_rect;
#pragma HLS RESOURCE  	  variable = strm_rect core = FIFO_LUTRAM
#pragma HLS STREAM    	  variable = strm_rect depth = 32
	      static hls::stream< bool > strm_e_tile;
#pragma HLS RESOURCE  	  variable = strm_e_tile core = FIFO_LUTRAM
#pragma HLS STREAM    	  variable = strm_e_tile depth = 32
    // clang-format on

    hls_config_gen(rect, strm_rect, strm_e_tile);

    for (int tby = 0; tby < rect.ysize_tiles; ++tby) {
        for (int tbx = 0; tbx < rect.xsize_tiles; ++tbx) { // block

            hls_tokenize_AC_tile_top(orders, strm_rect, strm_e_tile, strm_coef_raster,
                                     ac_static_context_map, // strm_blk_size,
                                     strm_token_addr, strm_token_symb, strm_token_bits, strm_e_addr, strm_e_token);
        }
    }
}
