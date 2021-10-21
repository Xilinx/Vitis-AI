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

#include "kernel3/ctrl_tokenize.hpp"

void hls_TokenizeAcStrategy(hls_Rect dc_rect,
                            hls::stream<bool>& strm_strategy_block0,
                            hls::stream<uint8_t>& strm_stragegy,

                            hls::stream<hls_Token>& strm_strategy_token,
                            hls::stream<bool>& strm_e) {
#pragma HLS INLINE OFF
    hls_Token out_token;

    for (int by = 0; by < dc_rect.ysize; by++) {
        for (int bx = 0; bx < dc_rect.xsize; bx++) {
#pragma HLS PIPELINE II = 1
            bool block = strm_strategy_block0.read();
            uint8_t strategy_ = strm_stragegy.read();
            if (!block) {
                out_token.context = 0;
                out_token.symbol = static_cast<uint8_t>(strategy_);
                out_token.nbits = 0;
                out_token.bits = 0;
                strm_strategy_token.write(out_token);
                strm_e.write(false);
            }
        }
    }
    strm_e.write(true);
}

inline void XAcc_PredictFromTopAndLeft_quant(quant_t here,
                                             uint32_t x,
                                             bool is_top_row,
                                             bool ping,
                                             quant_t abv1[hls_kDcGroupDimInBlocks],
                                             quant_t abv2[hls_kDcGroupDimInBlocks],
                                             quant_t& left,
                                             quant_t abv[hls_kDcGroupDimInBlocks][2],
                                             quant_t& predicted) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = abv complete dim = 2
    quant_t default_val = 32;
    quant_t pred;

    // read the regs and update
    if (x == 0 && is_top_row) {
        pred = default_val;
    } else if (x == 0) {
        pred = abv1[x];
    } else if (is_top_row) {
        pred = left;
    } else {
        pred = (abv1[x] + left + 1) >> 1;
    }
    left = here;

    abv2[x] = here;
    predicted = pred;
}

void XAcc_PredictFromTopAndLeft_q(
    quant_t here, uint32_t x, bool is_top_row, quant_t& left, quant_t abv, quant_t& predicted) {
#pragma HLS INLINE

    quant_t default_val = 32;
    quant_t pred;

    // read the regs and update
    if (x == 0 && is_top_row) {
        pred = default_val;
    } else if (x == 0) {
        pred = abv;
    } else if (is_top_row) {
        pred = left;
    } else {
        pred = (abv + left + 1) >> 1;
    }
    left = here;
    predicted = pred;
}

void streamDup(hls_Rect dc_rect,
               hls::stream<quant_t>& istrm,
               hls::stream<quant_t>& ostrm0,
               hls::stream<quant_t>& ostrm1) {
#pragma HLS INLINE OFF

    for (int by = 0; by < dc_rect.ysize; by++) {
        for (int bx = 0; bx < dc_rect.xsize; bx++) {
#pragma HLS PIPELINE II = 1
            quant_t in = istrm.read();

            ostrm0.write(in);
            ostrm1.write(in);
        }
    }
}

void hls_linebuffer_write(hls_Rect dc_rect,
                          quant_t array_above_ram[hls_kDcGroupDimInBlocks],
                          hls::stream<quant_t>& strm_out) {
#pragma HLS INLINE OFF
    for (int bx = 0; bx < dc_rect.xsize; bx++) {
#pragma HLS PIPELINE II = 1
        strm_out.write(array_above_ram[bx]);
    }
}

void hls_linebuffer_read(hls_Rect dc_rect,
                         hls::stream<quant_t>& strm_in,
                         quant_t array_above_ram[hls_kDcGroupDimInBlocks]) {
#pragma HLS INLINE OFF
    for (int bx = 0; bx < dc_rect.xsize; bx++) {
#pragma HLS PIPELINE II = 1
        quant_t tmp = strm_in.read();
        array_above_ram[bx] = tmp;
    }
}

void hls_linebuffer(hls_Rect dc_rect, hls::stream<quant_t>& strm_in, hls::stream<quant_t>& strm_out) {
#pragma HLS INLINE OFF
    static quant_t array_above_ram[hls_kDcGroupDimInBlocks];
#pragma HLS RESOURCE variable = array_above_ram core = RAM_2P_BRAM

    // no init, because the ram not used in the line[0]
    for (int by = 0; by < dc_rect.ysize; by++) {
        for (int bx = 0; bx < dc_rect.xsize; bx++) {
#pragma HLS PIPELINE II = 1
            strm_out.write(array_above_ram[bx]);
            quant_t tmp = strm_in.read();
            array_above_ram[bx] = tmp;
        }
    }
}

void hls_TokenizeQuantField_main2(hls_Rect dc_rect,
                                  hls::stream<bool>& strm_strategy_block,
                                  hls::stream<quant_t>& strm_quant_field,
                                  hls::stream<quant_t>& strm_quant_abv,

                                  hls::stream<hls_Token>& strm_quant_token,
                                  hls::stream<bool>& strm_e) {
#pragma HLS INLINE OFF
    hls_Token out_token;

    quant_t quant_left;
    quant_t predicted_quant;
    quant_t quant_here;
    bool is_locked = false;
    bool ping = false; // why pingpang?

    int bx = 0;
    int by = 0;
    bool is_top_row = (by == 0);

#ifdef DEBUG_QUANT
    int cnt = 0;
#endif

    while (by < dc_rect.ysize || is_locked) { // for flatten for and while loop
#pragma HLS PIPELINE II = 1

        if (is_locked) { // because of the q>=255 need 2 clk to write the stream
            is_locked = false;
            out_token.context = hls_QuantContext;
            out_token.symbol = quant_here - 1;
            out_token.nbits = 0;
            out_token.bits = 0;

            strm_quant_token.write(out_token);
            strm_e.write(false);

        } else {
            bool block_ = strm_strategy_block.read();

            quant_here = strm_quant_field.read();
            quant_t quant_abv = strm_quant_abv.read();

#ifdef DEBUG_QUANT
            cnt++;
#endif

            quant_t predicted_quant;
            XAcc_PredictFromTopAndLeft_q(quant_here, bx, is_top_row, quant_left, quant_abv, predicted_quant);

            if (!block_) {
                assert(quant_here < 32768);
                assert(quant_here > -32767);
                uint16_t q = hls_PackSigned_16b((int16_t)(quant_here - predicted_quant));

                if (q >= 255) {
                    _XF_IMAGE_PRINT("---quant_here = %d, predicted_quant = %d\n", quant_here, predicted_quant);
                    is_locked = true;

                    out_token.context = hls_QuantContext;
                    out_token.symbol = 255;
                    out_token.nbits = 0;
                    out_token.bits = 0;

                    strm_quant_token.write(out_token);
                    strm_e.write(false);

                    bx++;

                } else {
                    out_token.context = hls_QuantContext;
                    out_token.symbol = q;
                    out_token.nbits = 0;
                    out_token.bits = 0;

                    strm_quant_token.write(out_token);
                    strm_e.write(false);

                    bx++;
                }
            } else {
                bx++;
            }
        }

        if (bx == dc_rect.xsize && (!is_locked)) {
            by++;
            is_top_row = false;
            bx = 0;
            ping = !ping;
        }
    } // end by

    strm_e.write(true);

#ifdef DEBUG_QUANT
    std::cout << "read quant_field:" << cnt << std::endl;
#endif
}

void hls_TokenizeQuantField_main(hls_Rect dc_rect,
                                 hls::stream<bool>& strm_strategy_block,
                                 hls::stream<quant_t>& strm_quant_field,
                                 hls::stream<quant_t>& strm_quant_abv,

                                 hls::stream<hls_Token>& strm_quant_token,
                                 hls::stream<bool>& strm_e) {
#pragma HLS INLINE OFF
    hls_Token out_token;

    quant_t quant_left;
    quant_t predicted_quant;
    quant_t quant_here;
    bool is_locked = false;
    bool ping = false; // why pingpang?

    int bx = 0;
    int by = 0;
    bool is_top_row = (by == 0);

#ifdef DEBUG_QUANT
    int cnt = 0;
#endif

    for (int by = 0; by < dc_rect.ysize; by++) {
        is_top_row = (by == 0);
        bx = 0;

        while (bx < dc_rect.xsize || is_locked) { // for flatten for and while loop
#pragma HLS PIPELINE II = 1

            if (is_locked) { // because of the q>=255 need 2 clk to write the stream
                is_locked = false;
                out_token.context = hls_QuantContext;
                out_token.symbol = quant_here - 1;
                out_token.nbits = 0;
                out_token.bits = 0;

                strm_quant_token.write(out_token);
                strm_e.write(false);

            } else {
                bool block_ = strm_strategy_block.read();

                quant_here = strm_quant_field.read();
                quant_t quant_abv = strm_quant_abv.read();

#ifdef DEBUG_QUANT
                cnt++;
#endif

                quant_t predicted_quant;

                XAcc_PredictFromTopAndLeft_q(quant_here, bx, is_top_row, quant_left, quant_abv, predicted_quant);

                if (!block_) {
                    assert(quant_here < 32768);
                    assert(quant_here > -32767);
                    uint16_t q = hls_PackSigned_16b((int16_t)(quant_here - predicted_quant));

                    if (q >= 255) {
                        _XF_IMAGE_PRINT("---quant_here = %d, predicted_quant = %d\n", quant_here, predicted_quant);
                        is_locked = true;

                        out_token.context = hls_QuantContext;
                        out_token.symbol = 255;
                        out_token.nbits = 0;
                        out_token.bits = 0;

                        strm_quant_token.write(out_token);
                        strm_e.write(false);

                        bx++;

                    } else {
                        out_token.context = hls_QuantContext;
                        out_token.symbol = q;
                        out_token.nbits = 0;
                        out_token.bits = 0;

                        strm_quant_token.write(out_token);
                        strm_e.write(false);

                        bx++;
                    }
                } else {
                    bx++;
                }
            }
        } // end bx
    }     // end by

    strm_e.write(true);

#ifdef DEBUG_QUANT
    std::cout << "read quant_field:" << cnt << std::endl;
#endif
}

// goal: one Quant / clock
void hls_TokenizeQuantField(hls_Rect dc_rect,
                            hls::stream<bool>& strm_strategy_block,
                            hls::stream<quant_t>& quant_field,

                            hls::stream<hls_Token>& strm_quant_token,
                            hls::stream<bool>& strm_e) {
#pragma HLS INLINE OFF
#pragma HLS DATAFLOW

    hls::stream<quant_t> quant_here("quant_here");
#pragma HLS RESOURCE variable = quant_here core = FIFO_LUTRAM
#pragma HLS STREAM variable = quant_here depth = 32
    hls::stream<quant_t> quant_here1("quant_here1");
#pragma HLS RESOURCE variable = quant_here1 core = FIFO_LUTRAM
#pragma HLS STREAM variable = quant_here1 depth = 32
    hls::stream<quant_t> quant_abv("quant_abv");
#pragma HLS RESOURCE variable = quant_abv core = FIFO_LUTRAM
#pragma HLS STREAM variable = quant_abv depth = 32

    streamDup(dc_rect, quant_field, quant_here, quant_here1);
    hls_linebuffer(dc_rect, quant_here1, quant_abv);
    hls_TokenizeQuantField_main(dc_rect, strm_strategy_block, quant_here, quant_abv, strm_quant_token, strm_e);
}

// goal: one stragegy / clock
void hls_TokenizeARParameters(hls_Rect dc_rect,
                              hls::stream<bool>& strm_strategy_block,
                              hls::stream<arsigma_t>& strm_arsigma,

                              hls::stream<hls_Token>& strm_arsigma_token,
                              hls::stream<bool>& strm_e) {
#pragma HLS INLINE OFF

    hls_Token out_token;

    for (int by = 0; by < dc_rect.ysize; by++) {
        for (int bx = 0; bx < dc_rect.xsize; bx++) {
#pragma HLS PIPELINE II = 1
            bool block_ = strm_strategy_block.read();
            arsigma_t tmp = strm_arsigma.read();
            if (!block_) {
                out_token.context = hls_kARParamsContexts;
                out_token.symbol = tmp;
                out_token.nbits = 0;
                out_token.bits = 0;

                strm_arsigma_token.write(out_token);
                strm_e.write(false);
            }
        }
    }
    strm_e.write(true);
}

// goal: one token / cycle
void collect_ctrl_token(hls::stream<hls_Token>& strm_strategy_token,
                        hls::stream<bool>& strm_e_strategy,
                        hls::stream<addr_t>& strm_token_ct_addr,
                        hls::stream<bool>& strm_e_ct_addr,
                        hls::stream<hls_Token_symb>& strm_token_symb,
                        hls::stream<hls_Token_bits>& strm_token_bits,
                        hls::stream<bool>& strm_e_ctrl) {
#pragma HLS INLINE OFF
    hls_Token out_token;
    hls_Token_symb out_s;
    hls_Token_bits out_t;

    _XF_IMAGE_PRINT("---read the strategy_token \n");
    bool e = strm_e_strategy.read();

    while (!e) {
#pragma HLS PIPELINE II = 1
        out_token = strm_strategy_token.read();
        e = strm_e_strategy.read();
        out_s.symbol = out_token.symbol;
        out_s.context = out_token.context;
        out_t.nbits = out_token.nbits;
        out_t.bits = out_token.bits;
        strm_token_symb.write(out_s);
        strm_token_bits.write(out_t);
        strm_e_ctrl.write(false);

        addr_t addr = (out_token.context << 8) + out_token.symbol;
        strm_token_ct_addr.write(addr);
        strm_e_ct_addr.write(false);
        _XF_IMAGE_PRINT("---write token(%d,%d,%d,%d), %d \n", (int)(out_token.context), (int)(out_token.symbol),
                        out_token.nbits, out_token.bits, (int)addr);
    }

    _XF_IMAGE_PRINT("---read the quant_token \n");
    e = strm_e_strategy.read();

    while (!e) {
#pragma HLS PIPELINE II = 1
        out_token = strm_strategy_token.read();
        e = strm_e_strategy.read();
        out_s.symbol = out_token.symbol;
        out_s.context = out_token.context;
        out_t.nbits = out_token.nbits;
        out_t.bits = out_token.bits;
        strm_token_symb.write(out_s);
        strm_token_bits.write(out_t);
        strm_e_ctrl.write(false);

        addr_t addr = (out_token.context << 8) + out_token.symbol;
        strm_token_ct_addr.write(addr);
        strm_e_ct_addr.write(false);
        _XF_IMAGE_PRINT("---write token(%d,%d,%d,%d), %d \n", (int)(out_token.context), (int)(out_token.symbol),
                        out_token.nbits, out_token.bits, (int)addr);
    }

    _XF_IMAGE_PRINT("---read the arsigma_token \n");
    e = strm_e_strategy.read();

    while (!e) {
#pragma HLS PIPELINE II = 1
        out_token = strm_strategy_token.read();
        e = strm_e_strategy.read();
        out_s.symbol = out_token.symbol;
        out_s.context = out_token.context;
        out_t.nbits = out_token.nbits;
        out_t.bits = out_token.bits;
        strm_token_symb.write(out_s);
        strm_token_bits.write(out_t);
        strm_e_ctrl.write(false);

        addr_t addr = (out_token.context << 8) + out_token.symbol;
        strm_token_ct_addr.write(addr);
        strm_e_ct_addr.write(false);
        _XF_IMAGE_PRINT("---write token(%d,%d,%d,%d), %d \n", (int)(out_token.context), (int)(out_token.symbol),
                        out_token.nbits, out_token.bits, (int)addr);
    }

    strm_e_ctrl.write(true);
    strm_e_ct_addr.write(true);
}

void hls_TokenizeCtrlField_warpper(hls_Rect dc_rect,
                                   hls::stream<uint8_t>& strm_strategy,
                                   hls::stream<quant_t>& strm_quant_field,
                                   hls::stream<arsigma_t>& strm_arsigma,
                                   hls::stream<bool>& strm_strategy_block0,
                                   hls::stream<bool>& strm_strategy_block1,
                                   hls::stream<bool>& strm_strategy_block2,

                                   hls::stream<hls_Token>& strm_strategy_token,
                                   hls::stream<bool>& strm_e_strategy) {
// sequantial
#pragma HLS INLINE OFF

    hls_TokenizeAcStrategy(dc_rect, strm_strategy_block0, strm_strategy, strm_strategy_token, strm_e_strategy);

    hls_TokenizeQuantField(dc_rect, strm_strategy_block1, strm_quant_field, strm_strategy_token, strm_e_strategy);

    hls_TokenizeARParameters(dc_rect, strm_strategy_block2, strm_arsigma, strm_strategy_token, strm_e_strategy);
}

// goal: Save storage resources with build histogram
void Xacc_TokenizeCtrlField_top(hls_Rect dc_rect,
                                hls::stream<uint8_t>& strm_strategy,
                                hls::stream<quant_t>& strm_quant_field,
                                hls::stream<arsigma_t>& strm_arsigma,
                                hls::stream<bool>& strm_strategy_block0,
                                hls::stream<bool>& strm_strategy_block1,
                                hls::stream<bool>& strm_strategy_block2,

                                hls::stream<addr_t>& strm_token_ct_addr,
                                hls::stream<hls_Token_symb>& strm_token_symb,
                                hls::stream<hls_Token_bits>& strm_token_bits,
                                hls::stream<bool>& strm_e_ct_addr,
                                hls::stream<bool>& strm_e_ctrl) {
#pragma HLS INLINE OFF
#pragma HLS DATAFLOW

    static hls::stream<hls_Token> strm_strategy_token;
#pragma HLS DATA_PACK variable = strm_strategy_token
#pragma HLS RESOURCE variable = strm_strategy_token core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_strategy_token depth = 32
    static hls::stream<hls_Token> strm_quant_token;
#pragma HLS DATA_PACK variable = strm_quant_token
#pragma HLS RESOURCE variable = strm_quant_token core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_quant_token depth = 32
    static hls::stream<hls_Token> strm_arsigma_token;
#pragma HLS DATA_PACK variable = strm_arsigma_token
#pragma HLS RESOURCE variable = strm_arsigma_token core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_arsigma_token depth = 32
    static hls::stream<bool> strm_e_strategy;
#pragma HLS RESOURCE variable = strm_e_strategy core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_e_strategy depth = 32
    static hls::stream<bool> strm_e_quant;
#pragma HLS RESOURCE variable = strm_e_quant core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_e_quant depth = 32
    static hls::stream<bool> strm_e_arsigma;
#pragma HLS RESOURCE variable = strm_e_arsigma core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_e_arsigma depth = 32

    hls_TokenizeCtrlField_warpper(dc_rect, strm_strategy, strm_quant_field, strm_arsigma, strm_strategy_block0,
                                  strm_strategy_block1, strm_strategy_block2, strm_strategy_token, strm_e_strategy);

    collect_ctrl_token(strm_strategy_token, strm_e_strategy, strm_token_ct_addr, strm_e_ct_addr, strm_token_symb,
                       strm_token_bits, strm_e_ctrl);
}
