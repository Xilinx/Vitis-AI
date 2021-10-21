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
 * @file XAcc_jpegdecoder.cpp
 * @brief mcu_decoder template function implementation.
 *
 * This file is part of HLS algorithm library.
 */

#include "XAcc_jpegdecoder.hpp"
#include "utils/x_hls_utils.h"

#define DEVLI(s, n) ((s) == 0 ? (n) : (((n) >= (1 << ((s)-1))) ? (n) : (n) + 1 - (1 << (s))))

//**************************************

namespace xf {
namespace codec {
namespace details {

// ------------------------------------------------------------
void pick_huff_data(hls::stream<CHType>& image_strm,
                    hls::stream<bool>& eof_strm,
                    uint32_t& cnt_rst,
                    hls::stream<bool>& sign_no_huff,
                    hls::stream<sos_data>& huff_sos_strm) {
#pragma HLS INLINE off

    int test = 0; // for debug

    uint8_t EOI_marker = 0xD9;
    uint8_t RST_filter = 0xD0;
    sos_data huff_sos;
    uint8_t bytes[16];
    uint8_t tmp[8];
#pragma HLS ARRAY_PARTITION variable = bytes complete
#pragma HLS ARRAY_PARTITION variable = tmp complete
    bool is_ff = false;
    bool entropy_end = false; // other marker:d8~dd. is marker ff00 ffff
    int rst_cnt = 0;
    bool no_huff = false;

    // first read once
    bool eof = eof_strm.read();
    if (!eof) {
        ap_uint<CH_W> image = image_strm.read();
        eof = eof_strm.read();
        for (int i = 0; i < CH_W / 8; i++) {
#pragma HLS UNROLL
            bytes[i] = image(8 * i + 7, 8 * i);
        }
    } else {
        no_huff = true;
    }

    sign_no_huff.write(no_huff); // give a sign to next module if there is huffman stream

    bool eof_reg = eof;

// pipeline working
PICK_HUFF_LOOP:
    while (!eof_reg) { // eof_reg
#pragma HLS LOOP_TRIPCOUNT min = 5000 max = 5000
#pragma HLS PIPELINE II = 1
        eof_reg = eof; // read one more time to loop the shift buffer
        if (!eof) {
            ap_uint<CH_W> image = image_strm.read();
            eof = eof_strm.read();
            for (int i = 0; i < CH_W / 8; i++) {
#pragma HLS UNROLL
                bytes[CH_W / 8 + i] = image(8 * i + 7, 8 * i);
                tmp[i] = 0;
            }
        }

        // pick up
        int idx = 0;
        int garbage_bytes = 16;
        bool rst_flag = false;
        for (int i = 0; i < CH_W / 8; i++) { // check the ff
#pragma HLS UNROLL
            bool Redu_data = (bytes[i] == 0xFF) && ((bytes[i + 1] == 0x00));
            if (!Redu_data || entropy_end) {
                tmp[idx] = (is_ff | entropy_end) ? 0xFF : bytes[i];
                idx++; // when !Redu_data or marker entropy_end
            }

            if (is_ff && ((bytes[i] & RST_filter) == 0xD0)) { // ff dn or ff ff
                garbage_bytes = idx;
            }

            if (is_ff && (bytes[i] == 0xD0 + (rst_cnt & 7))) {
                rst_flag |= true;
            }

            entropy_end = entropy_end | (is_ff && (bytes[i] == EOI_marker));
            if (bytes[i] == 0xFF) {
                is_ff = true;
            } else {
                is_ff = false;
            }
        }

        if (rst_flag) {
            rst_cnt++;
        }

        for (int i = 0; i < CH_W / 8; i++) {
#pragma HLS UNROLL
            if (i >= idx) tmp[i] = 0;
        }

        // write out
        huff_sos.bits = idx * 8;
        huff_sos.data = tmp[0] << 8 | tmp[1];
        huff_sos.garbage_bits = garbage_bytes * 8;
        huff_sos.end_sos = false; // 7fff + 95 + false

        huff_sos_strm.write(huff_sos);

        // printf("\n  %.2x  %.2x  %.2x  %.2x",tmp[0],tmp[1],tmp[2],tmp[3]);
        for (int i = 0; i < CH_W / 8; i++) {
#pragma HLS UNROLL
            bytes[i] = bytes[i + CH_W / 8];
        }

#ifndef __SYNTHESIS__
        test++;
#endif
    } // endwhile

    cnt_rst = rst_cnt; // the count of reset marker

    // if there is no EOI marker, the stream will stop when AXI data is empty
    if (!no_huff) {
        huff_sos.bits = CH_W;
        huff_sos.data = 0xffff;
        huff_sos.garbage_bits = CH_W - 16;
        huff_sos.end_sos = true;
        huff_sos_strm.write(huff_sos);
    }
}

// ------------------------------------------------------------
void Huffman_decoder(
    // input
    hls::stream<sos_data>& huff_sos_strm,
    hls::stream<bool>& sign_no_huff,
    const uint16_t dht_tbl1[2][2][1 << DHT1],
    // const uint16_t dht_tbl2[2][2][1 << DHT2],
    //
    const uint8_t ac_val[2][165],
    const HCODE_T ac_huff_start_code[2][AC_N],
    const int16_t ac_huff_start_addr[2][16],
    //
    const uint8_t dc_val[2][12],
    const HCODE_T dc_huff_start_code[2][DC_N],
    const int16_t dc_huff_start_addr[2][16],
    //
    const ap_uint<12> cyc_cmp,
// regs
#ifndef __SYNTHESIS__
    const uint8_t hls_cs_cmpc,
    const uint16_t hls_mcuh,
#endif
    const uint8_t hls_mbs[MAX_NUM_COLOR],
    const uint32_t hls_mcuc,

    // output
    bool& rtn2,
    hls::stream<ap_uint<24> >& block_strm) {

#pragma HLS INLINE off

    ap_uint<12> hls_cmp = cyc_cmp;
#pragma HLS RESOURCE variable = hls_cmp core = FIFO_SRL
    int16_t lastDC[4] = {0, 0, 0, 0};
#pragma HLS ARRAY_PARTITION variable = lastDC complete

    // major parameter
    uint8_t huff_len = 1;    // the length of bits for huffman codes, eq with idx+1 1~16
    uint8_t run_len = 0;     // the number of zero before the non-zero ac coefficient 0~15
    uint8_t val_len = 0;     // the length of bits for value, 0~11
    uint8_t total_len = 0;   // huff_len + val_len 1~27
    uint8_t dec_len = 0;     // the length of tbl_data bits in buff 1~27
    ap_uint<24> block_coeff; // 23:is_bad, 22:is_endblock,21~16:bpos,15~0:block val

    int test = 0;
#ifndef __SYNTHESIS__
    uint8_t n_last = 0;
    uint8_t garbage_bits = 0;
    //    ap_uint<6> n_last = 0;
    //    ap_uint<6> garbage_bits = 0;
    int test_in8 = 0;
    int test_in16 = 0;
    int test_ov16 = 0;
    int cmp = 0;
    int mbs = 0;
    int n_mcu = 0;
#else
    ap_uint<6> n_last = 0;
    ap_uint<6> garbage_bits = 0;
#endif
    bool empty_n = false;
    bool e = false; // data end
    bool e_reg2 = sign_no_huff.read();
    bool e_reg1 = false;

    // tmp parameter
    ap_uint<16> input;
    ap_uint<8> input8;
    uint8_t bpos = 0;
    sos_data huff_sos;
    uint16_t val_i;
    int16_t block, block_tmp;
    bool ac = false;

    // major buffer
    ap_uint<2 * CH_W> buff_huff = 0; // the shift buffer
    ap_uint<2 * CH_W> buff_tail = 0; // the shift buffer
    // accurate shift control group, to adjust circuit timing
    ap_uint<2 * CH_W> buf_dat0 = 0;
    ap_uint<CH_W> buf_dat1 = 0;
    ap_uint<CH_W> buf_dat2 = 0;
    ap_uint<CH_W> buf_dat3 = 0;
    ap_uint<2 * CH_W> buf1 = 0;
    ap_uint<2 * CH_W> buf2 = 0;

    // major flag to control state machine
    bool lookup_tbl2 = false;
    // bool is_bad = false; // todo may be used in hls_next_mcupos2
    bool snd_loop = false; // first loop not decode the symbol, use snd
    bool thd_loop = false; // second loop not decode the symbol, use third loop
    bool fth_loop = false; // third loop not decode the symbol, use fourth loop
    bool fif_loop = false;
    bool next_block_reg = false;
    bool is_garbage = false;

    // tmp
    uint16_t tbl1 = 0;
    uint16_t tbl2 = 0;
    ap_uint<8> run_vlen = 0;
    uint16_t tbl_data = 0;
    int tmp_bits = 0;

    // add
    ap_uint<17> thermo_code;
    thermo_code[16] = 1;
    thermo_code[0] = 0;
    thermo_code[1] = 0;
    thermo_code[2] = 0;
    thermo_code[3] = 0;
    thermo_code[4] = 0;
    thermo_code[5] = 0;
    thermo_code[6] = 0;
    thermo_code[7] = 0;
    thermo_code[8] = 0;
    thermo_code[9] = 0;

    uint8_t huff_len_tbl2 = 0;
    uint8_t val_addr_dc = 0;
    uint8_t val_addr_ac = 0;
    ap_uint<16> input_reg;
    uint16_t huff[16]; // the input huffman codes
#pragma HLS ARRAY_PARTITION variable = huff complete

    const int cpmall = hls_mbs[0] + hls_mbs[1] + hls_mbs[2];

DECODE_LOOP:
    while (!e_reg2) {
#pragma HLS LOOP_TRIPCOUNT min = 10000 max = 10000
//#pragma HLS DEPENDENCE array inter false
#pragma HLS PIPELINE II = 1

        //----------
        // 1. shift all buffers and read huff_sos
        n_last = n_last - dec_len;
        garbage_bits = garbage_bits - dec_len;
        buff_huff <<= dec_len;

        buf2(2 * CH_W - 1, CH_W) = buf_dat1;
        buf2(1 * CH_W - 1, 0) = buf_dat2;
        buf1(2 * CH_W - 1, CH_W) = buf_dat2;
        buf1(1 * CH_W - 1, 0) = buf_dat3;
        buf1 <<= dec_len;
        buf2 <<= dec_len;
        buf_dat1 = (CHType)(buf2(2 * CH_W - 1, CH_W));
        buf_dat2 = (CHType)(buf1(2 * CH_W - 1, CH_W));
        buf_dat3 = buf1(CH_W - 1, 0);

        if (garbage_bits == 0) {
            e_reg2 = e_reg1; // end all blocks
            is_garbage = false;
        }
        if (((n_last < CH_W) && (empty_n)) || (e && (n_last <= CH_W))) { // prepare data

            buff_tail(2 * CH_W - 1, CH_W) = (CHType)(buf2(2 * CH_W - 1, CH_W));
            buff_tail(1 * CH_W - 1, 0) = (CHType)(buf1(2 * CH_W - 1, CH_W));

            buff_huff |= buff_tail;
            tmp_bits = huff_sos.garbage_bits;
            if (tmp_bits <= 2 * CH_W) {
                garbage_bits = n_last + huff_sos.garbage_bits;
                is_garbage = true;
            }
            n_last += huff_sos.bits;
            e_reg1 = e;

            empty_n = false;
        }
        if ((empty_n == false) && (!e)) { // read data

            huff_sos = huff_sos_strm.read();
            e = huff_sos.end_sos;
            buf_dat0 = huff_sos.data;
            buf_dat1 = buf_dat0 >> n_last;
            buf_dat2 = (buf_dat0 << CH_W) >> n_last;
            buf_dat3 = (buf_dat0 << (31 - n_last)) << 1; // CH_W==16

            empty_n = true;
        }

        // decode one huffman code from 32b
        bool freeze_out = (n_last < CH_W) && (!e_reg1); // unfreeze
        input = buff_huff(2 * CH_W - 1, 2 * CH_W - 16);
        // input8 = buff_huff(2 * CH_W - 1, 2 * CH_W - 8);
        // is_bad = false;

        //----------
        // 2. look up the table

        dec_len = 0;

        if (!snd_loop && (!thd_loop) && (!fth_loop) && (!fif_loop)) {
            // is the frist decode (total len<=15 && huffman_len<=DHT1 will done in first decode)
            // anyway the huffman_len will be shift out of data
            // tbl_data = (tbl1 >> 15) ? tbl2 : tbl1;
            ap_uint<DHT1> addr1 = input(15, 16 - DHT1);

            tbl1 = dht_tbl1[ac][hls_cmp[0]][addr1];

            if ((tbl1 >> 15) == 0) {
                tbl_data = tbl1;

                total_len = tbl_data & 0x1F;
                huff_len = (tbl_data >> 5) & 0x1F;
                run_len = (tbl_data >> 10) & 0x0F;
                val_len = total_len - huff_len;
                lookup_tbl2 = false;

            } else {
                lookup_tbl2 = true;
                input_reg = input;
                total_len = 16;
                huff_len = 0;
            }

            if (!freeze_out) { // if reset, false valid
                if (input == 0xFFFF) {
                    if (garbage_bits <= 16) {
                        dec_len = garbage_bits;
                    } else {
                        dec_len = garbage_bits - 16;
                    }

                    ac = false;
                    freeze_out = true;
                    snd_loop = false;
                    lastDC[0] = 0;
                    lastDC[1] = 0;
                    lastDC[2] = 0;
                } else {
                    if (total_len <= 15) {
                        dec_len = total_len;
                        freeze_out = false;
                        snd_loop = false;
                    } else {
                        dec_len = huff_len;
                        freeze_out = true;
                        snd_loop = true;
                    }
                }
            }

        } else if (snd_loop) {
            // decode one huffman length from 16b

            // compare, then decode a 17-bit thermometer code to huff_len
            for (int i = 10; i < 13; i++) {
#pragma HLS UNROLL
                if (ac) {
                    thermo_code[i] = (input_reg(13, 15 - i) >= ac_huff_start_code[hls_cmp[0]][i - 10]) ? 0 : 1;
                } else {
                    thermo_code[i] = (input_reg(13, 15 - i) >= dc_huff_start_code[hls_cmp[0]][i - 10]) ? 0 : 1;
                }
            }

            if (!freeze_out) { // wait until there is enough data
                if (lookup_tbl2) {
                    huff_len = 0;
                    // total_len = 16;
                    dec_len = huff_len;
                    freeze_out = true;
                    fif_loop = true;

                } else {
                    huff_len = 0;
                    total_len = val_len;
                    dec_len = total_len;
                    fif_loop = false;
                }

                snd_loop = false;
            }

        } else if (fif_loop) {
            // compare, then decode a 16-bit thermometer code to huff_len
            for (int i = 13; i < 16; i++) {
#pragma HLS UNROLL
                if (ac) {
                    thermo_code[i] = (input_reg(13, 15 - i) >= ac_huff_start_code[hls_cmp[0]][i - 10]) ? 0 : 1;
                } else {
                    _XF_IMAGE_PRINT(" there is an error DC code! \n");
                }
            }

            huff_len_tbl2 = (unsigned char)__builtin_ctz((unsigned int)thermo_code);

            if (!freeze_out) {
                huff_len = huff_len_tbl2;
                // total_len = 16;
                dec_len = huff_len;
                freeze_out = true;
                thd_loop = true;

                fif_loop = false;
            }

        } else if (thd_loop) {
            // is the second decode (total len>=16 && huffman_len>DHT1 will done in second decode)
            // anyway the huffman_len will be shift out of data
            // if (!freeze_out) { // wait until there is enough data

            if (ac) {
                // val_addr = huff[huff_len-1] - ac_huff_start_addr[hls_cmp[0]][huff_len-1];
                val_addr_ac = input_reg(15, 16 - huff_len) - ac_huff_start_addr[hls_cmp[0]][huff_len - 1];
            } else {
                val_addr_dc = input_reg(15, 16 - huff_len) - dc_huff_start_addr[hls_cmp[0]][huff_len - 1];
            }
            freeze_out = true;
            fth_loop = true;

            // defalut no shift
            thd_loop = false;
            //}

        } else if (fth_loop) {
            if (ac) {
                run_vlen = ac_val[hls_cmp[0]][val_addr_ac];
            } else {
                run_vlen = dc_val[hls_cmp[0]][val_addr_dc];
            }

            val_len = run_vlen(3, 0);
            run_len = run_vlen(7, 4);
            lookup_tbl2 = false;

            if (!freeze_out) { // wait until there is enough data
                huff_len = 0;
                total_len = val_len;
                dec_len = total_len;

                fth_loop = false;
            }
        }

        //----------
        // 3. get the value
        if (!freeze_out) {
#ifndef __SYNTHESIS__
            if (run_len > 0) _XF_IMAGE_PRINT(" run_len = %d \n", (int)run_len);
#endif

            if (val_len) {
                val_i = buff_huff(2 * CH_W - 1 - huff_len, 2 * CH_W - total_len);
            } else {
                val_i = 0;
            }
            block_tmp = DEVLI(val_len, reg(val_i));
        }

        bool eob = !freeze_out && ac && ((run_len | val_len) == 0);

        if (!freeze_out) {
            if (ac) {
                bpos = bpos + 1 + run_len;
                block = block_tmp;
                _XF_IMAGE_PRINT("AC: huff_len = %d , block[%d] = %d\n", (int)huff_len, bpos, (int)block);
            } else {
                ac = true;
                bpos = 0;
                block = lastDC[0] + block_tmp;
                lastDC[0] = block;
                _XF_IMAGE_PRINT("\nDC: huff_len = %d , dc_val_i = %d\n", (int)huff_len, (int)block_tmp);
            }
        }

        //----------
        // 4. write out
        if (!freeze_out) {
            if (!eob) {
                block_coeff[23] = 0; // is_bad && (bpos == 63);
                block_coeff[22] = (bpos == 63);
                block_coeff(21, 16) = (uint8_t)bpos;
                block_coeff(15, 0) = block;
                block_strm.write(block_coeff);

            } else { // is eob W [63]=0

                block_coeff[23] = 0; // is_bad;//not in use
                block_coeff[22] = 1;
                block_coeff(21, 16) = (uint8_t)(63);
                block_coeff(15, 0) = 0;
                block_strm.write(block_coeff);
                _XF_IMAGE_PRINT(" ================ eob [%d,63] \n", bpos);
            }
        }

        //----------
        // 5. next_block update and shift sampling cmp
        bool next_block = (eob || (!freeze_out && (bpos == 63)));
        if (next_block) {
            ac = false;
            ap_uint<1> tmp_sft = hls_cmp[0];
            if (hls_cmp[0] | hls_cmp[1]) {
                int16_t tmpDC = lastDC[0];

                lastDC[0] = lastDC[1];
                lastDC[1] = lastDC[2];
                lastDC[2] = tmpDC;
            }
            hls_cmp >>= 1;
            hls_cmp[11] = tmp_sft;

#ifndef __SYNTHESIS__

            if (cmp < hls_cs_cmpc - 1) {
                if (mbs < hls_mbs[cmp] - 1) {
                    mbs++;
                } else {
                    mbs = 0;
                    cmp++;
                }
            } else {
                cmp = 0;
                n_mcu++;
            }

            // cpmall = hls_mbs[0] + hls_mbs[1] + hls_mbs[2];

            // clang-format off
            _XF_IMAGE_PRINT(" block decode %d  times !! mcu [%d, %d][%d] block \n", test, (test / (cpmall)) % hls_mcuh,
                            test / ((cpmall)*hls_mcuh), test % (cpmall)); // test 420
            _XF_IMAGE_PRINT(" lft_in_buff = %d  n_mcu=%d *****\n\n *********************\n", (int)(n_last - dec_len),n_mcu);
            if (((test / (cpmall)) % hls_mcuh == 3) && (test / ((cpmall)*hls_mcuh) == 0) && (test % (cpmall) == 1)) {
                int tmp_test = hls_mbs[0] + hls_mbs[1] + hls_mbs[2];
            }
// clang-format on
// test++;
#endif
            // if the test is 14999 the decode is end, but there is still garbage,
            // so we use 15000 to forcibly stop the bad cases
            if (test == hls_mcuc * cpmall) {
                e_reg2 = true;
            }
            test++;

        } // end new block

    } // end decode one block and loop all mcu/cmp/mbs

    // clean up the fifo/stream and return the error
    if (test != hls_mcuc * cpmall) {
        _XF_IMAGE_PRINT("Warning : there is error number of blocks!\n");
        rtn2 = true;

        if (!huff_sos_strm.empty()) {
        INTER_LOOP:
            while (!e) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS PIPELINE II = 1
                huff_sos = huff_sos_strm.read();
                e = huff_sos.end_sos;
            }
        }
    } // end inter_loop
}

// ------------------------------------------------------------
// for JPEG-D
void hls_next_mcupos2(hls::stream<ap_uint<24> >& block_strm,
                      int16_t hls_block[MAX_NUM_COLOR * MAXCMP_BC * 64],
                      int hls_sfv[4],
                      int hls_sfh[4],
                      const uint8_t hls_mbs[4],
                      int hls_bch,
                      int hls_bc,
                      int32_t hls_mcuc,
                      uint8_t hls_cs_cmpc,
                      bool rtn2,
                      int& sta) {
    // int sta = 0; // status
    int test = 0;

    int n_mcu = 0;
    int cmp = 0;
    int mbs = 0;
    ap_uint<24> block_coeff;
    bool is_endblock;
    uint8_t bpos = 0;
    int16_t block;
    //  int lastdc[4] = {0, 0, 0, 0}; // last dc for each component
    int dpos[MAX_NUM_COLOR] = {0};
//#pragma HLS ARRAY_PARTITION variable = lastdc complete
#pragma HLS ARRAY_PARTITION variable = dpos complete

    while (!sta) {
#pragma HLS PIPELINE II = 1
        block_coeff = block_strm.read();
        is_endblock = block_coeff[22];
        bpos = block_coeff(21, 16);
        block = block_coeff(15, 0);

        hls_block[(cmp)*hls_bc * 64 + (dpos[cmp]) * 64 + bpos] = block;

        if (is_endblock) {
            unsigned int sfh = hls_sfh[cmp]; // 2   1    1
            unsigned int sfv = hls_sfv[cmp]; // 2   2    1
            if (sfh > 1) {                   // 420 cmp=0
                if (cmp != 0) {
                    _XF_IMAGE_PRINT("ERROR: next_mcu 420 case, cmp!=0");
                    sta = 2;
                }
                if (mbs == 0) {
                    dpos[cmp]++;
                } else if (mbs == 1) {
                    dpos[cmp] += hls_bch - 1;
                } else if (mbs == 2) {
                    dpos[cmp]++;
                } else {
                    if (dpos[cmp] % (2 * hls_bch) == 2 * hls_bch - 1) {
                        dpos[cmp]++;
                    } else {
                        dpos[cmp] -= hls_bch - 1;
                    }
                }
            } else if (sfv > 1) { // 422 cmp=0
                if (cmp != 0) {
                    _XF_IMAGE_PRINT("ERROR: next_mcu 422 case, cmp!=0");
                    sta = 2;
                }
                dpos[cmp]++;
            } else { // 420 cmp=1/2 422 cmp=1/2 444 cmp=0/1/2
                dpos[cmp]++;
            }
            if (n_mcu < hls_mcuc) {
                if (cmp < hls_cs_cmpc - 1) {
                    if (mbs < hls_mbs[cmp] - 1) { // 420:4/422:2/444:1
                        mbs++;
                    } else {
                        mbs = 0;
                        cmp++;
                    }
                } else {
                    cmp = 0;
                    n_mcu++;
                    if (n_mcu == hls_mcuc) {
                        sta = 2;
                    }
                }
            }

            test++;
        } // end one block

    } // end while

    while (!block_strm.empty()) {
#pragma HLS PIPELINE II = 1
        block_coeff = block_strm.read();
    }
}

// ------------------------------------------------------------
void Huffman_decoder2(
    // input
    hls::stream<sos_data>& huff_sos_strm,
    hls::stream<bool>& sign_no_huff,
    const uint16_t dht_tbl1[2][2][1 << DHT1],
    const uint16_t dht_tbl2[2][2][1 << DHT2],
    const ap_uint<12> cyc_cmp,
// regs
#ifndef __SYNTHESIS__
    const uint8_t hls_cs_cmpc,
    const uint16_t hls_mcuh,
#endif
    const uint8_t hls_mbs[MAX_NUM_COLOR],
    const uint32_t hls_mcuc,

    // output
    bool& rtn2,
    hls::stream<ap_uint<24> >& block_strm) {

#pragma HLS INLINE off

    ap_uint<12> hls_cmp = cyc_cmp;
#pragma HLS RESOURCE variable = hls_cmp core = FIFO_SRL
    int16_t lastDC[4] = {0, 0, 0, 0};
#pragma HLS ARRAY_PARTITION variable = lastDC complete

    // major parameter
    uint8_t huff_len = 0;    // the length of bits for huffman codes, eq with idx+1 1~16
    uint8_t run_len = 0;     // the number of zero before the non-zero ac coefficient 0~15
    uint8_t val_len = 0;     // the length of bits for value, 0~11
    uint8_t total_len = 0;   // huff_len + val_len 1~27
    uint8_t dec_len = 0;     // the length of tbl_data bits in buff 1~27
    ap_uint<24> block_coeff; // 23:is_bad, 22:is_endblock,21~16:bpos,15~0:block val

    int test = 0;
#ifndef __SYNTHESIS__
    uint8_t n_last = 0;
    uint8_t garbage_bits = 0;
    //    ap_uint<6> n_last = 0;
    //    ap_uint<6> garbage_bits = 0;
    int test_in8 = 0;
    int test_in16 = 0;
    int test_ov16 = 0;
    int cmp = 0;
    int mbs = 0;
    int n_mcu = 0;
#else
    ap_uint<6> n_last = 0;
    ap_uint<6> garbage_bits = 0;
#endif
    bool empty_n = false;
    bool e = false; // data end
    bool e_reg2 = sign_no_huff.read();
    bool e_reg1 = false;

    // tmp parameter
    ap_uint<16> input;
    ap_uint<8> input8;
    uint8_t bpos = 0;
    sos_data huff_sos;
    uint16_t val_i;
    int16_t block, block_tmp;
    bool ac = false;

    // major buffer
    ap_uint<2 * CH_W> buff_huff = 0; // the shift buffer
    ap_uint<2 * CH_W> buff_tail = 0; // the shift buffer
    // accurate shift control group, to adjust circuit timing
    ap_uint<2 * CH_W> buf_dat0 = 0;
    ap_uint<CH_W> buf_dat1 = 0;
    ap_uint<CH_W> buf_dat2 = 0;
    ap_uint<CH_W> buf_dat3 = 0;
    ap_uint<2 * CH_W> buf1 = 0;
    ap_uint<2 * CH_W> buf2 = 0;

    // major flag to control state machine
    bool lookup_tbl2 = false;
    bool is_bad = false; // todo may be used in hls_next_mcupos2
    bool val_loop = false;
    bool next_block_reg = false;
    bool is_garbage = false;

    // tmp
    uint16_t tbl1 = 0;
    uint16_t tbl2 = 0;
    uint16_t tbl_data = 0;
    int tmp_bits = 0;

    const int cpmall = hls_mbs[0] + hls_mbs[1] + hls_mbs[2];

DECODE_LOOP:
    while (!e_reg2) {
#pragma HLS LOOP_TRIPCOUNT min = 10000 max = 10000
#pragma HLS DEPENDENCE array inter false
#pragma HLS PIPELINE II = 1

        //----------
        // 1. shift all buffers and read huff_sos
        n_last = n_last - dec_len;
        garbage_bits = garbage_bits - dec_len;
        buff_huff <<= dec_len;

        buf2(2 * CH_W - 1, CH_W) = buf_dat1;
        buf2(1 * CH_W - 1, 0) = buf_dat2;
        buf1(2 * CH_W - 1, CH_W) = buf_dat2;
        buf1(1 * CH_W - 1, 0) = buf_dat3;
        buf1 <<= dec_len;
        buf2 <<= dec_len;
        buf_dat1 = (CHType)(buf2(2 * CH_W - 1, CH_W));
        buf_dat2 = (CHType)(buf1(2 * CH_W - 1, CH_W));
        buf_dat3 = buf1(CH_W - 1, 0);

        if (garbage_bits == 0) {
            e_reg2 = e_reg1; // end all blocks
            is_garbage = false;
        }
        if (((n_last < CH_W) && (empty_n)) || (e && (n_last <= CH_W))) { // prepare data

            buff_tail(2 * CH_W - 1, CH_W) = (CHType)(buf2(2 * CH_W - 1, CH_W));
            buff_tail(1 * CH_W - 1, 0) = (CHType)(buf1(2 * CH_W - 1, CH_W));

            buff_huff |= buff_tail;
            tmp_bits = huff_sos.garbage_bits;
            if (tmp_bits <= 2 * CH_W) {
                garbage_bits = n_last + huff_sos.garbage_bits;
                is_garbage = true;
            }
            n_last += huff_sos.bits;
            e_reg1 = e;

            empty_n = false;
        }
        if ((empty_n == false) && (!e)) { // read data

            huff_sos = huff_sos_strm.read();
            e = huff_sos.end_sos;
            buf_dat0 = huff_sos.data;
            buf_dat1 = buf_dat0 >> n_last;
            buf_dat2 = (buf_dat0 << CH_W) >> n_last;
            buf_dat3 = (buf_dat0 << (31 - n_last)) << 1; // CH_W==16

            empty_n = true;
        }

        // decode one huffman code from 32b
        bool freeze_out = (n_last < CH_W) && (!e_reg1); // unfreeze
        input = buff_huff(2 * CH_W - 1, 2 * CH_W - 16);
        input8 = buff_huff(2 * CH_W - 1, 2 * CH_W - 8);
        is_bad = false;

        //----------
        // 2. look up the table
        ap_uint<DHT1> addr1 = input(15, 16 - DHT1);
        ap_uint<DHT2> addr2 = input(DHT2 - 1, 0);

        tbl1 = dht_tbl1[ac][hls_cmp[0]][addr1];
        tbl2 = dht_tbl2[ac][hls_cmp[0]][addr2];

        dec_len = 0;

        if (!val_loop) {
            tbl_data = (tbl1 >> 15) ? tbl2 : tbl1;

            total_len = tbl_data & 0x1F;
            huff_len = (tbl_data >> 5) & 0x1F;
            run_len = (tbl_data >> 10) & 0x0F;
            val_len = total_len - huff_len;

            if (!freeze_out) { // if reset, false valid
                if (input == 0xFFFF) {
                    if (garbage_bits <= 16) {
                        dec_len = garbage_bits;
                    } else {
                        dec_len = garbage_bits - 16;
                    }

                    ac = false;
                    freeze_out = true;
                    val_loop = false;
                    lastDC[0] = 0;
                    lastDC[1] = 0;
                    lastDC[2] = 0;
                } else {
                    if (total_len <= 15) {
                        dec_len = total_len;
                        freeze_out = false;
                        val_loop = false;
                    } else {
                        dec_len = huff_len;
                        freeze_out = true;
                        val_loop = true;
                    }
                }
            }
        } else {
            if (!freeze_out) { // wait until there is enough data

                huff_len = 0;
                total_len = val_len;
                dec_len = total_len;
                val_loop = false;
            }
        }

        //----------
        // 3. get the value
        if (!freeze_out) {
#ifndef __SYNTHESIS__
            if (run_len > 0) _XF_IMAGE_PRINT(" run_len = %d \n", (int)run_len);
#endif

            if (val_len) {
                val_i = buff_huff(2 * CH_W - 1 - huff_len, 2 * CH_W - total_len);
            } else {
                val_i = 0;
            }
            block_tmp = DEVLI(val_len, val_i);
        }

        bool eob = !freeze_out && ac && ((run_len | val_len) == 0);

        if (!freeze_out) {
            if (ac) {
                bpos = bpos + 1 + run_len;
                block = block_tmp;
                _XF_IMAGE_PRINT("AC: huff_len = %d , block[%d] = %d\n", (int)huff_len, bpos, (int)block);
            } else {
                ac = true;
                bpos = 0;
                block = lastDC[0] + block_tmp;
                lastDC[0] = block;
                _XF_IMAGE_PRINT("\nDC: huff_len = %d , dc_val_i = %d\n", (int)huff_len, (int)block_tmp);
            }
        }

        //----------
        // 4. write out
        if (!freeze_out) {
            if (!eob) {
                block_coeff[23] = is_bad && (bpos == 63);
                block_coeff[22] = (bpos == 63);
                block_coeff(21, 16) = (uint8_t)bpos;
                block_coeff(15, 0) = block;
                block_strm.write(block_coeff);

            } else { // is eob W [63]=0

                block_coeff[23] = is_bad; // not in use
                block_coeff[22] = 1;
                block_coeff(21, 16) = (uint8_t)(63);
                block_coeff(15, 0) = 0;
                block_strm.write(block_coeff);
                _XF_IMAGE_PRINT(" ================ eob [%d,63] \n", bpos);
            }
        }

        //----------
        // 5. next_block update and shift sampling cmp
        bool next_block = (eob || (!freeze_out && (bpos == 63)));
        if (next_block) {
            ac = false;
            ap_uint<1> tmp_sft = hls_cmp[0];
            if (hls_cmp[0] | hls_cmp[1]) {
                int16_t tmpDC = lastDC[0];

                lastDC[0] = lastDC[1];
                lastDC[1] = lastDC[2];
                lastDC[2] = tmpDC;
            }
            hls_cmp >>= 1;
            hls_cmp[11] = tmp_sft;

#ifndef __SYNTHESIS__

            if (cmp < hls_cs_cmpc - 1) {
                if (mbs < hls_mbs[cmp] - 1) {
                    mbs++;
                } else {
                    mbs = 0;
                    cmp++;
                }
            } else {
                cmp = 0;
                n_mcu++;
            }

            // cpmall = hls_mbs[0] + hls_mbs[1] + hls_mbs[2];

            // clang-format off
            _XF_IMAGE_PRINT(" block decode %d  times !! mcu [%d, %d][%d] block \n", test, (test / (cpmall)) % hls_mcuh,
                            test / ((cpmall)*hls_mcuh), test % (cpmall)); // test 420
            _XF_IMAGE_PRINT(" lft_in_buff = %d  n_mcu=%d *****\n\n *********************\n", (int)(n_last - dec_len),n_mcu);
            if (((test / (cpmall)) % hls_mcuh == 27) && (test / ((cpmall)*hls_mcuh) == 58) && (test % (cpmall) == 0)) {
                //cpmall = hls_mbs[0] + hls_mbs[1] + hls_mbs[2];
            }
// clang-format on
// test++;
#endif
            // if the test is 14999 the decode is end, but there is still garbage,
            // so we use 15000 to forcibly stop the bad cases
            if (test == hls_mcuc * cpmall) {
                e_reg2 = true;
            }
            test++;

        } // end new block

    } // end decode one block and loop all mcu/cmp/mbs

    // clean up the fifo/stream and return the error
    if (test != hls_mcuc * cpmall) {
        _XF_IMAGE_PRINT("Warning : there is error number of blocks!\n");
        rtn2 = true;

        if (!huff_sos_strm.empty()) {
            while (!e) {
#pragma HLS LOOP_TRIPCOUNT min = 100 max = 100
#pragma HLS PIPELINE II = 1
                huff_sos = huff_sos_strm.read();
                e = huff_sos.end_sos;
            }
        }
    }
}

// ------------------------------------------------------------
void mcu_decoder(
    // input
    hls::stream<CHType>& image_strm,
    hls::stream<bool>& eof_strm,
    const uint16_t dht_tbl1[2][2][1 << DHT1],
    const uint16_t dht_tbl2[2][2][1 << DHT2],
    ap_uint<12> hls_cmp,

    // image info
    const uint8_t hls_cs_cmpc, // component count in current scan
    const uint8_t hls_mbs[MAX_NUM_COLOR],
    const uint16_t hls_mcuh, // the horizontal mcu
    const uint32_t hls_mcuc, // the total mcu

    // output
    bool& rtn2,
    uint32_t& rst_cnt,
    hls::stream<ap_uint<24> >& block_strm) {
#pragma HLS DATAFLOW

    // clang-format off
	static hls::stream<sos_data> huff_sos_strm;
#pragma HLS DATA_PACK variable = huff_sos_strm
#pragma HLS RESOURCE  variable = huff_sos_strm core = FIFO_LUTRAM
#pragma HLS STREAM    variable = huff_sos_strm depth = 32
	static hls::stream<bool> sign_no_huff;
#pragma HLS DATA_PACK variable = sign_no_huff
#pragma HLS RESOURCE  variable = sign_no_huff core = FIFO_LUTRAM
#pragma HLS STREAM    variable = sign_no_huff depth = 1
    // clang-format on

    pick_huff_data(image_strm, eof_strm, rst_cnt, sign_no_huff, huff_sos_strm);

    Huffman_decoder2(huff_sos_strm, sign_no_huff, dht_tbl1, dht_tbl2, hls_cmp,
#ifndef __SYNTHESIS__
                     hls_cs_cmpc, hls_mcuh,
#endif
                     hls_mbs, hls_mcuc, rtn2, block_strm);
}

} // namespace details
} // namespace codec
} // namespace xf

// ------------------------------------------------------------
void top_mcu_decoder(
    // input
    hls::stream<CHType>& image_strm,
    hls::stream<bool>& eof_strm,
    const uint16_t dht_tbl1[2][2][1 << DHT1],
    // const uint16_t dht_tbl2[2][2][1 << DHT2],
    //
    const uint8_t ac_val[2][165],
    const HCODE_T ac_huff_start_code[2][AC_N],
    const int16_t ac_huff_start_addr[2][16],
    //
    const uint8_t dc_val[2][12],
    const HCODE_T dc_huff_start_code[2][DC_N],
    const int16_t dc_huff_start_addr[2][16],
    //
    ap_uint<12> hls_cmp,

    // image info
    const uint8_t hls_cs_cmpc, // component count in current scan
    const uint8_t hls_mbs[MAX_NUM_COLOR],
    const uint16_t hls_mcuh, // the horizontal mcu
    const uint32_t hls_mcuc, // the total mcu

    // output
    bool& rtn2,
    uint32_t& rst_cnt,
    hls::stream<ap_uint<24> >& block_strm) {
#pragma HLS DATAFLOW

    // clang-format off
    hls::stream<xf::codec::sos_data> huff_sos_strm;
#pragma HLS DATA_PACK variable = huff_sos_strm
#pragma HLS RESOURCE  variable = huff_sos_strm core = FIFO_LUTRAM
#pragma HLS STREAM    variable = huff_sos_strm depth = 32
    hls::stream<bool> sign_no_huff;
#pragma HLS RESOURCE  variable = sign_no_huff core = FIFO_LUTRAM
#pragma HLS STREAM    variable = sign_no_huff depth = 1

//#pragma HLS RESOURCE        variable = dht_tbl1     core = RAM_2P_LUTRAM
//#pragma HLS ARRAY_PARTITION variable = dht_tbl1 complete dim = 0

#pragma HLS RESOURCE        variable = ac_huff_start_code core = RAM_2P_LUTRAM
#pragma HLS ARRAY_PARTITION variable = ac_huff_start_code complete dim = 2
#pragma HLS RESOURCE        variable = dc_huff_start_code core = RAM_2P_LUTRAM
#pragma HLS ARRAY_PARTITION variable = dc_huff_start_code complete dim = 2

#pragma HLS RESOURCE variable = ac_huff_start_addr core = RAM_2P_LUTRAM
#pragma HLS RESOURCE variable = ac_val             core = RAM_2P_LUTRAM
#pragma HLS RESOURCE variable = dc_huff_start_addr core = RAM_2P_LUTRAM
#pragma HLS RESOURCE variable = dc_val             core = RAM_2P_LUTRAM
    // clang-format on

    xf::codec::details::pick_huff_data(image_strm, eof_strm, rst_cnt, sign_no_huff, huff_sos_strm);

    xf::codec::details::Huffman_decoder(huff_sos_strm, sign_no_huff, dht_tbl1, ac_val, ac_huff_start_code,
                                        ac_huff_start_addr, dc_val, dc_huff_start_code, dc_huff_start_addr, hls_cmp,
#ifndef __SYNTHESIS__
                                        hls_cs_cmpc, hls_mcuh,
#endif
                                        hls_mbs, hls_mcuc, rtn2, block_strm);
}
