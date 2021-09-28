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

#include "kernel3/ans.hpp"

// ------------------------------------------------------------
void ANS_LookupInfo( // input
    const int start,
    const int end,
    hls::stream<hls_Token_symb>& strm_ac_token_reverse,
    const hls_ANSEncSymbolInfo codes[hls_kNumStaticContexts][hls_alphabet_size],
    const uint8_t context_map[hls_kNumContexts],
    const bool is_dc,
    uint8_t dc_context_map[MAX_NUM_COLOR],
    // output
    hls::stream<hls_TokenInfo>& strm_token_info) {
#pragma HLS INLINE OFF
    hls_TokenInfo token_info;
    int size_ac_token = end - start;

    // token index
    for (int i = start; i < end; i++) {
#pragma HLS PIPELINE II = 1
        hls_Token_symb token = strm_ac_token_reverse.read();
        const uint8_t histo_idx = is_dc ? dc_context_map[token.context] : context_map[token.context];
        token_info.info = codes[histo_idx][token.symbol]; // 0~23 * 0~255 = 6144's freq
        strm_token_info.write(token_info);
        // strm_e_info.write(false);
    }
    // strm_e_info.write(true);
}

void ANS_Renormalize( // input
    const int start,
    const int end,
    hls::stream<hls_TokenInfo>& strm_token_info_reverse,
    // output

    hls::stream<uint32_t>& strm_last_state,
    hls_Runbit_t ram_runbit[hls_kANSBufferSize >> 1],
    int& cntInt,
    hls::stream<int>& strm_start,
    hls::stream<int>& strm_end,
    hls::stream<int>& strm_cntInt) {
#pragma HLS INLINE OFF

    // const int start = 0;
    int size_ac_token = end - start;
    ap_uint<32> cnt = 0;
    // private
    uint32_t state_ = (hls_ANS_SIGNATURE << 16);
    uint32_t runbit_reg;

    _XF_IMAGE_PRINT("--1 start ac ANS loop %d- AC_ENCODE\n", (int)size_ac_token);

    // 2. check all nbits = 16 // loop 50243 because the size_ac_token<kANSBufferSize //
    // lookup all the token to form the strm_idx strm_bits
    // short cntInt;

    // for(int j = start; j <size_ac_token ; j+= hls_kANSBufferSize){
    // when the timing fixed the reset could be move into loop, but attention the index

    for (int i = start + 1; i <= end; i++) {
#pragma HLS PIPELINE II = 6
        hls_TokenInfo token_info = strm_token_info_reverse.read();
        // hls_Token_bits token_bit_plain = strm_token_bits.read();
        const hls_ANSEncSymbolInfo info = token_info.info;

        bool do_shift;
        uint16_t bits;

        // AMS_PutSymbol(state_, info, do_shift, bits);//state+info = bool + bits + next_state
        bits = 0;
        do_shift = false;
        if ((state_ >> (32 - hls_ANS_LOG_TAB_SIZE)) >= info.freq_) { // freq(0~1024)
            bits = state_ & 0xffff;
            state_ >>= 16;
            do_shift = true;
        }

// We use mult-by-reciprocal trick, but that requires 64b calc.

#if 1
        // We use mult-by-reciprocal trick, but that requires 64b calc.
        const uint32_t v = (state_ * info.ifreq_) >> hls_RECIPROCAL_PRECISION;
        const uint32_t offset = state_ - v * info.freq_ + info.start_;
        state_ = (v << hls_ANS_LOG_TAB_SIZE) + offset;
#else
        state_ = ((state_ / info.freq_) << ANS_LOG_TAB_SIZE) + (state_ % info.freq_) + info.start_;
#endif

        //_XF_IMAGE_PRINT("--2 check all nbits=16, bits=%d, idx=%d - AC_ENCODE\n", bits, (int)(end-i));
        if (do_shift) {
            _XF_IMAGE_PRINT("--2 check all nbits=16, bits=%d, idx=%d - AC_ENCODE\n", bits, (int)(end - i));
            uint16_t nRenormal = end - i;
            uint32_t tmp = nRenormal;
            tmp = (tmp << 16) | bits;

            if (!cnt[0]) {
                ram_runbit[cnt >> 1] = tmp;
                runbit_reg = tmp;
            } else { // write the same addr but with the new high 32bit
                uint64_t runbit = tmp;
                runbit = ((runbit << 32) | runbit_reg);
                ram_runbit[cnt >> 1] = runbit;
            }
            cnt++;
        }
    } // end i

    // last_state = state_;
    cntInt = cnt;
    strm_start.write(start);
    strm_end.write(end);
    strm_last_state.write(state_);
    strm_cntInt.write(cnt);

    // strm_e_run.write(true);
    _XF_IMAGE_PRINT("--3 nbits = 16 write Tocken, last=%.4x - AC_ENCODE\n", state_);
    _XF_IMAGE_PRINT("--flag cntInt=%d - AC_ENCODE\n", cntInt);
}

// read ram reseve
void ANS_read_ram(const int cntInt,
                  hls_Runbit_t ram_runbit[hls_kANSBufferSize >> 1],
                  hls::stream<hls_Runbit_t2>& strm_runbit) {
#pragma HLS INLINE OFF
    ap_int<18> cnt = cntInt;

    bool is_odd = cnt[0];

    ap_uint<64> runbit_reverse = ram_runbit[(cntInt - 1) >> 1];

    hls_Runbit_t2 runbit = is_odd ? runbit_reverse(31, 0) : runbit_reverse(63, 32);
    strm_runbit.write(runbit);
    is_odd = !is_odd;
    cnt--;

    while (cnt > 0) {
#pragma HLS PIPELINE II = 1

        if (is_odd) {
            runbit = runbit_reverse(31, 0);
        } else {
            runbit_reverse = ram_runbit[(cnt - 2) >> 1];
            runbit = runbit_reverse(63, 32);
        }

        strm_runbit.write(runbit);
        is_odd = !is_odd;
        cnt--;
    }
}

void ANS_enc_Pushbit2( // input
    const int start,
    const int end,
    hls::stream<hls_Token_bits>& strm_token_bit,
    hls::stream<hls_Runbit_t2>& strm_runbit,
    int cntInt,
    uint32_t last_state,
    // output
    uint32_t& pos,

    uint8_t& cnt_buffer,
    uint16_t& reg_buffer,
    hls::stream<uint16_t>& strm_pos_byte,
    hls::stream<bool>& strm_ac_e) {
#pragma HLS INLINE OFF

    _XF_IMAGE_PRINT("--4 nbits < 16 write Tocken - AC_ENCODE\n");

    int size_ac_token = end - start;
    int tokenidx = 0;

    uint16_t shortInt;
    // size_t num_extra_bits = 0;
    ap_int<18> cnt = cntInt; // there is 1 sign and 17 bit to represent 1~1<<16
    bool is_odd = cnt[0];    //  (even 32 | odd 32)= 64bits
    // loop 8668 // write out strm_bits

    // read the end runbit
    ap_uint<32> runbit;
    uint16_t nRenormal = 0;
    uint16_t Renormal_bits;

    //	    if(cntInt>0){
    //	    	runbit = ram_runbit[(cntInt-1)>>1];
    //			Renormal_bits = is_odd ? runbit(15,0) : runbit(47,32);
    //			nRenormal = is_odd ? runbit(31,16) : runbit(63,48);//= next index gap
    //	    	//cnt--;
    //	    	is_odd = !is_odd;
    //	    }
    if (cntInt > 0) {
        runbit = strm_runbit.read();
        Renormal_bits = runbit(15, 0);
        nRenormal = runbit(31, 16);
    }

    // for token loop
    bool is_extra_loop = nRenormal;
    // uint16_t reg_nRenormal = nRenormal;
    int i = 0;

    // for write byte count
    uint64_t storage_ix = 0;
    ap_uint<32> buffer = reg_buffer;
    uint8_t cnt16 = cnt_buffer;

    // pos of the byte
    // uint64_t pos = 32;
    ap_uint<32> l_state = last_state;
    _XF_IMAGE_PRINT("reg buffer = %.4x\n", reg_buffer);
    _XF_IMAGE_PRINT("last state = %.4x\n", last_state);
    // high
    buffer(cnt16 + 15, cnt16) = l_state(31, 16);
    shortInt = buffer(15, 0);
    strm_pos_byte.write(shortInt);
    strm_ac_e.write(false);
    buffer >>= 16;
    _XF_IMAGE_PRINT("state high buffer = %.4x\n", shortInt);
    // low
    buffer(cnt16 + 15, cnt16) = l_state(15, 0);
    shortInt = buffer(15, 0);
    strm_pos_byte.write(shortInt);
    strm_ac_e.write(false);
    buffer >>= 16;
    _XF_IMAGE_PRINT("state low buffer = %.4x\n", shortInt);
    pos += 32;

    while (cnt >= 0) {
#pragma HLS PIPELINE II = 1

        if (is_extra_loop) { // write some extra data ,usually not big than 16bits
            // _XF_IMAGE_PRINT("--4 W from %d to %d - AC_ENCODE\n", tokenidx, nRenormal);

            const hls_Token_bits token = strm_token_bit.read();
            // !!!to improve:checkout the nbits=0 in the front module
            if (token.nbits > 0) buffer(cnt16 + token.nbits - 1, cnt16) = token.bits;

            // write out when there is enough 8 bits
            if (token.nbits + cnt16 >= 16) {
                shortInt = buffer(15, 0);
                cnt16 = cnt16 + token.nbits - 16;
                buffer >>= 16;
                strm_pos_byte.write(shortInt);
                strm_ac_e.write(false);
                // storage_ix += 2;

                if (token.nbits > 0)
                    _XF_IMAGE_PRINT("---W--- n_bits=%ld, bits=%ld, pos=%ld\n", token.nbits, token.bits, pos);
                pos += token.nbits;
            } else {
                cnt16 = cnt16 + token.nbits;

                if (token.nbits > 0)
                    _XF_IMAGE_PRINT("---W--- n_bits=%ld, bits=%ld, pos=%ld, cnt16=%d\n", token.nbits, token.bits, pos,
                                    cnt16);
                pos += token.nbits;
            }
            // num_extra_bits += token.nbits;

            if (i == nRenormal - 1 || (i == size_ac_token - 1)) { // the last time not write out
                tokenidx = i + 1;
                is_extra_loop = false;

                if (!cnt && (i == size_ac_token - 1)) {
                    cnt--;
                }
            }
            i++;

        } else { // is_extra_loop=0 // write a 16bits Renormal_bits

            if (cnt > 0) {
                buffer(cnt16 + 15, cnt16) = Renormal_bits;

                _XF_IMAGE_PRINT("---W--- n_bits=%ld, bits=%ld, pos=%ld\n", 16, Renormal_bits, pos);
                pos += 16;
                // read next renormal
                if (cnt >= 2) {
                    runbit = strm_runbit.read();
                } else {
                    runbit = 0;
                }
                Renormal_bits = runbit(15, 0);
                nRenormal = runbit(31, 16);

                // if(cnt == 1)
                //  nRenormal = end;
                cnt--;

                shortInt = buffer(15, 0);
                buffer >>= 16;
                strm_pos_byte.write(shortInt);
                strm_ac_e.write(false);
                // storage_ix += 2;
            }

            is_extra_loop = true;
            // reg_nRenormal = nRenormal;
        }
    } // end while

    if (cnt16 > 0 && (cnt16 <= 8)) { // can we change cnt16 to pos?

        reg_buffer = buffer(15, 0);
        cnt_buffer = cnt16;
        _XF_IMAGE_PRINT("1 reg buffer = %.4x\n", reg_buffer);
    } else if (cnt16 > 8 && (cnt16 < 16)) {
        // shortInt = buffer( 7, 0 );
        // strm_pos_byte.write(shortInt);
        // buffer >>= 8;

        reg_buffer = buffer(15, 0);
        cnt_buffer = cnt16;
        _XF_IMAGE_PRINT("2 reg buffer = %.4x, cnt_buffer=%d\n", reg_buffer, cnt_buffer);
    }
}

void ANS_runbitram( // input
    const int start,
    const int end,
    hls::stream<hls_Token_symb>& strm_ac_token_reverse,
    const hls_ANSEncSymbolInfo codes[hls_kNumStaticContexts][hls_alphabet_size],
    const uint8_t context_map[hls_kNumContexts],
    const bool is_dc,
    uint8_t dc_context_map[MAX_NUM_COLOR],
    // output
    hls::stream<int>& strm_start,
    hls::stream<int>& strm_end,
    hls::stream<int>& strm_cntInt,
    int& cntInt,
    hls_Runbit_t ram_runbit[hls_kANSBufferSize >> 1],
    hls::stream<uint32_t>& strm_last_state) {
#pragma HLS INLINE OFF
#pragma HLS DATAFLOW

    // clang-format off
  hls::stream< hls_TokenInfo > strm_token_info_reverse;
#pragma HLS DATA_PACK 	  variable = strm_token_info_reverse
#pragma HLS RESOURCE  	  variable = strm_token_info_reverse core = FIFO_LUTRAM
#pragma HLS STREAM    	  variable = strm_token_info_reverse depth = 32
    // clang-format on

    ANS_LookupInfo(start, end, strm_ac_token_reverse, codes, context_map, is_dc, dc_context_map,
                   strm_token_info_reverse);

    ANS_Renormalize(start, end, strm_token_info_reverse, strm_last_state, ram_runbit, cntInt, strm_start, strm_end,
                    strm_cntInt);
}
// ------------------------------------------------------------
void ANS_runbitram2( // input
    const int ndataflow,
    const int start[3],
    const int end[3],
    hls::stream<hls_Token_symb>& strm_ac_token_reverse,
    const hls_ANSEncSymbolInfo codes[hls_kNumStaticContexts][hls_alphabet_size],
    const uint8_t context_map[hls_kNumContexts],
    const bool is_dc,
    uint8_t dc_context_map[MAX_NUM_COLOR],
    // output
    hls::stream<int>& strm_start,
    hls::stream<int>& strm_end,
    hls::stream<int>& strm_cntInt,

    hls::stream<hls_Runbit_t2>& strm_runbit,
    hls::stream<uint32_t>& strm_last_state) {
#pragma HLS INLINE OFF

LOOP_ANSBUFFER:
    for (int i = 0; i < ndataflow; i++) { // 0~65535 65536~..
#pragma HLS DATAFLOW

        // clang-format off
	  hls::stream< hls_TokenInfo > strm_token_info_reverse;
	#pragma HLS DATA_PACK 	  variable = strm_token_info_reverse
	#pragma HLS RESOURCE  	  variable = strm_token_info_reverse core = FIFO_LUTRAM
	#pragma HLS STREAM    	  variable = strm_token_info_reverse depth = 32
// clang-format on

#ifndef __SYNTHESIS__
        hls_Runbit_t* ram_runbit;
        ram_runbit = (hls_Runbit_t*)malloc((hls_kANSBufferSize >> 1) * sizeof(hls_Runbit_t));
#else
        hls_Runbit_t ram_runbit[hls_kANSBufferSize >> 1];
// or remove this pargma use bram instead
#pragma HLS RESOURCE variable = ram_runbit core = XPM_MEMORY uram
#endif

        int cntInt = 0;

        ANS_runbitram(start[i], end[i], strm_ac_token_reverse, codes, context_map, is_dc, dc_context_map, strm_start,
                      strm_end, strm_cntInt, cntInt, ram_runbit, strm_last_state);

        ANS_read_ram(cntInt, ram_runbit, strm_runbit);
    }
}

// ------------------------------------------------------------

void ANS_Rpushbit( // input
    const int start,
    const int end,
    hls::stream<hls_Token_bits>& strm_token_bit,
    hls_Runbit_t ram_runbit[hls_kANSBufferSize >> 1],
    int cntInt,
    uint32_t last_state,
    // output
    uint32_t& pos,
    // uint32_t& num_extra_bits,

    uint8_t& cnt_buffer,
    uint16_t& reg_buffer,
    // hls_PikImageSizeInfo& pik_info,
    hls::stream<uint16_t>& strm_pos_byte,
    hls::stream<bool>& strm_ac_e) { // structure
#pragma HLS INLINE OFF
#pragma HLS DATAFLOW
    // clang-format off
	hls::stream< hls_Runbit_t2 > strm_runbit;
#pragma HLS RESOURCE  	  variable = strm_runbit core = FIFO_LUTRAM
#pragma HLS STREAM    	  variable = strm_runbit depth = 32
    // clang-format on

    ANS_read_ram(cntInt, ram_runbit, strm_runbit);

    ANS_enc_Pushbit2(start, end, strm_token_bit, strm_runbit, cntInt, last_state, pos, cnt_buffer, // num_extra_bits,
                     reg_buffer, strm_pos_byte, strm_ac_e);                                        // pik_info,
}

// pingpang ram_runbit
void hls_InitEnd(const int total_token, int& ndataflow, int start[3], int end[3]) {
#pragma HLS INLINE OFF
    int cnt = 0;
LOOP_CONST_END:
    for (int i = 0; i < total_token; i += hls_kANSBufferSize) { // 0~65535 65536~..
#pragma HLS PIPELINE II = 1
        int left = total_token - i;
        start[cnt] = i;
        end[cnt] = (hls_kANSBufferSize <= left) ? (i + hls_kANSBufferSize) : total_token;
        cnt++;
    }
    ndataflow = cnt;
}

void ANS_enc_Pushbit3( // input
    const int ndataflow,
    hls::stream<int>& strm_start,
    hls::stream<int>& strm_end,
    hls::stream<hls_Token_bits>& strm_token_bit,
    hls::stream<hls_Runbit_t2>& strm_runbit,
    hls::stream<int>& strm_cntInt,
    hls::stream<uint32_t>& strm_last_state,
    // output
    uint32_t& pos,
    uint8_t& cnt_buffer,
    uint16_t& reg_buffer,
    hls::stream<uint16_t>& strm_pos_byte,
    hls::stream<bool>& strm_ac_e) {
PUSH_LOOP_ANSBUFFER:
    for (int i = 0; i < ndataflow; i++) { // 0~65535 65536~..
#pragma HLS DATAFLOW
        int start = strm_start.read();
        int end = strm_end.read();
        int cntInt = strm_cntInt.read();
        uint32_t last_state = strm_last_state.read();

        ANS_enc_Pushbit2(start, end, strm_token_bit, strm_runbit, cntInt, last_state, pos,
                         cnt_buffer, // num_extra_bits,
                         reg_buffer, strm_pos_byte, strm_ac_e);
    }
}
// ------------------------------------------------------------
// pingpang ram_runbit
void XAcc_WriteTokens_wapper3(

    // input
    const int total_token,
    const int ndataflow,
    const int start[3],
    const int end[3],
    hls::stream<hls_Token_symb>& strm_ac_token_reverse,
    hls::stream<hls_Token_bits>& strm_token_bit,
    hls_ANSEncSymbolInfo hls_codes[hls_kNumStaticContexts][hls_alphabet_size],
    uint8_t ac_static_context_map[hls_kNumContexts], // table
    const bool is_dc,
    uint8_t dc_context_map[MAX_NUM_COLOR],

    // output
    uint32_t& ans_pos,
    uint8_t& cnt_buffer,
    uint16_t& reg_buffer,
    hls::stream<uint16_t>& strm_ac_byte,
    hls::stream<bool>& strm_ac_e) {
#pragma HLS INLINE OFF
#pragma HLS DATAFLOW

    // clang-format off
		hls::stream< hls_Runbit_t2 > strm_runbit;
#pragma HLS DATA_PACK 	  variable = strm_runbit
#pragma HLS RESOURCE  	  variable = strm_runbit core = FIFO_LUTRAM
#pragma HLS STREAM    	  variable = strm_runbit depth = 32
			   hls::stream<uint32_t> strm_last_state;
#pragma HLS STREAM    	  variable = strm_last_state depth = 32
					hls::stream<int> strm_cntInt;
#pragma HLS STREAM    	  variable = strm_cntInt depth = 32
					hls::stream<int> strm_start;
#pragma HLS STREAM    	  variable = strm_start depth = 32
					hls::stream<int> strm_end;
#pragma HLS STREAM    	  variable = strm_end depth = 32
    // clang-format on

    ANS_runbitram2(ndataflow, start, end, strm_ac_token_reverse, hls_codes, ac_static_context_map, is_dc,
                   dc_context_map, strm_start, strm_end, strm_cntInt, strm_runbit, strm_last_state);

    ANS_enc_Pushbit3(ndataflow, strm_start, strm_end, strm_token_bit, strm_runbit, strm_cntInt, strm_last_state,
                     ans_pos, cnt_buffer, reg_buffer, strm_ac_byte, strm_ac_e); // pik_info,
}

// ------------------------------------------------------------
// pingpang ram_runbit
void hls_WriteTokensTop(

    // input
    const int total_token,
    hls::stream<hls_Token_symb>& strm_ac_token_reverse,
    hls::stream<hls_Token_bits>& strm_token_bit,
    hls_ANSEncSymbolInfo hls_codes[hls_kNumStaticContexts][hls_alphabet_size],
    uint8_t ac_static_context_map[hls_kNumContexts], // table
    const bool is_dc,
    uint8_t dc_context_map[MAX_NUM_COLOR],

    // output
    int& len_ac,
    hls::stream<uint16_t>& strm_ac_byte,
    hls::stream<bool>& strm_ac_e) {
#pragma HLS INLINE OFF

    uint32_t ans_pos = 0;
    uint16_t reg_buffer = 0;
    uint8_t cnt_buffer = 0;
    int ndataflow;
    int start[3];
    int end[3]; // 3 is Empirical values measured from groups with a lot of detail
    hls_InitEnd(total_token, ndataflow, start, end);

    XAcc_WriteTokens_wapper3(total_token, ndataflow, start, end, strm_ac_token_reverse, strm_token_bit, hls_codes,
                             ac_static_context_map, is_dc, dc_context_map, ans_pos, cnt_buffer, reg_buffer,
                             strm_ac_byte, strm_ac_e);

    if (cnt_buffer != 0) {
        strm_ac_byte.write(reg_buffer);
        strm_ac_e.write(false);
    }
    strm_ac_e.write(true);

    len_ac = (ans_pos + 7) >> 3;
}
