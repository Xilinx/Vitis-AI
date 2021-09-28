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

#include "pikEnc/XAccPIKKernel3.hpp"
#include "xf_utils_hw/axi_to_stream.hpp"
#include "xf_utils_hw/stream_to_axi.hpp"

// ------------------------------------------------------------
uint8_t ac_static_context_map[hls_kNumContexts] = {
    0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,
    1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,
    2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,
    0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  3,  3,  4,  4,  4,  4,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,
    7,  7,  7,  7,  4,  4,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  7,  7,  7,  7,  4,  4,  5,  5,  5,  5,  5,  6,
    6,  6,  6,  6,  8,  8,  8,  8,  8,  8,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  9,  9,  9,  9,  8,  8,  5,  5,
    5,  5,  5,  6,  6,  6,  6,  9,  9,  9,  9,  8,  8,  5,  5,  5,  5,  5,  6,  6,  6,  9,  9,  9,  9,  8,  8,  5,
    5,  5,  5,  5,  3,  10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 11, 11, 12,
    12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 15, 15, 15,
    15, 15, 15, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 16, 16, 16, 16, 15, 15, 12, 12, 12, 12, 12, 13, 13, 13, 13,
    16, 16, 16, 16, 15, 15, 12, 12, 12, 12, 12, 13, 13, 13, 16, 16, 16, 16, 15, 15, 12, 12, 12, 12, 12, 10, 17, 17,
    18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 21, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20,
    20, 20, 21, 21, 21, 21, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 22, 22, 22, 22, 22, 22, 19, 19, 19, 19,
    19, 20, 20, 20, 20, 20, 23, 23, 23, 23, 22, 22, 19, 19, 19, 19, 19, 20, 20, 20, 20, 23, 23, 23, 23, 22, 22, 19,
    19, 19, 19, 19, 20, 20, 20, 23, 23, 23, 23, 22, 22, 19, 19, 19, 19, 19, 17,
};
//----------------------------------------------------------
void buffer_tokens_onboard(
    // const int total_token,
    hls::stream<hls_Token_symb>& strm_token_symb,
    hls::stream<hls_Token_bits>& strm_token_bits,
    hls::stream<bool>& strm_e_token,

    int& total_token,
    ap_uint<72> ram_symb[hls_kMaxBufSize], // hls_Token_symb*3
    ap_uint<72> ram_bits[hls_kMaxBufSize]  // hls_Token_bits*3
    ) {
#pragma HLS INLINE OFF

    // for(int i=0; i<total_token; i++){
    ap_uint<2> cnt3 = 0;
    ap_uint<72> buffer_symb;
    ap_uint<72> buffer_bits;

    // total_token = 0;
    int addr = 0;
    bool e = strm_e_token.read();
    while (!e) {
#pragma HLS PIPELINE II = 1
        e = strm_e_token.read();

        hls_Token_symb token_symb = strm_token_symb.read();
        hls_Token_bits token_bits = strm_token_bits.read();

        ap_uint<16> context = token_symb.context;
        ap_uint<8> symbol = token_symb.symbol;
        ap_uint<16> bits = token_bits.bits;
        ap_uint<8> nbits = token_bits.nbits;

        buffer_symb(24 * (cnt3 + 1) - 1, 24 * cnt3) = (context, symbol);
        buffer_bits(24 * (cnt3 + 1) - 1, 24 * cnt3) = (nbits, bits);

        // write to ram_token
        if (cnt3 == 2) {
            ram_symb[addr] = buffer_symb;
            ram_bits[addr] = buffer_bits;
            addr++;
            cnt3 = 0;
        } else {
            cnt3++;
        }
    }
    if (cnt3 > 0) {
        ram_symb[addr] = buffer_symb;
        ram_bits[addr] = buffer_bits;
    }

    total_token = addr * 3 + cnt3;
}

//----------------------------------------------------------
void read_token_symb(const int total_token,
                     ap_uint<72> ram_symb[hls_kMaxBufSize], // hls_Token_symb*3

                     hls::stream<hls_Token_symb>& strm_ac_token_reverse) {
#pragma HLS INLINE OFF
    for (int i = 0; i < total_token; i += hls_kANSBufferSize) {
        int left = total_token - i;
        int end = hls_kANSBufferSize <= left ? (i + hls_kANSBufferSize) : total_token;

        int addr_reverse = (end) / 3;

        ap_uint<72> buffer_symb;

        int tmp = end - addr_reverse * 3; // 1,2,0
        ap_uint<2> cnt_r;
        if (!tmp) {
            cnt_r = 2;
            buffer_symb = ram_symb[addr_reverse - 1];
            addr_reverse -= 2;
        } else {
            cnt_r = tmp - 1;
            buffer_symb = ram_symb[addr_reverse];
            addr_reverse--;
        }

        _XF_IMAGE_PRINT("---cnt_r = %d, addr_reverse =%d, start=%d, end=%d\n", (int)cnt_r, addr_reverse, i, end);

        for (int j = i; j < end; j++) {
#pragma HLS PIPELINE II = 1
            // reverse sequence
            ap_uint<24> token_symb = buffer_symb(24 * (cnt_r + 1) - 1, 24 * cnt_r);
            hls_Token_symb ac_token_reverse;
            ac_token_reverse.context = token_symb(23, 8);
            ac_token_reverse.symbol = token_symb(7, 0);
            strm_ac_token_reverse.write(ac_token_reverse);

            if (cnt_r == 0) {
                if (addr_reverse >= 0) {
                    buffer_symb = ram_symb[addr_reverse];
                }
                addr_reverse--;
                cnt_r = 2;
            } else {
                cnt_r--;
            }
        }

        _XF_IMAGE_PRINT("last---cnt_r = %d, addr_reverse =%d, start=%d, end=%d\n", (int)cnt_r, addr_reverse, i, end);
    }
}

//----------------------------------------------------------
void read_token_bits(const int total_token,
                     ap_uint<72> ram_bits[hls_kMaxBufSize], // hls_Token_bits*3

                     hls::stream<hls_Token_bits>& strm_token_bit) {
#pragma HLS INLINE OFF
    for (int i = 0; i < total_token; i += hls_kANSBufferSize) {
        int left = total_token - i;
        int end = hls_kANSBufferSize <= left ? (i + hls_kANSBufferSize) : total_token;

        int addr = i / 3;
        ap_uint<72> buffer_bits;

        int tmp3 = i - addr * 3;
        ap_uint<2> cnt3 = tmp3;
        _XF_IMAGE_PRINT("---  cnt3=%d, start=%d, end=%d\n", (int)cnt3, i, end);

        buffer_bits = ram_bits[addr];
        addr++;

        for (int j = i; j < end; j++) {
#pragma HLS PIPELINE II = 1
            ap_uint<24> token_bits = buffer_bits(24 * (cnt3 + 1) - 1, 24 * cnt3);
            hls_Token_bits token_bit_plain;
            token_bit_plain.nbits = token_bits(23, 16);
            token_bit_plain.bits = token_bits(15, 0);
            strm_token_bit.write(token_bit_plain);

            if (cnt3 == 2) {
                buffer_bits = ram_bits[addr];
                addr++;
                cnt3 = 0;
            } else {
                cnt3++;
            }
        }

        _XF_IMAGE_PRINT("last---cnt3 = %d,  addr         =%d, start=%d, end=%d\n", (int)cnt3, addr, i, end);
    }
}

//----------------------------------------------------------
void ANS_top(const bool is_dc,
             uint8_t dc_context_map[MAX_NUM_COLOR],
             const int total_token,
             ap_uint<72> ram_symb[hls_kMaxBufSize], // hls_Token_symb*3
             ap_uint<72> ram_bits[hls_kMaxBufSize], // hls_Token_bits*3
             hls_ANSEncSymbolInfo hls_codes[hls_kNumStaticContexts][MAX_ALPHABET_SIZE],

             // uint32_t& num_extra_bits,
             int& len_ac,
             hls::stream<uint16_t>& strm_ac_dc_byte,
             hls::stream<bool>& strm_ac_dc_e) {
#pragma HLS INLINE OFF
#pragma HLS DATAFLOW

    hls::stream<hls_Token_symb> strm_ac_token_reverse;
#pragma HLS RESOURCE variable = strm_ac_token_reverse core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_ac_token_reverse depth = 32

    hls::stream<hls_Token_bits> strm_token_bit;
#pragma HLS RESOURCE variable = strm_token_bit core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_token_bit depth = 32

    read_token_symb(total_token, ram_symb, strm_ac_token_reverse);

    read_token_bits(total_token, ram_bits, strm_token_bit);

    hls_WriteTokensTop(total_token, strm_ac_token_reverse, strm_token_bit, hls_codes, ac_static_context_map, is_dc,
                       dc_context_map, len_ac, strm_ac_dc_byte, strm_ac_dc_e);
}

// ------------------------------------------------------------

// ------------------------------------------------------------

#define DEBUGCONFIG
void load_config_kernel3(ap_uint<32> in[MAX_NUM_CONFIG], ConfigKernel3 config[4]) {
#pragma HLS INLINE OFF

    ap_uint<32> tmp[MAX_NUM_CONFIG];
    for (int i = 0; i < MAX_NUM_CONFIG; i++) {
#pragma HLS PIPELINE II = 1
        tmp[i] = in[i];
    }

    for (int i = 0; i < 4; i++) {
#pragma HLS PIPELINE II = 1
        config[i].xsize = tmp[0];
        config[i].ysize = tmp[1];
        config[i].xblock8 = tmp[2];
        config[i].yblock8 = tmp[3];
        config[i].xblock32 = tmp[4];
        config[i].yblock32 = tmp[5];
        config[i].xblock64 = tmp[6];
        config[i].yblock64 = tmp[7];
        config[i].ac_xgroup = tmp[8];
        config[i].ac_ygroup = tmp[9];
        config[i].dc_xgroup = tmp[10];
        config[i].dc_ygroup = tmp[11];
        config[i].ac_group = tmp[12];
        config[i].dc_group = tmp[13];
        config[i].num_dc = tmp[14];
        config[i].num_ac = tmp[15];
    }

#ifndef __SYNTHESIS__
#ifdef DEBUGCONFIG
    std::cout << "k3 Config:" << std::endl;
    std::cout << "xsize:" << tmp[0] << std::endl;
    std::cout << "ysize:" << tmp[1] << std::endl;
    std::cout << "xblock8:" << tmp[2] << std::endl;
    std::cout << "yblock8:" << tmp[3] << std::endl;
    std::cout << "xblock32:" << tmp[4] << std::endl;
    std::cout << "yblock32:" << tmp[5] << std::endl;
    std::cout << "xblock64:" << tmp[6] << std::endl;
    std::cout << "yblock64:" << tmp[7] << std::endl;
    std::cout << "ac_xgroup:" << tmp[8] << std::endl;
    std::cout << "ac_ygroup:" << tmp[9] << std::endl;
    std::cout << "dc_xgroup:" << tmp[10] << std::endl;
    std::cout << "dc_ygroup:" << tmp[11] << std::endl;
    std::cout << "ac_group:" << tmp[12] << std::endl;
    std::cout << "dc_group:" << tmp[13] << std::endl;
    std::cout << "num_dc:" << tmp[14] << std::endl;
    std::cout << "num_ac:" << tmp[15] << std::endl;
#endif
#endif
};

template <typename _IStrm, typename _TStrm>
void streamRetype(hls::stream<_IStrm>& istrm, hls::stream<bool>& strm_e_in, hls::stream<_TStrm>& ostrm) {
#ifndef __SYNTHESIS__
    int addr = 0;
#endif

    bool e = strm_e_in.read();
    while (!e) {
#pragma HLS PIPELINE II = 1
        e = strm_e_in.read();
        _IStrm in = istrm.read();
        _TStrm out = (_TStrm)in;
        ostrm.write(out);
    }
}

template <typename _IStrm, typename _TStrm>
void streamRetype(hls::stream<_IStrm>& istrm,
                  hls::stream<bool>& strm_e_in,
                  hls::stream<_TStrm>& ostrm,
                  hls::stream<bool>& strm_e_o) {
    bool e = strm_e_in.read();
    while (!e) {
#pragma HLS PIPELINE II = 1
        e = strm_e_in.read();
        _IStrm in = istrm.read();
        _TStrm out = (_TStrm)in;
        ostrm.write(out);
        strm_e_o.write(false);
    }
    strm_e_o.write(true);
}

template <int _BurstLen, int _WAxi1, typename _TStrm1, typename _RStrm1>
void axiToStreamRetype(ap_uint<_WAxi1>* rbuf1, const int num1, hls::stream<_RStrm1>& ostrm1) {
#pragma HLS INLINE OFF
#pragma HLS DATAFLOW

    hls::stream<_TStrm1> strm_in1("strm_in1");
#pragma HLS STREAM variable = strm_in1 depth = 32
    hls::stream<bool> strm_e_in1("strm_e_in1");
#pragma HLS STREAM variable = strm_e_in1 depth = 32

    xf::common::utils_hw::axiToStream<_BurstLen, _WAxi1, _TStrm1>(rbuf1, num1, strm_in1, strm_e_in1);

    streamRetype<_TStrm1, _RStrm1>(strm_in1, strm_e_in1, ostrm1);
}

template <int _BurstLen, int _WAxi1, typename _RStrm1, bool _Padd1, int _WAxi2, typename _RStrm2, bool _Padd2>
void axiToStreamRetype_serial(ap_uint<_WAxi1>* rbuf1,
                              ap_uint<_WAxi2>* rbuf2,
                              const int num,
                              hls::stream<_RStrm1>& ostrm1,
                              hls::stream<_RStrm2>& ostrm2) {
#pragma HLS INLINE off
#pragma HLS FUNCTION_INSTANTIATE variable = ostrm1

    const int loop_num = num / _BurstLen;
    const int fraction = num % _BurstLen;

    int addr1 = 0;
    int addr2 = 0;
base:
    for (int i = 0; i < loop_num; i++) {
        for (int j = 0; j < _BurstLen; j++) {
#pragma HLS PIPELINE II = 1

            _RStrm1 tmp1;
            if (_Padd1)
                tmp1 = 0;
            else
                tmp1 = rbuf1[addr1];
            ostrm1.write(tmp1);
#ifndef __SYNTHESIS__
            std::cout << "base: addr1=" << addr1 << " strm1=" << (int)tmp1 << std::endl;
#endif
            addr1++;
        }

        for (int j = 0; j < _BurstLen; j++) {
#pragma HLS PIPELINE II = 1

            _RStrm2 tmp2;
            if (_Padd2)
                tmp2 = 0;
            else
                tmp2 = rbuf2[addr2];
#ifndef __SYNTHESIS__
            std::cout << "base: addr2=" << addr2 << " strm2=" << (int)tmp2 << std::endl;
#endif
            ostrm2.write(tmp2);
            addr2++;
        }
    }

fraction1:
    for (int i = 0; i < fraction; i++) {
#pragma HLS PIPELINE II = 1

        _RStrm1 tmp1;
        if (_Padd1)
            tmp1 = 0;
        else
            tmp1 = rbuf1[addr1];
#ifndef __SYNTHESIS__
        std::cout << "fraction: addr1=" << addr1 << " strm1=" << (int)tmp1 << std::endl;
#endif
        ostrm1.write(tmp1);
        addr1++;
    }
fraction2:
    for (int i = 0; i < fraction; i++) {
#pragma HLS PIPELINE II = 1

        _RStrm2 tmp2;
        if (_Padd2)
            tmp2 = 0;
        else
            tmp2 = rbuf2[addr2];
#ifndef __SYNTHESIS__
        std::cout << "fraction: addr2=" << addr2 << " strm2=" << (int)tmp2 << std::endl;
#endif
        ostrm2.write(tmp2);
        addr2++;
    }
}

template <int _BurstLen, int _WAxi1, typename _RStrm1, bool _Padd1, int _WAxi2, typename _RStrm2, bool _Padd2>
void axiToStreamRetype_parallel(ap_uint<_WAxi1>* rbuf1,
                                ap_uint<_WAxi2>* rbuf2,
                                const int num,
                                hls::stream<_RStrm1>& ostrm1,
                                hls::stream<_RStrm2>& ostrm2) {
#pragma HLS INLINE off

    const int loop_num = num / _BurstLen;
    const int fraction = num % _BurstLen;

    int addr1 = 0;
    int addr2 = 0;
base:
    for (int i = 0; i < loop_num; i++) {
        for (int j = 0; j < _BurstLen; j++) {
#pragma HLS PIPELINE II = 1

            _RStrm1 tmp1;
            if (_Padd1)
                tmp1 = 0;
            else
                tmp1 = rbuf1[addr1];
            ostrm1.write(tmp1);
#ifndef __SYNTHESIS__
            std::cout << "base: addr1=" << addr1 << " strm1=" << (int)tmp1 << std::endl;
#endif
            addr1++;

            _RStrm2 tmp2;
            if (_Padd2)
                tmp2 = 0;
            else
                tmp2 = rbuf2[addr2];
#ifndef __SYNTHESIS__
            std::cout << "base: addr2=" << addr2 << " strm2=" << (int)tmp2 << std::endl;
#endif
            ostrm2.write(tmp2);
            addr2++;
        }
    }

fraction:
    for (int i = 0; i < fraction; i++) {
#pragma HLS PIPELINE II = 1

        _RStrm1 tmp1;
        if (_Padd1)
            tmp1 = 0;
        else
            tmp1 = rbuf1[addr1];
#ifndef __SYNTHESIS__
        std::cout << "fraction: addr1=" << addr1 << " strm1=" << (int)tmp1 << std::endl;
#endif
        ostrm1.write(tmp1);
        addr1++;

        _RStrm2 tmp2;
        if (_Padd2)
            tmp2 = 0;
        else
            tmp2 = rbuf2[addr2];
#ifndef __SYNTHESIS__
        std::cout << "fraction: addr2=" << addr2 << " strm2=" << (int)tmp2 << std::endl;
#endif
        ostrm2.write(tmp2);
        addr2++;
    }
}

// ------------------------------------------------------------
template <int _WAxi, int _BurstLen>
void axiToPikAcStream(const ap_uint<_WAxi>* rbuf,
                      ConfigKernel3 config_dev,

                      hls::stream<dct_t>& strm_ac) {
#pragma HLS INLINE OFF

    const int hls_ac_groups = config_dev.ac_group;
    const int xsize_blocks = config_dev.xblock8;
    const int ysize_blocks = config_dev.yblock8;
    const int xsize_tiles = config_dev.xblock64;
    const int ysize_tiles = config_dev.yblock64;
    const int element_size = config_dev.num_dc * 64;
    int cnt = 0;

    for (int gy = 0; gy < config_dev.ac_ygroup; ++gy) {
        for (int gx = 0; gx < config_dev.ac_xgroup; ++gx) {
            hls_Rect group_rect;
            group_rect.x0 = gx * 8;
            group_rect.y0 = gy * 8;
            group_rect.xsize = (group_rect.x0 + 8 <= xsize_tiles) ? 8 : (xsize_tiles - group_rect.x0);
            group_rect.ysize = (group_rect.y0 + 8 <= ysize_tiles) ? 8 : (ysize_tiles - group_rect.y0);

            for (int tby = 0; tby < group_rect.ysize; ++tby) {
                for (int tbx = 0; tbx < group_rect.xsize; ++tbx) {
                    hls_Rect tile_rect;
                    tile_rect.x0 = (group_rect.x0 + tbx) * hls_kTileDimInBlocks;
                    tile_rect.y0 = (group_rect.y0 + tby) * hls_kTileDimInBlocks;

                    tile_rect.xsize = (tile_rect.x0 + hls_kTileDimInBlocks <= xsize_blocks)
                                          ? hls_kTileDimInBlocks
                                          : (xsize_blocks - tile_rect.x0);
                    tile_rect.ysize = (tile_rect.y0 + hls_kTileDimInBlocks <= ysize_blocks)
                                          ? hls_kTileDimInBlocks
                                          : (ysize_blocks - tile_rect.y0);

                    _XF_IMAGE_PRINT("-5 debug the axi(%d,%d... ,%d,%d) - E2B\n", tile_rect.x0, tile_rect.y0,
                                    tile_rect.xsize, tile_rect.ysize);

                    for (int c = 0; c < MAX_NUM_COLOR; ++c) {
                        for (int by = 0; by < tile_rect.ysize; ++by) {
#pragma HLS DATAFLOW
                            const ap_uint<_WAxi>* vec_ptr = rbuf + element_size * c +
                                                            xsize_blocks * (by + tby * 8 + gy * 64) * 64 + gx * 4096 +
                                                            8 * tbx * 64;
                            ap_uint<_WAxi> row_tile_ram[512]; // 512 = 8*64 for one row of a tile

                            for (int n = 0; n < 512; n++) {
#pragma HLS loop_tripcount min = 1 max = 1
#pragma HLS PIPELINE II = 1
                                row_tile_ram[n] = vec_ptr[n];
                            }

                            for (int bx = 0; bx < tile_rect.xsize; bx++) { // OUTPUT
                                for (ap_uint<7> i = 0; i < 64; i++) {
#pragma HLS loop_tripcount min = 1 max = 1
#pragma HLS PIPELINE II = 1
                                    dct_t ac = row_tile_ram[bx * 64 + i];
                                    //_XF_IMAGE_PRINT("%d,", (int)ac);
                                    strm_ac.write(ac);
                                    cnt++;
                                }
                                //_XF_IMAGE_PRINT("\n");
                            }
                        } // dataflow region
                    }     // color
                }         // tile x
            }             // tile y
        }                 // group x
    }                     // group y
#ifdef DEBUGAXItoPikAcStream
    std::cout << "all_ac = " << cnt << std::endl;
#endif
}

// ------------------------------------------------------------

void axiToPikAcwithOrder(ap_uint<32>* ac_buf,
                         ap_uint<32>* order_buf,
                         ConfigKernel3 config_dev,

                         hls::stream<dct_t>& strm_ac,
                         hls::stream<int>& strm_order) {
#pragma HLS INLINE OFF
#pragma HLS DATAFLOW
    axiToPikAcStream<32, 512>(ac_buf, config_dev, strm_ac); // 512 for 32w input//256 for 16w input

    axiToStreamRetype<128, 32, ap_uint<32>, int>(order_buf, (config_dev.ac_group * 3 * 64), strm_order);
}
// ------------------------------------------------------------
void localOrder(hls::stream<int>& strm_order, int hls_order[hls_kOrderContexts][64]) {
    // std::cout<<"k3_order:"<<std::endl;
    for (int i = 0; i < hls_kOrderContexts; i++) {
        for (int j = 0; j < 64; j++) {
#pragma HLS PIPELINE II = 1
            int tmp = strm_order.read();
            hls_order[i][j] = tmp;
            // std::cout<<tmp<<",";
        }
    }
    // std::cout<<std::endl;
}

// ------------------------------------------------------------
// sequential
void hls_burst_dc_ac_by_group2(ConfigKernel3 config_dev,

                               ap_uint<32>* ddr_dc,
                               ap_uint<32>* ddr_ac_strategy,
                               ap_uint<32>* ddr_block,
                               ap_uint<32>* ddr_quant_field,
                               ap_uint<32>* ddr_ac,
                               ap_uint<32>* ddr_order,

                               hls::stream<dct_t>& strm_dc_y1,
                               hls::stream<dct_t>& strm_dc_y2,
                               hls::stream<dct_t>& strm_dc_y3,
                               hls::stream<dct_t>& strm_dc_x,
                               hls::stream<dct_t>& strm_dc_b,

                               hls::stream<uint8_t>& strm_strategy,
                               hls::stream<quant_t>& strm_quant_field,
                               hls::stream<arsigma_t>& strm_arsigma,
                               hls::stream<bool>& strm_strategy_block0,
                               hls::stream<bool>& strm_strategy_block1,
                               hls::stream<bool>& strm_strategy_block2,

                               hls::stream<dct_t>& strm_ac,
                               hls::stream<int>& strm_order) {
#pragma HLS INLINE OFF
    assert(hls_kDcGroupDimInBlocks % (hls_kGroupDim / 8) == 0);

    ap_uint<32>* ddr_dc_x = ddr_dc;
    ap_uint<32>* ddr_dc_y = ddr_dc + config_dev.num_dc;
    ap_uint<32>* ddr_dc_b = ddr_dc + 2 * config_dev.num_dc;

    for (int i = 0; i < 7; i++) {
        if (i == 0)
            axiToStreamRetype_serial<64, 32, dct_t, false, 32, dct_t, false>(ddr_dc_x, ddr_dc_y, config_dev.num_dc,
                                                                             strm_dc_x, strm_dc_y1);
        else if (i == 1)
            axiToStreamRetype<64, 32, ap_uint<32>, dct_t>(ddr_dc_y, config_dev.num_dc, strm_dc_y2);
        else if (i == 2)
            axiToStreamRetype_serial<64, 32, dct_t, false, 32, dct_t, false>(ddr_dc_b, ddr_dc_y, config_dev.num_dc,
                                                                             strm_dc_b, strm_dc_y3);
        else if (i == 3)
            axiToStreamRetype_parallel<64, 32, uint8_t, false, 32, bool, false>(
                ddr_ac_strategy, ddr_block, config_dev.num_dc, strm_strategy, strm_strategy_block0);
        else if (i == 4)
            axiToStreamRetype_parallel<64, 32, quant_t, false, 32, bool, false>(
                ddr_quant_field, ddr_block, config_dev.num_dc, strm_quant_field, strm_strategy_block1);
        else if (i == 5)
            axiToStreamRetype_parallel<64, 32, arsigma_t, true, 32, bool, false>(nullptr, ddr_block, config_dev.num_dc,
                                                                                 strm_arsigma, strm_strategy_block2);
        else
            axiToPikAcwithOrder(ddr_ac, ddr_order, config_dev, strm_ac, strm_order);
    }
}

void hls_tokenize_dc_ac_sequential(ConfigKernel3 config_dev,

                                   // ac
                                   hls::stream<dct_t>& strm_coef_raster,
                                   hls::stream<int>& strm_order,

                                   // dc
                                   hls::stream<dct_t>& strm_dc_y1,
                                   hls::stream<dct_t>& strm_dc_y2,
                                   hls::stream<dct_t>& strm_dc_y3,
                                   hls::stream<dct_t>& strm_dc_x,
                                   hls::stream<dct_t>& strm_dc_b,

                                   hls::stream<uint8_t>& strm_strategy,
                                   hls::stream<quant_t>& strm_quant_field,
                                   hls::stream<arsigma_t>& strm_arsigma,
                                   hls::stream<bool>& strm_strategy_block0,
                                   hls::stream<bool>& strm_strategy_block1,
                                   hls::stream<bool>& strm_strategy_block2,

                                   hls::stream<addr_t>& strm_token_addr,
                                   hls::stream<hls_Token_symb>& strm_token_symb,
                                   hls::stream<hls_Token_bits>& strm_token_bits,
                                   hls::stream<bool>& strm_e_addr,
                                   hls::stream<bool>& strm_e_token) {
#pragma HLS INLINE OFF

    const int xsize_blocks = config_dev.xblock8;
    const int ysize_blocks = config_dev.yblock8;
    const int xsize_tiles = config_dev.xblock64;
    const int ysize_tiles = config_dev.yblock64;

    for (ap_uint<8> gy = 0; gy < config_dev.dc_ygroup; ++gy) {
        for (ap_uint<8> gx = 0; gx < config_dev.dc_xgroup; ++gx) {
            hls_Rect dc_rect;

            ap_uint<16> block_x0 = gx * 256;
            ap_uint<16> block_y0 = gy * 256;
            dc_rect.xsize = (block_x0 + 256 <= xsize_blocks) ? (int)(256) : (int)(xsize_blocks - block_x0);
            dc_rect.ysize = (block_y0 + 256 <= ysize_blocks) ? (int)(256) : (int)(ysize_blocks - block_y0);

            _XF_IMAGE_PRINT("dc_rect(%d,%d) \n", dc_rect.xsize, dc_rect.ysize);

            //----------------interleaving encode------------------------

            _XF_IMAGE_PRINT("\n************************************\n");
            _XF_IMAGE_PRINT("-5 Tokenize DC by GROUP - E2B\n");
            _XF_IMAGE_PRINT("**************************************\n");
            hls_encode_dc_top(false, dc_rect, strm_dc_y1, strm_dc_y2, strm_dc_y3, strm_dc_x, strm_dc_b, strm_token_addr,
                              strm_token_symb, strm_token_bits, strm_e_addr, strm_e_token);
            _XF_IMAGE_PRINT("\n************************************\n");
            _XF_IMAGE_PRINT("-5 Tokenize ctrl by GROUP - E2B\n");
            _XF_IMAGE_PRINT("**************************************\n");
            Xacc_TokenizeCtrlField_top(dc_rect, strm_strategy, strm_quant_field, strm_arsigma, strm_strategy_block0,
                                       strm_strategy_block1, strm_strategy_block2, strm_token_addr, strm_token_symb,
                                       strm_token_bits, strm_e_addr, strm_e_token);
        } // gx
    }     // gy

    for (ap_uint<8> gy = 0; gy < config_dev.ac_ygroup; ++gy) {
        for (ap_uint<8> gx = 0; gx < config_dev.ac_xgroup; ++gx) {
            group_rect ac_rect;

            ap_uint<16> block_x0 = gx * 64;
            ap_uint<16> block_y0 = gy * 64;
            ac_rect.xsize_blocks = (block_x0 + 64 <= xsize_blocks) ? (int)64 : (int)(xsize_blocks - block_x0);
            ac_rect.ysize_blocks = (block_y0 + 64 <= ysize_blocks) ? (int)64 : (int)(ysize_blocks - block_y0);

            ap_uint<16> tile_x0 = gx * 8;
            ap_uint<16> tile_y0 = gy * 8;

            ac_rect.xsize_tiles = (tile_x0 + 8 <= xsize_tiles) ? (int)8 : (int)(xsize_tiles - tile_x0);
            ac_rect.ysize_tiles = (tile_y0 + 8 <= ysize_tiles) ? (int)8 : (int)(ysize_tiles - tile_y0);

            _XF_IMAGE_PRINT("ac_rect(%d,%d,%d,%d) \n", ac_rect.xsize_tiles, ac_rect.ysize_tiles, ac_rect.xsize_blocks,
                            ac_rect.ysize_blocks);

            // 5. Tokenize Coefficients by tiles=8   blocks =ysize_blocks/8=ysize/64
            _XF_IMAGE_PRINT("\n************************************\n");
            _XF_IMAGE_PRINT("-5 Tokenize AC by Tiles - E2B\n");
            _XF_IMAGE_PRINT("**************************************\n");
            int hls_order[hls_kOrderContexts][64];

            localOrder(strm_order, hls_order);
            XAcc_TokenizeCoefficients6(hls_order, ac_rect, strm_coef_raster, ac_static_context_map,

                                       strm_token_addr, strm_token_symb, strm_token_bits, strm_e_addr, strm_e_token);
        } // gx
    }     // gy
}

void hls_encode_dc_ac(const int hls_dc_groups,
                      const int hls_ac_groups,

                      hls::stream<ap_uint<13> >& strm_token_addr,
                      hls::stream<hls_Token_symb>& strm_token_symb,
                      hls::stream<hls_Token_bits>& strm_token_bits,
                      hls::stream<bool>& strm_e_addr,
                      hls::stream<bool>& strm_e_token,

                      hist_t hls_histograms[hls_NumHistograms],
                      ap_uint<32> histo_cfg[2 * (2 * MAX_DC_GROUP + MAX_AC_GROUP)],
                      hls::stream<int>& histo_offset,
                      hls::stream<uint8_t>& strm_histo_byte,
                      hls::stream<bool>& strm_histo_e,
                      hls::stream<int>& ac_dc_offset,
                      hls::stream<uint16_t>& strm_ac_dc_byte,
                      hls::stream<bool>& strm_ac_dc_e) {
#pragma HLS INLINE OFF

    int len_dc_histo[2 * MAX_DC_GROUP] = {0};
    int len_dc[2 * MAX_DC_GROUP] = {0};
    int len_ac_histo[MAX_AC_GROUP] = {0};
    int len_ac[MAX_AC_GROUP] = {0};

    int offset_dc_histo = 0;
    int offset_dc = 0;
    int offset_ac_histo = 0;
    int offset_ac = 0;

#ifndef __SYNTHESIS__

    ap_uint<72>* ram_symb;
    ap_uint<72>* ram_bits;
    ram_symb = (ap_uint<72>*)malloc(hls_kMaxBufSize * sizeof(ap_uint<72>));
    ram_bits = (ap_uint<72>*)malloc(hls_kMaxBufSize * sizeof(ap_uint<72>));

#else

    ap_uint<72> ram_symb[hls_kMaxBufSize];
#pragma HLS resource variable = ram_symb core = XPM_MEMORY uram
    ap_uint<72> ram_bits[hls_kMaxBufSize];
#pragma HLS resource variable = ram_bits core = XPM_MEMORY uram

#endif

    ac_dc_offset.write(0);
    histo_offset.write(0);
encode_dc:
    for (ap_uint<16> group_index = 0; group_index < 2 * hls_dc_groups; ++group_index) {
        int total_token;
        // buffer_tokens_onboard and XAcc_EncodeHistogramsFast_top start at the
        // same time they both pingpang with the ANS_top
        // then they all ap_done, the ANS_top will ap_start
        buffer_tokens_onboard(strm_token_symb, strm_token_bits, strm_e_token, total_token, ram_symb, ram_bits);

#ifndef __SYNTHESIS__
        _XF_IMAGE_PRINT("size_token =%d\n", total_token);
#endif

        // 6. Build And Encode Histograms
        hls_ANSEncSymbolInfo hls_codes[hls_kNumStaticContexts][MAX_ALPHABET_SIZE];
        uint8_t dc_context_map[MAX_NUM_COLOR];
        int pos = 0;

#ifndef __SYNTHESIS__
        _XF_IMAGE_PRINT("\n************************************\n");
        _XF_IMAGE_PRINT("-6 BuildAndEncodeHistogramsFast - E2B\n");
        _XF_IMAGE_PRINT("**************************************\n");
#endif

        XAcc_EncodeHistogramsFast_top(true, dc_context_map, strm_token_addr, strm_e_addr, hls_codes, hls_histograms,
                                      pos, len_dc_histo[group_index], strm_histo_byte, strm_histo_e);

        offset_dc_histo += len_dc_histo[group_index];

#ifndef __SYNTHESIS__
        _XF_IMAGE_PRINT("len_histo = %d", (int)len_dc_histo[group_index]);
        _XF_IMAGE_PRINT("\n************************************\n");
        _XF_IMAGE_PRINT("-7 Write AC Tokens - E2B\n");
        _XF_IMAGE_PRINT("**************************************\n");
#endif

        ANS_top(true, dc_context_map, total_token, ram_symb, ram_bits, hls_codes, len_dc[group_index], strm_ac_dc_byte,
                strm_ac_dc_e);

        offset_dc += (len_dc[group_index] + 1) / 2;

        ac_dc_offset.write(offset_dc);
        histo_offset.write(offset_dc_histo);
    }

    ac_dc_offset.write(0);
    histo_offset.write(0);
encode_ac:
    for (ap_uint<16> group_index = 0; group_index < hls_ac_groups; ++group_index) {
        int total_token;
        // buffer_tokens_onboard and XAcc_EncodeHistogramsFast_top start at the
        // same time they both pingpang with the ANS_top
        // then they all ap_done, the ANS_top will ap_start
        buffer_tokens_onboard(strm_token_symb, strm_token_bits, strm_e_token, total_token, ram_symb, ram_bits);

#ifndef __SYNTHESIS__
        _XF_IMAGE_PRINT("size_token =%d\n", total_token);
#endif

        // 6. Build And Encode Histograms
        hls_ANSEncSymbolInfo hls_codes[hls_kNumStaticContexts][MAX_ALPHABET_SIZE];
        uint8_t ac_context_map[MAX_NUM_COLOR];
        int pos = 0;

#ifndef __SYNTHESIS__
        _XF_IMAGE_PRINT("\n************************************\n");
        _XF_IMAGE_PRINT("-6 BuildAndEncodeHistogramsFast - E2B\n");
        _XF_IMAGE_PRINT("**************************************\n");
#endif

        XAcc_EncodeHistogramsFast_top(false, ac_context_map, strm_token_addr, strm_e_addr, hls_codes, hls_histograms,
                                      pos, len_ac_histo[group_index], strm_histo_byte, strm_histo_e);

        offset_ac_histo += len_ac_histo[group_index];

#ifndef __SYNTHESIS__
        _XF_IMAGE_PRINT("len_histo = %d", (int)len_ac_histo[group_index]);
        _XF_IMAGE_PRINT("\n************************************\n");
        _XF_IMAGE_PRINT("-7 Write AC Tokens - E2B\n");
        _XF_IMAGE_PRINT("**************************************\n");
#endif

        ANS_top(false, ac_context_map, total_token, ram_symb, ram_bits, hls_codes, len_ac[group_index], strm_ac_dc_byte,
                strm_ac_dc_e);

        offset_ac += (len_ac[group_index] + 1) / 2;

        ac_dc_offset.write(offset_ac);
        histo_offset.write(offset_ac_histo);
    }

#ifndef __SYNTHESIS__
    std::cout << "ac_dc_byte_size:" << strm_ac_dc_byte.size() << " " << strm_ac_dc_e.size() << std::endl;
    free(ram_symb);
    free(ram_bits);
#endif

    ap_uint<32> cfg_addr = 0;
output_cfg0:
    for (int group_index = 0; group_index < 2 * hls_dc_groups; ++group_index) {
#pragma HLS PIPELINE II = 1
        histo_cfg[cfg_addr] = len_dc_histo[group_index];
        cfg_addr++;
    }

output_cfg1:
    for (int group_index = 0; group_index < hls_ac_groups; ++group_index) {
#pragma HLS PIPELINE II = 1
        histo_cfg[cfg_addr] = len_ac_histo[group_index];
        cfg_addr++;
    }

output_cfg2:
    for (int group_index = 0; group_index < 2 * hls_dc_groups; ++group_index) {
#pragma HLS PIPELINE II = 1
        histo_cfg[cfg_addr] = len_dc[group_index];
        cfg_addr++;
    }

output_cfg3:
    for (int group_index = 0; group_index < hls_ac_groups; ++group_index) {
#pragma HLS PIPELINE II = 1
        histo_cfg[cfg_addr] = len_ac[group_index];
        cfg_addr++;
    }
}

template <int _BurstLen, int _WAxi, typename _IStrm, int _WStrm>
void streamRetypeToAxi(ap_uint<_WAxi>* wbuf, hls::stream<_IStrm>& istrm, hls::stream<bool>& e_istrm) {
#pragma HLS INLINE OFF

    const int fifo_buf = 2 * _BurstLen;

#pragma HLS DATAFLOW

    hls::stream<ap_uint<_WAxi> > axi_strm;
    hls::stream<ap_uint<8> > nb_strm;
#pragma HLS stream variable = nb_strm depth = 2
#pragma HLS stream variable = axi_strm depth = fifo_buf

    hls::stream<ap_uint<_WStrm> > istrm2;
#pragma HLS stream variable = istrm2 depth = 32
    hls::stream<bool> e_istrm2;
#pragma HLS stream variable = e_istrm2 depth = 32

    streamRetype<_IStrm, ap_uint<_WStrm> >(istrm, e_istrm, istrm2, e_istrm2);

    xf::common::utils_hw::details::countForBurst<_WAxi, _WStrm, _BurstLen>(istrm2, e_istrm2, axi_strm, nb_strm);

    xf::common::utils_hw::details::burstWrite<_WAxi, _WStrm, _BurstLen>(wbuf, axi_strm, nb_strm);
}

void hls_writeout_sub(hls::stream<uint8_t>& strm_histo_byte,
                      hls::stream<bool>& strm_histo_e,

                      hls::stream<uint16_t>& strm_dc_ac_byte,
                      hls::stream<bool>& strm_dc_ac_e,

                      ap_uint<32>* histo_code_out,
                      ap_uint<32>* code_out) {
#pragma HLS INLINE OFF
    for (int i = 0; i < 2; i++) {
        if (i == 0) {
#ifdef DEBUG
            std::cout << "======================write out histo=====================" << std::endl;
#endif
            streamRetypeToAxi<64, 32, uint8_t, 32>(histo_code_out, strm_histo_byte, strm_histo_e);
#ifdef DEBUG
            std::cout << "=========================write done=======================" << std::endl;
#endif
        } else {
#ifdef DEBUG
            std::cout << "======================write out DC_AC=====================" << std::endl;
#endif
            streamRetypeToAxi<64, 32, uint16_t, 32>(code_out, strm_dc_ac_byte, strm_dc_ac_e);
#ifdef DEBUG
            std::cout << "=========================write done=======================" << std::endl;
#endif
        }
    }
}

void hls_writeout(const int hls_dc_groups,
                  const int hls_ac_groups,
                  hls::stream<int>& histo_offset,
                  hls::stream<uint8_t>& strm_histo_byte,
                  hls::stream<bool>& strm_histo_e,
                  hls::stream<int>& dc_ac_offset,
                  hls::stream<uint16_t>& strm_dc_ac_byte,
                  hls::stream<bool>& strm_dc_ac_e,

                  ap_uint<32> dc_histo_code_out[2 * MAX_DC_GROUP * MAX_DC_HISTO_SIZE],
                  ap_uint<32> dc_code_out[2 * MAX_DC_GROUP * MAX_DC_SIZE],

                  ap_uint<32> ac_histo_code_out[MAX_AC_GROUP * MAX_AC_HISTO_SIZE],
                  ap_uint<32> ac_code_out[MAX_AC_GROUP * MAX_AC_SIZE]) {
#pragma HLS INLINE OFF

#ifdef DEBUG
    std::cout << "ac_goutps=" << hls_ac_groups << " dc_groups=" << hls_dc_groups << std::endl;
#endif

    for (int group_index = 0; group_index < 2 * hls_dc_groups; ++group_index) {
#ifdef DEBUG
        std::cout << "======================write out DC=====================" << std::endl;
#endif

        int offset0 = histo_offset.read();
        int offset1 = dc_ac_offset.read();
        hls_writeout_sub(strm_histo_byte, strm_histo_e, strm_dc_ac_byte, strm_dc_ac_e, (dc_histo_code_out + offset0),
                         (dc_code_out + offset1));
    }

    // disgard padding info
    histo_offset.read();
    dc_ac_offset.read();

    for (int group_index = 0; group_index < hls_ac_groups; ++group_index) {
#ifdef DEBUG
        std::cout << "======================write out AC=====================" << std::endl;
#endif

        int offset0 = histo_offset.read();
        int offset1 = dc_ac_offset.read();
        hls_writeout_sub(strm_histo_byte, strm_histo_e, strm_dc_ac_byte, strm_dc_ac_e, (ac_histo_code_out + offset0),
                         (ac_code_out + offset1));
    }

    // disgard padding info
    histo_offset.read();
    dc_ac_offset.read();
}

void kernel3Wrapper(ap_uint<32> config[MAX_NUM_CONFIG],

                    ap_uint<32> ddr_dc[MAX_NUM_DC],
                    ap_uint<32> ddr_ac_strategy[MAX_NUM_BLOCK88],
                    ap_uint<32> ddr_block[MAX_NUM_BLOCK88],
                    ap_uint<32> ddr_quant_field[MAX_NUM_BLOCK88],
                    ap_uint<32> ddr_ac[ALL_PIXEL],
                    ap_uint<32> ddr_order[MAX_AC_GROUP * hls_kOrderContexts * 64],

                    ap_uint<32> histo_cfg[2 * (2 * MAX_DC_GROUP + MAX_AC_GROUP)],
                    ap_uint<32> dc_histo_code_out[2 * MAX_DC_GROUP * MAX_DC_HISTO_SIZE],
                    ap_uint<32> dc_code_out[2 * MAX_DC_GROUP * MAX_DC_SIZE],
                    ap_uint<32> ac_histo_code_out[MAX_AC_GROUP * MAX_AC_HISTO_SIZE],
                    ap_uint<32> ac_code_out[MAX_AC_GROUP * MAX_AC_SIZE]) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    _XF_IMAGE_PRINT("\n kernel 3  start! \n");

    ConfigKernel3 config_dev[4];
#pragma HLS ARRAY_PARTITION variable = config_dev complete dim = 1

#ifdef DEBUG
    std::cout << "===================Load config=================" << std::endl;
#endif

    load_config_kernel3(config, config_dev);

#ifdef DEBUG
    std::cout << "======================Scan=====================" << std::endl;
#endif

    hls::stream<dct_t> strm_dc_x("strm_dc_x");
#pragma HLS RESOURCE variable = strm_dc_x core = FIFO_BRAM
#pragma HLS STREAM variable = strm_dc_x depth = 512
    hls::stream<dct_t> strm_dc_y1;
#pragma HLS RESOURCE variable = strm_dc_y1 core = FIFO_BRAM
#pragma HLS STREAM variable = strm_dc_y1 depth = 512
    hls::stream<dct_t> strm_dc_y2("strm_dc_y");
#pragma HLS RESOURCE variable = strm_dc_y2 core = FIFO_BRAM
#pragma HLS STREAM variable = strm_dc_y2 depth = 512
    hls::stream<dct_t> strm_dc_y3;
#pragma HLS RESOURCE variable = strm_dc_y3 core = FIFO_BRAM
#pragma HLS STREAM variable = strm_dc_y3 depth = 512
    hls::stream<dct_t> strm_dc_b("strm_dc_b");
#pragma HLS RESOURCE variable = strm_dc_b core = FIFO_BRAM
#pragma HLS STREAM variable = strm_dc_b depth = 512

    hls::stream<uint8_t> strm_strategy("strm_strategy");
#pragma HLS RESOURCE variable = strm_strategy core = FIFO_BRAM
#pragma HLS STREAM variable = strm_strategy depth = 512
    hls::stream<quant_t> strm_quant_field("strm_quant");
#pragma HLS RESOURCE variable = strm_quant_field core = FIFO_BRAM
#pragma HLS STREAM variable = strm_quant_field depth = 512
    hls::stream<arsigma_t> strm_arsigma("strm_arsigma");
#pragma HLS RESOURCE variable = strm_arsigma core = FIFO_BRAM
#pragma HLS STREAM variable = strm_arsigma depth = 512
    hls::stream<bool> strm_strategy_block0("strm_strategy_block0");
#pragma HLS RESOURCE variable = strm_strategy_block0 core = FIFO_BRAM
#pragma HLS STREAM variable = strm_strategy_block0 depth = 512
    hls::stream<bool> strm_strategy_block1("strm_strategy_block1");
#pragma HLS RESOURCE variable = strm_strategy_block1 core = FIFO_BRAM
#pragma HLS STREAM variable = strm_strategy_block1 depth = 512
    hls::stream<bool> strm_strategy_block2("strm_strategy_block2");
#pragma HLS RESOURCE variable = strm_strategy_block2 core = FIFO_BRAM
#pragma HLS STREAM variable = strm_strategy_block2 depth = 512

    hls::stream<dct_t> strm_coef_raster_syn("strm_ac_raster_syn");
#pragma HLS RESOURCE variable = strm_coef_raster_syn core = FIFO_BRAM
#pragma HLS STREAM variable = strm_coef_raster_syn depth = 1024
    hls::stream<int> strm_order("strm_order");
#pragma HLS RESOURCE variable = strm_coef_raster_syn core = FIFO_BRAM
#pragma HLS STREAM variable = strm_order depth = 512

    hls_burst_dc_ac_by_group2(config_dev[0], ddr_dc, ddr_ac_strategy, ddr_block, ddr_quant_field, ddr_ac, ddr_order,

                              strm_dc_y1, strm_dc_y2, strm_dc_y3, strm_dc_x, strm_dc_b, strm_strategy, strm_quant_field,
                              strm_arsigma, strm_strategy_block0, strm_strategy_block1, strm_strategy_block2,
                              strm_coef_raster_syn, strm_order);

#ifdef DEBUG
    std::cout << "======================tokenize=====================" << std::endl;
#endif

    hls::stream<addr_t> strm_token_addr("strm_token_addr");
#pragma HLS RESOURCE variable = strm_token_addr core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_token_addr depth = 32
    hls::stream<bool> strm_e_addr;
#pragma HLS RESOURCE variable = strm_e_addr core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_e_addr depth = 32

    hls::stream<hls_Token_symb> strm_token_symb("strm_token_symb");
#pragma HLS RESOURCE variable = strm_token_symb core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_token_symb depth = 32
    hls::stream<hls_Token_bits> strm_token_bits("strm_token_bits");
#pragma HLS RESOURCE variable = strm_token_bits core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_token_bits depth = 32
    hls::stream<bool> strm_e_token("strm_e_token");
#pragma HLS RESOURCE variable = strm_e_token core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_e_token depth = 32

    hls_tokenize_dc_ac_sequential(config_dev[1], strm_coef_raster_syn, strm_order, strm_dc_y1, strm_dc_y2, strm_dc_y3,
                                  strm_dc_x, strm_dc_b, strm_strategy, strm_quant_field, strm_arsigma,
                                  strm_strategy_block0, strm_strategy_block1, strm_strategy_block2,

                                  strm_token_addr, strm_token_symb, strm_token_bits, strm_e_addr, strm_e_token);

#ifdef DEBUG
    std::cout << "======================encode=====================" << std::endl;
#endif

    hist_t hls_histograms[hls_NumHistograms];

    hls::stream<int> strm_histo_offset("strm_histo_offset");
#pragma HLS RESOURCE variable = strm_histo_offset core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_histo_offset depth = 32
    hls::stream<uint8_t> strm_histo_byte("strm_histo_byte");
#pragma HLS RESOURCE variable = strm_histo_byte core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_histo_byte depth = 32
    hls::stream<bool> strm_histo_e("strm_histo_e");
#pragma HLS RESOURCE variable = strm_histo_e core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_histo_e depth = 32

    hls::stream<int> strm_dc_ac_offset("strm_dc_ac_offset");
#pragma HLS RESOURCE variable = strm_dc_ac_offset core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_dc_ac_offset depth = 32
    hls::stream<uint16_t> strm_dc_ac_byte("dc_byte");
#pragma HLS RESOURCE variable = strm_dc_ac_byte core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_dc_ac_byte depth = 32
    hls::stream<bool> strm_dc_ac_e("dc_e");
#pragma HLS RESOURCE variable = strm_dc_ac_e core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_dc_ac_e depth = 32

    hls_encode_dc_ac(config_dev[2].dc_group, config_dev[2].ac_group, strm_token_addr, strm_token_symb, strm_token_bits,
                     strm_e_addr, strm_e_token,

                     hls_histograms, histo_cfg, strm_histo_offset, strm_histo_byte, strm_histo_e, strm_dc_ac_offset,
                     strm_dc_ac_byte, strm_dc_ac_e);

#ifdef DEBUG
    std::cout << "======================start write out=====================" << std::endl;
#endif

    hls_writeout(config_dev[3].dc_group, config_dev[3].ac_group, strm_histo_offset, strm_histo_byte, strm_histo_e,
                 strm_dc_ac_offset, strm_dc_ac_byte, strm_dc_ac_e, dc_histo_code_out, dc_code_out, ac_histo_code_out,
                 ac_code_out);

#ifdef DEBUG
    std::cout << "======================kernel3 done=====================" << std::endl;
#endif
}

extern "C" void kernel3Top(ap_uint<32>* config,

                           ap_uint<32>* ddr_ac,
                           ap_uint<32>* ddr_dc,
                           ap_uint<32>* ddr_quant_field,
                           ap_uint<32>* ddr_ac_strategy,
                           ap_uint<32>* ddr_block,
                           ap_uint<32>* hls_order,

                           ap_uint<32>* histo_cfg,
                           ap_uint<32>* dc_histo_code_out,
                           ap_uint<32>* dc_code_out,
                           ap_uint<32>* ac_histo_code_out,
                           ap_uint<32>* ac_code_out) {
#pragma HLS INLINE off

    const int max_num_dc = MAX_NUM_DC;
    const int max_num_block88 = MAX_NUM_BLOCK88;
    const int max_pixel = ALL_PIXEL;
    const int max_dc_histo = 2 * MAX_DC_GROUP * MAX_DC_HISTO_SIZE;
    const int max_dc = 2 * MAX_DC_GROUP * MAX_DC_SIZE;
    const int max_ac_histo = MAX_AC_GROUP * MAX_AC_HISTO_SIZE;
    const int max_ac = MAX_AC_GROUP * MAX_AC_SIZE;
    const int max_order = MAX_AC_GROUP * hls_kOrderContexts * 64;

// clang-format off

// cfg
#pragma HLS INTERFACE m_axi offset = slave latency = 64                        \
        num_write_outstanding = 4 max_write_burst_length = 8                   \
        num_read_outstanding = 4 max_read_burst_length = 8 bundle =            \
        gmem0_0 port = config depth = 32
#pragma HLS INTERFACE s_axilite port = config bundle = control

// dc
#pragma HLS INTERFACE m_axi offset = slave latency = 64                        \
        num_write_outstanding = 4 max_write_burst_length = 8                   \
        num_read_outstanding = 8 max_read_burst_length = 64 bundle =           \
        gmem0_1 port = ddr_dc depth = max_num_dc
#pragma HLS INTERFACE s_axilite port = ddr_dc bundle = control

// acs
#pragma HLS INTERFACE m_axi offset = slave latency = 64                        \
        num_write_outstanding = 4 max_write_burst_length = 8                   \
        num_read_outstanding = 8 max_read_burst_length = 64 bundle =           \
        gmem0_2 port = ddr_ac_strategy depth = max_num_block88
#pragma HLS INTERFACE s_axilite port = ddr_ac_strategy bundle = control

// qf
#pragma HLS INTERFACE m_axi offset = slave latency = 64                        \
        num_write_outstanding = 4 max_write_burst_length = 8                   \
        num_read_outstanding = 8 max_read_burst_length = 64 bundle =           \
        gmem0_3 port = ddr_quant_field depth = max_num_block88
#pragma HLS INTERFACE s_axilite port = ddr_quant_field bundle = control

// block
#pragma HLS INTERFACE m_axi offset = slave latency = 64                        \
        num_write_outstanding = 4 max_write_burst_length = 8                   \
        num_read_outstanding = 8 max_read_burst_length = 64 bundle =           \
        gmem0_4 port = ddr_block depth = max_num_block88
#pragma HLS INTERFACE s_axilite port = ddr_block bundle = control

// ac
#pragma HLS INTERFACE m_axi offset = slave latency = 64                        \
        num_write_outstanding = 4 max_write_burst_length = 8                   \
        num_read_outstanding = 8 max_read_burst_length = 64 bundle =           \
        gmem0_5 port = ddr_ac depth = max_pixel
#pragma HLS INTERFACE s_axilite port = ddr_ac bundle = control

// order
#pragma HLS INTERFACE m_axi offset = slave latency = 64                        \
        num_write_outstanding = 4 max_write_burst_length = 8                   \
        num_read_outstanding = 8 max_read_burst_length = 64 bundle =           \
        gmem0_6 port = hls_order depth = max_order
#pragma HLS INTERFACE s_axilite port = hls_order bundle = control


// output
#pragma HLS INTERFACE m_axi offset = slave latency = 64                        \
        num_write_outstanding = 8 max_write_burst_length = 64                  \
        num_read_outstanding = 4 max_read_burst_length = 8 bundle =            \
        gmem1_0 port = dc_histo_code_out depth = max_dc_histo
#pragma HLS INTERFACE s_axilite port = dc_histo_code_out bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64                        \
        num_write_outstanding = 8 max_write_burst_length = 64                  \
        num_read_outstanding = 4 max_read_burst_length = 8 bundle =            \
        gmem1_1 port = dc_code_out depth = max_dc
#pragma HLS INTERFACE s_axilite port = dc_code_out bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64                        \
        num_write_outstanding = 8 max_write_burst_length = 64                  \
        num_read_outstanding = 4 max_read_burst_length = 8 bundle =            \
        gmem1_2 port = ac_histo_code_out depth = max_ac_histo
#pragma HLS INTERFACE s_axilite port = ac_histo_code_out bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64                        \
        num_write_outstanding = 8 max_write_burst_length = 64                  \
        num_read_outstanding = 4 max_read_burst_length = 8 bundle =            \
        gmem1_3 port = ac_code_out depth = max_ac
#pragma HLS INTERFACE s_axilite port = ac_code_out bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64                        \
        num_write_outstanding = 4 max_write_burst_length = 8                   \
        num_read_outstanding = 4 max_read_burst_length = 8 bundle =            \
        gmem1_4 port = histo_cfg depth = 1024
#pragma HLS INTERFACE s_axilite port = histo_cfg bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

    // clang-format on

    kernel3Wrapper(config, ddr_dc, ddr_ac_strategy, ddr_block, ddr_quant_field, ddr_ac, hls_order, histo_cfg,
                   dc_histo_code_out, dc_code_out, ac_histo_code_out, ac_code_out);
}
