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
 * @file XAcc_jfifparser.cpp
 * @brief parser_jpg_top template function implementation and kernel_decoder warpper.
 *
 * This file is part of HLS algorithm library.
 */

#include "XAcc_jfifparser.hpp"

// ------------------------------------------------------------
#define B_SHORT(v1, v2) ((((int)v1) << 8) + ((int)v2))

namespace xf {
namespace codec {
namespace details {

inline void readBytes(int& j, const int& cnt, int& r, int& c) {
#pragma HLS INLINE
    j += cnt;
    r = j >> 1;
    c = j & 1;
}
inline void oneByte(int& j, int& r, int& c) {
#pragma HLS INLINE
    j += 1;
    r = j >> 1;
    c = j & 1;
}

// ------------------------------------------------------------
void SetOtherQtab(decOutput* pout) {
    // clang-format off
	ap_uint<16> freqmax_[3][64];
	static const unsigned short int freqmax[] =
		        {
		            1024, 931, 985, 968, 1020, 968, 1020, 1020,
		            932, 858, 884, 840, 932, 838, 854, 854,
		            985, 884, 871, 875, 985, 878, 871, 854,
		            967, 841, 876, 844, 967, 886, 870, 837,
		            1020, 932, 985, 967, 1020, 969, 1020, 1020,
		            969, 838, 878, 886, 969, 838, 969, 838,
		            1020, 854, 871, 870, 1010, 969, 1020, 1020,
		            1020, 854, 854, 838, 1020, 838, 1020, 838
		        };
// clang-format on
#pragma HLS ARRAY_PARTITION variable = pout->q_tables complete dim = 2
#pragma HLS ARRAY_PARTITION variable = pout->idct_q_table_x complete dim = 3
#pragma HLS ARRAY_PARTITION variable = pout->idct_q_table_y complete dim = 3
    unsigned short RESIDUAL_NOISE_FLOOR = 7;
    for (int idx_cmp = 0; idx_cmp < pout->axi_num_cmp_mcu; idx_cmp++) {
        uint8_t c = pout->axi_map_row2cmp[idx_cmp];

        for (int i = 0; i < 64; i++) {
#pragma HLS pipeline
            pout->idct_q_table_x[c][i >> 3][i & 7] =
                hls_icos_base_8192_scaled[(i & 7) << 3] * pout->q_tables[c][i & 7][i >> 3];
            pout->idct_q_table_y[c][i >> 3][i & 7] =
                hls_icos_base_8192_scaled[(i & 7) << 3] * pout->q_tables[c][i >> 3][i & 7];
            // pout->idct_q_table_l[c][i>>3][i&7] = hls_icos_idct_linear_8192_scaled[i]   * pout->q_tables[c][0][i&7];
            //}

            // for (int coord = 0; coord < 64; ++coord) {
            freqmax_[c][i] = (freqmax[i] + pout->q_tables[c][i >> 3][i & 7] - 1) / pout->q_tables[c][i >> 3][i & 7];
            // uint8_t max_len = uint16bit_length(freqmax_[c][i]);
            uint8_t max_len = 16 - freqmax_[c][i].countLeadingZeros();
            // bitlen_freqmax_[c][i] = max_len;
            if (max_len > (int)RESIDUAL_NOISE_FLOOR) {
                pout->min_nois_thld_x[c][i] = pout->min_nois_thld_y[c][i] = max_len - RESIDUAL_NOISE_FLOOR;
            } else {
                pout->min_nois_thld_x[c][i] = pout->min_nois_thld_y[c][i] = 0;
            }

        } // end for
    }
}

// ------------------------------------------------------------

void parser_jpg_top(ap_uint<AXI_WIDTH>* datatoDDR,
                    const int size,
                    int& r,
                    int& c,
                    uint16_t dht_tbl1[2][2][1 << DHT1],
                    // uint16_t dht_tbl2[2][2][1 << DHT2],
                    //
                    uint8_t ac_val[2][165],
                    HCODE_T ac_start_code[2][AC_N],
                    int16_t ac_huff_start_addr[2][16],
                    //
                    uint8_t dc_val[2][12],
                    HCODE_T dc_start_code[2][DC_N],
                    int16_t dc_huff_start_addr[2][16],
                    //
                    ap_uint<12>& hls_cmp,
                    int& left,

                    // image info
                    img_info& img_info,
                    uint8_t hls_mbs[MAX_NUM_COLOR],
                    hls_compInfo hls_compinfo[MAX_NUM_COLOR],
                    int& rtn,
                    decOutput* pout) {
    ap_uint<AXI_WIDTH>* segment = datatoDDR;
    int offset = 0;
    uint8_t b1, b2, b3, b4;

    // read
    b1 = segment[0](7, 0);
    b2 = segment[0](15, 8);
    if ((size < 127) | (b1 != 0xFF) | (b2 != 0xD8)) {
        _XF_IMAGE_PRINT("Header failed\n");
        rtn = 1;
    } else {
        rtn = 0;
    }
    // bill
    readBytes(offset, 2, r, c);
    bool scanned = false;

    while (!scanned && !rtn) {
        if (segment[r](c * 8 + 7, c * 8) != 0xFF) {
            _XF_IMAGE_PRINT("marker+length detect failed\n");
            rtn = 1;
            break;
        }

        // skip marker ff and protect 16 bit length
        oneByte(offset, r, c);
        b2 = segment[r](c * 8 + 7, c * 8);
        oneByte(offset, r, c);
        uint8_t l1 = segment[r](c * 8 + 7, c * 8);
        oneByte(offset, r, c);
        uint8_t l2 = segment[r](c * 8 + 7, c * 8);
        oneByte(offset, r, c);
        uint16_t len = B_SHORT(l1, l2);
        len -= 2;

        // marker read and bill
        if (((b2 & 0xF0) == 0xE0) || (b2 == 0xDD)) { // all APP

            _XF_IMAGE_PRINT("APP or DRI : OFFSET: %.8x\n", offset - 4);
            readBytes(offset, len, r, c);
            _XF_IMAGE_PRINT("skip %d Bytes of marker \n", len);

        } else if (b2 == 0xDB) { // marker

            _XF_IMAGE_PRINT("------------------------\n");
            // min 3 byte marker
            // 2B  4  4    B
            // L   p  T  Vij*64
            // syn_build_DQT(len, offset,r,c, segment, dqt, rtn);
            _XF_IMAGE_PRINT("DQT 0xDB: OFFSET: %.8x\n", offset - 4);

            while (len >= 64 + 1) {
                // first b + 64 b
                b1 = segment[r](c * 8 + 7, c * 8);
                oneByte(offset, r, c);
                if (b1 > 3) { // rtn = false;//19 byte marker
                    _XF_IMAGE_PRINT(" Warning: DQT, Warning idx \n");
                }

                for (int j = 0; j < 64; ++j) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
#pragma HLS PIPELINE

                    int jzz_x = hls_jpeg_zigzag_to_raster[j] & 7;
                    int jzz_y = hls_jpeg_zigzag_to_raster[j] >> 3;
                    if (b1 == 0)
                        pout->q_tables[0][jzz_y][jzz_x] = segment[r](c * 8 + 7, c * 8);
                    else {
                        pout->q_tables[1][jzz_y][jzz_x] = segment[r](c * 8 + 7, c * 8);
                        pout->q_tables[2][jzz_y][jzz_x] = segment[r](c * 8 + 7, c * 8);
                    }

                    oneByte(offset, r, c);
                }
                len -= 65;
            }
            // if(len) {_XF_IMAGE_PRINT("Decode DQT failed\n");}

        } else if (b2 == 0xC0) { // marker

            _XF_IMAGE_PRINT("------------------------\n");
            // sof ffc0//min (17-2) byte marker
            // 2B B 2B 2B B  ( B   4    4     B ) *3
            // L P Y  X  CMP idx   sfh  sfv   qidx
            // syn_frame_SOF(len, offset,r,c, segment, hls_mbs, rtn, hls_compinfo,
            //	  hls_mcuc,hls_mcuh,hls_mcuv,hls_cs_cmpc);
            _XF_IMAGE_PRINT("SOF 0xC0: OFFSET: %.8x\n", offset - 4);

            b1 = segment[r](c * 8 + 7, c * 8);
            if (b1 != 8) {
                // rtn=false;
                _XF_IMAGE_PRINT(" Warning: SOF, image precision is not 8bit \n");
            }
            oneByte(offset, r, c);

            // height and width
            b1 = segment[r](c * 8 + 7, c * 8);
            oneByte(offset, r, c);
            b2 = segment[r](c * 8 + 7, c * 8);
            oneByte(offset, r, c);
            int height = B_SHORT(b1, b2);

            b1 = segment[r](c * 8 + 7, c * 8);
            oneByte(offset, r, c);
            b2 = segment[r](c * 8 + 7, c * 8);
            oneByte(offset, r, c);
            int width = B_SHORT(b1, b2);
            _XF_IMAGE_PRINT("height=%d, width=%d, \n", height, width);

            img_info.hls_cs_cmpc = segment[r](c * 8 + 7, c * 8);
            oneByte(offset, r, c);
            if (img_info.hls_cs_cmpc != 3) {
                // rtn= false;
                _XF_IMAGE_PRINT("Warning: SOF, supports only 3 component color jpeg files\n");
            }

            uint8_t sfhm = 0, sfvm = 0;
            for (int cmp = 0; cmp < 3; ++cmp) {
#pragma HLS PIPELINE
                uint8_t sfv, sfh;
                oneByte(offset, r, c);
                b1 = segment[r](c * 8 + 7, c * 8);
                sfv = b1 >> 4;
                sfh = b1 & 0x0f;

                if ((sfv & (sfv - 1)) || (sfh & (sfh - 1))) {
                    // rtn= false;
                    _XF_IMAGE_PRINT("Warning: SOF, sfv of sfh \n");
                }

                hls_compinfo[cmp].sfv = sfv;
                hls_compinfo[cmp].sfh = sfh;
                if (hls_compinfo[cmp].sfh > sfhm) sfhm = hls_compinfo[cmp].sfh;
                if (hls_compinfo[cmp].sfv > sfvm) sfvm = hls_compinfo[cmp].sfv;
                hls_mbs[cmp] = hls_compinfo[cmp].sfv * hls_compinfo[cmp].sfh;
                _XF_IMAGE_PRINT("sfv = %d, sfh = %d\n", sfv, sfh);
                // if (cmp == 0) downsample = sfv > 1;
                readBytes(offset, 2, r, c);
            }

            if (hls_mbs[0] == 4) {
                hls_cmp = 0b110000110000;
            } else if (hls_mbs[0] == 2) {
                hls_cmp = 0b110011001100;
            } else if (hls_mbs[0] == 1) {
                hls_cmp = 0b110110110110;
            } else {
                _XF_IMAGE_PRINT("Warning: hls_cmpnfo[0].mbs is not 4/2/1 \n");
            }

            int sub_o_sfh = (height >> 3) / sfhm;
            int sub_o_sfv = (width >> 3) / sfvm;
            img_info.hls_mcuv = (height - (sub_o_sfh << 3) * sfhm) ? (sub_o_sfh + 1) : sub_o_sfh;
            img_info.hls_mcuh = (width - (sub_o_sfv << 3) * sfvm) ? (sub_o_sfv + 1) : sub_o_sfv;
            // hls_mcuv =  ( int ) ceil( (float) height / (float) ( 8 * sfhm ) );
            // hls_mcuh =  ( int ) ceil( (float) width  / (float) ( 8 * sfvm ) );
            img_info.hls_mcuc = img_info.hls_mcuv * img_info.hls_mcuh;

#ifndef __SYNTHESIS__
            printf("hls_mcuv=%d, hls_mcuh=%d, hls_mcuc=%d, \n", img_info.hls_mcuv, img_info.hls_mcuh,
                   img_info.hls_mcuc);
#endif
            for (int cmp = 0; cmp < 3; cmp++) {
#pragma HLS PIPELINE
                hls_compinfo[cmp].mbs = hls_mbs[cmp];
                hls_compinfo[cmp].bcv = img_info.hls_mcuv * hls_compinfo[cmp].sfh;
                hls_compinfo[cmp].bch = img_info.hls_mcuh * hls_compinfo[cmp].sfv;
                hls_compinfo[cmp].bc = hls_compinfo[cmp].bcv * hls_compinfo[cmp].bch;
            }

        } else if (b2 == 0xC4) { // marker

            _XF_IMAGE_PRINT("------------------------\n");
            // min 19 byte marker
            // 2B  4  4   B       B
            // L  ac cmp  Li*16  Vij*256
            // syn_DHT(len, offset,r,c, segment, dht_tbl1, dht_tbl2, rtn);
            _XF_IMAGE_PRINT("DHT 0xC4: OFFSET: %.8x\n", offset - 4);

            uint16_t huff_len = 1, cnt, addr_now, addr_gap;
            const int addr_all = 65536;
            uint16_t huff_cnt[16];

            uint16_t dc_huff_start_code[2][16];
            uint16_t ac_huff_start_code[2][16];

            while (len > 16 + 1) {
                // read fstb + Li*16
                int test = 0;
                // accumlator
                uint8_t val_addr = 0;
                cnt = 0;
                b1 = segment[r](c * 8 + 7, c * 8);
                oneByte(offset, r, c);

                bool ac = b1 & 0x10;
                int cmp_huff = b1 & 0x0f;
                _XF_IMAGE_PRINT(" ac = %d, cmp =%d", ac, cmp_huff);
                if ((b1 & 0xEC) || (b1 & 0x02)) {               // check 0bxxxdxxdd
                    _XF_IMAGE_PRINT(" Warning: DHT failed \n"); // rtn = false;
                }

                dc_huff_start_code[cmp_huff][0] = 0;
                ac_huff_start_code[cmp_huff][0] = 0;

            // init huff_cnt
            GEN_NEW_TBL_LOOP:
                for (huff_len = 1; huff_len <= 16; ++huff_len) {
#pragma HLS PIPELINE
                    huff_cnt[huff_len - 1] = segment[r](c * 8 + 7, c * 8);

                    uint16_t tbl3_code = dc_huff_start_code[cmp_huff][huff_len - 1];
                    uint16_t tbl4_code = ac_huff_start_code[cmp_huff][huff_len - 1];

                    if (huff_len <= 15) {
                        if (ac) {
                            ac_huff_start_code[cmp_huff][huff_len] = (tbl4_code + huff_cnt[huff_len - 1]) << 1;
                            ap_uint<16> tmp = (tbl4_code + huff_cnt[huff_len - 1]) << 1; // huff_len+1 bit
                            // origin huff_len 10bit~15, now count the 11bit start code(so, huff_len-3+1),  index from
                            // 10~15 cut to 0~5
                            if (huff_len > DHT1) {
                                ac_start_code[cmp_huff][huff_len - DHT1 - 1] = tmp(huff_len - 2, 0);
                            }
                        } else {
                            dc_huff_start_code[cmp_huff][huff_len] = (tbl3_code + huff_cnt[huff_len - 1]) << 1;
                            ap_uint<16> tmp = (tbl3_code + huff_cnt[huff_len - 1]) << 1;
                            // origin huff_len 10bit~15, cut 2 bit
                            if (huff_len > DHT1 && (huff_len < 12)) {
                                dc_start_code[cmp_huff][huff_len - DHT1 - 1] = tmp(huff_len - 2, 0);
                            }
                        }
                    }

                    if (ac) {
                        ac_huff_start_addr[cmp_huff][huff_len - 1] = tbl4_code - val_addr;
                    } else {
                        dc_huff_start_addr[cmp_huff][huff_len - 1] = tbl3_code - val_addr;
                    }

                    val_addr += huff_cnt[huff_len - 1];

                    oneByte(offset, r, c);
                }

                len -= 17;
                val_addr = 0;

                // init the val in each address
                for (huff_len = 1; huff_len <= 16; ++huff_len) {
                    addr_now = addr_all >> huff_len;

                    _XF_IMAGE_PRINT(" Codes of length %d bits (%.3d total):", huff_len, huff_cnt[huff_len - 1]);

                    for (int j = 0; j < huff_cnt[huff_len - 1]; ++j) {
                        //#pragma HLS PIPELINE

                        b1 = segment[r](c * 8 + 7, c * 8);
                        _XF_IMAGE_PRINT(" %.2x", b1);
                        oneByte(offset, r, c);

                        uint8_t run_vlen = b1;
                        uint8_t val_len = run_vlen & 0x0F;
                        uint8_t run_len = (run_vlen & 0xF0) >> 4;
                        uint8_t total_len = val_len + huff_len;
                        uint8_t is_writed = true;
                        ap_uint<16> data = 0;
                        data(4, 0) = total_len;
                        data(9, 5) = huff_len;
                        data(13, 10) = run_len;
                        data[14] = is_writed;
                        data[15] = huff_len > DHT1;
                    //_XF_IMAGE_PRINT(
                    //    " from [%d] to [%d],huff_len=%d, addr_org=%d\n",
                    //    (cnt >> DHT_S), (addr_now + cnt) >> DHT_S, huff_len, addr_now >> DHT_S);

                    GEN_VAL_LOOP:
                        for (int k = addr_now; k > 0; k -= huff_len > DHT1 ? addr_now : (1 << DHT_S)) {
#pragma HLS DEPENDENCE array inter false
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
#pragma HLS PIPELINE II = 1

                            dht_tbl1[ac][cmp_huff][(cnt >> (DHT_S))] = data;

                            if (ac) {
                                ac_val[cmp_huff][val_addr + j] = run_vlen;
                            } else {
                                dc_val[cmp_huff][val_addr + j] = run_vlen;
                            }

                            if (huff_len > DHT1)
                                cnt += addr_now;
                            else
                                cnt += (1 << DHT_S);

                        } // end one val

                    } // end one huff_len

                    val_addr += huff_cnt[huff_len - 1];
                    len -= huff_cnt[huff_len - 1];
                    _XF_IMAGE_PRINT(" \n");

                } // end all huff

// for print
#ifndef __SYNTHESIS__
                if (ac) {
                    for (int i = 0; i < 16; i++) {
                        std::cout << "huffman " << (i + 1) << " bits codes is :0b"
                                  << std::bitset<16>(ac_huff_start_code[cmp_huff][i]) << std::endl;
                    }
                    for (int i = 0; i < 16; i++) {
                        std::cout << "huffman " << (i + 1)
                                  << " bits start addr is :" << (ac_huff_start_addr[cmp_huff][i]) << std::endl;
                    }
                } else {
                    for (int i = 0; i < 16; i++) {
                        std::cout << "huffman " << (i + 1) << " bits codes is :0b"
                                  << std::bitset<16>(dc_huff_start_code[cmp_huff][i]) << std::endl;
                    }
                    for (int i = 0; i < 16; i++) {
                        std::cout << "huffman " << (i + 1)
                                  << " bits start addr is :" << (dc_huff_start_addr[cmp_huff][i]) << std::endl;
                    }
                }
#endif

                // printf(" Warning: huffman table2 is %d cases !\n", test);
                //                if(test > (1 << DHT2)){
                //                	rtn = 2;
                //                	_XF_IMAGE_PRINT(" Warning: huffman table is not logic \n");
                //                }

            } // save one FFC4 marker

        } else if (b2 == 0xDA) { // marker

            _XF_IMAGE_PRINT("------------------------\n");
            // min 12 byte marker
            // 2B  B      B    4    4 x3      B     B   	B
            // L  NS     CSj  Tdj  Taj      ss=0 se=63 ahal=0
            // syn_Scan_decode(size ,len, offset,r,c,  segment, dht_tbl1, dht_tbl2, rtn);
            _XF_IMAGE_PRINT("Scan 0xDA: OFFSET: %.8x\n", offset - 4);

            // read first b + 3*3 b
            b1 = segment[r](c * 8 + 7, c * 8);
            if (b1 != 3) {
                // rtn = false;
                _XF_IMAGE_PRINT(" Warning: SOS, error CMP \n");
            }

            readBytes(offset, 10, r, c);

            // non-interleaving//to be added
            _XF_IMAGE_PRINT("Scan DATA: OFFSET: %.8x\n", offset);

            scanned = true;

        } else {
            _XF_IMAGE_PRINT("Undefined segment  %d\n", b2);
            rtn = 1;
        }

    } // end while

    left = size - offset;
    // To full fill the structure pout with hls_compinfo
    // enum COLOR_FORMAT{C400=0, C420, C422, C444};
    // COLOR_FORMAT format;
    if (hls_compinfo[0].sfv == 2 && hls_compinfo[0].sfh == 2)
        pout->format = C420;
    else if (hls_compinfo[0].sfv == 1 && hls_compinfo[0].sfh == 1)
        pout->format = C444;
    else
        pout->format = C422;
    pout->axi_num_cmp = img_info.hls_cs_cmpc;
    pout->axi_map_row2cmp[0] = 2;
    pout->axi_map_row2cmp[1] = 1;
    pout->axi_map_row2cmp[2] = 0;
    pout->axi_map_row2cmp[3] = 0;
    if (pout->format == C400)
        pout->axi_num_cmp_mcu = 1; //? Not very sure
    else if (pout->format == C420)
        pout->axi_num_cmp_mcu = 4;
    else
        pout->axi_num_cmp_mcu = 3;
    pout->axi_width[0] = hls_compinfo[0].bch;
    pout->axi_width[1] =
        (pout->format == C400) ? 0 : (pout->format == C444) ? hls_compinfo[0].bch : (hls_compinfo[0].bch + 1) >> 1;
    pout->axi_width[2] =
        (pout->format == C400) ? 0 : (pout->format == C444) ? hls_compinfo[0].bch : (hls_compinfo[0].bch + 1) >> 1;
    //
    pout->axi_height[0] = hls_compinfo[0].bcv;
    pout->axi_height[1] =
        (pout->format == C400) ? 0 : (pout->format == C420) ? (hls_compinfo[0].bcv + 1) >> 1 : hls_compinfo[0].bcv;
    pout->axi_height[2] =
        (pout->format == C400) ? 0 : (pout->format == C420) ? (hls_compinfo[0].bcv + 1) >> 1 : hls_compinfo[0].bcv;
    pout->axi_mcuv =
        (pout->format == C400) ? 0 : (pout->format == C420) ? (hls_compinfo[0].bcv + 1) >> 1 : hls_compinfo[0].bcv;

    // SetOtherQtab(pout);//todo add IDCT

    // deal with the [code 1 or 2]
    if (rtn) {
        left = 0;
        c = 0;
    }
}

// ------------------------------------------------------------
void decoder_jpg_top(ap_uint<AXI_WIDTH>* ptr,
                     const int sz,
                     const int c,
                     const uint16_t dht_tbl1[2][2][1 << DHT1],
                     // const uint16_t dht_tbl2[2][2][1 << DHT2],
                     //
                     uint8_t ac_val[2][165],
                     HCODE_T ac_huff_start_code[2][AC_N],
                     int16_t ac_huff_start_addr[2][16],
                     //
                     uint8_t dc_val[2][12],
                     HCODE_T dc_huff_start_code[2][DC_N],
                     int16_t dc_huff_start_addr[2][16],
                     //
                     ap_uint<12> hls_cmp,

                     // image info
                     const uint8_t hls_mbs[MAX_NUM_COLOR],
                     const img_info img_info,

                     bool& rtn2,
                     uint32_t& rst_cnt,
                     hls::stream<ap_uint<24> >& block_strm) {
#pragma HLS DATAFLOW
    // clang-format off
    _XF_IMAGE_PRINT(" ************* start decode %d mcus in FPGA  *************\n", (int)img_info.hls_mcuc);
    _XF_IMAGE_PRINT(
    				"  hls_cs_cmpc=%d, hls_mbs[0]=%d, hls_mbs[1]=%d, hls_mbs[2]=%d, \n",
						img_info.hls_cs_cmpc, hls_mbs[0], hls_mbs[1], hls_mbs[2]);


#pragma HLS ARRAY_PARTITION variable = hls_mbs  complete

//#pragma HLS RESOURCE        variable = dht_tbl1 core = RAM_2P_LUTRAM
//#pragma HLS ARRAY_PARTITION variable = dht_tbl1 complete dim = 0


    hls::stream<CHType> image_strm;
    hls::stream<bool>   eof_strm;
#pragma HLS RESOURCE variable = image_strm core = FIFO_LUTRAM
#pragma HLS STREAM   variable = image_strm depth = 32
#pragma HLS RESOURCE variable = eof_strm core = FIFO_LUTRAM
#pragma HLS STREAM   variable = eof_strm depth = 32
    // clang-format on

    // in case AXI_WIDTH=16 cut down resource
    ap_uint<1> column = c;
    xf::common::utils_hw::axi_to_char_stream<BURST_LENTH, AXI_WIDTH, CHType>(ptr, image_strm, eof_strm, sz,
                                                                             (int)column);

    top_mcu_decoder(image_strm, eof_strm, dht_tbl1, // dht_tbl3, dht_tbl4,
                    ac_val, ac_huff_start_code, ac_huff_start_addr, dc_val, dc_huff_start_code, dc_huff_start_addr,
                    hls_cmp, img_info.hls_cs_cmpc, hls_mbs, img_info.hls_mcuh, img_info.hls_mcuc, rtn2, rst_cnt,
                    block_strm);
}

} // namespace details

// ------------------------------------------------------------
/**
 * @brief Level 2 : kernel for jfif parser + huffman decoder
 *
 * @tparam CH_W size of data path in dataflow region, in bit.
 *         when CH_W is 16, the decoder could decode one symbol per cycle in about 99% cases.
 *         when CH_W is 8 , the decoder could decode one symbol per cycle in about 80% cases, but use less resource.
 *
 * @param datatoDDR the pointer to DDR.
 * @param size the total bytes to be read from DDR.
 * @param img_info information to recovery the image.
 * @param hls_compInfo the component info used to generate the decOutput.
 * @param block_strm the stream of coefficients in block,23:is_rst, 22:is_endblock,21~16:bpos,15~0:block val
 * @param rtn the flag of the jfif parser succeed
 * @param rtn2 the flag of the decode succeed
 */
void kernelParserDecoderTop(ap_uint<AXI_WIDTH>* datatoDDR,
                            const int size,

                            xf::codec::img_info& img_info,
                            xf::codec::hls_compInfo hls_cmpnfo[MAX_NUM_COLOR],
                            hls::stream<ap_uint<24> >& block_strm,
                            int& rtn,
                            bool& rtn2,
                            xf::codec::decOutput* pout) {
    // clang-format off
		//uint64_t max_pix = MAX_NUM_PIX;//for 8K*8K
		const uint64_t max_pix = MAX_DEC_PIX;//for 800*800
		const uint64_t burst_lenth = BURST_LENTH;
		#pragma HLS INTERFACE m_axi port = datatoDDR depth = max_pix offset = slave bundle = gmem_in2 \
		latency = 125 max_read_burst_length = burst_lenth
		#pragma HLS INTERFACE s_axilite port=datatoDDR      bundle=control
		#pragma HLS INTERFACE s_axilite port=return         bundle=control

    // clang-format on

    // for offset = row*scale_char + column
    int r = 0, c = 0;
    int left = 0;
    ap_uint<12> hls_cmp;
    uint32_t rst_cnt;
    uint8_t hls_mbs[MAX_NUM_COLOR] = {0};

    // clang-format off
    uint16_t 					dht_tbl1[2][2][1 << DHT1];
#pragma HLS ARRAY_PARTITION variable = dht_tbl1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = dht_tbl1 complete dim = 2


	uint8_t ac_val[2][ 165 ];//every val relative with the huffman codes
	HCODE_T ac_huff_start_code[2][AC_N];// the huff_start_code<65535
	int16_t  ac_huff_start_addr[2][16];// the addr of the val huff_start_addr<256
	uint8_t dc_val[2][ 12 ];//every val relative with the huffman codes
	HCODE_T dc_huff_start_code[2][DC_N];// the huff_start_code<65535
	int16_t  dc_huff_start_addr[2][16];// the addr of the val huff_start_addr<256
//#pragma HLS RESOURCE 		variable = ac_huff_start_code core = RAM_2P_LUTRAM
#pragma HLS ARRAY_PARTITION variable = ac_huff_start_code complete dim = 2
//#pragma HLS RESOURCE 		variable = dc_huff_start_code core = RAM_2P_LUTRAM
#pragma HLS ARRAY_PARTITION variable = dc_huff_start_code complete dim = 2

#pragma HLS RESOURCE variable = ac_huff_start_addr  core = RAM_2P_LUTRAM
#pragma HLS RESOURCE variable = ac_val 				core = RAM_2P_LUTRAM
#pragma HLS RESOURCE variable = dc_huff_start_addr  core = RAM_2P_LUTRAM
#pragma HLS RESOURCE variable = dc_val 				core = RAM_2P_LUTRAM
    // clang-format on

    // Functions to parser the header before the data burst load from DDR
    //----------------------------------------------------------
    xf::codec::details::parser_jpg_top(datatoDDR, size, r, c, dht_tbl1, ac_val, ac_huff_start_code, ac_huff_start_addr,
                                       dc_val, dc_huff_start_code, dc_huff_start_addr, hls_cmp, left, img_info, hls_mbs,
                                       hls_cmpnfo, rtn, pout);

    ap_uint<AXI_WIDTH>* ptr = datatoDDR + r;

    // Functions to decode the huffman code to non(Inverse quantization+IDCT) block coefficient
    //----------------------------------------------------------
    xf::codec::details::decoder_jpg_top(ptr, left, c, dht_tbl1, // dht_tbl2,
                                        ac_val, ac_huff_start_code, ac_huff_start_addr, dc_val, dc_huff_start_code,
                                        dc_huff_start_addr, hls_cmp, hls_mbs, img_info, rtn2, rst_cnt, block_strm);

#ifndef __SYNTHESIS__
    if (rtn || (rtn2)) {
        fprintf(stderr, "Warning: parser the bad case input file! \n");
    }
#endif
}
} // namespace codec
} // namespace xf

void kernel_parser_decoder(ap_uint<AXI_WIDTH>* datatoDDR,
                           const int size,

                           xf::codec::img_info& img_info,
                           xf::codec::hls_compInfo hls_cmpnfo[MAX_NUM_COLOR],
                           hls::stream<ap_uint<24> >& block_strm,
                           int& rtn,
                           bool& rtn2,
                           xf::codec::decOutput* pout) {
    // clang-format off
		//uint64_t max_pix = MAX_NUM_PIX;//for 8K*8K
		const uint64_t max_pix = MAX_DEC_PIX;//for 800*800
		const uint64_t burst_lenth = BURST_LENTH;
		#pragma HLS INTERFACE m_axi port = datatoDDR depth = max_pix offset = slave bundle = gmem_in2 \
		latency = 125 max_read_burst_length = burst_lenth
		#pragma HLS INTERFACE s_axilite port=datatoDDR      bundle=control
		#pragma HLS INTERFACE s_axilite port=return         bundle=control
    // clang-format on
    kernelParserDecoderTop(datatoDDR, size, img_info, hls_cmpnfo, block_strm, rtn, rtn2, pout);
}
