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
 * @file sobol_rsg.hpp
 * @brief This file include first dimension sequence generator
 * and 128-dimension sobol sequence generator
 *
 */

#ifndef XF_FINTECH_RSG_H
#define XF_FINTECH_RSG_H

#include "ap_fixed.h"
#include "ap_int.h"

namespace xf {
namespace fintech {

/**
 * @brief SobolRsg is a multi-dimensional sobol sequence generator.
 *
 * @tparam DIM sobol sequence dimension, maximum is 128
 */

template <int DIM>
class SobolRsg {
   private:
    /*
     * initPara is the first 128 dimensions from the file new-joe-kuo-6.21201
     * in https://web.maths.unsw.edu.au/~fkuo/sobol/, where
     * bit 0~7 is a, bit 8 is m1, bit 9~10 is m2, bit 11~13 is m3,
     * bit 14~17 is m4, bit 18~22 is m5, bit 23~28 is m6,
     * bit 29~35 is m7, bit 36~43 is m8, bit 44~52 is m9,
     * bit 53~62 is m10.
     * */
    const ap_uint<63> initPara[128] = {0x0,
                                       0x100,
                                       0x701,
                                       0xF01,
                                       0xB02,
                                       0xDB01,
                                       0x36F04,
                                       0x456B02,
                                       0x156B04,
                                       0x4EFB07,
                                       0x46B0B,
                                       0x2CCB0D,
                                       0x7D6F0E,
                                       0x189E5F01,
                                       0xAD7CB0D,
                                       0x18EF4F10,
                                       0x29FCB13,
                                       0xCB7CF16,
                                       0x1ECD6B19,
                                       0xCE7DEFF01,
                                       0x8A7B77F04,
                                       0x7F19F5B07,
                                       0x6AC866F08,
                                       0xD71A74F0E,
                                       0x3FEED4F13,
                                       0x7B4CEEB15,
                                       0x8A68CEF1C,
                                       0x29877B1F,
                                       0x769B57F20,
                                       0x52EE65B25,
                                       0x6E0DF6F29,
                                       0x23DB4FF2A,
                                       0x8BA94CF32,
                                       0x1B0DD6B37,
                                       0xF7E85FB38,
                                       0x63EB67B3B,
                                       0x43B8D5F3E,
                                       0xF5626FFCF0E,
                                       0x617FDFFEF15,
                                       0xF99A5AECF16,
                                       0x98F5EECF26,
                                       0x2DA25D7FB2F,
                                       0x4F82FE4FF31,
                                       0xCD065CC4F32,
                                       0x9D3AACE6B34,
                                       0xB9B3086FF38,
                                       0x479E4BCDF43,
                                       0x1BEF3BEFF46,
                                       0xE1C2FAC5B54,
                                       0xB1735DCCB61,
                                       0x474A8C5FF67,
                                       0xD5F7FED4F73,
                                       0x856B5AD5B7A,
                                       0x1DFAD5E8F56F08,
                                       0x4509DA08EDF0D,
                                       0x157052F3C54B10,
                                       0x1F3673E7E54F16,
                                       0xB7697E8AECB19,
                                       0x16BE7C2EA6EB2C,
                                       0x17F07536CFEB2F,
                                       0xDD89A69FDFF34,
                                       0x53DFDE7DCCB37,
                                       0xA1196E7FF6B3B,
                                       0x101574F7E75B3E,
                                       0x125F9FBAD6CB43,
                                       0x1434F723AEFB4A,
                                       0x8303A26C56B51,
                                       0x1DBFB823DF7B52,
                                       0xB95075A46F57,
                                       0x1E7FF1A6FF5B5B,
                                       0x7F5BB3F945F5E,
                                       0xED7FF6984DB67,
                                       0x121F34AFDDEB68,
                                       0x1EBB7EBAC6EB6D,
                                       0x159D11A6854B7A,
                                       0x2107E7C87DB7C,
                                       0xAFCFA359ECF89,
                                       0x31FF7EDBC4F8A,
                                       0x131ABD3EECEF8F,
                                       0x95F972184EB91,
                                       0x9F0D1FC955B98,
                                       0xE18DD259ECB9D,
                                       0x10F65F3DED5FA7,
                                       0x733B678AE6FAD,
                                       0x1A347FB6DC7BB0,
                                       0x4B6DD22DD5BB5,
                                       0x1C5798659FFBB6,
                                       0x1C11B3E6A4FFB9,
                                       0xF594F3CFCFBF,
                                       0x17B91930844BC2,
                                       0x1E30D3B5BFCFC7,
                                       0x1AF83AADCCFBDA,
                                       0x15DC32F194DFDC,
                                       0x1293B4EDA5DFE3,
                                       0x9DF11A8AE5BE5,
                                       0xD5BD43CE7FFE6,
                                       0xD95393BA47BEA,
                                       0xF9712EDCF5FEC,
                                       0x1DFFD075DCEFF1,
                                       0xD9755A2AD6BF4,
                                       0x937B432F5DFFD,
                                       0x396DFE34A297CF04,
                                       0x3CE87FF7F3957B0D,
                                       0x4AED9F9AE3A5CF13,
                                       0x1EF6BE10F7A75B16,
                                       0x5C20943126CF7F32,
                                       0x54B3F290FDCD6F37,
                                       0x62ACF2B1FFFCEB40,
                                       0x153F12F073B67B45,
                                       0x7139F13C28D5CF62,
                                       0xFEA56F8EF8C7F6B,
                                       0x69ECB77A7E86EB73,
                                       0x462F6127EA75F79,
                                       0x3ABA15F7EEBDFB7F,
                                       0x302D5398E4E64F86,
                                       0x2AA2739CB7FF6F8C,
                                       0x44F6DADFBCFCDB91,
                                       0x587C39D87CB47F98,
                                       0x78BAD59D26D5CB9E,
                                       0x11A9D775B9C66BA1,
                                       0x5CA8109B76B5FFAB,
                                       0x476978D87CDC7FB5,
                                       0x13B776BBB7C6DBC2,
                                       0x726A93356AAD5FC7,
                                       0x4E3C743CBBBCEBCB,
                                       0x4A7592F3AB866FD0,
                                       0x49B439B3B895FFE3,
                                       0x59B053DFF495DFF2};
    // Bit width of element in state vector
    const static int W = 32;
    // Log2W = log2(W)
    const static int Log2W = 5;
    // addr is a counter represents the sequence output order
    ap_uint<W> addr;
    // s is the degree of the primitive polynomial
    ap_uint<4> s[DIM];
    // a is the number representing the coefficients
    ap_uint<8> a[DIM];
    // v include the list of initial direction numbers
    ap_uint<W> v[DIM][W];
    // last seqOut
    ap_uint<W> last_seqOut[DIM];
    ap_uint<Log2W> c_init[DIM];

   public:
    SobolRsg(){
#pragma HLS inline
#pragma HLS RESOURCE variable = initPara core = ROM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable = v dim = 0
#pragma HLS ARRAY_PARTITION variable = s dim = 0
#pragma HLS ARRAY_PARTITION variable = a dim = 0
#pragma HLS ARRAY_PARTITION variable = c_init dim = 0
#pragma HLS ARRAY_PARTITION variable = last_seqOut dim = 0
    };

    /**
     * @brief sobol parameter initialization
     *
     */
    void initialization() {
#pragma HLS inline off
// work-around for vivado_hls.
// It cannot correctly pass the pragma from class constructor to class member function.
// No one wants to touch the freezed vivado before 20.1 release.
// The corresponding CR-1055714 is assigned to Yi Gao.
// Will be fixed after 20.1 release.
#pragma HLS RESOURCE variable = initPara core = ROM_2P_BRAM
        ap_uint<8> i;
        ap_uint<5> j;
        ap_uint<7> begin;

        addr = 0;
        c_init[0] = 0;
        v[0][0] = 2147483648;

        for (i = 1; i < DIM; i++) {
#pragma HLS unroll
            ap_uint<63> init_para = initPara[i];
            if (i == 1)
                s[i] = 1;
            else if (i == 2)
                s[i] = 2;
            else if (i <= 4)
                s[i] = 3;
            else if (i <= 6)
                s[i] = 4;
            else if (i <= 12)
                s[i] = 5;
            else if (i <= 18)
                s[i] = 6;
            else if (i <= 36)
                s[i] = 7;
            else if (i <= 52)
                s[i] = 8;
            else if (i <= 100)
                s[i] = 9;
            else
                s[i] = 10;
            a[i] = init_para(7, 0); // 8bit
            begin = 8;
            for (j = 0; j < 10; j++) {
#pragma HLS unroll
                v[i][j] = (init_para(begin + j, begin)) << (W - j - 1);
                begin += 1 + j;
            }
            c_init[i] = s[i] - 1;
        }
    }
    /**
     * @brief each call of next() generates sobol sequence numbers in DIM
     * dimensions, one number per dimension
     *
     * @param seqOut sobol results in DIM dimensions
     */
    void next(ap_ufixed<W, 0> seqOut[DIM]) {
//#ddpragma HLS inline off
#pragma HLS PIPELINE
        ap_uint<Log2W> c;
        ap_uint<8> id;
        if (addr == 0) {
            for (id = 0; id < DIM; id++) {
#pragma HLS unroll
                last_seqOut[id] = 0;
                seqOut[id](W - 1, 0) = 0;
            }
        } else {
            ap_uint<W> temp;
            temp = addr - 1;
            c = 0;
            ap_uint<6> i;
            for (i = 0; i < W; ++i) {
#pragma HLS unroll
                if (~temp[i]) break;
            }
            c = i;

            ap_uint<W> v_now;
            v_now = 1 << (W - c - 1);
            last_seqOut[0] = last_seqOut[0] ^ v_now;
            seqOut[0](W - 1, 0) = last_seqOut[0](W - 1, 0);
        loop_0:
            for (id = 1; id < DIM; id++) {
#pragma HLS unroll
                ap_uint<W> v_tmp1;
                ap_uint<W> v_tmp2;
                if (c > c_init[id]) { // c_init
                    v_tmp1 = v[id][c - s[id]];
                    v_tmp2 = v_tmp1 >> s[id];
                    v_now = v_tmp1 ^ v_tmp2;
                    ap_uint<4> i;
                loop_1:
                    for (i = 1; i < 10; i++) {
                        if ((a[id] >> (s[id] - 1 - i)) & 1)
                            v_now ^= v[id][c - i];
                        else
                            v_now = v_now;
                    }
                    v[id][c] = v_now;
                } else {
                    v_now = v[id][c];
                }
                last_seqOut[id] = last_seqOut[id] ^ v_now;
                seqOut[id](W - 1, 0) = last_seqOut[id](W - 1, 0);
            }
        }
        addr++;
    }
};

/**
 *
 * @brief One dimensional sobol sequence generator.
 *
 */
class SobolRsg1D {
   private:
    /// Bit width of element in state vector
    static const int W = 32;
    // Log2W = log2(W)
    static const int Log2W = 5;

    // addr is a counter represents the sequence output order
    ap_uint<W> addr;
    // last seqOut
    ap_uint<W> last_seqOut;

   public:
    SobolRsg1D(){
#pragma HLS inline
    };

    /**
     * @brief sobol parameter initialization
     *
     */
    void initialization() { addr = 0; }

    /**
     * @brief each call of next() generate a sobol number
     *
     * @param seqOut 1d sobol result
     */
    void next(ap_ufixed<W, 0>* seqOut) {
//#pragma HLS inline off
#pragma HLS PIPELINE
        ap_uint<Log2W> c;
        ap_uint<W> v;
        ap_ufixed<W, 0> tmp_seqOut;
        ap_uint<W> tmp[8];

        if (addr == 0) {
            last_seqOut = 0;
            *seqOut = 0;
        } else {
            ap_uint<W> temp;
            temp = addr - 1;
            c = 0;
            ap_uint<6> i;
            for (i = 0; i < W; ++i) {
#pragma HLS unroll
                if (~temp[i]) break;
            }
            c = i;

            v = 1 << (W - c - 1);
            last_seqOut = last_seqOut ^ v;
            tmp_seqOut(W - 1, 0) = last_seqOut(W - 1, 0);
            *seqOut = tmp_seqOut;
        }
        addr++;
    }
};

} // namespace fintech
} // namespace xf
#endif // ifndef XF_FINTECH_RSG_H
