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
 * @file sm234.hpp
 * @brief header file for SM2 / SM3 / SM4 algorithm related functions.
 * This file is part of Vitis Security Library.
 */

#ifndef _XF_SECURITY_SM234_HPP_
#define _XF_SECURITY_SM234_HPP_

#include <ap_int.h>
#include "xf_security/ecc.hpp"
#include "xf_security/types.hpp"

namespace xf {
namespace security {
namespace internal {

struct sm3BlkPack {
    uint32_t M[16];

    sm3BlkPack() {
#pragma HLS inline
#pragma HLS array_partition variable = M dim = 1 complete
    }
};

static void sm3Packing(hls::stream<ap_uint<64> >& msgStrm,
                       hls::stream<ap_uint<64> >& lenStrm,
                       hls::stream<bool>& endLenStrm,
                       hls::stream<sm3BlkPack>& packStrm,
                       hls::stream<ap_uint<64> >& numPackStrm,
                       hls::stream<bool>& endNumPackStrm) {
    while (!endLenStrm.read()) {
        ap_uint<64> len = lenStrm.read();
        ap_uint<64> numPack = (len >> 9) + 1 + ((len & 0x1ff) > 447);
        numPackStrm.write(numPack);
        endNumPackStrm.write(false);
        for (ap_uint<64> i = 0; i < ap_uint<64>(len >> 9); i++) {
            sm3BlkPack blkPack;
            for (int j = 0; j < 16; j += 2) {
#pragma HLS pipeline II = 8
                uint64_t ll = msgStrm.read().to_uint64();
                uint32_t l = ll & 0xffffffffUL;
                l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8) |
                    ((0xff000000UL & l) >> 24);
                blkPack.M[j] = l;
                l = (ll >> 32) & 0xffffffffUL;
                l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8) |
                    ((0xff000000UL & l) >> 24);
                blkPack.M[j + 1] = l;
                if (j == 14) {
                    packStrm.write(blkPack);
                }
            }
        }
        // last pack
        ap_uint<10> left = len.range(8, 0);
        sm3BlkPack blkPack;
        for (ap_uint<10> i = 0; i < 512; i += 64) {
            if (i < left) {
                ap_uint<10> ii = (i >> 5);
                uint64_t ll = msgStrm.read().to_uint64();
                uint32_t l = ll & 0xffffffffUL;
                l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8) |
                    ((0xff000000UL & l) >> 24);
                blkPack.M[ii] = l;
                l = (ll >> 32) & 0xffffffffUL;
                l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8) |
                    ((0xff000000UL & l) >> 24);
                blkPack.M[ii + 1] = l;
            } else {
                ap_uint<10> ii = (i >> 5);
                blkPack.M[ii] = 0;
                blkPack.M[ii + 1] = 0;
            }
        }
        ap_uint<4> ptr = left >> 5;
        ap_uint<5> res = left & 0x1f;
        blkPack.M[ptr] |= (1 << (31 - res));
        if (left < 448) {
            blkPack.M[14] = len.range(63, 32);
            blkPack.M[15] = len.range(31, 0);
            packStrm.write(blkPack);
        } else {
            packStrm.write(blkPack);
            for (int i = 0; i < 14; i++) {
#pragma HLS unroll
                blkPack.M[i] = 0;
            }
            blkPack.M[14] = len.range(63, 32);
            blkPack.M[15] = len.range(31, 0);
            packStrm.write(blkPack);
        }
    }
    endNumPackStrm.write(true);
}

template <int k>
uint32_t sm3LSH(uint32_t x) {
#pragma HLS inline
    return ((x << k) | (x >> (32 - k)));
}

uint32_t sm3LSHD(uint32_t x, ap_uint<5> k) {
#pragma HLS inline
    return ((x << k) | (x >> (32 - k)));
}

static uint32_t sm3P0(uint32_t x) {
#pragma HLS inline
    return (x ^ sm3LSH<9>(x) ^ sm3LSH<17>(x));
}

static uint32_t sm3P1(uint32_t x) {
#pragma HLS inline
    return (x ^ sm3LSH<15>(x) ^ sm3LSH<23>(x));
}

static uint32_t sm3Tj(int j) {
#pragma HLS inline
    if (j < 16) {
        return 0x79cc4519UL;
    } else {
        return 0x7a879d8aUL;
    }
}

static uint32_t sm3FFj(uint32_t x, uint32_t y, uint32_t z, int j) {
#pragma HLS inline
    if (j < 16) {
        return x ^ y ^ z;
    } else {
        return (x & y) | (x & z) | (y & z);
    }
}

static uint32_t sm3GGj(uint32_t x, uint32_t y, uint32_t z, int j) {
#pragma HLS inline
    if (j < 16) {
        return x ^ y ^ z;
    } else {
        return (x & y) | ((~x) & z);
    }
}

static ap_uint<32> sm3ByteShift(ap_uint<32> x) {
    ap_uint<32> res;
    res.range(31, 24) = x.range(7, 0);
    res.range(23, 16) = x.range(15, 8);
    res.range(15, 8) = x.range(23, 16);
    res.range(7, 0) = x.range(31, 24);
    return res;
}

static void sm3Expand(hls::stream<sm3BlkPack>& packStrm,
                      hls::stream<ap_uint<64> >& numPackStrm,
                      hls::stream<bool>& endNumPackStrm,
                      hls::stream<ap_uint<256> >& digestStrm,
                      hls::stream<bool>& endDigestStrm) {
    while (!endNumPackStrm.read()) {
        ap_uint<64> numPack = numPackStrm.read();
        ap_uint<32> regA = ap_uint<32>("0x7380166f");
        ap_uint<32> regB = ap_uint<32>("0x4914b2b9");
        ap_uint<32> regC = ap_uint<32>("0x172442d7");
        ap_uint<32> regD = ap_uint<32>("0xda8a0600");
        ap_uint<32> regE = ap_uint<32>("0xa96f30bc");
        ap_uint<32> regF = ap_uint<32>("0x163138aa");
        ap_uint<32> regG = ap_uint<32>("0xe38dee4d");
        ap_uint<32> regH = ap_uint<32>("0xb0fb0e4e");
        ap_uint<32> recA = regA;
        ap_uint<32> recB = regB;
        ap_uint<32> recC = regC;
        ap_uint<32> recD = regD;
        ap_uint<32> recE = regE;
        ap_uint<32> recF = regF;
        ap_uint<32> recG = regG;
        ap_uint<32> recH = regH;

        sm3BlkPack pack;
        uint32_t msgWord[68];
#pragma HLS array_partition variable = msgWord dim = 1
        uint32_t msgWordP[64];
#pragma HLS array_partition variable = msgWordP dim = 1

        for (ap_uint<64> i = 0; i < numPack; i++) {
            for (int j = 0; j < 68; j++) {
#pragma HLS pipeline II = 1
                if (j == 0) {
                    pack = packStrm.read();
                }
                if (j < 16) {
                    msgWord[j] = pack.M[j];
                } else {
                    msgWord[j] = sm3P1(msgWord[j - 16] ^ msgWord[j - 9] ^ (sm3LSH<15>(msgWord[j - 3]))) ^
                                 (sm3LSH<7>(msgWord[j - 13])) ^ msgWord[j - 6];
                }
                if (j >= 4) {
                    msgWordP[j - 4] = msgWord[j - 4] ^ msgWord[j];
                    ap_uint<6> jj = j - 4;
                    ap_uint<5> jjj = jj.range(4, 0);

                    uint32_t ss1, ss2, tt1, tt2;
                    ss1 = sm3LSH<7>(sm3LSH<12>(regA) + regE + sm3LSHD(sm3Tj(jj), jjj));
                    ss2 = ss1 ^ sm3LSH<12>(regA);
                    tt1 = sm3FFj(regA, regB, regC, jj) + regD + ss2 + msgWordP[jj];
                    tt2 = sm3GGj(regE, regF, regG, jj) + regH + ss1 + msgWord[jj];
                    regD = regC;
                    regC = sm3LSH<9>(regB);
                    regB = regA;
                    regA = tt1;
                    regH = regG;
                    regG = sm3LSH<19>(regF);
                    regF = regE;
                    regE = sm3P0(tt2);
                }
                if (j == 67) {
                    regA ^= recA;
                    regB ^= recB;
                    regC ^= recC;
                    regD ^= recD;
                    regE ^= recE;
                    regF ^= recF;
                    regG ^= recG;
                    regH ^= recH;

                    recA = regA;
                    recB = regB;
                    recC = regC;
                    recD = regD;
                    recE = regE;
                    recF = regF;
                    recG = regG;
                    recH = regH;
                }
            }
        }
        ap_uint<256> digest;
        digest.range(31, 0) = sm3ByteShift(regA);
        digest.range(63, 32) = sm3ByteShift(regB);
        digest.range(95, 64) = sm3ByteShift(regC);
        digest.range(127, 96) = sm3ByteShift(regD);
        digest.range(159, 128) = sm3ByteShift(regE);
        digest.range(191, 160) = sm3ByteShift(regF);
        digest.range(223, 192) = sm3ByteShift(regG);
        digest.range(255, 224) = sm3ByteShift(regH);
        digestStrm.write(digest);
        endDigestStrm.write(false);
    }
    endDigestStrm.write(true);
}

template <int k>
uint32_t sm4LSH(uint32_t x) {
#pragma HLS inline
    return ((x << k) | (x >> (32 - k)));
}

} // namespace internal

/**
 * @brief SM2 algorithm related function.
 * This class provide signing and verifying functions.
 *
 * @tparam W Bit width of SM2 curve's parameters.
 */
template <int W>
class sm2 : public xf::security::ecc<W> {
   public:
    /// X coordinate of generation point
    ap_uint<W> Gx;
    /// Y coordinate of generation point
    ap_uint<W> Gy;
    /// Order of generation point
    ap_uint<W> n;

    sm2() {
#pragma HLS inline
    }

    /**
     * @brief Setup parameters for curve y^2 = x^3 + ax + b in GF(p)
     *
     * @param inputA Parameter a for y^2 = x^3 + ax + b in GF(p)
     * @param inputB Parameter b for y^2 = x^3 + ax + b in GF(p)
     * @param inputP Parameter p for y^2 = x^3 + ax + b in GF(p)
     * @param inputGx X coordinate of generation point G.
     * @param inputGy Y coordinate of generation point G.
     * @param inputN Order of generation point.
     */
    void init(ap_uint<W> inputA,
              ap_uint<W> inputB,
              ap_uint<W> inputP,
              ap_uint<W> inputGx,
              ap_uint<W> inputGy,
              ap_uint<W> inputN) {
        this->a = inputA;
        this->b = inputB;
        this->p = inputP;
        Gx = inputGx;
        Gy = inputGy;
        n = inputN;
    };

    /**
     * @brief signing function.
     * It will return true if input parameters are legal, otherwise return false.
     *
     * @param hashZaM Digest value of message to be signed.
     * @param k A random key to sign the message, should kept different each time to be used.
     * @param privateKey Private Key to sign the message
     * @param r part of signing pair {r, s}
     * @param s part of signing pair {r, s}
     */
    bool sign(ap_uint<W> hashZaM, ap_uint<W> k, ap_uint<W> privateKey, ap_uint<W>& r, ap_uint<256>& s) {
        bool valid = true;
        ap_uint<W> x, y;
        this->dotProduct(Gx, Gy, k, x, y);

        if (x >= n) {
            x -= n;
        }

        if (hashZaM >= n) {
            hashZaM -= n;
        }

        ap_uint<W> tmpr, tmps;
        tmpr = xf::security::internal::addMod<W>(hashZaM, x, n);
        if (tmpr == 0 || xf::security::internal::addMod<W>(tmpr, k, n) == 0) {
            valid = false;
        } else {
            ap_uint<W> da1 = xf::security::internal::addMod<W>(privateKey, 1, n);
            ap_uint<W> da1Inv = xf::security::internal::modularInv<W>(da1, n);
            ap_uint<W> rda = xf::security::internal::productMod<W>(privateKey, tmpr, n);
            ap_uint<W> krda = xf::security::internal::subMod<W>(k, rda, n);
            tmps = xf::security::internal::productMod<W>(da1Inv, krda, n);
            if (tmps == 0) {
                valid = false;
            }
        }

        r = tmpr;
        s = tmps;
        return valid;
    }

    /**
     * @brief verifying function.
     * It will return true if verified, otherwise false.
     *
     * @param r part of signing pair {r, s}
     * @param s part of signing pair {r, s}
     * @param hashZaM Digest value of message to be signed.
     * @param Px X coordinate of public key point P.
     * @param Py Y coordinate of public key point P.
     */
    bool verify(ap_uint<W> r, ap_uint<W> s, ap_uint<W> hashZaM, ap_uint<W> Px, ap_uint<W> Py) {
        bool valid = true;
        if (r == 0 || r >= n || s == 0 || s >= n) {
            valid = false;
        }

        ap_uint<W> t = xf::security::internal::addMod<W>(r, s, n);
        if (t == 0) {
            valid = false;
        }

        ap_uint<W> tmpX, tmpY, x, y;
        this->dotProduct(Px, Py, t, tmpX, tmpY);
        this->dotProduct(Gx, Gy, s, x, y);
        this->add(x, y, tmpX, tmpY, x, y);

        if (hashZaM >= n) {
            hashZaM -= n;
        }
        ap_uint<W> res = xf::security::internal::addMod<W>(hashZaM, x, n);

        if (res != r) {
            valid = false;
        }

        return valid;
    }
};

/**
 * @brief SM3 function to genrate digest value for input messages.
 * @param msgStrm Stream to input messages to be signed.
 * @param lenStrm Stream to input length of input messages.
 * @param endLenStrm Stream of end flag of lenStrm.
 * @param hashStrm Stream to output digests of messges.
 * @param endHashStrm Stream of end flag of hashStrm.
 */
void sm3(hls::stream<ap_uint<64> >& msgStrm,
         hls::stream<ap_uint<64> >& lenStrm,
         hls::stream<bool>& endLenStrm,
         hls::stream<ap_uint<256> >& hashStrm,
         hls::stream<bool>& endHashStrm) {
#pragma HLS dataflow
    hls::stream<xf::security::internal::sm3BlkPack> packStrm;
#pragma HLS stream variable = packStrm depth = 4
#pragma HLS resource variable = packStrm core = FIFO_LUTRAM
    hls::stream<ap_uint<64> > numPackStrm;
#pragma HLS stream variable = numPackStrm depth = 4
#pragma HLS resource variable = numPackStrm core = FIFO_LUTRAM
    hls::stream<bool> endNumPackStrm;
#pragma HLS stream variable = endNumPackStrm depth = 4
#pragma HLS resource variable = endNumPackStrm core = FIFO_LUTRAM

    xf::security::internal::sm3Packing(msgStrm, lenStrm, endLenStrm, packStrm, numPackStrm, endNumPackStrm);
    xf::security::internal::sm3Expand(packStrm, numPackStrm, endNumPackStrm, hashStrm, endHashStrm);
}

/**
 * @brief SM4 encryption / decryption functions.
 */
class sm4 {
   public:
    const ap_uint<8> sbox[256] = {
        0xd6, 0x90, 0xe9, 0xfe, 0xcc, 0xe1, 0x3d, 0xb7, 0x16, 0xb6, 0x14, 0xc2, 0x28, 0xfb, 0x2c, 0x05, 0x2b, 0x67,
        0x9a, 0x76, 0x2a, 0xbe, 0x04, 0xc3, 0xaa, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99, 0x9c, 0x42, 0x50, 0xf4,
        0x91, 0xef, 0x98, 0x7a, 0x33, 0x54, 0x0b, 0x43, 0xed, 0xcf, 0xac, 0x62, 0xe4, 0xb3, 0x1c, 0xa9, 0xc9, 0x08,
        0xe8, 0x95, 0x80, 0xdf, 0x94, 0xfa, 0x75, 0x8f, 0x3f, 0xa6, 0x47, 0x07, 0xa7, 0xfc, 0xf3, 0x73, 0x17, 0xba,
        0x83, 0x59, 0x3c, 0x19, 0xe6, 0x85, 0x4f, 0xa8, 0x68, 0x6b, 0x81, 0xb2, 0x71, 0x64, 0xda, 0x8b, 0xf8, 0xeb,
        0x0f, 0x4b, 0x70, 0x56, 0x9d, 0x35, 0x1e, 0x24, 0x0e, 0x5e, 0x63, 0x58, 0xd1, 0xa2, 0x25, 0x22, 0x7c, 0x3b,
        0x01, 0x21, 0x78, 0x87, 0xd4, 0x00, 0x46, 0x57, 0x9f, 0xd3, 0x27, 0x52, 0x4c, 0x36, 0x02, 0xe7, 0xa0, 0xc4,
        0xc8, 0x9e, 0xea, 0xbf, 0x8a, 0xd2, 0x40, 0xc7, 0x38, 0xb5, 0xa3, 0xf7, 0xf2, 0xce, 0xf9, 0x61, 0x15, 0xa1,
        0xe0, 0xae, 0x5d, 0xa4, 0x9b, 0x34, 0x1a, 0x55, 0xad, 0x93, 0x32, 0x30, 0xf5, 0x8c, 0xb1, 0xe3, 0x1d, 0xf6,
        0xe2, 0x2e, 0x82, 0x66, 0xca, 0x60, 0xc0, 0x29, 0x23, 0xab, 0x0d, 0x53, 0x4e, 0x6f, 0xd5, 0xdb, 0x37, 0x45,
        0xde, 0xfd, 0x8e, 0x2f, 0x03, 0xff, 0x6a, 0x72, 0x6d, 0x6c, 0x5b, 0x51, 0x8d, 0x1b, 0xaf, 0x92, 0xbb, 0xdd,
        0xbc, 0x7f, 0x11, 0xd9, 0x5c, 0x41, 0x1f, 0x10, 0x5a, 0xd8, 0x0a, 0xc1, 0x31, 0x88, 0xa5, 0xcd, 0x7b, 0xbd,
        0x2d, 0x74, 0xd0, 0x12, 0xb8, 0xe5, 0xb4, 0xb0, 0x89, 0x69, 0x97, 0x4a, 0x0c, 0x96, 0x77, 0x7e, 0x65, 0xb9,
        0xf1, 0x09, 0xc5, 0x6e, 0xc6, 0x84, 0x18, 0xf0, 0x7d, 0xec, 0x3a, 0xdc, 0x4d, 0x20, 0x79, 0xee, 0x5f, 0x3e,
        0xd7, 0xcb, 0x39, 0x48};

    const ap_uint<32> ck[32] = {0x00070e15, 0x1c232a31, 0x383f464d, 0x545b6269, 0x70777e85, 0x8c939aa1, 0xa8afb6bd,
                                0xc4cbd2d9, 0xe0e7eef5, 0xfc030a11, 0x181f262d, 0x343b4249, 0x50575e65, 0x6c737a81,
                                0x888f969d, 0xa4abb2b9, 0xc0c7ced5, 0xdce3eaf1, 0xf8ff060d, 0x141b2229, 0x30373e45,
                                0x4c535a61, 0x686f767d, 0x848b9299, 0xa0a7aeb5, 0xbcc3cad1, 0xd8dfe6ed, 0xf4fb0209,
                                0x10171e25, 0x2c333a41, 0x484f565d, 0x646b7279};

    ap_uint<32> rk[36]; // first 4 is not used

   private:
    ap_uint<32> T(ap_uint<32> x) {
#pragma HLS inline
        ap_uint<32> tmp;
        for (int i = 0; i < 4; i++) {
#pragma HLS unroll
            tmp.range(i * 8 + 7, i * 8) = sbox[x.range(i * 8 + 7, i * 8)];
        }
        return tmp ^ xf::security::internal::sm4LSH<2>(tmp) ^ xf::security::internal::sm4LSH<10>(tmp) ^
               xf::security::internal::sm4LSH<18>(tmp) ^ xf::security::internal::sm4LSH<24>(tmp);
    }

    ap_uint<32> TP(ap_uint<32> x) {
#pragma HLS inline
        ap_uint<32> tmp;
        for (int i = 0; i < 4; i++) {
#pragma HLS unroll
            tmp.range(i * 8 + 7, i * 8) = sbox[x.range(i * 8 + 7, i * 8)];
        }
        return tmp ^ xf::security::internal::sm4LSH<13>(tmp) ^ xf::security::internal::sm4LSH<23>(tmp);
    }

    ap_uint<32> F(ap_uint<32> x0, ap_uint<32> x1, ap_uint<32> x2, ap_uint<32> x3, ap_uint<32> rk) {
#pragma HLS inline
        return x0 ^ T(x1 ^ x2 ^ x3 ^ rk);
    }

   public:
    sm4() {
#pragma HLS inline
#pragma HLS array_partition variable = sbox dim = 1
    }

    /**
     * @brief Genrate extension key used in encryption/decryption.
     * @param key Key.
     */
    void keyExtension(ap_uint<128> key) {
        const ap_uint<32> FK[4] = {0xA3B1BAC6, 0x56AA3350, 0x677D9197, 0xB27022DC};
        for (int i = 0; i < 4; i++) {
#pragma HLS pipeline
            rk[i] = key.range(i * 32 + 31, i * 32) ^ FK[i];
        }
        for (int i = 4; i < 36; i++) {
#pragma HLS pipeline
            rk[i] = rk[i - 4] ^ TP(rk[i - 3] ^ rk[i - 2] ^ rk[i - 1] ^ ck[i - 4]);
        }
    }

    /**
     * @brief encryption function.
     * @param input Input block for encryption.
     * @param output Output of encrypted block.
     */
    void encrypt(ap_uint<128> input, ap_uint<128>& output) {
        for (int i = 0; i < 32; i++) {
#pragma HLS pipeline
            ap_uint<32> tmp =
                F(input.range(31, 0), input.range(63, 32), input.range(95, 64), input.range(127, 96), rk[i + 4]);
            input >>= 32;
            input.range(127, 96) = tmp;
        }
        ap_uint<128> tmp;
        tmp.range(127, 96) = input.range(31, 0);
        tmp.range(95, 64) = input.range(63, 32);
        tmp.range(63, 32) = input.range(95, 64);
        tmp.range(31, 0) = input.range(127, 96);
        output = tmp;
    }

    /**
     * @brief decryption function.
     * @param input Input block for decryption.
     * @param output Output of decrypted block.
     */
    void decrypt(ap_uint<128> input, ap_uint<128>& output) {
        for (int i = 0; i < 32; i++) {
#pragma HLS pipeline
            ap_uint<32> tmp =
                F(input.range(31, 0), input.range(63, 32), input.range(95, 64), input.range(127, 96), rk[35 - i]);
            input >>= 32;
            input.range(127, 96) = tmp;
        }
        ap_uint<128> tmp;
        tmp.range(127, 96) = input.range(31, 0);
        tmp.range(95, 64) = input.range(63, 32);
        tmp.range(63, 32) = input.range(95, 64);
        tmp.range(31, 0) = input.range(127, 96);
        output = tmp;
    }
};

} // namespace security
} // namespace xf

#endif
