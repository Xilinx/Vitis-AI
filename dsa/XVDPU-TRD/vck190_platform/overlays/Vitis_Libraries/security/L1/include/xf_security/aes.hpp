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
 * @file aes.hpp
 * @brief header file for Advanced Encryption Standard relate function.
 * This file part of Vitis Security Library.
 *
 * @detail Currently we have Aes256_Encryption for AES256 standard.
 */

#ifndef _XF_SECURITY_AES_HPP_
#define _XF_SECURITY_AES_HPP_

#include <ap_int.h>
#include <hls_stream.h>
#ifndef __SYNTHESIS__
#include <iostream>
#endif
namespace xf {
namespace security {

class aesTable {
   public:
    aesTable() {
#pragma HLS inline
#pragma HLS resource variable = ssbox core = ROM_nP_LUTRAM
#pragma HLS resource variable = iibox core = ROM_nP_LUTRAM
#pragma HLS resource variable = i32box core = ROM_nP_LUTRAM
#pragma HLS resource variable = p16box core = ROM_nP_LUTRAM
    }

    const ap_uint<16> p16box[256] = {
        0x0,    0x302,  0x604,  0x506,  0x0c08, 0x0f0a, 0x0a0c, 0x090e, 0x1810, 0x1b12, 0x1e14, 0x1d16, 0x1418, 0x171a,
        0x121c, 0x111e, 0x3020, 0x3322, 0x3624, 0x3526, 0x3c28, 0x3f2a, 0x3a2c, 0x392e, 0x2830, 0x2b32, 0x2e34, 0x2d36,
        0x2438, 0x273a, 0x223c, 0x213e, 0x6040, 0x6342, 0x6644, 0x6546, 0x6c48, 0x6f4a, 0x6a4c, 0x694e, 0x7850, 0x7b52,
        0x7e54, 0x7d56, 0x7458, 0x775a, 0x725c, 0x715e, 0x5060, 0x5362, 0x5664, 0x5566, 0x5c68, 0x5f6a, 0x5a6c, 0x596e,
        0x4870, 0x4b72, 0x4e74, 0x4d76, 0x4478, 0x477a, 0x427c, 0x417e, 0xc080, 0xc382, 0xc684, 0xc586, 0xcc88, 0xcf8a,
        0xca8c, 0xc98e, 0xd890, 0xdb92, 0xde94, 0xdd96, 0xd498, 0xd79a, 0xd29c, 0xd19e, 0xf0a0, 0xf3a2, 0xf6a4, 0xf5a6,
        0xfca8, 0xffaa, 0xfaac, 0xf9ae, 0xe8b0, 0xebb2, 0xeeb4, 0xedb6, 0xe4b8, 0xe7ba, 0xe2bc, 0xe1be, 0xa0c0, 0xa3c2,
        0xa6c4, 0xa5c6, 0xacc8, 0xafca, 0xaacc, 0xa9ce, 0xb8d0, 0xbbd2, 0xbed4, 0xbdd6, 0xb4d8, 0xb7da, 0xb2dc, 0xb1de,
        0x90e0, 0x93e2, 0x96e4, 0x95e6, 0x9ce8, 0x9fea, 0x9aec, 0x99ee, 0x88f0, 0x8bf2, 0x8ef4, 0x8df6, 0x84f8, 0x87fa,
        0x82fc, 0x81fe, 0x9b1b, 0x9819, 0x9d1f, 0x9e1d, 0x9713, 0x9411, 0x9117, 0x9215, 0x830b, 0x8009, 0x850f, 0x860d,
        0x8f03, 0x8c01, 0x8907, 0x8a05, 0xab3b, 0xa839, 0xad3f, 0xae3d, 0xa733, 0xa431, 0xa137, 0xa235, 0xb32b, 0xb029,
        0xb52f, 0xb62d, 0xbf23, 0xbc21, 0xb927, 0xba25, 0xfb5b, 0xf859, 0xfd5f, 0xfe5d, 0xf753, 0xf451, 0xf157, 0xf255,
        0xe34b, 0xe049, 0xe54f, 0xe64d, 0xef43, 0xec41, 0xe947, 0xea45, 0xcb7b, 0xc879, 0xcd7f, 0xce7d, 0xc773, 0xc471,
        0xc177, 0xc275, 0xd36b, 0xd069, 0xd56f, 0xd66d, 0xdf63, 0xdc61, 0xd967, 0xda65, 0x5b9b, 0x5899, 0x5d9f, 0x5e9d,
        0x5793, 0x5491, 0x5197, 0x5295, 0x438b, 0x4089, 0x458f, 0x468d, 0x4f83, 0x4c81, 0x4987, 0x4a85, 0x6bbb, 0x68b9,
        0x6dbf, 0x6ebd, 0x67b3, 0x64b1, 0x61b7, 0x62b5, 0x73ab, 0x70a9, 0x75af, 0x76ad, 0x7fa3, 0x7ca1, 0x79a7, 0x7aa5,
        0x3bdb, 0x38d9, 0x3ddf, 0x3edd, 0x37d3, 0x34d1, 0x31d7, 0x32d5, 0x23cb, 0x20c9, 0x25cf, 0x26cd, 0x2fc3, 0x2cc1,
        0x29c7, 0x2ac5, 0x0bfb, 0x08f9, 0x0dff, 0x0efd, 0x7f3,  0x4f1,  0x1f7,  0x2f5,  0x13eb, 0x10e9, 0x15ef, 0x16ed,
        0x1fe3, 0x1ce1, 0x19e7, 0x1ae5};

    const ap_uint<32> i32box[256] = {
        0x0,        0x0e0d0b09, 0x1c1a1612, 0x12171d1b, 0x38342c24, 0x3639272d, 0x242e3a36, 0x2a23313f, 0x70685848,
        0x7e655341, 0x6c724e5a, 0x627f4553, 0x485c746c, 0x46517f65, 0x5446627e, 0x5a4b6977, 0xe0d0b090, 0xeeddbb99,
        0xfccaa682, 0xf2c7ad8b, 0xd8e49cb4, 0xd6e997bd, 0xc4fe8aa6, 0xcaf381af, 0x90b8e8d8, 0x9eb5e3d1, 0x8ca2feca,
        0x82aff5c3, 0xa88cc4fc, 0xa681cff5, 0xb496d2ee, 0xba9bd9e7, 0xdbbb7b3b, 0xd5b67032, 0xc7a16d29, 0xc9ac6620,
        0xe38f571f, 0xed825c16, 0xff95410d, 0xf1984a04, 0xabd32373, 0xa5de287a, 0xb7c93561, 0xb9c43e68, 0x93e70f57,
        0x9dea045e, 0x8ffd1945, 0x81f0124c, 0x3b6bcbab, 0x3566c0a2, 0x2771ddb9, 0x297cd6b0, 0x35fe78f,  0x0d52ec86,
        0x1f45f19d, 0x1148fa94, 0x4b0393e3, 0x450e98ea, 0x571985f1, 0x59148ef8, 0x7337bfc7, 0x7d3ab4ce, 0x6f2da9d5,
        0x6120a2dc, 0xad6df676, 0xa360fd7f, 0xb177e064, 0xbf7aeb6d, 0x9559da52, 0x9b54d15b, 0x8943cc40, 0x874ec749,
        0xdd05ae3e, 0xd308a537, 0xc11fb82c, 0xcf12b325, 0xe531821a, 0xeb3c8913, 0xf92b9408, 0xf7269f01, 0x4dbd46e6,
        0x43b04def, 0x51a750f4, 0x5faa5bfd, 0x75896ac2, 0x7b8461cb, 0x69937cd0, 0x679e77d9, 0x3dd51eae, 0x33d815a7,
        0x21cf08bc, 0x2fc203b5, 0x5e1328a,  0x0bec3983, 0x19fb2498, 0x17f62f91, 0x76d68d4d, 0x78db8644, 0x6acc9b5f,
        0x64c19056, 0x4ee2a169, 0x40efaa60, 0x52f8b77b, 0x5cf5bc72, 0x6bed505,  0x08b3de0c, 0x1aa4c317, 0x14a9c81e,
        0x3e8af921, 0x3087f228, 0x2290ef33, 0x2c9de43a, 0x96063ddd, 0x980b36d4, 0x8a1c2bcf, 0x841120c6, 0xae3211f9,
        0xa03f1af0, 0xb22807eb, 0xbc250ce2, 0xe66e6595, 0xe8636e9c, 0xfa747387, 0xf479788e, 0xde5a49b1, 0xd05742b8,
        0xc2405fa3, 0xcc4d54aa, 0x41daf7ec, 0x4fd7fce5, 0x5dc0e1fe, 0x53cdeaf7, 0x79eedbc8, 0x77e3d0c1, 0x65f4cdda,
        0x6bf9c6d3, 0x31b2afa4, 0x3fbfa4ad, 0x2da8b9b6, 0x23a5b2bf, 0x09868380, 0x78b8889,  0x159c9592, 0x1b919e9b,
        0xa10a477c, 0xaf074c75, 0xbd10516e, 0xb31d5a67, 0x993e6b58, 0x97336051, 0x85247d4a, 0x8b297643, 0xd1621f34,
        0xdf6f143d, 0xcd780926, 0xc375022f, 0xe9563310, 0xe75b3819, 0xf54c2502, 0xfb412e0b, 0x9a618cd7, 0x946c87de,
        0x867b9ac5, 0x887691cc, 0xa255a0f3, 0xac58abfa, 0xbe4fb6e1, 0xb042bde8, 0xea09d49f, 0xe404df96, 0xf613c28d,
        0xf81ec984, 0xd23df8bb, 0xdc30f3b2, 0xce27eea9, 0xc02ae5a0, 0x7ab13c47, 0x74bc374e, 0x66ab2a55, 0x68a6215c,
        0x42851063, 0x4c881b6a, 0x5e9f0671, 0x50920d78, 0x0ad9640f, 0x4d46f06,  0x16c3721d, 0x18ce7914, 0x32ed482b,
        0x3ce04322, 0x2ef75e39, 0x20fa5530, 0xecb7019a, 0xe2ba0a93, 0xf0ad1788, 0xfea01c81, 0xd4832dbe, 0xda8e26b7,
        0xc8993bac, 0xc69430a5, 0x9cdf59d2, 0x92d252db, 0x80c54fc0, 0x8ec844c9, 0xa4eb75f6, 0xaae67eff, 0xb8f163e4,
        0xb6fc68ed, 0x0c67b10a, 0x26aba03,  0x107da718, 0x1e70ac11, 0x34539d2e, 0x3a5e9627, 0x28498b3c, 0x26448035,
        0x7c0fe942, 0x7202e24b, 0x6015ff50, 0x6e18f459, 0x443bc566, 0x4a36ce6f, 0x5821d374, 0x562cd87d, 0x370c7aa1,
        0x390171a8, 0x2b166cb3, 0x251b67ba, 0x0f385685, 0x1355d8c,  0x13224097, 0x1d2f4b9e, 0x476422e9, 0x496929e0,
        0x5b7e34fb, 0x55733ff2, 0x7f500ecd, 0x715d05c4, 0x634a18df, 0x6d4713d6, 0xd7dcca31, 0xd9d1c138, 0xcbc6dc23,
        0xc5cbd72a, 0xefe8e615, 0xe1e5ed1c, 0xf3f2f007, 0xfdfffb0e, 0xa7b49279, 0xa9b99970, 0xbbae846b, 0xb5a38f62,
        0x9f80be5d, 0x918db554, 0x839aa84f, 0x8d97a346};

    const ap_uint<8> ssbox[256] = {
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x1,  0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76, 0xca, 0x82,
        0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26,
        0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15, 0x4,  0xc7, 0x23, 0xc3, 0x18, 0x96,
        0x5,  0x9a, 0x7,  0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
        0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x0,  0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb,
        0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, 0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x2,  0x7f,
        0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff,
        0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32,
        0x3a, 0x0a, 0x49, 0x6,  0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d,
        0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, 0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6,
        0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, 0x70, 0x3e, 0xb5, 0x66, 0x48, 0x3,  0xf6, 0x0e,
        0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e,
        0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf, 0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f,
        0xb0, 0x54, 0xbb, 0x16};

    const ap_uint<8> iibox[256] = {
        0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB, 0x7C, 0xE3,
        0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB, 0x54, 0x7B, 0x94, 0x32,
        0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E, 0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9,
        0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25, 0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16,
        0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92, 0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15,
        0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84, 0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05,
        0xB8, 0xB3, 0x45, 0x06, 0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13,
        0x8A, 0x6B, 0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
        0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E, 0x47, 0xF1,
        0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B, 0xFC, 0x56, 0x3E, 0x4B,
        0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4, 0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07,
        0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F, 0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D,
        0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF, 0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB,
        0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61, 0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63,
        0x55, 0x21, 0x0C, 0x7D};
};
/**
 * @brief AES encryption
 *
 * @tparam W Bit width of AES key, which is 128, 192 or 256
 */
template <int W>
class aesEnc {
   public:
    /**
     * @brief Update key before using it to encrypt
     *
     * @param cipherkey Key to be used in encryption.
     */
    void updateKey(ap_uint<W> cipherkey) {}
    /**
     * @brief Encrypt message using AES algorithm
     *
     * @param plaintext Message to be encrypted.
     * @param cipherkey Key to be used in encryption.
     * @param ciphertext Encryption result.
     */
    void process(ap_uint<128> plaintext, ap_uint<W> cipherkey, ap_uint<128>& ciphertext) {}
};

template <>
class aesEnc<256> : public aesTable {
   public:
    ap_uint<128> key_list[16];

    aesEnc() {
#pragma HLS inline
#pragma HLS ARRAY_PARTITION variable = key_list complete dim = 1
    }

    void updateKey(ap_uint<256> cipherkey) {
#pragma HLS inline off
        const ap_uint<8> Rcon[10] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36};
        ap_uint<256> lastRound = cipherkey;
        key_list[0] = lastRound.range(127, 0);
        key_list[1] = lastRound.range(255, 128);
        for (ap_uint<5> iter = 2; iter < 15; iter += 2) {
#pragma HLS pipeline II = 1
            //#pragma HLS pipeline II = 1
            ap_uint<128> currRound_0, currRound_1;

            ap_uint<32> round_tmp = lastRound.range(255, 224);
            round_tmp = (round_tmp >> 8) | (round_tmp << 24);

            round_tmp.range(7, 0) = ssbox[round_tmp.range(7, 0)] ^ Rcon[(iter >> 1) - 1];
            round_tmp.range(15, 8) = ssbox[round_tmp.range(15, 8)];
            round_tmp.range(23, 16) = ssbox[round_tmp.range(23, 16)];
            round_tmp.range(31, 24) = ssbox[round_tmp.range(31, 24)];

            currRound_0.range(31, 0) = lastRound.range(31, 0) ^ round_tmp;
            currRound_0.range(63, 32) = lastRound.range(63, 32) ^ currRound_0.range(31, 0);
            currRound_0.range(95, 64) = lastRound.range(95, 64) ^ currRound_0.range(63, 32);
            currRound_0.range(127, 96) = lastRound.range(127, 96) ^ currRound_0.range(95, 64);

            ap_uint<32> round_tmp2 = currRound_0.range(127, 96);

            round_tmp2.range(7, 0) = ssbox[round_tmp2.range(7, 0)];
            round_tmp2.range(15, 8) = ssbox[round_tmp2.range(15, 8)];
            round_tmp2.range(23, 16) = ssbox[round_tmp2.range(23, 16)];
            round_tmp2.range(31, 24) = ssbox[round_tmp2.range(31, 24)];

            currRound_1.range(31, 0) = lastRound.range(159, 128) ^ round_tmp2;
            currRound_1.range(63, 32) = lastRound.range(191, 160) ^ currRound_1.range(31, 0);
            currRound_1.range(95, 64) = lastRound.range(223, 192) ^ currRound_1.range(63, 32);
            currRound_1.range(127, 96) = lastRound.range(255, 224) ^ currRound_1.range(95, 64);
            lastRound.range(127, 0) = currRound_0;
            lastRound.range(255, 128) = currRound_1;
            key_list[iter] = currRound_0;
            key_list[iter + 1] = currRound_1;
        }
    }

    void process(ap_uint<128> plaintext, ap_uint<256> cipherkey, ap_uint<128>& ciphertext) {
        ap_uint<128> state, state_1, state_2, state_3;
        ap_uint<8> tmp_1, tmp_2_1, tmp_2_2, tmp_3;
        ap_uint<4> round_counter;

        // state init and add roundkey[0]
        state = plaintext ^ key_list[0];

        // Start 14 rounds of process
        for (round_counter = 1; round_counter <= 14; round_counter++) {
            // SubByte
            for (int i = 0; i < 16; i++) {
#pragma HLS unroll
                state(i * 8 + 7, i * 8) = ssbox[state(i * 8 + 7, i * 8)];
            }
            // ShiftRow
            tmp_1 = state(15, 8);
            state(15, 8) = state(47, 40);
            state(47, 40) = state(79, 72);
            state(79, 72) = state(111, 104);
            state(111, 104) = tmp_1;

            tmp_2_1 = state(23, 16);
            state(23, 16) = state(87, 80);
            state(87, 80) = tmp_2_1;

            tmp_2_2 = state(55, 48);
            state(55, 48) = state(119, 112);
            state(119, 112) = tmp_2_2;

            tmp_3 = state(127, 120);
            state(127, 120) = state(95, 88);
            state(95, 88) = state(63, 56);
            state(63, 56) = state(31, 24);
            state(31, 24) = tmp_3;

            // MixColumn
            for (int i = 0; i < 16; i++) {
#pragma HLS unroll
                ap_uint<8> tmp = state(i * 8 + 7, i * 8);
                ap_uint<16> tmp23 = p16box[tmp];
                state_1(i * 8 + 7, i * 8) = tmp;
                state_2(i * 8 + 7, i * 8) = tmp23.range(7, 0);
                state_3(i * 8 + 7, i * 8) = tmp23.range(15, 8);
            }

            if (round_counter < 14) {
                for (int i = 0; i < 4; i++) {
#pragma HLS unroll

                    state(i * 32 + 7, i * 32) = state_2(i * 32 + 7, i * 32) ^ state_3(i * 32 + 15, i * 32 + 8) ^
                                                state_1(i * 32 + 23, i * 32 + 16) ^ state_1(i * 32 + 31, i * 32 + 24);
                    state(i * 32 + 15, i * 32 + 8) = state_1(i * 32 + 7, i * 32) ^ state_2(i * 32 + 15, i * 32 + 8) ^
                                                     state_3(i * 32 + 23, i * 32 + 16) ^
                                                     state_1(i * 32 + 31, i * 32 + 24);
                    state(i * 32 + 23, i * 32 + 16) = state_1(i * 32 + 7, i * 32) ^ state_1(i * 32 + 15, i * 32 + 8) ^
                                                      state_2(i * 32 + 23, i * 32 + 16) ^
                                                      state_3(i * 32 + 31, i * 32 + 24);
                    state(i * 32 + 31, i * 32 + 24) = state_3(i * 32 + 7, i * 32) ^ state_1(i * 32 + 15, i * 32 + 8) ^
                                                      state_1(i * 32 + 23, i * 32 + 16) ^
                                                      state_2(i * 32 + 31, i * 32 + 24);
                }
            } else {
                state = state_1;
            }

            state ^= key_list[round_counter];
        }
        ciphertext = state;
    }

    void updateKey_2(ap_uint<256> cipherkey) {
#pragma HLS inline off
        const ap_uint<8> Rcon[10] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36};
        ap_uint<128> lastRound[2];
        key_list[0] = cipherkey.range(127, 0);
        key_list[1] = cipherkey.range(255, 128);
        lastRound[0] = cipherkey.range(127, 0);
        lastRound[1] = cipherkey.range(255, 128);
        for (ap_uint<5> iter = 2; iter < 15; iter++) {
#pragma HLS pipeline II = 1
            ap_uint<32> round_tmp = lastRound[ap_uint<1>(1) - iter[0]].range(127, 96);
            if (iter[0] == ap_uint<1>(0)) {
                round_tmp = (round_tmp >> 8) | (round_tmp << 24);
            }

            round_tmp.range(7, 0) = ssbox[round_tmp.range(7, 0)];
            round_tmp.range(15, 8) = ssbox[round_tmp.range(15, 8)];
            round_tmp.range(23, 16) = ssbox[round_tmp.range(23, 16)];
            round_tmp.range(31, 24) = ssbox[round_tmp.range(31, 24)];

            if (iter[0] == ap_uint<1>(0)) {
                round_tmp.range(7, 0) ^= Rcon[(iter >> 1) - 1];
            }

            ap_uint<128> tmp_key;
            tmp_key.range(31, 0) = lastRound[iter[0]].range(31, 0) ^ round_tmp;
            tmp_key.range(63, 32) = lastRound[iter[0]].range(63, 32) ^ tmp_key.range(31, 0);
            tmp_key.range(95, 64) = lastRound[iter[0]].range(95, 64) ^ tmp_key.range(63, 32);
            tmp_key.range(127, 96) = lastRound[iter[0]].range(127, 96) ^ tmp_key.range(95, 64);

            key_list[iter] = tmp_key;
            lastRound[iter[0]] = tmp_key;
        }
    }
};

template <>
class aesEnc<192> : public aesTable {
   public:
    ap_uint<128> key_list[14];

    aesEnc() {
#pragma HLS inline
#pragma HLS ARRAY_PARTITION variable = key_list complete dim = 1
    }

    void updateKey(ap_uint<192> cipherkey) {
#pragma HLS inline off
        const ap_uint<8> Rcon[10] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36};
        ap_uint<192> lastRound = cipherkey;
        key_list[0] = lastRound.range(127, 0);
        key_list[1].range(63, 0) = lastRound.range(191, 128);
        ap_uint<4> rIter = 0;

        for (ap_uint<5> iter = 3; iter < 26; iter += 3) {
#pragma HLS pipeline II = 1

            ap_uint<192> thisRound = 0;

            ap_uint<32> round_tmp = lastRound.range(191, 160);
            round_tmp = (round_tmp >> 8) | (round_tmp << 24);

            round_tmp.range(7, 0) = ssbox[round_tmp.range(7, 0)] ^ Rcon[rIter++];
            round_tmp.range(15, 8) = ssbox[round_tmp.range(15, 8)];
            round_tmp.range(23, 16) = ssbox[round_tmp.range(23, 16)];
            round_tmp.range(31, 24) = ssbox[round_tmp.range(31, 24)];

            thisRound.range(31, 0) = lastRound.range(31, 0) ^ round_tmp;
            thisRound.range(63, 32) = lastRound.range(63, 32) ^ thisRound.range(31, 0);
            thisRound.range(95, 64) = lastRound.range(95, 64) ^ thisRound.range(63, 32);
            thisRound.range(127, 96) = lastRound.range(127, 96) ^ thisRound.range(95, 64);
            thisRound.range(159, 128) = lastRound.range(159, 128) ^ thisRound.range(127, 96);
            thisRound.range(191, 160) = lastRound.range(191, 160) ^ thisRound.range(159, 128);

            if (iter[0] == 1) {
                key_list[iter.range(4, 1)].range(127, 64) = thisRound.range(63, 0);
                key_list[iter.range(4, 1) + 1].range(127, 0) = thisRound.range(191, 64);
            } else {
                key_list[iter.range(4, 1)].range(127, 0) = thisRound.range(127, 0);
                key_list[iter.range(4, 1) + 1].range(63, 0) = thisRound.range(191, 128);
            }
            lastRound = thisRound;
        }
    }

    void process(ap_uint<128> plaintext, ap_uint<192> cipherkey, ap_uint<128>& ciphertext) {
        ap_uint<128> state, state_1, state_2, state_3;
        ap_uint<8> tmp_1, tmp_2_1, tmp_2_2, tmp_3;
        ap_uint<4> round_counter;

        // state init and add roundkey[0]
        state = plaintext ^ key_list[0];

        // Start 14 rounds of process
        for (round_counter = 1; round_counter <= 12; round_counter++) {
            // SubByte
            for (int i = 0; i < 16; i++) {
#pragma HLS unroll
                state(i * 8 + 7, i * 8) = ssbox[state(i * 8 + 7, i * 8)];
            }
            // ShiftRow
            tmp_1 = state(15, 8);
            state(15, 8) = state(47, 40);
            state(47, 40) = state(79, 72);
            state(79, 72) = state(111, 104);
            state(111, 104) = tmp_1;

            tmp_2_1 = state(23, 16);
            state(23, 16) = state(87, 80);
            state(87, 80) = tmp_2_1;

            tmp_2_2 = state(55, 48);
            state(55, 48) = state(119, 112);
            state(119, 112) = tmp_2_2;

            tmp_3 = state(127, 120);
            state(127, 120) = state(95, 88);
            state(95, 88) = state(63, 56);
            state(63, 56) = state(31, 24);
            state(31, 24) = tmp_3;

            // MixColumn
            for (int i = 0; i < 16; i++) {
#pragma HLS unroll
                ap_uint<8> tmp = state(i * 8 + 7, i * 8);
                ap_uint<16> tmp23 = p16box[tmp];
                state_1(i * 8 + 7, i * 8) = tmp;
                state_2(i * 8 + 7, i * 8) = tmp23.range(7, 0);
                state_3(i * 8 + 7, i * 8) = tmp23.range(15, 8);
            }

            if (round_counter < 12) {
                for (int i = 0; i < 4; i++) {
#pragma HLS unroll

                    state(i * 32 + 7, i * 32) = state_2(i * 32 + 7, i * 32) ^ state_3(i * 32 + 15, i * 32 + 8) ^
                                                state_1(i * 32 + 23, i * 32 + 16) ^ state_1(i * 32 + 31, i * 32 + 24);
                    state(i * 32 + 15, i * 32 + 8) = state_1(i * 32 + 7, i * 32) ^ state_2(i * 32 + 15, i * 32 + 8) ^
                                                     state_3(i * 32 + 23, i * 32 + 16) ^
                                                     state_1(i * 32 + 31, i * 32 + 24);
                    state(i * 32 + 23, i * 32 + 16) = state_1(i * 32 + 7, i * 32) ^ state_1(i * 32 + 15, i * 32 + 8) ^
                                                      state_2(i * 32 + 23, i * 32 + 16) ^
                                                      state_3(i * 32 + 31, i * 32 + 24);
                    state(i * 32 + 31, i * 32 + 24) = state_3(i * 32 + 7, i * 32) ^ state_1(i * 32 + 15, i * 32 + 8) ^
                                                      state_1(i * 32 + 23, i * 32 + 16) ^
                                                      state_2(i * 32 + 31, i * 32 + 24);
                }
            } else {
                state = state_1;
            }

            state ^= key_list[round_counter];
        }
        ciphertext = state;
    }
};

template <>
class aesEnc<128> : public aesTable {
   public:
    ap_uint<128> key_list[11];

    aesEnc() {
#pragma HLS inline
#pragma HLS ARRAY_PARTITION variable = key_list complete dim = 1
    }

    void updateKey(ap_uint<128> cipherkey) {
#pragma HLS inline off
        const ap_uint<8> Rcon[10] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36};
        ap_uint<128> lastRound = cipherkey;
        key_list[0] = lastRound;
        for (ap_uint<5> iter = 1; iter < 11; iter++) {
#pragma HLS pipeline II = 1

            ap_uint<32> round_tmp = lastRound.range(127, 96);
            round_tmp = (round_tmp >> 8) | (round_tmp << 24);

            round_tmp.range(7, 0) = ssbox[round_tmp.range(7, 0)] ^ Rcon[iter - 1];
            round_tmp.range(15, 8) = ssbox[round_tmp.range(15, 8)];
            round_tmp.range(23, 16) = ssbox[round_tmp.range(23, 16)];
            round_tmp.range(31, 24) = ssbox[round_tmp.range(31, 24)];

            key_list[iter].range(31, 0) = lastRound.range(31, 0) ^ round_tmp;
            key_list[iter].range(63, 32) = lastRound.range(63, 32) ^ key_list[iter].range(31, 0);
            key_list[iter].range(95, 64) = lastRound.range(95, 64) ^ key_list[iter].range(63, 32);
            key_list[iter].range(127, 96) = lastRound.range(127, 96) ^ key_list[iter].range(95, 64);

            lastRound.range(127, 0) = key_list[iter];
        }
    }

    void process(ap_uint<128> plaintext, ap_uint<128> cipherkey, ap_uint<128>& ciphertext) {
        ap_uint<128> state, state_1, state_2, state_3;
        ap_uint<8> tmp_1, tmp_2_1, tmp_2_2, tmp_3;
        ap_uint<4> round_counter;

        // state init and add roundkey[0]
        state = plaintext ^ key_list[0];

        // Start 14 rounds of process
        for (round_counter = 1; round_counter <= 10; round_counter++) {
            // SubByte
            for (int i = 0; i < 16; i++) {
#pragma HLS unroll
                state(i * 8 + 7, i * 8) = ssbox[state(i * 8 + 7, i * 8)];
            }
            // ShiftRow
            tmp_1 = state(15, 8);
            state(15, 8) = state(47, 40);
            state(47, 40) = state(79, 72);
            state(79, 72) = state(111, 104);
            state(111, 104) = tmp_1;

            tmp_2_1 = state(23, 16);
            state(23, 16) = state(87, 80);
            state(87, 80) = tmp_2_1;

            tmp_2_2 = state(55, 48);
            state(55, 48) = state(119, 112);
            state(119, 112) = tmp_2_2;

            tmp_3 = state(127, 120);
            state(127, 120) = state(95, 88);
            state(95, 88) = state(63, 56);
            state(63, 56) = state(31, 24);
            state(31, 24) = tmp_3;

            // MixColumn
            for (int i = 0; i < 16; i++) {
#pragma HLS unroll
                ap_uint<8> tmp = state(i * 8 + 7, i * 8);
                ap_uint<16> tmp23 = p16box[tmp];
                state_1(i * 8 + 7, i * 8) = tmp;
                state_2(i * 8 + 7, i * 8) = tmp23.range(7, 0);
                state_3(i * 8 + 7, i * 8) = tmp23.range(15, 8);
            }

            if (round_counter < 10) {
                for (int i = 0; i < 4; i++) {
#pragma HLS unroll

                    state(i * 32 + 7, i * 32) = state_2(i * 32 + 7, i * 32) ^ state_3(i * 32 + 15, i * 32 + 8) ^
                                                state_1(i * 32 + 23, i * 32 + 16) ^ state_1(i * 32 + 31, i * 32 + 24);
                    state(i * 32 + 15, i * 32 + 8) = state_1(i * 32 + 7, i * 32) ^ state_2(i * 32 + 15, i * 32 + 8) ^
                                                     state_3(i * 32 + 23, i * 32 + 16) ^
                                                     state_1(i * 32 + 31, i * 32 + 24);
                    state(i * 32 + 23, i * 32 + 16) = state_1(i * 32 + 7, i * 32) ^ state_1(i * 32 + 15, i * 32 + 8) ^
                                                      state_2(i * 32 + 23, i * 32 + 16) ^
                                                      state_3(i * 32 + 31, i * 32 + 24);
                    state(i * 32 + 31, i * 32 + 24) = state_3(i * 32 + 7, i * 32) ^ state_1(i * 32 + 15, i * 32 + 8) ^
                                                      state_1(i * 32 + 23, i * 32 + 16) ^
                                                      state_2(i * 32 + 31, i * 32 + 24);
                }
            } else {
                state = state_1;
            }

            state ^= key_list[round_counter];
        }
        ciphertext = state;
    }
};

/**
 * @brief AES decryption
 *
 * @tparam W Bit width of AES key, which is 128, 192 or 256
 */
template <int W>
class aesDec {
   public:
    /**
     * @brief Update key before using it to decrypt.
     *
     * @param cipherkey Key to be used in decryption.
     */
    void updateKey(ap_uint<W> cipherkey) {}
    /**
     * @brief Decrypt message using AES algorithm
     *
     * @param ciphertext Cipher text to be decrypted.
     * @param cipherkey Key to be used in decryption.
     * @param plaintext Decryption result.
     */
    void process(ap_uint<128> ciphertext, ap_uint<W> cipherkey, ap_uint<128>& plaintext) {}
};

template <>
class aesDec<256> : public aesTable {
   public:
    ap_uint<128> key_list[16];

    aesDec() {
#pragma HLS inline
#pragma HLS ARRAY_PARTITION variable = key_list complete dim = 1
    }

    void updateKey(ap_uint<256> cipherkey) {
#pragma HLS inline off
        const ap_uint<8> Rcon[10] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36};
        ap_uint<256> lastRound = cipherkey;
        key_list[0] = lastRound.range(127, 0);
        key_list[1] = lastRound.range(255, 128);
        for (ap_uint<5> iter = 2; iter < 15; iter += 2) {
#pragma HLS pipeline II = 1

            ap_uint<32> round_tmp = lastRound.range(255, 224);
            round_tmp = (round_tmp >> 8) | (round_tmp << 24);

            round_tmp.range(7, 0) = ssbox[round_tmp.range(7, 0)] ^ Rcon[(iter >> 1) - 1];
            round_tmp.range(15, 8) = ssbox[round_tmp.range(15, 8)];
            round_tmp.range(23, 16) = ssbox[round_tmp.range(23, 16)];
            round_tmp.range(31, 24) = ssbox[round_tmp.range(31, 24)];

            key_list[iter].range(31, 0) = lastRound.range(31, 0) ^ round_tmp;
            key_list[iter].range(63, 32) = lastRound.range(63, 32) ^ key_list[iter].range(31, 0);
            key_list[iter].range(95, 64) = lastRound.range(95, 64) ^ key_list[iter].range(63, 32);
            key_list[iter].range(127, 96) = lastRound.range(127, 96) ^ key_list[iter].range(95, 64);

            ap_uint<32> round_tmp2 = key_list[iter].range(127, 96);

            round_tmp2.range(7, 0) = ssbox[round_tmp2.range(7, 0)];
            round_tmp2.range(15, 8) = ssbox[round_tmp2.range(15, 8)];
            round_tmp2.range(23, 16) = ssbox[round_tmp2.range(23, 16)];
            round_tmp2.range(31, 24) = ssbox[round_tmp2.range(31, 24)];

            key_list[iter + 1].range(31, 0) = lastRound.range(159, 128) ^ round_tmp2;
            key_list[iter + 1].range(63, 32) = lastRound.range(191, 160) ^ key_list[iter + 1].range(31, 0);
            key_list[iter + 1].range(95, 64) = lastRound.range(223, 192) ^ key_list[iter + 1].range(63, 32);
            key_list[iter + 1].range(127, 96) = lastRound.range(255, 224) ^ key_list[iter + 1].range(95, 64);
            lastRound.range(127, 0) = key_list[iter];
            lastRound.range(255, 128) = key_list[iter + 1];
        }
    }

    void process(ap_uint<128> ciphertext, ap_uint<256> cipherkey, ap_uint<128>& plaintext) {
        ap_uint<128> state;
        ap_uint<4> round_counter;

        // state init and add roundkey[0]
        state = ciphertext ^ key_list[14];

        // Start 14 rounds of process
        for (round_counter = 1; round_counter <= 14; round_counter++) {
            ap_uint<8> tmp_1, tmp_2_1, tmp_2_2, tmp_3;
            ap_uint<128> state_9, state_b, state_d, state_e;
            // Inv ShiftRow
            tmp_1 = state(111, 104);
            state(111, 104) = state(79, 72);
            state(79, 72) = state(47, 40);
            state(47, 40) = state(15, 8);
            state(15, 8) = tmp_1;

            tmp_2_1 = state(87, 80);
            state(87, 80) = state(23, 16);
            state(23, 16) = tmp_2_1;

            tmp_2_2 = state(119, 112);
            state(119, 112) = state(55, 48);
            state(55, 48) = tmp_2_2;

            tmp_3 = state(31, 24);
            state(31, 24) = state(63, 56);
            state(63, 56) = state(95, 88);
            state(95, 88) = state(127, 120);
            state(127, 120) = tmp_3;
            // Inv SubByte
            for (int i = 0; i < 16; i++) {
#pragma HLS unroll
                state(i * 8 + 7, i * 8) = iibox[state(i * 8 + 7, i * 8)];
            }

            // Add Round Key
            state ^= key_list[14 - round_counter];

            // Inverse Mix Column
            for (int i = 0; i < 16; i++) {
                ap_uint<32> tmp_9bde = i32box[state.range(i * 8 + 7, i * 8)];
                state_9(i * 8 + 7, i * 8) = tmp_9bde.range(7, 0);
                state_b(i * 8 + 7, i * 8) = tmp_9bde.range(15, 8);
                state_d(i * 8 + 7, i * 8) = tmp_9bde.range(23, 16);
                state_e(i * 8 + 7, i * 8) = tmp_9bde.range(31, 24);
            }
            if (round_counter < 14) {
                for (int i = 0; i < 4; i++) {
#pragma HLS unroll
                    state(i * 32 + 7, i * 32) = state_e(i * 32 + 7, i * 32) ^ state_b(i * 32 + 15, i * 32 + 8) ^
                                                state_d(i * 32 + 23, i * 32 + 16) ^ state_9(i * 32 + 31, i * 32 + 24);
                    state(i * 32 + 15, i * 32 + 8) = state_9(i * 32 + 7, i * 32) ^ state_e(i * 32 + 15, i * 32 + 8) ^
                                                     state_b(i * 32 + 23, i * 32 + 16) ^
                                                     state_d(i * 32 + 31, i * 32 + 24);
                    state(i * 32 + 23, i * 32 + 16) = state_d(i * 32 + 7, i * 32) ^ state_9(i * 32 + 15, i * 32 + 8) ^
                                                      state_e(i * 32 + 23, i * 32 + 16) ^
                                                      state_b(i * 32 + 31, i * 32 + 24);
                    state(i * 32 + 31, i * 32 + 24) = state_b(i * 32 + 7, i * 32) ^ state_d(i * 32 + 15, i * 32 + 8) ^
                                                      state_9(i * 32 + 23, i * 32 + 16) ^
                                                      state_e(i * 32 + 31, i * 32 + 24);
                }
            }
        }
        plaintext = state;
    }
};

template <>
class aesDec<192> : public aesTable {
   public:
    ap_uint<128> key_list[14];

    aesDec() {
#pragma HLS inline
#pragma HLS ARRAY_PARTITION variable = key_list complete dim = 1
    }

    void updateKey(ap_uint<192> cipherkey) {
#pragma HLS inline off
        const ap_uint<8> Rcon[10] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36};
        ap_uint<192> lastRound = cipherkey;
        key_list[0] = lastRound.range(127, 0);
        key_list[1].range(63, 0) = lastRound.range(191, 128);
        ap_uint<4> rIter = 0;

        for (ap_uint<5> iter = 3; iter < 26; iter += 3) {
#pragma HLS pipeline II = 1

            ap_uint<192> thisRound = 0;

            ap_uint<32> round_tmp = lastRound.range(191, 160);
            round_tmp = (round_tmp >> 8) | (round_tmp << 24);

            round_tmp.range(7, 0) = ssbox[round_tmp.range(7, 0)] ^ Rcon[rIter++];
            round_tmp.range(15, 8) = ssbox[round_tmp.range(15, 8)];
            round_tmp.range(23, 16) = ssbox[round_tmp.range(23, 16)];
            round_tmp.range(31, 24) = ssbox[round_tmp.range(31, 24)];

            thisRound.range(31, 0) = lastRound.range(31, 0) ^ round_tmp;
            thisRound.range(63, 32) = lastRound.range(63, 32) ^ thisRound.range(31, 0);
            thisRound.range(95, 64) = lastRound.range(95, 64) ^ thisRound.range(63, 32);
            thisRound.range(127, 96) = lastRound.range(127, 96) ^ thisRound.range(95, 64);
            thisRound.range(159, 128) = lastRound.range(159, 128) ^ thisRound.range(127, 96);
            thisRound.range(191, 160) = lastRound.range(191, 160) ^ thisRound.range(159, 128);

            if (iter[0] == 1) {
                key_list[iter.range(4, 1)].range(127, 64) = thisRound.range(63, 0);
                key_list[iter.range(4, 1) + 1].range(127, 0) = thisRound.range(191, 64);
            } else {
                key_list[iter.range(4, 1)].range(127, 0) = thisRound.range(127, 0);
                key_list[iter.range(4, 1) + 1].range(63, 0) = thisRound.range(191, 128);
            }
            lastRound = thisRound;
        }
    }

    void process(ap_uint<128> ciphertext, ap_uint<192> cipherkey, ap_uint<128>& plaintext) {
        ap_uint<128> state;
        ap_uint<4> round_counter;

        // state init and add roundkey[0]
        state = ciphertext ^ key_list[12];

        // Start 14 rounds of process
        for (round_counter = 1; round_counter <= 12; round_counter++) {
            ap_uint<8> tmp_1, tmp_2_1, tmp_2_2, tmp_3;
            ap_uint<128> state_9, state_b, state_d, state_e;
            // Inv ShiftRow
            tmp_1 = state(111, 104);
            state(111, 104) = state(79, 72);
            state(79, 72) = state(47, 40);
            state(47, 40) = state(15, 8);
            state(15, 8) = tmp_1;

            tmp_2_1 = state(87, 80);
            state(87, 80) = state(23, 16);
            state(23, 16) = tmp_2_1;

            tmp_2_2 = state(119, 112);
            state(119, 112) = state(55, 48);
            state(55, 48) = tmp_2_2;

            tmp_3 = state(31, 24);
            state(31, 24) = state(63, 56);
            state(63, 56) = state(95, 88);
            state(95, 88) = state(127, 120);
            state(127, 120) = tmp_3;
            // Inv SubByte
            for (int i = 0; i < 16; i++) {
#pragma HLS unroll
                state(i * 8 + 7, i * 8) = iibox[state(i * 8 + 7, i * 8)];
            }

            // Add Round Key
            state ^= key_list[12 - round_counter];

            // Inverse Mix Column
            for (int i = 0; i < 16; i++) {
                ap_uint<32> tmp_9bde = i32box[state.range(i * 8 + 7, i * 8)];
                state_9(i * 8 + 7, i * 8) = tmp_9bde.range(7, 0);
                state_b(i * 8 + 7, i * 8) = tmp_9bde.range(15, 8);
                state_d(i * 8 + 7, i * 8) = tmp_9bde.range(23, 16);
                state_e(i * 8 + 7, i * 8) = tmp_9bde.range(31, 24);
            }
            if (round_counter < 12) {
                for (int i = 0; i < 4; i++) {
#pragma HLS unroll
                    state(i * 32 + 7, i * 32) = state_e(i * 32 + 7, i * 32) ^ state_b(i * 32 + 15, i * 32 + 8) ^
                                                state_d(i * 32 + 23, i * 32 + 16) ^ state_9(i * 32 + 31, i * 32 + 24);
                    state(i * 32 + 15, i * 32 + 8) = state_9(i * 32 + 7, i * 32) ^ state_e(i * 32 + 15, i * 32 + 8) ^
                                                     state_b(i * 32 + 23, i * 32 + 16) ^
                                                     state_d(i * 32 + 31, i * 32 + 24);
                    state(i * 32 + 23, i * 32 + 16) = state_d(i * 32 + 7, i * 32) ^ state_9(i * 32 + 15, i * 32 + 8) ^
                                                      state_e(i * 32 + 23, i * 32 + 16) ^
                                                      state_b(i * 32 + 31, i * 32 + 24);
                    state(i * 32 + 31, i * 32 + 24) = state_b(i * 32 + 7, i * 32) ^ state_d(i * 32 + 15, i * 32 + 8) ^
                                                      state_9(i * 32 + 23, i * 32 + 16) ^
                                                      state_e(i * 32 + 31, i * 32 + 24);
                }
            }
        }
        plaintext = state;
    }
};

template <>
class aesDec<128> : public aesTable {
   public:
    ap_uint<128> key_list[11];

    aesDec() {
#pragma HLS inline
#pragma HLS ARRAY_PARTITION variable = key_list complete dim = 1
    }

    void updateKey(ap_uint<128> cipherkey) {
#pragma HLS inline off
        const ap_uint<8> Rcon[10] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36};
        ap_uint<128> lastRound = cipherkey;
        key_list[0] = lastRound;
        for (ap_uint<5> iter = 1; iter < 11; iter++) {
#pragma HLS pipeline II = 1

            ap_uint<32> round_tmp = lastRound.range(127, 96);
            round_tmp = (round_tmp >> 8) | (round_tmp << 24);

            round_tmp.range(7, 0) = ssbox[round_tmp.range(7, 0)] ^ Rcon[iter - 1];
            round_tmp.range(15, 8) = ssbox[round_tmp.range(15, 8)];
            round_tmp.range(23, 16) = ssbox[round_tmp.range(23, 16)];
            round_tmp.range(31, 24) = ssbox[round_tmp.range(31, 24)];

            key_list[iter].range(31, 0) = lastRound.range(31, 0) ^ round_tmp;
            key_list[iter].range(63, 32) = lastRound.range(63, 32) ^ key_list[iter].range(31, 0);
            key_list[iter].range(95, 64) = lastRound.range(95, 64) ^ key_list[iter].range(63, 32);
            key_list[iter].range(127, 96) = lastRound.range(127, 96) ^ key_list[iter].range(95, 64);

            lastRound.range(127, 0) = key_list[iter];
        }
    }

    void process(ap_uint<128> ciphertext, ap_uint<128> cipherkey, ap_uint<128>& plaintext) {
        ap_uint<128> state;
        ap_uint<4> round_counter;

        // state init and add roundkey[0]
        state = ciphertext ^ key_list[10];

        // Start 14 rounds of process
        for (round_counter = 1; round_counter <= 10; round_counter++) {
            ap_uint<8> tmp_1, tmp_2_1, tmp_2_2, tmp_3;
            ap_uint<128> state_9, state_b, state_d, state_e;
            // Inv ShiftRow
            tmp_1 = state(111, 104);
            state(111, 104) = state(79, 72);
            state(79, 72) = state(47, 40);
            state(47, 40) = state(15, 8);
            state(15, 8) = tmp_1;

            tmp_2_1 = state(87, 80);
            state(87, 80) = state(23, 16);
            state(23, 16) = tmp_2_1;

            tmp_2_2 = state(119, 112);
            state(119, 112) = state(55, 48);
            state(55, 48) = tmp_2_2;

            tmp_3 = state(31, 24);
            state(31, 24) = state(63, 56);
            state(63, 56) = state(95, 88);
            state(95, 88) = state(127, 120);
            state(127, 120) = tmp_3;
            // Inv SubByte
            for (int i = 0; i < 16; i++) {
#pragma HLS unroll
                state(i * 8 + 7, i * 8) = iibox[state(i * 8 + 7, i * 8)];
            }

            // Add Round Key
            state ^= key_list[10 - round_counter];

            // Inverse Mix Column
            for (int i = 0; i < 16; i++) {
                ap_uint<32> tmp_9bde = i32box[state.range(i * 8 + 7, i * 8)];
                state_9(i * 8 + 7, i * 8) = tmp_9bde.range(7, 0);
                state_b(i * 8 + 7, i * 8) = tmp_9bde.range(15, 8);
                state_d(i * 8 + 7, i * 8) = tmp_9bde.range(23, 16);
                state_e(i * 8 + 7, i * 8) = tmp_9bde.range(31, 24);
            }
            if (round_counter < 10) {
                for (int i = 0; i < 4; i++) {
#pragma HLS unroll
                    state(i * 32 + 7, i * 32) = state_e(i * 32 + 7, i * 32) ^ state_b(i * 32 + 15, i * 32 + 8) ^
                                                state_d(i * 32 + 23, i * 32 + 16) ^ state_9(i * 32 + 31, i * 32 + 24);
                    state(i * 32 + 15, i * 32 + 8) = state_9(i * 32 + 7, i * 32) ^ state_e(i * 32 + 15, i * 32 + 8) ^
                                                     state_b(i * 32 + 23, i * 32 + 16) ^
                                                     state_d(i * 32 + 31, i * 32 + 24);
                    state(i * 32 + 23, i * 32 + 16) = state_d(i * 32 + 7, i * 32) ^ state_9(i * 32 + 15, i * 32 + 8) ^
                                                      state_e(i * 32 + 23, i * 32 + 16) ^
                                                      state_b(i * 32 + 31, i * 32 + 24);
                    state(i * 32 + 31, i * 32 + 24) = state_b(i * 32 + 7, i * 32) ^ state_d(i * 32 + 15, i * 32 + 8) ^
                                                      state_9(i * 32 + 23, i * 32 + 16) ^
                                                      state_e(i * 32 + 31, i * 32 + 24);
                }
            }
        }
        plaintext = state;
    }
};

} // namespace security
} // namespace xf
#endif
