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
 * @file dct.hpp
 */

#ifndef _XF_CODEC_DCT_HPP_
#define _XF_CODEC_DCT_HPP_

#ifndef __cplusplus
#error "Vitis Codec Library only works with C++."
#endif

#include <ap_int.h>
#include <hls_stream.h>

const float kDCTScales2[2] = {0.707106781186547524f, 0.707106781186547524f};

const float kIDCTScales2[2] = {0.707106781186547524f, 0.707106781186547524f};

const float kDCTScales4[4] = {0.5f, 0.653281482438188264f, 0.5f, 0.270598050073098492f};

const float kIDCTScales4[4] = {0.5f, 0.382683432365089772f, 0.5f, 0.923879532511286756f};

const float kDCTScales8[8] = {0.353553390593273762f, 0.254897789552079584f, 0.270598050073098492f,
                              0.30067244346752264f,  0.353553390593273762f, 0.449988111568207852f,
                              0.653281482438188264f, 1.28145772387075309f};

const float kIDCTScales8[8] = {0.353553390593273762f, 0.490392640201615225f, 0.461939766255643378f,
                               0.415734806151272619f, 0.353553390593273762f, 0.277785116509801112f,
                               0.191341716182544886f, 0.0975451610080641339f};

const float kIDCTScales16[16] = {0.25f,
                                 0.177632042131274808f,
                                 0.180239955501736978f,
                                 0.184731156892216368f,
                                 0.191341716182544886f,
                                 0.200444985785954314f,
                                 0.212607523691814112f,
                                 0.228686034616512494f,
                                 0.25f,
                                 0.278654739432954475f,
                                 0.318189645143208485f,
                                 0.375006192208515097f,
                                 0.461939766255643378f,
                                 0.608977011699708658f,
                                 0.906127446352887843f,
                                 1.80352839005774887f};

const float kDCTScales16[16] = {0.25f,
                                0.351850934381595615f,
                                0.346759961330536865f,
                                0.33832950029358817f,
                                0.326640741219094132f,
                                0.311806253246667808f,
                                0.293968900604839679f,
                                0.273300466750439372f,
                                0.25f,
                                0.224291896585659071f,
                                0.196423739596775545f,
                                0.166663914619436624f,
                                0.135299025036549246f,
                                0.102631131880589345f,
                                0.0689748448207357531f,
                                0.0346542922997728657f};

const float kIDCTScales32[32] = {
    0.176776695296636881f, 0.125150749558799075f, 0.125604821547038926f, 0.126367739974385915f, 0.127448894776039792f,
    0.128861827480656137f, 0.13062465373492222f,  0.132760647772446044f, 0.135299025036549246f, 0.138275974008611132f,
    0.141736008704089426f, 0.145733742051533468f, 0.15033622173376132f,  0.155626030758916204f, 0.161705445839997532f,
    0.168702085363751436f, 0.176776695296636881f, 0.186134067750574612f, 0.197038655862812556f, 0.20983741135388176f,
    0.224994055784103926f, 0.243142059465490173f, 0.265169421497586868f, 0.292359983358221239f, 0.326640741219094132f,
    0.371041154078541569f, 0.430611774559583482f, 0.514445252488352888f, 0.640728861935376545f, 0.851902104617179697f,
    1.27528715467229096f,  2.5475020308870142f};

const float kDCTScales32[32] = {
    0.176776695296636881f,  0.249698864051293098f,  0.248796181668049222f,  0.247294127491195243f,
    0.245196320100807612f,  0.242507813298635998f,  0.239235083933052216f,  0.235386016295755195f,
    0.230969883127821689f,  0.225997323280860833f,  0.220480316087088757f,  0.214432152500068017f,
    0.207867403075636309f,  0.200801882870161227f,  0.19325261334068424f,   0.185237781338739773f,
    0.176776695296636881f,  0.1678897387117546f,    0.158598321040911375f,  0.148924826123108336f,
    0.138892558254900556f,  0.128525686048305432f,  0.117849184206499412f,  0.106888773357570524f,
    0.0956708580912724429f, 0.0842224633480550127f, 0.0725711693136155919f, 0.0607450449758159725f,
    0.048772580504032067f,  0.0366826186138404379f, 0.0245042850823901505f, 0.0122669185818545036f};

const float kL1Norm2[2] = {
    1.0000000000000000000f, 1.0000000000000000000f,
};

const float kL1Norm4[4] = {
    1.0000000000000000000f, //
    0.9238795325112867561f, // cos(pi/8)
    1.0000000000000000000f, //
    0.9238795325112867561f, // cos(pi/8)
};

const float kL1Norm8[8] = {
    1.0000000000000000000f, //
    0.9061274463528878431f, // cos(pi/8) * cos(pi/16)
    0.9238795325112867561f, // cos(pi/8)
    0.9061274463528878431f, // cos(pi/8) * cos(pi/16)
    1.0000000000000000000f, //
    0.9061274463528878431f, // cos(pi/8) * cos(pi/16)
    0.9238795325112867561f, // cos(pi/8)
    0.9061274463528878431f, // cos(pi/8) * cos(pi/16)

};

const float kL1Norm16[16] = {
    1.0000000000000000000f, //
    0.9017641950288744354f, // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9061274463528878431f, // cos(pi/8) * cos(pi/16)
    0.9017641950288744354f, // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9238795325112867561f, // cos(pi/8)
    0.9017641950288744354f, // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9061274463528878431f, // cos(pi/8) * cos(pi/16)
    0.9017641950288744354f, // cos(pi/8) * cos(pi/16) * cos(pi/32)
    1.0000000000000000000f, //
    0.9017641950288744354f, // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9061274463528878431f, // cos(pi/8) * cos(pi/16)
    0.9017641950288744354f, // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9238795325112867561f, // cos(pi/8)
    0.9017641950288744354f, // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9061274463528878431f, // cos(pi/8) * cos(pi/16)
    0.9017641950288744354f, // cos(pi/8) * cos(pi/16) * cos(pi/32)
};

const float kL1Norm32[32] = {
    1.0000000000000000000f, //
    0.9006779805633546924f, // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9017641950288744354f, // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9006779805633546924f, // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9061274463528878431f, // cos(pi/8) * cos(pi/16)
    0.9006779805633546924f, // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9017641950288744354f, // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9006779805633546924f, // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9238795325112867561f, // cos(pi/8)
    0.9006779805633546924f, // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9017641950288744354f, // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9006779805633546924f, // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9061274463528878431f, // cos(pi/8) * cos(pi/16)
    0.9006779805633546924f, // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9017641950288744354f, // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9006779805633546924f, // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    1.0000000000000000000f, //
    0.9006779805633546924f, // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9017641950288744354f, // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9006779805633546924f, // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9061274463528878431f, // cos(pi/8) * cos(pi/16)
    0.9006779805633546924f, // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9017641950288744354f, // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9006779805633546924f, // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9238795325112867561f, // cos(pi/8)
    0.9006779805633546924f, // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9017641950288744354f, // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9006779805633546924f, // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9061274463528878431f, // cos(pi/8) * cos(pi/16)
    0.9006779805633546924f, // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9017641950288744354f, // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9006779805633546924f, // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
};

const float kL1NormInv2[2] = {
    1.000000000000000000f, 1.000000000000000000f,
};

const float kL1NormInv4[4] = {
    1.000000000000000000f, 1.082392200292393968f, 1.000000000000000000f, 1.082392200292393968f,
};

const float kL1NormInv8[8] = {
    1.000000000000000000f, 1.103597517131772049f, 1.082392200292393968f, 1.103597517131772049f,
    1.000000000000000000f, 1.103597517131772049f, 1.082392200292393968f, 1.103597517131772049f,
};

const float kL1NormInv16[16] = {
    1.000000000000000000f, 1.108937353592731700f, 1.103597517131772049f, 1.108937353592731700f,
    1.082392200292393968f, 1.108937353592731700f, 1.103597517131772049f, 1.108937353592731700f,
    1.000000000000000000f, 1.108937353592731700f, 1.103597517131772049f, 1.108937353592731700f,
    1.082392200292393968f, 1.108937353592731700f, 1.103597517131772049f, 1.108937353592731700f,
};

const float kL1NormInv32[32] = {
    1.000000000000000000, 1.110274728127050414, 1.108937353592731379, 1.110274728127050414, 1.103597517131772010,
    1.110274728127050636, 1.108937353592731379, 1.110274728127050414, 1.082392200292393580, 1.110274728127050414,
    1.108937353592730934, 1.110274728127050414, 1.103597517131771788, 1.110274728127050414, 1.108937353592731156,
    1.110274728127050414, 0.999999999999999556, 1.110274728127049970, 1.108937353592731601, 1.110274728127051080,
    1.103597517131771788, 1.110274728127050414, 1.108937353592732045, 1.110274728127050192, 1.082392200292394691,
    1.110274728127049526, 1.108937353592733155, 1.110274728127050858, 1.103597517131772232, 1.110274728127051969,
    1.108937353592732933, 1.110274728127050414,
};

template <int N>
float DCTScales(int x) {
#pragma HLS inline

    return N == 2
               ? kDCTScales2[x]
               : (N == 4 ? kDCTScales4[x] : (N == 8 ? kDCTScales8[x] : (N == 16 ? kDCTScales16[x] : kDCTScales32[x])));
}

template <int N>
float IDCTScales(int x) {
#pragma HLS inline

    return N == 2 ? kIDCTScales2[x]
                  : (N == 4 ? kIDCTScales4[x]
                            : (N == 8 ? kIDCTScales8[x] : (N == 16 ? kIDCTScales16[x] : kIDCTScales32[x])));
}

template <int N>
float L1Norm(int x) {
#pragma HLS inline

    return N == 2 ? kL1Norm2[x]
                  : (N == 4 ? kL1Norm4[x] : (N == 8 ? kL1Norm8[x] : (N == 16 ? kL1Norm16[x] : kL1Norm32[x])));
}

template <int N>
float L1NormInv(int x) {
#pragma HLS inline

    return N == 2
               ? kL1NormInv2[x]
               : (N == 4 ? kL1NormInv4[x] : (N == 8 ? kL1NormInv8[x] : (N == 16 ? kL1NormInv16[x] : kL1NormInv32[x])));
}

template <int N>
float DCTTotalScale(int x, int y) {
    return N * DCTScales<N>(x) * DCTScales<N>(y) * L1NormInv<N>(x) * L1NormInv<N>(y);
}

template <int N>
float DCTInvTotalScale(int x, int y) {
    return N * IDCTScales<N>(x) * IDCTScales<N>(y) * L1Norm<N>(x) * L1Norm<N>(y);
}

template <bool scale>
void dct4_block(float in[1024], float out[1024]) {
#pragma HLS INLINE off

    const float c2_8 = 1.414213562373095048f; // 2 * cos(2 * pi / 8)

    for (ap_uint<8> by = 0; by < 8; by++) {
        for (ap_uint<8> bx = 0; bx < 8; bx++) {
            for (ap_uint<8> x = 0; x < 4; x++) {
#pragma HLS DEPENDENCE variable = in inter false
#pragma HLS DEPENDENCE variable = out inter false
#pragma HLS LOOP_FLATTEN off
#pragma HLS pipeline

                float i0 = in[(by(2, 0), (ap_uint<2>)0, bx(2, 0), x(1, 0))];
                float i1 = in[(by(2, 0), (ap_uint<2>)1, bx(2, 0), x(1, 0))];
                float i2 = in[(by(2, 0), (ap_uint<2>)2, bx(2, 0), x(1, 0))];
                float i3 = in[(by(2, 0), (ap_uint<2>)3, bx(2, 0), x(1, 0))];

                float t0 = i0 + i3;
                float t1 = i1 + i2;
                float t2 = i0 - i3;
                float t3 = i1 - i2;

                float t4 = t0 + t1;
                float t5 = t0 - t1;
                float t6 = t2 - t3;
                float t7 = t3 * c2_8;
                float t8 = t6 + t7;
                float t9 = t6 - t7;

                if (scale) {
                    out[(by(2, 0), (ap_uint<2>)0, bx(2, 0), x(1, 0))] = t4 / 16;
                    out[(by(2, 0), (ap_uint<2>)1, bx(2, 0), x(1, 0))] = t8 / 16;
                    out[(by(2, 0), (ap_uint<2>)2, bx(2, 0), x(1, 0))] = t5 / 16;
                    out[(by(2, 0), (ap_uint<2>)3, bx(2, 0), x(1, 0))] = t9 / 16;
                } else {
                    out[(by(2, 0), (ap_uint<2>)0, bx(2, 0), x(1, 0))] = t4;
                    out[(by(2, 0), (ap_uint<2>)1, bx(2, 0), x(1, 0))] = t8;
                    out[(by(2, 0), (ap_uint<2>)2, bx(2, 0), x(1, 0))] = t5;
                    out[(by(2, 0), (ap_uint<2>)3, bx(2, 0), x(1, 0))] = t9;
                }

#ifdef DEBUF_DCT
                std::cout << "dc4_block: by=" << by << " bx=" << bx << " i0=" << i0 << " i1=" << i1 << " i2=" << i2
                          << " i3=" << i3 << std::endl;
                std::cout << "dc4_block: by=" << by << " bx=" << bx << " o0=" << t4 << " o1=" << t8 << " o2=" << t5
                          << " o3=" << t9 << std::endl;
#endif
            }
        }
    }
}

template <bool scale>
void dct8_block(float in[1024], float out[1024]) {
#pragma HLS INLINE off

    float c1 = 0.707106781186548f; // 1 / sqrt(2)
    float c2 = 0.382683432365090f; // cos(3 * pi / 8)
    float c3 = 1.30656296487638f;  // 1 / (2 * cos(3 * pi / 8))
    float c4 = 0.541196100146197f; // sqrt(2) * cos(3 * pi / 8)

    for (ap_uint<8> by = 0; by < 4; by++) {
        for (ap_uint<8> bx = 0; bx < 4; bx++) {
            for (ap_uint<8> x = 0; x < 8; x++) {
#pragma HLS DEPENDENCE variable = in inter false
#pragma HLS DEPENDENCE variable = out inter false
#pragma HLS LOOP_FLATTEN off
#pragma HLS pipeline

                float t00 =
                    in[(by(1, 0), (ap_uint<3>)0, bx(1, 0), x(2, 0))] + in[(by(1, 0), (ap_uint<3>)7, bx(1, 0), x(2, 0))];
                float t01 =
                    in[(by(1, 0), (ap_uint<3>)0, bx(1, 0), x(2, 0))] - in[(by(1, 0), (ap_uint<3>)7, bx(1, 0), x(2, 0))];
                float t02 =
                    in[(by(1, 0), (ap_uint<3>)3, bx(1, 0), x(2, 0))] + in[(by(1, 0), (ap_uint<3>)4, bx(1, 0), x(2, 0))];
                float t03 =
                    in[(by(1, 0), (ap_uint<3>)3, bx(1, 0), x(2, 0))] - in[(by(1, 0), (ap_uint<3>)4, bx(1, 0), x(2, 0))];
                float t04 =
                    in[(by(1, 0), (ap_uint<3>)2, bx(1, 0), x(2, 0))] + in[(by(1, 0), (ap_uint<3>)5, bx(1, 0), x(2, 0))];
                float t05 =
                    in[(by(1, 0), (ap_uint<3>)2, bx(1, 0), x(2, 0))] - in[(by(1, 0), (ap_uint<3>)5, bx(1, 0), x(2, 0))];
                float t06 =
                    in[(by(1, 0), (ap_uint<3>)1, bx(1, 0), x(2, 0))] + in[(by(1, 0), (ap_uint<3>)6, bx(1, 0), x(2, 0))];
                float t07 =
                    in[(by(1, 0), (ap_uint<3>)1, bx(1, 0), x(2, 0))] - in[(by(1, 0), (ap_uint<3>)6, bx(1, 0), x(2, 0))];

                float t08 = t00 + t02;
                float t09 = t00 - t02;
                float t10 = t06 + t04;
                float t11 = t06 - t04;
                float t12 = t07 + t05;
                float t13 = t01 + t07;
                float t14 = t05 + t03;

                float t15 = t11 + t09;
                float t16 = t14 - t13;

                float t17 = c1 * t15;
                float t18 = c1 * t12;
                float t19 = c2 * t16;
                float t20 = c3 * t13;
                float t21 = c4 * t14;

                float t22 = t20 + t19;
                float t23 = t21 + t19;
                float t24 = t01 + t18;
                float t25 = t01 - t18;

                float t26 = t08 + t10;
                float t27 = t24 + t22;
                float t28 = t09 + t17;
                float t29 = t25 - t23;
                float t30 = t08 - t10;
                float t31 = t25 + t23;
                float t32 = t09 - t17;
                float t33 = t24 - t22;

                if (scale) {
                    out[(by(1, 0), (ap_uint<3>)0, bx(1, 0), x(2, 0))] = t26 / 64;
                    out[(by(1, 0), (ap_uint<3>)1, bx(1, 0), x(2, 0))] = t27 / 64;
                    out[(by(1, 0), (ap_uint<3>)2, bx(1, 0), x(2, 0))] = t28 / 64;
                    out[(by(1, 0), (ap_uint<3>)3, bx(1, 0), x(2, 0))] = t29 / 64;
                    out[(by(1, 0), (ap_uint<3>)4, bx(1, 0), x(2, 0))] = t30 / 64;
                    out[(by(1, 0), (ap_uint<3>)5, bx(1, 0), x(2, 0))] = t31 / 64;
                    out[(by(1, 0), (ap_uint<3>)6, bx(1, 0), x(2, 0))] = t32 / 64;
                    out[(by(1, 0), (ap_uint<3>)7, bx(1, 0), x(2, 0))] = t33 / 64;
                } else {
                    out[(by(1, 0), (ap_uint<3>)0, bx(1, 0), x(2, 0))] = t26;
                    out[(by(1, 0), (ap_uint<3>)1, bx(1, 0), x(2, 0))] = t27;
                    out[(by(1, 0), (ap_uint<3>)2, bx(1, 0), x(2, 0))] = t28;
                    out[(by(1, 0), (ap_uint<3>)3, bx(1, 0), x(2, 0))] = t29;
                    out[(by(1, 0), (ap_uint<3>)4, bx(1, 0), x(2, 0))] = t30;
                    out[(by(1, 0), (ap_uint<3>)5, bx(1, 0), x(2, 0))] = t31;
                    out[(by(1, 0), (ap_uint<3>)6, bx(1, 0), x(2, 0))] = t32;
                    out[(by(1, 0), (ap_uint<3>)7, bx(1, 0), x(2, 0))] = t33;
                }
            }
        }
    }
}

template <bool scale>
void dct16_block(float in[1024], float out[1024]) {
#pragma HLS INLINE off

    float c1_16 = 1.9615705608064609f;  // 2 * cos(1 * pi / 16)
    float c2_16 = 1.8477590650225735f;  // 2 * cos(2 * pi / 16)
    float c3_16 = 1.6629392246050905f;  // 2 * cos(3 * pi / 16)
    float c4_16 = 1.4142135623730951f;  // 2 * cos(4 * pi / 16)
    float c5_16 = 1.1111404660392046f;  // 2 * cos(5 * pi / 16)
    float c6_16 = 0.7653668647301797f;  // 2 * cos(6 * pi / 16)
    float c7_16 = 0.39018064403225666f; // 2 * cos(7 * pi / 16)

    for (ap_uint<8> by = 0; by < 2; by++) {
        for (ap_uint<8> bx = 0; bx < 2; bx++) {
            for (ap_uint<8> x = 0; x < 16; x++) {
#pragma HLS DEPENDENCE variable = in inter false
#pragma HLS DEPENDENCE variable = out inter false
#pragma HLS LOOP_FLATTEN off
#pragma HLS pipeline

                float t00 = in[((ap_uint<1>)by[0], (ap_uint<4>)0, (ap_uint<1>)bx[0], x(3, 0))] +
                            in[((ap_uint<1>)by[0], (ap_uint<4>)15, (ap_uint<1>)bx[0], x(3, 0))];
                float t01 = in[((ap_uint<1>)by[0], (ap_uint<4>)1, (ap_uint<1>)bx[0], x(3, 0))] +
                            in[((ap_uint<1>)by[0], (ap_uint<4>)14, (ap_uint<1>)bx[0], x(3, 0))];
                float t02 = in[((ap_uint<1>)by[0], (ap_uint<4>)2, (ap_uint<1>)bx[0], x(3, 0))] +
                            in[((ap_uint<1>)by[0], (ap_uint<4>)13, (ap_uint<1>)bx[0], x(3, 0))];
                float t03 = in[((ap_uint<1>)by[0], (ap_uint<4>)3, (ap_uint<1>)bx[0], x(3, 0))] +
                            in[((ap_uint<1>)by[0], (ap_uint<4>)12, (ap_uint<1>)bx[0], x(3, 0))];
                float t04 = in[((ap_uint<1>)by[0], (ap_uint<4>)4, (ap_uint<1>)bx[0], x(3, 0))] +
                            in[((ap_uint<1>)by[0], (ap_uint<4>)11, (ap_uint<1>)bx[0], x(3, 0))];
                float t05 = in[((ap_uint<1>)by[0], (ap_uint<4>)5, (ap_uint<1>)bx[0], x(3, 0))] +
                            in[((ap_uint<1>)by[0], (ap_uint<4>)10, (ap_uint<1>)bx[0], x(3, 0))];
                float t06 = in[((ap_uint<1>)by[0], (ap_uint<4>)6, (ap_uint<1>)bx[0], x(3, 0))] +
                            in[((ap_uint<1>)by[0], (ap_uint<4>)9, (ap_uint<1>)bx[0], x(3, 0))];
                float t07 = in[((ap_uint<1>)by[0], (ap_uint<4>)7, (ap_uint<1>)bx[0], x(3, 0))] +
                            in[((ap_uint<1>)by[0], (ap_uint<4>)8, (ap_uint<1>)bx[0], x(3, 0))];
                float t08 = in[((ap_uint<1>)by[0], (ap_uint<4>)0, (ap_uint<1>)bx[0], x(3, 0))] -
                            in[((ap_uint<1>)by[0], (ap_uint<4>)15, (ap_uint<1>)bx[0], x(3, 0))];
                float t09 = in[((ap_uint<1>)by[0], (ap_uint<4>)1, (ap_uint<1>)bx[0], x(3, 0))] -
                            in[((ap_uint<1>)by[0], (ap_uint<4>)14, (ap_uint<1>)bx[0], x(3, 0))];
                float t10 = in[((ap_uint<1>)by[0], (ap_uint<4>)2, (ap_uint<1>)bx[0], x(3, 0))] -
                            in[((ap_uint<1>)by[0], (ap_uint<4>)13, (ap_uint<1>)bx[0], x(3, 0))];
                float t11 = in[((ap_uint<1>)by[0], (ap_uint<4>)3, (ap_uint<1>)bx[0], x(3, 0))] -
                            in[((ap_uint<1>)by[0], (ap_uint<4>)12, (ap_uint<1>)bx[0], x(3, 0))];
                float t12 = in[((ap_uint<1>)by[0], (ap_uint<4>)4, (ap_uint<1>)bx[0], x(3, 0))] -
                            in[((ap_uint<1>)by[0], (ap_uint<4>)11, (ap_uint<1>)bx[0], x(3, 0))];
                float t13 = in[((ap_uint<1>)by[0], (ap_uint<4>)5, (ap_uint<1>)bx[0], x(3, 0))] -
                            in[((ap_uint<1>)by[0], (ap_uint<4>)10, (ap_uint<1>)bx[0], x(3, 0))];
                float t14 = in[((ap_uint<1>)by[0], (ap_uint<4>)6, (ap_uint<1>)bx[0], x(3, 0))] -
                            in[((ap_uint<1>)by[0], (ap_uint<4>)9, (ap_uint<1>)bx[0], x(3, 0))];
                float t15 = in[((ap_uint<1>)by[0], (ap_uint<4>)7, (ap_uint<1>)bx[0], x(3, 0))] -
                            in[((ap_uint<1>)by[0], (ap_uint<4>)8, (ap_uint<1>)bx[0], x(3, 0))];

                float t16 = t00 + t07;
                float t17 = t01 + t06;
                float t18 = t02 + t05;
                float t19 = t03 + t04;
                float t20 = t00 - t07;
                float t21 = t01 - t06;
                float t22 = t02 - t05;
                float t23 = t03 - t04;
                float t24 = t16 + t19;
                float t25 = t17 + t18;
                float t26 = t16 - t19;
                float t27 = t17 - t18;
                float t30 = t26 - t27;
                float t31 = t27 * c4_16;
                float t34 = t20 - t23;
                float t35 = t21 - t22;
                float t36 = t22 * c4_16;
                float t37 = t23 * c4_16;
                float t38 = t34 + t36;
                float t39 = t35 + t37;
                float t40 = t34 - t36;
                float t41 = t35 - t37;
                float t42 = t38 - t39;
                float t43 = t39 * c2_16;
                float t46 = t40 - t41;
                float t47 = t41 * c6_16;
                float t50 = t08 - t15;
                float t51 = t09 - t14;
                float t52 = t10 - t13;
                float t53 = t11 - t12;
                float t54 = t12 * c4_16;
                float t55 = t13 * c4_16;
                float t56 = t14 * c4_16;
                float t57 = t15 * c4_16;
                float t58 = t50 + t54;
                float t59 = t51 + t55;
                float t60 = t52 + t56;
                float t61 = t53 + t57;
                float t62 = t50 - t54;
                float t63 = t51 - t55;
                float t64 = t52 - t56;
                float t65 = t53 - t57;
                float t66 = t58 - t61;
                float t67 = t59 - t60;
                float t68 = t60 * c2_16;
                float t69 = t61 * c2_16;
                float t70 = t66 + t68;
                float t71 = t67 + t69;
                float t72 = t66 - t68;
                float t73 = t67 - t69;
                float t74 = t70 - t71;
                float t75 = t71 * c1_16;
                float t78 = t72 - t73;
                float t79 = t73 * c7_16;
                float t82 = t62 - t65;
                float t83 = t63 - t64;
                float t84 = t64 * c6_16;
                float t85 = t65 * c6_16;
                float t86 = t82 + t84;
                float t87 = t83 + t85;
                float t88 = t82 - t84;
                float t89 = t83 - t85;
                float t90 = t86 - t87;
                float t91 = t87 * c3_16;
                float t94 = t88 - t89;
                float t95 = t89 * c5_16;

                float t96 = t24 + t25;
                float t97 = t24 - t25;
                float t98 = t30 + t31;
                float t99 = t30 - t31;
                float t100 = t42 + t43;
                float t101 = t42 - t43;
                float t102 = t46 + t47;
                float t103 = t46 - t47;
                float t104 = t74 + t75;
                float t105 = t74 - t75;
                float t106 = t78 + t79;
                float t107 = t78 - t79;
                float t108 = t90 + t91;
                float t109 = t90 - t91;
                float t110 = t94 + t95;
                float t111 = t94 - t95;

#ifdef DEBUG_DCT
                std::cout << "t0=" << t00 << std::endl;
                std::cout << "t1=" << t01 << std::endl;
                std::cout << "t2=" << t02 << std::endl;
                std::cout << "t3=" << t03 << std::endl;
                std::cout << "t4=" << t04 << std::endl;
                std::cout << "t5=" << t05 << std::endl;
                std::cout << "t6=" << t06 << std::endl;
                std::cout << "t7=" << t07 << std::endl;
                std::cout << "t8=" << t08 << std::endl;
                std::cout << "t9=" << t09 << std::endl;
                std::cout << "t10=" << t10 << std::endl;
                std::cout << "t11=" << t11 << std::endl;
                std::cout << "t12=" << t12 << std::endl;
                std::cout << "t13=" << t13 << std::endl;
                std::cout << "t14=" << t14 << std::endl;
                std::cout << "t15=" << t15 << std::endl;
#endif

                if (scale) {
                    out[((ap_uint<1>)by[0], (ap_uint<4>)0, (ap_uint<1>)bx[0], x(3, 0))] = t96 / 256;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)8, (ap_uint<1>)bx[0], x(3, 0))] = t97 / 256;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)4, (ap_uint<1>)bx[0], x(3, 0))] = t98 / 256;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)12, (ap_uint<1>)bx[0], x(3, 0))] = t99 / 256;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)2, (ap_uint<1>)bx[0], x(3, 0))] = t100 / 256;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)14, (ap_uint<1>)bx[0], x(3, 0))] = t101 / 256;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)6, (ap_uint<1>)bx[0], x(3, 0))] = t102 / 256;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)10, (ap_uint<1>)bx[0], x(3, 0))] = t103 / 256;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)1, (ap_uint<1>)bx[0], x(3, 0))] = t104 / 256;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)15, (ap_uint<1>)bx[0], x(3, 0))] = t105 / 256;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)7, (ap_uint<1>)bx[0], x(3, 0))] = t106 / 256;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)9, (ap_uint<1>)bx[0], x(3, 0))] = t107 / 256;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)3, (ap_uint<1>)bx[0], x(3, 0))] = t108 / 256;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)13, (ap_uint<1>)bx[0], x(3, 0))] = t109 / 256;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)5, (ap_uint<1>)bx[0], x(3, 0))] = t110 / 256;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)11, (ap_uint<1>)bx[0], x(3, 0))] = t111 / 256;
                } else {
                    out[((ap_uint<1>)by[0], (ap_uint<4>)0, (ap_uint<1>)bx[0], x(3, 0))] = t96;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)8, (ap_uint<1>)bx[0], x(3, 0))] = t97;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)4, (ap_uint<1>)bx[0], x(3, 0))] = t98;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)12, (ap_uint<1>)bx[0], x(3, 0))] = t99;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)2, (ap_uint<1>)bx[0], x(3, 0))] = t100;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)14, (ap_uint<1>)bx[0], x(3, 0))] = t101;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)6, (ap_uint<1>)bx[0], x(3, 0))] = t102;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)10, (ap_uint<1>)bx[0], x(3, 0))] = t103;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)1, (ap_uint<1>)bx[0], x(3, 0))] = t104;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)15, (ap_uint<1>)bx[0], x(3, 0))] = t105;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)7, (ap_uint<1>)bx[0], x(3, 0))] = t106;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)9, (ap_uint<1>)bx[0], x(3, 0))] = t107;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)3, (ap_uint<1>)bx[0], x(3, 0))] = t108;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)13, (ap_uint<1>)bx[0], x(3, 0))] = t109;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)5, (ap_uint<1>)bx[0], x(3, 0))] = t110;
                    out[((ap_uint<1>)by[0], (ap_uint<4>)11, (ap_uint<1>)bx[0], x(3, 0))] = t111;
                }
            }
        }
    }
}

template <bool scale>
void dct32_block(float in[1024], float out[1024]) {
#pragma HLS INLINE off

    float c2_64 = 1.990369453344393857f;  // 2 * cos(2 * pi / 64)
    float c4_64 = 1.961570560806460861f;  // 2 * cos(4 * pi / 64)
    float c6_64 = 1.913880671464417649f;  // 2 * cos(6 * pi / 64)
    float c8_64 = 1.847759065022573477f;  // 2 * cos(8 * pi / 64)
    float c10_64 = 1.763842528696710099f; // 2 * cos(10 * pi / 64)
    float c12_64 = 1.662939224605090471f; // 2 * cos(12 * pi / 64)
    float c14_64 = 1.546020906725473987f; // 2 * cos(14 * pi / 64)
    float c16_64 = 1.414213562373095145f; // 2 * cos(16 * pi / 64)
    float c18_64 = 1.268786568327290976f; // 2 * cos(18 * pi / 64)
    float c20_64 = 1.111140466039204577f; // 2 * cos(20 * pi / 64)
    float c22_64 = 0.942793473651995617f; // 2 * cos(22 * pi / 64)
    float c24_64 = 0.765366864730179675f; // 2 * cos(24 * pi / 64)
    float c26_64 = 0.580569354508924662f; // 2 * cos(26 * pi / 64)
    float c28_64 = 0.390180644032256663f; // 2 * cos(28 * pi / 64)
    float c30_64 = 0.196034280659121540f; // 2 * cos(30 * pi / 64)

    for (ap_uint<8> i = 0; i < 32; i++) {
#pragma HLS DEPENDENCE variable = in inter false
#pragma HLS DEPENDENCE variable = out inter false
#pragma HLS pipeline

        float t00 = in[((ap_uint<5>)0, i(4, 0))] + in[((ap_uint<5>)31, i(4, 0))];
        float t01 = in[((ap_uint<5>)1, i(4, 0))] + in[((ap_uint<5>)30, i(4, 0))];
        float t02 = in[((ap_uint<5>)2, i(4, 0))] + in[((ap_uint<5>)29, i(4, 0))];
        float t03 = in[((ap_uint<5>)3, i(4, 0))] + in[((ap_uint<5>)28, i(4, 0))];
        float t04 = in[((ap_uint<5>)4, i(4, 0))] + in[((ap_uint<5>)27, i(4, 0))];
        float t05 = in[((ap_uint<5>)5, i(4, 0))] + in[((ap_uint<5>)26, i(4, 0))];
        float t06 = in[((ap_uint<5>)6, i(4, 0))] + in[((ap_uint<5>)25, i(4, 0))];
        float t07 = in[((ap_uint<5>)7, i(4, 0))] + in[((ap_uint<5>)24, i(4, 0))];
        float t08 = in[((ap_uint<5>)8, i(4, 0))] + in[((ap_uint<5>)23, i(4, 0))];
        float t09 = in[((ap_uint<5>)9, i(4, 0))] + in[((ap_uint<5>)22, i(4, 0))];
        float t10 = in[((ap_uint<5>)10, i(4, 0))] + in[((ap_uint<5>)21, i(4, 0))];
        float t11 = in[((ap_uint<5>)11, i(4, 0))] + in[((ap_uint<5>)20, i(4, 0))];
        float t12 = in[((ap_uint<5>)12, i(4, 0))] + in[((ap_uint<5>)19, i(4, 0))];
        float t13 = in[((ap_uint<5>)13, i(4, 0))] + in[((ap_uint<5>)18, i(4, 0))];
        float t14 = in[((ap_uint<5>)14, i(4, 0))] + in[((ap_uint<5>)17, i(4, 0))];
        float t15 = in[((ap_uint<5>)15, i(4, 0))] + in[((ap_uint<5>)16, i(4, 0))];
        float t16 = in[((ap_uint<5>)0, i(4, 0))] - in[((ap_uint<5>)31, i(4, 0))];
        float t17 = in[((ap_uint<5>)1, i(4, 0))] - in[((ap_uint<5>)30, i(4, 0))];
        float t18 = in[((ap_uint<5>)2, i(4, 0))] - in[((ap_uint<5>)29, i(4, 0))];
        float t19 = in[((ap_uint<5>)3, i(4, 0))] - in[((ap_uint<5>)28, i(4, 0))];
        float t20 = in[((ap_uint<5>)4, i(4, 0))] - in[((ap_uint<5>)27, i(4, 0))];
        float t21 = in[((ap_uint<5>)5, i(4, 0))] - in[((ap_uint<5>)26, i(4, 0))];
        float t22 = in[((ap_uint<5>)6, i(4, 0))] - in[((ap_uint<5>)25, i(4, 0))];
        float t23 = in[((ap_uint<5>)7, i(4, 0))] - in[((ap_uint<5>)24, i(4, 0))];
        float t24 = in[((ap_uint<5>)8, i(4, 0))] - in[((ap_uint<5>)23, i(4, 0))];
        float t25 = in[((ap_uint<5>)9, i(4, 0))] - in[((ap_uint<5>)22, i(4, 0))];
        float t26 = in[((ap_uint<5>)10, i(4, 0))] - in[((ap_uint<5>)21, i(4, 0))];
        float t27 = in[((ap_uint<5>)11, i(4, 0))] - in[((ap_uint<5>)20, i(4, 0))];
        float t28 = in[((ap_uint<5>)12, i(4, 0))] - in[((ap_uint<5>)19, i(4, 0))];
        float t29 = in[((ap_uint<5>)13, i(4, 0))] - in[((ap_uint<5>)18, i(4, 0))];
        float t30 = in[((ap_uint<5>)14, i(4, 0))] - in[((ap_uint<5>)17, i(4, 0))];
        float t31 = in[((ap_uint<5>)15, i(4, 0))] - in[((ap_uint<5>)16, i(4, 0))];

        float t32 = t00 + t15;
        float t33 = t01 + t14;
        float t34 = t02 + t13;
        float t35 = t03 + t12;
        float t36 = t04 + t11;
        float t37 = t05 + t10;
        float t38 = t06 + t09;
        float t39 = t07 + t08;
        float t40 = t00 - t15;
        float t41 = t01 - t14;
        float t42 = t02 - t13;
        float t43 = t03 - t12;
        float t44 = t04 - t11;
        float t45 = t05 - t10;
        float t46 = t06 - t09;
        float t47 = t07 - t08;
        float t48 = t32 + t39;
        float t49 = t33 + t38;
        float t50 = t34 + t37;
        float t51 = t35 + t36;
        float t52 = t32 - t39;
        float t53 = t33 - t38;
        float t54 = t34 - t37;
        float t55 = t35 - t36;
        float t56 = t48 + t51;
        float t57 = t49 + t50;
        float t58 = t48 - t51;
        float t59 = t49 - t50;
        float t60 = t56 + t57;
        float t61 = t56 - t57;
        float t62 = t58 - t59;
        float t63 = t59 * c16_64;
        float t64 = t62 + t63;
        float t65 = t62 - t63;
        float t66 = t52 - t55;
        float t67 = t53 - t54;
        float t68 = t54 * c16_64;
        float t69 = t55 * c16_64;
        float t70 = t66 + t68;
        float t71 = t67 + t69;
        float t72 = t66 - t68;
        float t73 = t67 - t69;
        float t74 = t70 - t71;
        float t75 = t71 * c8_64;
        float t76 = t74 + t75;
        float t77 = t74 - t75;
        float t78 = t72 - t73;
        float t79 = t73 * c24_64;
        float t80 = t78 + t79;
        float t81 = t78 - t79;
        float t82 = t40 - t47;
        float t83 = t41 - t46;
        float t84 = t42 - t45;
        float t85 = t43 - t44;
        float t86 = t44 * c16_64;
        float t87 = t45 * c16_64;
        float t88 = t46 * c16_64;
        float t89 = t47 * c16_64;
        float t90 = t82 + t86;
        float t91 = t83 + t87;
        float t92 = t84 + t88;
        float t93 = t85 + t89;
        float t94 = t82 - t86;
        float t95 = t83 - t87;
        float t96 = t84 - t88;
        float t97 = t85 - t89;
        float t98 = t90 - t93;
        float t99 = t91 - t92;
        float t100 = t92 * c8_64;
        float t101 = t93 * c8_64;
        float t102 = t98 + t100;
        float t103 = t99 + t101;
        float t104 = t98 - t100;
        float t105 = t99 - t101;
        float t106 = t102 - t103;
        float t107 = t103 * c4_64;
        float t108 = t106 + t107;
        float t109 = t106 - t107;
        float t110 = t104 - t105;
        float t111 = t105 * c28_64;
        float t112 = t110 + t111;
        float t113 = t110 - t111;
        float t114 = t94 - t97;
        float t115 = t95 - t96;
        float t116 = t96 * c24_64;
        float t117 = t97 * c24_64;
        float t118 = t114 + t116;
        float t119 = t115 + t117;
        float t120 = t114 - t116;
        float t121 = t115 - t117;
        float t122 = t118 - t119;
        float t123 = t119 * c12_64;
        float t124 = t122 + t123;
        float t125 = t122 - t123;
        float t126 = t120 - t121;
        float t127 = t121 * c20_64;
        float t128 = t126 + t127;
        float t129 = t126 - t127;
        float t130 = t16 - t31;
        float t131 = t17 - t30;
        float t132 = t18 - t29;
        float t133 = t19 - t28;
        float t134 = t20 - t27;
        float t135 = t21 - t26;
        float t136 = t22 - t25;
        float t137 = t23 - t24;
        float t138 = t24 * c16_64;
        float t139 = t25 * c16_64;
        float t140 = t26 * c16_64;
        float t141 = t27 * c16_64;
        float t142 = t28 * c16_64;
        float t143 = t29 * c16_64;
        float t144 = t30 * c16_64;
        float t145 = t31 * c16_64;
        float t146 = t130 + t138;
        float t147 = t131 + t139;
        float t148 = t132 + t140;
        float t149 = t133 + t141;
        float t150 = t134 + t142;
        float t151 = t135 + t143;
        float t152 = t136 + t144;
        float t153 = t137 + t145;
        float t154 = t130 - t138;
        float t155 = t131 - t139;
        float t156 = t132 - t140;
        float t157 = t133 - t141;
        float t158 = t134 - t142;
        float t159 = t135 - t143;
        float t160 = t136 - t144;
        float t161 = t137 - t145;
        float t162 = t146 - t153;
        float t163 = t147 - t152;
        float t164 = t148 - t151;
        float t165 = t149 - t150;
        float t166 = t150 * c8_64;
        float t167 = t151 * c8_64;
        float t168 = t152 * c8_64;
        float t169 = t153 * c8_64;
        float t170 = t162 + t166;
        float t171 = t163 + t167;
        float t172 = t164 + t168;
        float t173 = t165 + t169;
        float t174 = t162 - t166;
        float t175 = t163 - t167;
        float t176 = t164 - t168;
        float t177 = t165 - t169;
        float t178 = t170 - t173;
        float t179 = t171 - t172;
        float t180 = t172 * c4_64;
        float t181 = t173 * c4_64;
        float t182 = t178 + t180;
        float t183 = t179 + t181;
        float t184 = t178 - t180;
        float t185 = t179 - t181;
        float t186 = t182 - t183;
        float t187 = t183 * c2_64;
        float t188 = t186 + t187;
        float t189 = t186 - t187;
        float t190 = t184 - t185;
        float t191 = t185 * c30_64;
        float t192 = t190 + t191;
        float t193 = t190 - t191;
        float t194 = t174 - t177;
        float t195 = t175 - t176;
        float t196 = t176 * c28_64;
        float t197 = t177 * c28_64;
        float t198 = t194 + t196;
        float t199 = t195 + t197;
        float t200 = t194 - t196;
        float t201 = t195 - t197;
        float t202 = t198 - t199;
        float t203 = t199 * c14_64;
        float t204 = t202 + t203;
        float t205 = t202 - t203;
        float t206 = t200 - t201;
        float t207 = t201 * c18_64;
        float t208 = t206 + t207;
        float t209 = t206 - t207;
        float t210 = t154 - t161;
        float t211 = t155 - t160;
        float t212 = t156 - t159;
        float t213 = t157 - t158;
        float t214 = t158 * c24_64;
        float t215 = t159 * c24_64;
        float t216 = t160 * c24_64;
        float t217 = t161 * c24_64;
        float t218 = t210 + t214;
        float t219 = t211 + t215;
        float t220 = t212 + t216;
        float t221 = t213 + t217;
        float t222 = t210 - t214;
        float t223 = t211 - t215;
        float t224 = t212 - t216;
        float t225 = t213 - t217;
        float t226 = t218 - t221;
        float t227 = t219 - t220;
        float t228 = t220 * c12_64;
        float t229 = t221 * c12_64;
        float t230 = t226 + t228;
        float t231 = t227 + t229;
        float t232 = t226 - t228;
        float t233 = t227 - t229;
        float t234 = t230 - t231;
        float t235 = t231 * c6_64;
        float t236 = t234 + t235;
        float t237 = t234 - t235;
        float t238 = t232 - t233;
        float t239 = t233 * c26_64;
        float t240 = t238 + t239;
        float t241 = t238 - t239;
        float t242 = t222 - t225;
        float t243 = t223 - t224;
        float t244 = t224 * c20_64;
        float t245 = t225 * c20_64;
        float t246 = t242 + t244;
        float t247 = t243 + t245;
        float t248 = t242 - t244;
        float t249 = t243 - t245;
        float t250 = t246 - t247;
        float t251 = t247 * c10_64;
        float t252 = t250 + t251;
        float t253 = t250 - t251;
        float t254 = t248 - t249;
        float t255 = t249 * c22_64;
        float t256 = t254 + t255;
        float t257 = t254 - t255;

        if (scale) {
            out[((ap_uint<5>)0, i(4, 0))] = t60 / 1024;
            out[((ap_uint<5>)1, i(4, 0))] = t188 / 1024;
            out[((ap_uint<5>)2, i(4, 0))] = t108 / 1024;
            out[((ap_uint<5>)3, i(4, 0))] = t236 / 1024;
            out[((ap_uint<5>)4, i(4, 0))] = t76 / 1024;
            out[((ap_uint<5>)5, i(4, 0))] = t252 / 1024;
            out[((ap_uint<5>)6, i(4, 0))] = t124 / 1024;
            out[((ap_uint<5>)7, i(4, 0))] = t204 / 1024;
            out[((ap_uint<5>)8, i(4, 0))] = t64 / 1024;
            out[((ap_uint<5>)9, i(4, 0))] = t208 / 1024;
            out[((ap_uint<5>)10, i(4, 0))] = t128 / 1024;
            out[((ap_uint<5>)11, i(4, 0))] = t256 / 1024;
            out[((ap_uint<5>)12, i(4, 0))] = t80 / 1024;
            out[((ap_uint<5>)13, i(4, 0))] = t240 / 1024;
            out[((ap_uint<5>)14, i(4, 0))] = t112 / 1024;
            out[((ap_uint<5>)15, i(4, 0))] = t192 / 1024;
            out[((ap_uint<5>)16, i(4, 0))] = t61 / 1024;
            out[((ap_uint<5>)17, i(4, 0))] = t193 / 1024;
            out[((ap_uint<5>)18, i(4, 0))] = t113 / 1024;
            out[((ap_uint<5>)19, i(4, 0))] = t241 / 1024;
            out[((ap_uint<5>)20, i(4, 0))] = t81 / 1024;
            out[((ap_uint<5>)21, i(4, 0))] = t257 / 1024;
            out[((ap_uint<5>)22, i(4, 0))] = t129 / 1024;
            out[((ap_uint<5>)23, i(4, 0))] = t209 / 1024;
            out[((ap_uint<5>)24, i(4, 0))] = t65 / 1024;
            out[((ap_uint<5>)25, i(4, 0))] = t205 / 1024;
            out[((ap_uint<5>)26, i(4, 0))] = t125 / 1024;
            out[((ap_uint<5>)27, i(4, 0))] = t253 / 1024;
            out[((ap_uint<5>)28, i(4, 0))] = t77 / 1024;
            out[((ap_uint<5>)29, i(4, 0))] = t237 / 1024;
            out[((ap_uint<5>)30, i(4, 0))] = t109 / 1024;
            out[((ap_uint<5>)31, i(4, 0))] = t189 / 1024;
        } else {
            out[((ap_uint<5>)0, i(4, 0))] = t60;
            out[((ap_uint<5>)1, i(4, 0))] = t188;
            out[((ap_uint<5>)2, i(4, 0))] = t108;
            out[((ap_uint<5>)3, i(4, 0))] = t236;
            out[((ap_uint<5>)4, i(4, 0))] = t76;
            out[((ap_uint<5>)5, i(4, 0))] = t252;
            out[((ap_uint<5>)6, i(4, 0))] = t124;
            out[((ap_uint<5>)7, i(4, 0))] = t204;
            out[((ap_uint<5>)8, i(4, 0))] = t64;
            out[((ap_uint<5>)9, i(4, 0))] = t208;
            out[((ap_uint<5>)10, i(4, 0))] = t128;
            out[((ap_uint<5>)11, i(4, 0))] = t256;
            out[((ap_uint<5>)12, i(4, 0))] = t80;
            out[((ap_uint<5>)13, i(4, 0))] = t240;
            out[((ap_uint<5>)14, i(4, 0))] = t112;
            out[((ap_uint<5>)15, i(4, 0))] = t192;
            out[((ap_uint<5>)16, i(4, 0))] = t61;
            out[((ap_uint<5>)17, i(4, 0))] = t193;
            out[((ap_uint<5>)18, i(4, 0))] = t113;
            out[((ap_uint<5>)19, i(4, 0))] = t241;
            out[((ap_uint<5>)20, i(4, 0))] = t81;
            out[((ap_uint<5>)21, i(4, 0))] = t257;
            out[((ap_uint<5>)22, i(4, 0))] = t129;
            out[((ap_uint<5>)23, i(4, 0))] = t209;
            out[((ap_uint<5>)24, i(4, 0))] = t65;
            out[((ap_uint<5>)25, i(4, 0))] = t205;
            out[((ap_uint<5>)26, i(4, 0))] = t125;
            out[((ap_uint<5>)27, i(4, 0))] = t253;
            out[((ap_uint<5>)28, i(4, 0))] = t77;
            out[((ap_uint<5>)29, i(4, 0))] = t237;
            out[((ap_uint<5>)30, i(4, 0))] = t109;
            out[((ap_uint<5>)31, i(4, 0))] = t189;
        }
    }
}

template <int N>
void TransposeN(float in[N * N], float out[N * N]) {
#pragma HLS INLINE off

    for (ap_uint<8> y = 0; y < N; y++) {
        for (ap_uint<8> x = 0; x < N; x++) {
#pragma HLS pipeline II = 1

            out[y * N + x] = in[x * N + y];
        }
    }
}

void TransposeBlock32(float in[1024], float out[1024]) {
#pragma HLS INLINE off

    for (ap_uint<8> y = 0; y < 32; y++) {
        for (ap_uint<8> x = 0; x < 32; x++) {
#pragma HLS pipeline II = 1

            ap_uint<10> addr_i, addr_o;
            addr_i(9, 5) = x(4, 0);
            addr_i(4, 0) = y(4, 0);
            addr_o(9, 5) = y(4, 0);
            addr_o(4, 0) = x(4, 0);

            out[addr_o] = in[addr_i];
        }
    }
}

void TransposeBlock16(float in[1024], float out[1024]) {
#pragma HLS INLINE off

    for (ap_uint<8> by = 0; by < 2; by++) {
        for (ap_uint<8> bx = 0; bx < 2; bx++) {
            for (ap_uint<8> y = 0; y < 16; y++) {
                for (ap_uint<8> x = 0; x < 16; x++) {
#pragma HLS pipeline II = 1

                    ap_uint<10> addr_i, addr_o;
                    addr_i[9] = (ap_uint<1>)by[0];
                    addr_i(8, 5) = x(3, 0);
                    addr_i[4] = (ap_uint<1>)bx[0];
                    addr_i(3, 0) = y(3, 0);
                    addr_o[9] = (ap_uint<1>)by[0];
                    addr_o(8, 5) = y(3, 0);
                    addr_o[4] = (ap_uint<1>)bx[0];
                    addr_o(3, 0) = x(3, 0);

                    out[addr_o] = in[addr_i];
                }
            }
        }
    }
}

void TransposeBlock8(float in[1024], float out[1024]) {
#pragma HLS INLINE off

    for (ap_uint<8> by = 0; by < 4; by++) {
        for (ap_uint<8> bx = 0; bx < 4; bx++) {
            for (ap_uint<8> y = 0; y < 8; y++) {
                for (ap_uint<8> x = 0; x < 8; x++) {
#pragma HLS pipeline II = 1

                    ap_uint<10> addr_i, addr_o;
                    addr_i(9, 8) = by(1, 0);
                    addr_i(7, 5) = x(2, 0);
                    addr_i(4, 3) = bx(1, 0);
                    addr_i(2, 0) = y(2, 0);
                    addr_o(9, 8) = by(1, 0);
                    addr_o(7, 5) = y(2, 0);
                    addr_o(4, 3) = bx(1, 0);
                    addr_o(2, 0) = x(2, 0);

                    out[addr_o] = in[addr_i];
                }
            }
        }
    }
}

void TransposeBlock4(float in[1024], float out[1024]) {
#pragma HLS INLINE off

    for (ap_uint<8> by = 0; by < 8; by++) {
        for (ap_uint<8> bx = 0; bx < 8; bx++) {
            for (ap_uint<8> y = 0; y < 4; y++) {
                for (ap_uint<8> x = 0; x < 4; x++) {
#pragma HLS pipeline II = 1

                    ap_uint<10> addr_i, addr_o;
                    addr_i(9, 7) = by(2, 0);
                    addr_i(6, 5) = x(1, 0);
                    addr_i(4, 2) = bx(2, 0);
                    addr_i(1, 0) = y(1, 0);
                    addr_o(9, 7) = by(2, 0);
                    addr_o(6, 5) = y(1, 0);
                    addr_o(4, 2) = bx(2, 0);
                    addr_o(1, 0) = x(1, 0);

                    out[addr_o] = in[addr_i];
                }
            }
        }
    }
}

void ComputeDC(float block0[4], float block1[4]) {
#pragma HLS INLINE off

    float a0, b0, c0, d0;
    a0 = block0[0] + block0[1];
    b0 = block0[0] - block0[1];
    c0 = block0[2] + block0[3];
    d0 = block0[2] - block0[3];

    float a1, b1, c1, d1;
    a1 = a0 + c0;
    b1 = a0 - c0;
    c1 = b0 + d0;
    d1 = b0 - d0;

    block1[0] = a1 * 0.25f;
    block1[1] = b1 * 0.25f;
    block1[2] = c1 * 0.25f;
    block1[3] = d1 * 0.25f;
}

void LoadBlock4to8(ap_uint<8> by, ap_uint<8> bx, float in[1024], float tmp[64], float block0[4]) {
#pragma HLS INLINE off

load:
    for (ap_uint<8> dy = 0; dy < 2; dy++) {
        for (ap_uint<8> dx = 0; dx < 2; dx++) {
            for (ap_uint<8> y = 0; y < 4; y++) {
                for (ap_uint<8> x = 0; x < 4; x++) {
#pragma HLS pipeline II = 1

                    ap_uint<10> addr_i;
                    ap_uint<6> addr_o;
                    addr_i(9, 8) = by(1, 0);
                    addr_i[7] = dy[0];
                    addr_i(6, 5) = y(1, 0);
                    addr_i(4, 3) = bx(1, 0);
                    addr_i[2] = dx[0];
                    addr_i(1, 0) = x(1, 0);

                    addr_o(5, 4) = y(1, 0);
                    addr_o[3] = dy[0];
                    addr_o(2, 1) = x(1, 0);
                    addr_o[0] = dx[0];

                    tmp[addr_o] = in[addr_i];

                    if (x == 0 && y == 0) block0[(dy[0], dx[0])] = in[addr_i];
                }
            }
        }
    }
}

void FeedBlock4to8(ap_uint<8> by, ap_uint<8> bx, float tmp[64], float block1[4], float out[1024]) {
#pragma HLS INLINE off

feed:
    for (ap_uint<8> y = 0; y < 8; y++) {
        for (ap_uint<8> x = 0; x < 8; x++) {
#pragma HLS pipeline II = 1

            ap_uint<6> addr_i = (y(2, 0), x(2, 0));
            ap_uint<10> addr_o = (by(1, 0), y(2, 0), bx(1, 0), x(2, 0));

            if (y < 2 && x < 2)
                out[addr_o] = block1[((ap_uint<1>)y[0], (ap_uint<1>)x[0])];
            else
                out[addr_o] = tmp[addr_i];
        }
    }
}

void TransformBlock4to8(float in[1024], float out[1024]) {
#pragma HLS INLINE off

    for (ap_uint<8> by = 0; by < 4; by++) {
        for (ap_uint<8> bx = 0; bx < 4; bx++) {
#pragma HLS dataflow

            float tmp[64];
            float block0[4];
            float block1[4];

            LoadBlock4to8(by, bx, in, tmp, block0);
            ComputeDC(block0, block1);
            FeedBlock4to8(by, bx, tmp, block1, out);
        }
    }
}

void DCT4x4_block(float in[1024], float out[1024]) {
#pragma HLS INLINE off
#pragma HLS dataflow

    float temp0[1024];
#pragma HLS RESOURCE variable = temp0 core = RAM_2P_BRAM
    float temp1[1024];
#pragma HLS RESOURCE variable = temp1 core = RAM_2P_BRAM
    float temp2[1024];
#pragma HLS RESOURCE variable = temp2 core = RAM_2P_BRAM

    dct4_block<false>(in, temp0);
    TransposeBlock4(temp0, temp1);
    dct4_block<true>(temp1, temp2);
    TransformBlock4to8(temp2, out);

#ifdef DEBUF_DCT
    for (ap_uint<8> by = 0; by < 8; by++) {
        for (ap_uint<8> bx = 0; bx < 8; bx++) {
            for (ap_uint<8> y = 0; y < 4; y++) {
                for (ap_uint<8> x = 0; x < 4; x++) {
                    ap_uint<10> addr;
                    addr(9, 7) = by(2, 0);
                    addr(6, 5) = y(1, 0);
                    addr(4, 2) = bx(2, 0);
                    addr(1, 0) = x(1, 0);

                    std::cout << "dct4_before_interleave: id=" << addr << " " << temp2[addr] << std::endl;
                }
            }
        }
    }

    for (ap_uint<8> by = 0; by < 4; by++) {
        for (ap_uint<8> bx = 0; bx < 4; bx++) {
            for (ap_uint<8> y = 0; y < 8; y++) {
                for (ap_uint<8> x = 0; x < 8; x++) {
                    ap_uint<10> addr;
                    addr(9, 8) = by(1, 0);
                    addr(7, 5) = y(2, 0);
                    addr(4, 3) = bx(1, 0);
                    addr(2, 0) = x(2, 0);

                    std::cout << "dct4_after_interleave: id=" << addr << " " << out[addr] << std::endl;
                }
            }
        }
    }
#endif
}

void DCT4x4Top(ap_uint<16> xblock, ap_uint<16> yblock, hls::stream<float>& in, hls::stream<float>& out) {
#pragma HLS INLINE off
    for (ap_uint<8> cnty = 0; cnty < yblock; cnty++) {
        for (ap_uint<8> cntx = 0; cntx < xblock; cntx++) {
#pragma HLS DATAFLOW

            float temp0[1024];
#pragma HLS RESOURCE variable = temp0 core = RAM_2P_BRAM
            float temp1[1024];
#pragma HLS RESOURCE variable = temp1 core = RAM_2P_BRAM
            float temp2[1024];
#pragma HLS RESOURCE variable = temp2 core = RAM_2P_BRAM
            float temp3[1024];
#pragma HLS RESOURCE variable = temp3 core = RAM_2P_BRAM
            float temp4[1024];
#pragma HLS RESOURCE variable = temp4 core = RAM_2P_BRAM

        load:
            for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1

                temp0[i] = in.read();
            }

            dct4_block<false>(temp0, temp1);
            TransposeBlock4(temp1, temp2);
            dct4_block<true>(temp2, temp3);
            TransformBlock4to8(temp3, temp4);

        feed:
            for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1

                out.write(temp4[i]);
            }
        }
    }
}

void DCT8x8_block(float in[1024], float out[1024]) {
#pragma HLS INLINE off
#pragma HLS dataflow

    float temp0[1024];
#pragma HLS RESOURCE variable = temp0 core = RAM_2P_BRAM
    float temp1[1024];
#pragma HLS RESOURCE variable = temp1 core = RAM_2P_BRAM

    dct8_block<false>(in, temp0);
    TransposeBlock8(temp0, temp1);
    dct8_block<true>(temp1, out);

#ifdef DEBUG_DCT
    for (ap_uint<8> by = 0; by < 4; by++) {
        for (ap_uint<8> bx = 0; bx < 4; bx++) {
            for (ap_uint<8> y = 0; y < 8; y++) {
                for (ap_uint<8> x = 0; x < 8; x++) {
                    ap_uint<10> addr;
                    addr(9, 8) = by(1, 0);
                    addr(7, 5) = y(2, 0);
                    addr(4, 3) = bx(1, 0);
                    addr(2, 0) = x(2, 0);

                    if (by == 0 && bx == 1)
                        std::cout << "dct8: id=" << addr << " in=" << in[addr] << " temp0=" << temp0[addr]
                                  << " temp1=" << temp1[addr] << " out=" << out[addr] << std::endl;
                }
            }
        }
    }
#endif
}

void DCT8x8Top(ap_uint<16> xblock, ap_uint<16> yblock, hls::stream<float>& in, hls::stream<float>& out) {
#pragma HLS INLINE off
    for (ap_uint<8> cnty = 0; cnty < yblock; cnty++) {
        for (ap_uint<8> cntx = 0; cntx < xblock; cntx++) {
#pragma HLS DATAFLOW

            float temp0[1024];
#pragma HLS RESOURCE variable = temp0 core = RAM_2P_BRAM
            float temp1[1024];
#pragma HLS RESOURCE variable = temp1 core = RAM_2P_BRAM
            float temp2[1024];
#pragma HLS RESOURCE variable = temp2 core = RAM_2P_BRAM
            float temp3[1024];
#pragma HLS RESOURCE variable = temp3 core = RAM_2P_BRAM

        load:
            for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1

                temp0[i] = in.read();
            }

            dct8_block<false>(temp0, temp1);
            TransposeBlock8(temp1, temp2);
            dct8_block<true>(temp2, temp3);

        feed:
            for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1

                out.write(temp3[i]);
            }
        }
    }
}

void DCT16x16_block(float in[1024], float out[1024]) {
#pragma HLS INLINE off
#pragma HLS dataflow

    float temp0[1024];
#pragma HLS RESOURCE variable = temp0 core = RAM_2P_BRAM
    float temp1[1024];
#pragma HLS RESOURCE variable = temp1 core = RAM_2P_BRAM

    dct16_block<false>(in, temp0);
    TransposeBlock16(temp0, temp1);
    dct16_block<true>(temp1, out);

#ifdef DEBUG_DCT
    for (ap_uint<8> by = 0; by < 2; by++) {
        for (ap_uint<8> bx = 0; bx < 2; bx++) {
            for (ap_uint<8> y = 0; y < 16; y++) {
                for (ap_uint<8> x = 0; x < 16; x++) {
                    ap_uint<10> addr;
                    addr[9] = (ap_uint<1>)by[0];
                    addr(8, 5) = y(3, 0);
                    addr[4] = (ap_uint<1>)bx[0];
                    addr(3, 0) = x(3, 0);

                    std::cout << "dct16: id=" << addr << " in=" << in[addr] << " temp0=" << temp0[addr]
                              << " temp1=" << temp1[addr] << " out=" << out[addr] << std::endl;
                }
            }
        }
    }
#endif
}

void DCT16x16Top(ap_uint<16> xblock, ap_uint<16> yblock, hls::stream<float>& in, hls::stream<float>& out) {
#pragma HLS INLINE off
    for (ap_uint<8> cnty = 0; cnty < yblock; cnty++) {
        for (ap_uint<8> cntx = 0; cntx < xblock; cntx++) {
#pragma HLS DATAFLOW

            float temp0[1024];
#pragma HLS RESOURCE variable = temp0 core = RAM_2P_BRAM
            float temp1[1024];
#pragma HLS RESOURCE variable = temp1 core = RAM_2P_BRAM
            float temp2[1024];
#pragma HLS RESOURCE variable = temp2 core = RAM_2P_BRAM
            float temp3[1024];
#pragma HLS RESOURCE variable = temp3 core = RAM_2P_BRAM

        load:
            for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1

                temp0[i] = in.read();
            }

            dct16_block<false>(temp0, temp1);
            TransposeBlock16(temp1, temp2);
            dct16_block<true>(temp2, temp3);

        feed:
            for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1

                out.write(temp3[i]);
            }
        }
    }
}

void DCT32x32_block(float in[1024], float out[1024]) {
#pragma HLS INLINE off
#pragma HLS dataflow

    float temp0[1024];
#pragma HLS RESOURCE variable = temp0 core = RAM_2P_BRAM
    float temp1[1024];
#pragma HLS RESOURCE variable = temp1 core = RAM_2P_BRAM

    dct32_block<false>(in, temp0);
    TransposeBlock32(temp0, temp1);
    dct32_block<true>(temp1, out);
}

void DCT32x32Top(ap_uint<16> xblock, ap_uint<16> yblock, hls::stream<float>& in, hls::stream<float>& out) {
#pragma HLS INLINE off
    for (ap_uint<8> cnty = 0; cnty < yblock; cnty++) {
        for (ap_uint<8> cntx = 0; cntx < xblock; cntx++) {
#pragma HLS DATAFLOW

            float temp0[1024];
#pragma HLS RESOURCE variable = temp0 core = RAM_2P_BRAM
            float temp1[1024];
#pragma HLS RESOURCE variable = temp1 core = RAM_2P_BRAM
            float temp2[1024];
#pragma HLS RESOURCE variable = temp2 core = RAM_2P_BRAM
            float temp3[1024];
#pragma HLS RESOURCE variable = temp3 core = RAM_2P_BRAM

        load:
            for (ap_uint<16> i = 0; i < 1024; i++) {
                temp0[i] = in.read();
            }

            dct32_block<false>(temp0, temp1);
            TransposeBlock32(temp1, temp2);
            dct32_block<true>(temp2, temp3);

        feed:
            for (ap_uint<16> i = 0; i < 1024; i++) {
                out.write(temp3[i]);
            }
        }
    }
}

void DCT2x2_block16(float in[16], float out[16]) {
#pragma HLS INLINE off

    for (ap_uint<8> by = 0; by < 2; by++) {
        for (ap_uint<8> bx = 0; bx < 2; bx++) {
#pragma HLS pipeline

            float a00 = in[((ap_uint<1>)by[0], (ap_uint<1>)0, (ap_uint<1>)bx[0], (ap_uint<1>)0)];
            float a01 = in[((ap_uint<1>)by[0], (ap_uint<1>)0, (ap_uint<1>)bx[0], (ap_uint<1>)1)];
            float a10 = in[((ap_uint<1>)by[0], (ap_uint<1>)1, (ap_uint<1>)bx[0], (ap_uint<1>)0)];
            float a11 = in[((ap_uint<1>)by[0], (ap_uint<1>)1, (ap_uint<1>)bx[0], (ap_uint<1>)1)];

            // std::cout<<"dct: a00="<<a00<<" a01="<<a01<<" a10="<<a10<<"
            // a11="<<a11<<std::endl;

            float t0 = a00 + a01;
            float t1 = a10 + a11;
            float t2 = a00 - a01;
            float t3 = a10 - a11;

            float o00 = t0 + t1;
            float o01 = t0 - t1;
            float o10 = t2 + t3;
            float o11 = t2 - t3;

            out[((ap_uint<1>)by[0], (ap_uint<1>)0, (ap_uint<1>)bx[0], (ap_uint<1>)0)] = o00 / 4;
            out[((ap_uint<1>)by[0], (ap_uint<1>)0, (ap_uint<1>)bx[0], (ap_uint<1>)1)] = o01 / 4;
            out[((ap_uint<1>)by[0], (ap_uint<1>)1, (ap_uint<1>)bx[0], (ap_uint<1>)0)] = o10 / 4;
            out[((ap_uint<1>)by[0], (ap_uint<1>)1, (ap_uint<1>)bx[0], (ap_uint<1>)1)] = o11 / 4;
        }
    }
}

template <bool scale>
void dct4_block16(float in[16], float out[16]) {
#pragma HLS INLINE off

    const float c2_8 = 0.7071067811865475244f; // 0.5 / cos(2 * pi / 8)

    for (ap_uint<8> x = 0; x < 4; x++) {
#pragma HLS DEPENDENCE variable = in inter false
#pragma HLS DEPENDENCE variable = out inter false
#pragma HLS pipeline

        float i0 = in[((ap_uint<2>)0, x(1, 0))];
        float i1 = in[((ap_uint<2>)1, x(1, 0))];
        float i2 = in[((ap_uint<2>)2, x(1, 0))];
        float i3 = in[((ap_uint<2>)3, x(1, 0))];

        float t0 = i0 + i2;
        float i2n = -i2;
        float t1 = i0 + i2n;
        float t2 = i1 + i3;
        float i3n = -i3;
        float t3 = i1 + i3n;
        float t4 = t3 * c2_8;
        float t5 = t2 + t4;
        float t6 = t0 + t5;
        float t7 = t1 + t4;
        float t5n = -t5;
        float t4n = -t4;
        float t8 = t0 + t5n;
        float t9 = t1 + t4n;

        if (scale) {
            out[((ap_uint<2>)0, x(1, 0))] = t6 / 16;
            out[((ap_uint<2>)1, x(1, 0))] = t7 / 16;
            out[((ap_uint<2>)2, x(1, 0))] = t9 / 16;
            out[((ap_uint<2>)3, x(1, 0))] = t8 / 16;
        } else {
            out[((ap_uint<2>)0, x(1, 0))] = t6;
            out[((ap_uint<2>)1, x(1, 0))] = t7;
            out[((ap_uint<2>)2, x(1, 0))] = t9;
            out[((ap_uint<2>)3, x(1, 0))] = t8;
        }
    }
}

void DCT4x4_block16(float in[16], float out[16]) {
#pragma HLS INLINE off
#pragma HLS dataflow

    float temp0[16];
#pragma HLS RESOURCE variable = temp0 core = RAM_2P_BRAM
    float temp1[16];
#pragma HLS RESOURCE variable = temp1 core = RAM_2P_BRAM

    dct4_block16<false>(in, temp0);
    TransposeN<4>(temp0, temp1);
    dct4_block16<false>(temp1, out);
}

void idct4_block16(float from[16], float to[16]) {
    const float c2_8 = 0.707106769;

LOOP_IDCT4X4:
    for (ap_uint<8> i = 0; i < 4; i++) {
#pragma HLS DEPENDENCE variable = from inter false
#pragma HLS DEPENDENCE variable = to inter false
#pragma HLS pipeline

        float i0 = from[((ap_uint<2>)0, i(1, 0))];
        float i1 = from[((ap_uint<2>)1, i(1, 0))];
        float i2 = from[((ap_uint<2>)2, i(1, 0))];
        float i3 = from[((ap_uint<2>)3, i(1, 0))];

        float t0 = i0 + i2;
        float t1 = i0 - i2;
        float t2 = i1 + i3;
        float t3 = i1 - i3;

        float t4 = t3 * c2_8;
        float t5 = t2 + t4;

        float t6 = t0 + t5;
        float t7 = t1 + t4;
        float t8 = t0 - t5;
        float t9 = t1 - t4;

        to[((ap_uint<2>)0, i(1, 0))] = t6;
        to[((ap_uint<2>)1, i(1, 0))] = t7;
        to[((ap_uint<2>)2, i(1, 0))] = t9;
        to[((ap_uint<2>)3, i(1, 0))] = t8;
    }
}

void IDCT2x2_block16(float from[16], float to[16]) {
    float dest[4];
LOOP_IDCT2X2:
    for (ap_uint<8> by = 0; by < 2; by++) {
        for (ap_uint<8> bx = 0; bx < 2; bx++) {
#pragma HLS pipeline

            float a00 = from[((ap_uint<1>)by[0], (ap_uint<1>)0, (ap_uint<1>)bx[0], (ap_uint<1>)0)];
            float a01 = from[((ap_uint<1>)by[0], (ap_uint<1>)0, (ap_uint<1>)bx[0], (ap_uint<1>)1)];
            float a10 = from[((ap_uint<1>)by[0], (ap_uint<1>)1, (ap_uint<1>)bx[0], (ap_uint<1>)0)];
            float a11 = from[((ap_uint<1>)by[0], (ap_uint<1>)1, (ap_uint<1>)bx[0], (ap_uint<1>)1)];

            // std::cout<<"idct: a00="<<a00<<" a01="<<a01<<" a10="<<a10<<"
            // a11="<<a11<<std::endl;

            float t0 = a00 + a01;
            float t1 = a00 - a01;
            float t2 = a10 + a11;
            float t3 = a10 - a11;

            dest[0] = t0 + t2;
            dest[1] = t0 - t2;
            dest[2] = t1 + t3;
            dest[3] = t1 - t3;

            to[((ap_uint<1>)by[0], (ap_uint<1>)0, (ap_uint<1>)bx[0], (ap_uint<1>)0)] = dest[0];
            to[((ap_uint<1>)by[0], (ap_uint<1>)0, (ap_uint<1>)bx[0], (ap_uint<1>)1)] = dest[1];
            to[((ap_uint<1>)by[0], (ap_uint<1>)1, (ap_uint<1>)bx[0], (ap_uint<1>)0)] = dest[2];
            to[((ap_uint<1>)by[0], (ap_uint<1>)1, (ap_uint<1>)bx[0], (ap_uint<1>)1)] = dest[3];
        }
    }
}

void IDCT4x4_block16(float from[16], float to[16]) {
#pragma HLS DATAFLOW

    float from0[16];
    float to0[16];

    idct4_block16(from, from0);
    TransposeN<4>(from0, to0);
    idct4_block16(to0, to);
}

#endif
