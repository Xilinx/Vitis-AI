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

#include "pikEnc/XAccPIKKernel2.hpp"
#include "pikEnc/dct.hpp"
#include "pikEnc/dequant.hpp"

#include "xf_utils_hw/axi_to_multi_stream.hpp"
#include "xf_utils_hw/axi_to_stream.hpp"
#include "xf_utils_hw/stream_to_axi.hpp"

#include <iomanip>
#include <iostream>

#ifndef __SYNTHESIS__
#define DEBUG true
#ifdef DEBUG
//#define DEBUG_ACSTRATEGY true
//#define DEBUG_DCT true
//#define DEBUG_IDCT true
//#define DEBUG_QUANTIZER true
//#define DEBUG_FRAMECACHE true
//#define DEBUG_COEFFS true
//#define DEBUG_REORDER true
//#define DEBUG_COEFFS_ORDER true
//#define DEBUG_SORT true
#endif
#endif

#define USE_HLS_SQRT true

float power(float a, float b) {
#pragma HLS PIPELINE

    return hls::pow(a, b);
}

#ifndef USE_HLS_SQRT

float Inv(float x) {
#pragma HLS INLINE

    return 1.0 / x;
}

float InvSqrt(float x) {
#pragma HLS INLINE

    float xhalf = 0.5f * x;
    int i = *(int*)&x;              // get bits for floating VALUE
    i = 0x5f3759df - (i >> 1);      // gives initial guess y0
    x = *(float*)&i;                // convert bits BACK to float
    x = x * (1.5f - xhalf * x * x); // Newton step, repeating increases accuracy
    x = x * (1.5f - xhalf * x * x); // Newton step, repeating increases accuracy
    return x;
}

float Sqrt2(float x) {
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS INLINE off

    float temp;
    temp = InvSqrt(x);
    return Inv(temp);
}

float Sqrt4(float x) {
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS INLINE off

    float temp = InvSqrt(x);
    return InvSqrt(temp);
}

float Sqrt8(float x) {
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS INLINE off

    float temp = Sqrt2(x);
    return Sqrt4(temp);
}

#else

float Sqrt2(float x) {
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS INLINE off

    return hls::sqrt(x);
}

float Sqrt4(float x) {
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS INLINE off

    float temp = hls::sqrt(x);
    return hls::sqrt(temp);
}

float Sqrt8(float x) {
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS INLINE off

    float temp0 = hls::sqrt(x);
    float temp1 = hls::sqrt(temp0);
    return hls::sqrt(temp1);
}

#endif

void get4x4block(ap_uint<8> bx, ap_uint<8> by, float src32x32[3][1024], hls::stream<float> src4x4[3][16]) {
#pragma HLS INLINE off

#ifdef DEBUG_ACSTRATEGY
    std::cout << "get4x4block:" << std::endl;
#endif

    ap_uint<10> addr_i;
    ap_uint<4> addr_o;

    for (ap_uint<8> block = 0; block < 4; block++)
        for (ap_uint<8> dy = 0; dy < 4; dy++) {
            for (ap_uint<8> dx = 0; dx < 4; dx++) {
#pragma HLS pipeline II = 1
                for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL

                    addr_i(9, 8) = by(1, 0);
                    addr_i[7] = block[1];
                    addr_i(6, 5) = dy(1, 0);
                    addr_i(4, 3) = bx(1, 0);
                    addr_i[2] = block[0];
                    addr_i(1, 0) = dx(1, 0);

                    addr_o(3, 2) = dy(1, 0);
                    addr_o(1, 0) = dx(1, 0);

                    src4x4[c][addr_o].write(src32x32[c][addr_i]);

#ifdef DEBUG_ACSTRATEGY
                    std::cout << "id=" << addr_i << " src=" << src32x32[c][addr_i] << std::endl;
#endif
                }
            }
        }
}

void DCT_collection4x4(hls::stream<float> src_in[3][16], const float kColorWeights[3], hls::stream<float>& blockval) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<float> sum_stream[3];
#pragma HLS stream variable = sum_stream depth = 8
#pragma HLS BIND_STORAGE variable = sum_stream type = fifo impl = srl

loop_sum:
    for (ap_uint<8> ix = 0; ix < 4; ix++) {
        for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS pipeline

            float sum = 0;
            float src[16];
            for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
                src[i] = src_in[c][i].read();
            }

            // x
            sum += hls::fabs(src[0] - src[1]);
            sum += hls::fabs(src[1] - src[2]);
            sum += hls::fabs(src[2] - src[3]);

            sum += hls::fabs(src[4] - src[5]);
            sum += hls::fabs(src[5] - src[6]);
            sum += hls::fabs(src[6] - src[7]);

            sum += hls::fabs(src[8] - src[9]);
            sum += hls::fabs(src[9] - src[10]);
            sum += hls::fabs(src[10] - src[11]);

            sum += hls::fabs(src[12] - src[13]);
            sum += hls::fabs(src[13] - src[14]);
            sum += hls::fabs(src[14] - src[15]);

            // y
            sum += hls::fabs(src[0] - src[4]);
            sum += hls::fabs(src[1] - src[5]);
            sum += hls::fabs(src[2] - src[6]);
            sum += hls::fabs(src[3] - src[7]);

            sum += hls::fabs(src[4] - src[8]);
            sum += hls::fabs(src[5] - src[9]);
            sum += hls::fabs(src[6] - src[10]);
            sum += hls::fabs(src[7] - src[11]);

            sum += hls::fabs(src[8] - src[12]);
            sum += hls::fabs(src[9] - src[13]);
            sum += hls::fabs(src[10] - src[14]);
            sum += hls::fabs(src[11] - src[15]);

#ifdef DEBUG_ACSTRATEGY
            std::cout << "DCT_collection4x4:" << std::endl;
            for (ap_uint<8> i = 0; i < 16; i++) {
                std::cout << "c=" << c << " id=" << i << " src=" << src[i] << std::endl;
            }
            std::cout << "sum=" << sum << std::endl;
#endif
            sum_stream[c].write(sum);
        }
    }

loop_total_sum:
    for (ap_uint<8> ix = 0; ix < 4; ix++) {
#pragma HLS pipeline

        float sum[3];
        for (ap_uint<8> c = 0; c < 3; c++) {
            sum[c] = sum_stream[c].read();
        }
        float total_sum = kColorWeights[0] * sum[0] + kColorWeights[1] * sum[1] + kColorWeights[2] * sum[2];
        blockval.write(total_sum);
    }
}

void DCT_normalize(hls::stream<float>& blockval, const float constant[3], hls::stream<bool>& result) {
#pragma HLS INLINE off

    float power2 = 0;
    float power4 = 0;
    float power8 = 0;

    float norm2;
    float norm4;
    float norm8;
    for (ap_uint<8> ix = 0; ix < 4; ix++) {
#pragma HLS pipeline

        float v = blockval.read();
        float v2, v4, v8;

        v2 = v * v;
        power2 += v2;
        v4 = v2 * v2;
        power4 += v4;
        v8 = v4 * v4;
        power8 += v8;
    }

    norm2 = Sqrt2(power2 * 0.25);
    norm4 = Sqrt4(power4 * 0.25);
    norm8 = Sqrt8(power8 * 0.25);
    norm2 += 0.03;

    float loss = constant[0] * norm8;
    loss += constant[1] * norm4;
    float loss_limit = constant[2] * norm2;

    bool result_tmp;
    if (loss >= loss_limit) {
        result_tmp = true;
    } else {
        result_tmp = false;
    }
    result.write(result_tmp);
}

void DCT_collection32x32(ap_uint<16> xblock,
                         ap_uint<16> yblock,
                         hls::stream<float> src[3],
                         const float kColorWeights[3],
                         const float constant[3],
                         hls::stream<bool>& result) {
#pragma HLS INLINE off

#ifdef DEBUG_ACSTRATEGY
    std::cout << "kColorWeights:" << std::endl;
    for (ap_uint<8> i = 0; i < 3; i++) {
        std::cout << "id=" << i << " " << kColorWeights[i] << std::endl;
    }

    std::cout << "constant:" << std::endl;
    for (ap_uint<8> i = 0; i < 3; i++) {
        std::cout << "id=" << i << " " << constant[i] << std::endl;
    }
#endif

// Process y
loop_y:
    for (ap_uint<16> ycnt = 0; ycnt < yblock; ycnt++) {
    // Process x
    loop_x:
        for (ap_uint<16> xcnt = 0; xcnt < xblock; xcnt++) {
#pragma HLS DATAFLOW
            // load src_buffer 32x32
            float src_buffer[3][1024];
#pragma HLS ARRAY_PARTITION variable = src_buffer complete dim = 1

            for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1
                for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL
                    src_buffer[c][i] = src[c].read();
                }
            }

            // divide 32x32 to 16 block of 8x8
            for (ap_uint<8> by = 0; by < 4; by++) {
                for (ap_uint<8> bx = 0; bx < 4; bx++) {
#pragma HLS DATAFLOW

                    hls::stream<float> src4x4[3][16];
#pragma HLS ARRAY_PARTITION variable = src4x4 complete
#pragma HLS stream variable = src4x4 depth = 8
#pragma HLS BIND_STORAGE variable = src4x4 type = fifo impl = srl
                    hls::stream<float> blockval;
#pragma HLS stream variable = blockval depth = 8
#pragma HLS BIND_STORAGE variable = blockval type = fifo impl = srl

                    // divide 8x8 to 4 block of 4x4
                    get4x4block(bx, by, src_buffer, src4x4);
                    DCT_collection4x4(src4x4, kColorWeights, blockval);
                    DCT_normalize(blockval, constant, result);
                }
            }
        }
    }
}

void min_max_entropy(float src[3][1024],
                     hls::stream<float> min_ext16_strm[3],
                     hls::stream<float> max_ext16_strm[3],
                     hls::stream<float> min_ext32_strm[3],
                     hls::stream<float> max_ext32_strm[3]) {
#pragma HLS INLINE off

    float min_ext32[3];
#pragma HLS ARRAY_PARTITION variable = min_ext32 complete
    float max_ext32[3];
#pragma HLS ARRAY_PARTITION variable = max_ext32 complete

    float min_ext16[3];
#pragma HLS ARRAY_PARTITION variable = min_ext16 complete
    float max_ext16[3];
#pragma HLS ARRAY_PARTITION variable = max_ext16 complete

loop_min_mix:
    for (ap_uint<8> by = 0; by < 2; by++) {
        for (ap_uint<8> bx = 0; bx < 2; bx++) {
            for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL
                float min8x8;
                float max8x8;
                for (ap_uint<8> iy = 0; iy < 2; iy++) {
                    for (ap_uint<8> ix = 0; ix < 2; ix++) {
                        for (ap_uint<8> dy = 0; dy < 8; ++dy) {
                            for (ap_uint<8> dx = 0; dx < 8; ++dx) {
#pragma HLS pipeline

                                // min-max 8x8
                                ap_uint<10> shift;
                                shift[9] = by[0];
                                shift[8] = iy[0];
                                shift(7, 5) = dy(2, 0);
                                shift[4] = bx[0];
                                shift[3] = ix[0];
                                shift(2, 0) = dx(2, 0);

                                if (dy == 0 && dx == 0) {
                                    min8x8 = 1e30;
                                    max8x8 = -1e30;

                                    if (iy == 0 && ix == 0 && dy == 0 && dx == 0) {
                                        min_ext16[c] = 1e30;
                                        max_ext16[c] = -1e30;

                                        if (by == 0 && bx == 0) {
                                            min_ext32[c] = 1e30;
                                            max_ext32[c] = -1e30;
                                        }
                                    }
                                }

                                float v = src[c][shift];
                                if (v < min8x8) min8x8 = v;
                                if (v > max8x8) max8x8 = v;

                                if (dy == 7 && dx == 7) {
                                    float ext = max8x8 - min8x8;
                                    if (ext < min_ext16[c]) min_ext16[c] = ext;
                                    if (ext > max_ext16[c]) max_ext16[c] = ext;

                                    if (ext < min_ext32[c]) min_ext32[c] = ext;
                                    if (ext > max_ext32[c]) max_ext32[c] = ext;

                                    if (iy == 1 && ix == 1) {
                                        min_ext16_strm[c].write(min_ext16[c]);
                                        max_ext16_strm[c].write(max_ext16[c]);
                                        if (by == 1 && bx == 1) {
                                            min_ext32_strm[c].write(min_ext32[c]);
                                            max_ext32_strm[c].write(max_ext32[c]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void entropy_accum0(float power0[3][1024],
                    float power1[3][1024],
                    hls::stream<float> entropy16_strm[3],
                    hls::stream<float> entropy32_strm[3]) {
#pragma HLS INLINE off

    static const float kColorWeights[3] = {
        0.65285453568125873, 2.4740163893371157, 2.0140216656143393,
    };

    const float kDiff16 = 0.2494383590606063;
    const float kDiff32 = 0.9539527585329598;

    const float kPow = 0.99263297216052859;
    const float kPow2 = 0.018823021573462634;
    const float kExtremityWeight = 7.77;

    float entropy32[3];
#pragma HLS ARRAY_PARTITION variable = entropy32 complete
    float entropy16[3];
#pragma HLS ARRAY_PARTITION variable = entropy16 complete

    for (ap_uint<8> by = 0; by < 2; by++) {
        for (ap_uint<8> bx = 0; bx < 2; bx++) {
            for (ap_uint<8> iy = 0; iy < 2; iy++) {
                for (ap_uint<8> ix = 0; ix < 2; ix++) {
                    for (ap_uint<8> dy = 0; dy < 8; ++dy) {
                        for (ap_uint<8> dx = 0; dx < 8; ++dx) {
#pragma HLS pipeline
                            for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL

                                // dequant 8x8, skip the dc values at 0 and 64.
                                if (dx == 0 && dy == 0) {
                                    if (ix == 0 && iy == 0) {
                                        entropy16[c] = 0;
                                        if (by == 0 && bx == 0) {
                                            entropy32[c] = 0;
                                        }
                                    }
                                } else {
                                    ap_uint<10> shift1;

                                    shift1[9] = by[0];
                                    shift1[8] = iy[0];
                                    shift1(7, 5) = dy(2, 0);
                                    shift1[4] = bx[0];
                                    shift1[3] = ix[0];
                                    shift1(2, 0) = dx(2, 0);

                                    float p0 = power0[c][shift1];
                                    float p1 = power1[c][shift1];
                                    entropy16[c] += 1 + kDiff16 - p0 - kDiff16 * p1;
                                    entropy32[c] += 1 + kDiff32 - p0 - kDiff32 * p1;

                                    if (dy == 7 && dx == 7 && iy == 1 && ix == 1) {
                                        entropy16_strm[c].write(entropy16[c]);
                                        if (by == 1 && bx == 1) {
                                            entropy32_strm[c].write(entropy32[c]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void entropy_accum1(hls::stream<float> entropy16_strm[3],
                    hls::stream<float> max_ext16_strm[3],
                    hls::stream<float> min_ext16_strm[3],
                    hls::stream<float>& reference_entropy16) {
#pragma HLS INLINE off

    static const float kColorWeights[3] = {
        0.65285453568125873, 2.4740163893371157, 2.0140216656143393,
    };
    const float kExtremityWeight = 7.77;

    float dct16x16_reference_entropy;
    for (ap_uint<8> by = 0; by < 2; by++) {
        for (ap_uint<8> bx = 0; bx < 2; bx++) {
            for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS pipeline

                if (c == 0) {
                    dct16x16_reference_entropy = 0;
                }

                float entropy16_tmp = entropy16_strm[c].read();
                float max_ext16_tmp = max_ext16_strm[c].read();
                float min_ext16_tmp = min_ext16_strm[c].read();
                entropy16_tmp -= kExtremityWeight * (max_ext16_tmp - min_ext16_tmp);
                dct16x16_reference_entropy += kColorWeights[c] * entropy16_tmp;

                if (c == 2) {
                    reference_entropy16.write(dct16x16_reference_entropy);
                }
            }
        }
    }
}

void entropy_accum2(hls::stream<float> entropy32_strm[3],
                    hls::stream<float> max_ext32_strm[3],
                    hls::stream<float> min_ext32_strm[3],
                    hls::stream<float>& reference_entropy32) {
#pragma HLS INLINE off

    static const float kColorWeights[3] = {
        0.65285453568125873, 2.4740163893371157, 2.0140216656143393,
    };
    const float kExtremityWeight = 7.77;

    float dct32x32_reference_entropy = 0;
    for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS pipeline

        float entropy32_tmp = entropy32_strm[c].read();
        float max_ext32_tmp = max_ext32_strm[c].read();
        float min_ext32_tmp = min_ext32_strm[c].read();
        entropy32_tmp -= kExtremityWeight * (max_ext32_tmp - min_ext32_tmp);
        dct32x32_reference_entropy += kColorWeights[c] * entropy32_tmp;
    }
    reference_entropy32.write(dct32x32_reference_entropy);
}

void load_dequant_src(hls::stream<float>& quant_field_strm,
                      hls::stream<float> src_strm[3],
                      hls::stream<float> coeffs_strm[3],
                      float quant_field[16],
                      float src[3][1024],
                      float coeffs[3][1024]) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

load_src:
    for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1
        for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL
            src[c][i] = src_strm[c].read();
        }
    }

load_coeffs:
    for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1
        for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL
            coeffs[c][i] = coeffs_strm[c].read();
        }
    }

load_qf:
    for (ap_uint<16> i = 0; i < 16; i++) {
#pragma HLS PIPELINE II = 1
        quant_field[i] = quant_field_strm.read();
    }
}

void dequant(float discretization_factor,
             const float inv_dequant[3][64],
             float quant_field[16],
             float src[3][1024],
             float coeffs[3][1024],
             float power0[3][1024],
             float power1[3][1024]) {
#pragma HLS INLINE off

    // The quantized symbol distribution contracts with the increasing
    // butteraugli_target.
    static const float kColorWeights[3] = {
        0.65285453568125873, 2.4740163893371157, 2.0140216656143393,
    };

    const float kDiff16 = 0.2494383590606063;
    const float kDiff32 = 0.9539527585329598;

    const float kPow = 0.99263297216052859;
    const float kPow2 = 0.018823021573462634;
    const float kExtremityWeight = 7.77;

loop_dequant:
    for (ap_uint<8> by = 0; by < 2; by++) {
        for (ap_uint<8> bx = 0; bx < 2; bx++) {
            for (ap_uint<8> iy = 0; iy < 2; iy++) {
                for (ap_uint<8> ix = 0; ix < 2; ix++) {
                    for (ap_uint<8> dy = 0; dy < 8; ++dy) {
                        for (ap_uint<8> dx = 0; dx < 8; ++dx) {
#pragma HLS pipeline
                            for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL
                                // dequant 8x8, skip the dc values at 0 and 64.
                                ap_uint<10> shift1;
                                ap_uint<6> shift2;
                                ap_uint<4> shift3;

                                shift1[9] = by[0];
                                shift1[8] = iy[0];
                                shift1(7, 5) = dy(2, 0);
                                shift1[4] = bx[0];
                                shift1[3] = ix[0];
                                shift1(2, 0) = dx(2, 0);

                                shift2(5, 3) = dy(2, 0);
                                shift2(2, 0) = dx(2, 0);

                                shift3[3] = by[0];
                                shift3[2] = iy[0];
                                shift3[1] = bx[0];
                                shift3[0] = ix[0];

                                float val = coeffs[c][shift1] * inv_dequant[c][shift2];
                                val *= quant_field[shift3];
                                float v = std::fabs(val) * discretization_factor;
                                power0[c][shift1] = std::pow(kPow, v);
                                power1[c][shift1] = std::pow(kPow2, v);
                            }
                        }
                    }
                }
            }
        }
    }
}

void reference_entropy(ap_uint<16> xblock,
                       ap_uint<16> yblock,
                       float discretization_factor,
                       hls::stream<float>& quant_field_strm,
                       const float inv_dequant[3][64],
                       hls::stream<float> src_strm[3],
                       hls::stream<float> coeffs_strm[3],
                       hls::stream<float>& reference_entropy16,
                       hls::stream<float>& reference_entropy32) {
#pragma HLS INLINE off

// Process y
loop_y:
    for (ap_uint<16> ycnt = 0; ycnt < yblock; ycnt++) {
    // Process x
    loop_x:
        for (ap_uint<16> xcnt = 0; xcnt < xblock; xcnt++) {
#pragma HLS DATAFLOW

            float quant_field[16];
#pragma HLS ARRAY_PARTITION variable = quant_field complete
            float src[3][1024];
#pragma HLS ARRAY_PARTITION variable = src complete dim = 1
#pragma HLS BIND_STORAGE variable = src type = ram_2p impl = bram
            float coeffs[3][1024];
#pragma HLS ARRAY_PARTITION variable = coeffs complete dim = 1
#pragma HLS BIND_STORAGE variable = coeffs type = ram_2p impl = bram

            float power0[3][1024];
#pragma HLS ARRAY_PARTITION variable = power0 complete dim = 1
#pragma HLS BIND_STORAGE variable = power0 type = ram_2p impl = bram
            float power1[3][1024];
#pragma HLS ARRAY_PARTITION variable = power1 complete dim = 1
#pragma HLS BIND_STORAGE variable = power1 type = ram_2p impl = bram

            hls::stream<float> min_ext32_strm[3];
#pragma HLS stream variable = min_ext32_strm depth = 512
#pragma HLS BIND_STORAGE variable = min_ext32_strm type = fifo impl = bram
            hls::stream<float> max_ext32_strm[3];
#pragma HLS stream variable = max_ext32_strm depth = 512
#pragma HLS BIND_STORAGE variable = max_ext32_strm type = fifo impl = bram
            hls::stream<float> entropy32_strm[3];
#pragma HLS stream variable = entropy32_strm depth = 8
#pragma HLS BIND_STORAGE variable = entropy32_strm type = fifo impl = srl

            hls::stream<float> min_ext16_strm[3];
#pragma HLS stream variable = min_ext16_strm depth = 512
#pragma HLS BIND_STORAGE variable = min_ext16_strm type = fifo impl = bram
            hls::stream<float> max_ext16_strm[3];
#pragma HLS stream variable = max_ext16_strm depth = 512
#pragma HLS BIND_STORAGE variable = max_ext16_strm type = fifo impl = bram
            hls::stream<float> entropy16_strm[3];
#pragma HLS stream variable = entropy16_strm depth = 8
#pragma HLS BIND_STORAGE variable = entropy16_strm type = fifo impl = srl

            load_dequant_src(quant_field_strm, src_strm, coeffs_strm, quant_field, src, coeffs);

            min_max_entropy(src, min_ext16_strm, max_ext16_strm, min_ext32_strm, max_ext32_strm);

            dequant(discretization_factor, inv_dequant, quant_field, src, coeffs, power0, power1);

            entropy_accum0(power0, power1, entropy16_strm, entropy32_strm);

            entropy_accum1(entropy16_strm, max_ext16_strm, min_ext16_strm, reference_entropy16);

            entropy_accum2(entropy32_strm, max_ext32_strm, min_ext32_strm, reference_entropy32);
        }
    }
}

void get_max_quant(float kMulInho16,
                   float kMulInho32,
                   float quant_field[16],
                   hls::stream<float>& quant_field_strm,
                   hls::stream<float>& inhomogeneity16,
                   hls::stream<float>& quant16,
                   hls::stream<float>& inhomogeneity32,
                   hls::stream<float>& quant32) {
#pragma HLS INLINE off

    float quant_inhomogeneity32;
    float max_quant32;
    float quant_inhomogeneity16;
    float max_quant16;

block4x4:
    for (ap_uint<8> by = 0; by < 2; by++) {
        for (ap_uint<8> bx = 0; bx < 2; bx++) {
            for (ap_uint<8> dy = 0; dy < 2; ++dy) {
                for (ap_uint<8> dx = 0; dx < 2; ++dx) {
#pragma HLS pipeline

                    ap_uint<4> shift;
                    shift[3] = by[0];
                    shift[2] = dy[0];
                    shift[1] = bx[0];
                    shift[0] = dx[0];

                    if ((dy == 0) && (dx == 0)) {
                        quant_inhomogeneity16 = 0;
                        max_quant16 = -1e30;
                        if ((by == 0) && (bx == 0)) {
                            quant_inhomogeneity32 = 0;
                            max_quant32 = -1e30;
                        }
                    }

                    float temp = quant_field[shift];
                    max_quant16 = hls::max(max_quant16, temp);
                    quant_inhomogeneity16 -= temp;

                    max_quant32 = hls::max(max_quant32, temp);
                    quant_inhomogeneity32 -= temp;

                    if ((dy == 1) && (dx == 1)) {
                        quant_inhomogeneity16 += 4 * max_quant16;
                        inhomogeneity16.write(kMulInho16 * quant_inhomogeneity16);
                        quant16.write(max_quant16);
                        if ((by == 1) && (bx == 1)) {
                            quant_inhomogeneity32 += 16 * max_quant32;
                            inhomogeneity32.write(kMulInho32 * quant_inhomogeneity32);
                            quant32.write(max_quant32);
                        }
                    }
                }
            }
        }
    }
}

void get_quant_inhomogeneity(ap_uint<16> xblock,
                             ap_uint<16> yblock,
                             float kMulInho16,
                             float kMulInho32,
                             hls::stream<float>& quant_field_strm,
                             hls::stream<float>& inhomogeneity16,
                             hls::stream<float>& quant16,
                             hls::stream<float>& inhomogeneity32,
                             hls::stream<float>& quant32) {
#pragma HLS INLINE off

// Process y
loop_y:
    for (ap_uint<16> ycnt = 0; ycnt < yblock; ycnt++) {
    // Process x
    loop_x:
        for (ap_uint<16> xcnt = 0; xcnt < xblock; xcnt++) {
#pragma HLS DATAFLOW

            float quant_field[16];

        load_qf:
            for (ap_uint<16> i = 0; i < 16; i++) {
                quant_field[i] = quant_field_strm.read();
            }

            get_max_quant(kMulInho16, kMulInho32, quant_field, quant_field_strm, inhomogeneity16, quant16,
                          inhomogeneity32, quant32);
        }
    }
}

float compute_entropy16_sub(float discretization_factor,
                            ap_uint<8> by,
                            ap_uint<8> bx,
                            float max_quant,
                            const float inv_dequant16[256],
                            float coeffs16[1024]) {
#pragma HLS INLINE off

    const float kFavor8x8Dct = 0.978192691479985;
    const float kDiff = 0.2494383590606063;
    const float kPow = 0.99263297216052859;
    const float kPow2 = 0.018823021573462634;
    const float kExtremityWeight = 7.77;

    float entropy = 0;
    for (ap_uint<8> dy = 0; dy < 16; dy++) {
        for (ap_uint<8> dx = 0; dx < 16; dx++) {
#pragma HLS pipeline
            if (dy < 2 && dx < 2) {
                // Leave out the lowest frequencies.
            } else {
                ap_uint<10> shift1;
                ap_uint<8> shift2;

                shift1[9] = by[0];
                shift1(8, 5) = dy(3, 0);
                shift1[4] = bx[0];
                shift1(3, 0) = dx(3, 0);

                shift2(7, 4) = dy(3, 0);
                shift2(3, 0) = dx(3, 0);

                float val = coeffs16[shift1] * inv_dequant16[shift2];
                val *= max_quant;
                float v = std::fabs(val) * discretization_factor;
                entropy += 1 + kDiff - std::pow(kPow, v) - kDiff * std::pow(kPow2, v);

#ifdef DEBUG_ACSTRATEGY
                std::cout << "entropy:" << entropy << std::endl;
#endif
            }
#ifdef DEBUG_ACSTRATEGY
            std::cout << "coeff16=" << coeffs16[shift1] << " inv_dequant16=" << inv_dequant16[shift2]
                      << " max_quant=" << max_quant << " discretization_factor=" << discretization_factor << " v=" << v
                      << " entropy=" << entropy << std::endl;
#endif
        }
    }
    return entropy;
}

void compute_entropy16(float discretization_factor,
                       float entropy8[2][2],
                       float quant_inhomogeneity[2][2],
                       float max_quant[2][2],
                       const float inv_dequant16[3][256],
                       float coeffs16[3][1024],
                       hls::stream<bool>& enableDCT16) {
#pragma HLS INLINE off
#pragma HLS allocation function instance = compute_entropy16_sub limit = 3

    const float kColorWeights[3] = {
        0.65285453568125873, 2.4740163893371157, 2.0140216656143393,
    };

    const float kFavor8x8Dct = 0.978192691479985;
    const float kDiff = 0.2494383590606063;
    const float kPow = 0.99263297216052859;
    const float kPow2 = 0.018823021573462634;
    const float kExtremityWeight = 7.77;

    float dct16x16_entropy;
    float dct8x8_entropy;
    float entropy0, entropy1, entropy2;
    bool result;

    for (ap_uint<8> by = 0; by < 2; by++) {
        for (ap_uint<8> bx = 0; bx < 2; bx++) {
            dct8x8_entropy = entropy8[by][bx] + quant_inhomogeneity[by][bx];

            entropy0 =
                compute_entropy16_sub(discretization_factor, by, bx, max_quant[by][bx], inv_dequant16[0], coeffs16[0]);

            entropy1 =
                compute_entropy16_sub(discretization_factor, by, bx, max_quant[by][bx], inv_dequant16[1], coeffs16[1]);

            entropy2 =
                compute_entropy16_sub(discretization_factor, by, bx, max_quant[by][bx], inv_dequant16[2], coeffs16[2]);

            dct16x16_entropy = kColorWeights[0] * entropy0 + kColorWeights[1] * entropy1 + kColorWeights[2] * entropy2;

            if (dct16x16_entropy < kFavor8x8Dct * dct8x8_entropy) {
                result = true;
            } else {
                result = false;
            }
            enableDCT16.write(result);
        }

#ifdef DEBUG_ACSTRATEGY
        std::cout << "entropy:" << entropy << std::endl;
        std::cout << "by=" << by << " bx=" << bx << " dct16x16_entropy=" << dct16x16_entropy
                  << " dct8x8_entropy=" << dct8x8_entropy << std::endl;
#endif
    }
}

void dct_entropy16x16(ap_uint<16> xblock,
                      ap_uint<16> yblock,
                      float discretization_factor,
                      hls::stream<float>& entropy8_strm,
                      hls::stream<float>& quant_inhomogeneity_strm,
                      hls::stream<float>& max_quant_strm,
                      const float inv_dequant16[3][256],
                      hls::stream<float> coeffs16_strm[3],
                      hls::stream<bool>& enableDCT16) {
#pragma HLS INLINE off

    float entropy8[2][2];
#pragma HLS ARRAY_PARTITION variable = entropy8 complete
    float quant_inhomogeneity[2][2];
#pragma HLS ARRAY_PARTITION variable = quant_inhomogeneity complete
    float max_quant[2][2];
#pragma HLS ARRAY_PARTITION variable = max_quant complete
    float coeffs16[3][1024];
#pragma HLS ARRAY_PARTITION variable = coeffs16 complete dim = 1
// Process y
loop_y:
    for (ap_uint<16> ycnt = 0; ycnt < yblock; ycnt++) {
    // Process x
    loop_x:
        for (ap_uint<16> xcnt = 0; xcnt < xblock; xcnt++) {
#pragma HLS DATAFLOW

        load_ctl:
            for (ap_uint<8> by = 0; by < 2; by++) {
                for (ap_uint<8> bx = 0; bx < 2; bx++) {
#pragma HLS PIPELINE II = 1
                    entropy8[by][bx] = entropy8_strm.read();
                    quant_inhomogeneity[by][bx] = quant_inhomogeneity_strm.read();
                    max_quant[by][bx] = max_quant_strm.read();
                }
            }

        load_coeffs:
            for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1
                for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL
                    coeffs16[c][i] = coeffs16_strm[c].read();
                }
            }

            compute_entropy16(discretization_factor, entropy8, quant_inhomogeneity, max_quant, inv_dequant16, coeffs16,
                              enableDCT16);
        }
    }
}

float compute_entropy32_sub(float discretization_factor,
                            float max_quant,
                            const float inv_dequant32[1024],
                            float coeffs32[1024]) {
#pragma HLS INLINE off

    const float kDiff = 0.9539527585329598;
    const float kPow = 0.99263297216052859;
    const float kPow2 = 0.018823021573462634;
    const float kExtremityWeight = 7.77;

    float entropy = 0;

compute_entropy32_sub:
    for (ap_uint<8> dy = 0; dy < 32; dy++) {
        for (ap_uint<8> dx = 0; dx < 32; dx++) {
#pragma HLS PIPELINE
            if (dy < 4 && dx < 4) {
                // Leave out the lowest frequencies.
            } else {
                ap_uint<10> shift;
                shift(9, 5) = dy(4, 0);
                shift(4, 0) = dx(4, 0);

                float val = coeffs32[shift] * inv_dequant32[shift];
                val *= max_quant;
                float v = std::fabs(val) * discretization_factor;
                entropy += 1 + kDiff - std::pow(kPow, v) - kDiff * std::pow(kPow2, v);
            }
        }
    }
    return entropy;
}

void compute_entropy32(float discretization_factor,
                       float butteraugli_target,
                       hls::stream<float>& entropy8_strm,
                       hls::stream<float>& quant_inhomogeneity_strm,
                       hls::stream<float>& max_quant_strm,
                       const float inv_dequant32[3][1024],
                       float coeffs32[3][1024],
                       hls::stream<bool>& enableDCT32) {
#pragma HLS INLINE off
#pragma HLS allocation function instance = compute_entropy32_sub limit = 3

    static const float kColorWeights[3] = {
        0.65285453568125873, 2.4740163893371157, 2.0140216656143393,
    };

    float kFavor8x8Dct = 0.74742417168628905;
    if (butteraugli_target >= 6.0) {
        kFavor8x8Dct = 0.737101360945845;
    }

    const float kDiff = 0.9539527585329598;
    const float kPow = 0.99263297216052859;
    const float kPow2 = 0.018823021573462634;
    const float kExtremityWeight = 7.77;

    float entropy8 = entropy8_strm.read();
    float quant_inhomogeneity = quant_inhomogeneity_strm.read();
    float max_quant = max_quant_strm.read();
    float dct8x8_entropy = entropy8 + quant_inhomogeneity;

    float entropy0, entropy1, entropy2;
    entropy0 = compute_entropy32_sub(discretization_factor, max_quant, inv_dequant32[0], coeffs32[0]);
    entropy1 = compute_entropy32_sub(discretization_factor, max_quant, inv_dequant32[1], coeffs32[1]);
    entropy2 = compute_entropy32_sub(discretization_factor, max_quant, inv_dequant32[2], coeffs32[2]);

    float dct32x32_entropy = kColorWeights[0] * entropy0 + kColorWeights[1] * entropy1 + kColorWeights[2] * entropy2;

    bool result;
    if (dct32x32_entropy < kFavor8x8Dct * dct8x8_entropy) {
        result = true;
    } else {
        result = false;
    }
    enableDCT32.write(result);
}

void dct_entropy32x32(ap_uint<16> xblock,
                      ap_uint<16> yblock,
                      float discretization_factor,
                      float butteraugli_target,
                      hls::stream<float>& entropy8_strm,
                      hls::stream<float>& quant_inhomogeneity_strm,
                      hls::stream<float>& max_quant_strm,
                      const float inv_dequant32[3][1024],
                      hls::stream<float> coeffs32_strm[3],
                      hls::stream<bool>& enableDCT32) {
#pragma HLS INLINE off

// Process y
loop_y:
    for (ap_uint<16> ycnt = 0; ycnt < yblock; ycnt++) {
    // Process x
    loop_x:
        for (ap_uint<16> xcnt = 0; xcnt < xblock; xcnt++) {
#pragma HLS DATAFLOW

            float coeffs32[3][1024];
#pragma HLS ARRAY_PARTITION variable = coeffs32 complete dim = 1

        load_coeffs:
            for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1
                for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL
                    coeffs32[c][i] = coeffs32_strm[c].read();
                }
            }

            compute_entropy32(discretization_factor, butteraugli_target, entropy8_strm, quant_inhomogeneity_strm,
                              max_quant_strm, inv_dequant32, coeffs32, enableDCT32);
        }
    }
}

void DCTWrapper(ap_uint<16> xblock,
                ap_uint<16> yblock,
                hls::stream<float> src[3],
                hls::stream<float> coeffs4_tmp[3],
                hls::stream<float> coeffs8_tmp0[3],
                hls::stream<float> coeffs16_tmp0[3],
                hls::stream<float> coeffs32_tmp0[3],
                hls::stream<float> coeffs8_tmp1[3],
                hls::stream<float> coeffs16_tmp1[3],
                hls::stream<float> coeffs32_tmp1[3]) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<float> src0[3];
#pragma HLS stream variable = src0 depth = 8
#pragma HLS BIND_STORAGE variable = src0 type = fifo impl = srl
    hls::stream<float> src1[3];
#pragma HLS stream variable = src1 depth = 8
#pragma HLS BIND_STORAGE variable = src1 type = fifo impl = srl
    hls::stream<float> src2[3];
#pragma HLS stream variable = src2 depth = 8
#pragma HLS BIND_STORAGE variable = src2 type = fifo impl = srl
    hls::stream<float> src3[3];
#pragma HLS stream variable = src3 depth = 8
#pragma HLS BIND_STORAGE variable = src3 type = fifo impl = srl

    hls::stream<float> coeffs8[3];
#pragma HLS stream variable = src1 depth = 8
#pragma HLS BIND_STORAGE variable = src1 type = fifo impl = srl
    hls::stream<float> coeffs16[3];
#pragma HLS stream variable = src2 depth = 8
#pragma HLS BIND_STORAGE variable = src2 type = fifo impl = srl
    hls::stream<float> coeffs32[3];
#pragma HLS stream variable = src3 depth = 8
#pragma HLS BIND_STORAGE variable = src3 type = fifo impl = srl

    // Process y
    for (ap_uint<16> ycnt = 0; ycnt < yblock; ycnt++) {
        // Process x
        for (ap_uint<16> xcnt = 0; xcnt < xblock; xcnt++) {
        duplicate_src:
            for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1
                for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL
                    float tmp = src[c].read();

                    src0[c].write(tmp);
                    src1[c].write(tmp);
                    src2[c].write(tmp);
                    src3[c].write(tmp);
                }
            }
        }
    }

    for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL
        DCT4x4Top(xblock, yblock, src0[c], coeffs4_tmp[c]);
    }

    for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL
        DCT8x8Top(xblock, yblock, src1[c], coeffs8[c]);
    }

    for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL
        DCT16x16Top(xblock, yblock, src2[c], coeffs16[c]);
    }

    for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL
        DCT32x32Top(xblock, yblock, src3[c], coeffs32[c]);
    }

    // Process y
    for (ap_uint<16> ycnt = 0; ycnt < yblock; ycnt++) {
        // Process x
        for (ap_uint<16> xcnt = 0; xcnt < xblock; xcnt++) {
#pragma HLS DATAFLOW

        duplicate_coeffs8:
            for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1
                for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL
                    float tmp1 = coeffs8[c].read();
                    coeffs8_tmp0[c].write(tmp1);
                    coeffs8_tmp1[c].write(tmp1);
                }
            }

        duplicate_coeffs16:
            for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1
                for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL
                    float tmp2 = coeffs16[c].read();
                    coeffs16_tmp0[c].write(tmp2);
                    coeffs16_tmp1[c].write(tmp2);
                }
            }

        duplicate_coeffs32:
            for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1
                for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL
                    float tmp3 = coeffs32[c].read();
                    coeffs32_tmp0[c].write(tmp3);
                    coeffs32_tmp1[c].write(tmp3);
                }
            }
        }
    }
}

void LoadAcStrategySrc(ap_uint<16> xblock,
                       ap_uint<16> yblock,
                       hls::stream<ap_uint<32> > src[3],
                       hls::stream<float> src0[3],
                       hls::stream<float> src1[3],
                       hls::stream<float> src2[3],
                       hls::stream<float> src3[3]) {
#pragma HLS INLINE off

// Process y
loop_y:
    for (ap_uint<16> ycnt = 0; ycnt < yblock; ycnt++) {
    // Process x
    loop_x:
        for (ap_uint<16> xcnt = 0; xcnt < xblock; xcnt++) {
        // load buffer of 32x32 region
        load32x32_region:
            for (ap_uint<8> y = 0; y < 32; y++) {
                for (ap_uint<8> x = 0; x < 32; x++) {
#pragma HLS pipeline II = 1
                    for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL
                        float temp = bitsToF<uint32_t, float>(src[c].read());
                        src0[c].write(temp);
                        src1[c].write(temp);
                        src2[c].write(temp);
                        src3[c].write(temp);

#ifdef DEBUG_ACSTRATEGY
                        std::cout << "y=" << y << " x=" << x << " " << temp << std::endl;
#endif
                    }
                }
            }
        }
    }
}

void LoadAcStrategyQF(ap_uint<16> xblock,
                      ap_uint<16> yblock,
                      hls::stream<float>& quant_field,
                      hls::stream<float>& quant_field_strm0,
                      hls::stream<float>& quant_field_strm1) {
#pragma HLS INLINE off

    // Disgard first 2 value
    quant_field.read();
    quant_field.read();

// Process y
loop_y:
    for (ap_uint<16> ycnt = 0; ycnt < yblock; ycnt++) {
    // Process x
    loop_x:
        for (ap_uint<16> xcnt = 0; xcnt < xblock; xcnt++) {
        // load quant field of 4x4 region
        load4x4_region:
            for (ap_uint<8> y = 0; y < 4; y++) {
                for (ap_uint<8> x = 0; x < 4; x++) {
#pragma HLS pipeline II = 1

                    ap_uint<4> shift;
                    shift(3, 2) = y(1, 0);
                    shift(1, 0) = x(1, 0);

                    float temp = quant_field.read();
                    quant_field_strm0.write(temp);
                    quant_field_strm1.write(temp);
                }
            }
        }
    }
}

void JudgeAcStrategy(ap_uint<16> xblock,
                     ap_uint<16> yblock,
                     ap_uint<16> xsize,
                     ap_uint<16> ysize,
                     hls::stream<bool>& enable_dct4_strm,
                     hls::stream<bool>& disable_dct16_strm,
                     hls::stream<bool>& enable_dct16_strm,
                     hls::stream<bool>& enable_dct32_strm,
                     hls::stream<ap_uint<4> >& ac_strategy) {
#pragma HLS INLINE off
// Process y
loop_y:
    for (ap_uint<16> ycnt = 0; ycnt < yblock; ycnt++) {
    // Process x
    loop_x:
        for (ap_uint<16> xcnt = 0; xcnt < xblock; xcnt++) {
            bool enable_dct4;
            bool disable_dct16;
            bool enable_dct16[2][2];
            bool enable_dct32;

        judge_ac_strategy:
            for (ap_uint<8> y = 0; y < 4; y++) {
                for (ap_uint<8> x = 0; x < 4; x++) {
#pragma HLS pipeline II = 1

                    enable_dct4 = enable_dct4_strm.read();
                    disable_dct16 = disable_dct16_strm.read();

                    if (x == 0 && y == 0) enable_dct32 = enable_dct32_strm.read();

                    if (x[0] == 0 && y[0] == 0) enable_dct16[y[1]][x[1]] = enable_dct16_strm.read();

                    uint8_t temp;
                    ap_uint<16> bx32 = xcnt;
                    ap_uint<16> by32 = ycnt;

                    ap_uint<16> bx16 = xcnt << 1;
                    ap_uint<16> by16 = ycnt << 1;

                    if (enable_dct4) {
                        temp = 3; // AcStrategy::Type::DCT4X4 3
                    } else if (!disable_dct16 && x == 0 && y == 0 && bx32 < xsize(15, 5) && by32 < ysize(15, 5) &&
                               enable_dct32) {
                        temp = 5; // AcStrategy::Type::DCT32X32 5
                    } else if (!disable_dct16 && x[0] == 0 && y[0] == 0 && bx16 < xsize(15, 4) && by16 < ysize(15, 4) &&
                               enable_dct16[y[1]][x[1]]) {
                        temp = 4; // AcStrategy::Type::DCT16X16 4
                    } else {
                        temp = 0; // AcStrategy::Type::DCT 0
                    }
                    ac_strategy.write(temp);
                }
            }
        }
    }
}

void ComposeDCT(ap_uint<16> xblock,
                ap_uint<16> yblock,
                bool kChooseAcStrategy,
                hls::stream<ap_uint<4> >& strategy_strm,
                hls::stream<float> coeffs4_strm[3],
                hls::stream<float> coeffs8_strm[3],
                hls::stream<float> coeffs16_strm[3],
                hls::stream<float> coeffs32_strm[3],
                hls::stream<uint8_t>& ac_strategy0,
                hls::stream<uint8_t>& ac_strategy1,
                hls::stream<uint8_t>& ac_strategy2,
                hls::stream<uint8_t>& ac_strategy3,
                hls::stream<float> dct[3],
                hls::stream<float> ac_dec[3]) {
#pragma HLS INLINE off

// Process y
loop_y:
    for (ap_uint<16> ycnt = 0; ycnt < yblock; ycnt++) {
    // Process x
    loop_x:
        for (ap_uint<16> xcnt = 0; xcnt < xblock; xcnt++) {
#pragma HLS DATAFLOW

            ap_uint<4> strategy[4][4];
#pragma HLS ARRAY_PARTITION variable = strategy complete
            float coeffs4[3][1024];
            float coeffs8[3][1024];
            float coeffs16[3][1024];
            float coeffs32[3][1024];

        load_coeffs:
            for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1
                for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL
                    coeffs4[c][i] = coeffs4_strm[c].read();
                    coeffs8[c][i] = coeffs8_strm[c].read();
                    coeffs16[c][i] = coeffs16_strm[c].read();
                    coeffs32[c][i] = coeffs32_strm[c].read();
                }
            }

        load4x4:
            for (ap_uint<8> y = 0; y < 4; y++) {
                for (ap_uint<8> x = 0; x < 4; x++) {
                    strategy[y][x] = strategy_strm.read();
                }
            }

        // compose dct
        compose_dct:
            for (ap_uint<8> by = 0; by < 4; by++) {
                for (ap_uint<8> bx = 0; bx < 4; bx++) {
                    for (ap_uint<8> y = 0; y < 8; y++) {
                        for (ap_uint<8> x = 0; x < 8; x++) {
#pragma HLS pipeline II = 1

                            ap_uint<8> strategy_block32 = strategy[0][0];
                            ap_uint<8> strategy_block16 = strategy[by[1] << 1][bx[1] << 1];
                            ap_uint<8> strategy_block8 = strategy[by][bx];

                            ap_uint<10> addr;
                            addr(9, 8) = by(1, 0);
                            addr(7, 5) = y(2, 0);
                            addr(4, 3) = bx(1, 0);
                            addr(2, 0) = x(2, 0);

                            uint8_t strategy_tmp;
                            for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL

                                float dct_tmp;
                                float ac_dec_tmp;

                                if (!kChooseAcStrategy) {
                                    strategy_tmp = 0;
                                    dct_tmp = coeffs8[c][addr];
                                    ac_dec_tmp = coeffs8[c][addr];
                                } else if (strategy_block32 == 5) {
                                    // DCT32
                                    strategy_tmp = 5;
                                    dct_tmp = coeffs32[c][addr];
                                    ac_dec_tmp = coeffs32[c][addr];
                                } else if (strategy_block16 == 4) {
                                    // DCT16
                                    strategy_tmp = 4;
                                    dct_tmp = coeffs16[c][addr];
                                    ac_dec_tmp = coeffs16[c][addr];
                                } else if (strategy_block8 == 0) {
                                    // DCT8
                                    strategy_tmp = 0;
                                    dct_tmp = coeffs8[c][addr];
                                    ac_dec_tmp = coeffs8[c][addr];
                                } else {
                                    // DCT4
                                    strategy_tmp = 3;
                                    dct_tmp = coeffs4[c][addr];
                                    ac_dec_tmp = coeffs4[c][addr];
                                }

                                dct[c].write(dct_tmp);
                                ac_dec[c].write(ac_dec_tmp);

#ifdef DEBUG_ACSTRATEGY
                                std::cout << "acs=" << strategy[by][bx] << " c=" << i << " by=" << by << " bx=" << bx
                                          << " y=" << y << " x=" << x << " coeff16=" << coeffs16[i][addr] << std::endl;
#endif
                            }
                            if (x == 0 && y == 0) {
                                ac_strategy0.write(strategy_tmp);
                                ac_strategy1.write(strategy_tmp);
                                ac_strategy2.write(strategy_tmp);
                                ac_strategy3.write(strategy_tmp);
                            }
                        }
                    }
                }
            }
        }
    }
}

void FindBestAcStrategy(Config config,
                        hls::stream<float>& quant_field,

                        const float inv_dequant8x8[3][64],
                        const float inv_dequant16x16[3][256],
                        const float inv_dequant32x32[3][1024],

                        hls::stream<ap_uint<32> > src[3],
                        hls::stream<uint8_t>& ac_strategy0,
                        hls::stream<uint8_t>& ac_strategy1,
                        hls::stream<uint8_t>& ac_strategy2,
                        hls::stream<uint8_t>& ac_strategy3,
                        hls::stream<float> dct[3],
                        hls::stream<float> ac_dec[3]) {
// TODO(veluca): this function does *NOT* know the actual quantization field
// values, and thus is not able to make choices taking into account the actual
// quantization matrix.
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    static const float kColorWeights16x16[3] = {0.60349588292079182, 1.5435289569786645, 0.33080849938060852};

    static const float kMul16x16[3] = {0.86101693093148191, -0.18168363725368566, 1.0861540086721586};

    static const float kColorWeights4x4[3] = {0.76084140985773008, 0.9344031093258709, 0.31536647913297183};

    static const float kMul4x4[3] = {0.84695221371792806, -0.012220022434342694, 1.079485914917413};

    ap_uint<16> xsize = config.xsize;
    ap_uint<16> ysize = config.ysize;
    ap_uint<16> xblock = config.xblock32;
    ap_uint<16> yblock = config.yblock32;
    float butteraugli_target = config.butteraugli_target;
    float discretization_factor = config.discretization_factor;
    float kMul16 = config.kMulInhomogeneity16x16;
    float kMul32 = config.kMulInhomogeneity32x32;

    hls::stream<float> src0[3]; // 3 plane of region 32x32
#pragma HLS stream variable = src0 depth = 4096
#pragma HLS BIND_STORAGE variable = src0 type = fifo impl = uram
    hls::stream<float> src1[3];
#pragma HLS stream variable = src1 depth = 4096
#pragma HLS BIND_STORAGE variable = src1 type = fifo impl = uram
    hls::stream<float> src2[3];
#pragma HLS stream variable = src2 depth = 8192
#pragma HLS BIND_STORAGE variable = src2 type = fifo impl = uram
    hls::stream<float> src3[3];
#pragma HLS stream variable = src3 depth = 16384
#pragma HLS BIND_STORAGE variable = src3 type = fifo impl = uram

    hls::stream<float> coeffs4[3];
#pragma HLS stream variable = coeffs4 depth = 16384
#pragma HLS BIND_STORAGE variable = coeffs4 type = fifo impl = uram

    hls::stream<float> coeffs8tmp0[3];
#pragma HLS stream variable = coeffs8tmp0 depth = 4096
#pragma HLS BIND_STORAGE variable = coeffs8tmp0 type = fifo impl = uram
    hls::stream<float> coeffs16tmp0[3];
#pragma HLS stream variable = coeffs16tmp0 depth = 8192
#pragma HLS BIND_STORAGE variable = coeffs16tmp0 type = fifo impl = uram
    hls::stream<float> coeffs32tmp0[3];
#pragma HLS stream variable = coeffs32tmp0 depth = 8192
#pragma HLS BIND_STORAGE variable = coeffs32tmp0 type = fifo impl = uram

    hls::stream<float> coeffs8tmp1[3];
#pragma HLS stream variable = coeffs8tmp1 depth = 16384
#pragma HLS BIND_STORAGE variable = coeffs8tmp1 type = fifo impl = uram
    hls::stream<float> coeffs16tmp1[3];
#pragma HLS stream variable = coeffs16tmp1 depth = 16384
#pragma HLS BIND_STORAGE variable = coeffs16tmp1 type = fifo impl = uram
    hls::stream<float> coeffs32tmp1[3];
#pragma HLS stream variable = coeffs32tmp1 depth = 16384
#pragma HLS BIND_STORAGE variable = coeffs32tmp1 type = fifo impl = uram

    hls::stream<float> quant_field_strm0("qf0");
#pragma HLS stream variable = quant_field_strm0 depth = 512
#pragma HLS BIND_STORAGE variable = quant_field_strm0 type = fifo impl = bram
    hls::stream<float> quant_field_strm1("qf1");
#pragma HLS stream variable = quant_field_strm1 depth = 1024
#pragma HLS BIND_STORAGE variable = quant_field_strm1 type = fifo impl = bram

    hls::stream<bool> enable_dct4("enable_dct4");
#pragma HLS stream variable = enable_dct4 depth = 2048
#pragma HLS BIND_STORAGE variable = enable_dct4 type = fifo impl = bram
    hls::stream<bool> disable_dct16("disable_dct16");
#pragma HLS stream variable = disable_dct16 depth = 2048
#pragma HLS BIND_STORAGE variable = disable_dct16 type = fifo impl = bram

    hls::stream<bool> enable_dct16("enable_dct16");
#pragma HLS stream variable = enable_dct16 depth = 128
#pragma HLS BIND_STORAGE variable = enable_dct16 type = fifo impl = srl
    hls::stream<bool> enable_dct32("enable_dct32");
#pragma HLS stream variable = enable_dct32 depth = 128
#pragma HLS BIND_STORAGE variable = enable_dct32 type = fifo impl = srl

    hls::stream<float> entropy16("entropy16");
#pragma HLS stream variable = entropy16 depth = 512
#pragma HLS BIND_STORAGE variable = entropy16 type = fifo impl = bram
    hls::stream<float> inhomogeneity16("inhomogeneity16");
#pragma HLS stream variable = inhomogeneity16 depth = 512
#pragma HLS BIND_STORAGE variable = inhomogeneity16 type = fifo impl = bram
    hls::stream<float> max_quant16("max_quant16");
#pragma HLS stream variable = max_quant16 depth = 512
#pragma HLS BIND_STORAGE variable = max_quant16 type = fifo impl = bram

    hls::stream<float> entropy32("entropy32");
#pragma HLS stream variable = entropy32 depth = 512
#pragma HLS BIND_STORAGE variable = entropy32 type = fifo impl = bram
    hls::stream<float> inhomogeneity32("inhomogeneity32");
#pragma HLS stream variable = inhomogeneity32 depth = 512
#pragma HLS BIND_STORAGE variable = inhomogeneity32 type = fifo impl = bram
    hls::stream<float> max_quant32("max_quant32");
#pragma HLS stream variable = max_quant32 depth = 512
#pragma HLS BIND_STORAGE variable = max_quant32 type = fifo impl = bram

    hls::stream<ap_uint<4> > strategy_temp("strategy_tmp");
#pragma HLS stream variable = strategy_temp depth = 512
#pragma HLS BIND_STORAGE variable = strategy_temp type = fifo impl = bram

// Load data
#ifdef DEBUG_ACSTRATEGY
    std::cout << "Load input" << std::endl;
#endif

    LoadAcStrategySrc(xblock, yblock, src, src0, src1, src2, src3);

    LoadAcStrategyQF(xblock, yblock, quant_field, quant_field_strm0, quant_field_strm1);

// Disable large transform
#ifdef DEBUG_ACSTRATEGY
    std::cout << "Disable large transform" << std::endl;
#endif
    DCT_collection32x32(xblock, yblock, src0, kColorWeights16x16, kMul16x16, disable_dct16);

// Enable DCT4x4
#ifdef DEBUG_ACSTRATEGY
    std::cout << "Enable DCT4x4" << std::endl;
#endif
    DCT_collection32x32(xblock, yblock, src1, kColorWeights4x4, kMul4x4, enable_dct4);

// DCT
#ifdef DEBUG_ACSTRATEGY
    std::cout << "DCT" << std::endl;
#endif
    DCTWrapper(xblock, yblock, src2, coeffs4, coeffs8tmp0, coeffs16tmp0, coeffs32tmp0, coeffs8tmp1, coeffs16tmp1,
               coeffs32tmp1);

// Reference entropy
#ifdef DEBUG_ACSTRATEGY
    std::cout << "Reference entropy" << std::endl;
#endif
    reference_entropy(xblock, yblock, discretization_factor, quant_field_strm0, inv_dequant8x8, src3, coeffs8tmp0,
                      entropy16, entropy32);

    get_quant_inhomogeneity(xblock, yblock, kMul16, kMul32, quant_field_strm1, inhomogeneity16, max_quant16,
                            inhomogeneity32, max_quant32);

    dct_entropy16x16(xblock, yblock, discretization_factor, entropy16, inhomogeneity16, max_quant16, inv_dequant16x16,
                     coeffs16tmp0, enable_dct16);

    dct_entropy32x32(xblock, yblock, discretization_factor, butteraugli_target, entropy32, inhomogeneity32, max_quant32,
                     inv_dequant32x32, coeffs32tmp0, enable_dct32);

// Judge
#ifdef DEBUG_ACSTRATEGY
    std::cout << "Judge AcStrategy" << std::endl;
#endif
    JudgeAcStrategy(xblock, yblock, xsize, ysize, enable_dct4, disable_dct16, enable_dct16, enable_dct32,
                    strategy_temp);

#ifdef DEBUG_ACSTRATEGY
    std::cout << "Compose DCT" << std::endl;
#endif
    ComposeDCT(xblock, yblock, config.kChooseAcStrategy, strategy_temp, coeffs4, coeffs8tmp1, coeffs16tmp1,
               coeffs32tmp1, ac_strategy0, ac_strategy1, ac_strategy2, ac_strategy3, dct, ac_dec);
}

void GetMaxQuant(hls::stream<float>& quant_field,
                 hls::stream<uint8_t>& ac_strategy,
                 ap_uint<8> strategy[4][4],
                 float max_quant8[4][4],
                 float max_quant16[2][2],
                 float& max_quant32) {
#pragma HLS INLINE off

get_max_quant:
    for (ap_uint<8> y = 0; y < 4; y++) {
        for (ap_uint<8> x = 0; x < 4; x++) {
#pragma HLS pipeline II = 1

            strategy[y][x] = ac_strategy.read();
            float quant = quant_field.read();

            if (x == 0 && y == 0) {
                max_quant32 = quant;
            } else {
                max_quant32 = hls::max(quant, max_quant32);
            }

            if (x[0] == 0 && y[0] == 0) {
                max_quant16[y[1]][x[1]] = quant;
            } else {
                max_quant16[y[1]][x[1]] = hls::max(quant, max_quant16[y[1]][x[1]]);
            }

            max_quant8[y][x] = quant;
        }
    }
}

void AdjustValue(ap_uint<8> strategy[4][4],
                 float max_quant8[4][4],
                 float max_quant16[2][2],
                 float max_quant32,
                 hls::stream<float>& o_quant_field) {
#pragma HLS INLINE off

adjust_quant_field:
    for (ap_uint<8> y = 0; y < 4; y++) {
        for (ap_uint<8> x = 0; x < 4; x++) {
#pragma HLS pipeline II = 1

            float quant;
            if (strategy[0][0] == 5) {
                // DCT32
                quant = max_quant32;
            } else if (strategy[y[1] << 1][x[1] << 1] == 4) {
                // DCT16
                quant = max_quant16[y[1]][x[1]];
            } else {
                quant = max_quant8[y][x];
            }
            o_quant_field.write(quant);

#ifdef DEBUG_QUANTIZER
            std::cout << "qf_post=" << quant << std::endl;
#endif
        }
    }
}

void AdjustQuantField(ap_uint<16> xblock,
                      ap_uint<16> yblock,
                      hls::stream<float>& i_quant_field,
                      hls::stream<uint8_t>& ac_strategy,
                      hls::stream<float>& o_quant_field) {
// Replace the whole quant_field in non-8x8 blocks with the maximum of each
// 8x8 block.
#pragma HLS INLINE off

    for (ap_uint<16> by = 0; by < xblock; ++by) {
        for (ap_uint<16> bx = 0; bx < yblock; ++bx) {
#pragma HLS DATAFLOW

            ap_uint<8> strategy[4][4];
#pragma HLS ARRAY_PARTITION variable = strategy complete
            float max_quant8[4][4];
#pragma HLS ARRAY_PARTITION variable = max_quant8 complete
            float max_quant16[2][2];
#pragma HLS ARRAY_PARTITION variable = max_quant16 complete
            float max_quant32;

            // get max quant
            GetMaxQuant(i_quant_field, ac_strategy, strategy, max_quant8, max_quant16, max_quant32);

            // adjust quant field
            AdjustValue(strategy, max_quant8, max_quant16, max_quant32, o_quant_field);
        }
    }
}

int ClampVal(float val) {
#pragma HLS INLINE
    const int kQuantMax = 256;
    return hls::min<float>((float)kQuantMax, hls::max<float>(1.0, val));
}

void SetQuantField(hls::stream<float>& inv_global_scale_strm,
                   ap_uint<16> xblock,
                   ap_uint<16> yblock,
                   hls::stream<float>& quant_field,
                   hls::stream<int32_t>& quant_img_ac0,
                   hls::stream<int32_t>& quant_img_ac1) {
#pragma HLS INLINE off

    float inv_global_scale = inv_global_scale_strm.read();

    for (ap_uint<16> by = 0; by < xblock; ++by) {
        for (ap_uint<16> bx = 0; bx < yblock; ++bx) {
            for (ap_uint<8> y = 0; y < 4; y++) {
                for (ap_uint<8> x = 0; x < 4; x++) {
#pragma HLS pipeline II = 1

                    float quant = quant_field.read();
                    int val = (int)ClampVal(quant * inv_global_scale + 0.5f);

                    quant_img_ac0.write(val);
                    quant_img_ac1.write(val);

#ifdef DEBUG_QUANTIZER
                    std::cout << "qf=" << quant << " quant_qf=" << val << std::endl;
#endif
                }
            }
        }
    }
}

void GetQuantizer(ap_uint<16> xblock,
                  ap_uint<16> yblock,
                  float quant_dc,
                  hls::stream<float>& i_quant_field,
                  hls::stream<float>& inv_global_scale_strm,
                  hls::stream<ap_uint<32> >& scale_out_strm,
                  hls::stream<float> quantizer[3],
                  hls::stream<float>& o_quant_field) {
#pragma HLS INLINE off

    // origin quantizer
    const int kGlobalScaleDenom = 1 << 16;

    const int quant_dc_origin = 64;
    const int global_scale = kGlobalScaleDenom / 64;

    const float global_scale_float = (float)(global_scale) * (1.0 / (float)kGlobalScaleDenom);
    const float inv_global_scale = 1.0 * (float)(kGlobalScaleDenom / global_scale);

    const float qdc = global_scale_float * (float)(quant_dc_origin);
    const float inv_quant_dc = 1.0f / qdc;

    float quant_median = i_quant_field.read();
    float quant_median_absd = i_quant_field.read();

    // Target value for the median value in the quant field.
    const float kQuantFieldTarget = 3.80987740592518214386;
    // We reduce the median of the quant field by the median absolute
    // deviation:
    // higher resolution on highly varying quant fields.

    // update new quantizer
    int global_scale_;
    int quant_dc_;
    float global_scale_float_;
    float inv_global_scale_;
    float inv_quant_dc_;
    bool changed;

    int new_global_scale = (int)((float)kGlobalScaleDenom * (quant_median - quant_median_absd) / kQuantFieldTarget);

    // Ensure that quant_dc_ will always be at least
    // kGlobalScaleDenom/kGlobalScaleNumerator.
    const int kGlobalScaleNumerator = 4096;

    if (new_global_scale > quant_dc * kGlobalScaleNumerator) {
        new_global_scale = quant_dc * kGlobalScaleNumerator;
    }

    // Ensure that new_global_scale is positive and no more than 1<<15.
    if (new_global_scale <= 0) new_global_scale = 1;
    if (new_global_scale > (1 << 15)) new_global_scale = 1 << 15;

    if (new_global_scale != global_scale) {
        global_scale_ = new_global_scale;
        global_scale_float_ = (float)global_scale_ * (1.0 / (float)kGlobalScaleDenom);
        inv_global_scale_ = 1.0 * ((float)kGlobalScaleDenom / (float)global_scale_);

        changed = true;
    } else {
        global_scale_ = global_scale;
        global_scale_float_ = global_scale_float;
        inv_global_scale_ = inv_global_scale;

        changed = false;
    }

    int val = ClampVal(quant_dc * inv_global_scale_ + 0.5f);
    if (val != quant_dc_origin) {
        quant_dc_ = val;
        changed = true;
    } else {
        quant_dc_ = quant_dc;
        changed = false;
    }

    if (changed) {
        float tmp = global_scale_float_ * (float)quant_dc_;
        inv_quant_dc_ = 1.0f / tmp;
    } else {
        inv_quant_dc_ = inv_quant_dc;
    }

    inv_global_scale_strm.write(inv_global_scale_);
    scale_out_strm.write(global_scale_);
    scale_out_strm.write(quant_dc_);

    for (ap_uint<8> i = 0; i < 3; i++) {
        quantizer[i].write(bitsToF<int32_t, float>(quant_dc_));
        quantizer[i].write(bitsToF<int32_t, float>(global_scale_));
        quantizer[i].write(inv_quant_dc_);
        quantizer[i].write(inv_global_scale_);
        quantizer[i].write(global_scale_float_);
    }

#ifdef DEBUG_QUANTIZER
    std::cout << "quant_dc=" << quant_dc_ << std::endl;
    std::cout << "global_scale=" << global_scale_ << std::endl;
    std::cout << "inv_quant_dc=" << inv_quant_dc_ << std::endl;
    std::cout << "inv_global_scale=" << inv_global_scale_ << std::endl;
    std::cout << "global_scale_float=" << global_scale_float_ << std::endl;
#endif

    for (ap_uint<16> y = 0; y < yblock; y++) {
        for (ap_uint<16> x = 0; x < xblock; x++) {
            for (ap_uint<16> k = 0; k < 16; k++) {
#pragma HLS pipeline II = 1

                float tmp = i_quant_field.read();
                o_quant_field.write(tmp);

#ifdef DEBUG_QUANTIZER
                std::cout << "qf_pre=" << tmp << std::endl;
#endif
            }
        }
    }
}

void FindBestQuantizer(Config config,
                       hls::stream<float>& quant_field,
                       hls::stream<uint8_t>& ac_strategy,
                       hls::stream<float> quantizer[3],
                       hls::stream<ap_uint<32> >& scale_out,
                       hls::stream<int32_t>& quant_img_ac0,
                       hls::stream<int32_t>& quant_img_ac1) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<float> inv_global_scale_strm;
#pragma HLS stream variable = inv_global_scale_strm depth = 4
#pragma HLS resource variable = inv_global_scale_strm core = FIFO_SRL
    hls::stream<float> quant_field_temp0;
#pragma HLS stream variable = quant_field_temp0 depth = 16
#pragma HLS resource variable = quant_field_temp0 core = FIFO_SRL
    hls::stream<float> quant_field_temp1;
#pragma HLS stream variable = quant_field_temp1 depth = 16
#pragma HLS resource variable = quant_field_temp1 core = FIFO_SRL

    GetQuantizer(config.xblock32, config.yblock32, config.quant_dc, quant_field, inv_global_scale_strm, scale_out,
                 quantizer, quant_field_temp0);

    AdjustQuantField(config.xblock32, config.yblock32, quant_field_temp0, ac_strategy, quant_field_temp1);

    SetQuantField(inv_global_scale_strm, config.xblock32, config.yblock32, quant_field_temp1, quant_img_ac0,
                  quant_img_ac1);
}

template <typename type_t, int N>
void Delay_block(type_t in[N], type_t out[N]) {
    for (ap_uint<8> i = 0; i < N; i++) {
#pragma HLS pipeline II = 1
        out[i] = in[i];
    }
}

#define SHIFT_SCL (6)
#define FACTOR_SCL (1 << SHIFT_SCL)
#define SCLF(a) ((int)((int)a * (int)FACTOR_SCL))

typedef ap_int<SHIFT_SCL + 1> ap_frac;
typedef ap_uint<SHIFT_SCL + 1> apu_frac;
typedef ap_int<SHIFT_SCL + 10> ap_frac16;

int16_t UpdateErr_int(ap_frac16 val_i, apu_frac thres_i, char k, ap_frac& err_left, ap_frac previous_row_err_i[8]) {
#pragma HLS INLINE

    int idx = k & 7;
    short err_i;

    if (k == 0)
        err_i = 0;
    else if ((idx) == 0) {
        err_i = previous_row_err_i[idx];
    } else {
        if (k > 7)
            err_i = err_left + previous_row_err_i[idx];
        else
            err_i = err_left;
    }
    bool isPos = val_i > 0;

    int val_org_i = val_i;
    bool isValOrg_1 = (val_org_i > FACTOR_SCL) || (0 - val_org_i > FACTOR_SCL);
    apu_frac val_frac = val_i & (FACTOR_SCL - 1);
    ap_frac16 val_int = (val_i - val_frac) >> SHIFT_SCL;
    bool isValIntZero = val_int == 0;
    bool isValNegOne = val_int == -1;

    bool isZero_u;
    bool isZero_Nu;
    bool isUseErr = (err_i > 0);

    ap_frac gap_u_Z_p = (0 << SHIFT_SCL) + val_frac + err_i / 2;
    ap_frac gap_u_Z_n = -(((-1) << SHIFT_SCL) + val_frac - err_i / 2);
    ap_frac gap_u_Nz_p = val_frac + err_i / 2 - (((val_frac + err_i / 2 + FACTOR_SCL / 2) >> SHIFT_SCL) << SHIFT_SCL);
    ap_frac gap_u_Nz_n = (((val_frac - err_i / 2 + FACTOR_SCL / 2) >> SHIFT_SCL) << SHIFT_SCL) - val_frac + err_i / 2;
    ap_frac gap_un_Z_p = val_frac;
    ap_frac gap_un_Z_n = (FACTOR_SCL - val_frac);
    ap_frac gap_un_Nz_p = val_frac - (((val_frac + FACTOR_SCL / 2) >> SHIFT_SCL) << SHIFT_SCL);
    ap_frac gap_un_Nz_n = (((val_frac + FACTOR_SCL / 2) >> SHIFT_SCL) << SHIFT_SCL) - val_frac;

    ap_frac err_i_gap = isValOrg_1 ? 0 : err_i;
    ap_frac err_i_gap_un_Z_p = err_i_gap + gap_un_Z_p;
    ap_frac err_i_gap_un_Z_n = err_i_gap + gap_un_Z_n;
    ap_frac err_i_gap_un_Nz_p = err_i_gap + gap_un_Nz_p;
    ap_frac err_i_gap_un_Nz_n = err_i_gap + gap_un_Nz_n;

    bool NoCarry_u_p = val_frac < (thres_i - err_i / 2);
    bool NoCarry_u_n = (FACTOR_SCL - val_frac) < (thres_i - err_i / 2);
    bool NoCarry_Nu_p = val_frac < thres_i;
    bool NoCarry_Nu_n = (FACTOR_SCL - val_frac) < thres_i;

    if (isUseErr) {
        if (isPos) {
            if (isValIntZero && NoCarry_u_p)
                isZero_u = true;
            else
                isZero_u = false;
        } else {
            if (isValNegOne && NoCarry_u_n)
                isZero_u = true;
            else
                isZero_u = false;
        }
    }
    if (!isUseErr) {
        if (isPos) {
            if (isValIntZero && NoCarry_Nu_p)
                isZero_Nu = true;
            else
                isZero_Nu = false;
        } else {
            if (isValNegOne && NoCarry_Nu_n)
                isZero_Nu = true;
            else
                isZero_Nu = false;
        }
    }

    if (k == 0 || (idx) == 7) {
        err_left = 0;
    } else if (isUseErr) {
        if (isZero_u) {
            if (isPos)
                err_left = gap_u_Z_p / 2;
            else
                err_left = gap_u_Z_n / 2;
        } else {
            if (isPos)
                err_left = gap_u_Nz_p / 2;
            else
                err_left = gap_u_Nz_n / 2;
        }
    } else {
        if (isZero_Nu) {
            if (isPos)
                err_left = err_i_gap_un_Z_p / 2; // + gap_un_Z_p;
            else
                err_left = err_i_gap_un_Z_n / 2; // + gap_un_Z_n;
        } else {
            if (isPos)
                err_left = err_i_gap_un_Nz_p / 2; // + gap_un_Nz_p;
            else
                err_left = err_i_gap_un_Nz_n / 2; // + gap_un_Nz_n;
        }
    }

    ap_frac err_new_i;
    if (k == 0) {
        err_new_i = 0;
    } else if (isUseErr) {
        if (isZero_u) {
            if (isPos)
                err_new_i = gap_u_Z_p;
            else
                err_new_i = gap_u_Z_n;
        } else {
            if (isPos)
                err_new_i = gap_u_Nz_p;
            else
                err_new_i = gap_u_Nz_n;
        }
    } else {
        if (isZero_Nu) {
            if (isPos)
                err_new_i = err_i_gap_un_Z_p; // + gap_un_Z_p;
            else
                err_new_i = err_i_gap_un_Z_n; // + gap_un_Z_n;
        } else {
            if (isPos)
                err_new_i = err_i_gap_un_Nz_p; // + gap_un_Nz_p;
            else
                err_new_i = err_i_gap_un_Nz_n; // + gap_un_Nz_n;
        }
    }

    if ((idx) == 7)
        previous_row_err_i[idx] = err_new_i;
    else
        previous_row_err_i[idx] = err_new_i / 2;

    int16_t v_i;
    if (isUseErr) {
        if (isPos)
            v_i = isZero_u ? 0 : (int16_t)(((val_i + err_i / 2 + FACTOR_SCL / 2) >> SHIFT_SCL));
        else
            v_i = isZero_u ? 0 : (int16_t)(((val_i - err_i / 2 + FACTOR_SCL / 2) >> SHIFT_SCL));
    } else // err is not used
        v_i = isZero_Nu ? 0 : (int16_t)(((val_i + FACTOR_SCL / 2) >> SHIFT_SCL));
    if (v_i > 32767) v_i = 32767;
    if (v_i < -32767) v_i = -32767;

    return v_i;
}

void QuantizeBlockAC0(hls::stream<float>& qac_strm,
                      hls::stream<uint8_t>& ac_strategy_strm,
                      float thres,
                      const float InvDequantMatrix4x4[64],
                      const float InvDequantMatrix8x8[64],
                      const float InvDequantMatrix16x16[256],
                      const float InvDequantMatrix32x32[1024],
                      float cplane[1024],
                      int16_t quantized[1024]) {
#pragma HLS INLINE off
    // Done in a somewhat weird way to preserve the previous behaviour of
    // dithering.
    // TODO(jyrki): properly dither DCT blocks larger than 8.

    for (ap_uint<8> iy = 0; iy < 4; iy++) {
        for (ap_uint<8> ix = 0; ix < 4; ix++) {
            ap_frac err = 0;
            ap_frac previous_row_err_i[8] = {0};
#pragma HLS ARRAY_PARTITION variable = previous_row_err_i complete

            ap_uint<8> strategy_block;
            float qa;
            for (ap_uint<8> k = 0; k < 64; ++k) {
#pragma HLS pipeline II = 1

                if (k == 0) {
                    qa = qac_strm.read();
                    strategy_block = ac_strategy_strm.read();
                }

                ap_uint<10> addr8x8;
                addr8x8(9, 8) = iy(1, 0);
                addr8x8(7, 5) = k(5, 3);
                addr8x8(4, 3) = ix(1, 0);
                addr8x8(2, 0) = k(2, 0);

                ap_uint<10> addr32x32;
                addr32x32(9, 7) = k(5, 3);
                addr32x32[6] = iy[1];
                addr32x32[5] = iy[0];
                addr32x32(4, 2) = k(2, 0);
                addr32x32[1] = ix[1];
                addr32x32[0] = ix[0];

                ap_uint<10> addr16x16;
                addr16x16[9] = iy[1];
                addr16x16(8, 6) = k(5, 3);
                addr16x16[5] = iy[0];
                addr16x16[4] = ix[1];
                addr16x16(3, 1) = k(2, 0);
                addr16x16[0] = ix[0];

                ap_uint<8> addr_quant16;
                addr_quant16(7, 5) = k(5, 3);
                addr_quant16[4] = iy[0];
                addr_quant16(3, 1) = k(2, 0);
                addr_quant16[0] = ix[0];

                float qm, plane;
                if (strategy_block == 5) {
                    // DCT32
                    qm = InvDequantMatrix32x32[addr32x32];
                    plane = cplane[addr32x32];
                } else if (strategy_block == 4) {
                    // DCT16
                    qm = InvDequantMatrix16x16[addr_quant16];
                    plane = cplane[addr16x16];
                } else if (strategy_block == 0) {
                    // DCT8
                    qm = InvDequantMatrix8x8[k];
                    plane = cplane[addr8x8];
                } else {
                    // DCT4
                    qm = InvDequantMatrix4x4[k];
                    plane = cplane[addr8x8];
                }

                float val = plane * (qm * qa);

                ap_frac16 val_i = SCLF(val);
                apu_frac thres_i = SCLF(thres);
                int16_t v_i = UpdateErr_int(val_i, thres_i, k, err, previous_row_err_i);

                ap_uint<10> addr_o;
                if (strategy_block == 5) {
                    // DCT32
                    addr_o = addr32x32;
                } else if (strategy_block == 4) {
                    // DCT16
                    addr_o = addr16x16;
                } else if (strategy_block == 0) {
                    // DCT8
                    addr_o = addr8x8;
                } else {
                    // DCT4
                    addr_o = addr8x8;
                }
                quantized[addr_o] = (int16_t)v_i;

#ifdef DEBUG_COEFFS
                if (ix == 0 && iy == 0) {
                    std::cout << std::hex << "k2_qua: k=" << k << " cplane=" << (int&)plane << " val=" << (int&)val
                              << " qm=" << (int&)qm << " qac=" << (int&)qa << " quantized=" << v_i << std::endl;

                    std::cout << std::setprecision(16) << "k2_qua: k=" << k << " cplane=" << plane << " val=" << val
                              << " qm=" << qm << " qac=" << qa << " quantized=" << v_i << std::endl;
                }
#endif
            }
        }
    }
}

void QuantizeBlockAC1(float qac[4][4],
                      uint8_t ac_strategy[4][4],
                      float thres,
                      const float InvDequantMatrix4x4[64],
                      const float InvDequantMatrix8x8[64],
                      const float InvDequantMatrix16x16[256],
                      const float InvDequantMatrix32x32[1024],
                      float cplane[1024],
                      int16_t quantized[1024]) {
#pragma HLS INLINE off
    // Done in a somewhat weird way to preserve the previous behaviour of
    // dithering.
    // TODO(jyrki): properly dither DCT blocks larger than 8.

    for (ap_uint<8> iy = 0; iy < 4; iy++) {
        for (ap_uint<8> ix = 0; ix < 4; ix++) {
            ap_frac err = 0;
            ap_frac previous_row_err_i[8] = {0};
#pragma HLS ARRAY_PARTITION variable = previous_row_err_i complete
            for (ap_uint<8> k = 0; k < 64; ++k) {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = ac_strategy inter false

                ap_uint<10> addr8x8;
                addr8x8(9, 8) = iy(1, 0);
                addr8x8(7, 5) = k(5, 3);
                addr8x8(4, 3) = ix(1, 0);
                addr8x8(2, 0) = k(2, 0);

                ap_uint<10> addr32x32;
                addr32x32(9, 7) = k(5, 3);
                addr32x32[6] = iy[1];
                addr32x32[5] = iy[0];
                addr32x32(4, 2) = k(2, 0);
                addr32x32[1] = ix[1];
                addr32x32[0] = ix[0];

                ap_uint<10> addr16x16;
                addr16x16[9] = iy[1];
                addr16x16(8, 6) = k(5, 3);
                addr16x16[5] = iy[0];
                addr16x16[4] = ix[1];
                addr16x16(3, 1) = k(2, 0);
                addr16x16[0] = ix[0];

                ap_uint<8> addr_quant16;
                addr_quant16(7, 5) = k(5, 3);
                addr_quant16[4] = iy[0];
                addr_quant16(3, 1) = k(2, 0);
                addr_quant16[0] = ix[0];

                ap_uint<8> strategy_block = ac_strategy[iy][ix];

                float qm, qa, plane;
                if (strategy_block == 5) {
                    // DCT32
                    qm = InvDequantMatrix32x32[addr32x32];
                    plane = cplane[addr32x32];
                    qa = qac[iy][ix];
                } else if (strategy_block == 4) {
                    // DCT16
                    qm = InvDequantMatrix16x16[addr_quant16];
                    plane = cplane[addr16x16];
                    qa = qac[iy][ix];
                } else if (strategy_block == 0) {
                    // DCT8
                    qm = InvDequantMatrix8x8[k];
                    plane = cplane[addr8x8];
                    qa = qac[iy][ix];
                } else {
                    // DCT4
                    qm = InvDequantMatrix4x4[k];
                    plane = cplane[addr8x8];
                    qa = qac[iy][ix];
                }

                float val = plane * (qm * qa);

                ap_frac16 val_i = SCLF(val);
                apu_frac thres_i = SCLF(thres);
                int16_t v_i = UpdateErr_int(val_i, thres_i, k, err, previous_row_err_i);

                ap_uint<10> addr_o;
                if (strategy_block == 5) {
                    // DCT32
                    addr_o = addr32x32;
                } else if (strategy_block == 4) {
                    // DCT16
                    addr_o = addr16x16;
                } else if (strategy_block == 0) {
                    // DCT8
                    addr_o = addr8x8;
                } else {
                    // DCT4
                    addr_o = addr8x8;
                }
                quantized[addr_o] = (int16_t)v_i;

#ifdef DEBUG_COEFFS
                if (ix == 0 && iy == 0) {
                    std::cout << std::hex << "k2_qua: k=" << k << " cplane=" << (int&)plane << " val=" << (int&)val
                              << " qm=" << (int&)qm << " qac=" << (int&)qa << " quantized=" << v_i << std::endl;

                    std::cout << std::setprecision(16) << "k2_qua: k=" << k << " cplane=" << plane << " val=" << val
                              << " qm=" << qm << " qac=" << qa << " quantized=" << v_i << std::endl;
                }
#endif
            }
        }
    }
}

void adjustAcDec(hls::stream<uint8_t>& acs1,
                 hls::stream<float>& inv_quant,
                 const float DequantMatrix4x4Y[64],
                 const float DequantMatrix8x8Y[64],
                 const float DequantMatrix16x16Y[256],
                 const float DequantMatrix32x32Y[1024],
                 int16_t quantized[1024],
                 float dec_ac_Y[1024]) {
#pragma HLS INLINE off

adjust:
    for (ap_uint<8> by = 0; by < 4; by++) {
        for (ap_uint<8> bx = 0; bx < 4; bx++) {
            ap_uint<8> strategy_block;
            float inv_qac;
            for (ap_uint<8> k = 0; k < 64; k++) {
#pragma HLS pipeline II = 1

                ap_uint<10> addr32x32;
                addr32x32(9, 8) = by(1, 0);
                addr32x32(7, 5) = k(5, 3);
                addr32x32(4, 3) = bx(1, 0);
                addr32x32(2, 0) = k(2, 0);

                ap_uint<8> addr16x16;
                addr16x16[7] = by[0];
                addr16x16(6, 4) = k(5, 3);
                addr16x16[3] = bx[0];
                addr16x16(2, 0) = k(2, 0);

                int16_t quantized_coeff = quantized[addr32x32];

                if (k == 0) {
                    strategy_block = acs1.read();
                    inv_qac = inv_quant.read();
                }

                float dequant;
                if (strategy_block == 5) {
                    // DCT32
                    dequant = DequantMatrix32x32Y[addr32x32];
                } else if (strategy_block == 4) {
                    // DCT16
                    dequant = DequantMatrix16x16Y[addr16x16];
                } else if (strategy_block == 0) {
                    // DCT8
                    dequant = DequantMatrix8x8Y[k];
                } else if (strategy_block == 3) {
                    // DCT4
                    dequant = DequantMatrix4x4Y[k];
                }

                float kBiasNumerator = 0.145f;
                float AdjustQuantBias;

                if (quantized_coeff == 0)
                    AdjustQuantBias = 0;
                else if (quantized_coeff == 1)
                    AdjustQuantBias = 1.0f - 0.07005449891748593f;
                else if (quantized_coeff == -1)
                    AdjustQuantBias = 0.07005449891748593f - 1.0f;
                else
                    AdjustQuantBias = quantized_coeff - kBiasNumerator / quantized_coeff;

                float out = AdjustQuantBias * dequant * inv_qac;
                dec_ac_Y[addr32x32] = out;

#ifdef DEBUG_COEFFS
                std::cout << "k2_quant_y=" << by << " bx=" << bx << " k=" << k << " quant_y=" << (int)quantized_coeff
                          << " dec_ac_Y=" << out << std::endl;
#endif
            }
        }
    }
}

void QuantizeRoundtripBlockAC(ap_uint<16> xblock,
                              ap_uint<16> yblock,
                              hls::stream<float>& quant,
                              hls::stream<float>& inv_quant,
                              hls::stream<uint8_t>& ac_strategy,

                              const float DequantMatrix4x4Y[64],
                              const float DequantMatrix8x8Y[64],
                              const float DequantMatrix16x16Y[256],
                              const float DequantMatrix32x32Y[1024],

                              const float InvDequantMatrix4x4Y[64],
                              const float InvDequantMatrix8x8Y[64],
                              const float InvDequantMatrix16x16Y[256],
                              const float InvDequantMatrix32x32Y[1024],

                              hls::stream<float>& plane_y,
                              hls::stream<float>& dec_ac_y) {
#pragma HLS INLINE off

    const float thres = 0.6f;

    for (ap_uint<16> y = 0; y < yblock; ++y) {
        for (ap_uint<16> x = 0; x < xblock; ++x) {
#pragma HLS DATAFLOW

            hls::stream<uint8_t> acs0;
#pragma HLS stream variable = acs0 depth = 512
#pragma HLS resource variable = acs0 core = FIFO_BRAM
            hls::stream<uint8_t> acs1;
#pragma HLS stream variable = acs1 depth = 1024
#pragma HLS resource variable = acs1 core = FIFO_BRAM

        duplicate_acs:
            for (ap_uint<8> dy = 0; dy < 4; dy++) {
                for (ap_uint<8> dx = 0; dx < 4; dx++) {
#pragma HLS pipeline II = 1

                    uint8_t tmp = ac_strategy.read();
                    acs0.write(tmp);
                    acs1.write(tmp);
                }
            }

            float y_plane[1024];
        load_src:
            for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1
                y_plane[i] = plane_y.read();
            }

            int16_t quantized[1024];
            QuantizeBlockAC0(quant, acs0, thres, InvDequantMatrix4x4Y, InvDequantMatrix8x8Y, InvDequantMatrix16x16Y,
                             InvDequantMatrix32x32Y, y_plane, quantized);

            float dec_ac_Y[1024];
            adjustAcDec(acs1, inv_quant, DequantMatrix4x4Y, DequantMatrix8x8Y, DequantMatrix16x16Y, DequantMatrix32x32Y,
                        quantized, dec_ac_Y);

        feed:
            for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1
                dec_ac_y.write(dec_ac_Y[i]);
            }
        }
    }
}

// "y_plane" may refer to plane#1 of "coeffs"; it is also organized in the
// block layout (consecutive block coefficient `pixels').
// Class Dequant applies color correlation maps back.
void UnapplyColorCorrelationAC(ap_uint<16> xblock,
                               ap_uint<16> yblock,
                               float ytob_map[16384],
                               float ytox_map[16384],
                               hls::stream<float>& plane_x,
                               hls::stream<float>& plane_y,
                               hls::stream<float>& plane_b,
                               hls::stream<float>& coeffs_Y,
                               hls::stream<float> coeffs_XYB[3]) {
#pragma HLS INLINE off

LOOP_Y:
    for (ap_uint<16> y = 0; y < yblock; ++y) {
    LOOP_X:
        for (ap_uint<16> x = 0; x < xblock; ++x) {
#pragma HLS DATAFLOW

            float x_plane[1024];
            float y_plane[1024];
            float b_plane[1024];
            float coeffs_y[1024];
            float coeffs[3][1024];
#pragma HLS ARRAY_PARTITION variable = coeffs complete dim = 1

        LOAD:
            for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1
                x_plane[i] = plane_x.read();
                y_plane[i] = plane_y.read();
                b_plane[i] = plane_b.read();

                coeffs_y[i] = coeffs_Y.read();
            }

        CALC:
            for (ap_uint<8> by = 0; by < 4; by++) {
                for (ap_uint<8> bx = 0; bx < 4; bx++) {
                    for (ap_uint<8> k = 0; k < 64; k++) {
#pragma HLS PIPELINE II = 1

                        ap_uint<14> block64 = xblock[0] ? ap_uint<14>(xblock(15, 1) + 1) : xblock(15, 1);
                        ap_uint<14> cmap_addr = y(15, 1) * block64 + x(15, 1);

                        float ytob = ytob_map[cmap_addr];
                        float ytox = ytox_map[cmap_addr];

                        ap_uint<10> addr;
                        addr(9, 8) = by(1, 0);
                        addr(7, 5) = k(5, 3);
                        addr(4, 3) = bx(1, 0);
                        addr(2, 0) = k(2, 0);

                        float in_y = y_plane[addr];
                        float in_b = b_plane[addr];
                        float in_x = x_plane[addr];

                        float y_tmp = coeffs_y[addr];

                        float out_b = in_b - ytob * in_y;
                        float out_x = in_x - ytox * in_y;

                        coeffs[0][addr] = out_x;
                        coeffs[1][addr] = y_tmp;
                        coeffs[2][addr] = out_b;

#ifdef DEBUG_COEFFS
                        std::cout << std::setprecision(16) << "k2_corr_in by=" << by << " bx=" << bx << " k=" << k
                                  << " x=" << in_x << " y=" << in_y << " b=" << in_b << std::endl;

                        std::cout << std::setprecision(16) << "k2_corr_out by=" << by << " bx=" << bx << " k=" << k
                                  << " x=" << out_x << " y=" << y_tmp << " b=" << out_b << std::endl;
#endif
                    }
                }
            }

        FEED:
            for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL
                for (ap_uint<16> i = 0; i < 1024; i++) {
                    coeffs_XYB[c].write(coeffs[c][i]);
                }
            }
        }
    }
}

void InitializeComputeCoeffocients(ap_uint<16> xblock,
                                   ap_uint<16> yblock,
                                   hls::stream<float>& quantizer,
                                   hls::stream<uint8_t>& ac_strategy,
                                   hls::stream<int32_t>& quant_field,

                                   hls::stream<uint8_t>& ac_strategy0,
                                   hls::stream<uint8_t>& ac_strategy1,
                                   hls::stream<uint8_t>& ac_strategy2,
                                   hls::stream<float>& inv_qac,
                                   hls::stream<float>& qac0,
                                   hls::stream<float>& qac1) {
#pragma HLS INLINE off

    Quantizer q;
    q.quant_dc = fToBits<float, int32_t>(quantizer.read());
    q.global_scale = fToBits<float, int32_t>(quantizer.read());
    q.inv_quant_dc = quantizer.read();
    q.inv_global_scale = quantizer.read();
    q.global_scale_float = quantizer.read();

LOOP_Y:
    for (ap_uint<16> by = 0; by < yblock; ++by) {
    LOOP_X:
        for (ap_uint<16> bx = 0; bx < xblock; ++bx) {
        initialize:
            for (ap_uint<8> y = 0; y < 4; y++) {
                for (ap_uint<8> x = 0; x < 4; x++) {
#pragma HLS pipeline II = 1

                    float acs = ac_strategy.read();
                    ac_strategy0.write(acs);
                    ac_strategy1.write(acs);
                    ac_strategy2.write(acs);

                    int32_t quant = quant_field.read();
                    inv_qac.write(q.inv_global_scale / quant);

                    float tmp = q.global_scale_float * quant;
                    qac0.write(tmp);
                    qac1.write(tmp);

#ifdef DEBUG_COEFFS
                    std::cout << "y=" << y << " x=" << x << " qac=" << tmp << " inv_qac=" << q.inv_global_scale / quant
                              << " qf=" << quant << std::endl;
#endif
                }
            }
        }
    }
}

void LoadAcsDc4x4(hls::stream<uint8_t>& ac_strategy,
                  uint8_t acs[3][4][4],
                  hls::stream<float> dc[3],
                  float dc0[3][16],
                  float dc1[3][16],
                  float dc2[3][16]) {
#pragma HLS INLINE off
load_acs_dc:
    for (ap_uint<8> y = 0; y < 4; y++) {
        for (ap_uint<8> x = 0; x < 4; x++) {
#pragma HLS pipeline II = 1

            uint8_t acs_tmp = ac_strategy.read();
            for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL

                ap_uint<4> addr;
                addr(3, 2) = y(1, 0);
                addr(1, 0) = x(1, 0);

                float dc_tmp = dc[c].read();
                dc0[c][addr] = dc_tmp;
                dc1[c][addr] = dc_tmp;
                dc2[c][addr] = dc_tmp;

                acs[c][y][x] = acs_tmp;

#ifdef DEBUG_COEFFS
                std::cout << "c=" << c << " y=" << y << " x=" << x << " dc_dec=" << dc_tmp << std::endl;
#endif
            }
        }
    }
}

void LoadAc32x32(hls::stream<float> ac[3], float ac_in[3][1024]) {
#pragma HLS INLINE off

load_ac:
    for (ap_uint<8> by = 0; by < 4; by++) {
        for (ap_uint<8> bx = 0; bx < 4; bx++) {
            for (ap_uint<8> y = 0; y < 8; y++) {
                for (ap_uint<8> x = 0; x < 8; x++) {
#pragma HLS pipeline II = 1
                    for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL

                        ap_uint<10> addr;
                        addr(9, 8) = by(1, 0);
                        addr(7, 5) = y(2, 0);
                        addr(4, 3) = bx(1, 0);
                        addr(2, 0) = x(2, 0);

                        ac_in[c][addr] = ac[c].read();
                    }
                }
            }
        }
    }
}

// Compute the lowest-frequency coefficients in the DCT block, then compose
// it with ac (1x1 for DCT8, 2x2 for DCT16, etc.)
void ComputeLlf(ap_uint<16> xblock,
                ap_uint<16> yblock,
                hls::stream<uint8_t>& ac_strategy,
                hls::stream<float> dc_in[3],
                hls::stream<float> ac_in[3],
                hls::stream<float>& plane_x,
                hls::stream<float>& plane_y0,
                hls::stream<float>& plane_y1,
                hls::stream<float>& plane_b) {
#pragma HLS INLINE off

LOOP_Y:
    for (ap_uint<16> dy = 0; dy < yblock; ++dy) {
    LOOP_X:
        for (ap_uint<16> dx = 0; dx < xblock; ++dx) {
#pragma HLS DATAFLOW

            uint8_t acs[3][4][4];
#pragma HLS ARRAY_PARTITION variable = acs complete
            float dc0[3][16];
#pragma HLS ARRAY_PARTITION variable = dc0 complete
            float dc1[3][16];
#pragma HLS ARRAY_PARTITION variable = dc1 complete
            float dc2[3][16];
#pragma HLS ARRAY_PARTITION variable = dc2 complete
            float ac[3][1024];
#pragma HLS ARRAY_PARTITION variable = ac complete dim = 1
            float plane[3][1024];
#pragma HLS ARRAY_PARTITION variable = plane complete dim = 1
            float dct1_o[3][16];
#pragma HLS ARRAY_PARTITION variable = dct1_o complete
            float dct2_o[3][16];
#pragma HLS ARRAY_PARTITION variable = dct2_o complete
            float dct4_o[3][16];
#pragma HLS ARRAY_PARTITION variable = dct4_o complete

#ifdef DEBUG_COEFFS
            std::cout << "Load data" << std::endl;
#endif

            LoadAcsDc4x4(ac_strategy, acs, dc_in, dc0, dc1, dc2);
            LoadAc32x32(ac_in, ac);

        LOOP_C0:
            for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL

#ifdef DEBUG_COEFFS
                std::cout << "DCT" << std::endl;
#endif

                Delay_block<float, 16>(dc0[c], dct1_o[c]);
                DCT2x2_block16(dc1[c], dct2_o[c]);
                DCT4x4_block16(dc2[c], dct4_o[c]);

#ifdef DEBUG_COEFFS
                int cnt = 0;
                for (int i = 0; i < 16; i++)
                    std::cout << "c=" << c << " id=" << i << " dct1_i=" << dc0[c][i] << " dct2_i=" << dc1[c][i]
                              << " dct4_i=" << dc2[c][i] << std::endl;

                for (ap_uint<8> i = 0; i < 16; i++)
                    std::cout << "c=" << c << " id=" << i << " dct1_o=" << dct1_o[c][i] << " dct2_o="
                              << dct2_o[c][i] * DCTTotalScale<2>(i(1, 0), i(3, 2)) *
                                     DCTInvTotalScale<16>(i(1, 0), i(3, 2))
                              << " dct4_o="
                              << dct4_o[c][i] * DCTTotalScale<4>(i(1, 0), i(3, 2)) *
                                     DCTInvTotalScale<32>(i(1, 0), i(3, 2))
                              << std::endl;
#endif
            }

        LOOP_C1:
            for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL

#ifdef DEBUG_COEFFS
                std::cout << "Compose AC" << std::endl;
#endif

            judge:
                for (ap_uint<8> by = 0; by < 4; by++) {
                    for (ap_uint<8> bx = 0; bx < 4; bx++) {
#ifdef DEBUG_COEFFS
                        if (c == 0) std::cout << "by=" << by << " bx=" << bx << std::endl;
#endif
                        for (ap_uint<8> y = 0; y < 8; y++) {
                            for (ap_uint<8> x = 0; x < 8; x++) {
#pragma HLS pipeline II = 1

                                ap_uint<8> strategy_block = acs[c][by][bx];

                                ap_uint<10> addr;
                                addr(9, 8) = by(1, 0);
                                addr(7, 5) = y(2, 0);
                                addr(4, 3) = bx(1, 0);
                                addr(2, 0) = x(2, 0);

                                ap_uint<4> addr_dct32;
                                addr_dct32(3, 2) = y(1, 0);
                                addr_dct32(1, 0) = x(1, 0);

                                ap_uint<4> addr_dct16;
                                addr_dct16[3] = by[1];
                                addr_dct16[2] = y[0];
                                addr_dct16[1] = bx[1];
                                addr_dct16[0] = x[0];

                                ap_uint<4> addr_dct8;
                                addr_dct8(3, 2) = by(1, 0);
                                addr_dct8(1, 0) = bx(1, 0);

                                float tmp;
                                if (strategy_block == 5) {
                                    // DCT32
                                    if (by == 0 && bx == 0 && y < 4 && x < 4) {
                                        tmp =
                                            dct4_o[c][addr_dct32] * DCTTotalScale<4>(x, y) * DCTInvTotalScale<32>(x, y);
#ifdef DEBUG_COEFFS
                                        std::cout << "k2_dc_:" << std::setprecision(8) << tmp << std::endl;
#endif
                                    } else
                                        tmp = ac[c][addr];
                                } else if (strategy_block == 4) {
                                    // DCT16
                                    if (by[0] == 0 && bx[0] == 0 && y < 2 && x < 2) {
                                        tmp =
                                            dct2_o[c][addr_dct16] * DCTTotalScale<2>(x, y) * DCTInvTotalScale<16>(x, y);
                                    } else
                                        tmp = ac[c][addr];
                                } else {
                                    // DCT8 && DCT4
                                    if (y == 0 && x == 0) {
                                        tmp = dct1_o[c][addr_dct8];
#ifdef DEBUG_COEFFS
                                        std::cout << "k2_dc_:" << std::setprecision(8) << tmp << std::endl;
#endif
                                    } else
                                        tmp = ac[c][addr];
                                }

                                plane[c][addr] = tmp;

#ifdef DEBUG_COEFFS
                                if (c == 0) {
                                    std::cout << plane[c][addr] << ",";
                                }
#endif
                            }
                        }
#ifdef DEBUG_COEFFS
                        if (c == 0) std::cout << std::endl;
#endif
                    }
                }
            }

#ifdef DEBUG_COEFFS
            std::cout << "Feed Data" << std::endl;
#endif
        FEED:
            for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1

                plane_x.write(plane[0][i]);
                plane_y0.write(plane[1][i]);
                plane_y1.write(plane[1][i]);
                plane_b.write(plane[2][i]);
            }
        }
    }
}

void ScatterCoefficient(uint8_t ac_strategy[4][4], int16_t quantized[1024], hls::stream<int16_t>& ac_quantized) {
#pragma HLS INLINE off

    for (ap_uint<8> by = 0; by < 4; by++) {
        for (ap_uint<8> bx = 0; bx < 4; bx++) {
            for (ap_uint<8> y = 0; y < 8; y++) {
                for (ap_uint<8> x = 0; x < 8; x++) {
#pragma HLS pipeline II = 1

                    uint8_t strategy = ac_strategy[by][bx];
                    ap_uint<10> addr, addr8, addr16, addr32;

                    addr8(9, 8) = by(1, 0);
                    addr8(7, 5) = y(2, 0);
                    addr8(4, 3) = bx(1, 0);
                    addr8(2, 0) = x(2, 0);

                    addr16[9] = by[1];
                    addr16(8, 6) = y(2, 0);
                    addr16[5] = by[0];
                    addr16[4] = bx[1];
                    addr16(3, 1) = x(2, 0);
                    addr16[0] = bx[0];

                    addr32(9, 7) = y(2, 0);
                    addr32[6] = by[1];
                    addr32[5] = by[0];
                    addr32(4, 2) = x(2, 0);
                    addr32[1] = bx[1];
                    addr32[0] = bx[0];

                    if (strategy == 5) {
                        // DCT32
                        addr = addr32;
                    } else if (strategy == 4) {
                        // DCT16
                        addr = addr16;
                    } else if (strategy == 0) {
                        // DCT8
                        addr = addr8;
                    } else {
                        // DCT4
                        addr = addr8;
                    }

                    ac_quantized.write(quantized[addr]);

#ifdef DEBUG_COEFFS
                    std::cout << "scatter by=" << by << " bx=" << bx << " y=" << y << " x=" << x << " addr=" << addr
                              << " v=" << quantized[addr] << std::endl;
#endif
                }
            }
        }
    }
}

void QuantizeAc(ap_uint<16> xblock,
                ap_uint<16> yblock,
                hls::stream<float> coeffs_in[3],
                hls::stream<uint8_t>& ac_strategy,
                hls::stream<float>& quant,

                const float DequantMatrix4x4[3][64],
                const float DequantMatrix8x8[3][64],
                const float DequantMatrix16x16[3][256],
                const float DequantMatrix32x32[3][1024],

                const float InvDequantMatrix4x4[3][64],
                const float InvDequantMatrix8x8[3][64],
                const float InvDequantMatrix16x16[3][256],
                const float InvDequantMatrix32x32[3][1024],

                hls::stream<int16_t> ac_quantized[3]) {
#pragma HLS INLINE off

    const float kZeroBiasDefault[3] = {0.65f, 0.6f, 0.7f};

LOOP_Y:
    for (ap_uint<16> by = 0; by < yblock; ++by) {
    LOOP_X:
        for (ap_uint<16> bx = 0; bx < xblock; ++bx) {
#pragma HLS DATAFLOW

            uint8_t acs0[3][4][4];
#pragma HLS ARRAY_PARTITION variable = acs0 complete
            uint8_t acs1[3][4][4];
#pragma HLS ARRAY_PARTITION variable = acs1 complete
            float qac[3][4][4];
#pragma HLS ARRAY_PARTITION variable = qac complete
            float coeffs[3][1024];
#pragma HLS ARRAY_PARTITION variable = coeffs complete dim = 1
            int16_t quantized[3][1024];
#pragma HLS ARRAY_PARTITION variable = quantized complete dim = 1

        load_acs:
            for (ap_uint<8> y = 0; y < 4; y++) {
                for (ap_uint<8> x = 0; x < 4; x++) {
#pragma HLS pipeline II = 1

                    uint8_t tmp0 = ac_strategy.read();
                    float tmp1 = quant.read();

                    for (ap_uint<8> c = 0; c < 3; c++) {
                        acs0[c][y][x] = tmp0;
                        acs1[c][y][x] = tmp0;
                        qac[c][y][x] = tmp1;
                    }
                }
            }

        quantize_ac:
            for (ap_uint<8> c = 0; c < 3; ++c) {
#pragma HLS UNROLL

            load_coeffs:
                for (ap_uint<16> i = 0; i < 1024; i++) {
#pragma HLS PIPELINE II = 1
                    coeffs[c][i] = coeffs_in[c].read();
                }

#ifdef DEBUG_COEFFS
                std::cout << "QuantizeBlockAC c=" << c << std::endl;
#endif

                QuantizeBlockAC1(qac[c], acs0[c], kZeroBiasDefault[c], InvDequantMatrix4x4[c], InvDequantMatrix8x8[c],
                                 InvDequantMatrix16x16[c], InvDequantMatrix32x32[c], coeffs[c], quantized[c]);

                ScatterCoefficient(acs1[c], quantized[c], ac_quantized[c]);
            }
        }
    }
}

void ComputeCoefficients(Config config,
                         hls::stream<float>& quantizer,
                         hls::stream<float> dc[3],
                         hls::stream<float> ac[3],
                         hls::stream<uint8_t>& ac_strategy,
                         hls::stream<int32_t>& quant_field,

                         float ytob_map[16384],
                         float ytox_map[16384],

                         const float DequantMatrix4x4Y[64],
                         const float DequantMatrix8x8Y[64],
                         const float DequantMatrix16x16Y[256],
                         const float DequantMatrix32x32Y[1024],

                         const float InvDequantMatrix4x4Y[64],
                         const float InvDequantMatrix8x8Y[64],
                         const float InvDequantMatrix16x16Y[256],
                         const float InvDequantMatrix32x32Y[1024],

                         const float DequantMatrix4x4[3][64],
                         const float DequantMatrix8x8[3][64],
                         const float DequantMatrix16x16[3][256],
                         const float DequantMatrix32x32[3][1024],

                         const float InvDequantMatrix4x4[3][64],
                         const float InvDequantMatrix8x8[3][64],
                         const float InvDequantMatrix16x16[3][256],
                         const float InvDequantMatrix32x32[3][1024],

                         hls::stream<int16_t> ac_quantized[3]) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<uint8_t> ac_strategy0("acs0");
#pragma HLS stream variable = ac_strategy0 depth = 512
#pragma HLS resource variable = ac_strategy0 core = FIFO_BRAM
    hls::stream<uint8_t> ac_strategy1("acs1");
#pragma HLS stream variable = ac_strategy1 depth = 2048
#pragma HLS resource variable = ac_strategy1 core = FIFO_BRAM
    hls::stream<uint8_t> ac_strategy2("acs2");
#pragma HLS stream variable = ac_strategy2 depth = 4096
#pragma HLS BIND_STORAGE variable = ac_strategy2 type = fifo impl = uram

    hls::stream<float> inv_qac("inv_qac");
#pragma HLS stream variable = inv_qac depth = 512
#pragma HLS resource variable = inv_qac core = FIFO_BRAM
    hls::stream<float> qac0("qac0");
#pragma HLS stream variable = qac0 depth = 512
#pragma HLS resource variable = qac0 core = FIFO_BRAM
    hls::stream<float> qac1("qac1");
#pragma HLS stream variable = qac1 depth = 4096
#pragma HLS BIND_STORAGE variable = qac1 type = fifo impl = uram

    hls::stream<float> plane_y0("plane_y0");
#pragma HLS stream variable = plane_y0 depth = 1024
#pragma HLS resource variable = plane_y0 core = FIFO_BRAM
    hls::stream<float> plane_x("plane_x");
#pragma HLS stream variable = plane_x depth = 8192
#pragma HLS BIND_STORAGE variable = plane_x type = fifo impl = uram
    hls::stream<float> plane_y1("plane_y1");
#pragma HLS stream variable = plane_y1 depth = 8192
#pragma HLS BIND_STORAGE variable = plane_y1 type = fifo impl = uram
    hls::stream<float> plane_b("plane_b");
#pragma HLS stream variable = plane_b depth = 8192
#pragma HLS BIND_STORAGE variable = plane_b type = fifo impl = uram

    hls::stream<float> dec_ac_Y("dec_ac_Y");
#pragma HLS stream variable = dec_ac_Y depth = 1024
#pragma HLS resource variable = dec_ac_Y core = FIFO_BRAM
    hls::stream<float> coeffs[3];
#pragma HLS stream variable = coeffs depth = 4192
#pragma HLS BIND_STORAGE variable = coeffs type = fifo impl = uram

    ap_uint<16> xblock = config.xblock32;
    ap_uint<16> yblock = config.yblock32;

// Pre-quantized, matches what decoder will see.

#ifdef DEBUG_COEFFS
    std::cout << "Initialize" << std::endl;
#endif

    InitializeComputeCoeffocients(xblock, yblock, quantizer, ac_strategy, quant_field, ac_strategy0, ac_strategy1,
                                  ac_strategy2, inv_qac, qac0, qac1);

#ifdef DEBUG_COEFFS
    std::cout << "Compute low frequency" << std::endl;
#endif

    ComputeLlf(xblock, yblock, ac_strategy0, dc, ac, plane_x, plane_y0, plane_y1, plane_b);

#ifdef DEBUG_COEFFS
    std::cout << "Quantize Y" << std::endl;
#endif

    QuantizeRoundtripBlockAC(xblock, yblock, qac0, inv_qac, ac_strategy1, DequantMatrix4x4Y, DequantMatrix8x8Y,
                             DequantMatrix16x16Y, DequantMatrix32x32Y, InvDequantMatrix4x4Y, InvDequantMatrix8x8Y,
                             InvDequantMatrix16x16Y, InvDequantMatrix32x32Y, plane_y0, dec_ac_Y);

#ifdef DEBUG_COEFFS
    std::cout << "Unapply color correlation" << std::endl;
#endif

    UnapplyColorCorrelationAC(xblock, yblock, ytob_map, ytox_map, plane_x, dec_ac_Y, plane_b, plane_y1, coeffs);

#ifdef DEBUG_COEFFS
    std::cout << "Quantize AC" << std::endl;
#endif

    QuantizeAc(xblock, yblock, coeffs, ac_strategy2, qac1, DequantMatrix4x4, DequantMatrix8x8, DequantMatrix16x16,
               DequantMatrix32x32, InvDequantMatrix4x4, InvDequantMatrix8x8, InvDequantMatrix16x16,
               InvDequantMatrix32x32, ac_quantized);
}

void Reorder_load_acs(ap_uint<16> xblock32,
                      hls::stream<uint8_t>& ac_strategy32,
                      uint8_t ac_strategy[4][1024],
                      uint8_t block[4][1024]) {
#pragma HLS INLINE off

    uint8_t acs[4][4];
Load_ac_strategy:
    for (ap_uint<16> bx = 0; bx < xblock32; ++bx) {
        for (ap_uint<8> y = 0; y < 4; ++y) {
            for (ap_uint<8> x = 0; x < 4; ++x) {
#pragma HLS pipeline II = 1

                ap_uint<10> cur;
                cur(9, 2) = bx(7, 0);
                cur(1, 0) = x(1, 0);

                uint8_t ac_tmp = ac_strategy32.read();
                ac_strategy[y][cur] = ac_tmp;

                acs[y][x] = ac_tmp;
                uint8_t blk;
                if (acs[0][0] == 5) {
                    blk = (y(1, 0), x(1, 0));
                } else if (acs[y[1] << 1][x[1] << 1] == 4) {
                    blk = ((ap_uint<1>)y[0], (ap_uint<1>)x[0]);
                } else {
                    blk = 0;
                }
                block[y][cur] = blk;

#ifdef DEBUG_REORDER
                std::cout << "acs_in y=" << y << " x=" << x << " acs=" << (uint32_t)ac_tmp << std::endl;
#endif
            }
        }
    }
}

void Reorder_load_qf(ap_uint<16> xblock32, hls::stream<int32_t>& quant_field32, int32_t quant_field[4][1024]) {
#pragma HLS INLINE off

Load_quant_field:
    for (ap_uint<16> bx = 0; bx < xblock32; ++bx) {
        for (ap_uint<8> y = 0; y < 4; ++y) {
            for (ap_uint<8> x = 0; x < 4; ++x) {
#pragma HLS pipeline II = 1

                ap_uint<10> cur;
                cur(9, 2) = bx(7, 0);
                cur(1, 0) = x(1, 0);

                quant_field[y][cur] = quant_field32.read();
            }
        }
    }
}

void Reorder_load_dc(ap_uint<16> xblock32, hls::stream<int16_t> dc32[3], int16_t dc[3][4][1024]) {
#pragma HLS INLINE off

Load_dc:
    for (ap_uint<16> bx = 0; bx < xblock32; ++bx) {
        for (ap_uint<8> c = 0; c < 3; ++c) {
#pragma HLS UNROLL
            for (ap_uint<8> y = 0; y < 4; ++y) {
                for (ap_uint<8> x = 0; x < 4; ++x) {
#pragma HLS pipeline II = 1

                    ap_uint<10> cur;
                    cur(9, 2) = bx(7, 0);
                    cur(1, 0) = x(1, 0);

                    dc[c][y][cur] = dc32[c].read();
                }
            }
        }
    }
}

void Reorder_load_ac(ap_uint<16> xblock32, hls::stream<int16_t> ac32[3], ap_uint<64> ac[3][65536]) {
#pragma HLS INLINE off

Load_ac:
    for (ap_uint<16> bx = 0; bx < xblock32; ++bx) {
        for (ap_uint<8> c = 0; c < 3; ++c) {
#pragma HLS UNROLL
            for (ap_uint<8> dy = 0; dy < 4; ++dy) {
                for (ap_uint<8> dx = 0; dx < 4; ++dx) {
                    for (ap_uint<8> y = 0; y < 8; ++y) {
#pragma HLS pipeline II = 8

                        ap_uint<16> cur0, cur1;
                        cur0(15, 11) = (dy(1, 0), y(2, 0));
                        cur0(10, 3) = bx(7, 0);
                        cur0(2, 1) = dx(1, 0);
                        cur0[0] = 0;

                        cur1(15, 11) = (dy(1, 0), y(2, 0));
                        cur1(10, 3) = bx(7, 0);
                        cur1(2, 1) = dx(1, 0);
                        cur1[0] = 1;

                        ap_uint<64> tmp0, tmp1;
                        ap_int<16> r0 = ac32[c].read();
                        ap_int<16> r1 = ac32[c].read();
                        ap_int<16> r2 = ac32[c].read();
                        ap_int<16> r3 = ac32[c].read();
                        tmp0 = (r3, r2, r1, r0);

                        ap_int<16> r4 = ac32[c].read();
                        ap_int<16> r5 = ac32[c].read();
                        ap_int<16> r6 = ac32[c].read();
                        ap_int<16> r7 = ac32[c].read();
                        tmp1 = (r7, r6, r5, r4);

                        ac[c][cur0] = tmp0;
                        ac[c][cur1] = tmp1;
                    }
                }
            }
        }
    }
}

void Reorder_feed_acs(ap_uint<16> by,
                      ap_uint<16> xblock8,
                      ap_uint<16> yblock8,
                      uint8_t ac_strategy[4][1024],
                      uint8_t block[4][1024],
                      hls::stream<ap_uint<32> >& o_ac_strategy,
                      hls::stream<ap_uint<32> >& o_block) {
#pragma HLS INLINE off

Feed_ac_strategy:
    for (ap_uint<8> y = 0; y < 4; ++y) {
        for (ap_uint<16> x = 0; x < xblock8; ++x) {
#pragma HLS pipeline II = 1

            ap_uint<10> cur;
            cur(9, 2) = by(7, 0);
            cur(1, 0) = y(1, 0);

            if (cur < yblock8) {
                o_ac_strategy.write(ac_strategy[y][x]);
                o_block.write(block[y][x]);

#ifdef DEBUG_REORDER
                std::cout << "acs_out y=" << y << " x=" << x << " acs=" << (uint32_t)ac_strategy[y][x] << std::endl;
#endif
            }
        }
    }
}

void Reorder_feed_qf(ap_uint<16> by,
                     ap_uint<16> xblock8,
                     ap_uint<16> yblock8,
                     int32_t quant_field[4][1024],
                     hls::stream<ap_uint<32> >& o_quant_field) {
#pragma HLS INLINE off

Feed_quant_field:
    for (ap_uint<8> y = 0; y < 4; ++y) {
        for (ap_uint<16> x = 0; x < xblock8; ++x) {
#pragma HLS pipeline II = 1
            ap_uint<10> cur;
            cur(9, 2) = by(7, 0);
            cur(1, 0) = y(1, 0);

            if (cur < yblock8) {
                o_quant_field.write(quant_field[y][x]);
            }
        }
    }
}

void Reorder_feed_dc(
    ap_uint<16> by, ap_uint<16> xblock8, ap_uint<16> yblock8, int16_t dc[3][4][1024], hls::stream<ap_uint<32> >& o_dc) {
#pragma HLS INLINE off

Feed_dc:
    for (ap_uint<8> y = 0; y < 4; ++y) {
        for (ap_uint<8> c = 0; c < 3; ++c) {
            for (ap_uint<16> x = 0; x < xblock8; ++x) {
#pragma HLS pipeline II = 1
                ap_uint<10> cur;
                cur(9, 2) = by(7, 0);
                cur(1, 0) = y(1, 0);

                if (cur < yblock8) {
                    o_dc.write(dc[c][y][x]);
                }
            }
        }
    }
}

void Reorder_feed_ac(ap_uint<16> by,
                     ap_uint<16> xblock8,
                     ap_uint<16> yblock8,
                     ap_uint<64> ac[3][65536],
                     hls::stream<int16_t>& o_ac0,
                     hls::stream<ap_uint<32> >& o_ac1) {
#pragma HLS INLINE off

Feed_ac_order:
    for (ap_uint<8> dy = 0; dy < 4; ++dy) {
        for (ap_uint<16> bx = 0; bx < xblock8; ++bx) {
            for (ap_uint<8> c = 0; c < 3; ++c) {
                for (ap_uint<8> y = 0; y < 8; ++y) {
                    for (ap_uint<8> x = 0; x < 8; ++x) {
#pragma HLS pipeline II = 1
                        ap_uint<10> cur;
                        cur(9, 2) = by(7, 0);
                        cur(1, 0) = dy(1, 0);

                        ap_uint<16> addr;
                        addr(15, 14) = dy(1, 0);
                        addr(13, 11) = y(2, 0);
                        addr(10, 1) = bx(9, 0);
                        addr[0] = x[2];

                        ap_uint<64> uram_tmp = ac[c][addr];
                        int16_t ac_tmp[4];
                        ac_tmp[0] = uram_tmp(15, 0);
                        ac_tmp[1] = uram_tmp(31, 16);
                        ac_tmp[2] = uram_tmp(47, 32);
                        ac_tmp[3] = uram_tmp(63, 48);

                        if (cur < yblock8) {
                            int16_t tmp = ac_tmp[x(1, 0)];

                            o_ac0.write(tmp);
                            o_ac1.write(tmp);
                        }
                    }
                }
            }
        }
    }
}

// 32x32 block scan to 8x8 block scan, disgard redundant result,
void Reorder(Config config,
             hls::stream<int16_t> dc32[3],
             hls::stream<int16_t> ac32[3],
             hls::stream<uint8_t>& ac_strategy32,
             hls::stream<int32_t>& quant_field32,

             hls::stream<ap_uint<32> >& o_dc,
             hls::stream<int16_t>& o_ac0,
             hls::stream<ap_uint<32> >& o_ac1,
             hls::stream<ap_uint<32> >& o_quant_field,
             hls::stream<ap_uint<32> >& o_ac_strategy,
             hls::stream<ap_uint<32> >& o_block) {
#pragma HLS INLINE off

    int16_t dc[3][4][1024];
    ap_uint<64> ac[3][65536];
#pragma HLS RESOURCE variable = ac core = RAM_2P_URAM // XPM_MEMORY uram
    uint8_t ac_strategy[4][1024];
    uint8_t block[4][1024];
    int32_t quant_field[4][1024];

    ap_uint<16> xblock8 = config.xblock8;
    ap_uint<16> yblock8 = config.yblock8;
    ap_uint<16> xblock32 = config.xblock32;
    ap_uint<16> yblock32 = config.yblock32;

    for (ap_uint<16> by = 0; by < yblock32; ++by) {
#pragma HLS DATAFLOW

        Reorder_load_acs(xblock32, ac_strategy32, ac_strategy, block);

        Reorder_load_qf(xblock32, quant_field32, quant_field);

        Reorder_load_dc(xblock32, dc32, dc);

        Reorder_load_ac(xblock32, ac32, ac);

        Reorder_feed_acs(by, xblock8, yblock8, ac_strategy, block, o_ac_strategy, o_block);

        Reorder_feed_qf(by, xblock8, yblock8, quant_field, o_quant_field);

        Reorder_feed_dc(by, xblock8, yblock8, dc, o_dc);

        Reorder_feed_ac(by, xblock8, yblock8, ac, o_ac0, o_ac1);
    }
}

template <typename T, int len>
void bubbleSort(T arr[len], T pld[len]) {
#pragma HLS INLINE off

    T tempKey, tempPld;
    int i, j;

    for (i = 0; i < len - 1; i++) {
        for (j = 0; j < len - 1 - i; j++) {
#pragma HLS PIPELINE

            if (arr[j] > arr[j + 1]) {
                tempKey = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = tempKey;

                tempPld = pld[j];
                pld[j] = pld[j + 1];
                pld[j + 1] = tempPld;
            }
        }
    }
}

void SortOrder(ap_uint<16> xgroup,
               ap_uint<16> ygroup,
               uint32_t num_zeros[16 * 16 * 4 * 64],
               hls::stream<ap_uint<32> >& order) {
#pragma HLS INLINE off

    // zig-zag addr
    const ap_uint<8> natural_coeff_order[8 * 8] = {0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
                                                   12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
                                                   35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
                                                   58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

    for (ap_uint<8> by = 0; by < ygroup; ++by) {
        for (ap_uint<8> bx = 0; bx < xgroup; ++bx) {
            for (ap_uint<8> c = 0; c < 3; ++c) {
                ap_uint<32> key[64];
                ap_uint<32> pld[64];

                // Apply zig-zag order.
                for (ap_uint<8> i = 0; i < 64; ++i) {
#pragma HLS pipeline II = 1

                    ap_uint<8> order = natural_coeff_order[i];

                    ap_uint<16> offset;
                    offset(15, 12) = by(3, 0);
                    offset(11, 8) = bx(3, 0);
                    offset(7, 6) = c(1, 0);
                    offset(5, 0) = order(5, 0);

                    pld[i] = order;
                    // We don't care for the exact number -> quantize number of zeros,
                    // to get less permuted order.
                    key[i] = num_zeros[offset] / 8;
                }

                // sort
                bubbleSort<ap_uint<32>, 64>(key, pld);

                // feed
                for (ap_uint<8> i = 0; i < 64; i++) {
#pragma HLS pipeline II = 1

                    order.write(pld[i]);
                }
            }
        }
    }
}

void ComputeCount(Config config, hls::stream<int16_t>& ac, uint32_t num_zeros[16 * 16 * 4 * 64]) {
#pragma HLS INLINE off

    ap_uint<16> xblock = config.xblock8;
    ap_uint<16> yblock = config.yblock8;

    ap_uint<16> xgroup = config.xgroup;
    ap_uint<16> ygroup = config.ygroup;

initilize:
    for (ap_uint<16> by = 0; by < yblock; ++by) {
        for (ap_uint<16> bx = 0; bx < xblock; ++bx) {
            for (ap_uint<8> c = 0; c < 3; ++c) {
                for (ap_uint<8> k = 0; k < 64; ++k) {
#pragma HLS pipeline II = 1

                    ap_uint<16> offset;
                    offset(15, 12) = by(9, 6);
                    offset(11, 8) = bx(9, 6);
                    offset(7, 6) = c(1, 0);
                    offset(5, 0) = k(5, 0);
                    num_zeros[offset] = 0;
                }
            }
        }
    }

Compute_count:
    for (ap_uint<16> by = 0; by < yblock; ++by) {
        for (ap_uint<16> bx = 0; bx < xblock; ++bx) {
            for (ap_uint<8> c = 0; c < 3; ++c) {
                for (ap_uint<8> k = 0; k < 64; ++k) {
#pragma HLS pipeline II = 1

                    ap_uint<16> offset;
                    offset(15, 12) = by(9, 6);
                    offset(11, 8) = bx(9, 6);
                    offset(7, 6) = c(1, 0);
                    offset(5, 0) = k(5, 0);
                    int16_t row = ac.read();
                    if (row == 0 && k != 0) ++num_zeros[offset];

#ifdef DEBUG_COEFFS_ORDER
                    cnt++;
#endif
                }
            }
        }
    }
}

void ComputeCoeffOrder(Config config, hls::stream<int16_t>& ac, hls::stream<ap_uint<32> >& order) {
#pragma HLS INLINE off

    ap_uint<16> xblock = config.xblock8;
    ap_uint<16> yblock = config.yblock8;

    ap_uint<16> xgroup = config.xgroup;
    ap_uint<16> ygroup = config.ygroup;

    // Count number of zero coefficients, separately for each DCT band.
    uint32_t num_zeros[16 * 16 * 4 * 64];
#pragma HLS BIND_STORAGE variable = num_zeros type = ram_2p impl = uram

    ComputeCount(config, ac, num_zeros);

    SortOrder(xgroup, ygroup, num_zeros, order);
}

void loadIDCTs(hls::stream<float>& dcts, float coeffs0[16], float coeffs1[16], float coeffs2[16]) {
#pragma HLS INLINE off

    ap_uint<4> p0 = 0, p1 = 0, p2 = 0;
LOOP_LOAD_DCTS:
    for (ap_uint<8> by = 0; by < 4; ++by) {
        for (ap_uint<8> bx = 0; bx < 4; ++bx) {
            for (ap_uint<8> y = 0; y < 8; ++y) {
                for (ap_uint<8> x = 0; x < 8; ++x) {
#pragma HLS pipeline II = 1

                    float tmp = dcts.read();
                    ap_uint<4> addr1 = (by(1, 0), bx(1, 0));
                    if (y == 0 && x == 0) {
                        coeffs0[addr1] = tmp; // IDCT1x1
#ifdef DEBUG_IDCT
                        std::cout << "idct1x1: id=" << addr1 << " value=" << tmp << " scaled=" << coeffs0[addr1]
                                  << std::endl;
#endif
                    }

                    ap_uint<4> addr2;
                    addr2[3] = by[1];
                    addr2[2] = y[0];
                    addr2[1] = bx[1];
                    addr2[0] = x[0];
                    if ((by[0] == 0) && (bx[0] == 0) && y < 2 && x < 2) {
                        coeffs1[addr2] = tmp * DCTInvTotalScale<2>(x, y) * DCTTotalScale<16>(x, y); // IDCT2x2
#ifdef DEBUG_IDCT
                        std::cout << "idct2x2: id=" << addr2 << " value=" << tmp << " scaled=" << coeffs1[addr2]
                                  << std::endl;
#endif
                    }

                    ap_uint<4> addr3 = (y(1, 0), x(1, 0));
                    if (by == 0 && bx == 0 && y < 4 && x < 4) {
                        coeffs2[addr3] = tmp * DCTInvTotalScale<4>(x, y) * DCTTotalScale<32>(x, y); // IDCT4x4
#ifdef DEBUG_IDCT
                        std::cout << "idct4x4: id=" << addr3 << " value=" << tmp << " scaled=" << coeffs2[addr3]
                                  << std::endl;
#endif
                    }
                }
            }
        } // bx
    }     // by
}

void IDCT(float from0[16], float from1[16], float from2[16], float to0[16], float to1[16], float to2[16]) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    Delay_block<float, 16>(from0, to0); // Delay
    IDCT2x2_block16(from1, to1);        // IDCT2x2
    IDCT4x4_block16(from2, to2);        // IDCT4x4
}

void feedIDCTs(uint8_t ac_strategy[4][4], float dcs0[16], float dcs1[16], float dcs2[16], hls::stream<float>& dc) {
#pragma HLS INLINE off

Judge:
    for (ap_uint<8> y = 0; y < 4; ++y) {
        for (ap_uint<8> x = 0; x < 4; ++x) {
#pragma HLS pipeline II = 1
            float tmp;
            ap_uint<4> addr;
            addr(3, 2) = y(1, 0);
            addr(1, 0) = x(1, 0);

            uint8_t acs = ac_strategy[y][x];

            if (acs == 5) { // dct32
                tmp = dcs2[addr];
            } else if (acs == 4) { // dct16
                tmp = dcs1[addr];
            } else { // dct4:3 && dct8:0
                tmp = dcs0[addr];
            }

#ifdef DEBUG_IDCT
            std::cout << "dc=" << tmp << " dct8x8=" << dcs0[addr] << " dct16x16=" << dcs1[addr]
                      << " dct32x32=" << dcs2[addr] << std::endl;
#endif

            dc.write(tmp);
        }
    }
}

void QuantizeDC(ap_uint<16> xblock32,
                ap_uint<16> yblock32,
                hls::stream<float>& quantizer,
                hls::stream<float> in[3],
                const float ytox,
                const float ytob,
                hls::stream<int16_t> dc[3],
                hls::stream<float> dc_dec[3]) {
#pragma HLS INLINE off

    Quantizer q;
    q.quant_dc = fToBits<float, int32_t>(quantizer.read());
    q.global_scale = fToBits<float, int32_t>(quantizer.read());
    q.inv_quant_dc = quantizer.read();
    q.inv_global_scale = quantizer.read();
    q.global_scale_float = quantizer.read();

    float mul_x = dequantDCx * q.inv_quant_dc;
    float mul_y = dequantDCy * q.inv_quant_dc;
    float mul_b = dequantDCb * q.inv_quant_dc;

    float inv_mul_x = invDequantDCx * q.quant_dc * q.global_scale_float;
    float inv_mul_y = invDequantDCy * q.quant_dc * q.global_scale_float;
    float inv_mul_b = invDequantDCb * q.quant_dc * q.global_scale_float;

#ifdef DEBUG_FRAMECACHE
    std::cout << "ytox=" << ytox << std::endl;
    std::cout << "ytob=" << ytob << std::endl;

    std::cout << "mul_x=" << mul_x << std::endl;
    std::cout << "mul_y=" << mul_y << std::endl;
    std::cout << "mul_b=" << mul_b << std::endl;

    std::cout << "inv_mul_x=" << inv_mul_x << std::endl;
    std::cout << "inv_mul_y=" << inv_mul_y << std::endl;
    std::cout << "inv_mul_b=" << inv_mul_b << std::endl;
#endif

LOOP_Y:
    for (ap_uint<16> iy = 0; iy < xblock32; ++iy) {
    LOOP_X:
        for (ap_uint<16> ix = 0; ix < yblock32; ++ix) {
        LOOP_QUANT_BY:
            for (ap_uint<8> y = 0; y < 4; y++) {
                for (ap_uint<8> x = 0; x < 4; x++) {
#pragma HLS PIPELINE // II=3 is ok

                    float in_x = in[0].read();
                    float in_y = in[1].read();
                    float in_b = in[2].read();

                    // QuantizeRoundtripDC
                    float coeff_dc_y = hls::round(in_y * inv_mul_y);
                    float dc_dec_y = coeff_dc_y * mul_y;

                    // ApplyColorCorrelationDC false
                    float out_x_f = in_x - ytox * dc_dec_y;
                    float out_b_f = in_b - ytob * dc_dec_y;

                    // QuantizeCoeffsDC
                    float coeff_dc_x = hls::round(out_x_f * inv_mul_x);
                    float coeff_dc_b = hls::round(out_b_f * inv_mul_b);

                    // Dequant
                    float dc_dec_x = coeff_dc_x * mul_x;
                    float dc_dec_b = coeff_dc_b * mul_b;

                    // ApplyColorCorrelationDC true
                    float out_x_t = dc_dec_x + ytox * dc_dec_y;
                    float out_b_t = dc_dec_b + ytob * dc_dec_y;

                    dc[0].write((int16_t)coeff_dc_x);
                    dc[1].write((int16_t)coeff_dc_y);
                    dc[2].write((int16_t)coeff_dc_b);

                    dc_dec[0].write(out_x_t);
                    dc_dec[1].write(dc_dec_y);
                    dc_dec[2].write(out_b_t);

#ifdef DEBUG_FRAMECACHE
                    std::cout << "dc_in: y=" << y << " x=" << x << " X=" << in_x << " Y=" << in_y << " B=" << in_b
                              << std::endl;
                    std::cout << "quantizeRound: y=" << y << " x=" << x << " X=" << in_x << " Y=" << dc_dec_y
                              << " B=" << in_b << std::endl;
                    std::cout << "cor_false: y=" << y << " x=" << x << " X=" << out_x_f << " Y=" << dc_dec_y
                              << " B=" << out_b_f << std::endl;
                    std::cout << "dc: y=" << y << " x=" << x << " X=" << coeff_dc_x << " Y=" << coeff_dc_y
                              << " B=" << coeff_dc_b << std::endl;
                    std::cout << "cor_true: y=" << y << " x=" << x << " X=" << out_x_t << " Y=" << dc_dec_y
                              << " B=" << out_b_t << std::endl;
#endif
                }
            }
        }
    }
}

void AdaptiveDCReconstruction(ap_uint<16> xblock32,
                              ap_uint<16> yblock32,
                              hls::stream<float>& quantizer,
                              hls::stream<int16_t> dc_quantized[3],
                              hls::stream<float> dc_dec_tmp[3],

                              hls::stream<int16_t> dc[3],
                              hls::stream<float> dc_dec[3]) {
#pragma HLS INLINE off

    Quantizer q;
    q.quant_dc = fToBits<float, int32_t>(quantizer.read());
    q.global_scale = fToBits<float, int32_t>(quantizer.read());
    q.inv_quant_dc = quantizer.read();
    q.inv_global_scale = quantizer.read();
    q.global_scale_float = quantizer.read();

    float half_step[3] = {q.inv_quant_dc * dequantDCx * 0.5f, q.inv_quant_dc * dequantDCy * 0.5f,
                          q.inv_quant_dc * dequantDCb * 0.5f};

#ifdef DEBUG_FRAMECACHE
    std::cout << "k2_AdaptiveDCReconstruction:";
#endif

LOOP_Y:
    for (ap_uint<16> iy = 0; iy < xblock32; ++iy) {
    LOOP_X:
        for (ap_uint<16> ix = 0; ix < yblock32; ++ix) {
        LOOP_ADAPTIVE:
            for (ap_uint<8> c = 0; c < 3; c++) {
#pragma HLS UNROLL
            Y:
                for (ap_uint<8> iy = 0; iy < 4; iy++) {
                X:
                    for (ap_uint<8> ix = 0; ix < 4; ix++) {
#pragma HLS PIPELINE II = 1

                        ap_uint<4> addr;
                        addr(3, 2) = iy(1, 0);
                        addr(1, 0) = ix(1, 0);

                        float tmp = dc_dec_tmp[c].read();
                        float out = hls::max(tmp - half_step[c], hls::min(tmp, tmp + half_step[c]));

                        int16_t dc_tmp = dc_quantized[c].read();

                        dc[c].write(dc_tmp);
                        dc_dec[c].write(out);

#ifdef DEBUG_FRAMECACHE
                        std::cout << std::setprecision(16) << out << ",";
#endif
                    } // c
                }     // ix
            }         // iy
        }
    }

#ifdef DEBUG_FRAMECACHE
    std::cout << std::endl;
#endif
}

void LoadAcStrategy4x4(hls::stream<uint8_t>& ac_strategy, uint8_t strategy[3][4][4]) {
#pragma HLS INLINE off

Load_ac_strategy:
    for (ap_uint<8> y = 0; y < 4; ++y) {
        for (ap_uint<8> x = 0; x < 4; ++x) {
#pragma HLS pipeline II = 1

            uint8_t acs_tmp = ac_strategy.read();
            for (ap_uint<8> c = 0; c < 3; c++) strategy[c][y][x] = acs_tmp;
        }
    }
}

void IDCTxyb(ap_uint<16> xblock32,
             ap_uint<16> yblock32,
             hls::stream<float> dcts[3],
             hls::stream<uint8_t>& ac_strategy,
             hls::stream<float> dc_tmp[3]) {
#pragma HLS INLINE off

IDCT:
    for (ap_uint<16> iy = 0; iy < xblock32; ++iy) {
        for (ap_uint<16> ix = 0; ix < yblock32; ++ix) {
#pragma HLS DATAFLOW

            uint8_t strategy[3][4][4];
#pragma HLS ARRAY_PARTITION variable = strategy complete
            float temp0[3][16];
#pragma HLS ARRAY_PARTITION variable = temp0 complete
            float temp1[3][16];
#pragma HLS ARRAY_PARTITION variable = temp1 complete
            float temp2[3][16];
#pragma HLS ARRAY_PARTITION variable = temp2 complete
            float temp3[3][16];
#pragma HLS ARRAY_PARTITION variable = temp3 complete
            float temp4[3][16];
#pragma HLS ARRAY_PARTITION variable = temp4 complete
            float temp5[3][16];
#pragma HLS ARRAY_PARTITION variable = temp5 complete

            LoadAcStrategy4x4(ac_strategy, strategy);

        loopLoadIDCT:
            for (ap_uint<8> c = 0; c < 3; ++c) {
#pragma HLS UNROLL
                loadIDCTs(dcts[c], temp0[c], temp1[c], temp2[c]);
            }

        loopIDCT:
            for (ap_uint<8> c = 0; c < 3; ++c) {
#pragma HLS UNROLL
                IDCT(temp0[c], temp1[c], temp2[c], temp3[c], temp4[c], temp5[c]);
            }

        loopFeedIDCT:
            for (ap_uint<8> c = 0; c < 3; ++c) {
#pragma HLS UNROLL
                feedIDCTs(strategy[c], temp3[c], temp4[c], temp5[c], dc_tmp[c]);
            }
        }
    }
}

void InitializeFrameEncCache(Config config,
                             hls::stream<float>& quantizer0,
                             hls::stream<float>& quantizer1,
                             hls::stream<float> dcts[3],
                             hls::stream<uint8_t>& ac_strategy,
                             float ytob,
                             float ytox,
                             hls::stream<int16_t> dc[3],
                             hls::stream<float> dc_dec[3]) {
#pragma HLS INLINE off
#pragma HLS dataflow

    int xblock32 = config.xblock32;
    int yblock32 = config.yblock32;

    hls::stream<float> dc_tmp[3];
#pragma HLS stream variable = dc_tmp depth = 32
#pragma HLS BIND_STORAGE variable = dc_tmp type = fifo impl = srl
    hls::stream<int16_t> dc_quantized[3];
#pragma HLS stream variable = dc_quantized depth = 32
#pragma HLS BIND_STORAGE variable = dc_quantized type = fifo impl = srl
    hls::stream<float> dc_dec_tmp[3];
#pragma HLS stream variable = dc_dec_tmp depth = 32
#pragma HLS BIND_STORAGE variable = dc_dec_tmp type = fifo impl = srl

    IDCTxyb(xblock32, yblock32, dcts, ac_strategy, dc_tmp);

    QuantizeDC(xblock32, yblock32, quantizer0, dc_tmp, ytox, ytob, dc_quantized, dc_dec_tmp);

    AdaptiveDCReconstruction(xblock32, yblock32, quantizer1, dc_quantized, dc_dec_tmp, dc, dc_dec);
}

void load_config(ap_uint<32> in[32], Config config[8]) {
/*

  uint32_t xsize;
  uint32_t ysize;
  uint32_t xblock8;
  uint32_t yblock8;
  uint32_t xblock32;
  uint32_t yblock32;
  uint32_t xgroup;
  uint32_t ygroup;

  int src_num;
  int in_quant_field_num;
  int cmap_num0;
  int cmap_num1;
  int ac_num;
  int dc_num;
  int acs_num;
  int out_quant_field_num;

  bool kChooseAcStrategy;
  float discretization_factor;
  float kMulInhomogeneity16x16;
  float kMulInhomogeneity32x32;
  float butteraugli_target;
  float intensity_multiplier;

 */

#ifdef DEBUG
    std::cout << "Config:" << std::endl;
    std::cout << "xsize:" << in[0] << std::endl;
    std::cout << "ysize:" << in[1] << std::endl;
    std::cout << "xblock8:" << in[2] << std::endl;
    std::cout << "yblock8:" << in[3] << std::endl;
    std::cout << "xblock32:" << in[4] << std::endl;
    std::cout << "yblock32:" << in[5] << std::endl;
    std::cout << "xgroup:" << in[6] << std::endl;
    std::cout << "ygroup:" << in[7] << std::endl;
    std::cout << "src_num:" << in[8] << std::endl;
    std::cout << "in_quant_field_num:" << in[9] << std::endl;
    std::cout << "cmap_num0:" << in[10] << std::endl;
    std::cout << "cmap_num1:" << in[11] << std::endl;
    std::cout << "ac_num:" << in[12] << std::endl;
    std::cout << "dc_num:" << in[13] << std::endl;
    std::cout << "acs_num:" << in[14] << std::endl;
    std::cout << "out_quant_field_num:" << in[15] << std::endl;
    std::cout << "choose_ac_astrategy:" << in[16] << std::endl;
    std::cout << "discretization_factor:" << bitsToF<uint32_t, float>(in[17]) << std::endl;
    std::cout << "kmul16:" << bitsToF<uint32_t, float>(in[18]) << std::endl;
    std::cout << "kmul32:" << bitsToF<uint32_t, float>(in[19]) << std::endl;
    std::cout << "butteraugli:" << bitsToF<uint32_t, float>(in[20]) << std::endl;
    std::cout << "intensity_mul:" << bitsToF<uint32_t, float>(in[21]) << std::endl;
    std::cout << "quant_dc:" << bitsToF<uint32_t, float>(in[22]) << std::endl;
#endif

    ap_uint<32> tmp[32];
    for (ap_uint<8> i = 0; i < 32; i++) {
#pragma HLS pipeline II = 1
        tmp[i] = in[i];
    }

    for (ap_uint<8> i = 0; i < 8; i++) {
        config[i].xsize = tmp[0];
        config[i].ysize = tmp[1];
        config[i].xblock8 = tmp[2];
        config[i].yblock8 = tmp[3];
        config[i].xblock32 = tmp[4];
        config[i].yblock32 = tmp[5];
        config[i].xgroup = tmp[6];
        config[i].ygroup = tmp[7];

        // config[i].src_num = tmp[8];
        config[i].in_quant_field_num = tmp[9];
        config[i].cmap_num0 = tmp[10];
        config[i].cmap_num1 = tmp[11];
        config[i].ac_num = tmp[12];
        config[i].dc_num = tmp[13];
        config[i].acs_num = tmp[14];
        config[i].out_quant_field_num = tmp[15];

        config[i].kChooseAcStrategy = tmp[16];
        config[i].discretization_factor = bitsToF<uint32_t, float>(tmp[17]);
        config[i].kMulInhomogeneity16x16 = bitsToF<uint32_t, float>(tmp[18]);
        config[i].kMulInhomogeneity32x32 = bitsToF<uint32_t, float>(tmp[19]);
        config[i].butteraugli_target = bitsToF<uint32_t, float>(tmp[20]);
        config[i].intensity_multiplier = bitsToF<uint32_t, float>(tmp[21]);
        config[i].quant_dc = bitsToF<uint32_t, float>(tmp[22]);

        config[i].src_num[0] = tmp[25];
        config[i].src_num[1] = tmp[26];
        config[i].src_num[2] = tmp[27];
        config[i].src_offset[0] = tmp[28];
        config[i].src_offset[1] = tmp[29];
        config[i].src_offset[2] = tmp[30];
    }
};

void streamDup(hls::stream<ap_uint<32> >& istrm,
               hls::stream<bool>& e_istrm,
               hls::stream<float>& ostrm0,
               hls::stream<float>& ostrm1) {
#pragma HLS INLINE off

    bool end = e_istrm.read();
    while (!end) {
        float in = bitsToF<uint32_t, float>(istrm.read());
        end = e_istrm.read();

        ostrm0.write(in);
        ostrm1.write(in);
    }
}

void eliminate_strm_end(hls::stream<bool>& strm_end) {
#pragma HLS INLINE off

    bool end = strm_end.read();
    while (!end) {
        end = strm_end.read();
    }
}

void loadCMap(hls::stream<ap_uint<32> >& cmap_strm,
              ap_uint<32> cmap_num,
              float& YtoB,
              float& YtoX,
              float ytob_map[16384],
              float ytox_map[16384]) {
#pragma HLS INLINE off

    const int32_t kColorFactorX = 256;
    const int32_t kColorOffsetX = 128;
    const float kColorScaleX = 1.0f / kColorFactorX;

    const int32_t kColorFactorB = 128;
    const int32_t kColorOffsetB = 0;
    const float kColorScaleB = 1.0f / kColorFactorB;

    int32_t ytox = cmap_strm.read();
    int32_t ytob = cmap_strm.read();

    YtoX = 1.0 * (ytox - kColorOffsetX) * kColorScaleX;
    YtoB = 1.0 * (ytob - kColorOffsetB) * kColorScaleB;

#ifdef DEBUG
    std::cout << "x=" << ytox << " ytox=" << YtoX << std::endl;
    std::cout << "b=" << ytob << " ytob=" << YtoB << std::endl;
#endif

    for (int i = 0; i < cmap_num; i++) {
#pragma HLS pipeline

        int x = cmap_strm.read();
        int b = cmap_strm.read();

        ytox_map[i] = 1.0 * (x - kColorOffsetX) * kColorScaleX;
        ytob_map[i] = 1.0 * (b - kColorOffsetB) * kColorScaleB;

#ifdef DEBUG
        std::cout << "id=" << i << " x=" << x << " ytox=" << ytox_map[i] << std::endl;
        std::cout << "id=" << i << " b=" << b << " ytob=" << ytob_map[i] << std::endl;
#endif
    }
}

template <int _BurstLen, int _WData>
void DcToAxi(ap_uint<32> xblock8,
             ap_uint<32> yblock8,
             ap_uint<_WData>* dc,
             hls::stream<ap_uint<_WData> >& dc_quantized) {
#pragma HLS INLINE off

    for (int y = 0; y < yblock8; y++) {
        for (int c = 0; c < 3; c++) {
            ap_uint<32> addr = c * yblock8 * xblock8 + y * xblock8;
            for (int x = 0; x < xblock8; x++) {
#pragma HLS PIPELINE II = 1
                dc[addr] = dc_quantized.read();
                addr++;
            }
        }
    }
}

template <int _BurstLen, int _WData>
void AcToAxi(ap_uint<32> xblock8,
             ap_uint<32> yblock8,
             ap_uint<_WData>* ac,
             hls::stream<ap_uint<_WData> >& ac_quantized) {
#pragma HLS INLINE off

    ap_uint<32> num_tile_y = yblock8(2, 0) == 0 ? (ap_uint<32>)yblock8(31, 3) : (ap_uint<32>)(yblock8(31, 3) + 1);
    ap_uint<32> num_tile_x = xblock8(2, 0) == 0 ? (ap_uint<32>)xblock8(31, 3) : (ap_uint<32>)(xblock8(31, 3) + 1);

    ap_uint<32> num_group_y = yblock8(5, 0) == 0 ? (ap_uint<32>)yblock8(31, 6) : (ap_uint<32>)(yblock8(31, 6) + 1);
    ap_uint<32> num_group_x = xblock8(5, 0) == 0 ? (ap_uint<32>)xblock8(31, 6) : (ap_uint<32>)(xblock8(31, 6) + 1);

    for (ap_uint<32> by = 0; by < yblock8; by++) {
        for (ap_uint<32> bx = 0; bx < xblock8; bx++) {
            for (int c = 0; c < 3; c++) {
                ap_uint<32> addr = c * xblock8 * yblock8 * 64 + by(31, 6) * 64 * xblock8 * 64 + bx(31, 6) * 4096 +
                                   (by(5, 0) * xblock8 + bx(5, 0)) * 64;

#ifdef DEBUG_AXI
                if (by(31, 6) > 0 || bx(31, 6) > 0) std::cout << "k2_group:";
#endif

                for (int i = 0; i < 64; i++) {
#pragma HLS PIPELINE II = 1
                    ap_uint<_WData> data = ac_quantized.read();
                    ac[addr] = data;
                    addr++;

                    if (by(31, 6) > 0 || bx(31, 6) > 0) std::cout << (int)data << ",";
                }

                if (by(31, 6) > 0 || bx(31, 6) > 0) std::cout << std::endl;
            }
        }
    }
}

template <int _BurstLen, int _WData>
void QfToAxi(ap_uint<32> xblock8,
             ap_uint<32> yblock8,
             ap_uint<_WData>* qf,
             hls::stream<ap_uint<_WData> >& qf_quantized,
             hls::stream<ap_uint<_WData> >& scale_strm) {
#pragma HLS INLINE off

    ap_uint<32> addr = 0;
    for (int y = 0; y < yblock8; y++) {
        for (int x = 0; x < xblock8; x++) {
#pragma HLS PIPELINE II = 1
            qf[addr] = qf_quantized.read();
            addr++;
        }
    }

    qf[addr] = scale_strm.read();
    qf[addr + 1] = scale_strm.read();
}

template <int _BurstLen, int _WData>
void AcsToAxi(ap_uint<32> xblock8,
              ap_uint<32> yblock8,
              ap_uint<_WData>* acs,
              hls::stream<ap_uint<_WData> >& acs_strm,
              ap_uint<_WData>* block,
              hls::stream<ap_uint<_WData> >& block_strm) {
#pragma HLS INLINE off

    ap_uint<32> addr = 0;
    for (int y = 0; y < yblock8; y++) {
        for (int x = 0; x < xblock8; x++) {
#pragma HLS PIPELINE II = 1
            acs[addr] = acs_strm.read();
            block[addr] = block_strm.read();
            addr++;
        }
    }
}

template <int _BurstLen, int _WData>
void OrderToAxi(ap_uint<32> xgroup,
                ap_uint<32> ygroup,
                ap_uint<_WData>* order,
                hls::stream<ap_uint<_WData> >& order_strm) {
#pragma HLS INLINE off

    ap_uint<32> addr = 0;
    for (int y = 0; y < ygroup; y++) {
        for (int x = 0; x < xgroup; x++) {
            for (int k = 0; k < 3 * 64; k++) {
#pragma HLS PIPELINE II = 1
                order[addr] = order_strm.read();
                addr++;
            }
        }
    }
}

template <int _BurstLen, int _WData>
void WriteToAxi(Config config,
                ap_uint<_WData>* ac,
                hls::stream<ap_uint<_WData> >& ac_quantized,
                ap_uint<_WData>* dc,
                hls::stream<ap_uint<_WData> >& dc_quantized,
                ap_uint<_WData>* qf,
                hls::stream<ap_uint<_WData> >& qf_quantized,
                hls::stream<ap_uint<_WData> >& scale_strm,
                ap_uint<_WData>* acs,
                hls::stream<ap_uint<_WData> >& acs_strm,
                ap_uint<_WData>* block,
                hls::stream<ap_uint<_WData> >& block_strm,
                ap_uint<_WData>* order,
                hls::stream<ap_uint<_WData> >& order_strm) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    DcToAxi<_BurstLen, _WData>(config.xblock8, config.yblock8, dc, dc_quantized);

    AcToAxi<_BurstLen, _WData>(config.xblock8, config.yblock8, ac, ac_quantized);

    QfToAxi<_BurstLen, _WData>(config.xblock8, config.yblock8, qf, qf_quantized, scale_strm);

    AcsToAxi<_BurstLen, _WData>(config.xblock8, config.yblock8, acs, acs_strm, block, block_strm);

    OrderToAxi<_BurstLen, _WData>(config.xgroup, config.ygroup, order, order_strm);
}

void kernel2Wrapper(ap_uint<AXI_SZ> config[MAX_NUM_CONFIG],

                    ap_uint<2 * AXI_SZ> src[AXI_OUT / 2],
                    ap_uint<AXI_SZ> quant_field_in[AXI_QF],
                    ap_uint<AXI_SZ> cmap[AXI_CMAP],

                    ap_uint<AXI_SZ> ac[MAX_NUM_AC],
                    ap_uint<AXI_SZ> dc[MAX_NUM_DC],

                    ap_uint<AXI_SZ> quant_field_out[MAX_NUM_BLOCK88],
                    ap_uint<AXI_SZ> ac_strategy[MAX_NUM_BLOCK88],
                    ap_uint<AXI_SZ> block[MAX_NUM_BLOCK88],
                    ap_uint<AXI_SZ> order[MAX_NUM_ORDER]) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    Config config_dev[8];
#pragma HLS ARRAY_PARTITION variable = config_dev complete dim = 1
#pragma HLS DISAGGREGATE variable = config_dev

#ifdef DEBUG
    std::cout << "==================Kernel2 Start================" << std::endl;
    std::cout << "===================Load config=================" << std::endl;
#endif

    load_config(config, config_dev);

// scan
#ifdef DEBUG
    std::cout << "======================Scan=====================" << std::endl;
#endif

    hls::stream<ap_uint<32> > src_strm[3];
#pragma HLS stream variable = src_strm depth = 4096
#pragma HLS BIND_STORAGE variable = src_strm type = fifo impl = uram

    hls::stream<bool> e_src_strm[3];
#pragma HLS stream variable = e_src_strm depth = 64
#pragma HLS resource variable = e_src_strm core = FIFO_SRL

    //  xf::common::utils_hw::axiToStream<64, 32, ap_uint<32> >(
    //      src, config_dev[0].src_num, src_strm, e_src_strm);

    xf::common::utils_hw::axiToMultiStream<1024, 64, ap_uint<32>, ap_uint<32>, ap_uint<32> >(
        src, src_strm[0], e_src_strm[0], src_strm[1], e_src_strm[1], src_strm[2], e_src_strm[2], config_dev[0].src_num,
        config_dev[0].src_offset);

    hls::stream<ap_uint<32> > quant_field_strm("quant_field_strm");
#pragma HLS stream variable = quant_field_strm depth = 512
#pragma HLS resource variable = quant_field_strm core = FIFO_BRAM

    hls::stream<bool> e_quant_field_strm;
#pragma HLS stream variable = e_quant_field_strm depth = 64
#pragma HLS resource variable = e_quant_field_strm core = FIFO_SRL

    xf::common::utils_hw::axiToStream<64, 32, ap_uint<32> >(quant_field_in, config_dev[0].in_quant_field_num,
                                                            quant_field_strm, e_quant_field_strm);

    hls::stream<ap_uint<32> > cmap_strm("cmap_strm");
#pragma HLS stream variable = cmap_strm depth = 8
#pragma HLS resource variable = cmap_strm core = FIFO_SRL

    hls::stream<bool> e_cmap_strm;
#pragma HLS stream variable = e_cmap_strm depth = 8
#pragma HLS resource variable = e_cmap_strm core = FIFO_SRL

    xf::common::utils_hw::axiToStream<64, 32, ap_uint<32> >(cmap, config_dev[0].cmap_num0, cmap_strm, e_cmap_strm);

// load
#ifdef DEBUG
    std::cout << "===================Load cmap===================" << std::endl;
#endif

    hls::stream<float> quant_field0("acs_quant_field");
#pragma HLS stream variable = quant_field0 depth = 1024
#pragma HLS resource variable = quant_field0 core = FIFO_BRAM
    hls::stream<float> quant_field1("quantizer_quant_field");
#pragma HLS stream variable = quant_field1 depth = 4096
#pragma HLS BIND_STORAGE variable = quant_field1 type = fifo impl = uram

    float ytob;
    float ytox;

    float ytob_map[16384];
#pragma HLS BIND_STORAGE variable = ytob_map type = ram_2p impl = uram
    float ytox_map[16384];
#pragma HLS BIND_STORAGE variable = ytox_map type = ram_2p impl = uram

    eliminate_strm_end(e_src_strm[0]);
    eliminate_strm_end(e_src_strm[1]);
    eliminate_strm_end(e_src_strm[2]);
    eliminate_strm_end(e_cmap_strm);

    streamDup(quant_field_strm, e_quant_field_strm, quant_field0, quant_field1);

    loadCMap(cmap_strm, config_dev[0].cmap_num1, ytob, ytox, ytob_map, ytox_map);

// Find Best AcStrategy
#ifdef DEBUG
    std::cout << "=============Find best ac_strategy=============" << std::endl;
#endif

    hls::stream<uint8_t> ac_strategy0("acs0");
#pragma HLS stream variable = ac_strategy0 depth = 1024
#pragma HLS resource variable = ac_strategy0 core = FIFO_BRAM
    hls::stream<uint8_t> ac_strategy1("acs1");
#pragma HLS stream variable = ac_strategy1 depth = 4096
#pragma HLS BIND_STORAGE variable = ac_strategy1 type = fifo impl = uram
    hls::stream<uint8_t> ac_strategy2("acs2");
#pragma HLS stream variable = ac_strategy2 depth = 8192
#pragma HLS BIND_STORAGE variable = ac_strategy2 type = fifo impl = uram
    hls::stream<uint8_t> ac_strategy3("acs3");
#pragma HLS stream variable = ac_strategy3 depth = 8192
#pragma HLS BIND_STORAGE variable = ac_strategy3 type = fifo impl = uram
    hls::stream<float> dct[3];
#pragma HLS stream variable = dct depth = 4096
#pragma HLS BIND_STORAGE variable = dct type = fifo impl = uram
    hls::stream<float> ac_dec[3];
#pragma HLS stream variable = ac_dec depth = 8192
#pragma HLS BIND_STORAGE variable = ac_dec type = fifo impl = uram

    FindBestAcStrategy(config_dev[1], quant_field0, inv_dequant0_matrix8x8, inv_dequant0_matrix16x16,
                       inv_dequant0_matrix32x32, src_strm, ac_strategy0, ac_strategy1, ac_strategy2, ac_strategy3, dct,
                       ac_dec);

// Find Best Quantizer (quantize quant field)
#ifdef DEBUG
    std::cout << "==============Find best quantizer==============" << std::endl;
#endif

    hls::stream<float> quantizer[3];
#pragma HLS stream variable = quantizer depth = 8
#pragma HLS ARRAY_PARTITION variable = quantizer complete
    hls::stream<ap_uint<32> > scale_strm;
#pragma HLS stream variable = scale_strm depth = 4
#pragma HLS resource variable = scale_strm core = FIFO_SRL
    hls::stream<int32_t> quant_img_ac0;
#pragma HLS stream variable = quant_img_ac0 depth = 4096
#pragma HLS BIND_STORAGE variable = quant_img_ac0 type = fifo impl = uram
    hls::stream<int32_t> quant_img_ac1;
#pragma HLS stream variable = quant_img_ac1 depth = 4096
#pragma HLS BIND_STORAGE variable = quant_img_ac1 type = fifo impl = uram

    FindBestQuantizer(config_dev[2], quant_field1, ac_strategy0, quantizer, scale_strm, quant_img_ac0, quant_img_ac1);

// Initialize FrameEncCache
#ifdef DEBUG
    std::cout << "===========Initialize Frame EncCache===========" << std::endl;
#endif

    hls::stream<float> dc_dec[3];
#pragma HLS stream variable = dc_dec depth = 1024
#pragma HLS resource variable = dc_dec core = FIFO_BRAM
    hls::stream<int16_t> dc_quantized0[3];
#pragma HLS stream variable = dc_quantized0 depth = 4096
#pragma HLS BIND_STORAGE variable = dc_quantized0 type = fifo impl = uram

    InitializeFrameEncCache(config_dev[3], quantizer[0], quantizer[1], dct, ac_strategy1, ytob, ytox, dc_quantized0,
                            dc_dec);

// Compute Coefficient
#ifdef DEBUG
    std::cout << "==============Compute coefficients=============" << std::endl;
#endif

    hls::stream<int16_t> ac_quantized0[3];
#pragma HLS stream variable = ac_quantized0 depth = 2048
#pragma HLS resource variable = ac_quantized0 core = FIFO_BRAM

    ComputeCoefficients(config_dev[4], quantizer[2], dc_dec, ac_dec, ac_strategy2, quant_img_ac0,

                        ytob_map, ytox_map,

                        dequantY_matrix4x4, dequantY_matrix8x8, dequantY_matrix16x16, dequantY_matrix32x32,

                        inv_dequantY_matrix4x4, inv_dequantY_matrix8x8, inv_dequantY_matrix16x16,
                        inv_dequantY_matrix32x32,

                        dequant1_matrix4x4, dequant1_matrix8x8, dequant1_matrix16x16, dequant1_matrix32x32,

                        inv_dequant_matrix4x4, inv_dequant_matrix8x8, inv_dequant_matrix16x16, inv_dequant_matrix32x32,

                        ac_quantized0);

// Reorder
#ifdef DEBUG
    std::cout << "======================Reorder==================" << std::endl;
#endif

    hls::stream<ap_uint<32> > dc_quantized("dc_quantized");
#pragma HLS stream variable = dc_quantized depth = 4096
#pragma HLS resource variable = dc_quantized core = FIFO_BRddAM

    hls::stream<int16_t> ac_quantized1("ac_quantized1");
#pragma HLS stream variable = ac_quantized1 depth = 4096
#pragma HLS resource variable = ac_quantized1 core = FIFO_BRAM

    hls::stream<ap_uint<32> > ac_quantized2("ac_quantized");
#pragma HLS stream variable = ac_quantized2 depth = 4096
#pragma HLS resource variable = ac_quantized2 core = FIFO_BRAM

    hls::stream<ap_uint<32> > quant_img_ac2("qf");
#pragma HLS stream variable = quant_img_ac2 depth = 4096
#pragma HLS resource variable = quant_img_ac2 core = FIFO_BRAM

    hls::stream<ap_uint<32> > ac_strategy4("acs4");
#pragma HLS stream variable = ac_strategy4 depth = 4096
#pragma HLS resource variable = ac_strategy4 core = FIFO_BRAM

    hls::stream<ap_uint<32> > ac_block("block");
#pragma HLS stream variable = ac_block depth = 4096
#pragma HLS resource variable = ac_block core = FIFO_BRAM

    Reorder(config_dev[5], dc_quantized0, ac_quantized0, ac_strategy3, quant_img_ac1, dc_quantized, ac_quantized1,
            ac_quantized2, quant_img_ac2, ac_strategy4, ac_block);

// Compute Order
#ifdef DEBUG
    std::cout << "================Compute Coeff order============" << std::endl;
#endif

    hls::stream<ap_uint<32> > order_strm("order");
#pragma HLS stream variable = order_strm depth = 4096
#pragma HLS resource variable = order_strm core = FIFO_BRAM

    ComputeCoeffOrder(config_dev[6], ac_quantized1, order_strm);

// write out
#ifdef DEBUG
    std::cout << "=====================Write out=================" << std::endl;
#endif

    WriteToAxi<64, 32>(config_dev[7], ac, ac_quantized2, dc, dc_quantized, quant_field_out, quant_img_ac2, scale_strm,
                       ac_strategy, ac_strategy4, block, ac_block, order, order_strm);

#ifdef DEBUG
    std::cout << "===================Kernel2 End=================" << std::endl;
#endif
}

extern "C" void kernel2Top(ap_uint<AXI_SZ> config[MAX_NUM_CONFIG],

                           ap_uint<2 * AXI_SZ> src[AXI_OUT / 2],
                           ap_uint<AXI_SZ> quant_field_in[AXI_QF],
                           ap_uint<AXI_SZ> cmap[AXI_CMAP],

                           ap_uint<AXI_SZ> ac[MAX_NUM_AC],
                           ap_uint<AXI_SZ> dc[MAX_NUM_DC],

                           ap_uint<AXI_SZ> quant_field_out[AXI_QF],
                           ap_uint<AXI_SZ> ac_strategy[MAX_NUM_BLOCK88],
                           ap_uint<AXI_SZ> block[MAX_NUM_BLOCK88],
                           ap_uint<AXI_SZ> order[MAX_NUM_ORDER]) {
#pragma HLS INLINE off

// clang-format off
#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
    num_read_outstanding = 4 num_write_outstanding = 4 max_write_burst_length = 8 \
    max_read_burst_length = 8 bundle = gmem0_0 port = config
#pragma HLS INTERFACE s_axilite port = config bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
    num_read_outstanding = 8 num_write_outstanding = 8 max_write_burst_length = 32 \
    max_read_burst_length = 32 bundle = gmem0_1 port = src 
#pragma HLS INTERFACE s_axilite port = src bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
    num_read_outstanding = 8 num_write_outstanding = 8 max_write_burst_length = 32 \
    max_read_burst_length = 32 bundle = gmem0_2 port = quant_field_in
#pragma HLS INTERFACE s_axilite port = quant_field_in bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
    num_read_outstanding = 8 num_write_outstanding = 8 max_write_burst_length = 32 \
    max_read_burst_length = 32 bundle = gmem0_3 port = cmap
#pragma HLS INTERFACE s_axilite port = cmap bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
    num_read_outstanding = 8 num_write_outstanding = 8 max_write_burst_length = 32 \
    max_read_burst_length = 32 bundle = gmem1_0 port = ac
#pragma HLS INTERFACE s_axilite port = ac bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
    num_read_outstanding = 8 num_write_outstanding = 8 max_write_burst_length = 32 \
    max_read_burst_length = 32 bundle = gmem1_1 port = dc
#pragma HLS INTERFACE s_axilite port = dc bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
    num_read_outstanding = 8 num_write_outstanding = 8 max_write_burst_length = 32 \
    max_read_burst_length = 32 bundle = gmem1_2 port = quant_field_out
#pragma HLS INTERFACE s_axilite port = quant_field_out bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
    num_read_outstanding = 8 num_write_outstanding = 8 max_write_burst_length = 32 \
    max_read_burst_length = 32 bundle = gmem1_3 port = ac_strategy 
#pragma HLS INTERFACE s_axilite port = ac_strategy bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
    num_read_outstanding = 8 num_write_outstanding = 8 max_write_burst_length = 32 \
    max_read_burst_length = 32 bundle = gmem1_4 port = block 
#pragma HLS INTERFACE s_axilite port = block bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
    num_read_outstanding = 8 num_write_outstanding = 8 max_write_burst_length = 32 \
    max_read_burst_length = 32 bundle = gmem1_5 port = order
#pragma HLS INTERFACE s_axilite port = order bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control
    // clang-format on

    kernel2Wrapper(config, src, quant_field_in, cmap, ac, dc, quant_field_out, ac_strategy, block, order);
}
