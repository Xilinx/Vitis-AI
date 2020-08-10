/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifdef __cplusplus
extern "C" {
#endif

#include "dpu_err.h"

#ifdef __cplusplus
}
#endif


#include "neonopt.h"
#include "neon_mathfun.h"

/**
 * Transform input tensor data for one byte.
 * Transform prinsple: dst = (src - mean) * scale
 */
inline void transform_input(int8_t* dst, const uint8_t* src, const float mean, const float scale) {
    int value = (int)((*src - mean) * scale);
    if ((value>127) || (value<-128)) {
        printf("[DNNDK] Invalid pixel value of input tensor: %d\n", value);
        printf("[DNNDK] Please check if decent tool produces correct quantization info.");
        exit(-1);
    };
    *dst = (char)value;
}

/**
 * As for input image, there may be extra bytes for stride.
 * We need to cut out these extra bytes so as to make input tensor continous for alignment.
 * @param buf - pointer to source image data
 * @param h - height of input input iamge
 * @param w - width of input input iamge
 * @param stride - the gap in bytes between two neighbour rows of image data
 */
inline void make_continuous(uint8_t* buf, int h, int w, int stride) {
    int i, j, c = 3;
    for(i = 1; i < h; i++) {
        int idx_base_i = i * stride;
        int idx_base_o = i * w * c;
        for(j = 0; j < w; j++) {
            buf[idx_base_o + j * c]     = buf[idx_base_i + j * c];
            buf[idx_base_o + j * c + 1] = buf[idx_base_i + j * c + 1];
            buf[idx_base_o + j * c + 2] = buf[idx_base_i + j * c + 2];
        }
    }
}

/**
 * @brief transform the format of input image(3 channels) in DPU order using arm neon
 *
 * @param dst - pointer to target data
 * @param src - pointer to source image data
 * @param h - height of input input iamge
 * @param w - width of input input iamge
 * @param shift - point to the shift value array, size equals to channel
 * @param scale - scale value of each pixel
 * @param stride - the gap in bytes between two neighbour rows of image data
 *
 * @return void
 */
void dpuProcessNormalizion(int8_t* dst, uint8_t* src, int h, int w, float* shift, float scale,
                        int stride) {
    int i;
    int channel = 3;
    int iter_size = channel*8;
    int total_size = h*w*channel;
    int align_times = total_size/iter_size;
    int remain_size = total_size%iter_size;

    uint8x8x3_t sbgr_u8;
    uint16x8x3_t sbgr_u16;
    uint32x4_t sb_low_u32, sg_low_u32, sr_low_u32;
    uint32x4_t sb_high_u32, sg_high_u32, sr_high_u32;
    float32x4_t sb_low_f32, sg_low_f32, sr_low_f32;
    float32x4_t sb_high_f32, sg_high_f32, sr_high_f32;
    int32x4_t db_low_s32, dg_low_s32, dr_low_s32;
    int32x4_t db_high_s32, dg_high_s32, dr_high_s32;
    int16x4_t db_low_s16, dg_low_s16, dr_low_s16;
    int16x4_t db_high_s16, dg_high_s16, dr_high_s16;
    int16x8_t db_s16, dg_s16, dr_s16;
    int8x8x3_t dbgr;

    float32x4_t shiftB = vdupq_n_f32(shift[0]);
    float32x4_t shiftG = vdupq_n_f32(shift[1]);
    float32x4_t shiftR = vdupq_n_f32(shift[2]);

    float32x4_t scaleB = vdupq_n_f32(scale);
    float32x4_t scaleG = vdupq_n_f32(scale);
    float32x4_t scaleR = vdupq_n_f32(scale);

    /* cut out extra bytes if src has stride */
    if (stride != w * 3) {
        make_continuous(src, h, w, stride);
    }

    /* deal with data which aligned */
    for (i = 0; i < align_times; i++) {
        // init
        sbgr_u8 = vld3_u8(src + i * iter_size);
        sbgr_u16.val[0] = vmovl_u8(sbgr_u8.val[0]);
        sbgr_u16.val[1] = vmovl_u8(sbgr_u8.val[1]);
        sbgr_u16.val[2] = vmovl_u8(sbgr_u8.val[2]);

        // get low part u32
        sb_low_u32 = vmovl_u16(vget_low_u16(sbgr_u16.val[0]));
        sg_low_u32 = vmovl_u16(vget_low_u16(sbgr_u16.val[1]));
        sr_low_u32 = vmovl_u16(vget_low_u16(sbgr_u16.val[2]));

        // get high part u32
        sb_high_u32 = vmovl_u16(vget_high_u16(sbgr_u16.val[0]));
        sg_high_u32 = vmovl_u16(vget_high_u16(sbgr_u16.val[1]));
        sr_high_u32 = vmovl_u16(vget_high_u16(sbgr_u16.val[2]));

        // get low part float
        sb_low_f32 = vcvtq_f32_u32(sb_low_u32);
        sg_low_f32 = vcvtq_f32_u32(sg_low_u32);
        sr_low_f32 = vcvtq_f32_u32(sr_low_u32);

        // get high part float
        sb_high_f32 = vcvtq_f32_u32(sb_high_u32);
        sg_high_f32 = vcvtq_f32_u32(sg_high_u32);
        sr_high_f32 = vcvtq_f32_u32(sr_high_u32);

        // calculate low part float
        sb_low_f32 = vmulq_f32(vsubq_f32(sb_low_f32, shiftB), scaleB);
        sg_low_f32 = vmulq_f32(vsubq_f32(sg_low_f32, shiftG), scaleG);
        sr_low_f32 = vmulq_f32(vsubq_f32(sr_low_f32, shiftR), scaleR);

        // calculate low part float
        sb_high_f32 = vmulq_f32(vsubq_f32(sb_high_f32, shiftB), scaleB);
        sg_high_f32 = vmulq_f32(vsubq_f32(sg_high_f32, shiftG), scaleG);
        sr_high_f32 = vmulq_f32(vsubq_f32(sr_high_f32, shiftR), scaleR);

        // get the result low part int32
        db_low_s32 = vcvtq_s32_f32(sb_low_f32);
        dg_low_s32 = vcvtq_s32_f32(sg_low_f32);
        dr_low_s32 = vcvtq_s32_f32(sr_low_f32);

        // get the result high part int32
        db_high_s32 = vcvtq_s32_f32(sb_high_f32);
        dg_high_s32 = vcvtq_s32_f32(sg_high_f32);
        dr_high_s32 = vcvtq_s32_f32(sr_high_f32);

        // get the result low part int16
        db_low_s16 = vmovn_s32(db_low_s32);
        dg_low_s16 = vmovn_s32(dg_low_s32);
        dr_low_s16 = vmovn_s32(dr_low_s32);

        // get the result high part int16
        db_high_s16 = vmovn_s32(db_high_s32);
        dg_high_s16 = vmovn_s32(dg_high_s32);
        dr_high_s16 = vmovn_s32(dr_high_s32);

        // combine low and high into int16x8
        db_s16 = vcombine_s16(db_low_s16, db_high_s16);
        dg_s16 = vcombine_s16(dg_low_s16, dg_high_s16);
        dr_s16 = vcombine_s16(dr_low_s16, dr_high_s16);

        // combine low and high into int16x8
        dbgr.val[0] = vmovn_s16(db_s16);
        dbgr.val[1] = vmovn_s16(dg_s16);
        dbgr.val[2] = vmovn_s16(dr_s16);

        // store...
        vst3_s8(dst + i * iter_size, dbgr);
    }

    /* deal with remain data not aligned */
    for (i = align_times * iter_size; i < total_size; i += 3) {
        transform_input(dst+i,   src+i,   shift[0], scale);
        transform_input(dst+i+1, src+i+1, shift[1], scale);
        transform_input(dst+i+2, src+i+2, shift[2], scale);
    }
}

/**
 * @brief softmax using CPU
 *
 *
 * @param output - pointer to target data
 * @param input - pointer to source data
 * @param size - the length of each array
 * @param scale - scale value in softmax
 *
 * @return void
 */
void softmax_one(float* output, const int8_t* input, unsigned int size, float scale) {
    float sum = 0.f;
    for (unsigned int i = 0; i < size; ++i) {
        output[i] = exp(input[i] * scale);
        sum += output[i];
    }

    for (unsigned int i = 0; i < size; ++i) output[i] /= sum;
}

/**
 * @brief batch softmax using CPU
 *
 *
 * @param output - pointer to target data
 * @param input - pointer to source data
 * @param size - the length of each array
 * @param scale - scale value in softmax
 * @param batch - how many array to be calculated
 *
 * @return void
 */
void softmax_batch(float* output, const int8_t* input, unsigned int size, float scale,
                    unsigned int batch) {
    for (unsigned int i = 0; i < batch; ++i) {
        softmax_one(output, input, size, scale);
        input += size;
        output += size;
    }
}

/**
 * @brief 4-class softmax using arm neon
 *
 * @note Assume group is divided by 8
 *
 * @param output - pointer to target data
 * @param input - pointer to source data
 * @param scale - scale value in softmax
 * @param group - how many array to be calculated
 *
 * @return void
 */
void neon_softmax4_internal(float* output, const int8_t* input, float scale, unsigned int group) {
    unsigned int batch = group / 8;

    for (unsigned int i = 0; i < batch; ++i) {
        /* Interleaved load 32 bytes into 4 NEON registers */
        int8x8x4_t q01 = vld4_s8(input);
        /* Convert to 16-bit integers */
        int16x8_t q2 = vmovl_s8(q01.val[0]);
        int16x8_t q3 = vmovl_s8(q01.val[1]);
        int16x8_t q4 = vmovl_s8(q01.val[2]);
        int16x8_t q5 = vmovl_s8(q01.val[3]);

        /* Process first 4 groups */
        int16x4_t d10 = vget_low_s16(q2);
        int16x4_t d11 = vget_low_s16(q3);
        int16x4_t d12 = vget_low_s16(q4);
        int16x4_t d13 = vget_low_s16(q5);

        float32x4_t q8 = vcvtq_f32_s32(vmovl_s16(d10));
        float32x4_t q9 = vcvtq_f32_s32(vmovl_s16(d11));
        float32x4_t q10 = vcvtq_f32_s32(vmovl_s16(d12));
        float32x4_t q11 = vcvtq_f32_s32(vmovl_s16(d13));

        q8 = exp_ps(vmulq_n_f32(q8, scale));
        q9 = exp_ps(vmulq_n_f32(q9, scale));
        q10 = exp_ps(vmulq_n_f32(q10, scale));
        q11 = exp_ps(vmulq_n_f32(q11, scale));

        float32x4_t q12 = vaddq_f32(q8, q9);
        q12 = vaddq_f32(q12, q10);
        q12 = vaddq_f32(q12, q11);
        q12 = vrecpeq_f32(q12);

        q8 = vmulq_f32(q12, q8);
        q9 = vmulq_f32(q12, q9);
        q10 = vmulq_f32(q12, q10);
        q11 = vmulq_f32(q12, q11);

        float32x4x4_t b0 = {q8, q9, q10, q11};
        vst4q_f32(output, b0);
        output += 16;

        /* Process last 4 groups */
        d10 = vget_high_s16(q2);
        d11 = vget_high_s16(q3);
        d12 = vget_high_s16(q4);
        d13 = vget_high_s16(q5);

        q8 = vcvtq_f32_s32(vmovl_s16(d10));
        q9 = vcvtq_f32_s32(vmovl_s16(d11));
        q10 = vcvtq_f32_s32(vmovl_s16(d12));
        q11 = vcvtq_f32_s32(vmovl_s16(d13));

        q8 = exp_ps(vmulq_n_f32(q8, scale));
        q9 = exp_ps(vmulq_n_f32(q9, scale));
        q10 = exp_ps(vmulq_n_f32(q10, scale));
        q11 = exp_ps(vmulq_n_f32(q11, scale));

        q12 = vaddq_f32(q8, q9);
        q12 = vaddq_f32(q12, q10);
        q12 = vaddq_f32(q12, q11);
        q12 = vrecpeq_f32(q12);

        q8 = vmulq_f32(q12, q8);
        q9 = vmulq_f32(q12, q9);
        q10 = vmulq_f32(q12, q10);
        q11 = vmulq_f32(q12, q11);

        float32x4x4_t b1 = {q8, q9, q10, q11};
        vst4q_f32(output, b1);
        output += 16;

        input += 32;
    }
}

/**
 * @brief 2-class softmax using arm neon
 *
 * @note Assume group is divided by 8
 *
 * @param output - pointer to target data
 * @param input - pointer to source data
 * @param scale - scale value in softmax
 * @param group - how many array to be calculated
 *
 * @return void
 */
void neon_softmax2_internal(float* output, const int8_t* input, float scale, unsigned int group) {
    unsigned int batch = group / 8;

    for (unsigned int i = 0; i < batch; ++i) {
        /* Interleaved load 16 bytes into 2 NEON registers */
        int8x8x2_t q0 = vld2_s8(input);
        /* Convert to 16-bit integers */
        int16x8_t q1 = vmovl_s8(q0.val[0]);
        int16x8_t q2 = vmovl_s8(q0.val[1]);

        int16x4_t d2 = vget_low_s16(q1);
        int16x4_t d3 = vget_high_s16(q1);
        int16x4_t d4 = vget_low_s16(q2);
        int16x4_t d5 = vget_high_s16(q2);

        /* Process first 4 groups */
        float32x4_t q3 = vcvtq_f32_s32(vmovl_s16(d2));
        float32x4_t q4 = vcvtq_f32_s32(vmovl_s16(d4));
        q3 = exp_ps(vmulq_n_f32(q3, scale));
        q4 = exp_ps(vmulq_n_f32(q4, scale));

        float32x4_t q7 = vaddq_f32(q3, q4);
        q7 = vrecpeq_f32(q7);
        q3 = vmulq_f32(q7, q3);
        q4 = vmulq_f32(q7, q4);

        /* Process last 4 groups */
        float32x4_t q5 = vcvtq_f32_s32(vmovl_s16(d3));
        float32x4_t q6 = vcvtq_f32_s32(vmovl_s16(d5));
        q5 = exp_ps(vmulq_n_f32(q5, scale));
        q6 = exp_ps(vmulq_n_f32(q6, scale));

        float32x4_t q8 = vaddq_f32(q5, q6);
        q8 = vrecpeq_f32(q8);
        q5 = vmulq_f32(q8, q5);
        q6 = vmulq_f32(q8, q6);

        /* Save to memory */
        float32x4x2_t b0 = {q3, q4};
        vst2q_f32(output, b0);
        output += 8;
        float32x4x2_t b1 = {q5, q6};
        vst2q_f32(output, b1);
        output += 8;

        input += 16;
    }
}

/**
 * @brief 4-class softmax
 *
 * @param output - pointer to target data
 * @param input - pointer to source data
 * @param scale - scale value in softmax
 * @param group - how many array to be calculated
 *
 * @return void
 */
void neon_softmax4(float* output, const int8_t* input, float scale, unsigned int group) {
    unsigned int aligned = group & (-8);
    neon_softmax4_internal(output, input, scale, aligned);
    unsigned int remain = group - aligned;
    input += (4 * aligned);
    output += (4 * aligned);
    softmax_batch(output, input, 4, scale, remain);
}

/**
 * @brief 2-class softmax
 *
 * @param output - pointer to target data
 * @param input - pointer to source data
 * @param scale - scale value in softmax
 * @param group - how many array to be calculated
 *
 * @return void
 */
void neon_softmax2(float* output, const int8_t* input, float scale, unsigned int group) {
    unsigned int aligned = group & (-8);
    neon_softmax2_internal(output, input, scale, aligned);
    unsigned int remain = group - aligned;
    input += (2 * aligned);
    output += (2 * aligned);
    softmax_batch(output, input, 2, scale, remain);
}
