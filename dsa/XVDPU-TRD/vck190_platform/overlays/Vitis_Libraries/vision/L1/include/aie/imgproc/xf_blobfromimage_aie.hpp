/*
 * Copyright 2021 Xilinx, Inc.
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

#include <adf.h>

#ifndef _AIE_BLOBFROMIAMGE_H_
#define _AIE_BLOBFROMIAMGE_H_

#define img_height 32
#define img_width 128

/*#define alpha   0.342
#define beta    0.0039215686274509803921568627451
#define gama    5.2432

#define threshold1 -127
#define threshold2 127*/

/**
 * ----------------------------------------------------------------------------
 * floating point blobFromImage
 * ----------------------------------------------------------------------------
*/
namespace xf {
namespace cv {
namespace aie {

enum ops { mean_sub, scale_n_clip, clip, scale_n_bias, scale_n_bias_mean_sub, fused_op };

void mean_subtraction(v8float* restrict ptr_in, v8float* restrict ptr_out, float alpha) {
    v8float data_buf1 = null_v8float();
    v8float chess_storage(WR2) alpha_acc = null_v8float();
    v8float chess_storage(WD0) data_out = null_v8float();
    for (int i = 0; i < 8; i++) {
        alpha_acc = upd_elem(alpha_acc, i, alpha);
    }
    for (int j = 0; j < (img_height * img_width); j += 8) // 8x samples per loop
        chess_prepare_for_pipelining chess_loop_range(14, ) {
            data_buf1 = *(ptr_in++); // in:00++8|_________|_________|_________
            data_out = fpsub(data_buf1, concat(alpha_acc, undef_v8float()), 0, 0x76543210);
            *(ptr_out++) = (v8float)data_out;
        }
}

void clip_fun(v8float* restrict ptr_in, v8float* restrict ptr_out, int th1, int th2) {
    v8float data_buf1 = null_v8float();
    v8float chess_storage(WR2) thresh1_acc = null_v8float();
    v8float chess_storage(WR3) thresh2_acc = null_v8float();

    v8float chess_storage(WD0) temp_out = null_v8float();
    v8float chess_storage(WD1) data_out = null_v8float();

    for (int i = 0; i < 8; i++) {
        thresh1_acc = upd_elem(thresh1_acc, i, th1);
        thresh2_acc = upd_elem(thresh2_acc, i, th2);
    }
    for (int j = 0; j < (img_height * img_width); j += 8) // 8x samples per loop
        chess_prepare_for_pipelining chess_loop_range(14, ) {
            data_buf1 = *(ptr_in++); // in:00++8|_________|_________|_________
            temp_out = fpmax(thresh1_acc, concat(data_buf1, undef_v8float()), 0, 0x76543210);
            data_out = fpmin(thresh2_acc, concat(temp_out, undef_v8float()), 0, 0x76543210);
            *(ptr_out++) = (v8float)data_out;
        }
}
void scale_n_bias_fun(v8float* restrict ptr_in, v8float* restrict ptr_out, float beta, float gama) {
    v8float data_buf1 = null_v8float();
    v8float bias_acc = null_v8float();
    v8float scale = null_v8float();

    v8float chess_storage(WD0) data_out = null_v8float();

    for (int i = 0; i < 8; i++) {
        scale = upd_elem(scale, i, beta);
        bias_acc = upd_elem(bias_acc, i, gama);
    }
    for (int j = 0; j < (img_height * img_width); j += 8) // 8x samples per loop
        chess_prepare_for_pipelining chess_loop_range(14, ) {
            data_buf1 = *(ptr_in++); // in:00++8|_________|_________|_________
            data_out = fpmac(bias_acc, concat(data_buf1, undef_v8float()), 0, 0x76543210, scale, 0, 0x76543210);
            *(ptr_out++) = (v8float)data_out;
        }
}

void scale_n_clip_fun(v8float* restrict ptr_in, v8float* restrict ptr_out, float beta, int th1, int th2) {
    v8float data_buf1 = null_v8float();
    v8float scale = null_v8float();
    v8float bias_acc = null_v8float();

    v8float chess_storage(WR2) thresh1_acc = null_v8float();
    v8float chess_storage(WR3) thresh2_acc = null_v8float();

    v8float chess_storage(WD0) temp_out = null_v8float();
    v8float chess_storage(WD1) data_out = null_v8float();

    v8float* restrict ptr_out_temp = ptr_out;

    for (int i = 0; i < 8; i++) {
        scale = upd_elem(scale, i, beta);
        thresh1_acc = upd_elem(thresh1_acc, i, th1);
        thresh2_acc = upd_elem(thresh2_acc, i, th2);
    }
    for (int j = 0; j < (img_height * img_width); j += 8) // 8x samples per loop
        chess_prepare_for_pipelining chess_loop_range(14, ) {
            data_buf1 = *(ptr_in++); // in:00++8|_________|_________|_________
            temp_out = fpmac(bias_acc, concat(data_buf1, undef_v8float()), 0, 0x76543210, scale, 0, 0x76543210);
            *(ptr_out_temp++) = (v8float)data_out;
        }

    ptr_out_temp = ptr_out_temp - ((img_height * img_width) / 8);

    for (int j = 0; j < (img_height * img_width); j += 8) // 8x samples per loop
        chess_prepare_for_pipelining chess_loop_range(14, ) {
            data_buf1 = *(ptr_out_temp++); // in:00++8|_________|_________|_________

            temp_out = fpmax(thresh1_acc, concat(data_buf1, undef_v8float()), 0, 0x76543210);
            data_out = fpmin(thresh2_acc, concat(temp_out, undef_v8float()), 0, 0x76543210);
            *(ptr_out++) = (v8float)data_out;
        }
}
void scale_n_bias_mean_sub_fun(
    v8float* restrict ptr_in, v8float* restrict ptr_out, float alpha, float beta, float gama) {
    v8float chess_storage(WR2) data_buf1 = null_v8float();
    v8float chess_storage(WR3) bias_acc = null_v8float();
    v8float scale = null_v8float();
    v8float alpha_acc = null_v8float();

    v8float chess_storage(WD0) temp_out = null_v8float();
    v8float chess_storage(WD1) data_out = null_v8float();

    for (int i = 0; i < 8; i++) {
        alpha_acc = upd_elem(alpha_acc, i, alpha);
        scale = upd_elem(scale, i, beta);
        bias_acc = upd_elem(bias_acc, i, gama);
    }
    for (int j = 0; j < (img_height * img_width); j += 8) // 8x samples per loop
        chess_prepare_for_pipelining chess_loop_range(14, ) {
            data_buf1 = *(ptr_in++); // in:00++8|_________|_________|_________
            temp_out = fpsub(data_buf1, concat(alpha_acc, undef_v8float()), 0, 0);
            data_out = fpmac(bias_acc, concat(temp_out, undef_v8float()), 0, 0x76543210, scale, 0, 0x76543210);
            *(ptr_out++) = (v8float)data_out;
        }
}
void fused_op_fun(
    v8float* restrict ptr_in, v8float* restrict ptr_out, float alpha, float beta, float gama, int th1, int th2) {
    v8float* restrict ptr_out_temp = ptr_out;

    v8float chess_storage(WR2) data_buf1 = null_v8float();
    v8float data_buf = null_v8float();

    v8float bias_acc = null_v8float();
    v8float scale = null_v8float();
    v8float alpha_acc = null_v8float();

    v8float chess_storage(WD0) temp_out = null_v8float();
    v8float chess_storage(WD1) data_out = null_v8float();

    for (int i = 0; i < 8; i++) {
        alpha_acc = upd_elem(alpha_acc, i, alpha);
        bias_acc = upd_elem(bias_acc, i, gama);
        scale = upd_elem(scale, i, beta);
    }
    for (int j = 0; j < (img_height * img_width); j += 8) // 8x samples per loop
        chess_prepare_for_pipelining chess_loop_range(14, ) {
            data_buf1 = *(ptr_in++); // in:00++8|_________|_________|_________

            temp_out = fpsub(data_buf1, concat(alpha_acc, undef_v8float()), 0, 0);
            data_out = fpmac(bias_acc, concat(temp_out, undef_v8float()), 0, 0x76543210, scale, 0, 0x76543210);
            *(ptr_out_temp++) = (v8float)data_out;
        }

    ptr_out_temp = ptr_out_temp - ((img_height * img_width) / 8);

    v8float chess_storage(WR2) thresh1_acc = null_v8float();
    v8float chess_storage(WR3) thresh2_acc = null_v8float();
    for (int i = 0; i < 8; i++) {
        thresh1_acc = upd_elem(thresh1_acc, i, th1);
        thresh2_acc = upd_elem(thresh2_acc, i, th2);
    }

    for (int j = 0; j < (img_height * img_width); j += 8) // 8x samples per loop
        chess_prepare_for_pipelining chess_loop_range(14, ) {
            data_buf = *(ptr_out_temp++); // in:00++8|_________|_________|_________

            temp_out = fpmax(thresh1_acc, concat(data_buf, undef_v8float()), 0, 0x76543210);
            data_out = fpmin(thresh2_acc, concat(temp_out, undef_v8float()), 0, 0x76543210);

            *(ptr_out++) = (v8float)data_out;
        }
}

// void blobFromImage( input_window_float * img_in, output_window_float * restrict img_out,float alpha, float beta,
// float gama,int threshold1,int threshold2)
void blobFromImage_api(input_window_float* img_in, output_window_float* img_out)
// void blobFromImage( input_window_float * img_in, output_window_float * img_out)
{
    float alpha = 0.342;
    float beta = 0.0039215686274509803921568627451;
    float gama = 5.2432;
    int threshold1 = -127;
    int threshold2 = 127;

    v8float* restrict ptr_img_buffer = (v8float*)img_in->ptr;
    v8float* restrict ptr_out_buffer = (v8float*)img_out->ptr;

    v8float* restrict ptr_in = (v8float*)ptr_img_buffer;
    v8float* restrict ptr_out = (v8float*)ptr_out_buffer;

    switch (OPMODE) {
        case mean_sub:
            mean_subtraction(ptr_in, ptr_out, alpha);
            break;
        case scale_n_clip:
            scale_n_clip_fun(ptr_in, ptr_out, beta, threshold1, threshold2);
            break;
        case clip:
            clip_fun(ptr_in, ptr_out, threshold1, threshold2);
            break;
        case scale_n_bias:
            scale_n_bias_fun(ptr_in, ptr_out, beta, gama);
            break;
        case scale_n_bias_mean_sub:
            scale_n_bias_mean_sub_fun(ptr_in, ptr_out, alpha, beta, gama);
            break;
        case fused_op:
            fused_op_fun(ptr_in, ptr_out, alpha, beta, gama, threshold1, threshold2);
            break;
    }
}

} // aie
} // cv
} // xf
#endif
