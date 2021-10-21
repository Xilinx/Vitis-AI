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
#include <aie_api/aie.hpp>
#include <common/xf_aie_utils.hpp>

#ifndef _AIE_CVT_COLOR_H_
#define _AIE_CVT_COLOR_H_

namespace xf {
namespace cv {
namespace aie {

/****************************************************************************
 * 	CalculateY - calculates the Y(luma) component using R,G,B values
 * 	Y = (0.257 * R) + (0.504 * G) + (0.098 * B) + 16
 * 	An offset of 16 is added to the resultant value
 ***************************************************************************/
// int16_t y_wei[16]={ 8422, 16516, 3212, 0};

/* const int16_t R_WEI=float2fix(0.257,11) ;
 const int16_t G_WEI=float2fix(0.504,11) ;
 const int16_t B_WEI=float2fix(0.098,11) ;
 const int16_t  WEI=float2fix(0.5,11) ;

 printf("weights are %d %d  %d\n",R_WEI,G_WEI,B_WEI,WEI);*/

int16_t y_wei[16] = {526, 1032, 201, 2048};
int16_t const_val1[16] = {16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16};
int16_t rounding_val[16] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
                            1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};

template <typename T, int N>
void calculate_Y(const T* restrict ptr1,
                 const T* restrict ptr2,
                 const T* restrict ptr3,
                 T* restrict ptr_out1,
                 const T& img_width,
                 const T& img_height) {
    ::aie::vector<T, N> data_buf1, data_buf2, data_buf3, data_out, round_buff;
    ::aie::vector<T, N> const_val(16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16);
    ::aie::vector<T, N> weights(526, 1032);
    ::aie::vector<T, N> weights1(201, 2048);
    ::aie::accum<acc32, N> acc;
    acc.from_vector(round_buff);

    for (int i = 0; i < (img_height * img_width); i += N) chess_prepare_for_pipelining chess_loop_range(14, ) {
            data_buf1 = ::aie::load_v<N>(ptr1);
            ptr1 += N;
            data_buf2 = ::aie::load_v<N>(ptr2);
            ptr2 += N;
            acc = ::aie::accumulate<N>(acc, weights, 0, data_buf1, data_buf2);
            data_buf3 = ::aie::load_v<N>(ptr3);
            ptr3 += N;
            acc = ::aie::accumulate<N>(acc, weights1, 0, data_buf3, const_val);
            ::aie::store_v(ptr_out1, acc.template to_vector<T>(11));
            ptr_out1 += N;
            acc.from_vector(round_buff);
        }
}

void calculate_Y_api(input_window_int16* ptr1_img_buffer,
                     input_window_int16* ptr2_img_buffer,
                     input_window_int16* ptr3_img_buffer,
                     output_window_int16* ptr_out1) {
    int16_t* r_in_ptr = (int16_t*)ptr1_img_buffer->ptr;
    int16_t* g_in_ptr = (int16_t*)ptr2_img_buffer->ptr;
    int16_t* b_in_ptr = (int16_t*)ptr3_img_buffer->ptr;

    int16_t* y_out_ptr = (int16_t*)ptr_out1->ptr;

    const int16_t img_width = xfcvGetTileWidth(r_in_ptr);
    const int16_t img_height = xfcvGetTileHeight(r_in_ptr);

    xfcvCopyMetaData(r_in_ptr, y_out_ptr);
    xfcvUnsignedSaturation(y_out_ptr);

    int16* restrict ptr1 = xfcvGetImgDataPtr(r_in_ptr);
    int16* restrict ptr2 = xfcvGetImgDataPtr(g_in_ptr);
    int16* restrict ptr3 = xfcvGetImgDataPtr(b_in_ptr);
    int16* restrict data_out = xfcvGetImgDataPtr(y_out_ptr);

    calculate_Y<int16_t, 16>(ptr1, ptr2, ptr3, data_out, img_width, img_height);
}
/***********************************************************************
*      CalculateU - calculates the U(Chroma) component using R,G,B values
*      U = -(0.148 * R) - (0.291 * G) + (0.439 * B) + 128
*      an offset of 128 is added to the resultant value
**********************************************************************/
/***********************************************************************
*      CalculateV - calculates the V(Chroma) component using R,G,B values
*      V = (0.439 * R) - (0.368 * G) - (0.071 * B) + 128
*      an offset of 128 is added to the resultant value
**********************************************************************/
/*    const int16_t VR_WEI=float2fix(0.439,7) ;
 const int16_t VG_WEI=float2fix(-0.368,7) ;
 const int16_t VB_WEI=float2fix(-0.071,7) ;
 const int16_t  V_WEI=float2fix(0.5,7) ;

 printf("weights are %d %d  %d\n",VR_WEI,VG_WEI,VB_WEI,V_WEI);
  const int16_t UR_WEI=float2fix(-0.148,7) ;
 const int16_t UG_WEI=float2fix(-0.291,7) ;
 const int16_t UB_WEI=float2fix(0.439,7) ;
 const int16_t  U_WEI=float2fix(0.5,7) ;
 printf("weights are %d %d  %d\n",UR_WEI,UG_WEI,UB_WEI,U_WEI);*/

int16_t UV_wei[16] = {-19, 0, -37, 0, 56, 0, 1, 0, 56, 0, -47, 0, -9, 0, 1, 0};
int16_t weight[16] = {16448, 16448, 16448, 16448, 16448, 16448, 16448, 16448,
                      16448, 16448, 16448, 16448, 16448, 16448, 16448, 16448};
// int16_t UV_wei[16]={  -38, 0, -74, 0, 112, 0, 256, 0 , 112, 0, -94, 0, -18, 0, 256, 0 };
// int16_t weight[16]={  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128 };

template <typename T, int N>
void calculate_UV(const T* restrict ptr1,
                  const T* restrict ptr2,
                  const T* restrict ptr3,
                  T* restrict ptr_out2,
                  T* restrict ptr_out3,
                  const T& img_width,
                  const T& img_height) {
    constexpr unsigned Lanes = 16;
    constexpr unsigned Points = 2;
    constexpr unsigned CoeffStep = 1;
    constexpr unsigned DataStepY = 2;

    using mul_ops = ::aie::sliding_mul_y_ops<Lanes, Points, CoeffStep, DataStepY, int16, int16>;

    ::aie::vector<int16_t, 16> kernel_coeff(-19, 0, -37, 0, 56, 0, 1, 0, 56, 0, -47, 0, -9, 0, 1, 0);
    ::aie::accum<acc48, 16> acc_u, acc_v;
    ::aie::vector<T, 32> data_buf1;
    ::aie::vector<T, 32> data_buf2;
    ::aie::vector<T, 16> weights(16448, 16448, 16448, 16448, 16448, 16448, 16448, 16448, 16448, 16448, 16448, 16448,
                                 16448, 16448, 16448, 16448);

    for (int i = 0; i < img_height; i += 2) chess_prepare_for_pipelining chess_loop_range(16, ) {
            for (int j = 0; j < img_width; j += (2 * 16)) chess_prepare_for_pipelining chess_loop_range(4, ) {
                    data_buf1.insert(0, ::aie::load_v<16>(ptr1));
                    ptr1 += 16;
                    data_buf1.insert(1, ::aie::load_v<16>(ptr1));
                    ptr1 += 16; //          |   loading R channel
                    acc_u = mul_ops::mul(kernel_coeff, 0, data_buf1, 0);
                    acc_v = mul_ops::mul(kernel_coeff, 8, data_buf1, 0);

                    data_buf2.insert(0, ::aie::load_v<16>(ptr2));
                    ptr2 += 16;
                    data_buf2.insert(1, ::aie::load_v<16>(ptr2));
                    ptr2 += 16;
                    acc_u = mul_ops::mac(acc_u, kernel_coeff, 2, data_buf2, 0);
                    acc_v = mul_ops::mac(acc_v, kernel_coeff, 10, data_buf2, 0);

                    data_buf1.insert(0, ::aie::load_v<16>(ptr3));
                    ptr3 += 16;
                    data_buf1.insert(1, ::aie::load_v<16>(ptr3));
                    ptr3 += 16;
                    acc_u = mul_ops::mac(acc_u, kernel_coeff, 4, data_buf1, 0);
                    acc_v = mul_ops::mac(acc_v, kernel_coeff, 12, data_buf1, 0);

                    data_buf2.insert(0, weights);
                    data_buf2.insert(1, weights);
                    acc_u = mul_ops::mac(acc_u, kernel_coeff, 6, data_buf2, 0);
                    acc_v = mul_ops::mac(acc_v, kernel_coeff, 14, data_buf2, 0);

                    ::aie::store_v(ptr_out2, acc_u.template to_vector<int16>(7));
                    ::aie::store_v(ptr_out3, acc_v.template to_vector<int16>(7));
                    ptr_out2 += 16;
                    ptr_out3 += 16;
                }

            ptr1 += img_width;
            ptr2 += img_width;
            ptr3 += img_width;
        }
}

void calculate_UV_api(input_window_int16* ptr1_img_buffer,
                      input_window_int16* ptr2_img_buffer,
                      input_window_int16* ptr3_img_buffer,
                      output_window_int16* ptr_out2,
                      output_window_int16* ptr_out3) {
    int16_t* r_in_ptr = (int16_t*)ptr1_img_buffer->ptr;
    int16_t* g_in_ptr = (int16_t*)ptr2_img_buffer->ptr;
    int16_t* b_in_ptr = (int16_t*)ptr3_img_buffer->ptr;

    int16_t* u_out_ptr = (int16_t*)ptr_out2->ptr;
    int16_t* v_out_ptr = (int16_t*)ptr_out3->ptr;

    const int16_t img_width = xfcvGetTileWidth(g_in_ptr);
    const int16_t img_height = xfcvGetTileHeight(b_in_ptr);

    xfcvCopyMetaData(g_in_ptr, u_out_ptr);
    xfcvCopyMetaData(b_in_ptr, v_out_ptr);
    xfcvUnsignedSaturation(u_out_ptr);
    xfcvUnsignedSaturation(v_out_ptr);
    xfcvSetUVMetaData(u_out_ptr);
    xfcvSetUVMetaData(v_out_ptr);

    int16* restrict ptr1 = xfcvGetImgDataPtr(r_in_ptr);
    int16* restrict ptr2 = xfcvGetImgDataPtr(g_in_ptr);
    int16* restrict ptr3 = xfcvGetImgDataPtr(b_in_ptr);

    int16* restrict data_out2 = xfcvGetImgDataPtr(u_out_ptr);
    int16* restrict data_out3 = xfcvGetImgDataPtr(v_out_ptr);

    calculate_UV<int16_t, 16>(ptr1, ptr2, ptr3, data_out2, data_out3, img_width, img_height);
}

void cvtcolor_api(input_window_int16* img_r,
                  input_window_int16* img_g,
                  input_window_int16* img_b,
                  output_window_int16* img_y,
                  output_window_int16* img_u,
                  output_window_int16* img_v) {
    calculate_Y_api(img_r, img_g, img_b, img_y);
    calculate_UV_api(img_r, img_g, img_b, img_u, img_v);
}

} // aie
} // cv
} // xf
#endif
