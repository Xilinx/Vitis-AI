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

#ifndef _XF_KALMANFILTER_HPP_
#define _XF_KALMANFILTER_HPP_

#define DEBUG 0
#include "common/xf_common.hpp"
#include "ap_int.h"

namespace xf {
namespace cv {
template <int PROC>
float KF_dotProduct(float dot_in1[PROC], float dot_in2[PROC]) {
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=dot_in1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=dot_in2 complete dim=1
    #pragma HLS inline off
    // clang-format on

    enum {
        TA_L1 = (PROC / 2 + (((PROC % 2) != 0) & (PROC != 1))),
        TA_L2 = (TA_L1 / 2 + (((TA_L1 % 2) != 0) & (TA_L1 != 1))),
        TA_L3 = (TA_L2 / 2 + (((TA_L2 % 2) != 0) & (TA_L2 != 1))),
        TA_L4 = (TA_L3 / 2 + (((TA_L3 % 2) != 0) & (TA_L3 != 1))),
        TA_L5 = (TA_L4 / 2 + (((TA_L4 % 2) != 0) & (TA_L4 != 1))),
        TA_L6 = (TA_L5 / 2 + (((TA_L5 % 2) != 0) & (TA_L5 != 1))),
        TA_L7 = (TA_L6 / 2 + (((TA_L6 % 2) != 0) & (TA_L6 != 1))),
        TA_L8 = (TA_L7 / 2 + (((TA_L7 % 2) != 0) & (TA_L7 != 1)))
    };

    float mul_out[PROC];
    for (ap_uint<10> idx = 0; idx < PROC; idx++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        mul_out[idx] = dot_in1[idx] * dot_in2[idx];
    }

    float add1_out[TA_L1];
    float add2_out[TA_L2];
    float add3_out[TA_L3];
    float add4_out[TA_L4];
    float add5_out[TA_L5];
    float add6_out[TA_L6];
    float add7_out[TA_L7];
    float add8_out[TA_L8];

    if (TA_L1 != 0) {
        for (ap_uint<10> idx = 0; idx < TA_L1; idx++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            if (idx == TA_L1 - 1 && PROC % 2 == 1)
                add1_out[idx] = mul_out[2 * idx];
            else
                add1_out[idx] = mul_out[2 * idx] + mul_out[2 * idx + 1];
        }
    }

    if (TA_L2 != 0) {
        for (ap_uint<10> idx = 0; idx < TA_L2; idx++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            if (idx == TA_L2 - 1 && TA_L1 % 2 == 1)
                add2_out[idx] = add1_out[2 * idx];
            else
                add2_out[idx] = add1_out[2 * idx] + add1_out[2 * idx + 1];
        }
    }

    if (TA_L3 != 0) {
        for (ap_uint<10> idx = 0; idx < TA_L3; idx++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            if (idx == TA_L3 - 1 && TA_L2 % 2 == 1)
                add3_out[idx] = add2_out[2 * idx];
            else
                add3_out[idx] = add2_out[2 * idx] + add2_out[2 * idx + 1];
        }
    }

    if (TA_L4 != 0) {
        for (ap_uint<10> idx = 0; idx < TA_L4; idx++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            if (idx == TA_L4 - 1 && TA_L3 % 2 == 1)
                add4_out[idx] = add3_out[2 * idx];
            else
                add4_out[idx] = add3_out[2 * idx] + add3_out[2 * idx + 1];
        }
    }

    if (TA_L5 != 0) {
        for (ap_uint<10> idx = 0; idx < TA_L5; idx++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            if (idx == TA_L5 - 1 && TA_L4 % 2 == 1)
                add5_out[idx] = add4_out[2 * idx];
            else
                add5_out[idx] = add4_out[2 * idx] + add4_out[2 * idx + 1];
        }
    }

    if (TA_L6 != 0) {
        for (ap_uint<10> idx = 0; idx < TA_L6; idx++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            if (idx == TA_L6 - 1 && TA_L5 % 2 == 1)
                add6_out[idx] = add5_out[2 * idx];
            else
                add6_out[idx] = add5_out[2 * idx] + add5_out[2 * idx + 1];
        }
    }

    if (TA_L7 != 0) {
        for (ap_uint<10> idx = 0; idx < TA_L7; idx++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            if (idx == TA_L7 - 1 && TA_L6 % 2 == 1)
                add7_out[idx] = add6_out[2 * idx];
            else
                add7_out[idx] = add6_out[2 * idx] + add6_out[2 * idx + 1];
        }
    }

    if (TA_L8 != 0) {
        for (ap_uint<10> idx = 0; idx < TA_L8; idx++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            if (idx == TA_L8 - 1 && TA_L7 % 2 == 1)
                add8_out[idx] = add7_out[2 * idx];
            else
                add8_out[idx] = add7_out[2 * idx] + add7_out[2 * idx + 1];
        }
    }

    float add_out;
    if (TA_L1 == 1)
        add_out = add1_out[0];
    else if (TA_L2 == 1)
        add_out = add2_out[0];
    else if (TA_L3 == 1)
        add_out = add3_out[0];
    else if (TA_L4 == 1)
        add_out = add4_out[0];
    else if (TA_L5 == 1)
        add_out = add5_out[0];
    else if (TA_L6 == 1)
        add_out = add6_out[0];
    else if (TA_L7 == 1)
        add_out = add7_out[0];
    else if (TA_L8 == 1)
        add_out = add8_out[0];
    else
        add_out = mul_out[0];

    return (add_out);
}

template <int DEPTH>
void KF_treeAdder(float in1[DEPTH], float* output) {
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=in1 complete dim=1
    #pragma HLS inline
    // clang-format on
    enum {
        TA_L1 = (DEPTH / 2 + (((DEPTH % 2) != 0) & (DEPTH != 1))),
        TA_L2 = (TA_L1 / 2 + (((TA_L1 % 2) != 0) & (TA_L1 != 1))),
        TA_L3 = (TA_L2 / 2 + (((TA_L2 % 2) != 0) & (TA_L2 != 1))),
        TA_L4 = (TA_L3 / 2 + (((TA_L3 % 2) != 0) & (TA_L3 != 1))),
        TA_L5 = (TA_L4 / 2 + (((TA_L4 % 2) != 0) & (TA_L4 != 1))),
        TA_L6 = (TA_L5 / 2 + (((TA_L5 % 2) != 0) & (TA_L5 != 1))),
        TA_L7 = (TA_L6 / 2 + (((TA_L6 % 2) != 0) & (TA_L6 != 1))),
        TA_L8 = (TA_L7 / 2 + (((TA_L7 % 2) != 0) & (TA_L7 != 1)))
    };

    float add1_out[TA_L1];
    float add2_out[TA_L2];
    float add3_out[TA_L3];
    float add4_out[TA_L4];
    float add5_out[TA_L5];
    float add6_out[TA_L6];
    float add7_out[TA_L7];
    float add8_out[TA_L8];

    if (TA_L1 != 0) {
        for (ap_uint<10> idx = 0; idx < TA_L1; idx++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            if (idx == TA_L1 - 1 && DEPTH % 2 == 1)
                add1_out[idx] = in1[2 * idx];
            else
                add1_out[idx] = in1[2 * idx] + in1[2 * idx + 1];
        }
    }

    if (TA_L2 != 0) {
        for (ap_uint<10> idx = 0; idx < TA_L2; idx++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            if (idx == TA_L2 - 1 && TA_L1 % 2 == 1)
                add2_out[idx] = add1_out[2 * idx];
            else
                add2_out[idx] = add1_out[2 * idx] + add1_out[2 * idx + 1];
        }
    }

    if (TA_L3 != 0) {
        for (ap_uint<10> idx = 0; idx < TA_L3; idx++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            if (idx == TA_L3 - 1 && TA_L2 % 2 == 1)
                add3_out[idx] = add2_out[2 * idx];
            else
                add3_out[idx] = add2_out[2 * idx] + add2_out[2 * idx + 1];
        }
    }

    if (TA_L4 != 0) {
        for (ap_uint<10> idx = 0; idx < TA_L4; idx++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            if (idx == TA_L4 - 1 && TA_L3 % 2 == 1)
                add4_out[idx] = add3_out[2 * idx];
            else
                add4_out[idx] = add3_out[2 * idx] + add3_out[2 * idx + 1];
        }
    }

    if (TA_L5 != 0) {
        for (ap_uint<10> idx = 0; idx < TA_L5; idx++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            if (idx == TA_L5 - 1 && TA_L4 % 2 == 1)
                add5_out[idx] = add4_out[2 * idx];
            else
                add5_out[idx] = add4_out[2 * idx] + add4_out[2 * idx + 1];
        }
    }

    if (TA_L6 != 0) {
        for (ap_uint<10> idx = 0; idx < TA_L6; idx++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            if (idx == TA_L6 - 1 && TA_L5 % 2 == 1)
                add6_out[idx] = add5_out[2 * idx];
            else
                add6_out[idx] = add5_out[2 * idx] + add5_out[2 * idx + 1];
        }
    }

    if (TA_L7 != 0) {
        for (ap_uint<10> idx = 0; idx < TA_L7; idx++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            if (idx == TA_L7 - 1 && TA_L6 % 2 == 1)
                add7_out[idx] = add6_out[2 * idx];
            else
                add7_out[idx] = add6_out[2 * idx] + add6_out[2 * idx + 1];
        }
    }

    if (TA_L8 != 0) {
        for (ap_uint<10> idx = 0; idx < TA_L8; idx++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            if (idx == TA_L8 - 1 && TA_L7 % 2 == 1)
                add8_out[idx] = add7_out[2 * idx];
            else
                add8_out[idx] = add7_out[2 * idx] + add7_out[2 * idx + 1];
        }
    }

    float add_out;
    if (TA_L1 == 1)
        add_out = add1_out[0];
    else if (TA_L2 == 1)
        add_out = add2_out[0];
    else if (TA_L3 == 1)
        add_out = add3_out[0];
    else if (TA_L4 == 1)
        add_out = add4_out[0];
    else if (TA_L5 == 1)
        add_out = add5_out[0];
    else if (TA_L6 == 1)
        add_out = add6_out[0];
    else if (TA_L7 == 1)
        add_out = add7_out[0];
    else if (TA_L8 == 1)
        add_out = add8_out[0];
    else
        add_out = in1[0];

    *output = (add_out);
}

template <int PROC>
void KF_scaleSub(float in1[PROC], float scale, float in2[PROC], float out[PROC]) {
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=in1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=in2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=out complete dim=1
    #pragma HLS inline off
    // clang-format on

    float scale_neg = -scale;

    for (int idx = 0; idx < PROC; idx++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        out[idx] = in1[idx] + (scale_neg * in2[idx]);
    }
}

template <int PROC>
void KF_scale(float in[PROC], float scale, float out[PROC]) {
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=in complete dim=1
    #pragma HLS ARRAY_PARTITION variable=out complete dim=1
    #pragma HLS inline off
    // clang-format on
    for (int idx = 0; idx < PROC; idx++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        out[idx] = scale * in[idx];
    }
}

template <int PROC>
void KF_add(float in1[PROC], float in2[PROC], float out[PROC]) {
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=in1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=in2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=out complete dim=1
    #pragma HLS inline off
    // clang-format on
    for (int idx = 0; idx < PROC; idx++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        out[idx] = in1[idx] + in2[idx];
    }
}

template <int N_STATE, int TYPE, int NPC>
void KF_X_write(float xu_vector[512], xf::cv::Mat<TYPE, N_STATE, 1, NPC>& Xout_mat) {
// clang-format off
    #pragma HLS inline off
    // clang-format on
    for (int ptr = 0; ptr < N_STATE; ptr++) {
// clang-format off
        #pragma HLS pipeline
        // clang-format on
        Xout_mat.write_float(ptr, xu_vector[ptr]);
    }
}

template <int N_STATE, int PROC_MU, int DEPTH_MU, int UMAT_DEPTH, int TYPE, int NPC>
void KF_UD_write(float U_matrix[PROC_MU][UMAT_DEPTH],
                 float D_vector[512],
                 xf::cv::Mat<TYPE, N_STATE, N_STATE, NPC>& Uout_mat,
                 xf::cv::Mat<TYPE, N_STATE, 1, NPC>& Dout_mat) {
// clang-format off
    #pragma HLS inline off
    // clang-format on

    ap_uint<32> counter1 = 0;
    ap_uint<32> counter1_1 = 0; // for dim2
    ap_uint<32> counter2 = 0;   // for dim1
    ap_uint<32> counter3 = 0;   // for dim2
LOOPI_U:
    for (int ptr = 0; ptr < N_STATE * N_STATE; ptr++) {
// clang-format off
        #pragma HLS pipeline
        // clang-format on

        ap_uint<8> dim1 = counter2;
        ap_uint<16> dim2 = counter1_1 + counter3;

        Uout_mat.write_float(ptr, U_matrix[dim1][dim2]);

        if (counter1 == N_STATE - 1) {
            if (counter2 == PROC_MU - 1) {
                counter2 = 0;
                counter3++;
            } else {
                counter2++;
            }

            counter1 = 0;
            counter1_1 = 0;
        } else {
            counter1++;
            counter1_1 += DEPTH_MU;
        }
    }

    for (int ptr = 0; ptr < N_STATE; ptr++) {
// clang-format off
        #pragma HLS pipeline
        // clang-format on
        Dout_mat.write_float(ptr, D_vector[ptr]);
    }
}

template <int N_STATE, int M_MEAS, int PROC_MU, int DEPTH_MU, int UMAT_DEPTH, bool URAM_EN, bool EKF_EN>
void MeasUpdate_1x(float Uin_matrix[PROC_MU][UMAT_DEPTH],
                   float Din_vector[N_STATE],
                   float Uout_matrix[PROC_MU][UMAT_DEPTH],
                   float Dout_vector[N_STATE],
                   float xu_vector[512],
                   float h_vector[PROC_MU][DEPTH_MU],
                   float r_value,
                   float z_value,
                   bool UDX_en) {
    // clang-format off
    //** comment for Ubar Dbar
    //    f1= h(1) & g1 = f1*D(1) & a0 = r
    //  ---------------------------
    //                  <a0,a1>       | Dbar(1) = D(1)*a0/a1            <a1,a2>      | Dbar(2) = D(2)*a1/a2         <a2,a3>     |..| Dbar(n) = D(n)*a(n-1)/a(n)
    // f1'=f1/a0        <f1',f2,g2>   | Ubar1= U1-f1'k1 -> f2'=f2/a1    <f2',f3,g3>  | Ubar2= U2-f2'k2 -> f3'=f3/a2 <f3',f4,g4> |..| Ubar(n)= Un-fn'kn -> f(n+1)'=f(n+1)/a(n)
    // k1 = {0,0..0}    <k1,g1>       | k2 = k1 + g1U1                    <k2,g2>    | k3 = k2 + g2U2               <k3,g3>     |..| k(n+1) = k(n) + g(n+1)U(n+1)_sv
    // f2 = U2*h                      | f3 = U3*h                                    | f4 = U4*h                                |..| f(n+2) = mulAcc
    // g2 = f2*D(2)                   | g3 = f3*D(3)                                 | g4 = f4*D(4)                             |..| g(n+2) = f(n+2)*g(n+2)
    // a1 = a0 + f1g1                 | a2 = a1 + f2g2                               | a3 = a2 + f3g3                           |..| a(n+1) = a(n) + f(n+1)g(n+1)
    //##############################################################################################################
    // a0 pass                        | a1 pass                                      | a2 pass                                  |..|
    // a1 compute/pass                | a2 compute/pass                              | a3 compute/pass                          |..|
    //###############################################################################################################
    // a_prev=a0,a_up=a1<a_prev,a_up> | Dbar(1)=D(1)*a_prev/a_up     <a_prev,a_up>   | Dbar(2)=D(2)*a_prev/a_up                 |..| Dbar(n)=D(n)*a_prev/a_up
    // f'=f1/a_prev   <f',f_nex,g_nex>| Ubar1=U1-f'K ->f'=f_nex/a_up <f',f_nex,g_nex>| Ubar2=U2-f'K -> f'=f/a_up                |..| Ubar(n)=U(n)-f'K -> f'=f/a_up
    // k=k1            <K,g>          | K= K + g*U1                  <k,g>           | K= K + g*U2                              |..| K= K + g*U2
    // g=g1                           | /*a1*/a_prev = a_up                          | /*a2*/a_prev = a_up                      |..| /*a(n)*/a_prev = a_up
    //                                | /*a2*/a_up   = a_up +f_nex*g_nex             | /*a3*/a_up   = a_up + f_nex*g_nex        |..| /*a(n+1)*/a_up   = a_up +f_nex*g_nex
    //                                | g = g_nex                                    | g = g_nex                                |..| g = g_nex
    // f=f2                           | f_nex = U3*h                                 | f_nex = f4 = U4*h                        |..| f_nex= f(n+2)
    // g=g2                           | g_nex = g3 = f*D(3)                          | g_nex = g4 = f4*D(4)                     |..| g_nex= g(n+2)=D(n+4)
    //*************
    // clang-format on
    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=Uin_matrix complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=Uin_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=Uin_matrix complete dim=1
        // clang-format on
    }
    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=Uout_matrix complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=Uout_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=Uout_matrix complete dim=1
        // clang-format on
    }
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=h_vector complete dim=0
    #pragma HLS inline off
    // clang-format on

    float res = z_value;
    float f1_value = h_vector[0][0];
    float g1_value = f1_value * Din_vector[0];
    float alpha_prev = r_value;                     // alpha0
    float alpha_up = r_value + f1_value * g1_value; // alpha1
    float f_dash_div = f1_value / r_value;          // f1' = f1/alpha0
    float f_dash;
    if (UDX_en == 0)
        f_dash = 0;
    else
        f_dash = f_dash_div;

    float kg_vector[PROC_MU][DEPTH_MU];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=kg_vector complete dim=0
    // clang-format on
    for (int i = 0; i < PROC_MU; i++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        for (int j = 0; j < DEPTH_MU; j++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            kg_vector[i][j] = 0;
        }
    }

    float h_value2;
    if (PROC_MU == 1)
        h_value2 = h_vector[0][1];
    else
        h_value2 = h_vector[1][0];

    float f_nex = h_vector[0][0] * Uin_matrix[0][1 * DEPTH_MU] + h_value2;
    float g_nex = f_nex * Din_vector[1];
    float fg_nex = f_nex * g_nex;
    float g_value = g1_value;

LOOP2:
    for (int state = 0, u_offset = 0; state < N_STATE; state++, u_offset += DEPTH_MU) {
// clang-format off
        #pragma HLS pipeline II=10
        // clang-format on

        if (EKF_EN == 0) {
            //### needed for X update
            float hval = h_vector[state % PROC_MU][state / PROC_MU];
            res -= hval * xu_vector[state];
        }

        //####Dbar update
        float Din0 = Din_vector[state];
        float Din2 = Din_vector[state + 2];

        float alpha_div;
        if (UDX_en == 0)
            alpha_div = 1;
        else
            alpha_div = alpha_prev / alpha_up;

        Dout_vector[state] = Din0 * alpha_div;

        // Read col_j and col_j+2 from U matrix
        // For timing sake & II , Uin_matrix data is loaded in Uin0_col & Uin2_col
        float Uin0_col[PROC_MU][DEPTH_MU];
        float Uin2_col[PROC_MU][DEPTH_MU];
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=Uin0_col complete dim=0
        #pragma HLS ARRAY_PARTITION variable=Uin2_col complete dim=0
        // clang-format on
        for (int i = 0; i < PROC_MU; i++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            for (int j = 0; j < DEPTH_MU; j++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on

                Uin0_col[i][j] = Uin_matrix[i][j + u_offset];
#if !__SYNTHESIS__

                if ((j + u_offset + 2 * DEPTH_MU) < UMAT_DEPTH)
                    Uin2_col[i][j] = Uin_matrix[i][j + u_offset + 2 * DEPTH_MU];
                else
                    Uin2_col[i][j] = 0;
#else
                Uin2_col[i][j] = Uin_matrix[i][j + u_offset + 2 * DEPTH_MU];
#endif
            }
        }

        float tmp_kg_vector[PROC_MU][DEPTH_MU];
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=tmp_kg_vector complete dim=0
        // clang-format on
        for (int i = 0; i < PROC_MU; i++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            for (int j = 0; j < DEPTH_MU; j++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                tmp_kg_vector[i][j] = kg_vector[i][j];
            }
        }

        float Uout_col[PROC_MU][DEPTH_MU];
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=Uout_col complete dim=0
    // clang-format on
    LOOP5:
        for (ap_uint<8> u_seq = 0, var = 0; u_seq < DEPTH_MU; u_seq++, var += PROC_MU) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            float u_readchunk[PROC_MU];
            float k_readchunk[PROC_MU];
// clang-format off
            #pragma HLS ARRAY_PARTITION variable=u_readchunk complete dim=1
            #pragma HLS ARRAY_PARTITION variable=k_readchunk complete dim=1
            // clang-format on
            for (ap_uint<8> loadin = 0; loadin < PROC_MU; loadin++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                u_readchunk[loadin] = Uin0_col[loadin][u_seq];
                k_readchunk[loadin] = tmp_kg_vector[loadin][u_seq];
            }
            float u_writechunk[PROC_MU];
            KF_scaleSub<PROC_MU>(u_readchunk, f_dash, k_readchunk, u_writechunk);

            for (ap_uint<8> loadin = 0; loadin < PROC_MU; loadin++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                Uout_col[loadin][u_seq] = u_writechunk[loadin];
            }

        } // u seq loop

        for (int i = 0; i < PROC_MU; i++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            for (int j = 0; j < DEPTH_MU; j++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on

                Uout_matrix[i][j + u_offset] = Uout_col[i][j];
            }
        }

        //###f_dash calculation
        float f_dash_temp = f_nex / alpha_up;

        if (UDX_en == 0)
            f_dash = 0;
        else
            f_dash = f_dash_temp;

        //##Update Kalman gain kg_vector
        float gu_vector[PROC_MU][DEPTH_MU];
    LOOP61:
        for (ap_uint<8> k_seq = 0, var = 0; k_seq < DEPTH_MU; k_seq++, var += PROC_MU) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            float u_readchunk[PROC_MU];
            for (ap_uint<8> loadin = 0; loadin < PROC_MU; loadin++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                u_readchunk[loadin] = Uin0_col[loadin][k_seq];
            }

            float gu_writechunk[PROC_MU];
            KF_scale<PROC_MU>(u_readchunk, g_value, gu_writechunk);

            for (ap_uint<8> loadin = 0; loadin < PROC_MU; loadin++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                gu_vector[loadin][k_seq] = gu_writechunk[loadin];
            }
        } // k seq loop

    LOOP62:
        for (ap_uint<8> k_seq = 0, var = 0; k_seq < DEPTH_MU; k_seq++, var += PROC_MU) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            float k_readchunk[PROC_MU];
            float gu_readchunk[PROC_MU];
            for (ap_uint<8> loadin = 0; loadin < PROC_MU; loadin++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                k_readchunk[loadin] = tmp_kg_vector[loadin][k_seq];
                gu_readchunk[loadin] = gu_vector[loadin][k_seq];
            }

            float k_writechunk[PROC_MU];
            KF_add<PROC_MU>(k_readchunk, gu_readchunk, k_writechunk);

            for (ap_uint<8> loadin = 0; loadin < PROC_MU; loadin++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                kg_vector[loadin][k_seq] = k_writechunk[loadin];
            }
        } // k seq loop

        //### update alpha
        alpha_prev = alpha_up;
        alpha_up = alpha_up + f_nex * g_nex;

        //### f and g calculation
        g_value = g_nex;
        float dot_out[DEPTH_MU];
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=dot_out complete dim=1
    // clang-format on
    LOOP7:
        for (ap_uint<10> dot_seq = 0, var = 0; dot_seq < DEPTH_MU; dot_seq++, var += PROC_MU) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            float dot_in1[PROC_MU];
            float dot_in2[PROC_MU];
// clang-format off
            #pragma HLS ARRAY_PARTITION variable=dot_in1 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=dot_in2 complete dim=1
            // clang-format on
            for (ap_uint<10> loadin = 0; loadin < PROC_MU; loadin++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                dot_in1[loadin] = Uin2_col[loadin][dot_seq];
                dot_in2[loadin] = h_vector[loadin][dot_seq];
            } // loadin loop

            dot_out[dot_seq] = KF_dotProduct<PROC_MU>(dot_in1, dot_in2);
        } // dot seq loop

        float tmp_ta;
        KF_treeAdder<DEPTH_MU>(dot_out, &tmp_ta);
        f_nex = tmp_ta;
        g_nex = tmp_ta * Din2;

    } // state loop

    for (ap_uint<8> x_update = 0; x_update < N_STATE; x_update++) {
// clang-format off
        #pragma HLS pipeline
        // clang-format on
        float kg_temp = kg_vector[x_update % PROC_MU][x_update / PROC_MU] / alpha_prev;
        float kg;

        if (UDX_en == 0)
            kg = 0;
        else
            kg = kg_temp;

        xu_vector[x_update] = xu_vector[x_update] + kg * res;

        Dout_vector[x_update + N_STATE] = Dout_vector[x_update];
    }
}

template <int N_STATE, int PROC_TU, int DEPTH_TU, int TMAT_DEPTH, int UQMAT_DEPTH, bool URAM_EN>
void load_Uq(float T_matrix[PROC_TU][TMAT_DEPTH], float Uq_matrix[UQMAT_DEPTH]) {
    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=T_matrix complete dim=1
        #pragma HLS resource variable=T_matrix core=RAM_S2P_BRAM
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=T_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=T_matrix complete dim=1
        // clang-format on
    }

    if (URAM_EN == 1) {
// clang-format off
        #pragma HLS RESOURCE variable=Uq_matrix core=RAM_S2P_URAM
        // clang-format on
    }
// clang-format off
    #pragma HLS inline off
    // clang-format on

    ap_uint<16> counter_trow = 0;
    ap_uint<32> offset_inc = N_STATE;
LOOPI_UQ:
    for (int ptr = 0; ptr < N_STATE * N_STATE; ptr++) {
// clang-format off
        #pragma HLS pipeline
        // clang-format on

        ap_uint<32> offset = offset_inc + counter_trow;
        ap_uint<8> dim1 = offset % PROC_TU;
        ap_uint<16> dim2 = offset / PROC_TU;

        T_matrix[dim1][dim2] = Uq_matrix[ptr];

        if (counter_trow == N_STATE - 1) {
            counter_trow = 0;
            offset_inc += DEPTH_TU * PROC_TU;
        } else
            counter_trow++;
    }
}

template <int N_STATE,
          int C_CTRL,
          int M_MEAS,
          int PROC_MU,
          int DEPTH_MU,
          int UMAT_DEPTH,
          int HMAT_DEPTH,
          bool URAM_EN,
          bool EKF_EN,
          int TYPE,
          int NPC>
void MeasUpdate(float U_matrix[PROC_MU][UMAT_DEPTH],
                float H_matrix[PROC_MU][HMAT_DEPTH],
                float D_vector[512],
                float xu_vector[512],
                float ry_vector[512],
#if KF_C != 0
                xf::cv::Mat<TYPE, C_CTRL, 1, NPC>& u_mat,
#endif
                xf::cv::Mat<TYPE, M_MEAS, 1, NPC>& y_mat,
                xf::cv::Mat<TYPE, M_MEAS, 1, NPC>& R_mat,
                xf::cv::Mat<TYPE, M_MEAS, N_STATE, NPC>& H_mat,
                xf::cv::Mat<TYPE, N_STATE, 1, NPC>& Xout_mat,
                xf::cv::Mat<TYPE, N_STATE, N_STATE, NPC>& Uout_mat,
                xf::cv::Mat<TYPE, N_STATE, 1, NPC>& Dout_mat,
                bool X_write_en,
                bool UD_write_en) {
    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=U_matrix complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=U_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=U_matrix complete dim=1
        // clang-format on
    }

    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=H_matrix complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=H_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=H_matrix complete dim=1
        // clang-format on
    }

// clang-format off
    #pragma HLS inline off
    // clang-format on

    enum { M_MEAS_align2 = (M_MEAS + (M_MEAS % 2)) };

    float Uint_matrix[PROC_MU][UMAT_DEPTH];
    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=Uint_matrix complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=Uint_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=Uint_matrix complete dim=1
        // clang-format on
    }

    float Dint_vector[512];

    ap_uint<8> meas_index;
    if (EKF_EN == 1)
        meas_index = xu_vector[511];
    else
        meas_index = 0;

    float hx, Zekf, Rekf;
    //##### Read Y mesurements
    if (EKF_EN == 0) {
    LOOP1:
        for (ap_uint<8> ddr_ptr = 0; ddr_ptr < M_MEAS; ddr_ptr++) {
// clang-format off
            #pragma HLS pipeline
            // clang-format on
            ry_vector[ddr_ptr + M_MEAS] = y_mat.read_float(ddr_ptr);
        }
    } else {
        Zekf = y_mat.read_float(0);
#if KF_C != 0
        hx = u_mat.read_float(0);
#endif
        Rekf = R_mat.read_float(0);

        ap_uint<32> offset_incH = 0;
    LOOPI_H:
        for (int ptr = 0; ptr < N_STATE; ptr++) {
// clang-format off
            #pragma HLS pipeline
            // clang-format on
            ap_uint<10> dim1 = ptr % PROC_MU;
            ap_uint<16> dim2 = ptr / PROC_MU;

            H_matrix[dim1][dim2] = H_mat.read_float(ptr);
        }
    }

    bool flip = 0;

    ap_uint<8> meas_loop_cnt;
    if (EKF_EN == 1)
        meas_loop_cnt = 2;
    else
        meas_loop_cnt = M_MEAS_align2;

LOOP2:
    for (ap_uint<8> meas = 0; meas < meas_loop_cnt; meas++) {
        bool UDX_en;
        if (EKF_EN == 0) {
            if (meas == M_MEAS)
                UDX_en = 0;
            else
                UDX_en = 1;
        } else {
            if (meas == 1)
                UDX_en = 0;
            else
                UDX_en = 1;
        }

        float h_vector[PROC_MU][DEPTH_MU];
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=h_vector complete dim=0
    // clang-format on
    LOOPHM:
        for (ap_uint<8> i = 0; i < DEPTH_MU; i++) {
// clang-format off
            #pragma HLS LOOP_FLATTEN off
            #pragma HLS pipeline
            // clang-format on
            for (ap_uint<8> j = 0; j < PROC_MU; j++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on

                h_vector[j][i] = H_matrix[j][meas * DEPTH_MU + i];
            }
        }
        float r_value; // = ry_vector[meas];

        float z_value;
        if (EKF_EN == 0) {
            z_value = ry_vector[meas + M_MEAS];
            r_value = ry_vector[meas];
        } else {
            z_value = Zekf - hx;
            r_value = Rekf;
        }

        if (flip == 0) {
            MeasUpdate_1x<N_STATE, M_MEAS, PROC_MU, DEPTH_MU, UMAT_DEPTH, URAM_EN, EKF_EN>(
                U_matrix, D_vector, Uint_matrix, Dint_vector, xu_vector, h_vector, r_value, z_value, UDX_en);
            flip = 1;
        } else {
            MeasUpdate_1x<N_STATE, M_MEAS, PROC_MU, DEPTH_MU, UMAT_DEPTH, URAM_EN, EKF_EN>(
                Uint_matrix, Dint_vector, U_matrix, D_vector, xu_vector, h_vector, r_value, z_value, UDX_en);
            flip = 0;
        }
    }

    //###### Write X corrected state vector
    if (X_write_en) KF_X_write<N_STATE, TYPE, NPC>(xu_vector, Xout_mat);

    //###### Write P corrected state vector
    if (UD_write_en)
        KF_UD_write<N_STATE, PROC_MU, DEPTH_MU, UMAT_DEPTH, TYPE, NPC>(U_matrix, D_vector, Uout_mat, Dout_mat);
}

template <int N_STATE,
          int C_CTRL,
          int M_MEAS,
          int PROC_TU,
          int DEPTH_TU,
          int PROC_MU,
          int DEPTH_MU,
          int UMAT_DEPTH,
          int HMAT_DEPTH,
          int TMAT_DEPTH,
          int UQMAT_DEPTH,
          bool URAM_EN,
          bool EKF_EN,
          int TYPE,
          int NPC>
void MeasUpdate_wrapper(float U_matrix[PROC_MU][UMAT_DEPTH],
                        float H_matrix[PROC_MU][HMAT_DEPTH],
                        float D_vector[512],
                        float xu_vector[512],
                        float ry_vector[512],
                        float T_matrix[PROC_TU][TMAT_DEPTH],
                        float Uq_matrix[UQMAT_DEPTH],
#if KF_C != 0
                        xf::cv::Mat<TYPE, C_CTRL, 1, NPC>& u_mat,
#endif
                        xf::cv::Mat<TYPE, M_MEAS, 1, NPC>& y_mat,
                        xf::cv::Mat<TYPE, M_MEAS, 1, NPC>& R_mat,
                        xf::cv::Mat<TYPE, M_MEAS, N_STATE, NPC>& H_mat,
                        xf::cv::Mat<TYPE, N_STATE, 1, NPC>& Xout_mat,
                        xf::cv::Mat<TYPE, N_STATE, N_STATE, NPC>& Uout_mat,
                        xf::cv::Mat<TYPE, N_STATE, 1, NPC>& Dout_mat,
                        bool X_write_en,
                        bool UD_write_en) {
    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=U_matrix complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=U_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=U_matrix complete dim=1
        // clang-format on
    }

    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=H_matrix complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=H_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=H_matrix complete dim=1
        // clang-format on
    }

    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=T_matrix complete dim=1
        #pragma HLS resource variable=T_matrix core=RAM_S2P_BRAM
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=T_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=T_matrix complete dim=1
        // clang-format on
    }

// clang-format off
    #pragma HLS inline off
// clang-format on

LOOP1:
    for (int itr1 = 0; itr1 < 1; itr1++) {
        load_Uq<N_STATE, PROC_TU, DEPTH_TU, TMAT_DEPTH, UQMAT_DEPTH, URAM_EN>(T_matrix, Uq_matrix);

        MeasUpdate<N_STATE, C_CTRL, M_MEAS, PROC_MU, DEPTH_MU, UMAT_DEPTH, HMAT_DEPTH, URAM_EN, EKF_EN, TYPE, NPC>(
            U_matrix, H_matrix, D_vector, xu_vector, ry_vector,
#if KF_C != 0
            u_mat,
#endif
            y_mat, R_mat, H_mat, Xout_mat, Uout_mat, Dout_mat, X_write_en, UD_write_en);
    }
}

template <int N_STATE, int U_SIZE, int TYPE, int NPC>
void load_control_input(
#if KF_C != 0
    xf::cv::Mat<TYPE, U_SIZE, 1, NPC>& control_input,
#endif
    float xu_vector[512]) {
    for (ap_uint<8> idx = 0; idx < U_SIZE; idx++) {
// clang-format off
#pragma HLS pipeline
// clang-format on
#if KF_C != 0
        xu_vector[N_STATE + idx] = control_input.read_float(idx);
#endif
    }
}

//########################################################################################//
// For gemv operation,
//...
//    0__ 4__ 8 __ 11__ 	 a|			  0a			4b			8c
//    12d 1__ 5__ 9 __ 12__ 	 b|			  1a			5b			9c
//    13d
// A = 2__ 6__ 10__ 13__  X = c|		1st = 2a	2nd +=  6b	3rd +=	10c	4th +=	14d
//    3__ 7__ 11__ 14__ 	 d|			  3a			7b			11c
//    15d
//##########################################################################################//
// for   x' = Ax, x_buffer[256-383] = A_buffer*x_buffer[0-127]
// for x_tu = Bu, x_buffer[0-127]   = B_buffer*x_buffer[256-383]
//##########################################################################################//
template <int N_STATE, int C_CTRL, int PROC_MU, int DEPTH_MU, int DEPTH_MU_CTRL, int ABMAT_DEPTH, int UMAT_DEPTH>
void gemv(float AB_matrix[PROC_MU][ABMAT_DEPTH],
          float xu_vector[512],
          ap_uint<16> matrix_offset,
          ap_uint<10> vector_offset_in,
          ap_uint<10> vector_offset_out,
          ap_uint<8> outer_loop_bound) {
// clang-format off
    #pragma HLS inline off
// clang-format on

// New Gemv design with 1 multipliers and 1 adders
LOOP1:
    for (ap_uint<8> outer_loop = 0; outer_loop < outer_loop_bound; outer_loop++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=128 max=128
        // clang-format on
        float input_x = xu_vector[vector_offset_in + outer_loop];

        ap_uint<16> buffer_idx = 0;
        ap_uint<16> idx_inc;
        if (vector_offset_in == 0)
            idx_inc = DEPTH_MU;
        else
            idx_inc = DEPTH_MU_CTRL;

    LOOPF1:
        for (ap_uint<10> inner_loop = 0; inner_loop < N_STATE; inner_loop++) {
// clang-format off
            #pragma HLS loop_flatten off
            #pragma HLS DEPENDENCE variable=xu_vector inter false
            #pragma HLS pipeline
            // clang-format on

            float input_A = AB_matrix[outer_loop % PROC_MU][matrix_offset + buffer_idx + outer_loop / PROC_MU];

            float mul_out = input_A * input_x;

            ap_uint<32> offset;
            if (vector_offset_in != 0 && outer_loop != 0)
                offset = 0;
            else
                offset = 256;

            float intermediate_x = xu_vector[inner_loop + offset];

            float add_input;
            if (vector_offset_in == 0 && outer_loop == 0)
                add_input = 0;
            else
                add_input = intermediate_x;

            float add_in2;

            if (C_CTRL == 0) {
                if (vector_offset_in != 0)
                    add_in2 = 0;
                else
                    add_in2 = mul_out;

            } else {
                add_in2 = mul_out;
            }

            xu_vector[inner_loop + vector_offset_out] = add_input + add_in2;

            buffer_idx += idx_inc;

        } // end proc loop
    }     // end outer loop
}

template <int N_STATE, int C_CTRL, int PROC_MU, int DEPTH_MU, int DEPTH_MU_CTRL, int ABMAT_DEPTH, int UMAT_DEPTH>
void state_predict(float AB_matrix[PROC_MU][ABMAT_DEPTH], float xu_vector[512]) {
// clang-format off
    #pragma HLS inline off
    // clang-format on
    for (ap_uint<2> iteration = 0; iteration < 2; iteration++) {
        ap_uint<16> matrix_offset;
        ap_uint<10> vector_offset_in;
        ap_uint<10> vector_offset_out;
        ap_uint<8> outer_loop_bound;
        if (iteration == 0) {
            matrix_offset = 0;
            vector_offset_in = 0;
            vector_offset_out = 256;
            outer_loop_bound = N_STATE;
        } else {
            matrix_offset = UMAT_DEPTH;
            vector_offset_in = N_STATE;
            vector_offset_out = 0;
            if (C_CTRL == 0)
                outer_loop_bound = 1;
            else
                outer_loop_bound = C_CTRL;
        }

        gemv<N_STATE, C_CTRL, PROC_MU, DEPTH_MU, DEPTH_MU_CTRL, ABMAT_DEPTH, UMAT_DEPTH>(
            AB_matrix, xu_vector, matrix_offset, vector_offset_in, vector_offset_out, outer_loop_bound);
    }
}

template <int N_STATE,
          int PROC_TU,
          int DEPTH_TU,
          int PROC_MU,
          int DEPTH_MU,
          int ABMAT_DEPTH,
          int UMAT_DEPTH,
          int TMAT_DEPTH,
          bool URAM_EN>
void gemm_update(float AB_matrix[PROC_MU][ABMAT_DEPTH],
                 float U_matrix[PROC_MU][UMAT_DEPTH],
                 float T_matrix[PROC_TU][TMAT_DEPTH],
                 ap_uint<10> out_col_start,
                 ap_uint<10> out_col_cnt,
                 ap_uint<10> iteration) {
    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=U_matrix complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=U_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=U_matrix complete dim=1
        // clang-format on
    }

    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=AB_matrix complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=AB_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=AB_matrix complete dim=1
        // clang-format on
    }

    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=T_matrix complete dim=1
        #pragma HLS resource variable=T_matrix core=RAM_S2P_BRAM
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=T_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=T_matrix complete dim=1
        // clang-format on
    }

// clang-format off
    #pragma HLS inline off
// clang-format on
LOOP2:
    for (ap_uint<10> out_row0 = 0; out_row0 < N_STATE; out_row0++) {
    LOOP3:
        for (ap_uint<10> out_col_idx = 0; out_col_idx < out_col_cnt; out_col_idx++) {
// clang-format off
            #pragma HLS loop_flatten
            #pragma HLS LOOP_TRIPCOUNT min=128 max=128
            #pragma HLS DEPENDENCE variable=T_matrix inter false
            #pragma HLS pipeline
            // clang-format on

            ap_uint<8> out_col = out_col_start + out_col_idx;

            ap_uint<20> out_index0 = out_row0 * (DEPTH_TU * PROC_TU) + out_col;
            ap_uint<8> dim1_0_Tmatrix = out_index0 % PROC_TU;
            ap_uint<16> dim2_0_Tmatrix = out_index0 / PROC_TU;

            ap_uint<16> dim2_0_Amatrix = out_row0 * (DEPTH_MU) + iteration * 2;
            ap_uint<16> dim2_1_Amatrix = dim2_0_Amatrix + 1;

            ap_uint<16> dim2_0_Umatrix = out_col * (DEPTH_MU) + iteration * 2;
            ap_uint<16> dim2_1_Umatrix = dim2_0_Umatrix + 1;

            bool pad_en = (2 * iteration + 2) > DEPTH_MU;

            float input1_dotproduct[PROC_MU * 2];
            float input2_dotproduct[PROC_MU * 2];
            for (ap_uint<8> idx = 0; idx < PROC_MU; idx++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                input1_dotproduct[idx] = AB_matrix[idx][dim2_0_Amatrix];
                if (pad_en)
                    input1_dotproduct[idx + PROC_MU] = 0;
                else
                    input1_dotproduct[idx + PROC_MU] = AB_matrix[idx][dim2_1_Amatrix];

                input2_dotproduct[idx] = U_matrix[idx][dim2_0_Umatrix];
                input2_dotproduct[idx + PROC_MU] = U_matrix[idx][dim2_1_Umatrix];
            }

            float dot_output = KF_dotProduct<2 * PROC_MU>(input1_dotproduct, input2_dotproduct);

            float read1 = T_matrix[dim1_0_Tmatrix][dim2_0_Tmatrix];
            float write1;
            if (iteration == 0)
                write1 = dot_output;
            else
                write1 = read1 + dot_output;

            T_matrix[dim1_0_Tmatrix][dim2_0_Tmatrix] = write1;
        }
    }
}

template <int N_STATE,
          int PROC_TU,
          int DEPTH_TU,
          int PROC_MU,
          int DEPTH_MU,
          int ABMAT_DEPTH,
          int UMAT_DEPTH,
          int TMAT_DEPTH,
          bool URAM_EN>
void AU_compute(float AB_matrix[PROC_MU][ABMAT_DEPTH],
                float U_matrix[PROC_MU][UMAT_DEPTH],
                float T_matrix[PROC_TU][TMAT_DEPTH]) {
    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=U_matrix complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=U_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=U_matrix complete dim=1
        // clang-format on
    }

    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=AB_matrix complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=AB_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=AB_matrix complete dim=1
        // clang-format on
    }

    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=T_matrix complete dim=1
        #pragma HLS resource variable=T_matrix core=RAM_S2P_BRAM
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=T_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=T_matrix complete dim=1
        // clang-format on
    }

// clang-format off
    #pragma HLS inline off
    // clang-format on

    enum { GEMM_ITERATION = ((DEPTH_MU / 2) + (DEPTH_MU % 2)) };

LOOP1:
    for (ap_uint<10> iteration = 0, out_col_start = 0, out_col_cnt = N_STATE; iteration < GEMM_ITERATION;
         iteration++, out_col_start += (2 * PROC_MU), out_col_cnt -= (2 * PROC_MU)) {
        gemm_update<N_STATE, PROC_TU, DEPTH_TU, PROC_MU, DEPTH_MU, ABMAT_DEPTH, UMAT_DEPTH, TMAT_DEPTH, URAM_EN>(
            AB_matrix, U_matrix, T_matrix, out_col_start, out_col_cnt, iteration);
    }
}

// In update_T_matrix function...
//
// for j= n..1
//{
//	Dbar[j] = trans(t[j]) * DpDq * t[j]; //Delta[j] = DpDq * t[j]
//
//	for i= 1..j-1
//	{
//		Ubar[i,j] = trans(t[i]) * Delta[j] / Dbar[j]
//		t[i] = t[i] - Ubar[i,j]*t[j]
//	}
//}
//
// Above psuedo code is modified as below
//
// for j= n..1
//{
//	for i= j..1
//	{
//		Delta[j] = DpDq * t[j]
//	  	Udash = dotproduct(trans(t[i]) ,Delta[j])
//	  	if(i=j)
//	  		Dbar[j] = Udash
//		Ubar[i,j] = Udash / Dbar[j]
//		t[i] = t[i] - Ubar[i,j]*t[j]
//	}
//}
template <int N_STATE,
          int PROC_TU,
          int DEPTH_TU,
          int PROC_MU,
          int DEPTH_MU,
          int TMAT_DEPTH,
          int DPDQ_DEPTH,
          int UMAT_DEPTH,
          bool URAM_EN>
void update_T_matrix(float Tj_vector[PROC_TU][DPDQ_DEPTH],
                     float Deltaj_vector[PROC_TU][DPDQ_DEPTH],
                     float T_matrix[PROC_TU][TMAT_DEPTH],
                     float U_matrix[PROC_MU][UMAT_DEPTH],
                     float D_vector[512],
                     ap_uint<10> u_col_num) {
    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=T_matrix complete dim=1
        #pragma HLS resource variable=T_matrix core=RAM_S2P_BRAM
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=T_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=T_matrix complete dim=1
        // clang-format on
    }
// clang-format off
    #pragma HLS inline off
    // clang-format on
    float Dn_value = 0;
    float dotOutInt_ti_Deltaj[DEPTH_TU];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=dotOutInt_ti_Deltaj complete dim=1
    // clang-format on
    for (int i = 0; i < DEPTH_TU; i++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        dotOutInt_ti_Deltaj[i] = 0;
    }
    float Un_dash = 0;
    float U_value_in = 0;

    float Ti_ping[PROC_TU][DEPTH_TU];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=Ti_ping complete dim=0
    // clang-format on
    float Ti_pong[PROC_TU][DEPTH_TU];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=Ti_pong complete dim=0
// clang-format on

//#########################
// 1st iteration of LOOPM_1, T matrix's rows will not be updated
// since this loop is running in ping-pong, 1st iteration will be ideal for U_value and T matrix
// After 1st iteration, T matrix row index = u_row_num+1

LOOPM_1:
    for (ap_int<16> u_row_num = u_col_num, start = 0; u_row_num >= -1; u_row_num--, start++) {
    LOOPM_2:
        for (ap_uint<10> depth_num = 0; depth_num < DEPTH_TU; depth_num++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=128*16 max=128*16
            #pragma HLS DEPENDENCE variable=T_matrix inter false
            #pragma HLS pipeline
            // clang-format on

            ap_uint<16> index_num;
            ap_uint<16> index_num2;
            index_num = u_row_num * DEPTH_TU + depth_num;
            index_num2 = (u_row_num + 1) * DEPTH_TU + depth_num;

            float Ti_chunk[PROC_TU];
// clang-format off
            #pragma HLS ARRAY_PARTITION variable=Ti_chunk complete dim=1
            // clang-format on

            for (ap_uint<10> idx = 0; idx < PROC_TU; idx++) {
// clang-format off
#pragma HLS unroll
// clang-format on
#if !__SYNTHESIS__
                float T_mat_read;
                if (u_row_num == -1)
                    T_mat_read = 0;
                else {
                    T_mat_read = T_matrix[idx][index_num];
                }
#else
                float T_mat_read = T_matrix[idx][index_num];
#endif
                Ti_chunk[idx] = T_mat_read;
                if (start[0] == 0)
                    Ti_ping[idx][depth_num] = T_mat_read;
                else
                    Ti_pong[idx][depth_num] = T_mat_read;
            } // idx loop

            float Tj_for_delta[PROC_TU];
// clang-format off
            #pragma HLS ARRAY_PARTITION variable=Tj_for_delta complete dim=1
            // clang-format on
            float deltaj_chunk[PROC_TU];
// clang-format off
            #pragma HLS ARRAY_PARTITION variable=deltaj_chunk complete dim=1
            // clang-format on

            for (ap_uint<10> idx = 0; idx < PROC_TU; idx++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                Tj_for_delta[idx] = Tj_vector[idx][depth_num];
                deltaj_chunk[idx] = Deltaj_vector[idx][depth_num];
            } // idx loop

            float temp_dotout;
            temp_dotout = KF_dotProduct<PROC_TU>(Ti_chunk, deltaj_chunk);

            dotOutInt_ti_Deltaj[depth_num] = temp_dotout;

            float Un_dash_temp;
            KF_treeAdder<DEPTH_TU>(dotOutInt_ti_Deltaj, &Un_dash_temp);

            if ((depth_num) == (DEPTH_TU - 1)) Un_dash = Un_dash_temp;

            if (u_row_num == u_col_num && (depth_num) == (DEPTH_TU - 1)) {
                Dn_value = Un_dash;
                D_vector[u_col_num] = Un_dash;
            }

            float Un_value = Un_dash / Dn_value;

            ap_uint<8> dim1_Umat = u_row_num % PROC_MU;
            ap_uint<16> dim2_Umat = u_col_num * (DEPTH_MU) + u_row_num / PROC_MU;

            if (u_row_num != -1) {
                U_matrix[dim1_Umat][dim2_Umat] = Un_value;
            }

            float Ti_select[PROC_TU];
            for (ap_uint<10> idx = 0; idx < PROC_TU; idx++) {
                if (start[0] == 1)
                    Ti_select[idx] = Ti_ping[idx][depth_num];
                else
                    Ti_select[idx] = Ti_pong[idx][depth_num];
            }

            float Ti_update_chunk[PROC_TU];

            KF_scaleSub<PROC_TU>(Ti_select, U_value_in, Tj_for_delta, Ti_update_chunk);

            for (ap_uint<10> idx = 0; idx < PROC_TU; idx++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                if (u_row_num != u_col_num) {
                    T_matrix[idx][index_num2] = Ti_update_chunk[idx];
                }
            } // idx loop

            if (depth_num == DEPTH_TU - 1) U_value_in = Un_value;

        } // depth_num
    }     // u_row_num
}

//####Load 1 ROW from T_matrix from Tj_vector. ROW id = u_col_num
template <int N_STATE, int PROC_TU, int DEPTH_TU, int TMAT_DEPTH, int DPDQ_DEPTH, bool URAM_EN>
void load_TjDeltaj_vector(float T_matrix[PROC_TU][TMAT_DEPTH],
                          float Tj_vector[PROC_TU][DPDQ_DEPTH],
                          float Deltaj_vector[PROC_TU][DPDQ_DEPTH],
                          float D_vector[512],
                          ap_uint<10> u_col_num) {
    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=T_matrix complete dim=1
        #pragma HLS resource variable=T_matrix core=RAM_S2P_BRAM
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=T_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=T_matrix complete dim=1
        // clang-format on
    }
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=Tj_vector complete dim=1
    #pragma HLS ARRAY_PARTITION variable=Deltaj_vector complete dim=1
    #pragma HLS inline off
    // clang-format on

    ap_uint<10> dim1_D = N_STATE;
    for (ap_uint<14> idx1 = 0; idx1 < (DEPTH_TU); idx1++) {
        for (ap_uint<8> idx2 = 0; idx2 < PROC_TU; idx2++) {
// clang-format off
            #pragma HLS pipeline
            // clang-format on
            ap_uint<16> dim2 = (idx1 + u_col_num * DEPTH_TU);

            float T_value = T_matrix[idx2][dim2];
            float D_value = D_vector[dim1_D++];

            Deltaj_vector[idx2][idx1] = T_value * D_value;
            Tj_vector[idx2][idx1] = T_value;
        }
    }
}

template <int N_STATE,
          int PROC_TU,
          int DEPTH_TU,
          int PROC_MU,
          int DEPTH_MU,
          int TMAT_DEPTH,
          int UMAT_DEPTH,
          int DPDQ_DEPTH,
          bool URAM_EN>
void UD_compute(float T_matrix[PROC_TU][TMAT_DEPTH], float U_matrix[PROC_MU][UMAT_DEPTH], float D_vector[512]) {
    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=U_matrix complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=U_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=U_matrix complete dim=1
        // clang-format on
    }

    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=T_matrix complete dim=1
        #pragma HLS resource variable=T_matrix core=RAM_S2P_BRAM
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=T_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=T_matrix complete dim=1
        // clang-format on
    }
// clang-format off
    #pragma HLS inline off
    // clang-format on

    float Tj_vector[PROC_TU][DPDQ_DEPTH];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=Tj_vector complete dim=1
    // clang-format on
    float Deltaj_vector[PROC_TU][DPDQ_DEPTH];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=Deltaj_vector complete dim=1
    // clang-format on

    for (ap_int<10> u_col_num = N_STATE - 1; u_col_num >= 0; u_col_num--) {
        load_TjDeltaj_vector<N_STATE, PROC_TU, DEPTH_TU, TMAT_DEPTH, DPDQ_DEPTH, URAM_EN>(
            T_matrix, Tj_vector, Deltaj_vector, D_vector, u_col_num);

        update_T_matrix<N_STATE, PROC_TU, DEPTH_TU, PROC_MU, DEPTH_MU, TMAT_DEPTH, DPDQ_DEPTH, UMAT_DEPTH, URAM_EN>(
            Tj_vector, Deltaj_vector, T_matrix, U_matrix, D_vector, u_col_num);
    }
}

template <int N_STATE,
          int M_MEAS,
          int C_CTRL,
          int PROC_TU,
          int DEPTH_TU,
          int PROC_MU,
          int DEPTH_MU,
          int DEPTH_MU_CTRL,
          int UMAT_DEPTH,
          int ABMAT_DEPTH,
          int DPDQ_DEPTH,
          int TMAT_DEPTH,
          bool URAM_EN,
          bool EKF_EN,
          int TYPE,
          int NPC>
void TimeUpdate(float T_matrix[PROC_TU][TMAT_DEPTH],
                float AB_matrix[PROC_MU][ABMAT_DEPTH],
                float xu_vector[512],
                float U_matrix[PROC_MU][UMAT_DEPTH],
                float D_vector[512],
#if KF_C != 0
                xf::cv::Mat<TYPE, C_CTRL, 1, NPC>& u_mat,
#endif
                xf::cv::Mat<TYPE, N_STATE, 1, NPC>& Xout_mat,
                xf::cv::Mat<TYPE, N_STATE, N_STATE, NPC>& Uout_mat,
                xf::cv::Mat<TYPE, N_STATE, 1, NPC>& Dout_mat,
                bool X_write_en,
                bool UD_write_en) {

// clang-format off
    #pragma HLS inline off
// clang-format on

LOOP1:
    for (int itr1 = 0; itr1 < 1; itr1++) {
#if KF_C != 0
        if (EKF_EN == 0) load_control_input<N_STATE, C_CTRL, TYPE, NPC>(u_mat, xu_vector);
#endif
        AU_compute<N_STATE, PROC_TU, DEPTH_TU, PROC_MU, DEPTH_MU, ABMAT_DEPTH, UMAT_DEPTH, TMAT_DEPTH, URAM_EN>(
            AB_matrix, U_matrix, T_matrix);
    }

LOOP2:
    for (int itr1 = 0; itr1 < 1; itr1++) {
        if (EKF_EN == 0)
            state_predict<N_STATE, C_CTRL, PROC_MU, DEPTH_MU, DEPTH_MU_CTRL, ABMAT_DEPTH, UMAT_DEPTH>(AB_matrix,
                                                                                                      xu_vector);

        UD_compute<N_STATE, PROC_TU, DEPTH_TU, PROC_MU, DEPTH_MU, TMAT_DEPTH, UMAT_DEPTH, DPDQ_DEPTH, URAM_EN>(
            T_matrix, U_matrix, D_vector);
    }

    if (X_write_en) KF_X_write<N_STATE, TYPE, NPC>(xu_vector, Xout_mat);

    if (UD_write_en)
        KF_UD_write<N_STATE, PROC_MU, DEPTH_MU, UMAT_DEPTH, TYPE, NPC>(U_matrix, D_vector, Uout_mat, Dout_mat);
}

template <int N_STATE,
          int M_MEAS,
          int C_CTRL,
          int PROC_TU,
          int DEPTH_TU,
          int PROC_MU,
          int DEPTH_MU,
          int DEPTH_MU_CTRL,
          int UMAT_DEPTH,
          int HMAT_DEPTH,
          int ABMAT_DEPTH,
          int DPDQ_DEPTH,
          int TMAT_DEPTH,
          int UQMAT_DEPTH,
          bool URAM_EN,
          bool EKF_EN,
          int TYPE,
          int NPC>
void initialization(xf::cv::Mat<TYPE, N_STATE, N_STATE, NPC>& A_mat,
#if KF_C != 0
                    xf::cv::Mat<TYPE, N_STATE, C_CTRL, NPC>& B_mat,
#endif
                    xf::cv::Mat<TYPE, N_STATE, N_STATE, NPC>& Uq_mat,
                    xf::cv::Mat<TYPE, N_STATE, 1, NPC>& Dq_mat,
                    xf::cv::Mat<TYPE, M_MEAS, N_STATE, NPC>& H_mat,
                    xf::cv::Mat<TYPE, N_STATE, 1, NPC>& X0_mat,
                    xf::cv::Mat<TYPE, N_STATE, N_STATE, NPC>& U0_mat,
                    xf::cv::Mat<TYPE, N_STATE, 1, NPC>& D0_mat,
                    xf::cv::Mat<TYPE, M_MEAS, 1, NPC>& R_mat,
                    float H_matrix[PROC_MU][HMAT_DEPTH],
                    float U_matrix[PROC_MU][UMAT_DEPTH],
                    float xu_vector[512],
                    float ry_vector[512],
                    float D_vector[512],
                    float AB_matrix[PROC_MU][ABMAT_DEPTH],
                    float T_matrix[PROC_TU][TMAT_DEPTH],
                    float Uq_matrix[UQMAT_DEPTH],
                    bool read_opt_flag) {
    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=H_matrix complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=H_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=H_matrix complete dim=1
        // clang-format on
    }

    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=U_matrix complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=U_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=U_matrix complete dim=1
        // clang-format on
    }

    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=AB_matrix complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=AB_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=AB_matrix complete dim=1
        // clang-format on
    }

    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=T_matrix complete dim=1
        #pragma HLS resource variable=T_matrix core=RAM_S2P_BRAM
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=T_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=T_matrix complete dim=1
        // clang-format on
    }

    if (URAM_EN == 1) {
// clang-format off
        #pragma HLS RESOURCE variable=Uq_matrix core=RAM_S2P_URAM
        // clang-format on
    }

// clang-format off
    #pragma HLS inline off
    // clang-format on

    int U0_loop_cnt;
    if (EKF_EN == 1 && read_opt_flag == 1)
        U0_loop_cnt = 0;
    else
        U0_loop_cnt = N_STATE * N_STATE;

    ap_uint<32> counter1 = 0;
    ap_uint<32> counter1_1 = 0; // for dim2
    ap_uint<32> counter2 = 0;   // for dim1
    ap_uint<32> counter3 = 0;   // for dim2
LOOPI_U:
    for (int ptr = 0; ptr < U0_loop_cnt; ptr++) {
// clang-format off
        #pragma HLS pipeline
        // clang-format on

        ap_uint<8> dim1 = counter2;
        ap_uint<16> dim2 = counter1_1 + counter3;

        U_matrix[dim1][dim2] = U0_mat.read_float(ptr);

        if (counter1 == N_STATE - 1) {
            if (counter2 == PROC_MU - 1) {
                counter2 = 0;
                counter3++;
            } else {
                counter2++;
            }

            counter1 = 0;
            counter1_1 = 0;
        } else {
            counter1++;
            counter1_1 += DEPTH_MU;
        }
    }

LOOPHZ:
    for (int ptr_zero = 0, dim2 = (DEPTH_MU - 1); ptr_zero < M_MEAS; ptr_zero++, dim2 += DEPTH_MU) {
// clang-format off
        #pragma HLS pipeline
        // clang-format on

        for (int dim1 = 0; dim1 < PROC_MU; dim1++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on

            H_matrix[dim1][dim2] = 0;
        }
    }

    if (EKF_EN == 0) {
        ap_uint<32> offset_incH = 0;
        ap_uint<32> counter_Hrow = 0;

    LOOPI_H:
        for (int ptr = 0; ptr < M_MEAS * N_STATE; ptr++) {
// clang-format off
            #pragma HLS pipeline
            // clang-format on
            ap_uint<32> offset = offset_incH + counter_Hrow;
            ap_uint<10> dim1 = offset % PROC_MU;
            ap_uint<16> dim2 = offset / PROC_MU;

            H_matrix[dim1][dim2] = H_mat.read_float(ptr);

            if (counter_Hrow == N_STATE - 1) {
                counter_Hrow = 0;
                offset_incH += DEPTH_MU * PROC_MU;
            } else
                counter_Hrow++;
        }
    }

    //******************************Load R ****************************//
    int R_loop_cnt;
    if (EKF_EN == 1 && read_opt_flag == 1)
        R_loop_cnt = 0;
    else
        R_loop_cnt = M_MEAS;

LOOPI_R:
    for (int ptr = 0; ptr < R_loop_cnt; ptr++) {
// clang-format off
        #pragma HLS pipeline
        // clang-format on

        ry_vector[ptr] = R_mat.read_float(ptr);
    }

//******************************Load X0 ****************************//
LOOPI_X:
    for (int ptr = 0; ptr < N_STATE; ptr++) {
// clang-format off
        #pragma HLS pipeline
        // clang-format on

        xu_vector[ptr] = X0_mat.read_float(ptr);
    }

    //******************************Load D0 ****************************//
    int D0_loop_cnt;
    if (EKF_EN == 1 && read_opt_flag == 1)
        D0_loop_cnt = 0;
    else
        D0_loop_cnt = N_STATE;

LOOPI_D:
    for (int ptr = 0; ptr < D0_loop_cnt; ptr++) {
// clang-format off
        #pragma HLS pipeline
        // clang-format on

        D_vector[ptr] = D0_mat.read_float(ptr);
    }

LOOPI_T1:
    for (int ptr = 0; ptr < D0_loop_cnt; ptr++) {
// clang-format off
        #pragma HLS pipeline
        // clang-format on

        D_vector[ptr + N_STATE] = D_vector[ptr];
    }

    //******************************Load A <Row major>****************************//

    ap_uint<16> dim2 = (DEPTH_MU - 1);
LOOPAZ:
    for (int ptr_zero = 0; ptr_zero < 2 * N_STATE; ptr_zero++) {
// clang-format off
        #pragma HLS pipeline
        // clang-format on

        for (int dim1 = 0; dim1 < PROC_MU; dim1++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on

            AB_matrix[dim1][dim2] = 0;
        }
        if (ptr_zero < (N_STATE - 1))
            dim2 += DEPTH_MU;
        else
            dim2 += DEPTH_MU_CTRL;
    }

    ap_uint<32> offset_incA = 0;
    ap_uint<32> counter_Arow = 0;
LOOPI_A:
    for (int ptr = 0; ptr < N_STATE * N_STATE; ptr++) {
// clang-format off
        #pragma HLS pipeline
        // clang-format on

        ap_uint<32> offset = offset_incA + counter_Arow;
        ap_uint<8> dim1 = offset % PROC_MU;
        ap_uint<16> dim2 = offset / PROC_MU;

        AB_matrix[dim1][dim2] = A_mat.read_float(ptr);

        if (counter_Arow == N_STATE - 1) {
            counter_Arow = 0;
            offset_incA += DEPTH_MU * PROC_MU;
        } else
            counter_Arow++;
    }

    ap_uint<32> offset_incB = 0;
    ap_uint<32> counter_Brow = 0;

    int B_loop_cnt;
    if (EKF_EN == 1)
        B_loop_cnt = 0;
    else
        B_loop_cnt = N_STATE * C_CTRL;

LOOPI_B:
    for (int ptr = 0; ptr < B_loop_cnt; ptr++) {
// clang-format off
        #pragma HLS pipeline
        // clang-format on

        ap_uint<32> offset = offset_incB + counter_Brow;
        ap_uint<8> dim1 = offset % PROC_MU;
        ap_uint<16> dim2 = offset / PROC_MU;
#if KF_C != 0
        AB_matrix[dim1][dim2 + (DEPTH_MU * N_STATE)] = B_mat.read_float(ptr);
#endif
        if (counter_Brow == C_CTRL - 1) {
            counter_Brow = 0;
            offset_incB += DEPTH_MU_CTRL * PROC_MU;
        } else
            counter_Brow++;
    }

    //******************************Load Dq only digonal elements****************************//
    int Dq_loop_cnt;
    if (EKF_EN == 1 && read_opt_flag == 1)
        Dq_loop_cnt = 0;
    else
        Dq_loop_cnt = N_STATE;

LOOPI_T2:
    for (int ptr = 0; ptr < Dq_loop_cnt; ptr++) {
// clang-format off
        #pragma HLS pipeline
        // clang-format on

        D_vector[ptr + 2 * N_STATE] = Dq_mat.read_float(ptr);
    }
    //******************************Load Uq <Row major> ****************************//
    int Uq_loop_cnt;
    if (EKF_EN == 1 && read_opt_flag == 1)
        Uq_loop_cnt = 0;
    else
        Uq_loop_cnt = N_STATE * N_STATE;

    ap_uint<16> counter_trow = 0;
    ap_uint<32> offset_inc = N_STATE;
LOOPI_UQ:
    for (int ptr = 0; ptr < Uq_loop_cnt; ptr++) {
// clang-format off
        #pragma HLS pipeline
        // clang-format on

        ap_uint<32> offset = offset_inc + counter_trow;
        ap_uint<8> dim1 = offset % PROC_TU;
        ap_uint<16> dim2 = offset / PROC_TU;

        float Uq_value = Uq_mat.read_float(ptr);

        T_matrix[dim1][dim2] = Uq_value;
        Uq_matrix[ptr] = Uq_value;

        if (counter_trow == N_STATE - 1) {
            counter_trow = 0;
            offset_inc += DEPTH_TU * PROC_TU;
        } else
            counter_trow++;
    }
}

template <int N_STATE, int M_MEAS, int C_CTRL, int PROC_TU, int PROC_MU, bool URAM_EN, bool EKF_EN, int TYPE, int NPC>
void KalmanFilter_def(xf::cv::Mat<TYPE, N_STATE, N_STATE, NPC>& A_mat,
#if KF_C != 0
                      xf::cv::Mat<TYPE, N_STATE, C_CTRL, NPC>& B_mat,
#endif
                      xf::cv::Mat<TYPE, N_STATE, N_STATE, NPC>& Uq_mat,
                      xf::cv::Mat<TYPE, N_STATE, 1, NPC>& Dq_mat,
                      xf::cv::Mat<TYPE, M_MEAS, N_STATE, NPC>& H_mat,
                      xf::cv::Mat<TYPE, N_STATE, 1, NPC>& X0_mat,
                      xf::cv::Mat<TYPE, N_STATE, N_STATE, NPC>& U0_mat,
                      xf::cv::Mat<TYPE, N_STATE, 1, NPC>& D0_mat,
                      xf::cv::Mat<TYPE, M_MEAS, 1, NPC>& R_mat,
#if KF_C != 0
                      xf::cv::Mat<TYPE, C_CTRL, 1, NPC>& u_mat,
#endif
                      xf::cv::Mat<TYPE, M_MEAS, 1, NPC>& y_mat,
                      xf::cv::Mat<TYPE, N_STATE, 1, NPC>& Xout_mat,
                      xf::cv::Mat<TYPE, N_STATE, N_STATE, NPC>& Uout_mat,
                      xf::cv::Mat<TYPE, N_STATE, 1, NPC>& Dout_mat,
                      unsigned char flag) {
// clang-format off
    #pragma HLS inline off
    // clang-format on

    enum {
        DEPTH_TU = ((2 * N_STATE) / PROC_TU + (((2 * N_STATE) % PROC_TU) != 0)),
        DEPTH_MU = (N_STATE / PROC_MU + ((N_STATE % PROC_MU) != 0)),
        DEPTH_MU_CTRL = (C_CTRL / PROC_MU + ((C_CTRL % PROC_MU) != 0)),
        UMAT_DEPTH = (DEPTH_MU * N_STATE),
        HMAT_DEPTH = (DEPTH_MU * M_MEAS),
        ABMAT_DEPTH = ((DEPTH_MU * N_STATE) + (DEPTH_MU_CTRL * N_STATE)),
        DPDQ_DEPTH = DEPTH_TU,
        TMAT_DEPTH = (DEPTH_TU * N_STATE),
        UQMAT_DEPTH = (N_STATE * N_STATE)
    };

    static float H_matrix[PROC_MU][HMAT_DEPTH];
    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=H_matrix complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=H_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=H_matrix complete dim=1
        // clang-format on
    }

    static float U_matrix[PROC_MU][UMAT_DEPTH];
    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=U_matrix complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=U_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=U_matrix complete dim=1
        // clang-format on
    }

    static float xu_vector[512];
    static float ry_vector[512];
    static float D_vector[512];

    static float AB_matrix[PROC_MU][ABMAT_DEPTH];
    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=AB_matrix complete dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=AB_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=AB_matrix complete dim=1
        // clang-format on
    }

    static float T_matrix[PROC_TU][TMAT_DEPTH];
    if (URAM_EN == 0) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=T_matrix complete dim=1
        #pragma HLS resource variable=T_matrix core=RAM_S2P_BRAM
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=T_matrix core=RAM_S2P_URAM
        #pragma HLS ARRAY_RESHAPE variable=T_matrix complete dim=1
        // clang-format on
    }

    static float Uq_matrix[UQMAT_DEPTH];
    if (URAM_EN == 1) {
// clang-format off
        #pragma HLS RESOURCE variable=Uq_matrix core=RAM_S2P_URAM
        // clang-format on
    }

    ap_uint<8> flag_reg = flag;

    if (EKF_EN == 1) {
        if (flag_reg[0] == 1) xu_vector[511] = 0;
    }

    if (flag_reg[0])
        initialization<N_STATE, M_MEAS, C_CTRL, PROC_TU, DEPTH_TU, PROC_MU, DEPTH_MU, DEPTH_MU_CTRL, UMAT_DEPTH,
                       HMAT_DEPTH, ABMAT_DEPTH, DPDQ_DEPTH, TMAT_DEPTH, UQMAT_DEPTH, URAM_EN, EKF_EN, TYPE, NPC>(
            A_mat,
#if KF_C != 0
            B_mat,
#endif
            Uq_mat, Dq_mat, H_mat, X0_mat, U0_mat, D0_mat, R_mat, H_matrix, U_matrix, xu_vector, ry_vector, D_vector,
            AB_matrix, T_matrix, Uq_matrix, flag_reg[7]);

    if (flag_reg[1])
        TimeUpdate<N_STATE, M_MEAS, C_CTRL, PROC_TU, DEPTH_TU, PROC_MU, DEPTH_MU, DEPTH_MU_CTRL, UMAT_DEPTH,
                   ABMAT_DEPTH, DPDQ_DEPTH, TMAT_DEPTH, URAM_EN, EKF_EN, TYPE, NPC>(
            T_matrix, AB_matrix, xu_vector, U_matrix, D_vector,
#if KF_C != 0
            u_mat,
#endif
            Xout_mat, Uout_mat, Dout_mat, flag_reg[3], flag_reg[4]);

    if (flag_reg[2])
        MeasUpdate_wrapper<N_STATE, C_CTRL, M_MEAS, PROC_TU, DEPTH_TU, PROC_MU, DEPTH_MU, UMAT_DEPTH, HMAT_DEPTH,
                           TMAT_DEPTH, UQMAT_DEPTH, URAM_EN, EKF_EN, TYPE, NPC>(
            U_matrix, H_matrix, D_vector, xu_vector, ry_vector, T_matrix, Uq_matrix,
#if KF_C != 0
            u_mat,
#endif
            y_mat, R_mat, H_mat, Xout_mat, Uout_mat, Dout_mat, flag_reg[5], flag_reg[6]);

    if (EKF_EN == 1) {
        if (flag_reg[2] == 1) xu_vector[511]++;
    }
}

#if KF_C != 0
#endif
#if KF_C != 0
#endif

template <int N_STATE,
          int M_MEAS,
          int C_CTRL,
          int MTU,
          int MMU,
          bool USE_URAM = 0,
          bool EKF_EN = 0,
          int TYPE,
          int NPC = 1>
void KalmanFilter(xf::cv::Mat<TYPE, N_STATE, N_STATE, NPC>& A_mat,
#if KF_C != 0
                  xf::cv::Mat<TYPE, N_STATE, C_CTRL, NPC>& B_mat,
#endif
                  xf::cv::Mat<TYPE, N_STATE, N_STATE, NPC>& Uq_mat,
                  xf::cv::Mat<TYPE, N_STATE, 1, NPC>& Dq_mat,
                  xf::cv::Mat<TYPE, M_MEAS, N_STATE, NPC>& H_mat,
                  xf::cv::Mat<TYPE, N_STATE, 1, NPC>& X0_mat,
                  xf::cv::Mat<TYPE, N_STATE, N_STATE, NPC>& U0_mat,
                  xf::cv::Mat<TYPE, N_STATE, 1, NPC>& D0_mat,
                  xf::cv::Mat<TYPE, M_MEAS, 1, NPC>& R_mat,
#if KF_C != 0
                  xf::cv::Mat<TYPE, C_CTRL, 1, NPC>& u_mat,
#endif
                  xf::cv::Mat<TYPE, M_MEAS, 1, NPC>& y_mat,
                  xf::cv::Mat<TYPE, N_STATE, 1, NPC>& Xout_mat,
                  xf::cv::Mat<TYPE, N_STATE, N_STATE, NPC>& Uout_mat,
                  xf::cv::Mat<TYPE, N_STATE, 1, NPC>& Dout_mat,
                  unsigned char flag) {
    assert((N_STATE > 0 && N_STATE <= 128) && "For N_STATE, possible options are 1 to 128");
    assert((M_MEAS > 0 && M_MEAS <= 128) && "For M_MEAS, possible options are 1 to 128");
    assert((C_CTRL >= 0 && C_CTRL <= 128) && "For C_CTRL, possible options are 0 to 128");
    assert((MTU > 0 && MTU <= N_STATE) && "For MTU, possible options are 1 to N_STATE");
    assert((MMU > 0 && MMU <= N_STATE) && "For MMU, possible options are 1 to N_STATE");
    assert(((A_mat.rows == N_STATE) && (A_mat.cols == N_STATE)) && "A matrix dimension must be N_STATE x N_STATE");
#if KF_C != 0
    assert(((B_mat.rows == N_STATE) && (B_mat.cols == C_CTRL)) && "B matrix dimension must be N_STATE x C_CTRL");
#endif
    assert(((Uq_mat.rows == N_STATE) && (Uq_mat.cols == N_STATE)) && "Uq matrix dimension must be N_STATE x N_STATE");
    assert(((Dq_mat.rows == N_STATE) && (Dq_mat.cols == 1)) && "Dq matrix dimension must be N_STATE x 1");
    assert(((H_mat.rows == M_MEAS) && (H_mat.cols == N_STATE)) && "H matrix dimension must be M_MEAS x N_STATE");
    assert(((X0_mat.rows == N_STATE) && (X0_mat.cols == 1)) && "X0 matrix dimension must be N_STATE x 1");
    assert(((U0_mat.rows == N_STATE) && (U0_mat.cols == N_STATE)) && "U0 matrix dimension must be N_STATE x N_STATE");
    assert(((D0_mat.rows == N_STATE) && (D0_mat.cols == 1)) && "D0 matrix dimension must be N_STATE x 1");
    assert(((R_mat.rows == M_MEAS) && (R_mat.cols == 1)) && "R matrix dimension must be M_MEAS x 1");
#if KF_C != 0
    assert(((u_mat.rows == C_CTRL) && (u_mat.cols == 1)) && "u matrix dimension must be C_CTRL x 1");
#endif
    assert(((y_mat.rows == M_MEAS) && (y_mat.cols == 1)) && "y matrix dimension must be M_MEAS x 1");
    assert(((Xout_mat.rows == N_STATE) && (Xout_mat.cols == 1)) && "Xout matrix dimension must be N_STATE x 1");
    assert(((Uout_mat.rows == N_STATE) && (Uout_mat.cols == N_STATE)) &&
           "Uout matrix dimension must be N_STATE x N_STATE");
    assert(((Dout_mat.rows == N_STATE) && (Dout_mat.cols == 1)) && "Dout matrix dimension must be N_STATE x 1");
    assert((TYPE == XF_32FC1) && "TYPE must be XF_32FC1");
    assert((NPC == XF_NPPC1) && "NPC must be XF_NPPC1");

    KalmanFilter_def<N_STATE, M_MEAS, C_CTRL, MTU, MMU, USE_URAM, EKF_EN, TYPE, NPC>(
        A_mat,
#if KF_C != 0
        B_mat,
#endif
        Uq_mat, Dq_mat, H_mat, X0_mat, U0_mat, D0_mat, R_mat,
#if KF_C != 0
        u_mat,
#endif
        y_mat, Xout_mat, Uout_mat, Dout_mat, flag);
}
} // namespace cv
} // namespace xf
#endif //_XF_KALMANFILTER_HPP_
