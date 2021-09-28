/*
 * Copyright 2021 Xilinx, Inc.
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
/*
This file holds the body of the kernel class for the Asymmetric Interpolation FIR.
Unlike single rate implementations, this interpolation FIR calculates sets of output
vectors in parallel such that the number of lanes in total is a multiple of the
interpolation factor.

Coding conventions
  TT_      template type suffix
  TP_      template parameter suffix
*/

#include <adf.h>

//#include <aie_api/aie_adf.hpp>
#ifndef __NEW_WINDOW_H__
#define __NEW_WINDOW_H__ 1
#endif
// if we use 1kb registers -> aie api uses 2x512b registers for 1024b so we need this for QoR
#ifndef __AIE_API_USE_NATIVE_1024B_VECTOR__
#define __AIE_API_USE_NATIVE_1024B_VECTOR__
#endif
#include "aie_api/aie_adf.hpp"

#include "matrix_mult.hpp" //hence including matrix_mult_traits.hpp too

//#define _DSPLIB_MATRIX_MULT_HPP_DEBUG_

namespace xf {
namespace dsp {
namespace aie {
namespace blas {
namespace matrix_mult {

// aie_api is external to xf::dsp::aie namespace
namespace aie = ::aie;
// doesn't import properly from aie_api/utils.hpp
template <typename T, unsigned Elems>
void print(const aie::vector<T, Elems>& v, bool nl = false, const char* prefix = nullptr) {
    if (prefix) printf("%s", prefix);

    using vector_type = aie::vector<T, Elems>;

    for (unsigned i = 0; i < Elems; ++i) {
        T e = v[i];

        if
            constexpr(vector_type::is_complex()) {
                if
                    constexpr(vector_type::is_floating_point()) printf("%f %f ", (float)e.real, (float)e.imag);
                else
                    printf("%d %d ", (int)e.real, (int)e.imag);
            }
        else {
            if
                constexpr(vector_type::is_floating_point()) printf("%f ", (float)e);
            else if
                constexpr(!vector_type::is_signed()) printf("%u ", (unsigned)e);
            else
                printf("%d ", (int)e);
        }
    }

    if (nl) printf("\n");
}

//-----------------------------------------------------------------------------------------------------
//#TEMPLATE_FUNCTION_DEFINITION
template <typename TT_DATA_A,
          typename TT_DATA_B,
          unsigned int TP_DIM_A,
          unsigned int TP_DIM_AB,
          unsigned int TP_DIM_B,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_DIM_A_LEADING, // = ROW_MAJOR,
          unsigned int TP_DIM_B_LEADING, // = COL_MAJOR,
          unsigned int TP_DIM_OUT_LEADING,
          unsigned int TP_INPUT_WINDOW_VSIZE_A,
          unsigned int TP_INPUT_WINDOW_VSIZE_B,
          bool TP_CASC_IN,
          bool TP_CASC_OUT,
          unsigned int TP_DIM_A_RANGE,
          unsigned int TP_DIM_AB_RANGE,
          unsigned int TP_DIM_B_RANGE,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
inline void kernelMatMultClass<TT_DATA_A,
                               TT_DATA_B,
                               TP_DIM_A,
                               TP_DIM_AB,
                               TP_DIM_B,
                               TP_SHIFT,
                               TP_RND,
                               TP_DIM_A_LEADING,
                               TP_DIM_B_LEADING,
                               TP_DIM_OUT_LEADING,
                               TP_INPUT_WINDOW_VSIZE_A,
                               TP_INPUT_WINDOW_VSIZE_B,
                               TP_CASC_IN,
                               TP_CASC_OUT,
                               TP_DIM_A_RANGE,
                               TP_DIM_AB_RANGE,
                               TP_DIM_B_RANGE,
                               TP_KERNEL_POSITION,
                               TP_CASC_LEN>::matMultKernel(T_inputIF<TP_CASC_IN, TT_DATA_A, TT_DATA_B> inInterface,
                                                           T_outputIF<TP_CASC_OUT, TT_DATA_A, TT_DATA_B> outInterface) {
    // This function hides exposure of the implementation choice from the user.
    matMult_impl1(inInterface, outInterface);
};

template <typename TT_DATA_A,
          typename TT_DATA_B,
          unsigned int TP_DIM_A,
          unsigned int TP_DIM_AB,
          unsigned int TP_DIM_B,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_DIM_A_LEADING, // = ROW_MAJOR,
          unsigned int TP_DIM_B_LEADING, // = COL_MAJOR,
          unsigned int TP_DIM_OUT_LEADING,
          unsigned int TP_INPUT_WINDOW_VSIZE_A,
          unsigned int TP_INPUT_WINDOW_VSIZE_B,
          bool TP_CASC_IN,
          bool TP_CASC_OUT,
          unsigned int TP_DIM_A_RANGE,
          unsigned int TP_DIM_AB_RANGE,
          unsigned int TP_DIM_B_RANGE,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
inline void kernelMatMultClass<TT_DATA_A,
                               TT_DATA_B,
                               TP_DIM_A,
                               TP_DIM_AB,
                               TP_DIM_B,
                               TP_SHIFT,
                               TP_RND,
                               TP_DIM_A_LEADING,
                               TP_DIM_B_LEADING,
                               TP_DIM_OUT_LEADING,
                               TP_INPUT_WINDOW_VSIZE_A,
                               TP_INPUT_WINDOW_VSIZE_B,
                               TP_CASC_IN,
                               TP_CASC_OUT,
                               TP_DIM_A_RANGE,
                               TP_DIM_AB_RANGE,
                               TP_DIM_B_RANGE,
                               TP_KERNEL_POSITION,
                               TP_CASC_LEN>::matMult_impl1(T_inputIF<TP_CASC_IN, TT_DATA_A, TT_DATA_B> inInterface,
                                                           T_outputIF<TP_CASC_OUT, TT_DATA_A, TT_DATA_B> outInterface) {
    set_rnd(TP_RND);
    set_sat();
    constexpr unsigned int M = tilingScheme.Atile;
    constexpr unsigned int N = tilingScheme.ABtile;
    constexpr unsigned int K = tilingScheme.Btile;
    constexpr unsigned int sizeTileA = M * N;
    constexpr unsigned int sizeTileB = N * K;
    constexpr unsigned int sizeTileC = M * K;
    constexpr bool parrallelA = (M < TP_DIM_A);
    constexpr bool parrallelB = (K < TP_DIM_B);
    using MMUL = aie::mmul<M, N, K, TT_DATA_A, TT_DATA_B, accType_t<TT_DATA_A, TT_DATA_B> >;

    const unsigned int numAReg = 2;
    const unsigned int numBReg = 2;
    // printf("M %d, N %d, K%d\n", M, N, K);
    // TT_DATA_A tiledWindowA[TP_DIM_A * TP_DIM_AB];
    // doTiling<M,N, TP_DIM_A, TP_DIM_AB, TP_DIM_A_LEADING>(inInterface.inWindowA, &tiledWindowA[0]);
    // TT_DATA_B tiledWindowB[TP_DIM_AB * TP_DIM_B];
    // doTiling<N,K, TP_DIM_AB, TP_DIM_B, TP_DIM_B_LEADING>(inInterface.inWindowB, &tiledWindowB[0]);
    TT_DATA_A* inputAPtr =
        //&tiledWindowA[0];
        (TT_DATA_A*)inInterface.inWindowA->ptr;
    TT_DATA_B* inputBPtr =
        //&tiledWindowB[0];
        (TT_DATA_B*)inInterface.inWindowB->ptr;
    // TT_OUT* outWindowPtr = (TT_OUT*) outInterface.outWindow->ptr;

    // TT_OUT tiledOutWindow[TP_DIM_A * TP_DIM_B];
    TT_OUT* tiledOutWindowPtr;
    if
        constexpr(TP_CASC_OUT == CASC_OUT_FALSE) { tiledOutWindowPtr = (TT_OUT*)outInterface.outWindow->ptr; }
    else {
        tiledOutWindowPtr = nullptr;
    }

    for (unsigned AChunk = 0; AChunk < TP_DIM_A / M; AChunk += numAReg)
        chess_prepare_for_pipelining chess_loop_count((TP_DIM_A / M) / numAReg) // TP_DIM_A)
        {
            // window pointer for output
            TT_OUT* restrict pC1 = (TP_CASC_OUT == CASC_OUT_FALSE)
                                       ? tiledOutWindowPtr + ((AChunk * TP_DIM_B / K + 0) * sizeTileC)
                                       : nullptr;
            TT_OUT* restrict pC2 = (TP_CASC_OUT == CASC_OUT_FALSE && parrallelA)
                                       ? tiledOutWindowPtr + (((AChunk + 1) * TP_DIM_B / K + 0) * sizeTileC)
                                       : nullptr;

            for (unsigned BChunk = 0; BChunk < TP_DIM_B / K; BChunk += numBReg)
                chess_prepare_for_pipelining chess_loop_count((TP_DIM_B / K) / numBReg) // TP_DIM_B/K)
                {
                    const TT_DATA_A* restrict pA1 = inputAPtr + ((AChunk * TP_DIM_AB / N + 0) * sizeTileA);
                    const TT_DATA_A* restrict pA2;
                    if
                        constexpr(parrallelA) { pA2 = inputAPtr + (((AChunk + 1) * TP_DIM_AB / N + 0) * sizeTileA); }
                    const TT_DATA_B* restrict pB1 = inputBPtr + ((0 * TP_DIM_B / K + BChunk) * sizeTileB);
                    const TT_DATA_B* restrict pB2;
                    if
                        constexpr(parrallelB) { pB2 = inputBPtr + ((0 * TP_DIM_B / K + (BChunk + 1)) * sizeTileB); }

                    aie::vector<TT_DATA_A, sizeTileA> A0 = aie::load_v<sizeTileA>(pA1);
                    pA1 += sizeTileA;
                    aie::vector<TT_DATA_A, sizeTileA> A1;
                    if
                        constexpr(parrallelA) {
                            A1 = aie::load_v<sizeTileA>(pA2);
                            pA2 += sizeTileA;
                        }

                    aie::vector<TT_DATA_B, sizeTileB> B0 = aie::load_v<sizeTileB>(pB1);
                    pB1 += sizeTileB * TP_DIM_B / K;
                    aie::vector<TT_DATA_B, sizeTileB> B1;
                    if
                        constexpr(parrallelB) {
                            B1 = aie::load_v<sizeTileB>(pB2);
                            pB2 += sizeTileB * TP_DIM_B / K;
                        }

                    // initial muls
                    MMUL C00;
                    MMUL C01;
                    MMUL C10;
                    MMUL C11;
                    if
                        constexpr(TP_CASC_IN == CASC_IN_TRUE) {
                            // hoping AIE API readincr_v will infer the lanes.
                            // guess for sizeTileC
                            // could also try to replae sizeTileC with MMUL::accum_type::size()   or just
                            // MMUL::accum_type::Elems
                            C00 = MMUL(readincr_v<sizeTileC>(inInterface.inCascade));
                            if
                                constexpr(parrallelB) C01 = MMUL(readincr_v<sizeTileC>(inInterface.inCascade));
                            if
                                constexpr(parrallelA) C10 = MMUL(readincr_v<sizeTileC>(inInterface.inCascade));
                            if
                                constexpr(parrallelA && parrallelB) C11 =
                                    MMUL(readincr_v<sizeTileC>(inInterface.inCascade));
                            C00.mac(A0, B0);
                            if
                                constexpr(parrallelB) C01.mac(A0, B1);
                            if
                                constexpr(parrallelA) C10.mac(A1, B0);
                            if
                                constexpr(parrallelA && parrallelB) C11.mac(A1, B1);
                        }
                    else {
                        C00.mul(A0, B0);
                        if
                            constexpr(parrallelB) C01.mul(A0, B1);
                        if
                            constexpr(parrallelA) C10.mul(A1, B0);
                        if
                            constexpr(parrallelA && parrallelB) C11.mul(A1, B1);
                    }

                    //#pragma unroll((TP_DIM_AB/N -1))
                    for (unsigned i = 1; i < TP_DIM_AB / N; ++i)
                        chess_loop_count((TP_DIM_AB / N - 1)) chess_prepare_for_pipelining {
                            // print(A0,true,"A0: ");
                            // print(A1,true,"A1: ");
                            // print(B0,true,"B0: ");
                            // print(B1,true,"B1: ");
                            A0 = aie::load_v<sizeTileA>(pA1);
                            pA1 += sizeTileA;
                            if
                                constexpr(parrallelA) {
                                    A1 = aie::load_v<sizeTileA>(pA2);
                                    pA2 += sizeTileA;
                                }

                            B0 = aie::load_v<sizeTileB>(pB1);
                            pB1 += sizeTileB * TP_DIM_B / K;
                            if
                                constexpr(parrallelB) {
                                    B1 = aie::load_v<sizeTileB>(pB2);
                                    pB2 += sizeTileB * TP_DIM_B / K;
                                }

                            // print(C00.template to_vector<TT_OUT>(TP_SHIFT),true,"C00: ");
                            // print(C01.template to_vector<TT_OUT>(TP_SHIFT),true,"C01: ");
                            // print(C10.template to_vector<TT_OUT>(TP_SHIFT),true,"C10: ");
                            // print(C11.template to_vector<TT_OUT>(TP_SHIFT),true,"C11: ");
                            C00.mac(A0, B0);
                            if
                                constexpr(parrallelB) { C01.mac(A0, B1); }
                            if
                                constexpr(parrallelA) { C10.mac(A1, B0); }
                            if
                                constexpr(parrallelA && parrallelB) { C11.mac(A1, B1); }
                        }

                    // print(A0,true,"FinalA0: ");
                    // print(A1,true,"FinalA1: ");
                    // print(B0,true,"FinalB0: ");
                    // print(B1,true,"FinalB1: ");

                    // print(C00.template to_vector<TT_OUT>(TP_SHIFT),true,"FinalC00: ");
                    // print(C01.template to_vector<TT_OUT>(TP_SHIFT),true,"FinalC01: ");
                    // print(C10.template to_vector<TT_OUT>(TP_SHIFT),true,"FinalC10: ");
                    // print(C11.template to_vector<TT_OUT>(TP_SHIFT),true,"FinalC11: ");
                    // TT_DATA_A * restrict pAtest1 = ((TT_DATA_A *)pA1)+4*sizeTileA; aie::store_v(pAtest1, C00.template
                    // to_vector<TT_OUT>(TP_SHIFT)); //pA1 -= 4*sizeTileA;
                    if
                        constexpr(TP_CASC_OUT == CASC_OUT_FALSE) {
                            aie::store_v(pC1, C00.template to_vector<TT_OUT>(TP_SHIFT));
                            pC1 += sizeTileC;
                            if
                                constexpr(parrallelB) {
                                    aie::store_v(pC1, C01.template to_vector<TT_OUT>(TP_SHIFT));
                                    pC1 += sizeTileC;
                                }
                            if
                                constexpr(parrallelA) {
                                    aie::store_v(pC2, C10.template to_vector<TT_OUT>(TP_SHIFT));
                                    pC2 += sizeTileC;
                                }
                            if
                                constexpr(parrallelA && parrallelB) {
                                    aie::store_v(pC2, C11.template to_vector<TT_OUT>(TP_SHIFT));
                                    pC2 += sizeTileC;
                                }
                        }
                    else {
                        writeincr(outInterface.outWindow, C00.to_accum());
                        if
                            constexpr(parrallelB) { writeincr(outInterface.outWindow, C01.to_accum()); }
                        if
                            constexpr(parrallelA) { writeincr(outInterface.outWindow, C10.to_accum()); }
                        if
                            constexpr(parrallelA && parrallelB) { writeincr(outInterface.outWindow, C11.to_accum()); }
                    }
                }
        }

    // doUnTiling<M,K, TP_DIM_A, TP_DIM_B, TP_DIM_OUT_LEADING>(tiledOutWindowPtr, outWindowPtr);
}

// function overloaded with cascade interface variations
// This is a specialization of the main class for when there is only one kernel for the whole function.
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA_A,
          typename TT_DATA_B,
          unsigned int TP_DIM_A,
          unsigned int TP_DIM_AB,
          unsigned int TP_DIM_B,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_DIM_A_LEADING, // = ROW_MAJOR,
          unsigned int TP_DIM_B_LEADING, // = COL_MAJOR,
          unsigned int TP_DIM_OUT_LEADING,
          unsigned int TP_INPUT_WINDOW_VSIZE_A,
          unsigned int TP_INPUT_WINDOW_VSIZE_B,
          bool TP_CASC_IN,
          bool TP_CASC_OUT,
          unsigned int TP_DIM_A_RANGE,
          unsigned int TP_DIM_AB_RANGE,
          unsigned int TP_DIM_B_RANGE,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void matrix_mult<TT_DATA_A,
                 TT_DATA_B,
                 TP_DIM_A,
                 TP_DIM_AB,
                 TP_DIM_B,
                 TP_SHIFT,
                 TP_RND,
                 TP_DIM_A_LEADING,
                 TP_DIM_B_LEADING,
                 TP_DIM_OUT_LEADING,
                 TP_INPUT_WINDOW_VSIZE_A,
                 TP_INPUT_WINDOW_VSIZE_B,
                 TP_CASC_IN,
                 TP_CASC_OUT,
                 TP_DIM_A_RANGE,
                 TP_DIM_AB_RANGE,
                 TP_DIM_B_RANGE,
                 TP_KERNEL_POSITION,
                 TP_CASC_LEN>::matMult(input_window<TT_DATA_A>* inWindowA,
                                       input_window<TT_DATA_B>* inWindowB,
                                       output_window<outType_t<TT_DATA_A, TT_DATA_B> >* outWindow) {
    T_inputIF<CASC_IN_FALSE, TT_DATA_A, TT_DATA_B> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA_A, TT_DATA_B> outInterface;
    inInterface.inWindowA = inWindowA;
    inInterface.inWindowB = inWindowB;
    outInterface.outWindow = outWindow;
    this->matMultKernel(inInterface, outInterface);
};

// function overloaded with cascade interface variations
// This is a specialization of the main class for the final kernel in a cascade chain.
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA_A,
          typename TT_DATA_B,
          unsigned int TP_DIM_A,
          unsigned int TP_DIM_AB,
          unsigned int TP_DIM_B,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_DIM_A_LEADING, // = ROW_MAJOR,
          unsigned int TP_DIM_B_LEADING, // = COL_MAJOR,
          unsigned int TP_DIM_OUT_LEADING,
          unsigned int TP_INPUT_WINDOW_VSIZE_A,
          unsigned int TP_INPUT_WINDOW_VSIZE_B,
          unsigned int TP_DIM_A_RANGE,
          unsigned int TP_DIM_AB_RANGE,
          unsigned int TP_DIM_B_RANGE,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void matrix_mult<TT_DATA_A,
                 TT_DATA_B,
                 TP_DIM_A,
                 TP_DIM_AB,
                 TP_DIM_B,
                 TP_SHIFT,
                 TP_RND,
                 TP_DIM_A_LEADING,
                 TP_DIM_B_LEADING,
                 TP_DIM_OUT_LEADING,
                 TP_INPUT_WINDOW_VSIZE_A,
                 TP_INPUT_WINDOW_VSIZE_B,
                 CASC_IN_TRUE,
                 CASC_OUT_FALSE,
                 TP_DIM_A_RANGE,
                 TP_DIM_AB_RANGE,
                 TP_DIM_B_RANGE,
                 TP_KERNEL_POSITION,
                 TP_CASC_LEN>::matMult(input_window<TT_DATA_A>* inWindowA,
                                       input_window<TT_DATA_B>* inWindowB,
                                       input_stream<accType_t<TT_DATA_A, TT_DATA_B> >* inCascade,
                                       output_window<outType_t<TT_DATA_A, TT_DATA_B> >* outWindow) {
    T_inputIF<CASC_IN_TRUE, TT_DATA_A, TT_DATA_B> inInterface;
    T_outputIF<CASC_OUT_FALSE, TT_DATA_A, TT_DATA_B> outInterface;
    inInterface.inWindowA = inWindowA;
    inInterface.inWindowB = inWindowB;
    inInterface.inCascade = inCascade;
    outInterface.outWindow = outWindow;
    this->matMultKernel(inInterface, outInterface);
};

// function overloaded with cascade interface variations
// This is a specialization of the main class for the first kernel in a cascade chain.
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA_A,
          typename TT_DATA_B,
          unsigned int TP_DIM_A,
          unsigned int TP_DIM_AB,
          unsigned int TP_DIM_B,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_DIM_A_LEADING, // = ROW_MAJOR,
          unsigned int TP_DIM_B_LEADING, // = COL_MAJOR,
          unsigned int TP_DIM_OUT_LEADING,
          unsigned int TP_INPUT_WINDOW_VSIZE_A,
          unsigned int TP_INPUT_WINDOW_VSIZE_B,
          unsigned int TP_DIM_A_RANGE,
          unsigned int TP_DIM_AB_RANGE,
          unsigned int TP_DIM_B_RANGE,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void matrix_mult<TT_DATA_A,
                 TT_DATA_B,
                 TP_DIM_A,
                 TP_DIM_AB,
                 TP_DIM_B,
                 TP_SHIFT,
                 TP_RND,
                 TP_DIM_A_LEADING,
                 TP_DIM_B_LEADING,
                 TP_DIM_OUT_LEADING,
                 TP_INPUT_WINDOW_VSIZE_A,
                 TP_INPUT_WINDOW_VSIZE_B,
                 CASC_IN_FALSE,
                 CASC_OUT_TRUE,
                 TP_DIM_A_RANGE,
                 TP_DIM_AB_RANGE,
                 TP_DIM_B_RANGE,
                 TP_KERNEL_POSITION,
                 TP_CASC_LEN>::matMult(input_window<TT_DATA_A>* inWindowA,
                                       input_window<TT_DATA_B>* inWindowB,
                                       output_stream<accType_t<TT_DATA_A, TT_DATA_B> >* outCascade) {
    T_inputIF<CASC_IN_FALSE, TT_DATA_A, TT_DATA_B> inInterface;
    T_outputIF<CASC_OUT_TRUE, TT_DATA_A, TT_DATA_B> outInterface;
    inInterface.inWindowA = inWindowA;
    inInterface.inWindowB = inWindowB;
    outInterface.outWindow = outCascade; // toodo rename outWindow to just outPort
    this->matMultKernel(inInterface, outInterface);
};

// function overloaded with cascade interface variations
// This is a specialization of the main class for any kernel within a cascade chain, but neither first nor last.
//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA_A,
          typename TT_DATA_B,
          unsigned int TP_DIM_A,
          unsigned int TP_DIM_AB,
          unsigned int TP_DIM_B,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_DIM_A_LEADING, // = ROW_MAJOR,
          unsigned int TP_DIM_B_LEADING, // = COL_MAJOR,
          unsigned int TP_DIM_OUT_LEADING,
          unsigned int TP_INPUT_WINDOW_VSIZE_A,
          unsigned int TP_INPUT_WINDOW_VSIZE_B,
          unsigned int TP_DIM_A_RANGE,
          unsigned int TP_DIM_AB_RANGE,
          unsigned int TP_DIM_B_RANGE,
          unsigned int TP_KERNEL_POSITION,
          unsigned int TP_CASC_LEN>
void matrix_mult<TT_DATA_A,
                 TT_DATA_B,
                 TP_DIM_A,
                 TP_DIM_AB,
                 TP_DIM_B,
                 TP_SHIFT,
                 TP_RND,
                 TP_DIM_A_LEADING,
                 TP_DIM_B_LEADING,
                 TP_DIM_OUT_LEADING,
                 TP_INPUT_WINDOW_VSIZE_A,
                 TP_INPUT_WINDOW_VSIZE_B,
                 CASC_IN_TRUE,
                 CASC_OUT_TRUE,
                 TP_DIM_A_RANGE,
                 TP_DIM_AB_RANGE,
                 TP_DIM_B_RANGE,
                 TP_KERNEL_POSITION,
                 TP_CASC_LEN>::matMult(input_window<TT_DATA_A>* inWindowA,
                                       input_window<TT_DATA_B>* inWindowB,
                                       input_stream<accType_t<TT_DATA_A, TT_DATA_B> >* inCascade,
                                       output_stream<accType_t<TT_DATA_A, TT_DATA_B> >* outCascade) {
    T_inputIF<CASC_IN_TRUE, TT_DATA_A, TT_DATA_B> inInterface;
    T_outputIF<CASC_OUT_TRUE, TT_DATA_A, TT_DATA_B> outInterface;
    inInterface.inWindowA = inWindowA;
    inInterface.inWindowB = inWindowB;
    inInterface.inCascade = inCascade;
    outInterface.outWindow = outCascade;
    this->matMultKernel(inInterface, outInterface);
};
}
}
}
}
}
