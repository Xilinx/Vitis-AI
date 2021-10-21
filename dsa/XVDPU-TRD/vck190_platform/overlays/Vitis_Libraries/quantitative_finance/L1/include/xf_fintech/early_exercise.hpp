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
 * @file early_exercise.hpp
 * @brief This file contains implementaiton of pricing early exericse option
 * withe least-square approch
 *
 */

#ifndef XF_FINTECH_EARLYEX_H
#define XF_FINTECH_EARLYEX_H
#include "xf_fintech/jacobi_svd.hpp"
#include "xf_fintech/mc_simulation.hpp"
#include "xf_fintech/path_generator.hpp"
#include "xf_fintech/path_pricer.hpp"
#include "xf_fintech/utils.hpp"
#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {
namespace fintech {
namespace internal {

/// @brief generate the AtA from price data
template <typename DT, int COEFNM, int SampNum, bool StepFirst>
void GenAtA(ap_uint<16> steps,
            DT underlying,
            DT strike,
            DT invStk,
            hls::stream<DT>& priceStrmIn,
            hls::stream<DT>& outStrm,
            hls::stream<DT> matrixOut[3 * (COEFNM - 1)]) {
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
    std::cout << "--------------------Start of GenAtA--------------------" << std::endl;
#endif
#endif
    static const int BUFDEP = 16;
    // ap_uint<4> index = 0;
    static const int OUTNM = 3 * (COEFNM - 1);
    DT buff[OUTNM][BUFDEP];
#pragma HLS array_partition variable = buff dim = 1
AtA_LOOP:
    for (int i = 0; i < steps; ++i) {
#pragma HLS loop_tripcount min = 8 max = 8
        for (int j = 0; j < SampNum; ++j) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1024 max = 1024
            DT in1 = priceStrmIn.read();
            DT in = FPTwoMul(underlying, in1);
            outStrm.write(in);
            // DT tempIn   = FPTwoMul(in1, 0.9);
            DT tempIn0 = FPTwoMul(in1, underlying);
            DT tempIn = FPTwoMul(tempIn0, invStk);

            DT sub = FPTwoSub<DT>(strike, in);
            DT exercise = MAX(sub, 0);
            DT squareIn = FPTwoMul<DT>(tempIn, tempIn);
            DT pre[OUTNM];
#pragma HLS array_partition variable = pre
            for (int k = 0; k < OUTNM; ++k) {
#pragma HLS unroll
                DT tp = buff[k][(i * SampNum + j) % 16];
                if (j < BUFDEP)
                    pre[k] = 0;
                else
                    pre[k] = tp;
                if (i != 0 && j < BUFDEP) matrixOut[k].write(tp);
            }
            DT newD[OUTNM];
#pragma HLS array_partition variable = newD
            DT mul = FPTwoMul(tempIn, squareIn);
            DT mul2 = FPTwoMul(squareIn, squareIn);
            DT mul3 = FPTwoMul(tempIn, exercise);
            DT mul4 = FPTwoMul(squareIn, exercise);
            DT mul5 = FPTwoMul(exercise, exercise);
            if (exercise > 0) {
                newD[0] = FPTwoAdd(pre[0], 1.0);
                newD[1] = FPTwoAdd(pre[1], tempIn);
                newD[2] = FPTwoAdd(pre[2], squareIn);
                newD[3] = FPTwoAdd(pre[3], mul);
                newD[4] = FPTwoAdd(pre[4], mul2);
                newD[5] = FPTwoAdd(pre[5], exercise);
                newD[6] = FPTwoAdd(pre[6], mul3);
                newD[7] = FPTwoAdd(pre[7], mul4);
                newD[8] = FPTwoAdd(pre[8], mul5);
            } else {
                for (int k = 0; k < OUTNM; ++k) {
#pragma HLS unroll
                    newD[k] = pre[k];
                }
            }
            for (int k = 0; k < OUTNM; ++k) {
#pragma HLS unroll
#ifndef __SYNTHESIS__
//               if(k==0) std::cout<<std::hex<<"buff
//               ="<<doubleToBits(newD[k])<<std::endl;
#endif
                buff[k][(i * SampNum + j) % 16] = newD[k];
            }
            // index++;
        }
    }
    for (int i = 0; i < BUFDEP; ++i) {
#pragma HLS pipeline II = 1
        for (int k = 0; k < OUTNM; k++) {
            matrixOut[k].write(buff[k][i]);
        }
    }
}
///@brief accumulate the AtA matrix data to a new AtA and output as stream
template <typename DT, int COEFNM>
void MergeBuff(ap_uint<27> steps, hls::stream<DT> matrixIn[3 * (COEFNM - 1)], hls::stream<DT>& outStrm) {
    // ii=1 latency = 288
    static const unsigned int INNM = 3 * (COEFNM - 1);
    const static int BuffDep = 16;
    DT sum[INNM]; // = 0;
    for (int i = 0; i < steps; ++i) {
#pragma HLS loop_tripcount min = 10 max = 10
        for (int m = 0; m < BuffDep; ++m) {
#pragma HLS pipeline II = 9
            for (int k = 0; k < INNM; ++k) {
#pragma HLS loop_tripcount min = 9 max = 9
                DT op1;
                if (m == 0) {
                    op1 = 0;
                } else {
                    op1 = sum[k];
                }
                DT op2 = matrixIn[k].read();
#ifndef __SYNTHESIS__
//            std::cout<<std::hex<<"martix piece
//            ="<<doubleToBits(op2)<<std::endl;
#endif
                DT t = FPTwoAdd(op1, op2);
                sum[k] = t;
                if (m == BuffDep - 1) {
#ifndef __SYNTHESIS__
//              std::cout<<std::hex<<"martix total
//              ="<<doubleToBits(t)<<std::endl;
#endif
                    outStrm.write(t);
                }
            }
        }
    }
}

template <typename DT, int COEFNM>
void CalcLinear(hls::stream<DT>& Ustrm, hls::stream<DT>& Vstrm, hls::stream<DT>& Sstrm, DT Y[COEFNM], DT coef[COEFNM]) {
#pragma HLS inline
    DT threshold = 1e-10; // COEFNM* precision * max;
    DT S[COEFNM][COEFNM];
    DT U[COEFNM][COEFNM];
    DT V[COEFNM][COEFNM];
    DT pdt[COEFNM];
    DT w[COEFNM];
    bool sGtZero[COEFNM];
ReadUVS_Init_Loop:
    for (int k = 0; k < COEFNM; ++k) {
        for (int l = 0; l < COEFNM; ++l) {
#pragma HLS pipeline II = 1
            U[k][l] = Ustrm.read();
            V[k][l] = Vstrm.read();
            S[k][l] = Sstrm.read();
            pdt[k] = 0;
            coef[k] = 0;
            w[k] = 0;
        }
    }
///////////////////////////////////
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
    std::cout << "S matrix" << std::endl;
    for (int m = 0; m < COEFNM; ++m) {
        for (int n = 0; n < COEFNM; ++n) {
            // std::cout<<std::hex<<doubleToBits(S[m][n])<<", ";
            std::cout << S[m][n] << ", ";
        }
        std::cout << std::endl;
        // coefStrmOut[m].write(coef[m]);
    }
    std::cout << "U matrix" << std::endl;
    for (int m = 0; m < COEFNM; ++m) {
        for (int n = 0; n < COEFNM; ++n) {
            // std::cout<<std::hex<<doubleToBits(U[m][n])<<", ";
            std::cout << U[m][n] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "V matrix" << std::endl;
    for (int m = 0; m < COEFNM; ++m) {
        for (int n = 0; n < COEFNM; ++n) {
            // std::cout<<std::hex<<doubleToBits(V[m][n])<<", ";
            std::cout << V[m][n] << ", ";
        }
        std::cout << std::endl;
    }
#endif
#endif
// Matrix 4*4 * 4*1
LinearSolve_Loop:
    for (int i = 0; i < COEFNM; ++i) {
#pragma HLS pipeline II = 12
        DT product = 0;
        for (int j = 0; j < COEFNM; ++j) {
            DT m = FPTwoMul(U[j][i], Y[j]);
            product = FPTwoAdd(product, m);
        }
        DT u = FPTwoMul(product, S[i][i]);
        for (int j = 0; j < COEFNM; ++j) {
            DT m1 = FPTwoMul(V[j][i], u);
            DT tmp_coef = FPTwoAdd(coef[j], m1);
            coef[j] = tmp_coef;
        }
    }
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
    std::cout << "The Y is " << Y[0] << " " << Y[1] << " " << Y[2] << " " << Y[3] << std::endl;
    std::cout << "The Ax=Y FOR 4X4 MATRIX solution is \n"
              << coef[0] << " " << coef[1] << " " << coef[2] << " " << coef[3] << std::endl;
#endif
#endif
}

template <typename DT, int COEFNM>
void calculateLinear(DT U[COEFNM][COEFNM], DT vS[COEFNM], DT V[COEFNM][COEFNM], DT Y[COEFNM], int mm, DT Coef[COEFNM]) {
    DT product;
    DT u, max, threshold, precision;
    // precision = 1.17549435e-38;//for float 1.17549435e-38
    precision = 2.2250738585072014e-308; // for float 1.17549435e-38
    // for double 2.2250738585072014e-308
    // threshold = 1e-2;//COEFNM* precision * max;
    max = 0;
    for (int j = 0; j < COEFNM; ++j) {
        Coef[j] = 0;
        max = (max < vS[j]) ? vS[j] : max;
    }
    threshold = 1e-10; // COEFNM* precision * max;
    for (int i = 0; i < COEFNM; ++i) {
        if (vS[i] > threshold) { // 1e-2){//3.706894e-4) {
            product = 0;
            for (int z = 0; z < mm; ++z) {
                product += U[z][i] * Y[z];
            }
            for (int j = 0; j < COEFNM; ++j) {
                u = product / vS[i];
                Coef[j] += u * V[j][i];
            }
        }
    }
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
    std::cout << "The Y is " << Y[0] << " " << Y[1] << " " << Y[2] << " " << Y[3] << std::endl;
    std::cout << "The Ax=Y FOR 4X4 MATRIX solution is \n"
              << Coef[0] << " " << Coef[1] << " " << Coef[2] << " " << Coef[3] << std::endl;
#endif
#endif
}
///@brief The American engine Monte Carlo Model for calibration. Process 1024
/// paths (samples) per time step
// @param priceStrm the price stream output
// @param matrixStrm the matrix stream output
template <typename DT,
          typename RNG,
          typename PathGeneratorT,
          typename PathPricerT,
          typename RNGSeqT,
          int VariateNum,
          int SampNum,
          int COEFNM>
void MonteCarloModel(ap_uint<16> steps,
                     DT underlying,
                     DT strike,
                     DT invStk,
                     RNG rngInst[VariateNum],
                     PathGeneratorT pathGenInst[1],
                     PathPricerT pathPriInst[1],
                     RNGSeqT rngSeqInst[1],
                     hls::stream<DT>& priceStrm,
                     hls::stream<DT>& matrixStrm) {
#pragma HLS inline off
#pragma HLS DATAFLOW
    const static unsigned int RandN = RNGSeqT::OutN;
    const static unsigned int PathN = PathPricerT::InN;

    hls::stream<DT> randNumberStrm[RandN];
#pragma HLS stream variable = randNumberStrm depth = 8
    hls::stream<DT> pathStrm[PathN];
#pragma HLS stream variable = pathStrm depth = 8

    hls::stream<DT> pStrm[PathN];
#pragma HLS stream variable = pStrm depth = 9

    hls::stream<DT> xStrm[3 * (COEFNM - 1)];
#pragma HLS stream variable = xStrm depth = 16

    // RNGSequence<DT, RNG, VariateNum, SampNum>(steps, rngInst, randNumberStrm);
    if (PathN == 1) {
        rngSeqInst[0].NextSeq(steps, SampNum, rngInst, randNumberStrm);
        pathGenInst[0].NextPath(steps, SampNum, randNumberStrm, pathStrm);
        pathPriInst[0].Pricing(steps, pathStrm, pStrm);
        GenAtA<DT, COEFNM, SampNum, false>(steps, underlying, strike, invStk, pStrm[0], priceStrm, xStrm);
        MergeBuff<DT, COEFNM>(steps, xStrm, matrixStrm);
    } else {
        // No Path Generator
        rngSeqInst[0].NextSeq(steps, SampNum, rngInst, randNumberStrm);
        //        pathPriInst[0].Pricing(steps, randNumberStrm, priceStrm);
    }
}

///@brief Monte Carlo with an unroll number UN,
//  each Monte Carlo Model processes 1024 samples(paths)
// @param priceOutStrm[UN] each monte carlo model generates one price stream
// data
// @param matrixStrm[UN] each monte Carlo model generates one matrix stream data
template <typename DT,
          typename RNG,
          int UN,
          typename PathGeneratorT,
          typename PathPricerT,
          typename RNGSeqT,
          int VariateNum,
          int SampNum,
          int COEFNM>
void MultiMonteCarloModel(ap_uint<16> steps,
                          DT underlying,
                          DT strike,
                          DT invStk,
                          RNG rngInst[UN][VariateNum],
                          PathGeneratorT pathGenInst[UN][1],
                          PathPricerT pathPriInst[UN][1],
                          RNGSeqT rngSeqInst[UN][1],
                          hls::stream<DT> priceOutStrm[UN],
                          hls::stream<DT> matrixStrm[UN]) {
    for (int i = 0; i < UN; ++i) {
#pragma HLS unroll
        MonteCarloModel<DT, RNG, PathGeneratorT, PathPricerT, RNGSeqT, VariateNum, SampNum, COEFNM>(
            steps, underlying, strike, invStk, rngInst[i], pathGenInst[i], pathPriInst[i], rngSeqInst[i],
            priceOutStrm[i], matrixStrm[i]);
    }
}

///@brief write the priceMat data to DDR
template <typename DT, int UN, int Size>
void write_ddr_price(int depth, int offset, hls::stream<DT> in_strm[UN], ap_uint<UN * Size>* Out) {
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
    std::cout << "Write price " << depth * UN << std::endl;
#endif
#endif
    for (int i = 0; i < depth; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS loop_tripcount min = 10240 max = 10240
        ap_uint<UN * Size> out_0;
        for (int k = 0; k < UN; ++k) {
            DT in_0 = in_strm[k].read();
            out_0((k + 1) * Size - 1, k * Size) = doubleToBits(in_0);
        }
        Out[offset + i] = out_0;
    }
}

///@brief write matrix data to DDR,
// the matrix data are the output of PreSamples kernel.
template <typename DT, int Size>
void write_ddr_matrix(int depth, DT* Buffer, ap_uint<Size>* Out) {
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
    std::cout << "Mat out Number " << depth << std::endl;
#endif
#endif
    for (int i = 0; i < depth; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS loop_tripcount min = 90 max = 90
        ap_uint<Size> out_0;
        DT in_0 = Buffer[i];
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
        if (i < 10) std::cout << "Mat out " << in_0 << std::endl;
#endif
#endif
        out_0(Size - 1, 0) = doubleToBits(in_0);
        Out[i] = out_0;
    }
}

/// @brief convert the UN streams from MultiMonteCarloModel to
//   one stream output. aka, matrixIN[UN] ==> UN*outStrm
//  @param depth the size of matdata
template <typename DT, int UN>
void MergeMatrixUN(int depth, hls::stream<DT> matrixIn[UN], hls::stream<DT>& outStrm) {
    DT sum = 0;
    for (int i = 0; i < depth; ++i) {
#pragma HLS loop_tripcount min = 72 max = 72
        for (int m = 0; m < UN; ++m) {
#pragma HLS pipeline II = 10
            DT op1;
            if (m == 0) {
                op1 = 0;
            } else {
                op1 = sum;
            }
            DT in = matrixIn[m].read();
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
            if (i == 0) std::cout << "mat out=" << in << std::endl;
#endif
#endif
            sum = FPTwoAdd(op1, in);
            if (m == UN - 1) {
                outStrm.write(sum);
            }
        }
    }
}

///@brief save iterations of matdata to Buffer
//  for each k, one Buffer is stored.
//
template <typename DT>
void MergeMatrixIter(int depth, hls::stream<DT>& matrixIn, DT* Buffer, int k) {
    for (int i = 0; i < depth; ++i) {
#pragma HLS loop_tripcount min = 72 max = 72
#pragma HLS pipeline II = 1
        DT op1;
        if (k == 0) {
            op1 = 0;
        } else {
            op1 = Buffer[i];
        }
        DT in = matrixIn.read();
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
        if (i < 10) std::cout << "mat iter =" << in << std::endl;
#endif
#endif
        Buffer[i] = FPTwoAdd(op1, in);
    }
}
///@brief Monte Carlo Process
template <typename DT,
          typename RNG,
          int UN,
          typename PathGeneratorT,
          typename PathPricerT,
          typename RNGSeqT,
          int VariateNum,
          int SampNum,
          int COEFNM,
          int SZ>
void MCProcess(ap_uint<16> steps,
               DT underlying,
               DT strike,
               DT invStk,
               int mat_nm,
               RNG rngInst[UN][VariateNum],
               PathGeneratorT pathGenInst[UN][1],
               PathPricerT pathPriInst[UN][1],
               RNGSeqT rngSeqInst[UN][1],
               DT* Buffer,
               int k,
               int offset,
               ap_uint<UN * 8 * sizeof(DT)>* pOut) {
#pragma HLS inline off
#pragma HLS dataflow
    hls::stream<DT> priceStrm[UN];
#pragma HLS STREAM variable = priceStrm depth = 64
#pragma HLS array_partition variable = priceStrm dim = 0
    hls::stream<DT> matrixStrm[UN];
#pragma HLS STREAM variable = matrixStrm depth = 18
#pragma HLS array_partition variable = matrixStrm dim = 0
    hls::stream<DT> xStrm;
#pragma HLS STREAM variable = &xStrm depth = 18
    MultiMonteCarloModel<DT, RNG, UN, PathGeneratorT, PathPricerT, RNGSeqT, VariateNum, SampNum, COEFNM>(
        steps, underlying, strike, invStk, rngInst, pathGenInst, pathPriInst, rngSeqInst, priceStrm, matrixStrm);
    MergeMatrixUN<DT, UN>(mat_nm, matrixStrm, xStrm);
    write_ddr_price<DT, UN, SZ>(steps * SampNum, offset, priceStrm, pOut);
    MergeMatrixIter<DT>(mat_nm, xStrm, Buffer, k);
}

/// @brief Run multiple times of Monte Carlo Process,
// running loop = iter, which eqauls to UN
template <typename DT,
          typename RNG,
          int UN,
          typename PathGeneratorT,
          typename PathPricerT,
          typename RNGSeqT,
          int VariateNum,
          int SampNum,
          int COEFNM,
          int ITER,
          int SZ>
void MCIteration(ap_uint<16> steps,
                 DT underlying,
                 DT strike,
                 DT invStk,
                 int mat_nm,
                 int iter,
                 RNG rngInst[UN][VariateNum],
                 PathGeneratorT pathGenInst[UN][1],
                 PathPricerT pathPriInst[UN][1],
                 RNGSeqT rngSeqInst[UN][1],
                 ap_uint<UN * 8 * sizeof(DT)>* pOut,
                 ap_uint<8 * sizeof(DT)>* mOut,
                 hls::stream<int>& phase_end) {
#pragma HLS inline off
    // set the upper bound (maximum) of timesteps
    const static int depthMax = 4096;
    DT Buffer[depthMax];
    for (int k = 0; k < iter; ++k) {
#pragma HLS loop_tripcount min = 5 max = 5
        MCProcess<DT, RNG, UN, PathGeneratorT, PathPricerT, RNGSeqT, VariateNum, SampNum, COEFNM, SZ>(
            steps, underlying, strike, invStk, mat_nm, rngInst, pathGenInst, pathPriInst, rngSeqInst, Buffer, k,
            k * steps * SampNum, pOut);
    }
    write_ddr_matrix<DT, SZ>(mat_nm, Buffer, mOut);
    phase_end.write(1);
}

/// @brief Read samples for external memory
template <typename DT, int COEFNM, int MAXPATH>
void GenAty(hls::stream<DT>& pStrm,
            ap_uint<16> paths,
            DT dF,
            DT y[MAXPATH],
            DT pBuff[MAXPATH],
            DT coef[COEFNM],
            DT outAty[COEFNM],
            bool optionType,
            DT strike,
            DT invStk) {
    const int buff_num = 16;
    // ap_uint<4> cnt = 0;
    DT AtyBuff[COEFNM][buff_num];
#pragma HLS array_partition variable = AtyBuff dim = 1
GENAty_Loop:
    for (int j = 0; j < paths; ++j) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1024 max = 1024
        int cnt = j % 16;
        DT in = pStrm.read();
        // current steps
        DT tempIn = FPTwoMul(in, invStk);
        DT subOp1 = 0;
        DT subOp2 = 0;
        if (optionType) {
            subOp1 = strike;
            subOp2 = in;
        } else {
            subOp1 = in;
            subOp2 = strike;
        }
        DT payoff = FPTwoSub(subOp1, subOp2);
        DT exercise = MAX(payoff, 0);

        DT squareIn = FPTwoMul(tempIn, tempIn);

        DT preY = y[j];
        DT newY;

        DT pPrice = pBuff[j];
        DT op1 = 0;
        DT op2 = 0;
        if (optionType) {
            op1 = strike;
            op2 = pPrice;
        } else {
            op1 = pPrice;
            op2 = strike;
        }
        DT pPayoff = FPTwoSub(op1, op2);
        DT pEx = MAX(pPayoff, 0);
        DT pP = FPTwoMul(pPrice, invStk);
        DT sqPP = FPTwoMul(pP, pP);
        DT m1 = FPTwoMul(coef[1], pP);
        DT m2 = FPTwoMul(coef[2], sqPP);
        DT m3 = FPTwoMul(coef[3], pEx);
        DT a1 = FPTwoAdd(coef[0], m1);
        DT a2 = FPTwoAdd(m2, m3);
        DT conVal = FPTwoAdd(a1, a2);
        // DT conVal = coef[0] + coef[1]*pP + coef[2]*sqPP + coef[3]*pEx;
        if (pEx > 0 && conVal < pEx)
            newY = pEx;
        else
            newY = preY;
        DT newY_1 = FPTwoMul(newY, dF);
        DT oldV[COEFNM];
        for (int k = 0; k < COEFNM; ++k) {
#pragma HLS unroll
            if (j < buff_num)
                oldV[k] = 0;
            else
                oldV[k] = AtyBuff[k][cnt];
        }
        DT addV[COEFNM];
        if (exercise > 0) {
            addV[0] = newY_1;
            addV[1] = FPTwoMul(newY_1, tempIn);
            addV[2] = FPTwoMul(newY_1, squareIn);
            addV[3] = FPTwoMul(newY_1, exercise);
        } else {
            for (int k = 0; k < COEFNM; ++k) {
#pragma HLS unroll
                addV[k] = 0;
            }
        }
        DT newV[COEFNM];
        for (int k = 0; k < COEFNM; ++k) {
#pragma HLS unroll
            newV[k] = FPTwoAdd(oldV[k], addV[k]);
        }
        for (int k = 0; k < COEFNM; ++k) {
#pragma HLS unroll
            AtyBuff[k][cnt] = newV[k];
        }
        y[j] = newY_1;
        pBuff[j] = in;
        // cnt++;
    }

Merge16To8_Loop:
    for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline II = 1
        for (int k = 0; k < COEFNM; ++k) {
            AtyBuff[k][i] = FPTwoAdd(AtyBuff[k][i], AtyBuff[k][i + 8]);
        }
    }
Merge8To4_Loop:
    for (int i = 0; i < 4; ++i) {
#pragma HLS pipeline II = 1
        for (int k = 0; k < COEFNM; ++k) {
            AtyBuff[k][i] = FPTwoAdd(AtyBuff[k][i], AtyBuff[k][i + 4]);
        }
    }
Merge4To1_Loop:
    for (int k = 0; k < COEFNM; ++k) {
#pragma HLS pipeline II = 1
        DT sum1 = FPTwoAdd(AtyBuff[k][0], AtyBuff[k][1]);
        DT sum2 = FPTwoAdd(AtyBuff[k][2], AtyBuff[k][3]);
        outAty[k] = FPTwoAdd(sum1, sum2);
    }
}

template <typename DT, int COEFNM, int SampNum, int UN>
void MultGenAty(hls::stream<DT> pStrm[UN],
                ap_uint<16> paths,
                DT dF,
                DT y[UN][SampNum],
                DT pBuff[UN][SampNum],
                DT coef[UN][COEFNM],
                DT outAty[COEFNM],
                bool optionType,
                DT strike,
                DT invStk) {
    DT tmpAty[UN][COEFNM];
#pragma HLS array_partition variable = tmpAty dim = 0
    for (int j = 0; j < COEFNM; ++j) {
#pragma HLS pipeline II = 1
        for (int k = 0; k < UN; ++k) {
            tmpAty[k][j] = 0;
            outAty[j] = 0;
        }
    }
    for (int k = 0; k < UN; ++k) {
#pragma HLS unroll
        GenAty<DT, COEFNM, SampNum>(pStrm[k], paths, dF, y[k], pBuff[k], coef[k], tmpAty[k], optionType, strike,
                                    invStk);
    }
Merge_AtY_LOOP:
    for (int k = 0; k < UN; ++k) {
        for (int j = 0; j < COEFNM; ++j) {
#pragma HLS pipeline II = 10
            outAty[j] = FPTwoAdd(outAty[j], tmpAty[k][j]);
        }
    }
}

/// SVD process
template <typename DT, int COEFNM>
void SVDDec(ap_uint<16> steps,
            hls::stream<DT>& inStrm,
            hls::stream<DT>& Ustrm,
            hls::stream<DT>& Vstrm,
            hls::stream<DT>& Sstrm) {
    const static int NM = 3 * (COEFNM - 1);
    const static double threshold = 1e-10;
Loop_SVDDec:
    for (int i = 0; i < steps; ++i) {
#pragma HLS loop_tripcount min = 8 max = 8
        DT A[COEFNM][COEFNM];
#pragma HLS array_partition variable = A complete dim = 0
        DT U[COEFNM][COEFNM];
#pragma HLS array_partition variable = U complete dim = 0
        DT V[COEFNM][COEFNM];
#pragma HLS array_partition variable = V complete dim = 0
        DT S[COEFNM][COEFNM];
#pragma HLS array_partition variable = S complete dim = 0
        DT AtA[NM];
#pragma HLS array_partition variable = AtA complete dim = 0
        for (int j = 0; j < NM; ++j) {
#pragma HLS pipeline II = 1
            AtA[j] = inStrm.read();
        }
        // Construct A
        for (int m = 0; m < COEFNM - 1; m++) {
            for (int n = 0; n < COEFNM - 1; n++) {
#pragma HLS unroll
                A[m][n] = AtA[m + n];
            }
        }
        for (int m = 0; m < COEFNM; m++) {
#pragma HLS unroll
            A[m][COEFNM - 1] = AtA[2 * (COEFNM - 1) - 1 + m];
            A[COEFNM - 1][m] = AtA[2 * (COEFNM - 1) - 1 + m];
        }
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUE
        std::cout << "A matrix" << std::endl;
        for (int m = 0; m < COEFNM; ++m) {
            for (int n = 0; n < COEFNM; ++n) {
                // std::cout<<std::hex<<doubleToBits(A[m][n])<<", ";
                std::cout << A[m][n] << ", ";
            }
            std::cout << std::endl;
        }
#endif
#endif
        // SVD decomposition
        xf::fintech::svd<DT, COEFNM>(A, S, U, V);
///////////////////////////////////
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
        std::cout << "S matrix" << std::endl;
        for (int m = 0; m < COEFNM; ++m) {
            for (int n = 0; n < COEFNM; ++n) {
                // std::cout<<std::hex<<doubleToBits(S[m][n])<<", ";
                std::cout << S[m][n] << ", ";
            }
            std::cout << std::endl;
            // coefStrmOut[m].write(coef[m]);
        }
        std::cout << "U matrix" << std::endl;
        for (int m = 0; m < COEFNM; ++m) {
            for (int n = 0; n < COEFNM; ++n) {
                // std::cout<<std::hex<<doubleToBits(U[m][n])<<", ";
                std::cout << U[m][n] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << "V matrix" << std::endl;
        for (int m = 0; m < COEFNM; ++m) {
            for (int n = 0; n < COEFNM; ++n) {
                // std::cout<<std::hex<<doubleToBits(V[m][n])<<", ";
                std::cout << V[m][n] << ", ";
            }
            std::cout << std::endl;
        }
#endif
#endif
        for (int k = 0; k < COEFNM; ++k) {
            for (int t = 0; t < COEFNM; ++t) {
#pragma HLS loop_tripcount min = 16 max = 16
#pragma HLS pipeline II = 1
                Ustrm.write(U[k][t]);
                Vstrm.write(V[k][t]);
                // move divide from calccoef to here.
                DT tmp = 0;
                DT s = S[k][t];
                if (k == t && s > threshold)
                    tmp = 1.0 / s;
                else
                    tmp = 0;
                // Sstrm.write(S[i][j]);
                Sstrm.write(tmp);
            }
        }
    }
}
///@brief execute SVD in parallel
// @tparam COEFNM is the number of coefficient, default = 4
// @tparam UN is the unroll number, it should be equal to UN_STEP
template <typename DT, int COEFNM, int UN>
void MultiSVD(ap_uint<16> steps,
              hls::stream<DT> inStrm[UN],
              hls::stream<DT> Ustrm[UN],
              hls::stream<DT> Vstrm[UN],
              hls::stream<DT> Sstrm[UN]) {
    for (int i = 0; i < UN; ++i) {
#pragma HLS loop_tripcount min = UN max = UN
#pragma HLS unroll
        SVDDec<DT, COEFNM>(steps, inStrm[i], Ustrm[i], Vstrm[i], Sstrm[i]);
    }
}

///@brief read in the price mat data from DDR to stream
// Because the matrix data in DDR is stored with width double*UN,
// the output stream is UN * double.
template <typename DT, int UN, int samplesNm>
void readin_ddr(const int loopNm,
                const int steps,
                ap_uint<8 * sizeof(DT) * UN>* in_data,
                hls::stream<ap_uint<8 * sizeof(DT) * UN> >& outStrm) {
    ap_uint<16> batch = steps * samplesNm;
    for (int i = steps - 1; i >= 0; --i) {
#pragma HLS loop_tripcount min = 8 max = 8
        for (int j = 0; j < loopNm; ++j) {
#pragma HLS loop_tripcount min = 2 max = 2
            for (int k = 0; k < samplesNm; ++k) {
#pragma HLS loop_tripcount min = samplesNm max = samplesNm
#pragma HLS pipeline II = 1
                ap_uint<8 * sizeof(DT)* UN> in = in_data[batch * j + i * samplesNm + k];
                outStrm.write(in);
            }
        }
    }
}
//@brief convert 1 streamed data from width UN*double to
// UN streamed data with width double
template <typename DT, int UN>
void read_merge(int loopNm, hls::stream<ap_uint<8 * sizeof(DT) * UN> >& inStrm, hls::stream<DT> outStrm[UN]) {
    const int SZ = 8 * sizeof(DT);
    for (int i = 0; i < loopNm; ++i) {
#pragma HLS loop_tripcount min = 1024 max = 1024
#pragma HLS pipeline II = 1
        ap_uint<SZ* UN> in = inStrm.read();
        for (int k = 0; k < UN; ++k) {
            int64_t i_i = in((k + 1) * SZ - 1, k * SZ);
            DT i_d = bitsToDouble(i_i);
            outStrm[k].write(i_d);
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
            if (i < 20) {
                std::cout << "pricedata is " << i_d << std::endl;
            }
#endif
#endif
        }
    }
}
///@brief read in matrix B data from DDR
template <typename DT, int UN, int ORDER>
void read_AtA(int steps, ap_uint<8 * sizeof(DT)>* in_data, hls::stream<DT>& dout_strm) {
    const static unsigned int n = 3 * (ORDER - 1);
    for (int i = steps - 1; i >= 0; --i) {
#pragma HLS loop_tripcount min = 8 max = 8
        for (int k = 0; k < n; ++k) {
#pragma HLS loop_tripcount min = 9 max = 9
#pragma HLS PIPELINE II = 1
            DT in = bitsToDouble(in_data[i * n + k]);
            dout_strm.write(in);
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
            if (i < 20) {
                std::cout << "matdata is " << in << std::endl;
            }
#endif
#endif
        }
    }
}
///@brief split the matdata by UN_STEP, prepare the data for svd
//  when UN_STEP >1, execute SVD in parallel
template <typename DT, int COEFNM, int UN>
void SplitStrm(ap_uint<16> steps, hls::stream<DT>& inStrm, hls::stream<DT> outStrm[UN]) {
    for (int i = 0; i < steps; ++i) {
#pragma HLS loop_tripcount min = 8 max = 8
        for (int k = 0; k < UN; ++k) {
#pragma HLS loop_tripcount min = UN max = UN
            for (int j = 0; j < 3 * (COEFNM - 1); ++j) {
#pragma HLS loop_tripcount min = 9 max = 9
#pragma HLS pipeline II = 1
                DT in = inStrm.read();
                outStrm[k].write(in);
            }
        }
    }
}
///@brief convert svd result from UN(UN_STEP) streams to 1 stream
// UN(UN_STEP) is the parallel unroll number while doing SVD
// @tparam COEFNM is the number of coefficient, default = 4
// @tparam UN is the unroll number, it should be equal to UN_STEP
template <typename DT, int COEFNM, int UN>
void MergeStrm(ap_uint<16> steps,
               hls::stream<DT> mUstrm[UN],
               hls::stream<DT> mVstrm[UN],
               hls::stream<DT> mSstrm[UN],
               hls::stream<DT>& Ustrm,
               hls::stream<DT>& Vstrm,
               hls::stream<DT>& Sstrm) {
    for (int i = 0; i < steps; ++i) {
#pragma HLS loop_tripcount min = 4 max = 4
        for (int k = 0; k < UN; ++k) {
#pragma HLS loop_tripcount min = UN max = UN
            for (int t = 0; t < COEFNM * COEFNM; ++t) {
#pragma HLS loop_tripcount min = 16 max = 16
#pragma HLS pipeline II = 1
                DT u = mUstrm[k].read();
                DT v = mVstrm[k].read();
                DT s = mSstrm[k].read();
                Ustrm.write(u);
                Vstrm.write(v);
                Sstrm.write(s);
            }
        }
    }
}
/// @brief calculate the coefficients and output as streams
template <typename DT, int COEFNM, int SamplesNm, int UN>
void CalCoef(ap_uint<16> steps,
             ap_uint<16> paths,
             bool optionType,
             DT dF,
             DT strike,
             DT invStk,
             hls::stream<DT>& Ustrm,
             hls::stream<DT>& Vstrm,
             hls::stream<DT>& Sstrm,
             hls::stream<DT> pStrm[UN],
             hls::stream<DT> coefStrm[COEFNM]) {
    DT y[UN][SamplesNm];
#pragma HLS array_partition variable = y dim = 1
    DT Aty[COEFNM];
#pragma HLS array_partition variable = Aty dim = 0
    DT pBuff[UN][SamplesNm];
#pragma HLS array_partition variable = pBuff dim = 1
    DT coef[UN][COEFNM];
#pragma HLS array_partition variable = coef dim = 0
    //    std::cout<<"path = "<<paths<<std::endl;
    for (int j = 0; j < paths; ++j) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1024 max = 1024
        for (int k = 0; k < UN; ++k) {
            y[k][j] = 0;
            pBuff[k][j] = pStrm[k].read();
        }
    }
    for (int i = 0; i < COEFNM; ++i) {
#pragma HLS unroll
        coef[0][i] = 0;
        Aty[i] = 0;
    }
    // Delete the last steps
    for (int i = 0; i < COEFNM; ++i) {
        for (int j = 0; j < COEFNM; ++j) {
#pragma HLS pipeline II = 1
            DT u = Ustrm.read();
            DT v = Vstrm.read();
            DT s = Sstrm.read();
        }
    }
BACKTRACE_LOOP:
    for (int i = steps - 2; i >= 0; --i) {
#pragma HLS loop_tripcount min = 7 max = 7
        MultGenAty<DT, COEFNM, SamplesNm, UN>(pStrm, paths, dF, y, pBuff, coef, Aty, optionType, strike, invStk);
        CalcLinear<DT, COEFNM>(Ustrm, Vstrm, Sstrm, Aty, coef[0]);
        for (int j = 0; j < COEFNM; ++j) {
#pragma HLS loop_tripcount min = 4 max = 4
#pragma HLS pipeline
            for (int k = 1; k < UN; ++k) {
#pragma HLS loop_tripcount min = UN max = UN
                coef[k][j] = coef[0][j];
            }
            coefStrm[j].write(coef[0][j]);
        }
    }
}
///@brief write the coeff data to DDR, the data width is COEFNM*double
template <typename DT, int UN, int Size>
void write_ddr(int depth, hls::stream<DT> in_strm[UN], ap_uint<UN * Size>* Out) {
    for (int i = 0; i < depth; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS loop_tripcount min = 10 max = 10
        ap_uint<UN * Size> out_0;
        for (int k = 0; k < UN; ++k) {
#pragma HLS loop_tripcount min = UN max = UN
            DT in_0 = in_strm[k].read();
            out_0((k + 1) * Size - 1, k * Size) = doubleToBits(in_0);
        }
        Out[i] = out_0;
    }
}

///@brief Read the coefficients from DDR,
//  then assign the coeff value to LPPricer
template <typename DT, int UN, int Size, int COEFNM, bool SF, int SN, int MaxSteps, bool Antithetic>
void read_coef(hls::stream<int>& phase_start,
               int depth,
               ap_uint<COEFNM * Size>* In,
               PathPricer<LongstaffSchwartz, DT, SF, SN, Antithetic, MaxSteps> pathPriInst[UN][1]) {
    int phase2_start = phase_start.read();
    if (phase2_start == 1) {
        for (int i = 0; i < depth; ++i) {
#pragma HLS loop_tripcount min = 7 max = 7
#pragma HLS PIPELINE II = 1
            ap_uint<COEFNM* Size> out_0 = In[i];
            DT tmp[4];
            for (int k = 0; k < COEFNM; ++k) {
#pragma HLS unroll
                tmp[k] = bitsToDouble(out_0((k + 1) * Size - 1, k * Size));
            }
            for (int j = 0; j < UN; ++j) {
#pragma HLS unroll
                pathPriInst[j][0].coefBuff_0[i] = tmp[0];
                pathPriInst[j][0].coefBuff_1[i] = tmp[1];
                pathPriInst[j][0].coefBuff_2[i] = tmp[2];
                pathPriInst[j][0].coefBuff_3[i] = tmp[3];
            }
        }
    }
}
template <typename DT = double, int UN = 2, int UN_STEP = 2>
void MCAmericanEngineCalibrate_dataflow(DT dt,
                                        DT riskFreeRate,
                                        DT strike,
                                        DT invStk,
                                        DT discount,
                                        bool optionType,
                                        ap_uint<8 * sizeof(DT) * UN>* priceIn,
                                        ap_uint<8 * sizeof(DT)>* matIn,
                                        ap_uint<8 * sizeof(DT) * 4>* coefOut,
                                        unsigned int calibSamples = 4096,
                                        unsigned int timeSteps = 100) {
#pragma HLS inline off

    // number of samples per simulation
    const static int SN = 1024;

    // number of variate
    const static int VN = 1;

    // order of polynomial for LongStaffShwartz
    const static int COEFNM = 4;

    // Max sampels for Caliration
    const static int CalSample = 4096;

    // intermediate streams used to buffer data between dataflow functions
    hls::stream<ap_uint<8 * sizeof(double) * UN> > pStrm;
#pragma HLS stream variable = &pStrm depth = 8
    hls::stream<DT> inStrm[UN];
#pragma HLS stream variable = inStrm depth = 8
#pragma HLS array_partition variable = inStrm dim = 0
    hls::stream<DT> xStrm;
#pragma HLS stream variable = &xStrm depth = 9
    hls::stream<DT> xStrm_un[UN_STEP];
#pragma HLS stream variable = xStrm_un depth = 9
#pragma HLS array_partition variable = xStrm_un dim = 0
    hls::stream<DT> mUstrm[UN_STEP];
#pragma HLS stream variable = mUstrm depth = 16
#pragma HLS array_partition variable = mUstrm dim = 0
    hls::stream<DT> mVstrm[UN_STEP];
#pragma HLS stream variable = mVstrm depth = 16
#pragma HLS array_partition variable = mVstrm dim = 0
    hls::stream<DT> mSstrm[UN_STEP];
#pragma HLS stream variable = mSstrm depth = 16
#pragma HLS array_partition variable = mSstrm dim = 0
    hls::stream<DT> Ustrm;
#pragma HLS stream variable = &Ustrm depth = 16
    hls::stream<DT> Vstrm;
#pragma HLS stream variable = &Vstrm depth = 16
    hls::stream<DT> Sstrm;
#pragma HLS stream variable = &Sstrm depth = 16
    hls::stream<DT> coefStrm[COEFNM];
#pragma HLS stream variable = coefStrm depth = 16
#pragma HLS array_partition variable = coefStrm dim = 0
#pragma HLS dataflow

    // read price mat data from DDR
    readin_ddr<DT, UN, SN>(calibSamples / UN / SN, timeSteps, priceIn, pStrm);

    // Because the data are stored in differnet locations of DDR, prepare the
    // complete data for calib process
    read_merge<DT, UN>(calibSamples * timeSteps / UN, pStrm, inStrm);

    // read m mat data from DDR
    read_AtA<DT, UN_STEP, COEFNM>(timeSteps, matIn, xStrm);

    SplitStrm<DT, COEFNM, UN_STEP>(timeSteps / UN_STEP, xStrm, xStrm_un);

    // calc SVD
    MultiSVD<DT, COEFNM, UN_STEP>(timeSteps / UN_STEP, xStrm_un, mUstrm, mVstrm, mSstrm);

    MergeStrm<DT, COEFNM, UN_STEP>(timeSteps / UN_STEP, mUstrm, mVstrm, mSstrm, Ustrm, Vstrm, Sstrm);

    // calculate the coeff // TODO iteration should be as a para in CalCoef
    // //iteration = calibratesamples / unroll / 1024
    CalCoef<DT, COEFNM, CalSample, UN>(timeSteps, calibSamples / UN, optionType, discount, strike, invStk, Ustrm, Vstrm,
                                       Sstrm, inStrm, coefStrm);

    // write the coeff data to DDR, the data width is COEFNM* double
    write_ddr<DT, COEFNM, 8 * sizeof(DT)>(timeSteps - 1, coefStrm, coefOut);
}

/**
 * @brief American Option Pricing Engine using Monte Carlo Method.
 * Calibrate kernel: this kernel reads the sample price data from external
 * memory and use them to calculate the coefficient
 *
 * @tparam DT supported data type including double and float data type, which
 * decides the precision of result, default double-precision data type.
 * @tparam UN number of Monte Carlo Module in parallel (in path dimension),
 * which affects the latency and resources utilization, default 2. [this unroll
 * number should be equal to UN in MCAmericanEnginePresample]
 * @tparam UN_STEP number of Monte Carlo Module in parallel (in time steps
 * dimension), which affects the latency and resources utilization, default 2.
 * [this Unroll is completely resource bounded, unrelated to other params]
 * @param phase_start phase start
 * @param phase_end phase end
 * @param timeLength the time length of contract from start to end.
 * @param riskFreeRate risk-free interest rate.
 * @param strike the strike price also known as exericse price, which is settled
 * in the contract.
 * @param optionType option type. 1: call option, 0: put option.
 * @param priceIn the price data, read in from DDR or HBM
 * @param matIn the matrix data, read in from DDR or HBM
 * @param coefOut the coef data that calculated by this kernel, the data can be
 * stored to DDR or HBM
 * @param calibSamples sample numbers that used in calibration, default 4096.
 * @param timeSteps the number of discrete steps from 0 to T, T is the expiry
 * time, default 100.
 */
template <typename DT = double, int UN = 2, int UN_STEP = 2>
void MCAmericanEngineCalibrateCalc(hls::stream<int>& phase_start,
                                   hls::stream<int>& phase_end,
                                   DT timeLength,
                                   DT riskFreeRate,
                                   DT strike,
                                   bool optionType,
                                   ap_uint<8 * sizeof(DT) * UN>* priceIn,
                                   ap_uint<8 * sizeof(DT)>* matIn,
                                   ap_uint<8 * sizeof(DT) * 4>* coefOut,
                                   unsigned int calibSamples = 4096,
                                   unsigned int timeSteps = 100) {
    int module_start = phase_start.read();

    if (module_start == 1) {
        // pre-process of "cold" logic
        DT dt = timeLength / timeSteps;
        DT invStk = 1 / strike;
        DT discount = FPExp(-1.0 * riskFreeRate * dt);

        MCAmericanEngineCalibrate_dataflow<DT, UN, UN_STEP>(dt, riskFreeRate, strike, invStk, discount, optionType,
                                                            priceIn, matIn, coefOut, calibSamples, timeSteps);
        phase_end.write(1);
    }
}

} // namespace internal
} // namespace fintech
} // namespace xf

#endif // ifndef XF_FINTECH_EARLYEX_H
