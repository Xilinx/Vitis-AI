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
 * @file polyfit.hpp
 * @brief Polynomial fitting and evaluating templated funtions implementation.
 *
 */

#ifndef _POLYFIT_H_
#define _POLYFIT_H_

#include "xf_solver_L2.hpp"

namespace xf {
namespace fintech {

namespace {

template <typename DT, unsigned int D, unsigned int MAX_WIDTH>
struct polyfitHelperBase {
    /**
     * Solves the least squares fitting linear equation with the polySolver for matrix M
     * Mx = evalPoints
     */
    static void polyfitSolve(const DT solverMatrix[MAX_WIDTH * D],
                             const DT evalPoints[MAX_WIDTH],
                             const unsigned int sizeEval,
                             DT coefficients[D]) {
        // Matrix vector product of polysolver with the points in order to get the polyfit, output in reverse order
        for (unsigned i = 0; i < D; i++) {
#pragma HLS UNROLL
            DT sum = 0;
            for (unsigned j = 0; j < sizeEval; j++) {
                sum += solverMatrix[i * sizeEval + j] * evalPoints[j];
            }
            coefficients[(D - 1) - i] = sum;
        }
    }
};

template <typename DT, unsigned int D, unsigned int MAX_WIDTH>
struct polyfitHelper {
    static void mmult(DT* a, DT* b, DT* s, unsigned rowsa, unsigned rowsb, unsigned colsa, unsigned colsb) {
    mmult_row_loop:
        for (unsigned i = 0; i < rowsa; i++) {
        mmult_col_loop:
            for (unsigned j = 0; j < colsb; j++) {
                DT sum = 0.0;
            mmult_accum_loop:
                for (unsigned k = 0; k < rowsb; k++) {
#pragma HLS PIPELINE II = 1
                    sum += a[i * colsa + k] * b[k * colsb + j];
                }
                s[i * colsb + j] = sum;
            }
        }
    }

    /*
     * Computes the least squares fitting solver matrix:
     * M = (vand(X)^T * vand(X))^-1 * vand(X)^T
     */
    static void computePolySolver(const DT evalX[MAX_WIDTH], unsigned int sizeEval, DT polySolver[D * MAX_WIDTH]) {
        // Create vandermonde matrix and vand_transpose
        DT vand[MAX_WIDTH * D], vandt[2][MAX_WIDTH * D];
#pragma HLS ARRAY_PARTITION variable = vandt dim = 1 complete
    Polyfit_vandermonde_loop:
        for (unsigned i = 0; i < sizeEval; i++) {
            DT v = 1.0;
            for (unsigned j = 0; j < D; j++) {
                vand[i * D + j] = v;
                vandt[0][j * sizeEval + i] = v;
                vandt[1][j * sizeEval + i] = v;
                v *= evalX[i];
            }
        }

        // (A.T * A)^-1
        DT vinv_res[D][D], vand_inv[D * D], vandnorm[D * D], invCore[1][D][D];
#pragma HLS ARRAY_PARTITION variable = invCore dim = 1 complete
        mmult(vandt[0], vand, vandnorm, D, sizeEval, sizeEval, D);
        for (unsigned i = 0; i < D; i++) {
            for (unsigned j = 0; j < D; j++) {
                invCore[0][i][j] = vandnorm[i * D + j];
            }
        }
        xf::solver::internal_pomi::inverse_core<DT, D, 1>(D, invCore, vinv_res);
        for (unsigned i = 0; i < D; i++) {
            for (unsigned j = 0; j < D; j++) {
                vand_inv[i * D + j] = vinv_res[i][j];
            }
        }

        mmult(vand_inv, vandt[1], polySolver, D, D, D, sizeEval);
    }
};

template <typename DT, unsigned int MAX_WIDTH>
struct polyfitHelper<DT, 1, MAX_WIDTH> {
    static void computePolySolver(const DT evalX[MAX_WIDTH], unsigned int sizeEval, DT polySolver[MAX_WIDTH]) {
        /*
         * With constant polynomial fitting, all poly solver matrix calculations simplify into a single vector
         * [1/sizeEval, 1/sizeEval, ...]
         */
        for (unsigned i = 0; i < sizeEval; i++) {
#pragma HLS PIPELINE
            polySolver[i] = 1.0 / sizeEval;
        }
    }
};

} // namespace

/**
 * @brief Calculates the polynomial fitting to the D degree of the discrete set of points in 'evalPoints'.
 *
 * @tparam DT: The data type of the points
 * @tparam D: The degree of the polynomial that will approximate the set of points.
 * @tparam MAX_WIDTH: The maximum synthetisable amount of discrete points.
 *
 * @param evalX: values on the X axis of the set of points.
 * @param evalPoints: Set of points to be approximated.
 * @param sizeEval: Length of 'evalPoints', must be <= MAX_WIDTH
 * @param coefficients: Output polynomial coefficients that approximate 'evalPoints' to the D degree, in decreasing
 * order of degree.
 */
template <typename DT, unsigned int D, unsigned int MAX_WIDTH>
void polyfit(const DT evalX[MAX_WIDTH], const DT evalPoints[MAX_WIDTH], unsigned int sizeEval, DT coefficients[D]) {
    // TODO: in the case of sequential X axis, poly_solver_matrix could be precomputed
    DT computedPolySolver[D * MAX_WIDTH];
    polyfitHelper<DT, D, MAX_WIDTH>::computePolySolver(evalX, sizeEval, computedPolySolver);
    polyfitHelperBase<DT, D, MAX_WIDTH>::polyfitSolve(computedPolySolver, evalPoints, sizeEval, coefficients);
}

/**
 * @brief Calculates the polynomial fitting to the D degree of the discrete set of points in 'evalPoints',
 * assuming values on the X axis = [0, 1, 2, 3, ..., MAX_WIDTH]
 *
 * @tparam DT: The data type of the points
 * @tparam D: The degree of the polynomial that will approximate the set of points.
 * @tparam MAX_WIDTH: The maximum synthetisable amount of discrete points.
 *
 * @param evalPoints: Set of points to be approximated.
 * @param sizeEval: Length of 'evalPoints', must be <= MAX_WIDTH
 * @param coefficients: Output polynomial coefficients that approximate 'evalPoints' to the D degree, in decreasing
 * order of degree.
 */
template <typename DT, unsigned int D, unsigned int MAX_WIDTH>
void polyfit(const DT evalPoints[MAX_WIDTH], const unsigned int sizeEval, DT coefficients[D]) {
    // Sequential X axis
    DT axis[MAX_WIDTH];
    for (unsigned i = 0; i < MAX_WIDTH; i++) {
        axis[i] = i;
    }
    polyfit<DT, D, MAX_WIDTH>(axis, evalPoints, sizeEval, coefficients);
}

/**
 * @brief Calculates the polynomial evaluation of a set of coefficients at the point 'x'
 *
 * @tparam DT: The data type to be used.
 * @tparam D: The degree of the polynomial
 *
 * @param coeff: The list of polynomial coefficients calculated with polyfit.
 * @param x: The point at which the polynomial should be evaluated.
 */
template <typename DT, unsigned int D>
DT polyval(const DT coeff[D], const DT x) {
#pragma HLS INLINE
    DT res = coeff[D - 1];
    DT xpow = x;
PolyVal_loop:
    for (unsigned i = 1; i < D; i++) {
#pragma HLS PIPELINE II = 1
        res += coeff[(D - 1) - i] * xpow;
        xpow *= x;
    }
    return res;
}

/**
 * @brief Performs the definite integral of a polynomial fitted function defined by its coefficients.
 *
 * @tparam DT: The data type to be used.
 * @tparam D: The degree of the fitted original polynomial.
 *
 * @param pf: The original polyfitted coefficients vector to be integrated.
 * @param pfInt: Output vector containing the definite integral of the original polynomial. Note that it's length must
 * be 1 more than the original polynomial to hold the extra degree.
 * @param c: Constant term of the new integral, defaults to 0.
 */
template <typename DT, unsigned int D>
void polyint(DT pf[D], DT pfInt[D + 1], DT c = 0.0) {
#pragma HLS INLINE
    pfInt[D] = c;
PolyInt_loop:
    for (unsigned int i = 0; i < D; i++) {
#pragma HLS PIPELINE
        pfInt[i] = pf[i] / (D - i);
    }
}

/**
 * @brief Performs the first derivate of a polynomial fitted function defined by its coefficients.
 *
 * @tparam DT: The data type to be used.
 * @tparam D: The degree of the fitted original polynomial
 *
 * @param pf: The original polyfitted coefficients vector to be integrated.
 * @param pfDer: Output vector containing the derivate of the original polynomial. Note that it's length must be 1
 * fewer than the original polynomial.
 */
template <typename DT, unsigned int D>
void polyder(DT pf[D], DT pfDer[D - 1]) {
#pragma HLS INLINE
PolyDer_loop:
    for (unsigned int i = 1; i < D; i++) {
#pragma HLS UNROLL
        pfDer[i - 1] = i * pf[i];
    }
}

} // namespace fintech
} // namespace xf

#endif // _PCA_H_