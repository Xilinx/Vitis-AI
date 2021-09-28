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
* @file pentadiag_cr.hpp
* @brief Main file containing solver function
*
* This algorithm solves pentadiagonal systems of linear equations using  \n
* parallel cyclic reduction (also known asodd-even elimination) \n
* This implementation is based on paper : \n
* C. Levit  <EM>Parallel Solution of Pentadiagonal Systems Using Generalized
* Odd-Even Elimination</EM> - 1989 \n
*
*
*
* @bug Algorithm is very sensitive to any zeros in the diagonals.
* Please be aware that zeros in diagonals could lead to division by zero and
* algorithm fails.
* @bug Experiments have shown that the algorithm failes for number of steps
* greater than 8.
* In that case it is recommend limit numebr of steps to 8. The same idea is used
* in LOGN calculation in pentadiag_top.h file.
*/

#ifndef XF_FINTECH_PENTADIAG_CR_HPP_
#define XF_FINTECH_PENTADIAG_CR_HPP_

namespace xf {
namespace fintech {
namespace internal {
/**
* @brief Executes one step of odd-even elimination.
* For each row it calculates new diagonal element and right hand side element
based on the paper.
*
* Structure of input matrix: \n
*
* \rst
*
* .. math::
*	\begin{vmatrix}
*	a & d & e & 0 & 0\\
*	b & c & d & e & 0\\
*	a & b & c & d & e\\
*	0 & a & b & c & d\\
*	0 & 0 & a & b & c
*	\end{vmatrix}
* \endrst
*
*@tparam T data type used in whole function (double by default)
*@tparam P_SIZE Size of the operating matrix
*@param[in] c - Main diagonal \n
*@param[in] b - First lower \n
*@param[in] a - Second lower \n
*@param[in] d - First upper \n
*@param[in] e - Second upper \n
*@param[in] r - Right hand side vector of length n \n
*@param[in] k - Number of current calculating step. It is used to calculate
indexes of diagonals \n

*@param[out] c_out - Main diagonal output \n
*@param[out] b_out - First lower output \n
*@param[out] a_out - Second lower output \n
*@param[out] d_out - First upper output \n
*@param[out] e_out - Second upper output \n
*@param[out] r_out - Right hand side vector of length n \n

*/
template <typename T, unsigned int P_SIZE>
void pentadiag_step(T a[P_SIZE],
                    T b[P_SIZE],
                    T c[P_SIZE],
                    T d[P_SIZE],
                    T e[P_SIZE],
                    T r[P_SIZE],
                    T a_out[P_SIZE],
                    T b_out[P_SIZE],
                    T c_out[P_SIZE],
                    T d_out[P_SIZE],
                    T e_out[P_SIZE],
                    T r_out[P_SIZE],
                    int k) {
    // Additional memmory to allow full pipelining of main loop
    T a_ram2[P_SIZE];
    T a_ram3[P_SIZE];
    T b_ram2[P_SIZE];
    T b_ram3[P_SIZE];
    T c_ram2[P_SIZE];
    T c_ram3[P_SIZE];
    T d_ram2[P_SIZE];
    T d_ram3[P_SIZE];
    T e_ram2[P_SIZE];
    T e_ram3[P_SIZE];
    T r_ram2[P_SIZE];
    T r_ram3[P_SIZE];
    // Copy input memmory to allow full pipelining of main loop
    for (int i = 0; i < P_SIZE; i++) {
#pragma HLS pipeline
        a_ram2[i] = a[i];
        a_ram3[i] = a[i];
        b_ram2[i] = b[i];
        b_ram3[i] = b[i];
        c_ram2[i] = c[i];
        c_ram3[i] = c[i];
        d_ram2[i] = d[i];
        d_ram3[i] = d[i];
        e_ram2[i] = e[i];
        e_ram3[i] = e[i];
        r_ram2[i] = r[i];
        r_ram3[i] = r[i];
    }

    T a1, a_1, a_2, a2;
    T b_1, b1, b_2, b2;
    T c1, c_1, c_2, c2;
    T d_1, d_2, d1, d2;
    T e_1, e1, e_2, e2;
    T r_1, r1, r_2, r2;
    T d_temp;
    T x1, x2, x3, x4;

    int ex = 1 << k;
    int ex1 = 1 << (k + 1);
    int index, index1, index_neg, index_neg_1;
    /*
    * Main calculation loop. It Iterates for every row and applies calculation of
    * new diagonals.
    */
    for (int i = 0; i < P_SIZE; i++) {
#pragma HLS pipeline
        // Calculating indexes for input diagonals
        index = ex + i;
        index1 = ex1 + i;
        index_neg = i - ex;
        index_neg_1 = i - ex1;
        /*
        * Reading right value from memmoory input based on calculated indexes.
        * Out of boundary read conditions applied based on the paper.
        */
        if (index1 > P_SIZE - 1) {
            a1 = 0;
            b1 = 0;
            c1 = 1;
            d1 = 1;
            e1 = 0;
            r1 = 0;
        } else {
            a1 = a_ram2[index1];
            b1 = b_ram2[index1];
            c1 = c_ram2[index1];
            d1 = d_ram2[index1];
            e1 = e_ram2[index1];
            r1 = r_ram2[index1];
        }

        if (index > P_SIZE - 1) {
            a2 = 0;
            b2 = 0;
            c2 = 1;
            d2 = 1;
            e2 = 0;
            r2 = 0;
        } else {
            a2 = a_ram3[index];
            b2 = b_ram3[index];
            c2 = c_ram3[index];
            d2 = d_ram3[index];
            e2 = e_ram3[index];
            r2 = r_ram3[index];
        }

        if (index_neg_1 < 0) {
            a_1 = 0;
            b_1 = 1;
            c_1 = 1;
            d_1 = 0;
            e_1 = 0;
            r_1 = 0;
        } else {
            a_1 = a_ram2[index_neg_1];
            b_1 = b_ram2[index_neg_1];
            c_1 = c_ram2[index_neg_1];
            d_1 = d_ram2[index_neg_1];
            e_1 = e_ram2[index_neg_1];
            r_1 = r_ram2[index_neg_1];
        }
        if (index_neg < 0) {
            a_2 = 0;
            b_2 = 1;
            c_2 = 1;
            d_2 = 0;
            e_2 = 0;
            r_2 = 0;
        } else {
            a_2 = a[index_neg];
            b_2 = b[index_neg];
            c_2 = c[index_neg];
            d_2 = d_ram3[index_neg];
            e_2 = e[index_neg];
            r_2 = r[index_neg];
        }

        // Calculation of part results
        T t1 = e2 * b1 - c2 * d1;
        T t2 = a2 * d1;
        T t3 = d_1 * a_2;
        T s1 = b[i] * t1 + d[i] * t2;
        T s2 = b_1 * (e_2 * b[i] - c_2 * d[i]) + t3 * d[i];
        T D = t3 * t1 - b_1 * (c_2 * t1 + e_2 * t2);

        /*
        * The weakest point of this algorithm is division.
        * When some of the input diagonals are zero it may lead to division by zero.
        */
        T D_inv = 1 / D;

        T x1_temp = s1 * D_inv;
        T x2_temp = s1 * D_inv;
        T x3_temp = s2 * D_inv;
        T x4_temp = s2 * D_inv;

        x1 = -a_2 * x1_temp;
        x2 = b_1 * x2_temp;
        x3 = d1 * x3_temp;
        x4 = -e2 * x4_temp;

        // Calculation of output diagonals
        a_out[i] = x1 * a_1;
        b_out[i] = x1 * c_1 + x2 * b_2 + a[i];
        c_out[i] = x1 * e_1 + x2 * d_2 + c[i] + x3 * b2 + x4 * a1;
        d_temp = e[i] + x3 * d2 + x4 * c1;
        d_out[i] = d_temp;
        e_out[i] = x4 * e1;

        r_out[i] = x1 * r_1 + x2 * r_2 + r[i] + x3 * r2 + x4 * r1;
    } // End of main loop
    // Applying boundary conditions to output diagonals after main loops finishes.
    d_out[P_SIZE - 1] = 1;
    e_out[P_SIZE - 2] = 0;
    e_out[P_SIZE - 1] = 0;
    b_out[0] = 1;
    a_out[0] = 0;
    a_out[1] = 0;
}

} // namespace internal

/**
* @brief Solves for u in linear system Pu = r \n
* It calls function pentadiag_step for each step until all diagonals instead of
* main are zeros (or very close to zero).
* Result U is made by dividing Each of elements of right hand vactor (\a v) by
* main diagonal (\a c) \n
* Structure of input matrix: \n
*  | c d e 0 0 | \n
*  | b c d e 0 | \n
*  | a b c d e | \n
*  | 0 a b c d | \n
*  | 0 0 a b c |
*@tparam T data type used in whole function (double by default)
*@tparam P_SIZE Size of the operating matrix
*@tparam logN Number of steps for algorithm
*@param[in] c Main diagonal
*@param[in] b First lower
*@param[in] a Second lower
*@param[in] d First upper
*@param[in] e Second upper
*@param[in]     v Right hand side vector of length n
*@param[out]    u Vectors of unknows to solve for
*/
template <typename T, unsigned int P_SIZE, unsigned int logN>
void pentadiagCr(T a[P_SIZE], T b[P_SIZE], T c[P_SIZE], T d[P_SIZE], T e[P_SIZE], T v[P_SIZE], T u[P_SIZE]) {
#pragma HLS INLINE off
    T a_out[P_SIZE];
    T d_out[P_SIZE];
    T e_out[P_SIZE];
    T b_out[P_SIZE];
    T c_out[P_SIZE];
    T r_out[P_SIZE];

    /*
    * These are boundary condition from paper to ensure the right values from
    * input.
    */
    d[P_SIZE - 1] = 1;
    e[P_SIZE - 2] = 0;
    e[P_SIZE - 1] = 0;
    b[0] = 1; //
    a[0] = 0;
    a[1] = 0;
    /*
    * Main loop calling step function. Referring to the paper it needs
    * log2(P_SIZE) steps.
    * Although experiments have shown that the algorithm failes for number of
    * steps greater than 8.
    * In that case it is recommend limit numebr of steps to 8. The same idea is
    * used in LOGN calculation in pentadiag_top.h file.
    */
    for (int k = 0; k < (LOGN >> 1); k++) {
        internal::pentadiag_step<double, P_SIZE>(a, b, c, d, e, v, a_out, b_out, c_out, d_out, e_out, r_out, 2 * k);
        internal::pentadiag_step<double, P_SIZE>(a_out, b_out, c_out, d_out, e_out, r_out, a, b, c, d, e, v, 2 * k + 1);
    }
    /*
     *Calculate output
    */
    if (LOGN % 2 == 1) {
        internal::pentadiag_step<double, P_SIZE>(a, b, c, d, e, v, a_out, b_out, c_out, d_out, e_out, r_out, LOGN - 1);
        for (int i = 0; i < P_SIZE; i++) {
#pragma HLS pipeline
            u[i] = r_out[i] / c_out[i];
        }
    } else {
        for (int i = 0; i < P_SIZE; i++) {
#pragma HLS pipeline
            u[i] = v[i] / c[i];
        }
    }
}
} // namespace solver
} // namespace xf

#endif
