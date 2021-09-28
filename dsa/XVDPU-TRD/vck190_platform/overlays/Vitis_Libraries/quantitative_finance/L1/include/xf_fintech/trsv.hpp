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
 *  @file trsv.hpp
 *  @brief Tridiagonal solver header file
 *
 *  $DateTime: 2019/04/09 12:00:00 $
 */

#ifndef _XSOLVER_CORE_TRSV_
#define _XSOLVER_CORE_TRSV_

namespace xf {
namespace fintech {
namespace internal {
/**
* @brief Executes one step of odd-even elimination. \n
* For each row it calculates new diagonal element and right hand side element.
\n
* \n
*
* Please note the algorithm is very sensitive to zeros in main diagonal. \n
* Any zeros in main diagonal will lead to division by zero and algorithm fail.
*@tparam T data type used in whole function (double by default)
*@tparam N Size of the operating matrix
*@tparam NCU Number of compute units working in parallel
*@param[in] inlow Input vector of lower diagonal
*@param[in] indiag Input vector of main diagonal
*@param[in] inup Input vector of upper diagonal
*@param[in] inrhs Input vector of Right hand side

*@param[out] outlow Output vector of lower diagonal
*@param[out] outdiag Output vector of main diagonal
*@param[out] outup Output vector of upper diagonal
*@param[out] outrhs Output vector of Right hand side
*/
template <class T, unsigned int N, unsigned int NCU>
void trsv_step(T inlow[N], T indiag[N], T inup[N], T inrhs[N], T outlow[N], T outdiag[N], T outup[N], T outrhs[N]) {
#pragma HLS dependence variable = inlow inter false
#pragma HLS dependence variable = indiag inter false
#pragma HLS dependence variable = inup inter false
#pragma HLS dependence variable = inrhs inter false

#pragma HLS dependence variable = outlow inter false
#pragma HLS dependence variable = outdiag inter false
#pragma HLS dependence variable = outup inter false
#pragma HLS dependence variable = outrhs inter false

    const unsigned int N2 = N >> 1;
    const unsigned int NCU2 = (NCU == 1) ? 0 : (NCU >> 1);

    T a[NCU + 2];
    T b[NCU + 2];
    T c[NCU + 2];
    T v[NCU + 2];
#pragma HLS array_partition variable = a complete
#pragma HLS array_partition variable = b complete
#pragma HLS array_partition variable = c complete
#pragma HLS array_partition variable = v complete

    // init read regs
    for (int r = 0; r < (NCU + 1); r++) {
#pragma HLS unroll
        a[r] = 0.0;
        b[r] = 1.0;
        c[r] = 0.0;
        v[r] = 0.0;
    };
    a[NCU + 1] = inlow[0];
    b[NCU + 1] = indiag[0];
    c[NCU + 1] = inup[0];
    v[NCU + 1] = inrhs[0];

    T reglow[2][NCU2];
    T regdiag[2][NCU2];
    T regup[2][NCU2];
    T regrhs[2][NCU2];
#pragma HLS array_partition variable = reglow complete dim = 0
#pragma HLS array_partition variable = regdiag complete dim = 0
#pragma HLS array_partition variable = regup complete dim = 0
#pragma HLS array_partition variable = regrhs complete dim = 0

LoopLines:
    for (unsigned int i = 0; i < (N / NCU); i++) {
#pragma HLS pipeline

        // update read regs
        a[0] = a[NCU];
        a[1] = a[NCU + 1];
        b[0] = b[NCU];
        b[1] = b[NCU + 1];
        c[0] = c[NCU];
        c[1] = c[NCU + 1];
        v[0] = v[NCU];
        v[1] = v[NCU + 1];

        for (unsigned int r = 0; r < NCU; r++) {
#pragma HLS unroll
            unsigned int addr = i * NCU + r + 1;

            if (addr < N) {
                a[2 + r] = inlow[addr];
                b[2 + r] = indiag[addr];
                c[2 + r] = inup[addr];
                v[2 + r] = inrhs[addr];
            } else {
                a[2 + r] = 0.0;
                b[2 + r] = 1.0;
                c[2 + r] = 0.0;
                v[2 + r] = 0.0;
            };
        };

        T low[NCU];
        T diag[NCU];
        T up[NCU];
        T rhs[NCU];
#pragma HLS array_partition variable = low complete
#pragma HLS array_partition variable = diag complete
#pragma HLS array_partition variable = up complete
#pragma HLS array_partition variable = rhs complete

    LoopCompute:
        for (int i = 0; i < NCU; i++) {
#pragma HLS unroll

            T a_1 = a[i];
            T a0 = a[i + 1];
            T a1 = a[i + 2];

            T b_1 = b[i];
            T b0 = b[i + 1];
            T b1 = b[i + 2];

            T c_1 = c[i];
            T c0 = c[i + 1];
            T c1 = c[i + 2];

            T v_1 = v[i];
            T v0 = v[i + 1];
            T v1 = v[i + 2];

            T k1 = a0 / b_1;
            T ak1 = a_1 * k1;
            T ck1 = c_1 * k1;
            T vk1 = v_1 * k1;

            T k2 = c0 / b1;
            T ak2 = a1 * k2;
            T ck2 = c1 * k2;
            T vk2 = v1 * k2;

            low[i] = -ak1;
            diag[i] = b0 - ck1 - ak2;
            up[i] = -ck2;
            rhs[i] = v0 - vk1 - vk2;
        };

        // write
        if (NCU == 1) {
            unsigned int i2 = (i >> 1);
            unsigned int addc = (i % 2 == 0) ? i2 : (i2 + N2);
            outlow[addc] = low[0];
            outdiag[addc] = diag[0];
            outup[addc] = up[0];
            outrhs[addc] = rhs[0];
        } else {
            unsigned int addr1 = i * NCU2;              // ((i*NCU)>>1);
            unsigned int addr2 = ((i - 1) * NCU2) + N2; // (((i-1)*NCU)>>1) + N2;
            unsigned int regidx1 = i % 2;
            unsigned int regidx2 = (i % 2 == 0) ? 1 : 0;

            for (unsigned int r = 0; r < NCU2; r++) {
#pragma HLS unroll
                outlow[addr1 + r] = low[r * 2];
                outdiag[addr1 + r] = diag[r * 2];
                outup[addr1 + r] = up[r * 2];
                outrhs[addr1 + r] = rhs[r * 2];

                if (i > 0) {
                    outlow[addr2 + r] = reglow[regidx1][r];
                    outdiag[addr2 + r] = regdiag[regidx1][r];
                    outup[addr2 + r] = regup[regidx1][r];
                    outrhs[addr2 + r] = regrhs[regidx1][r];
                }

                reglow[regidx2][r] = low[r * 2 + 1];
                regdiag[regidx2][r] = diag[r * 2 + 1];
                regup[regidx2][r] = up[r * 2 + 1];
                regrhs[regidx2][r] = rhs[r * 2 + 1];
            };
        };

    }; // end of LoopLines

    // write last element
    if (NCU > 1) {
        unsigned int addr2 = N - NCU2;
        unsigned int regidx = N % 2 == 0 ? 0 : 1;

        for (unsigned int r = 0; r < NCU2; r++) {
#pragma HLS unroll
            outlow[addr2 + r] = reglow[regidx][r];
            outdiag[addr2 + r] = regdiag[regidx][r];
            outup[addr2 + r] = regup[regidx][r];
            outrhs[addr2 + r] = regrhs[regidx][r];
        };
    };
};

} // namespace internal
/**
  @brief Tridiagonal linear solver
  It solves tridiagonal linear system of equations by eliminating upper and
  lower diagonals
  To get result (U) divide each element of \a inrhs by coresponding element of
  main diagonal \a indiag
  @tparam T data type
  @tparam N matrix size
  @tparam logN log2(N)(TOREMOVE)
  @tparam NCU number of compute units
  @param[in] inlow lower diagonal
  @param[in] indiag diagonal
  @param[in] inup upper diagonal
  @param[in] inrhs right-hand side
 */
template <class T, unsigned int N, unsigned int logN, unsigned int NCU>
void trsvCore(T inlow[N], T indiag[N], T inup[N], T inrhs[N]) {
    // TODO: remove logN
    // TODO: N is not power of 2

    const int N2 = N >> 1;
    const int N4 = N >> 2;

    T outlow[N];
    T outdiag[N];
    T outup[N];
    T outrhs[N];

#pragma HLS RESOURCE variable = outlow core = RAM_2P_BRAM
#pragma HLS RESOURCE variable = outdiag core = RAM_2P_BRAM
#pragma HLS RESOURCE variable = outup core = RAM_2P_BRAM
#pragma HLS RESOURCE variable = outrhs core = RAM_2P_BRAM

#pragma HLS array_partition variable = outlow cyclic factor = NCU
#pragma HLS array_partition variable = outdiag cyclic factor = NCU
#pragma HLS array_partition variable = outup cyclic factor = NCU
#pragma HLS array_partition variable = outrhs cyclic factor = NCU

LoopTop:
    for (int s = 0; s < (logN >> 1); s++) {
        internal::trsv_step<T, N, NCU>(inlow, indiag, inup, inrhs, outlow, outdiag, outup, outrhs);

        internal::trsv_step<T, N, NCU>(outlow, outdiag, outup, outrhs, inlow, indiag, inup, inrhs);
    };

    if (logN % 2 == 1) {
        internal::trsv_step<T, N, NCU>(inlow, indiag, inup, inrhs, outlow, outdiag, outup, outrhs);
    LoopWrite:
        for (int i = 0; i < N; i++) {
#pragma HLS pipeline
            inlow[i] = outlow[i];
            indiag[i] = outdiag[i];
            inup[i] = outup[i];
            inrhs[i] = outrhs[i];
        };
    };
};

} // namespace solver
} // namespace xf

#endif
