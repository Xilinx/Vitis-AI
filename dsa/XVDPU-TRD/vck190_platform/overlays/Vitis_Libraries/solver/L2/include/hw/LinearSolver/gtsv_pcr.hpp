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
 * @file gtsv_pcr.hpp
 * @brief Tri-diagonal linear solver. Parallel Cyclic Reduction method.
 */

#ifndef _XF_SOLVER_L1_GTSV_PCR_
#define _XF_SOLVER_L1_GTSV_PCR_

#ifndef _SYNTHESIS_
#include <iostream>
#endif

namespace xf {
namespace solver {

namespace internal {

// gtsv, log2 sweeps, 1 CU
template <typename T, unsigned int N>
void gtsv_multisweeps_1cu(T low1[N], T diag1[N], T up1[N], T rhs1[N], T low2[N], T diag2[N], T up2[N], T rhs2[N]) {
#pragma HLS dependence variable = low1 inter false
#pragma HLS dependence variable = diag1 inter false
#pragma HLS dependence variable = up1 inter false
#pragma HLS dependence variable = rhs1 inter false
#pragma HLS dependence variable = low2 inter false
#pragma HLS dependence variable = diag2 inter false
#pragma HLS dependence variable = up2 inter false
#pragma HLS dependence variable = rhs2 inter false

    const unsigned int N2u = N >> 1;
    const unsigned int N2o = N - N2u;

    const unsigned int clzN = __builtin_clz(N);
    const unsigned int ctzN = __builtin_ctz(N);
    const unsigned int nBits = sizeof(unsigned int) * 8 - 1;
    const unsigned int logN = nBits - clzN + (((clzN + ctzN) == nBits) ? 0 : 1);
    const unsigned int NSWEEP = logN;

    T a[3], b[3], c[3], v[3];
#pragma HLS array_partition variable = a complete
#pragma HLS array_partition variable = b complete
#pragma HLS array_partition variable = c complete
#pragma HLS array_partition variable = v complete

    const unsigned int nIter = N + 1;

LoopLines:
    for (unsigned int j = 0; j < nIter * NSWEEP; j++) {
#pragma HLS pipeline

        unsigned int i = j % nIter;
        unsigned int i2 = (i - 1) >> 1;
        bool sweepeven = ((j / nIter) % 2 == 0); // sweep is even or odd

        // i=0: read first elment and init a, b, c, v
        // i=1,2,...,N: read from the second elements and compute

        if (i == 0) {
            for (unsigned int r = 0; r < 2; r++) {
#pragma HLS unroll
                a[r] = 0.0;
                b[r] = 1.0;
                c[r] = 0.0;
                v[r] = 0.0;
            };
            if (sweepeven) {
                a[2] = low1[0];
                b[2] = diag1[0];
                c[2] = up1[0];
                v[2] = rhs1[0];
            } else {
                a[2] = low2[0];
                b[2] = diag2[0];
                c[2] = up2[0];
                v[2] = rhs2[0];
            }
        } else {
            // update read regs
            a[0] = a[1];
            a[1] = a[2];
            b[0] = b[1];
            b[1] = b[2];
            c[0] = c[1];
            c[1] = c[2];
            v[0] = v[1];
            v[1] = v[2];
            if (i < N) {
                if (sweepeven) {
                    a[2] = low1[i];
                    b[2] = diag1[i];
                    c[2] = up1[i];
                    v[2] = rhs1[i];
                } else {
                    a[2] = low2[i];
                    b[2] = diag2[i];
                    c[2] = up2[i];
                    v[2] = rhs2[i];
                };
            } else {
                a[2] = 0.0;
                b[2] = 1.0;
                c[2] = 0.0;
                v[2] = 0.0;
            }

            // Compute values
            T a_1 = a[0];
            T a0 = a[1];
            T a1 = a[2];
            T b_1 = b[0];
            T b0 = b[1];
            T b1 = b[2];
            T c_1 = c[0];
            T c0 = c[1];
            T c1 = c[2];
            T v_1 = v[0];
            T v0 = v[1];
            T v1 = v[2];

            T k1 = a0 / b_1;
            T ak1 = a_1 * k1;
            T ck1 = c_1 * k1;
            T vk1 = v_1 * k1;

            T k2 = c0 / b1;
            T ak2 = a1 * k2;
            T ck2 = c1 * k2;
            T vk2 = v1 * k2;

            T low = -ak1;
            T diag = b0 - ck1 - ak2;
            T up = -ck2;
            T rhs = v0 - vk1 - vk2;

            // write
            unsigned int addw = ((i - 1) % 2 == 0) ? i2 : (i2 + N2o);
            if (addw < N) {
                if (sweepeven) {
                    low2[addw] = low;
                    diag2[addw] = diag;
                    up2[addw] = up;
                    rhs2[addw] = rhs;
                } else {
                    low1[addw] = low;
                    diag1[addw] = diag;
                    up1[addw] = up;
                    rhs1[addw] = rhs;
                };
            };

        }; // end of if-else: i=0 or i>0
    };     // end of iterations

}; // end of gtsv_multisweeps_1cu

// gtsv, log2 sweeps, n CU (n>=2)
template <typename T, unsigned int N, unsigned int NCU>
void gtsv_multisweeps_ncu(T low1[N], T diag1[N], T up1[N], T rhs1[N], T low2[N], T diag2[N], T up2[N], T rhs2[N]) {
#pragma HLS dependence variable = low1 inter false
#pragma HLS dependence variable = diag1 inter false
#pragma HLS dependence variable = up1 inter false
#pragma HLS dependence variable = rhs1 inter false
#pragma HLS dependence variable = low2 inter false
#pragma HLS dependence variable = diag2 inter false
#pragma HLS dependence variable = up2 inter false
#pragma HLS dependence variable = rhs2 inter false

    const unsigned int N2u = N >> 1;
    const unsigned int N2o = N - N2u;
    const unsigned int NCU2 = NCU >> 1;
    const int Ns = N2o % NCU - NCU2;
    const unsigned int Nwc = (Ns >= 0) ? Ns : -Ns; // write cache size

    const unsigned int clzN = __builtin_clz(N);
    const unsigned int ctzN = __builtin_ctz(N);
    const unsigned int nBits = sizeof(unsigned int) * 8 - 1;
    const unsigned int logN = nBits - clzN + (((clzN + ctzN) == nBits) ? 0 : 1);
    const unsigned int NSWEEP = logN;

    T a[NCU + 2];
    T b[NCU + 2];
    T c[NCU + 2];
    T v[NCU + 2];
#pragma HLS array_partition variable = a complete
#pragma HLS array_partition variable = b complete
#pragma HLS array_partition variable = c complete
#pragma HLS array_partition variable = v complete

    unsigned int idxrow[NCU];
#pragma HLS array_partition variable = idxrow complete

    // init read regs
    T cachelow[2][NCU2];
    T cachediag[2][NCU2];
    T cacheup[2][NCU2];
    T cacherhs[2][NCU2];
#pragma HLS array_partition variable = cachelow complete
#pragma HLS array_partition variable = cachediag complete
#pragma HLS array_partition variable = cacheup complete
#pragma HLS array_partition variable = cacherhs complete

    T cacheadd[2][NCU2];
#pragma HLS array_partition variable = cacheadd complete

    const unsigned int nIter = ((unsigned int)(N + NCU - 1)) / NCU + 2;

LoopLines:
    for (unsigned int j = 0; j < nIter * NSWEEP; j++) {
#pragma HLS pipeline

        unsigned int i = j % nIter;
        unsigned int i2 = (i - 1) >> 1;
        bool sweepeven = ((j / nIter) % 2 == 0); // sweep is even or odd

        // i = 0: init a, b, c, v and read first elments
        // i = 1,2,...,nIter-2: read, compute and write
        // i = nIter-1: write

        if (i == 0) { // read first elment
            for (unsigned int r = 0; r < (NCU + 1); r++) {
#pragma HLS unroll
                a[r] = 0.0;
                b[r] = 1.0;
                c[r] = 0.0;
                v[r] = 0.0;
            };
            if (sweepeven) {
                a[NCU + 1] = low1[0];
                b[NCU + 1] = diag1[0];
                c[NCU + 1] = up1[0];
                v[NCU + 1] = rhs1[0];
            } else {
                a[NCU + 1] = low2[0];
                b[NCU + 1] = diag2[0];
                c[NCU + 1] = up2[0];
                v[NCU + 1] = rhs2[0];
            }
        } else { // read from the second element and compute

            // ***** update regs including reads ***** //
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
                unsigned int addr = (i - 1) * NCU + r + 1;

                if (addr < N) {
                    if (sweepeven) {
                        a[2 + r] = low1[addr];
                        b[2 + r] = diag1[addr];
                        c[2 + r] = up1[addr];
                        v[2 + r] = rhs1[addr];
                    } else {
                        a[2 + r] = low2[addr];
                        b[2 + r] = diag2[addr];
                        c[2 + r] = up2[addr];
                        v[2 + r] = rhs2[addr];
                    }
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

            // ***** compute values ***** //
            for (unsigned int r = 0; r < NCU; r++) {
#pragma HLS unroll
                T a_1 = a[r];
                T a0 = a[r + 1];
                T a1 = a[r + 2];
                T b_1 = b[r];
                T b0 = b[r + 1];
                T b1 = b[r + 2];
                T c_1 = c[r];
                T c0 = c[r + 1];
                T c1 = c[r + 2];
                T v_1 = v[r];
                T v0 = v[r + 1];
                T v1 = v[r + 2];

                T k1 = a0 / b_1;
                T ak1 = a_1 * k1;
                T ck1 = c_1 * k1;
                T vk1 = v_1 * k1;

                T k2 = c0 / b1;
                T ak2 = a1 * k2;
                T ck2 = c1 * k2;
                T vk2 = v1 * k2;

                low[r] = -ak1;
                diag[r] = b0 - ck1 - ak2;
                up[r] = -ck2;
                rhs[r] = v0 - vk1 - vk2;
            };

            // ***** write ***** //
            for (unsigned int r = 0; r < NCU; r++) {
#pragma HLS unroll

                if (r < NCU2) { // direct writes: 0 to NCU/2-1
                    unsigned int addr = r * 2;
                    unsigned int addw = (i - 1) * NCU2 + r;

                    if (addw < N2o) { // write out
                        if (sweepeven) {
                            low2[addw] = low[addr];
                            diag2[addw] = diag[addr];
                            up2[addw] = up[addr];
                            rhs2[addw] = rhs[addr];
                        } else {
                            low1[addw] = low[addr];
                            diag1[addw] = diag[addr];
                            up1[addw] = up[addr];
                            rhs1[addw] = rhs[addr];
                        };
                    };
                } else if (r < (NCU2 + Nwc)) { // cache write: NCU/2 to NCU/2+Nwc-1
                    unsigned int rNCU2 = r - NCU2;
                    unsigned int idx1 = (i % 2 == 0) ? 1 : 0;
                    unsigned int idx2 = (i % 2 == 0) ? 0 : 1;
                    unsigned int addr = (r - NCU2) * 2 + 1;
                    unsigned int addw = N2o + (i - 2) * NCU2 + rNCU2;

                    if ((addw >= N2o) && (addw < N)) {
                        if (sweepeven) {
                            low2[addw] = cachelow[idx2][rNCU2];
                            diag2[addw] = cachediag[idx2][rNCU2];
                            up2[addw] = cacheup[idx2][rNCU2];
                            rhs2[addw] = cacherhs[idx2][rNCU2];
                        } else {
                            low1[addw] = cachelow[idx2][rNCU2];
                            diag1[addw] = cachediag[idx2][rNCU2];
                            up1[addw] = cacheup[idx2][rNCU2];
                            rhs1[addw] = cacherhs[idx2][rNCU2];
                        };
                    };

                    // write to cache
                    cachelow[idx1][rNCU2] = low[addr];
                    cachediag[idx1][rNCU2] = diag[addr];
                    cacheup[idx1][rNCU2] = up[addr];
                    cacherhs[idx1][rNCU2] = rhs[addr];
                } else { // direct write: NCU/2+Nwc to NCU-1
                    unsigned int rNCU2 = r - NCU2;
                    unsigned int addr = rNCU2 * 2 + 1;
                    unsigned int addw = N2o + (i - 1) * NCU2 + rNCU2;

                    if (addw < N) { // write out
                        if (sweepeven) {
                            low2[addw] = low[addr];
                            diag2[addw] = diag[addr];
                            up2[addw] = up[addr];
                            rhs2[addw] = rhs[addr];
                        } else {
                            low1[addw] = low[addr];
                            diag1[addw] = diag[addr];
                            up1[addw] = up[addr];
                            rhs1[addw] = rhs[addr];
                        };
                    };
                }; // end of write out: NCU/2+Nwc to NCU-1
            }      // end of write out
        }
    }; // end of iterations
};     // end of gtsv_sweeps_ncu

// gtsv, 1 sweep, 1 CU
template <typename T, unsigned int N, unsigned int NCU>
void gtsv_singlesweep(
    T inlow[N], T indiag[N], T inup[N], T inrhs[N], T outlow[N], T outdiag[N], T outup[N], T outrhs[N]) {
#pragma HLS dependence variable = inlow inter false
#pragma HLS dependence variable = indiag inter false
#pragma HLS dependence variable = inup inter false
#pragma HLS dependence variable = inrhs inter false
#pragma HLS dependence variable = outlow inter false
#pragma HLS dependence variable = outdiag inter false
#pragma HLS dependence variable = outup inter false
#pragma HLS dependence variable = outrhs inter false

    const unsigned int N2u = N >> 1;
    const unsigned int N2o = N - N2u;
    const unsigned int NCU2 = NCU >> 1;
    const int Ns = N2o % NCU - NCU2;
    const unsigned int Nwc = (Ns >= 0) ? Ns : -Ns; // write cache size

    T a[NCU + 2];
    T b[NCU + 2];
    T c[NCU + 2];
    T v[NCU + 2];
#pragma HLS array_partition variable = a complete
#pragma HLS array_partition variable = b complete
#pragma HLS array_partition variable = c complete
#pragma HLS array_partition variable = v complete

    unsigned int idxrow[NCU];
#pragma HLS array_partition variable = idxrow complete

    // init read regs
    for (unsigned int r = 0; r < (NCU + 1); r++) {
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

    T cachelow[2][NCU2];
    T cachediag[2][NCU2];
    T cacheup[2][NCU2];
    T cacherhs[2][NCU2];
#pragma HLS array_partition variable = cachelow complete
#pragma HLS array_partition variable = cachediag complete
#pragma HLS array_partition variable = cacheup complete
#pragma HLS array_partition variable = cacherhs complete

    T cacheadd[2][NCU2];
#pragma HLS array_partition variable = cacheadd complete

    const unsigned int nIter = ((unsigned int)(N + NCU - 1)) / NCU;

LoopLines:
    for (unsigned int i = 0; i < nIter; i++) {
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

    LoopComputeValue:
        for (unsigned int r = 0; r < NCU; r++) {
#pragma HLS unroll
            T a_1 = a[r];
            T a0 = a[r + 1];
            T a1 = a[r + 2];
            T b_1 = b[r];
            T b0 = b[r + 1];
            T b1 = b[r + 2];
            T c_1 = c[r];
            T c0 = c[r + 1];
            T c1 = c[r + 2];
            T v_1 = v[r];
            T v0 = v[r + 1];
            T v1 = v[r + 2];

            T k1 = a0 / b_1;
            T ak1 = a_1 * k1;
            T ck1 = c_1 * k1;
            T vk1 = v_1 * k1;

            T k2 = c0 / b1;
            T ak2 = a1 * k2;
            T ck2 = c1 * k2;
            T vk2 = v1 * k2;

            low[r] = -ak1;
            diag[r] = b0 - ck1 - ak2;
            up[r] = -ck2;
            rhs[r] = v0 - vk1 - vk2;
        };

        unsigned int i2 = (i >> 1);

        // write
        if (NCU == 1) {
            unsigned int addw = (i % 2 == 0) ? i2 : (i2 + N2o);
            if (addw < N) {
                outlow[addw] = low[0];
                outdiag[addw] = diag[0];
                outup[addw] = up[0];
                outrhs[addw] = rhs[0];
            };
        } else {
            for (unsigned int r = 0; r < NCU2; r++) {
#pragma HLS unroll
                unsigned int addr = r * 2;
                unsigned int addw = i * NCU2 + r;

                if (addw < N2o) {
                    outlow[addw] = low[addr];
                    outdiag[addw] = diag[addr];
                    outup[addw] = up[addr];
                    outrhs[addw] = rhs[addr];
                };
            };

            for (unsigned int r = 0; r < Nwc; r++) {
#pragma HLS unroll
                unsigned int addr = r * 2 + 1;
                unsigned int addw = N2o + i * NCU2 + r;
                unsigned int idx1 = (i % 2 == 0) ? 0 : 1;
                unsigned int idx2 = (i % 2 == 0) ? 1 : 0;
                cachelow[idx1][r] = low[addr];
                cachediag[idx1][r] = diag[addr];
                cacheup[idx1][r] = up[addr];
                cacherhs[idx1][r] = rhs[addr];
                cacheadd[idx1][r] = addw;

                if (i > 0) {
                    unsigned int addwc = cacheadd[idx2][r];
                    if (addwc < N) {
                        outlow[addwc] = cachelow[idx2][r];
                        outdiag[addwc] = cachediag[idx2][r];
                        outup[addwc] = cacheup[idx2][r];
                        outrhs[addwc] = cacherhs[idx2][r];
                    };
                };
            };

            for (unsigned int r = Nwc; r < NCU2; r++) {
#pragma HLS unroll
                unsigned int addr = r * 2 + 1;
                unsigned int addw = N2o + i * NCU2 + r;
                if (addw < N) {
                    outlow[addw] = low[addr];
                    outdiag[addw] = diag[addr];
                    outup[addw] = up[addr];
                    outrhs[addw] = rhs[addr];
                }
            };

        }; // end of if-else
    };     // end of LoopLines

    // write last elements
    for (unsigned int r = 0; r < Nwc; r++) {
#pragma HLS unroll
        unsigned int idx2 = (N % 2 == 0) ? 1 : 0;
        unsigned int addwc = cacheadd[idx2][r];
        if (addwc < N) {
            outlow[addwc] = cachelow[idx2][r];
            outdiag[addwc] = cachediag[idx2][r];
            outup[addwc] = cacheup[idx2][r];
            outrhs[addwc] = cacherhs[idx2][r];
        };
    };

}; // end of gtsv_singlesweep

// gtsv, 1 sweep, 1 CU
template <typename T, unsigned int N, unsigned int NCU, bool POW2>
void gtsv_singlesweep(T inlow[N],
                      T indiag[N],
                      T inup[N],
                      T inrhs[N],
                      T outlow[N],
                      T outdiag[N],
                      T outup[N],
                      T outrhs[N],
                      unsigned int inidxrow[N],
                      unsigned int outidxrow[N]) {
#pragma HLS dependence variable = inlow inter false
#pragma HLS dependence variable = indiag inter false
#pragma HLS dependence variable = inup inter false
#pragma HLS dependence variable = inrhs inter false
#pragma HLS dependence variable = outlow inter false
#pragma HLS dependence variable = outdiag inter false
#pragma HLS dependence variable = outup inter false
#pragma HLS dependence variable = outrhs inter false

    const unsigned int N2u = N >> 1;
    const unsigned int N2o = N - N2u;
    const unsigned int NCU2 = NCU >> 1;
    const int Ns = N2o % NCU - NCU2;
    const unsigned int Nwc = (Ns >= 0) ? Ns : -Ns; // write cache size

    T a[NCU + 2];
    T b[NCU + 2];
    T c[NCU + 2];
    T v[NCU + 2];
#pragma HLS array_partition variable = a complete
#pragma HLS array_partition variable = b complete
#pragma HLS array_partition variable = c complete
#pragma HLS array_partition variable = v complete

    unsigned int idxrow[NCU];
#pragma HLS array_partition variable = idxrow complete

    // init read regs
    for (unsigned int r = 0; r < (NCU + 1); r++) {
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

    T cachelow[2][NCU2];
    T cachediag[2][NCU2];
    T cacheup[2][NCU2];
    T cacherhs[2][NCU2];
#pragma HLS array_partition variable = cachelow complete
#pragma HLS array_partition variable = cachediag complete
#pragma HLS array_partition variable = cacheup complete
#pragma HLS array_partition variable = cacherhs complete

    T cacheadd[2][NCU2];
#pragma HLS array_partition variable = cacheadd complete

    const unsigned int nIter = ((unsigned int)(N + NCU - 1)) / NCU;

LoopLines:
    for (unsigned int i = 0; i < nIter; i++) {
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

    LoopComputeValue:
        for (unsigned int r = 0; r < NCU; r++) {
#pragma HLS unroll
            T a_1 = a[r];
            T a0 = a[r + 1];
            T a1 = a[r + 2];
            T b_1 = b[r];
            T b0 = b[r + 1];
            T b1 = b[r + 2];
            T c_1 = c[r];
            T c0 = c[r + 1];
            T c1 = c[r + 2];
            T v_1 = v[r];
            T v0 = v[r + 1];
            T v1 = v[r + 2];

            T k1 = a0 / b_1;
            T ak1 = a_1 * k1;
            T ck1 = c_1 * k1;
            T vk1 = v_1 * k1;

            T k2 = c0 / b1;
            T ak2 = a1 * k2;
            T ck2 = c1 * k2;
            T vk2 = v1 * k2;

            low[r] = -ak1;
            diag[r] = b0 - ck1 - ak2;
            up[r] = -ck2;
            rhs[r] = v0 - vk1 - vk2;
        };

        unsigned int i2 = (i >> 1);

        // write
        if (NCU == 1) {
            unsigned int addw = (i % 2 == 0) ? i2 : (i2 + N2o);
            if (addw < N) {
                outlow[addw] = low[0];
                outdiag[addw] = diag[0];
                outup[addw] = up[0];
                outrhs[addw] = rhs[0];
            };
        } else {
            for (unsigned int r = 0; r < NCU2; r++) {
#pragma HLS unroll
                unsigned int addr = r * 2;
                unsigned int addw = i * NCU2 + r;

                if (addw < N2o) {
                    outlow[addw] = low[addr];
                    outdiag[addw] = diag[addr];
                    outup[addw] = up[addr];
                    outrhs[addw] = rhs[addr];
                };
            };

            for (unsigned int r = 0; r < Nwc; r++) {
#pragma HLS unroll
                unsigned int addr = r * 2 + 1;
                unsigned int addw = N2o + i * NCU2 + r;

                unsigned int idx1 = (i % 2 == 0) ? 0 : 1;
                unsigned int idx2 = (i % 2 == 0) ? 1 : 0;
                cachelow[idx1][r] = low[addr];
                cachediag[idx1][r] = diag[addr];
                cacheup[idx1][r] = up[addr];
                cacherhs[idx1][r] = rhs[addr];
                cacheadd[idx1][r] = addw;

                if (i > 0) {
                    unsigned int addwc = cacheadd[idx2][r];
                    if (addwc < N) {
                        outlow[addwc] = cachelow[idx2][r];
                        outdiag[addwc] = cachediag[idx2][r];
                        outup[addwc] = cacheup[idx2][r];
                        outrhs[addwc] = cacherhs[idx2][r];
                    };
                };
            };

            for (unsigned int r = Nwc; r < NCU2; r++) {
#pragma HLS unroll
                unsigned int addr = r * 2 + 1;
                unsigned int addw = N2o + i * NCU2 + r;

                if (addw < N) {
                    outlow[addw] = low[addr];
                    outdiag[addw] = diag[addr];
                    outup[addw] = up[addr];
                    outrhs[addw] = rhs[addr];
                }
            };

        }; // end of if-else
    };     // end of LoopLines

    // write last elements
    for (unsigned int r = 0; r < Nwc; r++) {
#pragma HLS unroll
        unsigned int idx2 = (N % 2 == 0) ? 1 : 0;
        unsigned int addwc = cacheadd[idx2][r];
        if (addwc < N) {
            outlow[addwc] = cachelow[idx2][r];
            outdiag[addwc] = cachediag[idx2][r];
            outup[addwc] = cacheup[idx2][r];
            outrhs[addwc] = cacherhs[idx2][r];
        };
    };
};

template <typename T, unsigned int N, unsigned int NCU>
int gtsv_core(unsigned int n, T matDiagLow[N], T matDiag[N], T matDiagUp[N], T rhs[N]) {
    // compute log2(N)
    const unsigned int clzN = __builtin_clz(N);
    const unsigned int ctzN = __builtin_ctz(N);
    const unsigned int nBits = sizeof(unsigned int) * 8 - 1;
    const unsigned int logN = nBits - clzN + (((clzN + ctzN) == nBits) ? 0 : 1);

#ifndef _SYNTHESIS_
    // check N
    if (N == 0) {
        std::cout << "ERROR: N should be positive." << std::endl;
        return 1;
    }
#endif

    const unsigned int clzNCU = __builtin_clz(NCU);
    const unsigned int ctzNCU = __builtin_ctz(NCU);
    const bool power2 = ((clzNCU + ctzNCU) == nBits) ? true : false;

#ifndef _SYNTHESIS_
    if (!power2) {
        std::cout << "ERROR: NCU should be power of 2." << std::endl;
        return 1;
    };
#endif

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

    unsigned int inidx[N];
    unsigned int outidx[N];
#pragma HLS array_partition variable = inidx cyclic factor = NCU
#pragma HLS array_partition variable = outidx cyclic factor = NCU

    if (!power2) {
        for (unsigned int i = 0; i < N; i++) {
            inidx[i] = i;
        };
    };

    for (unsigned int s = 0; s < (logN >> 1); s++) {
        internal::gtsv_singlesweep<T, N, NCU>(matDiagLow, matDiag, matDiagUp, rhs, outlow, outdiag, outup, outrhs);
        internal::gtsv_singlesweep<T, N, NCU>(outlow, outdiag, outup, outrhs, matDiagLow, matDiag, matDiagUp, rhs);
    };
    if (logN % 2 == 1) {
        internal::gtsv_singlesweep<T, N, NCU>(matDiagLow, matDiag, matDiagUp, rhs, outlow, outdiag, outup, outrhs);
    };

    const unsigned int nIter = ((unsigned int)(N + NCU - 1)) / NCU;

LoopWrite:
    for (unsigned int i = 0; i < nIter; i++) {
#pragma HLS pipeline
        for (unsigned int r = 0; r < NCU; r++) {
            unsigned int addr = i * NCU + r;
            if (addr < N) {
                if (logN % 2 == 0) {
                    rhs[addr] = rhs[addr] / matDiag[addr];
                } else {
                    rhs[addr] = outrhs[addr] / outdiag[addr];
                };
            };
        }
    }
    return 0;
};
} // end of internal name space

/*!
  @brief Tri-diagonal linear solver. Compute solution to linear system with a tridiagonal matrix. Parallel Cyclic
  Reduction method.
  @param T data type (support float and double)
  @param NMAX matrix size
  @param NCU number of compute units
  @param matDiagLow lower diagonal of matrix
  @param matDiag diagonal of matrix
  @param matDiagUp upper diagonal of matrix
  @param rhs right-hand side
*/
template <typename T, unsigned int NMAX, unsigned int NCU>
int gtsv(unsigned int n, T* matDiagLow, T* matDiag, T* matDiagUp, T* rhs) {
    T DiagLow[NMAX], Diag[NMAX], DiagUp[NMAX], RHS[NMAX];
#pragma HLS RESOURCE variable = DiagLow core = RAM_2P_BRAM
#pragma HLS RESOURCE variable = Diag core = RAM_2P_BRAM
#pragma HLS RESOURCE variable = DiagUp core = RAM_2P_BRAM
#pragma HLS RESOURCE variable = RHS core = RAM_2P_BRAM
#pragma HLS array_partition variable = DiagLow cyclic factor = NCU
#pragma HLS array_partition variable = Diag cyclic factor = NCU
#pragma HLS array_partition variable = DiagUp cyclic factor = NCU
#pragma HLS array_partition variable = RHS cyclic factor = NCU

LoopRead:
    for (int i = 0; i < n; i++) {
#pragma HLS pipeline
        DiagLow[i] = matDiagLow[i];
        Diag[i] = matDiag[i];
        DiagUp[i] = matDiagUp[i];
        RHS[i] = rhs[i];
    };

    internal::gtsv_core<T, NMAX, NCU>(n, DiagLow, Diag, DiagUp, RHS);

LoopWrite:
    for (int i = 0; i < n; i++) {
#pragma HLS pipeline
        matDiagLow[i] = DiagLow[i];
        matDiag[i] = Diag[i];
        matDiagUp[i] = DiagUp[i];
        rhs[i] = RHS[i];
    };

    return 0;
};

} // end of namespace solver
} // end of namespace xf

#endif
