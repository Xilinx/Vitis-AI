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
 *  @brief Kernel implementation file
 *
 *  $DateTime: 2019/04/09 12:00:00 $
 */

#ifndef _HLS_FD_SOLVER_H_
#define _HLS_FD_SOLVER_H_
#include "ap_int.h"
#include "xf_fintech/spmv.hpp"
#include "xf_fintech/dimv.hpp"
#include "xf_fintech/trsv.hpp"
#ifndef __SYNTHESIS__
#include <assert.h>
#include <boost/algorithm/string.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#endif

#define PRAGMA_SUB(x) _Pragma(#x)
#define DO_PRAGMA(x) PRAGMA_SUB(x)

namespace xf {
namespace fintech {
namespace internal {

/// @brief Get V-inner index from an S ordered vector
inline unsigned int V2S(const unsigned int index, const unsigned int m1, const unsigned int m2) {
    unsigned int row = index % m2;
    unsigned int col = index / m2;
    return row * m1 + col;
}

/// @brief Get S-inner index from an V ordered vector
inline unsigned int S2V(const unsigned int index, const unsigned int m1, const unsigned int m2) {
    unsigned int row = index % m1;
    unsigned int col = index / m1;
    return row * m2 + col;
}

/// @brief Class to encapsulate the Finite Difference engine components
template <typename DT,
          unsigned int M_SIZE,
          unsigned int LOG2_M_SIZE,
          unsigned int A_SIZE,
          unsigned int MEM_WIDTH,
          unsigned int DIM2_SIZE1 = 3,
          unsigned int DIM2_SIZE2 = 5>
class Solver {
   public:
    /// @brief default constructor
    Solver() {
#pragma HLS inline
    }
    /// @brief Copy and reorder S ordered vector into V inner form
    /// @param[in]  v_in  Vector representing array of size m1 x m2 flattened in S
    /// inner form
    /// @param[in]  m1   Size of array [0..m1-1] in S direction
    /// @param[in]  m2   Size of array [0..m2-1] in V direction
    /// @param[out] v_out Vector representing array of size m1 x m2 flattened in V
    /// inner form
    void reorderS2V(const DT v_in[M_SIZE], DT v_out[M_SIZE], const unsigned int m1, const unsigned int m2) {
#pragma HLS ARRAY_PARTITION variable = v_out cyclic factor = MEM_WIDTH dim = 1

        for (unsigned int i = 0; i < M_SIZE; ++i) {
#pragma HLS UNROLL factor = MEM_WIDTH
            v_out[S2V(i, m1, m2)] = v_in[i];
        }
    }

    /// @brief Copy and reorder V ordered vector into S inner form
    /// @param[in]  v_in  Vector representing array of size m1 x m2 flattened in V
    /// inner form
    /// @param[in]  m1   Size of array [0..m1] in S direction
    /// @param[in]  m2   Size of array [0..m2] in V direction
    /// @param[out] v_out Vector representing array of size m1 x m2 flattened in S
    /// inner form
    void reorderV2S(const DT v_in[M_SIZE], DT v_out[M_SIZE], const unsigned int m1, const unsigned int m2) {
#pragma HLS ARRAY_PARTITION variable = v_out cyclic factor = MEM_WIDTH dim = 1

        for (unsigned int i = 0; i < M_SIZE; ++i) {
#pragma HLS UNROLL factor = MEM_WIDTH
            v_out[V2S(i, m1, m2)] = v_in[i];
        }
    }

    /// @brief Utility function to copy a vector
    void CopyVector(const DT v_in[M_SIZE], DT v_out[M_SIZE]) {
        for (unsigned int i = 0; i < M_SIZE; ++i) {
            v_out[i] = v_in[i];
        }
    }

    /// @brief Utility function to add two vectors
    void vectorAdd(const DT v_in0[M_SIZE], const DT v_in1[M_SIZE], DT v_out[M_SIZE]) {
#pragma HLS ARRAY_PARTITION variable = v_out cyclic factor = MEM_WIDTH dim = 1

        for (int i = 0; i < M_SIZE; ++i) {
#pragma HLS UNROLL factor = MEM_WIDTH
            v_out[i] = v_in0[i] + v_in1[i];
        }
    }

    /// @brief Utility function to subtract two vectors
    void vectorSub(const DT v_in0[M_SIZE], const DT v_in1[M_SIZE], DT v_out[M_SIZE]) {
#pragma HLS ARRAY_PARTITION variable = v_out cyclic factor = MEM_WIDTH dim = 1

        for (int i = 0; i < M_SIZE; ++i) {
#pragma HLS UNROLL factor = MEM_WIDTH
            v_out[i] = v_in0[i] - v_in1[i];
        }
    }

    /// @brief Wrapper to PCR tridiagonal solver
    /// @details Solves tridiagonal linear system
    /// @param[in]  X1    M x 3 array holding lower/main/upper diagonals of X1
    /// tridiagonal matrix
    /// @param[in]  rhs   Right hand side of linear system to be solved
    /// @param[in]  m1    Size of array [0..m1-1] in S direction
    /// @param[in]  m2    Size of array [0..m2-1] in V direction
    /// @param[out] v_out  Solution vector representing array of size m1 x m2
    /// flattened in V inner form
    void triDiagSovlerPCR(
        DT X1[M_SIZE][DIM2_SIZE1], DT rhs[M_SIZE], DT v_out[M_SIZE], const unsigned int m1, const unsigned int m2) {
        DT r[M_SIZE];
        DT a[M_SIZE];
        DT b[M_SIZE];
        DT c[M_SIZE];
#pragma HLS RESOURCE variable = r core = RAM_2P_BRAM
#pragma HLS RESOURCE variable = a core = RAM_2P_BRAM
#pragma HLS RESOURCE variable = b core = RAM_2P_BRAM
#pragma HLS RESOURCE variable = c core = RAM_2P_BRAM
        DO_PRAGMA(HLS array_partition variable = r cyclic factor = FD_NUM_PCR)
        DO_PRAGMA(HLS array_partition variable = a cyclic factor = FD_NUM_PCR)
        DO_PRAGMA(HLS array_partition variable = b cyclic factor = FD_NUM_PCR)
        DO_PRAGMA(HLS array_partition variable = c cyclic factor = FD_NUM_PCR)

        // Working copies of diagonals as solver will overwrite them
        for (int i = 0; i < M_SIZE; i++) {
            //#pragma HLS pipeline
            r[i] = rhs[i];
            a[i] = X1[i][0];
            b[i] = X1[i][1];
            c[i] = X1[i][2];
        }

        // Call the PCR tridiagonal solver
        xf::fintech::trsvCore<DT, M_SIZE, LOG2_M_SIZE, FD_NUM_PCR>(a, b, c, r);

        // Final division for result
        for (int k = 0; k < M_SIZE; k++) {
            // Note that v_out is written in V inner order using the S2V function
            v_out[S2V(k, m1, m2)] = r[k] / b[k];
        }
    }

    /// @brief Solve pentadiagonal form linear system
    /// @details This is a highly serial algorithm and is the bottleneck in this
    /// solver
    /// Unfortunately due to the formulation used by In 'T Hout & Foulon, the
    /// pentadiagonal array contains diagonals
    /// which are not fully populated.  This causes the common parallel
    /// pentadiagonal systems to fail due to
    /// divide-by-zero errors or similar.
    /// @param[in]  A          M x 5 array holding lower/lower/main/upper/upper
    /// diagonals of X1 tridiagonal matrix
    /// @param[in]  rhs        Right hand side of linear system to be solved
    /// @param[in]  precompute Flag to indicate the scaling factors should be
    /// computed
    /// @param[out] v_out       Solution vector representing array of size m1 x m2
    /// flattened in V inner form
    void pendaDiagSovler(DT A[M_SIZE][DIM2_SIZE2], DT rhs[M_SIZE], DT v_out[M_SIZE], bool precompute) {
        // These are preprocessed versions of the diagonals.
        // For a given array these are constant so they are calculated once and
        // cached (by marking as static) and using precompute variable.
        static DT d[M_SIZE];
        static DT a[M_SIZE];
        static DT e[M_SIZE];
        static DT c[M_SIZE];
        static DT f[M_SIZE];

        static DT xmult0[M_SIZE];
        static DT xmult1[M_SIZE];

        static DT xmult_r;

        static DT d_inv[M_SIZE];

        // Not static as this changes each time
        DT r[M_SIZE];

        // Working variable
        DT xmult = 0.0;

        // One time precomputation of the vectors
        if (precompute) {
            // Main diagonal
            for (unsigned int i = 0; i < M_SIZE; ++i) {
                d[i] = A[i][2];
            }

            // First upper/lower, dropping padded zeros
            for (unsigned int i = 0; i < M_SIZE - 1; ++i) {
                a[i] = A[i + 1][1];
                c[i] = A[i][3];
            }

            // Second upper/lower, dropping padded zeros
            for (unsigned int i = 0; i < M_SIZE - 2; ++i) {
                e[i] = A[i + 2][0];
                f[i] = A[i][4];
            }

            // Scaling factors
            for (unsigned int i = 1; i < (M_SIZE - 1); ++i) {
                xmult = a[i - 1] / d[i - 1];
                d[i] = d[i] - xmult * c[i - 1];
                c[i] = c[i] - xmult * f[i - 1];
                xmult0[i] = xmult;
                xmult = e[i - 1] / d[i - 1];
                a[i] = a[i] - xmult * c[i - 1];
                d[i + 1] = d[i + 1] - xmult * f[i - 1];
                xmult1[i] = xmult;
            }

            // d manipulation
            xmult_r = a[M_SIZE - 2] / d[M_SIZE - 2];
            d[M_SIZE - 1] = d[M_SIZE - 1] - xmult_r * c[M_SIZE - 2];

            // Invert d (one time operation and allows multiplies to be used later on)
            for (unsigned int i = 0; i < M_SIZE; ++i) {
                d_inv[i] = 1.0 / d[i];
            }
        }

        // Input vector changes each time
        for (unsigned int i = 0; i < M_SIZE; ++i) {
            r[i] = rhs[i];
        }

        // Apply the scaling factors
        for (unsigned int i = 1; i < (M_SIZE - 1); ++i) {
            r[i] = r[i] - xmult0[i] * r[i - 1];
            r[i + 1] = r[i + 1] - xmult1[i] * r[i - 1];
        }

        // Back solve
        v_out[M_SIZE - 1] = (r[M_SIZE - 1] - xmult_r * r[M_SIZE - 2]) * d_inv[M_SIZE - 1];
        v_out[M_SIZE - 2] = (r[M_SIZE - 2] - c[M_SIZE - 2] * v_out[M_SIZE - 1]) * d_inv[M_SIZE - 2];
        unsigned int i = M_SIZE - 3;
        do {
            v_out[i] = (r[i] - f[i] * v_out[i + 2] - c[i] * v_out[i + 1]) * d_inv[i];
        } while (i-- > 0);
    }
};

/// @brief Utility class to encapsulate the multiplier elements
template <typename DT,
          unsigned int MEM_WIDTH,
          unsigned int INDEX_WIDTH,
          unsigned int A_SIZE,
          unsigned int M_SIZE,
          unsigned int LOG2_M_SIZE,
          unsigned int DIM2_SIZE1 = 3,
          unsigned int DIM2_SIZE2 = 5>
class StreamWrapper {
   private:
    static const unsigned int M_SIZE_BLOCKS = M_SIZE / MEM_WIDTH;

   public:
    typedef xf::fintech::blas::WideType<DT, MEM_WIDTH> WideDataType;
    typedef hls::stream<WideDataType> WideStreamType;

   public:
    /// @brief default constructor
    StreamWrapper() {
#pragma HLS inline
    }
    /// @brief Computes multiplication of tridiagonal matrix by a vector
    /// @param[in]  A1         M x 3 array holding lower/main/upper diagonals of
    /// A1 tridiagonal matrix
    /// @param[in]  u          Vector to be multiplied
    /// @param[out] rhs1_tmp0 Multiplication result
    /// @param[out] u_out     Stream form of U to pass to pentadiagonal multiplier
    void streamDimv3(DT A1[M_SIZE][DIM2_SIZE1], DT u[M_SIZE], DT rhs1_tmp0[M_SIZE], WideStreamType& u_out) {
        for (unsigned int i = 0; i < M_SIZE_BLOCKS; ++i) {
#pragma HLS PIPELINE
            WideDataType val;
#pragma HLS ARRAY_PARTITION variable = val complete
            for (unsigned int j = 0; j < MEM_WIDTH; ++j) {
                val[j] = u[i * MEM_WIDTH + j];
            }
            u_out.write(val);
        }
        xf::fintech::blas::dimv<DT, M_SIZE, DIM2_SIZE1, MEM_WIDTH>(A1, u, M_SIZE, rhs1_tmp0);
    }

    /// @brief Computes multiplication of pentadiagonal matrix by a vector
    /// @param[in]  A2         M x 5 array holding lower/main/upper diagonals of
    /// A2 tridiagonal matrix
    /// @param[in]  u_in      Vector to be multiplied [stream format]
    /// @param[in]  m1         Size of array [0..m1-1] in S direction
    /// @param[in]  m2         Size of array [0..m2-1] in V direction
    /// @param[out] rhs2_tmp0 Multiplication result
    /// @param[out] u_out     Vector output [stream format]
    void streamDimv5(DT A2[M_SIZE][DIM2_SIZE2],
                     WideStreamType& u_in,
                     unsigned int m1,
                     unsigned int m2,
                     DT rhs2_tmp0[M_SIZE],
                     WideStreamType& u_out) {
        DT u_r0[M_SIZE]; // S-inner
        DT u_r1[M_SIZE]; // V-inner
#pragma HLS ARRAY_PARTITION variable = u_r0 cyclic factor = MEM_WIDTH dim = 1
#pragma HLS ARRAY_PARTITION variable = u_r1 cyclic factor = MEM_WIDTH dim = 1

        for (unsigned int i = 0; i < M_SIZE_BLOCKS; ++i) {
#pragma HLS PIPELINE
            WideDataType val;
#pragma HLS ARRAY_PARTITION variable = val complete
            val = u_in.read();
            u_out.write(val);
            for (unsigned int j = 0; j < MEM_WIDTH; ++j) {
                u_r0[i * MEM_WIDTH + j] = val[j];
            }
        }
        // S2V conversion
        Solver<DT, M_SIZE, LOG2_M_SIZE, A_SIZE, MEM_WIDTH> solver;
        solver.reorderS2V(u_r0, u_r1, m1, m2);
        xf::fintech::blas::dimv<DT, M_SIZE, DIM2_SIZE2, MEM_WIDTH>(A2, u_r1, M_SIZE, rhs2_tmp0);
    }

    /// @brief Computes multiplication of sparse matrix by vector plus a constant
    /// @param[in]  A          Sparse matrix value
    /// @param[in]  Ar         Sparse matrix row
    /// @param[in]  Ac         Sparse matrix column
    /// @param[in]  u_in      Vector to be multiplied [stream format]
    /// @param[in]  b          Vector to be added after sparse-mult stage
    /// @param[in]  Annz       Number of non-zeros in sparse matrix (how many
    /// elements of A/Ar/Ac are valid)
    /// @param[in]  M          Matrix M-size === (m1+1) x (m2+1)
    /// @param[out] y0         Result of mult-add in flattened S-inner form
    void streamSparseMultAdd(DT A[A_SIZE],
                             unsigned int Ar[A_SIZE],
                             unsigned int Ac[A_SIZE],
                             WideStreamType& u_in,
                             DT b[M_SIZE],
                             unsigned int Annz,
                             unsigned int M,
                             DT y0[M_SIZE]) {
#pragma HLS ARRAY_PARTITION variable = y0 cyclic factor = MEM_WIDTH dim = 1
        DT u[M_SIZE];
#pragma HLS ARRAY_PARTITION variable = u cyclic factor = MEM_WIDTH dim = 1
        DT y0_tmp1[M_SIZE];
#pragma HLS ARRAY_PARTITION variable = y0_tmp1 cyclic factor = MEM_WIDTH dim = 1

        for (unsigned int i = 0; i < M_SIZE_BLOCKS; ++i) {
#pragma HLS PIPELINE
            WideDataType val = u_in.read();
#pragma HLS ARRAY_PARTITION variable = val complete
            for (unsigned int j = 0; j < MEM_WIDTH; ++j) {
                u[i * MEM_WIDTH + j] = val[j];
            }
        }
        xf::fintech::blas::Spmv<DT, MEM_WIDTH, INDEX_WIDTH, M_SIZE, M_SIZE, A_SIZE> spmv;
        spmv.sparseMultAdd(A, Ar, Ac, u, b, y0_tmp1, Annz, M);
        unsigned int vec_blocks = M / MEM_WIDTH;
        for (unsigned int i = 0; i < vec_blocks; ++i) {
#pragma HLS PIPELINE
            for (unsigned int j = 0; j < MEM_WIDTH; ++j) {
                y0[i * MEM_WIDTH + j] = u[i * MEM_WIDTH + j] + y0_tmp1[i * MEM_WIDTH + j];
            }
        }
    }

    /// @brief Wrapper function to combine multipliers into a dataflow region and
    /// allow parallelization
    void parallelBlocks(DT A[A_SIZE],
                        unsigned int Ar[A_SIZE],
                        unsigned int Ac[A_SIZE],
                        DT u[M_SIZE],
                        DT b[M_SIZE],
                        unsigned int Annz,
                        unsigned int M,
                        DT A1[M_SIZE][DIM2_SIZE1],
                        DT A2[M_SIZE][DIM2_SIZE2],
                        unsigned int m1,
                        unsigned int m2,
                        DT y0[M_SIZE],
                        DT rhs1_tmp0[M_SIZE],
                        DT rhs2_tmp0[M_SIZE]) {
        WideStreamType u0;
        WideStreamType u1;
#pragma HLS DATAFLOW
        streamDimv3(A1, u, rhs1_tmp0, u0);
        streamDimv5(A2, u0, m1, m2, rhs2_tmp0, u1);
        streamSparseMultAdd(A, Ar, Ac, u1, b, Annz, M, y0);
    }
};

} // end of internal namespace block

/// @brief Top level callable function to perform the Douglas ADI method
/// @details This function creates the solver/stream wrapper objects and connects
/// them up
/// It also provides the extra connectivity for the non-streaming blocks
/// @param[in]  A          Sparse matrix value
/// @param[in]  Ar         Sparse matrix row
/// @param[in]  Ac         Sparse matrix column
/// @param[in]  Annz       Number of non-zeros in sparse matrix (how many
/// elements of A/Ar/Ac are valid)
/// @param[in]  A1         Tridiagonal matrix stored as three vectors
/// lower/main/upper
/// @param[in]  A2         Pentadiagonal matrix stored as five vectors
/// lower/lower/main/upper/upper
/// @param[in]  X1         Tridiagonal matrix stored as three vectors
/// lower/main/upper
/// @param[in]  X2         Pentadiagonal matrix stored as five vectors
/// lower/lower/main/upper/upper
/// @param[in]  b          Boundary condition vector
/// @param[in]  u0         Initial condition (payoff condition for a call
/// option)
/// @param[in]  M1         Size of array [0..M1] in S direction
/// @param[in]  M2         Size of array [0..M2] in V direction
/// @param[in]  N          Iteration count
/// @param[out] u          Calculated price grid
template <typename DT,
          unsigned int MEM_WIDTH,
          unsigned int INDEX_WIDTH,
          unsigned int A_SIZE,
          unsigned int M_SIZE,
          unsigned int LOG2_M_SIZE,
          unsigned int DIM2_SIZE1,
          unsigned int DIM2_SIZE2>
void FdDouglas(DT A[A_SIZE],
               unsigned int Ar[A_SIZE],
               unsigned int Ac[A_SIZE],
               unsigned int Annz,
               DT A1[M_SIZE][DIM2_SIZE1],
               DT A2[M_SIZE][DIM2_SIZE2],
               DT X1[M_SIZE][DIM2_SIZE1],
               DT X2[M_SIZE][DIM2_SIZE2],
               DT b[M_SIZE],
               DT u0[M_SIZE],
               unsigned int M1,
               unsigned int M2,
               unsigned int N,
               DT u[M_SIZE]) {
    DT y0[M_SIZE];
#pragma HLS ARRAY_PARTITION variable = y0 cyclic factor = MEM_WIDTH dim = 1
    DT rhs1[M_SIZE];
    DT y1[M_SIZE];
    DT rhs2[M_SIZE];
    DT y2[M_SIZE];

    DT rhs1_tmp0[M_SIZE];
    DT rhs2_tmp0[M_SIZE];

    internal::Solver<DT, M_SIZE, LOG2_M_SIZE, A_SIZE, MEM_WIDTH, DIM2_SIZE1, DIM2_SIZE2> solver;
    internal::StreamWrapper<DT, MEM_WIDTH, INDEX_WIDTH, A_SIZE, M_SIZE, LOG2_M_SIZE> stream_wrapper;

    solver.CopyVector(u0, u);

    // Perform the Douglas iteration
    // After N iterations the u vector will hold the output
    for (int i = 0; i < N; ++i) {
        // These operations are performed in parallel with a stream passing vector
        // data between them
        stream_wrapper.parallelBlocks(A, Ar, Ac, u, b, Annz, M_SIZE, A1, A2, M1, M2, y0, rhs1_tmp0, rhs2_tmp0);

        // These operations depending on the previous one completing and run
        // sequentially
        solver.vectorSub(y0, rhs1_tmp0, rhs1);
        solver.triDiagSovlerPCR(X1, rhs1, y1, M1, M2);
        solver.vectorSub(y1, rhs2_tmp0, rhs2);
        solver.pendaDiagSovler(X2, rhs2, y2, i == 0);

        // Vector u has to be reordered back to S-inner form
        solver.reorderV2S(y2, u, M1, M2);
    }
}
}
} // End of namespace xf::fintech

#endif
