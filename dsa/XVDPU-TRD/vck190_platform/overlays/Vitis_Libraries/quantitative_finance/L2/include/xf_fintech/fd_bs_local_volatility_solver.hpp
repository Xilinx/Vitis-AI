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
 *  @brief FPGA FD accelerator kernel
 *
 *  $DateTime: 2018/02/05 02:36:41 $
 */

#ifndef _HLS_FD1D_SOLVER_H_
#define _HLS_FD1D_SOLVER_H_

#include <assert.h>
#include "ap_fixed.h"
#include "ap_shift_reg.h"
#include <hls_stream.h>
#include <math.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include "xf_fintech/trsv.hpp"

namespace xf {
namespace fintech {
namespace internal {

/// @brief Internal wrapper class to encapsulate the Black-Scholes local volatility solver
///
/// @tparam DT Data Type used for this function
/// @tparam DT_EQ_TYPE Integer data type of same width as DT
/// @tparam N Discretized spatial grid size
/// @tparam M Discretized temporal grid maximum possible size
template <typename DT, typename DT_EQ_TYPE, unsigned int N, unsigned int M>
class FdBsLvSolverWrapper {
   public:
    typedef struct ParallelDataType { DT data[1]; } ParallelDataType;
    typedef hls::stream<ParallelDataType> ParallelStreamType;

    typedef struct Parallel3DataType { DT data[3]; } Parallel3DataType;
    typedef hls::stream<Parallel3DataType> Parallel3StreamType;

    static const unsigned int BITS_PER_DATA_TYPE = 8 * sizeof(DT);
    static const unsigned int DATA_ELEMENTS_PER_DDR_WORD = 512 / BITS_PER_DATA_TYPE;

    /// @brief default constructor
    FdBsLvSolverWrapper() {
#pragma HLS inline
    }

    /// @brief Read 512-bit DDR bus into array
    ///
    /// @param[in] in Pointer to data location in DDR
    /// @param[out] out Array of data as type DT
    /// @param[in] size Size of data region to read (required to be multiple of data elements in DDR)
    void read_vector(const ap_uint<512>* in, DT out[], const unsigned int size) {
        for (unsigned int i = 0; i < size / DATA_ELEMENTS_PER_DDR_WORD; ++i) {
#pragma HLS PIPELINE
            ap_uint<512> wide_temp = in[i];
            for (unsigned int j = 0; j < DATA_ELEMENTS_PER_DDR_WORD; ++j) {
#pragma HLS UNROLL
                DT_EQ_TYPE temp = wide_temp.range(BITS_PER_DATA_TYPE * (j + 1) - 1, BITS_PER_DATA_TYPE * j);
                out[DATA_ELEMENTS_PER_DDR_WORD * i + j] = *(DT*)(&temp);
            }
        }
    }

    /// @brief Write array to 512-bit DDR bus
    ///
    /// @param[in] in Array of data as type DT
    /// @param[out] out Pointer to data location in DDR
    /// @param[in] size Size of data region to read (required to be multiple of data elements in DDR)
    void write_vector(const DT in[], ap_uint<512>* out, const unsigned int size) {
        for (unsigned int i = 0; i < size / DATA_ELEMENTS_PER_DDR_WORD; ++i) {
#pragma HLS PIPELINE
            ap_uint<512> wide_temp;
            for (unsigned int j = 0; j < DATA_ELEMENTS_PER_DDR_WORD; ++j) {
#pragma HLS UNROLL
                DT temp = in[DATA_ELEMENTS_PER_DDR_WORD * i + j];
                wide_temp.range(BITS_PER_DATA_TYPE * (j + 1) - 1, BITS_PER_DATA_TYPE * j) = *(DT_EQ_TYPE*)(&temp);
            }
            out[i] = wide_temp;
        }
    }

    /// @brief Copy array
    ///
    /// @param[in] in Array of data as type DT
    /// @param[out] out Array of data as type DT
    /// @param[in] size Size of data region to copy
    void copy_vector(const DT in[], DT out[], const unsigned int size) {
        for (unsigned int i = 0; i < size; ++i) {
            out[i] = in[i];
        }
    }

    /// @brief Read sigma(x,t) from DDR for t=m-1 and t=m
    ///
    /// @param[in] in Pointer to start of sigma data location in DDR
    /// @param[out] sig0 sigma vector for t=m-1
    /// @param[out] sig1 sigma vector for t=m
    /// @param[in] m time-step
    void read_sigma(const ap_uint<512>* in, ParallelStreamType& sig0, ParallelStreamType& sig1, const unsigned int m) {
        ParallelDataType parallel_temp;
        unsigned int offset;

        offset = (m - 1) * N;
        for (unsigned int i = 0; i < N / DATA_ELEMENTS_PER_DDR_WORD; ++i) {
#pragma HLS PIPELINE
            ap_uint<512> wide_temp = in[i + offset / DATA_ELEMENTS_PER_DDR_WORD];
            for (unsigned int j = 0; j < DATA_ELEMENTS_PER_DDR_WORD; ++j) {
#pragma HLS UNROLL
                DT_EQ_TYPE temp = wide_temp.range(BITS_PER_DATA_TYPE * (j + 1) - 1, BITS_PER_DATA_TYPE * j);
                parallel_temp.data[0] = *(DT*)(&temp);
                sig0.write(parallel_temp);
            }
        }

        offset = m * N;
        for (unsigned int i = 0; i < N / DATA_ELEMENTS_PER_DDR_WORD; ++i) {
#pragma HLS PIPELINE
            ap_uint<512> wide_temp = in[i + offset / DATA_ELEMENTS_PER_DDR_WORD];
            for (unsigned int j = 0; j < DATA_ELEMENTS_PER_DDR_WORD; ++j) {
#pragma HLS UNROLL
                DT_EQ_TYPE temp = wide_temp.range(BITS_PER_DATA_TYPE * (j + 1) - 1, BITS_PER_DATA_TYPE * j);
                parallel_temp.data[0] = *(DT*)(&temp);
                sig1.write(parallel_temp);
            }
        }
    }

    /// @brief Calculate difference between consecutive data elements
    ///
    /// @param[in] in Array of data as type DT
    /// @param[out] out Array of data as type DT
    /// @param[in] size Size of vector
    void calculateDeltas(const DT* in, DT* out, const unsigned int size) {
        out[0] = 0;
        for (unsigned int i = 1; i < size; ++i) {
            out[i] = in[i] - in[i - 1];
        }
    }

    /// @brief Convert stream to a vector
    ///
    /// @param[in] in Input stream
    /// @param[out] out Output array
    /// @param[in] size Number of data elements to extract from stream
    void stream_to_vector(ParallelStreamType& in, DT out[], unsigned int size) {
        ParallelDataType temp;
        for (unsigned int i = 0; i < size; ++i) {
            temp = in.read();
            out[i] = temp.data[0];
        }
    }

    /// @brief Convert vector to a stream
    ///
    /// @param[in] in Input array
    /// @param[out] out Output stream
    /// @param[in] size Vector size
    void vector_to_stream(DT in[], ParallelStreamType& out, unsigned int size) {
        ParallelDataType temp;
        for (unsigned int i = 0; i < size; ++i) {
            temp.data[0] = in[i];
            out.write(temp);
        }
    }

    /// @brief Convert stream3 to a vector
    ///
    /// @param[in] in Input stream
    /// @param[out] out Output array
    /// @param[in] size Number of data elements to extract from stream
    void stream3_to_vector3(Parallel3StreamType& in, DT out[][3], unsigned int size) {
        Parallel3DataType temp;
        for (unsigned int i = 0; i < size; ++i) {
            temp = in.read();
            for (unsigned int j = 0; j < 3; ++j) {
                out[i][j] = temp.data[j];
            }
        }
    }

    /// @brief Generate the left and right matrices in the Ax=B linear system
    ///
    /// @param[in] x Spatial grid
    /// @param[in] h Spatial grid spacings
    /// @param[in] t Temporal grid
    /// @param[in] dt Temporal grid spacings
    /// @param[in] sig0 Volatility vector sig(x) for t-1
    /// @param[in] sig1 Volatility vector sig(x) for t
    /// @param[in] r Interest rate r(t)
    /// @param[in] b Boundary vector b(x)
    /// @param[in] theta Controls explicit/implicit/Crank-Nicholson
    /// @param[in] ti Time index for which matrices are generated
    /// @param[out] lmatix Left-hand matrix as stream
    /// @param[out] rmatix Right-hand matrix as stream
    /// @param[out] discountedBoundary b(x) vector discounted with r(t) as a stream
    void generate_matrices(const DT x[],
                           const DT h[],
                           const DT t[],
                           const DT dt[],
                           ParallelStreamType& sig0,
                           ParallelStreamType& sig1,
                           const DT r[],
                           const DT boundary[],
                           const DT theta,
                           const unsigned int ti,
                           Parallel3StreamType& lmatrix,
                           Parallel3StreamType& rmatrix,
                           ParallelStreamType& discountedBoundary) {
        ParallelDataType sig;
        ParallelDataType dbtemp;
        Parallel3DataType temp3;
        DT a, b, c;
        DT a2, a2n, scale;

        for (unsigned int i = 0; i < N; ++i) {
            if ((i == 0) || (i == (N - 1))) {
                // Ensure we read from stream even though we don't need the data
                sig0.read();
                sig1.read();

                temp3.data[0] = (DT)0.0;
                temp3.data[1] = (DT)1.0;
                temp3.data[2] = (DT)0.0;
                lmatrix.write(temp3);

                temp3.data[0] = (DT)0.0;
                temp3.data[1] = (DT)1.0;
                temp3.data[2] = (DT)0.0;
                rmatrix.write(temp3);
            } else {
                // LHS first, including (I-theta*dt*A) term
                sig = sig0.read();
                a = (DT)0.5 * sig.data[0] * sig.data[0];
                b = r[ti - 1] - a;
                c = (DT)-1.0 * r[ti - 1];

                a2 = (DT)2.0 * a;
                a2n = (DT)-1.0 * a2;
                scale = dt[ti] / (h[i] + h[i + 1]);

                temp3.data[0] = (DT)0.0 - theta * (scale * ((a2 / h[i]) - b));
                temp3.data[1] = (DT)1.0 - theta * (scale * ((a2n / h[i]) - (a2 / h[i + 1])) + dt[ti] * c);
                temp3.data[2] = (DT)0.0 - theta * (scale * ((a2 / h[i + 1]) + b));
                lmatrix.write(temp3);

                // RHS side uses time t including (I+(1-theta)*dt*A) term
                sig = sig1.read();
                a = (DT)0.5 * sig.data[0] * sig.data[0];
                b = r[ti] - a;
                c = (DT)-1.0 * r[ti];

                a2 = (DT)2.0 * a;
                a2n = (DT)-1.0 * a2;
                scale = dt[ti] / (h[i] + h[i + 1]);

                temp3.data[0] = (DT)0.0 + ((DT)1.0 - theta) * (scale * ((a2 / h[i]) - b));
                temp3.data[1] = (DT)1.0 + ((DT)1.0 - theta) * (scale * ((a2n / h[i]) - (a2 / h[i + 1])) + dt[ti] * c);
                temp3.data[2] = (DT)0.0 + ((DT)1.0 - theta) * (scale * ((a2 / h[i + 1]) + b));
                rmatrix.write(temp3);
            }

            dbtemp.data[0] = boundary[i] * exp(-r[ti] * t[ti]);
            discountedBoundary.write(dbtemp);
        }
    }

    /// @brief Multiple array stream by vector stream
    ///
    /// @param[in] m Matrix as stream holding 3 data elements
    /// @param[in] v Input vector as stream
    /// @param[out] u Output vector as stream
    /// @param[in] size Vector length
    void diag3_mult(Parallel3StreamType& m, ParallelStreamType& v, ParallelStreamType& u, const unsigned int size) {
        static ap_shift_reg<DT, 3> sreg;
        Parallel3DataType temp3;
        ParallelDataType temp;
        ParallelDataType utemp;

        sreg.shift(0, 0); // Preload with zero
        temp = v.read();
        sreg.shift(temp.data[0], 0);

        for (unsigned int i = 0; i < size; ++i) {
            temp3 = m.read();
            if (i < size - 1) {
                temp = v.read();
            } else {
                temp.data[0] = 0;
            }
            sreg.shift(temp.data[0], 0);
            utemp.data[0] = temp3.data[0] * sreg.read(2) + temp3.data[1] * sreg.read(1) + temp3.data[2] * sreg.read(0);
            u.write(utemp);
        }
    }

    /// @brief Solve tridiagonal Mu = v linear system using PCR algorithm
    ///
    /// @param[in] m Matrix as stream holding 3 data elements
    /// @param[in] v Input vector as stream
    /// @param[out] u Output solution vector as stream
    /// @param[in] size Vector length
    void triDiagSolverPCR(Parallel3StreamType& m, ParallelStreamType& v, DT u[], const unsigned int size) {
        DT r[size];
        DT a[size];
        DT b[size];
        DT c[size];

        Parallel3DataType temp3;
        ParallelDataType temp;

        // Working copies of diagonals as solver will overwrite them
        for (unsigned int i = 0; i < size; i++) {
            //#pragma HLS pipeline
            temp3 = m.read();
            temp = v.read();
            r[i] = temp.data[0];
            a[i] = temp3.data[0];
            b[i] = temp3.data[1];
            c[i] = temp3.data[2];
        }

        // Call the PCR tridiagonal solver
        xf::fintech::trsvCore<DT, FD_N_SIZE, FD_LOG2_N_SIZE, FD_NUM_PCR>(a, b, c, r);

        // Final division for result
        for (unsigned int i = 0; i < size; i++) {
            u[i] = r[i] / b[i];
        }
    }

    /// @brief Modify vector with boundary conditions (replaces in[0] and in[size-1] with boundary vector)
    ///
    /// @param[in] in Input vector as stream
    /// @param[in] boundary Boundary vector as stream
    /// @param[out] u Output vector as stream
    /// @param[in] size Vector length
    void apply_boundary(ParallelStreamType& in,
                        ParallelStreamType& boundary,
                        ParallelStreamType& out,
                        unsigned int size) {
        ParallelDataType temp0;
        ParallelDataType temp1;

        for (unsigned int i = 0; i < size; ++i) {
            temp0 = in.read();
            temp1 = boundary.read();
            if ((i == 0) || (i == size - 1)) {
                out.write(temp1);
            } else {
                out.write(temp0);
            }
        }
    }

    /// @brief Wrapper function to allow parallelization
    ///
    /// @param[in] x Spatial grid
    /// @param[in] h Spatial grid spacings
    /// @param[in] t Temporal grid
    /// @param[in] dt Temporal grid spacings
    /// @param[in] r Interest rate r(t)
    /// @param[in] b Boundary vector b(x)
    /// @param[in] theta Controls explicit/implicit/Crank-Nicholson
    /// @param[in] ti Time index for which matrices are generated
    /// @param[in] u_in Input vector at time-step t-1
    /// @param[out] u_out Output vector at time-step t
    void parallel_block(ap_uint<512>* sigma,
                        DT x[],
                        DT h[],
                        DT t[],
                        DT dt[],
                        DT r[],
                        DT boundary[],
                        DT theta,
                        unsigned int ti,
                        DT u_in[],
                        DT u_out[]) {
#pragma HLS DATAFLOW
        ParallelStreamType sig0;
        ParallelStreamType sig1;
        Parallel3StreamType lmatrix;
        Parallel3StreamType rmatrix;
        ParallelStreamType u;
        ParallelStreamType mult;
        ParallelStreamType discountedBoundary;
        ParallelStreamType bounded;

#pragma HLS STREAM variable = sig0 depth = N
#pragma HLS STREAM variable = sig1 depth = N
#pragma HLS STREAM variable = lmatrix depth = N
#pragma HLS STREAM variable = rmatrix depth = N
#pragma HLS STREAM variable = u depth = N
#pragma HLS STREAM variable = mult depth = N
#pragma HLS STREAM variable = discountedBoundary depth = N
#pragma HLS STREAM variable = bounded depth = N

        vector_to_stream(u_in, u, N);
        read_sigma(sigma, sig0, sig1, ti);
        generate_matrices(x, h, t, dt, sig0, sig1, r, boundary, theta, ti, lmatrix, rmatrix, discountedBoundary);
        diag3_mult(rmatrix, u, mult, N);
        apply_boundary(mult, discountedBoundary, bounded, N);
        triDiagSolverPCR(lmatrix, bounded, u_out, N);
    }
};

} // Internal namespace

/// @brief Entry point to Fd1D Solver
///
/// @tparam DT Data Type used for this function
/// @tparam DT_EQ_TYPE Integer data type of same width as DT
/// @tparam N Discretized spatial grid size
/// @tparam M Discretized temporal grid maximum possible size
///
/// @param[in] xGrid Pointer to spatial grid
/// @param[in] tGrid Pointer to temporal grid
/// @param[in] sigma Pointer to 2D volatility array sigma(x,t)
/// @param[in] rate Pointer to interest rate vector
/// @param[in] boundary Pointer to boundary vector
/// @param[in] boundary Pointer to initial condition
/// @param[in] tGrid Pointer to temporal grid
/// @param[in] theta Controls explicit/implicit/Crank-Nicholson
/// @param[in] tSteps Size of tGrid
/// @param[out] solution Final solution vector
template <typename DT, typename DT_EQ_TYPE, unsigned int N, unsigned int M>
void FdBsLvSolver(ap_uint<512>* xGrid,
                  ap_uint<512>* tGrid,
                  ap_uint<512>* sigma,
                  ap_uint<512>* rate,
                  ap_uint<512>* initialCondition,
                  float theta,
                  DT boundary_lower,
                  DT boundary_upper,
                  unsigned int tSteps,
                  ap_uint<512>* solution) {
    DT x[N];
    DT t[M];
    DT r[M];
    DT u[N];
    DT un[N];
    DT boundary[N];
    DT h[N];
    DT dt[M];

    internal::FdBsLvSolverWrapper<DT, DT_EQ_TYPE, N, M> solver;

    // The vectors used for reading from DDR must be partitioned to match the number of data elements in each DDR word
    static const unsigned int f = solver.DATA_ELEMENTS_PER_DDR_WORD;
#pragma HLS array_partition variable = x cyclic factor = f dim = 1
#pragma HLS array_partition variable = t cyclic factor = f dim = 1
#pragma HLS array_partition variable = r cyclic factor = f dim = 1
#pragma HLS array_partition variable = boundary cyclic factor = f dim = 1
#pragma HLS array_partition variable = u cyclic factor = f dim = 1

    solver.read_vector(xGrid, x, N);
    solver.read_vector(tGrid, t, tSteps);
    solver.read_vector(rate, r, tSteps);
    solver.read_vector(initialCondition, u, N);

    // Boundary is put into a N vector to be passed as stream along with matrix data
    for (unsigned int i = 0; i < N; ++i) {
        if (i == 0) {
            boundary[i] = boundary_lower;
        } else if (i == (N - 1)) {
            boundary[N - 1] = boundary_upper;
        } else {
            boundary[i] = (DT)0.0;
        }
    }

    // This is a one-time step - TODO - work out parallelization here
    solver.calculateDeltas(x, h, N);
    solver.calculateDeltas(t, dt, tSteps);

    // for(unsigned int i=0;i<N;++i) std::cout << "x[" << i << "] = " << x[i] << std::endl;
    // for(unsigned int i=0;i<N;++i) std::cout << "h[" << i << "] = " << h[i] << std::endl;
    // for(unsigned int i=0;i<N;++i) std::cout << "t[" << i << "] = " << t[i] << std::endl;
    // for(unsigned int i=0;i<N;++i) std::cout << "dt[" << i << "] = " << dt[i] << std::endl;
    // for(unsigned int i=0;i<N;++i) std::cout << "r[" << i << "] = " << r[i] << std::endl;
    // for(unsigned int i=0;i<N;++i) std::cout << "ic[" << i << "] = " << u[i] << std::endl;
    // for(unsigned int i=0;i<N;++i) std::cout << "boundary[" << i << "] = " << boundary[i] << std::endl;

    // Iterate over each time-step
    for (int ti = 1; ti < tSteps; ++ti) {
        solver.parallel_block(sigma, x, h, t, dt, r, boundary, theta, ti, u, un);
        solver.copy_vector(un, u, N);
    }
    solver.write_vector(un, solution, N);
}

} // namespace fintech
} // namespace xf

#endif
