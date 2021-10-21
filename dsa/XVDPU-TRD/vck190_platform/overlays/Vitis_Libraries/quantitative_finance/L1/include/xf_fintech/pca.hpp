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
 * @file pca.hpp
 * @brief L2 Principal Component Analysis templated implementation
 *
 */

#ifndef _PCA_H_
#define _PCA_H_

#ifndef __SYNTHESIS__
//#define _PCA_DEBUG_ 1
#endif

#include <hls_stream.h>
#include <limits>
#include "xf_solver_L2.hpp"
#include "covariance.hpp"
#ifdef _PCA_DEBUG_
#include <iostream>
#endif

namespace xf {
namespace fintech {

/**
 * @brief Enum describing implementation types for Principal Component Analysis
 */
enum class pcaImplementationMethod {
    /* Input data is expected to be any matrix with variables in the rows and observations in the columns */
    Covariance = 0,
    /*
     * Input data is expected to be the correlation matrix of some input data.
     * Must be a square and standarized (mean = 0, std = 1) matrix.
     */
    Correlation = 1
};

/**
 * @brief Class that will compute the Principal Component Analysis of some input data. Data is assumed to be
 * a set of at most @c MAX_VARS variables in the rows and at most @c MAX_OBS observations in the columns.
 *
 * @tparam DT The data type to be used
 * @tparam N The number of components to select from the PCA.
 * @tparam NCU Number of computational units to improve parallelism
 * @tparam MAX_VARS Maximum number of synthetisable rows.
 * @tparam MAX_OBS Maximum number of synthetisable cols.
 * @tparam IMPL_METHOD Choice of implementation method. By default it will use the covariance matrix.
 */
template <typename DT,
          unsigned int N,
          unsigned int NCU,
          unsigned int MAX_VARS,
          unsigned int MAX_OBS,
          pcaImplementationMethod IMPL_METHOD = pcaImplementationMethod::Covariance>
class PCA {
    constexpr static unsigned int EIG_DUP = 2;

    const unsigned m_noVariables;

    // Outputs
    DT m_pcVals[EIG_DUP][N];
    DT m_pcVecs[EIG_DUP][MAX_VARS * N];

    /**
     * @brief default constructor
     */
    PCA() {
#pragma HLS inline
#pragma HLS ARRAY_PARTITION variable = m_pcVals complete
#pragma HLS ARRAY_PARTITION variable = m_pcVecs dim = 1 complete
    }

    /**
     * @brief Shifts the elements of a buffer right up to position 'from'
     */
    template <typename T, unsigned S>
    static void shiftBuffer(unsigned from, T buffer[S]) {
#pragma HLS INLINE
    NSort_shift_buf_Loop:
        for (unsigned i = S - 1; i > from; i--) {
#pragma HLS PIPELINE
            buffer[i] = buffer[i - 1];
        }
    }

    /**
     * @brief Sorts and selects the first 'N' items from the array 'data'
     *
     * @tparam IT Data type of the array's indexes.
     *
     * @param data input data to be sorted
     * @param dcount Length of the input data to sort
     * @param sorted First N elements of the sorted array.
     * @param indexSorted Indexes of the first N sorted elements from the array.
     */
    template <typename IT = unsigned int>
    void nsort(const DT data[MAX_VARS], const unsigned int dcount, DT sorted[N], IT indexSorted[N]) {
        DT dSortedBuf[N];
        IT iSortedBuf[N];
    NSort_Init_Loop:
        for (unsigned i = 0; i < N; i++) {
#pragma HLS UNROLL
            dSortedBuf[i] = -std::numeric_limits<DT>::max();
            iSortedBuf[i] = -1;
        }

    NSort_main_Loop:
        for (unsigned id = 0; id < dcount; id++) {
            for (unsigned j = 0; j < N; j++) {
                if (data[id] > dSortedBuf[j]) {
                    shiftBuffer<DT, N>(j, dSortedBuf);
                    shiftBuffer<IT, N>(j, iSortedBuf);
                    dSortedBuf[j] = data[id];
                    iSortedBuf[j] = id;
                    break;
                }
            }
        }

        for (unsigned i = 0; i < N; i++) {
#pragma HLS UNROLL
            sorted[i] = dSortedBuf[i];
            indexSorted[i] = iSortedBuf[i];
        }
    }

    /**
     * @brief Selects from a provided matrix a subset of its columns
     *
     * @tparam IT The data type of the matrix indexes, defaults to unsigned int.
     *
     * @param data The input matrix
     * @param rows Number of rows in the input matrix.
     * @param cols Number of columns in the input matrix.
     * @param indexes The ordered list of columns to select from the matrix.
     * @param output The output matrix containing just the columns from indexes[]
     */
    template <typename IT = unsigned int>
    void filterMat(DT* data, const unsigned int rows, const unsigned int cols, IT indexes[N], DT* output) {
        unsigned index = 0;
    PCA_filtermat_loop:
        for (unsigned i = 0; i < rows; i++) {
            // Output just the columns in the sequence of 'indexes'
            for (unsigned j = 0; j < N; j++) {
#pragma HLS PIPELINE
                output[index++] = data[i * cols + indexes[j]];
            }
        }
    }

    void implement(unsigned int noVars, DT standarisedData[MAX_VARS * MAX_VARS]) {
        /*
         * Calculate Eigen Values and Eigen Vectors
         */
        int info; // unused
        DT eigVals[MAX_VARS];
        DT eigVecs[MAX_VARS * MAX_VARS];
        xf::solver::syevj<DT, MAX_VARS, NCU>(noVars, standarisedData, noVars, eigVals, eigVecs, noVars, info);
#ifdef _PCA_DEBUG_
        std::cout << "Unordered eigen-values are:" << std::endl;
        for (unsigned i = 0; i < noVars; i++) {
            std::cout << eigVals[i] << std::endl;
        }
#endif

        /*
         * Sort Eigen Values
         */
        unsigned int eigIndexes[N];
        DT pcVals[N];
#pragma HLS ARRAY_PARTITION variable = eigIndexes complete
#pragma HLS ARRAY_PARTITION variable = pcVals complete
        nsort(eigVals, noVars, pcVals, eigIndexes);
#ifdef _PCA_DEBUG_
        std::cout << "Ordered first " << N << " eig-vals are:" << std::endl;
        for (unsigned i = 0; i < N; i++) {
            std::cout << pcVals[i] << std::endl;
        }
#endif

        /*
         * Filter Eigen Vectors
         */
        DT pcVecs[N * MAX_VARS];
#pragma HLS ARRAY_PARTITION variable = pcVecs cyclic factor = N
        filterMat(eigVecs, noVars, noVars, eigIndexes, pcVecs);

        /*
         * Normalise Eigen Vectors.
         * Since eigen-vector's sign is arbitrary, we will follow matlab's convention of flipping the sign if the first
         * element of an eigen-vector is negative.
         */
        DT pcVecsNorm[N * MAX_VARS];
#pragma HLS ARRAY_PARTITION variable = pcVecsNorm cyclic factor = N
        bool signFlip[N];
    PCA_Sign_Normalization_Loop:
        for (unsigned int i = 0; i < noVars; i++) {
            for (unsigned int j = 0; j < N; j++) {
#pragma HLS PIPELINE II = 1
                if (i == 0) {
                    signFlip[j] = pcVecs[j] < 0;
                }
                pcVecsNorm[i * N + j] = signFlip[j] ? -pcVecs[i * N + j] : pcVecs[i * N + j];
            }
        }
#ifdef _PCA_DEBUG_
        std::cout << "Ordered first " << N << " eig-vecs are:" << std::endl;
        for (unsigned i = 0; i < noVars; i++) {
            for (unsigned j = 0; j < N; j++) {
                std::cout << pcVecsNorm[i * N + j];
                if (j != N - 1) {
                    std::cout << ",";
                }
            }
            std::cout << std::endl;
        }
#endif

        /*
         * Save streams
         */
        for (unsigned int i = 0; i < N; i++) {
            for (unsigned int j = 0; j < EIG_DUP; j++) {
#pragma HLS UNROLL
                m_pcVals[j][i] = pcVals[i];
            }
        }
        for (unsigned int i = 0; i < noVars * N; i++) {
            for (unsigned int j = 0; j < EIG_DUP; j++) {
#pragma HLS UNROLL
                m_pcVecs[j][i] = pcVecsNorm[i];
            }
        }
    }

   public:
    /**
     * @brief Calculates the core Principal Component Analysis functionality from the provided matrix
     * with the given method.
     *
     * @param noVars Number of variables (rows) in the matrix.
     * @param noObs Number of observations (cols) in the matrix.
     * @param data The matrix with the data to be analysed.
     */
    PCA(const unsigned int noVars, const unsigned int noObs, DT data[MAX_VARS][MAX_OBS]) : m_noVariables(noVars) {
#ifndef __SYNTHESIS__
        assert(noVars > N && "Must provide more variables than the selected N");
        assert(noVars <= MAX_VARS && "Number of variables must be <= than MAX_VARS templated parameter");
        assert(noObs <= MAX_OBS && "Number of observations must be <= than MAX_OBS templated parameter");
#endif

        DT standarisedData[MAX_VARS * MAX_VARS];
        if (IMPL_METHOD == pcaImplementationMethod::Covariance) {
            /*
             * Calculate Covariance matrix
             */
            DT covMatrix[MAX_VARS][MAX_VARS];
            // TODO - Numbers above 1 for ROWS_UNROLL break the covariance calculation in hw
            constexpr unsigned ROWS_UNROLL = 1, COLS_UNROLL = 2;
            xf::fintech::covCoreMatrix<DT, MAX_VARS, MAX_OBS, ROWS_UNROLL, COLS_UNROLL>(noVars, noObs, data, covMatrix);
            for (unsigned i = 0; i < noVars; i++) {
                for (unsigned j = 0; j < noVars; j++) {
#pragma HLS PIPELINE II = 1
                    standarisedData[i * noVars + j] = covMatrix[i][j];
                }
            }
        } else if (IMPL_METHOD == pcaImplementationMethod::Correlation) {
/*
 * Data is already standarised, just make sure a square matrix is provided and pass
 * the input data directly to implement
 */
#ifndef __SYNTHESIS__
            assert(noVars == noObs && "When using PCA with Correlation method, a square matrix must be provided");
#endif
            for (unsigned i = 0; i < noVars; i++) {
                for (unsigned j = 0; j < noVars; j++) {
#pragma HLS PIPELINE II = 1
                    standarisedData[i * noVars + j] = data[i][j];
                }
            }
        }
        implement(noVars, standarisedData);
    }

    /**
     * @brief Gets the @c N principal components' variance of the input data.
     *
     * @param values Vector of @c N elements wide where the PCA values will be stored.
     */
    inline void getExplainedVariance(DT values[N]) {
#pragma HLS INLINE
        for (unsigned i = 0; i < N; i++) {
#pragma HLS UNROLL
            values[i] = m_pcVals[0][i];
        }
    }

    /**
     * @brief Gets the @c N principal components of the input data.
     *
     * @param vectors Matrix of @c MAX_VARS x @c N elements wide where the principal components will be stored.
     */
    inline void getComponents(DT vectors[MAX_VARS][N]) {
#pragma HLS INLINE
        for (unsigned i = 0; i < m_noVariables; i++) {
            for (unsigned j = 0; j < N; j++) {
#pragma HLS UNROLL
                vectors[i][j] = m_pcVecs[0][i * N + j];
            }
        }
    }

    /**
     * @brief Calculate the loadings matrix of the fitted data.\n
     *
     * \f[ loadings = components^T * \sqrt{explainedVariance} \f]
     *
     * @param loadings Matrix of @c MAX_VARS x @c N elements where the PCA loadings will be stored.
     */
    void getLoadingsMatrix(DT loadings[MAX_VARS][N]) {
        // loadings = pcVecs * sqrt(pcVals.T)
        DT sqrtVals[N];
#pragma HLS ARRAY_PARTITION variable = sqrtVals complete

        for (unsigned i = 0; i < N; i++) {
#pragma HLS UNROLL
            sqrtVals[i] = hls::sqrt(m_pcVals[1][i]);
#ifdef _PCA_DEBUG_
            std::cout << "eig_vals[" << i << "] = " << m_pcVals[1][i] << ", sqrt(eig_vals[" << i
                      << "]) = " << sqrtVals[i] << std::endl;
#endif
        }

        for (unsigned i = 0; i < m_noVariables; i++) {
            for (unsigned j = 0; j < N; j++) {
#pragma HLS PIPELINE
                loadings[i][j] = sqrtVals[j] * m_pcVecs[1][i * N + j];
#ifdef _PCA_DEBUG_
                std::cout << "L(" << i << "," << j << ") = " << loadings[i][j] << std::endl;
#endif
            }
        }
    }
};

} // namespace fintech
} // namespace xf

#endif // _PCA_H_
