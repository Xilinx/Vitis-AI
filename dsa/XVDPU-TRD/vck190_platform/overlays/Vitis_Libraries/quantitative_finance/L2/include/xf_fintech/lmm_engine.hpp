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
 * @file lmm_engine.hpp
 * @brief This file contains the implementation of the LIBOR market model path generation.
 */
#ifndef _LMM_ENGINE_H_
#define _LMM_ENGINE_H_

#include "xf_fintech/pca.hpp"
#include "xf_fintech/rng.hpp"
#include "xf_fintech/mc_simulation.hpp"
#include "xf_fintech/rng_sequence.hpp"
#include "xf_fintech/path_generator.hpp"
#include "xf_fintech/path_pricer.hpp"
#include "xf_fintech/lmm.hpp"

namespace xf {
namespace fintech {

/**
 * @brief Class that will perform a dimensionality reduction to @c N dimensions of a correlation matrix to be used by
 * the multi-factor LIBOR Market Model framework.
 *
 * @tparam DT The datatype of the correlation matrix elements.
 * @tparam MAX_TENORS Maximum number of synthetisable tenors.
 * @tparam N Number of dimensions to be kept from the original correlation matrix.
 * @tparam NCU Parallelisation factor for PCA implementation.
 */
template <typename DT, unsigned MAX_TENORS, unsigned N, unsigned NCU = 1>
class lmmReducedFactorCorrelationEngine {
    const unsigned m_noTenors;

    /**
     * @brief Matrix multiply by its transpose. The transposition can be embedded if each element of the
     * output matrix corresponds to the dot product of row_i and row_j
     */
    template <unsigned R, unsigned C>
    static void mmult_transp(unsigned r, unsigned c, DT matIn[R][C], DT matOut[R][R]) {
    mmult_transp_row_loop:
        for (unsigned i = 0; i < r; i++) {
        mmult_transp_col_loop:
            for (unsigned j = 0; j < r; j++) {
                // Accumulate the product of row_i by row_j
                DT accum = 0.0;
            mmult_transp_accum_loop:
                for (unsigned k = 0; k < c; k++) {
                    accum += matIn[i][k] * matIn[j][k];
                }
                matOut[i][j] = accum;
            }
        }
    }

   public:
    /* Correlations operate on 'tenors + 1' elements. */
    lmmReducedFactorCorrelationEngine(unsigned noTenors) : m_noTenors(noTenors + 1) {
#pragma HLS inline
    }

    /**
     * @brief takes the input from the @c rho stream and performs a dimensionality reduction to just @c N factors.
     *
     * @param rho Correlation matrix stream
     * @param rhoReduced Projection of @c rho matrix into @c N dimensions. This will be an input to the MC simulation.
     * @param eta Pseudo square root of the projected data. This will be an input to the MC simulation.
     */
    void reduceDimensionality(hls::stream<DT>& rho, DT rhoReduced[MAX_TENORS][MAX_TENORS], DT eta[MAX_TENORS][N]) {
#pragma HLS DATAFLOW
        /*
         * Apply PCA with the correlation method to N factors and get the loadings matrix
         */
        DT loadings[MAX_TENORS][N], loadingsStd[MAX_TENORS][N], loadingsEta[MAX_TENORS][N];
#pragma HLS ARRAY_PARTITION variable = loadings dim = 2 complete
#pragma HLS ARRAY_PARTITION variable = loadingsStd dim = 2 complete
#pragma HLS ARRAY_PARTITION variable = loadingsEta dim = 2 complete

        static DT rhoMat[MAX_TENORS][MAX_TENORS];
        for (unsigned i = 0; i < m_noTenors; i++) {
            for (unsigned j = 0; j < m_noTenors; j++) {
#pragma HLS PIPELINE
                rhoMat[i][j] = rho.read();
            }
        }

        xf::fintech::PCA<DT, N, NCU, MAX_TENORS, MAX_TENORS, xf::fintech::pcaImplementationMethod::Correlation> pca(
            m_noTenors, m_noTenors, rhoMat);
        pca.getLoadingsMatrix(loadings);

        // Take 2 copies of the loadings
        for (unsigned i = 0; i < m_noTenors; i++) {
            for (unsigned j = 0; j < N; j++) {
#pragma HLS PIPELINE
                loadingsStd[i][j] = loadings[i][j];
                loadingsEta[i][j] = loadings[i][j];
            }
        }

        /*
         * Calculate std_dev vector.
         */
        DT stdDevs[MAX_TENORS];
        for (unsigned i = 0; i < m_noTenors; i++) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min = MAX_TENORS max = MAX_TENORS
            DT accum = 0.0;
            for (unsigned j = 0; j < N; j++) {
                accum += loadingsStd[i][j] * loadingsStd[i][j];
            }
            stdDevs[i] = hls::sqrt(accum);
        }

        /*
         * Calculate eta (normalised loadings matrix)
         */
        DT etaCalc[MAX_TENORS][N];
#pragma HLS ARRAY_PARTITION variable = etaCalc dim = 2
        for (unsigned i = 0; i < m_noTenors; i++) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min = MAX_TENORS max = MAX_TENORS
            for (unsigned j = 0; j < N; j++) {
#pragma HLS UNROLL
                const DT etaEl = loadingsEta[i][j] / stdDevs[i];
                eta[i][j] = etaEl;
                etaCalc[i][j] = etaEl;
            }
        }

        /*
         * Calculate rho reduced (projected data normalised to N dimensions)
         */
        mmult_transp<MAX_TENORS, N>(m_noTenors, N, etaCalc, rhoReduced);
    }
};

/**
 * @brief Prepares and runs a Monte-Carlo simulation and pricing for the LIBOR Market Model framework
 * from a given correlation and volatility matrix.
 *
 * @tparam DT The data type of the internal simulation.
 * @tparam PT The class name for the LMM pricer.
 * @tparam NF Number of factors to use in the internal LMM Monte-Carlo simulation.
 * @tparam UN Unroll number for path generators and pricers. It will determine the parallelism level of the simulation.
 * @tparam PCA_NCU Unroll number for the dimensionality reduction of the correlation matrix stage.
 *
 * @param noTenors Number of tenors to simulate. It must be <= @c MAX_TENORS
 * @param noPaths Number of MonteCarlo paths to generate. It will determine the accuracy of the final price.
 * @param rho Stream with generated correlation matrix between tenors.
 * @param presentFc Current LIBOR rates for this tenor structure.
 * @param sigma Stream with lower triangular calibrated volatilities matrix.
 * @param pricer UN instances of the selected path pricer. Must be of @c PT class and implement the correct MC path
 * pricer method interface.
 * @param seeds Seeds for the RNGs in the simulation. There are @c UN RNGs in total.
 * @param outputPrice Calculated LMM MonteCarlo price with the selected pricer.
 */
template <typename DT, typename PT, unsigned MAX_TENORS, unsigned NF, unsigned UN = 5, unsigned PCA_NCU = 2>
void lmmEngine(unsigned noTenors,
               unsigned noPaths,
               hls::stream<DT>& rho,
               DT presentFc[MAX_TENORS],
               hls::stream<DT>& sigma,
               PT pricer[UN][1],
               ap_uint<32> seeds[UN],
               DT* outputPrice) {
#ifndef __SYNTHESIS__
    assert(noTenors <= MAX_TENORS && "Number of tenors must be less than MAX_TENORS");
    assert(noTenors > 2 && "Number of tenors must be at least 3");
#endif
#pragma HLS DATAFLOW

    const unsigned steps = noTenors - 1;
    const static unsigned VN = 1;

    // Aliases
    typedef xf::fintech::MT19937BoxMullerNormalRng RNG;
    typedef xf::fintech::internal::lmmPathGenerator<DT, NF, MAX_TENORS> PATH;
    typedef xf::fintech::internal::RNGSequence_1_N<DT, RNG, NF> RNG_SEQ;

    DT rhoReduced[MAX_TENORS + 1][MAX_TENORS + 1], eta[MAX_TENORS + 1][NF];
#pragma HLS ARRAY_PARTITION variable = rhoReduced dim = 2 complete
#pragma HLS ARRAY_PARTITION variable = eta dim = 2 complete

    // Correlations work on MAX_TENORS + 1 sized matrices
    lmmReducedFactorCorrelationEngine<DT, MAX_TENORS + 1, NF, PCA_NCU> corrEngine(noTenors);
    corrEngine.reduceDimensionality(rho, rhoReduced, eta);

    // LMM Data instance
    xf::fintech::lmmModelData<DT, NF, MAX_TENORS> lmmData[UN];
#pragma HLS ARRAY_PARTITION variable = lmmData complete

    // Path generator instance
    PATH pathGenInst[UN][1];
#pragma HLS ARRAY_PARTITION variable = pathGenInst complete
    // RNG sequence instance
    RNG_SEQ rngSeqInst[UN][1];
#pragma HLS ARRAY_PARTITION variable = rngSeqInst complete

    hls::stream<DT> sigmaDup[UN];
#pragma HLS ARRAY_PARTITION variable = sigmaDup complete
    // Duplicate sigma streams
    for (unsigned i = 0; i < steps; i++) {
        for (unsigned j = i; j < steps; j++) {
#pragma HLS PIPELINE II = 1
            DT sigmaEl = sigma.read();
            for (unsigned k = 0; k < UN; k++) {
#pragma HLS UNROLL
                sigmaDup[k] << sigmaEl;
            }
        }
    }

    DT presentFcDup[UN][MAX_TENORS], rhoReducedDup[UN][MAX_TENORS + 1][MAX_TENORS + 1], etaDup[UN][MAX_TENORS + 1][NF];
#pragma HLS ARRAY_PARTITION variable = presentFcDup dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = rhoReducedDup dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = etaDup dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = etaDup dim = 3 complete

    for (unsigned i = 0; i < noTenors; i++) {
#pragma HLS PIPELINE
        DT dRate = presentFc[i];
        for (unsigned j = 0; j < UN; j++) {
#pragma HLS UNROLL
            presentFcDup[j][i] = dRate;
        }
    }

    for (unsigned i = 0; i < noTenors + 1; i++) {
        for (unsigned j = 0; j < noTenors + 1; j++) {
#pragma HLS PIPELINE
            DT dRho = rhoReduced[i][j];
            for (unsigned k = 0; k < UN; k++) {
#pragma HLS UNROLL
                rhoReducedDup[k][i][j] = dRho;
            }
        }
    }

    for (unsigned i = 0; i < noTenors + 1; i++) {
        for (unsigned j = 0; j < NF; j++) {
#pragma HLS PIPELINE
            DT dEta = eta[i][j];
            for (unsigned k = 0; k < UN; k++) {
#pragma HLS UNROLL
                etaDup[k][i][j] = dEta;
            }
        }
    }

    // Configure the path generator and path pricer
    for (unsigned i = 0; i < UN; i++) {
#pragma HLS UNROLL
        // Path generator
        lmmData[i].initialise(noTenors, sigmaDup[i], rhoReducedDup[i], etaDup[i], presentFcDup[i]);
        pathGenInst[i][0].init(lmmData[i], steps);

        // RNGSequence
        rngSeqInst[i][0].seed = seeds[i];
    }

    // Call MonteCarlo simulation
    *outputPrice = xf::fintech::mcSimulation<DT, RNG, PATH, PT, RNG_SEQ, UN, VN, 1>(steps, 0, noPaths, 0.0, pathGenInst,
                                                                                    pricer, rngSeqInst);
}

} // namespace fintech
} // namespace xf

#endif // _LMM_ENGINE_H_
