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
 * @file lmm.hpp
 * @brief This file describes the LIBOR Market Model internal implementation
 */

#ifndef _LMM_H_
#define _LMM_H_

#include <ap_int.h>

namespace xf {
namespace fintech {

/**
 * @brief Class with LMM constants used in the LMM engine.
 */
struct lmmModelParams {
    // Evenly-spaced 6 month tenors
    constexpr static double tau = 0.5;
    // sqrt(tau)
    constexpr static double sqrtTau = 0.7071067811865476f;
};

/**
 * @brief Class with internal data used by the LMM path generation.
 */
template <typename DT, unsigned NF, unsigned MAX_TENORS>
struct lmmModelData : public lmmModelParams {
    constexpr static unsigned N_MATURITIES = MAX_TENORS - 1;
    constexpr static unsigned N_CORR = MAX_TENORS + 1;

   private:
    /**
     * @brief storage size of a triangular matrix given one dimension
     */
    static constexpr std::size_t triMatSize(unsigned T) { return (T * (T + 1)) / 2; }

    /**
     * @brief Calculates the 1-d index of a triangular matrix stored as an array
     */
    template <typename IT>
    static inline IT triMatIdx(IT i, IT j) {
#pragma HLS INLINE
        return j * N_MATURITIES - ((j - 1) * j * 0.5) + i - j;
    }

   public:
    /*
     * The path calculation will use a (N_MATURITIES) lower triangular matrix for rho and sigma.
     * We will store only the elements that we need for the path generation
     */
    DT m_rho[triMatSize(N_MATURITIES)];
    DT m_sigma[triMatSize(N_MATURITIES)];
    DT m_eta[N_CORR][NF];
    DT m_presentRate[MAX_TENORS];

    lmmModelData() {
#pragma HLS inline
#pragma HLS ARRAY_PARTITION variable = m_eta dim = 2 complete
    }

    inline void initialise(unsigned noTenors,
                           hls::stream<DT>& sigma,
                           DT rhoReduced[N_CORR][N_CORR],
                           DT eta[N_CORR][NF],
                           DT presentRate[MAX_TENORS]) {
#pragma HLS DATAFLOW

        for (unsigned i = 0; i < noTenors - 1; i++) {
            for (unsigned j = i; j < noTenors - 1; j++) {
#pragma HLS PIPELINE
                const unsigned idx = triMatIdx<unsigned>(j, i);
                sigma >> m_sigma[idx];
                m_rho[idx] = rhoReduced[j + 2][i + 1];
            }
        }

        for (unsigned i = 0; i < noTenors; i++) {
            for (unsigned j = 0; j < NF; j++) {
#pragma HLS UNROLL
                m_eta[i][j] = eta[i][j];
            }
        }

        for (unsigned i = 0; i < noTenors; i++) {
            m_presentRate[i] = presentRate[i];
        }
    }
};

/**
 * @brief Class with parametric correlation matrix generators for the LMM.
 * These functions will generate symmetric, positive and monotonically decreasing correlation matrices
 * based on user parameters.
 *
 * @tparam DT The data type for the correlation values.
 */
template <typename DT>
struct lmmCorrelationGenerator {
    /**
     * @brief Calculates 1-parametric correlation matrix, with evenly spaced tenors
     * @f[ rho_ij = exp(-beta * |T_i - T_j|) @f]
     */
    static void oneParametricCorrelation(DT beta, unsigned noTenors, hls::stream<DT>& rho) {
        for (int i = 0; i < noTenors + 1; i++) {
            for (int j = 0; j < noTenors + 1; j++) {
#pragma HLS PIPELINE
                rho << hls::exp(-beta * lmmModelParams::tau * hls::abs(i - j));
            }
        }
    }

    /**
     * @brief Calculates 2-parametric correlation matrix, with evenly spaced tenors
     * @f[ rho_ij = beta_0 + (1 - beta_0) * exp(-beta_1 * |T_i - T_j|) @f]
     */
    static void twoParametricCorrelation(DT beta[2], unsigned noTenors, hls::stream<DT>& rho) {
        for (int i = 0; i < noTenors + 1; i++) {
            for (int j = 0; j < noTenors + 1; j++) {
#pragma HLS PIPELINE
                rho << beta[0] + (1 - beta[0]) * hls::exp(-beta[1] * lmmModelParams::tau * hls::abs(i - j));
            }
        }
    }
};

/**
 * @brief Class with functions for volatility calibration and bootstrapping for the LMM.
 *
 * @tparam DT The data type for the volatility values.
 * @tparam MAX_TENORS Maximum number of synthetisable tenors.
 */
template <typename DT, unsigned MAX_TENORS>
struct lmmVolatilityGenerator {
    /**
     * @brief Implements time-homogeneous piecewise constant volatility functions.
     * The volatility functions will be calibrated to implied caplet volatilities.
     */
    static void piecewiseConstVolatility(unsigned noTenors, DT capletVola[MAX_TENORS], hls::stream<DT>& sigma) {
#ifndef __SYNTHESIS__
        assert(noTenors <= MAX_TENORS && "Number of tenors must be lower than MAX_TENORS parameter");
#endif
        // Bootstrapped volatilities
        static DT bsVolas[MAX_TENORS - 1];
        bsVolas[0] = capletVola[0];
        for (unsigned i = 1; i < noTenors - 1; i++) {
#pragma HLS PIPELINE
            bsVolas[i] = hls::sqrt((i + 1) * capletVola[i] * capletVola[i] - i * capletVola[i - 1] * capletVola[i - 1]);
        }

        // Output triangular matrix with constant volatilities per maturity time
        for (unsigned i = 0; i < noTenors - 1; i++) {
            for (unsigned j = i; j < noTenors - 1; j++) {
#pragma HLS PIPELINE
                sigma << bsVolas[j - i];
            }
        }
    }
};

} // namespace fintech
} // namespace xf

#endif // _LMM_H_
