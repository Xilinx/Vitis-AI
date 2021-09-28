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

#ifndef _HJM_ENGINE_HPP_
#define _HJM_ENGINE_HPP_

#include "xf_fintech/hjm_model.hpp"
#include "xf_fintech/path_generator.hpp"
#include "xf_fintech/path_pricer.hpp"
#include "xf_fintech/rng.hpp"
#include "xf_fintech/mc_simulation.hpp"
#include "xf_fintech/pca.hpp"
#include "xf_fintech/polyfit.hpp"

namespace xf {
namespace fintech {

namespace internal {

/**
 * @brief Class for calculating the inputs to the MonteCarlo simulation of Heath-Jarrow-Morton framework.
 * It deals with analysing historical data, calculating the risk neutral drift and the polynomial fitting of
 * the discrete volatility vectors.
 *
 * @tparam DT DataType for the internal calculations
 * @tparam MAX_TENORS Maximum synthetisable tenors supported by the framework.
 * @tparam MAX_CURVES Maximum number of days analyzable from historical data
 * @tparam NCU Number of compute units on the historical data matrix.
 */
template <typename DT, unsigned int MAX_TENORS, unsigned MAX_CURVES, unsigned NCU>
class HJMPcaEngine : public hjmModelParams {
    /**
     * Clones an input vector into C copies.
     */
    template <unsigned int N, unsigned int C>
    inline void split_vec(DT a[N], DT b[C][N]) {
#pragma HLS INLINE
        for (unsigned int i = 0; i < N; i++) {
#pragma HLS UNROLL
            for (unsigned int j = 0; j < C; j++) {
#pragma HLS UNROLL
                b[j][i] = a[i];
            }
        }
    }

    // Number of tenors to process in the framework
    const unsigned int m_noTenors;
    // Number of historical forward curves to process in the framework
    const unsigned int m_noObservations;

   public:
    HJMPcaEngine(const unsigned int noTenors, const unsigned int noObservations)
        : m_noTenors(noTenors), m_noObservations(noObservations) {
#pragma HLS inline
    }

    /**
     * @brief Risk Neutral drift calculation.
     * tau assumed to be 0.5 years
     *
     * RnD calculated as @f$ vol(t) * integral(vol(t)) @f$ for all volatilities and for @f$ t=1,...,n @f$
     */
    void riskNeutralDrift(DT vol_c1[PD[0]], DT vol_c2[PD[1]], DT vol_c3[PD[2]], DT rnd[MAX_TENORS]) {
#pragma HLS DATAFLOW
        DT c1_int[PD[0] + 1], c2_int[PD[1] + 1], c3_int[PD[2] + 1], p1_dup[2][PD[0]], p2_dup[2][PD[1]],
            p3_dup[2][PD[2]];
#pragma HLS ARRAY_PARTITION variable = c1_int complete
#pragma HLS ARRAY_PARTITION variable = c2_int complete
#pragma HLS ARRAY_PARTITION variable = c3_int complete
#pragma HLS ARRAY_PARTITION variable = p1_dup complete
#pragma HLS ARRAY_PARTITION variable = p2_dup complete
#pragma HLS ARRAY_PARTITION variable = p3_dup complete

        split_vec<PD[0], 2>(vol_c1, p1_dup);
        split_vec<PD[1], 2>(vol_c2, p2_dup);
        split_vec<PD[2], 2>(vol_c3, p3_dup);

        /*
         * RnD(i) = sum(j = 0 : 3, polyval(c_j, tau * i) * integral(0, i, polyval(c_j, tau * i)) )
         * c_j = [c0 c1 c2 ... cd]
         * cint_j = [c0/d c1/d-1 c2/d-2 ... cd/1 0]
         *
         * integral(0, i, polyval(c_j, i) ) == polyval(cint_j, i) - polyval(cint_j, 0) == polyval(cint_j, i)
         *
         * RnD(i) = sum(j = 0, 3,  polyval(c_j, tau * i) * polyval(cint_j, tau * i) )
         */
        xf::fintech::polyint<DT, PD[0]>(p1_dup[0], c1_int);
        xf::fintech::polyint<DT, PD[1]>(p2_dup[0], c2_int);
        xf::fintech::polyint<DT, PD[2]>(p3_dup[0], c3_int);

    HJM_Risk_Neutral_Drift_Loop:
        for (unsigned int t = 0; t < m_noTenors; t++) {
#pragma HLS PIPELINE
            const DT evalP = hjmModelParams::tau * t;
            const DT m1 =
                xf::fintech::polyval<DT, PD[0] + 1>(c1_int, evalP) * xf::fintech::polyval<DT, PD[0]>(p1_dup[1], evalP);
            const DT m2 =
                xf::fintech::polyval<DT, PD[1] + 1>(c2_int, evalP) * xf::fintech::polyval<DT, PD[1]>(p2_dup[1], evalP);
            const DT m3 =
                xf::fintech::polyval<DT, PD[2] + 1>(c3_int, evalP) * xf::fintech::polyval<DT, PD[2]>(p3_dup[1], evalP);
            rnd[t] = m1 + m2 + m3;
        }
    }

    /**
     * @brief Transforms each discrete volatility vector into a continuous least-squares fitting of a polynomial.
     * The chosen degrees for the polynomial is a model specific parameter selected for better convergence.
     * Each set of coefficients is also split into several copies for its use in different parts of the model.
     * The outputs is a set of coefficients, 'degree + 1' elements wide that represent the function
     * \f$ c_0 + c_1*x + c_2*x^2 + ... + c_n*x^n \f$
     *
     * @param discreteVols Matrix with discrete volatilities arranged in sets of N vectors.
     * @param vol1 Polynomial coefficients for the first volatility vector
     * @param vol2 Polynomial coefficients for the second volatility vector
     * @param vol3 Polynomial coefficients for the third volatility vector
     */
    void volatility_polyfit(DT discreteVols[MAX_TENORS][N],
                            DT vol1[VOL_POLYFIT_FANOUT][PD[0]],
                            DT vol2[VOL_POLYFIT_FANOUT][PD[1]],
                            DT vol3[VOL_POLYFIT_FANOUT][PD[2]]) {
#pragma HLS DATAFLOW
        /*
         * Split columns of loadings matrix stream into individual vectors
         */
        DT vola_discrete[N][MAX_TENORS];
#pragma HLS ARRAY_PARTITION variable = vola_discrete dim = 1 complete

        for (unsigned i = 0; i < m_noTenors; i++) {
#pragma HLS PIPELINE
            for (unsigned j = 0; j < N; j++) {
#pragma HLS UNROLL
                vola_discrete[j][i] = discreteVols[i][j];
            }
        }
        DT vol1_c[PD[0]], vol2_c[PD[1]], vol3_c[PD[2]];

        xf::fintech::polyfit<DT, PD[0], MAX_TENORS>(vola_discrete[0], m_noTenors, vol1_c);
        xf::fintech::polyfit<DT, PD[1], MAX_TENORS>(vola_discrete[1], m_noTenors, vol2_c);
        xf::fintech::polyfit<DT, PD[2], MAX_TENORS>(vola_discrete[2], m_noTenors, vol3_c);

        split_vec<PD[0], VOL_POLYFIT_FANOUT>(vol1_c, vol1);
        split_vec<PD[1], VOL_POLYFIT_FANOUT>(vol2_c, vol2);
        split_vec<PD[2], VOL_POLYFIT_FANOUT>(vol3_c, vol3);
    }

    /**
     * @brief matdiff function that computes the difference between rows in a matrix.
     *
     * @param input Input matrix data.
     * @param output Output containing the diff of the matrix data. It will contain 1 fewer row than the original.
     */
    template <unsigned R, unsigned C>
    void matdiffRows(DT input[R][C], DT output[R - 1][C], unsigned r, unsigned c) {
#ifndef __SYNTHESIS__
        assert((c <= C) && "Number of columns must be fewer than the max column value");
        assert((r <= R) && "Number of rows must be fewer than the max rows value");
#endif

        DT buffer[C];
    // Load the first row in the buffer
    MatDiffr_row_preload:
        for (unsigned i = 0; i < c; i++) {
#pragma HLS PIPELINE
            buffer[i] = input[0][i];
        }
    MatDiffr_main_loop:
        for (unsigned i = 0; i < r - 1; i++) {
            for (unsigned j = 0; j < c; j++) {
#pragma HLS PIPELINE
                output[i][j] = input[i + 1][j] - buffer[j];
                buffer[j] = input[i + 1][j];
            }
        }
    }

    /**
     * @brief Calculates the transpose of a matrix.
     */
    template <unsigned R, unsigned C>
    void mtransp(unsigned r, unsigned c, DT input[R][C], DT output[C][R]) {
        for (unsigned i = 0; i < r; i++) {
            for (unsigned j = 0; j < c; j++) {
#pragma HLS PIPELINE
                output[j][i] = input[i][j];
            }
        }
    }

    /**
     * @brief Extracts @c N discrete volatility vectors from a matrix of historical interest rates.
     * A Principal Component Analysis will be performed on the differences of the data, and a dimensionality reduction
     * will be performed to extract the @c noTenors datapoints for each vector.
     *
     * @param ratesIn Matrix with all the raw historical data, with size @c noTenors x @c noObservations
     * @param discreteVols Output vectors of the discrete volatilities arranged in sets of N. It will contain
     * @c noTenors elements.
     */
    void calculateDiscreteVolatility(DT ratesIn[MAX_CURVES][MAX_TENORS], DT discreteVols[MAX_TENORS][N]) {
        /*
         * Calculate delta of input data
         */
        DT deltaRates[MAX_CURVES - 1][MAX_TENORS], deltaTransp[MAX_TENORS][MAX_CURVES - 1];
        const unsigned int no_deltas = m_noObservations - 1;
        matdiffRows<MAX_CURVES, MAX_TENORS>(ratesIn, deltaRates, m_noObservations, m_noTenors);

        /*
         * Our variables live in the columns, so we need to transpose the diff matrix before sending it to PCA
         */
        mtransp<MAX_CURVES - 1, MAX_TENORS>(no_deltas, m_noTenors, deltaRates, deltaTransp);

        /*
         * Calculate PCA with N = 3 components
         */
        xf::fintech::PCA<DT, N, NCU, MAX_TENORS, MAX_CURVES - 1> pca(m_noTenors, no_deltas, deltaTransp);

        DT loadingsMat[MAX_TENORS][N];
#pragma HLS ARRAY_PARTITION variable = loadingsMat dim = 2 complete
        pca.getLoadingsMatrix(loadingsMat);

        /*
         * Annualise values: Scale volatility by 0.0252 (252 / 10000)
         * The factor is meant to represent a scaling in the eigen values of the PCA. Since we opererate in the loadings
         * matrix, which use the sqrt of the eig values, in order to apply the correct scaling, the scaling value needs
         * to be the sqrt of the expected scale.
         */
        constexpr DT annualisationFactor = 0.15874507866387544f; // = sqrt(0.0252f);
    HJM_PCA_Vola_Scale_loop:
        for (unsigned i = 0; i < m_noTenors; i++) {
#pragma HLS PIPELINE
            for (unsigned j = 0; j < N; j++) {
#pragma HLS UNROLL
                discreteVols[i][j] = loadingsMat[i][j] * annualisationFactor;
            }
        }
    }
};

} // namespace internal

/**
 * @brief Analyses raw historical data and calculates the input vectors for a Heath-Jarrow-Morton Monte-Carlo
 * simulation.
 *
 * @tparam DT The internal DataType of the calculations.
 * @tparam MAX_TENORS Maximum support synthetisable tenors.
 * @tparam MAX_CURVES Maximum synthetisable number of entries from the historial data.
 * @tparam NCU Number of parallel component units when processing the historical data matrix.
 *
 * @param ratesIn Matrix with the historical data.
 * @param noTenors Number of tenors in the simulation. Must be fewer than @c MAX_TENORS and be a multiple of @c NCU.
 * @param noObservations Number of forward curves in the historical data matrix.
 * @param riskNeutralDrift Output buffer for the Risk Neutral Drift vector for the MC simulation, @c tenors wide
 * @param volatilities Output vectors of the volatilities extracted from historical data. Consists on @c N vectors,
 * @c tenors wide.
 * @param presentForwardCurve Output vector with the forward curve at present date.
 */
template <typename DT, unsigned MAX_TENORS, unsigned MAX_CURVES, unsigned NCU = 1>
void hjmPcaEngine(DT ratesIn[MAX_CURVES * MAX_TENORS],
                  const unsigned int noTenors,
                  const unsigned int noObservations,
                  DT riskNeutralDrift[MAX_TENORS],
                  DT volatilities[hjmModelParams::N][MAX_TENORS],
                  DT presentForwardCurve[MAX_TENORS]) {
#ifndef __SYNTHESIS__
    assert(noObservations > 1 && "Number of observations must be larger than 1");
    assert((noTenors % NCU) == 0 && "Number of tenors must be a multiple of NCU");
#endif
#pragma HLS DATAFLOW
    internal::HJMPcaEngine<DT, MAX_TENORS, MAX_CURVES, NCU> hjmEngine(noTenors, noObservations);

    DT discreteVols[MAX_TENORS][hjmModelParams::N];
    DT ratesMat[MAX_CURVES][MAX_TENORS];
#pragma HLS STREAM variable = ratesMat
#pragma HLS STREAM variable = discreteVols

    // Load the matrix and calculate the present forward curve, last row of data as a %
    for (unsigned i = 0; i < noObservations; i++) {
        for (unsigned j = 0; j < noTenors; j++) {
            ratesMat[i][j] = ratesIn[i * noTenors + j];
            if (i == (noObservations - 1)) {
                // Last data row
                presentForwardCurve[j] = ratesIn[i * noTenors + j] * 0.01;
            }
        }
    }
    hjmEngine.calculateDiscreteVolatility(ratesMat, discreteVols);

    // We need one set of coefficients for drift calculation and another for volatility discretization
    DT vol1_c_split[hjmModelParams::VOL_POLYFIT_FANOUT][hjmModelParams::PD[0]],
        vol2_c_split[hjmModelParams::VOL_POLYFIT_FANOUT][hjmModelParams::PD[1]],
        vol3_c_split[hjmModelParams::VOL_POLYFIT_FANOUT][hjmModelParams::PD[2]];
#pragma HLS ARRAY_PARTITION variable = vol1_c_split complete
#pragma HLS ARRAY_PARTITION variable = vol2_c_split complete
#pragma HLS ARRAY_PARTITION variable = vol3_c_split complete
    hjmEngine.volatility_polyfit(discreteVols, vol1_c_split, vol2_c_split, vol3_c_split);

    /*
     * Calculate Risk Neutral Drift vector.
     */
    hjmEngine.riskNeutralDrift(vol1_c_split[0], vol2_c_split[0], vol3_c_split[0], riskNeutralDrift);

    /*
     * Calculate Volatility fitted to the coefficients
     */
    for (unsigned i = 0; i < noTenors; i++) {
#pragma HLS PIPELINE
        volatilities[0][i] = xf::fintech::polyval<DT, hjmModelParams::PD[0]>(vol1_c_split[1], (DT)i);
        volatilities[1][i] = xf::fintech::polyval<DT, hjmModelParams::PD[1]>(vol2_c_split[1], (DT)i);
        volatilities[2][i] = xf::fintech::polyval<DT, hjmModelParams::PD[2]>(vol3_c_split[1], (DT)i);
    }
}

/**
 * @brief Prepares and runs a Monte-Carlo simulation and pricing for the Heath-Jarrow-Morton framework
 *
 * @tparam DT The internal DataType in the simulation.
 * @tparam PT The class name for the HJM pricer.
 * @tparam MAX_TENORS The maximum number of supported tenors in the simulation.
 * @tparam UN The Unroll Number for the path generators and pricers. It will determine the level of parallelism of the
 * simulation.
 *
 * @param tenors Number of tenors to process. Must be <= @c MAX_TENORS.
 * @param simYears Number of years to simulate per path. Each path's IFR matrix is composed of @c simYears/dt rows.
 * @param noPaths Number of MonteCarlo paths to generate.
 * @param presentFc Present forward curve, determining the first row of every simulated path.
 * @param vol Volatility vectors for @c N factor model, @c tenors elements wide, describing the volatility per
 * tenor for each of the factors.
 * @param drift Risk Neutral Drift vector, @c tenors elements wide.
 * @param pricer @c UN instances of the selected path pricer. Must be of @c PT class and implement the correct MC path
 * pricer method interface.
 * @param seed Seeds for the RNGs in the simulation. There are @c N RNGs per path generator and @c UN path generators.
 * @param outputPrice Stream with the calculated HJM output price.
 */
template <typename DT, class PT, unsigned MAX_TENORS, unsigned UN = 1>
void hjmMcEngine(const unsigned tenors,
                 const float simYears,
                 const unsigned int noPaths,
                 DT presentFc[MAX_TENORS],
                 DT vol[hjmModelParams::N][MAX_TENORS],
                 DT drift[MAX_TENORS],
                 PT pricer[UN][1],
                 ap_uint<32> seed[UN][hjmModelParams::N],
                 hls::stream<DT>& outputPrice) {
#ifndef __SYNTHESIS__
    assert(tenors <= MAX_TENORS && "Number of tenors must be less than MAX_TENORS");
#endif

    // Aliases
    typedef xf::fintech::MT19937BoxMullerNormalRng RNG;
    typedef xf::fintech::internal::hjmPathGenerator<DT, MAX_TENORS> PATH;
    typedef xf::fintech::internal::RNGSequence_1_N<DT, RNG, hjmModelParams::N> RNG_SEQ;

    const DT simPoints = static_cast<DT>(simYears * hjmModelParams::dtInv);

    // HJM Model instance
    xf::fintech::hjmModelData<DT, MAX_TENORS> hjmImpl[UN];
#pragma HLS ARRAY_PARTITION variable = hjmImpl complete

    // Path generator instance
    PATH pathGenInst[UN][1];
#pragma HLS ARRAY_PARTITION variable = pathGenInst complete
    // RNG sequence instance
    RNG_SEQ rngSeqInst[UN][1];
#pragma HLS ARRAY_PARTITION variable = rngSeqInst complete
    // RNG Instance
    RNG rngInst[UN][hjmModelParams::N];
#pragma HLS ARRAY_PARTITION variable = rngInst complete

    // Configure the path generator and path pricer
    for (unsigned i = 0; i < UN; i++) {
#pragma HLS UNROLL
        // Path generator
        hjmImpl[i].initialise(tenors, vol[0], vol[1], vol[2], drift, presentFc);
        pathGenInst[i][0].init(hjmImpl[i]);

        // RNGSequence
        for (unsigned j = 0; j < hjmModelParams::N; j++) {
#pragma HLS UNROLL
            rngSeqInst[i][0].seed[j] = seed[i][j];
        }
        rngSeqInst[i][0].Init(rngInst[i]);
    }

    // Call Monte-Carlo simulation
    DT price = mcSimulation<DT, RNG, PATH, PT, RNG_SEQ, UN, hjmModelParams::N, 1>(simPoints, 0, noPaths, 0.0,
                                                                                  pathGenInst, pricer, rngSeqInst);
    outputPrice << price;
}

/**
 * @brief Prepares and runs a Monte-Carlo simulation and pricing for the Heath-Jarrow-Morton framework
 * from historical data. Combines @a hjmPcaEngine and @a hjmMcEngine into a single operation.
 *
 * @tparam DT The internal DataType in the simulation.
 * @tparam PT The class name for the HJM pricer.
 * @tparam MAX_TENORS The maximum number of supported tenors in the simulation.
 * @tparam MAX_CURVES Maximum synthetisable number of entries from the historial data.
 * @tparam PCA_NCU Number of parallel computing units when implementing the PCA engine.
 * @tparam MC_UN The Unroll Number for the path generators and pricers. It will determine the level of parallelism of
 * the simulation.
 *
 * @param tenors Number of tenors to process. Must be <= @c MAX_TENORS .
 * @param curves Number of curves in the historical data matrix.
 * @param simYears Number of years to simulate per path. Each path's IFR matrix is composed of @c simYears/dt rows.
 * @param noPaths Number of MonteCarlo paths to generate.
 * @param ratesIn Historical data matrix.
 * @param pricer UN instances of the selected path pricer. Must be of @c PT class and implement the correct MC path
 * pricer method interface.
 * @param seed Seeds for the RNGs in the simulation. There are @c N RNGs per path generator and @c UN path generators.
 * @param outputPrice Stream with the calculated HJM output price.
 */
template <typename DT, class PT, unsigned MAX_TENORS, unsigned MAX_CURVES, unsigned PCA_NCU = 1, unsigned MC_UN = 1>
void hjmEngine(const unsigned tenors,
               const unsigned curves,
               const float simYears,
               const unsigned noPaths,
               DT ratesIn[MAX_CURVES * MAX_TENORS],
               PT pricer[MC_UN][1],
               ap_uint<32> seeds[MC_UN][hjmModelParams::N],
               hls::stream<DT>& outPrice) {
#pragma HLS DATAFLOW
    DT presentFc[MAX_TENORS];
    DT riskNeutralDrift[MAX_TENORS];
    DT vols[hjmModelParams::N][MAX_TENORS];
#pragma HLS ARRAY_PARTITION variable = vols dim = 1 complete

    hjmPcaEngine<DT, MAX_TENORS, MAX_CURVES, PCA_NCU>(ratesIn, tenors, curves, riskNeutralDrift, vols, presentFc);

    hjmMcEngine<DT, PT, MAX_TENORS, MC_UN>(tenors, simYears, noPaths, presentFc, vols, riskNeutralDrift, pricer, seeds,
                                           outPrice);
}

} // namespace fintech
} // namespace xf

#endif // HJM_PCA_ENGINE_H_
