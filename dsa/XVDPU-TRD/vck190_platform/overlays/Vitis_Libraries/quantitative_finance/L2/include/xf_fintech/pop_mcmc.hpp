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
/**********
 * Copyright (c) 2019, Xilinx, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * **********/
/**
 *  @file pop_mcmc.hpp
 *  @brief  Implementation of Population Markov Chain Monte Carlo (MCMC)
 *
 *  $DateTime: 2019/07/24 12:00:00 $
 */

#ifndef _MCMC_CORE_
#define _MCMC_CORE_

#include "xf_fintech/rng.hpp"
#include "hls_math.h"
#include <hls_stream.h>

namespace xf {
namespace fintech {
namespace internal {
/**
* @brief Calculates target distribution density for a given sample and temperature.
* Calculated density is raised to power of temperature of target chain.
*
*@tparam DT data type used in whole function (double by default)
*@param[in] x - Sample to generate density for \n
*@param[in] temp_inv - Inverted temperature of the chain that density is generate for (1/Temp) \n
  *@return  Calculated density \n
  */
template <typename DT>
DT TargetDist(DT x, DT temp_inv) {
#pragma HLS inline
    DT val, result, result_log;
    DT gam = 4;
    val = gam * (x * x - 1) * (x * x - 1);
    result = hls::log(hls::exp(-val)) * temp_inv;
    return result;
}
/**
* @brief Calculates final transformation of Gaussian Sample.
*
*@tparam DT         Data type used in whole function (double by default)
*@param[in] in    - Sample from Uniform Distribution \n
*@param[in] mu    - Expected value for Normal Distribution  \n
*@param[in] sigma - Sigma for Proposal generation \n
*@return          - Generated Sample
*/
template <typename DT>
DT GaussTransform(DT in, DT mu, DT sigma) {
    DT result = in * sigma + mu;
    return result;
}

/**
* @brief Probability evaluation function. \n
* It Generates samples for all chains. Metropolis sampler is used in this function. \n
* Fully pipelined for chains. \n
* During Probability evaluation gauss sample for next sample is generated in parallel, \n
* this allows to save half of the time for probability evaluation. \n
* Part of the dataflow streaming region.
*
*@tparam DT data type used in whole function (double by default)
*@tparam NCHAINS Number of chains
*@param[in] chain_in    - Previous samples for each chains \n
*@param[in] gauss       - Gaussian sample proposal on [0:1] for current sample (1/Temp) \n
*@param[out] gauss_next - Gaussian sample proposal on [0:1] for next sample \n
*@param[out] chain_out  - Samples streaming output  \n
*@param[in] uniformRNG  - Pointer to Uniform RNG for Accept/Reject \n
*@param[in] temp_inv    - Array of Inverted temperatures of the chain that density is generate for (1/Temp) \n
*@param[in] sigma       - Array of sigmas for Proposal generation for each chain  \n
*/
template <typename DT, unsigned int NCHAINS>
void ProbEval(DT chain_in[NCHAINS],
              hls::stream<DT>& chain_out,
              DT gauss[NCHAINS],
              DT gauss_next[NCHAINS],
              xf::fintech::MT19937& uniformRNG,
              DT temp_inv[NCHAINS],
              DT sigma[NCHAINS]) {
#pragma HLS inline off
    DT xStar;
    static xf::fintech::MT19937 uniformRNG_2(71);
    DT alpha;
    DT u;
    DT sample_buff;

PROB_EVALUATION_LOOP:
    for (int n = 0; n < NCHAINS; n++) {
#pragma HLS pipeline
        // CALCULATE THE ACCEPTANCE PROBABILITY
        xStar = GaussTransform<DT>(gauss[n], chain_in[n], sigma[n]);
        // alpha = TargetDist<DT>(xStar,temp_inv[n])/TargetDist<DT>(chain_in[n],temp_inv[n]);
        alpha = TargetDist<DT>(xStar, temp_inv[n]) - TargetDist<DT>(chain_in[n], temp_inv[n]);

        DT in = uniformRNG_2.next();
        gauss_next[n] = xf::fintech::inverseCumulativeNormalAcklam<DT>(in);
        // ACCEPT OR REJECT?
        u = uniformRNG.next();
        if (hls::log(u) < alpha) {
            sample_buff = xStar;
        } else {
            sample_buff = chain_in[n];
        }
        chain_out << sample_buff;
    } // end of PROB_EVALUATION_LOOP
}
/**
* @brief Chain Exchange function. \n
* Calculates exchange ratio and exchanges chains if needed. \n
* Fully pipelined for chains. \n
* Part of the dataflow streaming region.
*
*@tparam DT data type used in whole function (double by default)
*@tparam NCHAINS Number of chains
*@param[in]  chain_in    - Current sample streaming input interface \n
*@param[in]  chain_out   - Array of generated samples for each chain samples.  \n
*@param[in]  temp_inv    - Array of Inverted temperatures of the chain that density is generate for (1/Temp) \n
*/
template <typename DT, unsigned int NCHAINS>
void ChainExchange(hls::stream<DT>& chain_in, DT chain_out[NCHAINS], DT temp_inv[NCHAINS]) {
    DT chain_buff[NCHAINS];
    static bool even;
    bool last_read = 0;
    DT u;
    DT alpha_ex;
    static xf::fintech::MT19937 uniformRNG_ex(71);
    // Do first read if even pairs are exchanging because first chain is not reached then.
    if (even) {
        chain_buff[0] = chain_in.read();
    }

EXCHANGE_LOOP:
    for (int n = 1 + even; n < NCHAINS; n = n + 2) { // exchange loop
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 5
#pragma HLS pipeline II = 2
        chain_buff[n - 1] = chain_in.read();
        chain_buff[n] = chain_in.read();
        // alpha_ex =
        // TargetDist<DT>(chain_buff[n],temp_inv[n-1])*TargetDist<DT>(chain_buff[n-1],temp_inv[n])/(TargetDist<DT>(chain_buff[n],temp_inv[n])*TargetDist<DT>(chain_buff[n-1],temp_inv[n-1]));
        alpha_ex = TargetDist<DT>(chain_buff[n], temp_inv[n - 1]) + TargetDist<DT>(chain_buff[n - 1], temp_inv[n]) -
                   (TargetDist<DT>(chain_buff[n], temp_inv[n]) + TargetDist<DT>(chain_buff[n - 1], temp_inv[n - 1]));
        u = uniformRNG_ex.next();
        if (hls::log(u) < alpha_ex) {
            chain_out[n - 1] = chain_buff[n];
            chain_out[n] = chain_buff[n - 1];
        } else {
            chain_out[n - 1] = chain_buff[n - 1];
            chain_out[n] = chain_buff[n];
        }
        if (NCHAINS - n == 2) {
            last_read = 1;
        }
    } // end of EXCHANGE_LOOP
    // Do last read from stream if last iteration hasn't reached last chain.
    if (last_read) {
        chain_out[NCHAINS - 1] = chain_in.read();
    }
    // Echagning odd or even pairs of chains
    even = !even;
}
/**
* @brief Wraping function for dataflow region. /n
*
*@tparam DT data type used in whole function (double by default)
*@tparam NCHAINS Number of chains
*@param[in] chain       - Previous samples for each chains \n
*@param[in] gauss       - Gaussian sample proposal on [0:1] for current sample (1/Temp) \n
*@param[out] gauss_next - Gaussian sample proposal on [0:1] for next sample \n
*@param[out] chain_out  - Array of generated samples  \n
*@param[in] uniformRNG  - Pointer to Uniform RNG for Accept/Reject \n
*@param[in] temp_inv    - Array of Inverted temperatures of the chain that density is generate for (1/Temp) \n
*@param[in] sigma       - Array of sigmas for Proposal generation for each chain  \n
*/
template <typename DT, unsigned int NCHAINS>
void SampleEval(DT chain[NCHAINS],
                DT chain_out[NCHAINS],
                DT gauss[NCHAINS],
                DT gauss_next[NCHAINS],
                xf::fintech::MT19937& uniformRNG,
                DT temp_inv[NCHAINS],
                DT sigma[NCHAINS]) {
    hls::stream<DT> chain_stream("chain_stream");
#pragma HLS stream variable = chain_stream depth = 4
#pragma HLS DATAFLOW
    ProbEval<DT, NCHAINS>(chain, chain_stream, gauss, gauss_next, uniformRNG, temp_inv, sigma);
    ChainExchange<DT, NCHAINS>(chain_stream, chain_out, temp_inv);
}

} // internal

/**
* @brief Top level Kernel function. Consists of INIT_LOOP and main sample loop: SAMPLES_LOOP \n
* \n
* Generates sample from target distribution function.\n
* Uses multiple Markov Chains to allow drawing samples from multi mode target distribution functions. \n
* Proposal is generated ussing Normal Distribution  \n
*@tparam DT             - Data type used in whole function (double by default)
*@tparam NCHAINS        - Number of chains
*@tparam NSAMPLES_MAX   - Maximum Number of chains for synthesis purpose
*@param[in] temp_inv    - Array of Inverted temperatures of the chain that density is generate for (1/Temp) \n
*@param[in] sigma       - Array of sigmas for Proposal generation for each chain  \n
*@param[in] nSamples    - Number of samples to generate  \n
*@param[out] x          - Sample output  \n
*/
template <typename DT, unsigned int NCHAINS, unsigned int NSAMPLES_MAX>
void McmcCore(DT temp_inv[NCHAINS], DT sigma[NCHAINS], DT x[NSAMPLES_MAX], unsigned int nSamples) {
    DT chain[NCHAINS];
#pragma HLS array_partition variable = chain complete
    DT chain_out[NCHAINS];
#pragma HLS array_partition variable = chain_out complete
    DT gauss[NCHAINS];
#pragma HLS array_partition variable = gauss complete
    DT gauss_next[NCHAINS];
#pragma HLS array_partition variable = gauss_next complete
    DT temp_inv_buff[NCHAINS];
#pragma HLS array_partition variable = temp_inv_buff complete
    DT sigma_buff[NCHAINS];
    xf::fintech::MT19937 uniformRNG(42);

INIT_LOOP:
    for (int n = 0; n < NCHAINS; n++) {
#pragma HLS pipeline
        chain[n] = 0.6;
        chain_out[n] = 0.6;
        DT in = uniformRNG.next();
        gauss[n] = xf::fintech::inverseCumulativeNormalAcklam<DT>(in);
        temp_inv_buff[n] = temp_inv[n];
        sigma_buff[n] = sigma[n];
    }

SAMPLES_LOOP:
    for (int t = 1; t < nSamples; t++) {
#pragma HLS LOOP_TRIPCOUNT min = 500 max = 5000
    SAMPLE_SWAP_LOOP:
        // Curent samples becoming previous samples in next step
        for (int n = 0; n < NCHAINS; n++) {
            chain[n] = chain_out[n];
        }
        internal::SampleEval<DT, NCHAINS>(chain, chain_out, gauss, gauss_next, uniformRNG, temp_inv_buff, sigma_buff);
        // Output is only first chain with temp=1
        x[t] = chain_out[0];
        // Replacing current gaussian proposal with the next one
        for (int n = 0; n < NCHAINS; n++) {
            gauss[n] = gauss_next[n];
        }
    } // end for sample loop
}

} // namespace solver
} // namespace xf

#endif
