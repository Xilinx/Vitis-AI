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
#include "xf_fintech/pop_mcmc.hpp"
#include "mcmc_kernel.hpp"
#ifndef __SYNTHESIS__
#include <iostream>
#endif
/**
* @brief Top level Kernel function.
*
*@tparam DT                 - Data type used in whole kernel (double by default)
*@tparam NCHAINS            - Number of chains
*@tparam NSAMPLES_MAX       - Maximum Number of chains for synthesis purpose
*@param[in] temp_inv        - Array of Inverted temperatures of the chain that density is generate for (1/Temp)
*@param[in] sigma           - Array of sigmas for Proposal generation for each chain
*@param[in] nSamples        - Number of samples to generate
*@param[out] sample_output  - Sample output
*/
extern "C" void mcmc_kernel(DT temp_inv[NCHAINS],
                            DT sigma[NCHAINS],
                            DT sample_output[NSAMPLES_MAX],
                            unsigned int nSamples) {
#pragma HLS INTERFACE m_axi port = sample_output bundle = gmem offset = slave
#pragma HLS INTERFACE m_axi port = sigma offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = temp_inv offset = slave bundle = gmem

#pragma HLS INTERFACE s_axilite port = temp_inv bundle = control
#pragma HLS INTERFACE s_axilite port = sigma bundle = control
#pragma HLS INTERFACE s_axilite port = sample_output bundle = control
#pragma HLS INTERFACE s_axilite port = nSamples bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    xf::fintech::McmcCore<DT, NCHAINS, NSAMPLES_MAX>(temp_inv, sigma, sample_output, nSamples);
}
