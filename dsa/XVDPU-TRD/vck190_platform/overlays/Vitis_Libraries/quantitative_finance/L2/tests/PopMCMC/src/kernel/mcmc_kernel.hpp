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
 *  @file mcmc_kernel.hpp
 *  @brief  Header file for kernel wrapper
 */

#ifndef _MCMC_KERNEL_H_
#define _MCMC_KERNEL_H_

#include "xf_fintech/rng.hpp"

/// @brief Specific implementation of this kernel
#define NCHAINS 10
#define NSAMPLES_MAX 5000
#define DT double

/**
* @brief Top level Kernel function.  \n
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
                            unsigned int nSamples);

#endif
