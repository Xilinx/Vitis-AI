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
 * @file bernoulli_distribution.hpp
 * @brief This file include the bernoulliCDF and bernoulliPMF
 *
 */

#ifndef __XF_FINTECH_BERNOULLI_DIST_HPP_
#define __XF_FINTECH_BERNOULLI_DIST_HPP_

#ifndef __SYNTHESIS__
#include "iostream"
#endif

namespace xf {

namespace fintech {

/**
 * @brief bernoulliPMF it implement a probability mass function for bernoulli distribution
 *
 * @tparam DT data type supported include float and double
 *
 * @param k k successes in 1 independent Bernoulli trial
 * @param p p is the probability of success of a single trial.
 * @return it belong to [0, 1] and also is a probability value.
 */
template <typename DT>
DT bernoulliPMF(int k, DT p) {
    if (p < 0.0 || p > 1.0) {
#ifndef __SYNTHESIS__
        std::cout << "[ERROR] p = " << p << ",p should belong to [0, 1]!\n";
#endif
        return -1.0;
    }
    if (k == 0)
        return 1 - p;
    else if (k == 1)
        return p;
    else
        return 0.0;
}

/**
 * @brief bernoulliCDF it implement a cumulative distribution function for bernoulli distribution
 *
 * @tparam DT data type supported include float and double
 *
 * @param k k successes in 1 independent Bernoulli trial
 * @param p p is the probability of success of a single trial.
 * @return it belong to [0, 1] and also is a cumulative probability value.
 */
template <typename DT>
DT bernoulliCDF(int k, DT p) {
    if (p < 0.0 || p > 1.0) {
#ifndef __SYNTHESIS__
        std::cout << "[ERROR] p = " << p << ",p should belong to [0, 1]!\n";
#endif
        return -1.0;
    }
    if (k == 0)
        return 1 - p;
    else if (k > 0)
        return 1.0;
    else
        return 0.0;
} // bernoulliCDF
}
}
#endif
