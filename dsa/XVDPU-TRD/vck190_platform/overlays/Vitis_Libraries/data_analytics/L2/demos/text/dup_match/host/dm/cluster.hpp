/*
 * Copyright 2021 Xilinx, Inc.
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
#ifndef _DUP_MATCH_CLUSTER_HPP_
#define _DUP_MATCH_CLUSTER_HPP_

#include "common.hpp"
#include "ext/ext.hpp"

namespace dup_match {
namespace internal {

class Cluster {
   private:
    void pair(std::vector<std::string>& blockKey, std::set<std::pair<uint32_t, uint32_t> >& pairId);
    double predictProba(std::array<double, 6>& scores);

    const double bias_ = 16.197957182332377;
    const std::array<double, 6> weights_ = {-4.30704455, -5.53147223, -0.18319828,
                                            -4.42798055, -0.18319828, 1.94724175};

    // parameter
    std::vector<std::pair<uint32_t, uint32_t> > pair_id_;
    std::vector<double> scores_;
    std::vector<GraphNode> graph_;

   public:
    void pairWrap(std::array<std::vector<std::string>, 3>& blockKey);
    void score(std::vector<std::vector<std::string> >& vec_fields);
    void connectedComponents();
    void hierCluster(std::vector<std::pair<uint32_t, double> >& membership);
};

} // internal
} // dup_match
#endif
