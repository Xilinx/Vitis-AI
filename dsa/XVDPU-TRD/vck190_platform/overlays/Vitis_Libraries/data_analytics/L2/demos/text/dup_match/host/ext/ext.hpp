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
#ifndef _EXT_HPP_
#define _EXT_HPP_

#include "rapidcsv/src/rapidcsv.h"
#include "distance.hpp"

struct GraphNode {
    uint32_t node1;
    uint32_t node2;
    double dist;
    uint32_t label;
};

#define fc_isnan(X) ((X) != (X))

#include "fastcluster/src/fastcluster.hpp"

void cluster_dist(GraphNode* Z, int* T, double cutoff, int n);

#endif
