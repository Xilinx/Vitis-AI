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
 * @file decision_tree.hpp
 * @brief DecisionTree Node data struct in kernel, for both train and predict.
 *
 * This file is part of Vitis Data Analytics Library.
 */

#ifndef _XF_DATA_ANALYTICS_L1_CLASSIFICATION_DECISIONTREE_HPP_
#define _XF_DATA_ANALYTICS_L1_CLASSIFICATION_DECISIONTREE_HPP_
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>

#ifdef INCLUDE_IN_L1_HOST
#ifndef __SYNTHESIS__
#include <stdio.h>
#endif
#endif

// warning: user should not change the max_nodes_num
// uram max depth
#define MAX_NODES_NUM 1023
#define TREE_SIZE (MAX_NODES_NUM)
#define INVALID_NODEID (MAX_NODES_NUM)

/**
 * @brief DecisionTree Node data struct in kernel, for both train and predict.
 *
 * As shown below in the struct, this data structure includes two elements, nodeInfo and threshold
 * nodeInfo matches width of uram, includes 4 detailed members:
 * 1.isLeaf (nodeInfo.range(0,0)) judges if current node is a leaf
 * for leaf node:
 * 2.leafCat (nodeInfo.range(15,1)) determines the final category id
 * for Non-leaf node
 * 3.featureId (nodeInfo.range(31,16)) determines the next featureId should be used to make decision
 * 4.chl (nodeInfo.range(71,32)) : left child id  in a Decision Tree node list
 *
 * threshold is corresponding to value of featureId in a sample
 * If value of featureId in a sample < threshold, go to left child , otherwise go to right child(chl+1)
 */
template <typename MType>
struct Node {
    ap_uint<72> nodeInfo;
    MType threshold;
};

#endif
