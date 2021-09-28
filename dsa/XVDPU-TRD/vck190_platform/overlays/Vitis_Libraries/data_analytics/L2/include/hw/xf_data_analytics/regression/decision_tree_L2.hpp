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
 * @brief interface of ML decisiontree train kernel.
 */

#ifndef _XF_DATA_ANALYTICS_L2_DECISIONTREE_TRAIN_HPP_
#define _XF_DATA_ANALYTICS_L2_DECISIONTREE_TRAIN_HPP_

#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>

// user should not change the macros
// size of samples buffer
#define DATASIZE (1 << 20)
// max nodes in each layer, matches uram depth
#define MAX_NODES_NUM 4095
#define TREE_SIZE (MAX_NODES_NUM)
#define INVALID_NODEID (MAX_NODES_NUM)

// user can change the macros
// kernel template for users
#define MAX_CAT_NUM_ 16
#define MAX_SPLITS_ 128
#define MAX_SPLITS_PARA_ 16
#define MAX_FEAS_ (64)
#define PARA_NUM_ 256 // for sw_emu, use larger hw_emu
//#define PARA_NUM_ 128 //use 128 when test benchmarks
#define MAX_TREE_DEPTH_ 12
typedef double DataType;
typedef ap_uint<64> TType;

/**
 * @brief Decision Tree Train enviroment context struct
 *
 * Note that the structure are used in both host and kernel
 *
 * As shown in the struct, there are 8 memebers.
 * 1. cretiea is the algorithm to compute node impurity and information gain
 *    0 : gini impurity(support)
 *    1 : entropy (not support)
 *    2 : Variance (not support)
 * 2. min_leaf_size A node stops splitting when it receives the number of training instances
 * 3. max_tree_depth is user input value in host, not transfer to device.
 * 4. (for extension) maxBins determines the max splits of each feature.
 *    in current version, splits are read from configuration.
 *    The parameter is for splits generation extension
 * 5. (for extension) min_samplecount_for_sample, for random forest and split generation sampling
 * 6. (for extension) sample_percent_if_sample, for random forest and split generation sampling
 * 7. (for extension) min_info_gain For entropy extension, a node stops splitting when its max gain <= this value
 */
struct Paras {
    unsigned cretiea; // 0:gain 1:gini not support variance:regression in current version
    unsigned max_tree_depth;
    unsigned min_leaf_size;
    float max_leaf_cat_per;
    float min_info_gain;
    // for large size sample,sampling
    int min_samplecount_for_sample;
    float sample_percent_if_sample;
    unsigned maxBins;
    float fea_fraction;
    ap_uint<64> seed;
};

template <typename MType>
struct Node {
    ap_uint<72> nodeInfo;
    MType threshold;
};

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
 * regValue is corresponding to the value of a leaf node
 * threshold is corresponding to value of featureId in a sample
 * If value of featureId in a sample < threshold, go to left child , otherwise go to right child(chl+1)
 */

template <typename MType>
struct NodeR {
    ap_uint<72> nodeInfo;
    MType regValue;
    MType threshold;
};

/**
 * @brief DecisionTree Regression Train Kernel
 * \rst
 * For detailed document, see :ref:`decisiontree_train_design`.
 * \endrst
 *
 * @param data input samples buffer,row-first tight storage
 * @param configs Configuration buffer
 * @param tree output decisiontree buffer
 *
 */
extern "C" void DecisionTree(ap_uint<512> data[DATASIZE], ap_uint<512> configs[30], ap_uint<512> tree[TREE_SIZE]);
#endif
