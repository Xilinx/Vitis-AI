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

#ifndef _XF_DATA_ANALYTICS_DECISIONTREE_TRAIN_HOST_HPP_
#define _XF_DATA_ANALYTICS_DECISIONTREE_TRAIN_HOST_HPP_
#include "xf_data_analytics/common/utils.hpp"

template <typename MType>
struct Node_H {
    bool isLeaf;
    ap_uint<8> leafCat;
    ap_uint<MAX_TREE_DEPTH_> chl;
    ap_uint<8> featureId;
    MType threshold; // for train, predict if a sample is in the node
};

void printTree(struct Node_H<DataType> nodes[MAX_NODES_NUM], int nodes_num) {
    for (int i = 0; i < nodes_num; i++) {
        std::cout << std::endl << "nodes" << i << std::endl;
        std::cout << "nodeId" << i << std::endl;
        std::cout << "featureId" << nodes[i].featureId << std::endl;
        std::cout << "threshod" << nodes[i].threshold << std::endl;
        if (nodes[i].isLeaf)
            std::cout << "leaf Node, the cat is : " << nodes[i].leafCat << std::endl;
        else {
            std::cout << "inner node,left chl is : " << nodes[i].chl << std::endl;
            std::cout << "inner node: right chl is : " << nodes[i].chl + 1 << std::endl;
        }
        // std::cout<<"nodeId"<<nodes[i].nodeId<<std::endl;
    }
}
void printTree_iter(struct Node_H<DataType> node, struct Node_H<DataType> nodes[MAX_NODES_NUM]) {
    if (node.isLeaf) {
        std::cout << "leaf Node, the cat is : " << node.leafCat << std::endl;
    } else {
        //    std::cout<<"nodeId:"<<node.nodeId<<std::endl;
        std::cout << "featureId:" << node.featureId << std::endl;
        std::cout << "threshod:" << node.threshold << std::endl;
        struct Node_H<DataType> nodel = nodes[node.chl];
        printTree_iter(nodel, nodes);

        struct Node_H<DataType> noder = nodes[node.chl + 1];
        printTree_iter(noder, nodes);
    }
}
ap_uint<MAX_TREE_DEPTH_> predict_host(DataType* onesample, struct Node_H<DataType> nodes[MAX_NODES_NUM]) {
    ap_uint<MAX_TREE_DEPTH_> node_id = 0;
    for (int i = 0; i < MAX_TREE_DEPTH_; i++) {
        if ((!nodes[node_id].isLeaf) && (nodes[node_id].chl < INVALID_NODEID)) // && nodes[node_id].chr != -1))
        {
            ap_uint<8> feature_id = nodes[node_id].featureId;
            DataType feature_val = onesample[feature_id];
            DataType threshold = nodes[node_id].threshold; // splits[nodes[node_id].splitId].threshold;
            if (feature_val <= threshold) {
                node_id = nodes[node_id].chl;
            } else {
                node_id = nodes[node_id].chl + 1;
            }
        }
    }
    return node_id;
}
int precisonAndRecall(DataType* testsamples,
                      ap_uint<30> samples_num,
                      ap_uint<8> features_num,
                      struct Node_H<DataType> nodes[MAX_NODES_NUM],
                      bool unorderedFeatures[MAX_FEAS_],
                      ap_uint<8> numClass) {
    int rightNum[MAX_CAT_NUM_] = {0};
    int testNum[MAX_CAT_NUM_] = {0};
    int predictNum[MAX_CAT_NUM_] = {0};
    DataType onesample[MAX_FEAS_ + 1];
    for (int i = 0; i < samples_num; i++) {
        int sample_start_position = (features_num + 1) * i;
        for (int j = 0; j < features_num + 1; j++) {
            onesample[j] = testsamples[sample_start_position + j];
        }
        int realClass = onesample[features_num];
        // int node_id = predict(onesample, nodes, unorderedFeatures);
        int node_id = predict_host(onesample, nodes);
        int testClass = nodes[node_id].leafCat;
        //        std::cout<<"sample "<<i<<"'s testClass is "<<testClass<<std::endl;
        //        std::cout<<"sample "<<i<<"'s realClass is "<<realClass<<std::endl;
        testNum[realClass]++;
        predictNum[testClass]++;
        if (testClass == realClass) rightNum[realClass]++;
    }
    for (int i = 0; i < numClass; i++) {
        std::cout << " class "
                  << " real count : " << testNum[i] << std::endl;
        std::cout << " class "
                  << " predict count : " << predictNum[i] << std::endl;
    }
    std::cout << "class     recall     precision" << std::endl;
    for (int i = 0; i < numClass; i++) {
        float recall = (float)rightNum[i] / testNum[i];
        if (i == 0 && fabs(recall - 0.87234) > 0.00001)
            return 1;
        else if (i == 1 && fabs(recall - 0.83019) > 0.00001)
            return 1;

        float precision = (float)rightNum[i] / predictNum[i];
        if (i == 0 && fabs(precision - 0.82) > 0.00001)
            return 1;
        else if (i == 1 && fabs(precision - 0.88) > 0.00001)
            return 1;
        std::cout << i << "   " << recall << "   " << precision << std::endl;
    }
    return 0;
}

// generate condig including:
// row1:numClass,features_num,samples_num
// row2:numsplits
// row3:splits
// row4
// row5
// row6
// row7:paras
//    max_leaf_cat_per(64),min_info_gain(64),min_leaf_size(16),max_tree_depth(16),maxBins(16),cretiea(2)
template <typename MType, unsigned WD>
void GenConfig(ap_uint<30> samples_num,
               ap_uint<8> features_num,
               ap_uint<8> numClass,
               Paras paras,
               ap_uint<8> numSplits[64],
               MType splits[128],
               ap_uint<512>* configs) {
    const unsigned elems_per_line = 512 / WD;
    int splitsnum = 0;
    for (int i = 0; i < features_num; i++) {
        splitsnum += numSplits[i];
    }
    configs[0].range(31, 0) = samples_num;
    configs[0].range(63, 32) = splitsnum;
    configs[0].range(95, 64) = features_num;
    configs[0].range(127, 96) = numClass;

    configs[1].range(31, 0) = paras.cretiea;
    configs[1].range(63, 32) = paras.maxBins;
    configs[1].range(95, 64) = paras.max_tree_depth;
    configs[1].range(127, 96) = paras.min_leaf_size;

    f_cast<float> feature_val_;
    feature_val_.f = paras.min_info_gain;
    configs[1].range(159, 128) = feature_val_.i;
    feature_val_.f = paras.max_leaf_cat_per;
    configs[1].range(191, 160) = feature_val_.i;

    int r = 1;
    for (int i = 0; i < features_num; i++) {
        int i_r = i & 0x3f;
        if (i_r == 0) r++;
        int index = i_r * 8;
        configs[r].range(index + 8 - 1, index) = numSplits[i];
    }
    r++;
    int c = 0;
    f_cast<MType> cc;
    for (int i = 0; i < splitsnum; i++) {
        cc.f = splits[i];
        configs[r].range(c * WD + WD - 1, c * WD) = cc.i;
        c++;
        if (c == elems_per_line) {
            r++;
            c = 0;
        }
    }
}
template <typename MType, unsigned WD>
void GenData(MType* samples, int total, ap_uint<512>* data) {
    int elem_per_line = 512 / WD;
    int r = 0;
    int c = 0;
    f_cast<MType> cc;
    for (int i = 0; i < total; i++) {
        r = i / elem_per_line + 1;
        c = i % elem_per_line;
        cc.f = samples[i];
        data[r].range(c * WD + WD - 1, c * WD) = cc.i;
    }
    data[0].range(31, 0) = r; // row count of data (from row 1) that is axiLen
    data[0].range(63, 32) = elem_per_line;
    data[0].range(95, 64) = total;
    data[0].range(127, 96) = WD;
    data[0].range(191, 128) = total;
    std::cout << "axiLen:" << r << std::endl;
    std::cout << "elem_per_line:" << elem_per_line << std::endl;
    std::cout << "total:" << total << std::endl;
    std::cout << "WD:" << WD << std::endl;
}

template <typename MType, unsigned WD>
void DataPreocess(MType* samples,
                  ap_uint<8> numCategories[MAX_FEAS_],
                  Paras paras,
                  std::string& in_dir,
                  ap_uint<30> samples_num,
                  ap_uint<8> features_num,
                  ap_uint<8> numClass,
                  ap_uint<512>* configs,
                  ap_uint<512>* data) {
    std::ifstream fin_config(in_dir + "/config.txt");
    std::string line;
    int read_int[128];
    ap_uint<8> numSplits[128] = {0};
    for (int i = 0; i < 128; i++) {
        read_int[i] = 129;
        numSplits[i] = 0;
    }
    double splits[128];
    int index = 0;
    while (getline(fin_config, line)) {
        std::istringstream sin(line);
        std::cout << line << std::endl;
        std::string attr_val;
        bool trans = true;
        while (getline(sin, attr_val, ',')) {
            if (trans) {
                read_int[index] = std::atoi(attr_val.c_str());
                trans = false;
                // std::cout<<"read int"<<read_int[index]<<std::endl;
            } else {
                splits[index] = std::atof(attr_val.c_str());
                // printf("%lf\n",splits[index] );
                trans = true;
                index++;
            }
        }
    }
    //#enddebug
    for (int i = 0; i < 128; i++) {
        int index = read_int[i];
        if (index < features_num) {
            numSplits[index]++;
        }
    }
    // debug
    printf("debug in host\n");
    printf("numSplits:");
    for (int i = 0; i < features_num; i++) {
        printf("%d,", numSplits[i].to_int());
    }
    printf("\n");
    index = 0;
    for (int i = 0; i < features_num; i++) {
        for (int j = 0; j < numSplits[i]; j++) {
            printf("%d,%lf\n", i, splits[index]);
            index++;
        }
    }
    GenConfig<MType, WD>(samples_num, features_num, numClass, paras, numSplits, splits, configs);
    int total = samples_num * (features_num + 1);
    GenData<MType, WD>(samples, total, data);
}
const unsigned treesize = ((MAX_NODES_NUM >> 1) + 1);
template <typename MType, unsigned WD>
void GetTreeFromBits(struct Node_H<MType> (&nodes)[MAX_NODES_NUM], ap_uint<512> tree[treesize], int& nodes_num) {
    nodes_num = tree[0].range(31, 0);
    f_cast<MType> ff;
    for (int i = 0; i < nodes_num; i++) {
        unsigned r = i >> 1;
        int offset = ((i & 0x01) == 0) ? 0 : 256;
        ap_uint<72> tmp = tree[r + 1].range(71 + offset, offset);
        nodes[i].isLeaf = tmp.range(0, 0);
        nodes[i].leafCat = tmp.range(15, 8);
        nodes[i].featureId = tmp.range(31, 16);
        nodes[i].chl = tmp.range(71, 32);
        ff.i = tree[r + 1].range(offset + 255, offset + 192);
        nodes[i].threshold = ff.f;
    }
}
#endif
