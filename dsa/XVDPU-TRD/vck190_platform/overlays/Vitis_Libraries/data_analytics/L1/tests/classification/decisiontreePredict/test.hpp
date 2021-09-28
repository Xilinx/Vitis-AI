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

#ifndef _XF_DATA_ANALYTICS_DECISIONTREE_PREDICT_L1_HOST_HPP_
#define _XF_DATA_ANALYTICS_DECISIONTREE_PREDICT_L1_HOST_HPP_

template <typename MType, unsigned MAX_TREE_DEPTH, unsigned MAX_CAT_BITS>
struct Node_H {
    bool isLeaf;
    ap_uint<MAX_CAT_BITS> leafCat;
    ap_uint<MAX_TREE_DEPTH> chl;
    ap_uint<8> featureId;
    MType threshold; // for train, predict if a sample is in the node
};

struct Paras {
    // decision tree stop-building condition
    unsigned maxBins;
    unsigned cretiea;        // 0:gain 1:gini not support variance:regression in current version
    unsigned max_tree_depth; // in spark
    unsigned min_leaf_size;
    float max_leaf_cat_per;
    float min_info_gain;

    // for large size sample,not used
    int min_samplecount_for_sample;
    float sample_percent_if_sample;
};

int genTreeFile(ap_uint<512>* tree, const std::string& fn, size_t n, size_t sizeT) {
    FILE* f = fopen(fn.c_str(), "wb");
    if (!f) {
        std::cerr << "ERROR: " << fn << " cannot be opened for binary write." << std::endl;
    }
    size_t cnt = fwrite((void*)tree, sizeT, n, f);
    fclose(f);
    if (cnt != n) {
        std::cerr << "ERROR: " << cnt << " entries write to " << fn << ", " << n << " entries required." << std::endl;
        return -1;
    }
    std::cout << "in genTreeFile" << std::endl;
    return 0;
}
int loadTreeFrmFile(ap_uint<512>* tree, const std::string& fn, size_t n, size_t sizeT) {
    FILE* f = fopen(fn.c_str(), "rb");
    if (!f) {
        std::cerr << "ERROR: " << fn << " cannot be opened for binary read." << std::endl;
    }
    // size_t cnt = fread(data, sizeof(T), n, f);
    size_t cnt = fread((void*)tree, sizeT, n, f);
    fclose(f);
    if (cnt != n) {
        std::cerr << "ERROR: " << cnt << " entries read from " << fn << ", " << n << " entries required." << std::endl;
        return -1;
    }
    return 0;
}
template <typename MType, unsigned MAX_TREE_DEPTH, unsigned MAX_CAT_BITS>
void printTree(struct Node_H<MType, MAX_TREE_DEPTH, MAX_CAT_BITS> nodes[MAX_NODES_NUM], int nodes_num) {
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

template <typename MType, unsigned MAX_TREE_DEPTH, unsigned MAX_CAT_BITS>
ap_uint<MAX_TREE_DEPTH> predict_host(DataType* onesample,
                                     struct Node_H<MType, MAX_TREE_DEPTH, MAX_CAT_BITS> nodes[MAX_NODES_NUM]) {
    ap_uint<MAX_TREE_DEPTH> node_id = 0;
    for (int i = 0; i < MAX_TREE_DEPTH; i++) {
        if ((!nodes[node_id].isLeaf) && (nodes[node_id].chl < INVALID_NODEID)) // && nodes[node_id].chr != -1))
        {
            int feature_id = nodes[node_id].featureId;
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
template <typename MType, unsigned MAX_TREE_DEPTH, unsigned MAX_CAT_BITS>
void precisonAndRecall(DataType* testsamples,
                       int samples_num,
                       int features_num,
                       struct Node_H<MType, MAX_TREE_DEPTH, MAX_CAT_BITS> nodes[MAX_NODES_NUM],
                       int numClass) {
    const int MAX_CAT_NUM = 1 << MAX_CAT_BITS;
    int rightNum[MAX_CAT_NUM] = {0};
    int testNum[MAX_CAT_NUM] = {0};
    int predictNum[MAX_CAT_NUM] = {0};
    DataType onesample[200];
    for (int i = 0; i < samples_num; i++) {
        int sample_start_position = (features_num + 1) * i;
        for (int j = 0; j < features_num + 1; j++) {
            onesample[j] = testsamples[sample_start_position + j];
        }
        int realClass = onesample[features_num];
        int node_id = predict_host<MType, MAX_TREE_DEPTH, MAX_CAT_BITS>(onesample, nodes);
        int testClass = nodes[node_id].leafCat;
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
        float precision = (float)rightNum[i] / predictNum[i];
        std::cout << i << "   " << recall << "   " << precision << std::endl;
    }
}

template <unsigned MAX_CAT_BITS>
void precisonAndRecall1(
    DataType* testsamples, ap_uint<512>* predictions, int samples_num, int features_num, int numClass) {
    const int MAX_CAT_NUM = 1 << MAX_CAT_BITS;
    int rightNum[MAX_CAT_NUM] = {0};
    int testNum[MAX_CAT_NUM] = {0};
    int predictNum[MAX_CAT_NUM] = {0};
    int elem_per_line = 512 / MAX_CAT_BITS;
    for (int i = 0; i < samples_num; i++) {
        int sample_start_position = (features_num + 1) * i;
        int realClass = testsamples[sample_start_position + features_num];
        int r = i / elem_per_line;
        int c = i % elem_per_line;
        int offset = c * MAX_CAT_BITS;
        int testClass = predictions[r].range(offset + MAX_CAT_BITS - 1, offset);
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
        float precision = (float)rightNum[i] / predictNum[i];
        std::cout << i << "   " << recall << "   " << precision << std::endl;
    }
}
template <unsigned MAX_CAT_BITS>
void precisonAndRecallKernel(DataType* testsamples,
                             hls::stream<ap_uint<MAX_CAT_BITS> >& predictionsStrm,
                             hls::stream<bool>& predictionsTag,
                             int features_num,
                             int numClass) {
    const int MAX_CAT_NUM = 1 << MAX_CAT_BITS;
    int rightNum[MAX_CAT_NUM] = {0};
    int testNum[MAX_CAT_NUM] = {0};
    int predictNum[MAX_CAT_NUM] = {0};
    int i = 0;
    bool e = predictionsTag.read();
    while (e) {
        int sample_start_position = (features_num + 1) * i;
        int realClass = testsamples[sample_start_position + features_num];
        int testClass = predictionsStrm.read();
        testNum[realClass]++;
        predictNum[testClass]++;
        if (testClass == realClass) rightNum[realClass]++;
        e = predictionsTag.read();
        i++;
    }
    for (i = 0; i < numClass; i++) {
        std::cout << " class "
                  << " real count : " << testNum[i] << std::endl;
        std::cout << " class "
                  << " predict count : " << predictNum[i] << std::endl;
    }
    std::cout << "class     recall     precision" << std::endl;
    for (i = 0; i < numClass; i++) {
        float recall = (float)rightNum[i] / testNum[i];
        float precision = (float)rightNum[i] / predictNum[i];
        std::cout << i << "   " << recall << "   " << precision << std::endl;
    }
}

template <typename MType, unsigned WD>
void GenData(MType* samples, int samples_num, int features_num, ap_uint<512>* data) {
    int total = samples_num * (features_num + 1);
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
    data[0].range(31, 0) = samples_num; // row count of data (from row 1) that is axiLen
    data[0].range(63, 32) = features_num;
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
                  Paras paras,
                  int samples_num,
                  int features_num,
                  int numClass,
                  std::string config_path,
                  ap_uint<512>* configs,
                  ap_uint<512>* data) {
    std::ifstream fin_config(config_path);
    std::string line;
    int read_int[128];
    int numSplits[128] = {0};
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
        printf("%d,", numSplits[i]);
    }
    printf("\n");
    index = 0;
    for (int i = 0; i < features_num; i++) {
        for (int j = 0; j < numSplits[i]; j++) {
            printf("%d,%lf\n", i, splits[index]);
            index++;
        }
    }
    // GenConfig<MType, WD>(samples_num, features_num, numClass, paras, numSplits, splits, configs);
    GenData<MType, WD>(samples, samples_num, features_num, data);
}
const unsigned treesize = ((MAX_NODES_NUM >> 1) + 1);
template <unsigned WD, typename MType, unsigned MAX_TREE_DEPTH, unsigned MAX_CAT_BITS>
void GetTreeFromBits(struct Node_H<MType, MAX_TREE_DEPTH, MAX_CAT_BITS> (&nodes)[MAX_NODES_NUM],
                     ap_uint<512> tree[treesize],
                     int& nodes_num) {
    nodes_num = tree[0].range(31, 0);
    f_cast<MType> ff;
    for (unsigned i = 0; i < nodes_num; i++) {
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
template <typename MType, unsigned MAX_TREE_DEPTH>
void loadTree(ap_uint<512>* tree, struct Node<MType> nodes[(MAX_TREE_DEPTH + 1) / 2 + 1][MAX_NODES_NUM]) {
    int nodes_num = tree[0].range(31, 0); // include header
    int axiLen = (nodes_num + 1) / 2 + 1;
    int dupnumPr = (MAX_TREE_DEPTH + 1) / 2 + 1;
    int x_id = 0;
    ap_uint<512> tmp;
    for (int i = 1; i < axiLen; i++) {
        tmp = tree[i];
        for (int d = 0; d < dupnumPr; d++) {
            for (int j = 0; j < 2; j++) {
                int offset = j * 256;
                nodes[d][x_id + j].nodeInfo = tmp.range(offset + 72, offset);
                f_cast<MType> thre;
                thre.i = tmp.range(offset + 255, offset + 192);
                nodes[d][x_id + j].threshold = thre.f;
            }
        }
        x_id += 2;
    }
}
#endif
