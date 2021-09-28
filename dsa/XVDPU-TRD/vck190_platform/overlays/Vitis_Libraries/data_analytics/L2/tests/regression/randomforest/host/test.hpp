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

#ifndef _XF_DATA_ANALYTICS_RANDOMFOREST_TRAIN_HOST_HPP_
#define _XF_DATA_ANALYTICS_RANDOMFOREST_TRAIN_HOST_HPP_
#define MAX_FEAS_H (MAX_FEAS)
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
void precisonAndRecall(DataType* testsamples,
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
        float precision = (float)rightNum[i] / predictNum[i];
        std::cout << i << "   " << recall << "   " << precision << std::endl;
    }
}
template <typename FType>
void gen_dat(const std::string& name, const std::string& dir, size_t samples_num, size_t features_num) {
    int WD = sizeof(FType) * 8;
    int elem_per_line = 512 / WD;
    size_t total_n = samples_num * (features_num + 1);
    size_t size_512 = (total_n + elem_per_line - 1) / elem_per_line;
    ap_uint<512>* data = (ap_uint<512>*)malloc(sizeof(ap_uint<512>) * size_512);
    std::string fi = dir + "/" + name + ".csv";
    std::string fo = dir + "/" + name + ".dat";
    std::ifstream fin(fi);
    std::string line;
    int row = 0;
    int col = 0;
    int r = 0;
    int c = 0;
    int i = 0;
    if (!fin) {
        std::cerr << "ERROR: " << fi << " cannot be opened for binary read." << std::endl;
    }
    while (getline(fin, line)) {
        std::istringstream sin(line);
        std::string attr_val;
        col = 0;
        while (getline(sin, attr_val, ',')) {
            FType elem = std::atof(attr_val.c_str());
            if (row < 10) {
                printf("%f ", elem);
                if (col == features_num) {
                    printf("\n");
                }
            }
            r = i / elem_per_line;
            c = i % elem_per_line;
            f_cast<FType> tmp_elem;
            tmp_elem.f = elem;
            data[r].range(c * WD + WD - 1, c * WD) = tmp_elem.i;
            col++;
            i++;
        }
        row++;
    }
    fin.close();
    std::ofstream fout(fo, std::ios::binary);
    fout.write((char*)data, sizeof(ap_uint<512>) * size_512);
    fout.close();
    printf("samples_num:%d(n:%d)\n", row, samples_num);
    printf("features_num:%d\n", col - 1);
    printf("total_n((feautes_num+1)*n=%d):%d\n", total_n, i);
    printf("size_512:%d\n", size_512);
}
template <typename T>
int load_dat(T* data, const std::string& name, const std::string& dir, size_t n, size_t sizeT) {
    if (!data) {
        return -1;
    }

    std::string fn = dir + "/" + name + ".dat";
    FILE* f = fopen(fn.c_str(), "rb");
    if (!f) {
        std::cerr << "ERROR: " << fn << " cannot be opened for binary read." << std::endl;
    }
    // size_t cnt = fread(data, sizeof(T), n, f);
    size_t cnt = fread((void*)data, sizeT, n, f);
    fclose(f);
    if (cnt != n) {
        std::cerr << "ERROR: " << cnt << " entries read from " << fn << ", " << n << " entries required." << std::endl;
        return -1;
    }
    return 0;
}
template <typename T>
int gen_dat(T* data, const std::string& name, const std::string& dir, size_t n, size_t sizeT) {
    if (!data) {
        return -1;
    }

    std::string fn = dir + "/" + name + ".dat";
    FILE* f = fopen(fn.c_str(), "wb");
    if (!f) {
        std::cerr << "ERROR: " << fn << " cannot be opened for binary read." << std::endl;
    }
    // size_t cnt = fread(data, sizeof(T), n, f);
    size_t cnt = fwrite((void*)data, sizeT, n, f);
    fclose(f);
    if (cnt != n) {
        std::cerr << "ERROR: " << cnt << " entries read from " << fn << ", " << n << " entries required." << std::endl;
        return -1;
    }
    return 0;
}
template <typename MType, unsigned WD>
void GenConfig(ap_uint<32> samples_num,
               ap_uint<32> features_num,
               ap_uint<32> numClass,
               Paras paras,
               ap_uint<8> numSplits[128],
               MType splits[128],
               ap_uint<512>* configs) {
    const unsigned elems_per_line = 512 / WD;
    int splitsnum = 0;
    for (int i = 0; i < features_num; i++) {
        splitsnum += numSplits[i];
    }
    configs[0].range(63, 32) = 1;
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
    configs[0].range(95, 64) = r + 1;
    for (int i = 0; i < features_num; i++) {
        int i_r = i & 0x3f;
        if (i_r == 0) r++;
        int index = i_r * 8;
        configs[r].range(index + 8 - 1, index) = numSplits[i];
    }
    r++;

    configs[0].range(127, 96) = r;
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
    if (c != elems_per_line) {
        r++;
    }

    configs[0].range(31, 0) = r;
    configs[0].range(159, 128) = splitsnum;
}
template <typename MType, unsigned WD>
void GenConfAll(ap_uint<32> samples_num,
                ap_uint<32> features_num,
                ap_uint<32> numClass,
                Paras paras,
                std::string path,
                ap_uint<512>* configs) {
    std::ifstream fin_config(path);
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
