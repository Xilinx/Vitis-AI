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
//#include "decision_tree.hpp"
#define INCLUDE_IN_L1_HOST
#include "xf_data_analytics/regression/decision_tree_predict.hpp"
typedef double DataType;
const int dw = sizeof(DataType) * 8;
#define MAX_FEA_NUM_ 64
#define MAX_TREE_DEPTH_ 10
void dut(hls::stream<ap_uint<dw> > dstrm_batch[MAX_FEA_NUM_],
         hls::stream<bool>& estrm_batch,
         hls::stream<ap_uint<512> >& treeStrm,
         hls::stream<bool>& treeTag,
         hls::stream<DataType>& predictionsStrm,
         hls::stream<bool>& predictionsTag) {
    xf::data_analytics::regression::decisionTreePredict<DataType, dw, MAX_FEA_NUM_, MAX_TREE_DEPTH_>(
        dstrm_batch, estrm_batch, treeStrm, treeTag, predictionsStrm, predictionsTag);
}

#ifndef __SYNTHESIS__
#include <iostream>
#include <fstream>
#include <string>
#include "test.hpp"
int main() {
    struct Node_H<DataType, MAX_TREE_DEPTH_> nodes[MAX_NODES_NUM];
    int nodes_num = 1;
    for (int i = 0; i < MAX_NODES_NUM; i++) {
        nodes[i].chl = INVALID_NODEID;
        nodes[i].isLeaf = 0;
    }
    int test_num = 100;
    int features_num = 26;

    std::ifstream fin_test("test_ss.csv");
    DataType* testsets = (DataType*)malloc(sizeof(DataType) * test_num * (features_num + 1));
    std::string line;
    int row = 0;
    int col = 0;
    while (getline(fin_test, line)) {
        std::istringstream sin(line);
        std::string attr_val;
        col = 0;
        while (getline(sin, attr_val, ',')) {
            testsets[(features_num + 1) * row + col] = std::atof(attr_val.c_str());
            col++;
        }
        row++;
    }

    ap_uint<512> tree[treesize];

    nodes_num = 167;
    int axiLen = (nodes_num + 1) / 2 + 1;
    loadTreeFrmFile(tree, "tree.data", axiLen, sizeof(ap_uint<512>));
    GetTreeFromBits<dw>(nodes, tree, nodes_num);
    printTree(nodes, nodes_num);
    precisonAndRecall<DataType, MAX_FEA_NUM_, MAX_TREE_DEPTH_>(testsets, test_num, features_num, nodes);

    // gen predict data
    hls::stream<ap_uint<dw> > dstrm_batch[MAX_FEA_NUM_];
    hls::stream<bool> estrm_batch;
    hls::stream<ap_uint<512> > treeStrm;
    hls::stream<bool> treeTag;
    hls::stream<DataType> predictionsStrm;
    hls::stream<bool> predictionsTag;

    for (int i = 0; i < test_num; i++) {
        for (int j = 0; j < MAX_FEA_NUM_; j++) {
            if (j < features_num) {
                f_cast<DataType> tdata;
                tdata.f = testsets[i * (features_num + 1) + j];
                dstrm_batch[j].write(tdata.i);
            } else {
                dstrm_batch[j].write(0);
            }
        }
        estrm_batch.write(true);
    }
    estrm_batch.write(false);

    for (int i = 1; i < axiLen; i++) {
        treeStrm.write(tree[i]);
        treeTag.write(true);
    }
    treeTag.write(false);
    // predict kernel
    dut(dstrm_batch, estrm_batch, treeStrm, treeTag, predictionsStrm, predictionsTag);
    precisonAndRecallKernel(testsets, predictionsStrm, predictionsTag, features_num);
    return 0;
}

#endif
