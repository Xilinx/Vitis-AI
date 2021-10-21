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
#include "xf_data_analytics/classification/svm_predict.hpp"
#include "xf_data_analytics/common/table_sample.hpp"
typedef double DataType;
const int dw = sizeof(DataType) * 8;
const int streamN = 8;
const int sampleD = 4;
#define MAX_CLASS_BITS_ 8
void dut(const int cols,
         hls::stream<DataType> sample_strm[8],
         hls::stream<bool>& e_sample_strm,
         hls::stream<ap_uint<512> >& weight_strm,
         hls::stream<bool>& eTag,
         hls::stream<ap_uint<1> >& predictionsStrm,
         hls::stream<bool>& predictionsTag) {
    xf::data_analytics::classification::svmPredict<DataType, dw, streamN, sampleD>(
        cols, sample_strm, e_sample_strm, weight_strm, eTag, predictionsStrm, predictionsTag);
}

#ifndef __SYNTHESIS__
#include <iostream>
#include <fstream>
#include <string>
int main() {
    int test_num = 999;
    int features_num = 28;
    int numClass = 2;

    std::ifstream fin_test("1000.csv");
    DataType* testsets = (DataType*)malloc(sizeof(DataType) * test_num * (features_num + 1));
    ap_uint<64>* datasets = (ap_uint<64>*)malloc(sizeof(ap_uint<64>) * test_num * (features_num + 1));
    std::string line;
    int row = 0;
    int col = 0;
    while (getline(fin_test, line)) {
        std::istringstream sin(line);
        std::string attr_val;
        col = 0;
        while (getline(sin, attr_val, ' ')) {
            size_t pos = attr_val.find(':');
            if (pos != attr_val.npos) {
                attr_val = attr_val.substr(pos + 1);
            }
            xf::data_analytics::common::internal::f_cast<DataType> w;
            w.f = (std::atof(attr_val.c_str()));
            datasets[(features_num + 1) * row + col] = w.i;
            testsets[(features_num + 1) * row + col] = (std::atof(attr_val.c_str()));
            col++;
        }
        row++;
    }
    // test csv read
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < col; j++) {
            std::cout << testsets[i * (features_num + 1) + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "rows num:" << row << std::endl;
    std::cout << "cols num:" << col << std::endl;

    ap_uint<512>* weight = (ap_uint<512>*)malloc(sizeof(ap_uint<512>) * (features_num / 8 + 1));

    for (int i = 0; i < features_num / 8 + 1; i++) {
        weight[i] = 0;
    }
    double init_weight[28] = {0.2936619965423339,    0.05043345524721191,  0.031392431301380294,  -0.2096483985054803,
                              -0.002805004135527406, 0.31402549753695486,  -0.053070948334048525, -0.006191672490869336,
                              0.11774994270906798,   0.007016734906995865, -0.003927265125801403, 0.025059226600449125,
                              0.07373436756434097,   0.1984081643964114,   0.012053003688647378,  -0.004846889579785206,
                              0.03069150997183022,   0.01741398466236531,  -0.022055060173911036, -0.03837669517039672,
                              0.02386278039997709,   0.012338757608139912, 0.28087785478663546,   0.4175200241772661,
                              -0.02757256316066491,  -0.3523018346852643,  -0.08882809901480027,  -0.32951255315353783};
    for (int i = 0; i < features_num; i++) {
        xf::data_analytics::common::internal::f_cast<DataType> w;
        w.f = init_weight[i];
        weight[i / 8](i % 8 * 64 + 63, i % 8 * 64) = w.i;
    }

    // gen predict data
    hls::stream<DataType> sample_strm[8];
    hls::stream<bool> e_sample_strm;
    hls::stream<ap_uint<512> > weight_strm;
    hls::stream<bool> eTag;
    hls::stream<ap_uint<1> > predictionsStrm;
    hls::stream<ap_uint<1> > goldenStrm;
    hls::stream<bool> predictionsTag;

    for (int i = 0; i < test_num; i++) {
        double temp = 0.0;
        for (int j = 0; j < features_num / 8 + 1; j++) {
            for (int k = 0; k < 8; k++) {
                if ((j * 8 + k) < features_num) {
                    sample_strm[k].write(testsets[i * (features_num + 1) + j * 8 + k]);
                    temp += testsets[i * (features_num + 1) + j * 8 + k] * init_weight[j * 8 + k];
                } else {
                    sample_strm[k].write(0.0);
                }
            }
            e_sample_strm.write(false);
        }
        if (temp > 0)
            goldenStrm.write(1);
        else
            goldenStrm.write(0);
    }
    e_sample_strm.write(true);

    for (int i = 0; i < features_num / 8 + 1; i++) {
        weight_strm.write(weight[i]);
        eTag.write(false);
    }
    eTag.write(true);
    // predict kernel
    dut(features_num, sample_strm, e_sample_strm, weight_strm, eTag, predictionsStrm, predictionsTag);
    bool e_out = predictionsTag.read();
    int count = 0;
    while (!e_out) {
        ap_uint<2> out = predictionsStrm.read();
        e_out = predictionsTag.read();
        if (out != goldenStrm.read()) {
            count++;
        }
    }

    free(testsets);
    free(datasets);
    free(weight);
    return count;
}

#endif
