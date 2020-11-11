/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <cassert>
#include <cmath>

#include <aks/AksKernelBase.h>
#include <aks/AksDataDescriptor.h>
#include <aks/AksNodeParams.h>

class ClassificationPostProc : public AKS::KernelBase
{
  public:
    int getNumCUs (void) { return 1; }
    int exec_async (
           std::vector<AKS::DataDescriptor*> &in, 
           std::vector<AKS::DataDescriptor*> &out, 
           AKS::NodeParams* nodeParams,
           AKS::DynamicParamValues* dynParams);
};

extern "C" { /// Add this to make this available for python bindings and 

AKS::KernelBase* getKernel (AKS::NodeParams *params)
{
  return new ClassificationPostProc();
}

}//extern "C"

/**
 * @brief calculate softmax
 *
 * @param data - pointer to input buffer
 * @param size - size of input buffer
 * @param result - calculation result
 *
 * @return none
 */
void CPUCalcSoftmax(const float *data, size_t size, float *result) {
    assert(data && result);
    double sum = 0.0f;

    for (size_t i = 0; i < size; i++) {
        result[i] = exp(data[i]);
        sum += result[i];

    }
    for (size_t i = 0; i < size; i++) {
        result[i] /= sum;
    }
}

/**
 * @brief Get top k results according to its probability
 *
 * @param d - pointer to input data
 * @param size - size of input data
 * @param k - calculation result
 *
 * @return vector of top 'k' pairs of <index, probability>
 */
std::vector<std::pair<int, float>> TopK(const float *d, int size, int k) {
    assert(d && size > 0 && k > 0);
    priority_queue<pair<float, int>> q;

    for (auto i = 0; i < size; ++i) {
        q.push(pair<float, int>(d[i], i));
    }
    std::vector<std::pair<int, float>> topKIndex;
    for (auto i = 0; i < k; ++i) {
        std::pair<float, int> ki = q.top();
        // printf("top[%d] index = %d prob = %-8f  \n", i, ki.second, d[ki.second]);
        q.pop();
        topKIndex.push_back(make_pair(ki.second, ki.first));
    }
    return topKIndex;
}

int ClassificationPostProc::exec_async (
           std::vector<AKS::DataDescriptor*> &in, 
           std::vector<AKS::DataDescriptor*> &out, 
           AKS::NodeParams* nodeParams,
           AKS::DynamicParamValues* dynParams)
{
  /// Get input and output data shapes
  std::vector <int> inShape  = in[0]->getShape();

  int inBatch    = inShape[0];
  int inChannels = inShape[1];
  int inHeight   = inShape[2];
  int inWidth    = inShape[3];

  /// Get input data 
  float * inData  = (float*) in[0]->data();

  /// Get K value for top "K"
  int k = nodeParams->_intParams.find("top_k") == nodeParams->_intParams.end() ?
                  5 : nodeParams->_intParams["top_k"];

  /// Create output data
  AKS::DataDescriptor *labels = new AKS::DataDescriptor ({inBatch, k, 1, 1}, AKS::DataType::INT32);
  AKS::DataDescriptor *probs  = new AKS::DataDescriptor ({inBatch, k, 1, 1}, AKS::DataType::FLOAT32);

  int *labelsData  = static_cast<int*>(labels->data());
  float *probsData = static_cast<float*>(probs->data());

  float * softmaxOut = new float[inChannels];
  
  for (int i = 0; i < inBatch; ++i) {

    /// Compute SoftMax
    CPUCalcSoftmax(inData + i * inChannels, inChannels, softmaxOut);
    std::vector<std::pair<int, float>> topKIndex = TopK(softmaxOut, inChannels, k);

    /// Create TopK for next node
    int idx = 0;
    for (auto& val: topKIndex) {
      labelsData[i*k + idx] = val.first;
      probsData[i*k + idx] = val.second;
      idx++;
    }
  }
  
  /// Push TopK indices to next node
  out.push_back(labels);
  out.push_back(probs);

  delete[] softmaxOut;
  return -1; /// No wait
}
