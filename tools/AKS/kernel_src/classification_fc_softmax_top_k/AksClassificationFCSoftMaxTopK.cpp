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
#include <map>
#include <algorithm>

#include <cblas.h>

#include <assert.h>
#include <math.h>
#include <queue>
#include <fstream>

#include <aks/AksKernelBase.h>
#include <aks/AksDataDescriptor.h>
#include <aks/AksNodeParams.h>
#include <aks/AksLogger.h>

#include <H5Cpp.h>

using namespace std;
using namespace AKS;

struct weightsDescriptor {
  std::string _weigthsFile;
  std::map<std::string, std::vector<float>> _weights;
  std::map<std::string, std::vector<float>> _bias;
};

class ClassificationFCSoftMaxTopK : public AKS::KernelBase
{
  public:
    ClassificationFCSoftMaxTopK () {}
    void nodeInit (AKS::NodeParams*);
    int exec_async (
        std::vector<AKS::DataDescriptor*> &in, 
        std::vector<AKS::DataDescriptor*> &out, 
        AKS::NodeParams* nodeParams,
        AKS::DynamicParamValues* dynParams);

    std::map<AKS::NodeParams*, weightsDescriptor> _wbParams;
  private:
    std::map<std::string, std::vector<float>> _weights;
    std::map<std::string, std::vector<float>> _bias;
    std::string _weigthsFile;
};

extern "C" { /// Add this to make this available for python bindings

  // weight - NxK
  // data   - KxM
  // output - weight x data + bias = NxM
  void computeFC(float *weight, float *bias, float *data,
      int M, int N, int K, float *output)
  {
    int lda = K;
    int ldb = K;
    for (int batch=0; batch<M; batch++) {
      float* to_fill = output + (batch*N);
      cblas_sgemv(CblasRowMajor,
          (CBLAS_TRANSPOSE)CblasNoTrans,
          N, K, 1.,
          weight, K, data + batch*K, 1, 0., to_fill, 1);
    }

    lda = 1;
    ldb = N;
    std::vector<float> bias_multiplier(M, 1.);
    for (int batch=0; batch<M; batch++) {
      float* to_fill = output + (batch*N);
      cblas_sgemv(
          CblasRowMajor,
          (CBLAS_TRANSPOSE)CblasNoTrans,
          N, 1, 1., bias, 1,
          &(bias_multiplier[0]), 1, 1., to_fill, 1);
    }
  }

  AKS::KernelBase* getKernel (AKS::NodeParams *params)
  {
    return new ClassificationFCSoftMaxTopK();
  }

}//extern "C"

void ClassificationFCSoftMaxTopK::nodeInit (AKS::NodeParams* params)
{
  struct weightsDescriptor wbDesc;
  wbDesc._weigthsFile = params->_stringParams["weights"];

  H5::Exception::dontPrint(); // Don't print H5 exceptions

  const H5std_string FILE_NAME(wbDesc._weigthsFile); // Open up weights
  H5::H5File file(FILE_NAME, H5F_ACC_RDONLY);        // as hdf5 file

  // Load up all weights
  int fi;
  for ( fi=0; ; fi++)
  {
    try
    {

      // Open the HDF5 Dataset for given layer
      std::string dsname = "fwbqb_" + std::to_string(fi);
      const H5std_string DATASET_NAME(dsname);
      H5::DataSet dataset = file.openDataSet(DATASET_NAME);
      /// Don't load these weights: Not required
    }
    catch( H5::FileIException error )
    {
      if (1)
        LOG(DEBUG) << "No more weights in HDF5" << std::endl;
      break;
    }
  }
  /// Load FC Weights
  fi = 0;
  for ( ; ; fi++)
  {
    try
    {
      // Open the HDF5 Dataset for given layer
      std::string dsname = "fc_" + std::to_string(fi);
      const H5std_string DATASET_NAME(dsname);
      H5::DataSet dataset = file.openDataSet(DATASET_NAME);

      // Get the dataspace // Defines number of dimensions, and the size of each dimension
      H5::DataSpace dataspace = dataset.getSpace();

      // Get the number of dimensions in the dataspace.
      int rank = dataspace.getSimpleExtentNdims();

      // Get the dimension size of each dimension in the dataspace
      std::vector<hsize_t> dims(rank);
      int ndims = dataspace.getSimpleExtentDims(&dims[0], NULL);

      // Get the layerName from the dataset's layer_name attribute
      const H5std_string  ATTR_NAME("layer_name");
      H5::Attribute attribute = dataset.openAttribute(ATTR_NAME);
      H5::StrType stype = attribute.getStrType();
      std::string layerName;
      attribute.read(stype,layerName);

      if (1)
      {
        SLOG(LogLevel::DEBUG, 
            _DEBUG << "Loading HDF5 dataset: " << DATASET_NAME << ", from file: " 
            << FILE_NAME << ",layerName: " << layerName << ", having dataspace:" << std::endl;
            _DEBUG << "rank: " << rank << ", dimensions: ";
            for (int i=0;i<dims.size();i++)
            std::cout << (unsigned long)(dims[i]) << " x ";
            std::cout << std::endl;);
      }

      // Get the raw data
      std::vector<float> weights;
      int space = 1;
      for (int i=0;i<rank;i++)
        space *= dims[i];
      weights.resize(space);
      dataset.read(&weights[0],H5::PredType::NATIVE_FLOAT,dataspace,dataspace);

      wbDesc._weights[dsname] = weights;
    }
    catch( H5::FileIException error )
    {
      if (1)
        LOG(DEBUG) << "No more weights in HDF5" << std::endl;
      break;
    }
  }

  // Load up all bias
  for (fi = 0; ; fi++)
  {
    try
    {
      // Open the HDF5 Dataset for given layer
      std::string dsname = "fwbqb_bias_" + std::to_string(fi);
      const H5std_string DATASET_NAME(dsname);
      H5::DataSet dataset = file.openDataSet(DATASET_NAME);
      /// Don't load these bias: Not required
    }
    catch( H5::FileIException error )
    {
      if (1)
        LOG(DEBUG) << "No more bias in HDF5" << std::endl;
      break;
    }
  }
  /// Load FC Bias
  fi = 0;
  for ( ; ; fi++)
  {
    try
    {
      // Open the HDF5 Dataset for given layer
      std::string dsname = "fc_bias_" + std::to_string(fi);
      const H5std_string DATASET_NAME(dsname);
      H5::DataSet dataset = file.openDataSet(DATASET_NAME);

      // Get the dataspace // Defines number of dimensions, and the size of each dimension
      H5::DataSpace dataspace = dataset.getSpace();

      // Get the number of dimensions in the dataspace.
      int rank = dataspace.getSimpleExtentNdims();

      // Get the dimension size of each dimension in the dataspace
      std::vector<hsize_t> dims(rank);
      int ndims = dataspace.getSimpleExtentDims(&dims[0], NULL);

      // Get the layerName from the dataset's layer_name attribute
      const H5std_string  ATTR_NAME("layer_name");
      H5::Attribute attribute = dataset.openAttribute(ATTR_NAME);
      H5::StrType stype = attribute.getStrType();
      std::string layerName;
      attribute.read(stype,layerName);

      if (1)
      {
        SLOG(LogLevel::DEBUG, 
            _DEBUG << "Loading HDF5 dataset: " << DATASET_NAME << ", from file: " 
            << FILE_NAME << ",layerName: " << layerName << ", having dataspace:" << std::endl;
            _DEBUG << "rank: " << rank << ", dimensions: ";
            for (int i=0;i<dims.size();i++)
            std::cout << (unsigned long)(dims[i]) << " x ";
            std::cout << std::endl;)
      }

      // Get the raw data
      std::vector<float> bias;
      int space = 1;
      for (int i=0;i<rank;i++)
        space *= dims[i];
      bias.resize(space);
      dataset.read(&bias[0],H5::PredType::NATIVE_FLOAT,dataspace,dataspace);

      wbDesc._bias[dsname] = bias;
    }
    catch( H5::FileIException error )
    {
      if (1)
        LOG(DEBUG) << "No more bias in HDF5" << std::endl;
      break;
    }
  }
  for (auto& w: _weights) {
    LOG(DEBUG) << "FC Weights Size: " << w.second.size() << std::endl;
  }
  for (auto& b: _bias) {
    LOG(DEBUG) << "FC Bias Size: " << b.second.size() << std::endl;
  }
  _wbParams[params] = std::move(wbDesc);
}

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

  float max = data[0];
  for (size_t i = 1; i < size; i++)
    if (data[i] > max) max = data[i];

  for (size_t i = 0; i < size; i++) {
    result[i] = exp(data[i] - max);
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

int ClassificationFCSoftMaxTopK::exec_async (
    std::vector<AKS::DataDescriptor*> &in, 
    std::vector<AKS::DataDescriptor*> &out, 
    AKS::NodeParams* nodeParams,
    AKS::DynamicParamValues* dynParams)
{
  /// Get input and output data shapes
  std::vector <int> inShape  = in[0]->getShape();

  int inBatch  = inShape[0];
  int inSize   = inShape[1];
  for(int i = 2; i < inShape.size(); ++i) inSize *= inShape[i];

  int outSize  = _wbParams[nodeParams]._bias["fc_bias_0"].size();
  int k = nodeParams->_intParams.find("top_k") == nodeParams->_intParams.end() ?
    5 : nodeParams->_intParams["top_k"];
  
  /// Get input data 
  float * inData  = (float*) in[0]->data();

  /// Create output buffer for FC layer
  AKS::DataDescriptor * fcOut = new AKS::DataDescriptor ({inBatch, outSize, 1, 1}, AKS::DataType::FLOAT32);
  float * fcOutPtr = static_cast<float*>(fcOut->data());
  /// Compute FC layer
  computeFC((float*)(&_wbParams[nodeParams]._weights["fc_0"][0]), 
      (float*)(&_wbParams[nodeParams]._bias["fc_bias_0"][0]), 
      inData, inBatch, outSize, inSize, fcOutPtr);

  AKS::DataDescriptor *labels = new AKS::DataDescriptor ({inBatch, k, 1, 1}, AKS::DataType::INT32);
  AKS::DataDescriptor *probs = new AKS::DataDescriptor ({inBatch, k, 1, 1}, AKS::DataType::FLOAT32);
  int *labelsData = static_cast<int*>(labels->data());
  float *probsData = static_cast<float*>(probs->data());

  float * softmaxOut = new float[outSize];
  for(int i=0; i<inBatch; ++i) {
    /// Create output buffer for SoftMax layer
    /// Compute SoftMax
    CPUCalcSoftmax(fcOutPtr + i*outSize, outSize, softmaxOut);

    std::vector<std::pair<int, float>> topKIndex = TopK(softmaxOut, outSize, k);

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

  delete fcOut;
  delete[] softmaxOut;
  return 0;
}

