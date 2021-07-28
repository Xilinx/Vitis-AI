/*
 * Copyright 2021 Xilinx Inc.
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
#include <cmath>
#include <vector>
#include <assert.h>
#include <fstream>

#include <aks/AksKernelBase.h>
#include <aks/AksNodeParams.h>
#include <aks/AksLogger.h>
#include <aks/AksTensorBuffer.h>

class OFClassificationPreProcess : public AKS::KernelBase
{
  public:
    int exec_async (
           std::vector<vart::TensorBuffer*> &in,
           std::vector<vart::TensorBuffer*> &out,
           AKS::NodeParams* nodeParams,
           AKS::DynamicParamValues* dynParams);
};

extern "C" { /// Add this to make this available for python bindings and

AKS::KernelBase* getKernel (AKS::NodeParams *params)
{
  return new OFClassificationPreProcess();
}


int OFClassificationPreProcess::exec_async (
           std::vector<vart::TensorBuffer*> &in,
           std::vector<vart::TensorBuffer*> &out,
           AKS::NodeParams* nodeParams,
           AKS::DynamicParamValues* dynParams)
{
  // std::cout << "[DBG] OFClassificationPreProcess: running now ... " << std::endl;
  assert(in.size() == 13);
  int numBuffersToStack = 10;
  int batchSize = 4;
  auto inShape = in[0]->get_tensor()->get_shape();
  int rows = inShape.at(1);
  int cols = inShape.at(2);
  int channels = inShape.at(3); // 2
  assert(channels == 2);
  int stackChannels = numBuffersToStack * channels;  // 20

  // for (int i=0; i<in.size(); i++) {
  //   int8_t* data = reinterpret_cast<int8_t*>(in[i]->data().first);
  //   ofstream myfile;
  //   myfile.open(std::to_string(i)+"_input.txt");
  //   for (int j=0; j<rows*cols*channels; j++) {
  //     myfile << (int)data[j]+128 << "\n";
  //   }
  //   myfile.close();
  // }

  vart::TensorBuffer *outDD = new AKS::AksTensorBuffer(
      xir::Tensor::create(
          "OFClassificationPreProcess",
          {batchSize, rows, cols, stackChannels},
          xir::create_data_type<float>()
      ));

  float* outData = reinterpret_cast<float*>(outDD->data().first);
  for (int b = 0; b < batchSize; b++) {
    float* batchData = outData + b*rows*cols*stackChannels;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        for (int k = 0; k < stackChannels; k++) {
          int8_t* data = reinterpret_cast<int8_t*>(in[b + k/channels]->data().first);
          batchData[i*cols*stackChannels + j*stackChannels + k] = \
              data[i*cols*channels + j*channels + k%channels];
        }
      }
    }

    // ofstream myfile;
    // myfile.open(std::to_string(b)+"_output.txt");
    // for (int i=0; i<rows*cols*stackChannels; i++) {
    //   myfile << batchData[i]+128 << "\n";
    // }
    // myfile.close();
  }

  out.push_back(outDD);
  // std::cout << "[DBG] OFClassificationPreProcess: Done!" << std::endl << std::endl;
  return -1; /// No wait
}

}//extern "C"
