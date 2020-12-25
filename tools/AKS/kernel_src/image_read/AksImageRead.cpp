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

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <aks/AksKernelBase.h>
#include <aks/AksDataDescriptor.h>
#include <aks/AksNodeParams.h>

class ImageRead : public AKS::KernelBase
{
  public:
    int exec_async (
        std::vector<AKS::DataDescriptor*> &in, 
        std::vector<AKS::DataDescriptor*> &out, 
        AKS::NodeParams* nodeParams,
        AKS::DynamicParamValues* dynParams);
};

extern "C" { /// Add this to make this available for python bindings

  AKS::KernelBase* getKernel (AKS::NodeParams *params)
  {
    return new ImageRead();
  }

}//extern "C"

int ImageRead::exec_async (
    std::vector<AKS::DataDescriptor*> &in, 
    std::vector<AKS::DataDescriptor*> &out, 
    AKS::NodeParams* nodeParams,
    AKS::DynamicParamValues* dynParams)
{
  /// Get the image path
  int batchSize = dynParams->imagePaths.size();
  out.push_back(new AKS::DataDescriptor({batchSize,}, AKS::DataType::AKSDD));
  auto outData = out[0]->data<AKS::DataDescriptor>();
  for(int b = 0; b < dynParams->imagePaths.size(); ++b) {

    cv::Mat inImage = cv::imread (dynParams->imagePaths[b].c_str());
    if (!inImage.data) {
      std::cerr << "[ERR] Unable to read image: " << dynParams->imagePaths[b] << std::endl;
      return -2;
    }

    /// Assign the image data to output vector
    std::vector<int> shape = { 1, inImage.rows, inImage.cols, inImage.channels() };
    AKS::DataDescriptor dd(shape, AKS::DataType::UINT8);
    unsigned long imgSize = inImage.channels() * inImage.rows * inImage.cols;
    memcpy(dd.data(), inImage.data, imgSize);
    outData[b] = std::move(dd);
  }
  return 0;
}

