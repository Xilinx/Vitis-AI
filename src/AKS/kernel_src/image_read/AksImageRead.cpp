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
#include <aks/AksNodeParams.h>
#include <aks/AksBatchTensorBuffer.h>
#include <xir/tensor/tensor.hpp>
#include <xir/util/data_type.hpp>

class ImageRead : public AKS::KernelBase
{
  public:
    int exec_async (
        std::vector<vart::TensorBuffer*> &in,
        std::vector<vart::TensorBuffer*> &out,
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
    std::vector<vart::TensorBuffer*> &in,
    std::vector<vart::TensorBuffer*> &out,
    AKS::NodeParams* nodeParams,
    AKS::DynamicParamValues* dynParams)
{
  /// Get the image path
  int batchSize = dynParams->imagePaths.size();
  std::vector<std::unique_ptr<xir::Tensor>> tensors; tensors.reserve(batchSize);
  std::vector<cv::Mat> images; images.reserve(batchSize);
  for(int b = 0; b < dynParams->imagePaths.size(); ++b) {

    cv::Mat inImage = cv::imread (dynParams->imagePaths[b].c_str());
    if (!inImage.data) {
      std::cerr << "[ERR] Unable to read image: " << dynParams->imagePaths[b] << std::endl;
      return -2;
    }

    /// Assign the image data to output vector
    std::vector<int> shape = { 1, inImage.rows, inImage.cols, inImage.channels() };

    auto tensorOut = xir::Tensor::create("imread_output", shape,
		                                     xir::DataType {xir::DataType::INT, 8U});
    tensors.push_back(std::move(tensorOut));
    images.push_back(std::move(inImage));
  }

  auto* buf =  new AKS::AksBatchTensorBuffer(std::move(tensors));
  for(int i=0; i<batchSize; ++i) {
    auto* bufptr = reinterpret_cast<uint8_t*>(buf->data({i}).first);
    auto* imgptr = images[i].data;
    memcpy(bufptr, imgptr, buf->get_tensors()[i]->get_data_size());
  }
  out.push_back(buf);
  return 0;
}

