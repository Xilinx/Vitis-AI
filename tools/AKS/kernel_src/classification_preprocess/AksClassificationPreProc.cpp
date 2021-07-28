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
#include <opencv2/imgproc/imgproc.hpp>

#include <aks/AksBatchTensorBuffer.h>
#include <aks/AksTensorBuffer.h>
#include <aks/AksKernelBase.h>
#include <aks/AksNodeParams.h>

class ClassificationPreProc : public AKS::KernelBase
{
  public:
    int exec_async (
        std::vector<vart::TensorBuffer*> &in,
        std::vector<vart::TensorBuffer*> &out,
        AKS::NodeParams* nodeParams,
        AKS::DynamicParamValues* dynParams);
};

extern "C" {

  AKS::KernelBase* getKernel (AKS::NodeParams *params)
  {
    return new ClassificationPreProc();
  }

} //extern "C"

int ClassificationPreProc::exec_async (
    std::vector<vart::TensorBuffer*> &in,
    std::vector<vart::TensorBuffer*> &out,
    AKS::NodeParams* nodeParams,
    AKS::DynamicParamValues* dynParams)
{
  std::vector<uint8_t*> buffers;
  std::vector<const xir::Tensor*> tensors;
  int batchSize = 1;
  if(auto* tb = dynamic_cast<AKS::AksBatchTensorBuffer*>(in[0])) {
    tensors = tb->get_tensors();
    batchSize = tensors.size();
    for(int b = 0; b < batchSize; ++b) {
      buffers.push_back(reinterpret_cast<uint8_t*>(tb->data({b}).first));
    }
  } else {
    const auto* tensor = in[0]->get_tensor();
    batchSize = tensor->get_shape()[0];
    for(int b = 0; b < batchSize; ++b) {
      tensors.push_back(tensor);
      buffers.push_back(reinterpret_cast<uint8_t*>(in[0]->data({b}).first));
    }
  }

  /// Get output Dimensions (Network input dim)
  int outHeight    = nodeParams->_intParams["net_h"];
  int outWidth     = nodeParams->_intParams["net_w"];
  int outChannels  = 3;
  int nOutElemsPerImg = outChannels * outHeight * outWidth;
  string outputLayout = nodeParams->hasKey<string>("output_layout") ?
                        nodeParams->getValue<string>("output_layout"): "NCHW";

  /// Create output data buffer
  auto shape = (outputLayout == "NCHW") ?
    std::vector<int>{ batchSize, 3, outHeight, outWidth }:
    std::vector<int>{ batchSize, outHeight, outWidth, 3 };

  std::string tensorName ("pre-output");
  AKS::AksTensorBuffer * outTB = new AKS::AksTensorBuffer(
                                   xir::Tensor::create(
                                     tensorName, shape,
                                     xir::create_data_type<float>()
                                 ));
  float * outData = reinterpret_cast<float*>(outTB->data().first);

  /// Get mean values
  auto meanIter = nodeParams->_floatVectorParams.find("mean");
  float mean [3];
  mean[0] = meanIter->second[0];
  mean[1] = meanIter->second[1];
  mean[2] = meanIter->second[2];

  /// Get input dims incase batch array
  int inChannel = 3;
  int inHeight  = 0;
  int inWidth   = 0;
  uint8_t* inData = nullptr;
  for (int b = 0; b < batchSize; ++b) {
    inData   = buffers[b];
    inHeight = tensors[b]->get_shape()[1];
    inWidth  = tensors[b]->get_shape()[2];

    /// Create a cv::Mat with input data
    cv::Mat inImage(inHeight, inWidth, CV_8UC3, inData);

    /// Resize the image to Network Shape
    cv::Mat resizedImage = cv::Mat(outHeight, outWidth, CV_8SC3);
    cv::resize(inImage, resizedImage, cv::Size(outWidth, outHeight));

    /// Pre-Processing loop
    float* out = outData + b * nOutElemsPerImg;
    if (outputLayout == "NCHW") {
      for (int c = 0; c < 3; c++) {
        for (int h = 0; h < outHeight; h++) {
          for (int w = 0; w < outWidth; w++) {
            out[(c*outHeight*outWidth)
              + (h*outWidth) + w]
              = resizedImage.at<cv::Vec3b>(h,w)[c]-mean[c];
          }
        }
      }
    } else if (outputLayout == "NHWC"){
      for (int h = 0; h < outHeight; h++) {
        for (int w = 0; w < outWidth; w++) {
          for (int c = 0; c < 3; c++) {
            out[h*outWidth*3 + w*3 + c]
              = resizedImage.at<cv::Vec3b>(h,w)[c]-mean[c];
          }
        }
      }
    }
  }

  /// Push back output
  out.push_back(outTB);
  return 0;
}

