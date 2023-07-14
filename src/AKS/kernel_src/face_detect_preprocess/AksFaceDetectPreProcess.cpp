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
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <aks/AksBatchTensorBuffer.h>
#include <aks/AksTensorBuffer.h>
#include <aks/AksKernelBase.h>
#include <aks/AksNodeParams.h>

class FaceDetectPreProcess : public AKS::KernelBase
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
    return new FaceDetectPreProcess();
  }

}//extern "C"

int FaceDetectPreProcess::exec_async (
  std::vector<vart::TensorBuffer*> &in, std::vector<vart::TensorBuffer*> &out,
  AKS::NodeParams* nodeParams, AKS::DynamicParamValues* dynParams)
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
  int outChannels  = nodeParams->_intParams["net_c"];

  std::string outputLayout =
    nodeParams->hasKey<string>("output_layout") ?
    nodeParams->getValue<string>("output_layout"): "NCHW";

  /// Create output data buffer
  auto shape = (outputLayout == "NCHW") ?
    std::vector<int>{ batchSize, outChannels, outHeight, outWidth }:
    std::vector<int>{ batchSize, outHeight, outWidth, outChannels };

  std::string tensorName ("face-pre-output");
  AKS::AksTensorBuffer * outTB =
    new AKS::AksTensorBuffer(xir::Tensor::create(
      tensorName, shape, xir::create_data_type<float>()));

  /// Get mean values
  auto meanIter = nodeParams->_floatVectorParams.find("mean");
  std::vector<float> mean { meanIter->second[0],
                            meanIter->second[1],
                            meanIter->second[2] };

  /// Get input dims incase batch array
  std::vector<int> imgDims; imgDims.reserve(3*batchSize);
  int inChannel = 3;
  int inHeight  = 0;
  int inWidth   = 0;

  const uint8_t* inData = nullptr;
  for (int b = 0; b < batchSize; ++b) {

    inData   = buffers[b];
    inHeight = tensors[b]->get_shape()[1];
    inWidth  = tensors[b]->get_shape()[2];

    auto imgPtr = (void*)const_cast<uint8_t*>(inData);
    /// Create a cv::Mat with input data
    cv::Mat inImage(inHeight, inWidth, CV_8UC3, imgPtr);

    /// Resize the image to Network Shape (LetterBox)
    cv::Mat resizedImage;
    cv::resize(inImage, resizedImage, cv::Size(outWidth, outHeight), 0);

    /// Pre-Processing loop
    float* batchData = reinterpret_cast<float*>(outTB->data({b}).first);
    if (outputLayout == "NCHW") {
      for (int c = 0; c < outChannels; c++) {
        for (int h = 0; h < outHeight; h++) {
          for (int w = 0; w < outWidth; w++) {
            batchData[(c*outHeight*outWidth) + (h*outWidth) + w]
              = float(resizedImage.at<cv::Vec3b>(h,w)[c]-mean[c]);
          }
        }
      }
    } else if (outputLayout == "NHWC"){
      for (int h = 0; h < outHeight; h++) {
        for (int w = 0; w < outWidth; w++) {
          for (int c = 0; c < outChannels; c++) {
            batchData[(h*outWidth*outChannels) + (w*outChannels) + c]
              = float(resizedImage.at<cv::Vec3b>(h,w)[c]-mean[c]);
          }
        }
      }
    }

    imgDims.insert(imgDims.end(), {inChannel, inHeight, inWidth});
  }
  /// Write back image shape to dynParams for PostProc
  dynParams->_intVectorParams["img_dims"] = std::move(imgDims);

  /// Push back output
  out.push_back(outTB);
  return 0;
}
