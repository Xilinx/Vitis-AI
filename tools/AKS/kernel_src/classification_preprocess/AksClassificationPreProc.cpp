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

#include <aks/AksKernelBase.h>
#include <aks/AksDataDescriptor.h>
#include <aks/AksNodeParams.h>

class ClassificationPreProc : public AKS::KernelBase
{
  public:
    int exec_async (
        std::vector<AKS::DataDescriptor*> &in, 
        std::vector<AKS::DataDescriptor*> &out, 
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
    std::vector<AKS::DataDescriptor*> &in, 
    std::vector<AKS::DataDescriptor*> &out, 
    AKS::NodeParams* nodeParams,
    AKS::DynamicParamValues* dynParams)
{
  /// Get input and output data shapes
  /// Input could be batch array or batch of images
  const std::vector<int>& inShape = in[0]->getShape();
  int batchSize = inShape[0];

  /// Get output Dimensions (Network input dim)
  int outHeight    = nodeParams->_intParams["net_h"];
  int outWidth     = nodeParams->_intParams["net_w"];
  int outChannels  = 3;
  int nOutElemsPerImg = outChannels * outHeight * outWidth;

  /// Create output data buffer
  std::vector<int> shape      = { batchSize, outChannels, outHeight, outWidth };
  AKS::DataDescriptor * outDD = new AKS::DataDescriptor(shape, AKS::DataType::FLOAT32);
  float * outData = (float*) outDD->data();

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
    if(in[0]->dtype() == AKS::DataType::AKSDD) {
      auto& dd = in[0]->data<AKS::DataDescriptor>()[b];
      inData   = dd.data<uint8_t>();
      // Shape = (Batch=1, H, W, C=3)
      inHeight = dd.getShape()[1];
      inWidth  = dd.getShape()[2];
    }
    else {
      inHeight = inShape[1];
      inWidth  = inShape[2];
      int nInElemsPerImg  = b * inChannel * inHeight * inWidth;
      inData   = in[0]->data<uint8_t>() + nInElemsPerImg;
    }

    /// Create a cv::Mat with input data
    cv::Mat inImage(inHeight, inWidth, CV_8UC3, inData);

    /// Resize the image to Network Shape
    cv::Mat resizedImage = cv::Mat(outHeight, outWidth, CV_8SC3);
    cv::resize(inImage, resizedImage, cv::Size(outWidth, outHeight));

    /// Pre-Processing loop
    float* out = outData + b * nOutElemsPerImg;
    if (1 /* TODO: Insert correct condition */ ) {
      for (int c = 0; c < 3; c++)
        for (int h = 0; h < outHeight; h++)
          for (int w = 0; w < outWidth; w++) {
            out[(c*outHeight*outWidth)
              + (h*outWidth) + w]
              = resizedImage.at<cv::Vec3b>(h,w)[c]-mean[c];
          }
    } else {
      for (int h = 0; h < outHeight; h++)
        for (int w = 0; w < outWidth; w++)
          for (int c = 0; c < 3; c++)
            out[h*outWidth*3 + w*3 + c]
              = resizedImage.at<cv::Vec3b>(h,w)[c]-mean[c];
    }
  }

  /// Push back output
  out.push_back(outDD);
  return 0;
}

