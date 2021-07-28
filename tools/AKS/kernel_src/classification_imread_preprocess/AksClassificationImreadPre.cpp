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
#include <opencv2/imgproc/imgproc.hpp>

#include <aks/AksTensorBuffer.h>
#include <aks/AksKernelBase.h>
#include <aks/AksNodeParams.h>

class ClassificationImreadPreKernel : public AKS::KernelBase
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
    return new ClassificationImreadPreKernel();
  }

}//extern "C"

int ClassificationImreadPreKernel::exec_async (
    std::vector<vart::TensorBuffer*> &in, 
    std::vector<vart::TensorBuffer*> &out, 
    AKS::NodeParams* nodeParams,
    AKS::DynamicParamValues* dynParams)
{
  int batchSize = dynParams->imagePaths.size();
  int outHeight = nodeParams->_intParams["net_h"];
  int outWidth  = nodeParams->_intParams["net_w"];
  string outputLayout = nodeParams->hasKey<string>("output_layout") ?
                        nodeParams->getValue<string>("output_layout"): "NCHW";

  /// Get mean values
  auto meanIter = nodeParams->_floatVectorParams.find("mean");
  float mean [3];
  mean[0] = meanIter->second[0]; 
  mean[1] = meanIter->second[1]; 
  mean[2] = meanIter->second[2]; 

  /// Create output data buffer for a batch data
  auto shape = (outputLayout == "NCHW") ?
    std::vector<int>{ batchSize, 3, outHeight, outWidth }:
    std::vector<int>{ batchSize, outHeight, outWidth, 3 };

  /// Create output Tensor buffer
  std::string tensorName ("pre-output");
  AKS::AksTensorBuffer * outTB = new AKS::AksTensorBuffer(
                                   xir::Tensor::create(
                                     tensorName, shape,
                                     xir::create_data_type<float>()
                                 ));

  float * outData = reinterpret_cast<float*>(outTB->data().first);

  const uint32_t nelemsPerImg = 3 * outHeight * outWidth;

  /// Load images and pre-process it.
  for(int i=0; i < batchSize; ++i) {
    cv::Mat inImage = cv::imread (dynParams->imagePaths[i].c_str());
    if (!inImage.data) {
      std::cerr << "[ERR] Unable to read image: " << dynParams->imagePaths[i] << std::endl;
      return -2;
    } 

    /// Resize the image to Network Shape
    cv::Mat resizedImage = cv::Mat(outHeight, outWidth, CV_8SC3);
    cv::resize(inImage, resizedImage, cv::Size(outWidth, outHeight));

    /// Pre-Processing loop
    float* batchData = outData + i * nelemsPerImg;
    if (outputLayout == "NCHW") {
      for (int c = 0; c < 3; c++)
        for (int h = 0; h < outHeight; h++)
          for (int w = 0; w < outWidth; w++) {
            batchData[(c*outHeight*outWidth)
              + (h*outWidth) + w]
              = resizedImage.at<cv::Vec3b>(h,w)[c]-mean[c];
          }
    } else if (outputLayout == "NHWC"){
      for (int h = 0; h < outHeight; h++)
        for (int w = 0; w < outWidth; w++)
          for (int c = 0; c < 3; c++)
            batchData[h*outWidth*3 + w*3 + c]
              = resizedImage.at<cv::Vec3b>(h,w)[c]-mean[c];
    }
  }

  /// Push back output
  out.push_back(outTB);
  return 0;
}

