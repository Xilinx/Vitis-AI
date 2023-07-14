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

#include <aks/AksTensorBuffer.h>
#include <aks/AksKernelBase.h>
#include <aks/AksNodeParams.h>

class FaceDetectImreadPre : public AKS::KernelBase
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
    return new FaceDetectImreadPre();
  }

}//extern "C"

int FaceDetectImreadPre::exec_async (
  std::vector<vart::TensorBuffer*> &in, std::vector<vart::TensorBuffer*> &out,
  AKS::NodeParams* nodeParams, AKS::DynamicParamValues* dynParams)
{
  int outHeight = nodeParams->_intParams["net_h"];
  int outWidth  = nodeParams->_intParams["net_w"];
  int outChannel= nodeParams->_intParams["net_c"];
  int batchSize = dynParams->imagePaths.size();
  string outputLayout = nodeParams->hasKey<string>("output_layout") ?
                        nodeParams->getValue<string>("output_layout"):
                        "NCHW";

  /// Get mean values
  auto meanIter = nodeParams->_floatVectorParams.find("mean");
  std::vector<float> mean { meanIter->second[0],
                            meanIter->second[1],
                            meanIter->second[2] };

  /// Create output data buffer for a batch data
  auto shape = (outputLayout == "NCHW") ?
    std::vector<int>{ batchSize, outChannel, outHeight, outWidth }:
    std::vector<int>{ batchSize, outHeight, outWidth, outChannel };

  AKS::AksTensorBuffer * outTB = new AKS::AksTensorBuffer(
                                   xir::Tensor::create(
                                     "face-preproc", shape,
                                     xir::create_data_type<float>()
                                 ));
  float * outData = reinterpret_cast<float*>(outTB->data().first);

  const uint32_t nelemsPerImg = outChannel * outHeight * outWidth;
  /// Load images and pre-process it.
  for(int i = 0; i < batchSize; ++i) {

    cv::Mat inImage = cv::imread (dynParams->imagePaths[i].c_str());
    if (!inImage.data) {
      std::cerr << "[ERR] Unable to read image: " << dynParams->imagePaths[i] << std::endl;
      return -2;
    }

    cv::Mat resizedImage;
    cv::resize(inImage, resizedImage, cv::Size(outWidth, outHeight), 0);

    /// Pre-Processing loop
    float* batchData = outData + i * nelemsPerImg;
    if (outputLayout == "NCHW") {
      for (int c = 0; c < outChannel; c++) {
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
          for (int c = 0; c < outChannel; c++) {
            batchData[(h*outWidth*outChannel) + (w*outChannel) + c]
              = float(resizedImage.at<cv::Vec3b>(h,w)[c]-mean[c]);
          }
        }
      }
    }
  }

  /// Push back output
  out.push_back(outTB);
  return 0;
}
