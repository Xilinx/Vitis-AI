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

class ClassificationMeanSubtract : public AKS::KernelBase
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
    return new ClassificationMeanSubtract();
  }

}//extern "C"

int ClassificationMeanSubtract::exec_async (
    std::vector<vart::TensorBuffer*> &in, 
    std::vector<vart::TensorBuffer*> &out, 
    AKS::NodeParams* nodeParams,
    AKS::DynamicParamValues* dynParams)
{
  int outHeight  = nodeParams->_intParams["net_h"];
  int outWidth   = nodeParams->_intParams["net_w"];
  int outChannel = nodeParams->_intParams["net_c"];
  int batchSize  = in[0]->get_tensor()->get_shape()[0];

  std::string outputLayout = 
    nodeParams->hasKey<string>("output_layout") ?
    nodeParams->getValue<string>("output_layout") : 
    "NCHW";

  /// Get mean values
  auto meanIter = nodeParams->_floatVectorParams.find("mean");
  float mean [] = { meanIter->second[0],
                    meanIter->second[1],
                    meanIter->second[2] };

  /// Create output data buffer for a batch data
  auto shape = (outputLayout == "NCHW") ?
    std::vector<int>{ batchSize, outChannel, outHeight, outWidth }:
    std::vector<int>{ batchSize, outHeight, outWidth, outChannel };

  std::string out_tensor_name ("mean-sub");
  AKS::AksTensorBuffer * outTB = new AKS::AksTensorBuffer(
                                   xir::Tensor::create(
                                     out_tensor_name, shape, 
                                     xir::create_data_type<float>()
                                 ));
  float * outData = reinterpret_cast<float*>(outTB->data().first);

  const uint32_t nelemsPerImg = outChannel * outHeight * outWidth;

  /// Load images and pre-process it.
  for(int i = 0; i < batchSize; ++i) {

    int nInElemsPerImg = i * nelemsPerImg;
    uint8_t* inData = reinterpret_cast<uint8_t*>(in[0]->data().first) 
                      + nInElemsPerImg;
    cv::Mat image (outHeight, outWidth, CV_8UC3, inData);

    /// Pre-Processing loop
    float* batchData = outData + i * nelemsPerImg;
    if (outputLayout == "NCHW") {
      for (int c = 0; c < outChannel; c++) {
        for (int h = 0; h < outHeight; h++) {
          for (int w = 0; w < outWidth; w++) {
            batchData[(c*outHeight*outWidth) + (h*outWidth) + w]
              = float(image.at<cv::Vec3b>(h,w)[c]-mean[c]);
          }
        }
      }
    } else if (outputLayout == "NHWC"){
      for (int h = 0; h < outHeight; h++) {
        for (int w = 0; w < outWidth; w++) {
          for (int c = 0; c < outChannel; c++) {
            batchData[(h*outWidth*outChannel) + (w*outChannel) + c]
              = float(image.at<cv::Vec3b>(h,w)[c]-mean[c]);
          }
        }
      }
    }
  }

  /// Push back output
  out.push_back(outTB);
  return 0;
}

