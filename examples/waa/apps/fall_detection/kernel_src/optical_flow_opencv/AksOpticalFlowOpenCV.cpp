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
#include <stdint.h>
#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video.hpp>

#include <aks/AksKernelBase.h>
#include <aks/AksNodeParams.h>
#include <aks/AksTensorBuffer.h>

class OpticalFlowOpenCV : public AKS::KernelBase
{
  public:
    void nodeInit (AKS::NodeParams*);
    int exec_async (
        std::vector<vart::TensorBuffer*> &in,
        std::vector<vart::TensorBuffer*> &out,
        AKS::NodeParams* nodeParams,
        AKS::DynamicParamValues* dynParams);
    void perform_preprocess(
        vart::TensorBuffer* &image, std::vector<vart::TensorBuffer*> &out,
        int outHeight, int outWidth);
  private:
    cv::Ptr<cv::DualTVL1OpticalFlow> alg_tvl1;
    std::string of_algo;
};


extern "C" { // Add this to make this available for python bindings

  AKS::KernelBase* getKernel (AKS::NodeParams *params)
  {
    return new OpticalFlowOpenCV();
  }

} //extern "C"

void OpticalFlowOpenCV::nodeInit (AKS::NodeParams* params)
{
  of_algo = params->getValue<string>("of_algorithm");
  if (strcmp(of_algo.c_str(), "DualTVL1") == 0) {
    alg_tvl1 = cv::DualTVL1OpticalFlow::create();
  }
}

int OpticalFlowOpenCV::exec_async (
    std::vector<vart::TensorBuffer*> &in,
    std::vector<vart::TensorBuffer*> &out,
    AKS::NodeParams* nodeParams,
    AKS::DynamicParamValues* dynParams)
{
  // in[0] contains  current image data
  // in[1] contains previous image data
  auto shape = in[0]->get_tensor()->get_shape();
  int rows = shape.at(1);
  int cols = shape.at(2);
  int channels = shape.at(3);
  assert(channels == 1);

  cv::Mat curr_gray(rows, cols, CV_8UC1, reinterpret_cast<void*>(in[0]->data().first));
  cv::Mat prev_gray(rows, cols, CV_8UC1, reinterpret_cast<void*>(in[1]->data().first));

  // curr_gray contains preprocessed  current image
  // prev_gray contains preprocessed previous image
  
  cv::Mat_<cv::Point2f> flow;
  if (strcmp(of_algo.c_str(), "Farneback") == 0) {
    cv::calcOpticalFlowFarneback(prev_gray, curr_gray, flow, 0.75, 5, 10, 5, 5, 1.2, 0);
  }
  else if (strcmp(of_algo.c_str(), "DualTVL1") == 0) {
    alg_tvl1->calc(prev_gray, curr_gray, flow);
  }
  else {
    std::cout << "[ERROR]: Parameter `of_algorithm` should be "
              << "either `Farneback` or `DualTVL1`" << std::endl;
    throw;
  }

  auto flowTB = new AKS::AksTensorBuffer(
                  xir::Tensor::create(
                    "OpenCVOF", {rows, cols, 2},
                    xir::create_data_type<float>()
                  ));
  float* flowData = reinterpret_cast<float*>(flowTB->data().first);

  std::memcpy(flowData, flow.data, rows*cols*2*sizeof(float));
  out.push_back(flowTB);

  return 0;
}


