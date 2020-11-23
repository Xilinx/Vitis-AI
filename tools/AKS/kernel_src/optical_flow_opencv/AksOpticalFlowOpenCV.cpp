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
#include <stdint.h>
#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video.hpp>

#include <aks/AksKernelBase.h>
#include <aks/AksDataDescriptor.h>
#include <aks/AksNodeParams.h>

class OpticalFlowOpenCV : public AKS::KernelBase
{
  public:
    void nodeInit (AKS::NodeParams*);
    int exec_async (
        std::vector<AKS::DataDescriptor*> &in,
        std::vector<AKS::DataDescriptor*> &out,
        AKS::NodeParams* nodeParams,
        AKS::DynamicParamValues* dynParams);
    void perform_preprocess(
        AKS::DataDescriptor* &image, std::vector<AKS::DataDescriptor*> &out,
        int outHeight, int outWidth);
  private:
    cv::Ptr<cv::DualTVL1OpticalFlow> alg_tvl1;
};


extern "C" { // Add this to make this available for python bindings

  AKS::KernelBase* getKernel (AKS::NodeParams *params)
  {
    return new OpticalFlowOpenCV();
  }

} //extern "C"

void OpticalFlowOpenCV::nodeInit (AKS::NodeParams* params)
{
  std::string of_algo = params->getValue<string>("of_algorithm");
  if (strcmp(of_algo.c_str(), "DualTVL1") == 0) {
    alg_tvl1 = cv::DualTVL1OpticalFlow::create();
  }
}

int OpticalFlowOpenCV::exec_async (
    std::vector<AKS::DataDescriptor*> &in, 
    std::vector<AKS::DataDescriptor*> &out, 
    AKS::NodeParams* nodeParams,
    AKS::DynamicParamValues* dynParams)
{
  // in[0] contains  current image data
  // in[1] contains previous image data
  std::string of_algo = \
                        nodeParams->_stringParams.find("of_algorithm") == nodeParams->_stringParams.end() ?
                        "Farneback" : nodeParams->_stringParams["of_algorithm"];
  auto shape = in[0]->getShape();
  int channels = shape[1];
  int rows = shape[2];
  int cols = shape[3];
  assert(channels == 1);

  cv::Mat curr_gray(rows, cols, CV_8UC1, in[0]->data());
  cv::Mat prev_gray(rows, cols, CV_8UC1, in[1]->data());

  // curr_gray contains preprocessed  current image
  // prev_gray contains preprocessed previous image
  cv::Mat flow(rows, cols, CV_32FC(2));
  if (strcmp(of_algo.c_str(), "Farneback") == 0) {
    cv::calcOpticalFlowFarneback(prev_gray, curr_gray, flow, 0.75, 5, 10, 5, 5, 1.2, 0);
  }
  else if (strcmp(of_algo.c_str(), "DualTVL1") == 0) {
    alg_tvl1->calc(prev_gray, curr_gray, flow);
  }
  else {
    std::cout << "[ERROR]: Parameter `of_algorithm` should be either `Farneback` or `DualTVL1`" << std::endl;
    throw;
  }

  AKS::DataDescriptor *FlowDD = new AKS::DataDescriptor(
      { rows, cols, 2 }, AKS::DataType::FLOAT32);  // channels = 2
  float* FlowData = static_cast<float*>(FlowDD->data());
  std::memcpy(FlowData, flow.data, rows*cols*2*sizeof(float));
  out.push_back(FlowDD);

  return 0;
}

