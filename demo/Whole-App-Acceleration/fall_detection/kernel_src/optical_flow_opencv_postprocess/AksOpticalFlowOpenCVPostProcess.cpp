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
#include <vector>

#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <aks/AksKernelBase.h>
#include <aks/AksNodeParams.h>
#include <aks/AksTensorBuffer.h>

class OpticalFlowOpenCVPostProcess : public AKS::KernelBase
{
  public:
    void nodeInit (AKS::NodeParams*);
    int exec_async (
        std::vector<vart::TensorBuffer*> &in,
        std::vector<vart::TensorBuffer*> &out,
        AKS::NodeParams* nodeParams,
        AKS::DynamicParamValues* dynParams);
  private:
    int bound;
    std::string output_folder;
};

void boundAndFormat(vart::TensorBuffer* src, int bound, int8_t* dst) {
  auto shape = src->get_tensor()->get_shape();
  int rows = shape[0];
  int cols = shape[1];
  int channels = shape[2];
  assert(channels == 2);
  float* srcPtr = reinterpret_cast<float*>(src->data().first);

  for (int i=0; i<rows; i++) {
    for (int j=0; j<cols; j++) {
      for (int k=0; k<channels; k++) {
        float x = srcPtr[i*cols*channels + j*channels + k];  // HWC format
        // Normalize pixel value and clip it to [0, 255]
        x = (x + bound) * (255.0 / (2 * bound));
        int y = cvRound(x);
        // HWC --> CHW
        dst[(i*cols*channels) + (j*channels) + k] = (y>255 ? 255 : (y<0 ? 0 : y)) - 128;
      }
    }
  }
}

void tensorBuffer2Mat(vart::TensorBuffer* src, cv::Mat &dst) {
  // Gray DataDescriptor --> Gray cv::Mat
  auto shape = src->get_tensor()->get_shape();
  int channels = shape[1];
  assert(channels == 2);
  int rows = shape[2];
  int cols = shape[3];
  int8_t* srcData = reinterpret_cast<int8_t*>(src->data().first);
  for (int k=0; k<channels; k++) {
    for (int i=0; i<rows; i++) {
      for (int j=0; j<cols; j++) {
        dst.at<cv::Vec<uint8_t, 2>>(i,j)[k] = srcData[(k*rows*cols) + (i*cols) + j] + 128;
      }
    }
  }
}


extern "C" { // Add this to make this available for python bindings

  AKS::KernelBase* getKernel (AKS::NodeParams *params)
  {
    return new OpticalFlowOpenCVPostProcess();
  }

} //extern "C"

void OpticalFlowOpenCVPostProcess::nodeInit (AKS::NodeParams* params)
{
  bound = params->getValue<int>("bound");
  output_folder = \
      params->_stringParams.find("visualize") == params->_stringParams.end() ?
      "" : params->getValue<string>("visualize");
}

int OpticalFlowOpenCVPostProcess::exec_async (
    std::vector<vart::TensorBuffer*> &in,
    std::vector<vart::TensorBuffer*> &out,
    AKS::NodeParams* nodeParams,
    AKS::DynamicParamValues* dynParams)
{

  auto shape = in[0]->get_tensor()->get_shape();
  int rows = shape.at(0);
  int cols = shape.at(1);
  int channels = shape.at(2); // should be 2

  auto flowTB = new AKS::AksTensorBuffer(
                  xir::Tensor::create(
                    "OpenCVOFPostProc", {1, rows, cols, 2},
                    xir::create_data_type<char>()
                  ));
  int8_t* flowData = reinterpret_cast<int8_t*>(flowTB->data().first);
  boundAndFormat(in[0], bound, flowData);
  out.push_back(flowTB);

  if (!output_folder.empty()) {
    boost::filesystem::path p(dynParams->imagePaths[0]);
    std::string filename = p.filename().string();
    std::string folder = p.parent_path().filename().string();
    std::string output_path = output_folder + "/" + folder;
    boost::filesystem::create_directories(output_path);
    std::string tmp = "_" + filename;
    cv::Mat image(rows, cols, CV_8UC(2));
    cv::Mat split[2];
    tensorBuffer2Mat(out[0], image);
    cv::split(image, split);
    cv::imwrite(output_path + "/flow_x" + tmp, split[0]);
    cv::imwrite(output_path + "/flow_y" + tmp, split[1]);
  }

  return 0;
}


