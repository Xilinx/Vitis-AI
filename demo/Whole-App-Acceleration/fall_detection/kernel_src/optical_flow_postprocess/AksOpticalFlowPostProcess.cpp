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
#include <atomic>

#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <aks/AksKernelBase.h>
#include <aks/AksNodeParams.h>
#include <aks/AksLogger.h>
#include <aks/AksTensorBuffer.h>

class OpticalFlowPostProcess : public AKS::KernelBase
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


void boundPixels(vart::TensorBuffer* src, cv::Mat& dst, int bound) {
  auto shape = src->get_tensor()->get_shape();
  int rows = shape[0];
  int cols = shape[1];
  float* srcPtr = reinterpret_cast<float*>(src->data().first);
  for (int i=0; i<rows; ++i) {
    for (int j=0; j<cols; ++j) {
      float x = srcPtr[i*cols+j];
      x = (x + bound) * (255.0 / (2 * bound));
      int y = cvRound(x);
      dst.at<uint8_t>(i, j) = (y>255 ? 255 : (y<0 ? 0 : y));
    }
  }
}


void mat2TensorBuffer(const cv::Mat &src, int8_t* dst) {
  // Gray scale cv::Mat --> Gray scale DataDescriptor
  int rows = src.rows;
  int cols = src.cols;
  int channels = src.channels();
  for (int i=0; i<rows; i++) {
    for (int j=0; j<cols; j++) {
      for (int k=0; k<channels; k++) {
        dst[(i*cols*channels) + (j*channels) + k] = \
            src.at<cv::Vec<uint8_t, 2>>(i,j)[k] - 128;
      }
    }
  }
}


extern "C" { // Add this to make this available for python bindings and

AKS::KernelBase* getKernel (AKS::NodeParams *params)
{
  return new OpticalFlowPostProcess();
}

void OpticalFlowPostProcess::nodeInit (AKS::NodeParams* nodeParams)
{
  output_folder = nodeParams->_stringParams.find("visualize") == nodeParams->_stringParams.end() ?
    "" : nodeParams->getValue<std::string>("visualize");
  bound = nodeParams->getValue<int>("bound");
}

int OpticalFlowPostProcess::exec_async (
      std::vector<vart::TensorBuffer*> &in,
      std::vector<vart::TensorBuffer*> &out,
      AKS::NodeParams* nodeParams,
      AKS::DynamicParamValues* dynParams)
{
    // in[0] contains flowx data
    // in[1] contains flowy data
    // std::cout << "[DBG] Starting OpticalFlowPostProcess... " << std::endl;
    auto shape = in[0]->get_tensor()->get_shape();
    int rows = shape.at(0);
    int cols = shape.at(1);

    cv::Mat boundedFlowX(rows, cols, CV_8UC1);
    cv::Mat boundedFlowY(rows, cols, CV_8UC1);

    float* flowxData = reinterpret_cast<float*>(in[0]->data().first);
    float* flowyData = reinterpret_cast<float*>(in[1]->data().first);

    boundPixels(in[0], boundedFlowX, bound);
    boundPixels(in[1], boundedFlowY, bound);

    cv::Mat flow[2] = { boundedFlowX, boundedFlowY };
    cv::Mat merged(rows, cols, CV_8UC(2));
    cv::merge(flow, 2, merged);

    auto flowDD = new AKS::AksTensorBuffer(
                  xir::Tensor::create(
                    "ofPostProc", {1, rows, cols, 2},
                    xir::create_data_type<char>()
                    // xir::DataType {xir::DataType::INT, 8u}
                  ));
    int8_t* flowData = reinterpret_cast<int8_t*>(flowDD->data().first);
    mat2TensorBuffer(merged, flowData);

    out.push_back(flowDD);

    if (!output_folder.empty()) {
        boost::filesystem::path p(dynParams->imagePaths.front());
        std::string filename = p.filename().string();
        std::string folder = p.parent_path().filename().string();
        std::string output_path = output_folder + "/" + folder;
        boost::filesystem::create_directories(output_path);
        std::string tmp = "_" + filename;
        cv::imwrite(output_path + "/flow_x" + tmp, boundedFlowX);
        cv::imwrite(output_path + "/flow_y" + tmp, boundedFlowY);
    }

    // std::cout << "[DBG] OpticalFlowPostProcess: Done!" << std::endl << std::endl;
    return -1; // No wait
}

} //extern "C"
