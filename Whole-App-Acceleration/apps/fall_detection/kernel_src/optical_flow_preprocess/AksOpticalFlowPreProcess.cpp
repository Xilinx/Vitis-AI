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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <xir/tensor/tensor.hpp>
#include <xir/util/data_type.hpp>

#include <aks/AksKernelBase.h>
#include <aks/AksNodeParams.h>
#include <aks/AksLogger.h>
#include <aks/AksTensorBuffer.h>

class OpticalFlowPreProcess : public AKS::KernelBase
{
  public:
    int exec_async (
        std::vector<vart::TensorBuffer*> &in,
        std::vector<vart::TensorBuffer*> &out,
        AKS::NodeParams* nodeParams,
        AKS::DynamicParamValues* dynParams);
    void letterBoxImage(cv::Mat &inImage, cv::Mat& outImage);
    void perform_preprocess(
        vart::TensorBuffer* inputTensorBuffer,
        std::vector<vart::TensorBuffer*> &out);
    void nodeInit(AKS::NodeParams* params);
  private:
    std::atomic<int> counter = 0;
    int outHeight;
    int outWidth;
};


extern "C" { // Add this to make this available for python bindings and

  AKS::KernelBase* getKernel (AKS::NodeParams *params)
  {
    return new OpticalFlowPreProcess();
  }

} //extern "C"

void embedImage(cv::Mat& source, cv::Mat& dest, int dx, int dy)
{
  uint8_t * srcData = (uint8_t*)source.data;
  uint8_t * dstData = (uint8_t*)dest.data;
  unsigned int dstStart;
  // Fill image data
  unsigned int size = source.rows * source.cols * source.channels();
  if (dx == 0) {
    // Horizontal letterbox (source.cols == dst.cols)
    unsigned int dstStart = source.cols * dy * source.channels();
    for (unsigned int i = 0; i < size; ++i) {
      dstData[dstStart+i] = srcData[i];
    }
  } else {
    // Vertial letterbox (source.rows == dst.rows)
    int moveBytes = source.cols * source.channels() * sizeof(uint8_t);
    for (int rows = 0; rows < source.rows; ++rows) {
      unsigned long srcOffset = rows * source.cols * source.channels();
      unsigned long dstOffset = rows * dest.cols * dest.channels() + (dx * dest.channels());
      memcpy(dstData + dstOffset, srcData+srcOffset, moveBytes);
    }
  }
}

void OpticalFlowPreProcess::nodeInit(AKS::NodeParams* params) {
  outHeight = params->getValue<int>("net_h");
  outWidth = params->getValue<int>("net_w");
}

void OpticalFlowPreProcess::letterBoxImage(cv::Mat &inImage, cv::Mat& outImage)
{
  int new_w = inImage.cols;
  int new_h = inImage.rows;

  // Find max dim
  if (((float)outWidth / inImage.cols) < ((float)outHeight / inImage.rows)) {
    new_w = outWidth;
    new_h = (inImage.rows * outWidth) / inImage.cols;
  } else {
    new_h = outHeight;
    new_w = (inImage.cols * outHeight) / inImage.rows;
  }

  // Resize image (keeping aspect ratio)
  cv::Mat resizedImage = cv::Mat(new_h, new_w, CV_8UC3);
  cv::resize(inImage, resizedImage, cv::Size(new_w, new_h));

  // Fill output image with 0 (for letterbox)
  outImage.setTo(cv::Scalar(0, 0, 0));
  embedImage(resizedImage, outImage, (outWidth-new_w)/2, (outHeight-new_h)/2);
}


void OpticalFlowPreProcess::perform_preprocess(
    vart::TensorBuffer* tensorBuffer, std::vector<vart::TensorBuffer*> &out) {
  auto inShape = tensorBuffer->get_tensor()->get_shape();
  int inHeight = inShape.at(1);
  int inWidth = inShape.at(2);
  int inChannel = inShape.at(3);
  uint8_t* inData = reinterpret_cast<uint8_t*>(tensorBuffer->data().first);
  cv::Mat inImage(inHeight, inWidth, CV_8UC3, inData);

  // Resize the image to Network Shape (LetterBox)
  cv::Mat outImage = cv::Mat(outHeight, outWidth, CV_8UC3);
  // Crop Letter-Box
  letterBoxImage(inImage, outImage);
  // cv::resize(inImage, outImage, cv::Size(inWidth, inHeight));

  // BGR --> Gray
  cv::Mat grayImage(outHeight, outWidth, CV_8UC1);
  cv::cvtColor(outImage, grayImage, cv::COLOR_BGR2GRAY);
  // Create output data buffer
  auto outDD = new AKS::AksTensorBuffer(
                xir::Tensor::create(
                  "of_preproc", { 1, outHeight, outWidth, 1 },
                  xir::create_data_type<unsigned char>()
                ));
  uint8_t* outData = reinterpret_cast<uint8_t*>(outDD->data().first);
  std::memcpy(outData, grayImage.data, outHeight*outWidth*sizeof(uint8_t));
  out.push_back(outDD);
}

int OpticalFlowPreProcess::exec_async (
    std::vector<vart::TensorBuffer*> &in,
    std::vector<vart::TensorBuffer*> &out,
    AKS::NodeParams* nodeParams,
    AKS::DynamicParamValues* dynParams)
{
  // in[0] contains current image data
  // in[1] contains previous image data
  // std::cout << "[DEBUG] Starting OpticalFlowPreProcess.." << std::endl;
  out.reserve(2);
  perform_preprocess(in[1], out);
  perform_preprocess(in[0], out);
  // std::cout << "[DEBUG] Finished OpticalFlowPreProcess.." << std::endl;
  return 0;
}

