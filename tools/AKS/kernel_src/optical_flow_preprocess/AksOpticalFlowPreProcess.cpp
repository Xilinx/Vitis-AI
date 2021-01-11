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
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <aks/AksKernelBase.h>
#include <aks/AksDataDescriptor.h>
#include <aks/AksNodeParams.h>

class OpticalFlowPreProcess : public AKS::KernelBase
{
  public:
    int exec_async (
        std::vector<AKS::DataDescriptor*> &in, 
        std::vector<AKS::DataDescriptor*> &out, 
        AKS::NodeParams* nodeParams,
        AKS::DynamicParamValues* dynParams);
    void perform_preprocess(
        AKS::DataDescriptor* &image, std::vector<AKS::DataDescriptor*> &out,
        int outHeight, int outWidth);
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

void letterBoxImage(cv::Mat &inImage, int resizeH, int resizeW, cv::Mat& outImage)
{
  int new_w = inImage.cols;
  int new_h = inImage.rows;

  // Find max dim
  if (((float)resizeW / inImage.cols) < ((float)resizeH / inImage.rows)) {
    new_w = resizeW;
    new_h = (inImage.rows * resizeW) / inImage.cols;
  } else {
    new_h = resizeH;
    new_w = (inImage.cols * resizeH) / inImage.rows;
  }
  // Resize image (keeping aspect ratio)
  cv::Mat resizedImage = cv::Mat(new_h, new_w, CV_8UC3);
  cv::resize(inImage, resizedImage, cv::Size(new_w, new_h));
  // Fill output image with 0.5 (for letterbox)
  outImage.setTo(cv::Scalar(0, 0, 0));
  embedImage(resizedImage, outImage, (resizeW-new_w)/2, (resizeH-new_h)/2);
}


void OpticalFlowPreProcess::perform_preprocess(
    AKS::DataDescriptor* &image, std::vector<AKS::DataDescriptor*> &out,
    int outHeight, int outWidth) {

  auto inShape = image->getShape();
  int inChannel = inShape[1];
  int inHeight = inShape[2];
  int inWidth = inShape[3];
  // Create a cv::Mat with input data
  cv::Mat inImage(inHeight, inWidth, CV_8UC3, image->data());

  // Resize the image to Network Shape (LetterBox)
  cv::Mat outImage = cv::Mat(outHeight, outWidth, CV_8UC3);
  // Crop Letter-Box
  letterBoxImage(inImage, outHeight, outWidth, outImage);

  // BGR --> Gray
  cv::Mat grayImage(outHeight, outWidth, CV_8UC1);
  cv::cvtColor(outImage, grayImage, cv::COLOR_BGR2GRAY);
  // Create output data buffer
  std::vector<int> shape = { 1, 1, outHeight, outWidth };
  AKS::DataDescriptor *outDD = new AKS::DataDescriptor(shape, AKS::DataType::UINT8);
  uint8_t *outData = (uint8_t*) outDD->data();

  std::memcpy(outData, grayImage.data, outHeight*outWidth*sizeof(uint8_t));
  out.push_back(outDD);
}

int OpticalFlowPreProcess::exec_async (
    std::vector<AKS::DataDescriptor*> &in, 
    std::vector<AKS::DataDescriptor*> &out, 
    AKS::NodeParams* nodeParams,
    AKS::DynamicParamValues* dynParams)
{
  // in[0] contains current image data
  // in[1] contains previous image data

  // out[0] contains preprocessed current image
  // out[1] contains preprocessed previous image

  int outHeight = nodeParams->_intParams["net_h"];
  int outWidth  = nodeParams->_intParams["net_w"];

  out.reserve(2);
  perform_preprocess(in[0], out, outHeight, outWidth);
  perform_preprocess(in[1], out, outHeight, outWidth);

  return 0;
}

