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

class DetectionImreadPre : public AKS::KernelBase
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
    return new DetectionImreadPre();
  }

}//extern C

void embedImage(cv::Mat& source, cv::Mat& dest, int dx, int dy)
{
  float * srcData = (float*)source.data;
  float * dstData = (float*)dest.data;
  unsigned int dstStart;
  /// Fill image data
  unsigned int size = source.rows * source.cols * source.channels();
  if (dx == 0) {
    /// Horizontal letterbox (source.cols == dst.cols)
    unsigned int dstStart = source.cols * dy * source.channels();
    for (unsigned int i = 0; i < size; ++i) {
      dstData[dstStart+i] = srcData[i];
    }
  } else {
    /// Vertial letterbox (source.rows == dst.rows)
    int moveBytes = source.cols * source.channels() * sizeof(float);
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

  /// Find max dim
  if (((float)resizeW / inImage.cols) < ((float)resizeH / inImage.rows)) {
    new_w = resizeW;
    new_h = (inImage.rows * resizeW) / inImage.cols;
  } else {
    new_h = resizeH;
    new_w = (inImage.cols * resizeH) / inImage.rows;
  }
  /// Resize image (keeping aspect ratio)
  cv::Mat resizedImage = cv::Mat(new_h, new_w, CV_8UC3);
  cv::resize(inImage, resizedImage, cv::Size(new_w, new_h));

  /// Convert image from BGR to RGB
  cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2RGB);

  /// Normalize image 
  cv::Mat scaledImage = cv::Mat(new_h, new_w, CV_32FC3); 
  resizedImage.convertTo(scaledImage, CV_32FC3, 1/255.0);

  /// Fill output image with 0.5 (for letterbox)
  outImage.setTo(cv::Scalar(0.5f, 0.5f, 0.5f));
  embedImage(scaledImage, outImage, (resizeW-new_w)/2, (resizeH-new_h)/2);
}


int DetectionImreadPre::exec_async (
    std::vector<vart::TensorBuffer*> &in,
    std::vector<vart::TensorBuffer*> &out,
    AKS::NodeParams* nodeParams,
    AKS::DynamicParamValues* dynParams)
{
  /// Get output Dimensions (Network input dim)
  int outHeight = nodeParams->_intParams["net_h"];
  int outWidth  = nodeParams->_intParams["net_w"];
  int outChannels = 3;
  int batchSize = dynParams->imagePaths.size();
  int nelemsPerImg = outChannels * outHeight * outWidth;

  std::string outputLayout = nodeParams->hasKey<string>("output_layout") ?
                             nodeParams->getValue<string>("output_layout"):
                             "NCHW";

  /// Create output data buffer
  auto shape = (outputLayout == "NCHW") ?
    std::vector<int>{ batchSize, 3, outHeight, outWidth }:
    std::vector<int>{ batchSize, outHeight, outWidth, 3 };

  std::string tensorName ("det-im-pre-out");
  AKS::AksTensorBuffer * outTB = new AKS::AksTensorBuffer(
                                   xir::Tensor::create(
                                     tensorName, shape,
                                     xir::create_data_type<float>()
                                 ));
  float * outData = reinterpret_cast<float*>(outTB->data().first);

  std::vector<int> imgDims; imgDims.reserve(3*batchSize);
  for(int i = 0; i < batchSize; ++i) {
    cv::Mat inImage = cv::imread(dynParams->imagePaths[i]);
    if (!inImage.data) {
      std::cerr << "[ERR] Unable to read image: " 
                << dynParams->imagePaths[0] << std::endl;
      return -2;
    }

    /// Get input and output data shapes
    int inChannel = inImage.channels();
    int inHeight  = inImage.rows;
    int inWidth   = inImage.cols;

    /// Resize the image to Network Shape (LetterBox)
    cv::Mat outImage = cv::Mat(outHeight, outWidth, CV_32FC3);
    /// Crop Letter-Box
    letterBoxImage(inImage, outHeight, outWidth, outImage);

    /// Transpose: HWC-->CHW
    float* batchData = outData + i * nelemsPerImg;
    if (outputLayout == "NCHW") {
      for (int c = 0; c < 3; c++) {
        for (int h = 0; h < outHeight; h++) {
          for (int w = 0; w < outWidth; w++) {
            batchData[(c*outHeight*outWidth) + (h*outWidth) + w]
              = outImage.at<cv::Vec3f>(h,w)[c];
          }
        }
      }
    } else if (outputLayout == "NHWC") {
      for (int h = 0; h < outHeight; h++) {
        for (int w = 0; w < outWidth; w++) {
          for (int c = 0; c < 3; c++) {
            batchData[(h*outWidth*3) + (w*3) + c]
              = outImage.at<cv::Vec3f>(h,w)[c];
          }
        }
      }
    }

    imgDims.insert(imgDims.end(), {inChannel, inHeight, inWidth});
  }

  /// Write back image shape to dynParams for PostProc
  dynParams->_intVectorParams["img_dims"] = std::move(imgDims);

  /// Push back output
  out.push_back(outTB);
  return 0;
}
