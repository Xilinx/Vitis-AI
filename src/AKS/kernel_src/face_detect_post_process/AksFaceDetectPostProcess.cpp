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
#include <assert.h>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <aks/AksTensorBuffer.h>
#include <aks/AksKernelBase.h>
#include <aks/AksNodeParams.h>

#include "AksFaceDetectPostProcess.hpp"

class FaceDetectPostProc : public AKS::KernelBase
{
  private:
    int _stride;
  public:
    void nodeInit(AKS::NodeParams*);
    int exec_async (
        std::vector<vart::TensorBuffer*> &in,
        std::vector<vart::TensorBuffer*> &out,
        AKS::NodeParams* nodeParams,
        AKS::DynamicParamValues* dynParams);
};

extern "C" {

  AKS::KernelBase* getKernel (AKS::NodeParams *params)
  {
    return new FaceDetectPostProc();
  }

}//extern C

void FaceDetectPostProc::nodeInit(AKS::NodeParams* nodeParams) {
  _stride = 8;
}

int FaceDetectPostProc::exec_async (
  std::vector<vart::TensorBuffer*> &in, std::vector<vart::TensorBuffer*> &out,
  AKS::NodeParams* nodeParams, AKS::DynamicParamValues* dynParams)
{
  int input_batch = in[0]->get_tensor()->get_shape()[0];
  int stride = 8;
  if (nodeParams->hasKey<int>("stride"))
    stride = nodeParams->getValue<int>("stride");
  assert(stride == 8);
  int stride_sq = stride * stride;

  bool do_gstiling = false;
  if (nodeParams->hasKey<int>("gs_tiling"))
    do_gstiling = static_cast<bool>(nodeParams->getValue<int>("gs_tiling"));

  auto tensor0_channel = in[0]->get_tensor()->get_shape()[3];
  auto tensor1_channel = in[1]->get_tensor()->get_shape()[3];

  if (do_gstiling) {
    tensor0_channel = in[0]->get_tensor()->get_shape()[3] / stride_sq;
    tensor1_channel = in[1]->get_tensor()->get_shape()[3] / stride_sq;
  }

  int pc_idx = 0;
  int bb_idx = 1;
  if (tensor0_channel == 2 && tensor1_channel == 4) {
    pc_idx = 0; bb_idx = 1;
  } else
  if (tensor0_channel == 4 && tensor1_channel == 2) {
    pc_idx = 1; bb_idx = 0;
  } else {
    std::cout
      << "Input Tensors channel error: "
      << "Tensor 0 channel is " << tensor0_channel
      << " "
      << "Tensor 1 channel is " << tensor1_channel
      << " "
      << std::endl;
  }

  /// Assumes input and NHWC format
  auto bbShape = in[bb_idx]->get_tensor()->get_shape();
  //# Read HW bb output params
  auto bb_in_channel = bbShape[3];
  auto bb_in_height  = bbShape[1];
  auto bb_in_width   = bbShape[2];
  auto bb_in_size    = std::accumulate(
                       std::next(bbShape.begin()), bbShape.end(),
                       1, std::multiplies<int>());

  int bb_out_channel;
  int bb_out_height;
  int bb_out_width;
  int bb_tiling_size;

  if (do_gstiling) {
    bb_out_channel = bb_in_channel / stride_sq;
    bb_out_height  = bb_in_height * stride;
    bb_out_width   = bb_in_width * stride;
    bb_tiling_size = bb_out_channel * bb_out_width * bb_out_height;
  } else {
    bb_out_channel = bb_in_channel;
    bb_out_height  = bb_in_height;
    bb_out_width   = bb_in_width;
    bb_tiling_size = bb_in_size;
  }

  /// Assumes input and NHWC format
  auto pcShape = in[pc_idx]->get_tensor()->get_shape();
  //# Read HW pixel conv output params
  auto pc_in_channel = pcShape[3];
  auto pc_in_height  = pcShape[1];
  auto pc_in_width   = pcShape[2];
  auto pc_in_size    = std::accumulate(
                       std::next(pcShape.begin()), pcShape.end(),
                       1, std::multiplies<int>());

  int pc_out_channel;
  int pc_out_height;
  int pc_out_width;
  int pc_tiling_size;

  if (do_gstiling) {
    pc_out_channel = pc_in_channel / stride_sq;
    pc_out_height  = pc_in_height * stride;
    pc_out_width   = pc_in_width * stride;
    pc_tiling_size = pc_out_channel * pc_out_height * pc_out_width;
  } else {
    pc_out_channel = pc_in_channel;
    pc_out_height  = pc_in_height;
    pc_out_width   = pc_in_width;
    pc_tiling_size = pc_in_size;
  }

  /// buffers for gst tiling o/p: Bounding Box
  auto gst_tiling_bb = std::make_unique<float[]>(bb_tiling_size);
  assert (gst_tiling_bb != nullptr);
  /// buffers for gst tiling o/p: Pixel Conv
  auto gst_tiling_pc = std::make_unique<float[]>(pc_tiling_size);
  assert (gst_tiling_pc != nullptr);

  /// Scales
  constexpr float bb_scale = 1.0;
  constexpr float pc_scale = 1.0;
  /// Thresholds
  constexpr float det_threshold = 0.7;
  constexpr float nms_threshold = 0.3;

  /// all boxes
  std::vector<float> batch_output;
  /// Process batch
  for (int b = 0; b < input_batch; ++b) {

    auto bb_out = reinterpret_cast<float*>(in[bb_idx]->data({b}).first);
    auto pc_out = reinterpret_cast<float*>(in[pc_idx]->data({b}).first);

    if (do_gstiling) {
      //# perform GSTiling on bb_out
      GSTilingLayer (bb_out, input_batch, bb_in_channel,
          bb_in_height, bb_in_width, bb_out_channel,
          stride, bb_out_height, bb_out_width,
          gst_tiling_bb.get()
      );
      //# perform GSTiling on pixel_conv
      GSTilingLayer (pc_out, input_batch, pc_in_channel,
          pc_in_height, pc_in_width, pc_out_channel,
          stride, pc_out_height, pc_out_width,
          gst_tiling_pc.get()
      );
      bb_out = gst_tiling_bb.get();
      pc_out = gst_tiling_pc.get();
    }

    vector<float> conf(pc_tiling_size);
    /// Perform SoftMax
    softmax (pc_out, pc_scale, pc_out_channel, pc_tiling_size / pc_out_channel, conf.data());

    /// Filter Boxes
    vector<vector<float>> boxes = FilterBox (bb_scale, det_threshold,
                                    bb_out, bb_out_width,
                                    bb_out_height, conf.data()
                                  );

    std::vector<std::vector<float>> results;
    /// Perform NMS
    NMS(nms_threshold, boxes, results);

    /// flatten results
    for (auto& result: results) {
      batch_output.push_back(b);
      for (auto r: result) {
        batch_output.push_back(r);
      }
    }

#ifdef DEBUG
    displayResults(results, dynParams->imagePaths[b], b);
#endif
  }
  // o/p format: [batch, x, y, sx, sy, score, ...]
  int out_tensor_size = batch_output.size();
  /// Create output tensor buffer and fill results
  AKS::AksTensorBuffer * outTB =
    new AKS::AksTensorBuffer(xir::Tensor::create(
      "face-detect-out", {out_tensor_size},
      xir::create_data_type<float>())
    );
  auto * tbPtr = reinterpret_cast<float*>(outTB->data().first);
  std::memcpy(tbPtr, batch_output.data(), outTB->get_tensor()->get_data_size());

  /// Return
  out.push_back(outTB);
  return 0;
}
