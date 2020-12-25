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

#include <aks/AksKernelBase.h>
#include <aks/AksDataDescriptor.h>
#include <aks/AksNodeParams.h>

#include "yolo.h"

class YoloPostProc : public AKS::KernelBase
{
  private:
    int _yolo_version;
    int _net_h;
    int _net_w;
    int _num_classes;
    int _anchor_cnt;
    float _conf_thresh;
    float _iou_thresh;
    std::vector<float> _biases;
  public:
    void nodeInit(AKS::NodeParams*);
    int exec_async (
        std::vector<AKS::DataDescriptor*> &in,
        std::vector<AKS::DataDescriptor*> &out,
        AKS::NodeParams* nodeParams,
        AKS::DynamicParamValues* dynParams);
};

extern "C" { /// Add this to make this available for python bindings and

  AKS::KernelBase* getKernel (AKS::NodeParams *params)
  {
    return new YoloPostProc();
  }

}//externC

void YoloPostProc::nodeInit(AKS::NodeParams* nodeParams) {
  _yolo_version = nodeParams->_intParams["yolo_version"];
  _net_h        = nodeParams->_intParams["net_h"];
  _net_w        = nodeParams->_intParams["net_w"];
  _num_classes  = nodeParams->_intParams["num_classes"];
  _anchor_cnt   = nodeParams->_intParams["anchor_cnt"];
  _conf_thresh  = nodeParams->_floatParams["conf_thresh"];
  _iou_thresh   = nodeParams->_floatParams["iou_thresh"];
  _biases       = nodeParams->_floatVectorParams["biases"];
  assert(_yolo_version == 2 || _yolo_version == 3);
}

int YoloPostProc::exec_async (
    std::vector<AKS::DataDescriptor*> &in,
    std::vector<AKS::DataDescriptor*> &out,
    AKS::NodeParams* nodeParams,
    AKS::DynamicParamValues* dynParams)
{
  int batchSize = in[0]->getShape()[0];
  int imgDims = 3;
  out.push_back(new AKS::DataDescriptor({batchSize}, AKS::DataType::AKSDD));

  // Get original image dimensions
  std::vector<int>& imgShape = dynParams->_intVectorParams.at("img_dims");

  for(int b=0; b<batchSize; ++b) {
    int img_h = imgShape[b*imgDims + 1];
    int img_w = imgShape[b*imgDims + 2];

    // 2...N inputs are coming from previous kernel (either Caffe or fpga)
    int nArrays = in.size();
    std::vector<Array> inputArrays;
    for(int i=0; i<in.size(); ++i) {
      auto inDD = in[i];
      auto& shape = inDD->getShape();
      int nelems = shape[1] * shape[2] * shape[3];
      inputArrays.push_back({shape[2], shape[3], shape[1], inDD->data<float>() + b*nelems});
    }

    float* boxes = nullptr;
    auto func = _yolo_version == 2 ? yolov2_postproc : yolov3_postproc;

    int nboxes = func(inputArrays.data(), nArrays, _biases.data(), _net_h, _net_w, _num_classes,
        _anchor_cnt, img_h, img_w, _conf_thresh, _iou_thresh, &boxes);

    // Push detected boxes to output (share the buffer instead of deepcopy)
    std::vector<int> shape {nboxes, 6};
    auto dd = AKS::DataDescriptor(shape, AKS::DataType::FLOAT32);
    std::copy(boxes, boxes + nboxes*6, dd.data<float>());
    clearBuffer(boxes);
    out[0]->data<AKS::DataDescriptor>()[b] = std::move(dd);
  }
  return 0;
}

