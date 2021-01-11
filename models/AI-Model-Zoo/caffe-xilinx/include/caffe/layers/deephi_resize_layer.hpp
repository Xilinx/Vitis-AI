/*
* Copyright 2019 Xilinx Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#ifndef CAFFE_DEEPHI_RESIZE_LAYER_HPP_
#define CAFFE_DEEPHI_RESIZE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Deephi resize feature map in consistant with DPU.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class DeephiResizeLayer : public Layer<Dtype> {
 public:
  explicit DeephiResizeLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DeephiResize"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
//  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//      const vector<Blob<Dtype>*>& top) {}
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
//  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  float scale_h_, scale_w_;
  int new_height_, new_width_;
  int channels_;
  int height_, width_;
  int resized_height_, resized_width_;
  bool use_scale_,inter_mode = true, use_dpu = false;
};

}  // namespace caffe

#endif  // CAFFE_DEEPHI_RESIZE_LAYER_HPP_
