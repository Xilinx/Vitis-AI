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

#ifndef CAFFE_CUDNN_BATCH_NORM_FIXED_LAYER_HPP_
#define CAFFE_CUDNN_BATCH_NORM_FIXED_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/cudnn_batch_norm_layer.hpp"

namespace caffe {

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNBatchNormFixedLayer : public CuDNNBatchNormLayer<Dtype> {
 public:
  explicit CuDNNBatchNormFixedLayer(const LayerParameter& param)
      : CuDNNBatchNormLayer<Dtype>(param) {}
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;

  inline const char* type() const override {
    return "BatchNormFixed"; }

 protected:
  void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) override;

  // fixed param
  FixedParameter_FixedMethod fixed_method_;
  bool enable_fix_;
  int bit_width_;
  int weight_dec_pos_;
  int bias_dec_pos_;

  shared_ptr<Layer<Dtype> > internal_bn_layer_;
  shared_ptr<Layer<Dtype> > fixed_forward_bn_layer_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_BATCH_NORM_FIXED_LAYER_HPP_
