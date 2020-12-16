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

#ifdef USE_CUDNN

#include <vector>

#include "caffe/layers/cudnn_batch_norm_fixed_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNBatchNormFixedLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  // // Set up convolution layer
  CuDNNBatchNormLayer<Dtype>::LayerSetUp(bottom, top);

  // Set up BatchNorm layer
  LayerParameter bn_layer_param(this->layer_param_);
  bn_layer_param.set_type("BatchNorm");
  internal_bn_layer_ = LayerRegistry<Dtype>::CreateLayer(bn_layer_param);
  internal_bn_layer_->SetUp(bottom, top);
  internal_bn_layer_->set_piter(this->piter_);

  // Set up fixed parameters
  FixedParameter fixed_param = this->layer_param_.fixed_param();
  enable_fix_ = fixed_param.enable();
  fixed_method_ = fixed_param.fixed_method();
  bit_width_ = fixed_param.bit_width();
  weight_dec_pos_ = 0;
  bias_dec_pos_ = 0;

  // Set up BatchNorm layer for fixed forward
  LayerParameter fixed_forward_bn_layer_param(this->layer_param_);
  fixed_forward_bn_layer_param.set_type("BatchNorm");
  fixed_forward_bn_layer_param.set_phase(TEST);
  fixed_forward_bn_layer_ = LayerRegistry<Dtype>::CreateLayer(fixed_forward_bn_layer_param);
  fixed_forward_bn_layer_->SetUp(bottom, top);

  // bind blobs_ to internal_bn_layer
  for (size_t i = 0; i< internal_bn_layer_->blobs().size(); ++i) {
    this->blobs_[i] = internal_bn_layer_->blobs()[i];
  }

}

template <typename Dtype>
void CuDNNBatchNormFixedLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    CuDNNBatchNormLayer<Dtype>::Reshape(bottom, top);
    internal_bn_layer_->Reshape(bottom, top);
    fixed_forward_bn_layer_->Reshape(bottom, top);
}

INSTANTIATE_CLASS(CuDNNBatchNormFixedLayer);

}  // namespace caffe

#endif
