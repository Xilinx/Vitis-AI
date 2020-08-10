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
#include <algorithm>
#include <vector>

#include "caffe/layers/cudnn_conv_fixed_layer.hpp"

namespace caffe {

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNConvolutionFixedLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  // Set up convolution layer
  CuDNNConvolutionLayer<Dtype>::LayerSetUp(bottom, top);

  // Set up fixed parameters
  FixedParameter fixed_param = this->layer_param_.fixed_param();
  enable_fix_ = fixed_param.enable();
  fixed_method_ = fixed_param.fixed_method();
  bit_width_ = fixed_param.bit_width();
  weight_dec_pos_ = 0;
  bias_dec_pos_ = 0;

  // Set up convolution layer for fixed forward
  LayerParameter layer_param(this->layer_param_);
  layer_param.set_type("Convolution");
  fixed_forward_conv_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
  fixed_forward_conv_layer_->SetUp(bottom, top);
}

template <typename Dtype>
void CuDNNConvolutionFixedLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  CuDNNConvolutionLayer<Dtype>::Reshape(bottom, top);
  fixed_forward_conv_layer_->Reshape(bottom, top);
}


INSTANTIATE_CLASS(CuDNNConvolutionFixedLayer);

} // namespace caffe
#endif
