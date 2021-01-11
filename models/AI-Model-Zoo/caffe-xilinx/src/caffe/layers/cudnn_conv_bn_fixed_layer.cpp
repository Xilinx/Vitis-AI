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

#include "caffe/layers/cudnn_conv_bn_fixed_layer.hpp"
#include "caffe/filler.hpp"
#include "boost/make_shared.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNConvolutionBNFixedLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  // No covolution bias in ConvolutionBNFixed layer
  CHECK(!this->layer_param_.convolution_param().bias_term());

  internal_conv_output_ = make_shared<Blob<Dtype> >();

  temp_scale_inv_var_ = make_shared<Blob<Dtype> >();

  // Create convolution layers
  LayerParameter conv_layer_param(this->layer_param_);
  conv_layer_param.set_type("Convolution");
  //ConvolutionParameter* convh_param = conv_layer_param.convolution_param();
  internal_conv_layer_ = LayerRegistry<Dtype>::CreateLayer(conv_layer_param);
  internal_conv_layer_->SetUp(bottom, {internal_conv_output_.get()});

  bn_channels_ = conv_layer_param.convolution_param().num_output();
  
  // Create batch norm layer
  LayerParameter bn_layer_param(this->layer_param_);
  bn_layer_param.set_type("BatchNorm");
  //BatchNormParameter* bn_param = layer_param.batch_norm_param();
  internal_bn_layer_ = LayerRegistry<Dtype>::CreateLayer(bn_layer_param);
  internal_bn_layer_->SetUp({internal_conv_output_.get()}, top);
  internal_bn_layer_->set_piter(this->piter_);

  eps_ = bn_layer_param.batch_norm_param().eps();

  this->blobs_.resize(internal_conv_layer_->blobs().size() + 
                      internal_bn_layer_->blobs().size());

  // Bind blobs
  this->blobs_[0] = internal_conv_layer_->blobs()[0];
  for (auto i = 0; i < internal_bn_layer_->blobs().size(); ++i) {
    this->blobs_[i+1] = internal_bn_layer_->blobs()[i];
  } 

  temp_scale_inv_var_->ReshapeLike(*this->blobs_[1]);

  // Set up fixed parameters
  FixedParameter fixed_param = this->layer_param_.fixed_param();
  this->enable_fix_ = fixed_param.enable();
  this->fixed_method_ = fixed_param.fixed_method();
  this->bit_width_ = fixed_param.bit_width();
  this->weight_dec_pos_ = 0;
  this->bias_dec_pos_ = 0;

  LayerParameter fixed_forward_conv_layer_param(this->layer_param_);
  fixed_forward_conv_layer_param.set_type("Convolution");
  fixed_forward_conv_layer_param.
      mutable_convolution_param()->set_bias_term(true);
  this->fixed_forward_conv_layer_ = 
      LayerRegistry<Dtype>::CreateLayer(fixed_forward_conv_layer_param);
  this->fixed_forward_conv_layer_->SetUp(bottom, top);
}

template <typename Dtype>
void CuDNNConvolutionBNFixedLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  internal_conv_layer_->Reshape(bottom, {internal_conv_output_.get()});
  internal_bn_layer_->Reshape({internal_conv_output_.get()}, top);
  this->fixed_forward_conv_layer_->Reshape(bottom, top);
}


INSTANTIATE_CLASS(CuDNNConvolutionBNFixedLayer);

} // namespace caffe
#endif
