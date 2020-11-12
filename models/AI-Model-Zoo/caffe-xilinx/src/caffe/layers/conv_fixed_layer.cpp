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

#include <algorithm>
#include <vector>

#include "caffe/layers/conv_fixed_layer.hpp"
#include "caffe/util/quantize.hpp"

namespace caffe {

/**
 */
template <typename Dtype>
void ConvolutionFixedLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  // Set up convolution layer
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);

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
void ConvolutionFixedLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  ConvolutionLayer<Dtype>::Reshape(bottom, top);
  fixed_forward_conv_layer_->Reshape(bottom, top);
}

template <typename Dtype>
void ConvolutionFixedLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  auto fixed_blobs = fixed_forward_conv_layer_->blobs();
  if (!enable_fix_) {
    caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),
               fixed_blobs[0]->mutable_cpu_data());
    if (this->bias_term_) {
      caffe_copy(this->blobs_[1]->count(), this->blobs_[1]->cpu_data(),
                 fixed_blobs[1]->mutable_cpu_data());
    }
    fixed_forward_conv_layer_->Forward(bottom, top);
    return;
  }

  // Fix weights and bias
  if ( !fixed_ || this->iter() == 0 ) {
    if (this->fixed_method_ == FixedParameter_FixedMethod_OVER_FLOW) {
      this->weight_dec_pos_ = (int)std::floor(caffe_cpu_fix_pos_overflow(
          this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),
          this->bit_width_));
      if (this->bias_term_) {
        this->bias_dec_pos_ = (int)std::floor(caffe_cpu_fix_pos_overflow(
            this->blobs_[1]->count(), this->blobs_[1]->cpu_data(),
            this->bit_width_));
      }
    } else if (this->fixed_method_ == FixedParameter_FixedMethod_DIFF_S) {
      this->weight_dec_pos_ = (int)std::floor(caffe_cpu_fix_pos_diffs(
          this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),
          this->bit_width_));
      if (this->bias_term_) {
        this->bias_dec_pos_ = (int)std::floor(caffe_cpu_fix_pos_diffs(
            this->blobs_[1]->count(), this->blobs_[1]->cpu_data(),
            this->bit_width_));
      }
    } else {
      LOG(FATAL) << "Unknown fixed method: " << this->fixed_method_;
    }

    fixed_blobs[0]->set_bit_width( this->bit_width_ );
    fixed_blobs[0]->set_fixed_pos( this->weight_dec_pos_ );
    caffe_cpu_fix(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),
                  fixed_blobs[0]->mutable_cpu_data(), this->bit_width_,
                  this->weight_dec_pos_);
    if (this->bias_term_) {
      fixed_blobs[1]->set_bit_width( this->bit_width_ );
      fixed_blobs[1]->set_fixed_pos( this->bias_dec_pos_ );
      caffe_cpu_fix(this->blobs_[1]->count(), this->blobs_[1]->cpu_data(),
                    fixed_blobs[1]->mutable_cpu_data(), this->bit_width_,
                    this->bias_dec_pos_);
    }
    fixed_ = true;
  } else if (this->phase_ == TEST) {
  }

  // enlarge weights if weight numbers are too small, at the same time
  // enlarge bias with the same scale
  if ( this->weight_dec_pos_ > 12 ) {
    caffe_cpu_scale( fixed_blobs[0]->count(), fixed_blobs[0]->cpu_data(),
                  fixed_blobs[0]->mutable_cpu_data(), this->weight_dec_pos_ );
    if (this->bias_term_) {
      caffe_cpu_scale( fixed_blobs[1]->count(), fixed_blobs[1]->cpu_data(),
                  fixed_blobs[1]->mutable_cpu_data(), this->weight_dec_pos_ );
    }
  }

  fixed_forward_conv_layer_->Forward(bottom, top);

  // shrink enlarged activation back
  if ( this->weight_dec_pos_ > 12 ) {
    caffe_cpu_scale( top[0]->count(), top[0]->cpu_data(),
                  top[0]->mutable_cpu_data(), -this->weight_dec_pos_ );

    caffe_cpu_scale( fixed_blobs[0]->count(), fixed_blobs[0]->cpu_data(),
                  fixed_blobs[0]->mutable_cpu_data(), -this->weight_dec_pos_ );
    if (this->bias_term_) {
      caffe_cpu_scale( fixed_blobs[1]->count(), fixed_blobs[1]->cpu_data(),
                  fixed_blobs[1]->mutable_cpu_data(), -this->weight_dec_pos_ );
    }
  }
}

template <typename Dtype>
void ConvolutionFixedLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {

  fixed_ = false;
  ConvolutionLayer<Dtype>::Backward_cpu(top, propagate_down, bottom);
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionFixedLayer);
#endif

INSTANTIATE_CLASS(ConvolutionFixedLayer);
} // namespace caffe
