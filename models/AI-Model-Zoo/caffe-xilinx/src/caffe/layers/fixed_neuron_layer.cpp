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

#include "caffe/layers/fixed_neuron_layer.hpp"
#include "caffe/util/quantize.hpp"

namespace caffe {

template <typename Dtype>
void FixedNeuronLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  FixedParameter fixed_param = this->layer_param_.fixed_param();
  this->enable_fix_ = fixed_param.enable();
  this->follow_data_layer = fixed_param.follow_data_layer();
  this->fixed_method_ = fixed_param.fixed_method();
  this->bit_width_ = fixed_param.bit_width();
  // this->moving_average_fraction_ = fixed_param.moving_average_fraction();
  // this->update_interval_ = fixed_param.update_interval();
  // this->data_fraction_ = fixed_param.data_fraction();
  CHECK_GT(this->bit_width_, 0) << "Bit width must be larger than 0.";
  this->dec_pos_ = 0;
  this->blobs_.resize(1);
  this->param_propagate_down_.resize(1);
  this->blobs_[0].reset(new Blob<Dtype>({1}));
  caffe_set(1, Dtype(0), this->blobs_[0]->mutable_cpu_data());
  this->param_propagate_down_[0] = false;
  // dist_ = std::uniform_real_distribution<>(0.0, 1-data_fraction_);
}

template <typename Dtype>
void FixedNeuronLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void FixedNeuronLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (!enable_fix_) {
    if (top[0] != bottom[0]) {
      caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(),
          top[0]->mutable_cpu_data());
    }
    return;
  }
  if (this->phase_ == TRAIN) {
    if (this->fixed_method_ == FixedParameter_FixedMethod_OVER_FLOW) {
      // overflow
      Dtype dec_pos = caffe_cpu_fix_pos_overflow(
          bottom[0]->count(), bottom[0]->cpu_data(), this->bit_width_);
      auto save_pos = this->blobs_[0]->mutable_cpu_data();
      if ( this->pos_converge() ) {
        if (this->iter() == 0) {
          *save_pos = (int)std::floor(dec_pos);
          pos_hist_[*save_pos]++;
        } else {
          pos_hist_[(int)std::floor(dec_pos)]++;
          auto max = std::max_element(
              pos_hist_.begin(), pos_hist_.end(),
              [](const pair<int, int> &p1, const pair<int, int> &p2) {
                return p1.second < p2.second;
              });
          *save_pos = max->first;
        }
        this->dec_pos_ = (int)std::floor(dec_pos);
        top[0]->set_bit_width( this->bit_width_ );
        top[0]->set_fixed_pos( this->dec_pos_ );
      } else {
        this->dec_pos_ = top[0]->fixed_pos();
        *save_pos = this->dec_pos_;
      }

      if ( !follow_data_layer )
        caffe_cpu_top_fix(bottom[0]->count(), bottom[0]->cpu_data(),
                          top[0]->mutable_cpu_data(), this->bit_width_,
                          this->dec_pos_);
      else
        caffe_cpu_fix(bottom[0]->count(), bottom[0]->cpu_data(),
                      top[0]->mutable_cpu_data(), this->bit_width_,
                      this->dec_pos_);
 
    } else if (this->fixed_method_ == FixedParameter_FixedMethod_DIFF_S || this->fixed_method_ == FixedParameter_FixedMethod_DIFF_A) {
      // Diff_S
      int dec_pos = caffe_cpu_fix_pos_diffs(
          bottom[0]->count(), bottom[0]->cpu_data(), this->bit_width_);
      auto save_pos = this->blobs_[0]->mutable_cpu_data();
      if ( this->pos_converge() ) {
        if (this->iter() == 0) {
          *save_pos = (int)std::floor(dec_pos);
          pos_hist_[*save_pos]++;
        } else {
          pos_hist_[(int)std::floor(dec_pos)]++;
          auto max = std::max_element(
              pos_hist_.begin(), pos_hist_.end(),
              [](const pair<int, int> &p1, const pair<int, int> &p2) {
                return p1.second < p2.second;
              });
          *save_pos = max->first;
        }
        this->dec_pos_ = (int)std::floor(*save_pos);
        top[0]->set_bit_width( this->bit_width_ );
        top[0]->set_fixed_pos( this->dec_pos_ );
      } else {
        this->dec_pos_ = top[0]->fixed_pos();
        *save_pos = this->dec_pos_;
      }

      if ( !follow_data_layer )
        caffe_cpu_top_fix(bottom[0]->count(), bottom[0]->cpu_data(),
                          top[0]->mutable_cpu_data(), this->bit_width_,
                          this->dec_pos_);
      else
        caffe_cpu_fix(bottom[0]->count(), bottom[0]->cpu_data(),
                      top[0]->mutable_cpu_data(), this->bit_width_,
                      this->dec_pos_);
 
    } else if (this->fixed_method_ == FixedParameter_FixedMethod_DIFF_S_SIGMOID) {
      // Diff_S_Sigmoid
      int dec_pos = caffe_cpu_fix_pos_diffs_sigmoid(
          bottom[0]->count(), bottom[0]->cpu_data(), this->bit_width_, 10);
      auto save_pos = this->blobs_[0]->mutable_cpu_data();
      if ( this->pos_converge() ) {
        if (this->iter() == 0) {
          *save_pos = (int)std::floor(dec_pos);
          pos_hist_[*save_pos]++;
        } else {
          pos_hist_[(int)std::floor(dec_pos)]++;
          auto max = std::max_element(
              pos_hist_.begin(), pos_hist_.end(),
              [](const pair<int, int> &p1, const pair<int, int> &p2) {
                return p1.second < p2.second;
              });
          *save_pos = max->first;
        }
        this->dec_pos_ = (int)std::floor(*save_pos);
        top[0]->set_bit_width( this->bit_width_ );
        top[0]->set_fixed_pos( this->dec_pos_ );
      } else {
        this->dec_pos_ = top[0]->fixed_pos();
        *save_pos = this->dec_pos_;
      }

      if ( !follow_data_layer )
        caffe_cpu_top_fix(bottom[0]->count(), bottom[0]->cpu_data(),
                          top[0]->mutable_cpu_data(), this->bit_width_,
                          this->dec_pos_);
      else
        caffe_cpu_fix(bottom[0]->count(), bottom[0]->cpu_data(),
                      top[0]->mutable_cpu_data(), this->bit_width_,
                      this->dec_pos_);

    } else {
      LOG(FATAL) << "Unknown fixed method: " << this->fixed_method_;
    }
  } else if (this->phase_ == TEST) {
    this->dec_pos_ = (int)std::floor(this->blobs_[0]->cpu_data()[0]);
    if ( !follow_data_layer )
      caffe_cpu_top_fix(bottom[0]->count(), bottom[0]->cpu_data(),
                        top[0]->mutable_cpu_data(), this->bit_width_, this->dec_pos_);
    else
    {
      caffe_cpu_trunc(bottom[0]->count(), bottom[0]->cpu_data(),
                    top[0]->mutable_cpu_data(), this->dec_pos_ );
    }
    top[0]->set_bit_width( this->bit_width_ );
    top[0]->set_fixed_pos( this->dec_pos_ );
  } else {
    LOG(FATAL) << "Unknown phase: " << this->phase_;
  }
}

template <typename Dtype>
void FixedNeuronLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    if (top[0] != bottom[0]) {
      caffe_copy(bottom[0]->count(), top[0]->cpu_diff(),
          bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FixedNeuronLayer);
#endif

INSTANTIATE_CLASS(FixedNeuronLayer);
REGISTER_LAYER_CLASS(FixedNeuron);

}  // namespace caffe
