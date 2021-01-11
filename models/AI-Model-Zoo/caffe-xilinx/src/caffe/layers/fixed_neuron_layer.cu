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
void FixedNeuronLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  if (!enable_fix_) {
    if (top[0] != bottom[0]) {
      caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(),
                 top[0]->mutable_gpu_data());
    }
    return;
  }
  if (this->phase_ == TRAIN) {
    if (this->fixed_method_ == FixedParameter_FixedMethod_OVER_FLOW) {

      Dtype dec_pos = caffe_gpu_fix_pos_overflow(
          bottom[0]->count(), bottom[0]->gpu_data(), this->bit_width_);
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
        caffe_gpu_top_fix(bottom[0]->count(), bottom[0]->gpu_data(),
                          top[0]->mutable_gpu_data(), this->bit_width_,
                          this->dec_pos_);
      else
        caffe_gpu_fix(bottom[0]->count(), bottom[0]->gpu_data(),
                      top[0]->mutable_gpu_data(), this->bit_width_,
                      this->dec_pos_);

      // Diff_S
    } else if (this->fixed_method_ == FixedParameter_FixedMethod_DIFF_S || this->fixed_method_ == FixedParameter_FixedMethod_DIFF_A) {
      int dec_pos = caffe_gpu_fix_pos_diffs(
          bottom[0]->count(), bottom[0]->gpu_data(), this->bit_width_);
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
        caffe_gpu_top_fix(bottom[0]->count(), bottom[0]->gpu_data(),
                          top[0]->mutable_gpu_data(), this->bit_width_,
                          this->dec_pos_);
      else
        caffe_gpu_fix(bottom[0]->count(), bottom[0]->gpu_data(),
                      top[0]->mutable_gpu_data(), this->bit_width_,
                      this->dec_pos_);

      // Diff_S_Sigmoid
    } else if (this->fixed_method_ == FixedParameter_FixedMethod_DIFF_S_SIGMOID) {
      int dec_pos = caffe_gpu_fix_pos_diffs_sigmoid(
          bottom[0]->count(), bottom[0]->gpu_data(), this->bit_width_, 10);
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
        caffe_gpu_top_fix(bottom[0]->count(), bottom[0]->gpu_data(),
                          top[0]->mutable_gpu_data(), this->bit_width_,
                          this->dec_pos_);
      else
        caffe_gpu_fix(bottom[0]->count(), bottom[0]->gpu_data(),
                      top[0]->mutable_gpu_data(), this->bit_width_,
                      this->dec_pos_);
    } else {
      LOG(FATAL) << "Unknown fixed method: " << this->fixed_method_;
    }
  } else if (this->phase_ == TEST) {
    this->dec_pos_ = (int)std::floor(this->blobs_[0]->cpu_data()[0]);
    /* LOG(INFO) << "test iter: " << this->iter() */
    /* << " layer: " << this->layer_param_.name() */
    /* << " saved pos :" << this->dec_pos_; */
    if ( !follow_data_layer )
      caffe_gpu_top_fix(bottom[0]->count(), bottom[0]->gpu_data(),
                        top[0]->mutable_gpu_data(), this->bit_width_, this->dec_pos_);
    else
    {
      // truncate input data numbers by ((int)(x * 0.5)) * 2
      caffe_gpu_trunc(bottom[0]->count(), bottom[0]->gpu_data(),
                    top[0]->mutable_gpu_data(), this->dec_pos_ );
      // quantize bottom data numbers
      //caffe_gpu_fix(bottom[0]->count(), bottom[0]->gpu_data(),
      //              top[0]->mutable_gpu_data(), this->bit_width_, this->dec_pos_);
    }
    top[0]->set_bit_width( this->bit_width_ );
    top[0]->set_fixed_pos( this->dec_pos_ );
  } else {
    LOG(FATAL) << "Unknown phase: " << this->phase_;
  }
}

template <typename Dtype>
void FixedNeuronLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    if (top[0] != bottom[0]) {
      caffe_copy(bottom[0]->count(), top[0]->gpu_diff(),
                 bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FixedNeuronLayer);

} // namespace caffe
