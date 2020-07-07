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
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/cudnn_batch_norm_fixed_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/quantize.hpp"

namespace caffe {

#ifdef USE_BOOST_SHARED_PTR
using boost::static_pointer_cast;
#else
using std::static_pointer_cast;
#endif

// bind scale: new scale = scale / sqrt(epsilon + var)
template <typename Dtype>
__global__ void TrainScaleInvVar(const int n, const Dtype eps,
                                 const Dtype *scale, const Dtype *inv_var,
                                 Dtype *scale_inv_var) {
  CUDA_KERNEL_LOOP(i, n) { scale_inv_var[i] = scale[i] * inv_var[i]; }
}

// bind scale: new scale = scale / sqrt(epsilon + var)
template <typename Dtype>
__global__ void TestScaleInvVar(const int n, const Dtype eps,
                                const Dtype *scale, const Dtype *var,
                                Dtype *scale_inv_var) {
  CUDA_KERNEL_LOOP(i, n) { scale_inv_var[i] = scale[i] / sqrt(var[i] + eps); }
}

// bind bias: new bias = bias - scale * mean / sqrt(epsilon + pow(var,2))
template <typename Dtype>
__global__ void BindBias(const int n, const Dtype *bias, const Dtype *mean,
                         const Dtype *scale_inv_var, Dtype *bind_bias) {
  CUDA_KERNEL_LOOP(i, n) {
    bind_bias[i] = bias[i] - mean[i] * scale_inv_var[i];
  }
}

template <typename Dtype>
void CuDNNBatchNormFixedLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  /* if (this->phase_ == TRAIN) { */
  internal_bn_layer_->Forward(bottom, top);
  /* } */

  // Transform scale and bias
  const Dtype *scale = this->blobs_[0]->gpu_data();
  const Dtype *bias = this->blobs_[1]->gpu_data();
  const Dtype *mean = this->blobs_[2]->gpu_data();
  const Dtype *var = this->blobs_[3]->gpu_data();

  auto save_mean =
      static_pointer_cast<CuDNNBatchNormLayer<Dtype>>(internal_bn_layer_)
          ->GetSaveMean()
          .gpu_data();
  auto save_inv_var =
      static_pointer_cast<CuDNNBatchNormLayer<Dtype>>(internal_bn_layer_)
          ->GetSaveInvVar()
          .gpu_data();

  double epsilon = max(this->eps_, CUDNN_BN_MIN_EPSILON);

  auto fixed_blobs = this->fixed_forward_bn_layer_->blobs();
  int bn_channels = fixed_blobs[0]->channels();

  if (this->phase_ == TRAIN) {

    TrainScaleInvVar<
        Dtype><<<CAFFE_GET_BLOCKS(bn_channels), CAFFE_CUDA_NUM_THREADS>>>(
        bn_channels, epsilon, scale, save_inv_var,
        fixed_blobs[0]->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;

    BindBias<Dtype><<<CAFFE_GET_BLOCKS(bn_channels), CAFFE_CUDA_NUM_THREADS>>>(
        bn_channels, bias, save_mean, fixed_blobs[0]->gpu_data(),
        fixed_blobs[1]->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;

  } else if (this->phase_ == TEST) {

    TestScaleInvVar<
        Dtype><<<CAFFE_GET_BLOCKS(bn_channels), CAFFE_CUDA_NUM_THREADS>>>(
        bn_channels, epsilon, scale, var, fixed_blobs[0]->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;

    BindBias<Dtype><<<CAFFE_GET_BLOCKS(bn_channels), CAFFE_CUDA_NUM_THREADS>>>(
        bn_channels, bias, mean, fixed_blobs[0]->gpu_data(),
        fixed_blobs[1]->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
  }

  // Reset mean and variance
  caffe_gpu_set<Dtype>(fixed_blobs[2]->count(), (Dtype)0,
                       fixed_blobs[2]->mutable_gpu_data());
  caffe_gpu_set<Dtype>(fixed_blobs[3]->count(), (Dtype)1,
                       fixed_blobs[3]->mutable_gpu_data());

  // fix scale and bias
  if (this->enable_fix_) {
    if (this->fixed_method_ == FixedParameter_FixedMethod_OVER_FLOW) {
      // scale
      this->weight_dec_pos_ = (int)std::floor(caffe_gpu_fix_pos_overflow(
          fixed_blobs[0]->count(), fixed_blobs[0]->gpu_data(),
          this->bit_width_));
      // bias
      this->bias_dec_pos_ = (int)std::floor(caffe_gpu_fix_pos_overflow(
          fixed_blobs[1]->count(), fixed_blobs[1]->gpu_data(),
          this->bit_width_));
    } else if (this->fixed_method_ == FixedParameter_FixedMethod_DIFF_S) {
      // scale
      this->weight_dec_pos_ = (int)std::floor(caffe_gpu_fix_pos_diffs(
          fixed_blobs[0]->count(), fixed_blobs[0]->gpu_data(),
          this->bit_width_));
      // bias
      this->bias_dec_pos_ = (int)std::floor(caffe_gpu_fix_pos_diffs(
          fixed_blobs[1]->count(), fixed_blobs[1]->gpu_data(),
          this->bit_width_));
    } else {
      LOG(FATAL) << "Unknown fixed method: " << this->fixed_method_;
    }
    caffe_gpu_fix(fixed_blobs[0]->count(), fixed_blobs[0]->gpu_data(),
                  fixed_blobs[0]->mutable_gpu_data(), this->bit_width_,
                  this->weight_dec_pos_);
    caffe_gpu_fix(fixed_blobs[1]->count(), fixed_blobs[1]->gpu_data(),
                  fixed_blobs[1]->mutable_gpu_data(), this->bit_width_,
                  this->bias_dec_pos_);
  }

  /* DLOG(INFO) << "iter: " << this->iter() */
  /* << " layer: " << this->layer_param().name() */
  /* << " fixed scale pos: " << this->weight_dec_pos_; */
  /* DLOG(INFO) << "iter: " << this->iter() */
  /* << " layer: " << this->layer_param().name() */
  /* << " fixed bias pos: " << this->bias_dec_pos_; */
  // fixed forward
  this->fixed_forward_bn_layer_->Forward(bottom, top);
}

template <typename Dtype>
void CuDNNBatchNormFixedLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {

  internal_bn_layer_->Backward(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNBatchNormFixedLayer);

} // namespace caffe

#endif
