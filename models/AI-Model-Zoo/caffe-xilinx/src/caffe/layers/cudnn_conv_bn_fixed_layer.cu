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

#include "caffe/layers/cudnn_conv_bn_fixed_layer.hpp"
#include <vector>
//#include <float.h>

#include "caffe/layers/cudnn_batch_norm_layer.hpp"
#include "caffe/util/quantize.hpp"
// #include <chrono>
// using Clock = std::chrono::high_resolution_clock;

namespace caffe {

#ifdef USE_BOOST_SHARED_PTR
using boost::static_pointer_cast;
#else
using std::static_pointer_cast;
#endif

/*
template <typename Dtype>
__global__ void VarToInvStd(const int n, const Dtype eps,
    const Dtype* var, Dtype* inv_std)  {
  CUDA_KERNEL_LOOP(i, n) {
    inv_std[i] = Dtype(1)/sqrt(var[i] + eps);
  }
}

template <typename Dtype>
__global__ void BindWeightsBias(
    const int weight_size, const int bias_size, const int kernel_dim,
    const Dtype* weights,
    const Dtype* scale, const Dtype* bias,
    const Dtype* mean, const Dtype* inv_std,
    Dtype* bind_weights, Dtype* bind_bias) {

  extern __shared__ Dtype scale_inv_std[];
  CUDA_KERNEL_LOOP(i, bias_size) {
    scale_inv_std[i] = scale[i] * inv_std[i];
  }
  CUDA_KERNEL_LOOP(i, bias_size) {
    bind_bias[i] = bias[i] - mean[i] * scale_inv_std[i];
  }
  CUDA_KERNEL_LOOP(i, weight_size) {
    int c = i / kernel_dim;
    bind_weights[i] = weights[i] * scale_inv_std[c];
  }
}

template <typename Dtype>
__global__ void BindWeights(const int n, const int kernel_dim, const Dtype*
weights,
    const Dtype* scale, const Dtype* inv_std, Dtype* bind_weights) {
  CUDA_KERNEL_LOOP(i, n) {
    int c = i / kernel_dim;
    bind_weights[i] = weights[i] * scale[c] * inv_std[c];
  }
}


// bias is batch norm bias
// bind_bias is for convolution
template <typename Dtype>
__global__ void BindBias(const int n, const Dtype* mean, const Dtype* inv_std,
    const Dtype* scale, const Dtype* bias, Dtype* bind_bias) {
  CUDA_KERNEL_LOOP(i, n) {
    bind_bias[i] = bias[i] - mean[i]*scale[i]*inv_std[i];
  }
}
*/

template <typename Dtype>
__global__ void TrainScaleInvVar(const int n, const Dtype eps,
                                 const Dtype *scale, const Dtype *inv_var,
                                 Dtype *scale_inv_var) {
  CUDA_KERNEL_LOOP(i, n) { scale_inv_var[i] = scale[i] * inv_var[i]; }
}

template <typename Dtype>
__global__ void TestScaleInvVar(const int n, const Dtype eps,
                                const Dtype *scale, const Dtype *var,
                                Dtype *scale_inv_var) {
  CUDA_KERNEL_LOOP(i, n) { scale_inv_var[i] = scale[i] / sqrt(var[i] + eps); }
}

template <typename Dtype>
__global__ void BindWeights(const int n, const int kernel_dim,
                            const Dtype *weights, const Dtype *scale_inv_var,
                            Dtype *bind_weights) {
  CUDA_KERNEL_LOOP(i, n) {
    int c = i / kernel_dim;
    bind_weights[i] = weights[i] * scale_inv_var[c];
  }
}

// bias is batch norm bias
// bind_bias is for convolution
template <typename Dtype>
__global__ void BindBias(const int n, const Dtype *bias, const Dtype *mean,
                         const Dtype *scale_inv_var, Dtype *bind_bias) {
  CUDA_KERNEL_LOOP(i, n) {
    bind_bias[i] = bias[i] - mean[i] * scale_inv_var[i];
  }
}

template <typename Dtype>
void CuDNNConvolutionBNFixedLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  if (this->phase_ == TRAIN) {
    internal_conv_layer_->Forward(bottom, {internal_conv_output_.get()});
    internal_bn_layer_->Forward({internal_conv_output_.get()}, top);
  }

  // Transform weights and bias

  auto weight_blob = this->blobs_[0];
  auto scale = this->blobs_[1]->gpu_data();
  auto bias = this->blobs_[2]->gpu_data();
  auto mean = this->blobs_[3]->gpu_data();
  auto var = this->blobs_[4]->gpu_data();

  auto save_mean =
      static_pointer_cast<CuDNNBatchNormLayer<Dtype>>(internal_bn_layer_)
          ->GetSaveMean()
          .gpu_data();
  auto save_inv_var =
      static_pointer_cast<CuDNNBatchNormLayer<Dtype>>(internal_bn_layer_)
          ->GetSaveInvVar()
          .gpu_data();

  double epsilon = max(eps_, CUDNN_BN_MIN_EPSILON);

  auto fixed_blobs = this->fixed_forward_conv_layer_->blobs();

  if (this->phase_ == TRAIN) {

    TrainScaleInvVar<
        Dtype><<<CAFFE_GET_BLOCKS(bn_channels_), CAFFE_CUDA_NUM_THREADS>>>(
        bn_channels_, epsilon, scale, save_inv_var,
        temp_scale_inv_var_->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;

    BindBias<Dtype><<<CAFFE_GET_BLOCKS(bn_channels_), CAFFE_CUDA_NUM_THREADS>>>(
        bn_channels_, bias, save_mean, temp_scale_inv_var_->gpu_data(),
        fixed_blobs[1]->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;

  } else {

    TestScaleInvVar<
        Dtype><<<CAFFE_GET_BLOCKS(bn_channels_), CAFFE_CUDA_NUM_THREADS>>>(
        bn_channels_, epsilon, scale, var,
        temp_scale_inv_var_->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;

    BindBias<Dtype><<<CAFFE_GET_BLOCKS(bn_channels_), CAFFE_CUDA_NUM_THREADS>>>(
        bn_channels_, bias, mean, temp_scale_inv_var_->gpu_data(),
        fixed_blobs[1]->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
  }

  BindWeights<Dtype><<<CAFFE_GET_BLOCKS(weight_blob->count()),
                       CAFFE_CUDA_NUM_THREADS>>>(
      weight_blob->count(), weight_blob->count(1), weight_blob->gpu_data(),
      temp_scale_inv_var_->gpu_data(), fixed_blobs[0]->mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;

  // Fix weights and bias
  if (this->enable_fix_) {
    if (this->fixed_method_ == FixedParameter_FixedMethod_OVER_FLOW) {
      this->weight_dec_pos_ = (int)std::floor(caffe_gpu_fix_pos_overflow(
          fixed_blobs[0]->count(), fixed_blobs[0]->gpu_data(),
          this->bit_width_));
      this->bias_dec_pos_ = (int)std::floor(caffe_gpu_fix_pos_overflow(
          fixed_blobs[1]->count(), fixed_blobs[1]->gpu_data(),
          this->bit_width_));
    } else if (this->fixed_method_ == FixedParameter_FixedMethod_DIFF_S) {
      this->weight_dec_pos_ = (int)std::floor(caffe_gpu_fix_pos_diffs(
          fixed_blobs[0]->count(), fixed_blobs[0]->gpu_data(),
          this->bit_width_));
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

  this->fixed_forward_conv_layer_->Forward(bottom, top);
}

template <typename Dtype>
void CuDNNConvolutionBNFixedLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {

  internal_bn_layer_->Backward(top, propagate_down,
                               {internal_conv_output_.get()});

  internal_conv_layer_->Backward({internal_conv_output_.get()}, propagate_down,
                                 bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionBNFixedLayer);

} // namespace caffe

#endif
