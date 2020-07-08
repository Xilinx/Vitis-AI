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
//#include <float.h>
#if 0
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "caffe/layers/cudnn_conv_fixed_layer.hpp"
#include "caffe/util/quantize.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNConvolutionFixedLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  auto fixed_blobs = fixed_forward_conv_layer_->blobs();
  if (!enable_fix_) {
    caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(),
               fixed_blobs[0]->mutable_gpu_data());
    if (this->bias_term_) {
      caffe_copy(this->blobs_[1]->count(), this->blobs_[1]->gpu_data(),
                 fixed_blobs[1]->mutable_gpu_data());
    }
    fixed_forward_conv_layer_->Forward(bottom, top);
    return;
  }

  // Fix weights and bias
  if ( !fixed_ || this->iter() == 0 ) {
    if (this->fixed_method_ == FixedParameter_FixedMethod_OVER_FLOW) {
      this->weight_dec_pos_ = (int)std::floor(caffe_gpu_fix_pos_overflow(
          this->blobs_[0]->count(), this->blobs_[0]->gpu_data(),
          this->bit_width_));
      if (this->bias_term_) {
        this->bias_dec_pos_ = (int)std::floor(caffe_gpu_fix_pos_overflow(
            this->blobs_[1]->count(), this->blobs_[1]->gpu_data(),
            this->bit_width_));
      }
    } else if (this->fixed_method_ == FixedParameter_FixedMethod_DIFF_S) {
      this->weight_dec_pos_ = (int)std::floor(caffe_gpu_fix_pos_diffs(
          this->blobs_[0]->count(), this->blobs_[0]->gpu_data(),
          this->bit_width_));
      if (this->bias_term_) {
        this->bias_dec_pos_ = (int)std::floor(caffe_gpu_fix_pos_diffs(
            this->blobs_[1]->count(), this->blobs_[1]->gpu_data(),
            this->bit_width_));
      }
    } else {
      LOG(FATAL) << "Unknown fixed method: " << this->fixed_method_;
    }

    fixed_blobs[0]->set_bit_width( this->bit_width_ );
    fixed_blobs[0]->set_fixed_pos( this->weight_dec_pos_ );
    caffe_gpu_fix(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(),
                  fixed_blobs[0]->mutable_gpu_data(), this->bit_width_,
                  this->weight_dec_pos_);
    if (this->bias_term_) {
      fixed_blobs[1]->set_bit_width( this->bit_width_ );
      fixed_blobs[1]->set_fixed_pos( this->bias_dec_pos_ );
      caffe_gpu_fix(this->blobs_[1]->count(), this->blobs_[1]->gpu_data(),
                    fixed_blobs[1]->mutable_gpu_data(), this->bit_width_,
                    this->bias_dec_pos_);
    }

    fixed_ = true;

  } else if (this->phase_ == TEST) {
  }
#if 0
  if (this->phase_ == TEST && this->iter() == 0) {
    if ( getenv( "XMODEL_GOLDEN" ) ) {
      LOG(INFO) << "Dumping weights...";
      string path = "dump_wb";
      path += '/';

      // dump int weights
      string filename = this->layer_param_.name();
      replace(filename.begin(), filename.end(), '/', '_');
      filename = path + filename + "_w.bin";
      FILE* fp = fopen( filename.c_str(), "wb" );
      float scale = pow(2.0, this->weight_dec_pos_);
      for (int i = 0; i < this->blobs_[0]->count(); ++i) {
        int num = int(fixed_blobs[0]->cpu_data()[i] * scale);
        fwrite((char *)&num, sizeof(char), 1, fp);
      }
      fclose(fp);
      /*
      // dump float weights
      string filename1 = this->layer_param_.name();
      replace(filename1.begin(), filename1.end(), '/', '_');
      filename1 = path + filename1 + "_wf.bin";

#if 0
      filename1 = path + filename1 + "_wf_en.bin";
      // encrypt fixed blob 0 
      auto data_sz = this->blobs_[0]->count() * sizeof(float);
      if (data_sz > 0) {
        auto en = new uint8_t[data_sz];
        F1Encrypt(data_sz, (const uint8_t*)(fixed_blobs[0]->cpu_data()), en);
        memcpy((void*)(fixed_blobs[0]->mutable_cpu_data()), en, data_sz);
        delete [] en;
      }
#endif

      FILE* fp1 = fopen( filename1.c_str(), "wb" );
      for (int i = 0; i < this->blobs_[0]->count(); ++i) {
        float num = fixed_blobs[0]->cpu_data()[i];
        fwrite((char *)&num, sizeof(char), 4, fp1);
      }
      fclose(fp1);
      */

      if (this->bias_term_) {
        // dump bias
        LOG(INFO) << "Dumping bias...";
        string filename = this->layer_param_.name();
        filename = path + filename + "_b.bin";
        FILE* fp = fopen( filename.c_str(), "wb" );
        float scale = pow(2.0, this->bias_dec_pos_);
        for (int i = 0; i < this->blobs_[1]->count(); ++i) {
          int num = int(fixed_blobs[1]->cpu_data()[i] * scale);
          fwrite((char *)&num, sizeof(char), 1, fp);
        }
        fclose(fp);
      }
    }
  }
#endif 

  // enlarge weights if weight numbers are too small, at the same time
  // enlarge bias with the same scale
  if ( this->weight_dec_pos_ > 12 ) {
    caffe_gpu_scale( fixed_blobs[0]->count(), fixed_blobs[0]->gpu_data(),
                  fixed_blobs[0]->mutable_gpu_data(), this->weight_dec_pos_ );
    if (this->bias_term_) {
      caffe_gpu_scale( fixed_blobs[1]->count(), fixed_blobs[1]->gpu_data(),
                  fixed_blobs[1]->mutable_gpu_data(), this->weight_dec_pos_ );
    }
  }

  fixed_forward_conv_layer_->Forward(bottom, top);

  // shrink enlarged activation back
  if ( this->weight_dec_pos_ > 12 ) {
    caffe_gpu_scale( top[0]->count(), top[0]->gpu_data(),
                  top[0]->mutable_gpu_data(), -this->weight_dec_pos_ );

    caffe_gpu_scale( fixed_blobs[0]->count(), fixed_blobs[0]->gpu_data(),
                  fixed_blobs[0]->mutable_gpu_data(), -this->weight_dec_pos_ );
    if (this->bias_term_) {
      caffe_gpu_scale( fixed_blobs[1]->count(), fixed_blobs[1]->gpu_data(),
                  fixed_blobs[1]->mutable_gpu_data(), -this->weight_dec_pos_ );
    }
  }
}

template <typename Dtype>
void CuDNNConvolutionFixedLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {

  fixed_ = false;
  CuDNNConvolutionLayer<Dtype>::Backward_gpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionFixedLayer);

} // namespace caffe
#endif
