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

#include <vector>

#include "caffe/layers/deephi_resize_layer.hpp"
#include "caffe/util/dpu_resize.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
namespace caffe {

template <typename Dtype>
void DeephiResizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  DeephiResizeParameter dr_param = this->layer_param_.deephi_resize_param();

  CHECK((dr_param.has_new_height() && dr_param.has_new_width()) ||
        (!dr_param.has_new_height() && !dr_param.has_new_width()))
      << "new_height and new_width should be set at the same time";
  CHECK((dr_param.has_scale_h() && dr_param.has_scale_w()) ||
        (!dr_param.has_scale_h() && !dr_param.has_scale_w()))
      << "h_scale and w_scale should be set at the same time";

  if (dr_param.has_new_height()) {
    new_height_ = dr_param.new_height();
    new_width_ = dr_param.new_width();
    use_scale_ = false;
  } else {
    scale_h_ = dr_param.scale_h();
    scale_w_ = dr_param.scale_w();
    use_scale_ = true;
  }
}

template <typename Dtype>
void DeephiResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  if (use_scale_) {
    resized_height_ = height_ * scale_h_;
    resized_width_ = width_ * scale_w_;
  } else {
    resized_height_ = new_height_;
    resized_width_ = new_width_;
  }
  top[0]->Reshape(bottom[0]->num(), channels_, resized_height_, resized_width_);
}

template <typename Dtype>
void DeephiResizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
 // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {

    const Dtype *batch_data = bottom_data + bottom[0]->offset(n);
    if (use_dpu){
      uint8_t *resize_input = new uint8_t[channels_ * height_ * width_];
      uint8_t *resize_output =
        new uint8_t[channels_ * resized_height_ * resized_width_];

      // Change Order for dpu_resize: CHW --> HWC
      for (int c = 0; c < channels_; ++c) {
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            int batch_index = c * height_ * width_ + h * width_ + w;
            int resize_index = h * width_ * channels_ + w * channels_ + c;
            resize_input[resize_index] = batch_data[batch_index];
            DLOG(INFO) << "resize_input[" << resize_index
              << "] = " << resize_input[resize_index]
              << " batch_data = " << batch_data[batch_index];
          }
        }
        batch_data += bottom[0]->offset(0, 1);
        resize_input += bottom[0]->offset(0, 1);
      }

      struct _config cfg;
      cfg.scale_w = round(1.0 * (1 << 15) * width_ / resized_width_);
      cfg.scale_h = round(1.0 * (1 << 15) * height_ / resized_height_);
      cfg.src_w = width_;
      cfg.src_h = height_;
      cfg.src_c = channels_;
      cfg.dst_w = resized_width_;
      cfg.dst_h = resized_height_;
      cfg.dst_c = channels_;
      if (this->layer_param_.deephi_resize_param().resize_type() ==
          DeephiResizeParameter_ResizeType_BILINEAR) {
        cfg.inter_mode = 0;
      } else {
        cfg.inter_mode = 1;
      }

      // Do resize
      dpu_resize dr(resize_input, resize_output, cfg);


      for (int i = 0; i < top[0]->offset(0); ++i) {
        top_data[i] = resize_output[i];
        DLOG(INFO) << "resize_output[" << i << "] = " << resize_output[i]
          << " top_data = " << top_data[i];
      }

      top_data += top[0]->offset(0);
    }
    else{

      if (this->layer_param_.deephi_resize_param().resize_type() ==
          DeephiResizeParameter_ResizeType_BILINEAR) {
        inter_mode = 0;
      } else {
        inter_mode = 1;
      }
      cv::Mat resize_input(cv::Size(height_,width_),CV_32FC1 , cv::Scalar(0));
      cv::Mat resize_output(cv::Size(resized_height_,resized_width_),CV_32FC1 , cv::Scalar(0));
      for (int c = 0; c < channels_; c++ ){
        for(int h = 0; h < height_; h++){
          for (int w = 0; w < width_; w ++){
            int batch_index = c * height_ * width_ + h * width_ + w;
            resize_input.at<float>(w ,h) = batch_data[batch_index];
          }
        }
        if (!inter_mode){
          cv::resize(resize_input , resize_output , cv::Size(resized_height_ , resized_width_));
        }
        else{
          cv::resize(resize_input , resize_output , cv::Size(resized_height_ , resized_width_),0 , 0 , cv::INTER_NEAREST);
        }
        for(int h = 0; h < resized_height_; h++){
          for (int w = 0; w < resized_width_; w ++){
            int batch_index =  c * resized_height_ * resized_width_ + h * resized_width_ + w;
            top_data[batch_index] = resize_output.at<float>(w ,h);
          }
        }
      }
    }
      top_data += top[0]->offset(1);
  }
}

INSTANTIATE_CLASS(DeephiResizeLayer);
REGISTER_LAYER_CLASS(DeephiResize);

} // namespace caffe
