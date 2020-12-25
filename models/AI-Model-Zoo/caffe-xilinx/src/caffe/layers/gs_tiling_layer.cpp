#include <vector>

#include "caffe/layers/gs_tiling_layer.hpp"

namespace caffe {

template <typename Dtype>
void GSTilingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  GSTilingParameter tiling_param = this->layer_param_.gs_tiling_param();
  stride_ = tiling_param.stride();
  stride_sq_ = stride_ * stride_;
  reverse_ = tiling_param.reverse();
  CHECK(stride_) << "tile_dim must be specified.";
  CHECK_GT(stride_, 0) << "tile_dim must be positive.";
}

template <typename Dtype>
void GSTilingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  input_channels_ = bottom[0]->channels();
  input_height_ = bottom[0]->height();
  input_width_ = bottom[0]->width();
  if(reverse_){
    output_channels_ = input_channels_ / stride_sq_;
    output_width_ = input_width_ * stride_;
    output_height_ = input_height_ * stride_;
    CHECK_EQ(0, input_channels_ % stride_sq_)
      << "The number of input channels for tiling layer must be multiples "
      << "of the stride.";
    
  }else {
    output_channels_ = input_channels_ * stride_sq_;
    output_width_ = input_width_ / stride_;
    output_height_ = input_height_ / stride_;
    CHECK_EQ(0, input_width_ % stride_)
      << "The width of input channels for tiling layer must be multiples "
      << "of the stride.";
    CHECK_EQ(0, input_height_ % stride_)
      << "The height of input channels for tiling layer must be multiples "
      << "of the stride.";
  }
  
  count_per_output_map_ = output_width_ * output_height_ * output_channels_;
  count_per_input_map_ = input_width_ * input_height_ * input_channels_;
  top[0]->Reshape(bottom[0]->num(), output_channels_,
      output_height_, output_width_);
}

template <typename Dtype>
void GSTilingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int ox,oy,oc,oi;
  for (int n = 0; n < bottom[0]->num(); ++n) {
    int ii = 0;
    for (int ic = 0; ic < input_channels_; ++ic) {
      for (int iy = 0; iy < input_height_; ++iy) {
        for (int ix = 0; ix < input_width_; ++ix) {
          if(!reverse_){
            ox = ix / stride_;
            oy = iy / stride_;
            oc = ((iy % stride_) * stride_ + ix % stride_) * input_channels_ + ic;
            //int oi = (oc * output_height_ + oy) * output_width_ + ox;
          }else
          {
            int off = ic / output_channels_;
            ox = ix * stride_ + off % stride_;
            oy = iy * stride_ + off / stride_;
            oc = ic % output_channels_;
          }
          oi = (oc * output_height_ + oy) * output_width_ + ox;
          top_data[oi] = bottom_data[ii];
          ii++;
        }
      }
    }
    bottom_data += count_per_input_map_;
    top_data += count_per_output_map_;
  }
}

template <typename Dtype>
void GSTilingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int ox,oy,oc,oi;
  for (int n = 0; n < bottom[0]->num(); ++n) {
    int ii = 0;
    for (int ic = 0; ic < input_channels_; ++ic) {
      for (int iy = 0; iy < input_height_; ++iy) {
        for (int ix = 0; ix < input_width_; ++ix) {
          if(!reverse_){
            ox = ix / stride_;
            oy = iy / stride_;
            oc = ((iy % stride_) * stride_ + ix % stride_) * input_channels_ + ic;
            //int oi = (oc * output_height_ + oy) * output_width_ + ox;
          }else
          {
            int off = ic / output_channels_;
            ox = ix * stride_ + off % stride_;
            oy = iy * stride_ + off / stride_;
            oc = ic % output_channels_;
          }
          oi = (oc * output_height_ + oy) * output_width_ + ox;
          bottom_diff[ii] = top_diff[oi];
          ii++;
        }
      }
    }
    bottom_diff += count_per_input_map_;
    top_diff += count_per_output_map_;
  }
}

#ifdef CPU_ONLY
STUB_GPU(GSTilingLayer);
#endif

INSTANTIATE_CLASS(GSTilingLayer);
REGISTER_LAYER_CLASS(GSTiling);

}  // namespace caffe
