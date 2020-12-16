#include <vector>

#include "caffe/layers/tiling_layer.hpp"

namespace caffe {

template <typename Dtype>
void TilingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  TilingParameter tiling_param = this->layer_param_.tiling_param();
  tile_dim_ = tiling_param.tile_dim();
  tile_dim_sq_ = tile_dim_ * tile_dim_;
  CHECK(tile_dim_) << "tile_dim must be specified.";
  CHECK_GT(tile_dim_, 0) << "tile_dim must be positive.";
}

template <typename Dtype>
void TilingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  input_channels_ = bottom[0]->channels();
  input_height_ = bottom[0]->height();
  input_width_ = bottom[0]->width();
  output_channels_ = bottom[0]->channels() / tile_dim_sq_;
  output_width_ = input_width_ * tile_dim_;
  output_height_ = input_height_ * tile_dim_;
  count_per_output_map_ = output_width_ * output_height_;
  count_per_input_map_ = input_width_ * input_height_;
  CHECK_EQ(0, input_channels_ % tile_dim_sq_)
      << "The number of input channels for tiling layer must be multiples "
      << "of the tile_dim.";
  top[0]->Reshape(bottom[0]->num(), input_channels_ / tile_dim_sq_,
      input_height_ * tile_dim_, input_width_ * tile_dim_);
}

template <typename Dtype>
void TilingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < output_channels_; ++c) {
      for (int iy = 0, oy = 0; iy < input_height_; ++iy, oy += tile_dim_) {
        for (int ix = 0, ox = 0; ix < input_width_; ++ix, ox += tile_dim_) {
          int input_channel_offset = 0;
          for (int ty = 0; ty < tile_dim_; ++ty) {
            for (int tx = 0; tx < tile_dim_; ++tx, input_channel_offset += count_per_input_map_) {
              top_data[(oy + ty) * output_width_ + ox + tx] =
                  bottom_data[input_channel_offset + iy * input_width_ + ix];
            }
          }
        }
      }
      bottom_data += count_per_output_map_;
      top_data += count_per_output_map_;
    }
  }
}

template <typename Dtype>
void TilingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < output_channels_; ++c) {
      for (int iy = 0, oy = 0; iy < input_height_; ++iy, oy += tile_dim_) {
        for (int ix = 0, ox = 0; ix < input_width_; ++ix, ox += tile_dim_) {
          int input_channel_offset = 0;
          for (int ty = 0; ty < tile_dim_; ++ty) {
            for (int tx = 0; tx < tile_dim_; ++tx, input_channel_offset += count_per_input_map_) {
              bottom_diff[input_channel_offset + iy * input_width_ + ix] =
                  top_diff[(oy + ty) * output_width_ + ox + tx];
            }
          }
        }
      }
      bottom_diff += count_per_output_map_;
      top_diff += count_per_output_map_;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TilingLayer);
#endif

INSTANTIATE_CLASS(TilingLayer);
REGISTER_LAYER_CLASS(Tiling);

}  // namespace caffe
