#include <vector>

#include "caffe/layers/reshape_label_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    template <typename Dtype>
    void ReshapeLabelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        int bottom_offset = 0;

        Dtype* top_data;
        int cnt[5] = {
                    8 * full_label_size_ * full_label_size_, 
                    full_label_size_ * full_label_size_ / 4, 
                    full_label_size_ * full_label_size_ / 16, 
                    full_label_size_ * full_label_size_ / 64,
                    4000};
        for (int i = 0; i < bottom[0]->num(); ++i) {
            for (int j = 0; j < 4; ++j) {
                top_data = top[3-j]->mutable_gpu_data();
                top_data += cnt[j] * i ;
                caffe_copy(cnt[j], bottom_data + bottom_offset, top_data);
                bottom_offset += cnt[j];
            }
            top_data = top[4]->mutable_gpu_data();
            top_data += cnt[4] * i;
            caffe_copy(cnt[4], bottom_data + bottom_offset, top_data);
            bottom_offset += cnt[4];
        }
    }
    
    template <typename Dtype>
    void ReshapeLabelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    }
    
    
INSTANTIATE_LAYER_GPU_FUNCS(ReshapeLabelLayer);

}
