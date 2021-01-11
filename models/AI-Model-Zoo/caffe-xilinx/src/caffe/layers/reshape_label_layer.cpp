#include <algorithm>
#include <vector>

#include "caffe/layers/reshape_label_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    template <typename Dtype>
    void ReshapeLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        const ReshapeLabelParameter& reshape_label_param = this->layer_param_.reshape_label_param();
        full_label_size_ = reshape_label_param.full_label_size();
    }
    
    template <typename Dtype>
    void ReshapeLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        int batch_size = bottom[0]->num();
        top[3]->Reshape(batch_size, 8, full_label_size_, full_label_size_);
        int iscale = 2;
        for (int i = 2; i >= 0; --i) {
            top[i]->Reshape(batch_size, 1, full_label_size_ / iscale, full_label_size_ / iscale);
            iscale *= 2;
        }
        top[4]->Reshape(batch_size,1,1000,4);
    }
    
    template <typename Dtype>
    void ReshapeLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        int bottom_offset = 0;
        
        Dtype* top_data;
        int cnt[5] = {
                    8 * full_label_size_ * full_label_size_, 
                    full_label_size_ * full_label_size_ / 4, 
                    full_label_size_ * full_label_size_ / 16, 
                    full_label_size_ * full_label_size_ / 64,
                    4000 };
        for (int i = 0; i < bottom[0]->num(); ++i) {
            for (int j = 0; j < 4; ++j) {
                top_data = top[3-j]->mutable_cpu_data();
                top_data += cnt[j] * i ;
                caffe_copy(cnt[j], bottom_data + bottom_offset, top_data);
                bottom_offset += cnt[j];
            }
            top_data = top[4]->mutable_cpu_data();
            top_data += cnt[4] * i;
            caffe_copy(cnt[4], bottom_data + bottom_offset, top_data);
            bottom_offset += cnt[4];
        }
    }
    
    template <typename Dtype>
    void ReshapeLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        
    }
    
#ifdef CPU_ONLY
STUB_GPU(ReshapeLabelLayer);
#endif

INSTANTIATE_CLASS(ReshapeLabelLayer);
REGISTER_LAYER_CLASS(ReshapeLabel);
    
}
