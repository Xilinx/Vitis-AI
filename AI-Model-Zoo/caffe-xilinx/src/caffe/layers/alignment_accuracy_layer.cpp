#include <vector>

#include <opencv2/core/core.hpp>

#include "caffe/layers/alignment_accuracy_layer.hpp"

using namespace cv;

namespace caffe {

template <typename Dtype>
static inline Dtype Distance(const cv::Point_<Dtype>& p1, const cv::Point_<Dtype>& p2) {
	return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

template <typename Dtype>
void AlignmentAccuracyLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const AlignmentAccuracyParameter& alignment_param = this->layer_param_.alignment_accuracy_param();

	ref_start_.clear();
	ref_end_.clear();

    std::copy(alignment_param.ref_start().begin(), alignment_param.ref_start().end(), std::back_inserter(ref_start_));
    std::copy(alignment_param.ref_end().begin(), alignment_param.ref_end().end(), std::back_inserter(ref_end_));

	CHECK_GT(ref_start_.size(), 0);
	CHECK_GT(ref_end_.size(), 0);

	accuracy_for_each_point_ = alignment_param.accuracy_for_each_point();
	threshold_ = alignment_param.threshold();
}

template <typename Dtype>
void AlignmentAccuracyLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom[0]->num(), bottom[1]->num());
	CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1));
	CHECK_EQ(bottom[0]->count(1)%2, 0);
	top[0]->Reshape(vector<int>(1, 1));
	if(accuracy_for_each_point_) {
		top[1]->Reshape(vector<int>(1, bottom[0]->count(1)/2));
		CHECK_EQ(top.size(), 2);
	}
}

template <typename Dtype>
Dtype AlignmentAccuracyLayer<Dtype>::GetRefDistance(const vector<cv::Point2f>& label) {
		Point2f start(0.0f, 0.0f);
		for(size_t i = 0; i < ref_start_.size(); i++) {
			start += label[ref_start_[i]];
		}
		start *= 1.0f / ref_start_.size();
		Point2f end(0.0f, 0.0f);
		for(size_t i = 0; i < ref_end_.size(); i++) {
			end += label[ref_end_[i]];
		}
		end *= 1.0f / ref_end_.size();
		return threshold_ * Distance(start, end);
}

template <typename Dtype>
void AlignmentAccuracyLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_result_with_offset = bottom[0]->cpu_data();
	const Dtype* bottom_label_with_offset = bottom[1]->cpu_data();
	int point_num = bottom[0]->count(1) / 2;

	vector<vector<Dtype> > weight_per_point(bottom[0]->num());
	vector<vector<bool> > hit_per_point(bottom[0]->num());
	for(size_t i = 0; i < hit_per_point.size(); i++) {
		weight_per_point[i].resize(point_num, Dtype(1));
		hit_per_point[i].resize(point_num, false);
	}

	for(int num = 0; num < bottom[0]->num(); num++) {
		vector<Point2f> result(point_num);
		for(size_t i = 0; i < result.size(); i++) {
			result[i].x = bottom_result_with_offset[i * 2];
			result[i].y = bottom_result_with_offset[i * 2 + 1];
		}
                
		vector<Point2f> label(point_num);
		for(size_t i = 0; i<label.size(); i++) {
			label[i].x = bottom_label_with_offset[i * 2];
			label[i].y = bottom_label_with_offset[i * 2 + 1];
		}

		Dtype ref = GetRefDistance(label);

		for(size_t i = 0; i < hit_per_point[num].size(); i++) {
			hit_per_point[num][i] = Distance(label[i], result[i]) < ref;
		}

		bottom_result_with_offset += 2 * point_num;
		bottom_label_with_offset += 2 * point_num;
	}

	if(bottom.size() == 3) {
		const Dtype* bottom_weight = bottom[2]->cpu_data();
		for(size_t i = 0; i < weight_per_point.size(); i++) {
			for(size_t j = 0; j < weight_per_point[i].size(); j++) {
				weight_per_point[i][j] = bottom_weight[(i * point_num + j) * 2];
			}
		}
	}

	Dtype pckh(0);
        for(int i = 0; i < point_num; i++) {
            Dtype pckh_point_ave(0);
            for(int j = 0; j < bottom[0]->num(); j++) {
                pckh_point_ave += (weight_per_point[j][i] == 1) ? hit_per_point[j][i] : 1;
            }
            pckh_point_ave /= bottom[0]->num();
        
            if (accuracy_for_each_point_)
                top[1]->mutable_cpu_data()[i] = pckh_point_ave;
            pckh += pckh_point_ave;
        }
        pckh /= point_num;
	top[0]->mutable_cpu_data()[0] = pckh;
}

INSTANTIATE_CLASS(AlignmentAccuracyLayer);
REGISTER_LAYER_CLASS(AlignmentAccuracy);

}  // namespace caffe
