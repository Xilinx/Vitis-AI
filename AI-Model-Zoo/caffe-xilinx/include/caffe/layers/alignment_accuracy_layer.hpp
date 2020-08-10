#ifndef CAFFE_ALIGNMENT_ACCURACY_LAYER_HPP_
#define CAFFE_ALIGNMENT_ACCURACY_LAYER_HPP_

#include <vector>

#include <opencv2/core/core.hpp>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class AlignmentAccuracyLayer : public Layer<Dtype> {
public:
	explicit AlignmentAccuracyLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "AlignmentAccuracy"; }

	virtual inline int MinNumBottomBlobs() const { return 2; }
	virtual inline int MaxNumBottomBlobs() const { return 3; }
	virtual inline int MinNumTopBlobs() const { return 1; }
	virtual inline int MaxNumTopBlobs() const { return 2; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		for (int i=0; i<propagate_down.size(); i++) {
			if(propagate_down[i]) {
				NOT_IMPLEMENTED;
			}
		}
	}
	Dtype GetRefDistance(const vector<cv::Point2f>& label);

	vector<int> ref_start_;
	vector<int> ref_end_;
	bool accuracy_for_each_point_;
	float threshold_;
};

}  // namespace caffe

#endif  // CAFFE_ALIGNMENT_ACCURACY_LAYER_HPP_
