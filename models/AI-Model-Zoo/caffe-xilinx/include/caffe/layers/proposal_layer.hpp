#ifndef CAFFE_PROPOSAL_LAYER_HPP_
#define CAFFE_PROPOSAL_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
class ProposalLayer : public Layer<Dtype> {
public:
	explicit ProposalLayer(const LayerParameter& param)
			: Layer<Dtype>(param){}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>&top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline int ExactNumBottomBlobs() const { return 3; }
	virtual inline int ExactNumTopBlobs() const { return 1; }
	virtual inline const char* type() const { return "Proposal"; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		for(size_t i=0; i<propagate_down.size(); i++) {
			if(propagate_down[i]) {
				NOT_IMPLEMENTED;
			}
		}
	}
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	void CalcBBoxes(int num, int width, int height, int anchor_num, Dtype stride_h,
			Dtype stride_w, const Dtype* img_info, const Dtype* reg, const Dtype* anchor, Dtype* bbox);
	vector<Dtype> Nms(int num, Dtype img_scale, const Dtype* bbox, const Dtype* score);
	vector<Dtype> Nms(int img_num, int num_per_img, const Dtype* img_info, const Dtype* bbox, const Dtype* score);
#ifndef CPU_ONLY
	vector<int> NmsGPU(int num, Dtype img_scale, const Dtype* bbox, const Dtype* score);
	void NmsGPU(int img_num, int num_per_img, const Dtype* img_info, const Dtype* bbox, const Dtype* score, Blob<Dtype>& output);
#endif

	Blob<Dtype> anchor;
	Blob<Dtype> bboxes;
	Blob<unsigned int> exist;
	Blob<unsigned int> overlap;
	Blob<int> index;
	int stride_h;
	int stride_w;
	int nms_input_num;
	int nms_output_num;
	Dtype min_width;
	Dtype min_height;
	Dtype nms_iou_th;
};

}  // namespace caffe

#endif // CAFFE_PROPOSAL_LAYER_HPP_
