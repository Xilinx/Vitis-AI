#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>
#include <string>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)

#include "caffe/common.hpp"
#include "caffe/layers/paf_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
PAFDataLayer<Dtype>::PAFDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    pe_transform_param_(param.pe_transform_param()){
}

template <typename Dtype>
PAFDataLayer<Dtype>::~PAFDataLayer() {
  this->StopInternalThread();
  delete[] meta_buf_;
}

template <typename Dtype>
void PAFDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  pe_data_transformer_.reset(
     new PEDataTransformer<Dtype>(pe_transform_param_, this->phase_));
  pe_data_transformer_->InitRand();
  int type = pe_transform_param_.type();
  const string& source = this->layer_param_.image_data_param().source();
  string path_source = source + "path.txt" ;
  string meta_source = source + "meta.bin";
  LOG(INFO) << "Opening file " << path_source << "," << meta_source;
  std::ifstream pathfile(path_source.c_str());
  std::ifstream metafile(meta_source.c_str());
  int meta_len = 0;
  metafile.seekg(0,ios::end);
  meta_len = metafile.tellg();
  meta_buf_ = new float[meta_len / 4];
  metafile.seekg(0,ios::beg);
  metafile.read((char*)meta_buf_,meta_len);
  metafile.close();
  string filename;
  float* label = meta_buf_;
  while (pathfile >> filename) {
    lines_.push_back(std::make_pair(filename, label));
    int len = 0;
    if (type == 1)
      len = 52+label[2]*45;
    else if(type == 2)
      len = 58+label[2]*51;
    label += len;
  }

  pathfile.close();
  lines_id_ = 0;

  // image
  cv::Mat cv_img = ReadImageToCVMat(lines_[lines_id_].first);
  const int crop_size = this->layer_param_.pe_transform_param().crop_size();
  const int batch_size = this->layer_param_.data_param().batch_size();

    const int stride = this->layer_param_.pe_transform_param().stride();
    /*
    bt = stride - cv_img.rows % stride;
    rt = stride - cv_img.cols % stride;
    if (bt == stride)
        bt = 0;
    if (rt == stride)
        rt = 0;
  if (this->phase_ != TRAIN)
      CHECK_EQ(batch_size,1) << "the batch size of TEST must be 1";
      */
  if (crop_size > 0) {
    // top[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
    // for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    //   this->prefetch_[i].data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
    // }
    // //this->transformed_data_.Reshape(1, 4, crop_size, crop_size);
    // this->transformed_data_.Reshape(1, 6, crop_size, crop_size);
  }
  else {
    const int height = this->phase_ != TRAIN ? this->layer_param_.pe_transform_param().crop_size_y(): //cv_img.rows + bt :
      this->layer_param_.pe_transform_param().crop_size_y();
    const int width = this->phase_ != TRAIN ? this->layer_param_.pe_transform_param().crop_size_x(): //cv_img.cols + rt :
      this->layer_param_.pe_transform_param().crop_size_x();
    LOG(INFO) << "PREFETCH_COUNT is " << this->PREFETCH_COUNT;
    top[0]->Reshape(batch_size, 3, height, width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(batch_size, 3, height, width);
    }
    //this->transformed_data_.Reshape(1, 4, height, width);
    this->transformed_data_.Reshape(1, 3, height, width);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // label
  if (this->output_labels_) {
    const int height = this->phase_ != TRAIN ? this->layer_param_.pe_transform_param().crop_size_y()://cv_img.rows + bt:
      this->layer_param_.pe_transform_param().crop_size_y();
    const int width = this->phase_ != TRAIN ? this->layer_param_.pe_transform_param().crop_size_x(): //cv_img.cols + rt:
      this->layer_param_.pe_transform_param().crop_size_x();

    int num_parts = this->layer_param_.pe_transform_param().num_parts();
    top[1]->Reshape(batch_size, 2*(num_parts+1), height/stride, width/stride);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(batch_size, 2*(num_parts+1), height/stride, width/stride);
    }
    this->transformed_label_.Reshape(1, 2*(num_parts+1), height/stride, width/stride);
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void PAFDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

  CPUTimer batch_timer;
  batch_timer.Start();
  double deque_time = 0;
  double decod_time = 0;
  double trans_time = 0;
  static int cnt = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape on single input batches for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  const int lines_size = lines_.size();
  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(lines_[lines_id_].first);
    float* data_buf = lines_[lines_id_].second;
    decod_time += timer.MicroSeconds();
    if (data_buf[9] == 0)
    {
      ++lines_id_;
      item_id--;
      continue;
    }
    // Apply data transformations (mirror, scale, crop...)
    timer.Start();
    const int offset_data = batch->data_.offset(item_id);
    const int offset_label = batch->label_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset_data);
    this->transformed_label_.set_cpu_data(top_label + offset_label);
    this->pe_data_transformer_->Transform(cv_img, data_buf, 
        &(this->transformed_data_),
        &(this->transformed_label_));
    ++cnt;
    // if (this->output_labels_) {
    //   top_label[item_id] = datum.label();
    // }
    ++lines_id_;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      //if (this->layer_param_.image_data_param().shuffle()) {
        //ShuffleImages();
      //}
    }
    trans_time += timer.MicroSeconds();

  }
  batch_timer.Stop();

  VLOG(2) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  VLOG(2) << "  Dequeue time: " << deque_time / 1000 << " ms.";
  VLOG(2) << "   Decode time: " << decod_time / 1000 << " ms.";
  VLOG(2) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(PAFDataLayer);
REGISTER_LAYER_CLASS(PAFData);

}  // namespace caffe
