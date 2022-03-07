#ifndef CAFFE_PAF_DATA_LAYER_HPP_
#define CAFFE_PAF_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/pe_data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class PAFDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit PAFDataLayer(const LayerParameter& param);
  virtual ~PAFDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // CPMDataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "PAFData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  Blob<Dtype> transformed_label_; // add another blob

  PETransformationParameter pe_transform_param_;
  shared_ptr<PEDataTransformer<Dtype> > pe_data_transformer_;
  float* meta_buf_;
  vector<std::pair<std::string, float*> > lines_;
  int lines_id_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
