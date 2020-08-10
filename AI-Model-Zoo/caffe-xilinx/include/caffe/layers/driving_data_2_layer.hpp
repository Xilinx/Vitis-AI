#ifndef CAFFE_DRIVING_DATA_LAYER_HPP_
#define CAFFE_DRIVING_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class DrivingData2Layer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DrivingData2Layer(const LayerParameter& param);
  virtual ~DrivingData2Layer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "DrivingData2"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  DataReader<DrivingData> reader_;
  int label_bb_offset_;
 private:
  bool ReadBoundingBoxLabel(const DrivingData& data, int car_width, int car_height, Dtype* label);
};

}  // namespace caffe

#endif  // CAFFE_DRIVING_DATA_LAYER_HPP_
