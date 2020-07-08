#ifndef CAFFE_IMAGE_SEG_DATA_LAYER_HPP_
#define CAFFE_IMAGE_SEG_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**  Lequan add
 * 
 * @brief Provides segmentation data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */

template <typename Dtype>
 class ImageSegDataLayer : public BasePrefetchingDataLayer<Dtype> {
  public:
   explicit ImageSegDataLayer(const LayerParameter& param)
       : BasePrefetchingDataLayer<Dtype>(param) {}
   virtual ~ImageSegDataLayer();
   virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top);
 
   virtual inline const char* type() const { return "ImageSegData"; }
   virtual inline int ExactNumBottomBlobs() const { return 0; }
   virtual inline int ExactNumTopBlobs() const { return 2; }
 
  protected:
   Blob<Dtype> transformed_label_;
   shared_ptr<Caffe::RNG> prefetch_rng_;
   virtual void ShuffleImages();
   virtual void load_batch(Batch<Dtype>* batch);
 
   vector<std::pair<std::string, std::string> > lines_;
   int lines_id_;
 };
/*
template <typename Dtype>
class ImageSegDataLayer : public ImageDimPrefetchingDataLayer<Dtype> {
 public:
  explicit ImageSegDataLayer(const LayerParameter& param)
    : ImageDimPrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageSegDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageSegData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }
  virtual inline bool AutoTopBlobs() const { return true; }

 protected:
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  Blob<Dtype> transformed_label_;
  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<std::pair<std::string, std::string> > lines_;
  int lines_id_;
  };
*/
}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
