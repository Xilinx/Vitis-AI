#ifndef CAFFE_ENHANCED_IMAGE_DATA_LAYER_HPP_
#define CAFFE_ENHANCED_IMAGE_DATA_LAYER_HPP_

#include "caffe/layers/image_data_layer.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class EnhancedImageDataLayer : public ImageDataLayer<Dtype> {
 public:
  explicit EnhancedImageDataLayer(const LayerParameter& param)
      : ImageDataLayer<Dtype>(param) {}
  virtual ~EnhancedImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "EnhancedImageData"; }
  // virtual inline int ExactNumBottomBlobs() const { return 0; }
  // virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  // shared_ptr<Caffe::RNG> prefetch_rng_;
  // virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, vector<int> > > lines_;
  // int lines_id_;
};


}  // namespace caffe

#endif  // CAFFE_ENHANCED_IMAGE_DATA_LAYER_HPP_
