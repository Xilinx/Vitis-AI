#ifdef USE_BOOST
#include <boost/thread.hpp>
#else
#include <mutex>
#endif
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void Layer<Dtype>::InitMutex() {
#ifdef USE_BOOST
  forward_mutex_.reset(new boost::mutex());
#else
  forward_mutex_.reset(new std::mutex());
#endif
}

template <typename Dtype>
void Layer<Dtype>::Lock() {
  if (IsShared()) {
    forward_mutex_->lock();
  }
}

template <typename Dtype>
void Layer<Dtype>::Unlock() {
  if (IsShared()) {
    forward_mutex_->unlock();
  }
}

INSTANTIATE_CLASS(Layer);

}  // namespace caffe
