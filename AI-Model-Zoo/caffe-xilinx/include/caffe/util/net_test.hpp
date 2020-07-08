#ifndef CAFFE_UTIL_NET_TEST_HPP_
#define CAFFE_UTIL_NET_TEST_HPP_

#include "caffe/caffe.hpp"

namespace caffe {

// test ssd/yolo
void test_detection(Net<float> &caffe_net, const string &ap_type,
                    const int &test_iter);

// test segmentation
void test_segmentation(Net<float> &caffe_net, const int &class_num,
                       const int &test_iter);

// others
void test_general(Net<float> &caffe_net, const int &test_iter);

// wrapper
void test_net_by_type(Net<float> &test_net, const string &net_type,
                      const int &test_iter, const string &ap_style,
                      const string &seg_metric, const int &seg_class_num);

} // namespace caffe

#endif // CAFFE_UTIL_NET_TEST_HPP_
