/*
* Copyright 2019 Xilinx Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "opencv2/opencv.hpp"

#include "caffe/blob.hpp"
#include "caffe/layers/segment_pixel_iou_layer.hpp"

using std::vector;
using caffe::Blob;

namespace caffe {

    template<typename Dtype>
    void SegmentPixelIOULayer<Dtype>::LayerSetUp(
            const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
        tp_.resize(bottom[0]->shape(1), 0);
        fp_.resize(bottom[0]->shape(1), 0);
        fn_.resize(bottom[0]->shape(1), 0);
    }

    template<typename Dtype>
    void SegmentPixelIOULayer<Dtype>::Reshape(
            const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
        CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
        CHECK_EQ(bottom[1]->shape(1), 1);

        vector<int> top_shape;
        top_shape.push_back(3);
        top_shape.push_back(bottom[0]->shape(1));
        top[0]->Reshape(top_shape);
        std::fill(tp_.begin(), tp_.end(), 0);
        std::fill(fp_.begin(), fp_.end(), 0);
        std::fill(fn_.begin(), fn_.end(), 0);
    }

    template<typename Dtype>
    void SegmentPixelIOULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                                  const vector<Blob<Dtype> *> &top) {
        const Blob<Dtype>* bottom_data = bottom[0];  // N * C * H' * W'
        const Blob<Dtype>* bottom_label = bottom[1];  // N * 1 * H * W

        vector<Dtype> iou(bottom_data->shape(1));
        for (int n = 0; n < bottom_data->shape(0); n++) {
            // get predict label on the max-pooling of channels
            cv::Mat src(bottom_data->shape(2), bottom_data->shape(3), CV_32S);

            for (int h = 0; h < bottom_data->shape(2); h++) {
                for (int w = 0; w < bottom_data->shape(3); w++) {
                    Dtype max = bottom_data->data_at(n, 0, h, w);
                    int idx = 0;
                    for (int c = 1; c < bottom_data->shape(1); c++) {
                        if (bottom_data->data_at(n, c, h, w) > max) {
                            max = bottom_data->data_at(n, c, h, w);
                            idx = c;
                        }
                    }
                    src.at<int>(h, w) = idx;
                }
            }

            // resize to H * W
            cv::Mat dst(bottom_label->shape(2), bottom_label->shape(3), CV_32S);
            cv::resize(src, dst, cv::Size(bottom_label->shape(3), bottom_label->shape(2)), 0, 0, cv::INTER_NEAREST);

            // compare dst with bottom_label
            for (int h = 0; h < bottom_label->shape(2); ++h) {
                for (int w = 0; w < bottom_label->shape(3); ++w) {
                    int c_label = bottom_label->data_at(n, 0, h, w);
//                    if (labels_map_.count(c_label)) {
//                        c_label = labels_map_[c_label];
//                    } else {
//                        continue;
//                    }
                    if (c_label == 255) {
                      continue;
                    }
                    int c_predict = dst.at<int>(h, w);

                    CHECK_LT(c_label, bottom_data->shape(1));
                    CHECK_LT(c_predict, bottom_data->shape(1));

                    if (c_label == c_predict) {
                        tp_[c_label] ++;
                    } else {
                        fn_[c_label] ++;
                        fp_[c_predict] ++;
                    }
                }
            }
        }

        for (int c = 0; c < bottom_data->shape(1); ++c) {
            top[0]->mutable_cpu_data()[c] = tp_[c];
            top[0]->mutable_cpu_data()[bottom_data->shape(1) + c] = fp_[c];
            top[0]->mutable_cpu_data()[bottom_data->shape(1) * 2 + c] = fn_[c];
        }
    }

//    STUB_GPU(SegmentPixelIOULayer);

    INSTANTIATE_CLASS(SegmentPixelIOULayer);
    REGISTER_LAYER_CLASS(SegmentPixelIOU);
}
