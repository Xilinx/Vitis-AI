#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#endif  // USE_OPENCV

#ifndef DEBUG_TEST
#define DEBUG_TEST
#undef DEBUG_TEST
#endif

#include <stdint.h>
#include <vector>
#include <math.h>
#include "caffe/data_transformer.hpp"
#include "caffe/layers/direct_regression_nnn_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
DirectRegressionNnnDataLayer<Dtype>::DirectRegressionNnnDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
DirectRegressionNnnDataLayer<Dtype>::~DirectRegressionNnnDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DirectRegressionNnnDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  DirectRegressionData& data = *(reader_.full().peek());
  const int grid_dim = data.text_label_resolution();
  const int width = data.text_label_width();
  const int height = data.text_label_height();
  LOG(INFO) << "width: " << width << " height: " << height;
  const int full_label_width = width * grid_dim;
  const int full_label_height = height * grid_dim;
  vector<int> top_shape;
  top_shape.push_back(1);
  top_shape.push_back(3);
  top_shape.push_back(data.text_cropped_height());
  top_shape.push_back(data.text_cropped_width());
  this->transformed_data_.Reshape(top_shape);
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    top_shape[1] = 12;   // mask(1), bb(4), norm(2), weight(1)
    top_shape[2] = full_label_height;
    top_shape[3] = full_label_width;
    LOG(ERROR) << "top_shape: " << full_label_width << " " << full_label_height; 
    top[1]->Reshape(top_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(top_shape);
    }
  }
}

template <typename Dtype>
bool DirectRegressionNnnDataLayer<Dtype>::ReadBoundingBoxLabel(const DirectRegressionData& data, int text_cropped_width, int text_cropped_height, Dtype* label) 
{
  const int grid_dim = data.text_label_resolution();
  const string image_source = data.text_img_source();
  const int width = data.text_label_width();
  const int height = data.text_label_height();
  const int full_label_width = width * grid_dim;
  const int full_label_height = height * grid_dim;
  const Dtype half_shrink_factor = data.text_shrink_factor() / 2;
  const Dtype whole_factor = data.text_whole_factor();
  const Dtype scaling = static_cast<Dtype>(full_label_width) / text_cropped_width;
  // 1 pixel label, 4 bounding box coordinates, 3 normalization labels.
  const int num_total_labels = 12;
  vector<cv::Mat *> labels;
  for (int i = 0; i < num_total_labels; ++i) {
    labels.push_back(
        new cv::Mat(full_label_height,
                    full_label_width, CV_32F,
                    cv::Scalar(0.0)));
  }
  #ifdef DEBUG_TEST
  LOG(INFO) << "image_source: case -1 " << image_source;
  LOG(INFO) << "box size: " << data.text_boxes_size();
  #endif                                              
  for (int i = 0; i < data.text_boxes_size(); ++i) {
    int x_1 = data.text_boxes(i).x_1();
    int y_1 = data.text_boxes(i).y_1();
    int x_2 = data.text_boxes(i).x_2();
    int y_2 = data.text_boxes(i).y_2();
    int x_3 = data.text_boxes(i).x_3();
    int y_3 = data.text_boxes(i).y_3();
    int x_4 = data.text_boxes(i).x_4();
    int y_4 = data.text_boxes(i).y_4();
    #ifdef DEBUG_TEST
    LOG(INFO) << "original: " << std::endl;
    LOG(INFO) << "x_1: " << x_1; 
    LOG(INFO) << "y_1: " << y_1;
    LOG(INFO) << "x_2: " << x_2; 
    LOG(INFO) << "y_2: " << y_2;
    LOG(INFO) << "x_3: " << x_3; 
    LOG(INFO) << "y_3: " << y_3;
    LOG(INFO) << "x_4: " << x_4; 
    LOG(INFO) << "y_4: " << y_4;
    #endif
    Dtype area_before = (Dtype)(y_4 + y_3 - y_1 - y_2) * (x_2 + x_3 - x_1 - x_4) / 4;
    x_1 = std::min(std::max(0, x_1), text_cropped_width - 1);
    x_2 = std::min(std::max(0, x_2), text_cropped_width - 1);
    x_3 = std::min(std::max(0, x_3), text_cropped_width - 1);
    x_4 = std::min(std::max(0, x_4), text_cropped_width - 1);
    y_1 = std::min(std::max(0, y_1), text_cropped_height - 1);
    y_2 = std::min(std::max(0, y_2), text_cropped_height - 1);
    y_3 = std::min(std::max(0, y_3), text_cropped_height - 1);
    y_4 = std::min(std::max(0, y_4), text_cropped_height - 1);
    Dtype w_1 = x_2 - x_1;
    Dtype w_2 = x_3 - x_4;
    Dtype h_1 = y_4 - y_1;
    Dtype h_2 = y_3 - y_2;
    Dtype area_after = (Dtype)(y_4 + y_3 - y_1 - y_2) * (x_2 + x_3 - x_1 - x_4) / 4; 
    if (w_1 < 4 || w_2 < 4 || h_1 < 4 || h_2 < 4) {
      #ifdef DEBUG_TEST
      LOG(INFO) << "image_source: case 0 " << image_source;
      LOG(INFO) << "x_1: " << x_1; 
      LOG(INFO) << "y_1: " << y_1;
      LOG(INFO) << "x_2: " << x_2; 
      LOG(INFO) << "y_2: " << y_2;
      LOG(INFO) << "x_3: " << x_3; 
      LOG(INFO) << "y_3: " << y_3;
      LOG(INFO) << "x_4: " << x_4; 
      LOG(INFO) << "y_4: " << y_4;
      #endif                                       
      continue;       
    }
    if ((area_after / area_before) < whole_factor) {
      #ifdef DEBUG_TEST
      LOG(INFO) << "image_source: case 1 " << image_source;
      #endif
      continue;
    }
    int gx_1 = cvFloor((x_1 + w_1 * half_shrink_factor / 2) * scaling);
    int gx_2 = cvCeil((x_2 - w_1 * half_shrink_factor / 2) * scaling);
    int gx_3 = cvCeil((x_3 - w_2 * half_shrink_factor / 2) * scaling);
    int gx_4 = cvFloor((x_4 + w_2 * half_shrink_factor / 2) * scaling);
    int gy_1 = cvFloor((y_1 + h_1 * half_shrink_factor) * scaling);
    int gy_2 = cvFloor((y_2 + h_2 * half_shrink_factor) * scaling);
    int gy_3 = cvCeil((y_3 - h_2 * half_shrink_factor) * scaling);
    int gy_4 = cvCeil((y_4 - h_1 * half_shrink_factor) * scaling);
    CHECK_LE(gx_1, gx_2);
    CHECK_LE(gx_4, gx_3);
    CHECK_LE(gy_1, gy_4);
    CHECK_LE(gy_2, gy_3);
    if (gx_1 >= full_label_width) {
      gx_1 = full_label_width - 1;
    }
    if (gx_4 >= full_label_width) {
      gx_4 = full_label_width - 1;
    }
    if (gy_1 >= full_label_height) {
      gy_1 = full_label_height - 1;
    }
    if (gy_2 >= full_label_height) {
      gy_2 = full_label_height - 1;
    }
    CHECK_LE(0, gx_1);
    CHECK_LE(0, gx_4);
    CHECK_LE(0, gy_1);
    CHECK_LE(0, gy_2);
    CHECK_LE(gx_2, full_label_width);
    CHECK_LE(gx_3, full_label_width);
    CHECK_LE(gy_3, full_label_height);
    CHECK_LE(gy_4, full_label_height);
    if (gx_1 == gx_2) {
      if (gx_2 < full_label_width - 1) {
        gx_2++;
      } else if (gx_1 > 0) {
        gx_1--;
      }
    }
    if (gx_3 == gx_4) {
      if (gx_3 < full_label_width - 1) {
        gx_3++;
      } else if (gx_4 > 0) {
        gx_4--;
      }
    }
    if (gy_1 == gy_4) {
      if (gy_4 < full_label_height - 1) {
        gy_4++;
      } else if (gy_1 > 0) {
        gy_1--;
      }
    }
    if (gy_2 == gy_3) {
      if (gy_3 < full_label_height - 1) {
        gy_3++;
      } else if (gy_2 > 0) {
        gy_2--;
      }
    }
    CHECK_LT(gx_1, gx_2);
    CHECK_LT(gx_4, gx_3);
    CHECK_LT(gy_1, gy_4);
    CHECK_LT(gy_2, gy_3);
    if (gx_2 == full_label_width) {
      gx_2--;
    }
    if (gx_3 == full_label_width) {
      gx_3--;
    }
    if (gy_3 == full_label_height) {
      gy_3--;
    }
    if (gy_4 == full_label_height) {
      gy_4--;
    }
    // cv::Rect r(gxmin, gymin, gxmax - gxmin + 1, gymax - gymin + 1);
    // Dtype w = sqrt((x_2 - x_1) * (x_2 - x_1) + (y_2 - y_1) * (y_2 - y_1));
    // Dtype h = sqrt((x_4 - x_1) * (x_4 - x_1) + (y_4 - y_1) * (y_4 - y_1));
    Dtype w = std::max(std::max(x_2, x_3) - std::min(x_1, x_4), 1);
    Dtype h = std::max(std::max(y_3, y_4) - std::min(y_1, y_2), 1);
    Dtype flabels[num_total_labels] = {(Dtype)1.0, (Dtype)x_1, (Dtype)y_1, (Dtype)x_2, (Dtype)y_2, (Dtype)x_3, (Dtype)y_3, (Dtype)x_4, (Dtype)y_4, 1 / w, 1 / h, (Dtype)1.0};
    Dtype  y_judge_1, y_judge_2, y_judge_3, y_judge_4;
    #ifdef DEBUG_TEST
    int case_section = 0; 
    LOG(INFO) << "image_source: case 2 " << image_source; 
    #endif
    for (int j = 0; j < num_total_labels; ++j) {
      for (int y = 0; y < full_label_height; y++) {
        for (int x = 0; x < full_label_width; x++) {
          if ((y < gy_1 && y < gy_2) || (y > gy_3 && y > gy_4)) {
            continue;  
          }
          if ((x < gx_1 && x < gx_4) || (x > gx_2 && x > gx_3)) {
            continue;
          }
          if (gy_1 < gy_2 - 1) {
            #ifdef DEBUG_TEST
            case_section = 3;
            #endif
            y_judge_1 = static_cast<Dtype>(x - gx_1) / std::max((gx_2 - gx_1), 1) * (gy_2 - gy_1) + gy_1;
            y_judge_2 = static_cast<Dtype>(x - gx_2) / std::min((gx_3 - gx_2), -1) * (gy_3 - gy_2) + gy_2;  
            y_judge_3 = static_cast<Dtype>(x - gx_3) / std::min((gx_4 - gx_3), -1) * (gy_4 - gy_3) + gy_3;
            y_judge_4 = static_cast<Dtype>(x - gx_4) / std::max((gx_1 - gx_4), 1) * (gy_1 - gy_4) + gy_4; 
            if (y_judge_1 <= y && y_judge_2 >= y &&  y_judge_3 >= y && y_judge_4 <= y) {
              labels[j]->at<Dtype>(y, x) = flabels[j];
            } 
          } else if (gy_1 - 1 > gy_2){
            #ifdef DEBUG_TEST
            case_section = 4;
            #endif
            y_judge_1 = static_cast<Dtype>(x - gx_1) / std::max((gx_2 - gx_1), 1) * (gy_2 - gy_1) + gy_1;
            y_judge_2 = static_cast<Dtype>(x - gx_2) / std::max((gx_3 - gx_2), 1) * (gy_3 - gy_2) + gy_2;  
            y_judge_3 = static_cast<Dtype>(x - gx_3) / std::min((gx_4 - gx_3), -1) * (gy_4 - gy_3) + gy_3;
            y_judge_4 = static_cast<Dtype>(x - gx_4) / std::min((gx_1 - gx_4), -1) * (gy_1 - gy_4) + gy_4; 
            if (y_judge_1 <= y && y_judge_2 <= y && y_judge_3 >= y && y_judge_4 >= y) {
              labels[j]->at<Dtype>(y, x) = flabels[j];
            } 
          } else {
            #ifdef DEBUG_TEST
            case_section = 5;
            #endif
            labels[j]->at<Dtype>(y, x) = flabels[j];
          } 
        }  
      }
    }
    #ifdef DEBUG_TEST
    LOG(INFO) <<"image_path: " << case_section << " " << image_source;
    #endif
  }
  Dtype reweight = 0;
  for (int y = 0; y < full_label_height; y++) {
    for (int x = 0; x < full_label_width; x++) {
      if (labels[0]->at<Dtype>(y, x) == 1) {
        reweight ++;
      }
    } 
  }
  reweight = 1 / reweight;
  CHECK_GT(reweight, 0.0);
  for (int y = 0; y < full_label_height; y++) {
    for (int x = 0; x < full_label_width; x++) {
      if (labels[11]->at<Dtype>(y, x) == 1) {
        labels[11]->at<Dtype>(y, x) = reweight;
      }
    } 
  }
  for (int m = 0; m < num_total_labels; ++m) {
    for (int y = 0; y < full_label_height; ++y) {
      for (int x = 0; x < full_label_width; ++x) {
        Dtype adjustment = 0;
        Dtype val = labels[m]->at<Dtype>(y, x);
        if (m == 0 || m > 8) {
          // do nothing
        } else if (labels[0]->at<Dtype>(y, x) == 0.0) {
          // do nothing
        } else if (m % 2 == 1) {
          // x coordinate
          adjustment = x / scaling;
        } else {
          // y coordinate
          adjustment = y / scaling;
        }
        *label = val - adjustment;
        label++;
      }
    }
  }

  #ifdef DEBUG_TEST
  for (int y = 0; y < full_label_height; y++) {
    for (int x = 0; x < full_label_width; x++) {
      labels[0]->at<Dtype>(y, x) = labels[0]->at<Dtype>(y, x) * 255;
    } 
  }
  int pos = image_source.find_last_of("/");
  // LOG(ERROR) << "pos: " << pos; 
  string image_name = image_source.substr(pos +1);
  const string &image_path = "./test_pic/" + image_name;
  // LOG(ERROR) <<"image_path: " << image_path;
  cv::imwrite(image_path, *(labels[0]));
  #endif

  for (int i = 0; i < num_total_labels; ++i) {
    delete labels[i];
  }
  
  return true;
}

// This function is called on prefetch thread
template<typename Dtype>
void DirectRegressionNnnDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  DirectRegressionData& data1 = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape;
  top_shape.push_back(1);
  top_shape.push_back(3);
  top_shape.push_back(data1.text_cropped_height());
  top_shape.push_back(data1.text_cropped_width());
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a data
    DirectRegressionData& data = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(data, &(this->transformed_data_));
    // Generate label.
    int offset1 = batch->label_.offset(item_id);
    if (this->output_labels_) {
        ReadBoundingBoxLabel(data, top_shape[3], top_shape[2], top_label + offset1);
    }
    trans_time += timer.MicroSeconds();
    reader_.free().push(const_cast<DirectRegressionData*>(&data));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DirectRegressionNnnDataLayer);
REGISTER_LAYER_CLASS(DirectRegressionNnnData);

}  
