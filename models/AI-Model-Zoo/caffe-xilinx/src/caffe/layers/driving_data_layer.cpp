#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/driving_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
DrivingDataLayer<Dtype>::DrivingDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
DrivingDataLayer<Dtype>::~DrivingDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DrivingDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  DrivingData& data = *(reader_.full().peek());

  const int grid_dim = data.car_label_resolution();
  const int width = data.car_label_width();
  const int height = data.car_label_height();
  const int full_label_width = width * grid_dim;
  const int full_label_height = height * grid_dim;

  vector<int> top_shape;
  top_shape.push_back(1);
  top_shape.push_back(3);
  top_shape.push_back(data.car_cropped_height());
  top_shape.push_back(data.car_cropped_width());
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
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
    top_shape[1] = 8;   // mask(1), bb(4), norm(2), weight(1)
    top_shape[2] = full_label_width;
    top_shape[3] = full_label_height;
    top[1]->Reshape(top_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(top_shape);
    }
  }
}

template <typename Dtype>
bool DrivingDataLayer<Dtype>::ReadBoundingBoxLabel(const DrivingData& data, int car_cropped_width, int car_cropped_height, Dtype* label) 
{
  const int grid_dim = data.car_label_resolution();
  const int width = data.car_label_width();
  const int height = data.car_label_height();
  const int full_label_width = width * grid_dim;
  const int full_label_height = height * grid_dim;
  const Dtype half_shrink_factor = data.car_shrink_factor() / 2;
  const Dtype whole_factor = data.car_whole_factor();
  const Dtype scaling = static_cast<Dtype>(full_label_width) / car_cropped_width;

  // 1 pixel label, 4 bounding box coordinates, 3 normalization labels.
  const int num_total_labels = 8;
  vector<cv::Mat *> labels;
  for (int i = 0; i < num_total_labels; ++i) {
    labels.push_back(
        new cv::Mat(full_label_height,
                    full_label_width, CV_32F,
                    cv::Scalar(0.0)));
  }

  for (int i = 0; i < data.car_boxes_size(); ++i) {
    int xmin = data.car_boxes(i).xmin();
    int ymin = data.car_boxes(i).ymin();
    int xmax = data.car_boxes(i).xmax();
    int ymax = data.car_boxes(i).ymax();
    Dtype area1 = (Dtype)(ymax - ymin) * (xmax - xmin);
    xmin = std::min(std::max(0, xmin), car_cropped_width);
    xmax = std::min(std::max(0, xmax), car_cropped_width);
    ymin = std::min(std::max(0, ymin), car_cropped_height);
    ymax = std::min(std::max(0, ymax), car_cropped_height);
    Dtype w = xmax - xmin;
    Dtype h = ymax - ymin;
    Dtype area2 = w * h;
    if (w < 4 || h < 4) {
      // drop boxes that are too small
      continue;
    }
    if ((area2 / area1) < whole_factor) {
      // drop boxes that out-of-bound
      continue;
    }
    // shrink bboxes
    int gxmin = cvFloor((xmin + w * half_shrink_factor) * scaling);
    int gxmax = cvCeil((xmax - w * half_shrink_factor) * scaling);
    int gymin = cvFloor((ymin + h * half_shrink_factor) * scaling);
    int gymax = cvCeil((ymax - h * half_shrink_factor) * scaling);

    CHECK_LE(gxmin, gxmax);
    CHECK_LE(gymin, gymax);
    if (gxmin >= full_label_width) {
      gxmin = full_label_width - 1;
    }
    if (gymin >= full_label_height) {
      gymin = full_label_height - 1;
    }
    CHECK_LE(0, gxmin);
    CHECK_LE(0, gymin);
    CHECK_LE(gxmax, full_label_width);
    CHECK_LE(gymax, full_label_height);
    if (gxmin == gxmax) {
      if (gxmax < full_label_width - 1) {
        gxmax++;
      } else if (gxmin > 0) {
        gxmin--;
      }
    }
    if (gymin == gymax) {
      if (gymax < full_label_height - 1) {
        gymax++;
      } else if (gymin > 0) {
        gymin--;
      }
    }
    CHECK_LT(gxmin, gxmax);
    CHECK_LT(gymin, gymax);
    if (gxmax == full_label_width) {
      gxmax--;
    }
    if (gymax == full_label_height) {
      gymax--;
    }
    cv::Rect r(gxmin, gymin, gxmax - gxmin + 1, gymax - gymin + 1);

    Dtype flabels[num_total_labels] = {(Dtype)1.0, (Dtype)xmin, (Dtype)ymin, (Dtype)xmax, (Dtype)ymax, 
        (Dtype)1.0 / w, (Dtype)1.0 / h, (Dtype)1.0};
    for (int j = 0; j < num_total_labels; ++j) {
      cv::Mat roi(*labels[j], r);
      roi = cv::Scalar(flabels[j]);
    }
  }

  int total_num_pixels = 0;
  for (int y = 0; y < full_label_height; ++y) {
    for (int x = 0; x < full_label_width; ++x) {
      if (labels[num_total_labels - 1]->at<Dtype>(y, x) == 1.0) {
        total_num_pixels++;
      }
    }
  }
  if (total_num_pixels != 0) {
    Dtype reweight_value = 1.0 / total_num_pixels;
    for (int y = 0; y < full_label_height; ++y) {
      for (int x = 0; x < full_label_width; ++x) {
        if (labels[num_total_labels - 1]->at<Dtype>(y, x) == 1.0) {
          labels[num_total_labels - 1]->at<Dtype>(y, x) = reweight_value;
        }
      }
    }
  }

  for (int m = 0; m < num_total_labels; ++m) {
    for (int y = 0; y < full_label_height; ++y) {
      for (int x = 0; x < full_label_width; ++x) {
        Dtype adjustment = 0;
        Dtype val = labels[m]->at<Dtype>(y, x);
        if (m == 0 || m > 4) {
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

  for (int i = 0; i < num_total_labels; ++i) {
    delete labels[i];
  }

  return true;
}

// This function is called on prefetch thread
template<typename Dtype>
void DrivingDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
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
  DrivingData& data1 = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape;
  top_shape.push_back(1);
  top_shape.push_back(3);
  top_shape.push_back(data1.car_cropped_height());
  top_shape.push_back(data1.car_cropped_width());
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
    DrivingData& data = *(reader_.full().pop("Waiting for data"));
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

    // save image and label for debug
    /*static int count = 0;
    if (count++ < 100) {
	char fileName[100];
	sprintf(fileName, "temp/image%03d.temp", count);
        FILE* fid = fopen(fileName, "wb");
        fwrite(top_data + offset, sizeof(float), top_shape[1] * top_shape[2] * top_shape[3], fid);
        fclose(fid);

	sprintf(fileName, "temp/label%03d.temp", count);
        fid = fopen(fileName, "wb");
        int size = top_shape[2] / 4;
        fwrite(top_label + offset1, sizeof(float), size * size * 8, fid);
        fclose(fid);
    }*/

    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<DrivingData*>(&data));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DrivingDataLayer);
REGISTER_LAYER_CLASS(DrivingData);

}  // namespace caffe
