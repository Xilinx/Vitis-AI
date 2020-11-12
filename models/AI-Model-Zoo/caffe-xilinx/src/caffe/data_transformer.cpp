#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/bbox_util.hpp"
#include "caffe/util/im_transforms.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/transformers/yolo_transformer.hpp"

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase), mean_values_gpu_ptr_(NULL) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
  if (param_.has_resize_param()) {
    CHECK_GT(param_.resize_param().height(), 0);
    CHECK_GT(param_.resize_param().width(), 0);
  }
  if (param_.has_expand_param()) {
    CHECK_GT(param_.expand_param().max_expand_ratio(), 1.);
  }
}

#ifdef USE_OPENCV
template <typename Dtype>
void DataTransformer<Dtype>::Copy(const cv::Mat& cv_img,
                                  Dtype *data) {
  const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  int top_index;
  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      const uchar* ptr = cv_img.ptr<uchar>(h);
      for (int w = 0; w < width; ++w) {
        int img_index = w*channels + c;
        top_index = (c * height + h) * width + w;
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index]);
        data[top_index] = pixel;
      }
    }
  }
}
#endif

template <typename Dtype>
void DataTransformer<Dtype>::Copy(const Datum& datum,
                                  Dtype *data) {
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    return Copy(cv_img, data);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const string& datum_data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const bool has_uint8 = datum_data.size() > 0;

  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < datum_height; ++h) {
      for (int w = 0; w < datum_width; ++w) {
        int idx = (c*datum_height + h)*datum_width + w;

        Dtype element;

        if (has_uint8) {
          element =
            static_cast<Dtype>(static_cast<uint8_t>(datum_data[idx]));
        } else {
          element = datum.float_data(idx);
        }

        data[idx] = element;
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::CopyPtrEntry(string* str,
                                          Dtype* transformed_ptr,
                                          bool output_labels, Dtype *label,
                                          BlockingQueue<string*>* free_) {
  Datum datum;
  datum.ParseFromString(*str);
  free_->push(str);

  if (output_labels) {
    *label = datum.label();
  }

  Copy(datum, transformed_ptr);

  // free_->push(datum);
}


#ifndef CPU_ONLY
template<typename Dtype>
void DataTransformer<Dtype>::TransformGPU(const Datum& datum,
                                          Dtype* transformed_data) {
  Blob<int> random_vec_;

  vector<int> random_vec_shape_;
  random_vec_shape_.push_back(3);
  random_vec_.Reshape(random_vec_shape_);

  int rand1 = 0, rand2 = 0, rand3 = 0;
  if (param_.mirror()) {
    rand1 = Rand(RAND_MAX)+1;
  }
  if (phase_ == TRAIN &&
      (param_.crop_size() || (param_.crop_h() && param_.crop_w()))) {
    rand2 = Rand(RAND_MAX)+1;
    rand3 = Rand(RAND_MAX)+1;
  }
  random_vec_.mutable_cpu_data()[0] = rand1;
  random_vec_.mutable_cpu_data()[1] = rand2;
  random_vec_.mutable_cpu_data()[2] = rand3;

  vector<int> datum_shape = InferBlobShape(datum, 1);
  Blob<Dtype> original_data;
  original_data.Reshape(datum_shape);

  Dtype* original_data_cpu_ptr = original_data.mutable_cpu_data();
  Copy(datum, original_data_cpu_ptr);
  Dtype* original_data_gpu_ptr = original_data.mutable_gpu_data();

  TransformGPU(1,
               datum.channels(),
               datum.height(),
               datum.width(),
               original_data_gpu_ptr,
               transformed_data,
               random_vec_.mutable_gpu_data());
}
#endif

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels)
        << "Specify either 1 mean_value or as many as channels: "
        << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;

  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  CHECK_GE(datum_height, crop_h);
  CHECK_GE(datum_width, crop_w);
  if ((crop_h > 0) && (crop_w > 0)) {
    height = crop_h;
    width = crop_w;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_h + 1);
      w_off = Rand(datum_width - crop_w + 1);
    } else {
      h_off = (datum_height - crop_h) / 2;
      w_off = (datum_width - crop_w) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum, const Datum& seg_datum,
                                       Dtype* transformed_data, Dtype* transformed_segmap) {
  NormalizedBBox crop_bbox;
  bool do_mirror;
  Transform(datum, seg_datum, transformed_data, transformed_segmap, &crop_bbox, &do_mirror);
}


// Image and Segment_label transform
//Add by LiWang
template<typename Dtype>
void DataTransformer<Dtype>::TransformImgAndSeg(const std::vector<cv::Mat>& cv_img_seg,
  Blob<Dtype>* transformed_data_blob, Blob<Dtype>* transformed_label_blob, const int ignore_label) {
  CHECK(cv_img_seg.size() == 2) << "Input must contain image and seg.";
  const int img_channels = cv_img_seg[0].channels();
  // height and width may change due to pad for cropping
  int img_height   = cv_img_seg[0].rows;
  int img_width    = cv_img_seg[0].cols;

  const int seg_channels = cv_img_seg[1].channels();
  int seg_height   = cv_img_seg[1].rows;
  int seg_width    = cv_img_seg[1].cols;

  const int data_channels = transformed_data_blob->channels();
  const int data_height   = transformed_data_blob->height();
  const int data_width    = transformed_data_blob->width();

  const int label_channels = transformed_label_blob->channels();
  const int label_height   = transformed_label_blob->height();
  const int label_width    = transformed_label_blob->width();
  CHECK_EQ(seg_channels, 1);
  CHECK_EQ(img_channels, data_channels);


  CHECK_EQ(label_channels, 1);
  CHECK_EQ(data_height, label_height);
  CHECK_EQ(data_width, label_width);
  CHECK_EQ(img_height, seg_height);
  CHECK_EQ(img_width, seg_width);
  //CHECK((cv_img_seg[0].depth() == CV_8U) || (cv_img_seg[0].depth() == CV_32F)) << "Image data type must be unsigned byte";
  //CHECK(cv_img_seg[1].depth() == CV_8U) << "Seg data type must be unsigned byte";

 const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img_seg[0];
  cv::Mat cv_cropped_seg = cv_img_seg[1];
  // transform to double, since we will pad mean pixel values
  cv_cropped_img.convertTo(cv_cropped_img, CV_64F);

  // copymakeborder
  int pad_height = std::max(crop_size - img_height, 0);
  int pad_width  = std::max(crop_size - img_width, 0);
  if (pad_height > 0 || pad_width > 0) {
    cv::copyMakeBorder(cv_cropped_img, cv_cropped_img, 0, pad_height,
          0, pad_width, cv::BORDER_CONSTANT,
          cv::Scalar(mean_values_[0], mean_values_[1], mean_values_[2]));
    cv::copyMakeBorder(cv_cropped_seg, cv_cropped_seg, 0, pad_height,
          0, pad_width, cv::BORDER_CONSTANT,
           cv::Scalar(ignore_label));
    // update height/width
    img_height   = cv_cropped_img.rows;
    img_width    = cv_cropped_img.cols;

    seg_height   = cv_cropped_seg.rows;
    seg_width    = cv_cropped_seg.cols;
  }
  // crop img/seg
  if (crop_size) {
    CHECK_EQ(crop_size, data_height);
    CHECK_EQ(crop_size, data_width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      // CHECK: use middle crop
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_cropped_img(roi);
    cv_cropped_seg = cv_cropped_seg(roi);
  }


  CHECK(cv_cropped_img.data);
  CHECK(cv_cropped_seg.data);

    Dtype* transformed_data  = transformed_data_blob->mutable_cpu_data();
  Dtype* transformed_label = transformed_label_blob->mutable_cpu_data();
  //float scale_esp[3] = {0.02228, 0.02167, 0.022156};
  int top_index;
  const double* data_ptr;
  const uchar* label_ptr;
  // cv::imwrite("/home/lqyu/img.jpg",cv_cropped_img);
  // cv::imwrite("/home/lqyu/seg.jpg",cv_cropped_seg);
  for (int h = 0; h < data_height; ++h) {
    data_ptr = cv_cropped_img.ptr<double>(h);
    label_ptr = cv_cropped_seg.ptr<uchar>(h);

    int data_index = 0;
    int label_index = 0;

    for (int w = 0; w < data_width; ++w) {
      // for image
      for (int c = 0; c < img_channels; ++c) {
      if (do_mirror) {
          top_index = (c * data_height + h) * data_width + (data_width - 1 - w);
        } else {
          top_index = (c * data_height + h) * data_width + w;
        }
        Dtype pixel = static_cast<Dtype>(data_ptr[data_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            //LOG(INFO)<<"mean: "<<mean_values_[c];
            //LOG(INFO)<<"scale: "<<scale_esp[c];
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }

      // for segmentation
      if (do_mirror) {
        top_index = h * data_width + data_width - 1 - w;
      } else {
        top_index = h * data_width + w;
      }
      Dtype pixel = static_cast<Dtype>(label_ptr[label_index++]);
      transformed_label[top_index] = pixel;
    }
  }
}

// do_mirror, h_off, w_off require that random values be passed in,
// because the random draws should have been taken in deterministic order
template<typename Dtype>
void DataTransformer<Dtype>::TransformPtrInt(Datum* datum,
                                             Dtype* transformed_data,
                                             int rand1, int rand2, int rand3) {
  const string& data = datum->data();
  const int datum_channels = datum->channels();
  const int datum_height = datum->height();
  const int datum_width = datum->width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && (rand1%2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels)
        << "Specify either 1 mean_value or as many as channels: "
        << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;

  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  CHECK_GE(datum_height, crop_h);
  CHECK_GE(datum_width, crop_w);
  if ((crop_h > 0) && (crop_w > 0)) {
    height = crop_h;
    width = crop_w;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = rand2 % (datum_height - crop_h + 1);
      w_off = rand3 % (datum_width - crop_w + 1);
    } else {
      h_off = (datum_height - crop_h) / 2;
      w_off = (datum_width - crop_w) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum->float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformPtrEntry(
    string* str, Dtype* transformed_ptr, int rand1, int rand2, int rand3,
    bool output_labels, Dtype *label, BlockingQueue<string*>* free_) {
  // Parse a datum from the string
  Datum datum;
  datum.ParseFromString(*str);
  free_->push(str);

  // Get label from datum if needed
  if (output_labels) {
    *label = datum.label();
  }

  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    TransformPtr(cv_img, transformed_ptr, rand1, rand2, rand3);
    return;
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  TransformPtrInt(&datum, transformed_ptr,
                  rand1, rand2, rand3);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob) {
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    return Transform(cv_img, transformed_blob);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  if ((crop_h > 0) && (crop_w > 0)) {
    CHECK_EQ(crop_h, height);
    CHECK_EQ(crop_w, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

#ifndef CPU_ONLY
  bool use_gpu_transform = param_.use_gpu_transform() &&
                           (Caffe::mode() == Caffe::GPU);
  if (use_gpu_transform) {
    Dtype* transformed_data_gpu = transformed_blob->mutable_gpu_data();
    TransformGPU(datum, transformed_data_gpu);
    transformed_blob->cpu_data();
  } else {
    Dtype* transformed_data = transformed_blob->mutable_cpu_data();
    Transform(datum, transformed_data);
  }
#else
  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data);
#endif
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(DrivingData& data,
                                       Blob<Dtype>* transformed_blob) {
  // If datum is encoded, decoded and transform the cv::image.
  const Datum& datum = data.car_image_datum();
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    Transform(cv_img, transformed_blob, data);
    // tranform bbox
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    LOG(ERROR) << "force_color and force_gray only for encoded datum";
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(DirectRegressionData& data,
                                       Blob<Dtype>* transformed_blob) {
  // If datum is encoded, decoded and transform the cv::image.
  const Datum& datum = data.text_image_datum();
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    Transform(cv_img, transformed_blob, data);
    // tranform bbox
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    LOG(ERROR) << "force_color and force_gray only for encoded datum";
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector, const vector<Datum> & seg_datum_vector,
                                       Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob) {
  const int datum_num = datum_vector.size();
  const int seg_datum_num = seg_datum_vector.size();
  const int num = transformed_blob->num();
  const int seg_num = transformed_segblob->num();
  const int channels = transformed_blob->channels();
  const int seg_channels = transformed_segblob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_GT(datum_num, 0) << "There is no seg datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
    CHECK_LE(seg_datum_num, seg_num) <<
    "The size of seg_datum_vector must be no greater than transformed_segblob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
  Blob<Dtype> uni_segblob(1, seg_channels, height, width);
  for (int item_id = 0; item_id < seg_datum_num; ++item_id) {
    int offset = transformed_segblob->offset(item_id);
    uni_segblob.set_cpu_data(transformed_segblob->mutable_cpu_data() + offset);
    Transform(seg_datum_vector[item_id], &uni_segblob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data,
                                       NormalizedBBox* crop_bbox,
                                       bool* do_mirror) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  *do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;

  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  CHECK_GE(datum_height, crop_h);
  CHECK_GE(datum_width, crop_w);
  if ((crop_h > 0) && (crop_w > 0)) {
    height = crop_h;
    width = crop_w;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_h + 1);
      w_off = Rand(datum_width - crop_w + 1);
    } else {
      h_off = (datum_height - crop_h) / 2;
      w_off = (datum_width - crop_w) / 2;
    }
  }

  // Return the normalized crop bbox.
  crop_bbox->set_xmin(Dtype(w_off) / datum_width);
  crop_bbox->set_ymin(Dtype(h_off) / datum_height);
  crop_bbox->set_xmax(Dtype(w_off + width) / datum_width);
  crop_bbox->set_ymax(Dtype(h_off + height) / datum_height);

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (*do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum, const Datum& seg_datum,
                                       Dtype* transformed_data, Dtype* transformed_segmap,
                                       NormalizedBBox* crop_bbox,
                                       bool* do_mirror) { // 3
  const string& data = datum.data();
  const string& segdata = seg_datum.data();
  const int datum_channels = datum.channels();
  const int seg_datum_channels = seg_datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  *do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GT(seg_datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  // Return the normalized crop bbox.
  crop_bbox->set_xmin(Dtype(w_off) / datum_width);
  crop_bbox->set_ymin(Dtype(h_off) / datum_height);
  crop_bbox->set_xmax(Dtype(w_off + width) / datum_width);
  crop_bbox->set_ymax(Dtype(h_off + height) / datum_height);

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (*do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
  for (int c = 0; c < seg_datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (*do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(segdata[data_index]));
        } else {
          datum_element = seg_datum.float_data(data_index);
        }
        
        transformed_segmap[top_index] = datum_element;
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob,
                                       NormalizedBBox* crop_bbox,
                                       bool* do_mirror) {
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    return Transform(cv_img, transformed_blob, crop_bbox, do_mirror);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  if ((crop_h > 0) && (crop_w > 0)) {
    CHECK_EQ(crop_h, height);
    CHECK_EQ(crop_w, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data, crop_bbox, do_mirror);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum, const Datum& seg_datum,
                                       Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob,
                                       NormalizedBBox* crop_bbox,
                                       bool* do_mirror) {
  // If datum is encoded, decoded and transform the cv::image.
  // LOG(INFO) << "datum encoded: " << datum.encoded();
  if (datum.encoded()) {
#ifdef USE_OPENCV
    //LOG(INFO) << "use opencv, run here";
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    cv::Mat cv_segmap;
    cv_segmap = DecodeDatumToCVMatNative(seg_datum);
    // Transform the cv::image into blob.
    return Transform(cv_img, cv_segmap, transformed_blob, transformed_segblob, crop_bbox, do_mirror);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int seg_datum_channels = seg_datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int seg_channels = transformed_segblob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_EQ(seg_channels, seg_datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Dtype* transformed_segdata = transformed_segblob->mutable_cpu_data();
  Transform(datum, seg_datum, transformed_data, transformed_segdata, crop_bbox, do_mirror);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, Blob<Dtype>* transformed_blob,
    RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all,
    bool* do_mirror) {
  // Transform datum.
  const Datum& datum = anno_datum.datum();
  NormalizedBBox crop_bbox;
  Transform(datum, transformed_blob, &crop_bbox, do_mirror);

  // Transform annotation.
  const bool do_resize = true;
  TransformAnnotation(anno_datum, do_resize, crop_bbox, *do_mirror,
                      transformed_anno_group_all);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, Blob<Dtype>* transformed_blob,
    RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all,
    RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all_2,
    bool* do_mirror) {
  // Transform datum.
  const Datum& datum = anno_datum.datum();
  NormalizedBBox crop_bbox;
  Transform(datum, transformed_blob, &crop_bbox, do_mirror);

  // Transform annotation.
  const bool do_resize = true;
  TransformAnnotation(anno_datum, do_resize, crop_bbox, *do_mirror,
                      transformed_anno_group_all, transformed_anno_group_all_2);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob,
    RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all,
    bool* do_mirror) { // 2
  // Transform datum.
  const Datum& datum = anno_datum.datum();
  const Datum& seg_datum = anno_datum.seg_datum();
  NormalizedBBox crop_bbox;
  Transform(datum, seg_datum, transformed_blob, transformed_segblob, &crop_bbox, do_mirror);

  // Transform annotation.
  const bool do_resize = true;
  TransformAnnotation(anno_datum, do_resize, crop_bbox, *do_mirror,
                      transformed_anno_group_all);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob,
    RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all, RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all_2,
    bool* do_mirror) { // 2
  // Transform datum.
  const Datum& datum = anno_datum.datum();
  const Datum& seg_datum = anno_datum.seg_datum();
  NormalizedBBox crop_bbox;
  Transform(datum, seg_datum, transformed_blob, transformed_segblob, &crop_bbox, do_mirror);

  // Transform annotation.
  const bool do_resize = true;
  TransformAnnotation(anno_datum, do_resize, crop_bbox, *do_mirror,
                      transformed_anno_group_all, transformed_anno_group_all_2);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, Blob<Dtype>* transformed_blob,
    RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all) {
  bool do_mirror;
  Transform(anno_datum, transformed_blob, transformed_anno_group_all,
            &do_mirror);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob,
    RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all) {
  bool do_mirror;
  Transform(anno_datum, transformed_blob, transformed_segblob, transformed_anno_group_all,
            &do_mirror);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, Blob<Dtype>* transformed_blob,
    vector<AnnotationGroup>* transformed_anno_vec, bool* do_mirror) { // 1
  RepeatedPtrField<AnnotationGroup> transformed_anno_group_all;
  Transform(anno_datum, transformed_blob, &transformed_anno_group_all,
            do_mirror);
  for (int g = 0; g < transformed_anno_group_all.size(); ++g) {
    transformed_anno_vec->push_back(transformed_anno_group_all.Get(g));
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, Blob<Dtype>* transformed_blob,
    vector<AnnotationGroup>* transformed_anno_vec, vector<AnnotationGroup>* transformed_anno_vec_2, bool* do_mirror) { // 1
  RepeatedPtrField<AnnotationGroup> transformed_anno_group_all;
  RepeatedPtrField<AnnotationGroup> transformed_anno_group_all_2;
  Transform(anno_datum, transformed_blob, &transformed_anno_group_all, &transformed_anno_group_all_2,
            do_mirror);
  for (int g = 0; g < transformed_anno_group_all.size(); ++g) {
    transformed_anno_vec->push_back(transformed_anno_group_all.Get(g));
  }
  for (int g = 0; g < transformed_anno_group_all_2.size(); ++g) {
    transformed_anno_vec_2->push_back(transformed_anno_group_all_2.Get(g));
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob,
    vector<AnnotationGroup>* transformed_anno_vec, bool* do_mirror) { // 1
  RepeatedPtrField<AnnotationGroup> transformed_anno_group_all;
  Transform(anno_datum, transformed_blob, transformed_segblob, &transformed_anno_group_all,
            do_mirror);
  for (int g = 0; g < transformed_anno_group_all.size(); ++g) {
    transformed_anno_vec->push_back(transformed_anno_group_all.Get(g));
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob,
    vector<AnnotationGroup>* transformed_anno_vec, vector<AnnotationGroup>* transformed_anno_vec_2, bool* do_mirror) { // 1
  RepeatedPtrField<AnnotationGroup> transformed_anno_group_all;
  RepeatedPtrField<AnnotationGroup> transformed_anno_group_all_2;
  Transform(anno_datum, transformed_blob, transformed_segblob, &transformed_anno_group_all, &transformed_anno_group_all_2,
            do_mirror);
  for (int g = 0; g < transformed_anno_group_all.size(); ++g) {
    transformed_anno_vec->push_back(transformed_anno_group_all.Get(g));
  }
  for (int g = 0; g < transformed_anno_group_all_2.size(); ++g) {
    transformed_anno_vec_2->push_back(transformed_anno_group_all_2.Get(g));
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, Blob<Dtype>* transformed_blob,
    vector<AnnotationGroup>* transformed_anno_vec) {
  bool do_mirror;
  Transform(anno_datum, transformed_blob, transformed_anno_vec, &do_mirror);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, Blob<Dtype>* transformed_blob,
    vector<AnnotationGroup>* transformed_anno_vec,
    vector<AnnotationGroup>* transformed_anno_vec_2) {
  bool do_mirror;
  Transform(anno_datum, transformed_blob, transformed_anno_vec, transformed_anno_vec_2, &do_mirror);
}


template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob,
    vector<AnnotationGroup>* transformed_anno_vec) { // 1
  bool do_mirror;
  Transform(anno_datum, transformed_blob, transformed_segblob, transformed_anno_vec, &do_mirror);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob,
    vector<AnnotationGroup>* transformed_anno_vec, vector<AnnotationGroup>* transformed_anno_vec_2) { // 1_1
  bool do_mirror;
  Transform(anno_datum, transformed_blob, transformed_segblob, transformed_anno_vec, transformed_anno_vec_2, &do_mirror);
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformAnnotation(
    const AnnotatedDatum& anno_datum, const bool do_resize,
    const NormalizedBBox& crop_bbox, const bool do_mirror,
    RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all) { // 4
  const int img_height = anno_datum.datum().height();
  const int img_width = anno_datum.datum().width();
  if (anno_datum.type() == AnnotatedDatum_AnnotationType_BBOX) {
    // Go through each AnnotationGroup.
    for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
      const AnnotationGroup& anno_group = anno_datum.annotation_group(g);
      AnnotationGroup transformed_anno_group;
      // Go through each Annotation.
      bool has_valid_annotation = false;
      for (int a = 0; a < anno_group.annotation_size(); ++a) {
        const Annotation& anno = anno_group.annotation(a);
        const NormalizedBBox& bbox = anno.bbox();
        // Adjust bounding box annotation.
        NormalizedBBox resize_bbox = bbox;
        if (do_resize && param_.has_resize_param()) {
          CHECK_GT(img_height, 0);
          CHECK_GT(img_width, 0);
          UpdateBBoxByResizePolicy(param_.resize_param(), img_width, img_height,
                                   &resize_bbox);
        }
        if (param_.has_emit_constraint() &&
            !MeetEmitConstraint(crop_bbox, resize_bbox,
                                param_.emit_constraint())) {
          continue;
        }
        NormalizedBBox proj_bbox;
        if (ProjectBBox(crop_bbox, resize_bbox, &proj_bbox)) {
          has_valid_annotation = true;
          Annotation* transformed_anno =
              transformed_anno_group.add_annotation();
          transformed_anno->set_instance_id(anno.instance_id());
          NormalizedBBox* transformed_bbox = transformed_anno->mutable_bbox();
          transformed_bbox->CopyFrom(proj_bbox);
          if (do_mirror) {
            Dtype temp = transformed_bbox->xmin();
            transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
            transformed_bbox->set_xmax(1 - temp);
          }
          if (do_resize && param_.has_resize_param()) {
            ExtrapolateBBox(param_.resize_param(), img_height, img_width,
                crop_bbox, transformed_bbox);
          }
        }
      }
      // Save for output.
      if (has_valid_annotation) {
        transformed_anno_group.set_group_label(anno_group.group_label());
        transformed_anno_group_all->Add()->CopyFrom(transformed_anno_group);
      }
    }
  } else {
    LOG(FATAL) << "Unknown annotation type.";
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformAnnotation(
    const AnnotatedDatum& anno_datum, const bool do_resize,
    const NormalizedBBox& crop_bbox, const bool do_mirror,
    RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all,
    RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all_2) { // 4
  const int img_height = anno_datum.datum().height();
  const int img_width = anno_datum.datum().width();
  if (anno_datum.type() == AnnotatedDatum_AnnotationType_BBOX) {
    // Go through each AnnotationGroup.
    for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
      const AnnotationGroup& anno_group = anno_datum.annotation_group(g);
      AnnotationGroup transformed_anno_group;
      // Go through each Annotation.
      bool has_valid_annotation = false;
      for (int a = 0; a < anno_group.annotation_size(); ++a) {
        const Annotation& anno = anno_group.annotation(a);
        const NormalizedBBox& bbox = anno.bbox();
        // Adjust bounding box annotation.
        NormalizedBBox resize_bbox = bbox;
        if (do_resize && param_.has_resize_param()) {
          CHECK_GT(img_height, 0);
          CHECK_GT(img_width, 0);
          UpdateBBoxByResizePolicy(param_.resize_param(), img_width, img_height,
                                   &resize_bbox);
        }
        if (param_.has_emit_constraint() &&
            !MeetEmitConstraint(crop_bbox, resize_bbox,
                                param_.emit_constraint())) {
          continue;
        }
        NormalizedBBox proj_bbox;
        if (ProjectBBox(crop_bbox, resize_bbox, &proj_bbox)) {
          has_valid_annotation = true;
          Annotation* transformed_anno =
              transformed_anno_group.add_annotation();
          transformed_anno->set_instance_id(anno.instance_id());
          NormalizedBBox* transformed_bbox = transformed_anno->mutable_bbox();
          transformed_bbox->CopyFrom(proj_bbox);
          if (do_mirror) {
            Dtype temp = transformed_bbox->xmin();
            transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
            transformed_bbox->set_xmax(1 - temp);
          }
          if (do_resize && param_.has_resize_param()) {
            ExtrapolateBBox(param_.resize_param(), img_height, img_width,
                crop_bbox, transformed_bbox);
          }
        }
      }
      // Save for output.
      if (has_valid_annotation) {
        transformed_anno_group.set_group_label(anno_group.group_label());
        transformed_anno_group_all->Add()->CopyFrom(transformed_anno_group);
      }
    }
    for (int g = 0; g < anno_datum.lane_annotation_group_size(); ++g) {
      const AnnotationGroup& anno_group = anno_datum.lane_annotation_group(g);
      AnnotationGroup transformed_anno_group;
      // Go through each Annotation.
      bool has_valid_annotation = false;
      for (int a = 0; a < anno_group.annotation_size(); ++a) {
        const Annotation& anno = anno_group.annotation(a);
        const NormalizedBBox& bbox = anno.bbox();
        // Adjust bounding box annotation.
        NormalizedBBox resize_bbox = bbox;
        if (do_resize && param_.has_resize_param()) {
          CHECK_GT(img_height, 0);
          CHECK_GT(img_width, 0);
          UpdateBBoxByResizePolicy(param_.resize_param(), img_width, img_height,
                                   &resize_bbox);
        }
        if (param_.has_emit_constraint() &&
            !MeetEmitConstraint(crop_bbox, resize_bbox,
                                param_.emit_constraint())) {
          continue;
        }
        NormalizedBBox proj_bbox;
        if (ProjectBBox(crop_bbox, resize_bbox, &proj_bbox)) {
          has_valid_annotation = true;
          Annotation* transformed_anno =
              transformed_anno_group.add_annotation();
          transformed_anno->set_instance_id(anno.instance_id());
          NormalizedBBox* transformed_bbox = transformed_anno->mutable_bbox();
          transformed_bbox->CopyFrom(proj_bbox);
          if (do_mirror) {
            Dtype temp = transformed_bbox->xmin();
            transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
            transformed_bbox->set_xmax(1 - temp);
          }
          if (do_resize && param_.has_resize_param()) {
            ExtrapolateBBox(param_.resize_param(), img_height, img_width,
                crop_bbox, transformed_bbox);
          }
        }
      }
      // Save for output.
      if (has_valid_annotation) {
        transformed_anno_group.set_group_label(anno_group.group_label());
        transformed_anno_group_all_2->Add()->CopyFrom(transformed_anno_group);
      }
    }
  } else {
    LOG(FATAL) << "Unknown annotation type.";
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::CropImage(const Datum& datum,
                                       const NormalizedBBox& bbox,
                                       Datum* crop_datum) {
  // If datum is encoded, decode and crop the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Crop the image.
    cv::Mat crop_img;
    CropImage(cv_img, bbox, &crop_img);
    // Save the image into datum.
    EncodeCVMatToDatum(crop_img, "png", crop_datum);
    crop_datum->set_label(datum.label());
    return;
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Get the bbox dimension.
  NormalizedBBox clipped_bbox;
  ClipBBox(bbox, &clipped_bbox);
  NormalizedBBox scaled_bbox;
  ScaleBBox(clipped_bbox, datum_height, datum_width, &scaled_bbox);
  const int w_off = static_cast<int>(scaled_bbox.xmin());
  const int h_off = static_cast<int>(scaled_bbox.ymin());
  const int width = static_cast<int>(scaled_bbox.xmax() - scaled_bbox.xmin());
  const int height = static_cast<int>(scaled_bbox.ymax() - scaled_bbox.ymin());

  // Crop the image using bbox.
  crop_datum->set_channels(datum_channels);
  crop_datum->set_height(height);
  crop_datum->set_width(width);
  crop_datum->set_label(datum.label());
  crop_datum->clear_data();
  crop_datum->clear_float_data();
  crop_datum->set_encoded(false);
  const int crop_datum_size = datum_channels * height * width;
  const std::string& datum_buffer = datum.data();
  std::string buffer(crop_datum_size, ' ');
  for (int h = h_off; h < h_off + height; ++h) {
    for (int w = w_off; w < w_off + width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        int crop_datum_index = (c * height + h - h_off) * width + w - w_off;
        buffer[crop_datum_index] = datum_buffer[datum_index];
      }
    }
  }
  crop_datum->set_data(buffer);
}

template<typename Dtype>
void DataTransformer<Dtype>::CropImage(const AnnotatedDatum& anno_datum,
                                       const NormalizedBBox& bbox,
                                       AnnotatedDatum* cropped_anno_datum) {
  // Crop the datum.
  CropImage(anno_datum.datum(), bbox, cropped_anno_datum->mutable_datum());
  cropped_anno_datum->set_type(anno_datum.type());

  // Transform the annotation according to crop_bbox.
  const bool do_resize = false;
  const bool do_mirror = false;
  NormalizedBBox crop_bbox;
  ClipBBox(bbox, &crop_bbox);
  TransformAnnotation(anno_datum, do_resize, crop_bbox, do_mirror,
                      cropped_anno_datum->mutable_annotation_group(),
		      cropped_anno_datum->mutable_lane_annotation_group());
}

template<typename Dtype>
void DataTransformer<Dtype>::CropImage(const AnnotatedDatum& anno_datum,
                                       const NormalizedBBox& bbox,
                                       AnnotatedDatum* cropped_anno_datum,
                                       bool has_seg_datum) {
  // Crop the datum.
  CropImage(anno_datum.datum(), bbox, cropped_anno_datum->mutable_datum());
  CropImage(anno_datum.seg_datum(), bbox, cropped_anno_datum->mutable_seg_datum());
  cropped_anno_datum->set_type(anno_datum.type());

  // Transform the annotation according to crop_bbox.
  const bool do_resize = false;
  const bool do_mirror = false;
  NormalizedBBox crop_bbox;
  ClipBBox(bbox, &crop_bbox);
  TransformAnnotation(anno_datum, do_resize, crop_bbox, do_mirror,
                      cropped_anno_datum->mutable_annotation_group(),
		      cropped_anno_datum->mutable_lane_annotation_group());
}

template<typename Dtype>
void DataTransformer<Dtype>::ExpandImage(const Datum& datum,
                                         const float expand_ratio,
                                         NormalizedBBox* expand_bbox,
                                         Datum* expand_datum) {
  // If datum is encoded, decode and crop the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Expand the image.
    cv::Mat expand_img;
    ExpandImage(cv_img, expand_ratio, expand_bbox, &expand_img);
    // Save the image into datum.
    EncodeCVMatToDatum(expand_img, "png", expand_datum);
    expand_datum->set_label(datum.label());
    return;
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Get the bbox dimension.
  int height = static_cast<int>(datum_height * expand_ratio);
  int width = static_cast<int>(datum_width * expand_ratio);
  float h_off, w_off;
  caffe_rng_uniform(1, 0.f, static_cast<float>(height - datum_height), &h_off);
  caffe_rng_uniform(1, 0.f, static_cast<float>(width - datum_width), &w_off);
  h_off = floor(h_off);
  w_off = floor(w_off);
  expand_bbox->set_xmin(-w_off/datum_width);
  expand_bbox->set_ymin(-h_off/datum_height);
  expand_bbox->set_xmax((width - w_off)/datum_width);
  expand_bbox->set_ymax((height - h_off)/datum_height);

  // Crop the image using bbox.
  expand_datum->set_channels(datum_channels);
  expand_datum->set_height(height);
  expand_datum->set_width(width);
  expand_datum->set_label(datum.label());
  expand_datum->clear_data();
  expand_datum->clear_float_data();
  expand_datum->set_encoded(false);
  const int expand_datum_size = datum_channels * height * width;
  const std::string& datum_buffer = datum.data();
  std::string buffer(expand_datum_size, ' ');
  for (int h = h_off; h < h_off + datum_height; ++h) {
    for (int w = w_off; w < w_off + datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index =
            (c * datum_height + h - h_off) * datum_width + w - w_off;
        int expand_datum_index = (c * height + h) * width + w;
        buffer[expand_datum_index] = datum_buffer[datum_index];
      }
    }
  }
  expand_datum->set_data(buffer);
}

template<typename Dtype>
void DataTransformer<Dtype>::ExpandImage(const AnnotatedDatum& anno_datum,
                                         AnnotatedDatum* expanded_anno_datum) {
  if (!param_.has_expand_param()) {
    expanded_anno_datum->CopyFrom(anno_datum);
    return;
  }
  const ExpansionParameter& expand_param = param_.expand_param();
  const float expand_prob = expand_param.prob();
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob > expand_prob) {
    expanded_anno_datum->CopyFrom(anno_datum);
    return;
  }
  const float max_expand_ratio = expand_param.max_expand_ratio();
  if (fabs(max_expand_ratio - 1.) < 1e-2) {
    expanded_anno_datum->CopyFrom(anno_datum);
    return;
  }
  float expand_ratio;
  caffe_rng_uniform(1, 1.f, max_expand_ratio, &expand_ratio);
  // Expand the datum.
  NormalizedBBox expand_bbox;
  ExpandImage(anno_datum.datum(), expand_ratio, &expand_bbox,
              expanded_anno_datum->mutable_datum());
  expanded_anno_datum->set_type(anno_datum.type());

  // Transform the annotation according to crop_bbox.
  const bool do_resize = false;
  const bool do_mirror = false;
  TransformAnnotation(anno_datum, do_resize, expand_bbox, do_mirror,
                      expanded_anno_datum->mutable_annotation_group(),
		      expanded_anno_datum->mutable_lane_annotation_group());
}

template<typename Dtype>
void DataTransformer<Dtype>::DistortImage(const Datum& datum,
                                          Datum* distort_datum) {
  if (!param_.has_distort_param()) {
    distort_datum->CopyFrom(datum);
    return;
  }
  // If datum is encoded, decode and crop the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Distort the image.
    cv::Mat distort_img = ApplyDistort(cv_img, param_.distort_param());
    // Save the image into datum.
    EncodeCVMatToDatum(distort_img, "png", distort_datum);
    distort_datum->set_label(datum.label());
    return;
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    LOG(ERROR) << "Only support encoded datum now";
  }
}

#ifdef USE_OPENCV
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       const vector<cv::Mat> & seg_mat_vector,
                                       Blob<Dtype>* transformed_blob,
                                       Blob<Dtype>* transformed_segblob) {
  const int mat_num = mat_vector.size();
  const int seg_mat_num = seg_mat_vector.size();
  const int num = transformed_blob->num();
  const int seg_num = transformed_segblob->num();
  const int channels = transformed_blob->channels();
  const int seg_channels = transformed_segblob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_GT(seg_mat_num, 0) << "There is no SEG MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  CHECK_EQ(seg_mat_num, seg_num) <<
    "The size of seg_mat_vector must be equals to transformed_segblob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
  Blob<Dtype> uni_segblob(1, seg_channels, height, width);
  for (int item_id = 0; item_id < seg_mat_num; ++item_id) {
    int offset = transformed_segblob->offset(item_id);
    uni_segblob.set_cpu_data(transformed_segblob->mutable_cpu_data() + offset);
    Transform(seg_mat_vector[item_id], &uni_segblob);
  }
}

 template<typename Dtype>                                                                                                          
 void DataTransformer<Dtype>::Transform(cv::Mat& cv_img,
                                        Blob<Dtype>* transformed_blob,
                                        DirectRegressionData& data) {
   const int crop_size = param_.crop_size();
   const int img_channels = cv_img.channels();
   int img_height = cv_img.rows;
   int img_width = cv_img.cols;
 
   // Check dimensions.
   const int channels = transformed_blob->channels();
   const int height = transformed_blob->height();
   const int width = transformed_blob->width();
   const int num = transformed_blob->num();
 
   CHECK_EQ(channels, img_channels);
   CHECK_GE(num, 1);
   CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
 
   const Dtype scale = param_.scale();
   const bool do_mirror = param_.mirror() && Rand(2);  // yes or no
   const bool has_mean_file = param_.has_mean_file();
   const bool has_mean_values = mean_values_.size() > 0;
   int do_rotate = 0;
   if (param_.rotate())
     do_rotate = Rand(4) * 90; // 0, 90, 180, 270
   CHECK_GT(img_channels, 0);
   Dtype* mean = NULL;
   if (has_mean_file) {
     CHECK_EQ(img_channels, data_mean_.channels());
     CHECK_EQ(img_height, data_mean_.height());
     CHECK_EQ(img_width, data_mean_.width());
     mean = data_mean_.mutable_cpu_data();
   }
   if (has_mean_values) {
     CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
      "Specify either 1 mean_value or as many as channels: " << img_channels;
     if (img_channels > 1 && mean_values_.size() == 1) {
       // Replicate the mean_value for simplicity
       for (int c = 1; c < img_channels; ++c) {
         mean_values_.push_back(mean_values_[0]);
       }
     }
   }
 
   // if image is too small, padding
   if (img_height < height || img_width < height) {
     int left = std::max(0, (width - img_width) / 2);
     int top = std::max(0, (height - img_height) / 2);
     int right = std::max(0, width - img_width - left);
     int bottom = std::max(0, height - img_height - top);
 
     img_height = std::max(img_height, height);
     img_width = std::max(img_width, width);
 
     cv::Mat temp(img_height, img_width, img_channels);
     cv::copyMakeBorder(cv_img, temp, top, bottom, left, right,
         cv::BORDER_CONSTANT, 0);
     cv_img = temp;
 
     // translate bbox
     for (int i = 0; i < data.text_boxes_size(); ++i) {
       int x_1 = data.text_boxes(i).x_1();
       int y_1 = data.text_boxes(i).y_1();
       int x_2 = data.text_boxes(i).x_2();
       int y_2 = data.text_boxes(i).y_2();
       int x_3 = data.text_boxes(i).x_3();
       int y_3 = data.text_boxes(i).y_3();
       int x_4 = data.text_boxes(i).x_4();
       int y_4 = data.text_boxes(i).y_4();
       data.mutable_text_boxes(i)->set_x_1(x_1 + left);
       data.mutable_text_boxes(i)->set_y_1(y_1 + top);
       data.mutable_text_boxes(i)->set_x_2(x_2 + left);
       data.mutable_text_boxes(i)->set_y_2(y_2 + top);
       data.mutable_text_boxes(i)->set_x_2(x_3 + left);
       data.mutable_text_boxes(i)->set_y_3(y_3 + top);
       data.mutable_text_boxes(i)->set_x_4(x_4 + left);
       data.mutable_text_boxes(i)->set_y_4(y_4 + top);
     }
   }
 
   int h_off = 0;
   int w_off = 0;
   cv::Mat cv_cropped_img = cv_img;
   if (crop_size) {
     CHECK_EQ(crop_size, height);
     CHECK_EQ(crop_size, width);
     h_off = Rand(img_height - crop_size + 1);
     w_off = Rand(img_width - crop_size + 1);
     cv::Rect roi(w_off, h_off, crop_size, crop_size);
     cv_cropped_img = cv_img(roi);
     // translate bbox
     for (int i = 0; i < data.text_boxes_size(); ++i) {
       int x_1 = data.text_boxes(i).x_1();
       int y_1 = data.text_boxes(i).y_1();
       int x_2 = data.text_boxes(i).x_2();
       int y_2 = data.text_boxes(i).y_2();
       int x_3 = data.text_boxes(i).x_3();
       int y_3 = data.text_boxes(i).y_3();
       int x_4 = data.text_boxes(i).x_4();
       int y_4 = data.text_boxes(i).y_4();
       data.mutable_text_boxes(i)->set_x_1(x_1 - w_off);
       data.mutable_text_boxes(i)->set_y_1(y_1 - h_off);
       data.mutable_text_boxes(i)->set_x_2(x_2 - w_off);
       data.mutable_text_boxes(i)->set_y_2(y_2 - h_off);
       data.mutable_text_boxes(i)->set_x_3(x_3 - w_off);
       data.mutable_text_boxes(i)->set_y_3(y_3 - h_off);
       data.mutable_text_boxes(i)->set_x_4(x_4 - w_off);
       data.mutable_text_boxes(i)->set_y_4(y_4 - h_off);
     }
   } else {
     CHECK_EQ(img_height, height);
     CHECK_EQ(img_width, width);
   }
 
   CHECK(cv_cropped_img.data);
 
   // mirror
   Dtype* transformed_data = transformed_blob->mutable_cpu_data();
   int top_index;
   for (int h = 0; h < height; ++h) {
     const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
     int img_index = 0;
     for (int w = 0; w < width; ++w) {
       for (int c = 0; c < img_channels; ++c) {
         if (do_mirror) {
           top_index = (c * height + h) * width + (width - 1 - w);
         } else {
           top_index = (c * height + h) * width + w;
         }
         // int top_index = (c * height + h) * width + w;
         Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
         if (has_mean_file) {
           int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
           transformed_data[top_index] =
             (pixel - mean[mean_index]) * scale;
         } else {
           if (has_mean_values) {
             transformed_data[top_index] =
               (pixel - mean_values_[c]) * scale;
           } else {
             transformed_data[top_index] = pixel * scale;
           }
         }
       }
     }
   }
 
   if (do_mirror) {
     for (int i = 0; i < data.text_boxes_size(); ++i) {
       int x_1 = data.text_boxes(i).x_1();
       int y_1 = data.text_boxes(i).y_1();
       int x_2 = data.text_boxes(i).x_2();
       int y_2 = data.text_boxes(i).y_2();
       int x_3 = data.text_boxes(i).x_3();
       int y_3 = data.text_boxes(i).y_3();
       int x_4 = data.text_boxes(i).x_4();
       int y_4 = data.text_boxes(i).y_4();
       data.mutable_text_boxes(i)->set_x_1(width - 1 - x_2);
       data.mutable_text_boxes(i)->set_y_1(y_2);
       data.mutable_text_boxes(i)->set_x_2(width - 1 - x_1);
       data.mutable_text_boxes(i)->set_y_2(y_1);
       data.mutable_text_boxes(i)->set_x_3(width - 1 - x_4);
       data.mutable_text_boxes(i)->set_y_3(y_4);
       data.mutable_text_boxes(i)->set_x_4(width - 1 - x_3);
       data.mutable_text_boxes(i)->set_y_4(y_3);
     }
   }
 
   // rotate (only support square)
   if (crop_size && do_rotate) {
     Dtype* buffer = new Dtype[width * height];
     Dtype* transformed_data = transformed_blob->mutable_cpu_data();
     memcpy(buffer, transformed_data, width * height * sizeof(Dtype));
     for (int h = 0; h < height; ++h) {
       for (int w = 0; w < width; ++w) {
         // backward
         int x, y;
         if (do_rotate == 90) {
           x = height - 1 - h;
           y = w;
         } else if (do_rotate == 180) {
           x = width - 1 - w;
           y = height - 1 - h;
         } else {
           x = h;
           y = width - 1 - w;
         }
         for (int c = 0; c < img_channels; ++c) {
           top_index = (c * height * width) + h * width + w;
           int index = (c * height * width) + y * width + x;
           transformed_data[top_index] = buffer[index];
         }
       }
     }
     delete[] buffer;
  
     for (int i = 0; i < data.text_boxes_size(); ++i) {
       int x1 = data.text_boxes(i).x_1();
       int y1 = data.text_boxes(i).y_1();
       int x2 = data.text_boxes(i).x_2();
       int y2 = data.text_boxes(i).y_2();
       int x3 = data.text_boxes(i).x_3();
       int y3 = data.text_boxes(i).y_3();
       int x4 = data.text_boxes(i).x_4();
       int y4 = data.text_boxes(i).y_4();

       int x11, y11, x22, y22, x33, y33, x44, y44;
       // forward
       if (do_rotate == 90) {
         x11 = y2;
         y11 = width - 1 - x2;
         x22 = y3;
         y22 = width - 1 - x3;
         x33 = y4;
         y33 = width - 1 - x4;
         x44 = y1;
         y44 = width - 1 - x1; 
       } else if (do_rotate == 180) {
         x11 = width - 1 - x3;
         y11 = height - 1 - y3;
         x22 = width - 1 - x4;
         y22 = height - 1 - y4;
         x33 = width - 1 - x1;
         y33 = height - 1 - y1;
         x44 = width - 1 - x2;
         y44 = height - 1 - y2;
       } else {
         x11 = height - 1 - y4;
         y11 = x4;
         x22 = height - 1 - y1;
         y22 = x1;
         x33 = height - 1 - y2;
         y33 = x2;                   
         x44 = height - 1 - y3;
         y44 = x3;
       }
       data.mutable_text_boxes(i)->set_x_1(x11);
       data.mutable_text_boxes(i)->set_y_1(y11);
       data.mutable_text_boxes(i)->set_x_2(x22);
       data.mutable_text_boxes(i)->set_y_2(y22);
       data.mutable_text_boxes(i)->set_x_3(x33);
       data.mutable_text_boxes(i)->set_y_3(y33);
       data.mutable_text_boxes(i)->set_x_4(x44);
       data.mutable_text_boxes(i)->set_y_4(y44);
     }
   }
 }

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {
//#ifdef USE_OPENCV
  const int yolo_height = param_.yolo_height();
  const int yolo_width = param_.yolo_width();
  if (yolo_height>0 && yolo_width>0) {
    yolo_transform(cv_img, transformed_blob);
    return;
  }
//#endif
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U || cv_img.depth() == CV_16U)
    << "Image data type must be unsigned byte, or unsigned short";

  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);
  if ((crop_h > 0) && (crop_w > 0)) {
    CHECK_EQ(crop_h, height);
    CHECK_EQ(crop_w, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_h + 1);
      w_off = Rand(img_width - crop_w + 1);
    } else {
      h_off = (img_height - crop_h) / 2;
      w_off = (img_width - crop_w) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_w, crop_h);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    // const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = cv_cropped_img.depth() == CV_8U ?
          static_cast<Dtype>(cv_cropped_img.ptr<uchar>(h)[img_index++]):
          static_cast<Dtype>(cv_cropped_img.ptr<ushort>(h)[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img, const cv::Mat& cv_segmap,
                                       Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob) {
  NormalizedBBox crop_bbox;
  bool do_mirror;
  Transform(cv_img, cv_segmap, transformed_blob, transformed_segblob, &crop_bbox, &do_mirror);
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformPtr(const cv::Mat& cv_img,
                                          Dtype* transformed_ptr,
                                          int rand1, int rand2, int rand3
    ) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && (rand1%2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  int height = img_height;
  int width = img_width;
  cv::Mat cv_cropped_img = cv_img;
  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);
  if ((crop_h > 0) && (crop_w > 0)) {
    height = crop_h;
    width = crop_w;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = rand2 % (img_height - crop_h + 1);
      w_off = rand3 % (img_width - crop_w + 1);
    } else {
      h_off = (img_height - crop_h) / 2;
      w_off = (img_width - crop_w) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_w, crop_h);
    cv_cropped_img = cv_img(roi);
  }

  CHECK(cv_cropped_img.data);

  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_ptr[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_ptr[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_ptr[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob,
                                       NormalizedBBox* crop_bbox,
                                       bool* do_mirror) {
  // Check dimensions.
  const int img_channels = cv_img.channels();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_GT(img_channels, 0);
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  CHECK_EQ(channels, img_channels);
  CHECK_GE(num, 1);

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  *do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
        "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }

  cv::Mat cv_resized_image, cv_noised_image, cv_cropped_image;
  if (param_.has_resize_param()) {
    cv_resized_image = ApplyResize(cv_img, param_.resize_param());
  } else {
    cv_resized_image = cv_img;
  }
  if (param_.has_noise_param()) {
    cv_noised_image = ApplyNoise(cv_resized_image, param_.noise_param());
  } else {
    cv_noised_image = cv_resized_image;
  }
  int img_height = cv_noised_image.rows;
  int img_width = cv_noised_image.cols;
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);

  int h_off = 0;
  int w_off = 0;
  if ((crop_h > 0) && (crop_w > 0)) {
    CHECK_EQ(crop_h, height);
    CHECK_EQ(crop_w, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_h + 1);
      w_off = Rand(img_width - crop_w + 1);
    } else {
      h_off = (img_height - crop_h) / 2;
      w_off = (img_width - crop_w) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_w, crop_h);
    cv_cropped_image = cv_noised_image(roi);
  } else {
    cv_cropped_image = cv_noised_image;
  }

  // Return the normalized crop bbox.
  crop_bbox->set_xmin(Dtype(w_off) / img_width);
  crop_bbox->set_ymin(Dtype(h_off) / img_height);
  crop_bbox->set_xmax(Dtype(w_off + width) / img_width);
  crop_bbox->set_ymax(Dtype(h_off + height) / img_height);

  if (has_mean_file) {
    CHECK_EQ(cv_cropped_image.rows, data_mean_.height());
    CHECK_EQ(cv_cropped_image.cols, data_mean_.width());
  }
  CHECK(cv_cropped_image.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_image.ptr<uchar>(h);
    int img_index = 0;
    int h_idx = h;
    for (int w = 0; w < width; ++w) {
      int w_idx = w;
      if (*do_mirror) {
        w_idx = (width - 1 - w);
      }
      int h_idx_real = h_idx;
      int w_idx_real = w_idx;
      for (int c = 0; c < img_channels; ++c) {
        top_index = (c * height + h_idx_real) * width + w_idx_real;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h_idx_real) * img_width
              + w_off + w_idx_real;
          transformed_data[top_index] =
              (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
                (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img, const cv::Mat& cv_segmap,
                                       Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob,
                                       NormalizedBBox* crop_bbox,
                                       bool* do_mirror) {
  // Check dimensions.
  const int img_channels = cv_img.channels();
  const int seg_channels = cv_segmap.channels();
  const int channels = transformed_blob->channels();
  const int segmap_channels = transformed_segblob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_GT(img_channels, 0);
  CHECK_GT(seg_channels, 0);
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  CHECK_EQ(channels, img_channels);
  CHECK_EQ(seg_channels, segmap_channels);
  CHECK_GE(num, 1);

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  *do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
        "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }

  cv::Mat cv_resized_image, cv_noised_image, cv_cropped_image;
  cv::Mat cv_resized_segmap, cv_cropped_segmap;
  if (param_.has_resize_param()) {
    cv_resized_image = ApplyResize(cv_img, param_.resize_param());
    cv_resized_segmap = ApplyResize(cv_segmap, param_.resize_param());
  } else {
    cv_resized_image = cv_img;
    cv_resized_segmap = cv_segmap;
  }
  if (param_.has_noise_param()) {
    cv_noised_image = ApplyNoise(cv_resized_image, param_.noise_param());
  } else {
    cv_noised_image = cv_resized_image;
  }
  int img_height = cv_noised_image.rows;
  int img_width = cv_noised_image.cols;
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);

  int h_off = 0;
  int w_off = 0;
  if ((crop_h > 0) && (crop_w > 0)) {
    CHECK_EQ(crop_h, height);
    CHECK_EQ(crop_w, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_h + 1);
      w_off = Rand(img_width - crop_w + 1);
    } else {
      h_off = (img_height - crop_h) / 2;
      w_off = (img_width - crop_w) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_w, crop_h);
    cv_cropped_image = cv_noised_image(roi);
    cv_cropped_segmap = cv_resized_segmap(roi);
  } else {
    cv_cropped_image = cv_noised_image;
    cv_cropped_segmap = cv_resized_segmap;
  }

  // Return the normalized crop bbox.
  crop_bbox->set_xmin(Dtype(w_off) / img_width);
  crop_bbox->set_ymin(Dtype(h_off) / img_height);
  crop_bbox->set_xmax(Dtype(w_off + width) / img_width);
  crop_bbox->set_ymax(Dtype(h_off + height) / img_height);

  if (has_mean_file) {
    CHECK_EQ(cv_cropped_image.rows, data_mean_.height());
    CHECK_EQ(cv_cropped_image.cols, data_mean_.width());
  }
  CHECK(cv_cropped_image.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_image.ptr<uchar>(h);
    int img_index = 0;
    int h_idx = h;
    for (int w = 0; w < width; ++w) {
      int w_idx = w;
      if (*do_mirror) {
        w_idx = (width - 1 - w);
      }
      int h_idx_real = h_idx;
      int w_idx_real = w_idx;
      for (int c = 0; c < img_channels; ++c) {
        top_index = (c * height + h_idx_real) * width + w_idx_real;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h_idx_real) * img_width
              + w_off + w_idx_real;
          transformed_data[top_index] =
              (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
                (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
  Dtype* transformed_segmap = transformed_segblob->mutable_cpu_data();
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_segmap.ptr<uchar>(h);
    int img_index = 0;
    int h_idx = h;
    for (int w = 0; w < width; ++w) {
      int w_idx = w;
      if (*do_mirror) {
        w_idx = (width - 1 - w);
      }
      int h_idx_real = h_idx;
      int w_idx_real = w_idx;
      for (int c = 0; c < segmap_channels; ++c) {
        top_index = (c * height + h_idx_real) * width + w_idx_real;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        transformed_segmap[top_index] = pixel;
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformInv(const Dtype* data, cv::Mat* cv_img,
                                          const int height, const int width,
                                          const int channels) {
  const Dtype scale = param_.scale();
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(channels, data_mean_.channels());
    CHECK_EQ(height, data_mean_.height());
    CHECK_EQ(width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == channels) <<
        "Specify either 1 mean_value or as many as channels: " << channels;
    if (channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  const int img_type = channels == 3 ? CV_8UC3 : CV_8UC1;
  cv::Mat orig_img(height, width, img_type, cv::Scalar(0, 0, 0));
  for (int h = 0; h < height; ++h) {
    uchar* ptr = orig_img.ptr<uchar>(h);
    int img_idx = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < channels; ++c) {
        int idx = (c * height + h) * width + w;
        if (has_mean_file) {
          ptr[img_idx++] = static_cast<uchar>(data[idx] / scale + mean[idx]);
        } else {
          if (has_mean_values) {
            ptr[img_idx++] =
                static_cast<uchar>(data[idx] / scale + mean_values_[c]);
          } else {
            ptr[img_idx++] = static_cast<uchar>(data[idx] / scale);
          }
        }
      }
    }
  }

  if (param_.has_resize_param()) {
    *cv_img = ApplyResize(orig_img, param_.resize_param());
  } else {
    *cv_img = orig_img;
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformInv(const Blob<Dtype>* blob,
                                          vector<cv::Mat>* cv_imgs) {
  const int channels = blob->channels();
  const int height = blob->height();
  const int width = blob->width();
  const int num = blob->num();
  CHECK_GE(num, 1);
  const Dtype* image_data = blob->cpu_data();

  for (int i = 0; i < num; ++i) {
    cv::Mat cv_img;
    TransformInv(image_data, &cv_img, height, width, channels);
    cv_imgs->push_back(cv_img);
    image_data += blob->offset(1);
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::CropImage(const cv::Mat& img,
                                       const NormalizedBBox& bbox,
                                       cv::Mat* crop_img) {
  const int img_height = img.rows;
  const int img_width = img.cols;

  // Get the bbox dimension.
  NormalizedBBox clipped_bbox;
  ClipBBox(bbox, &clipped_bbox);
  NormalizedBBox scaled_bbox;
  ScaleBBox(clipped_bbox, img_height, img_width, &scaled_bbox);

  // Crop the image using bbox.
  int w_off = static_cast<int>(scaled_bbox.xmin());
  int h_off = static_cast<int>(scaled_bbox.ymin());
  int width = static_cast<int>(scaled_bbox.xmax() - scaled_bbox.xmin());
  int height = static_cast<int>(scaled_bbox.ymax() - scaled_bbox.ymin());
  cv::Rect bbox_roi(w_off, h_off, width, height);

  img(bbox_roi).copyTo(*crop_img);
}

template <typename Dtype>
void DataTransformer<Dtype>::ExpandImage(const cv::Mat& img,
                                         const float expand_ratio,
                                         NormalizedBBox* expand_bbox,
                                         cv::Mat* expand_img) {
  const int img_height = img.rows;
  const int img_width = img.cols;
  const int img_channels = img.channels();

  // Get the bbox dimension.
  int height = static_cast<int>(img_height * expand_ratio);
  int width = static_cast<int>(img_width * expand_ratio);
  float h_off, w_off;
  caffe_rng_uniform(1, 0.f, static_cast<float>(height - img_height), &h_off);
  caffe_rng_uniform(1, 0.f, static_cast<float>(width - img_width), &w_off);
  h_off = floor(h_off);
  w_off = floor(w_off);
  expand_bbox->set_xmin(-w_off/img_width);
  expand_bbox->set_ymin(-h_off/img_height);
  expand_bbox->set_xmax((width - w_off)/img_width);
  expand_bbox->set_ymax((height - h_off)/img_height);

  expand_img->create(height, width, img.type());
  expand_img->setTo(cv::Scalar(0));
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(height, data_mean_.height());
    CHECK_EQ(width, data_mean_.width());
    Dtype* mean = data_mean_.mutable_cpu_data();
    for (int h = 0; h < height; ++h) {
      uchar* ptr = expand_img->ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < img_channels; ++c) {
          int blob_index = (c * height + h) * width + w;
          ptr[img_index++] = static_cast<char>(mean[blob_index]);
        }
      }
    }
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
        "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
    vector<cv::Mat> channels(img_channels);
    cv::split(*expand_img, channels);
    CHECK_EQ(channels.size(), mean_values_.size());
    for (int c = 0; c < img_channels; ++c) {
      channels[c] = mean_values_[c];
    }
    cv::merge(channels, *expand_img);
  }

  cv::Rect bbox_roi(w_off, h_off, img_width, img_height);
  img.copyTo((*expand_img)(bbox_roi));
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob,
                                       DrivingData& data) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  int img_height = cv_img.rows;
  int img_width = cv_img.cols;

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);  // yes or no
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;
  int do_rotate = 0;
  if (param_.rotate())
    do_rotate = Rand(4) * 90; // 0, 90, 180, 270

  CHECK_GT(img_channels, 0);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  // if image is too small, padding
  if (img_height < height || img_width < height) {
    int left = std::max(0, (width - img_width) / 2);
    int top = std::max(0, (height - img_height) / 2);
    int right = std::max(0, width - img_width - left);
    int bottom = std::max(0, height - img_height - top);

    img_height = std::max(img_height, height);
    img_width = std::max(img_width, width);

    cv::Mat temp(img_height, img_width, img_channels);
    cv::copyMakeBorder(cv_img, temp, top, bottom, left, right,
        cv::BORDER_CONSTANT, 0);
    cv_img = temp;

    // translate bbox
    for (int i = 0; i < data.car_boxes_size(); ++i) {
      int xmin = data.car_boxes(i).xmin();
      int ymin = data.car_boxes(i).ymin();
      int xmax = data.car_boxes(i).xmax();
      int ymax = data.car_boxes(i).ymax();
      data.mutable_car_boxes(i)->set_xmin(xmin + left);
      data.mutable_car_boxes(i)->set_ymin(ymin + top);
      data.mutable_car_boxes(i)->set_xmax(xmax + left);
      data.mutable_car_boxes(i)->set_ymax(ymax + top);
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    h_off = Rand(img_height - crop_size + 1);
    w_off = Rand(img_width - crop_size + 1);
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
    // translate bbox
    for (int i = 0; i < data.car_boxes_size(); ++i) {
      int xmin = data.car_boxes(i).xmin();
      int ymin = data.car_boxes(i).ymin();
      int xmax = data.car_boxes(i).xmax();
      int ymax = data.car_boxes(i).ymax();
      data.mutable_car_boxes(i)->set_xmin(xmin - w_off);
      data.mutable_car_boxes(i)->set_ymin(ymin - h_off);
      data.mutable_car_boxes(i)->set_xmax(xmax - w_off);
      data.mutable_car_boxes(i)->set_ymax(ymax - h_off);
    }
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  // mirror
  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }

  if (do_mirror) {
    for (int i = 0; i < data.car_boxes_size(); ++i) {
      int xmin = data.car_boxes(i).xmin();
      int xmax = data.car_boxes(i).xmax();
      data.mutable_car_boxes(i)->set_xmin(width - 1 - xmax);
      data.mutable_car_boxes(i)->set_xmax(width - 1 - xmin);
    }
  }

  // rotate (only support square)
  if (crop_size && do_rotate) {
    Dtype* buffer = new Dtype[width * height];
    Dtype* transformed_data = transformed_blob->mutable_cpu_data();
    memcpy(buffer, transformed_data, width * height * sizeof(Dtype));
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        // backward
        int x, y;
        if (do_rotate == 90) {
          x = height - 1 - h;
          y = w;
        } else if (do_rotate == 180) {
          x = width - 1 - w;
          y = height - 1 - h;
        } else {
          x = h;
          y = width - 1 - w;
        }
        for (int c = 0; c < img_channels; ++c) {
          top_index = (c * height * width) + h * width + w;
          int index = (c * height * width) + y * width + x;
          transformed_data[top_index] = buffer[index];
        }
      }
    }
    delete[] buffer;

    for (int i = 0; i < data.car_boxes_size(); ++i) {
      int x1 = data.car_boxes(i).xmin();
      int y1 = data.car_boxes(i).ymin();
      int x2 = data.car_boxes(i).xmax();
      int y2 = data.car_boxes(i).ymax();
      int x11, y11, x22, y22;
      // forward
      if (do_rotate == 90) {
        x11 = y1;
        y11 = width - 1 - x1;
        x22 = y2;
        y22 = width - 1 - x2;
      } else if (do_rotate == 180) {
        x11 = width - 1 - x1;
        y11 = height - 1 - y1;
        x22 = width - 1 - x2;
        y22 = height - 1 - y2;
      } else {
        x11 = height - 1 - y1;
        y11 = x1;
        x22 = height - 1 - y2;
        y22 = x2;
      }
      data.mutable_car_boxes(i)->set_xmin(std::min(x11, x22));
      data.mutable_car_boxes(i)->set_ymin(std::min(y11, y22));
      data.mutable_car_boxes(i)->set_xmax(std::max(x11, x22));
      data.mutable_car_boxes(i)->set_ymax(std::max(y11, y22));
    }
  }
}

#endif  // USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();
  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if ((crop_h > 0) && (crop_w > 0)) {
      transformed_blob->Reshape(input_num, input_channels,
                                crop_h, crop_w);
    } else {
      transformed_blob->Reshape(input_num, input_channels,
                                input_height, input_width);
    }
  }

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);


  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if ((crop_h > 0) && (crop_w > 0)) {
    CHECK_EQ(crop_h, height);
    CHECK_EQ(crop_w, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_h + 1);
      w_off = Rand(input_width - crop_w + 1);
    } else {
      h_off = (input_height - crop_h) / 2;
      w_off = (input_width - crop_w) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
                data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels)
        << "Specify either 1 mean_value or as many as channels: "
        << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
                           input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum,
                                                   bool use_gpu) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img, use_gpu);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }
  const int crop_size = param_.crop_size();
  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  const int datum_channels = datum.channels();
  int datum_height = datum.height();
  int datum_width = datum.width();
  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  if (param_.has_resize_param()) {
    InferNewSize(param_.resize_param(), datum_width, datum_height,
                 &datum_width, &datum_height);
  }
  CHECK_GE(datum_height, crop_h);
  CHECK_GE(datum_width, crop_w);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  // if using GPU transform, don't crop
  if (use_gpu) {
    shape[2] = datum_height;
    shape[3] = datum_width;
  } else {
    shape[2] = (crop_h)? crop_h: datum_height;
    shape[3] = (crop_w)? crop_w: datum_width;
  }
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img,
                                                   bool use_gpu) {
  const int crop_size = param_.crop_size();
  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  const int img_channels = cv_img.channels();
  int img_height = cv_img.rows;
  int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  if (param_.has_resize_param()) {
    InferNewSize(param_.resize_param(), img_width, img_height,
                 &img_width, &img_height);
  }
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  if (use_gpu) {
    shape[2] = img_height;
    shape[3] = img_width;
  } else {
    shape[2] = (crop_h)? crop_h: img_height;
    shape[3] = (crop_w)? crop_w: img_width;
    if (param_.yolo_height()>0 && param_.yolo_width()>0) {
      shape[2] = param_.yolo_height();
      shape[3] = param_.yolo_width();
    }
  }
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}
#endif  // USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      (phase_ == TRAIN &&
          (param_.crop_size() || (param_.crop_h() && param_.crop_w())));
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  // this doesn't actually produce a uniform distribution
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
