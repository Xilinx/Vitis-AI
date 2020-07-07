#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <sstream>
#include <iterator>

#include "caffe/data_transformer.hpp"
// #include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/enhanced_image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
EnhancedImageDataLayer<Dtype>::~EnhancedImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void EnhancedImageDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  bool use_gpu_transform = this->transform_param_.use_gpu_transform() &&
      (Caffe::mode() == Caffe::GPU);
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  const int label_num = this->layer_param_.image_data_param().label_num();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  while (getline(infile, line)) {
    if (std::all_of(line.begin(), line.end(),
        [](unsigned char c){ return std::isspace(c); }))
      continue;
    std::istringstream ss_line(line);
    vector<string> tokens{std::istream_iterator<string>{ss_line},
                          std::istream_iterator<string>{}};
    CHECK_GE(tokens.size(), 1 + label_num);
    string filename = tokens[0];
    vector<int> labels;
    std::transform(tokens.begin() + 1, tokens.end(),
        std::back_inserter(labels),
        [](const string& s){ return std::stoi(s); });
    lines_.emplace_back(filename, std::move(labels));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    this->prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    this->ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  this->lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    this->lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(
      root_folder + lines_[this->lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[this->lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img,
                          use_gpu_transform);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // labels
  vector<int> label_shape{batch_size, label_num};
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

/*
template <typename Dtype>
void EnhancedImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}
*/

// This function is called on prefetch thread
template <typename Dtype>
void EnhancedImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  bool use_gpu_transform = this->transform_param_.use_gpu_transform() &&
                           (Caffe::mode() == Caffe::GPU);
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int label_num = image_data_param.label_num();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  if (use_gpu_transform) {
      vector<int> random_vec_shape_;
      random_vec_shape_.push_back(batch_size * 3);
      batch->random_vec_.Reshape(random_vec_shape_);
  }

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(
      root_folder + lines_[this->lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[this->lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img,
                                use_gpu_transform);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    for (auto i = 0; i < label_num; ++i) {
      prefetch_label[item_id * label_num + i] =
          lines_[this->lines_id_].second[i];
    }
    Dtype* tmp_label = prefetch_label + item_id * label_num;
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, this->lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(
        root_folder + lines_[this->lines_id_].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[this->lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    if (this->phase_ == TRAIN ){//|| this->phase_ == TEST
        // Adding Transformation by ziheng
        // Generate Do_Mirror
        
        int do_mirror = caffe_rng_rand() % 2;
        if(this->phase_ == TEST){
          do_mirror = 0;
        }
        // Generate Do_Crop 0-means "do crop", otherwise, "don't crop"
        int do_crop = caffe_rng_rand() % 4;
        if(this->phase_ == TEST){
          do_crop = 0;
        }
        // Generate Do_Rotate 0-means "do rotate", otherwise, "don't rotate"
        //int do_rotate = 0;//caffe_rng_rand() % 2;

        // if Do_Crop then Generate Crop_Parameters
        // Now we only crop body part.
        cv::Rect crop_roi;
        Dtype crop_val;
        int crop_height = new_height;
        int crop_width = new_width;
        int crop_x = 0;
        int crop_y = 0;
        if (!do_crop)
        {
          //caffe_rng_uniform(1, (const Dtype)0.6, (const Dtype)0.9, &crop_val);
          if(this->phase_ == TEST){
              crop_val = 0.7;
          }else{
              caffe_rng_uniform(1, (const Dtype)0.7, (const Dtype)0.85, &crop_val);
          }
          crop_height = new_height * crop_val;
          //crop_width = new_width * crop_val;
          crop_width = new_width;
          //crop_y = caffe_rng_rand() % (new_height - crop_height);
          //crop_x = caffe_rng_rand() % (new_width - crop_width);
          crop_y = new_height - crop_height - 1;
          crop_x = 0;
          crop_roi.x = crop_x;
          crop_roi.y = crop_y;
          crop_roi.width = crop_width;
          crop_roi.height = crop_height;
        }

        //

        cv::Mat mirror_img = cv_img;
        int point_num = label_num / 5;
        if (do_mirror)
        {
          //img
          cv::flip(cv_img, mirror_img, 1);
          //label
          for (int i = 0; i < point_num; i++)
          {
            tmp_label[i*2] = new_width - (tmp_label[i*2] + 1);
            //tmp_label[i*2+1] = new_height - (tmp_label[i*2+1] + 1);
          }
          //flip hand
          for (int i = 0; i < 3; i++)
          {
            Dtype tmp_x, tmp_y;
            tmp_x = tmp_label[i*2];
            tmp_y = tmp_label[i*2+1];
            tmp_label[i*2] = tmp_label[(i+3)*2];
            tmp_label[i*2+1] = tmp_label[(i+3)*2+1];
            tmp_label[(i+3)*2] = tmp_x;
            tmp_label[(i+3)*2+1] = tmp_y;
            //weight
            tmp_x = tmp_label[point_num*2+i*2];
            tmp_y = tmp_label[point_num*2+i*2+1];
            tmp_label[point_num*2+i*2] = tmp_label[point_num*2+(i+3)*2];
            tmp_label[point_num*2+i*2+1] = tmp_label[point_num*2+(i+3)*2+1];
            tmp_label[point_num*2+(i+3)*2] = tmp_x;
            tmp_label[point_num*2+(i+3)*2+1] = tmp_y;
            //label
            tmp_x = tmp_label[point_num*4+i];
            tmp_label[point_num*4+i] = tmp_label[point_num*4+(i+3)];
            tmp_label[point_num*4+(i+3)] = tmp_x;
          }
          //flip foot
          for (int i = 6; i < 9; i++)
          {
            int tmp_x, tmp_y;
            tmp_x = tmp_label[i*2];
            tmp_y = tmp_label[i*2+1];
            tmp_label[i*2] = tmp_label[(i+3)*2];
            tmp_label[i*2+1] = tmp_label[(i+3)*2+1];
            tmp_label[(i+3)*2] = tmp_x;
            tmp_label[(i+3)*2+1] = tmp_y;
            //weight
            tmp_x = tmp_label[point_num*2+i*2];
            tmp_y = tmp_label[point_num*2+i*2+1];
            tmp_label[point_num*2+i*2] = tmp_label[point_num*2+(i+3)*2];
            tmp_label[point_num*2+i*2+1] = tmp_label[point_num*2+(i+3)*2+1];
            tmp_label[point_num*2+(i+3)*2] = tmp_x;
            tmp_label[point_num*2+(i+3)*2+1] = tmp_y;
            //label
            tmp_x = tmp_label[point_num*4+i];
            tmp_label[point_num*4+i] = tmp_label[point_num*4+(i+3)];
            tmp_label[point_num*4+(i+3)] = tmp_x;
          }
        }
        cv::Mat crop_img;
        if (!do_crop)
        {
          //img
          crop_img = mirror_img(crop_roi);
          //label
          for (int i = 0; i < point_num; i++)
          {
            tmp_label[i*2] = tmp_label[i*2] - crop_x;
            tmp_label[i*2+1] = tmp_label[i*2+1] - crop_y;
            if ( tmp_label[i*2] < 0 || tmp_label[i*2+1] < 0
                || tmp_label[i*2] > crop_width || tmp_label[i*2+1] > crop_height)
            {
              tmp_label[i*2] = 0;
              tmp_label[i*2+1] = 0;
              tmp_label[point_num*2+i*2] = 0;
              tmp_label[point_num*2+i*2+1] = 0;
              tmp_label[point_num*4+i] = 2;
            }
          }
          //resize img
          cv::resize(crop_img, cv_img, cv::Size(new_width, new_height));
          //resize label
          Dtype w_ratio = (Dtype) new_width / (Dtype) crop_width;
          Dtype h_ratio = (Dtype) new_height / (Dtype) crop_height;
          //w_ratio == h_ratio,actually
          for (int i = 0; i < point_num; i++)
          {
            tmp_label[i*2] = tmp_label[i*2] * w_ratio;
            tmp_label[i*2+1] = tmp_label[i*2+1] * h_ratio;
          }

        }else
        {
          cv_img = mirror_img;
        }

    
    }
    
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    
    // go to the next iter
    this->lines_id_++;
    if (this->lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      this->lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        this->ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(EnhancedImageDataLayer);
REGISTER_LAYER_CLASS(EnhancedImageData);

}  // namespace caffe
#endif  // USE_OPENCV
