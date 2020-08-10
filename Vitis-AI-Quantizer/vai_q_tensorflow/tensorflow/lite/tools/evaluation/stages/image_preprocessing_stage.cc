/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h"

#include <cmath>
#include <fstream>
#include <streambuf>
#include <string>

#include "absl/base/casts.h"
#include "tensorflow/core/lib/jpeg/jpeg_handle.h"
#include "tensorflow/core/lib/jpeg/jpeg_mem.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tflite {
namespace evaluation {
namespace {

// We assume 3-channel RGB images.
const int kNumChannels = 3;

// Returns the offset for the element in the raw image array based on the image
// height/weight & coordinates of a pixel (h, w, c).
inline int ImageArrayOffset(int height, int width, int h, int w, int c) {
  return (h * width + w) * kNumChannels + c;
}

inline void Crop(int input_height, int input_width, int start_h, int start_w,
                 int crop_height, int crop_width, const uint8_t* input_data,
                 std::vector<float>* output_data) {
  const int stop_h = start_h + crop_height;
  const int stop_w = start_w + crop_width;

  for (int in_h = start_h; in_h < stop_h; ++in_h) {
    for (int in_w = start_w; in_w < stop_w; ++in_w) {
      for (int c = 0; c < kNumChannels; ++c) {
        output_data->push_back(static_cast<float>(input_data[ImageArrayOffset(
            input_height, input_width, in_h, in_w, c)]));
      }
    }
  }
}

// Performs billinear interpolation for 3-channel RGB image.
// See: https://en.wikipedia.org/wiki/Bilinear_interpolation
template <typename T>
inline void ResizeBilinear(int input_height, int input_width,
                           const std::vector<float>& input_data,
                           int output_height, int output_width, int total_size,
                           std::vector<T>& output_data, float input_mean,
                           float scale) {
  tflite::ResizeBilinearParams resize_params;
  resize_params.align_corners = false;
  tflite::RuntimeShape input_shape(
      {1, input_height, input_width, kNumChannels});
  tflite::RuntimeShape output_size_dims({1, 1, 1, 2});
  std::vector<int32_t> output_size_data = {output_height, output_width};
  tflite::RuntimeShape output_shape(
      {1, output_height, output_width, kNumChannels});
  std::vector<float> temp_float_data;
  temp_float_data.reserve(total_size);
  for (int i = 0; i < total_size; ++i) {
    temp_float_data.push_back(0);
  }
  tflite::reference_ops::ResizeBilinear(
      resize_params, input_shape, input_data.data(), output_size_dims,
      output_size_data.data(), output_shape, temp_float_data.data());

  // Normalization.
  output_data.clear();
  output_data.reserve(total_size);
  for (int i = 0; i < total_size; ++i) {
    output_data.push_back(
        static_cast<T>((temp_float_data[i] - input_mean) * scale));
  }
}
}  // namespace

TfLiteStatus ImagePreprocessingStage::Init() {
  auto& params = config_.specification().image_preprocessing_params();
  if (params.image_height() <= 0 || params.image_width() <= 0) {
    LOG(ERROR) << "Invalid image dimensions to ImagePreprocessingStage";
    return kTfLiteError;
  }
  cropping_fraction_ = params.cropping_fraction();
  if (cropping_fraction_ > 1.0 || cropping_fraction_ < 0) {
    LOG(ERROR) << "Invalid cropping fraction";
    return kTfLiteError;
  } else if (cropping_fraction_ == 0) {
    cropping_fraction_ = 1.0;
  }
  input_mean_value_ = 0;
  scale_ = 1.0;
  output_type_ = static_cast<TfLiteType>(params.output_type());
  total_size_ = params.image_height() * params.image_width() * kNumChannels;
  if (output_type_ == kTfLiteUInt8) {
  } else if (output_type_ == kTfLiteInt8) {
    input_mean_value_ = 128.0;
  } else if (output_type_ == kTfLiteFloat32) {
    input_mean_value_ = 127.5;
    scale_ = 1.0 / 127.5;
  } else {
    LOG(ERROR) << "Wrong TfLiteType for ImagePreprocessingStage";
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus ImagePreprocessingStage::Run() {
  if (!image_path_) {
    LOG(ERROR) << "Image path not set";
    return kTfLiteError;
  }
  auto& params = config_.specification().image_preprocessing_params();

  int64_t start_us = profiling::time::NowMicros();

  // Read image.
  std::ifstream t(*image_path_);
  std::string image_str((std::istreambuf_iterator<char>(t)),
                        std::istreambuf_iterator<char>());
  const int fsize = image_str.size();
  auto temp = absl::bit_cast<const uint8_t*>(image_str.data());
  std::unique_ptr<uint8_t[]> original_image;
  int original_width, original_height, original_channels;
  tensorflow::jpeg::UncompressFlags flags;
  // JDCT_ISLOW performs slower but more accurate pre-processing.
  // This isn't always obvious in unit tests, but makes a difference during
  // accuracy testing with ILSVRC dataset.
  flags.dct_method = JDCT_ISLOW;
  // We necessarily require a 3-channel image as the output.
  flags.components = kNumChannels;
  original_image.reset(Uncompress(temp, fsize, flags, &original_width,
                                  &original_height, &original_channels,
                                  nullptr));

  // Central Crop.
  const int left = static_cast<int>(
      std::round(original_width * (1 - cropping_fraction_) / 2));
  const int top = static_cast<int>(
      std::round(original_height * (1 - cropping_fraction_) / 2));
  const int crop_width =
      static_cast<int>(std::round(original_width * cropping_fraction_));
  const int crop_height =
      static_cast<int>(std::round(original_height * cropping_fraction_));
  std::vector<float> cropped_image;
  cropped_image.reserve(crop_height * crop_width * kNumChannels);
  Crop(original_height, original_width, top, left, crop_height, crop_width,
       original_image.get(), &cropped_image);

  // Billinear-Resize & apply mean & scale.
  if (output_type_ == kTfLiteUInt8) {
    ResizeBilinear(crop_height, crop_width, cropped_image,
                   params.image_height(), params.image_width(), total_size_,
                   uint8_preprocessed_image_, input_mean_value_, scale_);
  } else if (output_type_ == kTfLiteInt8) {
    ResizeBilinear(crop_height, crop_width, cropped_image,
                   params.image_height(), params.image_width(), total_size_,
                   int8_preprocessed_image_, input_mean_value_, scale_);
  } else if (output_type_ == kTfLiteFloat32) {
    ResizeBilinear(crop_height, crop_width, cropped_image,
                   params.image_height(), params.image_width(), total_size_,
                   float_preprocessed_image_, input_mean_value_, scale_);
  }

  latency_stats_.UpdateStat(profiling::time::NowMicros() - start_us);
  return kTfLiteOk;
}

void* ImagePreprocessingStage::GetPreprocessedImageData() {
  if (latency_stats_.count() == 0) return nullptr;

  if (output_type_ == kTfLiteUInt8) {
    return uint8_preprocessed_image_.data();
  } else if (output_type_ == kTfLiteInt8) {
    return int8_preprocessed_image_.data();
  } else if (output_type_ == kTfLiteFloat32) {
    return float_preprocessed_image_.data();
  }
  return nullptr;
}

EvaluationStageMetrics ImagePreprocessingStage::LatestMetrics() {
  EvaluationStageMetrics metrics;
  auto* latency_metrics =
      metrics.mutable_process_metrics()->mutable_total_latency();
  latency_metrics->set_last_us(latency_stats_.newest());
  latency_metrics->set_max_us(latency_stats_.max());
  latency_metrics->set_min_us(latency_stats_.min());
  latency_metrics->set_sum_us(latency_stats_.sum());
  latency_metrics->set_avg_us(latency_stats_.avg());
  metrics.set_num_runs(static_cast<int>(latency_stats_.count()));
  return metrics;
}

}  // namespace evaluation
}  // namespace tflite
