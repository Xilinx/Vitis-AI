/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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
#pragma once
#include <glog/logging.h>

#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace vitis {
namespace ai {
class ImageList {
 public:
  explicit ImageList(const std::string& filename, bool lazy_load_image);
  ImageList(const ImageList&) = delete;
  ImageList& operator=(const ImageList& other) = delete;
  virtual ~ImageList();

 public:
  const cv::Mat operator[](long i) const;

  std::string to_string();
  void resize_images();

  bool empty() const { return list_.empty(); }
  size_t size() const { return list_.size(); }
  std::string getName(size_t i) const;

 public:
  struct Image {
    std::string name;
    cv::Mat mat;
  };

 private:
  std::vector<Image> list_;
  bool lazy_load_image_;
};

static std::vector<ImageList::Image> get_list(const std::string& filename,
                                              bool lazy_load_image) {
  std::ifstream fs(filename.c_str());
  std::string line;
  auto ret = std::vector<ImageList::Image>{};
  while (getline(fs, line)) {
    // LOG(INFO) << "line = [" << line << "]";
    if (!lazy_load_image) {
      auto image = cv::imread(line);
      if (image.empty()) {
        LOG(WARNING) << "cannot read image: " << line;
      } else {
        ret.emplace_back(ImageList::Image{line, image});
      }
    } else {
      ret.emplace_back(ImageList::Image{line, cv::Mat{}});
    }
  }
  fs.close();
  return ret;
}

inline ImageList::ImageList(const std::string& filename, bool lazy_load_image)
    : list_{get_list(filename, lazy_load_image)},
      lazy_load_image_{lazy_load_image} {}

inline ImageList::~ImageList() {}

inline std::string ImageList::to_string() {
  std::ostringstream str;
  int c = 0;
  for (const auto& image : list_) {
    if (c++ != 0) {
      str << ",";
    }
    str << image.name << image.mat.size();
  }
  // str << "\n";
  return str.str();
}

inline const cv::Mat ImageList::operator[](long i) const {
  const auto& name = list_[i % list_.size()].name;
  const auto& mat = list_[i % list_.size()].mat;
  if (!lazy_load_image_) {
    return mat;
  }
  return cv::imread(name);
}

inline std::string ImageList::getName(size_t i) const {
  const auto& name = list_[i % list_.size()].name;
  return name;
}
cv::Mat resize_and_crop_image(cv::Mat image) {
  auto original_height = 900, original_width = 1600;
  auto final_height = 256, final_width = 704;
  auto resize_scale = 0.48;  //

  int resized_width = original_width * resize_scale;
  int resized_height = original_height * resize_scale;
  cv::Size resize_dims(resized_width, resized_height);
  auto crop_h = std::max(0, resized_height - final_height);
  auto crop_w = int(std::max(0, (resized_width - final_width) / 2));
  cv::Rect box(crop_w, crop_h, final_width, final_height);

  if (image.size() == resize_dims) {
    LOG(INFO) << "no resize";
    return image(box).clone();
  }
  if (image.size() == cv::Size(final_width, final_height)) {
    LOG(INFO) << "no crop";
    return image;
  }
  cv::Mat img;
  cv::resize(image, img, resize_dims, 0, 0, cv::INTER_LINEAR);
  return img(box).clone();
}
inline void ImageList::resize_images() {
  CHECK(!lazy_load_image_) << "only for eager load mode";
  for (auto& image : list_) {
    auto old = cv::Mat(std::move(image.mat));
    image.mat = resize_and_crop_image(old);
  }
}
class BinList {
 public:
  struct Bin {
    std::string name;
    std::vector<char> bin;
  };
  explicit BinList(const std::string& filename) {
    std::ifstream fs(filename.c_str());
    std::string line;
    while (getline(fs, line)) {
      auto infile = std::ifstream(line, std::ios_base::binary);
      list_.emplace_back(
          Bin{line, std::vector<char>(std::istreambuf_iterator<char>(infile),
                                      std::istreambuf_iterator<char>())});
    }
    fs.close();
  }
  BinList(const BinList&) = delete;
  BinList& operator=(const BinList& other) = delete;
  virtual ~BinList() {}

 public:
  const std::vector<char> operator[](long unsigned int i) const {
    return list_[i % list_.size()].bin;
  }

  bool empty() const { return list_.empty(); }
  size_t size() const { return list_.size(); }
  std::string getName(size_t i) { return list_[i % list_.size()].name; }

 private:
  std::vector<Bin> list_;
};
}  // namespace ai
}  // namespace vitis
