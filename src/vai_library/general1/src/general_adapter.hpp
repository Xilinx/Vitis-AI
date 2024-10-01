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
#include "vitis/ai/general.hpp"

namespace vitis {
namespace ai {
template <typename ResultType>
vitis::ai::proto::DpuModelResult process_result(const ResultType& result);

template <typename T>
class GeneralAdapter : public General {
 public:
  explicit GeneralAdapter(typename std::unique_ptr<T>&& target);
  GeneralAdapter(const GeneralAdapter&) = delete;
  virtual ~GeneralAdapter();

 private:
  virtual vitis::ai::proto::DpuModelResult run(const cv::Mat& image) override;
  virtual std::vector<vitis::ai::proto::DpuModelResult> run(
      const std::vector<cv::Mat>& images) override;
  virtual int getInputWidth() const override;
  virtual int getInputHeight() const override;
  virtual size_t get_input_batch() const override;

 private:
  typename std::unique_ptr<T> target_;
};

template <typename T>
GeneralAdapter<T>::GeneralAdapter(typename std::unique_ptr<T>&& target)
    : target_{std::move(target)} {}

template <typename T>
GeneralAdapter<T>::~GeneralAdapter() {}

template <typename T>
vitis::ai::proto::DpuModelResult GeneralAdapter<T>::run(const cv::Mat& image) {
  auto result = target_->run(image);
  return process_result(result);
}

template <typename T>
std::vector<vitis::ai::proto::DpuModelResult> GeneralAdapter<T>::run(
    const std::vector<cv::Mat>& images) {
  auto results = target_->run(images);
  auto ret = std::vector<vitis::ai::proto::DpuModelResult>();
  ret.reserve(results.size());
  for (auto& result : results) {
    ret.emplace_back(process_result(result));
  }
  return ret;
}

template <typename T>
int GeneralAdapter<T>::getInputWidth() const {
  return target_->getInputWidth();
}

template <typename T>
int GeneralAdapter<T>::getInputHeight() const {
  return target_->getInputHeight();
}
template <typename T>
size_t GeneralAdapter<T>::get_input_batch() const {
  return target_->get_input_batch();
}

template <typename T>
std::unique_ptr<General> createInternal(typename std::unique_ptr<T> target) {
  return std::unique_ptr<General>(new GeneralAdapter<T>(std::move(target)));
}

}  // namespace ai
}  // namespace vitis
