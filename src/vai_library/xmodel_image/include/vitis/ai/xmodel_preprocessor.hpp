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
#include <memory>
#include <vector>

#include "vart/tensor_buffer.hpp"
#include "vitis/ai/Mat.hpp"
#include "xir/graph/graph.hpp"

namespace vitis {
namespace ai {

class XmodelPreprocessor {
 public:
  explicit XmodelPreprocessor(const xir::Graph* graph,
                              const xir::Tensor* tensor);
  virtual ~XmodelPreprocessor() = default;
  XmodelPreprocessor(const XmodelPreprocessor& other) = delete;
  XmodelPreprocessor& operator=(const XmodelPreprocessor& rhs) = delete;

 public:
  static std::unique_ptr<XmodelPreprocessor> create(const xir::Graph* graph,
                                                    const xir::Tensor* tensor);

  virtual void process(const std::vector<Mat>& image_buffers,
                       vart::TensorBuffer* tensor_buffer) = 0;

 protected:
  size_t get_batch() const;
  size_t get_width() const;
  size_t get_height() const;
  size_t get_depth() const;
  void set_input_image(const void* data, size_t batch_index,
                       vart::TensorBuffer* tensor_buffer);

 protected:
  size_t batch_;
  size_t height_;
  size_t width_;
  size_t depth_;
  std::vector<float> mean_;
  std::vector<float> scale_;
  bool do_mean_scale_;
  bool is_rgb_input_;
};

}  // namespace ai
}  // namespace vitis
