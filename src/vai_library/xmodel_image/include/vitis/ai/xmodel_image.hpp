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
#include <vitis/ai/proto/dpu_model_param.pb.h>

#include <memory>

#include "vitis/ai/Mat.hpp"

namespace vitis {
namespace ai {

class XmodelImage {
 public:
  XmodelImage() = default;
  virtual ~XmodelImage() = default;
  XmodelImage(const XmodelImage& other) = delete;
  XmodelImage& operator=(const XmodelImage& rhs) = delete;

 public:
  static std::unique_ptr<XmodelImage> create(const std::string& filename);
  virtual size_t get_batch() const = 0;
  virtual size_t get_width() const = 0;
  virtual size_t get_height() const = 0;
  virtual size_t get_depth() const = 0;
  virtual std::vector<vitis::ai::proto::DpuModelResult> run(
      const std::vector<Mat>& image_buffers) = 0;
};

}  // namespace ai
}  // namespace vitis
