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
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace xir {

class SfmController {
 public:
  static std::shared_ptr<SfmController> get_instance();

 protected:
  explicit SfmController() = default;
  virtual ~SfmController() = default;

 public:
  SfmController(const SfmController& rhs) = delete;
  SfmController& operator=(const SfmController& rhs) = delete;

 public:
  virtual void run(const int8_t* input, float scale, unsigned int cls,
                   unsigned int group, float* output) = 0;
  virtual void run_xrt_cu(size_t core_idx, const uint64_t input,
                          const unsigned int cls, const unsigned int group,
                          const int scale, uint64_t output,
                          uint32_t offset) = 0;
  virtual bool supported(float scale, unsigned int cls,
                         unsigned int group) const = 0;
};
}  // namespace xir
