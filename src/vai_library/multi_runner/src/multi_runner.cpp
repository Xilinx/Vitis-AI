/*
 * Copyright 2021 xilinx Inc.
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
#include "vitis/ai/multi_runner.hpp"

#include <string>

#include "./multi_runner_imp.hpp"

namespace vitis {
namespace ai {

std::unique_ptr<vart::RunnerExt> MultiRunner::create(std::string model_name) {
  return std::unique_ptr<vart::RunnerExt>(new MultiRunnerImp(model_name));
}

std::vector<float> getMean(vart::RunnerExt* runner) {
  auto r = dynamic_cast<MultiRunnerImp*>(runner);
  if (r) return r->getMean();
  LOG(FATAL) << "The input runner is not vitis::ai::MultiRunner";
  return {0, 0, 0};
}
std::vector<float> getScale(vart::RunnerExt* runner) {
  auto r = dynamic_cast<MultiRunnerImp*>(runner);
  if (r) return r->getScale();
  LOG(FATAL) << "The input runner is not vitis::ai::MultiRunner";
  return {0, 0, 0};
}
}  // namespace ai
}  // namespace vitis
