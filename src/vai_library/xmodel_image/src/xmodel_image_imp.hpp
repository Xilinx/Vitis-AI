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

#include "./graph_holder.hpp"
#include "vart/runner_ext.hpp"
#include "vitis/ai/xmodel_image.hpp"
#include "vitis/ai/xmodel_postprocessor.hpp"
#include "vitis/ai/xmodel_preprocessor.hpp"

namespace vitis {
namespace ai {

class XmodelImageImp : public XmodelImage {
 public:
  XmodelImageImp(const std::string& filename);
  virtual ~XmodelImageImp();
  XmodelImageImp(const XmodelImageImp& other) = delete;
  XmodelImageImp& operator=(const XmodelImageImp& rhs) = delete;

 private:
  virtual size_t get_batch() const override;
  virtual size_t get_width() const override;
  virtual size_t get_height() const override;
  virtual size_t get_depth() const override;
  virtual std::vector<vitis::ai::proto::DpuModelResult> run(
      const std::vector<Mat>& image_buffers) override;

 private:
  XmodelPostprocessorInputs build_post_processor_inputs(
      const xir::OpDef& opdef);

 private:
  std::shared_ptr<GraphHolder> graph_;
  std::unique_ptr<xir::Attrs> attrs_;
  std::unique_ptr<vart::RunnerExt> runner_;
  size_t batch_;
  size_t height_;
  size_t width_;
  size_t depth_;
  std::vector<vart::TensorBuffer*> input_tensor_buffers_;
  std::vector<vart::TensorBuffer*> output_tensor_buffers_;
  XmodelPostprocessorInputs post_processor_inputs_;
  std::unique_ptr<XmodelPreprocessor> preprocessor_;
  std::unique_ptr<XmodelPostprocessorBase> postprocessor_;
};

}  // namespace ai
}  // namespace vitis
