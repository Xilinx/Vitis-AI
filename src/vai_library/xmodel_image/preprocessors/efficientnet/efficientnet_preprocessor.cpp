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
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "vart/runner_helper.hpp"
#include "vart/tensor_buffer.hpp"
#include "vitis/ai/xmodel_preprocessor.hpp"
#include "xir/graph/graph.hpp"

namespace {
class XmodelPreprocessorEfficientNet : public vitis::ai::XmodelPreprocessor {
 public:
  explicit XmodelPreprocessorEfficientNet(const xir::Graph* graph,
                                 const xir::Tensor* tensor);
  virtual ~XmodelPreprocessorEfficientNet() = default;
  XmodelPreprocessorEfficientNet(const XmodelPreprocessorEfficientNet& other) = delete;
  XmodelPreprocessorEfficientNet& operator=(const XmodelPreprocessorEfficientNet& rhs) = delete;

 public:
  static std::unique_ptr<XmodelPreprocessorEfficientNet> create(
      const xir::Graph* graph, const xir::Tensor* tensor);

 private:
  virtual void process(const std::vector<vitis::ai::Mat>& image_buffers,
                       vart::TensorBuffer* tensor_buffer) override;

 private:
  void process(cv::Mat input, cv::Mat& output);
};

XmodelPreprocessorEfficientNet::XmodelPreprocessorEfficientNet(const xir::Graph* graph,
                                             const xir::Tensor* tensor)
    : vitis::ai::XmodelPreprocessor(graph, tensor) {}

void XmodelPreprocessorEfficientNet::process(
    const std::vector<vitis::ai::Mat>& image_buffers,
    vart::TensorBuffer* tensor_buffer) {
  auto batch_index = 0u;
  for (auto& input : image_buffers) {
    cv::Mat cv_input(input.rows, input.cols, input.type, input.data,
                     input.step);
    // auto data = vart::get_tensor_buffer_data(tensor_buffer, batch_index);
    cv::Mat cv_output((int)get_height(), (int)get_width(),
                      CV_8UC3);  //, data.data,
    // get_width() * get_depth());
    // LOG_IF(INFO, true) << "processing begin" << cv_input.size();
    process(cv_input, cv_output);
    set_input_image((void*)cv_output.data, batch_index, tensor_buffer);
    batch_index = batch_index + 1;
    // LOG_IF(INFO, true) << "processing end" << cv_input.size();
  }
}

void XmodelPreprocessorEfficientNet::process(cv::Mat input, cv::Mat& output) {
  int CROP_PADDING = 32;
  int height = get_height();
  int width = get_width();
  CHECK_EQ(height, width)
      << "width must be equal with height";
  int output_size = height;
  int input_height = input.rows;
  int input_width = input.cols;

  float scale = (float)(output_size) / (output_size + CROP_PADDING);
  int padded_center_crop_size = 
      (int)(scale * ((input_height > input_width) ? input_width : input_height));
  int offset_height = ((input_height - padded_center_crop_size) + 1) / 2;
  int offset_width = ((input_width - padded_center_crop_size) + 1) / 2;

  cv::Mat cropped_img;
  cv::Rect box(offset_width, offset_height, padded_center_crop_size,  padded_center_crop_size);
  input(box).copyTo(cropped_img);

  cv::resize(cropped_img, output,
             cv::Size(output_size, output_size), 0, 0, cv::INTER_CUBIC);
}
}  // namespace

extern "C" std::unique_ptr<vitis::ai::XmodelPreprocessor>
create_xmodel_preprocessor(const xir::Graph* graph, const xir::Tensor* tensor) {
  return std::make_unique<XmodelPreprocessorEfficientNet>(graph, tensor);
}
