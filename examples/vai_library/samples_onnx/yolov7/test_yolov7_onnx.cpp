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
#include "yolov7_onnx.hpp"

static void process_result(cv::Mat& image, const Yolov7OnnxResult& result) {
  for (auto& box : result.bboxes) {
    int label = box.label;
    float xmin = box.x * image.cols + 1;
    float ymin = box.y * image.rows + 1;
    float xmax = xmin + box.width * image.cols;
    float ymax = ymin + box.height * image.rows;
    if (xmin < 0.) xmin = 1.;
    if (ymin < 0.) ymin = 1.;
    if (xmax > image.cols) xmax = image.cols;
    if (ymax > image.rows) ymax = image.rows;
    float confidence = box.score;

    std::cout << "RESULT: " << label << "\t" << xmin << "\t" << ymin << "\t" << xmax
         << "\t" << ymax << "\t" << confidence << "\n";
    cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), 1,
              1, 0);
  }
  return;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << "<model_name> <image_file_url>" << std::endl;
    return -1;
  }

  cv::Mat image = cv::imread(argv[2]);
  if (image.empty()) {
    std::cerr << "cannot load " << argv[2] << std::endl;
    return -1;
  }

  auto model = Yolov7Onnx::create(argv[1], 0.5);
  if (!model) {  // supress coverity complain
    std::cerr << "failed to create model\n";
    return -1;
  }

  auto batch = model->get_input_batch();
  std::vector<cv::Mat> images(batch);
  for (auto i = 0u; i < batch; ++i) {
    image.copyTo(images[i]);
  }

  __TIC__(ONNX_RUN)
  auto results = model->run(images);
  __TOC__(ONNX_RUN)

  for (auto i = 0u; i < results.size(); i++) {
    std::cout << "batch " << i << std::endl;
    process_result(images[i], results[i]);
    auto out_file = std::to_string(i) + "_result.jpg";
    cv::imwrite(out_file, images[i]);
  }

  return 0;
}
