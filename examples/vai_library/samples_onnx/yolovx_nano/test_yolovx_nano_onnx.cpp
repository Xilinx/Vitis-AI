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
#include "yolovx_nano_onnx.hpp"

static void process_result(cv::Mat& image, const YolovxnanoOnnxResult& result) {
  for (auto& res : result.bboxes) {
    int label = res.label;
    auto& box = res.box;

    std::cout << "RESULT: " << label << "\t" << std::fixed << std::setprecision(6)
         << box[0] << "\t" << box[1] << "\t" << box[2] << "\t" << box[3] << "\t"
         << std::setprecision(6) << res.score << "\n";
    cv::rectangle(image, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]),
              cv::Scalar(0, 255, 0), 1, 1, 0);
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

  auto model = YolovxnanoOnnx::create(argv[1], 0.3);
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

