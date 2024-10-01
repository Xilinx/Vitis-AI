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
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/multi_runner.hpp>
#include <vitis/ai/profiling.hpp>
#include <xir/tensor/tensor.hpp>
using namespace std;
using namespace cv;

void LoadImageNames(std::string const& filename,
                    std::vector<std::string>& images) {
  images.clear();

  /*Check if path is a valid directory path. */
  FILE* fp = fopen(filename.c_str(), "r");
  if (NULL == fp) {
    fprintf(stdout, "open file: %s  error\n", filename.c_str());
    exit(1);
  }

  char buffer[256] = {0};
  while (fgets(buffer, 256, fp) != NULL) {
    int n = strlen(buffer);
    buffer[n - 1] = '\0';
    std::string name = buffer;
    images.push_back(name);
  }

  fclose(fp);
}
void copy_input_from_image(const vector<string>& image_names,
                           vart::TensorBuffer* input, vart::RunnerExt* runner) {
  const xir::Tensor* tensor = input->get_tensor();
  auto mean = vitis::ai::getMean(runner)[0];
  auto scale = vitis::ai::getScale(runner)[0] * vart::get_input_scale(tensor);
  auto shape = tensor->get_shape();
  uint64_t datatmp;
  size_t datasize;

  std::tie(datatmp, datasize) =
      input->data(std::vector<int>(input->get_tensor()->get_shape().size(), 0));
  int8_t* data = (int8_t*)datatmp;
  auto size = Size(shape[2], shape[1]);
  size_t idx = 0;
  for (auto&& name : image_names) {
    auto image = cv::imread(name, cv::IMREAD_GRAYSCALE);
    cv::Mat image_resize;
    // resize
    if (size != image.size()) {
      cv::resize(image, image_resize, size, 0, 0, cv::INTER_LINEAR);
    } else {
      image_resize = image;
    }
    // mean scale
    for (auto h = 0; h < size.height; h++) {
      auto img_row = image.ptr(h);
      for (auto w = 0; w < size.width; w++) {
        data[idx++] = std::round((float(img_row[w]) - mean) * scale);
      }
    }
  }
}
int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage : " << argv[0] << "<model_name0>  <image_path_file> "
              << std::endl;
    abort();
  }

  auto runner = vitis::ai::MultiRunner::create(argv[1]);
  if (!runner) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

  LOG(INFO) << "Get input information and input data space";
  auto inputs = runner->get_inputs();
  auto outputs = runner->get_outputs();
  std::vector<std::string> names;
  LoadImageNames(argv[2], names);
  // copy input
  LOG(INFO) << "copy_input_from_image";
  copy_input_from_image(names, inputs[0], runner.get());
  LOG(INFO) << "run";
  runner->execute_async(inputs, outputs);
  LOG(INFO) << "get output ";
  uint64_t datatmp;
  size_t datasize;
  std::tie(datatmp, datasize) = outputs[0]->data(
      std::vector<int>(outputs[0]->get_tensor()->get_shape().size(), 0));
  LOG(INFO) << "datasize " << datasize << " " << ((float*)datatmp)[0];

  return 0;
}
