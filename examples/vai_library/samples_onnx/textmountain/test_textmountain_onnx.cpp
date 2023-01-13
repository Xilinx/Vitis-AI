/*
 * Copyright 2019 Xilinx Inc.
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

#include "textmountain_onnx.hpp"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << "  <pic1_url>" << std::endl;
    abort();
  }
  auto model_name = std::string(argv[1]);

  cv::Mat image = cv::imread(argv[2]);
  CHECK(!image.empty()) << "cannot read image from " << argv[2];

  auto model = OnnxTextMountain::create(model_name);

  auto batch = model->get_input_batch();
  std::vector<cv::Mat> images(batch);
  for (auto i = 0u; i < batch; ++i) {
    image.copyTo(images[i]);
  }

  auto r = model->run(images);
  for (int k = 0; k < (int)r.size(); k++) {
    std::cout << "batch-" << k << "\n";
    for (int i = 0; i < (int)r[k].res.size(); i++) {
      std::cout << r[k].res[i].box[0].x << "," << r[k].res[i].box[0].y << ","
                << r[k].res[i].box[1].x << "," << r[k].res[i].box[1].y << ","
                << r[k].res[i].box[2].x << "," << r[k].res[i].box[2].y << ","
                << r[k].res[i].box[3].x << "," << r[k].res[i].box[3].y << ","
                << r[k].res[i].score << "\n";
    }
  }
  return 0;
}

