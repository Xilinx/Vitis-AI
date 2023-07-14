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

cv::Mat process_result(cv::Mat& image, vitis::ai::ClassificationResult& result,
                       bool is_jpeg) {
  auto x = 10;
  auto y = 20;
  auto i = 0;
  for (auto& r : result.scores) {
    i++;
    LOG_IF(INFO, is_jpeg) << "r.index " << r.index << " "  //
                          << result.lookup(r.index) << " "
                          << "r.score " << r.score << " "  //
                          << std::endl;
    auto cls = std::string("") + result.lookup(r.index) + " prob. " +
               std::to_string(r.score);
    cv::putText(image, cls, cv::Point(x, y + 20 * i), cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(20, 20, 180), 1, 1);
  }
  return image;
}
