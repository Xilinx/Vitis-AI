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
#include <stdio.h>
#include <sys/mman.h>
#include <xrt.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vitis/ai/facedetect.hpp>
#include <vitis/ai/profiling.hpp>

static void my_setImageBGR(const cv::Mat& image, void* data1, float scale) {
  signed char* data = (signed char*)data1;
  int c = 0;
  for (auto row = 0; row < image.rows; row++) {
    for (auto col = 0; col < image.cols; col++) {
      auto v = image.at<cv::Vec3b>(row, col);
      auto B = (float)v[0];
      auto G = (float)v[1];
      auto R = (float)v[2];
      auto nB = (B - 128.0f) * scale;
      auto nG = (G - 128.0f) * scale;
      auto nR = (R - 128.0f) * scale;
      nB = std::max(std::min(nB, 127.0f), -128.0f);
      nG = std::max(std::min(nG, 127.0f), -128.0f);
      nR = std::max(std::min(nR, 127.0f), -128.0f);
      data[c++] = (int)(nB);
      data[c++] = (int)(nG);
      data[c++] = (int)(nR);
    }
  }
}

static void fillin_input_images(int argc, char* argv[], int batch,
                                std::vector<cv::Mat>& batch_images,
                                std::vector<std::string>& batch_images_names) {
  std::vector<cv::Mat> arg_input_images;
  std::vector<std::string> arg_input_images_names;
  for (auto i = 2; i < argc; i++) {
    cv::Mat img = cv::imread(argv[i]);
    if (img.empty()) {
      std::cout << "Cannot load " << argv[i] << std::endl;
      continue;
    }
    arg_input_images.push_back(img);
    arg_input_images_names.push_back(argv[i]);
  }
  if (arg_input_images.empty()) {
    std::cerr << "No image load success!" << std::endl;
    abort();
  }
  for (auto batch_idx = 0; batch_idx < batch; batch_idx++) {
    batch_images.push_back(
        arg_input_images[batch_idx % arg_input_images.size()]);
    batch_images_names.push_back(
        arg_input_images_names[batch_idx % arg_input_images.size()]);
  }
}

static void resize_images(std::vector<cv::Mat>& batch_images, int width,
                          int height, std::vector<cv::Mat>& new_images) {
  auto batch = batch_images.size();
  for (auto batch_idx = 0u; batch_idx < batch; batch_idx++) {
    cv::Mat img_resize;
    cv::Mat canvas =
        cv::Mat(cv::Size(width, height), CV_8UC3, cv::Scalar{0, 0, 0});
    auto ratio = std::min(static_cast<float>(width) /
                              static_cast<float>(batch_images[batch_idx].cols),
                          static_cast<float>(height) /
                              static_cast<float>(batch_images[batch_idx].rows));
    ratio = std::min(1.0f, ratio);
    auto new_w = batch_images[batch_idx].cols * ratio;
    auto new_h = batch_images[batch_idx].rows * ratio;
    auto new_size = cv::Size{(int)new_w, (int)new_h};
    cv::resize(batch_images[batch_idx], img_resize, new_size, 0, 0,
               cv::INTER_NEAREST);
    img_resize.copyTo(canvas(cv::Rect{cv::Point{0, 0}, new_size}));
    new_images.push_back(canvas);
  }
}
int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_url> [<image_url> ...]" << std::endl;
    abort();
  }

  __TIC__(FACE_DET_TOTAL)
  auto det = vitis::ai::FaceDetect::create(argv[1], false);
  auto batch = det->get_input_batch();

  // 1, get input tensor buffer's size & input offset
  auto input_tensor_size = det->get_input_buffer_size();
  auto offset = det->get_input_offset();

  // 2, Simulatio client code application xrt bos
  auto h = xclOpen(0, NULL, XCL_INFO);
  auto bo_for_inputs = std::vector<vart::xrt_bo_t>();
  for (auto batch_idx = 0u; batch_idx < batch; ++batch_idx) {
    // use input_tensor_size application bo
    auto bo = xclAllocBO(h, input_tensor_size, 0, 0);
    bo_for_inputs.push_back(vart::xrt_bo_t{h, bo});
  }

  // 3.Simulation hardware do pre-processing and fillin xrt bos
  int width = det->getInputWidth();
  int height = det->getInputHeight();
  std::vector<cv::Mat> batch_images;
  std::vector<std::string> batch_images_names;
  // read input images
  fillin_input_images(argc, argv, batch, batch_images, batch_images_names);
  std::vector<cv::Mat> new_images;
  // resize images (a part of pre-processing)
  resize_images(batch_images, width, height, new_images);
  //  fillin input xrt bos with pre-processed data
  for (auto batch_idx = 0u; batch_idx < bo_for_inputs.size(); batch_idx++) {
    // get xrt bo's addr
    auto data =
        (int*)xclMapBO(bo_for_inputs[batch_idx].xrt_handle,
                       bo_for_inputs[batch_idx].xrt_bo_handle, true);  //
    CHECK(data != nullptr) << "can not map bo";
    int fix_point = det->get_input_fix_point();
    float quant_scale = std::exp2f(1.0f * (float)fix_point);
    float norm_scale = 1.0;
    // Start writing data from the offset position
    my_setImageBGR(new_images[batch_idx], (void*)((char*)data + offset),
                   quant_scale * norm_scale);
    xclUnmapBO(bo_for_inputs[batch_idx].xrt_handle,
               bo_for_inputs[batch_idx].xrt_bo_handle, (void*)(data));
  }

  // 4, run with vector<xrt_bo_t> (include dpu run & post-processing) and get
  // results
  auto results = det->run(bo_for_inputs);
  __TOC__(FACE_DET_TOTAL)

  // print results
  for (auto batch_idx = 0u; batch_idx < results.size(); batch_idx++) {
    std::cout << "batch_index: " << batch_idx << "   "                   //
              << "image_name: " << batch_images_names[batch_idx] << " "  //
              << std::endl;
    auto canvas = new_images[batch_idx];
    for (const auto& r : results[batch_idx].rects) {
      std::cout << " " << r.score << " "  //
                << r.x << " "             //
                << r.y << " "             //
                << r.width << " "         //
                << r.height << " "        //
                << std::endl;
      cv::rectangle(canvas,
                    cv::Rect{cv::Point(r.x * canvas.cols, r.y * canvas.rows),
                             cv::Size{(int)(r.width * canvas.cols),
                                      (int)(r.height * canvas.rows)}},
                    0xff);
    }
    std::cout << std::endl;
    cv::imwrite("out_" + batch_images_names[batch_idx], canvas);
  }

  // release xrt bo
  for (auto bo : bo_for_inputs) {
    xclFreeBO(bo.xrt_handle, bo.xrt_bo_handle);
  }
  xclClose(h);
  return 0;
}
