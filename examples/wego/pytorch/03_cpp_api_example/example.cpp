/**
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <string>
#include <vector>
#include <queue>
#include <memory>
#include <unordered_map>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "torch/script.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "wego_torch/core/compiler.h"
#include <chrono>

torch::jit::script::Module loadJITModule(const std::string &model_path) {
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(model_path);
  } catch (const c10::Error &e) {
    std::cout << e.msg() << std::endl;
    exit(1);
  }

  return module;
}

void center_crop(int height, int width,
                        const cv::Mat& image, 
                        cv::Mat& cropped_img) {
  int offset_h = (image.rows - height) / 2;
  int offset_w = (image.cols - width) / 2;
  cv::Rect box(offset_w, offset_h, width, height);
  cropped_img = image(box).clone();
  return;
}

void get_img_tensor(const std::string &img_path, 
                    torch::Tensor &img_tensor) {
  auto img = cv::imread(img_path);
  int height=299;
  cv::Mat res;
  auto og_height = img.rows;
  auto og_width = img.cols;
  int dst_width = int((float(height) / og_height) * og_width);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  cv::resize(img, res, cv::Size(dst_width, height),cv::INTER_LINEAR);
  cv::Mat output;
  center_crop(299, 299, res, output);

  output.convertTo(output, CV_32FC3, 1.0f / 255.0f);
  img_tensor = torch::from_blob(output.data, {1, 299, 299, 3}, at::kFloat);
  img_tensor = img_tensor.permute({0, 3, 1, 2});
  auto mean = torch::tensor({0.485, 0.456, 0.406}, torch::kFloat32).unsqueeze(1).unsqueeze(2);
  auto std = torch::tensor({0.229, 0.224, 0.225}, torch::kFloat32).unsqueeze(1).unsqueeze(2);
  img_tensor[0] = img_tensor[0].sub(mean).div(std); 

  return;
}

void get_categories(const std::string &file_path, 
                    std::vector<std::string> &cat_vec){
  cat_vec.clear();
  std::fstream cat_file(file_path);
  if (cat_file.fail()) {
    std::cout << "open file failed" << std::endl;
    exit(1);
  }
  std::string kind;
  while (getline(cat_file, kind)){
    cat_vec.push_back(kind);
  }
  cat_file.close();

  return;
}

void post_process(const torch::Tensor &output, 
                  torch::Tensor &top_prob, 
                  torch::Tensor &top_idx) {

  auto prob = torch::softmax(output.squeeze(), 0);
  std::tuple<torch::Tensor, torch::Tensor> result = torch::sort(prob, 0, true);
  top_prob = std::get<0>(result)[0];
  top_idx = std::get<1>(result)[0];

  return;
}

void run_normal(const torch::jit::IValue &input,
                       torch::jit::script::Module &wego_mod) {
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input);

  std::vector<std::string> cat_vec;
  auto file_path = "imagenet_classes.txt";
  get_categories(file_path, cat_vec);

  auto output = wego_mod.forward(inputs).toTensor();
  torch::Tensor top_prob, top_idx;
  post_process(output, top_prob, top_idx);

  auto top_prob_data = top_prob.data_ptr<float>();
  auto top_idx_data = top_idx.data_ptr<int64_t>();

  std::cout << "===================== Perf Result =====================" << std::endl;
  for (int i = 0; i < 5; ++i) {
    std::cout<< cat_vec[top_idx_data[i]] 
             << " "<< top_prob_data[i] << std::endl;
  }

  return;
}

void run_thread(size_t start, size_t n_threads,
                const std::vector<torch::Tensor> &all_imgs,
                torch::jit::script::Module &wego_mod) {

  for (size_t i = start; i < all_imgs.size(); i += n_threads) {
    auto output = wego_mod.forward({all_imgs[i]}).toTensor();
    torch::Tensor top_prob, top_idx;
    post_process(output, top_prob, top_idx);
  }
  return;
}

void run_perf(const int &batch, const int &n_threads,
              const torch::Tensor &input_img, 
              torch::jit::script::Module &wego_mod) {
  std::vector<torch::Tensor> img_vec;
  for(int i = 0; i < batch; ++i)
    img_vec.push_back(input_img);
  
  torch::TensorList tensor_list{img_vec};
  auto batched_imgs = torch::cat(tensor_list);

  int repeat_batch = 400;
  std::vector<torch::Tensor> all_imgs;
  for(int i = 0; i < repeat_batch; ++i)
    all_imgs.push_back(batched_imgs);

  int n_images = batch * repeat_batch; 
  std::cout << "[Info] begin to run inference with "<< n_images <<" images." << std::endl;
  int r_n = 20;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  for(int r = 0; r < r_n; ++r ) {
    std::vector<std::thread> threads(n_threads);
    int i = 0;
    for(auto it = std::begin(threads); it != std::end(threads); ++it) {
      *it = std::thread(run_thread, i++, n_threads, std::ref(all_imgs), std::ref(wego_mod));
    }

    for(auto&& i : threads) {
      i.join();
    }
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  auto time_duration =  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/1000000.0;
  std::cout << "===================== Perf Result =====================" << std::endl;
  std::cout << "[Total Images] " << r_n * n_images << std::endl;
  std::cout << "[Total Time]   " << std::fixed << std::setprecision(6) << time_duration << std::endl;
  std::cout << "[FPS]          " << std::fixed << std::setprecision(2) << (n_images * r_n) / time_duration  << std::endl;

  return;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
      std::cerr << "Number of arguments is not equal to 4."  << std::endl;
      exit(0);
    }

    auto model_path = argv[2];
    auto mod = loadJITModule(model_path);
    mod.eval();
    wego_torch::AccuracyMode accuracy_mode{};
    auto mode = argv[1];
    if (strcmp(mode, "normal") == 0) {
      accuracy_mode = wego_torch::AccuracyMode::kReserveFixNeuron;
    }
    else if (strcmp(mode, "perf") == 0) {
      accuracy_mode = wego_torch::AccuracyMode::kDefaultRemoveFixNeuron;
    }
    else {
      std::cerr << "Unsupport running mode: " << mode << std::endl;
      exit(0);
    }

    std::vector<int64_t> shape {1, 3, 299, 299};
    std::vector<wego_torch::InputMeta> inputs_meta{
      {wego_torch::DataType::kFloat32, shape}};

    wego_torch::core::CompileOptions options{
      .accuracy_mode = accuracy_mode,
      .inputs_meta = std::move(inputs_meta),
    };
    auto rewrite_mod = wego_torch::core::Compile(mod, options);

    auto img_path = argv[3];
    torch::Tensor img_tensor;
    get_img_tensor(img_path,img_tensor);

    auto target_info = wego_torch::core::GetTargetInfo();
    auto batch = target_info.getBatch();

    int threads = std::stoi(argv[4]);
    if (strcmp(mode, "normal") == 0) {
      run_normal(img_tensor, rewrite_mod);
    }
    else if (strcmp(mode, "perf") == 0) {
      run_perf(batch, threads, img_tensor, rewrite_mod);
    }
    else {
      std::cerr << "Unsupport running mode: " << mode << std::endl;
      exit(0);
    }

    return 0;
}

