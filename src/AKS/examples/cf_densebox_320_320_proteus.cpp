/*
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
 */

#include <iostream>
#include <chrono>
#include <queue>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <aks/AksSysManagerExt.h>
#include <aks/AksNodeParams.h>
#include <aks/AksTensorBuffer.h>
#include <aks/AksBatchTensorBuffer.h>

using namespace AKS;

void usage (const char* exename) {
  std::cout << "[INFO] Usage: " << std::endl;
  std::cout << "[INFO] ---------------------- " << std::endl;
  std::cout << "[INFO] " << exename << " <Image Directory Path>" << std::endl;
  std::cout << std::endl;
}

// Leave this empty to stop writing o/p to disk
std::string _output_dir ("facedetect_outputs");

// Queue and Mutex to push/pop futures
using aks_future_t = std::future<std::vector<std::unique_ptr<vart::TensorBuffer>>>;
using aks_batch_img_t = std::vector<std::string>;
std::queue<std::pair<aks_future_t, aks_batch_img_t>> future_q;
std::mutex g_mtx;
std::atomic<bool> thread_continue {true};

void write_output (
    std::vector<std::unique_ptr<vart::TensorBuffer>> results,
    aks_batch_img_t img_paths)
{
  auto nboxes = results.front()->get_tensor()->get_element_num() / 6;
  if(!_output_dir.empty()) {
    if(!boost::filesystem::exists(_output_dir))
      boost::filesystem::create_directory(_output_dir);

    int boxcnt = 0;
    for(int b = 0; b < img_paths.size(); ++b) {
      std::vector<std::string> tokens;
      boost::split(tokens, img_paths[b], boost::is_any_of("/,."));
      auto& imgFile = tokens[tokens.size()-2];

      // Append output_dir and .txt to get output file
      std::string output_file = _output_dir + "/" + imgFile + ".txt";
      ofstream f(output_file);
      if(!f) {
        std::cerr << "[WARNING] : Couldn't open " << output_file << std::endl;
        std::cerr << "[WARNING] : Check if path is correct" << std::endl;
        return;
      }

      auto* boxptr = reinterpret_cast<float*>(results.front()->data().first);
      for (int box = 0; box < nboxes; ++box) {
        if (boxptr[box*6] == b) {
          float score    = boxptr[(box*6)+5];
          float x        = boxptr[(box*6)+1];
          float y        = boxptr[(box*6)+2];
          float w        = boxptr[(box*6)+3];
          float h        = boxptr[(box*6)+4];

          f << score << " ";
          f << x << " " << y << " ";
          f << w << " " << h << '\n';
        }
      }
      f.close();
    }
  }
}

// Wait for jobs and get results
void get_results (void)
{
  while (true) {
    g_mtx.lock();
    if (future_q.empty()) {
      g_mtx.unlock();
      if (!thread_continue) break;
    } else {
      auto element = std::move(future_q.front());
      future_q.pop();
      g_mtx.unlock();
      // Get data from future
      auto results = element.first.get();
      auto img_paths = element.second;
      // Write to disk
      write_output (std::move(results), std::move(img_paths));
    }
  }
}

// Face-Detection Inference using
// DPUCADF8H on Alveo-u200/u250 FPGAs
int main(int argc, char **argv)
{
  int ret = 0;
  if (argc != 2) {
    std::cout << "[ERROR] Usage invalid!" << std::endl;
    usage(argv[0]);
    return -1;
  }

  // Get image directory path
  std::string imgDirPath (argv[1]);

  // Get AKS System Manager instance
  AKS::SysManagerExt * sysMan = AKS::SysManagerExt::getGlobal();

  // Load all kernels
  sysMan->loadKernels("kernel_zoo");

  // Load graph
  sysMan->loadGraphs("graph_zoo/graph_facedetect_u200_u250_proteus.json");

  // Get graph instance
  AKS::AIGraph *graph = sysMan->getGraph("facedetect");

  if(!graph){
    cout<<"[ERROR] Couldn't find requested graph"<<endl;
    AKS::SysManagerExt::deleteGlobal();
    return -1;
  }

  std::vector<std::string> images;
  int i = 0;
  // Load Dataset
  for (boost::filesystem::directory_iterator it {imgDirPath};
      it != boost::filesystem::directory_iterator{}; it++) {
    std::string fileExtension = it->path().extension().string();
    if(fileExtension == ".jpg" || fileExtension == ".JPEG" || fileExtension == ".png")
      images.push_back((*it).path().string());
  }

  constexpr int bt = 4; // DPU batch size
  int left_out = images.size() % bt;
  if (left_out) { // Make a batch complete
    for (int b = 0; b < (bt-left_out); ++b) {
      std::string s = images.back();
      images.push_back(s);
    }
  }

  int nImages = images.size();
  std::cout << "[INFO] Running " << nImages << " Images" << std::endl;

  // Start a thread to wait for results
  std::thread wait_thread (get_results);

  sysMan->resetTimer();
  auto t1 = std::chrono::steady_clock::now();

  // User input
  std::cout << "[INFO] Starting enqueue ... " << std::endl;
  for (int i = 0; i < images.size(); i+=bt) {
    // Create input tensors
    std::vector<std::unique_ptr<xir::Tensor>> tensors;

    // Create batch of images
    std::vector<cv::Mat> batchimgs;
    std::vector<std::string> image_paths;

    for (int b = 0; b < bt; ++b) {
      cv::Mat img = cv::imread(images[i+b]);
      std::vector<int> shape = { 1, img.rows, img.cols, img.channels() };
      auto tensorOut =
        xir::Tensor::create("imread_output", shape,
          xir::create_data_type<unsigned char>());

      tensors.push_back(std::move(tensorOut));
      batchimgs.push_back(std::move(img));
      image_paths.push_back(images[i+b]);
    }

    // Create input buffer & fill image data
    std::unique_ptr<AKS::AksBatchTensorBuffer> tb =
      std::make_unique<AKS::AksBatchTensorBuffer>(std::move(tensors));

    for (int b = 0; b < bt; ++b) {
      auto* imgptr = batchimgs[b].data;
      auto size    = tb->get_tensors()[b]->get_data_size();
      auto* bufptr = reinterpret_cast<uint8_t*>(tb.get()->data({b}).first);
      // Copy image data
      memcpy(bufptr, imgptr, size);
    }

    // Fill input vector with input buffer
    std::vector<std::unique_ptr<vart::TensorBuffer>> inputs;
    inputs.push_back(std::move(tb));

    // Enqueue input & push future to queue
    {
      std::unique_lock<std::mutex> lock(g_mtx);
      future_q.push({
          std::move(sysMan->enqueueJob (graph, "", std::move(inputs), nullptr)),
          std::move(image_paths)
          });
    }
    batchimgs.clear();
  }

  // Wait for results
  std::cout << "[INFO] Waiting for results ... " << std::endl;

  thread_continue = false;
  wait_thread.join();

  std::cout << "[INFO] Waiting for results ... Done!" << std::endl;

  auto t2 = std::chrono::steady_clock::now();

  auto time_taken = std::chrono::duration<double>(t2-t1).count();
  auto throughput = static_cast<double>(nImages)/time_taken;

  // Print Stats
  std::cout << "[INFO] Total Images : " << nImages << std::endl;
  std::cout << "[INFO] Total Time (s): " << time_taken << std::endl;
  std::cout << "[INFO] Overall FPS : " << throughput << std::endl;

  sysMan->printPerfStats();

  // Clean-up
  AKS::SysManagerExt::deleteGlobal();
  return ret;
}

