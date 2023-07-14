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
#include <boost/filesystem.hpp>

using namespace std;

#include <aks/AksSysManagerExt.h>
#include <aks/AksNodeParams.h>
#include "AksGraphMeta.h"

using namespace AKS;

void usage(const char *exename) {
  std::cout << "[INFO] Usage: " << std::endl;
  std::cout << "[INFO] ---------------------- " << std::endl;
  std::cout << "[INFO] " << exename << " <Image Directory Path>" << std::endl;
  std::cout << std::endl;
}

int main(int argc, char **argv)
{
  int ret = 0;
  if (argc != 3) {
    std::cout << "[ERROR] Usage invalid!" << std::endl;
    usage(argv[0]);
    return -1;
  }

  /// Get device
  std::string device = argv[1];
  if (tf_resnet_v1_50.find(device) == tf_resnet_v1_50.end()) {
    std::cerr << "[ERROR] Couldn't find graph for requested device!";
    std::cerr << std::endl;
    return -1;
  } else {
    std::cout << "[INFO] Found ResNet50 graph for " << device;
    std::cout << std::endl;
  }

  /// Get AKS System Manager instance
  AKS::SysManagerExt * sysMan = AKS::SysManagerExt::getGlobal();

  /// Load all kernels
  sysMan->loadKernels("kernel_zoo");

  /// Load graph
  sysMan->loadGraphs(tf_resnet_v1_50[device].second);

  /// Get graph instance
  AKS::AIGraph *graph = sysMan->getGraph(tf_resnet_v1_50[device].first);
  if(!graph) {
    cout<<"[ERROR] Couldn't find requested graph"<<endl;
    AKS::SysManagerExt::deleteGlobal();
    return -1;
  }

  /// Get image directory path
  std::string imgDirPath (argv[2]);

  std::vector<std::string> images;
  /// Read Dataset
  for (boost::filesystem::directory_iterator it {imgDirPath};
      it != boost::filesystem::directory_iterator{}; it++) {
    std::string fileExtension = it->path().extension().string();
    if(fileExtension == ".jpg" || fileExtension == ".JPEG" || fileExtension == ".png")
    { images.push_back((*it).path().string()); }
  }

  const int batch = dpu_batch[device];
  int left_out = images.size() % batch;
  if (left_out) { // Make a batch complete
    for (int b = 0; b < (batch-left_out); ++b) {
      std::string s = images.back();
      images.push_back(s);
    }
  }

  int nImages = images.size();
  std::cout << "[INFO] Running " << nImages << " Images" << std::endl;

  sysMan->resetTimer();
  auto t1 = std::chrono::steady_clock::now();

  /// User input
  std::cout << "[INFO] Starting enqueue ... " << std::endl;
  for (int i = 0; i < images.size(); i+=batch) {
    std::vector<std::string> batch_imgs;
    for (int b = 0; b < batch; ++b) {
      batch_imgs.push_back(images[i+b]);
    }
    sysMan->enqueueJob (graph, batch_imgs);
    batch_imgs.clear();
  }

  /// Wait for results
  sysMan->waitForAllResults();

  /// Report - applicable for accuracy kernel
  sysMan->report(graph);

  auto t2 = std::chrono::steady_clock::now();

  auto time_taken = std::chrono::duration<double>(t2-t1).count();
  auto throughput = static_cast<double>(nImages)/time_taken;

  /// Print Stats
  std::cout << "[INFO] Total Images : " << nImages << std::endl;
  std::cout << "[INFO] Total Time (s): " << time_taken << std::endl;
  std::cout << "[INFO] Overall FPS : " << throughput << std::endl;

  sysMan->printPerfStats();

  AKS::SysManagerExt::deleteGlobal();
  return ret;
}

