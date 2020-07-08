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
#include <thread>
#include <chrono>
#include <boost/filesystem.hpp>

using namespace std;

#include "ext/AksSysManagerExt.h"
#include "ext/AksNodeParams.h"

using namespace AKS;

void usage (const char* exename) {
  std::cout << "[INFO] Usage: " << std::endl;
  std::cout << "[INFO] ---------------------- " << std::endl;
  std::cout << "[INFO] " << exename << " <image-dir-for-googlenet> <img-dir-for-tinyyolov3>" << std::endl;
  std::cout << std::endl;
}

void enqueueGoogleNetJobs(AKS::AIGraph* graph, std::vector<std::string>& images)
{
  /// Get System Manager
  AKS::SysManagerExt * sysMan = AKS::SysManagerExt::getGlobal();

  int nImages = images.size();
  std::cout << "[INFO] Running GoogleNet with " << nImages << " Images" << std::endl; 

  auto t1 = std::chrono::steady_clock::now();

  // User input
  std::cout << "[INFO] Starting enqueue ... " << std::endl;
  for(auto& imagePath: images) {
    std::vector<AKS::DataDescriptor> v; v.reserve(3);
    sysMan->enqueueJob (graph, imagePath, std::move(v), nullptr);
  }
  // Wait for results
  sysMan->waitForAllResults(graph);
 
  auto t2 = std::chrono::steady_clock::now();
  /// Report - applicable for accuracy kernel 
  std::cout << "\n[INFO] Report for GoogleNet: " << std::endl;
  sysMan->report(graph);

  auto time_taken = std::chrono::duration<double>(t2-t1).count();
  auto throughput = static_cast<double>(nImages)/time_taken;
  std::cout << "[INFO] Total Images : " << nImages << std::endl;
  std::cout << "[INFO] Total Time (s): " << time_taken << std::endl;
  std::cout << "[INFO] Overall FPS : " << throughput << std::endl;

}

void enqueueTinyYolov3Jobs(AKS::AIGraph* graph, std::vector<std::string>& images)
{
  /// Get System Manager
  AKS::SysManagerExt * sysMan = AKS::SysManagerExt::getGlobal();

  int nImages = images.size();
  std::cout << "[INFO] Running TinyYolo-v3 with " << nImages << " Images" << std::endl; 

  auto t1 = std::chrono::steady_clock::now();

  // User input
  std::cout << "[INFO] Starting enqueue ... " << std::endl;
  for(auto& imagePath: images) {
    std::vector<AKS::DataDescriptor> v; v.reserve(3);
    sysMan->enqueueJob (graph, imagePath, std::move(v), nullptr);
  }
  // Wait for results
  sysMan->waitForAllResults(graph);

  auto t2 = std::chrono::steady_clock::now();
  /// Report - applicable for accuracy kernel 
  std::cout << "\n[INFO] Report for TinyYolo-v3: " << std::endl;
  sysMan->report(graph);


  auto time_taken = std::chrono::duration<double>(t2-t1).count();
  auto throughput = static_cast<double>(nImages)/time_taken;
  std::cout << "[INFO] Total Images : " << nImages << std::endl;
  std::cout << "[INFO] Total Time (s): " << time_taken << std::endl;
  std::cout << "[INFO] Overall FPS : " << throughput << std::endl;

}

int main(int argc, char **argv)
{  
  int ret = 0;
  if (argc != 3) {
    std::cout << "[ERROR] Usage invalid!" << std::endl;
    usage(argv[0]);
    return -1;
  }

  /// Graphs
  std::vector<std::string> graphJsons = {
    "graph_zoo/graph_googlenet_no_runner.json", 
    "graph_zoo/graph_tinyyolov3_no_runner.json"
  };
  std::vector<std::string> graphNames = {
    "googlenet_no_runner", 
    "tinyyolov3_no_runner"
  };

  /// Get image directory path
  std::vector<std::string> imageDirs = {
    std::string(argv[1]),
    std::string(argv[2]),
  };

  /// Get System Manager
  AKS::SysManagerExt * sysMan = AKS::SysManagerExt::getGlobal();

  /// Load Kernels
  sysMan->loadKernels("kernel_zoo");

  /// Load Graphs
  sysMan->loadGraphs(graphJsons[0]);
  sysMan->loadGraphs(graphJsons[1]);

  AKS::AIGraph *graph_g = sysMan->getGraph(graphNames[0]);
  if(!graph_g){
    cout<<"[ERROR] Couldn't find requested graph: " << graphNames[0] <<endl;
    AKS::SysManagerExt::deleteGlobal();
    return -1;
  }

  AKS::AIGraph *graph_y = sysMan->getGraph(graphNames[1]);
  if(!graph_y){
    cout<<"[ERROR] Couldn't find requested graph: " << graphNames[1] <<endl;
    AKS::SysManagerExt::deleteGlobal();
    return -1;
  }

  /// Load Dataset
  std::vector<std::string> images_g;
  for (boost::filesystem::directory_iterator it {imageDirs[0]}; 
      it != boost::filesystem::directory_iterator{}; it++) {
    std::string fileExtension = it->path().extension().string();
    if(fileExtension == ".jpg" || fileExtension == ".JPEG" || fileExtension == ".png")
      images_g.push_back((*it).path().string());
  }

  std::vector<std::string> images_y;
  for (boost::filesystem::directory_iterator it {imageDirs[1]}; 
      it != boost::filesystem::directory_iterator{}; it++) {
    std::string fileExtension = it->path().extension().string();
    if(fileExtension == ".jpg" || fileExtension == ".JPEG" || fileExtension == ".png")
      images_y.push_back((*it).path().string());
  }

  sysMan->resetTimer();
  /// Create threads to enqueue jobs
  std::thread t_googlenet (enqueueGoogleNetJobs, graph_g, std::ref(images_g));
  std::thread t_tinyyolov3 (enqueueTinyYolov3Jobs, graph_y, std::ref(images_y));

  t_googlenet.join();
  t_tinyyolov3.join();

  /// Print stats
  sysMan->printPerfStats();
  /// Destroy Sys Manager instance
  AKS::SysManagerExt::deleteGlobal();
}

