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

#include "ext/AksSysManagerExt.h"
#include "ext/AksParamValues.h"

using namespace AKS;

void usage (const char* exename) {
  std::cout << "[INFO] Usage: " << std::endl;
  std::cout << "[INFO] ---------------------- " << std::endl;
  std::cout << "[INFO] " << exename << " <Image Directory Path>" << std::endl;
  std::cout << std::endl;
}

/// Routine to process a set of images on a particular graph
int executeGraph(
    std::string graphName,
    std::vector<std::string>& images)
{
  /// Access the global system manager
  auto sysMan = AKS::SysManagerExt::getGlobal();

  /// Get graph instance
  AKS::AIGraph *graph = sysMan->getGraph(graphName);

  if(!graph){
    cout<<"[ERROR] Couldn't find requested graph"<<endl;
    AKS::SysManagerExt::deleteGlobal();
    return -1;
  }

  int nImages = images.size();

  auto t1 = std::chrono::steady_clock::now();

  /// Enqueue the images to graph for execution
  std::cout << "[INFO] Starting enqueue ... " << std::endl;
  for(auto& imagePath: images) {
    std::vector<AKS::DataDescriptor> v; v.reserve(3);
    sysMan->enqueueJob (graph, imagePath , std::move(v), nullptr);
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
  return 0;
}

int main(int argc, char **argv)
{  
  int ret = 0;
  if (argc != 2) {
    std::cout << "[ERROR] Usage invalid!" << std::endl;
    usage(argv[0]);
    return -1;
  }

  std::string graphJson = "graph_zoo/graph_googlenet.json";
  std::string graphName = "googlenet";

  /// Get image directory path
  std::string imgDirPath (argv[1]);

  /// Get AKS System Manager instance
  AKS::SysManagerExt * sysMan = AKS::SysManagerExt::getGlobal();

  /// Load all kernels
  sysMan->loadKernels("kernel_zoo");

  /// Load graph
  sysMan->loadGraphs(graphJson);

  /// Get all the images in the given input directory.
  std::vector<std::string> images;
  int i = 0;
  for (boost::filesystem::directory_iterator it {imgDirPath}; 
      it != boost::filesystem::directory_iterator{}; it++) {
    std::string fileExtension = it->path().extension().string();
    if(fileExtension == ".jpg" || fileExtension == ".JPEG" || fileExtension == ".png")
      images.push_back((*it).path().string());
  }

  int nImages = images.size();
  std::cout << "[INFO] Running " << nImages << " Images" << std::endl; 

  /// Start the graph execution on a separate thread 
  /// so that main thread is free to do its own work.
  std::thread t(executeGraph, graphName, std::ref(images));

  std::cout << "[INFO] Waiting for Results ... " << std::endl;
  t.join();

  AKS::SysManagerExt::deleteGlobal();
	return ret;
}

