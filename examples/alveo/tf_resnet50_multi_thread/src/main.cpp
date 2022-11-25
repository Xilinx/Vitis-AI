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
// Vitis API execute_async example
#include <algorithm>
#include <chrono>
#include <iostream>
#include "cxxopts.hpp"
#include "tests.hpp"

int main(int argc, char** argv) {
  cxxopts::Options options("AppTests", "Vitis application-level tests");
  options.add_options()
    ("r,runnermeta", "Path to DpuRunner meta.json", cxxopts::value<std::string>())
    ("d,imgdir", "Image Directory", cxxopts::value<std::string>()->default_value("tests/dpuv3int8/models/commonImgLabelDir/imageDir"))
    ("n,numImgs", "Number of Images", cxxopts::value<int>()->default_value("4"))
    ("g,golden", "Prints top-1 and top-5 results, given golden labels file", cxxopts::value<bool>()->default_value("false"))
    ("v,verbose", "cout each image details", cxxopts::value<bool>()->default_value("false"))
    ("c,numCUs", "Number of CUs To Utilize", cxxopts::value<int>()->default_value("1"))
    ("e,numThreads", "Number of Threads for multi thread test", cxxopts::value<int>()->default_value("-1"))
    ;
  auto result = options.parse(argc, argv);
  if (!result.count("runnermeta"))
    throw std::runtime_error("Usage: DPUCADF8H.exe -r path_to_xmodel.xmodel");

  const auto runnerMeta = result["runnermeta"].as<std::string>();
  const std::string imgDir = result["imgdir"].as<std::string>();
  const int numImgs = result["numImgs"].as<int>();
  const bool golden = result["golden"].as<bool>();
  const bool verbose = result["verbose"].as<bool>();
  const int numCUs = result["numCUs"].as<int>();
  const int numThreadsMt = result["numThreads"].as<int>();

  const int batchSz = 4;
  assert((numImgs%batchSz)==0);
  const unsigned numQueries = numImgs/batchSz;
  unsigned numThreads = 0;

  if(numQueries < 20)
  {
    numThreads = numQueries;
  }
  else
  {
    for(int i=20; i>0; i--)
    {
      if(numQueries%i==0)
      {
        numThreads = i;
        break;
      }
    }
  }
  if(numThreadsMt!=-1)
    numThreads = numThreadsMt;
  if(numQueries%numThreads!=0)
  {
    std::cout<<"Number of queries provided is (Number of Images/Batch Size): "<<numImgs<<"/"<<batchSz<<" = "<<numQueries<<std::endl;
    std::cout<<"Number of threads provided is: "<<numThreads<<std::endl;
    throw std::runtime_error("ERROR: Non-integer division of queries and threads is not supported");
  }
  std::cout << std::endl << "Test Classify Multi Thread..." << std::endl;
  auto t3 = std::chrono::high_resolution_clock::now();
  TestClassifyMultiThread testClassifyMultiThread(runnerMeta, numQueries, numThreads, numCUs, imgDir, golden, verbose);
  auto t4 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed0 = t4-t3;

  std::cout<<"(Load Images + Resize + Flatten) for "<<numQueries*4<<" Images, Time (ms): "<<elapsed0.count()*1000<<std::endl;
  std::cout<<"********************************"<<std::endl;

  auto t1 = std::chrono::high_resolution_clock::now();
  testClassifyMultiThread.run();
  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = t2-t1;

  std::cout<<"-----------------------------------------------"<<std::endl;
  std::cout<<"Summary of Multi Thread Execution: "<<std::endl;
  std::cout<<"Num CUs used: "<<numCUs<<std::endl;
  std::cout<<"Postprocess includes output reorganization, softmax"<<std::endl;  
  std::cout << "(Preprocess + Kernel + Postprocess) Execution time for "<<numQueries*4<<" images (ms): "<<elapsed.count()*1000 << std::endl;
  std::cout << "Average (Preprocess + Kernel + Postprocess) Execution time for 4 or 1 img (ms): "<<(elapsed.count()*1000)/(numQueries) << std::endl;

  std::cout << "Average (Preprocess + Kernel + Postprocess) imgs per second: " << (numQueries*4)/elapsed.count() << std::endl;
  std::cout<<"-----------------------------------------------"<<std::endl;



  return 0;
}
