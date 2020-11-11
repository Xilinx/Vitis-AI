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
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <boost/algorithm/string.hpp>
#include <chrono>

//#include "opencv2/opencv.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"

#include <aks/AksKernelBase.h>
#include <aks/AksDataDescriptor.h>
#include <aks/AksNodeParams.h>
#include <aks/AksLogger.h>


struct AccuracyData
{
  std::map<std::string, int> _groundTruth;
  int _imagesProcessed = 0;
  int _top1 = 0;
  int _top5 = 0;
};

class ClassificationAccuracy : public AKS::KernelBase
{
  public:
    int getNumCUs (void) { return 1; }
    void nodeInit (AKS::NodeParams*);
    int exec_async (
           std::vector<AKS::DataDescriptor*> &in, 
           std::vector<AKS::DataDescriptor*> &out, 
           AKS::NodeParams* nodeParams,
           AKS::DynamicParamValues* dynParams);

    void loadGroundTruth (AKS::NodeParams* nodeParams);
    void updateTopK (AKS::NodeParams* nodeParams,std::string & path, int *topK);
    void report(AKS::NodeParams* nodeParams);
    ~ClassificationAccuracy();

  private:
    AccuracyData* getAccuracyData(AKS::NodeParams*);
    void setAccuracyData(AKS::NodeParams*,AccuracyData*);

    map<AKS::NodeParams*, AccuracyData*> _accuracyDatas;        
    std::chrono::time_point<std::chrono::steady_clock> _t0, _t1;
    bool _is_timer_started = false;
};

extern "C" { /// Add this to make this available for python bindings and 


AKS::KernelBase* getKernel (AKS::NodeParams *params)
{
  return new ClassificationAccuracy();
}

}//extern "C"

ClassificationAccuracy::~ClassificationAccuracy()
{
  for (const auto& accuData : _accuracyDatas) delete accuData.second;
}

void ClassificationAccuracy::setAccuracyData(AKS::NodeParams* nodeParams,AccuracyData* accuData)
{
  _accuracyDatas[nodeParams] = accuData;
}

AccuracyData* ClassificationAccuracy::getAccuracyData(AKS::NodeParams *nodeParams)
{
  auto itr = _accuracyDatas.find(nodeParams);
  if(itr != _accuracyDatas.end()) return itr->second;
  return nullptr;
}

void ClassificationAccuracy::nodeInit (AKS::NodeParams* nodeParams)
{
  //std::cout << "\n[DBG] ClassificationAccuracy Node: Labels: " << labels << std::endl;
 
  AccuracyData *accuData = getAccuracyData(nodeParams);
  
  if (!accuData) {
    /// Create entry for accuracy data
    accuData = new AccuracyData;
    setAccuracyData(nodeParams, accuData);
    /// Get Ground Truth file
    std::string path = nodeParams->_stringParams["ground_truth"];
    fstream gtFile(path);

    if (gtFile.fail()) {
        fprintf(stderr, "Error : Open %s failed.\n", path.c_str());
        exit(1);
    }
    std::string line;
    while (getline(gtFile, line)) {
      /// Split Image Path & Ground Truth
      std::vector<std::string> imgNameAndIndex;
      boost::split(imgNameAndIndex, line, boost::is_any_of(" "));
      accuData->_groundTruth[imgNameAndIndex[0]] = std::stoi(imgNameAndIndex[1]);
      imgNameAndIndex.clear();
    }
    gtFile.close();
    /*
    for (auto &gt: accuData->_groundTruth) {
      std::cout << gt.first << " " << gt.second << std::endl;
    }
    */
  }
}

void ClassificationAccuracy::updateTopK (AKS::NodeParams* nodeParams,std::string & path, int *topK)
{
  AccuracyData *accuData = getAccuracyData(nodeParams);
  if(accuData){
    std::vector<std::string> imgPathSplit; 
    boost::split(imgPathSplit, path, boost::is_any_of("/"));
    int gtIndex = accuData->_groundTruth.find(imgPathSplit.back())->second;
    accuData->_imagesProcessed++;  
    if (topK[0] == gtIndex) {
      accuData->_top1++;
      accuData->_top5++;
      return;
    }
    for (int i = 1; i < 5; ++i) {
      if (topK[i] == gtIndex) {
        accuData->_top5++;
        break;
      }
    }
  }
}

int ClassificationAccuracy::exec_async (
           std::vector<AKS::DataDescriptor*> &in, 
           std::vector<AKS::DataDescriptor*> &out, 
           AKS::NodeParams* nodeParams,
           AKS::DynamicParamValues* dynParams)
{
  //std::cout << "\n[DBG] ClassificationAccuracy Node start ..... \n" << std::endl;

  /// Load Labels file: Moved to node init
  //loadGroundTruth(nodeParams);

  // Start the timer
  if(!_is_timer_started) {
    _t0 = std::chrono::steady_clock::now();
    _is_timer_started = true;
  }

  int batchSize = in[0]->getShape()[0];
  int K = in[0]->getShape()[1];

  /// Get Top K 
  int * topKData = static_cast<int*>(in[0]->data());

  /// Update Accuracy
  //std::cout << "[DBG] ClassificationAccuracy Node: Image: " << dynParams->imagePaths[0]<< std::endl;
  for(int i=0; i<batchSize; ++i)
    updateTopK (nodeParams,dynParams->imagePaths[i], topKData + i*K);
  
  //std::cout << "\n[DBG] ClassificationAccuracy Node end ..... \n" << std::endl;
  return -1; /// No wait
}

void ClassificationAccuracy::report(AKS::NodeParams* nodeParams)
{
  _t1 = std::chrono::steady_clock::now();
  AccuracyData *accuData = getAccuracyData(nodeParams);
  if(accuData){
    _INFO << "Accuracy: Top 1: " << float(accuData->_top1) / float(accuData->_imagesProcessed) << std::endl;
    _INFO << "Accuracy: Top 5: " << float(accuData->_top5) / float(accuData->_imagesProcessed) << std::endl;
  }

  auto dur = std::chrono::duration<float>{_t1-_t0}.count() ;
  LOG(DEBUG) << "Total time (s) around Accuracy Layer : " << dur << std::endl;
  LOG(DEBUG) << "FPS around Accuracy Layer : " << accuData->_imagesProcessed / dur << std::endl;
}

