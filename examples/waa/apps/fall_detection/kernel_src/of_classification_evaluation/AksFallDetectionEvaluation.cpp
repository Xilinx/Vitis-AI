/*
 * Copyright 2021 Xilinx Inc.
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
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <chrono>

#include <aks/AksKernelBase.h>
#include <aks/AksNodeParams.h>
#include <aks/AksLogger.h>
#include <aks/AksTensorBuffer.h>


struct EvaluationData
{
  std::map<std::string, int> _groundTruth;
  int _imagesProcessed = 0;
  int accuracy = 0;
  int truePos = 0;
  int trueNeg = 0;
  int falsePos = 0;
  int falseNeg = 0;
};

class FallDetectionEvaluation : public AKS::KernelBase
{
  public:
    int getNumCUs (void) { return 1; }
    void nodeInit (AKS::NodeParams*);
    int exec_async (
        std::vector<vart::TensorBuffer*> &in,
        std::vector<vart::TensorBuffer*> &out,
        AKS::NodeParams* nodeParams,
        AKS::DynamicParamValues* dynParams);

    void loadGroundTruth(AKS::NodeParams* nodeParams);
    void evaluate(
        AKS::NodeParams* nodeParams, std::vector<std::string> &paths,
        float* probabilities, int8_t* outData);
    void report(AKS::NodeParams* nodeParams);
    ~FallDetectionEvaluation();

  private:
    EvaluationData* getEvaluationData(AKS::NodeParams*);
    void setEvaluationData(AKS::NodeParams*,EvaluationData*);

    map<AKS::NodeParams*, EvaluationData*> _EvaluationDatas;
    std::chrono::time_point<std::chrono::steady_clock> _t0, _t1;
    bool _is_timer_started = false;
    std::string output_folder;
    float threshold;
    std::string path;
    int default_label;
};

extern "C" { /// Add this to make this available for python bindings and


AKS::KernelBase* getKernel (AKS::NodeParams *params)
{
  // SET_LOGGING_LEVEL();
  return new FallDetectionEvaluation();
}

}//extern "C"

FallDetectionEvaluation::~FallDetectionEvaluation()
{
  for (const auto& accuData : _EvaluationDatas) delete accuData.second;
}

void FallDetectionEvaluation::setEvaluationData(AKS::NodeParams* nodeParams,EvaluationData* accuData)
{
  _EvaluationDatas[nodeParams] = accuData;
}

EvaluationData* FallDetectionEvaluation::getEvaluationData(AKS::NodeParams *nodeParams)
{
  auto itr = _EvaluationDatas.find(nodeParams);
  if(itr != _EvaluationDatas.end()) return itr->second;
  return nullptr;
}

void FallDetectionEvaluation::nodeInit (AKS::NodeParams* params)
{
  //std::cout << "\n[DBG] FallDetectionEvaluation Node: Labels: " << labels << std::endl;

  threshold = params->getValue<float>("threshold");
  path = params->getValue<std::string>("ground_truth");
  output_folder = params->_stringParams.find("visualize") == \
    params->_stringParams.end() ? "" : params->getValue<std::string>("visualize");
  default_label = params->getValue<int>("default_label");

  EvaluationData *accuData = getEvaluationData(params);
  
  if (!accuData) {
    /// Create entry for accuracy data
    accuData = new EvaluationData;
    setEvaluationData(params, accuData);
    /// Get Ground Truth file
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

void FallDetectionEvaluation::evaluate(
    AKS::NodeParams* nodeParams, std::vector<std::string> &paths,
    float* probs, int8_t* outData)
{
  EvaluationData *accuData = getEvaluationData(nodeParams);
  if(accuData){
    for (int i=0; i<paths.size(); i++) {
      auto path = paths.at(i);
      float prob = probs[i];
      std::vector<std::string> imgPathSplit;
      boost::split(imgPathSplit, path, boost::is_any_of("/"));
      int gtLabel;
      if (accuData->_groundTruth.find(imgPathSplit.back()) == accuData->_groundTruth.end()) {
        gtLabel = default_label;
      } else {
        gtLabel = accuData->_groundTruth.find(imgPathSplit.back())->second;
      }
      accuData->_imagesProcessed++;

      // prob > threshold ? no_fall : fall
      int label = prob > threshold ? 1 : 0;
      outData[i] = label;
      if (label == gtLabel) {
        accuData->accuracy++;
        if (label == 0)
          accuData->truePos++;
        else
          accuData->trueNeg++;
      }
      else {
        if (label == 0)
          accuData->falsePos++;
        else
          accuData->falseNeg++;
      }
      if (!output_folder.empty()) {
        boost::filesystem::path p(path);
        std::string filename = p.filename().string();
        std::string folder = p.parent_path().filename().string();
        boost::filesystem::create_directories(output_folder);
        std::string output_path = output_folder + "/" + folder + ".txt";
        ofstream fout;
        fout.open(output_path, ios::app);
        fout << filename << " " << prob << std::endl;
        fout.close();
      }
    }
  }
}

float calcSigmoid(float input) {
  return 1 / (1 + exp(-input));
}

int FallDetectionEvaluation::exec_async (
           std::vector<vart::TensorBuffer*> &in,
           std::vector<vart::TensorBuffer*> &out,
           AKS::NodeParams* nodeParams,
           AKS::DynamicParamValues* dynParams)
{
  // std::cout << "\n[DBG] FallDetectionEvaluation Node start ..... \n" << std::endl;
  int batchSize = in[0]->get_tensor()->get_shape().at(0);
  
  // Start the timer
  if(!_is_timer_started) {
    _t0 = std::chrono::steady_clock::now();
    _is_timer_started = true;
  }
  
  // Expects the data to be the output from Sigmoid (only to be used for binary classification)
  float* inData = reinterpret_cast<float*>(in[0]->data().first);
  
  auto *probsBuffer = new AKS::AksTensorBuffer(
                xir::Tensor::create(
                  "probabilities", {batchSize},
                  xir::create_data_type<float>()
                ));
  float* probsData = reinterpret_cast<float*>(probsBuffer->data().first);
  for (int i = 0; i < batchSize; i++) {
    probsData[i] = calcSigmoid(inData[i]);
  }
 
  auto *labelsBuffer = new AKS::AksTensorBuffer(
                xir::Tensor::create(
                  "labels", {batchSize},
                  xir::create_data_type<char>()
                ));
  int8_t* labelsData = reinterpret_cast<int8_t*>(labelsBuffer->data().first);

  evaluate(nodeParams, dynParams->imagePaths, probsData, labelsData);
  // for (int i=0; i<batchSize; i++) {
  //   std::cout << (int)labelsData[i] << std::endl;
  // }
  out.push_back(probsBuffer);
  out.push_back(labelsBuffer);
 //  std::cout << "\n[DBG] FallDetectionEvaluation Node end ..... \n" << std::endl;
  return -1; /// No wait
}

void FallDetectionEvaluation::report(AKS::NodeParams* nodeParams)
{
  EvaluationData *accuData = getEvaluationData(nodeParams);
  if(accuData){
    std::cout << "Total images processed: " << accuData->_imagesProcessed << std::endl;
    std::cout << "Accuracy: " << float(accuData->accuracy) / float(accuData->_imagesProcessed) << std::endl;
    std::cout << "Sensitivity/Recall: " << float(accuData->truePos) / float(accuData->truePos + accuData->falseNeg) << std::endl;
    std::cout << "Specificity: " << float(accuData->trueNeg) / float(accuData->trueNeg + accuData->falsePos) << std::endl;
    std::cout << "FAR/FPR: " << float(accuData->falsePos) / float(accuData->falsePos + accuData->trueNeg) << std::endl;
    std::cout << "MDR/FNR: " << float(accuData->falseNeg) / float(accuData->falseNeg + accuData->truePos) << std::endl;
  }
}

