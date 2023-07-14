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
#include "aks/AksSysManagerExt.h"
#include "AksSysManager.h"

using namespace std;
using namespace AKS;

extern "C" {
  AKS::SysManagerExt* createSysManagerExt();
  void loadKernels(AKS::SysManagerExt* sysMan, const char* kernelDir);
  void loadGraphs(AKS::SysManagerExt* sysMan, const char* graphPath);
  AKS::AIGraph* getGraph(AKS::SysManagerExt* sysMan, const char* graphName);
  void enqueueJob(AKS::SysManagerExt* sysMan, AKS::AIGraph* graph,
      const char* imagePath, AKS::NodeParams* params);
  void waitForAllResults(AKS::SysManagerExt* sysMan, AKS::AIGraph* graph);
  void resetTimer(AKS::SysManagerExt* sysMan);
  void report(AKS::SysManagerExt* sysMan, AKS::AIGraph* graph);
  void deleteSysManagerExt();
}

SysManagerExt* SysManagerExt::_global = nullptr;

SysManagerExt* SysManagerExt::getGlobal()
{
  if (!_global) {
    _global = new SysManagerExt();
  }
  return _global;
}

void SysManagerExt::deleteGlobal()
{
  AKS::SysManager::deleteGlobal();
  if (_global) {
    delete _global;
    _global = nullptr;
  }
}

SysManagerExt::SysManagerExt()
{
}

SysManagerExt::~SysManagerExt()
{
}

void SysManagerExt::loadGraphs(std::string graphJson)
{
  return AKS::SysManager::getGlobal()->loadGraphs(graphJson);
}

int SysManagerExt::loadKernels(std::string dir)
{
  return AKS::SysManager::getGlobal()->readKernelJsons(dir);
}

AIGraph* SysManagerExt::getGraph(std::string graphName)
{
  return AKS::SysManager::getGlobal()->getGraph(graphName);
}

//std::future<std::vector<DataDescriptor>> SysManagerExt::enqueueJob(
std::future<VecPtr<vart::TensorBuffer>> SysManagerExt::enqueueJob(
    AIGraph* graph, const std::string& filePath,
    VecPtr<vart::TensorBuffer> inputs,AKS::NodeParams* userArgs)
    //std::vector<DataDescriptor> inputs,AKS::NodeParams* userArgs)
{
  return AKS::SysManager::getGlobal()->enqueueJob(graph,filePath,std::move(inputs),userArgs);
}

//std::future<std::vector<DataDescriptor>> SysManagerExt::enqueueJob(
std::future<VecPtr<vart::TensorBuffer>> SysManagerExt::enqueueJob(
    AIGraph* graph, const std::vector<std::string>& filePaths,
    VecPtr<vart::TensorBuffer> inputs,AKS::NodeParams* userArgs)
    //std::vector<DataDescriptor> inputs,AKS::NodeParams* userArgs)
{
  return AKS::SysManager::getGlobal()->enqueueJob(graph,filePaths,std::move(inputs),userArgs);
}

void SysManagerExt::waitForAllResults()
{
  return AKS::SysManager::getGlobal()->waitForAllResults();
}

void SysManagerExt::waitForAllResults(AIGraph* graph)
{
  return AKS::SysManager::getGlobal()->waitForAllResults(graph);
}

void SysManagerExt::report(AIGraph* graph)
{
  return AKS::SysManager::getGlobal()->report(graph);
}

void SysManagerExt::printPerfStats()
{
  return AKS::SysManager::getGlobal()->printPerfStats();
}

void SysManagerExt::resetTimer() 
{
  return AKS::SysManager::getGlobal()->resetTimer();
}

void SysManagerExt::pyEnqueueJob(AIGraph* graph, const std::string& filePath)
{
  enqueueJob(graph, filePath);
}

void SysManagerExt::pyEnqueueJob(AIGraph* graph, const std::vector<std::string>& filePaths)
{
  enqueueJob(graph, filePaths);
}


// -----------  C API -------------------- //

AKS::SysManagerExt* createSysManagerExt() {
  AKS::SysManagerExt* sysMan = AKS::SysManagerExt::getGlobal();
  return sysMan;
}

void loadKernels(AKS::SysManagerExt* sysMan, const char* kernelDir) {
  sysMan->loadKernels(kernelDir);
}

void loadGraphs(AKS::SysManagerExt* sysMan, const char* graphPath) {
  sysMan->loadGraphs(graphPath);
}

AKS::AIGraph* getGraph(AKS::SysManagerExt* sysMan, const char* graphName) {
  return sysMan->getGraph(graphName);
}

void enqueueJob(AKS::SysManagerExt* sysMan, AKS::AIGraph* graph,
    const char* imagePath, AKS::NodeParams* params) {
  VecPtr<vart::TensorBuffer> v;
  sysMan->enqueueJob(graph, imagePath, std::move(v), params);
}

void waitForAllResults(AKS::SysManagerExt* sysMan, AKS::AIGraph* graph) {
  if(graph)
    sysMan->waitForAllResults(graph);
  else
    sysMan->waitForAllResults();
}

void resetTimer(AKS::SysManagerExt* sysMan) {
  sysMan->resetTimer();
}

void report(AKS::SysManagerExt* sysMan, AKS::AIGraph* graph) {
  sysMan->report(graph);
}

void deleteSysManagerExt() {
  AKS::SysManagerExt::deleteGlobal();
}
