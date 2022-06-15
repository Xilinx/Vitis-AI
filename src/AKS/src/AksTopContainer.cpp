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
#include "AksTopContainer.h"
#include "AksKernelDef.h"
#include "AksAIGraph.h"
#include "AksCommonDefs.h"
#include "aks/AksLogger.h"
#include <iostream>

using namespace std;
using namespace AKS;

TopContainer* TopContainer::_global = nullptr;

TopContainer* TopContainer::getGlobal()
{
  if(!_global){
    _global = new TopContainer();
  }
  return _global;
}

TopContainer::~TopContainer()
{
  for (const auto& kernel : _kernels) delete kernel.second;
  for (const auto& graph : _graphs) delete graph.second;
}
  

void TopContainer::deleteGlobal()
{
  delete _global;
}

void TopContainer::addKernel(KernelDef *def)
{
  if(!def) return;

  if(_kernels.find(def->getName()) == _kernels.end()){
    _kernels[def->getName()] = def;
    LOG_X(DEBUG) <<"Adding kernel: "<< def->getName()<<endl;
  }else{
    LOG_X(WARNING) <<"Kernel already exists: "<< def->getName()<<endl;
  }
}

void TopContainer::addGraph(AIGraph *graph)
{
  if(!graph) return;

  if(_graphs.find(graph->getName()) == _graphs.end()){
    _graphs[graph->getName()] = graph;
    LOG_X(INFO) <<"Adding graphs: "<< graph->getName()<<endl;
  }else{
    LOG_X(INFO) <<"Graph already exists:"<< graph->getName()<<endl;
  }
}

set<KernelDef*> TopContainer::getUniqueKernelsForLoadedGraphs()
{
  set<KernelDef*> ret;
  for (const auto& gpair : _graphs) {
    AIGraph *graph = gpair.second;
    set<KernelDef*> temp = graph->getUniqueKernels();
    ret.insert(temp.begin(),temp.end());
  }
  return ret;
}

KernelDef* TopContainer::getKernel(string str)
{
  auto itr = _kernels.find(str);
  if(itr != _kernels.end()){
    return itr->second;
  }
  return nullptr;
}

AIGraph* TopContainer::getGraph(string str)
{
  auto itr = _graphs.find(str);
  if(itr != _graphs.end()){
    return itr->second;
  }
  return nullptr;
}
