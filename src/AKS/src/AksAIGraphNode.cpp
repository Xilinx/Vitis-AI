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

#include "AksAIGraphNode.h"
#include "AksAIGraph.h"
#include "AksCommonDefs.h"
#include "AksKernelDef.h"
#include "aks/AksNodeParams.h"

using namespace AKS;

AIGraphNode::AIGraphNode(string name,AIGraph* parentGraph):_name(name),_parentGraph(parentGraph)
{
  //
}

AIGraphNode::~AIGraphNode()
{
  //Don't delete the kernel pointer. that's owned by TopContainer 
  for (const auto& kernel : _execKernels){
    if(kernel.second) delete kernel.second;
  }
}

void AIGraphNode::addExecKernel(KernelDef* kernel,NodeParams *pvalues)
{
  if(_execKernels.find(kernel) != _execKernels.end()){
    cout<<"[Warning] Kernel "<<kernel->getName()<<" already exists for node "<< _parentGraph->getName()<<"."<<_name<<endl;
  }
  _execKernels[kernel] = pvalues;
}

void AIGraphNode::addNextNodeStr(string nname)
{
  _nextNodesStr.insert(nname);
}

void AIGraphNode::dump(string prefix)
{
  cout<<prefix<<"Node Name: "<< _name<<endl;
  cout<<prefix<<"  Index: "<< _index<<endl;
  cout<<prefix<<"  Node Exec Kernels: "<<endl;
  for (const auto& kpair : _execKernels) {
    cout<<prefix<<"    Kernel Name: "<<kpair.first->getName()<<endl;
    cout<<prefix<<"    Param Values:"<<endl;
    if(kpair.second){
      kpair.second->dump(prefix + "      ");
    }
  }
  cout<<prefix<<"  Next Nodes:";
  for(auto &nextNode: _nextNodes){
      cout<<nextNode->getName()<<",";
  }
  cout<<endl;
  cout<<prefix<<"  Previous Nodes:";
  for(auto prevNode: _prevNodes){
      cout<<prevNode->getName()<<",";
  }
  cout<<endl;
}

void AIGraphNode::provideExecKernels(set<KernelDef*> &kernels)
{
  for (const auto& kpair : _execKernels) {
    kernels.insert(kpair.first);
  }
}
void AIGraphNode::addNextNode(AIGraphNode *nextNode)
{
  if(nextNode) _nextNodes.push_back(nextNode);
}

void AIGraphNode::addPrevNode(AIGraphNode *prevNode)
{
  if(prevNode) _prevNodes.push_back(prevNode);
}

bool AIGraphNode::isNodeANextNode(string nname)
{
  if(_nextNodesStr.find(nname) != _nextNodesStr.end()) return true;
  return false;
}

bool AIGraphNode::isNodeOK(string &err)
{
  if(!_parentGraph){
    err = "No parent graph for node \'"+_name+"\'";
    return false;
  }
  if(_nextNodes.size() < 1 && _prevNodes.size() < 1 && _parentGraph->getNodesVector().size() > 1){
    err = _name + " is a hanging node";
    return false;
  }
  //TODO - Check if Kernel and Opparam values match
  return true;
}
