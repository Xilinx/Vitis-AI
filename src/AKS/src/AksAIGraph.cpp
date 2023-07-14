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
#include <list>

#include "AksAIGraph.h"
#include "AksAIGraphNode.h"

using namespace std;
using namespace AKS;

AIGraph::AIGraph(string name):_name(name)
{
  //
}

AIGraph::~AIGraph()
{
  for (auto& node : _nodesVector) delete node;
}

void AIGraph::addNode(AIGraphNode *node)
{
  if(node){
    //TODO - Check if same name node doesn't already exist
    node->setIndex(_nodesVector.size());
    _nodesVector.push_back(node);
  }
}

set<KernelDef*> AIGraph::getUniqueKernels()
{
  set<KernelDef*> ret;
  for (auto& node : _nodesVector) {
    node->provideExecKernels(ret);
  }
  return ret;
}

bool AIGraph::isGraphOK(string &err)
{
  if(_name.empty()){
    //no name
    err = "Graph doesn't have a name";
    return false;
  }
  _inDegree.clear();
  _outDegree.clear();
  int index = 0;
  //update the previous and next nodes 
  for (auto &currNode : _nodesVector){
    currNode->clearNextPrev();
    //next nodes
    const set<string> nextNodesStr = currNode->getNextNodesStr();
    for(auto nextNodeStr: nextNodesStr){
      AIGraphNode *res = findNode(nextNodeStr);
      if(res){
        currNode->addNextNode(res);      
      }else{
        err = "Unknown next node \'"+nextNodeStr+"\' for node \'"+currNode->getName()+"\'";
        return false;
      }
    }
    //prev nodes
    for (auto &node2 : _nodesVector){
      if(node2->isNodeANextNode(currNode->getName())){
        currNode->addPrevNode(node2);
      }
    }
    _inDegree.push_back(currNode->getPrevNodes().size());
    _outDegree.push_back(currNode->getNextNodes().size());
    if(_inDegree[_inDegree.size()-1] == 0){
      _firstNode = currNode;
      //cout<<"Setting first node as "<<currNode->getName()<<endl;
    }
  }
  //Now check
  for (auto &node : _nodesVector){
    if(!node->isNodeOK(err)) return false;
  }
  return true;
}

void AIGraph::dump(string prefix)
{
  cout<<prefix<<"Graph Name:"<<_name<<endl;
  cout<<prefix<<"  Version:"<<_version<<endl;
  cout<<prefix<<"  Nodes:"<<endl;
  for (int i = 0; i < _nodesVector.size();++i){
    _nodesVector[i]->dump(prefix+"    ");
    cout<<"    Index of node in graph(must be same as in node):"<<i<<endl;
    cout<<"    In-degree of node:"<<_inDegree[i]<<endl;
    cout<<"    Out-degree of node:"<<_outDegree[i]<<endl;
  }
}
  
AIGraphNode* AIGraph::findNode(string nodeName)
{
  if(!nodeName.empty()){
    for (auto &node : _nodesVector){
      if(node->getName() == nodeName) return node;
    }
  }
  return nullptr;
}

