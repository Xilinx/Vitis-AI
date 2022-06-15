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
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <chrono>

#include <pybind11/pybind11.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "AksKernelDef.h"
#include "AksTopContainer.h"
#include "AksCommonUtils.h"
#include "AksCommonDefs.h"
#include "AksParamProps.h"
#include "AksAIGraphNode.h"
#include "AksSysManager.h"
#include "aks/AksNodeParams.h"
#include "aks/AksKernelBase.h"
#include "aks/AksTensorBuffer.h"
#include "aks/AksLogger.h"
#include "AksAIJob.h"
#include "AksTracer.h"
#include "AksAIGraph.h"
#include "AksAIGraphInAction.h"
#include "AksAliveNode.h"

using namespace std;
using namespace AKS;
using boost::property_tree::ptree;
using namespace std::chrono;
namespace py=pybind11;

#include <dlfcn.h>

SysManager* SysManager::_global = nullptr;

SysManager* SysManager::getGlobal()
{
  if (!_global) {
    _global = new SysManager();
  }
  return _global;
}

void SysManager::deleteGlobal()
{
  if (_global) {
    delete _global;
    _global = nullptr;
  }
  AKS::TopContainer::deleteGlobal();
}

SysManager::SysManager()
{
  Logger::setMinLogLevel();
  char* s_maxJobs = std::getenv("AKS_MAX_CONCURRENT_JOBS");
  if(s_maxJobs) {
    _max_concurrent_jobs = std::atoi(s_maxJobs);
  }
}

SysManager::~SysManager()
{
  // SLOG(LogLevel::DEBUG,
  //     _DEBUG    << "Job Enqueue Time (s): " << _t_enq_dur.count() << ", ";
  //     std::cout << "Total jobs : " << _jobID << "\n";)

  for(auto& q: _queues) {
    q.second->close();
  }
  _waitQueue.close();

  for(int i=0; i<_workers.size(); ++i) {
    _workers[i].join();
  }

  for(auto& q: _queues) {
    delete q.second;
  }
  //for(auto i: _graphia) {
  //  delete i;
  //}
}

void* findFunc(std::string func,std::string soPath)
{
  if(func.empty() || soPath.empty()){
    return nullptr;
  }

  // reset errors
  dlerror();

  /* open the needed object */
  void    *handle = dlopen(soPath.c_str(), RTLD_LOCAL | RTLD_LAZY);
  const char* dlopen_error = dlerror();
  if(dlopen_error){
    LOG_X(WARNING)  << dlopen_error
                  << ". Skipping this kernel." << endl;
    return nullptr;
  }

  /* find the address of function  */
  void *fptr = dlsym(handle, func.c_str());

  const char* dlsym_error = dlerror();
  if (dlsym_error) {
    LOG_X(WARNING)  << "Could not find " << func << " in "<< soPath << " : " << dlsym_error
                  << ". Skipping this kernel." << endl;
    return nullptr;
  }else{
    return fptr;
  }
}

inline bool ends_with(std::string const & value, std::string const & ending)
{
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

void print_ptree( const ptree& tree, int level = 0 )
{
  if( tree.empty()) {
    LOG_X(DEBUG) << '"' << tree.get_value<std::string>() << "\"\n";
  } else {
    for( const auto& kv : tree ) {
      LOG_X(DEBUG) << std::string( level, '\t' ) << '"' << kv.first << "\": ";
      print_ptree( kv.second, level + 1 );
    }
  }
}

int SysManager::readKernelJsons(std::string dir)
{
  if(dir.empty()) return 0;

  boost::filesystem::path p(dir);

  boost::filesystem::directory_iterator it{p};
  for (;it != boost::filesystem::directory_iterator{};it++){
    std::string jsonFile = (*it).path().string();
    if(ends_with(jsonFile,".json")){
      LOG_X(INFO) << "Reading "<< jsonFile << '\n';
      ifstream file(jsonFile.c_str());
      if (!file.good()){
        LOG_X(ERROR) <<"Could not open "<<jsonFile<<endl; 
        continue;
      }

      ptree pt;
      try {
        read_json(file, pt);
      } catch(std::exception & e){
        LOG_X(ERROR) <<"Error reading "<<jsonFile<<endl;
        LOG_X(ERROR) <<e.what()<<endl;
      }
      
      //print_ptree(pt);

      //Kernel Name
      std::string kname = pt.get<std::string>("kernel_name","");
      if(kname.empty()){
        LOG_X(WARNING) <<"No kernel name found in "<<jsonFile<<". Skipping"<<endl;
        continue;
      }
      AKS::KernelDef *kernel = new AKS::KernelDef(kname,(*it).path());

      //Kernel Description : Mandatory
      std::string desc = pt.get<std::string>("description","");
      if(desc.empty()){
        LOG_X(WARNING) <<"No kernel description found in "<<jsonFile<<". Skipping"<<endl;
        continue;
      }

      //Kernel Lib
      string klib = pt.get<string>("kernel_lib","");
      if(!klib.empty()){
        bool found = false;
        boost::filesystem::path p;
        //assume full path or relative to current dir
        if (boost::filesystem::exists(klib)){
          boost::filesystem::path p(klib);
          kernel->setLibPath(p);
          found = true;
        }else{
          //Check in LD_LIBRARY_PATH
          //TODO: decide which environment variable(s) to read 
          if(const char* env_p = std::getenv("LD_LIBRARY_PATH")){
            vector<string> dirs; 
            boost::split(dirs, env_p, boost::is_any_of(":")); 
            for (int i = 0; i < dirs.size(); i++){ 
              string fname = dirs[i] + boost::filesystem::path::preferred_separator + klib;
              //cout << "Looking for "<< klib <<" in "<<dirs[i] << endl;
              if (boost::filesystem::exists(fname)){
                boost::filesystem::path p(fname);
                kernel->setLibPath(p);
                found = true;
                break;
              }
            }
          }
        }
        if(!found){
          LOG_X(WARNING) << "Couldn't find kernel_lib \'"<<klib<< "\' for "<<kernel->getName()<<". Skipping this kernel."<<endl;
          delete kernel;
          continue;
        }
      }
      
      //Number of CUs (optional)
      int kNumCUs = pt.get<int>("num_cu",1);
      kernel->setJsonNumCUs(kNumCUs);

      //Kernel Type
      string ktype = pt.get<string>("kernel_type","");
      if(ktype.empty()){
        LOG_X(WARNING) <<"No kernel type found in "<<jsonFile<<". Skipping"<<endl;
        delete kernel;
        continue;
      }
      kernel->setKernelType(AKS::CommonUtils::getKernelTypeForStr(ktype));

      //Device Type
      string dtype = pt.get<string>("device_type","");
      if(dtype.empty()){
        LOG_X(WARNING) <<"No device type mentioned in "<<jsonFile<<". Skipping this kernel."<<endl;
        delete kernel;
        continue;
      }else{
        kernel->setDeviceType(AKS::CommonUtils::getDeviceTypeForStr(dtype));
      }

      //Queue size
      string qSize = pt.get<string>("kernel_queue_size","");
      if(!qSize.empty()){
        // TODO ARK: setQueueSize is redundant now
        kernel->setQueueSize(stoi(qSize));
        kernel->setAsyncQueueSize(stoi(qSize));
      }

      //param list
      bool paramsListFound = false;
      try{
        auto plist = pt.get_child("param_list");
        paramsListFound = true;
      }catch(std::exception & e){
        LOG_X(WARNING) <<"Missing or bad parameter \'param_list\' in "<<jsonFile<<endl;
        LOG_X(WARNING) <<e.what()<<endl;
      }
      if(paramsListFound){
        BOOST_FOREACH(boost::property_tree::ptree::value_type &v, pt.get_child("param_list"))
        {
          //cout<<"Param name:"<<v.first.data()<<endl;
          AKS::ParamProps *props = new AKS::ParamProps();
          kernel->addParam(v.first.data(),props);
          for (ptree::iterator pos = v.second.begin(); pos != v.second.end(); pos++) {
            //cout<<"  Property Name:"<<pos->first<<endl;
            if(pos->first == "type"){
              props->type = AKS::CommonUtils::getKernelParamTypeForStr(pos->second.data()); 
            }else if(pos->first == "default"){
              //TODO - support for default values
            }else if (pos->first == "optional"){
              props->optional = AKS::CommonUtils::getKernelParamOptionalForStr(pos->second.data()); 
            }else{
              LOG_X(WARNING) <<"Unknown param property: "<< pos->first<<endl;
            }
          }
        }
      }
      string klibPath = kernel->getLibPathStr();
      //check for getKernel function in the kernel lib
      void *funcPtr = findFunc("getKernel",klibPath);
      if(!funcPtr){
        delete kernel;
        continue;
      }else{
        kernel->setInitFunc(funcPtr);
      }
      /*
      //exec function
      func = pt.get<string>("exec","");
      if(!func.empty()){
        void *funcPtr = findFunc(func,klibPath);
        if(!funcPtr){
          cout<<"Exec function "<<func<<" not found in "<<klibPath<<". Skipping this kernel."<<endl;
          delete kernel;
          continue;
        }else{
          kernel->setExecFunc(funcPtr);
        }
      }
      //execAsync function
      func = pt.get<string>("execAsync","");
      if(!func.empty()){
        void *funcPtr = findFunc(func,klibPath);
        if(!funcPtr){
          cout<<"ExecAsync function "<<func<<" not found in "<<klibPath<<". Skipping this kernel."<<endl;
          delete kernel;
          continue;
        }else{
          kernel->setExecAsyncFunc(funcPtr);
        }
      }
      //wait function
      func = pt.get<string>("wait","");
      if(!func.empty()){
        void *funcPtr = findFunc(func,klibPath);
        if(!funcPtr){
          cout<<"wait function "<<func<<" not found in "<<klibPath<<". Skipping this kernel."<<endl;
          delete kernel;
          continue;
        }else{
          kernel->setWaitFunc(funcPtr);
        }
      }
      //cleanup function
      func = pt.get<string>("cleanup","");
      if(!func.empty()){
        void *funcPtr = findFunc(func,klibPath);
        if(!funcPtr){
          cout<<"cleanup function "<<func<<" not found in "<<klibPath<<". Skipping this kernel."<<endl;
          delete kernel;
          continue;
        }else{
          kernel->setCleanupFunc(funcPtr);
        }
      }*/
      string err;
      if(kernel->isKernelOK(err)){
        AKS::TopContainer::getGlobal()->addKernel(kernel);
        SLOG(LogLevel::DEBUG, kernel->dump("");)
      }else{
        LOG_X(ERROR) <<"Not adding kernel: "<<kernel->getName()<<"(Reason: "<<err<<")"<<endl;
        delete kernel;
        continue;
      }
    }
  }
  return 0;
}

int SysManager::loadGraphJson(string jsonFile)
{
  ifstream file(jsonFile.c_str());
  if (!file.good()) {
    LOG_X(ERROR) <<"Could not open "<<jsonFile<<endl; 
    return -1;
  }
  
  ptree pt;
  try {
    read_json(file, pt);
  } catch (std::exception & e) {
    LOG_X(ERROR) << e.what() << endl;
  }
  
  //print_ptree(pt);
  //Graph Name
  string gname = pt.get<string>("graph_name","");
  if(gname.empty()){
    LOG_X(ERROR) << "No graph name found in " << jsonFile << ". Skipping" << endl;
    return -1;
  }

  AKS::AIGraph *graph = new AKS::AIGraph(gname);
  
  //Version
  string version = pt.get<string>("version","");
  graph->setVersion(version);

  //node list
  try{
    auto nlist = pt.get_child("node_list");
  }catch(std::exception & e){
    LOG_X(ERROR) <<"Missing or bad parameter \'node_list\' in "<<jsonFile<<endl;
    LOG_X(ERROR) <<e.what()<<endl;
    LOG_X(ERROR) <<"Not adding graph: " << graph->getName()<< endl;
    delete graph;
    return -1;
  }
  BOOST_FOREACH(boost::property_tree::ptree::value_type &v, pt.get_child("node_list"))
  {
    string nname = v.second.get<string>("node_name","");
    AKS::AIGraphNode *node = new AKS::AIGraphNode(nname,graph);
    graph->addNode(node);
    //next_node
    try{
      BOOST_FOREACH(boost::property_tree::ptree::value_type &nextnodes, v.second.get_child("next_node"))
      {
        assert(nextnodes.first.empty()); // array elements have no names
        node->addNextNodeStr(nextnodes.second.data());
        //cout<<"Next node for "<<nname<<" is "<<nextnodes.second.data()<<endl;
      }
    }catch(std::exception & e){
      //next_node may or may not be there
    }
    //node_params
    try{
      BOOST_FOREACH(boost::property_tree::ptree::value_type &nparams, v.second.get_child("node_params"))
      {
        string kernelName = nparams.first.data();
        KernelDef* k = AKS::TopContainer::getGlobal()->getKernel(kernelName);
        if(!k){
          LOG_X(ERROR) << "Couldn't find kernel \'" << kernelName << "\' for node \'" << nname << "\' in " << jsonFile << endl;
          LOG_X(ERROR) << "Not adding graph: " << graph->getName() << endl;
          delete graph;
          return -1;
        }
        //found the kernel, now add it with the params
        NodeParams *pVals = new NodeParams();
        for (ptree::iterator pos = nparams.second.begin(); pos != nparams.second.end(); pos++) {
          ParamProps* pProps = k->getParamProps(pos->first);
          if(!pProps){
            LOG_X(WARNING) <<pos->first<<": no such parameter present for the kernel \'"<<k->getName()<<endl;
          }else{
            //cout<<"Node parameter "<<pos->first<<" is of type: "<<AKS::CommonUtils::getStrForKernelParamType(pProps->type)<<endl;
            switch (pProps->type){
              case AKS::KernelParamType::INT: 
                {
                  int val = stoi(pos->second.data());
                  pVals->_intParams[pos->first] = val;
                  break;
                }
              case AKS::KernelParamType::INT_ARR: 
                {
                  vector<int> vals;
                  BOOST_FOREACH(boost::property_tree::ptree::value_type &arrElems, pos->second)
                  {
                    vals.push_back(stoi(arrElems.second.data()));
                  }
                  pVals->_intVectorParams[pos->first] = vals;
                  break;
                }
              case AKS::KernelParamType::FLOAT: 
                {
                  float val = stof(pos->second.data());
                  pVals->_floatParams[pos->first] = val;
                  break;
                }
              case AKS::KernelParamType::FLOAT_ARR: 
                {
                  vector<float> vals;
                  BOOST_FOREACH(boost::property_tree::ptree::value_type &arrElems, pos->second)
                  {
                    vals.push_back(stof(arrElems.second.data()));
                  }
                  pVals->_floatVectorParams[pos->first] = vals;
                  break;
                }
              case AKS::KernelParamType::STRING: 
                {
                  pVals->_stringParams[pos->first] = pos->second.data();
                  break;
                }
              case AKS::KernelParamType::STRING_ARR: 
                {
                  vector<string> vals;
                  BOOST_FOREACH(boost::property_tree::ptree::value_type &arrElems, pos->second)
                  {
                    vals.push_back(arrElems.second.data());
                  }
                  pVals->_stringVectorParams[pos->first] = vals;
                  break;
                }
              default:
                {
                  //TODO - error
                }
            }
          }
        }
        node->addExecKernel(k,pVals);
      }
    }catch(std::exception & e){
      //Every node should have node_params
      LOG_X(ERROR) <<"Missing or bad parameter \'node_params\' for node "<<nname<<"in "<<jsonFile<<endl;
      LOG_X(ERROR) <<e.what()<<endl;
      LOG_X(ERROR) <<"Not adding graph: " << graph->getName()<< endl;
      delete graph;
      return -1;
    }
  }
  string err;
  if(graph->isGraphOK(err)){
    AKS::TopContainer::getGlobal()->addGraph(graph);
    SLOG(LogLevel::DEBUG, graph->dump(""); )
  }else{
    LOG_X(ERROR) <<"Not adding graph: " << graph->getName() << "(Reason: " << err << ")" << endl;
    delete graph;
    return -1;
  }
  return 0;
}

int SysManager::readGraphJsons(std::string pathToJsons)
{
  if(pathToJsons.empty()) return 0;
  
  boost::filesystem::path jsonPath (pathToJsons);
  bool pathExists = false;
  try {
    if (pathExists = boost::filesystem::exists(jsonPath)) {
      LOG_X(DEBUG) << "Path: " << pathToJsons << " - Exists!" << endl;
    } else {
      LOG_X(ERROR) << "Path: " << pathToJsons << " - Does not exists!" << endl;
    }
  } catch (boost::filesystem::filesystem_error & e) {
    LOG_X(ERROR) << e.what() << endl;
  }

  bool isRegularFile;
  int ret;
  if (isRegularFile = boost::filesystem::is_regular_file(pathToJsons)){
    LOG_X(DEBUG) << "File Path: " << pathToJsons << endl;
    string jsonFile = pathToJsons;
    if(ends_with(jsonFile,".json")) {
        LOG_X(INFO) << "Reading "<< jsonFile << '\n';
        ret = loadGraphJson(jsonFile);     
    }
  } else {
    LOG_X(DEBUG) << "Directory Path: " << pathToJsons << endl;
    for (boost::filesystem::directory_iterator it {jsonPath}; 
          it != boost::filesystem::directory_iterator{}; it++) {
      string jsonFile = (*it).path().string();
      if(ends_with(jsonFile,".json")){ 
        LOG_X(INFO) << "Reading "<< jsonFile << '\n';
        ret = loadGraphJson(jsonFile);     
      }
    }
  }
  return 0;
}

// load the user defined graphs
void SysManager::loadGraphs(std::string graphJsonPath)
{
  int ret;
  if (ret = readGraphJsons(graphJsonPath) != 0) {
    LOG_X(ERROR) << "Unable to read graph JSONs." << endl;
  }
  updateQueuesAndWorkers();
}

void SysManager::addNewQueue(AKS::Queue<AKS::AIJob*>* que)
{
  LOG_X(DEBUG) << "Adding New Queue: " << que->getName() << std::endl;
  _queues[que->getName()] = que;
}

bool SysManager::isQueueActive(std::string name)
{
  if (_queues.find(name) != _queues.end()) {
    return true;
  } else {
    return false;
  }
}

void SysManager::listActiveQueues(void)
{
  for (std::map<std::string, AKS::Queue<AKS::AIJob*>*>::iterator it = _queues.begin();
      it != _queues.end(); ++it) 
  {
    LOG_X(DEBUG) << "Active Queue: " << it->first << std::endl;
  }
}

AKS::AIGraph* SysManager::getGraph(std::string graphName) {
  
  return AKS::TopContainer::getGlobal()->getGraph(graphName);
}

void SysManager::updateQueuesAndWorkers(void)
{
  /// Get unique kernels
  std::set<KernelDef*> uniqueKernels = AKS::TopContainer::getGlobal()->getUniqueKernelsForLoadedGraphs();
  int totalUniqueKernels = uniqueKernels.size();
  /// Kernel Init & Queue Creation
  LOG_X(DEBUG) << "----------------------------------------------" << std::endl;

  /// List of newly added kernels: clear this at the end of this function
  std::vector<std::string> newKernels;

  for (const auto & kernel : uniqueKernels) {
    LOG_X(DEBUG) << "Kernel Name: " << kernel->getName() << std::endl;
    if (!isQueueActive(kernel->getName())) {

      /// Get Kernel Init
      AKS::KernelBase* (*initializer) (AKS::NodeParams*) = 
                (AKS::KernelBase* (*) (AKS::NodeParams*)) kernel->getInitFunc();

      /// Get Kernel runtime params (Creating a dummy for kernel init)
      AKS::NodeParams *nodeParams = new AKS::NodeParams;

      //std::string pfx ("");
      //nodeParams->dump(pfx);
      
      /// Initialize device with kernel
      auto _tmpKernel = initializer(nodeParams);
      LOG_X(DEBUG) << "Node Initialize : " << _tmpKernel << std::endl;
      kernel->setKernelHandle(_tmpKernel);          /// Get Kernel Init

      /// Check if kernel is asyncKernel
      if(_tmpKernel->isExecAsync()) kernel->setKernelAsyncFlag(true);
  
      /// Create Queue for kernel
      AKS::Queue<AKS::AIJob*> * que = new AKS::Queue<AKS::AIJob*>(kernel->getName(), 512);
      addNewQueue (que);

      /// Add newly added kernel to list
      newKernels.push_back(kernel->getName());

      /// Delete temp node param
      delete nodeParams;
    } else {
      LOG_X(DEBUG) << "Queue/Kernel: " << kernel->getName() << " is already active !" << std::endl;
    }
  }

  /// Get numWaitThreads per asyncKernel and assign them to each kernel
  int numAsyncKernels = 0;
  for(auto& kernel: uniqueKernels) {
    if(kernel->isKernelAsync()) {
      ++numAsyncKernels;
    }
  }

  int numThreadsPerAsyncKernel = numAsyncKernels > 0 ? 
                                  MAX_NUM_THREADS / numAsyncKernels :
                                  MAX_NUM_THREADS;
  assert(numThreadsPerAsyncKernel > 0);

  std::for_each(uniqueKernels.begin(), uniqueKernels.end(), 
      [numThreadsPerAsyncKernel](KernelDef* k) { k->setNumWaitThreads(numThreadsPerAsyncKernel); });

  for(auto& kernel: uniqueKernels) {
    SLOG(LogLevel::DEBUG, kernel->dump("--");)
  }
  
  LOG_X(DEBUG) << "----------------------------------------------" << std::endl;
  LOG_X(DEBUG) << "Active Queues " << std::endl;
  listActiveQueues();
  LOG_X(DEBUG) << "----------------------------------------------" << std::endl;

  // Get initial timestamp
  _t_start = std::chrono::steady_clock::now();

  /// Create Worker threads
  // workerID = 0 means main thread. Push its log by default.
  _logs.push_back(WorkerLog(0, std::this_thread::get_id(), "main"));

  LOG_X(DEBUG) << "Creating Workers " << std::endl;
  LOG_X(DEBUG) << "----------------------------------------------" << std::endl;
  /// Iterate over kernels but create threads for only 
  /// newly created kernel
  for (const auto & kernel : uniqueKernels) {

    /// Check if the kernel is newly added
    auto it = std::find (newKernels.begin(), newKernels.end(), kernel->getName());
    if (it != newKernels.end()) {
      LOG_X(DEBUG) << "Kernel Name: " << kernel->getName() << std::endl;

      /// Create workers threads for new kernel
      AKS::Queue<AKS::AIJob*>* que = getQueue(kernel->getName());
      AKS::KernelBase * handle = kernel->getKernelHandle();

      //Check the KernelBase first for NumCUs
      int numCUs = handle->getNumCUs();
      if (numCUs == -1) {
        //means the default value from KernelBase, take from KernelDef
        numCUs = kernel->getJsonNumCUs();
      }
      if (numCUs == 0) {
        LOG_X(ERROR) << "Kernel: " << kernel->getName() << " has no CUs! Skipping thread creation." << std::endl;
        continue;
      }
      for (int n = 0; n < numCUs; ++n) {
        int workerID = _logs.size();
        LOG_X(DEBUG) << "worker : " << workerID << " : " << kernel->getName() << std::endl;
        std::thread t(&SysManager::performJob, this, que, que, workerID);
        _workers.push_back(std::move(t));
        _logs.push_back({workerID, _workers.back().get_id(), kernel->getName()+"_"+std::to_string(n)});
      }
    }
  }

  // Last workers are wait threads and their logs
  for(int i = 0; i < 1; ++i) {
    // std::cout << "wait worker : " << workerID << std::endl;
    // std::thread t(&SysManager::waitRoutine, this, workerID);
    // _workers.push_back(std::move(t));
    int workerID = _logs.size();
    _logs.push_back({workerID, std::thread::id(), "wait_"+std::to_string(i)});
    workerID++;
  }

  /// Node Inits
  LOG_X(DEBUG) << "Initialize Kernels for graphs" << std::endl;

  /// get graphs
  std::map<std::string, AKS::AIGraph*> & graphs = AKS::TopContainer::getGlobal()->getGraphs();
  /// Iterate over nodes of each graph
  for (auto graph : graphs) {
    LOG_X(DEBUG) << "Graph Name: " << graph.first << std::endl;
    /// Get all its nodes
    for (auto node : graph.second->getNodesVector()) {
      LOG_X(DEBUG) << "Node Name: " << node->getName() << std::endl;
      if (node->isInitialized()) {
        LOG_X(DEBUG) << "Node Name: " << node->getName() << " already intialized" << std::endl;
        continue;
      }
      /// Get kernel for each node
      std::map<AKS::KernelDef*, AKS::NodeParams*> & kernels = node->getKernels();
      for (const auto kernel : kernels) {
        LOG_X(DEBUG) << "Kernel Name: " << kernel.first->getName() << std::endl;
        if (kernel.second == nullptr) 
          LOG_X(WARNING) << kernel.first->getName() << ": Kernel Params: ----- NULL -----" << std::endl;
        auto kernelHandle  = kernel.first->getKernelHandle();
        if (kernelHandle) {
          /// Get Kernel runtime params
          AKS::NodeParams *nodeParams = kernel.second;
          if (nodeParams == nullptr) {
            LOG_X(ERROR) << "Not Found: Node Parameters" << std::endl;
            LOG_X(ERROR) << "Skipped Kernel Initialization" << std::endl;
            continue;
          }
          SLOG(LogLevel::DEBUG, nodeParams->dump("");)
          //call nodeInit now
          kernelHandle->nodeInit(nodeParams);
          node->initDone();
        }
      }//kernels
    }//nodes
  }//graphs

  /// Clear list of new kernels
  newKernels.clear();
  LOG_X(DEBUG) << "----------------------------------------------" << std::endl;
}

//std::future<std::vector<DataDescriptor>> SysManager::enqueueJob(
std::future<VecPtr<vart::TensorBuffer>> SysManager::enqueueJob(
    AIGraph* graph, const std::vector<std::string>& filePaths,
    //std::vector<DataDescriptor> inputs,
    VecPtr<vart::TensorBuffer> inputs,
    AKS::NodeParams* userArgs)
{
  std::chrono::time_point<std::chrono::steady_clock> t0, t1, t2;
  if(_TRACE_) t0 = std::chrono::steady_clock::now();

  uint64_t jid = _jobID++;

  /// #. Prepare any user input buffers
  // std::vector<AKS::DataDescriptor*> inputPtrs(inputs.size());
  std::vector<vart::TensorBuffer*> inputPtrs(inputs.size());
  for(int i=0; i<inputs.size(); ++i) {
    //inputPtrs[i] = new AKS::DataDescriptor(std::move(inputs[i]));
    inputPtrs[i] = inputs[i].release();
  }

  /// #. Create DynamicParamValues for this job
  AKS::DynamicParamValues* dynParams;
  if(userArgs)
    dynParams = new AKS::DynamicParamValues(*userArgs, filePaths);
  else
    dynParams = new AKS::DynamicParamValues(NodeParams(), filePaths);

  /// #. Create AIGraphInAction for this job and get the associated future
  AKS::AIGraphInAction* curGraphInAction = new AKS::AIGraphInAction(graph, jid, dynParams);
  // _graphia.push_back(curGraphInAction);
  std::future<VecPtr<vart::TensorBuffer>> fut = curGraphInAction->getFuture();

  /// #. Create AIJob for the first node
  AKS::AIGraphNode* firstNode = graph->getFirstNode();
  AKS::AIJob* job = new AKS::AIJob(curGraphInAction, firstNode);

  /// #. Create a dummy aliveNode and push it into active nodes
  // TODO @abidk keeping it in nullptr is bad idea.
  curGraphInAction->insertAliveNode(nullptr, new AKS::AliveNode(std::move(inputPtrs)));

  /// #. Push the job to input queue
  auto _inputQueue = getQueue(firstNode->getKernel()->getName());
  if(_TRACE_) t1 = std::chrono::steady_clock::now();
  if(_max_concurrent_jobs > 0) {
    std::unique_lock<std::mutex> lock(_mtx);
    _cv.wait(lock, [this]() { return _numJobs.load() < _max_concurrent_jobs; });
  }
  _inputQueue->push(job);
  _numJobs++;

  if(_TRACE_) t2 = std::chrono::steady_clock::now();

  if(_TRACE_) {
    _logs[0].addEntry("enQ", t0, t2, jid, 'X');
    // _logs[0].addEntry("push", t1, t2, jid, 'X'); 
  }

  /// #. Return the associated future.
  return fut;
}

//std::future<std::vector<DataDescriptor>> SysManager::enqueueJob(
std::future<VecPtr<vart::TensorBuffer>> SysManager::enqueueJob(
    AIGraph* graph, const std::string& filePath,
    //std::vector<DataDescriptor> inputs,
    VecPtr<vart::TensorBuffer> inputs,
    AKS::NodeParams* userArgs)
{
  return enqueueJob(graph, std::vector<std::string>{filePath}, std::move(inputs), userArgs);
}

void SysManager::performJob(AKS::Queue<AIJob*>* qin, AKS::Queue<AIJob*>* qout, int workerID)
{
  // Required below line to link python libs
  // bool gil_status = PyGILState_Check();

  std::chrono::time_point<std::chrono::steady_clock> t0, t1, t2, t3, t4, t5;
  std::chrono::duration<float> total{0};
  std::chrono::duration<float> waitTime{0};
  bool isKernelAsync = false;
  int totalJobs = 0;
  auto t_start = std::chrono::steady_clock::now();

  // To get kernelDef after all jobs are done
  // TODO : abidk : This is fine now, but not a good idea 
  // if you go for load-balancing among workers
  KernelDef* commonKernelDef = nullptr;

  while(true) {
    ///. # Pop the job from qin
    AKS::AIJob* curJob = nullptr;
    if(_TRACE_) t0 = std::chrono::steady_clock::now();
    bool status = qin->pop(curJob);
    if(_TRACE_) t1 = std::chrono::steady_clock::now();
    if(!status || !curJob) break;

    ///. # Prepare params for execution
    AKS::AIGraphInAction* curGraphIA = curJob->getCurrentGraphInAction();
    AKS::AIGraphNode* curNode        = curJob->getCurrentNode();

    auto curJobID      = curGraphIA->getJobID();
    auto kernelDef     = curNode->getKernel();
    commonKernelDef    = kernelDef;
    auto nodeParams    = curNode->getOpParam(kernelDef);
    auto dynamicParams = curGraphIA->getDynamicParams();
    auto kernelHandle  = kernelDef->getKernelHandle();

    ///. # Get the alive Nodes
    auto prevNodes = curNode->getPrevNodes();
    int num_prevNodes = prevNodes.size();
    //std::vector<AKS::DataDescriptor*> inputs;
    std::vector<vart::TensorBuffer*> inputs;
    std::vector<AKS::AliveNode*> parentAliveNodes;
    if(num_prevNodes > 0) {
      inputs.reserve(num_prevNodes);
      parentAliveNodes.reserve(num_prevNodes);

      ///. # Prepare input buffers.
      ///. Since they are spread across multiple parent alive nodes, first get a list of parent alive nodes.
      ///. And combine all the input vectors to a single vector.
      for(auto& prevNode: curNode->getPrevNodes()) {
        auto prevAliveNode = curGraphIA->getAliveNode(prevNode);
        parentAliveNodes.push_back(prevAliveNode);
        auto prevNodeOutputs = prevAliveNode->getOutputs();
        inputs.insert(inputs.end(), prevNodeOutputs.begin(), prevNodeOutputs.end());
      }
    } else {
      auto prevAliveNode = curGraphIA->getAliveNode(nullptr);
      parentAliveNodes.push_back(prevAliveNode);
      auto prevNodeOutputs = prevAliveNode->getOutputs();
      inputs.insert(inputs.end(), prevNodeOutputs.begin(), prevNodeOutputs.end());
    }

    ///. # Prepare output buffer holder
    // std::vector<AKS::DataDescriptor*> outputs;
    std::vector<vart::TensorBuffer*> outputs;

    ///. # Execute the job
    if (!kernelDef->isKernelAsync()) {
      t2 = std::chrono::steady_clock::now();
      int jid = kernelHandle->exec_async(inputs, outputs, nodeParams, dynamicParams);
      t3 = std::chrono::steady_clock::now();
      postExecRoutine(curJob, outputs, workerID);
    } else {
      isKernelAsync = true;
      t5 = std::chrono::steady_clock::now();
      std::unique_lock<std::mutex> waitLK(_waitMtx);
      _waitCV.wait(waitLK, [this, kernelDef](){
          auto t = kernelDef->getNumWaitThreads();
          // std::cout << kernelDef->getName() << " " << t << '\n';
          bool cond =  t > 0;
          return cond;
          });

      t2 = std::chrono::steady_clock::now();
      int jid = kernelHandle->exec_async(inputs, outputs, nodeParams, dynamicParams);
      t3 = std::chrono::steady_clock::now();

      // increment it after submitting job since there could be a delay in
      // submission itself (for eg: waiting for free jobID in dpuv1)
      kernelDef->incNumExistingJobs();

      curJob->setJobID(jid);
      curJob->setWorkerID(workerID);
      curJob->setOutputs(outputs);
      kernelDef->decNumWaitThreads();
      std::thread t(&SysManager::waitRoutine, this, curJob);
      t.detach();
      waitLK.unlock();
      waitTime += (t2-t5);
      // _waitQueue.push(curJob);
    }

    if(_TRACE_) t4 = std::chrono::steady_clock::now();

    total += (t3-t2);
    totalJobs++;

    if(_TRACE_) {
      _logs[workerID].addEntry("pop", t0, t1, curJobID, 'X'); 
      if(!kernelDef->isKernelAsync()) {
        _logs[workerID].addEntry(curNode->getName(), t2, t3, curJobID, 'K'); 
        _logs[workerID].addEntry("post", t3, t4, curJobID, 'X'); 
      } else {
        _logs[workerID].addEntry(curNode->getName(), t2, t3, curJobID, 'W'); 
        _logs[workerID].addEntry("wPush", t3, t4, curJobID, 'X'); 
      }
    }
  }

  // Report Performance Metrics
  t_start                 = std::max(_t_start, t_start);
  auto t_end              = std::chrono::steady_clock::now();
  auto total_worker_time  = std::chrono::duration<float>{t_end - t_start}.count();
  auto kernel_exec_time   = total.count();
  auto peak_fps           = totalJobs / kernel_exec_time;
  auto worker_utilization = kernel_exec_time * 100.0f / total_worker_time;
  auto async_wait_time    = waitTime.count();
  auto async_kernel_time  = commonKernelDef ? commonKernelDef->getKernelActiveTime() : 0.0f;
  auto kernel_utilization = async_kernel_time * 100.0f / total_worker_time;

  _mtx.lock();
  auto coutflags = std::cout.flags();
  std::cout.precision(2);
  std::cout << std::fixed;
  SLOG(LogLevel::DEBUG,
      _DEBUG    << "Worker: " << _logs[workerID].name << " - ";
      std::cout << "Total jobs : " << totalJobs << "\n";)
  if(isKernelAsync) {
    SLOG(LogLevel::DEBUG,
        _DEBUG << "|--- Async Kernel : ";
        std::cout << "Submit time (s) : " << kernel_exec_time << ", ";
        std::cout << "Wait time (s) : " << async_wait_time << ", ";
        std::cout << "Kernel Active Time (s): " << async_kernel_time << "\n";)
  } else {
    SLOG(LogLevel::DEBUG,
        _DEBUG << "|--- Blocking Kernel : ";
        std::cout << "Exec time (s) : " << kernel_exec_time << ", ";
        std::cout << "Peak FPS possible: " << peak_fps << ", ";
        std::cout << "Utilization : " << worker_utilization << "%" << "\n";)
  }
  std::cout.flags(coutflags);
  _mtx.unlock();
}

void SysManager::waitForAllResults()
{
  std::unique_lock<std::mutex> lock(_mtx);
  _cv.wait(lock, [this]() { return _numJobs.load() <= 0; });
  LOG_X(DEBUG) << "All jobs are done... " << std::endl;
  lock.unlock();
}

void SysManager::waitForAllResults(AIGraph* graph)
{
  std::unique_lock<std::mutex> lock(_mtx);
  _cv.wait(lock, [graph]() { return graph->getNumJobs() <= 0; });
  LOG_X(DEBUG) << "All jobs are done for graph :  " << graph->getName() << std::endl;
  lock.unlock();
}

//void SysManager::postExecRoutine(AIJob* curJob, std::vector<AKS::DataDescriptor*>& outputs, int workerID)
void SysManager::postExecRoutine(AIJob* curJob, std::vector<vart::TensorBuffer*>& outputs, int workerID)
{
  std::chrono::time_point<std::chrono::steady_clock> t0, t1;
  auto curNode = curJob->getCurrentNode();
  auto curGraphIA = curJob->getCurrentGraphInAction();
  auto curJobID = curGraphIA->getJobID();

  //. Create alive node for curNode and push it to curGraphIA
  auto curAliveNode = new AKS::AliveNode(std::move(outputs));
  curGraphIA->insertAliveNode(curNode, curAliveNode);

  // If it is not last node, decrement in_degree of each nextnodes 
  // and if any reaches zero, enqueue its Job
  std::vector<AIJob*> nextJobs;
  auto nextNodes = curNode->getNextNodes();
  if(!nextNodes.empty()) {
    nextJobs.reserve(nextNodes.size());
    for(auto nextNode: nextNodes) {
      auto nextNodeID = nextNode->getIndex();

      curGraphIA->degreeMtx.lock();
      int newInRefCount = curGraphIA->decInRefCount(nextNodeID);
      if(newInRefCount == 0) {
        nextJobs.push_back(new AKS::AIJob(curGraphIA, nextNode));
      }
      curGraphIA->degreeMtx.unlock();
    }
    if(_TRACE_) t0 = std::chrono::steady_clock::now();
    for(auto nextJob: nextJobs) {
      auto nextNodeQueue = getQueue(nextJob->getCurrentNode()->getKernel()->getName());
      nextNodeQueue->push(nextJob);
    }
    if(_TRACE_) t1 = std::chrono::steady_clock::now();

    // Save the trace points
    if(_TRACE_) {
      if(workerID == -1) {
        _mtx.lock();
        _logs.back().addEntry("tpush", t0, t1, curJobID, 'X'); 
        _mtx.unlock();
      }
      else {
        _logs[workerID].addEntry("push", t0, t1, curJobID, 'X'); 
      }
    }
  }

  //. TODO : @abidk : memory optimization as follows
  //. Decrement OutRefCount of each of parent Node
  //. If anyone reaches zero, delete its alive Node (along with its DataDescriptors)

  //. Delete the job
  //. If this node is the last, cleanUp graphInAction also
  //. TODO : @abidk : Extend it for multiple OPs
  if(!nextNodes.empty()) {
    delete curJob; curJob = nullptr;
  } else {
    auto curGraph = curGraphIA->getGraph();
    curGraphIA->setFinished(curJob);
    delete curJob; curJob = nullptr;
    auto curJobID = curGraphIA->getJobID();
    delete curGraphIA; curGraphIA = nullptr;
    if(_TRACE_) {
      auto _ts_ = std::chrono::steady_clock::now();
      _logs[workerID].addEntry("Done", _ts_, _ts_, curJobID, 'F'); 
    }
    // Notify enqueueJob()
    int count = --_numJobs;
    if(_max_concurrent_jobs > 0 && count <= _max_concurrent_jobs) {
      _cv.notify_all();
    }
    // Notify waitForAllJobs
    if(count <= 0) {
      _cv.notify_all();
    }
    // Notify waitForAllJobs(graph)
    auto graphJobCount = curGraph->getNumJobs();
    if(graphJobCount <= 0) {
      _cv.notify_all();
    }
  }
}

// Thread based wait
void SysManager::waitRoutine(AIJob* curJob)
{
  std::chrono::time_point<std::chrono::steady_clock> t0, t1;
  auto curNode      = curJob->getCurrentNode();
  auto kernelDef    = curNode->getKernel();
  auto kernelHandle = kernelDef->getKernelHandle();
  auto curJobID     = curJob->getCurrentGraphInAction()->getJobID();
  auto rtJID        = curJob->getJobID();
  auto nodeParam    = curNode->getOpParam(kernelDef);

  // Wait for the job to finish and do postExecRoutine
  if(_TRACE_) t0 = std::chrono::steady_clock::now();
  kernelHandle->wait(rtJID, nodeParam);
  if(_TRACE_) t1 = std::chrono::steady_clock::now();
  kernelDef->decNumExistingJobs();

  auto outputs = curJob->getOutputs();
  postExecRoutine(curJob, outputs, -1 /*curJob->getWorkerID()*/);

  if(_TRACE_) {
    _mtx.lock();
    _logs.back().addEntry(curNode->getName(), t0, t1, curJobID, 'T'); 
    _mtx.unlock();
  }
  kernelDef->incNumWaitThreads();
  _waitCV.notify_all();
}

// waitQ based wait
// void SysManager::waitRoutine(int workerID)
// {
//   std::chrono::time_point<std::chrono::steady_clock> t0, t1, t2, t3;
//   std::chrono::duration<float> total{0};
//   Queue<AIJob*>* qin = &_waitQueue;
//   int totalJobs = 0;
//   auto t_start = std::chrono::steady_clock::now();
//   while(true) {
//     ///. # Pop the job from qin
//     AKS::AIJob* curJob = nullptr;
//     if(_TRACE_) t0 = std::chrono::steady_clock::now();
//     bool status = qin->pop(curJob);
//     if(_TRACE_) t1 = std::chrono::steady_clock::now();
//     if(!status || !curJob) break;
// 
//     auto curNode      = curJob->getCurrentNode();
//     auto kernelDef    = curNode->getKernel();
//     auto kernelHandle = kernelDef->getKernelHandle();
//     auto curJobID = curJob->getCurrentGraphInAction()->getJobID(); 
// 
//     // Wait for the job to finish and do postExecRoutine
//     t2 = std::chrono::steady_clock::now();
//     kernelHandle->wait(curJob->getJobID());
//     t3 = std::chrono::steady_clock::now();
//     auto outputs = curJob->getOutputs();
//     postExecRoutine(curJob, outputs, curJob->getWorkerID());
//     total += (t3-t2);
//     totalJobs++;
//     if(_TRACE_) {
//       _logs[workerID].addEntry("wPop", t0, t1-t0, curJobID, 'X'); 
//       _logs[workerID].addEntry(curNode->getName(), t2, t3-t2, curJobID, 'T'); 
//     }
//   }
//   auto t_end = std::chrono::steady_clock::now();
//   auto t_dur = std::chrono::duration<float>{t_end - t_start};
//   _mtx.lock();
//   std::cout << "Wait Worker: " << workerID << " - ";
//   std::cout << "Total time (s) : " << std::chrono::duration<float>{total}.count() << ", ";
//   std::cout << "Total jobs : " << totalJobs << ", ";
//   std::cout << "Max. FPS possible: " << totalJobs/std::chrono::duration<float>{total}.count() << ", ";
//   std::cout << "Utilization : " << total.count() * 100.0/t_dur.count() << "%" << std::endl;
//   _mtx.unlock();
// }

void SysManager::report(AIGraph* graph)
{
  if(!graph) return;

  //std::cout << "\n[AKS] Reporting for graph: " << graph->getName() << std::endl;
  /// Get all its nodes
  for (auto node : graph->getNodesVector()) {
    //std::cout << "  Reporting for node: " << node->getName() << std::endl;
    /// Get kernel for this node
    std::map<AKS::KernelDef*, AKS::NodeParams*> & kernels = node->getKernels();
    for (const auto kernel : kernels) {
      //std::cout << "    Reporting for kernel: " << kernel.first->getName() << std::endl;
      auto kernelHandle  = kernel.first->getKernelHandle();
      if(kernelHandle){
        /// Get Kernel runtime params
        AKS::NodeParams *nodeParams = kernel.second;
        //call report now
        kernelHandle->report(nodeParams);
      }
    }//kernels
  }//nodes
}

void SysManager::saveTrace(const std::string& filename) {
  std::ofstream f(filename);

  f << "[\n" ;   // starting
  
  // dump each dataPoint
  int workerID=0;
  int nLogs = _logs.size();
  for(int i=0; i<nLogs; ++i) {
    auto name = _logs[i].name.empty() ? to_string(i) : _logs[i].name;
    int nEntries = _logs[i].tracePoints.size();
    for(int j=0; j<nEntries; ++j) {
      auto& t = _logs[i].tracePoints[j];
      f << t.getTraceFormat(workerID, name, _t_start); 
      if(i!=nLogs-1 || j!=nEntries-1) f << ",\n";
      else f << "\n";
    }
    ++workerID;
  }
  f << "]";
  f.close();
}

void SysManager::printPerfStats() {
  // First, get the latency for first image
  if(!_TRACE_) return;
  const WorkerLog& main_thread_log = _logs.at(0);
  const TraceInfo& firstEnq = main_thread_log.tracePoints.at(0);

  std::chrono::duration<float> dur;
  bool found = false;
  for(auto& log: _logs) {
    for(auto &tpoint: log.tracePoints) {
      if(tpoint.jobID == 0 && tpoint.ph == 'F') {
        dur = tpoint.ts - firstEnq.ts;
        found = true;
        break;
      }
    }
    if(found) break;
  }
        
  LOG_X(INFO) << "Latency for first image : " 
    << std::chrono::duration<float, std::milli>{dur}.count() << " ms" << std::endl;

  // get the node-wise latencies
  std::cout << std::endl;
  std::cout << "Node-wise Average Latency (ms): \n";
  std::cout << "================================\n";

  using tp = std::chrono::time_point<std::chrono::steady_clock>;
  using sync_latency = std::pair<int, std::chrono::duration<float>>; // [njobs, dur]
  using async_latency = std::pair<tp, tp>; // [tstart, tend]

  std::map<std::string, sync_latency> lat; // {nodeName -> [nJobs, dur]}

  // first go through non-wait kernels
  for(auto& log: _logs) {
    for(auto& tinfo: log.tracePoints) {
      if(tinfo.ph == 'K') {
        if(lat.find(tinfo.name) == lat.end()) {
          lat[tinfo.name] = std::make_pair(0, std::chrono::seconds{0});
        }
        sync_latency& t = lat[tinfo.name];
        t.first++;
        t.second += (tinfo.te - tinfo.ts);
      }
    }
  }

  for(auto& t: lat) {
    LOG_X(INFO) << t.first << " : " 
      << "Total (ms) : " << std::chrono::duration<float, std::milli>{t.second.second}.count()
      << ", #Jobs : " << t.second.first
      << ", Avg. (ms) : " << std::chrono::duration<float, std::milli>{t.second.second}.count() / t.second.first 
      << std::endl;
  }
  lat.clear();

  // Now go through wait kernels
  // First, collect ts&te for all the images and all async nodes in each image
  using Key = std::pair<std::string, uint64_t>;
  std::map<Key, async_latency> lat_async; // {[nodeName, jid] -> [tstart, tend]}
  for(auto& log: _logs) {
    for(auto& tinfo: log.tracePoints) {
      if(tinfo.ph == 'W' || tinfo.ph == 'T') {
        Key key = {tinfo.name, tinfo.jobID};
        if(lat_async.find(key) == lat_async.end()) {
          lat_async[key] = std::make_pair(_t_start, _t_start);
        }
        async_latency& t = lat_async[key];
        if(tinfo.ph == 'W') t.first = tinfo.ts;
        else if(tinfo.ph == 'T') t.second = tinfo.te;
      }
    }
  }

  // Now, accumulate latencies of each node type
  for(auto& item: lat_async) {
    auto& key = item.first; // [nodename, jid]
    auto& val = item.second; // [ts, te]
    auto dur = val.second - val.first;
    if(lat.find(key.first) == lat.end()) {
      lat[key.first] = std::make_pair(0, std::chrono::seconds{0});
    }
    sync_latency& t = lat[key.first];
    t.first++;
    t.second += dur;
  }

  for(auto& t: lat) {
    LOG_X(INFO) << t.first << " : " 
      << "Total (ms) : " << std::chrono::duration<float, std::milli>{t.second.second}.count()
      << ", #Jobs : " << t.second.first
      << ", Avg. (ms) : " << std::chrono::duration<float, std::milli>{t.second.second}.count() / t.second.first 
      << std::endl;
  }
  std::cout << "================================\n\n";

  lat.clear();
  lat_async.clear();
}

void SysManager::resetTimer() {
  _t_start = std::chrono::steady_clock::now();
}
