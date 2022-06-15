/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 i* Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <fstream>
#include <iterator>
#include <algorithm>
#include <typeinfo>

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>


#include <bits/stdc++.h>
#include <vector>

#include <aks/AksTensorBuffer.h>
#include <aks/AksKernelBase.h>
#include <aks/AksNodeParams.h>

#include <cstdio>
#include <future>
#include <unistd.h>
#include <chrono>
#include <mutex>

#include "concurrentqueue.h"


using namespace std;
using namespace std::chrono;

vector<vector<int64_t>> get_out_shape( tvm::runtime::Module mod, DLTensor* in_data)
{
  tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
  set_input("data", in_data);
  tvm::runtime::PackedFunc run = mod.GetFunction("run");
  run();
  tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
  tvm::runtime::PackedFunc get_num_outputs_func = mod.GetFunction("get_num_outputs");
  int get_num_outputs = get_num_outputs_func();
  list<vector<int64_t>> listOfVec;
  for(int i=0;i<get_num_outputs;i++)
  {
   tvm::runtime::NDArray res = get_output(i);
   tvm::runtime::ShapeTuple out_shape = res.Shape();
   size_t ndim = out_shape.get()->size;
   int64_t shape_arr[ndim];
   std::copy(out_shape.begin(), out_shape.end(), shape_arr);
   std::vector <int64_t> out_shape_vec(shape_arr, shape_arr + sizeof(shape_arr) / sizeof(int64_t));
   listOfVec.push_back(out_shape_vec);
  }
  vector<vector<int64_t>> v(listOfVec.begin(), listOfVec.end());
  return v;
}

struct TVMNodeObject
{
  moodycamel::ConcurrentQueue<tvm::runtime::Module> _runnerq;
  std::string lib;
  std::string input_name;
  DLDevice ctx{kDLCPU, 0};
  std::vector<DLTensor*> in_data;
  std::vector<std::vector<DLTensor*>> out_data;
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
  int in_ndim = 4;
  std::atomic<unsigned int> core_id {0};
  unsigned int core_count;
  vector<vector<int64_t>> out_shape;
 
  
};

class TvmKernelBase : public AKS::KernelBase
{
public:
  int exec_async(
      std::vector<vart::TensorBuffer *> &in,
      std::vector<vart::TensorBuffer *> &out,
      AKS::NodeParams *params,
      AKS::DynamicParamValues *dynParams);
      void nodeInit(AKS::NodeParams *);
private:
  std::map<AKS::NodeParams *, TVMNodeObject> nodes;
  string _device = "CPU";
};

extern "C"
{
  AKS::KernelBase *getKernel(AKS::NodeParams *params)
  {
    return new TvmKernelBase();
  }
} // extern C

void TvmKernelBase::nodeInit(AKS::NodeParams *params)
{
  nodes.emplace(std::piecewise_construct,
                       std::forward_as_tuple(params),
                       std::forward_as_tuple());

  auto num_runners = params->hasKey<int>("num_runners") ? params->getValue<int>("num_runners") : 1;
  nodes[params].core_count = num_runners;
  auto lib = params->_stringParams["lib"];
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
  int in_ndim = 4;

  for (int i = 0; i < num_runners; i++)
  {
    auto input_name = params->_stringParams["input_name"];
    // load network
    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(lib);
    DLDevice ctx{kDLCPU, 0};
    tvm::runtime::Module mod = mod_factory.GetFunction("default")(ctx);
    nodes[params]._runnerq.enqueue(mod);
    nodes[params].lib = lib;
    nodes[params].input_name = input_name;

    auto indimIter = params->_intVectorParams.find("in_dim");
    int64_t in_dim[4];
    in_dim[0] = indimIter->second[0]; //N
    in_dim[1] = indimIter->second[1]; //C;
    in_dim[2] = indimIter->second[2]; //H;
    in_dim[3] = indimIter->second[3]; //W;

    vector<int64_t> in_dim_vec{in_dim[0], in_dim[1], in_dim[2], in_dim[3]};
    DLTensor* dl_data_in;
    nodes[params].in_data.push_back(dl_data_in);
    TVMArrayAlloc(in_dim, in_ndim, dtype_code,
		     dtype_bits, dtype_lanes,
		     device_type, device_id, &(nodes[params].in_data[i]));

    if (i == 0) //get output shapes from the first runner
      nodes[params].out_shape = get_out_shape(mod, nodes[params].in_data[0]);
      
    vector<DLTensor*>dl_data_out_vec;
    for (auto k = 0; k<nodes[params].out_shape.size();k++){   
      DLTensor* dl_data_out;
      dl_data_out_vec.push_back(dl_data_out);
    }

      for (auto j = 0; j<nodes[params].out_shape.size();j++){   
      int64_t shape_arr[nodes[params].out_shape[j].size()];
      std::copy(nodes[params].out_shape[j].begin(), nodes[params].out_shape[j].end(), shape_arr);
    
      nodes[params].out_data.push_back(dl_data_out_vec);
      TVMArrayAlloc(shape_arr, nodes[params].out_shape[j].size(), dtype_code, 
                    dtype_bits, dtype_lanes, 
                    device_type, device_id, 
                    &(nodes[params].out_data[i][j]));

  }
  }
  nodes[params].core_count = num_runners;
}

int TvmKernelBase::exec_async(
    vector<vart::TensorBuffer*> &in, vector<vart::TensorBuffer*> &out,
    AKS::NodeParams *params, AKS::DynamicParamValues *dynParams)
{
  auto &curNode = nodes[params];
  unsigned int tmpID = curNode.core_id++;
  unsigned int runnerID = tmpID % curNode.core_count;

  tvm::runtime::Module curRunner;
  while (!curNode._runnerq.try_dequeue(curRunner)) {}

  auto indimIter = params->_intVectorParams.find("in_dim");
  int64_t in_dim[4];
  in_dim[0] = indimIter->second[0]; //N
  in_dim[1] = indimIter->second[1]; //C;
  in_dim[2] = indimIter->second[2]; //H;
  in_dim[3] = indimIter->second[3]; //W;
  vector<int64_t> in_dim_vec{in_dim[0], in_dim[1], in_dim[2], in_dim[3]};

  for (int i = 0; i < 1; ++i)
  {
    
    float *inData= reinterpret_cast<float*>(in[i]->data().first);
    tvm::runtime::PackedFunc set_input = curRunner.GetFunction("set_input");
    int64_t size = std::accumulate(in_dim_vec.begin(), in_dim_vec.end(), 1, std::multiplies<int64_t>());
    curNode.in_data[runnerID]->data = inData;
    set_input(curNode.input_name, curNode.in_data[runnerID]);
  }
  
  tvm::runtime::PackedFunc run = curRunner.GetFunction("run");
  run();

  for (int i = 0; i < curNode.out_shape.size(); ++i)
  {
    vector<int> out_shape(curNode.out_shape[i].begin(), curNode.out_shape[i].end());
    int64_t shape_arr[curNode.out_shape[i].size()];
    std::copy(curNode.out_shape[i].begin(), curNode.out_shape[i].end(), shape_arr);
    std::string tensorName ("pre-output");
    tensorName.append(to_string(i));
    tensorName.append(to_string(runnerID));
    AKS::AksTensorBuffer * outDD = new AKS::AksTensorBuffer(
                                     xir::Tensor::create(
                                      tensorName, //check name
                                      out_shape, xir::create_data_type<float>()
                                   ));
    
    out.push_back(outDD);
    float *outData = reinterpret_cast<float*>(outDD->data().first);
    curNode.out_data[runnerID][i]->data = outData;
    tvm::runtime::PackedFunc get_output = curRunner.GetFunction("get_output");
    get_output(i, curNode.out_data[runnerID][i]);

  }
  
  curNode._runnerq.enqueue(std::move(curRunner));

  return -1;
}

