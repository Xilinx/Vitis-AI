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
// Kernel Functions Implementation

#include <vector>
#include <fstream>

#include <aks/AksKernelBase.h>
#include <aks/AksDataDescriptor.h>
#include <aks/AksNodeParams.h>
#include <aks/AksLogger.h>

#include <dpu/dpu_runner.hpp>

class DPUCADX8GBase: public AKS::KernelBase {
  public:
    std::unique_ptr<vitis::ai::DpuRunner> runner;
    bool isExecAsync() { return true; }
    void nodeInit(AKS::NodeParams*);
    int exec_async (
        std::vector<AKS::DataDescriptor *> &in, 
        std::vector<AKS::DataDescriptor *> &out, 
        AKS::NodeParams* params, 
        AKS::DynamicParamValues* dynParams);
    int getNumCUs(void);
    void wait (int, AKS::NodeParams*);
  private:
    // Filled at node init
    std::map<std::string, vitis::ai::Tensor*> _ioTensors;
    // Updated for each frame
    std::map<int, std::vector<vitis::ai::CpuFlatTensorBuffer*>> _ftbs;
};

extern "C" {

  AKS::KernelBase* getKernel (AKS::NodeParams* params) {
    /// Create kernel object
    DPUCADX8GBase * kbase = new DPUCADX8GBase();
    return kbase;
  }

} // extern C

void DPUCADX8GBase::nodeInit(AKS::NodeParams* params)
{
  auto rundir = params->_stringParams["vitis_rundir"];
  auto xclbin = params->_stringParams["xclbin"];
  int batch_size = params->_intParams.find("batch_size") == params->_intParams.end() ?
    1 : params->_intParams["batch_size"];
  const char* libpath = std::getenv("LIBXDNN_PATH");

  /// Number of FPGAs/CUs
  bool acquireCU;
  if (params->_intParams.find("acquire_cu") != params->_intParams.end()) {
    /// Just acquire 1 CU
    if ((params->_intParams["acquire_cu"] > 1) || 
        (params->_intParams["acquire_cu"] < 0)) {
      LOG(WARNING) << "Invalid \"acquire_cu\" value." << std::endl; 
      LOG(WARNING) << "Default to 0." << std::endl;
      acquireCU = false;
    } else {
      acquireCU = params->_intParams["acquire_cu"] ? true : false;
    }
  }

  int numfpga;
  if (acquireCU) {
    numfpga = 1;
  } else if(getenv("NUM_FPGA")) {
    /// Override #FPGAs from env variable
    acquireCU = false;
    numfpga = std::atoi(std::getenv("NUM_FPGA"));
  } else if (params->_intParams.find("num_fpga") != params->_intParams.end()) {
    /// Use #FPGAs mentioned in graph node
    acquireCU = false;
    numfpga = params->_intParams["num_fpga"];
  }

  /// Create meta file on the go
  std::ofstream metaFp;
  std::string metaPath = rundir + std::string("/meta.json");
  metaFp.open(metaPath);
  metaFp << "{" << std::endl;
  metaFp << "\t" << "\"target\": \"xdnn\"" << "," << std::endl;;
  metaFp << "\t" << "\"filename\": \"\"" << ","  << std::endl;;
  metaFp << "\t" << "\"kernel\": \"xdnn\"" << ","  << std::endl;;
  metaFp << "\t" << "\"config_file\": \"\"" << ","  << std::endl;;
  metaFp << "\t" << "\"lib\": " << "\"" << libpath << "\"" << ","  << std::endl;;
  metaFp << "\t" << "\"xclbin\": " << "\"" << xclbin << "\"" << ","  << std::endl;;
  metaFp << "\t" << "\"acquire_cu\": " << std::boolalpha << "\"" << acquireCU << "\"" << ","  << std::endl;;
  metaFp << "\t" << "\"num_fpga\": " << numfpga << std::endl;;
  metaFp << "}" << std::endl;
  metaFp.close();

  auto runners = vitis::ai::DpuRunner::create_dpu_runner(rundir); 

  auto outputTensors = runners[0]->get_output_tensors();
  for (auto & oTensor: outputTensors) {
    /// Get output dims
    auto out_dims = oTensor->get_dims();
    out_dims[0] = batch_size;
    /// Create output tensors for runner
    int outSize = oTensor->get_element_num() / oTensor->get_dim_size(0);
    auto outTensor = new vitis::ai::Tensor(oTensor->get_name(), out_dims, oTensor->get_data_type());
    _ioTensors[oTensor->get_name()] = outTensor;
  }

  auto inputTensors = runners[0]->get_input_tensors();
  for (auto & iTensor: inputTensors) {
    /// Get input dims
    auto in_dims = iTensor->get_dims();
    in_dims[0] = batch_size;
    /// Create input tensors for runner
    auto inTensor = new vitis::ai::Tensor(iTensor->get_name(), in_dims, iTensor->get_data_type());
    _ioTensors[iTensor->get_name()] = inTensor;
  }

  this->runner = std::move(runners[0]);
}

int DPUCADX8GBase::getNumCUs(void)
{
  return 1;
}

int DPUCADX8GBase::exec_async (
    vector<AKS::DataDescriptor *>& in, vector<AKS::DataDescriptor *>& out, 
    AKS::NodeParams* params, AKS::DynamicParamValues* dynParams) 
{ 
  DPUCADX8GBase* kbase = this;
  vitis::ai::DpuRunner* runner = kbase->runner.get();

  std::vector<vitis::ai::CpuFlatTensorBuffer*> ftb;
  std::vector<vitis::ai::TensorBuffer*> inputsPtr, outputsPtr;

  auto outputTensors = runner->get_output_tensors();
  int outDataIdx = 0;
  for (auto & oTensor: outputTensors) {
    /// Create output tensors for runner
    auto outTensor = _ioTensors[oTensor->get_name()];
    /// Get output dims
    auto out_dims = outTensor->get_dims();
    /// Create output descriptors
    out.push_back(new AKS::DataDescriptor(out_dims, AKS::DataType::FLOAT32));
    auto cpuTenBuff = new vitis::ai::CpuFlatTensorBuffer(out[outDataIdx]->data(), outTensor); 
    outputsPtr.push_back(cpuTenBuff); 
    ftb.push_back(cpuTenBuff);
    outDataIdx++;
  }

  auto inputTensors = runner->get_input_tensors();
  int inDataIdx = 0;
  for (auto & iTensor: inputTensors) {
    /// Create input tensors for runner
    auto inTensor = _ioTensors[iTensor->get_name()];
    auto cpuTenBuff = new vitis::ai::CpuFlatTensorBuffer(in[inDataIdx]->data(), inTensor);
    inputsPtr.push_back(cpuTenBuff); 
    ftb.push_back(cpuTenBuff);
    inDataIdx++;
  }

  auto job_id = runner->execute_async(inputsPtr, outputsPtr);

  /// Delete previously saved flatTensorbuffers
  auto t = _ftbs.find(job_id.first);
  if (t != _ftbs.end()){
    for(int i = 0; i < t->second.size(); ++i){
      delete t->second[i];
    }
  }

  /// Save current flatTensorbuffers
  _ftbs[job_id.first] = std::move(ftb);
  return job_id.first;
}

void DPUCADX8GBase::wait (int jobId, AKS::NodeParams* params) 
{
  DPUCADX8GBase* kbase = this;
  auto runner = kbase->runner.get();
  runner->wait(jobId, -1);
}
