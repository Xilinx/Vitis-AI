#include <iostream>
#include <vector>
#include <assert.h>

#include <aks/AksKernelBase.h>
#include <aks/AksDataDescriptor.h>
#include <aks/AksNodeParams.h>

class OFClassificationPreProcess : public AKS::KernelBase
{
  public:
    int exec_async (
           std::vector<AKS::DataDescriptor*> &in, 
           std::vector<AKS::DataDescriptor*> &out, 
           AKS::NodeParams* nodeParams,
           AKS::DynamicParamValues* dynParams);
};

extern "C" { /// Add this to make this available for python bindings and 

AKS::KernelBase* getKernel (AKS::NodeParams *params)
{
  return new OFClassificationPreProcess();
}


int OFClassificationPreProcess::exec_async (
           std::vector<AKS::DataDescriptor*> &in, 
           std::vector<AKS::DataDescriptor*> &out, 
           AKS::NodeParams* nodeParams,
           AKS::DynamicParamValues* dynParams)
{
  // std::cout << "[DBG] OFClassificationPreProcess: running now ... " << std::endl;

  uint8_t numDD = in.size();
  assert(numDD > 0);

  auto dd = in[0];
  auto shape = dd->getShape();
  int channels = shape[1];
  int rows = shape[2];
  int cols = shape[3];

  AKS::DataDescriptor *outDD = new AKS::DataDescriptor(
    {1, channels*numDD, rows, cols}, AKS::DataType::FLOAT32);
  float* outData = outDD->data<float>();
  // merge at Channels
  for (int i=0; i<numDD*channels; i++) {
    int8_t* data = in[i/channels]->data<int8_t>();
    for (int j=0; j<rows; j++) {
      for (int k=0; k<cols; k++) {
          outData[i*rows*cols + j*cols + k] = (float)data[(i%channels)*rows*cols + j*cols + k];
      }
    }
  }
  out.push_back(outDD);
  // std::cout << "[DBG] OFClassificationPreProcess: Done!" << std::endl << std::endl;
  return -1; /// No wait
}

}//extern "C"
