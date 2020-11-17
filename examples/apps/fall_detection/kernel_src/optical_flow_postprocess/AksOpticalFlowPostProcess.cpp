#include <iostream>
#include <stdint.h>
#include <vector>

#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <aks/AksKernelBase.h>
#include <aks/AksDataDescriptor.h>
#include <aks/AksNodeParams.h>

class OpticalFlowPostProcess : public AKS::KernelBase
{
  public:
    int id =0;
    int exec_async (
           std::vector<AKS::DataDescriptor*> &in, 
           std::vector<AKS::DataDescriptor*> &out, 
           AKS::NodeParams* nodeParams,
           AKS::DynamicParamValues* dynParams);
};


void boundPixels(AKS::DataDescriptor* src, cv::Mat& dst, int bound) {
  auto shape = src->getShape();
  int rows = shape[0];
  int cols = shape[1];
  float* srcPtr = static_cast<float*>(src->data());
  for (int i=0; i<rows; ++i) {
    for (int j=0; j<cols; ++j) {
      float x = srcPtr[i*cols+j];
      x = (x + bound) * (255.0 / (2 * bound));
      int y = cvRound(x);
      dst.at<uint8_t>(i, j) = (y>255 ? 255 : (y<0 ? 0 : y));
    }
  }
}


void mat2DD(const cv::Mat &src, int8_t* dst) {
  // Gray scale cv::Mat --> Gray scale DataDescriptor
  int channels = src.channels();
  int rows = src.rows;
  int cols = src.cols;
  for (int k=0; k<channels; k++) {
    for (int i=0; i<rows; i++) {
      for (int j=0; j<cols; j++) {
        dst[(k*rows*cols) + (i*cols) + j] = src.at<cv::Vec<uint8_t, 2>>(i,j)[k] - 128;
      }
    }
  }
}


extern "C" { // Add this to make this available for python bindings and 

AKS::KernelBase* getKernel (AKS::NodeParams *params)
{
  return new OpticalFlowPostProcess();
}


int OpticalFlowPostProcess::exec_async (
                      std::vector<AKS::DataDescriptor*> &in, 
                      std::vector<AKS::DataDescriptor*> &out, 
                      AKS::NodeParams* nodeParams,
                      AKS::DynamicParamValues* dynParams)
{
    // in[0] contains flowx data
    // in[1] contains flowy data

    // std::cout << "[DBG] OpticalFlowPostProcess: running now ... " << std::endl;
    auto shape = in[0]->getShape();
    int rows = shape[0];
    int cols = shape[1];

    int bound = nodeParams->_intParams["bound"];
    cv::Mat boundedFlowX(rows, cols, CV_8UC1);
    cv::Mat boundedFlowY(rows, cols, CV_8UC1);

    float* flowxData = static_cast<float*>(in[0]->data());
    float* flowyData = static_cast<float*>(in[1]->data());

    boundPixels(in[0], boundedFlowX, bound);
    boundPixels(in[1], boundedFlowY, bound);

    cv::Mat flow[2] = { boundedFlowX, boundedFlowY };
    cv::Mat bounded(rows, cols, CV_8UC(2));
    cv::merge(flow, 2, bounded);

    AKS::DataDescriptor *flowDD = new AKS::DataDescriptor(
        { 1, 2, rows, cols }, AKS::DataType::INT8);
    int8_t* flowData = static_cast<int8_t*>(flowDD->data());
    mat2DD(bounded, flowData);
    out.push_back(flowDD);

    std::string output_folder = \
      nodeParams->_stringParams.find("visualize") == nodeParams->_stringParams.end() ?
        "" : nodeParams->_stringParams["visualize"];
    if (!output_folder.empty()) {
        boost::filesystem::path p(dynParams->imagePaths.front());
        std::string filename = p.filename().string();
        std::string folder = p.parent_path().filename().string();
        std::string output_path = output_folder + "/" + folder;
        boost::filesystem::create_directories(output_path);
        std::string tmp = "_" + filename;
        cv::imwrite(output_path + "/flow_x" + tmp, boundedFlowX);
        cv::imwrite(output_path + "/flow_y" + tmp, boundedFlowY);
    }

    // std::cout << "[DBG] OpticalFlowPostProcess: Done!" << std::endl << std::endl;
    return -1; // No wait
}

} //extern "C"
