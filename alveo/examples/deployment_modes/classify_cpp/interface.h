// SPDX-License-Identifier: BSD-3-CLAUSE
//
// (C) Copyright 2018, Xilinx, Inc.
//

#ifndef __INTERFACE_H__
#define __INTERFACE_H__

#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include "xblas.h"
#include "xdnn.h"
#include "xdnn_fcweightsload_cpp_infer.h"
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <getopt.h>
#include <dirent.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#define PFX "[CXDNN] "
#define EXECUTOR_MAX_BATCH_SZ 1
using namespace std;
using boost::property_tree::ptree;


	
struct x_blob_dim{
	int batch;
	int depth;
	int height;
	int width;
};
typedef x_blob_dim x_blob_dim_t;
class classifycpp{

public: 

	XBLASHandle *handle = NULL;
    XDNNScriptExecutor<float> *executor;
	unordered_map <string, vector<const float*> > input_ptrs;
	unordered_map <string, vector<float*> > output_ptrs;
	unordered_map<int, vector<vector<float>> >  fc_wb_map;
	string input_layer_name;
	string output_layer_name;
	string xclbin;
    string netCfgFile; 
    string quantCfgFile;
    string dataDir;
	string labelFile;
	vector<string> labels;
	int batch_sz;
	string dir_path;
	vector<string> image_path;
	float mean[3];
	int numClasses;
	x_blob_dim_t in_dim;
	x_blob_dim_t out_dim;
	float *input;
	float *output;
    
    //# member functions
    classifycpp();
	classifycpp(string xclbin_t, string dataDir_t, string netCfgFile_t,string quantCfgFile_t, 
	string labelFile_t, int batch_sz_t, string dir_path_t, vector<string> image_path_t);

    //# Create FPGA handle and initialize script executor
	int xdnn_infer_init(void);
	int xdnn_infer_preprocess(void);
	int xdnn_infer_postprocess(void);
	int xdnn_infer_Execute(void);
	int json_search(boost::property_tree::ptree const& pt);
	~classifycpp();
};

//# Argument parser
void ProcessArgs(int argc, char** argv, string &xclbin, string &dataDir, string &netCfgFile,
    string &quantCfgFile, string &labelFile,int &batch_sz, string &dir_path, vector<string> &image_path
    );
int prepareInputData(cv::Mat &in_frame,int img_h, int img_w, int img_depth, float *data_ptr,int *act_img_h, int *act_img_w);
void softmax(vector<float> &input);
vector<string> getLabels(string fname);

#endif /*__INTERFACE_H__*/
