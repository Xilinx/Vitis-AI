//
// SPDX-License-Identifier: BSD-3-CLAUSE
//
// (C) Copyright 2018, Xilinx, Inc.
//

#include "interface.h"

//# Classify infernce on xfdnn                                
int main(int argc, char** argv)
{
    
    vector<string> img_path;
    string xclbin, netCfgFile, quantCfgFile, labelFile;
    int height, width, out_h, out_w, out_d, batch_sz;
    string dir_path_s, dataDir;
    int err_status=0;
    int ret=0;
    
	//# Arguments parser
    ProcessArgs(argc, argv, xclbin, dataDir, netCfgFile, quantCfgFile,
    labelFile,batch_sz, dir_path_s, img_path);

    //# create classifion Object and initialization 
    classifycpp classify_obj(xclbin,dataDir,netCfgFile,quantCfgFile,labelFile,batch_sz,dir_path_s,img_path);
	
    
    cout << PFX << " Running xdnn_infer_init() .......... " << endl;

    //# Create FPGA handle and initialize script executor
	classify_obj.xdnn_infer_init();

    //# Run on FPGA
    struct timeval start, end;
    double total_latency=0.0, total_hw=0.0, total_sw=0.0;
    //# xdnn_infer_preprocess do image read and subtraction mean from it
    cout<< PFX <<" Prepare input "<<endl;
    
    gettimeofday(&start, 0);
    err_status = classify_obj.xdnn_infer_preprocess();
    if(err_status != 0)
    {
        cout <<PFX<< " Failed to prepare input" << endl;
    }
    gettimeofday(&end, 0);
    cout << PFX <<" Profile pre process : " << (end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec) <<" usec" << endl;

    //# Run inference       
    int exec_count=0;
    double latency_num=0.0;
    gettimeofday(&start, 0);
    err_status = classify_obj.xdnn_infer_Execute();

    if(err_status != 0)
    {
        cout <<PFX <<" Failed to execute inference" << endl;
    }

    gettimeofday(&end, 0);

    latency_num = latency_num + (end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec);
    cout << PFX <<" Profile latency number for xfdnnExecute : " << latency_num <<" usec"<< endl;


    // run xdnn_infer_postprocess on CPU,FC & Softmax
    gettimeofday(&start, 0);
    err_status = classify_obj.xdnn_infer_postprocess();
    if(err_status != 0)
    {
        cout <<PFX <<" Failed to execute FC and Softmax" << endl;
    }
    gettimeofday(&end, 0);
    cout << PFX << " Profile postprocess sw layers : " << (end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec) <<" usec"<<endl;
    return 0;
}
