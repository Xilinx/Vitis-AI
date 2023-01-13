

#pragma once
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <dirent.h>
#include <xrt.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>



typedef struct AcceleratorHandle_t
{
    xrt::kernel kernel;
    xrt::device device;
    xrt::run runner;
    xrt::bo input;
    xrt::bo dummy1;
	xrt::bo dummy2;
    xrt::bo output;
    xrt::bo params;
    uint64_t dpu_conf_out_phy_addr;
	uint64_t dpu_box_out_phy_addr;
    void *input_m;
    void *output_m;
    void *dummy1_m;
	void *dummy2_m; 
    void *params_m;
} AcceleratorHandle;




AcceleratorHandle* pp_kernel_init();

int preprocess(unsigned char *in_image_data,
    int img_ht, int img_wt, int out_ht, int out_wt, uint64_t dpu_input_buf_addr,AcceleratorHandle* preproc_handle);

AcceleratorHandle* hw_sort_init(const short *&fx_pror);

int runHWSort(AcceleratorHandle* postproc_handle, short *&nms_out);



