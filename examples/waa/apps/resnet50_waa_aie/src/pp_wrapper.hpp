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

#include <sys/time.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include "adf/adf_api/XRTConfig.h"
#include <common/xf_aie_sw_utils.hpp>
#include <common/xfcvDataMovers.h>
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_graph.h"
#include "xrt/xrt_bo.h"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_graph.h"

using namespace adf;

typedef struct AcceleratorHandle_t
{
	
	xrtBufferHandle src_hndl1;
	xrtBufferHandle src_hndl2;
	xrtBufferHandle dst_hndl;
	xrtGraphHandle xrt_resize_norm_ghdl;

	void* srcData1;
	void* srcData2;
	void* dstData;

    xF::xfcvDataMovers<xF::TILER, uint8_t, TILE_HEIGHT_IN, TILE_WIDTH_IN, 16> *tiler;
    xF::xfcvDataMovers<xF::STITCHER, uint8_t, TILE_HEIGHT_OUT, TILE_WIDTH_OUT, 16> *stitcher;
    xF::xfcvDataMoverParams params;

} AcceleratorHandle;

template<int INPUT_BW=8, int INPUT_FBITS=0>
void get_alpha_beta(float mean[4], float scale[4], char alpha[4], char beta[4])
{
  for(int i=0; i<4;i++)
  {
     if(i<3){
        float a_v = mean[i]*(1<<INPUT_FBITS);
        float b_v = (scale[i])*(1<<4);

	alpha[i] = (unsigned char)a_v;
	beta[i] = (char)b_v;
     
        assert((a_v < (1<<8)) &&
		           "alpha values exceeds 8 bit precison");
        assert((b_v < (1<<8)) &&
		           "beta values exceeds 8 bit precison");
     }
     else
     {
	alpha[i] = 0;
	beta[i] = 0;
     }

  }
}

int ImgOddEvenFlag = 1;

int preprocess(AcceleratorHandle* preprocessor,cv::Mat image1, int img_ht, int img_wt, int out_ht, int out_wt, uint64_t dpu_input_buf_addr, int no_zcpy)
{	
	int ret;
	const int in_image_size_in_bytes = image1.rows * image1.cols * 3* sizeof(char);
	size_t out_image_size_in_bytes = out_ht * out_wt * 4 *sizeof(char);
	unsigned char *in_image_data1; 
	unsigned char *in_image_data2; 
	
	if(ImgOddEvenFlag==1)
	{
		auto tiles_sz = preprocessor->tiler->host2aie_nb(preprocessor->src_hndl1, image1.size(), preprocessor->params);

		if(no_zcpy)
			preprocessor->stitcher->aie2host_nb(preprocessor->dst_hndl, cv::Size(IMAGE_HEIGHT_OUT,IMAGE_WIDTH_OUT), tiles_sz);
		else	
			preprocessor->stitcher->aie2host_nb(dpu_input_buf_addr, cv::Size(IMAGE_HEIGHT_OUT,IMAGE_WIDTH_OUT), tiles_sz);

		in_image_data2 = image1.data;
		memcpy(preprocessor->srcData2, in_image_data2,  in_image_size_in_bytes);
		ImgOddEvenFlag = 0;
	}
	else
	{
		auto tiles_sz = preprocessor->tiler->host2aie_nb(preprocessor->src_hndl2, image1.size(), preprocessor->params);
		
		if(no_zcpy)
			preprocessor->stitcher->aie2host_nb(preprocessor->dst_hndl, cv::Size(IMAGE_HEIGHT_OUT,IMAGE_WIDTH_OUT), tiles_sz);
		else	
			preprocessor->stitcher->aie2host_nb(dpu_input_buf_addr, cv::Size(IMAGE_HEIGHT_OUT,IMAGE_WIDTH_OUT), tiles_sz);

		in_image_data1 = image1.data;
		memcpy(preprocessor->srcData1, in_image_data1,  in_image_size_in_bytes);
	
		ImgOddEvenFlag = 1;
	}		
	
	preprocessor->tiler->wait();
	
	preprocessor->stitcher->wait();
	
	if(no_zcpy)
	{
		int8_t *out_data =(int8_t *)dpu_input_buf_addr;
		std::memcpy(out_data, preprocessor->dstData, IMAGE_HEIGHT_OUT*IMAGE_WIDTH_OUT*3);
	}
	
	return 0;
}

AcceleratorHandle * pp_kernel_init(int out_ht, int out_wt, int no_zcpy)
{
	// Initializa device
	size_t out_image_size_in_bytes = out_ht * out_wt * 4 *sizeof(char);
	int ret;
	
	const char *xclBinName = std::getenv("XLNX_VART_FIRMWARE");
	xF::deviceInit(xclBinName);
	
	// Creating memory for full-HD image
	const int in_image_size_in_bytes = IMAGE_WIDTH_IN* IMAGE_HEIGHT_IN * 3 *sizeof(char);
	
	xrtDeviceLoadXclbinFile(xF::gpDhdl,xclBinName);
	
	xuid_t uuid;
	xrtDeviceGetXclbinUUID(xF::gpDhdl, uuid);
	
	auto tiler_accel = xrtPLKernelOpen(xF::gpDhdl,uuid,"Tiler_top");
	auto stitcher_accel = xrtPLKernelOpen(xF::gpDhdl,uuid,"stitcher_top");
	
	void* srcData1 = nullptr;
	xrtBufferHandle src_hndl1 = xrtBOAlloc(xF::gpDhdl, in_image_size_in_bytes, 0, xrtKernelArgGroupId(tiler_accel,2));
	srcData1 = xrtBOMap(src_hndl1);
	void* srcData2 = nullptr;
	xrtBufferHandle src_hndl2 = xrtBOAlloc(xF::gpDhdl, in_image_size_in_bytes, 0, xrtKernelArgGroupId(tiler_accel,2));
	srcData2 = xrtBOMap(src_hndl2);
	
	auto accel_handle = new AcceleratorHandle();

    if(no_zcpy)
	{
		void* dstData = nullptr;
		xrtBufferHandle dst_hndl = xrtBOAlloc(xF::gpDhdl, out_image_size_in_bytes, 0, xrtKernelArgGroupId(stitcher_accel,2));
		dstData = xrtBOMap(dst_hndl);
		accel_handle->dst_hndl=dst_hndl;
	    accel_handle->dstData=dstData;
	}

	float mean[4] = {104, 107, 123, 0};
	float scale[4] = {0.5, 0.5, 0.5, 0}; 

	char alpha[4];
	char beta[4];

	const int IN_BW = 8;
	const int IN_FBITS=0;
	const int OUT_BW=8;
	const int OUT_FBITS=0;

	get_alpha_beta<IN_BW, IN_FBITS>(mean, scale, alpha, beta);
	
	cv::Mat image=cv::Mat(IMAGE_HEIGHT_IN,IMAGE_WIDTH_IN,CV_8UC3);

	xF::xfcvDataMoverParams params(image.size(), cv::Size(IMAGE_HEIGHT_OUT, IMAGE_WIDTH_OUT));

	//resize_norm.init();
	auto xrt_resize_norm_ghdl = xrtGraphOpen(xF::gpDhdl, uuid, "resize_norm");
	if(!xrt_resize_norm_ghdl){
		printf("resize_norm graph open error\r\n");
	}

	accel_handle->tiler=new xF::xfcvDataMovers<xF::TILER, uint8_t, TILE_HEIGHT_IN, TILE_WIDTH_IN, 16> (0,0);
	accel_handle->stitcher= new xF::xfcvDataMovers<xF::STITCHER, uint8_t, TILE_HEIGHT_OUT, TILE_WIDTH_OUT, 16> ;
	//std::cout << "Graph init. This does nothing because CDO in boot PDI already configures AIE.\n";
	
	ret=xrtGraphUpdateRTP(xrt_resize_norm_ghdl,"resize_norm.k.in[1]",(char*)(&alpha[0]),1*sizeof(int));
	ret=xrtGraphUpdateRTP(xrt_resize_norm_ghdl,"resize_norm.k.in[2]",(char*)(&alpha[1]),1*sizeof(int));
	ret=xrtGraphUpdateRTP(xrt_resize_norm_ghdl,"resize_norm.k.in[3]",(char*)(&alpha[2]),1*sizeof(int));
	ret=xrtGraphUpdateRTP(xrt_resize_norm_ghdl,"resize_norm.k.in[4]",(char*)(&alpha[3]),1*sizeof(int));
	ret=xrtGraphUpdateRTP(xrt_resize_norm_ghdl,"resize_norm.k.in[5]",(char*)(&beta[0]),1*sizeof(int));
	ret=xrtGraphUpdateRTP(xrt_resize_norm_ghdl,"resize_norm.k.in[6]",(char*)(&beta[1]),1*sizeof(int));
	ret=xrtGraphUpdateRTP(xrt_resize_norm_ghdl,"resize_norm.k.in[7]",(char*)(&beta[2]),1*sizeof(int));
	ret=xrtGraphUpdateRTP(xrt_resize_norm_ghdl,"resize_norm.k.in[8]",(char*)(&beta[3]),1*sizeof(int));

	accel_handle->tiler->compute_metadata(image.size(), cv::Size(IMAGE_HEIGHT_OUT, IMAGE_WIDTH_OUT));

	accel_handle->src_hndl1=src_hndl1;
	accel_handle->src_hndl2=src_hndl2;
	accel_handle->srcData1=srcData1;
	accel_handle->srcData2=srcData2;
	accel_handle->xrt_resize_norm_ghdl=xrt_resize_norm_ghdl;
	accel_handle->params=params;
	
	return accel_handle;
}

int pp_graph_close(AcceleratorHandle* preprocessor)
{
	int ret;
	xrtGraphClose(preprocessor->xrt_resize_norm_ghdl);
	if(ret){
		printf("resize_norm graph close error\r\n");
	}
	return 0;
}
