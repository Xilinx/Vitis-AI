// Copyright 2020 Xilinx Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//Host code to run the kalman filter


//#include "common/xf_headers.hpp"
//#include "./xf_kalmanfilter_config.h"
#define KF_N 7
#define KF_M 4

#include <sys/time.h>

// driver includes
#include "ert.h"

// host_src includes
#include "xclhal2.h"
#include "xclbin.h"

// lowlevel common include
#include "utils.h"

#include "xkf_hw_64.h"      

#include <numeric>
#include <chrono>

//#include <gmp.h>
//#define __gmp_const const

//Creating a class containing cl objects
class KFMOT {
	
private:

	xclDeviceHandle handle;
	unsigned bo_iny_Handle;
	unsigned bo_inX0_Handle;
	unsigned bo_inU0_Handle;
	unsigned bo_inD0_Handle;
	unsigned bo_outX_Handle;
	unsigned bo_outU_Handle;
	unsigned bo_outD_Handle;
	float *inX0_ptr_xrt;
	float *inU0_ptr_xrt;
	float *inD0_ptr_xrt;
	float *outX_ptr_xrt;
	float *outU_ptr_xrt;
	float *outD_ptr_xrt;
	float *iny_ptr_xrt;
	unsigned execHandle = 0;
	void *execData = nullptr;
	int KF_INIT = 1;
	int KF_PREDICT_EN = 2;
	int KF_CORRECT_EN = 4;
	int KF_PREDICT_WRITE_STATE = 8;
	int KF_PREDICT_WRIET_CONV = 16;
	int KF_CORRECT_WRITE_STATE = 32;
	int KF_CORRECT_WRIET_CONV = 64;

public:
	KFMOT() = default;
	int kalmanfilter_init(char *xclbin,
			const char *kernelName,
			int deviceIdx)
	{

		float A_ptr[49]={ 1, 0, 0, 0, 1, 0, 0,
                	        0, 1, 0, 0, 0, 1, 0,
			        0, 0, 1, 0, 0, 0, 1,
				0, 0, 0, 1, 0, 0, 0,
				0, 0, 0, 0, 1, 0, 0,
				0, 0, 0, 0, 0, 1, 0,
				0, 0, 0, 0, 0, 0, 1};

	        float Q_ptr[49] = { 0.0099999998, 0, 0, 0, 0, 0, 0,
			           0, 0.0099999998, 0, 0, 0, 0, 0,
			           0, 0, 0.0099999998, 0, 0, 0, 0,
			           0, 0, 0, 0.0099999998, 0, 0, 0,
			           0, 0, 0, 0, 0.0099999998, 0, 0,
			           0, 0, 0, 0, 0, 0.0099999998, 0,
			           0, 0, 0, 0, 0, 0, 0.0099999998};
		float R_ptr[16] = { 0.1, 0, 0, 0,
	                            0, 0.1, 0, 0,
			            0, 0, 0.1, 0,
			            0, 0, 0, 0.1};

       		float H_ptr[28] = { 1, 0, 0, 0, 0, 0, 0,
		                    0, 1, 0, 0, 0, 0, 0,
		                    0, 0, 1, 0, 0, 0, 0,
		                    0, 0, 0, 1, 0, 0, 0};

    		// Vector sizes:
    		size_t vec_nn_size_bytes = KF_N * KF_N * sizeof(float);
    		size_t vec_mn_size_bytes = KF_M * KF_N * sizeof(float);
    		size_t vec_n_size_bytes = KF_N * sizeof(float);
    		size_t vec_m_size_bytes = KF_M * sizeof(float);

		float* Uq_ptr = (float*)malloc(vec_nn_size_bytes);
    		float* Dq_ptr = (float*)malloc(vec_n_size_bytes);
	
		for(int i=0;i<KF_N;i++)
		{
	   		for(int j=0;j<KF_N;j++)
	   		{
	   			if(i==j){
			  		Uq_ptr[i*KF_N + j] = 1;
		  			Dq_ptr[i] = Q_ptr[i*KF_N + j];
				}else
		  			Uq_ptr[i*KF_N + j] = 0;

	   		}
		}
/*
		float U0_ptr[] = {1,	0,	0,	0,	0.999999,		0,		0,	
				  0,	1,	0,	0,	0,		0.999999,		0,	
				  0,	0,	1,	0,	0,		0,		1,	
				  0,	0,	0,	1,	0,		0,		0,	
				  0,	0,	0,	0,	1,		0,		0,	
				  0,	0,	0,	0,	0,		1,		0,	
				  0,	0,	0,	0,	0,		0,		1};	
		float D0_ptr[] = {11.01,   11.01,   11.0001, 11,      10000,   10000,   10000};	
		float X0_ptr[] = {1419.23, 594.655, 43626.7, 0.331505,        0,       0,       0};
		float y_ptr[] = {1419.23, 594.655, 43626.7, 0.331505};
*/
		float* U0_ptr = (float*)malloc(vec_nn_size_bytes);
    		float* D0_ptr = (float*)malloc(vec_n_size_bytes);
    		float* X0_ptr = (float*)malloc(vec_n_size_bytes);
    		float* y_ptr  = (float*)malloc(vec_m_size_bytes);

		//std::cout << "\n[CPP KF_INIT] Load R diag Matrix"<< std::endl;
    		float* R_diag_ptr = (float*)malloc(vec_m_size_bytes);
		for(int i=0;i<KF_M;i++)
		{
			for(int j=0;j<KF_M;j++)
			{
				if(i==j)
					R_diag_ptr[i] = R_ptr[i*KF_M + j];
			}
		}

    		bool load_x0ud0_en=0;
    		unsigned char control_flag =KF_INIT;
    		int num_of_kf_instance = 1;

		//************ XRT code ********//
		unsigned index = 0;
		std::string halLogfile;
		unsigned cu_index = 0;

		//xclDeviceHandle handle;
		uint64_t cu_base_addr = 0;
		uuid_t xclbinId;
		int first_mem = -1;
		bool ret_initXRT=0;
		bool ret_checkDevMem=0;

		//Load xclbin
		if (initXRT((char *)xclbin, index, halLogfile.c_str(), handle, cu_index, cu_base_addr, first_mem, xclbinId))
			ret_initXRT=1;
		if(xclOpenContext(handle, xclbinId, cu_index, true))
			throw std::runtime_error("Cannot create context");

		//create XRT buffer
		unsigned bo_inA_Handle = xclAllocBO(handle, vec_nn_size_bytes, 0, first_mem);
		float *inA_ptr_xrt = (float*)xclMapBO(handle, bo_inA_Handle, true);
		unsigned bo_inUq_Handle = xclAllocBO(handle, vec_nn_size_bytes, 0, first_mem);
		float *inUq_ptr_xrt = (float*)xclMapBO(handle, bo_inUq_Handle, true);
		unsigned bo_inDq_Handle = xclAllocBO(handle, vec_n_size_bytes, 0, first_mem);
		float *inDq_ptr_xrt = (float*)xclMapBO(handle, bo_inDq_Handle, true);
		unsigned bo_inH_Handle = xclAllocBO(handle, vec_mn_size_bytes, 0, first_mem);
		float *inH_ptr_xrt = (float*)xclMapBO(handle, bo_inH_Handle, true);
		bo_inX0_Handle = xclAllocBO(handle, 500*vec_n_size_bytes, 0, first_mem);
		inX0_ptr_xrt = (float*)xclMapBO(handle, bo_inX0_Handle, true);
		bo_inU0_Handle = xclAllocBO(handle, 500*vec_nn_size_bytes, 0, first_mem);
		inU0_ptr_xrt = (float*)xclMapBO(handle, bo_inU0_Handle, true);
		bo_inD0_Handle = xclAllocBO(handle, 500*vec_n_size_bytes, 0, first_mem);
		inD0_ptr_xrt = (float*)xclMapBO(handle, bo_inD0_Handle, true);
		unsigned bo_inR_Handle = xclAllocBO(handle, vec_m_size_bytes, 0, first_mem);
		float *inR_ptr_xrt = (float*)xclMapBO(handle, bo_inR_Handle, true);
		bo_iny_Handle = xclAllocBO(handle, 500*vec_m_size_bytes, 0, first_mem);
		iny_ptr_xrt = (float*)xclMapBO(handle, bo_iny_Handle, true);
		bo_outX_Handle = xclAllocBO(handle, 500*vec_n_size_bytes, 0, first_mem);
		bo_outU_Handle = xclAllocBO(handle, 500*vec_nn_size_bytes, 0, first_mem);
		bo_outD_Handle = xclAllocBO(handle, 500*vec_n_size_bytes, 0, first_mem);
		outX_ptr_xrt = (float*)xclMapBO(handle, bo_outX_Handle, false);
		outU_ptr_xrt = (float*)xclMapBO(handle, bo_outU_Handle, false);
		outD_ptr_xrt = (float*)xclMapBO(handle, bo_outD_Handle, false);

		//copy data to XRT buffer
		std::memcpy(inA_ptr_xrt, A_ptr, vec_nn_size_bytes);
		std::memcpy(inUq_ptr_xrt, Uq_ptr, vec_nn_size_bytes);
		std::memcpy(inDq_ptr_xrt, Dq_ptr, vec_n_size_bytes);
		std::memcpy(inH_ptr_xrt, H_ptr, vec_mn_size_bytes);
		std::memcpy(inX0_ptr_xrt, X0_ptr, vec_n_size_bytes);
		std::memcpy(inU0_ptr_xrt, U0_ptr, vec_nn_size_bytes);
		std::memcpy(inD0_ptr_xrt, D0_ptr, vec_n_size_bytes);
		std::memcpy(inR_ptr_xrt, R_diag_ptr, vec_m_size_bytes);
		std::memcpy(iny_ptr_xrt, y_ptr, vec_m_size_bytes);

		// Send data to the device memory
		if(xclSyncBO(handle, bo_inA_Handle, XCL_BO_SYNC_BO_TO_DEVICE , vec_nn_size_bytes, 0)) {
			return 1;
		}
		if(xclSyncBO(handle, bo_inUq_Handle, XCL_BO_SYNC_BO_TO_DEVICE , vec_nn_size_bytes, 0)) {
			return 1;
		}
		if(xclSyncBO(handle, bo_inDq_Handle, XCL_BO_SYNC_BO_TO_DEVICE , vec_n_size_bytes, 0)) {
			return 1;
		}
		if(xclSyncBO(handle, bo_inH_Handle, XCL_BO_SYNC_BO_TO_DEVICE , vec_mn_size_bytes, 0)) {
			return 1;
		}
		if(xclSyncBO(handle, bo_inX0_Handle, XCL_BO_SYNC_BO_TO_DEVICE , vec_n_size_bytes, 0)) {
			return 1;
		}
		if(xclSyncBO(handle, bo_inU0_Handle, XCL_BO_SYNC_BO_TO_DEVICE , vec_nn_size_bytes, 0)) {
			return 1;
		}
		if(xclSyncBO(handle, bo_inD0_Handle, XCL_BO_SYNC_BO_TO_DEVICE , vec_n_size_bytes, 0)) {
			return 1;
		}
		if(xclSyncBO(handle, bo_inR_Handle, XCL_BO_SYNC_BO_TO_DEVICE , vec_m_size_bytes, 0)) {
			return 1;
		}
		if(xclSyncBO(handle, bo_iny_Handle, XCL_BO_SYNC_BO_TO_DEVICE , vec_m_size_bytes, 0)) {
			return 1;
		}

		// Get & check the device memory address
		xclBOProperties p;
		uint64_t bo_inA_devAddr = !xclGetBOProperties(handle, bo_inA_Handle, &p) ? p.paddr : -1;
		if( (bo_inA_devAddr == (uint64_t)(-1)) ){
			ret_checkDevMem=1;
		}
		uint64_t bo_inUq_devAddr = !xclGetBOProperties(handle, bo_inUq_Handle, &p) ? p.paddr : -1;
		if( (bo_inUq_devAddr == (uint64_t)(-1)) ){
			ret_checkDevMem=1;
		}
		uint64_t bo_inDq_devAddr = !xclGetBOProperties(handle, bo_inDq_Handle, &p) ? p.paddr : -1;
		if( (bo_inDq_devAddr == (uint64_t)(-1)) ){
			ret_checkDevMem=1;
		}
		uint64_t bo_inH_devAddr = !xclGetBOProperties(handle, bo_inH_Handle, &p) ? p.paddr : -1;
		if( (bo_inH_devAddr == (uint64_t)(-1)) ){
			ret_checkDevMem=1;
		}
		uint64_t bo_inX0_devAddr = !xclGetBOProperties(handle, bo_inX0_Handle, &p) ? p.paddr : -1;
		if( (bo_inX0_devAddr == (uint64_t)(-1)) ){
			ret_checkDevMem=1;
		}
		uint64_t bo_inU0_devAddr = !xclGetBOProperties(handle, bo_inU0_Handle, &p) ? p.paddr : -1;
		if( (bo_inU0_devAddr == (uint64_t)(-1)) ){
			ret_checkDevMem=1;
		}
		uint64_t bo_inD0_devAddr = !xclGetBOProperties(handle, bo_inD0_Handle, &p) ? p.paddr : -1;
		if( (bo_inD0_devAddr == (uint64_t)(-1)) ){
			ret_checkDevMem=1;
		}
		uint64_t bo_inR_devAddr = !xclGetBOProperties(handle, bo_inR_Handle, &p) ? p.paddr : -1;
		if( (bo_inR_devAddr == (uint64_t)(-1)) ){
			ret_checkDevMem=1;
		}
		uint64_t bo_iny_devAddr = !xclGetBOProperties(handle, bo_iny_Handle, &p) ? p.paddr : -1;
		if( (bo_iny_devAddr == (uint64_t)(-1)) ){
			ret_checkDevMem=1;
		}
		uint64_t bo_outX_devAddr = !xclGetBOProperties(handle, bo_outX_Handle, &p) ? p.paddr : -1;
		if( (bo_outX_devAddr == (uint64_t)(-1)) ){
			ret_checkDevMem=1;
		}
		uint64_t bo_outU_devAddr = !xclGetBOProperties(handle, bo_outU_Handle, &p) ? p.paddr : -1;
		if( (bo_outU_devAddr == (uint64_t)(-1)) ){
			ret_checkDevMem=1;
		}
		uint64_t bo_outD_devAddr = !xclGetBOProperties(handle, bo_outD_Handle, &p) ? p.paddr : -1;
		if( (bo_outD_devAddr == (uint64_t)(-1)) ){
			ret_checkDevMem=1;
		}

		if(execHandle == 0) execHandle = xclAllocBO(handle, 4096, xclBOKind(0), (1<<31));
		if(execData == nullptr) execData = xclMapBO(handle, execHandle, true);
 
		try {

			auto ecmd2 = reinterpret_cast<ert_start_kernel_cmd*>(execData);

			// Clear the command in case it was recycled
			size_t regmap_size = XKF_CONTROL_ADDR_out_D_DATA/4 + 2; // regmap
			std::memset(ecmd2,0,(sizeof *ecmd2) + regmap_size);

			// Program the command packet header
			ecmd2->state = ERT_CMD_STATE_NEW;
			ecmd2->opcode = ERT_START_CU;
			ecmd2->count = 1 + regmap_size;  // cu_mask + regmap

			// Program the CU mask. One CU at index 0
			ecmd2->cu_mask = 0x1;

			// Program the register map
			ecmd2->data[XKF_CONTROL_ADDR_AP_CTRL] = 0x0; // ap_start
	
			ecmd2->data[XKF_CONTROL_ADDR_in_A_DATA/4] = bo_inA_devAddr & 0xFFFFFFFF;
			ecmd2->data[XKF_CONTROL_ADDR_in_A_DATA/4 + 1] = (bo_inA_devAddr >> 32) & 0xFFFFFFFF;

			ecmd2->data[XKF_CONTROL_ADDR_in_Uq_DATA/4] = bo_inUq_devAddr & 0xFFFFFFFF;
			ecmd2->data[XKF_CONTROL_ADDR_in_Uq_DATA/4 + 1] = (bo_inUq_devAddr >> 32) & 0xFFFFFFFF;

			ecmd2->data[XKF_CONTROL_ADDR_in_Dq_DATA/4] = bo_inDq_devAddr & 0xFFFFFFFF;
			ecmd2->data[XKF_CONTROL_ADDR_in_Dq_DATA/4 + 1] = (bo_inDq_devAddr >> 32) & 0xFFFFFFFF;

			ecmd2->data[XKF_CONTROL_ADDR_in_H_DATA/4] = bo_inH_devAddr & 0xFFFFFFFF;
			ecmd2->data[XKF_CONTROL_ADDR_in_H_DATA/4 + 1] = (bo_inH_devAddr >> 32) & 0xFFFFFFFF;

			ecmd2->data[XKF_CONTROL_ADDR_in_X0_DATA/4] = bo_inX0_devAddr & 0xFFFFFFFF;
			ecmd2->data[XKF_CONTROL_ADDR_in_X0_DATA/4 + 1] = (bo_inX0_devAddr >> 32) & 0xFFFFFFFF;

			ecmd2->data[XKF_CONTROL_ADDR_in_U0_DATA/4] = bo_inU0_devAddr & 0xFFFFFFFF;
			ecmd2->data[XKF_CONTROL_ADDR_in_U0_DATA/4 + 1] = (bo_inU0_devAddr >> 32) & 0xFFFFFFFF;

			ecmd2->data[XKF_CONTROL_ADDR_in_D0_DATA/4] = bo_inD0_devAddr & 0xFFFFFFFF;
			ecmd2->data[XKF_CONTROL_ADDR_in_D0_DATA/4 + 1] = (bo_inD0_devAddr >> 32) & 0xFFFFFFFF;

			ecmd2->data[XKF_CONTROL_ADDR_in_R_DATA/4] = bo_inR_devAddr & 0xFFFFFFFF;
			ecmd2->data[XKF_CONTROL_ADDR_in_R_DATA/4 + 1] = (bo_inR_devAddr >> 32) & 0xFFFFFFFF;

			ecmd2->data[XKF_CONTROL_ADDR_in_y_DATA/4] = bo_iny_devAddr & 0xFFFFFFFF;
			ecmd2->data[XKF_CONTROL_ADDR_in_y_DATA/4 + 1] = (bo_iny_devAddr >> 32) & 0xFFFFFFFF;

			ecmd2->data[XKF_CONTROL_ADDR_SCALAR_flag_DATA/4] = control_flag;
		
			ecmd2->data[XKF_CONTROL_ADDR_SCALAR_loop_DATA/4] = num_of_kf_instance;
		
			ecmd2->data[XKF_CONTROL_ADDR_SCALAR_XUDEN_DATA/4] = load_x0ud0_en;
		
			ecmd2->data[XKF_CONTROL_ADDR_out_X_DATA/4] = bo_outX_devAddr & 0xFFFFFFFF;
			ecmd2->data[XKF_CONTROL_ADDR_out_X_DATA/4 + 1] = (bo_outX_devAddr >> 32) & 0xFFFFFFFF;
	
			ecmd2->data[XKF_CONTROL_ADDR_out_U_DATA/4] = bo_outU_devAddr & 0xFFFFFFFF;
			ecmd2->data[XKF_CONTROL_ADDR_out_U_DATA/4 + 1] = (bo_outU_devAddr >> 32) & 0xFFFFFFFF;

			ecmd2->data[XKF_CONTROL_ADDR_out_D_DATA/4] = bo_outD_devAddr & 0xFFFFFFFF;
			ecmd2->data[XKF_CONTROL_ADDR_out_D_DATA/4 + 1] = (bo_outD_devAddr >> 32) & 0xFFFFFFFF;
		
			int ret;
			if ((ret = xclExecBuf(handle, execHandle)) != 0) {
				std::cout << "Unable to trigger preprocess, error:" << ret << std::endl;
				return ret;
			}
			do {
				ret = xclExecWait(handle, 1000);
				if (ret == 0) {
					std::cout << "preprocess Task Time out, state =" << ecmd2->state << "cu_mask = " << ecmd2->cu_mask << std::endl;

				} else if (ecmd2->state == ERT_CMD_STATE_COMPLETED) {
	
					break;
				}
			} while (1);
		
		}

	
		catch (std::exception const& e)
		{
			std::cout << "Exception: " << e.what() << "\n";
			std::cout << "FAILED TEST\n";
			return 1;
		}

		return 0;
	}//end of kalmainfilter_init


	//Kalman-filter Predict(Time Update) kernel execution
	int kalmanfilter_predict( int num_of_kf_instance, 
				float *&Xin_ptr, float *&Uin_ptr, float *&Din_ptr,
				float *&Xout_ptr, float *&Uout_ptr, float *&Dout_ptr)
	{
    		// Vector sizes:
	    	size_t vec_nn_size_bytes = num_of_kf_instance * KF_N * KF_N * sizeof(float);
   		size_t vec_n_size_bytes = num_of_kf_instance * KF_N * sizeof(float);

		// Control flag for Xilinx Kalman Filter:
    		bool load_x0ud0_en=1;
    		unsigned char control_flag = KF_INIT + KF_PREDICT_EN + KF_PREDICT_WRITE_STATE + KF_PREDICT_WRIET_CONV;

		/////////////////////////////////////// XRT ///////////////////////////////////////
		
		std::memcpy(inX0_ptr_xrt, Xin_ptr, vec_n_size_bytes);
		std::memcpy(inU0_ptr_xrt, Uin_ptr, vec_nn_size_bytes);
		std::memcpy(inD0_ptr_xrt, Din_ptr, vec_n_size_bytes);

		if(xclSyncBO(handle, bo_inX0_Handle, XCL_BO_SYNC_BO_TO_DEVICE , vec_n_size_bytes, 0)) {
			return 1;
		}
		if(xclSyncBO(handle, bo_inU0_Handle, XCL_BO_SYNC_BO_TO_DEVICE , vec_nn_size_bytes, 0)) {
			return 1;
		}
		if(xclSyncBO(handle, bo_inD0_Handle, XCL_BO_SYNC_BO_TO_DEVICE , vec_n_size_bytes, 0)) {
			return 1;
		}

		auto ecmd2 = reinterpret_cast<ert_start_kernel_cmd*>(execData);

		ecmd2->data[XKF_CONTROL_ADDR_SCALAR_flag_DATA/4] = control_flag;
		
		ecmd2->data[XKF_CONTROL_ADDR_SCALAR_loop_DATA/4] = num_of_kf_instance;
		
		ecmd2->data[XKF_CONTROL_ADDR_SCALAR_XUDEN_DATA/4] = load_x0ud0_en;
	

		int ret;
		if ((ret = xclExecBuf(handle, execHandle)) != 0) {
			std::cout << "Unable to trigger preprocess, error:" << ret << std::endl;
			return ret;
		}
		do {
			ret = xclExecWait(handle, 1000);
			if (ret == 0) {
				std::cout << "preprocess Task Time out, state =" << ecmd2->state << "cu_mask = " << ecmd2->cu_mask << std::endl;

			} else if (ecmd2->state == ERT_CMD_STATE_COMPLETED) {

				break;
			}
		} while (1);


		if(xclSyncBO(handle, bo_outX_Handle, XCL_BO_SYNC_BO_FROM_DEVICE, vec_n_size_bytes, 0)) {
			return 1;
		}
		if(xclSyncBO(handle, bo_outU_Handle, XCL_BO_SYNC_BO_FROM_DEVICE, vec_nn_size_bytes, 0)) {
			return 1;
		}
		if(xclSyncBO(handle, bo_outD_Handle, XCL_BO_SYNC_BO_FROM_DEVICE, vec_n_size_bytes, 0)) {
			return 1;
		}
											
		Xout_ptr=(float *)outX_ptr_xrt;
        	Uout_ptr=(float *)outU_ptr_xrt;
        	Dout_ptr=(float *)outD_ptr_xrt;

		return 0;

	}//end of kalmanfilter_predict


	//Kalman-filter Correct(Measurement Update) kernel execution
	int kalmanfilter_correct(int num_of_kf_instance, 
				float *&Xin_ptr, float *&Uin_ptr, float *&Din_ptr, float *&y_ptr,
				float *&Xout_ptr, float *&Uout_ptr, float *&Dout_ptr)
	{
    		// Vector sizes:
    		size_t vec_nn_size_bytes = num_of_kf_instance * KF_N * KF_N * sizeof(float);
    		size_t vec_n_size_bytes = num_of_kf_instance * KF_N * sizeof(float);
    		size_t vec_m_size_bytes = num_of_kf_instance * KF_M * sizeof(float);

		// Control flag for Xilinx Kalman Filter:
    		unsigned char control_flag = KF_INIT + KF_CORRECT_EN + KF_CORRECT_WRITE_STATE + KF_CORRECT_WRIET_CONV;;
    		bool load_x0ud0_en=1;

		std::memcpy(inX0_ptr_xrt, Xin_ptr, vec_n_size_bytes);
		std::memcpy(inU0_ptr_xrt, Uin_ptr, vec_nn_size_bytes);
		std::memcpy(inD0_ptr_xrt, Din_ptr, vec_n_size_bytes);
		std::memcpy(iny_ptr_xrt, y_ptr, vec_m_size_bytes);

		if(xclSyncBO(handle, bo_inX0_Handle, XCL_BO_SYNC_BO_TO_DEVICE , vec_n_size_bytes, 0)) {
			return 1;
		}
		if(xclSyncBO(handle, bo_inU0_Handle, XCL_BO_SYNC_BO_TO_DEVICE , vec_nn_size_bytes, 0)) {
			return 1;
		}
		if(xclSyncBO(handle, bo_inD0_Handle, XCL_BO_SYNC_BO_TO_DEVICE , vec_n_size_bytes, 0)) {
			return 1;
		}
		if(xclSyncBO(handle, bo_iny_Handle, XCL_BO_SYNC_BO_TO_DEVICE , vec_m_size_bytes, 0)) {
			return 1;
		}

		auto ecmd2 = reinterpret_cast<ert_start_kernel_cmd*>(execData);

		ecmd2->data[XKF_CONTROL_ADDR_SCALAR_flag_DATA/4] = control_flag;
		
		ecmd2->data[XKF_CONTROL_ADDR_SCALAR_loop_DATA/4] = num_of_kf_instance;
		
		ecmd2->data[XKF_CONTROL_ADDR_SCALAR_XUDEN_DATA/4] = load_x0ud0_en;	

		int ret;
		if ((ret = xclExecBuf(handle, execHandle)) != 0) {
			std::cout << "Unable to trigger preprocess, error:" << ret << std::endl;
			return ret;
		}
		do {
			ret = xclExecWait(handle, 1000);
			if (ret == 0) {
				std::cout << "preprocess Task Time out, state =" << ecmd2->state << "cu_mask = " << ecmd2->cu_mask << std::endl;

			} else if (ecmd2->state == ERT_CMD_STATE_COMPLETED) {

				break;
			}
		} while (1);

		if(xclSyncBO(handle, bo_outX_Handle, XCL_BO_SYNC_BO_FROM_DEVICE, vec_n_size_bytes, 0)) {
			return 1;
		}
		if(xclSyncBO(handle, bo_outU_Handle, XCL_BO_SYNC_BO_FROM_DEVICE, vec_nn_size_bytes, 0)) {
			return 1;
		}
		if(xclSyncBO(handle, bo_outD_Handle, XCL_BO_SYNC_BO_FROM_DEVICE, vec_n_size_bytes, 0)) {
			return 1;
		}
											
		Xout_ptr=(float *)outX_ptr_xrt;
        	Uout_ptr=(float *)outU_ptr_xrt;
        	Dout_ptr=(float *)outD_ptr_xrt;
		return 0;
	}//end of kalmanfilter_correct

}; // end of class

