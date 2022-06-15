# /*******************************************************************************
# /*                                                                         
# * Copyright 2019 Xilinx Inc.                                               
# *                                                                          
# * Licensed under the Apache License, Version 2.0 (the "License");          
# * you may not use this file except in compliance with the License.         
# * You may obtain a copy of the License at                                  
# *                                                                          
# *    http://www.apache.org/licenses/LICENSE-2.0                            
# *                                                                          
# * Unless required by applicable law or agreed to in writing, software      
# * distributed under the License is distributed on an "AS IS" BASIS,        
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# * See the License for the specific language governing permissions and      
# * limitations under the License.                                           
# */
# *******************************************************************************/

# Setting the arch of XVDPU. For more details, please refer to PG389.
# Default parameters: LOAD_PARALLEL_IMG = 2

#Compute Unit (CU) number . The Max is 3.
CU_N     = 1

#CPB_N can be 32, or 64
CPB_N       = 32

#For Single CU (CU_N=1), BATCH number can be 1 ~ 6 for CPB_N=32, and BATCH number can be 1 ~ 5 for CPB_N=64
BATCH_SingleCU = 3

ifneq ($(CU_N), 1)
	#When CU_N=2 or CU_N=3, BATCH_N must be 1
	BATCH_N    = 1 
else
	BATCH_N    = $(BATCH_SingleCU) 
endif
	
#PL_FREQ is the frequency of XVDPU's 'm_axi_aclk'. Default is 333M Hz.
PL_FREQ     = 333000000

#Max value of 'UBANK_IMG_N' is 16, and Max value of 'UBANK_WGT_N' is 17.
UBANK_IMG_N = 16
UBANK_WGT_N = 17

#For Psmnet
PSMNET_EN  = 0

ifeq ($(PSMNET_EN), 1)
	CU_N       = 1
	CPB_N      = 32
	BATCH_N    = 3  
endif