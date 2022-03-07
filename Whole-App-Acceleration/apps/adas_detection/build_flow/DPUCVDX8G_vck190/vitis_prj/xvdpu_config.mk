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
# Default parameters:CPB_N = 32, LOAD_PARALLEL_IMG = 2

# BATCH_N can be 1 ~ 6
BATCH_N     = 3

#PL_FREQ is the frequency of XVDPU's 'm_axi_aclk'. Default is 333M Hz.
PL_FREQ     = 333

#Max value of 'UBANK_IMG_N' is 16, and Max value of 'UBANK_WGT_N' is 17.
UBANK_IMG_N = 16
UBANK_WGT_N = 17
