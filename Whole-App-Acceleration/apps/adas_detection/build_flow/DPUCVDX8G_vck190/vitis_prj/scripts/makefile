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

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CFG_VH_PATH         = xvdpu/vitis_cfg/vitis_cfg.vh
CPB_N               = 32
LOAD_PARALLEL_IMG   = 2
SAVE_PARALLEL_IMG   = 2
BATCH_SHRWGT_N      = 4
AIE_PL_FREQ         = 250
DBG_ENA             = 0
PROF_ENA            = 0

#Prepare project files
.PHONY: files
files: 
	@echo "INFO:- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
	@echo "INFO:Preparing files for project"	
	mkdir -p aie/ip
	cp $(FATHER_PATH)/xvdpu_ip/aie/C$(CPB_N)B$(BATCH_N)/libadf.*   aie/ip	
	gunzip aie/ip/libadf.a.gz
	cp -r $(FATHER_PATH)/xvdpu_ip/rtl/hdl         xvdpu/.
	cp -r $(FATHER_PATH)/xvdpu_ip/rtl/vitis_cfg   xvdpu/.
	cp -r $(FATHER_PATH)/xvdpu_ip/rtl/inc         xvdpu/.
	cp -r $(FATHER_PATH)/xvdpu_ip/rtl/ttcl        xvdpu/.
	sed -i "s/set pl_freq.*/set pl_freq $(PL_FREQ)/g" scripts/post_linker.tcl
	sed -i "s/set CPB.*/set CPB      $(CPB_N)/g"      scripts/post_linker.tcl
	cd scripts && python xvdpu_aie.py $(BATCH_N) $(CPB_N) $(BATCH_SHRWGT_N)&& python kernel_xml.py $(BATCH_N) $(LOAD_PARALLEL_IMG) $(CPB_N) $(BATCH_SHRWGT_N)
	cd ..
	#Change parameter of XVDPU
	sed -i "s/define wrp_CPB_N               .*/define wrp_CPB_N               $(CPB_N)/g" $(CFG_VH_PATH)
	sed -i "s/define wrp_BATCH_N             .*/define wrp_BATCH_N             $(BATCH_N)/g" $(CFG_VH_PATH)
	sed -i "s/define wrp_BATCH_SHRWGT_N      .*/define wrp_BATCH_SHRWGT_N      $(BATCH_SHRWGT_N)/g" $(CFG_VH_PATH)
	sed -i "s/define wrp_UBANK_IMG_N         .*/define wrp_UBANK_IMG_N         $(UBANK_IMG_N)/g" $(CFG_VH_PATH)
	sed -i "s/define wrp_UBANK_WGT_N         .*/define wrp_UBANK_WGT_N         $(UBANK_WGT_N)/g" $(CFG_VH_PATH)
	sed -i "s/define wrp_LOAD_PARALLEL_IMG   .*/define wrp_LOAD_PARALLEL_IMG   $(LOAD_PARALLEL_IMG)/g" $(CFG_VH_PATH)
	sed -i "s/define wrp_SAVE_PARALLEL_IMG   .*/define wrp_SAVE_PARALLEL_IMG   $(SAVE_PARALLEL_IMG)/g" $(CFG_VH_PATH)

ifeq  ($(DBG_ENA), 1)
	echo -e "\`define wrp_DBG_ENA             1"  >> $(CFG_VH_PATH)
endif

ifeq  ($(PROF_ENA), 1)
	echo -e "\`define wrp_PROF_ENA            1"  >> $(CFG_VH_PATH)
endif

ifeq  ($(BATCH_N), 6)
	echo -e "\`define wrp_UBANK_IMG_MRS       1" >> $(CFG_VH_PATH)
	echo -e "\`define wrp_UBANK_WGT_MRS       1" >> $(CFG_VH_PATH)
	echo -e "\`define wrp_ELEW_MULT_ENA       0" >> $(CFG_VH_PATH)
endif
