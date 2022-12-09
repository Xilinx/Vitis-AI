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
CFG_VH_PATH         = ../xvdpu/vitis_cfg/vitis_cfg.vh

ifeq ($(PSMNET_EN), 1)
	LOAD_PARALLEL_IMG   = 4
	SAVE_PARALLEL_IMG   = 4
	RAM_IMG_ADDR_BW     = 14
	UBANK_IMG_MRS       = 1
	UBANK_WGT_MRS       = 1
else
	LOAD_PARALLEL_IMG   = 2
	SAVE_PARALLEL_IMG   = 2
	RAM_IMG_ADDR_BW     = 13
	UBANK_IMG_MRS       = 0
	UBANK_WGT_MRS       = 0
endif

RAM_WGT_ADDR_BW     = 13
UBANK_BIAS          = 1
DBG_ENA             = 0
PROF_ENA            = 0
LOAD_PARALLEL_WGT   = 4

ifeq  ($(CPB_N), 32)
	BATCH_LIST          = 1 2 3 4 5 6
endif

ifeq  ($(CPB_N), 64)
	BATCH_LIST          = 1 2 3 4 5
endif


CU_NUM_LIST         = 1 2 3
	
#Prepare project files: copy xvdpu IP to project folder, generate scripts for special configuration of xvdpu.
.PHONY: all 
all: 
	@valid=0; \
	for p in $(BATCH_LIST); do \
	  if [ "$$p" = "$(BATCH_N)" ]; then \
	    valid=1; \
	    break; \
	  fi \
	done; \
	if [ "$$valid" -ne 1 ]; then \
	  echo 'Invalid batch number $(BATCH_N). Choose one of: $(BATCH_LIST)'; \
	  exit 1; \
	fi; 
	
	@valid_cu=0; \
	for p in $(CU_NUM_LIST); do \
	  if [ "$$p" = "$(CU_N)" ]; then \
		valid_cu=1; \
		break; \
	  fi \
	done; \
	if [ "$$valid_cu" -ne 1 ]; then \
	  echo 'Invalid CU number $(CU_N). "CU_N" should be one of: 1, 2, or 3'; \
	  exit 1; \
	fi; 
	
#Preparing files for project
	cp -r ../../xvdpu_ip/aie       ../aie
	
	if [ $(PSMNET_EN) -eq 1 ]; then \
		cp -r ../../xvdpu_ip/psmnet     ../aie ; \
		mv ../aie/psmnet/xo_files       ../    ; \
		rm -rf ../aie/graph_comb.cc                ; \
		rm -rf ../aie/scripts/compile_ip.csh.$(AIETOOL_VERSION); \
		mv ../aie/psmnet/graph_comb.cc  ../aie ; \
		mv ../aie/psmnet/compile_ip.csh ../aie/scripts/compile_ip.csh.$(AIETOOL_VERSION); \
	fi; 	
	
	# cp -r ../../xvdpu_ip/aie             ../.
	cp -r ../../xvdpu_ip/rtl/hdl         ../xvdpu/.
	cp -r ../../xvdpu_ip/rtl/vitis_cfg   ../xvdpu/.
	cp -r ../../xvdpu_ip/rtl/inc         ../xvdpu/.
	cp -r ../../xvdpu_ip/rtl/ttcl        ../xvdpu/.	
#Change parameters in 'vitis_cfg.vh'
	sed -i "s/define wrp_CPB_N               .*/define wrp_CPB_N               $(CPB_N)/g" $(CFG_VH_PATH)
	sed -i "s/define wrp_BATCH_N             .*/define wrp_BATCH_N             $(BATCH_N)/g" $(CFG_VH_PATH)
	sed -i "s/define wrp_BATCH_SHRWGT_N      .*/define wrp_BATCH_SHRWGT_N      $(BATCH_SHRWGT_N)/g" $(CFG_VH_PATH)
	sed -i "s/define wrp_UBANK_IMG_N         .*/define wrp_UBANK_IMG_N         $(UBANK_IMG_N)/g" $(CFG_VH_PATH)
	sed -i "s/define wrp_UBANK_WGT_N         .*/define wrp_UBANK_WGT_N         $(UBANK_WGT_N)/g" $(CFG_VH_PATH)
	sed -i "s/define wrp_LOAD_PARALLEL_IMG   .*/define wrp_LOAD_PARALLEL_IMG   $(LOAD_PARALLEL_IMG)/g" $(CFG_VH_PATH)
	sed -i "s/define wrp_SAVE_PARALLEL_IMG   .*/define wrp_SAVE_PARALLEL_IMG   $(SAVE_PARALLEL_IMG)/g" $(CFG_VH_PATH)
	printf "\`define wrp_LOAD_PARALLEL_WGT   $(LOAD_PARALLEL_WGT)\n"  >> $(CFG_VH_PATH)
	printf "\`define wrp_RAM_IMG_ADDR_BW     $(RAM_IMG_ADDR_BW)\n" >> $(CFG_VH_PATH)
	printf "\`define wrp_RAM_WGT_ADDR_BW     $(RAM_WGT_ADDR_BW)\n" >> $(CFG_VH_PATH)
	printf "\`define wrp_UBANK_IMG_MRS       $(UBANK_IMG_MRS)\n" >> $(CFG_VH_PATH)
	printf "\`define wrp_UBANK_WGT_MRS       $(UBANK_WGT_MRS)\n" >> $(CFG_VH_PATH)
	printf "\`define wrp_UBANK_BIAS          $(UBANK_BIAS)\n" >> $(CFG_VH_PATH)
		
	@if [ $(DBG_ENA) -eq 1 ]; then \
		printf "\`define wrp_DBG_ENA             1\n"  >> $(CFG_VH_PATH) ; \
	fi; 

	@if [ $(PROF_ENA) -eq 1 ]; then \
		printf "\`define wrp_PROF_ENA            1\n"  >> $(CFG_VH_PATH) ; \
	fi; 

	@if [ $(BATCH_N) -eq 6 ]; then \
		sed -i "s/define wrp_UBANK_IMG_MRS   .*/define wrp_UBANK_IMG_MRS       1/g"  $(CFG_VH_PATH) ; \
		sed -i "s/define wrp_UBANK_WGT_MRS   .*/define wrp_UBANK_WGT_MRS       1/g"  $(CFG_VH_PATH) ; \
	fi;

#Generate kernel.xml of DPUCVDX8G	
	@echo "INFO: Generate kernel.xml of DPUCVDX8G "
	python kernel_xml.py $(BATCH_N) $(LOAD_PARALLEL_IMG) $(CPB_N) $(BATCH_SHRWGT_N) $(LOAD_PARALLEL_WGT)
	
#Generate 'xvdpu_aie_noc.cfg'	
	@echo "INFO: Generate 'xvdpu_aie_noc.cfg' "
	@python xvdpu_aie_noc.py $(BATCH_N) $(CPB_N) $(LOAD_PARALLEL_IMG) $(BATCH_SHRWGT_N) $(LOAD_PARALLEL_WGT) $(CU_N) $(PSMNET_EN)	