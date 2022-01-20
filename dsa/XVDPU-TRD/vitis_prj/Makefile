############################## Help ##############################
.PHONY: help
help:
	@echo "Makefile Usage:"
	@echo ""
	@echo "  Note: Firstly specify the configuration of XVDPU by changing 'xvdpu_config.mk' "
	@echo ""
	@echo "  make files"
	@echo "      Command to generate specifical scripts and copy IP files to vitis_prj folder"
	@echo ""
	@echo "  make aie"
	@echo "      Command to generate the libadf.a of AIE"
	@echo ""
	@echo "  make xo"
	@echo "      Command to generate the xo files of XVDPU"
	@echo ""
	@echo "  make all"
	@echo "      Command to generate all design"
	@echo ""
	@echo "  make clean "
	@echo "      Command to remove IP files from vitis_prj folder and all the generated files"
	@echo ""
############################ Included files ####################################################
include xvdpu_config.mk
############################ Setting Up Project Variables #######################################
ABS_PATH                := $(shell pwd -P)
$(eval FATHER_PATH      := $(realpath $(ABS_PATH)/..))
#Platform setting
PLATFORM                 = $(FATHER_PATH)/vck190_platform/platforms/xilinx_vck190_mipiRxSingle_hdmiTx_202120_1/vck190_mipiRxSingle_hdmiTx.xpfm
DEVICE                  ?= vck190
XCL_EMULATION_MODE      ?= hw
unexport XCL_EMULATION_MODE

#Build directory
BUILD_DIR                = $(XCL_EMULATION_MODE)
#Default v++ compiler flags
VXXFLAGS                += -t $(XCL_EMULATION_MODE) --platform $(PLATFORM) --save-temps --temp_dir $(BUILD_DIR)/binary_container_1 -g
VXXFLAGS                += --xp vivado_prop:run.impl_1.STEPS.PLACE_DESIGN.TCL.PRE=$(ABS_PATH)/scripts/pre_place.tcl
VXXFLAGS                += --xp vivado_prop:run.impl_1.STEPS.PLACE_DESIGN.TCL.POST=$(ABS_PATH)/scripts/post_place.tcl
VXXFLAGS                += --xp param:compiler.userPostSysLinkOverlayTcl=$(ABS_PATH)/scripts/post_linker.tcl
VXXFLAGS                += --xp vivado_prop:run.impl_1.GEN_FULL_BITSTREAM=1
VXXFLAGS                += --log_dir $(BUILD_DIR)/binary_container_1/logs
VXXFLAGS                += --report_dir $(BUILD_DIR)/binary_container_1/reports
VXXFLAGS                += --xp vivado_prop:run.impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED=false

VXX                      = $(XILINX_VITIS)/bin/v++
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Build products
AIE_OBJECTS              = $(ABS_PATH)/aie/ip/libadf.a
BINARY_CONTAINERS       += $(BUILD_DIR)/binary_container_1.xclbin
BUILD_SUBDIRS           += $(BUILD_DIR)/binary_container_1
BINARY_CONTAINER_1_OBJS += $(BUILD_DIR)/binary_container_1/DPUCVDX8G.xo $(BUILD_DIR)/binary_container_1/libadf.a
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Package RTL kernel 
VIVADO_ROOT             := $(XILINX_VIVADO)
VIVADO                  := $(VIVADO_ROOT)/bin/vivado
xvdpu_TCL                = $(ABS_PATH)/scripts/gen_xvdpu_xo.tcl
DPU_KERN_NAME            = DPUCVDX8G
BATCH_SHRWGT_N           = 4
AIE_PL_FREQ              = 250
############################ Building steps ###################################################

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
.PHONY: all
all: files $(AIE_OBJECTS) $(BINARY_CONTAINERS) $(BUILD_DIR)/dpu.xclbin
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Prepare files for special configuration 
files:
	@echo "INFO:- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
	@echo "INFO:Preparing files for project"
	${MAKE} -C scripts all CPB_N=$(CPB_N) BATCH_N=$(BATCH_N) UBANK_IMG_N=$(UBANK_IMG_N) UBANK_WGT_N=$(UBANK_WGT_N) BATCH_SHRWGT_N=$(BATCH_SHRWGT_N)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Compile AIE IP as 'libadf.a'
.PHONY: aie
aie: $(ABS_PATH)/aie/ip/libadf.a

$(ABS_PATH)/aie/ip/libadf.a:
	@echo "INFO:- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
	@echo "INFO:Buiding $(notdir $@)..."
	@XILINX_VITIS_AIETOOLS=$(XILINX_VITIS)/aietools source $(XILINX_VITIS)/settings64.sh && ${MAKE} --no-print-directory -C aie ip CPB=$(CPB_N) BAT_NUM=$(BATCH_N) vver=212 pl_freq=$(AIE_PL_FREQ) aie_activations_en=1 bat_sharewgt=$(BATCH_SHRWGT_N)	
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Copy AIE's libadf.a to folder $(BUILD_SUBDIRS)
$(BUILD_DIR)/binary_container_1/libadf.a:
	@echo "INFO:- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
	@echo "INFO:Copying $(notdir $@)..."
	@mkdir -p $(@D)
	@cp aie/ip/libadf.a "$@"
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Compile xvdpu kernel as XO file
.PHONY: xo
xo: $(BUILD_DIR)/binary_container_1/DPUCVDX8G.xo

$(BUILD_DIR)/binary_container_1/DPUCVDX8G.xo:
	@echo "INFO:- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
	@echo "INFO:Building $(notdir $@) for $(BUILD_DIR)..."
	@mkdir -p $(@D)
	-@rm -rf $@
	$(VIVADO) -mode batch -source $(xvdpu_TCL) -tclargs $@ $(DPU_KERN_NAME) $(XCL_EMULATION_MODE) $(DEVICE)
	
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Link xvdpu kernel and AIE graph with Platform
$(BUILD_DIR)/binary_container_1.xclbin: $(BINARY_CONTAINER_1_OBJS)
	@echo "INFO:- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
	@echo "INFO:Building $(notdir $@) for $(BUILD_DIR)..."
	@mkdir -p $(@D)
	@$(VXX) $(VXXFLAGS) -l --config scripts/system.cfg --config scripts/xvdpu_aie_noc.cfg \
	  --clock.freqHz $(PL_FREQ):DPUCVDX8G_1.m_axi_aclk \
	  --clock.freqHz $(PL_FREQ):ai_engine_0.aclk0 \
	  --clock.freqHz 150000000:DPUCVDX8G_1.s_axi_aclk \
	  -o "$@" $(BINARY_CONTAINER_1_OBJS) 
	  
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Package design as SD Card Image
.PHONY: package
package: $(BUILD_DIR)/dpu.xclbin
$(BUILD_DIR)/dpu.xclbin: $(BUILD_DIR)/binary_container_1.xclbin $(BUILD_DIR)/binary_container_1/libadf.a
	@echo "INFO:- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
	@echo "INFO:Packaging $(notdir $@) for $(BUILD_DIR)..."
	-@rm -rf package_out
	@cd $(BUILD_DIR); $(VXX) -t $(XCL_EMULATION_MODE) --platform $(PLATFORM)  --save-temps --temp_dir binary_container_1 -p binary_container_1.xclbin binary_container_1/libadf.a -o "$@" \
	  --package.out_dir ../package_out \
	  --package.boot_mode sd  \
	  --package.rootfs $(dir $(PLATFORM))sw/*/xrt/filesystem/rootfs.ext4 \
	  --package.sd_file $(dir $(PLATFORM))sw/*/xrt/image/Image 	
	cp  $(ABS_PATH)/hw/binary_container_1/link/vivado/vpl/prj/prj.gen/sources_1/bd/*/ip/*_DPUCVDX8G_1_0/arch.json $(ABS_PATH)/package_out/sd_card
	gzip package_out/sd_card.img
	-@rm -rf .ipcache .Xil packaged_kernel_*_hw_* tmp_kernel_pack_*_hw_* vivado* *.log sample*.ini

############################ Clean ############################################################			
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Clean build products
.PHONY: clean
clean:
	-@rm -rf .Xil vivado* *.txt *.log *.xsa .ipcache sample*.ini packaged_kernel_*_hw_* tmp_kernel_pack_*_hw_*
	-@rm -rf hw package_out scripts/kernel.xml scripts/*xvdpu_aie_noc.cfg xvdpu/hdl xvdpu/inc xvdpu/vitis_cfg xvdpu/ttcl
	-@rm -rf aie	