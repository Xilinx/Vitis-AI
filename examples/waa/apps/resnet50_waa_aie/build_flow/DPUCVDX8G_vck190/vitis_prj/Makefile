############################## Help ##############################
.PHONY: help
help:
	@echo "Makefile Usage:"
	@echo ""
	@echo "  Note: Firstly specify the configuration of XVDPU by changing 'xvdpu_config.mk' "
	@echo ""
	@echo "  make files"
	@echo "      Generate specifical scripts and copy IP files to vitis_prj folder"
	@echo ""
	@echo "  make aie"
	@echo "      Generate the libadf.a of AIE"
	@echo ""
	@echo "  make xo"
	@echo "      Generate the xo files of XVDPU"
	@echo ""
	@echo "  make link"
	@echo "      Link kernels with platform"
	@echo ""
	@echo "  make package"
	@echo "      Package SD image"
	@echo ""
	@echo "  make all"
	@echo "      Generate all design"
	@echo ""
	@echo "  make clean "
	@echo "      Remove IP files from vitis_prj folder and all the generated files"
	@echo ""
############################ Included files ####################################################
include xvdpu_config.mk

############################ Setting Up Project Variables #######################################
ABS_PATH                := $(shell pwd -P)
$(eval FATHER_PATH      := $(realpath $(ABS_PATH)/..))
$(eval PLUGINS_PATH      := $(realpath $(FATHER_PATH)/../../../../plugins/))
#Platform setting
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
BINARY_CONTAINERS       += $(BUILD_DIR)/binary_container_1.xsa
BUILD_SUBDIRS           += $(BUILD_DIR)/binary_container_1
BINARY_CONTAINER_1_OBJS += $(BUILD_DIR)/binary_container_1/DPUCVDX8G.xo $(BUILD_DIR)/binary_container_1/libadf.a $(PLUGINS_PATH)/include/aie/lib/hw/tiler.xo $(PLUGINS_PATH)/include/aie/lib/hw/stitcher.xo

ifeq ($(PSMNET_EN), 1)
	BINARY_CONTAINER_1_OBJS   += ./xo_files/mm2s.xo ./xo_files/s2mm.xo
endif

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Package RTL kernel 
VIVADO_ROOT             := $(XILINX_VIVADO)
VIVADO                  := $(VIVADO_ROOT)/bin/vivado

xvdpu_TCL                = $(ABS_PATH)/scripts/gen_xvdpu_xo.tcl
DPU_KERN_NAME            = DPUCVDX8G

ifneq ($(CU_N), 1)
	BATCH_SHRWGT_N       = 1
else
	BATCH_SHRWGT_N       = 4
endif	

AIE_BATCH_N         = $(shell expr $(CU_N) \* $(BATCH_N)) 

AIE_PL_FREQ         = 250
AIETOOL_VERSION     = 221

ifeq ($(PSMNET_EN), 1)
	AIE_ACTIVA_EN   = 0
else
	AIE_ACTIVA_EN   = 1
endif

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
ifeq ($(CU_N), 1)
	DPU_m_axi_clk    = DPUCVDX8G_1.m_axi_aclk 
	DPU_s_axi_clk    = DPUCVDX8G_1.s_axi_aclk
else ifeq ($(CU_N),2)
	DPU_m_axi_clk    = DPUCVDX8G_1.m_axi_aclk,DPUCVDX8G_2.m_axi_aclk
	DPU_s_axi_clk    = DPUCVDX8G_1.s_axi_aclk,DPUCVDX8G_2.s_axi_aclk
else ifeq ($(CU_N),3)
	DPU_m_axi_clk    = DPUCVDX8G_1.m_axi_aclk,DPUCVDX8G_2.m_axi_aclk,DPUCVDX8G_3.m_axi_aclk
	DPU_s_axi_clk    = DPUCVDX8G_1.s_axi_aclk,DPUCVDX8G_2.s_axi_aclk,DPUCVDX8G_3.s_axi_aclk
endif

ifeq ($(PSMNET_EN), 1)
	kernel_clk    = ai_engine_0.aclk0,mm2s_1.ap_clk,s2mm_1.ap_clk
else
	kernel_clk    = ai_engine_0.aclk0
endif
############################ Building steps ###################################################

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
.PHONY: all
all: files aie xo link package

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Prepare files for special configuration 
files:
	@echo "INFO:- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
	@echo "INFO:Preparing files for project"
	${MAKE} -C scripts all CU_N=$(CU_N) CPB_N=$(CPB_N) BATCH_N=$(BATCH_N) UBANK_IMG_N=$(UBANK_IMG_N) UBANK_WGT_N=$(UBANK_WGT_N) BATCH_SHRWGT_N=$(BATCH_SHRWGT_N) AIETOOL_VERSION=$(AIETOOL_VERSION) PSMNET_EN=$(PSMNET_EN)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Compile AIE IP as 'libadf.a'
.PHONY: aie
aie: $(ABS_PATH)/aie/ip/libadf.a

$(ABS_PATH)/aie/ip/libadf.a:
	@echo "INFO:- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
	@echo "INFO:Buiding $(notdir $@)..."
	XILINX_VITIS_AIETOOLS=$(XILINX_VITIS)/aietools source $(XILINX_VITIS)/settings64.sh && ${MAKE} --no-print-directory -C aie ip CPB=$(CPB_N) BAT_NUM=$(AIE_BATCH_N) vver=$(AIETOOL_VERSION) pl_freq=$(AIE_PL_FREQ) aie_activations_en=$(AIE_ACTIVA_EN) bat_sharewgt=$(BATCH_SHRWGT_N) PLUGINS_PATH=$(PLUGINS_PATH)
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
.PHONY:link
link:  $(BINARY_CONTAINERS)

$(BINARY_CONTAINERS): $(BINARY_CONTAINER_1_OBJS)
	@echo "INFO:- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
	@echo "INFO:Building $(notdir $@) for $(BUILD_DIR)..."
	@echo " DPU_m_axi_clk = $(DPU_m_axi_clk)"
	@echo " DPU_s_axi_clk = $(DPU_s_axi_clk)"
	@echo " BINARY_CONTAINER_1_OBJS = $(BINARY_CONTAINER_1_OBJS)"	
	@mkdir -p $(@D)
	@$(VXX) $(VXXFLAGS) -l --config scripts/system.cfg --config scripts/xvdpu_aie_noc.cfg  -o "$@" $(BINARY_CONTAINER_1_OBJS) \
	  --clock.freqHz $(PL_FREQ):$(DPU_m_axi_clk) \
	  --clock.freqHz 150000000:$(DPU_s_axi_clk)  \
	  --clock.freqHz 400000000:Tiler_top_1 \
	  --clock.freqHz 400000000:stitcher_top_1 \
	  --clock.freqHz $(PL_FREQ):$(kernel_clk) 	  
	  
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Package design as SD Card Image
.PHONY: package
package: $(BUILD_DIR)/dpu.xclbin
$(BUILD_DIR)/dpu.xclbin: $(BINARY_CONTAINERS) $(BUILD_DIR)/binary_container_1/libadf.a
	@echo "INFO:- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
	@echo "INFO:Packaging $(notdir $@) for $(BUILD_DIR)..."
	-@rm -rf package_out
	@cd $(BUILD_DIR); $(VXX) -t $(XCL_EMULATION_MODE) --platform $(PLATFORM)  --save-temps --temp_dir binary_container_1 -p binary_container_1.xsa binary_container_1/libadf.a -o "$@" \
	  --package.out_dir ../package_out \
	  --package.boot_mode sd  \
	  --package.rootfs $(EDGE_COMMON_SW)/rootfs.ext4 \
	  --package.sd_file $(EDGE_COMMON_SW)/Image 	
	cp  $(ABS_PATH)/hw/binary_container_1/link/vivado/vpl/prj/prj.gen/sources_1/bd/*/ip/*_DPUCVDX8G_1_0/arch.json $(ABS_PATH)/package_out/sd_card
#Uncomment the following to gzip the sd_card.img file
#	gzip package_out/sd_card.img
	-@rm -rf .ipcache .Xil packaged_kernel_*_hw_* tmp_kernel_pack_*_hw_* vivado* *.log sample*.ini

############################ Clean ############################################################			
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Clean build products
.PHONY: clean
clean:
	-@rm -rf .Xil vivado* *.txt *.log *.xsa .ipcache sample*.ini packaged_kernel_*_hw_* tmp_kernel_pack_*_hw_*
	-@rm -rf hw package_out scripts/kernel.xml scripts/*xvdpu_aie_noc.cfg xvdpu/hdl xvdpu/inc xvdpu/vitis_cfg xvdpu/ttcl xo_files
	-@rm -rf aie