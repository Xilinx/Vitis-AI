################################## user config start######################

# Supported board list:
# u200 | u250
BOARD = u250


# Supported platform list: 
# xilinx_u200_xdma_201830_2 | xilinx_u200_gen3x16_xdma_base_1  | xilinx_u250_xdma_201830_2  | xilinx_u250_gen3x16_xdma_3_1
PLATFORM = xilinx_u250_gen3x16_xdma_3_1

# Supported freq list: 
# 50 | 400 | 500 | 600
FREQ = 600

#xilinx_u250_xdma_201830_2        0 1 2 3
#xilinx_u250_gen3x16_xdma_3_1     0 2 3
#xilinx_u200_xdma_201830_2        0 2
#xilinx_u200_gen3x16_xdma_base_1  0 2
DPU_SLR_IDX =0 2 3

###platform xpfm path
ifeq ($(PLATFORM), xilinx_u250_xdma_201830_2)
	DEVICES:= /opt/xilinx/platforms/xilinx_u250_xdma_201830_2/xilinx_u250_xdma_201830_2.xpfm
else  ifeq ($(PLATFORM), xilinx_u250_gen3x16_xdma_3_1)
	DEVICES:=/opt/xilinx/platforms/xilinx_u250_gen3x16_xdma_3_1_202020_1/xilinx_u250_gen3x16_xdma_3_1_202020_1.xpfm
else ifeq ($(PLATFORM), xilinx_u200_xdma_201830_2)
	DEVICES:= /opt/xilinx/platforms/xilinx_u200_xdma_201830_2/xilinx_u200_xdma_201830_2.xpfm
else ifeq ($(PLATFORM), xilinx_u200_gen3x16_xdma_base_1)
	DEVICES:= /opt/xilinx/platforms/xilinx_u200_gen3x16_xdma_1_202110_1/xilinx_u200_gen3x16_xdma_1_202110_1.xpfm
endif

###user xo path
BINARY_CONTAINER_dpdpuv3_wrapper_OBJS +=

###user xo num/slr/ddr config
CLFLAGS_SLR +=
################################## user config end#######################

USE_ENC := true

ENC_VERSION := 1

STEP = 0

TARGET := hw


PWD_PATH =$(shell pwd)
SDA_PATH =$(PWD_PATH)

DPU_NUM:= $(words $(DPU_SLR_IDX))

ifeq ($(BOARD)_$(DPU_NUM)_$(word 1,$(DPU_SLR_IDX)), u200_1_2)
	DPU_DDR_IDX = 3
else ifeq ($(BOARD)_$(DPU_NUM)_$(word 1,$(DPU_SLR_IDX)), u200_2_2)
	DPU_DDR_IDX = 3 0
else ifeq ($(BOARD)_$(DPU_NUM)_$(word 2,$(DPU_SLR_IDX)), u200_2_2)
	DPU_DDR_IDX = 0 3
else
	DPU_DDR_IDX = $(DPU_SLR_IDX)
endif

PLACE_DIRECTIVE = SSI_HighUtilSLRs

ROUTE_DIRECTIVE = NoTimingRelaxation
DEVICE := $(DEVICES)
TARGETS := hw
TARGET := $(TARGETS)
VPP := v++
SDCARD := sd_card
XSA := $(strip $(patsubst %.xpfm, % , $(shell basename $(DEVICE))))
BUILD_DIR := build_dir.$(TARGET).$(XSA)
TEMP_DIR := ./_x.$(TARGET).$(XSA).enc_$(USE_ENC)

CLFLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps  --jobs 16 \
--xp "param:compiler.userPostSysLinkTcl=$(SDA_PATH)/scripts/syn_tcl/link_post.tcl" \
--xp "vivado_prop:run.my_rm_synth_1.STEPS.SYNTH_DESIGN.TCL.PRE={$(SDA_PATH)/scripts/syn_tcl/synth_pre.tcl}" \
--xp "vivado_prop:run.my_rm_synth_1.STEPS.SYNTH_DESIGN.TCL.POST={$(SDA_PATH)/scripts/syn_tcl/synth_post.tcl}" \
--xp "vivado_prop:run.impl_1.STEPS.OPT_DESIGN.TCL.PRE={$(SDA_PATH)/scripts/syn_tcl/opt_pre.tcl}" \
--xp "vivado_prop:run.impl_1.STEPS.OPT_DESIGN.TCL.POST={$(SDA_PATH)/scripts/syn_tcl/opt_post.tcl}" \
--xp "vivado_prop:run.impl_1.STEPS.PLACE_DESIGN.TCL.PRE={$(SDA_PATH)/scripts/syn_tcl/place_pre.tcl}" \
--xp "vivado_prop:run.impl_1.STEPS.PLACE_DESIGN.TCL.POST={$(SDA_PATH)/scripts/syn_tcl/place_post.tcl}" \
--xp "vivado_prop:run.impl_1.STEPS.ROUTE_DESIGN.TCL.PRE={$(SDA_PATH)/scripts/syn_tcl/route_pre.tcl}" \
--xp "vivado_prop:run.impl_1.STEPS.ROUTE_DESIGN.TCL.POST={$(SDA_PATH)/scripts/syn_tcl/route_post.tcl}" \
\
--xp "vivado_prop:run.impl_1.STEPS.OPT_DESIGN.ARGS.DIRECTIVE=Explore" \
--xp "vivado_prop:run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=$(PLACE_DIRECTIVE)" \
--xp "vivado_prop:run.impl_1.STEPS.PLACE_DESIGN.ARGS.MORE OPTIONS=-debug_log" \
--xp "vivado_prop:run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=AggressiveExplore" \
--xp "vivado_prop:run.impl_1.STEPS.ROUTE_DESIGN.ARGS.MORE OPTIONS=-tns_cleanup" \
--xp "vivado_prop:run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=$(ROUTE_DIRECTIVE)" \
--xp "vivado_prop:run.impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE=AggressiveExplore" \
\
--xp "param:compiler.interfaceWrBurstLen=16" \
--xp "param:compiler.interfaceRdBurstLen=16" \
--xp "param:compiler.interfaceWrOutstanding=32" \
--xp "param:compiler.interfaceRdOutstanding=32" \
\
--xp "param:compiler.worstNegativeSlack=-2"


CLFLAGS += --kernel_frequency "0:300|1:300"


ifeq ($(DPU_NUM),1)
CLFLAGS_SLR += \
--connectivity.nk DPUCADF8H:1:DPUCADF8H_1
CLFLAGS_SLR += \
--connectivity.slr DPUCADF8H_1:SLR$(word 1,$(DPU_SLR_IDX))   \
--connectivity.sp DPUCADF8H_1.m00_axi:DDR[$(word 1,$(DPU_DDR_IDX))]
endif

ifeq ($(DPU_NUM),2)
CLFLAGS_SLR += \
--connectivity.nk DPUCADF8H:2:DPUCADF8H_1.DPUCADF8H_2
CLFLAGS_SLR += \
--connectivity.slr DPUCADF8H_1:SLR$(word 1,$(DPU_SLR_IDX)) \
--connectivity.slr DPUCADF8H_2:SLR$(word 2,$(DPU_SLR_IDX))   \
--connectivity.sp DPUCADF8H_1.m00_axi:DDR[$(word 1,$(DPU_DDR_IDX))] \
--connectivity.sp DPUCADF8H_2.m00_axi:DDR[$(word 2,$(DPU_DDR_IDX))]
endif

ifeq ($(DPU_NUM),3)
CLFLAGS_SLR += \
--connectivity.nk DPUCADF8H:3:DPUCADF8H_1.DPUCADF8H_2.DPUCADF8H_3  \
--connectivity.slr DPUCADF8H_1:SLR$(word 1,$(DPU_SLR_IDX))  \
--connectivity.slr DPUCADF8H_2:SLR$(word 2,$(DPU_SLR_IDX))  \
--connectivity.slr DPUCADF8H_3:SLR$(word 3,$(DPU_SLR_IDX))  \
--connectivity.sp DPUCADF8H_1.m00_axi:DDR[$(word 1,$(DPU_DDR_IDX))] \
--connectivity.sp DPUCADF8H_2.m00_axi:DDR[$(word 2,$(DPU_DDR_IDX))] \
--connectivity.sp DPUCADF8H_3.m00_axi:DDR[$(word 3,$(DPU_DDR_IDX))]
endif

ifeq ($(DPU_NUM),4)
CLFLAGS_SLR += \
--connectivity.nk DPUCADF8H:4:DPUCADF8H_1.DPUCADF8H_2.DPUCADF8H_3.DPUCADF8H_4
CLFLAGS_SLR += \
--connectivity.slr DPUCADF8H_1:SLR$(word 1,$(DPU_SLR_IDX)) \
--connectivity.slr DPUCADF8H_2:SLR$(word 2,$(DPU_SLR_IDX)) \
--connectivity.slr DPUCADF8H_3:SLR$(word 3,$(DPU_SLR_IDX)) \
--connectivity.slr DPUCADF8H_4:SLR$(word 4,$(DPU_SLR_IDX)) \
--connectivity.sp DPUCADF8H_1.m00_axi:DDR[$(word 1,$(DPU_DDR_IDX))] \
--connectivity.sp DPUCADF8H_2.m00_axi:DDR[$(word 2,$(DPU_DDR_IDX))] \
--connectivity.sp DPUCADF8H_3.m00_axi:DDR[$(word 3,$(DPU_DDR_IDX))] \
--connectivity.sp DPUCADF8H_4.m00_axi:DDR[$(word 4,$(DPU_DDR_IDX))]
endif


ifneq ($(TARGET), hw)
	CLFLAGS += -g
endif


ifeq ($(STEP), 1)
	CLFLAGS += --from_step vpl.impl
endif
     

CMD_ARGS = $(BUILD_DIR)/dpdpuv3_wrapper.$(TARGET).$(XSA).xclbin
EMCONFIG_DIR = $(TEMP_DIR)
EMU_DIR = $(SDCARD)/data/emulation

BINARY_CONTAINERS += $(BUILD_DIR)/dpdpuv3_wrapper.$(TARGET).$(XSA).xclbin
BINARY_CONTAINER_dpdpuv3_wrapper_OBJS += $(TEMP_DIR)/DPUCADF8H.xo

VIVADO := $(XILINX_VIVADO)/bin/vivado
SOURCE_TCL := scripts/syn_tcl/gen_xo.tcl
$(TEMP_DIR)/DPUCADF8H.xo: src/kernel.xml scripts/syn_tcl/package_kernel.tcl scripts/syn_tcl/gen_xo.tcl
	mkdir -p $(TEMP_DIR)
	$(VIVADO) -mode batch -source $(SOURCE_TCL) -tclargs $(TEMP_DIR)/DPUCADF8H.xo dpdpuv3_wrapper $(TARGET) $(XSA)


CP = cp -rf

.PHONY: all clean cleanall docs emconfig all_step xo_rtl gen_xo gen_xclbin

######################user make option start#################################
gen_xo:  clean  user_set $(BINARY_CONTAINER_dpdpuv3_wrapper_OBJS)
gen_xclbin: clean user_set $(BINARY_CONTAINER_dpdpuv3_wrapper_OBJS) $(BINARY_CONTAINERS) save_xclbin emconfig
######################user make option end###################################


.PHONY: build
build: $(BINARY_CONTAINERS)

.PHONY: user_set
user_set:
	sed -i "s/set BOARD.*/set BOARD \"$(BOARD)\"/"                  $(SDA_PATH)/scripts/user_setup/env_config.tcl; \
	sed -i "s/set PLATFORM.*/set PLATFORM \"$(PLATFORM)\"/"      $(SDA_PATH)/scripts/user_setup/env_config.tcl; \
	sed -i "s/set FREQ.*/set FREQ $(FREQ)/"                         $(SDA_PATH)/scripts/user_setup/env_config.tcl; \
	sed -i "s/set DPU_NUM.*/set DPU_NUM $(DPU_NUM)/"            	$(SDA_PATH)/scripts/user_setup/env_config.tcl; \
	sed -i "/DPU_SLR_IDX/d"    $(SDA_PATH)/scripts/user_setup/env_config.tcl; \
	if [ $(DPU_NUM) = 1 ];then \
        sed -i "/DPU_NUM/i set DPU_SLR_IDX\(0\)   $(word 1,$(DPU_SLR_IDX))"    $(SDA_PATH)/scripts/user_setup/env_config.tcl; \
	fi
	if [ $(DPU_NUM) = 2 ];then \
        sed -i "/DPU_NUM/i set DPU_SLR_IDX\(0\)   $(word 1,$(DPU_SLR_IDX))"    $(SDA_PATH)/scripts/user_setup/env_config.tcl; \
        sed -i "/DPU_NUM/i set DPU_SLR_IDX\(1\)   $(word 2,$(DPU_SLR_IDX))"    $(SDA_PATH)/scripts/user_setup/env_config.tcl; \
	fi
	if [ $(DPU_NUM) = 3 ];then \
        sed -i "/DPU_NUM/i set DPU_SLR_IDX\(0\)   $(word 1,$(DPU_SLR_IDX))"    $(SDA_PATH)/scripts/user_setup/env_config.tcl; \
        sed -i "/DPU_NUM/i set DPU_SLR_IDX\(1\)   $(word 2,$(DPU_SLR_IDX))"    $(SDA_PATH)/scripts/user_setup/env_config.tcl; \
        sed -i "/DPU_NUM/i set DPU_SLR_IDX\(2\)   $(word 3,$(DPU_SLR_IDX))"    $(SDA_PATH)/scripts/user_setup/env_config.tcl; \
	fi
	if [ $(DPU_NUM) = 4 ];then \
        sed -i "/DPU_NUM/i set DPU_SLR_IDX\(0\)   $(word 1,$(DPU_SLR_IDX))"    $(SDA_PATH)/scripts/user_setup/env_config.tcl; \
        sed -i "/DPU_NUM/i set DPU_SLR_IDX\(1\)   $(word 2,$(DPU_SLR_IDX))"    $(SDA_PATH)/scripts/user_setup/env_config.tcl; \
        sed -i "/DPU_NUM/i set DPU_SLR_IDX\(2\)   $(word 3,$(DPU_SLR_IDX))"    $(SDA_PATH)/scripts/user_setup/env_config.tcl; \
        sed -i "/DPU_NUM/i set DPU_SLR_IDX\(3\)   $(word 4,$(DPU_SLR_IDX))"    $(SDA_PATH)/scripts/user_setup/env_config.tcl; \
	fi
	sed -i "s/set BUILD_DIR.*//g"									$(SDA_PATH)/scripts/user_setup/env_config.tcl; \
	echo -n "set BUILD_DIR $(SDA_PATH)/$(BUILD_DIR)" >> 			$(SDA_PATH)/scripts/user_setup/env_config.tcl; \
	\
	sed -i "s/set USE_ENC .*/set USE_ENC \"$(USE_ENC)\"/"     $(SDA_PATH)/scripts/syn_tcl/package_kernel.tcl; \
	echo -n "BOARD $(BOARD) \n" >> 			                        $(SDA_PATH)/outputs/gitversion; \
	echo -n "PLATFORM  $(PLATFORM) \n" >> 			            $(SDA_PATH)/outputs/gitversion; \
	echo -n "FREQ $(FREQ) \n" >> 			                        $(SDA_PATH)/outputs/gitversion; \
	echo -n "DPU_NUM  $(DPU_NUM) \n" >> 			                $(SDA_PATH)/outputs/gitversion; \
	echo -n "DEBUG   $(DEBUG) \n" >> 			                    $(SDA_PATH)/outputs/gitversion; \
	echo -n "USE_ENC  $(USE_ENC) \n" >> 			                        $(SDA_PATH)/outputs/gitversion; \
	echo -n "STEP  $(STEP) \n" >> 			                        $(SDA_PATH)/outputs/gitversion; \




$(BUILD_DIR)/dpdpuv3_wrapper.$(TARGET).$(XSA).xclbin:
	mkdir -p $(BUILD_DIR)
	$(VPP) $(CLFLAGS)  $(CLFLAGS_SLR)  --temp_dir $(BUILD_DIR) -l  -o'$@' $(BINARY_CONTAINER_dpdpuv3_wrapper_OBJS)


emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od .


.PHONY: save_xclbin
save_xclbin: 
	$(CP) $(SDA_PATH)/$(BUILD_DIR)/dpdpuv3_wrapper.$(TARGET).$(XSA).* $(SDA_PATH)/outputs/xclbin/ 

clean: 
	-rm -rf  profile_* TempConfig system_estimate.xtxt *.rpt *.csv 
	-rm -rf  src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb
	-rm -rf  build_dir* sd_card*
	-rm -rf  _x* *xclbin.run_summary qemu-memory-_* emulation/ _vimage/ pl* start_simulation.sh *.xclbin xclbin sample_link.ini
	-rm -rf  ./tmp_kernel_pack* ./packaged_kernel* 
	-rm -rf  ./outputs/checkpoints/* ./outputs/logs/* ./outputs/reports/* ./outputs/xclbin/*


