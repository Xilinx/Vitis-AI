.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> BOARD=<BOARD> HOST_ARCH=<aarch32/aarch64/x86> SYSROOT=<sysroot_path>"
	$(ECHO) "      Command to generate the design for specified Target and Shell."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and SYSROOT is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO) "  make sd_card TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> SYSROOT=<sysroot_path>"
	$(ECHO) "      Command to prepare sd_card files."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and SYSROOT is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make check TARGET=<sw_emu/hw_emu/hw> BOARD=<BOARD> HOST_ARCH=<aarch32/aarch64/x86> SYSROOT=<sysroot_path>"
	$(ECHO) "      Command to run application in emulation."
	sed -i "s/set VIVADO_VER.*/set VIVADO_VER \"$(VIVADO_VER)\"/"   $(SDA_PATH)/scripts/user_setup/env_config.tcl; \
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and SYSROOT is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  Supported options:"
	$(ECHO) "     <Board>: "
	$(ECHO) "       u200"
	$(ECHO) "       u250"
	$(ECHO) "     <Vivado>: "
	$(ECHO) "       201802 (for vivado_2018.2.xdf)"
	$(ECHO) "       201901 (for vivado_2019.1) "
	$(ECHO) "     <Shell>: "
	$(ECHO) "       201802 (for xilinx_u200_xdma-gen3x4_201820_2 or xilinx_u250_xdma_201820_1)"
	$(ECHO) "       201803 (for xilinx_u200_xdma-gen3x4_201830_2 or xilinx_u250_xdma_201830_1)"
	$(ECHO) ""
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> SYSROOT=<sysroot_path>"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and SYSROOT is required for SoC shells"
	$(ECHO) ""


##################################
TARGET := hw
HOST_ARCH := x86
SYSROOT := 
#DEVICE = xilinx_u200_xdma_201920_1

include ./scripts/utility/utils.mk

# Points to Utility Directory
COMMON_REPO = ./scripts
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))

PWD_PATH =$(shell pwd)
SDA_PATH =$(PWD_PATH)

##################################
## User settings

# Supported board list: 
# u200 | u250
BOARD = u200

# Supported vivado list: 
# 201902 | 202002 | 202101
VIVADO_VER = 202002

# Supported shell list: 
# 201803 | 202002
SHELL_VER = 202002

# Supported freq list: 
# 50 | 400 | 500 | 600 | 666 
FREQ = 600

# Supported DPU_NUM list: 
# u200: 1 | 2
# u250: 1 | 2 | 3 | 4
DPU_NUM = 2

# Supported SLR list: 
# u200: 0 | 1 | 2 | 3
# u250: 0 | 1 | 2 | 3
SLR = 0

# Insert AXI monitor and Protocol checker
# False | True
DEBUG = Fasle

# true | false
ENC := false
# 1 | 2
ENC_VERSION := 1

# true | false
XO_INCLUDE_XDC:= false

STEP = 0
# Supported MSS_STRATEGY list: 
# default | MinBRAM
MSS_STRATEGY = default

ifeq ($(BOARD), u200)
    PLACE_DIRECTIVE = AltSpreadLogic_high
else 
    PLACE_DIRECTIVE = AltSpreadLogic_high
endif

ROUTE_DIRECTIVE = NoTimingRelaxation
##################################
# Context process
#DEVICE := $(DEVICES)
DEVICE := $(SDX_PLATFORM)
TARGETS := hw
TARGET := $(TARGETS)
VPP := v++
SDCARD := sd_card
XSA := $(call device2xsa, $(DEVICE))
BUILD_DIR := build_dir.$(TARGET).$(XSA)
TEMP_DIR := ./_x.$(TARGET).$(XSA).enc_$(ENC).xdc_$(XO_INCLUDE_XDC)

##################################
# Kernel compiler global settings
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


ifeq ($(SHELL_VER), aws)
CLFLAGS += --kernel_frequency "0:250|1:250"
else
CLFLAGS += --kernel_frequency "0:300|1:300"
endif

ifeq ($(SHELL_VER), microsoft)
CLFLAGS += --advanced.param compiler.acceleratorBinaryContent=dcp
endif

ifeq ($(DPU_NUM), 1)
ifeq ($(SHELL_VER), aws)
CLFLAGS_SLR += \
--connectivity.nk DPUCADF8H:1:DPUCADF8H_1\
--connectivity.slr DPUCADF8H_1:SLR2 \
--connectivity.sp DPUCADF8H_1.m00_axi:DDR[1]
DBG_CHIPSCOPE += \
--dk chipscope:DPUCADF8H_1
else
CLFLAGS_SLR += \
--connectivity.nk DPUCADF8H:1:DPUCADF8H_1\
--connectivity.slr DPUCADF8H_1:SLR$(SLR) \
--connectivity.sp DPUCADF8H_1.m00_axi:DDR[$(SLR)]
DBG_CHIPSCOPE += \
--dk chipscope:DPUCADF8H_1
endif
else ifeq ($(DPU_NUM), 2)
ifeq ($(BOARD), u200)
CLFLAGS_SLR += \
--connectivity.nk DPUCADF8H:2:DPUCADF8H_1.DPUCADF8H_2 \
--connectivity.slr DPUCADF8H_1:SLR0 \
--connectivity.slr DPUCADF8H_2:SLR2 \
--connectivity.sp DPUCADF8H_1.m00_axi:DDR[1] \
--connectivity.sp DPUCADF8H_2.m00_axi:DDR[2] \
--connectivity.nk blobfromimage_accel:1:blobfromimage_accel_1 \
--connectivity.slr blobfromimage_accel_1:SLR1 \
--connectivity.sp blobfromimage_accel_1.Y_OR_RGB_img_inp:DDR[1:2] \
--connectivity.sp blobfromimage_accel_1.U_img_inp:DDR[1:2] \
--connectivity.sp blobfromimage_accel_1.V_img_inp:DDR[1:2] \
--connectivity.sp blobfromimage_accel_1.img_out:DDR[1:2] \
--advanced.param compiler.worstNegativeSlack=-1 \

DBG_CHIPSCOPE += \
--dk chipscope:DPUCADF8H_1 \
--dk chipscope:DPUCADF8H_2
endif
ifeq ($(BOARD), u250)
ifeq ($(SHELL_VER), 201803)
CLFLAGS_SLR += \
--connectivity.nk DPUCADF8H:2:DPUCADF8H_1.DPUCADF8H_2 \
--connectivity.slr DPUCADF8H_1:SLR0 \
--connectivity.slr DPUCADF8H_2:SLR1 \
--connectivity.sp DPUCADF8H_1.m00_axi:DDR[0] \
--connectivity.sp DPUCADF8H_2.m00_axi:DDR[1]
DBG_CHIPSCOPE += \
--dk chipscope:DPUCADF8H_1 \
--dk chipscope:DPUCADF8H_2
endif
ifeq ($(SHELL_VER), 202002)
CLFLAGS_SLR += \
--connectivity.nk DPUCADF8H:2:DPUCADF8H_1.DPUCADF8H_2 \
--connectivity.slr DPUCADF8H_1:SLR0 \
--connectivity.slr DPUCADF8H_2:SLR2 \
--connectivity.sp DPUCADF8H_1.m00_axi:DDR[0] \
--connectivity.sp DPUCADF8H_2.m00_axi:DDR[2]
DBG_CHIPSCOPE += \
--dk chipscope:DPUCADF8H_1 \
--dk chipscope:DPUCADF8H_2
endif
ifeq ($(SHELL_VER), microsoft)
CLFLAGS_SLR += \
--connectivity.nk DPUCADF8H:2:DPUCADF8H_1.DPUCADF8H_2 \
--connectivity.slr DPUCADF8H_1:SLR0 \
--connectivity.slr DPUCADF8H_2:SLR2 \
--connectivity.sp DPUCADF8H_1.m00_axi:DDR[0] \
--connectivity.sp DPUCADF8H_2.m00_axi:DDR[2]
DBG_CHIPSCOPE += \
--dk chipscope:DPUCADF8H_1 \
--dk chipscope:DPUCADF8H_2
endif
endif
else ifeq ($(DPU_NUM), 3)
ifeq ($(SHELL_VER), 201803)
CLFLAGS_SLR += \
--connectivity.nk DPUCADF8H:3:DPUCADF8H_1.DPUCADF8H_2.DPUCADF8H_3  \
--connectivity.slr DPUCADF8H_1:SLR0 \
--connectivity.slr DPUCADF8H_2:SLR1 \
--connectivity.slr DPUCADF8H_3:SLR2 \
--connectivity.sp DPUCADF8H_1.m00_axi:DDR[0] \
--connectivity.sp DPUCADF8H_2.m00_axi:DDR[1] \
--connectivity.sp DPUCADF8H_3.m00_axi:DDR[2]
DBG_CHIPSCOPE += \
--dk chipscope:DPUCADF8H_1 \
--dk chipscope:DPUCADF8H_2 \
--dk chipscope:DPUCADF8H_3
endif
ifeq ($(SHELL_VER), 202002)
CLFLAGS_SLR += \
--connectivity.nk DPUCADF8H:3:DPUCADF8H_1.DPUCADF8H_2.DPUCADF8H_3  \
--connectivity.slr DPUCADF8H_1:SLR0 \
--connectivity.slr DPUCADF8H_2:SLR2 \
--connectivity.slr DPUCADF8H_3:SLR3 \
--connectivity.sp DPUCADF8H_1.m00_axi:DDR[0] \
--connectivity.sp DPUCADF8H_2.m00_axi:DDR[2] \
--connectivity.sp DPUCADF8H_3.m00_axi:DDR[3]
DBG_CHIPSCOPE += \
--dk chipscope:DPUCADF8H_1 \
--dk chipscope:DPUCADF8H_2 \
--dk chipscope:DPUCADF8H_3
endif
ifeq ($(SHELL_VER), microsoft)
CLFLAGS_SLR += \
--connectivity.nk DPUCADF8H:3:DPUCADF8H_1.DPUCADF8H_2.DPUCADF8H_3  \
--connectivity.slr DPUCADF8H_1:SLR0 \
--connectivity.slr DPUCADF8H_2:SLR2 \
--connectivity.slr DPUCADF8H_3:SLR3 \
--connectivity.sp DPUCADF8H_1.m00_axi:DDR[0] \
--connectivity.sp DPUCADF8H_2.m00_axi:DDR[2] \
--connectivity.sp DPUCADF8H_3.m00_axi:DDR[3]
DBG_CHIPSCOPE += \
--dk chipscope:DPUCADF8H_1 \
--dk chipscope:DPUCADF8H_2 \
--dk chipscope:DPUCADF8H_3
endif
else ifeq ($(DPU_NUM), 4)
CLFLAGS_SLR += \
--connectivity.nk DPUCADF8H:4:DPUCADF8H_1.DPUCADF8H_2.DPUCADF8H_3.DPUCADF8H_4  \
--connectivity.slr DPUCADF8H_1:SLR0 \
--connectivity.slr DPUCADF8H_2:SLR1 \
--connectivity.slr DPUCADF8H_3:SLR2 \
--connectivity.slr DPUCADF8H_4:SLR3 \
--connectivity.sp DPUCADF8H_1.m00_axi:DDR[0] \
--connectivity.sp DPUCADF8H_2.m00_axi:DDR[1] \
--connectivity.sp DPUCADF8H_3.m00_axi:DDR[2] \
--connectivity.sp DPUCADF8H_4.m00_axi:DDR[3]
DBG_CHIPSCOPE += \
--dk chipscope:DPUCADF8H_1 \
--dk chipscope:DPUCADF8H_2 \
--dk chipscope:DPUCADF8H_3 \
--dk chipscope:DPUCADF8H_4
endif


##################################
# Debug config
DBG_CHECKER = --dk protocol:all:all
DBG_PROFILE = --profile_kernel data:all:all:all:all
ifeq ($(DEBUG), True)
CLFLAGS_DBG += $(DBG_CHECKER)
else
CLFLAGS_DBG += 
endif


XOCC := $(XILINX_SDX)/bin/xocc

#Include Libraries
include $(ABS_COMMON_REPO)/libs/opencl/opencl.mk
include $(ABS_COMMON_REPO)/libs/xcl2/xcl2.mk
CXXFLAGS += $(xcl2_CXXFLAGS)
LDFLAGS += $(xcl2_LDFLAGS)
HOST_SRCS += $(xcl2_SRCS)

CXXFLAGS += $(opencl_CXXFLAGS) -Wall -O0 -g -std=c++11
LDFLAGS += $(opencl_LDFLAGS)

#HOST_SRCS += src/host.cpp


# Host compiler global settings
CXXFLAGS += -fmessage-length=0
CXXFLAGS += -I$(ABS_COMMON_REPO)/libs/userlib
CXXFLAGS +=  -lxilinxopencl
LDFLAGS += -lrt -lstdc++ 

ifneq ($(HOST_ARCH), x86)
	LDFLAGS += --sysroot=$(SYSROOT)
endif

# Kernel compiler global settings
ifneq ($(TARGET), hw)
	CLFLAGS += -g
endif

ifeq ($(STEP), 1)
	CLFLAGS += --from_step vpl.impl
endif
     

EXECUTABLE = host
CMD_ARGS = $(BUILD_DIR)/dpdpuv3_wrapper.$(TARGET).$(XSA).xclbin
EMCONFIG_DIR = $(TEMP_DIR)
EMU_DIR = $(SDCARD)/data/emulation

BINARY_CONTAINERS += $(BUILD_DIR)/dpdpuv3_wrapper.$(TARGET).$(XSA).xclbin
BINARY_CONTAINER_dpdpuv3_wrapper_OBJS += $(DPU_XO)/DPUCADF8H.xo $(blob_ACCEL)/blobfromimage_accel.xo

##################################
VIVADO := $(XILINX_VIVADO)/bin/vivado
##################################
CP = cp -rf

.PHONY: all clean cleanall docs emconfig
all: clean $(BINARY_CONTAINER_dpdpuv3_wrapper_OBJS) $(BINARY_CONTAINERS) ch_ctrl_mod save_xclbin emconfig 
all_step: clean_xclbin $(BINARY_CONTAINERS) ch_ctrl_mod save_xclbin emconfig
.PHONY: save_xclbin
save_xclbin: 
	$(CP) $(SDA_PATH)/$(BUILD_DIR)/dpdpuv3_wrapper.$(TARGET).$(XSA).* $(SDA_PATH)/outputs/xclbin/ 
	
.PHONY:ch_ctrl_mod 
ch_ctrl_mod : 
	rm -f IP_LAYOUT.json
	cp -f $(BINARY_CONTAINERS)  $(BINARY_CONTAINERS)_HS
	$(XILINX_XRT)/bin/xclbinutil --dump-section    IP_LAYOUT:JSON:IP_LAYOUT.json -i $(BINARY_CONTAINERS)_HS
	sed -i "s/AP_CTRL_HS/AP_CTRL_CHAIN/" IP_LAYOUT.json
	$(XILINX_XRT)/bin/xclbinutil --replace-section IP_LAYOUT:JSON:IP_LAYOUT.json -i $(BINARY_CONTAINERS)_HS  -o  $(BINARY_CONTAINERS)_CHAIN
	mv -f $(BINARY_CONTAINERS)_CHAIN  $(BINARY_CONTAINERS)
	
.PHONY: build
build: $(BINARY_CONTAINERS)

.PHONY: report_route
report_route:
	$(VIVADO) -mode tcl -verbose -source ./scripts/syn_tcl/report_route.tcl

# Building kernel
$(BUILD_DIR)/dpdpuv3_wrapper.$(TARGET).$(XSA).xclbin:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(SDA_PATH)/outputs
	mkdir -p $(SDA_PATH)/outputs/checkpoints
	mkdir -p $(SDA_PATH)/outputs/logs
	mkdir -p $(SDA_PATH)/outputs/reports
	mkdir -p $(SDA_PATH)/outputs/xclbin
	$(VPP) $(CLFLAGS)  $(CLFLAGS_SLR) $(CLFLAGS_DBG) --temp_dir $(BUILD_DIR) -l $(LDCLFLAGS) -o'$@' $(BINARY_CONTAINER_dpdpuv3_wrapper_OBJS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od .

# Cleaning stuff
clean: 
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv 
	-$(RMDIR) *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb
	-$(RMDIR) build_dir* sd_card*
	-$(RMDIR) _x* *xclbin.run_summary qemu-memory-_* emulation/ _vimage/ pl* start_simulation.sh *.xclbin xclbin sample_link.ini
	-$(RMDIR) ./tmp_kernel_pack* ./packaged_kernel* 
	-$(RMDIR) ./outputs/checkpoints/* ./outputs/logs/* ./outputs/reports/* ./outputs/xclbin/*

clean_xclbin:
	rm -f $(BUILD_DIR)/dpdpuv3_wrapper.$(TARGET).$(XSA).xclbin
