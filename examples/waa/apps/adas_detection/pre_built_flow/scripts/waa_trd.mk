# device, shell and DPU info
DEVICE ?= zcu102
PREBUILT_DPU ?= DPUCZDX8G
DPUver ?= DPUCZDX8G
BOARD ?= zynq

# set directory structure variables
DIR_PRJ := $(shell pwd)
ACCEL_DIR := $(shell readlink -f $(DIR_PRJ)/../../../../plugins)
PREBUILT_DPU_DIR := $(shell readlink -f ${DIR_PRJ}/../../../../../dsa/$(PREBUILT_DPU)-XO/xo_release)
PRE_IMPL_COMMON := $(shell readlink -f ${DIR_PRJ}/../scripts)
TRD_PATH := $(shell readlink -f ${DIR_PRJ}/DPU-TRD/DPUCZDX8G/)

# Xilinx Vivado and orhter Exec variables
VIVADO_ROOT := $(XILINX_VIVADO)
VIVADO:=${VIVADO_ROOT}/bin/vivado
MPSOC_CXX:=aarch64-linux-gnu-g++
RM = rm -f
RMDIR = rm -rf

# default target set to hardware
TARGET := hw

# Default flow - prebuilt; can also use full flow
FLOW ?= prebuilt

# Default output directory.
OUTPUT_DIR ?= _x_output

# setting the XOCC link options
XOCC_OPTS = -t ${TARGET} \
			--platform ${SDX_PLATFORM} \
			--save-temps \
			--config ${DIR_PRJ}/config_file/prj_config \
			--xp param:compiler.userPostSysLinkOverlayTcl=${DIR_PRJ}/config_file/sys_link_post.tcl \
			--xp param:compiler.skipTimingCheckAndFrequencyScaling=true \
			--xp param:compiler.enableAutoFrequencyScaling=false

ifeq ($(FLOW),prebuilt)
	XOCC_OPTS += --config $(OUTPUT_DIR)/scripts/pre-built_scripts.ini
endif

# DPU regeneration sources
dpu_HDLSRCS= ${DIR_PRJ}/kernel_xml/dpu/kernel.xml\
	     ${DIR_PRJ}/scripts/package_dpu_kernel.tcl\
	     ${DIR_PRJ}/scripts/gen_dpu_xo.tcl\
	     ${DIR_PRJ}/dpu_conf.vh\
	     ${TRD_PATH}/dpu_ip/Vitis/dpu/hdl/DPUCZDX8G.v\
	     ${TRD_PATH}/dpu_ip/Vitis/dpu/inc/arch_def.vh\
	     ${TRD_PATH}/dpu_ip/Vitis/dpu/xdc/*.xdc\
	     ${TRD_PATH}/dpu_ip/DPUCZDX8G_*/hdl/DPUCZDX8G_*_dpu.sv\
	     ${TRD_PATH}/dpu_ip/DPUCZDX8G_*/inc/function.vh\
         ${TRD_PATH}/dpu_ip/DPUCZDX8G_*/inc/arch_para.vh

dpu_TCL=${DIR_PRJ}/scripts/gen_dpu_xo.tcl

# Kernel name must match kernel name in kernel.xml
DPU_KERN_NAME = DPUCZDX8G

ifeq ($(DPUver), DPUCZDX8G)
DPU_XO = xo_release/dpu.xo
else
DPU_XO = xo_release/$(DPUver).xo
endif
$(ACCEL)_XO = $(ACCEL_DIR)/$(ACCEL)/pl/$(ACCEL)_accel.xo
JPEG_XO = $(ACCEL_DIR)/jpeg_decoder/pl/jpeg_decoder.xo

KERNEL_XO = $(DPU_XO)
KERNEL_XO += $($(ACCEL)_XO)

ifeq ($(ADDON),JPEG)
KERNEL_XO += $(JPEG_XO)
endif

.PHONY: all clean package

all : dpu.xclbin package

dpu.xclbin : $(OUTPUT_DIR)/dpu.xclbin

xo_release/dpu.xo: $(dpu_HDLSRCS)
	@echo -e "----\nGenerating DPU XO $@..."
	@mkdir -p $(@D)
	-@$(RM) $@
	$(VIVADO) -mode batch -source $(dpu_TCL) -notrace -tclargs $@ $(DPU_KERN_NAME) ${TARGET} ${DEVICE}

$($(ACCEL)_XO):
	@echo -e "----\nGenerating ACCEL $(ACCEL) XO $@..."
ifeq ($(DPUver), DPUCZDX8G)
	$(MAKE) -C $(ACCEL_DIR)/$(ACCEL)/pl xo TARGET=hw BOARD=Zynq ARCH=aarch64 BLOB_CHANNEL_SWAP_EN=1 BLOB_CROP_EN=0 BLOB_LETTERBOX_EN=1 BLOB_JPEG_EN=0 BLOB_NPC=4
else ifeq ($(DPUver), DPUCAHX8H_3ENGINE)
	$(MAKE) -C $(ACCEL_DIR)/$(ACCEL)/pl xo TARGET=hw BLOB_CHANNEL_SWAP_EN=1 BLOB_CROP_EN=0 BLOB_LETTERBOX_EN=1 BLOB_JPEG_EN=0 BLOB_NPC=4
endif

$(OUTPUT_DIR)/dpu.xclbin: $(KERNEL_XO)
	mkdir -p $(OUTPUT_DIR)/scripts
	cp -rf $(DIR_PRJ)/pre-built_info.json $(OUTPUT_DIR)/scripts/
	cp -rf $(PRE_IMPL_COMMON)/*.tcl $(PRE_IMPL_COMMON)/pre-built_scripts.ini $(OUTPUT_DIR)/scripts/
	sed -i 's#\("Path": \)"\S\+"#\1"${DIR_PRJ}"#g' $(OUTPUT_DIR)/scripts/pre-built_info.json
	sed -i "s#RUNDIR_CUSTOM#$(DIR_PRJ)/$(OUTPUT_DIR)/scripts#g" $(OUTPUT_DIR)/scripts/pre-built_scripts.ini $(OUTPUT_DIR)/scripts/*.tcl
	v++ $(XOCC_OPTS) -l --temp_dir $(OUTPUT_DIR) --log_dir $(OUTPUT_DIR)/logs --remote_ip_cache $(OUTPUT_DIR)/ip_cache -o "$@" $(+) -R 2

package:
ifeq ($(DPUver), DPUCZDX8G)
	v++ -t ${TARGET} --platform ${SDX_PLATFORM} -p $(OUTPUT_DIR)/dpu.xclbin -o dpu.xclbin --package.out_dir $(OUTPUT_DIR) --package.rootfs $(EDGE_COMMON_SW)/rootfs.ext4 --package.sd_file $(EDGE_COMMON_SW)/Image
	cp ./${OUTPUT_DIR}/link/vivado/vpl/prj/prj*/sources_1/bd/*/hw_handoff/*.hwh ./${OUTPUT_DIR}/sd_card
	#cp ./${OUTPUT_DIR}/link/vivado/vpl/prj/prj.gen/sources_1/bd/*/ip/*_DPUCZDX8G_1_0/arch.json ./${OUTPUT_DIR}/sd_card
	cp $(DIR_PRJ)/config_file/arch.json ./${OUTPUT_DIR}/sd_card
endif

clean:
	$(MAKE) -C $(ACCEL_DIR)/$(ACCEL)/pl cleanall
	${RM} *.o *.elf *.log *.jou sample* v++*
	${RMDIR} $(OUTPUT_DIR)/ packaged_*/ tmp_*/ .Xil/ _x/ xo_release/

