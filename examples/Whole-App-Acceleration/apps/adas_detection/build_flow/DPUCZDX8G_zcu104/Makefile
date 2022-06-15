ifndef DIR_PRJ
 DIR_PRJ = $(shell pwd)
endif
ifndef TRD_PATH
 TRD_PATH = ./DPU-TRD/DPUCZDX8G/
endif

LOAD_PARALLER = 1
VIVADO_ROOT = $(XILINX_VIVADO)
RM = rm -f
RMDIR = rm -rf

VIVADO:=${VIVADO_ROOT}/bin/vivado
MPSOC_CXX:=aarch64-linux-gnu-g++
TARGET := hw

XOCC_OPTS = -t ${TARGET} --platform ${SDX_PLATFORM} --save-temps --config ${DIR_PRJ}/config_file/prj_config --xp param:compiler.userPostSysLinkOverlayTcl=${DIR_PRJ}/syslink/strip_interconnects.tcl 
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


softmax_HDLSRCs=${DIR_PRJ}/kernel_xml/sfm/kernel.xml\
               ${DIR_PRJ}/scripts/package_sfm_kernel.tcl\
               ${DIR_PRJ}/scripts/gen_sfm_xo.tcl\
		${TRD_PATH}/dpu_ip/Vitis/sfm/hdl/*.v\
		${TRD_PATH}/dpu_ip/DPUCZDX8G_*/hdl/DPUCZDX8G_*_sfm.sv\
       ${TRD_PATH}/dpu_ip/DPUCZDX8G_*/xci/sfm/fp_*/*.xci		


dpu_TCL=${DIR_PRJ}/scripts/gen_dpu_xo.tcl
softmax_TCL=${DIR_PRJ}/scripts/gen_sfm_xo.tcl

# Kernel name must match kernel name in kernel.xml
DPU_KERN_NAME = DPUCZDX8G
SM_KERN_NAME = sfm_xrt_top

BUILD_DIR = binary_container_1

dpu_sm_xo = $(BUILD_DIR)/dpu.xo $(BUILD_DIR)/softmax.xo
dpu_xo = $(BUIDL_DIR)/dpu.xo

ifeq ($(KERNEL),DPU_SM)
kernel_xo = $(BUILD_DIR)/dpu.xo $(BUILD_DIR)/softmax.xo
else ifeq ($(KERNEL),DPU)
kernel_xo = $(BUILD_DIR)/dpu.xo
else ifeq ($(KERNEL),DPU_PP)
kernel_xo = $(BUILD_DIR)/dpu.xo $(BLOB_ACCEL)/blobfromimage_accel.xo
else
kernel_xo = $(BUILD_DIR)/dpu.xo $(BLOB_ACCEL)/blobfromimage_accel.xo
endif

.PHONY: help
help:
	@echo ' Usage:'
	@echo ''
	@echo ' make all KERNEL=<DPU/DPU_SM> DEVICE=<suffix_name>'
	@echo '     DPU   , will only integrate DPU.'
	@echo '     DPU_SM, will integrate DPU + Softmax.'
	@echo '     suffix_name, set target name, such as zcu102, zcu104, zcu106, SOM'
	@echo ''
	@echo ' The default settings are a simple case for familiar with the flow of DPU TRD. Please refer the chapter 5.5 of README.md for the released Image.'
	@echo ''
.PHONY: xml
xml : 
	python scripts/kernel_xml.py ${LOAD_PARALLER}

.PHONY: all clean 

all : $(BUILD_DIR)/$(BUILD_DIR).xclbin package

$(BUILD_DIR)/dpu.xo: $(dpu_HDLSRCS)
	@mkdir -p $(@D)
	-@$(RM) $@
	$(VIVADO) -mode batch -source $(dpu_TCL) -notrace -tclargs $@ $(DPU_KERN_NAME) ${TARGET} ${DEVICE}

$(BUILD_DIR)/softmax.xo: $(softmax_HDLSRCS)
	@mkdir -p $(@D)
	-@$(RM) $@
	$(VIVADO) -mode batch -source $(softmax_TCL) -notrace -tclargs $@ $(SM_KERN_NAME) ${TARGET} ${DEVICE}


$(BUILD_DIR)/$(BUILD_DIR).xclbin: $(kernel_xo)
	v++ $(XOCC_OPTS) -l --temp_dir $(BUILD_DIR) --log_dir $(BUILD_DIR)/logs --remote_ip_cache $(BUILD_DIR)/ip_cache -o "$@" $(+)

package:
	v++ -t ${TARGET} --platform ${SDX_PLATFORM} -p $(BUILD_DIR)/$(BUILD_DIR).xclbin -o $(BUILD_DIR)/dpu.xclbin --package.out_dir $(BUILD_DIR) --package.rootfs $(EDGE_COMMON_SW)/rootfs.ext4 --package.sd_file $(EDGE_COMMON_SW)/Image 
	cp ./binary_*/link/vivado/vpl/prj/prj*/sources_1/bd/*/hw_handoff/*.hwh ./binary_*/sd_card
	cp ./binary_*/link/vivado/vpl/prj/prj.gen/sources_1/bd/*/ip/*_DPUCZDX8G_1_0/arch.json ./binary_*/sd_card

clean:
	${RM} *.o *.elf *.log *.jou sample* v++* *.xclbin *.xclbin*
	${RMDIR} binary_container_1/ packaged_*/ tmp_*/ .Xil/ _x/

