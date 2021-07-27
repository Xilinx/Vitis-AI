ifndef DIR_PRJ
 DIR_PRJ = $(shell pwd)
endif
ifndef TRD_PATH
 TRD_PATH = ../../
endif

VIVADO_ROOT := $(XILINX_VIVADO)

RM = rm -f
RMDIR = rm -rf

VIVADO:=${VIVADO_ROOT}/bin/vivado
MPSOC_CXX:=aarch64-linux-gnu-g++
TARGET := hw

#EDGE_COMMON_SW := /proj/rdi/staff/jiaz/common_rootfs/xilinx-zynqmp-common-v2020.2
#SDX_PLATFORM = /proj/xbuilds/2021.1_daily_latest/internal_platforms/xilinx_zcu102_base_202110_1/xilinx_zcu102_base_202110_1.xpfm

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

dpu_sm_xo = binary_container_1/dpu.xo binary_container_1/softmax.xo
dpu_xo = binary_container_1/dpu.xo


ifeq ($(KERNEL),DPU_SM)
kernel_xo = binary_container_1/dpu.xo binary_container_1/softmax.xo
else ifeq ($(KERNEL),DPU)
kernel_xo = binary_container_1/dpu.xo
else
kernel_xo = binary_container_1/dpu.xo
endif

.PHONY: all clean package

all : binary_container_1/dpu.xclbin package

binary_container_1/dpu.xo: $(dpu_HDLSRCS)
	@mkdir -p $(@D)
	-@$(RM) $@
	$(VIVADO) -mode batch -source $(dpu_TCL) -notrace -tclargs $@ $(DPU_KERN_NAME) ${TARGET} ${DEVICE}

binary_container_1/softmax.xo: $(softmax_HDLSRCS)
	@mkdir -p $(@D)
	-@$(RM) $@
	$(VIVADO) -mode batch -source $(softmax_TCL) -notrace -tclargs $@ $(SM_KERN_NAME) ${TARGET} ${DEVICE}


binary_container_1/dpu.xclbin: $(kernel_xo)
	v++ $(XOCC_OPTS) -l --temp_dir binary_container_1 --log_dir binary_container_1/logs --remote_ip_cache binary_container_1/ip_cache -o "$@" $(+)

package:
	v++ -t ${TARGET} --platform ${SDX_PLATFORM} -p binary_container_1/dpu.xclbin  -o dpu.xclbin --package.out_dir binary_container_1 --package.rootfs $(EDGE_COMMON_SW)/rootfs.ext4 --package.sd_file $(EDGE_COMMON_SW)/Image 
	cp ./binary_*/link/vivado/vpl/prj/prj*/sources_1/bd/*/hw_handoff/*.hwh ./binary_*/sd_card
	cp ./binary_*/link/vivado/vpl/prj/prj.gen/sources_1/bd/*/ip/*_DPUCZDX8G_1_0/arch.json ./binary_*/sd_card

clean:
	${RM} *.o *.elf *.log *.jou sample* v++* *.xclbin *.xclbin*
	${RMDIR} binary_container_1/ packaged_*/ tmp_*/ .Xil/ _x/

