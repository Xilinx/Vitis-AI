DIR_PRJ = $(shell pwd)
VIVADO_ROOT := $(XILINX_VIVADO)

RM = rm -f
RMDIR = rm -rf

VIVADO:=${VIVADO_ROOT}/bin/vivado
MPSOC_CXX:=aarch64-linux-gnu-g++
TARGET := hw

#EDGE_COMMON_SW := /proj/rdi/staff/jiaz/common_rootfs/xilinx-zynqmp-common-v2020.1
#SDX_PLATFORM = /proj/xbuilds/2020.1_0603_1/internal_platforms/xilinx_zcu102_base_202010_1/xilinx_zcu102_base_202010_1.xpfm

XOCC_OPTS = -t ${TARGET} --platform ${SDX_PLATFORM} --save-temps --config ${DIR_PRJ}/config_file/prj_config --config $(PP_ACCEL)/config_file/prj_config --xp param:compiler.userPostSysLinkOverlayTcl=${DIR_PRJ}/syslink/strip_interconnects.tcl 
dpu_HDLSRCS=kernel_xml/dpu/kernel.xml\
	     scripts/package_dpu_kernel.tcl\
	     scripts/gen_dpu_xo.tcl\
	     ./dpu_conf.vh\
	     $(DPU_IP)/Vitis/dpu/hdl/DPUCZDX8G.v\
	     $(DPU_IP)/Vitis/dpu/inc/arch_def.vh\
	     $(DPU_IP)/Vitis/dpu/xdc/*.xdc\
	     $(DPU_IP)/DPUCZDX8G_*/hdl/DPUCZDX8G_*_dpu.sv\
	     $(DPU_IP)/DPUCZDX8G_*/inc/function.vh\
             $(DPU_IP)/DPUCZDX8G_*/inc/arch_para.vh


dpu_TCL=scripts/gen_dpu_xo.tcl

# Kernel name must match kernel name in kernel.xml
DPU_KERN_NAME = DPUCZDX8G

kernel_xo = binary_container_1/dpu.xo $(PP_ACCEL)/pp_pipeline_accel.xo

.PHONY: all clean package

all : binary_container_1/dpu.xclbin package

binary_container_1/dpu.xo: $(dpu_HDLSRCS)
	@mkdir -p $(@D)
	-@$(RM) $@
	$(VIVADO) -mode batch -source $(dpu_TCL) -tclargs $@ $(DPU_KERN_NAME) ${TARGET} ${DEVICE}


binary_container_1/dpu.xclbin: $(kernel_xo)
	v++ $(XOCC_OPTS) -l --temp_dir binary_container_1 --log_dir binary_container_1/logs --remote_ip_cache binary_container_1/ip_cache -o "$@" $(+)

package:
	v++ -t ${TARGET} --platform ${SDX_PLATFORM} -p binary_container_1/dpu.xclbin  -o dpu.xclbin --package.out_dir binary_container_1 --package.rootfs $(EDGE_COMMON_SW)/rootfs.ext4 --package.sd_file $(EDGE_COMMON_SW)/Image 
	cp ./binary_*/link/vivado/vpl/prj/prj*/sources_1/bd/*/hw_handoff/*.hwh ./binary_*/sd_card
	cp ./binary_*/link/vivado/vpl/prj/prj.gen/sources_1/bd/*/ip/*_DPUCZDX8G_1_0/arch.json ./binary_*/sd_card

clean:
	${RM} *.o *.elf *.log *.jou sample* v++* *.xclbin *.xclbin*
	${RMDIR} binary_container_1/ packaged_*/ tmp_*/ .Xil/ _x/

