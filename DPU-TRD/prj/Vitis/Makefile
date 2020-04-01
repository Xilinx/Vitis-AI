DIR_PRJ = $(shell pwd)
VIVADO_ROOT := $(XILINX_VIVADO)

RM = rm -f
RMDIR = rm -rf

VIVADO:=${VIVADO_ROOT}/bin/vivado
MPSOC_CXX:=aarch64-linux-gnu-g++

TARGET := hw

#SDX_PLATFORM = /proj/rdi/staff/jiaz/scout/platform_trd32/zcu102_0205/zcu102_dpu.xpfm
XOCC_OPTS = -t ${TARGET} --platform ${SDX_PLATFORM} --save-temps --config ${DIR_PRJ}/config_file/prj_config --xp param:compiler.userPostSysLinkTcl=${DIR_PRJ}/strip_interconnects.tcl 
dpu_HDLSRCS=kernel_xml/dpu/kernel.xml\
	     scripts/package_dpu_kernel.tcl\
	     scripts/gen_dpu_xo.tcl\
	     ./dpu_conf.vh\
	     ../../dpu_ip/Vitis/dpu/hdl/dpu_xrt_top.v\
	     ../../dpu_ip/Vitis/dpu/inc/arch_def.vh\
	     ../../dpu_ip/Vitis/dpu/xdc/*.xdc\
	     ../../dpu_ip/dpu_eu_*/hdl/dpu_eu_*_dpu.sv\
	     ../../dpu_ip/dpu_eu_*/inc/function.vh\
             ../../dpu_ip/dpu_eu_*/inc/arch_para.vh


softmax_HDLSRCs=kernel_xml/sfm/kernel.xml\
                scripts/package_sfm_kernel.tcl\
                scripts/gen_sfm_xo.tcl\
		../../dpu_ip/Vitis/sfm/hdl/*.v\
		../../dpu_ip/dpu_eu_*/hdl/dpu_eu_*_sfm.sv\
		../../dpu_ip/dpu_eu_*/xci/sfm/fp_*/*.xci		


dpu_TCL=scripts/gen_dpu_xo.tcl
softmax_TCL=scripts/gen_sfm_xo.tcl

# Kernel name must match kernel name in kernel.xml
DPU_KERN_NAME = dpu_xrt_top
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

.PHONY: all clean

all : binary_container_1/dpu.xclbin

binary_container_1/dpu.xo: $(dpu_HDLSRCS)
	@mkdir -p $(@D)
	-@$(RM) $@
	$(VIVADO) -mode batch -source $(dpu_TCL) -tclargs $@ $(DPU_KERN_NAME) ${TARGET} ${DEVICE}

binary_container_1/softmax.xo: $(softmax_HDLSRCS)
	@mkdir -p $(@D)
	-@$(RM) $@
	$(VIVADO) -mode batch -source $(softmax_TCL) -tclargs $@ $(SM_KERN_NAME) ${TARGET} ${DEVICE}


binary_container_1/dpu.xclbin: $(kernel_xo)
	v++ $(XOCC_OPTS) -l --temp_dir binary_container_1 --log_dir binary_container_1/logs --remote_ip_cache binary_container_1/ip_cache -o "$@" $(+)
	cp ./binary_*/link/vivado/vpl/prj/prj.srcs/sources_1/bd/*/hw_handoff/*.hwh ./binary_*/sd_card
    
clean:
	${RM} *.o *.elf *.log *.jou sample*
	${RMDIR} binary_container_1/ packaged_*/ tmp_*/ .Xil/

