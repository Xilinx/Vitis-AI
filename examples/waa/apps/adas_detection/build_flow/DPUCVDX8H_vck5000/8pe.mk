#IP name: DPUCVDX8H
#Release date: 2022 May15Sun2100
#VITIS version: 2021.2
#Support frequency: 100/200/300
#Input files inlcude:
#	XO file: ./8pe/DPUCVDX8H.xo
#	AIE libadf.a: ./8pe/libadf.a
#	DPU connection file: ./8pe/dpu.connection.cfg
#Output file inlcude:
#	xcblin file: work/2021.2/package.xclbin
#How to run:
#	1: modify PLATFORM below to point to a valid VCK5000 xpfm
#	2: make xclbin
#
XLNX = 2021.2
#PLATFORM = <please provide xpfm here>
BUILD_SUBDIRS=work
FREQ_0 = 300
AIE_LIBADF = ./8pe/libadf.a
XO = ./8pe/DPUCVDX8H.xo $(BLOB_ACCEL)/blobfromimage_accel.xo
DPU_CONNECTION_CFG = ./8pe/dpu.connection.cfg
DPU_CONNECTION_CFG1 = ./dpu_advanced.cfg
AIE_LIBADF_ABS_PATH = $(abspath $(AIE_LIBADF))

VPPFLAGS  = -l -g --platform $(PLATFORM) --save-temps --temp_dir $(BUILD_SUBDIRS) -R 2
VPPFLAGS += --include ./inc
VPPFLAGS += --log_dir $(BUILD_SUBDIRS)/logs
VPPFLAGS += --report_dir $(BUILD_SUBDIRS)/reports
VPPFLAGS += --config $(DPU_CONNECTION_CFG)
VPPFLAGS += --config $(DPU_CONNECTION_CFG1)
VPPFLAGS += --kernel_frequency="0:$(FREQ_0)"
VPPFLAGS += --xp vivado_prop:run.impl_1.GEN_FULL_BITSTREAM=1

.PHONY:xclbin
xclbin:$(BUILD_SUBDIRS)/complete.pdi $(BUILD_SUBDIRS)/$(XLNX)/package.xclbin
$(BUILD_SUBDIRS)/complete.xclbin:$(XO) $(FST_XO) $(AIE_LIBADF)
	@ mkdir -p $(@D)
	@ v++ $(VPPFLAGS) -t hw -o $@ $(XO) $(FST_XO) $(AIE_LIBADF)
$(BUILD_SUBDIRS)/$(XLNX)/package.xclbin: $(BUILD_SUBDIRS)/complete.xclbin $(AIE_LIBADF)
	@ mkdir -p $(@D)
	@ v++ -p -t hw  --platform $(PLATFORM) --save-temps --temp_dir $(@D) --clock.freqHz 300000000:blobfromimage_accel_1 -o "$@" $^ --package.boot_mode=ospi
$(BUILD_SUBDIRS)/complete.pdi: $(BUILD_SUBDIRS)/complete.xclbin $(AIE_LIBADF)
	@ cd $(BUILD_SUBDIRS)/link/vivado/vpl/prj/prj.runs/impl_1 ; \
	prep_target -target hw -pdi level0_wrapper.pdi -aie-archive $(AIE_LIBADF_ABS_PATH) -out-dir prep_target_output -enable-aie-cores -platform $(PLATFORM); \
	cp prep_target_output/BOOT.BIN ../../../../../../$(@F) ;

.PHONY: clean
clean:
	-@rm -rf .Xil vivado* *.txt *.log *.xsa .ipcache sample*.ini packaged_kernel_*_hw_* tmp_kernel_pack_*_hw_*
	-@rm -rf work
		