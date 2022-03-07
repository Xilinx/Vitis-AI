#IP name: DPUCVDX8H
#Release date: 2021 Dec14Tue1433
#VITIS version: 2021.2
#Support frequency: 100/200/300
#Input files inlcude:
#	XO file: ./2pe/DPUCVDX8H.xo
#	AIE libadf.a: ./2pe/libadf.a
#	DPU connection file: ./2pe/dpu.connection.cfg
#Output file inlcude:
#	xcblin file: work/2021.2/package.xclbin
#How to run:
#	1: modify PLATFORM below to point to a valid VCK5000 xpfm
#	2: make xclbin
#
#Note: Occupied AIE cores
#	section 1: column-12 to column-17 / row-0 to row-7
#	section 2: column-32 to column-37 / row-0 to row-7
#	section 3: column-22 to column-23 / row-0 to row-7
#	section 3: column-24 to column-25 / row-0 to row-7
XLNX = 2021.2
PLATFORM = <Please insert xpfm here>
BUILD_SUBDIRS=work
FREQ_0 = 300
AIE_LIBADF = ./2pe/libadf.a
XO = ./2pe/DPUCVDX8H.xo
DPU_CONNECTION_CFG = ./2pe/dpu.connection.cfg
AIE_LIBADF_ABS_PATH = $(abspath $(AIE_LIBADF))

VPPFLAGS  = -l -g --platform $(PLATFORM) --save-temps --temp_dir $(BUILD_SUBDIRS) -R 2
VPPFLAGS += --include ./inc
VPPFLAGS += --log_dir $(BUILD_SUBDIRS)/logs
VPPFLAGS += --report_dir $(BUILD_SUBDIRS)/reports
VPPFLAGS += --config $(DPU_CONNECTION_CFG)
VPPFLAGS += --kernel_frequency="0:$(FREQ_0)"
VPPFLAGS += --xp vivado_prop:run.impl_1.GEN_FULL_BITSTREAM=1

.PHONY:xclbin
xclbin:$(BUILD_SUBDIRS)/complete.pdi $(BUILD_SUBDIRS)/$(XLNX)/package.xclbin
$(BUILD_SUBDIRS)/complete.xclbin:$(XO) $(FST_XO) $(AIE_LIBADF)
	@ mkdir -p $(@D)
	@ v++ $(VPPFLAGS) -t hw -o $@ $(XO) $(FST_XO) $(AIE_LIBADF)
$(BUILD_SUBDIRS)/$(XLNX)/package.xclbin: $(BUILD_SUBDIRS)/complete.xclbin $(AIE_LIBADF)
	@ mkdir -p $(@D)
	@ v++ -p -t hw  --platform $(PLATFORM) --save-temps --temp_dir $(@D) -o "$@" $^ --package.boot_mode=ospi
$(BUILD_SUBDIRS)/complete.pdi: $(BUILD_SUBDIRS)/complete.xclbin $(AIE_LIBADF)
	@ cd $(BUILD_SUBDIRS)/link/vivado/vpl/prj/prj.runs/impl_1 ; \
	prep_target -target hw -pdi level0_wrapper.pdi -aie-archive $(AIE_LIBADF_ABS_PATH) -out-dir prep_target_output -enable-aie-cores -platform $(PLATFORM); \
	cp prep_target_output/BOOT.BIN ../../../../../../$(@F) ;
