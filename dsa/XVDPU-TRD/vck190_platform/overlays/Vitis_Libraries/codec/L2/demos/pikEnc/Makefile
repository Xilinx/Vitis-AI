#
# Copyright 2019-2021 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# makefile-generator v1.0.4
#

# ####################################### Help Section #####################################
.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<x86>"
	$(ECHO) "      Command to generate the design for specified Target and Shell."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO) "  make run TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<x86>"
	$(ECHO) "      Command to run application in emulation or on board."
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<x86>"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) ""
	$(ECHO) "  make host HOST_ARCH=<x86>"
	$(ECHO) "      Command to build host application."
	$(ECHO) ""

# ##################### Setting up default value of TARGET ##########################
TARGET ?= sw_emu

# ################### Setting up default value of DEVICE ##############################
DEVICE ?= xilinx_u200_xdma_201830_2

# ###################### Setting up default value of HOST_ARCH ####################### 
HOST_ARCH ?= x86

# #################### Checking if DEVICE in blacklist #############################
ifeq ($(findstring zc, $(DEVICE)), zc)
$(error [ERROR]: This project is not supported for $(DEVICE).)
endif

# #################### Checking if DEVICE in whitelist ############################
ifneq ($(findstring u200, $(DEVICE)), u200)
$(warning [WARNING]: This project has not been tested for $(DEVICE). It may or may not work.)
endif

# ######################## Setting up Project Variables #################################
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
XF_PROJ_ROOT ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%/L2/*}')
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))
XFLIB_DIR = $(XF_PROJ_ROOT)

# ######################### Include environment variables in utils.mk ####################
include ./utils.mk
XDEVICE := $(call device2xsa, $(DEVICE))
TEMP_DIR := _x_temp.$(TARGET).$(XDEVICE)
TEMP_REPORT_DIR := $(CUR_DIR)/reports/_x.$(TARGET).$(XDEVICE)
BUILD_DIR := build_dir.$(TARGET).$(XDEVICE)
BUILD_REPORT_DIR := $(CUR_DIR)/reports/_build.$(TARGET).$(XDEVICE)
EMCONFIG_DIR := $(BUILD_DIR)
XCLBIN_DIR := $(CUR_DIR)/$(BUILD_DIR)
export XCL_BINDIR = $(XCLBIN_DIR)

# ######################### Setting up Host Variables #########################
#Include Required Host Source Files
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/cpik_main.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/ac_predictions.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/ac_strategy.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/adaptive_quantization.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/adaptive_reconstruction.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/alpha.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/ans_common.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/ans_decode.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/ans_encode.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/ar_control_field.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/arch_specific.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/bilinear_transform.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/block_dictionary.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/butteraugli_comparator.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/butteraugli_distance.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/cache_aligned.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/chroma_from_luma.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/cmdline.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/codec_impl.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/codec_png.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/codec_pnm.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/color_correlation.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/color_encoding.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/compressed_dc.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/compressed_image.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/context_map_decode.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/context_map_encode.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/cpik.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/data_parallel.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/dc_predictor.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/dct.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/dct_util.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/deconvolve.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/descriptive_statistics.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/detect_dots.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/entropy_coder.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/epf.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/epf_target.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/external_image.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/gaborish.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/gauss_blur.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/gradient_map.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/headers.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/huffman_decode.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/huffman_encode.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/image.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/lehmer_code.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/linalg.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/lossless16.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/lossless8.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/lossless_entropy.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/metadata.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/noise.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/opsin_image.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/opsin_inverse.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/opsin_params.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/os_specific.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/padded_bytes.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/pik.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/pik_frame.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/pik_info.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/quant_weights.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/quantizer.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/saliency_map.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/single_image_handler.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/status.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/upscaler.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/yuv_convert.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/yuv_opsin_convert.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/butteraugli/butteraugli.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/simd/targets.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/brotli.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik/color_management.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/encode_order.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/pik_codec_common.cc
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/host_dev.cc
HOST_SRCS += $(XFLIB_DIR)/ext/xcl2/xcl2.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmsalpha.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmscam02.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmscgats.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmscnvrt.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmserr.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmsgamma.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmsgmt.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmshalf.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmsintrp.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmsio0.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmsio1.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmslut.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmsmd5.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmsmtrx.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmsnamed.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmsopt.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmspack.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmspcs.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmsplugin.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmsps2.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmssamp.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmssm.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmstypes.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmsvirt.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmswtpnt.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/src/cmsxform.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lodepng/lodepng.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lodepng/lodepng_util.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/FiniteStateEntropy/lib/entropy_common.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/FiniteStateEntropy/lib/fseU16.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/FiniteStateEntropy/lib/fse_compress.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/FiniteStateEntropy/lib/fse_decompress.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/FiniteStateEntropy/lib/huf_compress.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/FiniteStateEntropy/lib/huf_decompress.cpp
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/enc/backward_references.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/enc/backward_references_hq.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/enc/bit_cost.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/enc/block_splitter.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/enc/brotli_bit_stream.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/enc/cluster.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/enc/compress_fragment.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/enc/compress_fragment_two_pass.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/enc/dictionary_hash.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/enc/encode.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/enc/encoder_dict.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/enc/entropy_encode.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/enc/histogram.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/enc/literal_cost.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/enc/memory.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/enc/metablock.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/enc/static_dict.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/enc/utf8_util.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/common/dictionary.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/common/transform.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/dec/bit_reader.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/dec/decode.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/dec/huffman.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/dec/state.c
HOST_SRCS += $(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/fuzz/decode_fuzzer.c
CXXFLAGS += -I$(XFLIB_DIR)/../utils/L1/include/
CXXFLAGS += -I$(XFLIB_DIR)/L2/include/hw/pikEnc/
CXXFLAGS += -I$(XFLIB_DIR)/L2/include/hw/pikEnc/kernel3/
CXXFLAGS += -I$(XFLIB_DIR)/L2/demos/pikEnc/host
CXXFLAGS += -I$(XFLIB_DIR)/L2/demos/pikEnc/kernel
CXXFLAGS += -I$(XFLIB_DIR)/ext/xcl2
CXXFLAGS += -I$(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lcms/include
CXXFLAGS += -I$(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/brotli/c/include
CXXFLAGS += -I$(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/lodepng
CXXFLAGS += -I$(XFLIB_DIR)/L2/demos/pikEnc/host/third_party/FiniteStateEntropy/lib
CXXFLAGS += -I$(XFLIB_DIR)/L2/demos/pikEnc/host/third_party

ifeq ($(TARGET),sw_emu)
CXXFLAGS += -D SW_EMU_TEST
endif

ifeq ($(TARGET),hw_emu)
CXXFLAGS += -D HW_EMU_TEST
endif

# ######################### Host compiler global settings ############################
CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_HLS)/include -std=c++11 -O3 -Wall -Wno-unknown-pragmas -Wno-unused-label
LDFLAGS += -L$(XILINX_XRT)/lib -lOpenCL -lpthread -lrt -Wno-unused-label -Wno-narrowing -DVERBOSE
CXXFLAGS += -fmessage-length=0 -O3 
CXXFLAGS += -I$(CUR_DIR)/src/ 

ifeq ($(HOST_ARCH), x86)
LDFLAGS += -L$(XILINX_HLS)/lnx64/tools/fpo_v7_0 -Wl,--as-needed -lgmp -lmpfr -lIp_floating_point_v7_0_bitacc_cmodel
endif

# ################### Setting package and image directory #######################

EXE_NAME := host.exe
EXE_FILE := $(BUILD_DIR)/$(EXE_NAME)
HOST_ARGS :=  --xclbin $(BUILD_DIR)/pikEnc.xclbin images/small32x32.png test.pik --fast

# ##################### Kernel compiler global settings ##########################
VPP_FLAGS += -t $(TARGET) --platform $(XPLATFORM) --save-temps --optimize 2
VPP_FLAGS += --hls.jobs 8
VPP_LDFLAGS += --vivado.synth.jobs 8 --vivado.impl.jobs 8
ifneq (,$(shell echo $(XPLATFORM) | awk '/u200/'))
VPP_FLAGS += --config $(CUR_DIR)/conn_u200.cfg
endif


VPP_FLAGS += -I$(XFLIB_DIR)/../utils/L1/include/
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include/hw
VPP_FLAGS += -I$(XFLIB_DIR)/L2/include/hw/pikEnc
VPP_FLAGS += -I$(XFLIB_DIR)/L2/demos/pikEnc/kernel

kernel1Top_VPP_FLAGS +=  -D KERNEL_NAME=kernel1Top
kernel1Top_VPP_FLAGS += --hls.clock 300000000:kernel1Top
kernel2Top_VPP_FLAGS +=  -D KERNEL_NAME=kernel2Top
kernel2Top_VPP_FLAGS += --hls.clock 300000000:kernel2Top
kernel3Top_VPP_FLAGS +=  -D KERNEL_NAME=kernel3Top
kernel3Top_VPP_FLAGS += --hls.clock 300000000:kernel3Top
VPP_LDFLAGS_pikEnc += --kernel_frequency 200


# Kernel args
kernel3Top_ARGS += $(XFLIB_DIR)/L2/demos/pikEnc/kernel/kernel3/kernel3_common.cpp
kernel3Top_ARGS += $(XFLIB_DIR)/L2/demos/pikEnc/kernel/kernel3/ans.cpp
kernel3Top_ARGS += $(XFLIB_DIR)/L2/demos/pikEnc/kernel/kernel3/ac_tokenize.cpp
kernel3Top_ARGS += $(XFLIB_DIR)/L2/demos/pikEnc/kernel/kernel3/build_table_encode_histo.cpp
kernel3Top_ARGS += $(XFLIB_DIR)/L2/demos/pikEnc/kernel/kernel3/dc_shrink.cpp
kernel3Top_ARGS += $(XFLIB_DIR)/L2/demos/pikEnc/kernel/kernel3/dc_tokenize.cpp
kernel3Top_ARGS += $(XFLIB_DIR)/L2/demos/pikEnc/kernel/kernel3/ctrl_tokenize.cpp
kernel3Top_ARGS += $(XFLIB_DIR)/L2/demos/pikEnc/kernel/kernel3/build_cluster.cpp

# ############################ Declaring Binary Containers ##########################

BINARY_CONTAINERS += $(BUILD_DIR)/pikEnc.xclbin
BINARY_CONTAINER_pikEnc_OBJS += $(TEMP_DIR)/kernel1Top.xo
BINARY_CONTAINER_pikEnc_OBJS += $(TEMP_DIR)/kernel2Top.xo
BINARY_CONTAINER_pikEnc_OBJS += $(TEMP_DIR)/kernel3Top.xo

# ######################### Setting Targets of Makefile ################################

.PHONY: all clean cleanall docs emconfig
ifeq ($(HOST_ARCH), x86)
all:  check_version check_vpp check_platform check_xrt $(EXE_FILE) $(BINARY_CONTAINERS) emconfig
else
all:  check_version check_vpp check_platform check_sysroot $(EXE_FILE) $(BINARY_CONTAINERS) emconfig  sd_card
endif

.PHONY: host
ifeq ($(HOST_ARCH), x86)
host:   check_xrt $(EXE_FILE)
else
host:   check_sysroot $(EXE_FILE)
endif

.PHONY: xclbin
ifeq ($(HOST_ARCH), x86)
xclbin: check_vpp  check_xrt  $(BINARY_CONTAINERS)
else
xclbin: check_vpp  check_sysroot  $(BINARY_CONTAINERS)
endif

.PHONY: build
build: xclbin

# ################ Setting Rules for Binary Containers (Building Kernels) ################
$(TEMP_DIR)/kernel1Top.xo: $(XFLIB_DIR)/L2/demos/pikEnc/kernel/XAccPIKKernel1.cpp $(kernel1Top_ARGS)
	$(ECHO) "Compiling Kernel: kernel1Top"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(kernel1Top_VPP_FLAGS) $(VPP_FLAGS) -k kernel1Top -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' $^
$(TEMP_DIR)/kernel2Top.xo: $(XFLIB_DIR)/L2/demos/pikEnc/kernel/XAccPIKKernel2.cpp $(kernel2Top_ARGS)
	$(ECHO) "Compiling Kernel: kernel2Top"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(kernel2Top_VPP_FLAGS) $(VPP_FLAGS) -k kernel2Top -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' $^
$(TEMP_DIR)/kernel3Top.xo: $(XFLIB_DIR)/L2/demos/pikEnc/kernel/XAccPIKKernel3.cpp $(kernel3Top_ARGS)
	$(ECHO) "Compiling Kernel: kernel3Top"
	mkdir -p $(TEMP_DIR)
	$(VPP) -c $(kernel3Top_VPP_FLAGS) $(VPP_FLAGS) -k kernel3Top -I'$(<D)' --temp_dir $(TEMP_DIR) --report_dir $(TEMP_REPORT_DIR) -o'$@' $^


$(BUILD_DIR)/pikEnc.xclbin: $(BINARY_CONTAINER_pikEnc_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -l $(VPP_FLAGS) --temp_dir $(TEMP_DIR) --report_dir $(BUILD_REPORT_DIR)/pikEnc $(VPP_LDFLAGS) $(VPP_LDFLAGS_pikEnc) -o '$@' $(+)


# ################# Setting Rules for Host (Building Host Executable) ################
ifeq ($(HOST_ARCH), x86)
$(EXE_FILE):  $(HOST_SRCS) | check_xrt 
else
$(EXE_FILE):  $(HOST_SRCS) | check_sysroot 
endif

	mkdir -p $(BUILD_DIR)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(XPLATFORM) --od $(EMCONFIG_DIR)


# ###############Setting Essential Checks And Running Rules For Vitis Flow #############
run: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) $(EXE_FILE) $(HOST_ARGS)
else
	$(EXE_FILE) $(HOST_ARGS)
endif

# ################################# Cleaning Rules ##################################
cleanh:
	-$(RMDIR) $(EXE_FILE) vitis_* TempConfig system_estimate.xtxt *.rpt .run/
	-$(RMDIR) src/*.ll _xocc_* .Xil dltmp* xmltmp* *.log *.jou *.wcfg *.wdb sample_link.ini sample_compile.ini obj* bin* *.csv *.jpg *.jpeg *.png

cleank:
	-$(RMDIR) $(BUILD_DIR)/*.xclbin _vimage *xclbin.run_summary qemu-memory-_* emulation/ _vimage/ start_simulation.sh *.xclbin
	-$(RMDIR) _x_temp.*/_x.* _x_temp.*/.Xil _x_temp.*/profile_summary.* xo_* _x*
	-$(RMDIR) _x_temp.*/dltmp* _x_temp.*/kernel_info.dat _x_temp.*/*.log 
	-$(RMDIR) _x_temp.* 

cleanall: cleanh cleank
	-$(RMDIR) $(BUILD_DIR)  build_dir.* emconfig.json *.html $(TEMP_DIR) $(CUR_DIR)/reports *.csv *.run_summary $(CUR_DIR)/*.raw package_* run_script.sh .ipcache *.str
	-$(RMDIR) $(XFLIB_DIR)/common/data/*.xe2xd* $(XFLIB_DIR)/common/data/*.orig*

	-$(RMDIR) $(AIE_CONTAINERS) $(CUR_DIR)/Work $(CUR_DIR)/*.xpe $(CUR_DIR)/hw.o $(CUR_DIR)/*.xsa $(CUR_DIR)/xnwOut aiesimulator_output .AIE_SIM_CMD_LINE_OPTIONS

clean: cleanh
