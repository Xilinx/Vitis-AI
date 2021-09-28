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
#

#+-------------------------------------------------------------------------------
# The following parameters are assigned with default values. These parameters can
# be overridden through the make command line
#+-------------------------------------------------------------------------------

REPORT := no
PROFILE := no
DEBUG := no

#'estimate' for estimate report generation
#'system' for system report generation
ifneq ($(REPORT), no)
VXX_FLAGS += --report estimate
VXX_FLAGS += --report system
endif

#Generates profile summary report
ifeq ($(PROFILE), yes)
VXX_FLAGS += --profile_kernel data:all:all:all
endif

#Generates debug summary report
ifeq ($(DEBUG), yes)
VXX_FLAGS += --dk protocol:all:all:all
endif

#Checks for XILINX_VITIS
#ifndef XILINX_VITIS
#$(error XILINX_VITIS variable is not set, please set correctly and rerun)
#endif

#   sanitize_xsa - create a filesystem friendly name from xsa name
#   $(1) - name of xsa
COLON=:
PERIOD=.
UNDERSCORE=_
sanitize_xsa = $(strip $(subst $(PERIOD),$(UNDERSCORE),$(subst $(COLON),$(UNDERSCORE),$(1))))

device2xsa = $(if $(filter $(suffix $(1)),.xpfm),$(shell $(XFCMP_DIR)/common/utility//parsexpmf.py $(1) xsa 2>/dev/null),$(1))
device2sanxsa = $(call sanitize_xsa,$(call device2xsa,$(1)))
device2dep = $(if $(filter $(suffix $(1)),.xpfm),$(dir $(1))/$(shell $(XFCMP_DIR)/common/utility//parsexpmf.py $(1) hw 2>/dev/null) $(1),)

# Cleaning stuff
RM = rm -f
RMDIR = rm -rf

ECHO := @echo

