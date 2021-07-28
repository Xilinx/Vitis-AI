# rules.mk - defines basic rules for building executables and xclbins

# Defines the prefix for each kernel.
XCLBIN_DIR=xclbin

ECHO:= @echo

.PHONY: help
help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "	make all TARGETS=<sw_emu/hw_emu/hw>"
	$(ECHO) "		Command to generate the design for specified Target."
	$(ECHO) ""
	$(ECHO) "	make clean"
	$(ECHO) "		Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "	make cleanall"
	$(ECHO) "		Command to remove all the generated files."
	$(ECHO) ""

target_blacklist = $(if $($(1)_NTARGETS), $($(1)_NTARGETS),)
device_blacklist = $(if $($(1)_NDEVICES), $($(1)_NDEVICES),)

# mk_exe - build an exe from host code
#   CXX - compiler to use
#   CXXFLAGS - base compiler flags to use
#   LDFLAGS - base linker flags to use
#   $(1) - name of exe
#   $(1)_SRCS - the source files to compile
#   $(1)_HDRS - the header files used by sources
#   $(1)_CXXFLAGS - extra flags specific to this exe
#   $(1)_LDFLAGS - extra linkder flags

define mk_exe

$(1): $($(1)_SRCS) $($(1)_HDRS)
	$(CXX) $(CXXFLAGS) $($(1)_CXXFLAGS) $($(1)_SRCS) -o $$@ $($(1)_LDFLAGS) $(LDFLAGS)

EXE_GOALS+= $(1)

endef

# mk_xo - create an xo from a set of kernel sources
#  CLC - kernel compiler to use
#  CLFLAGS - flags to pass to the compiler
#  $(1) - base name for this kernel
#  $(1)_SRCS - set of source kernel
#  $(1)_HDRS - set of header kernel
#  $(1)_CLFLAGS - set clflags per kernel 
#  $(1)_NDEVICES - set blacklist for devices
#  $(1)_NTARGETS - set blacklist for targets
#  $(2) - compilation target (i.e. hw, hw_emu, sw_emu)
#  $(3) - device name (i.e. xilinx:adm-pcie-ku3:1ddr:3.0)
#  $(3)_CLFLAGS - set clflags per device

define mk_xo

ifneq ($(filter $(2),$(call target_blacklist,$(1))),$(2))
ifneq ($(filter $(3),$(call device_blacklist,$(1))),$(3))

$(XCLBIN_DIR)/$(1).$(2).$(call device2sandsa,$(3)).xo: $($(1)_SRCS) $($(1)_HDRS) $(call device2dep,$(3))
	mkdir -p ${XCLBIN_DIR}
	$(CLC) -c $(CLFLAGS) $($(1)_CLFLAGS) $($(1)_$(call device2sandsa,$(3))_CLFLAGS) -o $$@ -t $(2) --platform $(3) $($(1)_SRCS)

XO_GOALS+= $(XCLBIN_DIR)/$(1).$(2).$(call device2sandsa,$(3)).xo

endif
endif

endef

# mk_rtlxo - create an xo from a tcl and RTL sources
#   VIVADO - version of Vivado to use
#   $(1) - base name for this kernel
#   $(1)_HDLSRCS - source files used in compilation
#   $(1)_TCL - tcl file to use for build
#   $(2) - target to build for
#   $(3) - device to build for

define mk_rtlxo

ifneq ($(filter $(2),$(call target_blacklist,$(1))),$(2))
ifneq ($(filter $(3),$(call device_blacklist,$(1))),$(3))

$(XCLBIN_DIR)/$(1).$(2).$(call device2sandsa,$(3)).xo: $($(1)_HDLSRCS) $(call device2dep,$(3))
	mkdir -p $(XCLBIN_DIR)
	$(VIVADO) -mode batch -source $($(1)_TCL) -tclargs $(XCLBIN_DIR)/$(1).$(2).$(call device2sandsa,$(3)).xo $(1) $(2) $(call device2sandsa,$(3))

XO_GOALS+=$(XCLBIN_DIR)/$(1).$(2).$(call device2sandsa,$(3)).xo

endif
endif

endef

# mk_xclbin - create an xclbin from a set of krnl sources
#  LDCLC - kernel linker to use
#  LDCLFLAGS - flags to pass to the linker
#  $(1) - base name for this xclbin
#  $(1)_XOS - list of xos to link
#  $(1)_NDEVICES - set blacklist for devices
#  $(1)_NTARGETS - set blacklist for targets
#  $(2) - compilation target (i.e. hw, hw_emu, sw_emu)
#  $(3) - device name (i.e. xilinx:adm-pcie-ku3:1ddr:3.0)
#  $(3)_LDCLFLAGS - set linker flags per device

define mk_xclbin

ifneq ($(filter $(2),$(call target_blacklist,$(1))),$(2))
ifneq ($(filter $(3),$(call device_blacklist,$(1))),$(3))

$(XCLBIN_DIR)/$(1).$(2).$(call device2sandsa,$(3)).xclbin: $(addprefix $(XCLBIN_DIR)/,$(addsuffix .$(2).$(call device2sandsa,$(3)).xo, $($(1)_XOS))) $(call device2dep,$(3))

	mkdir -p ${XCLBIN_DIR}
	$(LDCLC) -l $(LDCLFLAGS) $($(1)_LDCLFLAGS) $($(1)_$(call device2sandsa,$(3))_LDCLFLAGS) -o $$@ -t $(2) --platform $(3) $(addprefix $(XCLBIN_DIR)/,$(addsuffix .$(2).$(call device2sandsa,$(3)).xo,$($(1)_XOS)))

XCLBIN_GOALS+= $(XCLBIN_DIR)/$(1).$(2).$(call device2sandsa,$(3)).xclbin

endif
endif

endef


$(foreach exe,$(EXES),$(eval $(call mk_exe,$(exe))))

$(foreach xo,$(XOS),$(foreach target,$(TARGETS),$(foreach device,$(DEVICES),$(eval $(call mk_xo,$(xo),$(target),$(device))))))
$(foreach rtlxo,$(RTLXOS),$(foreach target,$(TARGETS),$(foreach device,$(DEVICES),$(eval $(call mk_rtlxo,$(rtlxo),$(target),$(device))))))
$(foreach xclbin,$(XCLBINS),$(foreach target,$(TARGETS),$(foreach device,$(DEVICES),$(eval $(call mk_xclbin,$(xclbin),$(target),$(device))))))

.PHONY: all
all: user_set $(EXE_GOALS) $(XCLBIN_GOALS)  report_route

.PHONY: exe
exe: $(EXE_GOALS)

.PHONY: xo
xo: $(XO_GOALS)

.PHONY: xclbin
xclbin: $(XCLBIN_GOALS)

.PHONY: user_set 
user_set: 
	$(USER_SET)

.PHONY: gen_board_top
gen_board_top: 
	$(GEN_BOARD_TOP)

.PHONY: docs
docs: README.md

.PHONY: cleanall
cleanall: clean
	rm -rf $(XCLBIN_DIR)

.PHONY: report_route 
report_route: 
	$(REPORT_ROUTE)


.PHONY: clean
clean:
	rm -rf $(EXE_GOALS) $(XCLBIN_DIR)/{*sw_emu*,*hw_emu*} sdaccel_* TempConfig system_estimate.xtxt *.rpt
	rm -rf src/*.ll _xocc_* .Xil emconfig.json $(EXTRA_CLEAN) dltmp* xmltmp* *.log *.jou *.wcfg *.wdb
	rm -rf outputs/checkpoints/*.dcp
	rm -rf outputs/reports/*.rpt
	rm -rf outputs/logs/*.log

README.md: description.json
	$(COMMON_REPO)/utility/readme_gen/readme_gen.py description.json

include $(COMMON_REPO)/utility/check.mk

# copy_library_sources - copy library source files to the local source directory
#   $(1) - name of exe
#   $(1)_SRCS - the source files
#   $(1)_HDRS - the header files

define copy_library_sources

local-files: $(1)_local_files

# note:
# this does not update the makefile to reference the src/ copy of the
# library files, so gui and command line projects can get out of sync.
$(1)_local_files: $($(1)_SRCS) $($(1)_HDRS)
	@echo "Copying library sources to project:"
	-@mkdir -p src
	-@tar c $($(1)_SRCS) $($(1)_HDRS) -P --exclude="src/*" --xform="s/^.*\///" | tar xv -C src
	@echo "Library sources were copied to the src/ directory."

endef

$(foreach exe,$(EXES),$(eval $(call copy_library_sources,$(exe))))
