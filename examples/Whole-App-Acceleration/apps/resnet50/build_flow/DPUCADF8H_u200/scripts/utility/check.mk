# check.mk - defines rules for testing

NIMBIX_DSA_xilinx_adm-pcie-ku3_2ddr-xpr = nx1
NIMBIX_DSA_xilinx_adm-pcie-7v3_1ddr = nx2
NIMBIX_DSA_xilinx_xil-accel-rd-ku115_4ddr-xpr = nx3
NIMBIX_DSA_xilinx_xil-accel-rd-vu9p_4ddr-xpr = nx4

dsa2type = $(NIMBIX_DSA_$(call sanitize_dsa,$(1)))

nimbix_RUNNER = $(COMMON_REPO)/utility/nimbix/nimbix-run.py $(NIMBIXFLAGS) --type $(call dsa2type,$(1)) --
sw_emu_RUNNER = XCL_EMULATION_MODE=sw_emu
hw_emu_RUNNER = XCL_EMULATION_MODE=hw_emu

loader = $(call $(1)_RUNNER,$(2))

# mk_check - run a check against the application
#  $(1) - base name for the check
#  $(1)_EXE - executable to run
#  $(1)_ARGS - arguments to use by default for the check
#  $(1)_DEVICES - whitelist of devices to run the check
#  $(1)_DEPS - extra dependencies of the check
#  $(2) - compilation target (i.e. hw, hw_emu, sw_emu)
#  $(1)_$(2)_ARGS - specialization of arguments for specific targets
#  $(3) - device name (i.e. xilinx:adm-pcie-ku3:1ddr:3.0)
#  $(1)_$(3)_ARGS - specialization of arguments for specific devices (has
#                   priority over targets)
#  $(1)_$(2)_$(3)_ARGS - specialization of arguments for specific targets and
#                        devices
define mk_check

ifneq ($(filter $(2),$(call target_blacklist,$(1))),$(2))
ifneq ($(filter $(3),$(call device_blacklist,$(1))),$(3))

.PHONY: $(1)_$(2)_$(call device2sandsa,$(3))_check
$(1)_$(2)_$(call device2sandsa,$(3))_check: $($(1)_DEPS) $($(1)_EXE) $(foreach xclbin,$($(1)_XCLBINS),$(XCLBIN_DIR)/$(xclbin).$(2).$(call device2sandsa,$(3)).xclbin)
ifneq ($(2),hw)
	$(EMCONFIGUTIL) --platform $(3) --nd $(NUM_DEVICES)
endif
ifdef $(1)_$(2)_$(call device2sandsa,$(3))_ARGS
	$(call loader,$(2),$(3)) ./$($(1)_EXE) $($(1)_$(2)_$(call device2sandsa,$(3))_ARGS)
else
ifdef $(1)_$(call device2sandsa,$(3))_ARGS
	$(call loader,$(2),$(3)) ./$($(1)_EXE) $($(1)_$(call device2sandsa,$(3))_ARGS)
else
ifdef $(1)_$(2)_ARGS
	$(call loader,$(2),$(3)) ./$($(1)_EXE) $($(1)_$(2)_ARGS)
else
	$(call loader,$(2),$(3)) ./$($(1)_EXE) $($(1)_ARGS)
endif
endif
endif

CHECK_GOALS += $(1)_$(2)_$(call device2sandsa,$(3))_check

endif
endif

endef

$(foreach check,$(CHECKS),$(foreach target,$(TARGETS),$(foreach device,$(DEVICES),$(eval $(call mk_check,$(check),$(target),$(device))))))

ifdef CHECK_GOALS
ECHO:= @echo

#Extended help messages
help::
	$(ECHO) "	make check TARGETS=<sw_emu/hw_emu/hw>"
	$(ECHO) "		Command to run application/kernel either in emulation(sw_emu/hw_emu) or on board(hw)."
	$(ECHO) ""
endif

.PHONY: check
check: $(CHECK_GOALS)

