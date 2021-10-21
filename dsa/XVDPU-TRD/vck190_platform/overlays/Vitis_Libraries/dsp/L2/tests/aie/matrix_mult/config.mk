
ifneq ($(filter hw hw_emu, $(TARGET)), )
AIE_CXXFLAGS += -Xpreproc=-DUSING_PL_MOVER=1
endif

ifneq (,$(shell echo $(XPLATFORM) | awk '/vck190/'))
AIE_LDFLAGS := --config $(CUR_DIR)/system.cfg
endif

CXXFLAGS += -DUSING_PL_MOVER=1 -DUSING_UUT=1


