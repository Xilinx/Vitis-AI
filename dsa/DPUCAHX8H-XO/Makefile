help:
	@echo "make build ALVEO=u50 USE_2LV=true SLR0=4ENGINE SLR1=4ENGINE ACLK_FREQ=275 DWC_EN=1"
	@echo "make build ALVEO=u55c USE_2LV=false SLR0=3ENGINE SLR1=4ENGINE SLR2=4ENGINE ACLK_FREQ=300 DWC_EN=1"
	@echo "make build ALVEO=u280 USE_2LV=false SLR0=4ENGINE SLR1=5ENGINE SLR2=5ENGINE ACLK_FREQ=250 DWC_EN=0"
clean:
	#-rm -rf release_*xo*
	-rm bit_gen/vivado*
	-rm -rf release_*xo*/tmp_kernel_pack_DPUCAHX8H_*
	-rm -rf release_*xo*/packaged_kernel/
	-rm -rf release_*xo*/vivado*
	-rm -rf bit_gen/*log
	-rm -rf bit_gen/*_imp
	-rm -rf bit_gen/hbm_assignment.txt
	-rm -rf bit_gen/*ltx
	-rm -rf bit_gen/*xclbin*
	-rm -rf bit_gen/.*swp
	-rm -rf bit_gen/.*swo
	-rm -rf bit_gen/.ipcache
	-rm -rf bit_gen/.Xil/
	#-rm -rf mPUCAHX8H_SRC/DPU
#release_u50lv_xo_withdwc
ifeq ($(USE_2LV),true)
lv_xo=lv
else
lv_xo=
endif
ifeq ($(DWC_EN),1)
xo_dwc=_withdwc
else
xo_dwc=
endif

GIT=/group/dphi_cloud/jackiez/DPU_V3E_XO
update_dpu: check_tcl $(GIT)/option_list* 
	-rm -rf DPUCAHX8H_SRC/DPU
	perl ./xo_release/script/set_option.pl -DPU_GIT $(GIT) -option_list $(GIT)/option_list  
	cd $(GIT) && \
	sed -i 's/set enc \"false/set enc \"true/g' vivado_rel_ip/scripts/setup.tcl 
	cd $(GIT)/vivado_rel_ip/ && ./run
	cd $(GIT) && git branch >./vivado_rel_ip/outputs/version.info && git rev-parse HEAD >> ./vivado_rel_ip/outputs/version.info && \
	git diff >> ./vivado_rel_ip/outputs/version.info
	cp -r $(GIT)/vivado_rel_ip/outputs ./DPUCAHX8H_SRC/DPU
	cp $(GIT)/option_list ./DPUCAHX8H_SRC
	@ echo "update DPU source done"

u50lv: release_u50lv_xo 
	cd ./bit_gen/constraints  && cat opt_design.pre.common.u50_6cr.tcl v3e.u50_6cr.timing.slr0.xdc v3e.u50_6cr.physical.slr0.5ENGINE.xdc v3e.u50_6cr.timing.slr1.xdc v3e.u50_6cr.physical.slr1.5ENGINE.xdc  > opt_design.pre.u50lv.tcl
	cd ./bit_gen; v++ -t hw --platform xilinx_u50lv_gen3x4_xdma_2_202010_1 --save-temps --temp_dir u50lv_imp -l --config "./script/cons_u50lv.ini" -o u50lv.xclbin ../$</DPUCAHX8H_5ENGINE.xo
	sed -i 's/"m_interrupt_id": "1"/"m_interrupt_id": "temp"/g' ./bit_gen/u50lv_imp/link/int/u50lv.rtd
	sed -i 's/"m_interrupt_id": "0"/"m_interrupt_id": "1"/g' ./bit_gen/u50lv_imp/link/int/u50lv.rtd
	sed -i 's/"m_interrupt_id": "temp"/"m_interrupt_id": "0"/g' ./bit_gen/u50lv_imp/link/int/u50lv.rtd
	cd ./bit_gen/u50lv_imp  && /proj/xbuilds/SWIP/2020.1_0526_1803/installs/lin64/Vitis/2020.1/bin/xclbinutil --add-section BITSTREAM:RAW:./link/int/partial.bit --force --target hw --key-value SYS:dfx_enable:true --add-section :JSON:./link/int/u50lv.rtd --append-section :JSON:./link/int/appendSection.rtd --add-section CLOCK_FREQ_TOPOLOGY:JSON:./link/int/u50lv_xml.rtd --add-section BUILD_METADATA:JSON:./link/int/u50lv_build.rtd --add-section EMBEDDED_METADATA:RAW:./link/int/u50lv.xml --add-section SYSTEM_METADATA:RAW:./link/int/systemDiagramModelSlrBaseAddress.json --key-value SYS:PlatformVBNV:xilinx_u50lv_gen3x4_xdma_2_202010_1 --output ../u50lv.xclbin

u50lv_withdwc: release_u50lv_xo_withdwc 
	cd ./bit_gen/constraints  && cat opt_design.pre.common.u50_6cr.tcl v3e.u50_6cr.timing.slr0.xdc v3e.u50_6cr.physical.slr0.4ENGINE_withdwc.xdc v3e.u50_6cr.timing.slr1.xdc v3e.u50_6cr.physical.slr1.4ENGINE_withdwc.xdc  > opt_design.pre.u50lv.tcl
	cd ./bit_gen; v++ -t hw --platform xilinx_u50lv_gen3x4_xdma_2_202010_1 --save-temps --temp_dir u50lv_imp -l --config "./script/cons_u50lv_withdwc.ini" -o u50lv.xclbin ../$</DPUCAHX8H_4ENGINE.xo
	sed -i 's/"m_interrupt_id": "1"/"m_interrupt_id": "temp"/g' ./bit_gen/u50lv_imp/link/int/u50lv.rtd
	sed -i 's/"m_interrupt_id": "0"/"m_interrupt_id": "1"/g' ./bit_gen/u50lv_imp/link/int/u50lv.rtd
	sed -i 's/"m_interrupt_id": "temp"/"m_interrupt_id": "0"/g' ./bit_gen/u50lv_imp/link/int/u50lv.rtd
	cd ./bit_gen/u50lv_imp  && /proj/xbuilds/SWIP/2020.1_0526_1803/installs/lin64/Vitis/2020.1/bin/xclbinutil --add-section BITSTREAM:RAW:./link/int/partial.bit --force --target hw --key-value SYS:dfx_enable:true --add-section :JSON:./link/int/u50lv.rtd --append-section :JSON:./link/int/appendSection.rtd --add-section CLOCK_FREQ_TOPOLOGY:JSON:./link/int/u50lv_xml.rtd --add-section BUILD_METADATA:JSON:./link/int/u50lv_build.rtd --add-section EMBEDDED_METADATA:RAW:./link/int/u50lv.xml --add-section SYSTEM_METADATA:RAW:./link/int/systemDiagramModelSlrBaseAddress.json --key-value SYS:PlatformVBNV:xilinx_u50lv_gen3x4_xdma_2_202010_1 --output ../u50lv.xclbin


u50: release_u50_xo
	cd ./bit_gen/constraints  && cat opt_design.pre.common.u50_6cr.tcl v3e.u50_6cr.timing.slr0.xdc v3e.u50_6cr.physical.slr0.3ENGINE.xdc v3e.u50_6cr.timing.slr1.xdc v3e.u50_6cr.physical.slr1.3ENGINE.xdc  > opt_design.pre.u50.tcl
	cd ./bit_gen; v++ -t hw --platform xilinx_u50_gen3x4_xdma_2_202010_1 --save-temps --temp_dir u50_imp -l --config "./script/cons_u50.ini" -o u50.xclbin ../$</DPUCAHX8H_3ENGINE.xo

SLR0_ENGINE_NUM := $(if $(SLR0), $(subst ENGINE,,$(SLR0)) ,0 )
SLR1_ENGINE_NUM := $(if $(SLR1), $(subst ENGINE,,$(SLR1)) ,0 )
SLR2_ENGINE_NUM := $(if $(SLR2), $(subst ENGINE,,$(SLR2)) ,0 )
FORCE_D_LOC = balance
FORCE_I_LOC = balance
FORCE_W_LOC = balance
ifeq ($(ALVEO),u50)
ifeq ($(SLR0)_$(SLR1),3ENGINE_3ENGINE)
MANUALLY_HBM_ASSIGN = ./bit_gen/manual/u50.hbm_assignment.manually.33.txt
else ifeq ($(SLR0)_$(SLR1)),4ENGINE_5ENGINE)
MANUALLY_HBM_ASSIGN = ./bit_gen/manual/u50.hbm_assignment.manually.45.txt
else ifeq ($(SLR0)_$(SLR1),5ENGINE_5ENGINE)
MANUALLY_HBM_ASSIGN = ./bit_gen/manual/u50.hbm_assignment.manually.55.txt
else
MANUALLY_HBM_ASSIGN =
endif
else
ifeq ($(SLR0)_$(SLR1)_$(SLR2),3ENGINE_3ENGINE_3ENGINE)
MANUALLY_HBM_ASSIGN = ./bit_gen/manual/u280.hbm_assignment.manually.333.txt
else ifeq ($(SLR0)_$(SLR1)_$(SLR2),4ENGINE_4ENGINE_4ENGINE)
MANUALLY_HBM_ASSIGN = ./bit_gen/manual/u280.hbm_assignment.manually.444.txt
else ifeq ($(SLR0)_$(SLR1)_$(SLR2),4ENGINE_5ENGINE_5ENGINE)
MANUALLY_HBM_ASSIGN = ./bit_gen/manual/u280.hbm_assignment.manually.455.txt
else
MANUALLY_HBM_ASSIGN =
endif
endif
USE_2LV = false
DWC_EN = 0

ALVEO=u50
ifeq ($(ALVEO), u280)
override DEVICE = xilinx_u280_xdma_201920_3
override HBM_FREQ = 900
else ifeq ($(ALVEO),u55c)
override DEVICE = xilinx_u55c_gen3x16_xdma_2_202110_1
override HBM_FREQ = 900
else ifeq ($(ALVEO),u50)
ifeq ($(USE_2LV),true)
override DEVICE = xilinx_u50lv_gen3x4_xdma_2_202010_1
else
override DEVICE = xilinx_u50_gen3x4_xdma_2_202010_1
endif
override HBM_FREQ = 600
else
$(error ALVEO=$(ALVEO) is not supported for now)
endif

OPT_DESIGN_DIRECTIVE = Explore
#Explore, EarlyBlockPlacement, WLDrivenBlockPlacement, ExtraNetDelay_high, ExtraNetDelay_low, AltSpreadLogic_high, AltSpreadLogic_medium, AltSpreadLogic_low, ExtraPostPlacementOpt, ExtraTimingOpt, SSI_SpreadLogic_high, SSI_SpreadLogic_low, SSI_SpreadSLLs, SSI_BalanceSLLs, SSI_BalanceSLRs, SSI_HighUtilSLRs, RuntimeOptimized, Quick, Default
#set PPLACE_POWER_OPT_DESIGN_DIRECTIVE "";
PLACE_DIRECTIVE = Explore


BK_MODE = 2M
ifeq ($(ALVEO),u50)
SLR0_TIMING_XDC   := $(if $(SLR0), ./bit_gen/constraints/v3e.$(ALVEO)_6cr.timing.slr0.xdc,)
SLR1_TIMING_XDC   := $(if $(SLR1), ./bit_gen/constraints/v3e.$(ALVEO)_6cr.timing.slr1.xdc,)
SLR2_TIMING_XDC   := $(if $(SLR2), ./bit_gen/constraints/v3e.$(ALVEO).timing.slr2.xdc,)
else
SLR0_TIMING_XDC   := $(if $(SLR0), ./bit_gen/constraints/v3e.$(ALVEO).timing.slr0.xdc,)
SLR1_TIMING_XDC   := $(if $(SLR1), ./bit_gen/constraints/v3e.$(ALVEO).timing.slr1.xdc,)
SLR2_TIMING_XDC   := $(if $(SLR2), ./bit_gen/constraints/v3e.$(ALVEO).timing.slr2.xdc,)
endif


ifeq ($(ALVEO),u50)
SLR0_PHYSICAL_XDC := $(if $(SLR0), ./bit_gen/constraints/v3e.$(ALVEO)_6cr.physical.slr0.$(SLR0)$(xo_dwc).xdc,)
SLR1_PHYSICAL_XDC := $(if $(SLR1), ./bit_gen/constraints/v3e.$(ALVEO)_6cr.physical.slr1.$(SLR1)$(xo_dwc).xdc,)
SLR2_PHYSICAL_XDC := $(if $(SLR2), ./bit_gen/constraints/v3e.$(ALVEO).physical.slr2.$(SLR2).xdc,)
else
SLR0_PHYSICAL_XDC := $(if $(SLR0), ./bit_gen/constraints/v3e.$(ALVEO).physical.slr0.$(SLR0)$(BK_MODE)$(xo_dwc).xdc,)
SLR1_PHYSICAL_XDC := $(if $(SLR1), ./bit_gen/constraints/v3e.$(ALVEO).physical.slr1.$(SLR1)$(BK_MODE)$(xo_dwc).xdc,)
SLR2_PHYSICAL_XDC := $(if $(SLR2), ./bit_gen/constraints/v3e.$(ALVEO).physical.slr2.$(SLR2)$(BK_MODE)$(xo_dwc).xdc,)
endif


SLR0_OBJ := $(if $(SLR0), ../release_$(ALVEO)$(lv_xo)_xo$(xo_dwc)/DPUCAHX8H_$(SLR0).xo,)
SLR1_OBJ := $(if $(SLR1), ../release_$(ALVEO)$(lv_xo)_xo$(xo_dwc)/DPUCAHX8H_$(SLR1).xo,)
SLR2_OBJ := $(if $(SLR2), ../release_$(ALVEO)$(lv_xo)_xo$(xo_dwc)/DPUCAHX8H_$(SLR2).xo,)

XO_OBJS := $(sort $(SLR0_OBJ) $(SLR1_OBJ) $(SLR2_OBJ)) 
XO_PATH = release_$(ALVEO)$(lv_xo)_xo$(xo_dwc)

PHYSICAL_XDC := $(sort $(SLR0_PHYSICAL_XDC) $(SLR1_PHYSICAL_XDC) $(SLR2_PHYSICAL_XDC))
TIMING_XDC := $(sort $(SLR0_TIMING_XDC) $(SLR1_TIMING_XDC) $(SLR2_TIMING_XDC))

FORCE_ALL_HBM_ASSIGN = true
AXI_SHARE = false

SLR0_STR := $(if $(SLR0),SLR0=$(SLR0),)
SLR1_STR := $(if $(SLR1),SLR1=$(SLR1),)
SLR2_STR := $(if $(SLR2),SLR2=$(SLR2),)

check_tcl:
	#@touch $(GIT)/option_list*
	@touch ./bit_gen/deephi.sys_link_post.pl
	@touch ./bit_gen/deephi.ini.pl
	@touch ./bit_gen/constraints/opt_design.pre.common.$(ALVEO).tcl
build:  check_tcl ./bit_gen/constraints/HBM_RS_pblock.xdc ./bit_gen/constraints/sys_link_post.tcl ./bit_gen/constraints/cons.ini ./bit_gen/constraints/opt_design.pre.tcl
#	cd ./bit_gen/constraints  && cat opt_design.pre.common.u280.tcl HBM_RS_pblock.xdc v3e.u280.timing.slr0.xdc v3e.u280.physical.slr0.$(SLR0)2M.xdc v3e.u280.timing.slr1.xdc v3e.u280.physical.slr1.$(SLR1)2M.xdc v3e.u280.timing.slr2.xdc v3e.u280.physical.slr2.$(SLR2)2M.xdc    > opt_design.pre.tcl
	cd ./bit_gen; v++ -t hw --platform $(DEVICE) --save-temps --temp_dir $(ALVEO)$(lv_xo)_imp -l --config "./constraints/cons.ini" -o $(ALVEO)$(lv_xo).xclbin $(XO_OBJS)
	sed -i 's/"m_interrupt_id": "1"/"m_interrupt_id": "temp"/g' ./bit_gen/u50lv_imp/link/int/u50lv.rtd
	sed -i 's/"m_interrupt_id": "0"/"m_interrupt_id": "1"/g' ./bit_gen/u50lv_imp/link/int/u50lv.rtd
	sed -i 's/"m_interrupt_id": "temp"/"m_interrupt_id": "0"/g' ./bit_gen/u50lv_imp/link/int/u50lv.rtd
	cd ./bit_gen/u50lv_imp  && /proj/xbuilds/SWIP/2020.1_0526_1803/installs/lin64/Vitis/2020.1/bin/xclbinutil --add-section BITSTREAM:RAW:./link/int/partial.bit --force --target hw --key-value SYS:dfx_enable:true --add-section :JSON:./link/int/u50lv.rtd --append-section :JSON:./link/int/appendSection.rtd --add-section CLOCK_FREQ_TOPOLOGY:JSON:./link/int/u50lv_xml.rtd --add-section BUILD_METADATA:JSON:./link/int/u50lv_build.rtd --add-section EMBEDDED_METADATA:RAW:./link/int/u50lv.xml --add-section SYSTEM_METADATA:RAW:./link/int/systemDiagramModelSlrBaseAddress.json --key-value SYS:PlatformVBNV:xilinx_u50lv_gen3x4_xdma_2_202010_1 --output ../u50lv.xclbin


./bit_gen/constraints/opt_design.pre.tcl: ./bit_gen/constraints/opt_design.pre.common.$(ALVEO).tcl \
	./bit_gen/constraints/HBM_RS_pblock.xdc 
	@ echo "DWC_EN is $(DWC_EN) xo_dwc is $(xo_dwc) "
	@ echo "SLR0 constraints: $(SLR0_TIMING_XDC) $(SLR0_PHYSICAL_XDC)"
	@ echo "SLR1 constraints: $(SLR1_TIMING_XDC) $(SLR1_PHYSICAL_XDC)"
	@ echo "SLR2 constraints: $(SLR2_TIMING_XDC) $(SLR2_PHYSICAL_XDC)"
	-cat $^ $(SLR0_TIMING_XDC) $(SLR0_PHYSICAL_XDC) $(SLR1_TIMING_XDC) $(SLR1_PHYSICAL_XDC) $(SLR2_TIMING_XDC) $(SLR2_PHYSICAL_XDC) > $@


./bit_gen/constraints/sys_link_post.tcl ./bit_gen/hbm_assignment.txt ./bit_gen/constraints/HBM_RS_pblock.xdc: ./bit_gen/deephi.sys_link_post.pl
	@ echo "SLR0_ENGINE_NUM: $(SLR0_ENGINE_NUM)"
	@ echo "SLR1_ENGINE_NUM: $(SLR1_ENGINE_NUM)"
	@ echo "SLR2_ENGINE_NUM: $(SLR2_ENGINE_NUM)"
	perl $< -SLR0 $(SLR0_ENGINE_NUM) -SLR1 $(SLR1_ENGINE_NUM) -SLR2 $(SLR2_ENGINE_NUM) -FORCE_D_LOC "$(FORCE_D_LOC)" -FORCE_W_LOC "$(FORCE_W_LOC)" -FORCE_I_LOC "$(FORCE_I_LOC)" -ALVEO "$(ALVEO)" -SYS_LINK_POST ./bit_gen/constraints/sys_link_post.tcl -TXT ./bit_gen/hbm_assignment.txt -FORCE_ALL_HBM_ASSIGN "$(FORCE_ALL_HBM_ASSIGN)" -MANUALLY_HBM_ASSIGN "$(MANUALLY_HBM_ASSIGN)" -AXI_SHARE "$(AXI_SHARE)" -DEVICE "${DEVICE}" -HBM_FREQ $(HBM_FREQ)

./bit_gen/constraints/cons.ini: ./bit_gen/deephi.ini.pl ./bit_gen/hbm_assignment.txt
	perl ./bit_gen/deephi.ini.pl  \
		 ALVEO=$(ALVEO) HBM_PORT_ALLOCATION=./bit_gen/hbm_assignment.txt \
		FORCE_ALL_HBM_ASSIGN=$(FORCE_ALL_HBM_ASSIGN) \
		KRNL=DPUCAHX8H STRATEGY=Explore FREQ0=$(ACLK_FREQ) FREQ1=100 $(SLR0_STR) $(SLR1_STR) $(SLR2_STR) USE_2LV=$(USE_2LV) OPT_DESIGN_DIRECTIVE=$(OPT_DESIGN_DIRECTIVE) PLACE_DIRECTIVE=$(PLACE_DIRECTIVE) XO_PATH=$(XO_PATH)


u55c_withdwc: release_u55c_xo_withdwc
	cd ./bit_gen/constraints  && cat opt_design.pre.common.u280.tcl HBM_RS_pblock.xdc v3e.u280.timing.slr0.xdc v3e.u55c.physical.slr0.3ENGINE2M.xdc v3e.u280.timing.slr1.xdc v3e.u55c.physical.slr1.4ENGINE2M.xdc v3e.u280.timing.slr2.xdc v3e.u55c.physical.slr2.4ENGINE2M.xdc    > opt_design.pre.u55c.tcl
	cd ./bit_gen; v++ -t hw --platform xilinx_u55c_gen3x16_xdma_2_202110_1 --save-temps --temp_dir u55c_imp -l --config "./script/cons_u55c_withdwc.ini" -o u280.xclbin ../$</DPUCAHX8H_4ENGINE.xo ../$</DPUCAHX8H_3ENGINE.xo
u280_2DPU: release_u280_xo
	cd ./bit_gen/constraints  && cat opt_design.pre.common.u280.tcl HBM_RS_pblock.xdc v3e.u280.timing.slr1.xdc v3e.u280.physical.slr1.5ENGINE2M.xdc v3e.u280.timing.slr2.xdc v3e.u280.physical.slr2.5ENGINE2M.xdc    > opt_design.pre.u280.tcl
	cd ./bit_gen; v++ -t hw --platform xilinx_u280_xdma_201920_3 --save-temps --temp_dir u280_imp -l --config "./script/cons_u280_2DPU.ini" -o u280_2DPU.xclbin  ../$</DPUCAHX8H_5ENGINE.xo


release_u50_xo: update_dpu
	@ echo "packaging $@ XO-IP ..."
	-mkdir $@
	cp -r xo_release/* $@
	cp -r DPUCAHX8H_SRC $@
	cd  $@ &&\
	rm -rf *.xo && \
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  1ENGINE 300 u50 && \
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  2ENGINE 300 u50 && \
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  3ENGINE 300 u50

release_u50lv_xo: update_dpu
	@ echo "packaging $@ XO-IP ..."
	-mkdir $@
	cp -r xo_release/* $@
	cp -r DPUCAHX8H_SRC $@
	cd  $@ &&\
	rm -rf *.xo && \
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  1ENGINE 275 u50lv &&\
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  2ENGINE 275 u50lv &&\
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  3ENGINE 275 u50lv &&\
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  4ENGINE 275 u50lv &&\
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  5ENGINE 275 u50lv
release_u280_xo: update_dpu
	@ echo "packaging $@ XO-IP ..."
	-mkdir $@
	cp -r xo_release/* $@
	cp -r DPUCAHX8H_SRC $@
	cd  $@ &&\
	rm -rf *.xo && \
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  1ENGINE 250 u280 &&\
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  2ENGINE 250 u280 &&\
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  3ENGINE 250 u280 &&\
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  4ENGINE 250 u280 &&\
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  5ENGINE 250 u280
release_u280_xo_withdwc: update_dpu
	@ echo "packaging $@ XO-IP ..."
	-mkdir $@
	cp -r xo_release/* $@
	cp -r DPUCAHX8H_SRC $@
	cd  $@ &&\
	rm -rf *.xo && \
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  1ENGINE 250 u280 &&\
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  2ENGINE 250 u280 &&\
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  3ENGINE 250 u280 &&\
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  4ENGINE 250 u280 &&\
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  5ENGINE 250 u280

release_u50lv_xo_withdwc: update_dpu
	@ echo "packaging $@ XO-IP ..."
	-mkdir $@
	cp -r xo_release/* $@
	cp -r DPUCAHX8H_SRC $@
	cd  $@ &&\
	rm -rf *.xo && \
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  4ENGINE 275 u50lv_withdwc &&\
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  3ENGINE 275 u50lv_withdwc &&\
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  2ENGINE 275 u50lv_withdwc &&\
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  1ENGINE 275 u50lv_withdwc 
release_u55c_xo_withdwc: update_dpu
	@ echo "packaging $@ XO-IP ..."
	-mkdir $@
	cp -r xo_release/* $@
	cp -r DPUCAHX8H_SRC $@
	cd  $@ &&\
	sed -i "s/32:0/33:0/g" ./src/DPUCAHX8H_*ENGINE.v &&\
	rm -rf *.xo && \
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  1ENGINE 300 u55c_withdwc &&\
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  2ENGINE 300 u55c_withdwc &&\
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  3ENGINE 300 u55c_withdwc &&\
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  4ENGINE 300 u55c_withdwc
release_xo1: release_u50_xo release_u50lv_xo release_u280_xo release_u50lv_xo_withdwc release_u55c_xo_withdwc

release_xo:
	cp 	$(GIT)/option_list.u50lv_withdwc $(GIT)/option_list
	make release_u50lv_xo_withdwc
	cp 	$(GIT)/option_list.u55c_withdwc $(GIT)/option_list
	make release_u55c_xo_withdwc
	cp 	$(GIT)/option_list.u50lv $(GIT)/option_list
	make release_u50lv_xo
	cp 	$(GIT)/option_list.u280 $(GIT)/option_list
	make release_u280_xo
	cp 	$(GIT)/option_list.u50 $(GIT)/option_list
	make release_u50_xo
	cp 	$(GIT)/option_list.u280_withdwc $(GIT)/option_list
	make release_u280_xo_withdwc
