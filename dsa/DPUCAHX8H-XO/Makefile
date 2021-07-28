clean:
	-rm -rf rm xo_release/packaged_kernel*
	-rm bit_gen/vivado*
	-rm -rf xo_release/tmp_kernel_pack_DPUCAHX8H_*
	-rm -rf bit_gen/*log
	-rm -rf bit_gen/*_imp
	-rm -rf bit_gen/*ltx
	-rm -rf bit_gen/*xclbin*
	-rm -rf bit_gen/.*swp
	-rm -rf bit_gen/.*swo
	-rm -rf bit_gen/.ipcache
	-rm -rf bit_gen/.Xil/
u50lv: release_u50lv_xo
	cd ./bit_gen/constraint  && cat opt_design.pre.common.u50_6cr.tcl v3e.u50_6cr.timing.slr0.xdc v3e.u50_6cr.physical.slr0.5ENGINE.xdc v3e.u50_6cr.timing.slr1.xdc v3e.u50_6cr.physical.slr1.5ENGINE.xdc  > opt_design.pre.u50lv.tcl
	cd ./bit_gen; v++ -t hw --platform xilinx_u50lv_gen3x4_xdma_2_202010_1 --save-temps --temp_dir u50lv_imp -l --config "./script/cons_u50lv.ini" -o u50lv.xclbin ../xo_release/DPUCAHX8H_5ENGINE.xo
u50: release_u50_xo
	cd ./bit_gen/constraint  && cat opt_design.pre.common.u50_6cr.tcl v3e.u50_6cr.timing.slr0.xdc v3e.u50_6cr.physical.slr0.3ENGINE.xdc v3e.u50_6cr.timing.slr1.xdc v3e.u50_6cr.physical.slr1.3ENGINE.xdc  > opt_design.pre.u50.tcl
	cd ./bit_gen; v++ -t hw --platform xilinx_u50_gen3x4_xdma_2_202010_1 --save-temps --temp_dir u50_imp -l --config "./script/cons_u50.ini" -o u50.xclbin ../xo_release/DPUCAHX8H_3ENGINE.xo
u280: release_u280_xo
	cd ./bit_gen/constraint  && cat opt_design.pre.common.u280.tcl HBM_RS_pblock.xdc v3e.u280.timing.slr0.xdc v3e.u280.physical.slr0.4ENGINE2M.xdc v3e.u280.timing.slr1.xdc v3e.u280.physical.slr1.5ENGINE2M.xdc v3e.u280.timing.slr2.xdc v3e.u280.physical.slr2.5ENGINE2M.xdc    > opt_design.pre.u280.tcl
	cd ./bit_gen; v++ -t hw --platform xilinx_u280_xdma_201920_3 --save-temps --temp_dir u280_imp -l --config "./script/cons_u280.ini" -o u280.xclbin ../xo_release/DPUCAHX8H_4ENGINE.xo ../xo_release/DPUCAHX8H_5ENGINE.xo

release_u50_xo:
	@ echo "packaging DPUCAHX8H_3ENGINE XO-IP ..."
	cd xo_release &&\
	rm -rf DPUCAHX8H_3ENGINE.xo && \
	 vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  3ENGINE 300 u50
release_u50lv_xo:
	@ echo "packaging DPUCAHX8H_5ENGINE XO-IP ..."
	cd xo_release &&\
	rm -rf DPUCAHX8H_5ENGINE.xo && \
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  5ENGINE 275 u50lv
release_u280_xo:
	@ echo "packaging DPUCAHX8H_5ENGINE XO-IP & DPUCAHX8H_4ENGINE XO-IP..."
	cd xo_release &&\
	rm -rf DPUCAHX8H_4ENGINE.xo && \
	rm -rf DPUCAHX8H_5ENGINE.xo && \
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  4ENGINE 250 u280 && \
	vivado -mode tcl  -source ./script/gen_DPUCAHX8H_ENGINE_xo.tcl -log ./log/vivado.log -jou ./log/vivado.jou -tclargs  5ENGINE 250 u280

