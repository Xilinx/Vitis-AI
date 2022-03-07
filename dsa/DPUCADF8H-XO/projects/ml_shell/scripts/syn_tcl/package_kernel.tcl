
set path_to_hdl "./src/hdl"
set path_to_packaged "./packaged_kernel_${suffix}"
set path_to_tmp_project "./tmp_kernel_pack_${suffix}"
set USE_ENC "true"
set ENC_VERSION 1




source ./scripts/user_setup/env_config.tcl
if { $BOARD=="u200" } {
    create_project -force kernel_pack $path_to_tmp_project -part xcu200-fsgd2104-2-e
} elseif { $BOARD=="u250" } {
    create_project -force kernel_pack $path_to_tmp_project -part xcu250-figd2104-2L-e
}



  
 

if { $USE_ENC == "true" } {
  read_verilog -sv [glob \
  ../../ml_ip/cnnv3/rtl_enc/dpdpuv3_wrapper.v \
  ../../ml_ip/cnnv3/rtl_enc/dpdpuv3_all.sv \
  ../../ml_ip/cnnv3/rtl_enc/dpdpuv3_def.vh \
  ../../ml_ip/cnnv3/rtl_enc/dpdpuv3_func.vh \
  ]
} else {
  source ./scripts/syn_tcl/read_fileset.tcl
}




source ./scripts/syn_tcl/gen_ips.tcl

set_property top dpdpuv3_wrapper [current_fileset]

update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

ipx::package_project -root_dir $path_to_packaged -vendor xilinx.com -library RTLKernel -taxonomy /KernelIP -import_files -set_current false
ipx::unload_core $path_to_packaged/component.xml
ipx::edit_ip_in_project -upgrade true -name tmp_edit_project -directory $path_to_packaged $path_to_packaged/component.xml
set_property core_revision 2 [ipx::current_core]
foreach up [ipx::get_user_parameters] {
  ipx::remove_user_parameter [get_property NAME $up] [ipx::current_core]
}
set_property sdx_kernel true [ipx::current_core]
set_property sdx_kernel_type rtl [ipx::current_core]

ipx::create_xgui_files [ipx::current_core]
ipx::associate_bus_interfaces -busif m00_axi -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif s_axi_control -clock ap_clk [ipx::current_core]


ipx::infer_bus_interface ap_clk_2 xilinx.com:signal:clock_rtl:1.0 [ipx::current_core]
ipx::infer_bus_interface ap_rst_n_2 xilinx.com:signal:reset_rtl:1.0 [ipx::current_core]

set_property xpm_libraries {XPM_CDC XPM_MEMORY XPM_FIFO} [ipx::current_core]
set_property supported_families { } [ipx::current_core]
set_property auto_family_support_level level_2 [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]
close_project -delete
