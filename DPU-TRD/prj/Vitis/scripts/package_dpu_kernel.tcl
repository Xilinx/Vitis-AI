# /*
# * Copyright 2019 Xilinx Inc.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *    http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */


set path_to_hdl "../../dpu_ip"
set path_to_packaged "./packaged_kernel_${suffix}"
set path_to_tmp_project "./tmp_kernel_pack_${suffix}"

create_project -force kernel_pack $path_to_tmp_project 
add_files -norecurse [glob $path_to_hdl/Vitis/dpu/hdl/*.v $path_to_hdl/Vitis/dpu/inc/*.vh $path_to_hdl/dpu_eu_*/hdl/dpu_eu_*_dpu.sv $path_to_hdl/dpu_eu_*/inc/arch_para.vh $path_to_hdl/dpu_eu_*/inc/function.vh ./dpu_conf.vh]
add_files -norecurse [glob $path_to_hdl/Vitis/dpu/xdc/*.xdc] -fileset constrs_1
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
ipx::package_project -root_dir $path_to_packaged -vendor xilinx.com -library RTLKernel -taxonomy /KernelIP -import_files -set_current false
ipx::unload_core $path_to_packaged/component.xml
ipx::edit_ip_in_project -upgrade true -name tmp_edit_project -directory $path_to_packaged $path_to_packaged/component.xml
set_property core_revision 0 [ipx::current_core]
foreach up [ipx::get_user_parameters] {
  ipx::remove_user_parameter [get_property NAME $up] [ipx::current_core]
}
set_property sdx_kernel true [ipx::current_core]
set_property sdx_kernel_type rtl [ipx::current_core]
set clk_bus1  [ipx::get_bus_interfaces aclk -of_objects [ipx::current_core]]
set clk_freq1 [ipx::get_bus_parameters FREQ_HZ -of_objects $clk_bus1]
set_property value_resolve_type {user} $clk_freq1
set clk_bus2  [ipx::get_bus_interfaces ap_clk_2 -of_objects [ipx::current_core]]
set clk_freq2 [ipx::get_bus_parameters FREQ_HZ -of_objects $clk_bus2]
set_property value_resolve_type {user} $clk_freq2
ipx::create_xgui_files [ipx::current_core]
ipx::associate_bus_interfaces -busif M_AXI_GP0 -clock aclk [ipx::current_core]
ipx::associate_bus_interfaces -busif M_AXI_HP0 -clock aclk [ipx::current_core]
ipx::associate_bus_interfaces -busif M_AXI_HP2 -clock aclk [ipx::current_core]
ipx::associate_bus_interfaces -busif S_AXI_CONTROL -clock aclk [ipx::current_core]
ipx::infer_bus_interface ap_clk_2 xilinx.com:signal:clock_rtl:1.0 [ipx::current_core]
ipx::infer_bus_interface ap_rst_n_2 xilinx.com:signal:reset_rtl:1.0 [ipx::current_core]
set_property xpm_libraries {XPM_CDC XPM_MEMORY XPM_FIFO} [ipx::current_core]
set_property supported_families { } [ipx::current_core]
set_property auto_family_support_level level_2 [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]
close_project -delete
