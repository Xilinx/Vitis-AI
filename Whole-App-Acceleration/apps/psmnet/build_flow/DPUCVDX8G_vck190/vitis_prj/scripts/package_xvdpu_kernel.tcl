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

set path_to_hdl "./xvdpu"
set path_to_scripts "./scripts"
set path_to_packaged "./packaged_kernel_${suffix}"
set path_to_tmp_project "./tmp_kernel_pack_${suffix}"

source -notrace $path_to_scripts/bip_proc.tcl

create_project -force kernel_pack $path_to_tmp_project 
set_msg_config -id "IP_Flow 19-5107" -suppress
set_msg_config -id "IP_Flow 19-3158" -suppress
add_files -norecurse [glob $path_to_hdl/hdl/*.sv $path_to_hdl/inc/*.vh $path_to_hdl/vitis_cfg/*.vh]
add_files -norecurse [glob $path_to_hdl/ttcl/*_json.ttcl]
add_files -norecurse [glob $path_to_hdl/xdc/*.xdc] -fileset constrs_1
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
set_property PROCESSING_ORDER LATE [get_files timing_clocks.xdc]
set_property file_type TCL [get_files [glob $path_to_hdl/ttcl/*_json.ttcl]]
ipx::package_project -root_dir $path_to_packaged -vendor xilinx.com -library ip -taxonomy /KernelIP -import_files -set_current false
ipx::unload_core $path_to_packaged/component.xml
ipx::edit_ip_in_project -upgrade true -name tmp_edit_project -directory $path_to_packaged $path_to_packaged/component.xml
bip_add_bd  $path_to_scripts $path_to_packaged bd bd.tcl
set_property display_name  "Versal Deep Learning Processing Unit (XVDPU)" [ipx::current_core]
set_property core_revision 0 [ipx::current_core]
ipgui::remove_page -component [ipx::current_core] [bip_pagespec "Page 0"]
set_property type ttcl [ipx::get_files src/*.ttcl -of_objects [ipx::get_file_groups xilinx_anylanguagesynthesis            -of_objects [ipx::current_core]]]
set_property type ttcl [ipx::get_files src/*.ttcl -of_objects [ipx::get_file_groups xilinx_anylanguagebehavioralsimulation -of_objects [ipx::current_core]]]
bip_set_user_parameter
bip_set_wgtbc
bip_set_bus_interfaces
bip_set_bus_enablement_dependency
set_property sdx_kernel true [ipx::current_core]
set_property sdx_kernel_type rtl [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
set_property xpm_libraries {XPM_CDC XPM_MEMORY XPM_FIFO} [ipx::current_core]
set_property supported_families { } [ipx::current_core]
set_property auto_family_support_level level_2 [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]
close_project -delete
