# (C) Copyright 2020 - 2021 Xilinx, Inc.
# SPDX-License-Identifier: Apache-2.0

set jobs 8

# parse arguments
for { set i 0 } { $i < $argc } { incr i } {
  # jobs
  if { [lindex $argv $i] == "-jobs" } {
    incr i
    set jobs [lindex $argv $i]
  }
}

set proj_name vck190_mipiRxSingle_hdmiTx
set proj_board [get_board_parts "*:vck190:*" -latest_file_version]
puts "Board Part: $proj_board"
create_project -name ${proj_name} -force -dir ./project -part [get_property PART_NAME [get_board_parts $proj_board]] 
set_property board_part $proj_board [current_project]
       

set proj_dir ./project
set bd_tcl_dir ./scripts
set device vc1902
set output {zip xsa bit}
set xdc_list {xdc/mipi_vck190.xdc xdc/hdmi_tx_vck190.xdc xdc/pl_clk_uncertainty.xdc xdc/timing.xdc xdc/default.xdc}
set ip_repo_path {./ip}
      
    
import_files -fileset constrs_1 $xdc_list
        
    
set_property ip_repo_paths $ip_repo_path [current_project] 
update_ip_catalog
    
# Create block diagram design and set as current design
set design_name $proj_name
create_bd_design $proj_name
current_bd_design $proj_name

# Set current bd instance as root of current design
set parentCell [get_bd_cells /]
set parentObj [get_bd_cells $parentCell]
current_bd_instance $parentObj
        
source $bd_tcl_dir/config_bd.tcl
save_bd_design
    

make_wrapper -files [get_files $proj_dir/${proj_name}.srcs/sources_1/bd/$proj_name/${proj_name}.bd] -top
import_files -force -norecurse $proj_dir/${proj_name}.srcs/sources_1/bd/$proj_name/hdl/${proj_name}_wrapper.v
update_compile_order
set_property top ${proj_name}_wrapper [current_fileset]
update_compile_order -fileset sources_1

save_bd_design
validate_bd_design

#Set Platform properties
set_property platform.board_id $proj_name [current_project]
            
set_property platform.default_output_type "xclbin" [current_project]
            
set_property platform.design_intent.datacenter false [current_project]
            
set_property platform.design_intent.embedded true [current_project]
            
set_property platform.design_intent.external_host false [current_project]
            
set_property platform.design_intent.server_managed false [current_project]
            
set_property platform.extensible true [current_project]
            
set_property platform.platform_state "pre_synth" [current_project]
            
set_property platform.full_pdi_file "$proj_dir/${proj_name}.runs/impl_1/${proj_name}_wrapper.pdi" [current_project]
            
set_property platform.name $proj_name [current_project]
            
set_property platform.vendor "xilinx" [current_project]
            
set_property platform.version "1.0" [current_project]

set_property strategy Performance_ExploreWithRemap [get_runs impl_1] 

# Generate IPs and Implement design
generate_target all [get_files $proj_dir/${proj_name}.srcs/sources_1/bd/$proj_name/${proj_name}.bd]        

set_property synth_checkpoint_mode Hierarchical [get_files $proj_dir/${proj_name}.srcs/sources_1/bd/$proj_name/${proj_name}.bd]
launch_runs synth_1 -jobs ${jobs}
wait_on_run synth_1

launch_runs impl_1 -to_step write_device_image
            
wait_on_run impl_1

# Generate XSA
write_hw_platform -hw -force -file $proj_dir/${proj_name}.xsa
validate_hw_platform -verbose $proj_dir/${proj_name}.xsa

exit



