
enable_beta_device *  
set_param ips.UseCIPSv2 1
        
create_project -name vck190_es1_base_trd_platform1 -force -dir ./project -part xcvc1902-vsva2197-2MP-e-S-es1

set part xcvc1902-vsva2197-2MP-e-S-es1
        
set proj_name vck190_es1_base_trd_platform1
set proj_dir ./project
set bd_tcl_dir ./scripts
set board vck190
set device vc1902
set rev es1
set output {zip xsa bit}
set xdc_list {./xdc/vck190_vmk180_lpddr4dualinter.xdc ./xdc/default.xdc ./xdc/mipi_vck190.xdc ./xdc/hdmi_tx_vck190.xdc ./xdc/pl_clk_uncertainty.xdc ./xdc/timing.xdc}
set ip_repo_path {./source/ip}
	        
    
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
file mkdir ${proj_dir}/${proj_name}.sdk
        

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

launch_runs synth_1 -jobs 20
wait_on_run synth_1

launch_runs impl_1
wait_on_run impl_1

launch_runs impl_1 -to_step write_device_image
wait_on_run impl_1

open_run impl_1

write_hw_platform -include_bit -force  ${proj_name}.xsa
validate_hw_platform ${proj_name}.xsa -verbose

