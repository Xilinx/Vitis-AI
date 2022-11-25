rename link_design link_design_orig;
proc link_design {args} {
	puts "TCL_LIBS: skipping VPL link_design with $args";
	catch {close_design}
	remove_files [get_files]
	
	##################### get cell info from pre defined JSON file
	######### get project and kernel info from design_info dict
	set design_info_json [open "../../design_info.json" "r"];
	if { [info exists json] } {unset json};
	set design_info [read $design_info_json];
	set design_info [json::json2dict $design_info];
	close $design_info_json;
	
	set project_name [dict get $design_info project];
	set top_synth_run [dict get $design_info project top_synth_run];
	set design_mode [dict get $design_info project design_mode];
	dict for {kern_run kern_info} [dict get $design_info kernels] {dict with kern_info {lappend krnl_runs $kern_run} };
	# set kernel_run [lindex $krnl_runs 0];
	# set kernel_top [dict get $design_info kernels $kernel_run top];
	
	######### get abstract shell info from abs_shell_info dict
	set abs_shell_info_json [open "RUNDIR_CUSTOM/pre-built_info.json" "r"];
	if { [info exists json] } {unset json};
	set abs_shell_info [read $abs_shell_info_json];
	set abs_shell_info [json::json2dict $abs_shell_info];
	close $abs_shell_info_json;
	
	set abs_sources_path [dict get $abs_shell_info Path];
	set abs_shell_dcp [dict get $abs_shell_info PreImplementedComponents KernelAbs];
	set timing_xdc [dict get $abs_shell_info Constraints Timing];
	set num_modules [dict get $abs_shell_info Accel num_modules];
	set sub_accel_modules [split [dict get $abs_shell_info Accel modules] ","];
	set rm_module [dict get $abs_shell_info Design AccelModule];
	set rm_module_hier_cell [lindex [split $rm_module "/"] end];
	set top_cell [dict get $abs_shell_info Design Top];
	set part [dict get $abs_shell_info Platform Part];
	
	# enough of setup; now start working on design
	##################### method 1: 
	## this is default method; add_files followed by link_design
	## not sure what is the drawback of using other method - open_checkpoint followed by read_checkpoint -cell
	# add_files ${abs_sources_path}/${abs_shell_dcp};
	# if { $num_modules > 1 } {
	# 	add_files ../${top_synth_run}/${rm_module_hier_cell}.dcp;
	# 	set_property SCOPED_TO_CELLS ${rm_module} [get_files ${rm_module_hier_cell}.dcp];
	# 	# foreach sub_accel $sub_accel_modules {
	# 	# 	set accel_ip_runs [dict get $design_info kernels $sub_accel runs];
	# 	# 	set accel_top [dict get $design_info kernels $sub_accel top];
	# 	# 	add_files ../${accel_ip_runs}/${accel_top}.dcp
	# 	# 	set_property SCOPED_TO_CELLS ${rm_module}/${sub_accel} [get_files ${accel_top}.dcp];
	# 	# }
	# } else {
	# 	add_files ../${kernel_run}/${kernel_top}.dcp;
	# 	set_property SCOPED_TO_CELLS ${rm_module} [get_files ${kernel_top}.dcp]
	# }
	# read_xdc ${abs_sources_path}/${timing_xdc};
	# link_design_orig -top ${top_cell} -part ${part} -reconfig_partitions ${rm_module}

	##################### method 2: 
	## open_checkpoint followed by read_checkpoint -cell
	## drawback of using other method - add_files followed by link_design
	### port mismatch error when trying to read in hierarchy cell synthesized separately
	open_checkpoint ${abs_sources_path}/checkpoints/${abs_shell_dcp};
	if { $num_modules > 1 } {
		read_checkpoint -cell ${rm_module} ../${top_synth_run}/${rm_module_hier_cell}.dcp;
	# 	foreach sub_accel $sub_accel_modules {
	# 		set accel_ip_runs [dict get $design_info kernels $sub_accel runs];
	# 		set accel_top [dict get $design_info kernels $sub_accel top];
	# 		read_checkpoint -cell ${rm_module}/${sub_accel} ../${accel_ip_runs}/${accel_top}.dcp
	# 	}
	} else {
	 	set accel_ip_runs [dict get $design_info kernels $sub_accel_modules runs];
		set accel_top [dict get $design_info kernels $sub_accel_modules top];
		read_checkpoint -cell ${rm_module} ../${accel_ip_runs}/${accel_top}.dcp;
	}
	# read_checkpoint -cell ${rm_module} ../synth_1/${rm_module_hier_cell}.dcp
	# read_checkpoint -cell xilinx_zcu102_base_i/pp_pipeline_jpeg ../synth_1/pp_pipeline_jpeg.dcp
	# if { [get_property IS_BLACKBOX [get_cells xilinx_zcu102_base_i/pp_pipeline_jpeg/jpeg_decoder_1]] } {
	# 	read_checkpoint -cell xilinx_zcu102_base_i/pp_pipeline_jpeg/jpeg_decoder_1 ../pp_pipeline_jpeg_inst_0_jpeg_decoder_1_0_synth_1/pp_pipeline_jpeg_inst_0_jpeg_decoder_1_0.dcp
	# }
	# if { [get_property IS_BLACKBOX [get_cells xilinx_zcu102_base_i/pp_pipeline_jpeg/pp_pipeline_accel_1]] } {
	# 	read_checkpoint -cell xilinx_zcu102_base_i/pp_pipeline_jpeg/pp_pipeline_accel_1 ../pp_pipeline_jpeg_inst_0_pp_pipeline_accel_1_0_synth_1/pp_pipeline_jpeg_inst_0_pp_pipeline_accel_1_0.dcp
	# }
	read_xdc ${abs_sources_path}/constraints/${timing_xdc};

	# TODO: <REPORT_UTIL_CHECK>
	
	##################### method 2 alternative: get cell info from abs dcp
	## this method won't require ABS_SHELL_INFO > design and ABS_SHELL_INFO > part
	## open_checkpoint ${abs_sources_path}/${abs_shell_dcp};
	## set rm_module [get_cells -hier -filter {IS_BLACKBOX == 1}];
	## 
	## read_checkpoint -cell [get_cells $rm_module] ../${kernel_run}/${kernel_top}.dcp
	## read_xdc ${abs_sources_path}/${timing_xdc};
	
	## old method to find kernel shell for abstract flow and reset other modules
	## disconnect kernel cells which are not part of abstract shell
	set kernel_cells [get_cells -hier -filter SDX_KERNEL==true]
	foreach kernel_cell $kernel_cells {
		if { [get_property KEEP_PRSHELL_DISCONNECT [get_cells $kernel_cell]] eq "" } {
			puts "TCL_LIBS: ABS $kernel_cell";
		} elseif { [get_property KEEP_PRSHELL_DISCONNECT [get_cells $kernel_cell]] } {
			puts "TCL_LIBS: DISCONNECT $kernel_cell";
			set_property SDX_KERNEL 0 [get_cells $kernel_cell];
		}
	}
	
}
