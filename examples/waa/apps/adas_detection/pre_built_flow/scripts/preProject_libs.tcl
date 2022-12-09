# synth_design -top pfm_dynamic_axi_gpio_null_0 -part xcu200-fsgd2104-2-e -mode out_of_context
puts "TCL_LIBS: sourcing TCL Libs";
proc get_all_kernel_synth_runs_and_disable_others { design_info } {
	# get abstract shell info from abs_shell_info dict
	set cand_runs [get_runs -filter {IS_SYNTHESIS == 1}]
	dict for {kernel_name kernel_info} [dict get $design_info kernels] {
		lappend sub_accel_ips [dict get $design_info kernels $kernel_name IP];
	};
	foreach cand_run $cand_runs {
		set cand_fs [get_property srcset $cand_run]
		puts "TCL_LIBS: Cand_run $cand_run with fileset $cand_fs";
		if {[get_fileset $cand_fs] == {} } {
			puts "TCL_LIBS: no cand_fs for $cand_run";
			continue;
		}
		if {[get_property fileset_type $cand_fs] != "BlockSrcs"} {
			puts "TCL_LIBS: skipping non BlockSrcs fileset type $cand_fs for $cand_run run";
			continue;
		}
		set cand_files [get_files -of_objects $cand_fs -norecurse];
		foreach cand_file [lsort -uniq $cand_files] {
			set cand_ip [get_ips -all [get_property IP_TOP $cand_file]]
			if { [lsearch -exact $sub_accel_ips $cand_ip] < 0 } {
				set_property IS_ENABLED 0 [get_filesets $cand_fs];
				puts "TCL_LIBS: disabling fileset $cand_fs with not IP";
				continue;
			} else {
				puts "TCL_LIBS: VOILA.!!! Sub Accel IP found $cand_ip for fileset $cand_fs of $cand_run run";
				set top [get_property top [get_property srcset [get_runs $cand_run]]];
				set run_fs [get_property srcset [get_runs $cand_run]];
				dict for {kernel_name kernel_info} [dict get $design_info kernels] {
					if { [dict get $design_info kernels $kernel_name IP] eq $cand_ip } {
						dict set design_info kernels $kernel_name runs $cand_run;
						dict set design_info kernels $kernel_name top $top;
						dict set design_info kernels $kernel_name fileset $run_fs;
					}
				}
			}
		}
	}
	return $design_info
}

proc custom_dict2json { design_info } {
	if { [info exists json] } {unset json};
	dict for {type value} [dict get $design_info] {
		if { [llength $value] > 1 } {
			set json_value [custom_dict2json $value];
			set json_value "\"$type\":$json_value";
		} else {
			puts "	DGB: No llength $value";
			set json_value "\"$type\":\"$value\"";
		}
		puts "	DGB: JSON $json_value";
		lappend json $json_value;
	}
	set json "{[join $json ","]}";
	return $json;
}

rename launch_runs launch_runs_orig;
proc launch_runs {args} {
	set cwd [pwd];
	set proj_dir [get_property DIRECTORY [current_project]];

	# get abstract shell info from abs_shell_info dict
	set abs_shell_info_json [open "RUNDIR_CUSTOM/pre-built_info.json" "r"];
	if { [info exists json] } {unset json};
	set abs_shell_info [read $abs_shell_info_json];
	set abs_shell_info [json::json2dict $abs_shell_info];
	close $abs_shell_info_json;
	set num_modules [dict get $abs_shell_info Accel num_modules];
	set accel_modules [split [dict get $abs_shell_info Accel modules] ","];
	set bd_name [dict get $abs_shell_info Design BD];
	set rm_module [dict get $abs_shell_info Design AccelModule];
	set rm_module_hier_cell [lindex [split $rm_module "/"] end];
	set top_cell [dict get $abs_shell_info Design Top];
	set part [dict get $abs_shell_info Platform Part];
	set pblock_xdc [dict get $abs_shell_info Constraints Pblock];
	set abs_sources_path [dict get $abs_shell_info Path];
	
	set top_design_fileset [get_filesets -filter {FILESET_TYPE == "DesignSrcs"}];
	set design_mode [get_property DESIGN_MODE [get_filesets $top_design_fileset]];
	
	## Design Mode = RTL for full flow (non-DFx) design
	## Design Mode = GateLvl for Partial Reconfiguraion flow design
	## PR_FLOW will be set for DFx flow
	### if not PR_FLOW and Design Mode is full flow RTL,
	#### then check get the synth run name which uses top design fileset (sources_1).
	### if using PR_FLOW then get the reconfig_modules and get its run_name
	if { [string equal $design_mode "RTL"] && ![get_property PR_FLOW [current_project]] } {
		# synth run = synth_1
		set synth_run_name [get_runs -filter "IS_SYNTHESIS == 1 && SRCSET == $top_design_fileset"];
	} elseif { [get_property PR_FLOW [current_project]] } {
		# synth run = my_rm_synth_1
		set rm_fileset [get_filesets -of [get_reconfig_modules]];
		set synth_run_name [get_runs -filter "IS_SYNTHESIS == 1 && SRCSET == $rm_fileset"];
	} else {
		# default it to "synth_1"
		set synth_run_name "synth_1";
	}

	puts "TCL_LIBS: custom launch_runs_orig proc";
	if { [lsearch -exact $args "-scripts_only"] < 0 } {
		if { [lsearch -exact $args $synth_run_name] >= 0 } {
			puts "TCL_LIBS: launch_runs for $synth_run_name";

			### modifying design as per the requirement
			### open BD and get information later needed in Synth, Impl
			open_bd_design [get_files $bd_name];
			#### if more than 1 modules are present, merge to 1 hierarchy - done in user post sys link TCL
			#### difficulty of doing it here is, requirement to update, save, validate, generate_target etc,
			# if { $num_modules > 1 } { 
			# 	group_bd_cells ${rm_module_hier_cell} [get_bd_cells ${accel_modules}];
			# 	validate_bd_design;
			# 	save_bd_design;

			# 	update_compile_order -fileset sources_1;
			# 	generate_target all [get_files $bd_name];
			# 	eval "launch_runs_orig $synth_run_name -scripts_only";
			# 	reset_run $synth_run_name;
			# }

			### get the info of project which is later needed in Synth and Impl runs
			set design_info [dict create];
			dict set design_info project name [current_project];
			dict set design_info project design_mode $design_mode;
			dict set design_info project top_fileset $top_design_fileset;
			dict set design_info project top_name [get_property top [current_fileset]];
			dict set design_info project top_synth_run $synth_run_name;
			dict set design_info kernels {};
			if { [info exists sub_accel_ips] } {unset sub_accel_ips};
			
			### get the IP name of the kernel - needed to know which IP is to be run and used for linking
			foreach sub_accel $accel_modules {
				set accel_bd_cell [get_bd_cells -hierarchical -filter "NAME == $sub_accel"];
				set accel_ip [get_ips -filter "CONFIG.Component_Name == [get_property CONFIG.Component_Name $accel_bd_cell]"];
				lappend sub_accel_ips $accel_ip;
				dict set design_info kernels $sub_accel IP $accel_ip;
			}
			puts "TCL_LIBS: These IPs are ACCEL:\n[join $sub_accel_ips "\n"]\n";
			
			puts "TCL_LIBS: disabling other blocksets";
			set design_info [get_all_kernel_synth_runs_and_disable_others $design_info];

			### get the hierarchy cell's actual ref name - to synthesize this cell rather than whole top in case of merging of XOs
			if { [string equal $design_mode "RTL"] && ![get_property PR_FLOW [current_project]] } {
				set verilog_source_file_list [get_files -compile_order sources -used_in synthesis *.v];
			} else {
				set verilog_source_file_list [get_files *.v];
			}
			if { [info exists rm_ref_name] } {unset rm_ref_name};
			foreach verilog_source_file [lreverse $verilog_source_file_list] {
				set source_file [open $verilog_source_file "r"];
				while { [set gets_val [gets $source_file line]] >= 0} {
					if { [regexp -- [subst -nocommands -nobackslashes {^\s*module\s+(${rm_module_hier_cell}\w*)\s*$}] $line whole_match_line rm_ref_name] } {
						puts "TCL_LIBS: reconfig module hier declared as $rm_ref_name in file $verilog_source_file";
						break;
					}
				}
				close $source_file;
				if { [info exists rm_ref_name] } {
					break;
				}
			}
			if { ![info exists rm_ref_name] } {
				set rm_ref_name $rm_module_hier_cell;
			}
			dict set design_info project rm_top_ref_name $rm_ref_name;
			dict set design_info project rm_top_source_verilog $verilog_source_file;

			if { [info exists kernel_ips] } {unset kernel_ips};
			if { [info exists kernel_runs] } {unset kernel_runs};
			if { [info exists kernel_names] } {unset kernel_names};
			dict for {kernel_name kernel_info} [dict get $design_info kernels] {
				lappend kernel_names $kernel_name;
				lappend kernel_ips [dict get $design_info kernels $kernel_name IP];
				lappend kernel_runs [dict get $design_info kernels $kernel_name fileset];
			}

			##################################################
			# save above data to a file
			set design_info_json [open "${proj_dir}/design_info.json" "w"];
			puts $design_info_json [custom_dict2json $design_info];
			catch {close $design_info_json};
			puts "TCL_LIBS: ${proj_dir}/design_info.json file generated.";

			### load the pblock constraints which are already not part of default design - specific to TRD flow
			read_xdc ${abs_sources_path}/constraints/${pblock_xdc};

			puts "TCL_LIBS: disabled blockset runs except $kernel_runs";
			eval "launch_runs_orig $args";
		} else {
			puts "TCL_LIBS: launch_runs others";
			eval "launch_runs_orig $args";
		}
	} else {
		puts "TCL_LIBS: launch_runs -scripts_only";
		eval "launch_runs_orig $args";
	}
}
puts "TCL_LIBS: end of TCL Libs";

