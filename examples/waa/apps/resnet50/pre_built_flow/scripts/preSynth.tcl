rename synth_design synth_design_orig;
proc synth_design {args} {
	package require json;
	puts "TCL_LIBS: custom synth_design_orig proc";

	# get abstract shell info from abs_shell_info dict
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
	
	# get design info from JSON file
	set design_info_json [open "../../design_info.json" "r"];
	if { [info exists json] } {unset json};
	set design_info [read $design_info_json];
	set design_info [json::json2dict $design_info];
	close $design_info_json;

	# get project and kernel info from design info dict
	set project_name [dict get $design_info project];
	set top_synth_run [dict get $design_info project top_synth_run];
	set top_name [dict get $design_info project top_name];
	set design_mode [dict get $design_info project design_mode];
	set rm_top_ref_name [dict get $design_info project rm_top_ref_name];
	set rm_top_source_verilog [dict get $design_info project rm_top_source_verilog];
	dict for {kern_run kern_info} [dict get $design_info kernels] {
		dict with kern_info {
			lappend krnl_runs $kern_run;
			lappend krnl_tops [dict get $kern_info top];
		}
	}

	if { [set top [lsearch -exact $args "-top"]] >= 0 } {
		incr top;
		if { [lsearch -exact $krnl_tops [lindex $args $top]] >= 0 } {
			puts "TCL_LIBS: running kernel synthesis for synth_design $args";
			eval "synth_design_orig $args";
		} elseif { [lsearch -exact $top_name [lindex $args $top]] >= 0 } {
			puts "TCL_LIBS: running top level synthesis for synth_design $args";
			catch {remove_file [get_files]};
			if { $num_modules > 1 } {
				add_files $rm_top_source_verilog;
				foreach sub_accel $sub_accel_modules {
					set accel_ip_runs [dict get $design_info kernels $sub_accel runs];
					set accel_top [dict get $design_info kernels $sub_accel top];
					add_files ../${accel_ip_runs}/${accel_top}.dcp;
					set_property USED_IN_IMPLEMENTATION TRUE [get_files ../${accel_ip_runs}/${accel_top}.dcp];
				}
				set new_top_args [lreplace $args [lsearch -exact $args $top_name] [lsearch -exact $args $top_name] ${rm_top_ref_name}];
				eval "synth_design_orig $new_top_args -mode out_of_context";
				write_checkpoint ${rm_module_hier_cell}.dcp;
			} else {
				if { $design_mode eq "RTL" } {
					eval "link_design";
				} else {
					eval "link_design -mode out_of_context";
				}
			}
		} else {
			puts "TCL_LIBS: running non-kernel / non-top dependent module synthesis for synth_design $args";
			eval "synth_design_orig $args";
		}
	} else {
		puts "TCL_LIBS: default flow for synth_runs without -top only args $args";
		eval "synth_design_orig $args";
	}
}
