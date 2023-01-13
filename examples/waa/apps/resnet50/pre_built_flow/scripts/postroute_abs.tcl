## setup before actual work
# get abstract shell info from abs_shell_info dict
set abs_shell_info_json [open "RUNDIR_CUSTOM/pre-built_info.json" "r"];
if { [info exists json] } {unset json};
set abs_shell_info [read $abs_shell_info_json];
set abs_shell_info [json::json2dict $abs_shell_info];
close $abs_shell_info_json;

set abs_sources_path [dict get $abs_shell_info Path];
set route_kernel_bb_dcp [dict get $abs_shell_info PreImplementedComponents RouteKernelBB];
set timing_xdc [dict get $abs_shell_info Constraints Timing];
set design_top [dict get $abs_shell_info Design Top];
set rm_module [dict get $abs_shell_info Design AccelModule];

# get project and kernel info from design_info dict
set design_info_json [open "../../design_info.json" "r"];
if { [info exists json] } {unset json};
set design_info [read $design_info_json];
set design_info [json::json2dict $design_info];
close $design_info_json;
	
set design_mode [dict get $design_info project design_mode];
set rm_module_ref [dict get $design_info project rm_top_ref_name];

# orig method to get project and kernel info:
# set design_top [get_property top [current_design]];
# set rm_module [get_cells -hierarchical -filter {HD.RECONFIGURABLE == 1}];
# set rm_module_ref [get_property REF_NAME [get_cells $rm_module]];

## actual work on post route:
write_checkpoint -force ${design_top}_routed_abs.dcp;
write_checkpoint -cell $rm_module -force ${rm_module_ref}_routed_abs.dcp;
write_xdc -type timing -force ${design_top}_routed_abs_timing.xdc;

close_design;
catch {close_project};

# reattach with top level design - waa_script json file
open_checkpoint ${abs_sources_path}/checkpoints/${route_kernel_bb_dcp};
read_checkpoint -cell [get_cells $rm_module] ${rm_module_ref}_routed_abs.dcp;

# reset_timing
read_xdc ${abs_sources_path}/constraints/${timing_xdc};

if { [catch [set orig_dynamic_module [get_cells -hierarchical -filter {HD.RECONFIGURABLE_CONTAINER == 1} -quiet]]] && $design_mode eq "GateLvl"} {
	pr_recombine -cell $orig_dynamic_module;
} else {
	set_property HD.RECONFIGURABLE FALSE [get_cells ${rm_module}];
}

