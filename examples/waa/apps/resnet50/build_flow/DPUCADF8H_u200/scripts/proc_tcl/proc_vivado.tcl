proc make_inputs { fileset type } {

if { [info exists fileset] } {
    foreach files $fileset {
	foreach f [split $files " "] {
	    if { $f == ""} { continue }

	    if { [glob -nocomplain $f] == "" } {
		puts "ERROR: Not found $f when reading in $type"
	    } else {
	        puts "INFO: Reading $type - $f"
		switch -glob $type {
		    "IPS"		    {read_ip -verbose [glob $f]}
		    "VERILOG"		{read_verilog -verbose -sv [glob $f]}
		    "SVERILOG"		{read_verilog -verbose -sv [glob $f]}
		    "SYNTH_IPS"		{synth_ip -verbose -force [glob $f]}
		    "UPGRADE_IPS"	{upgrade -verbose -force [glob $f]}
		    "GENERATE_IPS"	{generate_target -verbose [glob $f]}
		    "XDC"		    {read_xdc -verbose [glob $f]}
		    "PBLOCK"		{source -verbose [glob $f]}
		}
	    }
	}
    }
}


}



proc make_outputs { stage } {
global REPORTS_DIR
global DESIGN_NAME
global RESULTS_DIR
global DESIGN_CELLS 
global CORE0_CELLS 
global LOG_DIR
global SDA_PATH 
global VIVADO_VER    
global BUILD_DIR

if {$VIVADO_VER == "201802"} {
    foreach f [glob $SDA_PATH/_x/link/vivado/*.log] {file copy -force $f $LOG_DIR}
} elseif {$VIVADO_VER == "201901"} {
    foreach f [glob $SDA_PATH/_x/link/vivado/vpl/*.log] {file copy -force $f $LOG_DIR}
} elseif {$VIVADO_VER == "201902"} {
    foreach f [glob $BUILD_DIR/link/vivado/vpl/*.log] {file copy -force $f $LOG_DIR}
} elseif {$VIVADO_VER == "202002"} {
    foreach f [glob $BUILD_DIR/link/vivado/vpl/*.log] {file copy -force $f $LOG_DIR}
}

report_utilization -hierarchical -hierarchical_percentages -hierarchical_depth 3 -file ${REPORTS_DIR}/${DESIGN_NAME}.${stage}.util_depth.rpt
#if { $DESIGN_CELLS != "" } {
    #report_utilization -hierarchical -hierarchical_percentages -cells $DESIGN_CELLS -hierarchical_depth 2 -file ${REPORTS_DIR}/${DESIGN_NAME}.${stage}.dpdpuv3_top.util_depth.rpt
    #report_utilization -hierarchical -hierarchical_percentages -cells $CORE0_CELLS -hierarchical_depth 2 -file ${REPORTS_DIR}/${DESIGN_NAME}.${stage}.dpdpuv3_core0.util_depth.rpt
#}
report_timing -delay_type max -max_path 1000 -file ${REPORTS_DIR}/${DESIGN_NAME}.${stage}.max_timing.rpt
report_timing -delay_type min -max_path 1000 -file ${REPORTS_DIR}/${DESIGN_NAME}.${stage}.min_timing.rpt
report_timing_summary  -file ${REPORTS_DIR}/${DESIGN_NAME}.${stage}.timing_summary.rpt
report_utilization -file ${REPORTS_DIR}/${DESIGN_NAME}.${stage}.util.rpt

if { $stage != "synth" && $stage != "opt_design" && $stage != "power_opt_design"} {
    report_design_analysis -congestion -file ${REPORTS_DIR}/${DESIGN_NAME}.${stage}.congestion.rpt
    #report_design_analysis -complex -file ${REPORTS_DIR}/${DESIGN_NAME}.${stage}.complex.rpt
}

if { $stage == "route" || $stage == "final" || $stage == "post_route_phys_opt_design" } {
    report_timing -delay_type min -max_path 1000 -file ${REPORTS_DIR}/${DESIGN_NAME}.${stage}.min_timing.rpt
    #report_route_status -show_all -file ${REPORTS_DIR}/${DESIGN_NAME}.${stage}.route_status.rpt
    #report_drc -file ${REPORTS_DIR}/${DESIGN_NAME}.${stage}.drc.rpt
}

#report_design_analysis -qor_summary -file ${REPORTS_DIR}/${DESIGN_NAME}.${stage}.qor.rpt

write_checkpoint -force -verbose ${RESULTS_DIR}/${DESIGN_NAME}.${stage}.dcp

}

proc make_ips { ips action} {
    foreach ip [split $ips " "] {
        if { $ip == ""} { continue }
	if { $action == "upgrade" || $action == "both" } {
	    puts "INFO: Upgrading IP - $ip"
	    upgrade_ip [get_ips $ip] -verbose
	}
	if { $action == "synth" || $action == "both" } {
	    puts "INFO: Synthesizing IP - $ip"
	    synth_ip -force [get_ips $ip] -verbose
	}
    }   
}

proc resyn_ips { ip_files } {
    foreach files $ip_files {
        foreach ip_file [split $files " "] {
            if { $ip_file == ""} { continue }
            puts "INFO: Resynthesizing IP - ${ip_file}"
            set_property GENERATE_SYNTH_CHECKPOINT FALSE [get_files "$ip_file"]
        }   
    }
}


proc pb { pb_files } {
    foreach files $pb_files {
        source $pb_files
    }
}



