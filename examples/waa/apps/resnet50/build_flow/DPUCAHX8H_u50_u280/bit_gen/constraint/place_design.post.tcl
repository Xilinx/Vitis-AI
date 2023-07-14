set stage place_design
report_utilization -hierarchical -file ./${stage}.util.rpt
report_timing_summary -slack_lesser_than 0.000 -delay_type max -max_path 100 -file ./${stage}.max_timing.rpt
report_timing_summary -slack_lesser_than 0.000 -delay_type min -max_path 100 -file ./${stage}.min_timing.rpt
report_design_analysis -congestion -file ./${stage}.congestion.rpt
#write_checkpoint -force -verbose ./${stage}.dcp
