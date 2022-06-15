
opt_design -directive Explore
set stage opt_design
report_utilization -hierarchical -file ./${stage}.util.rpt
report_timing_summary -slack_lesser_than 0.000 -delay_type max -max_path 100 -file ./${stage}.max_timing.rpt
#write_checkpoint -force -verbose ./${stage}.dcp


#############################################
