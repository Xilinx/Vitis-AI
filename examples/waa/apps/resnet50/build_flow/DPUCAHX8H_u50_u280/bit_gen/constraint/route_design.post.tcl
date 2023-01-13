###
set vivado_clk [clock seconds]
set vivado_dir [file normalize [get_property DIRECTORY [current_project]]]
set origin_dir [file dirname [file normalize [info script]]]
#
puts "**********************************************************"
puts "In Script : [file tail [file normalize [info script]]]"
puts "**********************************************************"
puts "YangEnshan @ [clock format $vivado_clk -format %D:%H:%M:%S] : vivado_dir = $vivado_dir\n"
puts "YangEnshan @ [clock format $vivado_clk -format %D:%H:%M:%S] : origin_dir = $origin_dir\n"

#add the following to solve 1 net route fail issue
route_design -nets [get_nets -hier -top_net_of_hierarchical_group -filter {ROUTE_STATUS == "UNROUTED"}]
set stage route_design
report_utilization -hierarchical -file ./${stage}.util.rpt
report_timing_summary -slack_lesser_than 0.000 -delay_type max -max_path 100 -file ./${stage}.max_timing.rpt
report_timing_summary -slack_lesser_than 0.000 -delay_type min -max_path 100 -file ./${stage}.min_timing.rpt
report_route_status -show_all -file ./${stage}.route_status.rpt
report_design_analysis -congestion -file ./${stage}.congestion.rpt
#write_checkpoint -force -verbose ./${stage}.dcp

#set reports_dir [file normalize $origin_dir/../imp/deephi/System/reports]
#set scripts_dir [file normalize $origin_dir/../imp/deephi/System/scripts]
##Generate more timing reports
#set intradie_path [get_timing_path -max_paths 10000 -unique_pins -filter "INTER_SLR_COMPENSATION!=0"]
#report_timing -of_objects $intradie_path -name intradie
#report_timing -of_objects $intradie_path -file ${reports_dir}/intradie.rpt
#exec perl $scripts_dir/filter_timing_rpt.pl -rpt ${reports_dir}/intradie.rpt -type Source -tcl ${reports_dir}/intradie.tcl -report filter_intradie
#source ${reports_dir}/intradie.tcl
#
#set interdie_path_ACLK [get_timing_path -from [get_clocks *dpu_clk] -to [get_clocks *dpu_clk] -max_paths 10000 -unique_pins -filter {INTER_SLR_COMPENSATION==""}]
#report_timing -of_objects $interdie_path_ACLK -name interdie_ACLK
#report_timing -of_objects $interdie_path_ACLK -file ${reports_dir}/interdie_ACLK.rpt
#exec perl $scripts_dir/filter_timing_rpt.pl -rpt ${reports_dir}/interdie_ACLK.rpt -type Source -tcl ${reports_dir}/interdie_ACLK_Source.tcl -report filter_interdie_ACLK_Source
#source ${reports_dir}/interdie_ACLK_Source.tcl
#exec perl $scripts_dir/filter_timing_rpt.pl -rpt ${reports_dir}/interdie_ACLK.rpt -type Destination -tcl ${reports_dir}/interdie_ACLK_Destination.tcl -report filter_interdie_ACLK_Destination
#source ${reports_dir}/interdie_ACLK_Destination.tcl
