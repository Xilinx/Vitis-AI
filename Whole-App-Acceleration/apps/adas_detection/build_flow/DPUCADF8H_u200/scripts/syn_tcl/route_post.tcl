#####################################
# Tcl file used in ROUTE_DESIGN.TCL.POST
#
puts "Start to source [info script]"

phys_opt_design
phys_opt_design -slr_crossing_opt 
phys_opt_design -slr_crossing_opt 
phys_opt_design -directive Explore
phys_opt_design -directive Explore
phys_opt_design -directive Explore
phys_opt_design -directive Explore
phys_opt_design -directive Explore

make_outputs "route"
