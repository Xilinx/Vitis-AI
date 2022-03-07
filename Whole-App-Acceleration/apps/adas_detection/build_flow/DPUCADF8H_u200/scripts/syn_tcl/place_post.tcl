#####################################
# Tcl file used in PLACE_DESIGN.TCL.POST
#
puts "Start to source [info script]"


# source $SDA_PATH/scripts/constraints/timing/set_acc_group.tcl
# source $SDA_PATH/scripts/constraints/timing/opt_acc.tcl

if { $BOARD == "u200" } {
    phys_opt_design -fanout_opt
    phys_opt_design -slr_crossing_opt 
#    phys_opt_design -directive AggressiveFanoutOpt
    phys_opt_design -directive AggressiveFanoutOpt
#    phys_opt_design -directive AggressiveExplore
    phys_opt_design -directive AggressiveExplore
} else {

    phys_opt_design -directive AggressiveExplore
    phys_opt_design -directive AggressiveExplore
    phys_opt_design -directive AggressiveFanoutOpt

    set_clock_uncertainty -setup 0.0 [get_clocks SLR*_ACLK_DR]
    set_clock_uncertainty -setup 0.0 [get_clocks SLR*_ACLK]



    # phys_opt_design -directive AggressiveExplore
    # phys_opt_design -directive AggressiveFanoutOpt
    # phys_opt_design -directive AggressiveExplore
}

#if { ($SHELL_VER =="202002")&&($BOARD == "u200") } {
#    set_clock_uncertainty -setup 0.0 [get_clocks SLR*_ACLK_DR]
#    set_clock_uncertainty -setup 0.0 [get_clocks SLR*_ACLK]
#}
# source $SDA_PATH/scripts/constraints/timing/opt_acc.tcl

make_outputs "place"
