puts "Start to source [info script]"



if { $BOARD == "u200" } {
    phys_opt_design -fanout_opt
    phys_opt_design -slr_crossing_opt 
    phys_opt_design -directive AggressiveFanoutOpt
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



make_outputs "place"
